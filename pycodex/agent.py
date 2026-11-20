import asyncio
import typing
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from .context import ContextConfig, ContextManager
from .events import (
    BASE_EVENT_HANDLER,
    AssistantDeltaEvent,
    AutoCompactCompletedEvent,
    AutoCompactFailedEvent,
    AutoCompactStartedEvent,
    CompactCompletedEvent,
    CompactFailedEvent,
    CompactStartedEvent,
    Event,
    EventHandler,
    EventScope,
    MessagesAddedEvent,
    ModelCalledEvent,
    ModelCompletedEvent,
    ModelEvent,
    ModelRequestInterrupted,
    ModelRequestTimingEvent,
    StreamErrorEvent,
    TokenCountEvent,
    ToolCompletedEvent,
    ToolStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnInputReadyEvent,
    TurnInterrupted,
    TurnInterruptedEvent,
    TurnStartedEvent,
)
from .model import (
    ContextLengthExceeded,
    ModelClient,
    ModelControl,
    ResponsesIncompleteError,
)
from .protocol import (
    AgentMessage,
    AssistantMessage,
    ConversationItem,
    ModelResponse,
    Prompt,
    ReasoningItem,
    ToolCall,
    ToolResult,
    TurnResult,
    UserMessage,
)
from .tools import ToolContext, ToolRegistry
from .utils.compactor import DEFAULT_COMPACT_PROMPT, CompactResult, prune_oldest_tool_response
from .utils.random_ids import uuid7_string
from .utils.session_persist import (
    SessionRolloutRecorder,
    load_resumed_session_path,
    resolve_codex_home,
)
from .utils.truncation import truncate_tool_result_for_history


@dataclass(frozen=True)
class ExecutionSnapshot:
    session_id: "str"
    running: "bool"
    stop_requested: "bool"
    usage_tokens: "typing.Union[int, None]"


class TurnContext:
    """Per-turn execution state and model stream filter owned by an Agent."""

    def __init__(
        self,
        observer: "EventHandler",
        turn_id: "typing.Union[str, None]" = None,
    ) -> "None":
        self.turn_id = turn_id or uuid7_string()
        self.messages: "typing.Tuple[ConversationItem, ...]" = ()
        self.iteration = 0
        self.output_text: "typing.Union[str, None]" = None
        self.assistant_text = ""
        self.model_errors: "typing.List[str]" = []
        self.stop_requested = False
        self.follow_up_requested = False
        self.finished = asyncio.Event()
        self.compact_input: "typing.Union[typing.List[ConversationItem], None]" = None
        self.deliver_messages: "typing.Union[typing.Callable[[], None], None]" = None
        self._observer = (
            EventScope.current()
            if observer is BASE_EVENT_HANDLER
            else EventScope.wrap(observer)
        )

    def bind_items(
        self, items: "typing.Iterable[ConversationItem]",
    ) -> "typing.Tuple[ConversationItem, ...]":
        return tuple(
            item if isinstance(item, (ToolCall, ToolResult, ReasoningItem))
            else replace(item, turn_id=self.turn_id)
            for item in items
        )

    @EventScope.router
    def dispatch(self, event: "Event") -> "None":
        if self.finished.is_set():
            return
        if isinstance(event, ModelEvent) and not event.turn_id:
            if self.compact_input is not None and not isinstance(
                event, (StreamErrorEvent, ModelRequestTimingEvent),
            ):
                return
            if isinstance(event, TokenCountEvent) and event.usage is not None:
                return
            if isinstance(event, AssistantDeltaEvent):
                self.assistant_text += event.delta
            elif isinstance(event, StreamErrorEvent):
                self.assistant_text = ""
                self.model_errors.append(event.detail())
            event = replace(event, turn_id=self.turn_id)
        self._observer(event)


class Agent:
    """Minimal Python port of Codex's turn loop.

    The core idea mirrors the Rust implementation:
    build a prompt from history, ask the model for output items, run any tool
    calls, append tool results to history, and keep going until the model emits
    a pure assistant response.
    """

    def __init__(
        self,
        model_client: "ModelClient",
        tool_registry: "ToolRegistry",
        context_config: "ContextConfig",
        session_id: "str",
        parallel_tool_calls: "bool" = True,
        event_handler: "EventHandler" = BASE_EVENT_HANDLER,
        initial_history: "typing.Tuple[ConversationItem, ...]" = (),
        session_file_path: "typing.Union[str, Path, None]" = None,
        agent_path: "str" = "/root",
    ) -> "None":
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.context_manager = ContextManager(
            replace(context_config, model=model_client.model)
        )
        self._parallel_tool_calls = parallel_tool_calls
        self.event_handler = event_handler
        self.runtime_scope = None
        self._history: "typing.List[ConversationItem]" = list(initial_history)
        # Agent identity stays fixed when resume changes the rollout session.
        self.agent_id = session_id
        self.agent_path = agent_path
        self._configure_recording(session_id, session_file_path)
        self._last_total_usage_tokens: "typing.Union[int, None]" = None
        self._turn: "typing.Union[TurnContext, None]" = None
        self.is_shutdown = False
        self.accepts_input = True

    @property
    def history(self) -> "typing.Tuple[ConversationItem, ...]":
        return tuple(self._history)

    @property
    def session_file_path(self) -> "typing.Union[Path, None]":
        recorder = self._rollout_recorder
        return recorder.rollout_path if recorder is not None else None

    @property
    def recorded_session_file_path(self) -> "typing.Union[Path, None]":
        recorder = self._rollout_recorder
        return recorder.recorded_path if recorder is not None else None

    @property
    def is_running(self) -> "bool":
        return self._turn is not None

    def execution_snapshot(self) -> "ExecutionSnapshot":
        """Read execution facts without retaining a second lifecycle state."""
        return ExecutionSnapshot(
            self.session_id, self.is_running,
            self._turn is not None and self._turn.stop_requested, self._last_total_usage_tokens,
        )

    async def wait_until_idle(self) -> "None":
        while self._turn is not None:
            await self._turn.finished.wait()

    def stop_asap(self) -> "None":
        if self._turn is not None:
            self._turn.stop_requested = True
            self.model_client.stop_retry()

    def request_follow_up(self) -> "None":
        """Keep the active turn open for input at its next safe boundary."""
        if self._turn is not None:
            self._turn.follow_up_requested = True

    def receive_messages(
        self, messages: "typing.Tuple[typing.Union[UserMessage, AgentMessage, ToolResult], ...]",
    ) -> "None":
        """Commit messages delivered at a sampling input boundary."""
        self._append_history(messages)

    @property
    def model_name(self) -> "str":
        return self.model_client.model

    def set_model(self, model: "str") -> "None":
        if self.is_running:
            raise RuntimeError("cannot change model while agent is running")
        client = typing.cast(ModelControl, self.model_client)
        client.model = model
        self.context_manager.set_model(model)
        self._last_total_usage_tokens = None

    def set_cwd(self, cwd: "typing.Union[str, Path]") -> "None":
        if self.is_running:
            raise RuntimeError("cannot change directory while agent is running")
        if not self.accepts_input:
            raise RuntimeError("cannot change directory while agent is closing or closed")
        self.context_manager.set_cwd(cwd)

    def resume(
        self, session_file_path: "typing.Union[str, Path, None]" = None
    ) -> "typing.Union[typing.Dict[str, object], None]":
        if self.is_running:
            raise RuntimeError("cannot restore session while agent is running")
        resumed = None
        if session_file_path is not None:
            resumed = load_resumed_session_path(session_file_path)
            session_id = str(resumed["session_id"])
            if not session_id:
                raise ValueError("session file has no session ID")
            restored_history = list(resumed["history"])
            self._configure_recording(session_id, resumed["rollout_path"], resume=True)
            self._history = restored_history
            self.context_manager.reset_history_context()
            self._last_total_usage_tokens = None
        self.is_shutdown = False
        self.accepts_input = True
        return resumed

    def ask(self, text: "str") -> "TurnResult":
        from .utils.async_bridge import run_async

        return run_async(self.run_turn([UserMessage(text)]))

    def _configure_recording(
        self,
        session_id: "str",
        session_file_path: "typing.Union[str, Path, None]" = None,
        resume: "bool" = False,
    ) -> "None":
        if session_file_path is None:
            recorder = None
        elif resume:
            recorder = SessionRolloutRecorder.resume(session_file_path)
            environment_state = recorder.read_environment_state()
            self.tool_registry.runtime_environment.validate_state(environment_state)
        else:
            recorder = SessionRolloutRecorder.create(
                self.context_manager.config.codex_home or resolve_codex_home(),
                session_id,
                self.context_manager.cwd,
                self.model_client.recording_metadata.originator,
                self.model_client.recording_metadata.provider_name,
                self.context_manager.resolve_base_instructions(),
                session_file_path,
            )
        self.session_id = session_id
        self._rollout_recorder = recorder
        if resume:
            self.tool_registry.runtime_environment.restore_state(environment_state)

    async def compact(
        self,
        prune_tool_results_on_context_error: "bool" = True,
    ) -> "typing.Union[CompactResult, None]":
        with self._turn_context(busy_error="cannot compact while agent is running") as turn:
            if not self._history:
                return None
            details = (turn.turn_id, "manual", None, None)
            try:
                with self._execution(turn, manual=True):
                    EventScope.emit(CompactStartedEvent(*details))
                    result = await self._run_compaction(
                        turn, prune_on_overflow=prune_tool_results_on_context_error,
                        record_usage=True,
                    )
            except Exception as exc:
                self._emit_terminal(CompactFailedEvent, *details, str(exc), type(exc).__name__)
                raise
            self._emit_terminal(
                CompactCompletedEvent, *details, result.original_item_count,
                result.retained_item_count, result.pruned_tool_results,
            )
            return result

    async def run_turn(
        self, messages: "typing.Iterable[typing.Union[UserMessage, AgentMessage, ToolResult]]",
        turn_id: "typing.Union[str, None]" = None,
    ) -> "TurnResult":
        with self._open_turn(messages, turn_id) as turn:
            while True:
                first = turn.iteration == 0
                limit = self.context_manager.resolve_auto_compact_token_limit()
                used = self._last_total_usage_tokens
                if limit is not None and used is not None and used >= limit:
                    phase = "pre_turn" if first else "mid_turn"
                    await self._auto_compact(turn, phase, used, limit)
                if first:
                    self._append_history(turn.messages)
                turn.follow_up_requested = False
                if turn.deliver_messages is not None:
                    turn.deliver_messages()
                EventScope.emit(TurnInputReadyEvent(turn.turn_id))
                if turn.stop_requested:
                    raise TurnInterrupted()

                turn.iteration += 1
                response = await self._sample(turn)
                if await self._run_tools(turn, response):
                    continue
                self._append_history(response.items)
                if turn.stop_requested:
                    raise TurnInterrupted()
                if not turn.follow_up_requested:
                    break

            return TurnResult(
                turn_id=turn.turn_id,
                output_text=turn.output_text,
                iterations=turn.iteration,
                response_items=tuple(response.items),
                history=self.history,
            )

    @contextmanager
    def _turn_context(self, turn_id=None, busy_error="agent already has an active turn"):
        if self.is_shutdown:
            raise RuntimeError("agent is shutdown")
        if self.is_running:
            raise RuntimeError(busy_error)
        turn = TurnContext(self.event_handler, turn_id)
        try:
            with EventScope.observe(turn.dispatch):
                yield turn
        finally:
            turn.finished.set()

    @contextmanager
    def _open_turn(self, messages, turn_id):
        with self._turn_context(turn_id) as turn:
            try:
                with self._execution(turn, messages):
                    EventScope.emit(TurnStartedEvent(turn.turn_id, turn.messages))
                    yield turn
            except TurnInterrupted:
                self._emit_terminal(
                    TurnInterruptedEvent, turn.turn_id, turn.iteration, turn.output_text,
                )
                raise
            except Exception as exc:
                self._emit_terminal(
                    TurnFailedEvent, turn.turn_id, turn.iteration, str(exc), type(exc).__name__,
                )
                raise
            self._emit_terminal(TurnCompletedEvent, turn.turn_id, turn.iteration, turn.output_text)

    @contextmanager
    def _bind_turn(self, turn, messages):
        self._turn = turn
        try:
            turn.messages = turn.bind_items(messages)
            yield turn
        finally:
            self._turn = None

    @contextmanager
    def _execution(self, turn, messages=(), manual=False):
        """Let the Runtime observe input binding, tool cleanup and release of the Agent."""
        try:
            turn_scope = self._bind_turn(turn, messages)
            if self.runtime_scope is not None:
                turn_scope = self.runtime_scope(turn_scope, manual=manual)
            with turn_scope, self.tool_registry.runtime_environment.execution(
                self._save_environment_state, manual=manual,
            ):
                yield
        except ModelRequestInterrupted as exc:
            raise TurnInterrupted() from exc

    def _emit_terminal(self, event_type, *args):
        try:
            count = self.tool_registry.runtime_environment.background_work_count(
                event_type in (TurnCompletedEvent, CompactCompletedEvent)
            )
        except (Exception, asyncio.CancelledError) as exc:
            EventScope.report_observer_error(
                "Agent background-work observer failed: " + event_type.kind, exc,
            )
            count = None
        EventScope.emit(event_type(*args, count))

    async def _sample(self, turn: "TurnContext") -> "ModelResponse":
        for attempt in range(2):
            prompt = self._build_prompt(turn.turn_id, self._history)
            EventScope.emit(ModelCalledEvent(
                turn.turn_id, turn.iteration, len(prompt.input), len(prompt.tools),
            ))
            try:
                response = await self.model_client.complete(prompt)
                break
            except ResponsesIncompleteError as exc:
                if exc.reason == "max_output_tokens":
                    self._append_history(
                        item for item in exc.partial_items
                        if isinstance(item, (AssistantMessage, ReasoningItem))
                    )
                raise
            except ContextLengthExceeded as exc:
                self._record_usage(turn.turn_id, exc.usage)
                if attempt == 1:
                    raise
                await self._auto_compact(
                    turn, phase="context_length_exceeded",
                    total_tokens=exc.usage.get("total_tokens") if exc.usage is not None else None,
                    token_limit=exc.token_limit,
                )
                if turn.stop_requested:
                    raise TurnInterrupted()
        self._record_usage(turn.turn_id, response.usage)
        EventScope.emit(ModelCompletedEvent(turn.turn_id, turn.iteration, len(response.items)))
        for item in response.items:
            if isinstance(item, AssistantMessage):
                turn.output_text = item.text
        return response

    def shutdown(self) -> "None":
        if self.is_shutdown:
            return
        self.accepts_input = False
        self.is_shutdown = True
        self.tool_registry.runtime_environment.shutdown(self._save_environment_state)

    async def _run_tools(self, turn: "TurnContext", response: "ModelResponse") -> "bool":
        """Record a tool-producing response and run its calls before the next sample."""
        tool_calls = [item for item in response.items if isinstance(item, ToolCall)]
        if not tool_calls:
            return False
        self._append_history(response.items)
        batches: "typing.List[typing.List[ToolCall]]" = []
        parallel_batch: "typing.List[ToolCall]" = []
        for call in tool_calls:
            can_run_parallel = (
                self._parallel_tool_calls
                and self.tool_registry.supports_parallel(call.name)
            )
            if can_run_parallel:
                parallel_batch.append(call)
                continue

            if parallel_batch:
                batches.append(parallel_batch)
            parallel_batch = []
            batches.append([call])
        if parallel_batch:
            batches.append(parallel_batch)

        results: "typing.List[ToolResult]" = []
        for batch in batches:
            history = self.history if turn.compact_input is None else tuple(turn.compact_input)
            if len(batch) == 1:
                results.append(await self._run_single_tool(turn, batch[0], history))
                continue
            outcomes = await asyncio.gather(
                *(self._run_single_tool(turn, call, history) for call in batch),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
                results.append(outcome)
        self._append_history(self.tool_registry.follow_up_messages(results))
        self._append_history(self.tool_registry.runtime_environment.follow_up_messages())
        if turn.stop_requested:
            raise TurnInterrupted()
        return True

    async def _run_single_tool(
        self,
        turn: "TurnContext",
        call: "ToolCall",
        history: "typing.Tuple[ConversationItem, ...]",
    ) -> "ToolResult":
        context = ToolContext(
            turn_id=turn.turn_id,
            history=history,
            cwd=self.context_manager.cwd,
            agent_id=self.agent_id,
            encrypted_messages=call.encrypted_function_args != [],
            session_id=self.session_id,
        )
        EventScope.emit(ToolStartedEvent(turn.turn_id, call))
        result = await self.tool_registry.execute(call, context)
        self._append_history([truncate_tool_result_for_history(result)])
        turn.assistant_text = ""
        self._save_environment_state()
        EventScope.emit(ToolCompletedEvent(turn.turn_id, call, result))
        return result

    def _save_environment_state(self) -> "None":
        if self.recorded_session_file_path is not None and (
            self._turn is None or self._turn.compact_input is None
        ):
            self._rollout_recorder.save_environment_state(
                self.tool_registry.runtime_environment.dump_state()
            )

    def _append_history(
        self,
        items: "typing.Iterable[ConversationItem]",
    ) -> "None":
        turn = self._turn
        items = tuple(items) if turn is None else turn.bind_items(items)
        if turn is not None and turn.compact_input is not None:
            turn.compact_input.extend(
                replace(item, visible=False) if isinstance(item, AssistantMessage) else item
                for item in items
            )
            return
        if self._rollout_recorder is not None:
            self._rollout_recorder.append_history_items(items, self._history)
        self._history.extend(items)
        messages = tuple(
            item for item in items if isinstance(item, (UserMessage, AgentMessage, AssistantMessage))
        )
        if messages:
            if turn is not None and any(isinstance(item, AssistantMessage) for item in messages):
                turn.assistant_text = ""
            EventScope.emit(MessagesAddedEvent(messages[0].turn_id or "", messages))

    def _record_usage(self, turn_id, usage, update_context=True):
        if usage is None:
            return
        if update_context:
            try:
                self._last_total_usage_tokens = int(usage["total_tokens"])
            except (KeyError, TypeError, ValueError):
                pass
        self.tool_registry.runtime_environment.goal_manager.record_usage(usage)
        EventScope.emit(TokenCountEvent(usage, turn_id))

    def _build_prompt(self, turn_id: "str", history, capture_context: "bool" = True) -> "Prompt":
        return self.context_manager.build_prompt(
            history,
            self.tool_registry.model_visible_specs(
                self.context_manager.resolve_code_mode_only(),
            ),
            self._parallel_tool_calls,
            turn_id=turn_id,
            reasoning_effort=self.model_client.reasoning_effort,
            capture_context=capture_context,
            session_id=self.session_id,
        )

    async def _auto_compact(
        self,
        turn: "TurnContext",
        phase: "str",
        total_tokens: "typing.Union[int, None]",
        token_limit: "typing.Union[int, None]",
    ) -> "None":
        if not self._history:
            return
        details = (turn.turn_id, phase, total_tokens, token_limit)
        try:
            try:
                EventScope.emit(AutoCompactStartedEvent(*details))
                result = await self._run_compaction(turn)
            finally:
                self._save_environment_state()
        except (TurnInterrupted, ModelRequestInterrupted):
            raise
        except Exception as exc:
            EventScope.emit(AutoCompactFailedEvent(*details, str(exc), type(exc).__name__))
            if isinstance(exc, ContextLengthExceeded):
                self._record_usage(turn.turn_id, exc.usage)
            raise
        EventScope.emit(AutoCompactCompletedEvent(
            *details, result.original_item_count, result.retained_item_count, result.pruned_tool_results,
        ))

    async def _run_compaction(
        self, turn: "TurnContext", prune_on_overflow: "bool" = True, record_usage: "bool" = False,
    ) -> "CompactResult":
        committed = self._history
        working = [*committed, UserMessage(DEFAULT_COMPACT_PROMPT)]
        first_exchange = len(working)
        history = working
        pruned = 0
        recorder = self._rollout_recorder
        turn.compact_input = working
        try:
            while True:
                prompt = self._build_prompt(turn.turn_id, history, capture_context=False)
                try:
                    response = await self.model_client.complete(prompt)
                except ContextLengthExceeded:
                    if not prune_on_overflow:
                        raise
                    history = prune_oldest_tool_response(history)
                    if history is None:
                        raise
                    pruned += 1
                    continue
                except ResponsesIncompleteError as exc:
                    if (
                        exc.reason != "max_output_tokens"
                        or not any(isinstance(item, AssistantMessage) for item in exc.partial_items)
                        or any(isinstance(item, ToolCall) for item in exc.partial_items)
                    ):
                        raise
                    response = ModelResponse(list(exc.partial_items))
                if record_usage:
                    self._record_usage(turn.turn_id, response.usage, update_context=False)
                if not await self._run_tools(turn, response):
                    break
                history = prune_oldest_tool_response(working, pruned)

            result = CompactResult.from_response(
                response, self.session_file_path, len(committed), pruned,
            )
            if recorder is not None:
                recorder.append_compacted_history(result.history, committed)
        except BaseException:
            exchanges = working[first_exchange:]
            if exchanges:
                if recorder is not None:
                    recorder.append_history_items(exchanges, committed)
                self._history = committed + exchanges
            raise
        finally:
            turn.compact_input = None

        self._history = list(result.history)
        self.context_manager.reset_history_context()
        self._last_total_usage_tokens = None
        return result
