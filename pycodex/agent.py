import asyncio
import json
import typing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from .context import ContextConfig, ContextManager
from .events import (
    AutoCompactCompletedEvent,
    AutoCompactFailedEvent,
    AutoCompactStartedEvent,
    CompactCompletedEvent,
    CompactFailedEvent,
    CompactStartedEvent,
    Event,
    ModelCalledEvent,
    ModelCompletedEvent,
    ModelEvent,
    StreamErrorEvent,
    TerminalEvent,
    TokenCountEvent,
    ToolCompletedEvent,
    ToolStartedEvent,
    TurnCompletedEvent,
    TurnEvent,
    TurnFailedEvent,
    TurnInterruptedEvent,
    TurnStartedEvent,
)
from .model import (
    DEFAULT_ORIGINATOR,
    ContextLengthExceeded,
    ModelClient,
    ModelControl,
    ResponsesIncompleteError,
)
from .protocol import (
    AssistantMessage,
    ConversationItem,
    ModelResponse,
    ReasoningItem,
    ToolCall,
    ToolResult,
    TurnResult,
    UserMessage,
)
from .tools import ToolContext, ToolRegistry
from .utils import uuid7_string
from .utils.session_persist import (
    SessionRolloutRecorder,
    load_resumed_session_path,
    resolve_codex_home,
    rollout_path_for_session,
)
from .utils.truncation import truncate_tool_result_for_history

if typing.TYPE_CHECKING:
    from .utils.compactor import CompactResult


EventHandler = Callable[[Event], None]
BASE_EVENT_HANDLER: "EventHandler" = lambda _event: None


class TurnInterrupted(RuntimeError):
    def __init__(self):
        super().__init__("turn interrupted")


@dataclass
class _TurnState:
    turn_id: "str"
    iteration: "int" = 0
    output_text: "typing.Union[str, None]" = None


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
        parallel_tool_calls: "bool" = True,
        event_handler: "EventHandler" = BASE_EVENT_HANDLER,
        initial_history: "typing.Tuple[ConversationItem, ...]" = (),
        session_file_path: "typing.Union[str, Path, None]" = None,
        session_id: "typing.Union[str, None]" = None,
    ) -> "None":
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.context_manager = ContextManager(
            replace(context_config, model=model_client.model)
        )
        self._parallel_tool_calls = parallel_tool_calls
        self.event_handler = event_handler
        self._history: "typing.List[ConversationItem]" = list(initial_history)
        self._configure_recording(session_id or uuid7_string(), session_file_path)
        self._last_total_usage_tokens: "typing.Union[int, None]" = None
        self._idle = asyncio.Event()
        self._idle.set()
        self.is_shutdown = False
        self.accepts_input = True
        self._stop_requested = False
        for tool in self.tool_registry.tools():
            tool.bind_agent(self)

    @property
    def history(self) -> "typing.Tuple[ConversationItem, ...]":
        return tuple(self._history)

    @property
    def session_file_path(self) -> "typing.Union[Path, None]":
        recorder = self._rollout_recorder
        return recorder.rollout_path if recorder is not None else None

    @property
    def is_running(self) -> "bool":
        return not self._idle.is_set()

    async def wait_until_idle(self) -> "None":
        while self.is_running:
            await self._idle.wait()

    def stop_asap(self) -> "None":
        if self.is_running:
            self._stop_requested = True

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
            self._last_total_usage_tokens = None
        self.is_shutdown = False
        self.accepts_input = True
        for tool in self.tool_registry.tools():
            tool.bind_agent(self)
        return resumed

    def ask(self, text: "str") -> "TurnResult":
        from .utils.async_bridge import run_async

        return run_async(self.run_turn([text]))

    def fork(self) -> "None":
        if self.is_running:
            raise RuntimeError("cannot fork session while agent is running")
        session_id = uuid7_string()
        path = None
        if self._rollout_recorder is not None:
            path = rollout_path_for_session(
                self.context_manager._config.codex_home or resolve_codex_home(),
                session_id,
            )
        self._configure_recording(session_id, path)

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
        else:
            recorder = SessionRolloutRecorder.create(
                self.context_manager._config.codex_home or resolve_codex_home(),
                session_id,
                self.context_manager.cwd,
                getattr(self.model_client, "_originator", DEFAULT_ORIGINATOR),
                getattr(
                    getattr(self.model_client, "_config", None), "provider_name", None
                ),
                self.context_manager.resolve_base_instructions(),
                session_file_path,
            )
        if hasattr(self.model_client, "_session_id"):
            self.model_client._session_id = session_id
        self.session_id = session_id
        self._rollout_recorder = recorder

    async def compact(
        self,
        prune_tool_results_on_context_error: "bool" = True,
    ) -> "typing.Union[CompactResult, None]":
        if self.is_shutdown:
            raise RuntimeError("agent is shutdown")
        if self.is_running:
            raise RuntimeError("cannot compact while agent is running")
        self._idle.clear()
        try:
            return await self._compact_history(
                uuid7_string(),
                "manual",
                None,
                None,
                prune_tool_results_on_context_error,
            )
        finally:
            self._idle.set()

    async def run_turn(
        self, texts: "typing.List[str]", turn_id: "typing.Union[str, None]" = None
    ) -> "TurnResult":
        if self.is_shutdown:
            raise RuntimeError("agent is shutdown")
        if self.is_running:
            raise RuntimeError("agent already has an active turn")
        self._stop_requested = False
        self._idle.clear()
        turn = _TurnState(turn_id or uuid7_string())
        try:
            self._emit(TurnStartedEvent(turn.turn_id, tuple(texts)))
            while True:
                phase = "pre_turn" if turn.iteration == 0 else "mid_turn"
                await self._maybe_auto_compact(turn.turn_id, phase)
                if turn.iteration == 0:
                    self._append_history(UserMessage(text=text) for text in texts)
                response = await self._sample(turn)
                self._emit(
                    ModelCompletedEvent(
                        turn.turn_id, turn.iteration, len(response.items)
                    )
                )
                self._append_history(response.items)
                tool_calls = []
                for item in response.items:
                    if isinstance(item, AssistantMessage):
                        turn.output_text = item.text
                    elif isinstance(item, ToolCall):
                        tool_calls.append(item)

                if tool_calls:
                    await self._execute_tool_batch(turn.turn_id, tool_calls)
                if self._stop_requested:
                    raise TurnInterrupted()
                if not tool_calls:
                    break

            self._emit(
                TurnCompletedEvent(
                    turn.turn_id,
                    turn.iteration,
                    turn.output_text,
                    self._background_work_count(TurnCompletedEvent),
                )
            )
            result = TurnResult(
                turn_id=turn.turn_id,
                output_text=turn.output_text,
                iterations=turn.iteration,
                response_items=tuple(response.items),
                history=self.history,
            )
            return result
        except TurnInterrupted:
            self._emit(
                TurnInterruptedEvent(
                    turn.turn_id,
                    turn.iteration,
                    turn.output_text,
                    self._background_work_count(TurnInterruptedEvent),
                )
            )
            raise
        except Exception as exc:
            if isinstance(exc, ContextLengthExceeded) and exc.usage is not None:
                self._remember_token_usage(exc.usage)
                self._emit(TokenCountEvent(exc.usage, turn.turn_id))
            self._emit(
                TurnFailedEvent(
                    turn.turn_id,
                    turn.iteration,
                    str(exc),
                    type(exc).__name__,
                    self._background_work_count(TurnFailedEvent),
                )
            )
            raise
        finally:
            self._stop_requested = False
            self._idle.set()

    async def _sample(self, turn: "_TurnState") -> "ModelResponse":
        for attempt in range(2):
            if self._stop_requested:
                raise TurnInterrupted()
            if attempt == 0:
                turn.iteration += 1
            try:
                return await self._complete_model_request(turn.turn_id, turn.iteration)
            except ContextLengthExceeded as exc:
                if attempt == 1:
                    raise
                if exc.usage is not None:
                    self._remember_token_usage(exc.usage)
                    self._emit(TokenCountEvent(exc.usage, turn.turn_id))
                await self._compact_history(
                    turn.turn_id,
                    phase="context_length_exceeded",
                    total_tokens=(
                        exc.usage.get("total_tokens") if exc.usage is not None else None
                    ),
                    token_limit=exc.token_limit,
                    prune_tool_results_on_context_error=True,
                )

    async def maybe_invoke(self, event: "typing.Dict[str, object]") -> "bool":
        if self.is_running or not self.accepts_input:
            return False
        tag = event["type"]
        if not isinstance(tag, str) or not tag.isidentifier():
            raise ValueError("invoke event type must be an identifier")
        payload = {key: value for key, value in event.items() if key != "type"}
        text = (
            f"<{tag}>\n"
            f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
            f"</{tag}>"
        )
        await self.run_turn([text])
        return True

    def shutdown(self) -> "None":
        if self.is_shutdown:
            return
        self.accepts_input = False
        self.is_shutdown = True
        for tool in self.tool_registry.tools():
            tool.shutdown()

    async def _execute_tool_batch(
        self,
        turn_id: "str",
        tool_calls: "typing.List[ToolCall]",
    ) -> "None":
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
            context = ToolContext(
                turn_id=turn_id,
                history=self.history,
            )
            if len(batch) == 1:
                results.append(await self._run_single_tool(turn_id, batch[0], context))
                continue
            outcomes = await asyncio.gather(
                *(self._run_single_tool(turn_id, call, context) for call in batch),
                return_exceptions=True,
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
                results.append(outcome)
        self._append_history(self.tool_registry.follow_up_messages(results))

    async def _run_single_tool(
        self,
        turn_id: "str",
        call: "ToolCall",
        context: "ToolContext",
    ) -> "ToolResult":
        self._emit(ToolStartedEvent(turn_id, call))
        result = await self.tool_registry.execute(call, context)
        self._append_history([truncate_tool_result_for_history(result)])
        self._emit(ToolCompletedEvent(turn_id, call, result))
        return result

    def _emit(self, event: "TurnEvent") -> "None":
        handlers = [self.event_handler]
        if isinstance(event, (TerminalEvent, TurnStartedEvent, CompactStartedEvent)):
            handlers = [
                tool.handle_agent_event for tool in self.tool_registry.tools()
            ] + handlers
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                asyncio.get_running_loop().call_exception_handler(
                    {
                        "message": "Agent event observer failed: " + event.kind,
                        "exception": exc,
                    }
                )

    def _background_work_count(self, event_type) -> "typing.Union[int, None]":
        try:
            return sum(
                tool.background_work_count(
                    event_type in (TurnCompletedEvent, CompactCompletedEvent)
                )
                for tool in self.tool_registry.tools()
            )
        except Exception as exc:
            asyncio.get_running_loop().call_exception_handler(
                {
                    "message": "Agent background-work observer failed: "
                    + event_type.kind,
                    "exception": exc,
                }
            )
            return None

    def _append_history(
        self,
        items: "typing.Iterable[ConversationItem]",
    ) -> "None":
        items = tuple(items)
        if self._rollout_recorder is not None:
            self._rollout_recorder.append_history_items(items, self._history)
        self._history.extend(items)

    def _handle_model_stream_event(self, turn_id: "str", event: "ModelEvent") -> "None":
        if isinstance(event, TokenCountEvent):
            self._remember_token_usage(event.usage)
        self._emit(replace(event, turn_id=turn_id))

    def _remember_token_usage(self, usage: "object") -> "None":
        if not isinstance(usage, dict):
            return
        try:
            self._last_total_usage_tokens = int(usage["total_tokens"])
        except (KeyError, TypeError, ValueError):
            return

    async def _complete_model_request(
        self,
        turn_id: "str",
        iteration: "int",
    ) -> "ModelResponse":
        prompt = self.context_manager.build_prompt(
            self._history,
            self.tool_registry.model_visible_specs(),
            self._parallel_tool_calls,
            turn_id=turn_id,
        )
        self._emit(
            ModelCalledEvent(turn_id, iteration, len(prompt.input), len(prompt.tools))
        )
        try:
            return await self.model_client.complete(
                prompt,
                lambda event: self._handle_model_stream_event(turn_id, event),
            )
        except ResponsesIncompleteError as exc:
            if exc.reason == "max_output_tokens":
                self._append_history(
                    item
                    for item in exc.partial_items
                    if isinstance(item, (AssistantMessage, ReasoningItem))
                )
            raise

    async def _maybe_auto_compact(
        self,
        turn_id: "str",
        phase: "str",
    ) -> "None":
        limit = self.context_manager.resolve_auto_compact_token_limit()
        total_tokens = self._last_total_usage_tokens
        if limit is None or total_tokens is None:
            return
        if total_tokens < limit or not self._history:
            return

        await self._compact_history(
            turn_id,
            phase=phase,
            total_tokens=total_tokens,
            token_limit=limit,
            prune_tool_results_on_context_error=True,
        )

    async def _compact_history(
        self,
        turn_id: "str",
        phase: "str",
        total_tokens: "typing.Union[int, None]" = None,
        token_limit: "typing.Union[int, None]" = None,
        prune_tool_results_on_context_error: "bool" = False,
    ) -> "typing.Union[CompactResult, None]":
        from .utils.compactor import compact_history

        if not self._history:
            return None
        details = (turn_id, phase, total_tokens, token_limit)
        started_type = (
            CompactStartedEvent if phase == "manual" else AutoCompactStartedEvent
        )
        self._emit(started_type(*details))

        def handle_compact_stream_event(event: "ModelEvent") -> "None":
            if isinstance(event, StreamErrorEvent) or (
                phase == "manual" and isinstance(event, TokenCountEvent)
            ):
                self._emit(replace(event, turn_id=turn_id))

        try:
            recorder = self._rollout_recorder
            compact_result = await compact_history(
                self.history,
                self.model_client,
                self.context_manager,
                handle_compact_stream_event,
                prune_tool_results_on_context_error,
                str(recorder.rollout_path) if recorder is not None else None,
                turn_id,
            )
            if recorder is not None:
                recorder.append_compacted_history(compact_result.history, self._history)
            self._history = list(compact_result.history)
            self._last_total_usage_tokens = None
        except Exception as exc:
            if phase == "manual":
                self._emit(
                    CompactFailedEvent(
                        *details,
                        str(exc),
                        type(exc).__name__,
                        self._background_work_count(CompactFailedEvent),
                    )
                )
            else:
                self._emit(
                    AutoCompactFailedEvent(*details, str(exc), type(exc).__name__)
                )
            raise

        completed = details + (
            compact_result.original_item_count,
            compact_result.retained_item_count,
            compact_result.pruned_tool_results,
        )
        if phase == "manual":
            self._emit(
                CompactCompletedEvent(
                    *completed,
                    self._background_work_count(CompactCompletedEvent),
                )
            )
        else:
            self._emit(AutoCompactCompletedEvent(*completed))
        return compact_result
