
import asyncio
import json
import re
from typing import Callable

from .context import ContextManager
from .model import ModelClient, ResponsesIncompleteError
from .protocol import (
    AgentEvent,
    AssistantMessage,
    ConversationItem,
    ModelStreamEvent,
    ReasoningItem,
    ToolCall,
    ToolResult,
    TurnResult,
    UserMessage,
)
from .tools import ExecCommandTool, ToolContext, ToolRegistry, UnifiedExecManager
from .utils import uuid7_string
import typing

if typing.TYPE_CHECKING:
    from .utils.session_persist import SessionRolloutRecorder
    from .runtime_services import AgentRuntimeEnvironment


EventHandler = Callable[[AgentEvent], None]
BASE_EVENT_HANDLER: 'EventHandler' = lambda _event: None
_REQUESTED_TOKENS_RE = re.compile(
    r"requested\s+([0-9,]+)\s+tokens",
    re.IGNORECASE,
)
_REQUESTED_TOKEN_SPLIT_RE = re.compile(
    r"\(([0-9,]+)\s+in\s+the\s+messages,\s+([0-9,]+)\s+in\s+the\s+completion\)",
    re.IGNORECASE,
)
_MAX_CONTEXT_TOKENS_RE = re.compile(
    r"maximum\s+context\s+length\s+is\s+([0-9,]+)\s+tokens",
    re.IGNORECASE,
)
_CONTEXT_LENGTH_ERROR_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "exceeds the context window",
    "exceeded the context window",
)
TERMINAL_TURN_EVENTS = {"turn_completed", "turn_failed", "turn_interrupted"}


class TurnInterrupted(RuntimeError):
    pass


class Agent:
    """Minimal Python port of Codex's turn loop.

    The core idea mirrors the Rust implementation:
    build a prompt from history, ask the model for output items, run any tool
    calls, append tool results to history, and keep going until the model emits
    a pure assistant response.
    """

    def __init__(
        self,
        model_client: 'ModelClient',
        tool_registry: 'ToolRegistry',
        context_manager: 'typing.Union[ContextManager, None]' = None,
        parallel_tool_calls: 'bool' = True,
        event_handler: 'EventHandler' = BASE_EVENT_HANDLER,
        initial_history: 'typing.Tuple[ConversationItem, ...]' = (),
        rollout_recorder: 'typing.Union[SessionRolloutRecorder, None]' = None,
        runtime_environment: 'AgentRuntimeEnvironment' = None,
    ) -> 'None':
        self._model_client = model_client
        self._tool_registry = tool_registry
        self._context_manager = context_manager or ContextManager()
        self._parallel_tool_calls = parallel_tool_calls
        self._event_handler = event_handler
        self._history: 'typing.List[ConversationItem]' = list(initial_history)
        self._rollout_recorder = rollout_recorder
        self._auto_compact_token_limit = (
            self._context_manager.resolve_auto_compact_token_limit()
        )
        self._last_total_usage_tokens: 'typing.Union[int, None]' = None
        self.runtime_environment = runtime_environment
        self.interrupt_asap = False
        self._turn_running = False
        exec_command_tool = self._tool_registry.get_tool("exec_command")
        self._exec_manager = (
            exec_command_tool._manager
            if isinstance(exec_command_tool, ExecCommandTool)
            else None
        )
        if self._exec_manager is not None:
            self._exec_manager.set_notify_hook(self.maybe_invoke)

    @property
    def history(self) -> 'typing.Tuple[ConversationItem, ...]':
        return tuple(self._history)

    def set_event_handler(
        self, event_handler: 'EventHandler' = BASE_EVENT_HANDLER
    ) -> 'None':
        self._event_handler = event_handler

    def replace_history(
        self,
        history: 'typing.Iterable[ConversationItem]',
    ) -> 'None':
        self._history = list(history)

    def set_rollout_recorder(
        self,
        rollout_recorder: 'typing.Union[SessionRolloutRecorder, None]',
    ) -> 'None':
        self._rollout_recorder = rollout_recorder

    def ask(self, text: 'str') -> 'TurnResult':
        from .utils.async_bridge import run_async

        return run_async(self.run_turn([text]))

    def _raise_if_interrupt_requested(
        self,
        turn_id: 'str',
        iteration: 'int',
        output_text: 'typing.Union[str, None]' = None,
    ) -> 'None':
        if self.interrupt_asap:
            self.interrupt_asap = False
            payload: 'typing.Dict[str, object]' = {"iteration": iteration}
            if output_text is not None:
                payload["output_text"] = output_text
            self._emit("turn_interrupted", turn_id, **payload)
            raise TurnInterrupted("turn interrupted")

    async def run_turn(
        self, texts: 'typing.List[str]', turn_id: 'typing.Union[str, None]' = None
    ) -> 'TurnResult':
        self._turn_running = True
        turn_id = turn_id or uuid7_string()
        self.interrupt_asap = False
        new_user_messages = [UserMessage(text=text) for text in texts]

        self._emit(
            "turn_started",
            turn_id,
            user_text="\n".join(texts),
            user_texts=list(texts),
        )
        await self._maybe_auto_compact(turn_id, phase="pre_turn")
        self._history.extend(new_user_messages)
        self._persist_history_items(new_user_messages)

        last_assistant_message: 'typing.Union[str, None]' = None
        final_response_items: 'typing.Tuple[\n    typing.Union[typing.Union[AssistantMessage, ToolCall], ReasoningItem], ...\n]' = ()

        iteration = 0
        try:
            while True:
                self._raise_if_interrupt_requested(
                    turn_id,
                    iteration,
                    output_text=last_assistant_message,
                )
                await self._maybe_auto_compact(turn_id, phase="mid_turn")
                iteration += 1
                response = await self._complete_model_request(
                    turn_id,
                    iteration,
                )
                final_response_items = tuple(response.items)
                self._emit(
                    "model_completed",
                    turn_id,
                    iteration=iteration,
                    item_count=len(response.items),
                )

                recorded_items = self._record_model_response_items(response.items)
                tool_calls = recorded_items[1]
                if recorded_items[2] is not None:
                    last_assistant_message = recorded_items[2]

                if not tool_calls:
                    self._raise_if_interrupt_requested(
                        turn_id,
                        iteration,
                        output_text=last_assistant_message,
                    )
                    self._emit(
                        "turn_completed",
                        turn_id,
                        iteration=iteration,
                        output_text=last_assistant_message,
                    )
                    self._turn_running = False
                    return TurnResult(
                        turn_id=turn_id,
                        output_text=last_assistant_message,
                        iterations=iteration,
                        response_items=final_response_items,
                        history=tuple(self._history),
                    )

                tool_results = await self._execute_tool_batch(turn_id, tool_calls)
                self._history.extend(tool_results)
                self._persist_history_items(tool_results)
                follow_up_messages = self._build_follow_up_messages(tool_results)
                self._history.extend(follow_up_messages)
                self._persist_history_items(follow_up_messages)
                self._raise_if_interrupt_requested(
                    turn_id,
                    iteration,
                    output_text=last_assistant_message,
                )
        except TurnInterrupted:
            self._turn_running = False
            raise
        except asyncio.CancelledError:
            self._turn_running = False
            raise
        except Exception as exc:
            context_usage = _usage_from_context_length_error(str(exc))
            if context_usage is not None:
                self._remember_token_usage(context_usage)
                self._emit("token_count", turn_id, usage=context_usage)
            self._emit(
                "turn_failed",
                turn_id,
                iteration=iteration,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            self._turn_running = False
            raise

    async def maybe_invoke(self, event: 'typing.Dict[str, object]') -> 'bool':
        if self._turn_running or event.get("type") != "exec_command_completed":
            return False
        payload = {
            "session_id": event.get("session_id"),
            "exit_code": event.get("exit_code"),
            "command": event.get("command"),
        }
        text = (
            "<exec_command_completed>\n"
            f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
            "</exec_command_completed>"
        )
        self._turn_running = True
        task = asyncio.create_task(self.run_turn([text]))
        task.add_done_callback(
            lambda task: None if task.cancelled() else task.exception()
        )
        return True

    async def _execute_tool_batch(
        self,
        turn_id: 'str',
        tool_calls: 'typing.List[ToolCall]',
    ) -> 'typing.List[ToolResult]':
        results: 'typing.List[ToolResult]' = []
        parallel_batch: 'typing.List[ToolCall]' = []

        for call in tool_calls:
            can_run_parallel = (
                self._parallel_tool_calls
                and self._tool_registry.supports_parallel(call.name)
            )
            if can_run_parallel:
                parallel_batch.append(call)
                continue

            if parallel_batch:
                prior_results = tuple(results)
                results.extend(
                    await asyncio.gather(
                        *(
                            self._run_single_tool(turn_id, batched_call, prior_results)
                            for batched_call in parallel_batch
                        )
                    )
                )
            parallel_batch = []
            results.append(await self._run_single_tool(turn_id, call, tuple(results)))

        if parallel_batch:
            prior_results = tuple(results)
            results.extend(
                await asyncio.gather(
                    *(
                        self._run_single_tool(turn_id, batched_call, prior_results)
                        for batched_call in parallel_batch
                    )
                )
            )
        return results

    async def _run_single_tool(
        self,
        turn_id: 'str',
        call: 'ToolCall',
        prior_results: 'typing.Tuple[ToolResult, ...]' = (),
    ) -> 'ToolResult':
        payload: 'typing.Dict[str, object]' = {
            "tool_name": call.name,
            "call_id": call.call_id,
            "call": call,
        }
        self._emit("tool_started", turn_id, **payload)
        result = await self._tool_registry.execute(
            call,
            ToolContext(
                turn_id=turn_id,
                history=tuple(self._history) + prior_results,
                collaboration_mode=self._context_manager.collaboration_mode,
            ),
        )
        payload["result"] = result
        payload["is_error"] = result.is_error
        self._emit("tool_completed", turn_id, **payload)
        return result

    def _emit(self, kind: 'str', turn_id: 'str', **payload: 'object') -> 'None':
        if kind in TERMINAL_TURN_EVENTS:
            payload["background_exec_count"] = self._background_exec_count()
        self._event_handler(
            AgentEvent(kind=kind, turn_id=turn_id, payload=dict(payload))
        )

    def _background_exec_count(self) -> 'int':
        manager: 'typing.Union[UnifiedExecManager, None]' = self._exec_manager
        if manager is None:
            return 0
        return manager.running_session_count()

    def _persist_history_items(
        self,
        items: 'typing.Iterable[ConversationItem]',
    ) -> 'None':
        recorder = self._rollout_recorder
        if recorder is None:
            return
        try:
            recorder.append_history_items(items)
        except Exception:  # pragma: no cover - persistence should not break turns
            return

    def _record_model_response_items(
        self,
        items: 'typing.Iterable[object]',
        include_tool_calls: 'bool' = True,
    ) -> 'typing.Tuple[typing.Tuple[ConversationItem, ...], typing.List[ToolCall], typing.Union[str, None]]':
        persisted_response_items: 'typing.List[ConversationItem]' = []
        tool_calls: 'typing.List[ToolCall]' = []
        last_assistant_message = None
        for item in items:
            if isinstance(item, ToolCall) and not include_tool_calls:
                continue
            if not isinstance(item, (AssistantMessage, ToolCall, ReasoningItem)):
                continue
            self._history.append(item)
            persisted_response_items.append(item)
            if isinstance(item, AssistantMessage):
                last_assistant_message = item.text
            elif isinstance(item, ToolCall):
                tool_calls.append(item)
        self._persist_history_items(persisted_response_items)
        return tuple(persisted_response_items), tool_calls, last_assistant_message

    def _handle_model_stream_event(self, turn_id: 'str', event: 'ModelStreamEvent') -> 'None':
        if event.kind == "token_count":
            self._remember_token_usage(event.payload.get("usage"))
        if event.kind == "assistant_delta":
            self._emit("assistant_delta", turn_id, **event.payload)
        elif event.kind == "tool_call":
            self._emit("tool_called", turn_id, **event.payload)
        elif event.kind == "token_count":
            self._emit("token_count", turn_id, **event.payload)
        elif event.kind == "stream_error":
            self._emit("stream_error", turn_id, **event.payload)

    def _remember_token_usage(self, usage: 'object') -> 'None':
        if not isinstance(usage, dict):
            return
        try:
            self._last_total_usage_tokens = int(usage["total_tokens"])
        except (KeyError, TypeError, ValueError):
            return

    async def _complete_model_request(
        self,
        turn_id: 'str',
        iteration: 'int',
    ) -> 'typing.Any':
        attempted_context_compact = False
        while True:
            prompt = self._context_manager.build_prompt(
                self._history,
                self._tool_registry.model_visible_specs(),
                self._parallel_tool_calls,
                turn_id=turn_id,
            )
            self._emit(
                "model_called",
                turn_id,
                iteration=iteration,
                history_size=len(prompt.input),
                tool_count=len(prompt.tools),
            )
            try:
                return await self._model_client.complete(
                    prompt,
                    lambda event: self._handle_model_stream_event(turn_id, event),
                )
            except ResponsesIncompleteError as exc:
                if exc.reason == "max_output_tokens":
                    self._record_model_response_items(
                        exc.partial_items,
                        include_tool_calls=False,
                    )
                raise
            except Exception as exc:
                error_message = str(exc)
                if (
                    not _is_context_length_error_message(error_message)
                    or attempted_context_compact
                ):
                    raise
                attempted_context_compact = True
                context_usage = _usage_from_context_length_error(error_message)
                if context_usage is not None:
                    self._remember_token_usage(context_usage)
                    self._emit("token_count", turn_id, usage=context_usage)
                await self._run_auto_compact(
                    turn_id,
                    phase="context_length_exceeded",
                    total_tokens=(
                        context_usage.get("total_tokens")
                        if context_usage is not None
                        else None
                    ),
                    token_limit=_context_length_error_token_limit(error_message),
                    prune_tool_results_on_context_error=True,
                )
                self._raise_if_interrupt_requested(turn_id, iteration)

    async def _maybe_auto_compact(
        self,
        turn_id: 'str',
        phase: 'str',
    ) -> 'None':
        limit = self._auto_compact_token_limit
        total_tokens = self._last_total_usage_tokens
        if limit is None or total_tokens is None:
            return
        if total_tokens < limit or not self._history:
            return

        await self._run_auto_compact(
            turn_id,
            phase=phase,
            total_tokens=total_tokens,
            token_limit=limit,
            prune_tool_results_on_context_error=True,
        )

    async def _run_auto_compact(
        self,
        turn_id: 'str',
        phase: 'str',
        total_tokens: 'typing.Union[int, None]' = None,
        token_limit: 'typing.Union[int, None]' = None,
        prune_tool_results_on_context_error: 'bool' = False,
    ) -> 'None':
        from .utils.compactor import compact_agent

        payload: 'typing.Dict[str, object]' = {"phase": phase}
        if total_tokens is not None:
            payload["total_tokens"] = total_tokens
        if token_limit is not None:
            payload["token_limit"] = token_limit
        self._emit(
            "auto_compact_started",
            turn_id,
            **payload,
        )

        def handle_compact_stream_event(event: 'ModelStreamEvent') -> 'None':
            if event.kind == "stream_error":
                self._emit("stream_error", turn_id, **event.payload)

        try:
            compact_result = await compact_agent(
                self,
                handle_compact_stream_event,
                prune_tool_results_on_context_error,
            )
        except Exception as exc:
            failed_payload = dict(payload)
            failed_payload.update(
                {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            self._emit(
                "auto_compact_failed",
                turn_id,
                **failed_payload,
            )
            raise

        self._last_total_usage_tokens = None
        if compact_result is None:
            return
        completed_payload = dict(payload)
        completed_payload.update(
            {
                "original_item_count": compact_result.original_item_count,
                "retained_item_count": compact_result.retained_item_count,
                "summary": compact_result.display_text(),
            }
        )
        if compact_result.pruned_tool_results:
            completed_payload["pruned_tool_results"] = compact_result.pruned_tool_results
        self._emit(
            "auto_compact_completed",
            turn_id,
            **completed_payload,
        )

    def _build_follow_up_messages(
        self,
        tool_results: 'typing.List[ToolResult]',
    ) -> 'typing.List[UserMessage]':
        follow_ups: 'typing.List[UserMessage]' = []
        for result in tool_results:
            statuses = None
            if (
                result.name == "wait_agent"
                and not result.is_error
                and isinstance(result.output, dict)
            ):
                statuses = result.output.get("status")
            if isinstance(statuses, dict):
                for agent_id, status in statuses.items():
                    if isinstance(agent_id, str) and isinstance(status, dict):
                        payload = {
                            "agent_id": agent_id,
                            "status": status,
                        }
                        follow_ups.append(
                            UserMessage(
                                text=(
                                    "<subagent_notification>\n"
                                    f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
                                    "</subagent_notification>"
                                )
                            )
                        )
        return follow_ups


def _usage_from_context_length_error(
    message: 'str',
) -> 'typing.Union[typing.Dict[str, int], None]':
    if not _is_context_length_error_message(message):
        return None

    requested_match = _REQUESTED_TOKENS_RE.search(message)
    if requested_match is None:
        return None

    usage = {"total_tokens": _parse_token_count(requested_match.group(1))}
    split_match = _REQUESTED_TOKEN_SPLIT_RE.search(message)
    if split_match is not None:
        usage["input_tokens"] = _parse_token_count(split_match.group(1))
        usage["output_tokens"] = _parse_token_count(split_match.group(2))
    else:
        usage["input_tokens"] = usage["total_tokens"]
    return usage


def _is_context_length_error_message(message: 'str') -> 'bool':
    lower = message.lower()
    return any(marker in lower for marker in _CONTEXT_LENGTH_ERROR_MARKERS)


def _context_length_error_token_limit(message: 'str') -> 'typing.Union[int, None]':
    limit_match = _MAX_CONTEXT_TOKENS_RE.search(message)
    if limit_match is None:
        return None
    return _parse_token_count(limit_match.group(1))


def _parse_token_count(value: 'str') -> 'int':
    return int(value.replace(",", ""))
