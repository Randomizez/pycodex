
import asyncio
import json
from typing import Callable

from .context import ContextManager
from .model import ModelClient
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
from .tools import ToolContext, ToolRegistry
from .utils import uuid7_string
import typing

if typing.TYPE_CHECKING:
    from .utils.session_persist import SessionRolloutRecorder


EventHandler = Callable[[AgentEvent], None]
NOOP_EVENT_HANDLER: 'EventHandler' = lambda _event: None


class TurnInterrupted(RuntimeError):
    pass


class AgentLoop:
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
        event_handler: 'EventHandler' = NOOP_EVENT_HANDLER,
        initial_history: 'typing.Tuple[ConversationItem, ...]' = (),
        rollout_recorder: 'typing.Union[SessionRolloutRecorder, None]' = None,
    ) -> 'None':
        self._model_client = model_client
        self._tool_registry = tool_registry
        self._context_manager = context_manager or ContextManager()
        self._parallel_tool_calls = parallel_tool_calls
        self._event_handler = event_handler
        self._history: 'typing.List[ConversationItem]' = list(initial_history)
        self._rollout_recorder = rollout_recorder
        self.interrupt_asap = False

    @property
    def history(self) -> 'typing.Tuple[ConversationItem, ...]':
        return tuple(self._history)

    def set_event_handler(
        self, event_handler: 'EventHandler' = NOOP_EVENT_HANDLER
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
        turn_id = turn_id or uuid7_string()
        self.interrupt_asap = False
        new_user_messages = [UserMessage(text=text) for text in texts]
        self._history.extend(new_user_messages)
        self._persist_history_items(new_user_messages)

        self._emit(
            "turn_started",
            turn_id,
            user_text="\n".join(texts),
            user_texts=list(texts),
        )

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
                iteration += 1
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
                response = await self._model_client.complete(
                    prompt,
                    lambda event: self._handle_model_stream_event(turn_id, event),
                )
                final_response_items = tuple(response.items)
                self._emit(
                    "model_completed",
                    turn_id,
                    iteration=iteration,
                    item_count=len(response.items),
                )

                tool_calls: 'typing.List[ToolCall]' = []
                persisted_response_items: 'typing.List[ConversationItem]' = []
                for item in response.items:
                    self._history.append(item)
                    persisted_response_items.append(item)
                    if isinstance(item, AssistantMessage):
                        last_assistant_message = item.text
                    elif isinstance(item, ToolCall):
                        tool_calls.append(item)
                self._persist_history_items(persisted_response_items)

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
            raise
        except Exception as exc:
            self._emit(
                "turn_failed",
                turn_id,
                iteration=iteration,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise

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
        self._event_handler(
            AgentEvent(kind=kind, turn_id=turn_id, payload=dict(payload))
        )

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

    def _handle_model_stream_event(self, turn_id: 'str', event: 'ModelStreamEvent') -> 'None':
        if event.kind == "assistant_delta":
            self._emit("assistant_delta", turn_id, **event.payload)
        elif event.kind == "tool_call":
            self._emit("tool_called", turn_id, **event.payload)
        elif event.kind == "token_count":
            self._emit("token_count", turn_id, **event.payload)
        elif event.kind == "stream_error":
            self._emit("stream_error", turn_id, **event.payload)

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
