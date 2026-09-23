import typing
from dataclasses import dataclass

from ..protocol import (
    AssistantMessage,
    ConversationItem,
    ModelResponse,
    ToolCall,
    ToolResult,
    UserMessage,
)
from .random_ids import uuid7_string

if typing.TYPE_CHECKING:
    from ..context import ContextManager
    from ..events import ModelEvent
    from ..model import ModelClient

DEFAULT_COMPACT_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue
- Preserve concise verbatim excerpts of the latest user request and active constraints in their original language, and write the summary in the user's primary language.

Be concise, structured, and focused on helping the next LLM seamlessly continue the work."""

SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary "
    "of its thinking process. You also have access to the state of the tools "
    "that were used by that language model. Use this to build on the work that "
    "has already been done and avoid duplicating work. Here is the summary "
    "produced by the other language model, use the information in this summary "
    "to assist with your own analysis. Continue the current task directly; do "
    "not merely repeat or acknowledge the handoff:"
)


@dataclass(frozen=True)
class CompactResult:
    history: "typing.Tuple[ConversationItem, ...]"
    original_item_count: "int"
    pruned_tool_results: "int" = 0

    @property
    def retained_item_count(self) -> "int":
        return max(len(self.history) - 1, 0)


def compact(
    history: "typing.Sequence[ConversationItem]",
    session_file: "typing.Union[str, None]" = None,
) -> "typing.Tuple[ConversationItem, ...]":
    summary_text = _build_summary_message(_last_assistant_message(history))
    if session_file:
        summary_text += (
            f"\n\nCurrent session file: {session_file}\n"
            "If you need details from earlier conversation, inspect this file "
            "selectively with targeted searches and bounded reads (for example, "
            "`rg -n` followed by `sed -n` or `tail`). Do not read the entire "
            "file at once."
        )
    return build_compacted_history(summary_text)


async def compact_history(
    history: "typing.Sequence[ConversationItem]",
    model_client: "ModelClient",
    context_manager: "ContextManager",
    stream_event_handler: "typing.Union[typing.Callable[[ModelEvent], None], None]" = None,
    prune_tool_results_on_context_error: "bool" = False,
    session_file: "typing.Union[str, None]" = None,
    turn_id: "typing.Union[str, None]" = None,
) -> "typing.Union[CompactResult, None]":
    from ..model import ContextLengthExceeded, ResponsesIncompleteError

    if not history:
        return None
    original_item_count = len(history)
    pruned_tool_results = 0
    turn_id = turn_id or uuid7_string()

    noop_stream_event_handler = lambda _event: None
    while True:
        compact_prompt = UserMessage(text=DEFAULT_COMPACT_PROMPT)
        prompt = context_manager.build_prompt(
            list(history) + [compact_prompt],
            [],
            False,
            turn_id=turn_id,
        )
        try:
            response = await model_client.complete(
                prompt,
                stream_event_handler or noop_stream_event_handler,
            )
            break
        except ResponsesIncompleteError as exc:
            if (
                exc.reason != "max_output_tokens"
                or _last_assistant_message(exc.partial_items) is None
            ):
                raise
            response = ModelResponse(items=list(exc.partial_items))
            break
        except ContextLengthExceeded:
            if not prune_tool_results_on_context_error:
                raise
            pruned_history = prune_oldest_tool_response(history)
            if pruned_history is None:
                raise
            history = pruned_history
            pruned_tool_results += 1

    summary = _last_assistant_message(response.items)
    if summary is None or not summary.strip():
        raise ValueError("compact response contains no assistant summary")
    compacted_history = compact(
        response.items,
        session_file,
    )
    return CompactResult(
        history=compacted_history,
        original_item_count=original_item_count,
        pruned_tool_results=pruned_tool_results,
    )


def prune_oldest_tool_response(
    history: "typing.Sequence[ConversationItem]",
) -> "typing.Union[typing.Tuple[ConversationItem, ...], None]":
    items = list(history)
    tool_result_index = None
    call_id = None
    for index, item in enumerate(items):
        if isinstance(item, ToolResult):
            tool_result_index = index
            call_id = item.call_id
            break
    if tool_result_index is None:
        return None

    indexes_to_remove = {tool_result_index}
    for index, item in enumerate(items[:tool_result_index]):
        if isinstance(item, ToolCall) and item.call_id == call_id:
            indexes_to_remove.add(index)
            break

    return tuple(
        item for index, item in enumerate(items) if index not in indexes_to_remove
    )


def build_compacted_history(
    summary_text: "str",
) -> "typing.Tuple[ConversationItem, ...]":
    return (UserMessage(text=summary_text or _build_summary_message(None)),)


def _last_assistant_message(
    history: "typing.Sequence[ConversationItem]",
) -> "typing.Union[str, None]":
    for item in reversed(tuple(history)):
        if isinstance(item, AssistantMessage):
            return item.text
    return None


def _build_summary_message(summary_text: "typing.Union[str, None]") -> "str":
    normalized = (summary_text or "").strip() or "(no summary available)"
    return f"{SUMMARY_PREFIX}\n{normalized}"
