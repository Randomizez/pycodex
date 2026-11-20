import typing
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from ..protocol import (
    AssistantMessage,
    ContextMessage,
    ConversationItem,
    ToolCall,
    ToolResult,
)

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

    @classmethod
    def from_response(cls, response, session_file, original_item_count, pruned_tool_results):
        summary = _last_assistant_message(response.items)
        if summary is None or not summary.strip():
            raise ValueError("compact response contains no assistant summary")
        return cls(compact(response.items, session_file), original_item_count, pruned_tool_results)


def compact(
    history: "typing.Sequence[ConversationItem]",
    session_file: "typing.Union[str, Path, None]" = None,
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


def prune_oldest_tool_response(
    history: "typing.Sequence[ConversationItem]", count: "int" = 1,
) -> "typing.Optional[typing.Sequence[ConversationItem]]":
    """Drop the oldest count results together with their matching calls."""
    if count == 0:
        return history
    calls = defaultdict(deque)
    removed = set()
    for index, item in enumerate(history):
        if isinstance(item, ToolCall):
            calls[item.call_id].append(index)
        elif isinstance(item, ToolResult):
            removed.add(index)
            if calls[item.call_id]:
                removed.add(calls[item.call_id].popleft())
            count -= 1
            if count == 0:
                return tuple(item for i, item in enumerate(history) if i not in removed)
    return None


def build_compacted_history(
    summary_text: "str",
) -> "typing.Tuple[ConversationItem, ...]":
    return (ContextMessage(text=summary_text or _build_summary_message(None)),)


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
