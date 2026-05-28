from dataclasses import dataclass

from ..protocol import (
    AssistantMessage,
    ConversationItem,
    ModelStreamEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)
from .random_ids import uuid7_string
import typing

if typing.TYPE_CHECKING:
    from ..agent import Agent

DEFAULT_COMPACT_PROMPT = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work."""

SUMMARY_PREFIX = (
    "Another language model started to solve this problem and produced a summary "
    "of its thinking process. You also have access to the state of the tools "
    "that were used by that language model. Use this to build on the work that "
    "has already been done and avoid duplicating work. Here is the summary "
    "produced by the other language model, use the information in this summary "
    "to assist with your own analysis:"
)

COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000
_APPROX_CHARS_PER_TOKEN = 4
_SUBAGENT_NOTIFICATION_PREFIX = "<subagent_notification>\n"


@dataclass(frozen=True)
class CompactResult:
    history: 'typing.Tuple[ConversationItem, ...]'
    original_item_count: 'int'
    pruned_tool_results: 'int' = 0

    @property
    def retained_item_count(self) -> 'int':
        return max(len(self.history) - 1, 0)

    def display_text(self) -> 'str':
        retained_label = _pluralize("item", self.retained_item_count)
        original_label = _pluralize("item", self.original_item_count)
        text = (
            f"compact({self.original_item_count} {original_label}) -> "
            f"{self.retained_item_count} {retained_label} + [summary]"
        )
        if self.pruned_tool_results:
            tool_label = _pluralize("tool response", self.pruned_tool_results)
            text += f" (dropped {self.pruned_tool_results} old {tool_label})"
        return text


def compact(
    history: 'typing.Sequence[ConversationItem]',
) -> 'typing.Tuple[ConversationItem, ...]':
    summary_text = _build_summary_message(_last_assistant_message(history))
    user_messages = collect_user_messages(history)
    return build_compacted_history(user_messages, summary_text)


async def compact_agent(
    agent: 'Agent',
    stream_event_handler: 'typing.Union[typing.Callable[[ModelStreamEvent], None], None]' = None,
    prune_tool_results_on_context_error: 'bool' = False,
) -> 'typing.Union[CompactResult, None]':
    history = agent.history
    if not history:
        return None
    original_item_count = len(history)
    pruned_tool_results = 0

    noop_stream_event_handler = lambda _event: None
    while True:
        compact_prompt = UserMessage(text=DEFAULT_COMPACT_PROMPT)
        prompt = agent._context_manager.build_prompt(
            list(history) + [compact_prompt],
            [],
            False,
            turn_id=uuid7_string(),
        )
        try:
            response = await agent._model_client.complete(
                prompt,
                stream_event_handler or noop_stream_event_handler,
            )
            break
        except Exception as exc:
            if (
                not prune_tool_results_on_context_error
                or not _is_context_length_error(str(exc))
            ):
                raise
            pruned_history = prune_oldest_tool_response(history)
            if pruned_history is None:
                raise
            history = pruned_history
            pruned_tool_results += 1
            agent.replace_history(history)

    compacted_history = compact(
        list(history) + [compact_prompt] + list(response.items)
    )
    agent.replace_history(compacted_history)
    rollout_recorder = agent._rollout_recorder
    if rollout_recorder is not None:
        rollout_recorder.append_compacted_history(compacted_history)
    return CompactResult(
        history=compacted_history,
        original_item_count=original_item_count,
        pruned_tool_results=pruned_tool_results,
    )


def prune_oldest_tool_response(
    history: 'typing.Sequence[ConversationItem]',
) -> 'typing.Union[typing.Tuple[ConversationItem, ...], None]':
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


def collect_user_messages(
    history: 'typing.Sequence[ConversationItem]',
) -> 'typing.Tuple[str, ...]':
    compact_prompt = _normalize_for_compare(DEFAULT_COMPACT_PROMPT)
    collected: 'typing.List[str]' = []
    for item in history:
        if not isinstance(item, UserMessage):
            continue
        if is_summary_message(item.text):
            continue
        if _normalize_for_compare(item.text) == compact_prompt:
            continue
        if _is_synthetic_user_message(item.text):
            continue
        collected.append(item.text)
    return tuple(collected)


def is_summary_message(message: 'str') -> 'bool':
    return message.startswith(f"{SUMMARY_PREFIX}\n")


def build_compacted_history(
    user_messages: 'typing.Sequence[str]',
    summary_text: 'str',
    max_tokens: 'int' = COMPACT_USER_MESSAGE_MAX_TOKENS,
) -> 'typing.Tuple[ConversationItem, ...]':
    selected_messages: 'typing.List[str]' = []
    if max_tokens > 0:
        remaining = max_tokens
        for message in reversed(tuple(user_messages)):
            if remaining <= 0:
                break
            tokens = _approx_token_count(message)
            if tokens <= remaining:
                selected_messages.append(message)
                remaining -= tokens
                continue
            selected_messages.append(_truncate_text_to_tokens(message, remaining))
            break
        selected_messages.reverse()

    compacted: 'typing.List[ConversationItem]' = [
        UserMessage(text=message) for message in selected_messages
    ]
    compacted.append(UserMessage(text=summary_text or _build_summary_message(None)))
    return tuple(compacted)


def _last_assistant_message(
    history: 'typing.Sequence[ConversationItem]',
) -> 'typing.Union[str, None]':
    for item in reversed(tuple(history)):
        if isinstance(item, AssistantMessage):
            return item.text
    return None


def _build_summary_message(summary_text: 'typing.Union[str, None]') -> 'str':
    normalized = (summary_text or "").strip() or "(no summary available)"
    return f"{SUMMARY_PREFIX}\n{normalized}"


def _approx_token_count(text: 'str') -> 'int':
    if not text:
        return 0
    return max(1, (len(text) + _APPROX_CHARS_PER_TOKEN - 1) // _APPROX_CHARS_PER_TOKEN)


def _truncate_text_to_tokens(text: 'str', max_tokens: 'int') -> 'str':
    if max_tokens <= 0:
        return ""
    max_chars = max(max_tokens, 1) * _APPROX_CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text

    removed_tokens = _approx_token_count(text[max_chars:])
    suffix = f"\n...[{removed_tokens} tokens truncated]..."
    available = max_chars - len(suffix)
    if available <= 0:
        return suffix.lstrip()
    return text[:available].rstrip() + suffix


def _normalize_for_compare(text: 'str') -> 'str':
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def _pluralize(noun: 'str', count: 'int') -> 'str':
    if count == 1:
        return noun
    return f"{noun}s"


def _is_synthetic_user_message(text: 'str') -> 'bool':
    return text.startswith(_SUBAGENT_NOTIFICATION_PREFIX)


def _is_context_length_error(message: 'str') -> 'bool':
    lower = message.lower()
    return (
        "context_length_exceeded" in lower
        or "maximum context length" in lower
        or "exceeds the context window" in lower
        or "exceeded the context window" in lower
    )
