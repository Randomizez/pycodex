from pycodex.protocol import AssistantMessage, ToolCall, ToolResult, UserMessage
from pycodex.utils.compactor import (
    DEFAULT_COMPACT_PROMPT,
    SUMMARY_PREFIX,
    compact,
    prune_oldest_tool_response,
)


def test_compact_replaces_all_prior_context_with_language_preserving_summary():
    history = (
        UserMessage("real user"),
        AssistantMessage("old answer"),
        UserMessage(SUMMARY_PREFIX + "\nold summary"),
        UserMessage('<subagent_notification>{"status":"done"}</subagent_notification>'),
        UserMessage('<exec_command_completed>{"exit_code":0}</exec_command_completed>'),
        UserMessage(DEFAULT_COMPACT_PROMPT),
        AssistantMessage("new summary"),
    )
    assert compact(history) == (UserMessage(SUMMARY_PREFIX + "\nnew summary"),)
    assert "Continue the current task directly" in SUMMARY_PREFIX
    for instruction in (
        "concise verbatim excerpts",
        "in their original language",
        "user's primary language",
    ):
        assert instruction in DEFAULT_COMPACT_PROMPT


def test_prune_oldest_tool_response_keeps_other_pairs_intact():
    history = (
        UserMessage("first"),
        ToolCall("old", "echo", {}),
        ToolResult("old", "echo", "large"),
        AssistantMessage("after first"),
        ToolCall("new", "echo", {}),
        ToolResult("new", "echo", "recent"),
    )
    assert prune_oldest_tool_response(history) == (history[0],) + history[3:]
