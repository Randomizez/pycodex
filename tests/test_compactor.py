from pycodex.protocol import AssistantMessage, ToolCall, ToolResult, UserMessage
from pycodex.utils.compactor import (
    DEFAULT_COMPACT_PROMPT,
    SUMMARY_PREFIX,
    compact,
    prune_oldest_tool_response,
)


def test_compact_replaces_history_with_user_messages_and_summary() -> 'None':
    history = (
        UserMessage(text="first user"),
        AssistantMessage(text="first assistant"),
        UserMessage(text="second user"),
        UserMessage(text=DEFAULT_COMPACT_PROMPT),
        AssistantMessage(text="checkpoint summary"),
    )

    compacted = compact(history)

    assert [type(item).__name__ for item in compacted] == [
        "UserMessage",
        "UserMessage",
        "UserMessage",
    ]
    assert compacted[0].text == "first user"
    assert compacted[1].text == "second user"
    assert compacted[2].text == f"{SUMMARY_PREFIX}\ncheckpoint summary"


def test_compact_filters_previous_summary_messages() -> 'None':
    history = (
        UserMessage(text="real user"),
        UserMessage(text=f"{SUMMARY_PREFIX}\nold summary"),
        UserMessage(text=DEFAULT_COMPACT_PROMPT),
        AssistantMessage(text="new summary"),
    )

    compacted = compact(history)

    assert [item.text for item in compacted] == [
        "real user",
        f"{SUMMARY_PREFIX}\nnew summary",
    ]


def test_compact_filters_synthetic_subagent_notifications() -> 'None':
    history = (
        UserMessage(text="real user"),
        UserMessage(
            text=(
                "<subagent_notification>\n"
                '{"agent_id":"agent_1","status":{"completed":"done"}}\n'
                "</subagent_notification>"
            )
        ),
        AssistantMessage(text="new summary"),
    )

    compacted = compact(history)

    assert [item.text for item in compacted] == [
        "real user",
        f"{SUMMARY_PREFIX}\nnew summary",
    ]


def test_compact_filters_synthetic_exec_completion_notifications() -> 'None':
    history = (
        UserMessage(text="real user"),
        UserMessage(
            text=(
                "<exec_command_completed>\n"
                '{"session_id":1000,"exit_code":0,"command":"sleep 1"}\n'
                "</exec_command_completed>"
            )
        ),
        AssistantMessage(text="new summary"),
    )

    compacted = compact(history)

    assert [item.text for item in compacted] == [
        "real user",
        f"{SUMMARY_PREFIX}\nnew summary",
    ]


def test_prune_oldest_tool_response_removes_matching_call_pair() -> 'None':
    history = (
        UserMessage(text="first"),
        ToolCall(call_id="call_1", name="echo", arguments={}),
        ToolResult(call_id="call_1", name="echo", output="old large output"),
        AssistantMessage(text="after first"),
        ToolCall(call_id="call_2", name="echo", arguments={}),
        ToolResult(call_id="call_2", name="echo", output="new output"),
    )

    pruned = prune_oldest_tool_response(history)

    assert pruned is not None
    assert [type(item).__name__ for item in pruned] == [
        "UserMessage",
        "AssistantMessage",
        "ToolCall",
        "ToolResult",
    ]
    assert isinstance(pruned[2], ToolCall)
    assert pruned[2].call_id == "call_2"
    assert isinstance(pruned[3], ToolResult)
    assert pruned[3].call_id == "call_2"
