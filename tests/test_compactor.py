from pycodex.protocol import AssistantMessage, UserMessage
from pycodex.utils.compactor import (
    DEFAULT_COMPACT_PROMPT,
    SUMMARY_PREFIX,
    compact,
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
