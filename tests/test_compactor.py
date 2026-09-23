import json

import pytest

from pycodex.protocol import (
    AssistantMessage,
    ContextMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from pycodex.utils.compactor import (
    DEFAULT_COMPACT_PROMPT,
    SUMMARY_PREFIX,
    compact,
    prune_oldest_tool_response,
)
from pycodex.utils.session_persist import (
    conversation_history_to_turns,
    load_resumed_session_path,
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
    assert compact(history) == (ContextMessage(SUMMARY_PREFIX + "\nnew summary"),)
    assert (
        compact(history)[0].serialize()
        == UserMessage(SUMMARY_PREFIX + "\nnew summary").serialize()
    )
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


def test_compact_summary_is_context_but_identical_user_input_is_visible():
    summary = compact([AssistantMessage("summary")])
    assert conversation_history_to_turns(summary) == ()
    assert conversation_history_to_turns(
        summary + (AssistantMessage("continued answer"), UserMessage(summary[0].text))
    ) == (("", "continued answer"), (summary[0].text, ""))


@pytest.mark.parametrize("has_summary_metadata", [False, True])
def test_resume_summary_checkpoint_keeps_follow_up_items(
    tmp_path, has_summary_metadata
):
    text = (
        "Upstream handoff with its own prefix"
        if has_summary_metadata
        else SUMMARY_PREFIX + "\nold local summary"
    )
    summary = UserMessage(text, id="summary-id").serialize()
    payload = {"replacement_history": [summary]}
    if has_summary_metadata:
        payload["message"] = text
    follow_up = (
        ToolCall("call", "echo", {}, raw_arguments="{}"),
        ToolResult("call", "echo", "output"),
        AssistantMessage("continued answer"),
    )
    entries = (
        [
            {"type": "session_meta", "payload": {"id": "saved-session"}},
            {"type": "compacted", "payload": payload},
        ]
        + [{"type": "response_item", "payload": item.serialize()} for item in follow_up]
        + [
            {"type": "event_msg", "payload": {"type": "user_message", "message": text}},
            {
                "type": "response_item",
                "payload": AssistantMessage("next answer").serialize(),
            },
        ]
    )
    path = tmp_path / "checkpoint.jsonl"
    path.write_text("\n".join(json.dumps(entry) for entry in entries), encoding="utf-8")

    resumed = load_resumed_session_path(path)

    assert resumed["history"] == (
        (ContextMessage(text, id="summary-id"),)
        + follow_up
        + (UserMessage(text), AssistantMessage("next answer"))
    )
    assert resumed["history"][0].serialize() == summary
    assert resumed["turns"] == (("", "continued answer"), (text, "next answer"))
