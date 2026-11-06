from datetime import datetime

import pytest

from pycodex import ContextConfig, ToolRegistry
from pycodex.protocol import (
    AgentMessage, AssistantMessage, ContextMessage, UserMessage,
)
from pycodex.utils.history import conversation_history_to_turns
from pycodex.utils.session_persist import (
    SessionRolloutRecorder, load_display_history, load_resumed_session_path,
)
from tests.fakes import ScriptedModelClient, make_agent, make_runtime


def mail():
    return AgentMessage("/root/helper", "/root", (
        {"type": "input_text", "text": "internal report"},
    ))


@pytest.mark.parametrize("items,expected", [
    (
        (UserMessage("question"), AssistantMessage("working", phase="commentary"),
         mail(), mail(), AssistantMessage("answer", phase="final")),
        (("question", "answer"),),
    ),
    (
        (mail(), AssistantMessage("first"), mail(), AssistantMessage("second")),
        (("", "first"), ("", "second")),
    ),
    (
        (UserMessage("hidden", visible=False), AssistantMessage("answer", phase="final"),
         AssistantMessage("private", visible=False)),
        (("", "answer"),),
    ),
])
def test_runtime_history_and_resume_use_same_projection(tmp_path, items, expected):
    recorder = SessionRolloutRecorder.create(
        tmp_path, "session", tmp_path, "test", None, "", tmp_path / "rollout.jsonl",
    )
    recorder.append_history_items(items)
    runtime = make_runtime(make_agent(
        ScriptedModelClient([]), ToolRegistry(), ContextConfig(), initial_history=items,
    ))
    assert runtime.snapshot()["history"] == expected
    assert load_display_history(recorder.rollout_path) == expected


@pytest.mark.parametrize("message", [
    UserMessage("private", visible=False),
    mail(),
    AgentMessage("/root/helper", "/root", ({"type": "input_text", "text": "public"},),
                 visible=True),
    AssistantMessage("private", visible=False),
])
def test_visibility_round_trip_keeps_model_serialization_unchanged(tmp_path, message):
    assert "visible" not in message.serialize()
    recorder = SessionRolloutRecorder.create(
        tmp_path, "session", tmp_path, "test", None, "", tmp_path / "rollout.jsonl",
    )
    recorder.append_history_items((UserMessage("question"), message))
    assert load_resumed_session_path(recorder.rollout_path)["history"][-1] == message
    recorder.append_compacted_history((ContextMessage("summary"), message))
    assert load_resumed_session_path(recorder.rollout_path)["history"][-1] == message


def test_hidden_mail_does_not_consume_display_window_or_hide_compaction(tmp_path):
    recorder = SessionRolloutRecorder.create(
        tmp_path, "session", tmp_path, "test", None, "", tmp_path / "rollout.jsonl",
    )
    recorder.append_history_items((
        UserMessage("question"), AssistantMessage("answer", phase="final"),
    ) + (mail(),) * 100)
    recorder.append_compacted_history((ContextMessage("summary"),))
    recorder.append_history_items((AssistantMessage("continued", phase="final"),))
    assert load_display_history(recorder.rollout_path, limit=3) == (
        ("question", "answer"), None, ("", "continued"),
    )
    # Visibility defaults also apply to records produced before the field existed.
    assert conversation_history_to_turns((
        AgentMessage.from_response_item(mail().serialize()), AssistantMessage("reply"),
    )) == (("", "reply"),)


def test_display_history_can_include_record_times(tmp_path):
    recorder = SessionRolloutRecorder.create(
        tmp_path, "session", tmp_path, "test", None, "", tmp_path / "rollout.jsonl",
    )
    recorder.append_history_items((
        UserMessage("question"), AssistantMessage("working", phase="commentary"),
        mail(), AssistantMessage("answer", phase="final"),
    ))
    recorder.append_compacted_history((ContextMessage("summary"),))
    recorder.append_history_items((AssistantMessage("continued", phase="final"),))
    plain = load_display_history(recorder.rollout_path)
    timed = load_display_history(recorder.rollout_path, with_times=True)
    assert plain == (("question", "answer"), None, ("", "continued"))
    assert [None if turn is None else turn[:2] for turn in timed] == list(plain)
    (prompt, _, prompt_at, response_at), _, (_, _, no_prompt_at, continued_at) = timed
    assert prompt == "question"
    assert no_prompt_at is None
    for value in (prompt_at, response_at, continued_at):
        assert datetime.fromisoformat(value).tzinfo is not None
    assert prompt_at <= response_at <= continued_at
