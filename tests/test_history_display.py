import json
from dataclasses import replace

import pytest

from pycodex import ContextConfig, ToolRegistry
from pycodex.protocol import (
    AgentMessage, AssistantMessage, ContextMessage, ToolResult, UserMessage,
)
from pycodex.utils import session_persist
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
    turns = runtime.snapshot()["history"]
    assert tuple(("\n".join(message.text for message in turn["messages"]), turn["response"])
                 for turn in turns) == expected
    assert load_display_history(recorder.rollout_path) == turns
    replies = [item for item in items if isinstance(item, AssistantMessage) and item.visible]
    assert turns[-1]["response_id"] == replies[-1].message_id
    assert turns[-1]["response_at"] == replies[-1].created_at


@pytest.mark.parametrize("message", [
    UserMessage("private", visible=False),
    mail(),
    AgentMessage("/root/helper", "/root", ({"type": "input_text", "text": "public"},),
                 visible=True),
    AssistantMessage("private", visible=False),
])
def test_visibility_round_trip_keeps_model_serialization_unchanged(tmp_path, message):
    for key in ("visible", "message_id", "created_at", "turn_id"):
        assert key not in message.serialize()
    recorder = SessionRolloutRecorder.create(
        tmp_path, "session", tmp_path, "test", None, "", tmp_path / "rollout.jsonl",
    )
    recorder.append_history_items((UserMessage("question"), message))
    restored = load_resumed_session_path(recorder.rollout_path)["history"][-1]
    assert restored == message
    assert (restored.message_id, restored.created_at) == (message.message_id, message.created_at)
    recorder.append_compacted_history((ContextMessage("summary"), message))
    restored = load_resumed_session_path(recorder.rollout_path)["history"][-1]
    assert restored == message
    assert (restored.message_id, restored.created_at) == (message.message_id, message.created_at)


def test_hidden_mail_does_not_consume_display_window_or_hide_compaction(tmp_path):
    recorder = SessionRolloutRecorder.create(
        tmp_path, "session", tmp_path, "test", None, "", tmp_path / "rollout.jsonl",
    )
    recorder.append_history_items((
        UserMessage("question"), AssistantMessage("answer", phase="final"),
    ) + (mail(),) * 100)
    recorder.append_compacted_history((ContextMessage("summary"),))
    recorder.append_history_items((AssistantMessage("continued", phase="final"),))
    turns = load_display_history(recorder.rollout_path, limit=3)
    assert [turn["kind"] for turn in turns] == ["assistant", "compaction", "assistant"]
    assert [turn["response"] for turn in turns] == ["answer", "", "continued"]
    assert turns[0]["messages"] == (UserMessage("question"),)
    assert turns[1]["messages"] == turns[2]["messages"] == ()
    # Visibility defaults also apply to records produced before the field existed.
    turns = conversation_history_to_turns((
        AgentMessage.from_response_item(mail().serialize()), AssistantMessage("reply"),
    ))
    assert len(turns) == 1
    assert turns[0]["messages"] == ()
    assert turns[0]["response"] == "reply"


def test_display_history_always_preserves_message_identity_and_creation_time(tmp_path):
    recorder = SessionRolloutRecorder.create(
        tmp_path, "session", tmp_path, "test", None, "", tmp_path / "rollout.jsonl",
    )
    question = UserMessage("question", name="Alice", message_id="input", turn_id="turn",
                           created_at="2026-01-01T00:00:00.000+00:00")
    answer = AssistantMessage("answer", id="upstream", message_id="reply", turn_id="turn",
                              created_at="2026-01-01T00:01:00.000+00:00", phase="final")
    recorder.append_history_items((question, answer))
    expected = ({
        "turn_id": "turn", "messages": (question,), "response": "answer",
        "response_id": "reply", "response_at": answer.created_at,
        "response_phase": "final", "kind": "assistant",
    },)
    assert load_display_history(recorder.rollout_path) == expected
    restored = load_resumed_session_path(recorder.rollout_path)["history"]
    assert restored[0].message_id == "input"
    assert restored[0].created_at == question.created_at
    assert restored[0].name == "Alice"
    assert restored[1].message_id == "reply"
    assert restored[1].id == "upstream"
    recorder.append_compacted_history((ContextMessage("summary"), question, answer))
    checkpoint = load_resumed_session_path(recorder.rollout_path)
    assert checkpoint["turns"] == expected
    assert checkpoint["history"][1].message_id == "input"
    assert checkpoint["history"][1].created_at == question.created_at


def test_incremental_projection_replaces_same_id_and_preserves_distinct_authors():
    queued = UserMessage("same", name="Alice")
    admitted = replace(queued, turn_id="turn")
    bob = UserMessage("same", name="Bob", turn_id="turn")
    previous = conversation_history_to_turns((admitted,))
    previous[0]["messages"] = (queued,)
    previous[0]["status"] = "queued"
    turns = conversation_history_to_turns((admitted, bob, admitted), previous)
    assert turns[0]["messages"] == (admitted, bob)
    assert [message.message_id for message in turns[0]["messages"]] == [queued.message_id, bob.message_id]
    assert turns[0]["messages"][0] is admitted
    assert turns[0]["messages"][0].created_at == queued.created_at
    assert turns[0]["messages"][0].turn_id == "turn"
    assert previous[0]["messages"] == (queued,)
    assert previous[0]["messages"][0].turn_id is None
    assert turns[0] is not previous[0]
    assert "status" not in turns[0]


@pytest.mark.parametrize("indent", [None, 2])
def test_legacy_ids_are_stable_for_repeated_text_and_unknown_times_stay_unknown(
    tmp_path, monkeypatch, indent,
):
    monkeypatch.setattr(session_persist, "ROLLOUT_READ_CHUNK_SIZE", 37)
    path = tmp_path / "legacy.jsonl"
    records = [{"type": "session_meta", "payload": {"id": "legacy"}}]
    for _ in range(2):
        records.extend([
            {"type": "event_msg", "payload": {"type": "user_message", "message": "重复", "name": "Alice"}},
            {"type": "response_item", "timestamp": "2020-01-01T00:00:00Z",
             "payload": AssistantMessage("same answer", phase="final").serialize()},
        ])
    original = "\n".join(json.dumps(record, ensure_ascii=False, indent=indent) for record in records)
    original += '\n{"unfinished":'
    path.write_text(original, encoding="utf-8")
    turns = load_display_history(path)
    resumed = load_resumed_session_path(path)
    assert turns == resumed["turns"] == load_display_history(path)
    assert load_display_history(path, limit=1) == turns[-1:]
    assert load_display_history(path, limit=None) == turns
    ids = [message.message_id for turn in turns for message in turn["messages"]]
    assert len(set(ids)) == 2
    assert all(value.startswith("legacy-") for value in ids)
    assert all(turn["messages"][0].created_at is None for turn in turns)
    assert all(turn["response_at"] == "2020-01-01T00:00:00Z" for turn in turns)
    assert len({turn["response_id"] for turn in turns}) == 2
    assert [item.message_id for item in resumed["history"] if isinstance(item, UserMessage)] == ids
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("limit", [0, 1, 2, 5])
def test_display_limit_reads_only_complete_tail_turns(tmp_path, monkeypatch, limit):
    path = tmp_path / "large.jsonl"
    recorder = SessionRolloutRecorder.create(tmp_path, "session", tmp_path, "test", None, "", path)
    recorder.append_history_items((
        UserMessage("old", turn_id="old"), ToolResult("old-call", "echo", "x" * (2 * 1024 * 1024)),
    ))
    prefix_end = path.stat().st_size
    recorder.append_history_items((AssistantMessage("old answer", turn_id="old", phase="final"),))
    for index in range(3):
        turn_id = "turn-{0}".format(index)
        recorder.append_history_items((
            UserMessage("first", name="Alice", turn_id=turn_id),
            AssistantMessage("working", turn_id=turn_id, phase="commentary"),
            UserMessage("steer", name="Bob", turn_id=turn_id),
            AssistantMessage("answer", turn_id=turn_id, phase="final"),
        ))
        if index < 2:
            recorder.append_compacted_history((ContextMessage("summary"),))
    expected = load_display_history(path, limit=None)
    reads = []
    original_mmap = session_persist.mmap.mmap

    class TrackedMap:
        def __init__(self, *args, **kwargs):
            self.mapped = original_mmap(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.mapped.close()

        def __len__(self):
            return len(self.mapped)

        def __getitem__(self, key):
            reads.append((key.start, key.stop))
            return self.mapped[key]

        def rfind(self, value, start, end):
            found = self.mapped.rfind(value, start, end)
            reads.append((found if found >= 0 else start, end))
            return found

    monkeypatch.setattr(session_persist.mmap, "mmap", TrackedMap)
    assert load_display_history(path, limit=limit) == (expected[-limit:] if limit else ())
    assert all(start >= prefix_end - 1 for start, _end in reads)
    assert sum(end - start for start, end in reads) < 64 * 1024


def test_display_limit_keeps_steer_grouping_across_compaction(tmp_path):
    path = tmp_path / "steer.jsonl"
    recorder = SessionRolloutRecorder.create(tmp_path, "session", tmp_path, "test", None, "", path)
    recorder.append_history_items((
        UserMessage("old", turn_id="old"), AssistantMessage("done", turn_id="old", phase="final"),
        UserMessage("first", turn_id="active"),
        AssistantMessage("working", turn_id="active", phase="commentary"),
    ))
    recorder.append_compacted_history((ContextMessage("summary"),))
    recorder.append_history_items((
        UserMessage("steer", turn_id="active"),
        AssistantMessage("answer", turn_id="active", phase="final"),
    ))
    expected = load_display_history(path, limit=None)
    assert [message.text for message in expected[1]["messages"]] == ["first", "steer"]
    for limit in (1, 2, 3):
        assert load_display_history(path, limit=limit) == expected[-limit:]
