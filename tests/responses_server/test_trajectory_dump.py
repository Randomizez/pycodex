import json

import pytest

from responses_server.trajectory_dump import TrajectoryDumpWriter


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls", "length"])
def test_dump_retains_finish_reason_after_usage_only_chunk(tmp_path, finish_reason):
    request = {"model": "test", "messages": [], "stream": True}
    usage = {"prompt_tokens": 2, "completion_tokens": 3}
    chunks = [
        {
            "prompt_token_ids": [11, 12],
            "choices": [{"token_ids": [21], "finish_reason": None}],
        },
        {"choices": [{"token_ids": [22, 23], "finish_reason": finish_reason}]},
        {"choices": [], "usage": usage},
    ]
    writer = TrajectoryDumpWriter(str(tmp_path))
    assert list(writer.wrap_stream(iter(chunks), request)) == chunks
    record = json.loads((tmp_path / "dump.jsonl").read_text(encoding="utf-8"))
    assert record["request"] == request
    assert record["usage"] == usage
    assert record["finish_reason"] == finish_reason
    assert record["tokens"] == {"prefill": [11, 12], "decode": [21, 22, 23]}
    assert record["stream_completed"] is True
    assert record["stream_error_type"] is None


def test_interrupted_stream_preserves_partial_dump_and_raises(tmp_path):
    def stream():
        yield {
            "prompt_token_ids": [11, 12],
            "choices": [{"token_ids": [21], "finish_reason": None}],
        }
        raise RuntimeError("stream disconnected")

    writer = TrajectoryDumpWriter(str(tmp_path))
    with pytest.raises(RuntimeError, match="stream disconnected"):
        list(writer.wrap_stream(stream(), {"model": "test"}))
    record = json.loads((tmp_path / "dump.jsonl").read_text(encoding="utf-8"))
    assert record["finish_reason"] is None
    assert record["usage"] == {}
    assert record["tokens"] == {"prefill": [11, 12], "decode": [21]}
    assert record["stream_completed"] is False
    assert record["stream_error_type"] == "RuntimeError"


def test_failure_before_first_chunk_preserves_empty_attempt(tmp_path):
    def stream():
        raise OSError("private-backend-error")
        yield

    writer = TrajectoryDumpWriter(str(tmp_path))
    with pytest.raises(OSError, match="private-backend-error"):
        list(writer.wrap_stream(stream(), {"model": "test"}))
    text = (tmp_path / "dump.jsonl").read_text(encoding="utf-8")
    record = json.loads(text)
    assert record["tokens"] == {"prefill": [], "decode": []}
    assert record["usage"] == {}
    assert record["stream_completed"] is False
    assert record["stream_error_type"] == "OSError"
    assert "private-backend-error" not in text


def test_closed_stream_is_not_marked_completed(tmp_path):
    writer = TrajectoryDumpWriter(str(tmp_path))
    stream = writer.wrap_stream(
        iter([{"prompt_token_ids": [11], "choices": []}]), {"model": "test"}
    )
    next(stream)
    stream.close()
    record = json.loads((tmp_path / "dump.jsonl").read_text(encoding="utf-8"))
    assert record["tokens"] == {"prefill": [11], "decode": []}
    assert record["stream_completed"] is False
    assert record["stream_error_type"] is None
