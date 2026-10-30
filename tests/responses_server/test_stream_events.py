import json
from copy import deepcopy

import pytest
from starlette.testclient import TestClient

from responses_server import CompatServerConfig, ManagedResponseServer, StreamRouter
from responses_server.session_store import StoredResponse
from responses_server.request_translation import ToolNameMap, build_outcomming_request
from responses_server.transport import OutcommingChatError
from tests.responses_server.fake_chat_completions_server import (
    CaptureStore,
    build_messages_server,
    build_messages_text_events,
    build_test_server,
    build_text_chunks,
)


class ScriptedTransport:
    def __init__(self, streams):
        self.streams = iter(streams)
        self.requests = []

    def open_stream(self, request):
        self.requests.append(deepcopy(request))
        return iter(next(self.streams))


def _chunk(**delta):
    return {"choices": [{"index": 0, "delta": delta}]}


def _call(index, name, arguments):
    return {
        "index": index,
        "id": "call_%d" % index,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _stream(transport, tools=None):
    names = ToolNameMap()
    request = build_outcomming_request({
        "model": "test", "input": [], "tools": tools or [],
    }, names)
    return StreamRouter(transport).route_stream(StoredResponse("resp_test", None, "test", 0), request, names)


def _check_lifecycle(events, allow_abandoned=False):
    """Act like a sequential client; every delta must have a live parent item."""
    active = None
    indexes = {}
    added = {}
    deltas = {}
    completed = []
    text_parts = set()
    for event in events:
        kind = event["type"]
        if kind == "response.output_item.added":
            assert active is None or allow_abandoned
            item = event["item"]
            active = item["id"]
            assert active and active not in added
            assert item["status"] == "in_progress"
            assert event["output_index"] not in indexes.values()
            indexes[active] = event["output_index"]
            added[active] = item
            deltas[active] = []
        elif kind == "response.output_item.done":
            item = event["item"]
            assert item["id"] == active
            assert event["output_index"] == indexes[active]
            assert item["status"] == "completed"
            assert item["type"] == added[active]["type"]
            if item["type"] in ("message", "reasoning"):
                assert item["content"][0]["text"] == "".join(deltas[active])
            elif item["type"] in ("function_call", "custom_tool_call"):
                field = "arguments" if item["type"] == "function_call" else "input"
                assert added[active][field] == ""
                assert item[field] == "".join(deltas[active])
                assert added[active]["call_id"] == item["call_id"]
                assert added[active].get("namespace") == item.get("namespace")
            completed.append(item)
            active = None
        elif "item_id" in event:
            assert event["item_id"] == active
            assert event["output_index"] == indexes[active]
            if kind == "response.content_part.added":
                assert event["content_index"] == 0
                text_parts.add(active)
            elif kind.endswith(".delta"):
                if kind == "response.output_text.delta":
                    assert active in text_parts
                    assert event["content_index"] == 0
                if kind == "response.reasoning_text.delta":
                    assert added[active]["type"] == "reasoning"
                    assert event["content_index"] == 0
                deltas[active].append(event["delta"])
            elif kind.endswith("_text.done"):
                assert event["text"] == "".join(deltas[active])
            elif kind == "response.content_part.done":
                assert event["part"]["text"] == "".join(deltas[active])
        elif kind == "response.completed":
            assert active is None
            assert event["response"]["status"] == "completed"
            assert event["response"]["output"] == completed
    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.completed"
    if not allow_abandoned:
        assert len(added) == len(completed)
    return completed


def test_stream_preserves_segments_and_namespaced_tool_lifecycles():
    transport = ScriptedTransport([[
        _chunk(reasoning="First ", reasoning_content="First "),
        _chunk(reasoning_content="thought", content="First answer"),
        _chunk(reasoning="Second thought"),
        _chunk(content="Second ", tool_calls=[_call(0, "agents__run", '{"x":')]),
        _chunk(content="answer", tool_calls=[
            {"index": 0, "function": {"arguments": "1}"}},
            _call(1, "functions__apply_patch", '{"input":"*** Begin Patch\\n*** End Patch"}'),
        ]),
    ]])
    tools = [
        {"type": "namespace", "name": "agents", "tools": [
            {"type": "function", "name": "run", "parameters": {"type": "object"}},
        ]},
        {"type": "namespace", "name": "functions", "tools": [
            {"type": "custom", "name": "apply_patch"},
        ]},
    ]
    events = [payload for _, payload in _stream(transport, tools)]
    items = _check_lifecycle(events)
    assert [item["type"] for item in items] == [
        "reasoning", "message", "reasoning", "message", "function_call", "custom_tool_call",
    ]
    assert [item["content"][0]["text"] for item in items[:4]] == [
        "First thought", "First answer", "Second thought", "Second answer",
    ]
    assert items[-2]["name"] == "run" and items[-2]["namespace"] == "agents"
    assert items[-2]["arguments"] == '{"x":1}'
    assert items[-1]["namespace"] == "functions"
    assert items[-1]["input"] == "*** Begin Patch\n*** End Patch"


def test_deltas_arrive_before_reading_the_rest_of_the_backend_stream():
    consumed = []

    def chunks():
        yield _chunk(reasoning="thinking")
        consumed.append("read second chunk")
        yield _chunk(content="answer")
        consumed.append("read third chunk")
        yield _chunk(content="!")

    transport = ScriptedTransport([chunks()])
    events = []
    for _, event in _stream(transport):
        events.append(event)
        if event["type"] == "response.reasoning_text.delta":
            assert consumed == []
        if event["type"] == "response.output_text.delta" and event["delta"] == "answer":
            assert consumed == ["read second chunk"]
    _check_lifecycle(events)


def test_mock_search_followup_finishes_old_items_before_new_deltas():
    transport = ScriptedTransport([
        [
            _chunk(reasoning="Need search", content="Searching"),
            _chunk(tool_calls=[_call(0, "web_search", '{"query":"example"}')]),
        ],
        [_chunk(reasoning="Found it", content="Answer")],
    ])
    events = [payload for _, payload in _stream(transport, [{"type": "web_search"}])]
    items = _check_lifecycle(events)
    assert [item["type"] for item in items] == [
        "reasoning", "message", "web_search_call", "reasoning", "message",
    ]
    assert len(transport.requests) == 2
    assert transport.requests[1]["messages"][-1]["role"] == "tool"


@pytest.mark.parametrize("retry_reasoning", ["", "successful thought"])
def test_reasoning_retry_discards_unfinished_item_and_uses_fresh_identity(retry_reasoning):
    transport = ScriptedTransport([
        [_chunk(reasoning="discard this")],
        [_chunk(reasoning=retry_reasoning, content="answer")],
    ])
    events = [payload for _, payload in _stream(transport)]
    items = _check_lifecycle(events, allow_abandoned=True)
    abandoned = next(event["item"] for event in events if event["type"] == "response.output_item.added")
    assert abandoned["id"] not in [item["id"] for item in items]
    assert all("discard this" not in json.dumps(item) for item in items)
    assert len(transport.requests) == 2
    assert transport.requests[0] == transport.requests[1]


def test_invalid_parallel_call_never_finishes_any_tool():
    transport = ScriptedTransport([[
        _chunk(reasoning="thinking", content="calling tools"),
        _chunk(tool_calls=[_call(0, "one", "{}"), _call(1, "two", '{"invalid":')]),
    ]])
    events = []
    with pytest.raises(OutcommingChatError, match="invalid JSON"):
        for _, event in _stream(transport):
            events.append(event)
    assert not any(
        event.get("item", {}).get("type") == "function_call" for event in events
    )
    assert not any(event["type"] == "response.completed" for event in events)


@pytest.mark.parametrize("backend", ["chat_completions", "messages"])
def test_http_text_and_reasoning_lifecycle(tmp_path, backend):
    store = CaptureStore(tmp_path / "capture")
    if backend == "messages":
        chunks = build_messages_text_events("answer")
        for event in chunks[1:]:
            if "index" in event["data"]:
                event["data"]["index"] = 1
        chunks[1:1] = [
            {"event": "content_block_start", "data": {
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            }},
            {"event": "content_block_delta", "data": {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "thinking"},
            }},
            {"event": "content_block_stop", "data": {
                "type": "content_block_stop", "index": 0,
            }},
        ]
        server = build_messages_server(store, chunks)
    else:
        server = build_test_server(store, [
            _chunk(reasoning="thinking"), *build_text_chunks("answer"),
        ])
    server.start()
    app = ManagedResponseServer.build_app(CompatServerConfig(
        outcomming_base_url=server.base_url + "/v1", outcomming_api=backend,
    ))
    try:
        with TestClient(app) as client:
            response = client.post("/v1/responses", json={"model": "test", "input": []})
        assert response.status_code == 200
        events = [
            json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        items = _check_lifecycle(events)
        assert [item["content"][0]["text"] for item in items] == ["thinking", "answer"]
    finally:
        server.stop()
