import copy
import json

import pytest
from starlette.testclient import TestClient

from responses_server import CompatServerConfig, ManagedResponseServer
from responses_server.request_translation import build_outcomming_request
from responses_server.tools.custom_adapter import APPLY_PATCH_CHAT_DESCRIPTION
from tests.responses_server.fake_chat_completions_server import (
    CaptureStore,
    build_messages_server,
    build_messages_text_events,
    build_messages_tool_use_events,
    build_test_server,
    build_text_chunks,
    build_tool_call_chunks,
)


def _events(response):
    assert response.status_code == 200
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def _function(name):
    return {
        "type": "function",
        "name": name,
        "description": "Run the task.",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _namespace(name, tools, description=""):
    return {
        "type": "namespace", "name": name, "description": description, "tools": tools,
    }


def _items(events):
    return [
        event["item"] for event in events
        if event["type"] == "response.output_item.done"
    ]


@pytest.mark.parametrize("backend", ["chat_completions", "messages"])
@pytest.mark.parametrize(
    "namespace,name,kind",
    [
        ("multi_agent_v1", "spawn_agent", "function"),
        ("collaboration", "send_message", "function"),
        ("functions", "apply_patch", "custom"),
    ],
)
def test_namespace_tool_round_trip(tmp_path, backend, namespace, name, kind):
    alias = namespace + "__" + name
    value = "*** Begin Patch\n*** End Patch" if kind == "custom" else "inspect"
    arguments = json.dumps({"input" if kind == "custom" else "message": value})
    if kind == "function":
        tool = _function(name)
    else:
        tool = {"type": "custom", "name": name, "description": "Apply a patch."}
    store = CaptureStore(tmp_path / "capture")
    if backend == "messages":
        server = build_messages_server(store, [
            build_messages_tool_use_events(
                "call_ns", alias, [arguments[:8], arguments[8:]],
            ),
            build_messages_text_events("done"),
        ])
    else:
        server = build_test_server(store, [
            build_tool_call_chunks("call_ns", alias, [arguments[:8], arguments[8:]]),
            build_text_chunks("done"),
        ])
    server.start()
    app = ManagedResponseServer.build_app(CompatServerConfig(
        outcomming_base_url=server.base_url + "/v1",
        outcomming_api=backend,
    ))
    request = {
        "model": "step-test",
        "stream": True,
        "input": [{"type": "message", "role": "user", "content": "Run the task."}],
        "tools": [
            _function(name),
            _namespace("other", [_function(name)]),
            _namespace(namespace, [tool], "Tools for this workspace."),
        ],
        "tool_choice": {"type": kind, "namespace": namespace, "name": name},
    }
    original = copy.deepcopy(request)
    try:
        with TestClient(app) as client:
            events = _events(client.post("/v1/responses", json=request))
            assert events[-1]["type"] == "response.completed"
            items = _items(events)
            assert len(items) == 1
            call = items[0]
            expected = {
                "type": "custom_tool_call" if kind == "custom" else "function_call",
                "call_id": "call_ns",
                "namespace": namespace,
                "name": name,
                "id": call["id"],
                "status": "completed",
            }
            assert call["id"]
            added = next(
                event["item"] for event in events
                if event["type"] == "response.output_item.added"
            )
            assert added["id"] == call["id"]
            assert added["status"] == "in_progress"
            if kind == "custom":
                expected["input"] = value
            else:
                expected.update(arguments=arguments, encrypted_function_args=[])
            assert call == expected
            followup = copy.deepcopy(request)
            followup["input"] += [
                call,
                {
                    "type": call["type"] + "_output",
                    "call_id": "call_ns",
                    "output": "completed",
                },
            ]
            # History still needs its qualified name after this tool is removed.
            followup["tools"] = [_function("finish")]
            followup["tool_choice"] = "auto"
            events = _events(client.post("/v1/responses", json=followup))
            assert events[-1]["type"] == "response.completed"
    finally:
        server.stop()
    assert request == original
    captures = [
        json.loads(path.read_text())["body"]
        for path in sorted((tmp_path / "capture").glob("*_POST_*.json"))
    ]
    assert len(captures) == 2
    first, second = captures
    assert first["model"] == "step-test"
    definitions = first["tools"]
    if backend == "chat_completions":
        definitions = [entry["function"] for entry in definitions]
        assert first["tool_choice"] == {"type": "function", "function": {"name": alias}}
        historical_call = second["messages"][-2]["tool_calls"][0]
        assert historical_call["id"] == "call_ns"
        assert historical_call["function"]["name"] == alias
        assert json.loads(historical_call["function"]["arguments"]) == json.loads(arguments)
        assert second["messages"][-1] == {
            "role": "tool", "tool_call_id": "call_ns", "content": "completed",
        }
    else:
        assert first["tool_choice"] == {"type": "tool", "name": alias}
        assert second["messages"][-2]["content"] == [{
            "type": "tool_use", "id": "call_ns", "name": alias, "input": json.loads(arguments),
        }]
        assert second["messages"][-1]["content"] == [{
            "type": "tool_result", "tool_use_id": "call_ns", "content": "completed",
        }]
    assert [entry["name"] for entry in definitions] == [name, "other__" + name, alias]
    selected = definitions[-1]
    expected_description = (
        tool["description"] if kind == "function" else APPLY_PATCH_CHAT_DESCRIPTION
    )
    assert selected["description"] == "Tools for this workspace.\n\n" + expected_description
    if kind == "function":
        schema_key = "parameters" if backend == "chat_completions" else "input_schema"
        assert selected[schema_key] == tool["parameters"]
        if backend == "chat_completions":
            assert selected["strict"] is True


@pytest.mark.parametrize("backend", ["chat_completions", "messages"])
def test_namespace_mapping_survives_mock_search_and_is_request_local(tmp_path, backend):
    store = CaptureStore(tmp_path / "capture")
    calls = [
        ("search", "web_search", '{"query":"test"}'),
        ("namespaced", "workspace__web_search", '{"message":"run"}'),
        ("flat", "workspace__web_search", '{"message":"run"}'),
    ]
    if backend == "messages":
        server = build_messages_server(store, [
            build_messages_tool_use_events(call_id, name, [arguments])
            for call_id, name, arguments in calls
        ])
    else:
        server = build_test_server(store, [
            build_tool_call_chunks(call_id, name, [arguments])
            for call_id, name, arguments in calls
        ])
    server.start()
    app = ManagedResponseServer.build_app(CompatServerConfig(
        outcomming_base_url=server.base_url + "/v1", outcomming_api=backend,
    ))
    try:
        with TestClient(app) as client:
            response = client.post("/v1/responses", json={
                "model": "test", "input": [], "tools": [
                    {"type": "web_search"},
                    _namespace("workspace", [_function("web_search")]),
                ],
            })
            events = _events(response)
            assert events[-1]["type"] == "response.completed"
            items = _items(events)
            assert [item["type"] for item in items] == ["web_search_call", "function_call"]
            assert items[1]["namespace"] == "workspace"
            assert items[1]["name"] == "web_search"
            response = client.post("/v1/responses", json={
                "model": "test", "input": [], "tools": [_function("workspace__web_search")],
            })
            events = _events(response)
            assert events[-1]["type"] == "response.completed"
            call = _items(events)[0]
            assert call["name"] == "workspace__web_search"
            assert "namespace" not in call
    finally:
        server.stop()
    captures = sorted((tmp_path / "capture").glob("*_POST_*.json"))
    assert len(captures) == 3
    followup = json.loads(captures[1].read_text())["body"]
    assert len(followup["tools"]) == 1


@pytest.mark.parametrize(
    "tools,input_items,choice,error",
    [
        ([_namespace("", [])], [], "auto", "non-empty `name`"),
        ([_namespace("a", {})], [], "auto", "`tools` must be a list"),
        ([_namespace("a", [None])], [], "auto", "must be objects"),
        ([_namespace("a", [{"type": "web_search"}])], [], "auto", "function or custom"),
        ([_namespace("a", [_namespace("b", [])])], [], "auto", "function or custom"),
        (
            [_function("a__run"), _namespace("a", [_function("run")])],
            [], "auto", "collision",
        ),
        (
            [_namespace("a__b", [_function("c")]), _namespace("a", [_function("b__c")])],
            [], "auto", "collision",
        ),
        (
            [_function("a__run")],
            [{"type": "function_call", "namespace": "a", "name": "run", "arguments": "{}"}],
            "auto", "collision",
        ),
        (
            [_function("a__run")], [],
            {"type": "function", "namespace": "a", "name": "run"}, "collision",
        ),
        (
            [],
            [{"type": "function_call", "namespace": 1, "name": "run", "arguments": "{}"}],
            "auto", "namespace must be a non-empty string",
        ),
    ],
)
def test_invalid_namespace_requests_fail_before_backend(tools, input_items, choice, error):
    app = ManagedResponseServer.build_app(CompatServerConfig(
        outcomming_base_url="http://127.0.0.1:1/v1",
    ))
    with TestClient(app) as client:
        response = client.post("/v1/responses", json={
            "model": "test", "input": input_items, "tools": tools, "tool_choice": choice,
        })
    assert response.status_code == 501
    assert error in response.json()["error"]["message"]


def test_namespace_definition_groups_merge_without_mutating_request():
    request = {"model": "test", "tools": [
        _namespace("a", [_function("one")]),
        _namespace("empty", []),
        _namespace("a", [_function("two")]),
    ]}
    original = copy.deepcopy(request)
    outgoing = build_outcomming_request(request)
    assert [tool["name"] for tool in outgoing["tools"]] == ["a__one", "a__two"]
    assert request == original
