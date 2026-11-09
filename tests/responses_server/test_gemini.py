import json
from contextlib import contextmanager
from copy import deepcopy

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from pycodex.model import ResponsesModelClient, ResponsesProviderConfig
from pycodex.context import ContextConfig, ContextManager
from pycodex.protocol import ToolResult, UserMessage
from pycodex.utils.session_persist import SessionRolloutRecorder, load_resumed_session_path
from responses_server import CompatServerConfig, ManagedResponseServer
from responses_server.gemini_api import GeminiAPIAdapterError, build_gemini_request
from responses_server.request_translation import build_outcomming_request
from tests.responses_server.fake_chat_completions_server import RunningASGITestServer
from tests.responses_server.test_stream_events import _check_lifecycle


def _response(parts, finish="STOP", usage=None):
    candidate = {"content": {"role": "model", "parts": parts}}
    if finish:
        candidate["finishReason"] = finish
    payload = {"candidates": [candidate]}
    if usage is not None:
        payload["usageMetadata"] = usage
    return payload


@contextmanager
def _bridge(streams, status=200, api_key_env=None):
    pending = iter(streams)
    captures = []

    async def generate(request):
        if api_key_env:
            assert request.headers["x-goog-api-key"] == "synthetic-key"
            assert "authorization" not in request.headers
        captures.append({"path": request.url.path, "query": request.url.query, "body": await request.json()})
        if status != 200:
            return JSONResponse({"error": {"message": "synthetic failure"}}, status_code=status)

        async def events():
            for payload in next(pending):
                # Exercise comments and CRLF framing.
                yield ": heartbeat\r\ndata: " + json.dumps(payload) + "\r\n\r\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    async def models(request):
        if api_key_env:
            assert request.headers["x-goog-api-key"] == "synthetic-key"
        if request.query_params.get("pageToken") == "next/page+token":
            return JSONResponse({"models": [{"name": "models/gemini-other"}]})
        return JSONResponse({
            "models": [{"name": "models/gemini-test"}], "nextPageToken": "next/page+token",
        })

    server = RunningASGITestServer(Starlette(routes=[
        Route("/v1beta/models/{model_path}", generate, methods=["POST"]),
        Route("/v1beta/models", models),
    ]))
    server.start()
    try:
        app = ManagedResponseServer.build_app(CompatServerConfig(
            outcomming_base_url=server.base_url + "/v1beta", outcomming_api="gemini",
            outcomming_api_key_env=api_key_env,
        ))
        with TestClient(app) as client:
            yield client, captures
    finally:
        server.stop()


def _post(client, request):
    response = client.post("/v1/responses", json=request)
    assert response.status_code == 200
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def _tool(name="run", kind="function"):
    if kind == "custom":
        return {"type": "custom", "name": name}
    return {"type": "function", "name": name, "parameters": {
        "type": "object", "properties": {"value": {"type": "integer"}},
        "required": ["value"], "additionalProperties": False,
    }}


def _streamed_parts(part):
    call = part["functionCall"]
    first = deepcopy(part)
    first["functionCall"] = {key: value for key, value in call.items() if key != "args"}
    first["functionCall"]["willContinue"] = True
    yield first
    for key, value in call["args"].items():
        if isinstance(value, str):
            for text in (value[:2], value[2:]):
                yield {"functionCall": {"partialArgs": [{
                    "jsonPath": "$." + key, "stringValue": text, "willContinue": True,
                }], "willContinue": True}}
            field, value = "stringValue", ""
        else:
            field = "numberValue"
        yield {"functionCall": {"partialArgs": [{"jsonPath": "$." + key, field: value}],
                                "willContinue": True}}
    yield {"functionCall": {}}


@pytest.mark.parametrize("instructions", [None, "Explicit system rules"])
def test_argon_model_prompt_reaches_native_system_instruction(tmp_path, instructions):
    model = "gemini-4-argon-preview"
    context = ContextManager(ContextConfig(
        model=model, personality="pragmatic", cwd=tmp_path,
        base_instructions=instructions,
        include_skills_instructions=False, include_permissions_instructions=False,
    ))
    history = [UserMessage("Hello")]
    prompt = context.build_prompt(history, [], True)
    client = ResponsesModelClient(ResponsesProviderConfig(
        model=model, provider_name="gemini", base_url="https://example.invalid/v1beta",
        api_key_env=None, responses_lite_override=False,
    ))
    request = client._build_payload(prompt)
    with _bridge([[_response([{"text": "Hello"}])]]) as (bridge, captures):
        _check_lifecycle(_post(bridge, request))
    system_text = "\n".join(part["text"] for part in captures[0]["body"]["systemInstruction"]["parts"])
    if instructions is None:
        assert system_text.startswith("You are Codex, a coding agent based on Gemini 4 Argon.")
        assert "You are a deeply pragmatic, effective software engineer." in system_text
        assert "Do NOT batch operations if they depend on the output" in system_text
        assert "{{" not in system_text
        assert "task_boundary" not in system_text
        assert "notify_user" not in system_text
    else:
        assert system_text == instructions
    assert request["model"] == model
    assert captures[0]["path"] == "/v1beta/models/" + model + ":streamGenerateContent"
    assert history == [UserMessage("Hello")]


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
def test_signed_parallel_tools_survive_stream_recording_and_resume(tmp_path, checkpoint, streamed):
    replayed_parts = [
        {"text": "Need both tools.", "thought": True, "thoughtSignature": "reasoning-signature"},
        {"text": "Running tools."},
        {"functionCall": {"id": "one", "name": "workspace__run", "args": {"value": 7}}, "thoughtSignature": "call-signature"},
        {"functionCall": {"id": "two", "name": "functions__apply_patch", "args": {"input": "*** Begin Patch\n*** End Patch"}}},
        {"text": "", "thoughtSignature": "trailing-signature"},
    ]
    # A streamed tool call can be followed by an unsigned empty text part.
    # Preserve the original state in recordings, but omit this invalid input
    # part when replaying it to Gemini. Signed empty parts must survive.
    parts = replayed_parts[:4] + [{"text": ""}] + replayed_parts[4:]
    request = {
        "model": "gemini-test", "input": [UserMessage("Do both tasks.").serialize()],
        "tools": [
            {"type": "namespace", "name": "workspace", "tools": [_tool()]},
            {"type": "namespace", "name": "functions", "tools": [_tool("apply_patch", "custom")]},
        ], "tool_choice": "required", "parallel_tool_calls": True,
    }
    wire_parts = []
    for part in parts:
        wire_parts.extend(_streamed_parts(part) if streamed and "functionCall" in part else [part])
    streams = [
        [_response([part], finish=None) for part in wire_parts] + [_response([])],
        [_response([{"text": "Done."}]), _response([{"text": "", "thoughtSignature": "answer-signature"}])],
        [_response([{"text": "Still done."}])],
    ]
    with _bridge(streams) as (client, captures):
        events = _post(client, request)
        items = _check_lifecycle(events)
        assert [item["type"] for item in items] == ["reasoning", "message", "reasoning", "function_call", "custom_tool_call"]
        assert [entry["part"] for entry in items[2]["provider_data"]] == parts
        assert all("provider_data" not in item for item in items if item["type"] != "reasoning")
        assert items[-2]["encrypted_function_args"] == []
        assert items[-2]["namespace"] == "workspace"
        assert items[-1]["input"] == "*** Begin Patch\n*** End Patch"
        history = [UserMessage("Do both tasks.")] + [ResponsesModelClient._parse_output_item(item) for item in items]
        history.extend([
            ToolResult("one", "workspace.run", "ok"),
            ToolResult("two", "apply_patch", "ok", tool_type="custom"),
        ])
        recorder = SessionRolloutRecorder.create(tmp_path, "synthetic", tmp_path, "test", "gemini", "")
        recorder.append_history_items(history)
        if checkpoint:
            recorder.append_compacted_history(history)
        restored = load_resumed_session_path(recorder.rollout_path)["history"]
        assert [item.serialize() for item in restored] == [item.serialize() for item in history]
        context = ContextManager(ContextConfig(base_instructions="test", cwd=tmp_path))
        fork = context.snapshot_for_fork(restored, [], None)
        assert [item.serialize() for item in fork.history] == [item.serialize() for item in restored]
        assert fork.history[3].payload["provider_data"] is not restored[3].payload["provider_data"]
        request["input"] = [item.serialize() for item in fork.history]
        # Signed calls must replay even after their declarations are removed.
        request["tools"] = [_tool("finish")]
        request["tool_choice"] = "auto"
        answer = _check_lifecycle(_post(client, request))
        assert answer[0]["content"][0]["text"] == "Done."
        assert answer[1]["provider_data"][-1]["part"]["thoughtSignature"] == "answer-signature"
        request["input"] += [ResponsesModelClient._parse_output_item(item).serialize() for item in answer]
        request["input"].append(UserMessage("Confirm.").serialize())
        _check_lifecycle(_post(client, request))
        assert client.get("/v1/models").json()["data"][0]["id"] == request["model"]
    assert captures[0]["path"] == "/v1beta/models/gemini-test:streamGenerateContent"
    assert captures[0]["query"] == "alt=sse"
    assert "model" not in captures[0]["body"]
    assert captures[1]["body"]["contents"][-2] == {"role": "model", "parts": replayed_parts}
    results = captures[1]["body"]["contents"][-1]["parts"]
    assert [part["functionResponse"]["name"] for part in results] == ["workspace__run", "functions__apply_patch"]
    assert [part["functionResponse"]["id"] for part in results] == ["one", "two"]
    assert captures[2]["body"]["contents"][-2]["parts"] == [
        {"text": "Done."}, {"text": "", "thoughtSignature": "answer-signature"},
    ]


def test_stream_text_reasoning_usage_and_partial_usage_updates():
    streams = [[
        _response([{"text": "Think", "thought": True}], finish=None),
        _response([{"text": "Hello "}], finish=None),
        _response([{"text": "world"}], finish=None),
        _response([{"text": "", "thoughtSignature": "signature"}], usage={
            "promptTokenCount": 35, "candidatesTokenCount": 5,
            "thoughtsTokenCount": 104, "cachedContentTokenCount": 12,
        }),
        {"usageMetadata": {"candidatesTokenCount": 6, "totalTokenCount": 145}},
        {"usageMetadata": {"trafficType": "ON_DEMAND"}},
        {"modelVersion": "gemini-test"},
    ]]
    with _bridge(streams) as (client, captures):
        events = _post(client, {"model": "argon", "input": []})
    items = _check_lifecycle(events)
    assert [item["content"][0]["text"] for item in items if item.get("content")] == ["Think", "Hello world"]
    assert events[-1]["response"]["usage"] == {
        "input_tokens": 35, "output_tokens": 110, "total_tokens": 145,
        "input_tokens_details": {"cached_tokens": 12}, "output_tokens_details": {"reasoning_tokens": 104},
    }
    assert [event["delta"] for event in events if event["type"] == "response.output_text.delta"] == ["Hello ", "world"]


@pytest.mark.parametrize("choice,mode,names", [
    ("auto", "AUTO", None), ("required", "ANY", None), ("none", "NONE", None),
    ({"type": "function", "namespace": "workspace", "name": "run"}, "ANY", ["workspace__run"]),
])
def test_request_instructions_images_tool_results_schema_and_choice(choice, mode, names):
    request = {
        "model": "argon", "instructions": "Base rules", "max_output_tokens": 2000,
        "reasoning": {"effort": "xhigh"}, "tool_choice": choice,
        "tools": [{"type": "namespace", "name": "workspace", "description": "Workspace tools", "tools": [_tool()]}],
        "input": [
            {"type": "message", "role": "system", "content": "System rules"},
            {"type": "message", "role": "developer", "content": "Developer rules"},
            {"type": "agent_message", "author": "one", "recipient": "two", "content": [{"type": "input_text", "text": "Inspect"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,YQ=="}]},
            {"type": "function_call", "name": "run", "namespace": "workspace", "call_id": "call", "arguments": '{"value":1}'},
            {"type": "function_call_output", "call_id": "call", "output": [
                {"type": "input_text", "text": "Image result"},
                {"type": "input_image", "image_url": "data:image/png;base64,Yg=="},
            ]},
        ],
    }
    original = deepcopy(request)
    payload = build_gemini_request(build_outcomming_request(request))
    assert request == original
    assert payload["systemInstruction"]["parts"] == [{"text": text} for text in ["Base rules", "System rules", "Developer rules"]]
    assert payload["generationConfig"] == {"maxOutputTokens": 2000, "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "high"}}
    assert payload["contents"][0]["parts"] == [{"text": "Agent message from one to two:\nInspect"}]
    assert payload["contents"][1]["parts"] == [{"inlineData": {"mimeType": "image/png", "data": "YQ=="}}]
    assert payload["contents"][-2]["parts"] == [
        {"functionResponse": {"id": "call", "name": "workspace__run", "response": {"result": "Image result"}}},
    ]
    assert payload["contents"][-1] == {"role": "user", "parts": [
        {"inlineData": {"mimeType": "image/png", "data": "Yg=="}},
    ]}
    declaration = payload["tools"][0]["functionDeclarations"][0]
    assert declaration["parametersJsonSchema"] == _tool()["parameters"]
    assert declaration["description"].startswith("Workspace tools")
    expected = {"mode": mode, "streamFunctionCallArguments": True}
    if names:
        expected["allowedFunctionNames"] = names
    assert payload["toolConfig"]["functionCallingConfig"] == expected


@pytest.mark.parametrize("explicit_end", [False, True])
def test_streamed_arguments_preserve_nested_values_and_escaped_paths(explicit_end):
    partials = [
        {"jsonPath": "$.items[]"},
        {"jsonPath": "$.items[0]{}"},
        {"jsonPath": "$.items[0].text", "stringValue": '中"\\\n', "willContinue": True},
        {"jsonPath": "$.items[0].text", "stringValue": "文"},
        {"jsonPath": "$.items[1]", "numberValue": 2.5},
        {"jsonPath": "$.enabled", "boolValue": False},
        {"jsonPath": "$.nothing", "nullValue": None if explicit_end else "NULL_VALUE"},
        {"jsonPath": "$.zero", "numberValue": 0},
        {"jsonPath": "$.empty", "stringValue": ""},
        {"jsonPath": "$.empty_list[]"},
        {"jsonPath": "$.empty_object{}"},
        {"jsonPath": '$["a.b"][0]', "stringValue": "quoted"},
        {"jsonPath": "$['single\\'quote']", "stringValue": "single"},
        {"jsonPath": '$["double\\\"quote"]', "stringValue": "double"},
    ]
    first = {"functionCall": {"name": "store", "id": "call", "willContinue": True,
                             "partialArgs": [partials.pop(0)]},
             "thoughtSignature": "signed"}
    end = {"willContinue": False} if explicit_end else {}
    stream = [_response([first], finish=None)]
    stream.extend(_response([{"functionCall": {
        "partialArgs": [partial], "willContinue": True,
    }}], finish=None) for partial in partials)
    stream.append(_response([{"functionCall": end, "thoughtSignature": "tail"}]))
    with _bridge([stream, [_response([{"text": "saved"}])]]) as (client, captures):
        request = {"model": "gemini-test", "input": [], "tools": [_tool("store")]}
        items = _check_lifecycle(_post(client, request))
        arguments = json.loads(items[-1]["arguments"])
        assert arguments == {
            "items": [{"text": '中"\\\n文'}, 2.5], "enabled": False,
            "nothing": None, "zero": 0, "empty": "", "a.b": ["quoted"],
            "empty_list": [], "empty_object": {},
            "single'quote": "single", 'double"quote': "double",
        }
        request["input"] = items + [{"type": "function_call_output", "call_id": "call", "output": "ok"}]
        _check_lifecycle(_post(client, request))
    assert captures[1]["body"]["contents"][0]["parts"] == [
        {"functionCall": {"name": "store", "id": "call", "args": arguments},
         "thoughtSignature": "signed"},
        {"text": "", "thoughtSignature": "tail"},
    ]


def test_zero_argument_call_preserves_its_signed_native_part():
    part = {"functionCall": {"name": "ping", "id": "one"}, "thoughtSignature": "signed"}
    with _bridge([[_response([part])]]) as (client, captures):
        items = _check_lifecycle(_post(client, {
            "model": "gemini-test", "input": [],
            "tools": [{"type": "function", "name": "ping", "parameters": {"type": "object"}}],
        }))
    assert items[-1]["arguments"] == "{}"
    assert items[0]["provider_data"][0]["part"] == part


@pytest.mark.parametrize("parts,error", [
    ([{"functionCall": {}}], "invalid Gemini function call"),
    ([{"functionCall": {"name": "run", "willContinue": True}}], "incomplete function call"),
    ([
        {"functionCall": {"name": "run", "willContinue": True}},
        {"functionCall": {"partialArgs": [
            {"jsonPath": "$.value", "stringValue": "unfinished", "willContinue": True},
        ]}},
    ], "incomplete arguments"),
    ([
        {"functionCall": {"name": "run", "willContinue": True}},
        {"functionCall": {"name": "other", "args": {}}},
    ], "conflicting Gemini function call fragment"),
])
def test_invalid_streamed_call_never_publishes_a_tool(parts, error):
    with _bridge([[_response(parts)]]) as (client, captures):
        events = _post(client, {"model": "gemini-test", "input": [], "tools": [_tool()]})
    assert events[-1]["type"] == "response.failed"
    assert error in events[-1]["response"]["error"]["message"]
    assert not any(event.get("item", {}).get("type") == "function_call" for event in events)


@pytest.mark.parametrize("partial", [
    {"jsonPath": "$..secret", "stringValue": "hidden"},
    {"jsonPath": "$.items[*]", "stringValue": "hidden"},
    {"jsonPath": "$.items[1000000]", "stringValue": "hidden"},
    {"jsonPath": "$.value", "stringValue": "hidden", "numberValue": 2},
    {"jsonPath": "$.value", "boolValue": "false"},
    {"jsonPath": "$.value", "numberValue": True},
    {"jsonPath": "$.value", "nullValue": 99},
    {"jsonPath": "$.items[]", "stringValue": "hidden"},
])
def test_invalid_partial_argument_fails_explicitly(partial):
    stream = [
        _response([{"functionCall": {"name": "run", "willContinue": True}}], finish=None),
        _response([{"functionCall": {"partialArgs": [partial]}}]),
    ]
    with _bridge([stream]) as (client, captures):
        events = _post(client, {"model": "gemini-test", "input": [], "tools": [_tool()]})
    assert events[-1]["type"] == "response.failed"
    assert "hidden" not in events[-1]["response"]["error"]["message"]
    assert not any(event.get("item", {}).get("type") == "function_call" for event in events)


@pytest.mark.parametrize("chunks,error", [
    ([_response([{"text": "partial"}], finish=None)], "before `finishReason`"),
    ([_response([], finish="MAX_TOKENS")], "MAX_TOKENS"),
    ([_response([], finish="SAFETY")], "SAFETY"),
    ([_response([], finish="MALFORMED_FUNCTION_CALL")], "MALFORMED_FUNCTION_CALL"),
    ([{"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}], "PROHIBITED_CONTENT"),
    ([{"error": {"message": "overloaded"}}], "overloaded"),
    ([_response([{"inlineData": {"mimeType": "image/png", "data": "YQ=="}}])], "unsupported Gemini output part"),
    ([_response([{"functionCall": {"name": "run", "args": []}}])], "invalid Gemini function call"),
    ([[]], "must be an object"),
])
def test_failed_or_truncated_stream_never_completes(chunks, error):
    with _bridge([chunks]) as (client, captures):
        events = _post(client, {"model": "argon", "input": [], "tools": [_tool()]})
    assert events[-1]["type"] == "response.failed"
    assert error in events[-1]["response"]["error"]["message"]
    assert "phase=" in events[-1]["response"]["error"]["message"]
    assert not any(event["type"] == "response.completed" for event in events)
    assert not any(event.get("item", {}).get("type") == "function_call" for event in events)


def test_http_error_and_mock_search_signature_followup():
    with _bridge([], status=429) as (client, captures):
        events = _post(client, {"model": "argon", "input": []})
        assert events[-1]["type"] == "response.failed"
        assert "status 429" in events[-1]["response"]["error"]["message"]
    parts = [
        {"text": "Searching", "thought": True},
        {"functionCall": {"name": "web_search", "args": {"query": "test"}}, "thoughtSignature": "search-signature"},
    ]
    with _bridge([[_response(parts)], [_response([{"text": "No results"}])]]) as (client, captures):
        events = _post(client, {"model": "argon", "input": [], "tools": [{"type": "web_search"}]})
    _check_lifecycle(events)
    assert captures[1]["body"]["contents"][-2]["parts"] == parts
    assert "tools" not in captures[1]["body"]
    assert captures[1]["body"]["contents"][-1]["parts"][0]["functionResponse"]["name"] == "web_search"


def test_invalid_request_contracts_are_explicit():
    with pytest.raises(GeminiAPIAdapterError, match="matching function call"):
        build_gemini_request({"messages": [{"role": "tool", "tool_call_id": "missing", "content": "result"}]})
    with pytest.raises(GeminiAPIAdapterError, match="base64"):
        build_gemini_request({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png,abc"}}]}]})
    with pytest.raises(GeminiAPIAdapterError, match="reasoning effort"):
        build_gemini_request({"chat_template_kwargs": {"reasoning_effort": "invalid"}})
    with pytest.raises(GeminiAPIAdapterError, match="base64"):
        build_gemini_request({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.test/image.png"}}]}]})


def test_native_authentication_and_paginated_model_list(monkeypatch):
    monkeypatch.setenv("TEST_GEMINI_API_KEY", "synthetic-key")
    with _bridge([[_response([{"text": "Done"}])]], api_key_env="TEST_GEMINI_API_KEY") as (client, captures):
        _check_lifecycle(_post(client, {"model": "gemini-test", "input": []}))
        assert client.get("/v1/models").json() == {"object": "list", "data": [
            {"id": "gemini-test", "object": "model"},
            {"id": "gemini-other", "object": "model"},
        ]}


@pytest.mark.parametrize("base,expected", [
    ("https://example.test", "https://example.test/v1beta"),
    ("https://example.test/custom/v1beta/", "https://example.test/custom/v1beta"),
])
def test_gemini_endpoint_uses_configured_api_root(base, expected):
    config = CompatServerConfig.from_base_url(base, outcomming_api="gemini")
    assert config.outcomming_models_url() == expected + "/models"
    assert config.outcomming_gemini_url("gemini-test+preview") == (
        expected + "/models/gemini-test%2Bpreview:streamGenerateContent?alt=sse"
    )


@pytest.mark.parametrize("streamed", [False, True])
def test_continuation_state_does_not_replay_pruned_tool_calls(streamed):
    parts = [
        {"functionCall": {"id": "keep", "name": "run", "args": {"value": 1}}, "thoughtSignature": "keep-signature"},
        {"functionCall": {"id": "prune", "name": "run", "args": {"value": 2}}},
        {"text": "", "thoughtSignature": "pruned-tail"},
    ]
    wire_parts = []
    for part in parts:
        wire_parts.extend(_streamed_parts(part) if streamed and "functionCall" in part else [part])
    with _bridge([[_response(wire_parts)]]) as (client, captures):
        items = _check_lifecycle(_post(client, {"model": "argon", "input": [], "tools": [_tool()]}))
    restored = [ResponsesModelClient._parse_output_item(item).serialize() for item in items]
    retained = [item for item in restored if item.get("call_id") != "prune"]
    retained.append({"type": "function_call_output", "call_id": "keep", "output": "ok"})
    native = build_gemini_request(build_outcomming_request({"model": "argon", "input": retained}))
    assert native["contents"][0]["parts"] == parts[:1]


def test_reasoning_only_retry_discards_its_continuation_state():
    with _bridge([
        [_response([{"text": "discard", "thought": True, "thoughtSignature": "discarded"}])],
        [_response([{"text": "answer", "thoughtSignature": "retained"}])],
    ]) as (client, captures):
        items = _check_lifecycle(_post(client, {"model": "argon", "input": []}), allow_abandoned=True)
    assert "discarded" not in json.dumps(items)
    state = [item for item in items if "provider_data" in item]
    assert len(state) == 1
    assert state[0]["provider_data"][0]["part"]["thoughtSignature"] == "retained"
