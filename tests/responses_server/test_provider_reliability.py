import asyncio
import json

import pytest
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Route

from pycodex import (
    AssistantMessage, Prompt, ResponsesModelClient, ResponsesProviderConfig,
    ToolCall, ToolSpec, UserMessage,
)
from pycodex.model import ResponsesApiError, ResponsesRetryableError
from responses_server import CompatServerConfig, ManagedResponseServer
from tests.responses_server.fake_chat_completions_server import (
    CaptureStore, RunningASGITestServer, build_messages_server,
    build_messages_text_events, build_messages_tool_use_events,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("api, traffic", [
    ("messages", "tool_arguments"),
    ("messages", "ping"),
    ("gemini", "comments"),
    ("gemini", "metadata"),
    ("gemini", "silent"),
])
async def test_downstream_activity_preserves_client_idle_timeout(api, traffic):
    requests = []

    async def endpoint(request):
        requests.append(await request.json())

        async def events():
            if traffic == "tool_arguments":
                for entry in build_messages_tool_use_events(
                    "call_one", "echo", ['{"text":"'] + ["x"] * 15 + ['"}'],
                ):
                    yield "data: " + json.dumps(entry["data"]) + "\n\n"
                    await asyncio.sleep(0.1)
                return
            for _ in range(15):
                if traffic == "comments":
                    yield ": downstream alive\n\n"
                elif traffic == "ping":
                    yield 'event: ping\ndata: {"type":"ping"}\n\n'
                elif traffic == "metadata":
                    yield 'data: {"usageMetadata":{"promptTokenCount":10}}\n\n'
                await asyncio.sleep(0.1)
            if api == "messages":
                for entry in build_messages_text_events("done"):
                    yield "data: " + json.dumps(entry["data"]) + "\n\n"
            else:
                yield "data: " + json.dumps({"candidates": [{
                    "content": {"role": "model", "parts": [{"text": "done"}]},
                    "finishReason": "STOP",
                }]}) + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    backend = RunningASGITestServer(Starlette(routes=[
        Route("/{path:path}", endpoint, methods=["POST"]),
    ]))
    backend.start()
    compat = ManagedResponseServer(CompatServerConfig(
        outcomming_base_url=backend.base_url + "/v1", outcomming_api=api,
    ))
    client = ResponsesModelClient(ResponsesProviderConfig(
        "synthetic", "fixture", compat.base_url, None,
        stream_idle_timeout_ms=1000, stream_max_retries=0,
    ))
    observed = []
    try:
        compat.start()
        try:
            prompt = Prompt([UserMessage("test")], [ToolSpec(
                "echo", "Echo text", {"type": "object"},
            )])
            if traffic == "silent":
                with pytest.raises(ResponsesRetryableError, match="timed out"):
                    await client.complete(prompt, observed.append)
            else:
                response = await client.complete(prompt, observed.append)
                if traffic == "tool_arguments":
                    assert len(response.items) == 1
                    assert isinstance(response.items[0], ToolCall)
                    assert response.items[0].arguments == {"text": "x" * 15}
                else:
                    assert [item.text for item in response.items
                            if isinstance(item, AssistantMessage)] == ["done"]
                assert not [event for event in observed if event.kind == "stream_error"]
        finally:
            compat.stop()
    finally:
        backend.stop()
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("text, explanation", [
    ("", "Blocked by the upstream policy."),
    ("I cannot help with that.", None),
])
async def test_messages_refusal_is_terminal_and_does_not_retry(tmp_path, text, explanation):
    entries = build_messages_text_events(text)
    for entry in entries:
        if entry["data"]["type"] == "message_delta":
            delta = entry["data"]["delta"]
            delta["stop_reason"] = "refusal"
            if explanation is not None:
                delta["stop_details"] = {"type": "refusal", "explanation": explanation}
    captures = tmp_path / "captures"
    backend = build_messages_server(CaptureStore(captures), entries)
    backend.start()
    compat = ManagedResponseServer(CompatServerConfig(
        outcomming_base_url=backend.base_url + "/v1", outcomming_api="messages",
    ))
    client = ResponsesModelClient(ResponsesProviderConfig(
        "synthetic", "fixture", compat.base_url, None, stream_max_retries=2,
    ))
    observed = []
    try:
        compat.start()
        try:
            with pytest.raises(ResponsesApiError, match="stop_reason=refusal") as caught:
                await client.complete(Prompt([UserMessage("test")], []), observed.append)
            assert "invalid_prompt" in str(caught.value)
            if explanation is not None:
                assert explanation in str(caught.value)
        finally:
            compat.stop()
    finally:
        backend.stop()
    assert len(list(captures.glob("*_POST_*.json"))) == 1
    assert not [event for event in observed if event.kind == "stream_error"]
