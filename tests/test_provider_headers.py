import json
from contextlib import closing

import pytest

from pycodex.bootstrap import build_model
from pycodex.model import ResponsesModelClient, ResponsesProviderConfig
from pycodex.protocol import AssistantMessage, Prompt, UserMessage
from responses_server import CompatServerConfig, OutcommingTransport
from tests.responses_server.fake_chat_completions_server import (
    CaptureStore, build_test_server, build_text_chunks,
    build_messages_server, build_messages_text_events,
)


def test_provider_headers_follow_upstream_resolution(monkeypatch):
    monkeypatch.setenv("HEADER_TOKEN", "environment-token")
    monkeypatch.setenv("HEADER_EMPTY", " \t ")
    monkeypatch.setenv("HEADER_INVALID", "bad\r\nheader")
    monkeypatch.delenv("HEADER_MISSING", raising=False)
    config = ResponsesProviderConfig(
        model="test", provider_name="test", base_url="https://example.invalid", api_key_env=None,
        http_headers={
            "X-Token": "static", "X-Empty": "", "X-Missing": "keep",
            "X-Blank": "keep", "X-Invalid-Env": "keep", "X-Utf8": "中文",
            "Invalid Name": "ignored", "X-Invalid": "bad\x00value",
        },
        env_http_headers={
            "x-token": "HEADER_TOKEN", "X-Missing": "HEADER_MISSING",
            "X-Blank": "HEADER_EMPTY", "X-Invalid-Env": "HEADER_INVALID",
        },
    )
    client = ResponsesModelClient(config)
    child = client.with_overrides(model="child")
    monkeypatch.setenv("HEADER_TOKEN", "later-token")
    for current in (client, child):
        for headers in (current._build_headers(Prompt([], [])), current._build_model_list_headers()):
            custom = {name: value for name, value in headers.items() if name.startswith("x-") and name != "x-client-request-id"}
            assert custom == {
                "x-token": "environment-token", "x-empty": "", "x-missing": "keep",
                "x-blank": "keep", "x-invalid-env": "keep",
                "x-utf8": "中文".encode("utf-8").decode("latin-1"),
            }
            assert "authorization" not in headers


def test_native_custom_headers_override_defaults_before_request_auth(monkeypatch):
    monkeypatch.setenv("HEADER_TOKEN", "auth-token")
    client = ResponsesModelClient(ResponsesProviderConfig(
        model="test", provider_name="test", base_url="https://example.invalid",
        api_key_env="HEADER_TOKEN", http_headers={
            "User-Agent": "custom-client", "Originator": "custom-originator",
            "Authorization": "custom-auth", "Accept": "custom-accept",
            "Session_Id": "custom-session",
        },
    ), session_id="real-session")
    response_headers = client._build_headers(Prompt([], []))
    model_headers = client._build_model_list_headers()
    for headers in (response_headers, model_headers):
        assert headers["user-agent"] == "custom-client"
        assert headers["originator"] == "custom-originator"
        assert headers["authorization"] == "Bearer auth-token"
    assert response_headers["session_id"] == "real-session"
    assert response_headers["accept"] == "text/event-stream"
    assert model_headers["accept"] == "application/json"


@pytest.mark.parametrize("api", ["chat_completions", "messages", "gemini"])
def test_transport_auth_overrides_same_named_custom_headers(monkeypatch, api):
    monkeypatch.setenv("HEADER_TOKEN", "auth-token")
    headers = OutcommingTransport(CompatServerConfig(
        outcomming_api=api, outcomming_api_key_env="HEADER_TOKEN",
        outcomming_http_headers={
            "X-Custom": "custom", "Authorization": "custom-auth", "X-Goog-Api-Key": "custom-key",
        },
    ))._build_headers("text/event-stream")
    assert headers["x-custom"] == "custom"
    if api == "gemini":
        assert headers["x-goog-api-key"] == "auth-token"
    else:
        assert headers["authorization"] == "Bearer auth-token"


@pytest.mark.parametrize("api", ["chat_completions", "messages"])
@pytest.mark.asyncio
async def test_custom_headers_reach_compat_backend_and_models(tmp_path, monkeypatch, api):
    monkeypatch.setenv("HEADER_TOKEN", "downstream-token")
    store = CaptureStore(tmp_path / "capture")
    if api == "messages":
        backend = build_messages_server(store, build_messages_text_events("OK"))
    else:
        backend = build_test_server(store, build_text_chunks("OK"))
    backend.start()
    provider = ResponsesProviderConfig(
        model="gpt-5.4", provider_name="test", base_url=backend.base_url + "/v1",
        api_key_env=None, compat_api=api, stream_max_retries=0,
        http_headers={"X-Static": "static", "X-Custom-Token": "old"},
        env_http_headers={"x-custom-token": "HEADER_TOKEN"},
    )
    try:
        with closing(build_model(provider)) as built:
            prompt = Prompt([UserMessage("hello")], [])
            assert "x-custom-token" not in built.client._build_headers(prompt)
            assert "x-static" not in built.client._build_model_list_headers()
            assert await built.client.list_models() == ["gpt-5.4"]
            output = await built.client.complete(prompt)
            assert len(output.items) == 1
            assert isinstance(output.items[0], AssistantMessage)
            assert output.items[0].text == "OK"
    finally:
        backend.stop()
    captures = [json.loads(path.read_text()) for path in (tmp_path / "capture").glob("*.json")]
    assert len(captures) == 2
    for capture in captures:
        headers = {name.lower(): value for name, value in capture["headers"].items()}
        assert headers["x-static"] == "static"
        assert headers["x-custom-token"] == "downstream-token"
        assert "authorization" not in headers
