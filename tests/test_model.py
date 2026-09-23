import asyncio
import json
import re
import threading
import typing
import uuid
from dataclasses import replace
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest
import requests

import pycodex.utils.get_env as get_env
from pycodex import (
    NOOP_MODEL_STREAM_EVENT_HANDLER,
    AssistantMessage,
    ContextMessage,
    ModelEvent,
    ModelResponse,
    Prompt,
    ReasoningItem,
    ResponsesIncompleteError,
    ResponsesModelClient,
    ResponsesProviderConfig,
    ToolCall,
    ToolResult,
    ToolSpec,
    UserMessage,
)
from pycodex.compat import ThreadingHTTPServer
from pycodex.events import AssistantDeltaEvent, TokenCountEvent, ToolCalledEvent
from pycodex.model import RESPONSES_LITE_HEADER
from pycodex.utils.session_persist import (
    SessionRolloutRecorder,
    load_resumed_session_path,
)
from tests.fake_responses_server import CaptureStore, build_handler

EXPECTED_RESPONSES_REQUEST_BODY_JSON = """{
  "model": "demo-model",
  "instructions": "Be concise.",
  "input": [
    {
      "type": "message",
      "role": "developer",
      "content": [
        {
          "type": "input_text",
          "text": "first developer block"
        },
        {
          "type": "input_text",
          "text": "second developer block"
        }
      ]
    },
    {
      "type": "message",
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "hi"
        }
      ]
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "hello"
        }
      ]
    },
    {
      "type": "function_call",
      "name": "echo",
      "arguments": "{\\"text\\":\\"hello\\"}",
      "call_id": "call_1"
    },
    {
      "type": "function_call_output",
      "call_id": "call_1",
      "output": "{\\"text\\":\\"hello\\"}"
    },
    {
      "type": "custom_tool_call",
      "name": "apply_patch",
      "input": "*** Begin Patch\\n*** End Patch",
      "call_id": "call_2"
    },
    {
      "type": "custom_tool_call_output",
      "call_id": "call_2",
      "output": "patched",
      "name": "apply_patch"
    }
  ],
  "tools": [
    {
      "type": "function",
      "name": "echo",
      "description": "Echo text.",
      "parameters": {
        "type": "object"
      },
      "strict": false
    },
    {
      "type": "custom",
      "name": "apply_patch",
      "description": "Apply a patch.",
      "format": {
        "type": "grammar",
        "syntax": "lark",
        "definition": "start: /.+/"
      }
    },
    {
      "type": "web_search",
      "external_web_access": true
    }
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": true,
  "store": false,
  "stream": true,
  "include": [
    "reasoning.encrypted_content"
  ],
  "prompt_cache_key": "00000000-0000-7000-8000-000000000000",
  "reasoning": {
    "effort": "medium",
    "summary": "detailed"
  },
  "text": {
    "verbosity": "low"
  }
}"""


def _normalized_headers(headers: "typing.Dict[str, str]") -> "typing.Dict[str, str]":
    return {key.lower(): value for key, value in headers.items()}


async def test_cancelled_provider_stream_cannot_deliver_late_events():
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finished = asyncio.Event()
    release = threading.Event()
    callback_threads = []
    owner_thread = threading.get_ident()

    class LocalClient(ResponsesModelClient):
        def _complete_sync(self, prompt, event_handler):
            if prompt.turn_id == "old":
                loop.call_soon_threadsafe(started.set)
                if not release.wait(3):
                    raise RuntimeError("test stream was not released")
                event_handler(TokenCountEvent({"total_tokens": 777}))
                event_handler(AssistantDeltaEvent("stale"))
                loop.call_soon_threadsafe(finished.set)
            else:
                event_handler(TokenCountEvent({"total_tokens": 22}))
            return ModelResponse([AssistantMessage("answer")])

    client = LocalClient(
        ResponsesProviderConfig(
            model="test",
            provider_name="test",
            base_url="https://example.invalid/v1",
            api_key_env=None,
        )
    )
    events = []

    def observe(event):
        callback_threads.append(threading.get_ident())
        events.append(event)

    old_task = asyncio.create_task(
        client.complete(Prompt([], [], turn_id="old"), observe)
    )
    try:
        await asyncio.wait_for(started.wait(), 1)
        old_task.cancel()
        await asyncio.gather(old_task, return_exceptions=True)
        await client.complete(Prompt([], [], turn_id="new"), observe)
        assert [
            event.usage["total_tokens"]
            for event in events
            if event.kind == "token_count"
        ] == [22]
        events_after_new = len(events)
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
        assert len(events) == events_after_new
        assert set(callback_threads) == {owner_thread}
    finally:
        release.set()
        await asyncio.gather(old_task, return_exceptions=True)
        await asyncio.wait_for(finished.wait(), 1)


def test_provider_config_reads_codex_style_config_with_profile_override(
    tmp_path, monkeypatch
) -> "None":
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "base-model"',
                'model_provider = "primary"',
                'model_reasoning_effort = "high"',
                'model_verbosity = "medium"',
                'service_tier = "fast"',
                "",
                "[profiles.fast]",
                'model = "fast-model"',
                'model_provider = "backup"',
                "",
                "[model_providers.primary]",
                'base_url = "https://example.com/v1"',
                'env_key = "PRIMARY_KEY"',
                'wire_api = "responses"',
                "",
                "[model_providers.backup]",
                'base_url = "https://backup.example.com/openai"',
                'env_key = "BACKUP_KEY"',
                'wire_api = "responses"',
                "",
                "[model_providers.backup.query_params]",
                'api_version = "2026-03-01"',
                "",
                "[features]",
                "guardian_approval = true",
            ]
        )
    )
    monkeypatch.setenv("BACKUP_KEY", "test-key")

    provider = ResponsesProviderConfig.from_codex_config(config_path, "fast")
    client = ResponsesModelClient(provider)

    assert provider.model == "fast-model"
    assert provider.provider_name == "backup"
    assert provider.api_key() == "test-key"
    assert (
        client.responses_url()
        == "https://backup.example.com/openai/responses?api_version=2026-03-01"
    )
    assert provider.reasoning_effort == "high"
    assert provider.verbosity == "medium"
    assert provider.service_tier == "fast"
    assert provider.beta_features_header == "guardian_approval"


def test_build_payload_omits_tool_choice_when_no_tools() -> "None":
    client = ResponsesModelClient(
        ResponsesProviderConfig(
            model="gpt-test",
            provider_name="demo",
            base_url="https://example.com/v1",
            api_key_env=None,
        ),
        session_id="00000000-0000-7000-8000-000000000000",
    )

    payload = client._build_payload(
        Prompt(
            input=[UserMessage(text="hello")],
            tools=[],
            parallel_tool_calls=False,
            base_instructions="Be concise.",
        )
    )

    assert "tool_choice" not in payload


def test_provider_config_allows_provider_without_env_key(tmp_path) -> "None":
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "demo-model"',
                'model_provider = "local"',
                "",
                "[profiles.step]",
                'model = "step-3.5-flash-2603"',
                'model_provider = "stepfun-chat-zhy"',
                "",
                "[model_providers.local]",
                'base_url = "http://localhost:8000/v1"',
                'wire_api = "responses"',
                "",
                "[model_providers.stepfun-chat-zhy]",
                'base_url = "http://100.96.255.200:8001/v1"',
                'wire_api = "responses"',
            ]
        )
    )

    provider = ResponsesProviderConfig.from_codex_config(config_path, "step")

    assert provider.provider_name == "stepfun-chat-zhy"
    assert provider.base_url == "http://100.96.255.200:8001/v1"
    assert provider.api_key_env is None
    assert provider.api_key() is None


@pytest.mark.parametrize(
    ("model", "expected_effort"),
    [
        ("gpt-5.6-sol", "low"),
        ("gpt-5.6-terra", "medium"),
        ("gpt-5.6-luna", "medium"),
    ],
)
def test_provider_config_uses_gpt56_model_metadata_defaults(
    model,
    expected_effort,
) -> "None":
    provider = ResponsesProviderConfig(
        model=model,
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env=None,
    )

    assert provider.use_responses_lite() is True
    assert provider.effective_reasoning_effort() == expected_effort
    assert provider.effective_reasoning_summary() is None
    assert provider.effective_verbosity() == "low"


@pytest.mark.parametrize(
    ("service_tier", "expected"),
    [
        ("fast", "priority"),
        ("priority", "priority"),
        ("default", None),
        ("flex", None),
        ("unsupported", None),
    ],
)
def test_provider_config_resolves_supported_service_tier(
    service_tier,
    expected,
) -> "None":
    provider = ResponsesProviderConfig(
        model="gpt-5.6-sol",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env=None,
        service_tier=service_tier,
    )

    assert provider.effective_service_tier() == expected


def test_provider_config_omits_service_tier_for_unknown_model() -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env=None,
        service_tier="fast",
    )

    assert provider.effective_service_tier() is None


def test_responses_model_client_builds_responses_payload() -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env="DUMMY_KEY",
        reasoning_effort="medium",
        reasoning_summary="detailed",
        verbosity="low",
    )
    client = ResponsesModelClient(provider)
    prompt = Prompt(
        input=[
            ContextMessage(
                role="developer",
                content_items=(
                    {"type": "input_text", "text": "first developer block"},
                    {"type": "input_text", "text": "second developer block"},
                ),
            ),
            UserMessage(text="hi"),
            AssistantMessage(text="hello"),
            ToolCall(call_id="call_1", name="echo", arguments={"text": "hello"}),
            ToolResult(call_id="call_1", name="echo", output={"text": "hello"}),
            ToolCall(
                call_id="call_2",
                name="apply_patch",
                arguments="*** Begin Patch\n*** End Patch",
                tool_type="custom",
            ),
            ToolResult(
                call_id="call_2",
                name="apply_patch",
                output="patched",
                tool_type="custom",
            ),
        ],
        tools=[
            ToolSpec(
                name="echo",
                description="Echo text.",
                input_schema={"type": "object"},
                output_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            ),
            ToolSpec(
                name="apply_patch",
                description="Apply a patch.",
                tool_type="custom",
                format={
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": "start: /.+/",
                },
            ),
            ToolSpec(
                name="web_search",
                description="Web search.",
                tool_type="web_search",
                options={"external_web_access": True},
            ),
        ],
        parallel_tool_calls=True,
        base_instructions="Be concise.",
    )

    payload = client._build_payload(prompt)

    assert payload["model"] == "demo-model"
    assert payload["instructions"] == "Be concise."
    assert payload["parallel_tool_calls"] is True
    assert payload["prompt_cache_key"] is not None
    assert payload["reasoning"] == {"effort": "medium", "summary": "detailed"}
    assert payload["text"] == {"verbosity": "low"}
    assert payload["input"][0]["role"] == "developer"
    assert payload["input"][0]["content"][0]["type"] == "input_text"
    assert payload["input"][0]["content"][0]["text"] == "first developer block"
    assert payload["input"][0]["content"][1]["text"] == "second developer block"
    assert payload["input"][1]["content"][0]["type"] == "input_text"
    assert payload["input"][2]["content"][0]["type"] == "output_text"
    assert payload["input"][3]["type"] == "function_call"
    assert json.loads(payload["input"][3]["arguments"]) == {"text": "hello"}
    assert payload["input"][4]["type"] == "function_call_output"
    assert payload["input"][5] == {
        "type": "custom_tool_call",
        "name": "apply_patch",
        "input": "*** Begin Patch\n*** End Patch",
        "call_id": "call_2",
    }
    assert payload["input"][6] == {
        "type": "custom_tool_call_output",
        "call_id": "call_2",
        "name": "apply_patch",
        "output": "patched",
    }
    assert payload["tools"][0]["type"] == "function"
    assert "output_schema" not in payload["tools"][0]
    assert payload["tools"][1] == {
        "type": "custom",
        "name": "apply_patch",
        "description": "Apply a patch.",
        "format": {
            "type": "grammar",
            "syntax": "lark",
            "definition": "start: /.+/",
        },
    }
    assert payload["tools"][2] == {
        "type": "web_search",
        "external_web_access": True,
    }


def test_responses_model_client_builds_responses_lite_payload() -> "None":
    provider = ResponsesProviderConfig(
        model="gpt-5.6-sol",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env=None,
        service_tier="fast",
    )
    client = ResponsesModelClient(
        provider,
        session_id="00000000-0000-7000-8000-000000000000",
    )
    prompt = Prompt(
        input=[
            ContextMessage(
                role="user",
                content_items=(
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,abc",
                        "detail": "high",
                    },
                ),
            ),
            UserMessage(text="hi"),
        ],
        tools=[
            ToolSpec(
                name="echo",
                description="Echo text.",
                input_schema={"type": "object"},
            ),
        ],
        parallel_tool_calls=True,
        base_instructions="Be concise.",
    )

    payload = client._build_payload(prompt)
    headers = client._build_headers(prompt)

    assert "instructions" not in payload
    assert "tools" not in payload
    assert payload["parallel_tool_calls"] is False
    assert payload["tool_choice"] == "auto"
    assert payload["reasoning"] == {"effort": "low", "context": "all_turns"}
    assert payload["text"] == {"verbosity": "low"}
    assert payload["service_tier"] == "priority"
    assert payload["input"][0] == {
        "type": "additional_tools",
        "id": payload["input"][0]["id"],
        "role": "developer",
        "tools": [
            {
                "type": "function",
                "name": "echo",
                "description": "Echo text.",
                "parameters": {"type": "object"},
                "strict": False,
            }
        ],
    }
    assert payload["input"][1] == {
        "type": "message",
        "id": payload["input"][1]["id"],
        "role": "developer",
        "content": [{"type": "input_text", "text": "Be concise."}],
    }
    assert payload["input"][2]["content"][0] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,abc",
    }
    assert payload["input"][3]["content"][0]["text"] == "hi"
    assert headers[RESPONSES_LITE_HEADER] == "true"
    namespace = uuid.uuid5(uuid.NAMESPACE_OID, client._session_id)
    assert payload["input"][0]["id"] == "at_" + str(
        uuid.uuid5(
            namespace,
            json.dumps(
                payload["input"][0]["tools"], ensure_ascii=False, separators=(",", ":")
            ),
        )
    )
    assert payload["input"][1]["id"] == "msg_" + str(
        uuid.uuid5(namespace, "Be concise.")
    )
    assert client._build_payload(prompt) == payload
    changed = client._build_payload(
        replace(prompt, base_instructions="Different instructions.")
    )
    assert changed["input"][0] == payload["input"][0]
    assert changed["input"][1]["id"] != payload["input"][1]["id"]
    resumed = ResponsesModelClient(provider, session_id=client._session_id)
    assert resumed._build_payload(prompt) == payload


def test_suffixed_model_keeps_wire_name_and_uses_matching_metadata():
    client = ResponsesModelClient(
        ResponsesProviderConfig(
            model="gpt-6-astra:caishi-azure",
            provider_name="capture",
            base_url="http://127.0.0.1:1/v1",
            api_key_env=None,
            reasoning_effort="xhigh",
        )
    )
    payload = client._build_payload(
        Prompt(input=[], tools=[], base_instructions="Fixture.")
    )
    assert payload["model"] == "gpt-6-astra:caishi-azure"
    assert "instructions" not in payload and "tools" not in payload
    assert payload["input"][0]["type"] == "additional_tools"
    assert payload["input"][1]["content"] == [
        {"type": "input_text", "text": "Fixture."}
    ]
    assert payload["reasoning"] == {"effort": "xhigh", "context": "all_turns"}
    assert payload["text"] == {"verbosity": "low"}


def test_model_items_replay_metadata_and_raw_arguments_after_resume(tmp_path):
    client = ResponsesModelClient(
        ResponsesProviderConfig(
            model="gpt-5.4",
            provider_name="capture",
            base_url="http://127.0.0.1:1/v1",
            api_key_env=None,
        )
    )
    payloads = [
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_comment",
            "phase": "commentary",
            "content": [
                {"type": "output_text", "text": "First."},
                {"type": "output_text", "text": "Second."},
            ],
        },
        {
            "type": "reasoning",
            "id": "rs_fixture",
            "summary": [],
            "encrypted_content": "fixture-only",
            "content": None,
        },
        {
            "type": "function_call",
            "id": "fc_fixture",
            "namespace": "functions",
            "call_id": "call_fixture",
            "name": "exec_command",
            "arguments": '{ "cmd": "printf 测试", "max_output_tokens": 100 }',
        },
        {
            "type": "custom_tool_call",
            "id": "ctc_fixture",
            "status": "completed",
            "call_id": "custom_fixture",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch",
        },
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_answer",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "Done."}],
        },
    ]
    items = [client._parse_output_item(payload) for payload in payloads]
    assert [item.serialize() for item in items] == payloads
    assert items[0].text == "First.Second."
    assert items[2].arguments == {"cmd": "printf 测试", "max_output_tokens": 100}
    history = (
        [UserMessage("start")]
        + items[:3]
        + [
            ToolResult("call_fixture", "exec_command", "result", id="fco_fixture"),
            items[3],
            ToolResult(
                "custom_fixture",
                "apply_patch",
                "patched",
                tool_type="custom",
                id="cto_fixture",
            ),
            items[4],
        ]
    )
    recorder = SessionRolloutRecorder.create(
        tmp_path,
        "00000000-0000-7000-8000-000000000000",
        tmp_path,
        "codex_exec",
        "capture",
        "Fixture instructions.",
    )
    recorder.append_history_items(history)
    resumed = load_resumed_session_path(recorder.rollout_path)
    assert [item.serialize() for item in resumed["history"]] == [
        item.serialize() for item in history
    ]
    recorder.append_compacted_history(history)
    compacted = load_resumed_session_path(recorder.rollout_path)
    assert [item.serialize() for item in compacted["history"]] == [
        item.serialize() for item in history
    ]


@pytest.mark.parametrize(
    "content, retained",
    [
        (None, True),
        ([], False),
        ([{"type": "reasoning_text", "text": "Fixture"}], True),
    ],
)
def test_reasoning_replay_matches_upstream_optional_fields(content, retained):
    original = {"type": "reasoning", "summary": [], "content": content}
    serialized = ReasoningItem(original).serialize()
    assert ("content" in serialized) is retained
    assert serialized["encrypted_content"] is None
    assert "encrypted_content" not in original
    assert (
        ReasoningItem({"type": "reasoning", "summary": []}).serialize()["content"]
        is None
    )


def test_responses_model_client_can_disable_lite_for_compat_transport() -> "None":
    provider = ResponsesProviderConfig(
        model="gpt-6-astra",
        provider_name="vllm",
        base_url="http://127.0.0.1:18000/v1",
        api_key_env=None,
        responses_lite_override=False,
    )
    client = ResponsesModelClient(provider)

    prompt = Prompt(
        input=[UserMessage(text="hi")],
        tools=[],
        base_instructions="Be concise.",
    )
    payload = client._build_payload(prompt)
    headers = client._build_headers(prompt)

    assert provider.use_responses_lite() is False
    assert payload["instructions"] == "Be concise."
    assert payload["tools"] == []
    assert payload["input"][0]["type"] == "message"
    assert RESPONSES_LITE_HEADER not in headers


def test_responses_lite_payload_respects_explicit_reasoning_and_verbosity() -> "None":
    provider = ResponsesProviderConfig(
        model="gpt-5.6-sol",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env=None,
        reasoning_effort="ultra",
        reasoning_summary="auto",
        verbosity="medium",
    )
    client = ResponsesModelClient(provider)

    payload = client._build_payload(
        Prompt(
            input=[UserMessage(text="hi")],
            tools=[],
            base_instructions="Be concise.",
        )
    )

    assert payload["reasoning"] == {
        "effort": "ultra",
        "summary": "auto",
        "context": "all_turns",
    }
    assert payload["text"] == {"verbosity": "medium"}


def test_responses_payload_omits_default_service_tier() -> "None":
    provider = ResponsesProviderConfig(
        model="gpt-5.6-sol",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env=None,
        service_tier="default",
    )
    client = ResponsesModelClient(provider)

    payload = client._build_payload(
        Prompt(
            input=[UserMessage(text="hi")],
            tools=[],
        )
    )

    assert "service_tier" not in payload


@pytest.mark.asyncio
async def test_responses_model_client_lists_models_from_provider(monkeypatch) -> "None":
    requests_seen: "typing.List[str]" = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: "str", *args) -> "None":
            del format, args

        def do_GET(self) -> "None":
            requests_seen.append(self.path)
            body = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {"id": "demo-model", "object": "model"},
                        {"id": "alt-model", "object": "model"},
                    ],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    monkeypatch.setenv("DUMMY_KEY", "test-key")

    client = ResponsesModelClient(
        ResponsesProviderConfig(
            model="demo-model",
            provider_name="demo",
            base_url=f"http://127.0.0.1:{httpd.server_port}/v1",
            api_key_env="DUMMY_KEY",
        )
    )

    try:
        models = await client.list_models()
    finally:
        httpd.shutdown()
        server_thread.join(timeout=5)
        httpd.server_close()

    assert models == ["demo-model", "alt-model"]
    assert requests_seen == ["/v1/models"]


def test_responses_model_client_payload_matches_hardcoded_reference() -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env="DUMMY_KEY",
        reasoning_effort="medium",
        reasoning_summary="detailed",
        verbosity="low",
    )
    client = ResponsesModelClient(
        provider,
        session_id="00000000-0000-7000-8000-000000000000",
    )
    prompt = Prompt(
        input=[
            ContextMessage(
                role="developer",
                content_items=(
                    {"type": "input_text", "text": "first developer block"},
                    {"type": "input_text", "text": "second developer block"},
                ),
            ),
            UserMessage(text="hi"),
            AssistantMessage(text="hello"),
            ToolCall(call_id="call_1", name="echo", arguments={"text": "hello"}),
            ToolResult(call_id="call_1", name="echo", output={"text": "hello"}),
            ToolCall(
                call_id="call_2",
                name="apply_patch",
                arguments="*** Begin Patch\n*** End Patch",
                tool_type="custom",
            ),
            ToolResult(
                call_id="call_2",
                name="apply_patch",
                output="patched",
                tool_type="custom",
            ),
        ],
        tools=[
            ToolSpec(
                name="echo",
                description="Echo text.",
                input_schema={"type": "object"},
                output_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            ),
            ToolSpec(
                name="apply_patch",
                description="Apply a patch.",
                tool_type="custom",
                format={
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": "start: /.+/",
                },
            ),
            ToolSpec(
                name="web_search",
                description="Web search.",
                tool_type="web_search",
                options={"external_web_access": True},
            ),
        ],
        parallel_tool_calls=True,
        base_instructions="Be concise.",
    )

    payload = client._build_payload(prompt)

    assert payload == json.loads(EXPECTED_RESPONSES_REQUEST_BODY_JSON)


def test_responses_model_client_builds_codex_headers_and_stable_session_id(
    monkeypatch,
) -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env="DUMMY_KEY",
        sandbox_mode="danger-full-access",
        beta_features_header="guardian_approval",
    )
    monkeypatch.setenv("DUMMY_KEY", "test-key")
    client = ResponsesModelClient(provider, originator="codex_exec")
    prompt = Prompt(
        input=[UserMessage(text="hi")],
        tools=[],
        turn_id="turn_123",
        turn_metadata={"turn_id": "turn_123", "sandbox": "none"},
    )

    headers = client._build_headers(prompt)
    payload = client._build_payload(prompt)
    overridden = client.with_overrides(model="other-model")
    subagent = client.with_overrides(
        model="other-model",
        session_id="agent_123",
        openai_subagent="collab_spawn",
    )
    overridden_headers = overridden._build_headers(prompt)
    subagent_headers = subagent._build_headers(prompt)
    subagent_payload = subagent._build_payload(prompt)

    assert headers["x-client-request-id"] == payload["prompt_cache_key"]
    assert headers["session_id"] == payload["prompt_cache_key"]
    assert headers["originator"] == "codex_exec"
    assert headers["x-codex-beta-features"] == "guardian_approval"
    assert headers["x-codex-turn-metadata"] == '{"turn_id":"turn_123","sandbox":"none"}'
    assert re.match(
        r"^codex_exec/.+ \(.+; .+\) .+ \(codex-exec; .+\)$", headers["user-agent"]
    )
    assert "accept-encoding" not in headers
    assert "connection" not in headers
    assert overridden_headers["x-client-request-id"] == headers["x-client-request-id"]
    assert subagent_headers["x-client-request-id"] == "agent_123"
    assert subagent_headers["session_id"] == "agent_123"
    assert subagent_headers["x-openai-subagent"] == "collab_spawn"
    assert subagent_payload["prompt_cache_key"] == "agent_123"


def test_responses_model_client_omits_authorization_when_provider_has_no_env_key() -> (
    "None"
):
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env=None,
    )
    client = ResponsesModelClient(provider)
    prompt = Prompt(
        input=[UserMessage(text="hi")],
        tools=[],
        turn_id="turn_123",
        turn_metadata={"turn_id": "turn_123", "sandbox": "none"},
    )

    headers = client._build_headers(prompt)
    model_headers = client._build_model_list_headers()

    assert "authorization" not in headers
    assert "authorization" not in model_headers


def test_responses_model_client_wire_headers_and_body_match_builders(
    tmp_path,
    monkeypatch,
) -> "None":
    capture_root = tmp_path / "capture"
    capture_store = CaptureStore(capture_root)
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_handler(capture_store, "demo-model", "OK"),
    )
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url=f"http://127.0.0.1:{httpd.server_port}/v1",
        api_key_env="DUMMY_KEY",
        sandbox_mode="danger-full-access",
        beta_features_header="guardian_approval",
    )
    monkeypatch.setenv("DUMMY_KEY", "test-key")
    client = ResponsesModelClient(
        provider,
        session_id="00000000-0000-7000-8000-000000000000",
        originator="codex_exec",
    )
    prompt = Prompt(
        input=[UserMessage(text="hi")],
        tools=[],
        turn_id="turn_123",
        turn_metadata={"turn_id": "turn_123", "sandbox": "none"},
        base_instructions="Be concise.",
    )

    try:
        response = client._complete_sync(prompt, NOOP_MODEL_STREAM_EVENT_HANDLER)
    finally:
        httpd.shutdown()
        server_thread.join(timeout=5)
        httpd.server_close()

    assert response.items == [AssistantMessage(text="OK")]

    request_files = sorted(capture_root.glob("*_POST_*.json"))
    assert len(request_files) == 1
    request = json.loads(request_files[0].read_text())
    headers = _normalized_headers(request["headers"])

    assert request["body"] == client._build_payload(prompt)
    for key, value in client._build_headers(prompt).items():
        assert headers[key.lower()] == value
    assert headers.get("accept-encoding") == "identity"
    assert "connection" not in headers


def test_responses_model_client_builds_tui_user_agent(monkeypatch) -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env="DUMMY_KEY",
    )
    monkeypatch.setenv("DUMMY_KEY", "test-key")
    client = ResponsesModelClient(provider, originator="codex-tui")

    headers = client._build_headers(Prompt(input=[UserMessage(text="hi")], tools=[]))

    assert headers["originator"] == "codex-tui"
    assert re.match(
        r"^codex-tui/.+ \(.+; .+\) .+ \(codex-tui; .+\)$", headers["user-agent"]
    )


def test_get_package_version_reads_distribution_name(monkeypatch) -> "None":
    def fake_version(name: "str") -> "str":
        if name == "python-codex":
            return "0.1.3"
        raise get_env.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(get_env, "_detect_upstream_codex_version", lambda: None)
    monkeypatch.setattr(get_env.importlib_metadata, "version", fake_version)

    assert get_env.get_package_version() == "0.1.3"


def test_get_package_version_falls_back_to_local_pyproject(monkeypatch) -> "None":
    def fake_missing_version(_name: "str") -> "str":
        raise get_env.importlib_metadata.PackageNotFoundError

    monkeypatch.setattr(get_env, "_detect_upstream_codex_version", lambda: None)
    monkeypatch.setattr(get_env.importlib_metadata, "version", fake_missing_version)
    monkeypatch.setattr(get_env, "_read_local_package_version", lambda: "0.1.3")

    assert get_env.get_package_version() == "0.1.3"


def test_responses_model_client_serializes_prompt_turn_metadata(monkeypatch) -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env="DUMMY_KEY",
        sandbox_mode="danger-full-access",
    )
    monkeypatch.setenv("DUMMY_KEY", "test-key")
    client = ResponsesModelClient(provider, originator="codex-tui")

    headers = client._build_headers(
        Prompt(
            input=[UserMessage(text="hi")],
            tools=[],
            turn_id="turn_123",
            turn_metadata={
                "turn_id": "turn_123",
                "sandbox": "none",
                "workspaces": {
                    "/repo": {
                        "latest_git_commit_hash": "abc123",
                        "associated_remote_urls": {
                            "origin": "git@example.com:repo.git"
                        },
                        "has_changes": True,
                    }
                },
            },
        )
    )
    metadata = json.loads(headers["x-codex-turn-metadata"])

    assert metadata == {
        "turn_id": "turn_123",
        "sandbox": "none",
        "workspaces": {
            "/repo": {
                "latest_git_commit_hash": "abc123",
                "associated_remote_urls": {"origin": "git@example.com:repo.git"},
                "has_changes": True,
            }
        },
    }


def test_responses_model_client_serializes_reasoning_item_in_input() -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env="DUMMY_KEY",
    )
    client = ResponsesModelClient(
        provider, session_id="00000000-0000-7000-8000-000000000000"
    )
    prompt = Prompt(
        input=[
            ReasoningItem(
                payload={
                    "type": "reasoning",
                    "summary": [],
                    "content": None,
                    "encrypted_content": "encrypted-token",
                }
            )
        ],
        tools=[],
    )

    payload = client._build_payload(prompt)

    assert payload["input"] == [
        {
            "type": "reasoning",
            "summary": [],
            "content": None,
            "encrypted_content": "encrypted-token",
        }
    ]


def test_responses_model_client_parses_sse_stream() -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env="DUMMY_KEY",
    )
    client = ResponsesModelClient(provider)
    events: "typing.List[ModelEvent]" = []
    stream = [
        b"event: response.created\n",
        b'data: {"type":"response.created"}\n',
        b"\n",
        b"event: response.output_item.done\n",
        b'data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"echo","arguments":"{\\"text\\":\\"hello\\"}"}}\n',
        b"\n",
        b"event: response.output_item.done\n",
        b'data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}}\n',
        b"\n",
        b"event: response.completed\n",
        b'data: {"type":"response.completed","response":{"usage":{"input_tokens":11,"output_tokens":7,"total_tokens":18}}}\n',
        b"\n",
    ]

    response = client._parse_stream(stream, events.append)

    assert len(response.items) == 2
    assert isinstance(response.items[0], ToolCall)
    assert response.items[0].arguments == {"text": "hello"}
    assert isinstance(response.items[1], AssistantMessage)
    assert response.items[1].text == "done"
    assert events[-1] == TokenCountEvent(
        {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
        }
    )


def test_responses_model_client_formats_transport_stream_errors(monkeypatch) -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="http://127.0.0.1:8000/v1",
        api_key_env=None,
    )
    client = ResponsesModelClient(
        provider,
        session_id="00000000-0000-7000-8000-000000000000",
    )

    class _BrokenResponse:
        status_code = 200
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def iter_lines(self, chunk_size=1, decode_unicode=False):
            del chunk_size, decode_unicode
            yield b"event: response.created"
            yield b'data: {"type":"response.created"}'
            yield b""
            yield b"event: response.output_text.delta"
            yield b'data: {"type":"response.output_text.delta","delta":"partial"}'
            yield b""
            raise requests.exceptions.ChunkedEncodingError("Response ended prematurely")

    def fake_send(self, request, timeout, allow_redirects, **settings):
        del self, request, timeout, allow_redirects, settings
        return _BrokenResponse()

    monkeypatch.setattr(requests.Session, "send", fake_send)

    with pytest.raises(RuntimeError) as exc_info:
        client._complete_sync(
            Prompt(input=[UserMessage(text="hi")], tools=[]),
            NOOP_MODEL_STREAM_EVENT_HANDLER,
        )

    message = str(exc_info.value)
    assert message.startswith("responses request failed while reading the HTTP stream")
    assert "- provider: demo" in message
    assert "- model: demo-model" in message
    assert "- request: POST http://127.0.0.1:8000/v1/responses" in message
    assert "- session_id: 00000000-0000-7000-8000-000000000000" in message
    assert "- raw_lines_received: 6" in message
    assert "- sse_events_received: 2" in message
    assert "- output_items_received: 0" in message
    assert "- last_sse_event: response.output_text.delta" in message
    assert "- last_event_type: response.output_text.delta" in message
    assert (
        '- last_payload_excerpt: {"type":"response.output_text.delta","delta":"partial"}'
        in message
    )
    assert "- exception: ChunkedEncodingError" in message
    assert "- detail: Response ended prematurely" in message
    assert "local `responses_server`" in message


def test_responses_model_client_formats_incomplete_stream_errors() -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env="DUMMY_KEY",
    )
    client = ResponsesModelClient(provider)
    stream = [
        b"event: response.output_text.delta\n",
        b'data: {"type":"response.output_text.delta","delta":"partial"}\n',
        b"\n",
    ]

    with pytest.raises(RuntimeError) as exc_info:
        client._parse_stream(stream, NOOP_MODEL_STREAM_EVENT_HANDLER)

    message = str(exc_info.value)
    assert message.startswith("responses stream ended before `response.completed`")
    assert "- provider: demo" in message
    assert "- model: demo-model" in message
    assert "- request: POST https://example.com/v1/responses" in message
    assert "- last_event: response.output_text.delta" in message
    assert "- output_items_received: 0" in message
    assert "the stream ended without a terminal `response.completed` event" in message


def test_responses_model_client_raises_response_incomplete_with_partial_items() -> (
    "None"
):
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env="DUMMY_KEY",
    )
    client = ResponsesModelClient(provider)
    events: "typing.List[ModelEvent]" = []
    stream = [
        b"event: response.output_text.delta\n",
        b'data: {"type":"response.output_text.delta","delta":"partial summary"}\n',
        b"\n",
        b"event: response.output_item.done\n",
        b'data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"partial summary"}]}}\n',
        b"\n",
        b"event: response.incomplete\n",
        b'data: {"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}}\n',
        b"\n",
    ]

    with pytest.raises(ResponsesIncompleteError) as exc_info:
        client._parse_stream(stream, events.append)

    message = str(exc_info.value)
    assert message.startswith("responses stream ended with `response.incomplete`")
    assert "- reason: max_output_tokens" in message
    assert exc_info.value.reason == "max_output_tokens"
    assert "- output_items_received: 1" in message
    assert [
        item.text
        for item in exc_info.value.partial_items
        if isinstance(item, AssistantMessage)
    ] == ["partial summary"]
    assert events == [AssistantDeltaEvent("partial summary")]


def test_responses_model_client_incomplete_does_not_promote_text_deltas_to_items() -> (
    "None"
):
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env="DUMMY_KEY",
    )
    client = ResponsesModelClient(provider)
    events: "typing.List[ModelEvent]" = []
    stream = [
        b"event: response.output_text.delta\n",
        b'data: {"type":"response.output_text.delta","delta":"partial delta only"}\n',
        b"\n",
        b"event: response.incomplete\n",
        b'data: {"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}}\n',
        b"\n",
    ]

    with pytest.raises(ResponsesIncompleteError) as exc_info:
        client._parse_stream(stream, events.append)

    assert "- output_items_received: 0" in str(exc_info.value)
    assert exc_info.value.partial_items == ()
    assert events == [AssistantDeltaEvent("partial delta only")]


def test_responses_provider_default_stream_max_retries_is_five() -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env=None,
    )

    assert provider.effective_stream_max_retries() == 5


@pytest.mark.asyncio
async def test_responses_model_client_retries_retryable_stream_failures() -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env=None,
        stream_max_retries=1,
    )
    client = ResponsesModelClient(provider)
    send_calls: "typing.List[int]" = []

    class _RetryableFailureResponse:
        status_code = 200
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def iter_lines(self, chunk_size=1, decode_unicode=False):
            del chunk_size, decode_unicode
            yield b"event: response.failed"
            yield (
                b'data: {"type":"response.failed","response":{"error":{"code":"rate_limit_exceeded",'
                b'"message":"Please try again in 0ms."}}}'
            )
            yield b""

    class _SuccessfulResponse:
        status_code = 200
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def iter_lines(self, chunk_size=1, decode_unicode=False):
            del chunk_size, decode_unicode
            yield b"event: response.output_item.done"
            yield (
                b'data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant",'
                b'"content":[{"type":"output_text","text":"done"}]}}'
            )
            yield b""
            yield b"event: response.completed"
            yield b'data: {"type":"response.completed"}'
            yield b""

    original_send = requests.Session.send

    def fake_send(self, request, timeout, allow_redirects, **settings):
        del self, request, timeout, allow_redirects, settings
        send_calls.append(1)
        if len(send_calls) == 1:
            return _RetryableFailureResponse()
        return _SuccessfulResponse()

    requests.Session.send = fake_send
    try:
        events: "typing.List[ModelEvent]" = []
        response = await client.complete(
            Prompt(input=[UserMessage(text="hi")], tools=[]),
            events.append,
        )
    finally:
        requests.Session.send = original_send

    assert [
        item.text for item in response.items if isinstance(item, AssistantMessage)
    ] == ["done"]
    assert len(send_calls) == 2
    assert len(events) == 2
    assert events[0].kind == "stream_error"
    assert events[0].message == "Reconnecting... 1/1"
    assert events[1] == TokenCountEvent(None)


@pytest.mark.asyncio
async def test_responses_model_client_retries_truncated_stream_without_changing_session() -> (
    "None"
):
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env=None,
        stream_max_retries=1,
    )
    client = ResponsesModelClient(
        provider,
        session_id="00000000-0000-7000-8000-000000000000",
    )
    requests_seen: "typing.List[typing.Tuple[str, str]]" = []

    class _IncompleteResponse:
        status_code = 200
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def iter_lines(self, chunk_size=1, decode_unicode=False):
            del chunk_size, decode_unicode
            yield b"event: response.output_text.delta"
            yield (b'data: {"type":"response.output_text.delta","delta":"partial"}')
            yield b""

    class _SuccessfulResponse:
        status_code = 200
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def iter_lines(self, chunk_size=1, decode_unicode=False):
            del chunk_size, decode_unicode
            yield b"event: response.output_item.done"
            yield (
                b'data: {"type":"response.output_item.done","item":{"type":"message",'
                b'"role":"assistant","content":[{"type":"output_text","text":"done"}]}}'
            )
            yield b""
            yield b"event: response.completed"
            yield b'data: {"type":"response.completed"}'
            yield b""

    original_send = requests.Session.send

    def fake_send(self, request, timeout, allow_redirects, **settings):
        del self, timeout, allow_redirects, settings
        body = json.loads(request.body.decode("utf-8"))
        requests_seen.append((request.headers["session_id"], body["prompt_cache_key"]))
        if len(requests_seen) == 1:
            return _IncompleteResponse()
        return _SuccessfulResponse()

    requests.Session.send = fake_send
    try:
        events: "typing.List[ModelEvent]" = []
        response = await client.complete(
            Prompt(input=[UserMessage(text="hi")], tools=[]),
            events.append,
        )
    finally:
        requests.Session.send = original_send

    assert [
        item.text for item in response.items if isinstance(item, AssistantMessage)
    ] == ["done"]
    assert requests_seen == [
        (
            "00000000-0000-7000-8000-000000000000",
            "00000000-0000-7000-8000-000000000000",
        ),
        (
            "00000000-0000-7000-8000-000000000000",
            "00000000-0000-7000-8000-000000000000",
        ),
    ]
    stream_errors = [event for event in events if event.kind == "stream_error"]
    assert len(stream_errors) == 1
    assert stream_errors[0].message == "Reconnecting... 1/1"
    assert "response.completed" in stream_errors[0].error


@pytest.mark.asyncio
async def test_responses_model_client_does_not_retry_terminal_incomplete(monkeypatch):
    client = ResponsesModelClient(
        ResponsesProviderConfig(
            model="demo",
            provider_name="demo",
            base_url="https://example.invalid/v1",
            api_key_env=None,
            stream_max_retries=1,
        )
    )
    calls = []
    events = []

    def complete_sync(prompt, event_handler):
        calls.append(prompt)
        raise ResponsesIncompleteError(
            "terminal response.incomplete",
            [AssistantMessage("done item")],
            "max_output_tokens",
        )

    monkeypatch.setattr(client, "_complete_sync", complete_sync)
    monkeypatch.setattr(client, "_retry_delay_seconds", lambda retries: 0)
    with pytest.raises(ResponsesIncompleteError):
        await client.complete(Prompt(input=[], tools=[]), events.append)
    assert len(calls) == 1
    assert events == []


@pytest.mark.asyncio
async def test_responses_model_client_does_not_retry_context_length_failed_stream() -> (
    "None"
):
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env=None,
        stream_max_retries=5,
    )
    client = ResponsesModelClient(provider)
    send_calls: "typing.List[int]" = []

    class _ContextLengthFailureResponse:
        status_code = 200
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def iter_lines(self, chunk_size=1, decode_unicode=False):
            del chunk_size, decode_unicode
            yield b"event: response.failed"
            yield (
                b'data: {"type":"response.failed","response":{"error":{"message":'
                b"\"This model's maximum context length is 262144 tokens. However, "
                b"you requested 264568 tokens (264568 in the messages, 0 in the "
                b'completion). Please reduce the length of the messages or completion.",'
                b'"type":"context_length_exceeded"}}}'
            )
            yield b""

    original_send = requests.Session.send

    def fake_send(self, request, timeout, allow_redirects, **settings):
        del self, request, timeout, allow_redirects, settings
        send_calls.append(1)
        return _ContextLengthFailureResponse()

    requests.Session.send = fake_send
    try:
        events: "typing.List[ModelEvent]" = []
        with pytest.raises(RuntimeError, match="maximum context length"):
            await client.complete(
                Prompt(input=[UserMessage(text="hi")], tools=[]),
                events.append,
            )
    finally:
        requests.Session.send = original_send

    assert len(send_calls) == 1
    assert events == []


@pytest.mark.asyncio
async def test_responses_model_client_preserves_response_failed_error_code() -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env=None,
        stream_max_retries=5,
    )
    client = ResponsesModelClient(provider)

    class _ContextWindowFailureResponse:
        status_code = 200
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def iter_lines(self, chunk_size=1, decode_unicode=False):
            del chunk_size, decode_unicode
            yield b"event: response.failed"
            yield (
                b'data: {"type":"response.failed","response":{"error":{"message":'
                b'"Your input exceeds the context window of this model. Please adjust '
                b'your input and try again.","type":"context_length_exceeded"}}}'
            )
            yield b""

    original_send = requests.Session.send

    def fake_send(self, request, timeout, allow_redirects, **settings):
        del self, request, timeout, allow_redirects, settings
        return _ContextWindowFailureResponse()

    requests.Session.send = fake_send
    try:
        with pytest.raises(RuntimeError) as excinfo:
            await client.complete(
                Prompt(input=[UserMessage(text="hi")], tools=[]),
                NOOP_MODEL_STREAM_EVENT_HANDLER,
            )
    finally:
        requests.Session.send = original_send

    assert "- code: context_length_exceeded" in str(excinfo.value)


@pytest.mark.asyncio
async def test_responses_model_client_does_not_retry_invalid_model_output_stream() -> (
    "None"
):
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env=None,
        stream_max_retries=5,
    )
    client = ResponsesModelClient(provider)
    send_calls: "typing.List[int]" = []

    class _InvalidModelOutputFailureResponse:
        status_code = 200
        text = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def iter_lines(self, chunk_size=1, decode_unicode=False):
            del chunk_size, decode_unicode
            yield b"event: response.failed"
            yield (
                b'data: {"type":"response.failed","response":{"error":{"message":'
                b'"outcomming chat completion ended without assistant content",'
                b'"type":"model_output_invalid"}}}'
            )
            yield b""

    original_send = requests.Session.send

    def fake_send(self, request, timeout, allow_redirects, **settings):
        del self, request, timeout, allow_redirects, settings
        send_calls.append(1)
        return _InvalidModelOutputFailureResponse()

    requests.Session.send = fake_send
    try:
        events: "typing.List[ModelEvent]" = []
        with pytest.raises(RuntimeError, match="without assistant content"):
            await client.complete(
                Prompt(input=[UserMessage(text="hi")], tools=[]),
                events.append,
            )
    finally:
        requests.Session.send = original_send

    assert len(send_calls) == 1
    assert events == []


def test_responses_model_client_parses_reasoning_output_item() -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env="DUMMY_KEY",
    )
    client = ResponsesModelClient(provider)

    item = client._parse_output_item(
        {
            "type": "reasoning",
            "summary": [],
            "content": None,
            "encrypted_content": "encrypted-token",
        }
    )

    assert isinstance(item, ReasoningItem)
    assert item.serialize() == {
        "type": "reasoning",
        "summary": [],
        "content": None,
        "encrypted_content": "encrypted-token",
    }


def test_responses_model_client_parses_custom_tool_call_item() -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env="DUMMY_KEY",
    )
    client = ResponsesModelClient(provider)

    item = client._parse_output_item(
        {
            "type": "custom_tool_call",
            "call_id": "call_custom",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch",
        }
    )

    assert isinstance(item, ToolCall)
    assert item.tool_type == "custom"
    assert item.arguments == "*** Begin Patch\n*** End Patch"


def test_responses_model_client_emits_event_for_web_search_call() -> "None":
    provider = ResponsesProviderConfig(
        model="demo-model",
        provider_name="demo",
        base_url="https://example.com/v1",
        api_key_env="DUMMY_KEY",
    )
    client = ResponsesModelClient(provider)
    seen = []
    stream = [
        b"event: response.output_item.done\n",
        b'data: {"type":"response.output_item.done","item":{"type":"web_search_call","id":"ws_1","action":{"type":"search","query":"github codex","queries":["github codex"]}}}\n',
        b"\n",
        b"event: response.output_item.done\n",
        b'data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}}\n',
        b"\n",
        b"event: response.completed\n",
        b'data: {"type":"response.completed"}\n',
        b"\n",
    ]

    response = client._parse_stream(stream, seen.append)

    assert response.items == [AssistantMessage(text="done")]
    assert (
        ToolCalledEvent(
            "ws_1",
            "web_search",
            "search",
            "github codex",
            ("github codex",),
        )
        in seen
    )
