from __future__ import annotations

import json

from fastapi.testclient import TestClient
from responses_server import CompatServerConfig, ManagedResponseServer
from responses_server.tools.custom_adapter import (
    APPLY_PATCH_CHAT_DESCRIPTION,
    APPLY_PATCH_CHAT_INPUT_DESCRIPTION,
)
from tests.responses_server.fake_chat_completions_server import (
    CaptureStore,
    build_test_server as build_fake_chat_server,
    build_text_chunks,
    build_tool_call_chunks,
)


def test_responses_server_streams_text_from_chat_backend(tmp_path) -> None:
    capture_store = CaptureStore(tmp_path / "chat_capture")
    fake_chat_server = build_fake_chat_server(
        capture_store,
        build_text_chunks("Hello"),
    )
    fake_chat_server.start()

    app = ManagedResponseServer.build_app(
        CompatServerConfig(
            outcomming_base_url=f"http://127.0.0.1:{fake_chat_server.server_port}/v1",
        )
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "gpt-5.4",
                    "instructions": "Be concise.",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "hi"}],
                        }
                    ],
                    "tools": [],
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                    "stream": True,
                },
                headers={"Accept": "text/event-stream"},
            )
            status = response.status_code
            body = response.text
    finally:
        fake_chat_server.stop()

    assert status == 200
    assert "event: response.created" in body
    assert '"type": "response.output_text.delta", "delta": "Hello"' in body
    assert '"type": "response.output_item.done"' in body
    assert '"type": "message"' in body
    assert '"text": "Hello"' in body
    assert "event: response.completed" in body

    request_files = sorted((tmp_path / "chat_capture").glob("*_POST_*.json"))
    assert len(request_files) == 1
    request = json.loads(request_files[0].read_text())
    assert request["path"] == "/v1/chat/completions"
    assert request["body"]["stream"] is True
    assert request["body"]["messages"] == [
        {"role": "developer", "content": "Be concise."},
        {"role": "user", "content": "hi"},
    ]


def test_responses_server_preserves_request_model_without_default_override(
    tmp_path,
) -> None:
    capture_store = CaptureStore(tmp_path / "chat_capture")
    fake_chat_server = build_fake_chat_server(
        capture_store,
        build_text_chunks("Hello"),
    )
    fake_chat_server.start()

    app = ManagedResponseServer.build_app(
        CompatServerConfig(
            outcomming_base_url=f"http://127.0.0.1:{fake_chat_server.server_port}/v1",
        )
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "gpt-5.4",
                    "instructions": "",
                    "input": [],
                    "tools": [],
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                    "stream": True,
                },
                headers={"Accept": "text/event-stream"},
            )
            status = response.status_code
    finally:
        fake_chat_server.stop()

    assert status == 200
    request_files = sorted((tmp_path / "chat_capture").glob("*_POST_*.json"))
    assert len(request_files) == 1
    request = json.loads(request_files[0].read_text())
    assert request["body"]["model"] == "gpt-5.4"


def test_responses_server_translates_chat_tool_calls_to_incomming_items(tmp_path) -> None:
    capture_store = CaptureStore(tmp_path / "chat_capture")
    fake_chat_server = build_fake_chat_server(
        capture_store,
        build_tool_call_chunks("call_1", "echo", ['{"text":"', 'hello"}']),
    )
    fake_chat_server.start()

    app = ManagedResponseServer.build_app(
        CompatServerConfig(
            outcomming_base_url=f"http://127.0.0.1:{fake_chat_server.server_port}/v1",
        )
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "gpt-5.4",
                    "instructions": "",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "call echo"}],
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "name": "echo",
                            "description": "Echo text.",
                            "parameters": {"type": "object"},
                            "strict": False,
                        }
                    ],
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                    "stream": True,
                },
                headers={"Accept": "text/event-stream"},
            )
            status = response.status_code
            body = response.text
    finally:
        fake_chat_server.stop()

    assert status == 200
    assert '"type": "function_call"' in body
    assert '"call_id": "call_1"' in body
    assert '"name": "echo"' in body
    assert '\\"text\\":\\"hello\\"' in body


def test_responses_server_reconstructs_tool_history_for_outcomming_chat(tmp_path) -> None:
    capture_store = CaptureStore(tmp_path / "chat_capture")
    fake_chat_server = build_fake_chat_server(
        capture_store,
        build_text_chunks("done"),
    )
    fake_chat_server.start()

    app = ManagedResponseServer.build_app(
        CompatServerConfig(
            outcomming_base_url=f"http://127.0.0.1:{fake_chat_server.server_port}/v1",
        )
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "gpt-5.4",
                    "instructions": "Be concise.",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "hi"}],
                        },
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "echo",
                            "arguments": '{"text":"hello"}',
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": "hello",
                        },
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "name": "echo",
                            "description": "Echo text.",
                            "parameters": {"type": "object"},
                            "strict": False,
                        }
                    ],
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                    "stream": True,
                },
                headers={"Accept": "text/event-stream"},
            )
            status = response.status_code
    finally:
        fake_chat_server.stop()

    assert status == 200
    request_files = sorted((tmp_path / "chat_capture").glob("*_POST_*.json"))
    assert len(request_files) == 1
    request = json.loads(request_files[0].read_text())
    assert request["body"]["messages"] == [
        {"role": "developer", "content": "Be concise."},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": '{"text":"hello"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "hello",
        },
    ]


def test_responses_server_adapts_custom_tools_for_chat_backend(tmp_path) -> None:
    capture_store = CaptureStore(tmp_path / "chat_capture")
    fake_chat_server = build_fake_chat_server(
        capture_store,
        build_text_chunks("done"),
    )
    fake_chat_server.start()

    app = ManagedResponseServer.build_app(
        CompatServerConfig(
            outcomming_base_url=f"http://127.0.0.1:{fake_chat_server.server_port}/v1",
        )
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "gpt-5.4",
                    "instructions": "",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "patch it"}],
                        }
                    ],
                    "tools": [
                        {
                            "type": "custom",
                            "name": "apply_patch",
                            "description": "Use apply_patch.",
                            "format": {
                                "type": "grammar",
                                "syntax": "lark",
                                "definition": "start: PATCH",
                            },
                        }
                    ],
                    "tool_choice": {"type": "custom", "name": "apply_patch"},
                    "parallel_tool_calls": False,
                    "stream": True,
                },
                headers={"Accept": "text/event-stream"},
            )
            status = response.status_code
    finally:
        fake_chat_server.stop()

    assert status == 200
    request_files = sorted((tmp_path / "chat_capture").glob("*_POST_*.json"))
    assert len(request_files) == 1
    request = json.loads(request_files[0].read_text())
    assert request["body"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "apply_patch"},
    }
    assert request["body"]["tools"] == [
        {
            "type": "function",
            "name": "apply_patch",
            "function": {
                "name": "apply_patch",
                "description": APPLY_PATCH_CHAT_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": APPLY_PATCH_CHAT_INPUT_DESCRIPTION,
                        }
                    },
                    "required": ["input"],
                    "additionalProperties": False,
                },
                "strict": False,
            },
        }
    ]


def test_responses_server_reconstructs_custom_tool_history_for_outcomming_chat(
    tmp_path,
) -> None:
    capture_store = CaptureStore(tmp_path / "chat_capture")
    fake_chat_server = build_fake_chat_server(
        capture_store,
        build_text_chunks("done"),
    )
    fake_chat_server.start()

    app = ManagedResponseServer.build_app(
        CompatServerConfig(
            outcomming_base_url=f"http://127.0.0.1:{fake_chat_server.server_port}/v1",
        )
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "gpt-5.4",
                    "instructions": "",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "patch it"}],
                        },
                        {
                            "type": "custom_tool_call",
                            "call_id": "call_1",
                            "name": "apply_patch",
                            "input": "*** Begin Patch\n*** End Patch\n",
                        },
                        {
                            "type": "custom_tool_call_output",
                            "call_id": "call_1",
                            "name": "apply_patch",
                            "output": "patched",
                        },
                    ],
                    "tools": [
                        {
                            "type": "custom",
                            "name": "apply_patch",
                            "description": "Use apply_patch.",
                            "format": {
                                "type": "grammar",
                                "syntax": "lark",
                                "definition": "start: PATCH",
                            },
                        }
                    ],
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "stream": True,
                },
                headers={"Accept": "text/event-stream"},
            )
            status = response.status_code
    finally:
        fake_chat_server.stop()

    assert status == 200
    request_files = sorted((tmp_path / "chat_capture").glob("*_POST_*.json"))
    assert len(request_files) == 1
    request = json.loads(request_files[0].read_text())
    assert request["body"]["messages"] == [
        {"role": "user", "content": "patch it"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": (
                            '{"input":"*** Begin Patch\\n*** End Patch\\n"}'
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "patched",
        },
    ]


def test_responses_server_translates_adapted_custom_calls_back_to_custom_items(
    tmp_path,
) -> None:
    capture_store = CaptureStore(tmp_path / "chat_capture")
    fake_chat_server = build_fake_chat_server(
        capture_store,
        build_tool_call_chunks(
            "call_1",
            "apply_patch",
            ['{"input":"', "*** Begin Patch\\n*** End Patch\\n\"}"],
        ),
    )
    fake_chat_server.start()

    app = ManagedResponseServer.build_app(
        CompatServerConfig(
            outcomming_base_url=f"http://127.0.0.1:{fake_chat_server.server_port}/v1",
        )
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "gpt-5.4",
                    "instructions": "",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "patch it"}],
                        }
                    ],
                    "tools": [
                        {
                            "type": "custom",
                            "name": "apply_patch",
                            "description": "Use apply_patch.",
                            "format": {
                                "type": "grammar",
                                "syntax": "lark",
                                "definition": "start: PATCH",
                            },
                        }
                    ],
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "stream": True,
                },
                headers={"Accept": "text/event-stream"},
            )
            status = response.status_code
            body = response.text
    finally:
        fake_chat_server.stop()

    assert status == 200
    assert '"type": "custom_tool_call"' in body
    assert '"call_id": "call_1"' in body
    assert '"name": "apply_patch"' in body
    assert '"input": "*** Begin Patch\\n*** End Patch\\n"' in body


def test_responses_server_mocks_web_search_and_continues_chat(tmp_path) -> None:
    capture_store = CaptureStore(tmp_path / "chat_capture")
    fake_chat_server = build_fake_chat_server(
        capture_store,
        [
            [
                {
                    "id": "chatcmpl_mock",
                    "object": "chat.completion.chunk",
                    "model": "gpt-5.4",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "ws_1",
                                        "type": "function",
                                        "function": {
                                            "arguments": '{"query":"github codex"}'
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl_mock",
                    "object": "chat.completion.chunk",
                    "model": "gpt-5.4",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            ],
            build_text_chunks("done"),
        ],
    )
    fake_chat_server.start()

    app = ManagedResponseServer.build_app(
        CompatServerConfig(
            outcomming_base_url=f"http://127.0.0.1:{fake_chat_server.server_port}/v1",
        )
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "gpt-5.4",
                    "instructions": "Be concise.",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Search the web, then answer.",
                                }
                            ],
                        }
                    ],
                    "tools": [
                        {
                            "type": "web_search",
                            "external_web_access": True,
                        }
                    ],
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "stream": True,
                },
                headers={"Accept": "text/event-stream"},
            )
            status = response.status_code
            body = response.text
    finally:
        fake_chat_server.stop()

    assert status == 200
    assert '"type": "web_search_call"' in body
    assert '"id": "ws_1"' in body
    assert '"query": "github codex"' in body
    assert '"text": "done"' in body

    request_files = sorted((tmp_path / "chat_capture").glob("*_POST_*.json"))
    assert len(request_files) == 2
    first_request = json.loads(request_files[0].read_text())
    second_request = json.loads(request_files[1].read_text())

    assert first_request["body"]["tools"] == [
        {
            "type": "function",
            "name": "web_search",
            "function": {
                "name": "web_search",
                "description": (
                    "Mock web search tool for Responses compatibility. "
                    "Returns empty results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Primary search query.",
                        },
                        "queries": {
                            "type": "array",
                            "description": "Optional batch of search queries.",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["query"],
                },
                "strict": False,
            },
        }
    ]
    assert second_request["body"]["messages"][-2] == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "ws_1",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query":"github codex"}',
                },
            }
        ],
    }
    assert second_request["body"]["messages"][-1]["role"] == "tool"
    assert second_request["body"]["messages"][-1]["tool_call_id"] == "ws_1"
    assert json.loads(second_request["body"]["messages"][-1]["content"]) == {
        "query": "github codex",
        "queries": ["github codex"],
        "results": [],
        "mock": True,
    }
