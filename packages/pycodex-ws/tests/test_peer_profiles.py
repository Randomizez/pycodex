import asyncio
from contextlib import asynccontextmanager
import gzip
import json
import os
from pathlib import Path
import secrets
import socket
from urllib.parse import urlsplit

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from websockets.exceptions import ConnectionClosedError, InvalidStatusCode
from websockets.legacy.client import connect

from pycodex_router.client import RouterClient
from pycodex_router.protocol import decode, encode, parse_target, tunnel_url
from pycodex_router.server import create_app as create_router_app, create_tunnel_app
from tests.responses_server.fake_chat_completions_server import RunningASGITestServer
from workspace_server import WorkspaceStateStore, build_parser
from workspace_server import app as app_module
from workspace_server.owned_process import run_owned_server
from workspace_server.peer_profiles import PEER_BRIDGE_API_KEY_ENV, PeerProfileProvider

from test_router import client_target, eventually, ip_browser, login, serve, verify_device

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _reset_peer_bridge_env(monkeypatch):
    monkeypatch.delenv("PYCODEX_PEER_BRIDGE_KEY", raising=False)


async def wait_async_snapshot(client, url, predicate, timeout=10.0):
    deadline = asyncio.get_running_loop().time() + timeout
    last = None
    while asyncio.get_running_loop().time() < deadline:
        resp = await client.get(url)
        assert resp.status_code == 200, resp.text
        last = resp.json()["snapshot"]
        if predicate(last):
            return last
        await asyncio.sleep(0.02)
    raise AssertionError("timed out waiting for snapshot; last={0!r}".format(last))


def build_cli_workspace_app(
    root_dir: Path,
    config_path: Path,
    workspace_id: str = "work",
    password=None,
    launch_profile=None,
    toolset=(),
    enable_board: bool = True,
):
    work_dir = root_dir / "project"
    work_dir.mkdir(parents=True, exist_ok=True)
    ws_cfg = root_dir / "workspaces.json"
    ws_cfg.write_text(
        json.dumps(
            {
                "workspaces": [
                    {
                        "id": workspace_id,
                        "work_dir": str(work_dir),
                        "state_path": str(root_dir / ("state." + workspace_id)),
                        "enable_board": enable_board,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    argv = [
        "--workspace-config",
        str(ws_cfg),
        "--config",
        str(config_path),
    ]
    if toolset is not None:
        argv.extend(["--toolset", *toolset])
    if launch_profile is not None:
        argv.extend(["--profile", launch_profile])
    if password is not None:
        argv.extend(["--password", password])
    args = build_parser().parse_args(argv)
    return app_module.build_app(args), ws_cfg, work_dir


@asynccontextmanager
async def serve_router_node(app, target: str, name: str, identity_key=None):
    """Start a workspace and verify it through the advertised router API."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    client = RouterClient(
        app,
        target,
        name,
        port,
        app.state.router_inventory,
        refresh_seconds=0.05,
        reconnect_seconds=0.05,
        identity_key=identity_key,
    )
    app.state.router_client = client
    async with serve(app, listener=listener) as local_url:
        await client.start()
        try:
            await eventually(lambda: client.connected)
            async with ip_browser(client.api_url) as admin:
                await login(admin)
                await verify_device(admin, client.node_id)
            yield client, local_url
        finally:
            await client.close()


async def test_peer_node_discovery_and_profile_catalog(tmp_path):
    cfg_a = tmp_path / "node_a" / "config.toml"
    cfg_a.parent.mkdir(parents=True)
    cfg_a.write_text(
        'profile = "fast"\nmodel = "base-a"\n'
        '[profiles.fast]\nmodel = "fast-a"\n'
        '[profiles.deep]\nmodel = "deep-a"\n',
        encoding="utf-8",
    )
    cfg_b = tmp_path / "node_b" / "config.toml"
    cfg_b.parent.mkdir(parents=True)
    cfg_b.write_text(
        'model = "base-b"\n[profiles.local]\nmodel = "local-b"\n', encoding="utf-8"
    )
    cfg_c = tmp_path / "node_c" / "config.toml"
    cfg_c.parent.mkdir(parents=True)
    cfg_c.write_text('model = "base-c"\n', encoding="utf-8")

    app_b_standalone, _, _ = build_cli_workspace_app(tmp_path / "standalone_b", cfg_b)
    with TestClient(app_b_standalone) as standalone:
        catalog = standalone.get("/w/work/api/sessions").json()
        assert catalog["profiles"] == ["local"]
        assert "router_connected" not in catalog
        assert "peer_nodes" not in catalog
        offline_peer = standalone.get(
            "/w/work/api/peer-nodes/" + ("a" * 52) + "/profiles"
        )
        assert offline_peer.status_code == 503

    gate = create_router_app("router-password", aliases_path=tmp_path / "aliases.json")
    tunnel = create_tunnel_app(gate.state.router)
    async with serve(gate) as router_addr, serve(tunnel) as tunnel_addr:
        gate.state.router.api_port = urlsplit(router_addr).port
        target = client_target(tunnel_addr, gate.state.router.registration_key)

        app_a, _, _ = build_cli_workspace_app(tmp_path / "ws_a", cfg_a)
        app_b, _, _ = build_cli_workspace_app(tmp_path / "ws_b", cfg_b)
        app_c, _, _ = build_cli_workspace_app(tmp_path / "ws_c", cfg_c)

        async with serve_router_node(app_a, target, "host-a") as (
            client_a,
            a_url,
        ), serve_router_node(app_b, target, "host-b") as (
            client_b,
            b_url,
        ), serve_router_node(app_c, target, "host-c") as (
            client_c,
            c_url,
        ):
            async with ip_browser(router_addr) as admin:
                await login(admin)
                rename = await admin.post(
                    "/api/nodes/" + client_a.node_id + "/alias",
                    json={"alias": "GPU Server A"},
                )
                assert rename.status_code == 200
                reorder = await admin.post(
                    "/api/nodes/order",
                    json={
                        "order": [
                            client_c.node_id,
                            client_a.node_id,
                            client_b.node_id,
                        ]
                    },
                )
                assert reorder.status_code == 200

            # Standby /tunnel/data and warm /tunnel/peer channels pre-warm on registration
            await eventually(
                lambda: len(gate.state.router.nodes[client_a.node_id].standby) == 2
                and len(client_b._peer_pool) >= 1
            )
            warm_peer_ch = client_b._peer_pool[0]

            async with httpx.AsyncClient(
                base_url=b_url, trust_env=False
            ) as browser_b:
                sessions_resp = await browser_b.get("/w/work/api/sessions")
                assert sessions_resp.status_code == 200
                data = sessions_resp.json()
                assert data["router_connected"] is True
                assert data["profiles"] == ["local"]
                assert "peer_nodes" not in data

                peers_only = await browser_b.get("/w/work/api/peer-nodes")
                assert peers_only.status_code == 200
                expected_peers = [
                    {
                        "id": client_c.node_id,
                        "name": "{0}/127.0.0.1:{1}".format(
                            client_c.name, urlsplit(c_url).port
                        ),
                        "alias": "",
                    },
                    {
                        "id": client_a.node_id,
                        "name": "{0}/127.0.0.1:{1}".format(
                            client_a.name, urlsplit(a_url).port
                        ),
                        "alias": "GPU Server A",
                    },
                ]
                assert peers_only.json() == {"peer_nodes": expected_peers}
                # Warm /tunnel/peer WebSocket was reused across list_peers and returned to pool!
                assert client_b._peer_pool[0] is warm_peer_ch

                peer_catalog = await browser_b.get(
                    "/w/work/api/peer-nodes/{0}/profiles".format(client_a.node_id)
                )
                assert peer_catalog.status_code == 200
                assert peer_catalog.json() == {
                    "profiles": ["deep", "fast"],
                    "default_profile": "fast",
                }
                assert client_b._peer_pool[0] is warm_peer_ch
                # Consumed standby channel on Node A automatically replenishes back to 2
                await eventually(
                    lambda: len(gate.state.router.nodes[client_a.node_id].standby) == 2
                )

                (cfg_a.parent / "extra.config.toml").write_text(
                    'model = "extra-a"\n', encoding="utf-8"
                )
                updated_catalog = await browser_b.get(
                    "/w/work/api/peer-nodes/{0}/profiles".format(client_a.node_id)
                )
                assert updated_catalog.status_code == 200
                assert updated_catalog.json() == {
                    "profiles": ["deep", "extra", "fast"],
                    "default_profile": "fast",
                }


async def test_peer_compat_profile_turn_local_tools_and_key_rotation_recovery(
    tmp_path, monkeypatch
):
    key_env = "PYCODEX_TEST_PEER_HOST_A_KEY"
    monkeypatch.delenv(key_env, raising=False)
    active_key = {"expected": "valid-key-v1"}
    captured_requests = []

    async def chat_completions(request):
        auth = request.headers.get("authorization", "")
        body = json.loads((await request.body()).decode("utf-8"))
        captured_requests.append(
            {
                "auth": auth,
                "custom_header": request.headers.get("x-node-a-secret", ""),
                "body": body,
            }
        )
        if auth != "Bearer " + active_key["expected"]:
            return JSONResponse(
                {"error": {"message": "upstream key expired"}}, status_code=401
            )
        messages = body.get("messages", [])
        has_tool_result = any(msg.get("role") == "tool" for msg in messages)
        if not has_tool_result:
            chunk = {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_local_1",
                                    "type": "function",
                                    "function": {
                                        "name": "exec_command",
                                        "arguments": json.dumps(
                                            {"cmd": "pwd && cat local_marker.txt"}
                                        ),
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        else:
            tool_msg = [m for m in messages if m.get("role") == "tool"][-1]
            tool_payload = json.loads(str(tool_msg.get("content", "{}")))
            marker_line = (
                str(tool_payload.get("output", "")).strip().splitlines()[-1]
            )
            chunk = {
                "id": "chatcmpl-2",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "Peer reply using {0}: {1}".format(
                                active_key["expected"],
                                marker_line,
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 6,
                    "total_tokens": 18,
                },
            }

        async def stream():
            yield "data: " + json.dumps(chunk) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    downstream = RunningASGITestServer(
        Starlette(
            routes=[
                Route("/v1/chat/completions", chat_completions, methods=["POST"])
            ]
        )
    )
    downstream.start()
    try:
        node_a_home = tmp_path / "node_a_home"
        node_a_home.mkdir()
        (node_a_home / ".env").write_text(
            key_env + "=valid-key-v1\n", encoding="utf-8"
        )
        cfg_a = node_a_home / "config.toml"
        cfg_a.write_text(
            'model = "chat-default-a"\n'
            'model_provider = "corp"\n'
            'compat_api = "chat_completions"\n'
            'base_instructions = "Instructions from Node A."\n'
            '[model_providers.corp]\n'
            'base_url = "{0}/v1"\n'
            'env_key = "{1}"\n'
            'stream_max_retries = 0\n'
            'http_headers = {{"X-Node-A-Secret" = "only-on-a"}}\n'
            '[profiles.specialist]\n'
            'model = "chat-specialist-a"\n'
            'base_instructions = "Specialist instructions from Node A."\n'.format(
                downstream.base_url, key_env
            ),
            encoding="utf-8",
        )

        node_b_home = tmp_path / "node_b_home"
        node_b_home.mkdir()
        cfg_b = node_b_home / "config.toml"
        cfg_b.write_text('model = "unused-local-b"\n', encoding="utf-8")

        app_a, _, work_a = build_cli_workspace_app(
            tmp_path / "ws_a", cfg_a, toolset=("exec_command",)
        )
        (work_a / "AGENTS.md").write_text(
            "NODE_A_PROJECT_DOC_MUST_NOT_LEAK\n", encoding="utf-8"
        )
        (work_a / "local_marker.txt").write_text(
            "FROM_NODE_A_WRONG\n", encoding="utf-8"
        )

        app_b, _, work_b = build_cli_workspace_app(
            tmp_path / "ws_b", cfg_b, toolset=("exec_command",)
        )
        (work_b / "AGENTS.md").write_text(
            "CLIENT_B_LOCAL_AGENTS_DOC\n", encoding="utf-8"
        )
        (work_b / "local_marker.txt").write_text(
            "MARKER_ON_CLIENT_B\n", encoding="utf-8"
        )

        gate = create_router_app("router-password")
        tunnel = create_tunnel_app(gate.state.router)
        async with serve(gate) as router_addr, serve(tunnel) as tunnel_addr:
            gate.state.router.api_port = urlsplit(router_addr).port
            target = client_target(tunnel_addr, gate.state.router.registration_key)
            async with serve_router_node(app_a, target, "node-a") as (
                client_a,
                _a_url,
            ), serve_router_node(app_b, target, "node-b") as (
                _client_b,
                b_url,
            ):
                peer_profile_id = "peer:{0}:specialist".format(client_a.node_id)
                async with httpx.AsyncClient(
                    base_url=b_url, trust_env=False
                ) as browser_b:
                    created = await browser_b.post(
                        "/w/work/api/sessions",
                        json={
                            "profile": peer_profile_id,
                            "title": "Remote specialist",
                            "prompt": "Check local marker",
                        },
                    )
                    assert created.status_code == 200, created.text
                    session_id = created.json()["session_id"]
                    assert created.json()["sessions"][-1]["profile"] == peer_profile_id

                    snap = await wait_async_snapshot(
                        browser_b,
                        "/w/work/api/session?session_id=" + session_id,
                        lambda s: s["turns"]
                        and s["turns"][-1]["response"].startswith("Peer reply using"),
                    )
                    assert (
                        snap["turns"][-1]["response"]
                        == "Peer reply using valid-key-v1: MARKER_ON_CLIENT_B"
                    )

                    cache_dir = node_b_home / "peer-profiles-cache"
                    cache_files = list(cache_dir.glob("*.json"))
                    assert len(cache_files) == 1
                    cache_raw = cache_files[0].read_text(encoding="utf-8")
                    assert "valid-key-v1" not in cache_raw

                    first_req = captured_requests[0]
                    assert first_req["auth"] == "Bearer valid-key-v1"
                    assert first_req["custom_header"] == "only-on-a"
                    assert first_req["body"]["model"] == "chat-specialist-a"
                    serialized_msgs = json.dumps(first_req["body"]["messages"])
                    assert "Specialist instructions from Node A." in serialized_msgs
                    assert "CLIENT_B_LOCAL_AGENTS_DOC" in serialized_msgs
                    assert str(work_b) in serialized_msgs
                    assert "NODE_A_PROJECT_DOC_MUST_NOT_LEAK" not in serialized_msgs

                    # Simulate key expiration on upstream, then key rotation in Node A's .env!
                    active_key["expected"] = "rotated-key-v2"
                    fail_send = await browser_b.post(
                        "/w/work/api/session/message",
                        json={
                            "session_id": session_id,
                            "prompt": "Try with expired key",
                        },
                    )
                    assert fail_send.status_code == 200
                    failed_snap = await wait_async_snapshot(
                        browser_b,
                        "/w/work/api/session?session_id=" + session_id,
                        lambda s: len(s["turns"]) >= 2
                        and s["turns"][-1]["status"] == "error",
                    )
                    assert "upstream key expired" in failed_snap["turns"][-1]["error"]

                    (node_a_home / ".env").write_text(
                        key_env + "=rotated-key-v2\n", encoding="utf-8"
                    )
                    retry_send = await browser_b.post(
                        "/w/work/api/session/message",
                        json={
                            "session_id": session_id,
                            "prompt": "Retry after key rotation",
                        },
                    )
                    assert retry_send.status_code == 200
                    recovered_snap = await wait_async_snapshot(
                        browser_b,
                        "/w/work/api/session?session_id=" + session_id,
                        lambda s: len(s["turns"]) >= 3
                        and s["turns"][-1]["response"].startswith(
                            "Peer reply using rotated-key-v2:"
                        ),
                    )
                    assert (
                        recovered_snap["turns"][-1]["response"]
                        == "Peer reply using rotated-key-v2: MARKER_ON_CLIENT_B"
                    )
    finally:
        downstream.stop()


async def test_peer_native_openai_responses_preserves_lite_and_provider_headers(
    tmp_path, monkeypatch
):
    openai_key_env = "PYCODEX_TEST_NATIVE_OPENAI_KEY"
    header_env = "PYCODEX_TEST_NATIVE_HEADER_ENV"
    monkeypatch.delenv(openai_key_env, raising=False)
    monkeypatch.delenv(header_env, raising=False)
    captured = []

    async def native_responses(request):
        body = json.loads((await request.body()).decode("utf-8"))
        captured.append(
            {
                "query": dict(request.query_params),
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "body": body,
            }
        )
        events = [
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Native OpenAI OK"}],
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_native_1",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 4,
                        "total_tokens": 16,
                    },
                },
            },
        ]

        async def sse():
            for item in events:
                yield "data: " + json.dumps(item) + "\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    downstream = RunningASGITestServer(
        Starlette(routes=[Route("/v1/responses", native_responses, methods=["POST"])])
    )
    downstream.start()
    try:
        node_a_home = tmp_path / "native_a"
        node_a_home.mkdir()
        (node_a_home / ".env").write_text(
            "{0}=sk-native-secret\n{1}=env-header-secret\n".format(
                openai_key_env, header_env
            ),
            encoding="utf-8",
        )
        cfg_a = node_a_home / "config.toml"
        cfg_a.write_text(
            'model = "gpt-6-astra"\n'
            'model_provider = "native"\n'
            '[model_providers.native]\n'
            'base_url = "{0}/v1"\n'
            'env_key = "{1}"\n'
            'stream_max_retries = 0\n'
            'query_params = {{"api-version" = "2026-03-01"}}\n'
            'http_headers = {{"X-Static-Gateway" = "gw-1"}}\n'
            'env_http_headers = {{"X-Dynamic-Gateway" = "{2}"}}\n'.format(
                downstream.base_url, openai_key_env, header_env
            ),
            encoding="utf-8",
        )
        cfg_b = tmp_path / "native_b" / "config.toml"
        cfg_b.parent.mkdir()
        cfg_b.write_text('model = "unused"\n', encoding="utf-8")

        app_a, _, _ = build_cli_workspace_app(tmp_path / "ws_native_a", cfg_a)
        app_b, _, _ = build_cli_workspace_app(tmp_path / "ws_native_b", cfg_b)

        gate = create_router_app("router-password")
        tunnel = create_tunnel_app(gate.state.router)
        async with serve(gate) as router_addr, serve(tunnel) as tunnel_addr:
            gate.state.router.api_port = urlsplit(router_addr).port
            target = client_target(tunnel_addr, gate.state.router.registration_key)
            async with serve_router_node(app_a, target, "native-a") as (
                client_a,
                _a_url,
            ), serve_router_node(app_b, target, "native-b") as (
                _client_b,
                b_url,
            ):
                async with httpx.AsyncClient(
                    base_url=b_url, trust_env=False
                ) as browser_b:
                    created = await browser_b.post(
                        "/w/work/api/sessions",
                        json={
                            "profile": "peer:" + client_a.node_id,
                            "prompt": "Hello native",
                        },
                    )
                    assert created.status_code == 200, created.text
                    session_id = created.json()["session_id"]
                    await wait_async_snapshot(
                        browser_b,
                        "/w/work/api/session?session_id=" + session_id,
                        lambda s: s["turns"]
                        and s["turns"][-1]["response"] == "Native OpenAI OK",
                    )
                    assert len(captured) == 1
                    req = captured[0]
                    assert req["query"] == {"api-version": "2026-03-01"}
                    assert req["headers"]["authorization"] == "Bearer sk-native-secret"
                    assert req["headers"]["x-static-gateway"] == "gw-1"
                    assert req["headers"]["x-dynamic-gateway"] == "env-header-secret"
                    assert (
                        req["headers"].get("x-openai-internal-codex-responses-lite")
                        == "true"
                    )
                    assert req["body"]["model"] == "gpt-6-astra"
    finally:
        downstream.stop()


async def test_peer_pyagy_profile_executes_local_tools_and_forwards_web_search(
    tmp_path, monkeypatch
):
    agy_key_env = "PYCODEX_TEST_PEER_AGY_KEY"
    monkeypatch.delenv(agy_key_env, raising=False)
    seen_gemini_calls = []
    work_b_holder = {}

    async def gemini_endpoint(request):
        model_path = request.path_params["model_path"]
        body = json.loads((await request.body()).decode("utf-8"))
        seen_gemini_calls.append(
            {
                "model_path": model_path,
                "api_key": request.headers.get("x-goog-api-key", ""),
                "body": body,
            }
        )
        if any("googleSearch" in tool for tool in body.get("tools", [])):
            grounded = {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "Grounded answer from Node A search."}
                            ],
                        },
                        "groundingMetadata": {
                            "groundingChunks": [
                                {
                                    "web": {
                                        "title": "PyCodex Docs",
                                        "uri": "https://example.com/pycodex",
                                    }
                                }
                            ]
                        },
                    }
                ]
            }

            async def grounded_sse():
                yield "data: " + json.dumps(grounded) + "\n\n"

            return StreamingResponse(
                grounded_sse(), media_type="text/event-stream"
            )

        contents = body.get("contents", [])
        func_responses = [
            part["functionResponse"]
            for content in contents
            for part in content.get("parts", [])
            if "functionResponse" in part
        ]
        if len(func_responses) == 0:
            target_file = str(work_b_holder["path"] / "agy_local.txt")
            payload = {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "write_file",
                                        "args": {
                                            "Path": target_file,
                                            "Content": "written locally on Client B",
                                            "Overwrite": True,
                                        },
                                    }
                                }
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        elif len(func_responses) == 1:
            payload = {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "search_web",
                                        "args": {
                                            "query": "pycodex router peer profiles",
                                            "domain": "example.com",
                                        },
                                    }
                                }
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        else:
            search_output = func_responses[-1]["response"]
            payload = {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": "AGY finished: "
                                    + json.dumps(search_output, ensure_ascii=False)
                                }
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ]
            }

        async def sse():
            yield "data: " + json.dumps(payload) + "\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    downstream = RunningASGITestServer(
        Starlette(
            routes=[
                Route(
                    "/v1beta/models/{model_path:path}",
                    gemini_endpoint,
                    methods=["POST"],
                )
            ]
        )
    )
    downstream.start()
    try:
        node_a_home = tmp_path / "agy_a"
        node_a_home.mkdir()
        (node_a_home / ".env").write_text(
            agy_key_env + "=agy-gemini-secret\n", encoding="utf-8"
        )
        cfg_a = node_a_home / "config.toml"
        cfg_a.write_text(
            'model = "base-unused"\n'
            '[profiles.agy]\n'
            'harness = "pyagy"\n'
            'model = "gemini-3.1-pro-high"\n'
            'model_provider = "google_agy"\n'
            'compat_api = "gemini"\n'
            '[profiles.agy.model_providers.google_agy]\n'
            'base_url = "{0}/v1beta"\n'
            'env_key = "{1}"\n'
            'stream_max_retries = 0\n'.format(downstream.base_url, agy_key_env),
            encoding="utf-8",
        )
        cfg_b = tmp_path / "agy_b" / "config.toml"
        cfg_b.parent.mkdir()
        cfg_b.write_text('model = "unused"\n', encoding="utf-8")

        app_a, _, work_a = build_cli_workspace_app(
            tmp_path / "ws_agy_a", cfg_a, toolset=None
        )
        app_b, _, work_b = build_cli_workspace_app(
            tmp_path / "ws_agy_b", cfg_b, toolset=None
        )
        work_b_holder["path"] = work_b

        gate = create_router_app("router-password")
        tunnel = create_tunnel_app(gate.state.router)
        async with serve(gate) as router_addr, serve(tunnel) as tunnel_addr:
            gate.state.router.api_port = urlsplit(router_addr).port
            target = client_target(tunnel_addr, gate.state.router.registration_key)
            async with serve_router_node(app_a, target, "agy-host-a") as (
                client_a,
                _a_url,
            ), serve_router_node(app_b, target, "agy-client-b") as (
                _client_b,
                b_url,
            ):
                async with httpx.AsyncClient(
                    base_url=b_url, trust_env=False
                ) as browser_b:
                    created = await browser_b.post(
                        "/w/work/api/sessions",
                        json={
                            "profile": "peer:{0}:agy".format(client_a.node_id),
                            "prompt": "Write a local file and search the web",
                        },
                    )
                    assert created.status_code == 200, created.text
                    session_id = created.json()["session_id"]
                    snap = await wait_async_snapshot(
                        browser_b,
                        "/w/work/api/session?session_id=" + session_id,
                        lambda s: s["turns"]
                        and s["turns"][-1]["response"].startswith("AGY finished:"),
                    )
                    assert (work_b / "agy_local.txt").read_text(encoding="utf-8") == (
                        "written locally on Client B"
                    )
                    assert not (work_a / "agy_local.txt").exists()
                    assert (
                        "Grounded answer from Node A search."
                        in snap["turns"][-1]["response"]
                    )
                    assert (
                        "https://example.com/pycodex" in snap["turns"][-1]["response"]
                    )
                    assert all(
                        call["api_key"] == "agy-gemini-secret"
                        for call in seen_gemini_calls
                    )
    finally:
        downstream.stop()


async def test_broken_local_config_on_client_b_still_allows_selecting_peer_profile(
    tmp_path,
):
    cfg_a = tmp_path / "ok_a" / "config.toml"
    cfg_a.parent.mkdir()
    cfg_a.write_text('model = "healthy-a"\n', encoding="utf-8")

    cfg_b = tmp_path / "broken_b" / "config.toml"
    cfg_b.parent.mkdir()
    cfg_b.write_text(
        'model = "local-b"\n[profiles.dup]\nmodel = "one"\n',
        encoding="utf-8",
    )
    (cfg_b.parent / "dup.config.toml").write_text(
        'model = "two"\n', encoding="utf-8"
    )

    app_a, _, _ = build_cli_workspace_app(tmp_path / "ws_ok_a", cfg_a)
    app_b, _, work_b = build_cli_workspace_app(tmp_path / "ws_broken_b", cfg_b)

    gate = create_router_app("router-password")
    tunnel = create_tunnel_app(gate.state.router)
    async with serve(gate) as router_addr, serve(tunnel) as tunnel_addr:
        gate.state.router.api_port = urlsplit(router_addr).port
        target = client_target(tunnel_addr, gate.state.router.registration_key)
        async with serve_router_node(app_a, target, "healthy-a") as (
            client_a,
            _a_url,
        ), serve_router_node(app_b, target, "broken-b") as (
            client_b,
            _b_url,
        ):
            async with ip_browser(router_addr) as router_browser:
                await login(router_browser)
                b_prefix = "/n/{0}/w/work".format(client_b.node_id)
                initial_view = await router_browser.get(
                    b_prefix + "/api/session?workspace=true"
                )
                assert initial_view.status_code == 200
                degraded_id = initial_view.json()["session_id"]

                degraded_send = await router_browser.post(
                    b_prefix + "/api/session/message",
                    json={"session_id": degraded_id, "prompt": "Hello"},
                )
                assert degraded_send.status_code == 400
                assert "dup" in degraded_send.json()["error"]

                catalog = await router_browser.get(b_prefix + "/api/sessions")
                assert catalog.status_code == 200
                body = catalog.json()
                assert body["router_connected"] is True
                assert "dup" in body["local_profile_error"]

                peers_resp = await router_browser.get(b_prefix + "/api/peer-nodes")
                assert peers_resp.status_code == 200
                assert [n["id"] for n in peers_resp.json()["peer_nodes"]] == [client_a.node_id]

                local_fail = await router_browser.post(
                    b_prefix + "/api/sessions", json={}
                )
                assert local_fail.status_code == 400

                peer_ok = await router_browser.post(
                    b_prefix + "/api/sessions",
                    json={"profile": "peer:" + client_a.node_id, "title": "Peer Tab"},
                )
                assert peer_ok.status_code == 200
                assert peer_ok.json()["snapshot"]["model"] == "healthy-a:default"

        state_store = WorkspaceStateStore(
            tmp_path / "ws_broken_b" / "state.work.pycodex-ws.json"
        )
        saved_before = state_store.load_tabs()
        assert len(saved_before) == 2
        assert "profile" not in saved_before[0]
        assert [t.get("profile") for t in saved_before] == [
            None,
            "peer:" + client_a.node_id + "@base",
        ]

        app_b_mixed, _, _ = build_cli_workspace_app(tmp_path / "ws_broken_b", cfg_b)
        async with serve_router_node(app_a, target, "healthy-a") as (
            _ca2,
            _a2,
        ), serve_router_node(app_b_mixed, target, "broken-b-2") as (
            _client_b2,
            b2_url,
        ):
            assert state_store.load_tabs() == saved_before
            async with httpx.AsyncClient(
                base_url=b2_url, trust_env=False
            ) as b2_browser:
                listed = (await b2_browser.get("/w/work/api/sessions")).json()[
                    "sessions"
                ]
                assert [t["profile"] for t in listed] == [
                    None,
                    "peer:" + client_a.node_id + "@base",
                ]
                (cfg_b.parent / "dup.config.toml").unlink()
                cfg_b.write_text(
                    'profile = "office"\nmodel = "base-local-b"\n'
                    '[profiles.office]\nmodel = "office-local-b"\n',
                    encoding="utf-8",
                )
                promoted = await b2_browser.post(
                    "/w/work/api/session/message",
                    json={
                        "session_id": listed[0]["id"],
                        "prompt": "Staged after local fix",
                        "trigger": False,
                    },
                )
                assert promoted.status_code == 200, promoted.text
                assert promoted.json()["snapshot"]["model"] == "office-local-b:default"
                assert str(work_b) in promoted.json()["snapshot"]["cwd"]
                after_promote = (
                    await b2_browser.get("/w/work/api/sessions")
                ).json()["sessions"]
                assert after_promote[0]["profile"] == "office"


async def test_offline_peer_session_restore_and_persistent_node_identity(tmp_path):
    async def chat_completions(_request):
        chunk = {
            "id": "chatcmpl-persist",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Reply across restart"},
                    "finish_reason": "stop",
                }
            ],
        }

        async def stream():
            yield "data: " + json.dumps(chunk) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    downstream = RunningASGITestServer(
        Starlette(
            routes=[
                Route("/v1/chat/completions", chat_completions, methods=["POST"])
            ]
        )
    )
    downstream.start()
    try:
        node_a_home = tmp_path / "persist_a"
        node_a_home.mkdir()
        (node_a_home / ".env").write_text("PERSIST_A_KEY=ok\n", encoding="utf-8")
        cfg_a = node_a_home / "config.toml"
        cfg_a.write_text(
            'model = "persist-model"\n'
            'model_provider = "p"\n'
            'compat_api = "chat_completions"\n'
            '[model_providers.p]\n'
            'base_url = "{0}/v1"\n'
            'env_key = "PERSIST_A_KEY"\n'
            'stream_max_retries = 0\n'.format(downstream.base_url),
            encoding="utf-8",
        )
        node_b_home = tmp_path / "persist_b"
        node_b_home.mkdir()
        cfg_b = node_b_home / "config.toml"
        cfg_b.write_text('model = "local-b"\n', encoding="utf-8")

        gate = create_router_app("router-password")
        tunnel = create_tunnel_app(gate.state.router)
        async with serve(gate) as router_addr, serve(tunnel) as tunnel_addr:
            gate.state.router.api_port = urlsplit(router_addr).port
            target = client_target(tunnel_addr, gate.state.router.registration_key)

            def make_app_a():
                app, _, _ = build_cli_workspace_app(tmp_path / "owned_ws_a", cfg_a)
                return app

            app_a_holder = {}

            def build_a():
                app_a_holder["app"] = make_app_a()
                return app_a_holder["app"]

            r_a1, w_a1 = os.pipe()
            task_a1 = asyncio.get_running_loop().run_in_executor(
                None, run_owned_server, build_a, "127.0.0.1", 0, r_a1, target
            )
            try:
                await eventually(
                    lambda: (
                        "app" in app_a_holder
                        and getattr(app_a_holder["app"].state, "router_client", None)
                        is not None
                        and app_a_holder["app"].state.router_client.connected
                    )
                    or task_a1.done()
                )
                if task_a1.done():
                    task_a1.result()
                first_node_a_id = app_a_holder["app"].state.router_client.node_id
                async with ip_browser(router_addr) as admin:
                    await login(admin)
                    await verify_device(admin, first_node_a_id)

                app_b1, _, _ = build_cli_workspace_app(tmp_path / "owned_ws_b", cfg_b)
                async with serve_router_node(app_b1, target, "node-b") as (
                    _client_b1,
                    b1_url,
                ):
                    async with httpx.AsyncClient(
                        base_url=b1_url, trust_env=False
                    ) as browser_b:
                        initial_tabs = (
                            await browser_b.get("/w/work/api/sessions")
                        ).json()["sessions"]
                        created = await browser_b.post(
                            "/w/work/api/sessions",
                            json={
                                "profile": "peer:" + first_node_a_id,
                                "title": "Persisted peer tab",
                                "prompt": "First turn before restart",
                            },
                        )
                        assert created.status_code == 200
                        peer_session_id = created.json()["session_id"]
                        await wait_async_snapshot(
                            browser_b,
                            "/w/work/api/session?session_id=" + peer_session_id,
                            lambda s: s["turns"]
                            and s["turns"][-1]["response"] == "Reply across restart",
                        )
                        deleted = await browser_b.delete(
                            "/w/work/api/sessions/" + initial_tabs[0]["id"]
                        )
                        assert deleted.status_code == 200
            finally:
                os.close(w_a1)
                try:
                    await asyncio.wait_for(asyncio.shield(task_a1), 10)
                finally:
                    os.close(r_a1)

            app_b2, _, _ = build_cli_workspace_app(tmp_path / "owned_ws_b", cfg_b)
            with TestClient(app_b2) as offline_browser:
                snap_resp = offline_browser.get("/w/work/api/session")
                assert snap_resp.status_code == 200
                restored = snap_resp.json()
                assert len(restored["sessions"]) == 1
                assert (
                    restored["sessions"][0]["profile"]
                    == "peer:" + first_node_a_id + "@base"
                )
                assert restored["sessions"][0]["title"] == "Persisted peer tab"
                assert (
                    restored["snapshot"]["turns"][0]["response"]
                    == "Reply across restart"
                )

            app_a_holder.clear()
            r_a2, w_a2 = os.pipe()
            task_a2 = asyncio.get_running_loop().run_in_executor(
                None, run_owned_server, build_a, "127.0.0.1", 0, r_a2, target
            )
            try:
                await eventually(
                    lambda: (
                        "app" in app_a_holder
                        and getattr(app_a_holder["app"].state, "router_client", None)
                        is not None
                        and app_a_holder["app"].state.router_client.connected
                    )
                    or task_a2.done()
                )
                if task_a2.done():
                    task_a2.result()
                assert app_a_holder["app"].state.router_client.node_id == first_node_a_id

                app_b3, _, _ = build_cli_workspace_app(tmp_path / "owned_ws_b", cfg_b)
                async with serve_router_node(app_b3, target, "node-b") as (
                    _client_b3,
                    b3_url,
                ):
                    async with httpx.AsyncClient(
                        base_url=b3_url, trust_env=False
                    ) as browser_b3:
                        sessions = (
                            await browser_b3.get("/w/work/api/session")
                        ).json()["sessions"]
                        restored_id = sessions[0]["id"]
                        second = await browser_b3.post(
                            "/w/work/api/session/message",
                            json={
                                "session_id": restored_id,
                                "prompt": "Second turn after Node A & B restart",
                            },
                        )
                        assert second.status_code == 200
                        snap2 = await wait_async_snapshot(
                            browser_b3,
                            "/w/work/api/session?session_id=" + restored_id,
                            lambda s: len(s["turns"]) >= 2
                            and s["turns"][-1]["response"] == "Reply across restart",
                        )
                        assert len(snap2["turns"]) == 2
            finally:
                os.close(w_a2)
                try:
                    await asyncio.wait_for(asyncio.shield(task_a2), 10)
                finally:
                    os.close(r_a2)
    finally:
        downstream.stop()


async def test_peer_security_boundaries(tmp_path):
    cfg_a = tmp_path / "sec_a" / "config.toml"
    cfg_a.parent.mkdir()
    cfg_a.write_text('model = "sec-a"\n', encoding="utf-8")
    cfg_b = tmp_path / "sec_b" / "config.toml"
    cfg_b.parent.mkdir()
    cfg_b.write_text('model = "sec-b"\n', encoding="utf-8")

    app_a, _, _ = build_cli_workspace_app(tmp_path / "ws_sec_a", cfg_a, password=None)
    app_b, _, _ = build_cli_workspace_app(tmp_path / "ws_sec_b", cfg_b, password=None)

    gate = create_router_app("router-password")
    tunnel = create_tunnel_app(gate.state.router)
    async with serve(gate) as router_addr, serve(tunnel) as tunnel_addr:
        gate.state.router.api_port = urlsplit(router_addr).port
        target = client_target(tunnel_addr, gate.state.router.registration_key)
        async with serve_router_node(app_a, target, "sec-a") as (
            client_a,
            a_url,
        ), serve_router_node(app_b, target, "sec-b") as (
            client_b,
            _b_url,
        ):
            async with httpx.AsyncClient(
                base_url=a_url, trust_env=False
            ) as direct_http:
                for method, path in (
                    ("GET", "/api/peer-profiles"),
                    ("GET", "/api/peer-profiles/default"),
                    ("GET", "/api/peer-proxy"),
                    ("POST", "/api/peer-proxy"),
                ):
                    resp = await direct_http.request(
                        method,
                        path,
                        headers={"X-Peer-Target-Url": "http://127.0.0.1:9/x"},
                        json={"query": "x"},
                    )
                    assert resp.status_code == 403

            async with ip_browser(router_addr) as router_browser:
                await login(router_browser)
                proxied = await router_browser.get(
                    "/n/{0}/api/peer-profiles".format(client_a.node_id)
                )
                assert proxied.status_code == 403
                proxied_proxy = await router_browser.post(
                    "/n/{0}/api/peer-proxy".format(client_a.node_id),
                    headers={"X-Peer-Target-Url": "http://127.0.0.1:9/x"},
                    json={"query": "x"},
                )
                assert proxied_proxy.status_code == 403

            address, reg_key = parse_target(target)
            peer_ws_url = tunnel_url(address, "peer")

            with pytest.raises(InvalidStatusCode):
                async with connect(
                    peer_ws_url, extra_headers={"Authorization": "Bearer wrongkey"}
                ):
                    pass

            async with connect(
                peer_ws_url, extra_headers={"Authorization": "Bearer " + reg_key}
            ) as ws:
                await ws.send(encode({"type": "peers", "identity_key": "00" * 32}))
                with pytest.raises(ConnectionClosedError) as closed_exc:
                    await ws.recv()
                assert closed_exc.value.rcvd.code == 1008

            def full_http_scope(method: str, path: str):
                return {
                    "type": "http",
                    "asgi": {"version": "3.0", "spec_version": "2.3"},
                    "http_version": "1.1",
                    "method": method,
                    "scheme": "http",
                    "path": path,
                    "raw_path": path.encode("ascii"),
                    "root_path": "",
                    "query_string": b"",
                    "headers": [],
                    "client": ("127.0.0.1", 0),
                    "server": ("127.0.0.1", 80),
                }

            async with connect(
                peer_ws_url, extra_headers={"Authorization": "Bearer " + reg_key}
            ) as ws:
                await ws.send(
                    encode(
                        {
                            "type": "open",
                            "identity_key": client_b._identity_key,
                            "target": client_a.node_id,
                            "scope": full_http_scope("GET", "/api/peer-profiles"),
                        }
                    )
                )
                await ws.send(
                    encode({"type": "http.request", "body": b"", "more_body": False})
                )
                ok_start = decode(await ws.recv())
                assert ok_start["type"] == "http.response.start"
                assert ok_start["status"] == 200

            for bad_method, forbidden_path in (
                ("GET", "/api/workspaces"),
                ("GET", "/w/work/api/reload"),
                ("GET", "/api/peer-profiles/../workspaces"),
                ("POST", "/api/peer-profiles"),
                ("POST", "/api/peer-profiles/default/v1/responses"),
            ):
                async with connect(
                    peer_ws_url, extra_headers={"Authorization": "Bearer " + reg_key}
                ) as ws:
                    await ws.send(
                        encode(
                            {
                                "type": "open",
                                "identity_key": client_b._identity_key,
                                "target": client_a.node_id,
                                "scope": full_http_scope(bad_method, forbidden_path),
                            }
                        )
                    )
                    with pytest.raises(ConnectionClosedError) as closed_exc:
                        await ws.recv()
                    assert closed_exc.value.rcvd.code == 1008

            offline_status, _, offline_body = await client_b.peer_request(
                "b" * 52, "GET", "/api/peer-profiles"
            )
            assert offline_status == 503
            assert (
                "offline"
                in json.loads(offline_body.decode("utf-8"))["error"]["message"].lower()
            )


async def test_base_pinning_stream_disconnect_and_trajectory_isolation(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PYCODEX_DUMP", raising=False)
    monkeypatch.setenv("PIN_BASE_KEY", "k-base")
    monkeypatch.setenv("PIN_FAST_KEY", "k-fast")
    monkeypatch.setenv("PIN_OTHER_KEY", "k-other")

    test_loop = asyncio.get_running_loop()
    upstream_stream_aborted = asyncio.Event()

    async def handle_chat(_request, label):
        payload = {
            "id": "chatcmpl-" + label,
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "reply-" + label}},
                {"index": 0, "delta": {}, "finish_reason": "stop"},
            ],
        }

        async def sse():
            yield "data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    async def slow_hang_sse(_request):
        async def sse():
            try:
                yield b'data: {"type":"response.created"}\n\n'
                await asyncio.Future()
            finally:
                test_loop.call_soon_threadsafe(upstream_stream_aborted.set)

        return StreamingResponse(sse(), media_type="text/event-stream")

    async def base_chat(request):
        return await handle_chat(request, "base")

    async def fast_chat(request):
        return await handle_chat(request, "fast")

    async def other_chat(request):
        return await handle_chat(request, "other")

    downstream = RunningASGITestServer(
        Starlette(
            routes=[
                Route("/base/v1/chat/completions", base_chat, methods=["POST"]),
                Route("/base/v1/hang-sse", slow_hang_sse, methods=["POST"]),
                Route("/fast/v1/chat/completions", fast_chat, methods=["POST"]),
                Route("/other/v1/chat/completions", other_chat, methods=["POST"]),
            ]
        )
    )
    downstream.start()
    try:
        blocker_file = tmp_path / "dump_blocker.txt"
        blocker_file.write_text("not-a-directory", encoding="utf-8")

        home_a = tmp_path / "pin_a"
        home_a.mkdir(parents=True)
        (home_a / ".env").write_text(
            "PYCODEX_DUMP={0}\n".format(blocker_file), encoding="utf-8"
        )
        cfg_a = home_a / "config.toml"

        def write_cfg_a(top_prefix=""):
            cfg_a.write_text(
                top_prefix
                + 'model = "base-model"\n'
                'model_provider = "pbase"\n'
                'compat_api = "chat_completions"\n'
                '[model_providers.pbase]\n'
                'base_url = "{0}/base/v1"\n'
                'env_key = "PIN_BASE_KEY"\n'
                'stream_idle_timeout_ms = 600000\n'
                '[model_providers.pfast]\n'
                'base_url = "{0}/fast/v1"\n'
                'env_key = "PIN_FAST_KEY"\n'
                '[model_providers.pother]\n'
                'base_url = "{0}/other/v1"\n'
                'env_key = "PIN_OTHER_KEY"\n'
                '[profiles.fast]\n'
                'model = "fast-model"\n'
                'model_provider = "pfast"\n'
                'compat_api = "chat_completions"\n'
                '[profiles.other]\n'
                'model = "other-model"\n'
                'model_provider = "pother"\n'
                'compat_api = "chat_completions"\n'.format(downstream.base_url),
                encoding="utf-8",
            )

        write_cfg_a()
        peer_prov = PeerProfileProvider(cfg_a)
        peer_prov._load_proxy_provider_and_credentials("default")
        assert "PYCODEX_DUMP" not in os.environ
        peer_test_app = Starlette(routes=peer_prov.routes())
        with TestClient(peer_test_app) as prov_client:
            denied_resp = prov_client.post(
                "/api/peer-proxy",
                headers={"X-Peer-Profile": "default", "X-Peer-Subpath": "chat/completions"},
                json={"model": "base-model", "messages": []},
            )
            assert denied_resp.status_code == 403
        (home_a / ".env").unlink()

        codex_b = tmp_path / "pin_b"
        codex_b.mkdir(parents=True)
        cfg_b = codex_b / "config.toml"
        cfg_b.write_text('model = "local-b"\n', encoding="utf-8")
        ws_b_dir = tmp_path / "ws_pin_b"

        node_a_key = secrets.token_hex(32)
        app_a, _, _ = build_cli_workspace_app(tmp_path / "ws_pin_a", cfg_a)
        app_b1, _, _ = build_cli_workspace_app(ws_b_dir, cfg_b)

        gate = create_router_app("router-password")
        tunnel = create_tunnel_app(gate.state.router)
        async with serve(gate) as router_addr, serve(tunnel) as tunnel_addr:
            gate.state.router.api_port = urlsplit(router_addr).port
            target = client_target(tunnel_addr, gate.state.router.registration_key)
            async with serve_router_node(
                app_a, target, "pin-a", identity_key=node_a_key
            ) as (client_a, _a_url):
                async with serve_router_node(app_b1, target, "pin-b1") as (
                    _client_b1,
                    b1_url,
                ):
                    async with httpx.AsyncClient(
                        base_url=b1_url, trust_env=False
                    ) as browser_b1:
                        init_tabs = (
                            await browser_b1.get("/w/work/api/sessions")
                        ).json()["sessions"]
                        s1 = await browser_b1.post(
                            "/w/work/api/sessions",
                            json={
                                "profile": "peer:" + client_a.node_id,
                                "title": "Base Pinned",
                            },
                        )
                        assert s1.status_code == 200
                        await browser_b1.delete(
                            "/w/work/api/sessions/" + init_tabs[0]["id"]
                        )

                        write_cfg_a('profile = "fast"\n')
                        s2 = await browser_b1.post(
                            "/w/work/api/sessions",
                            json={
                                "profile": "peer:" + client_a.node_id,
                                "title": "Fast Pinned",
                            },
                        )
                        assert s2.status_code == 200

                        # Real end-to-end SSE disconnect: closing Client B's stream aborts
                        # Peer A's upstream HTTP connection immediately without leaking threads.
                        bridge = app_b1.state.peer_bridge
                        target_url = bridge.url_for(client_a.node_id, "base") + "/hang-sse"
                        hdrs = {
                            "Authorization": "Bearer " + os.environ[PEER_BRIDGE_API_KEY_ENV]
                        }
                        async with httpx.AsyncClient(trust_env=False) as hc:
                            async with hc.stream(
                                "POST", target_url, headers=hdrs, json={"input": "hi"}
                            ) as resp:
                                assert resp.status_code == 200
                                first = await resp.aiter_bytes().__anext__()
                                assert b"response.created" in first
                        await asyncio.wait_for(upstream_stream_aborted.wait(), timeout=2.0)

                write_cfg_a('profile = "other"\n')
                for cache_file in (codex_b / "peer-profiles-cache").glob("*.json"):
                    cache_file.unlink()

                app_b2, _, _ = build_cli_workspace_app(ws_b_dir, cfg_b)
                async with serve_router_node(app_b2, target, "pin-b2") as (
                    _cb2,
                    b2_url,
                ):
                    async with httpx.AsyncClient(
                        base_url=b2_url, trust_env=False
                    ) as browser_b2:
                        listed = (
                            await browser_b2.get("/w/work/api/sessions")
                        ).json()["sessions"]
                        by_title = {item["title"]: item for item in listed}
                        base_id = by_title["Base Pinned"]["id"]
                        msg = await browser_b2.post(
                            "/w/work/api/session/message",
                            json={"session_id": base_id, "prompt": "verify base"},
                        )
                        assert msg.status_code == 200
                        snap = await wait_async_snapshot(
                            browser_b2,
                            "/w/work/api/session?session_id=" + base_id,
                            lambda s: s["turns"]
                            and s["turns"][-1]["response"] == "reply-base",
                        )
                        assert snap["turns"][-1]["response"] == "reply-base"
    finally:
        downstream.stop()


async def test_unconditional_peer_http_proxy_methods_headers_gzip_and_self_ssrf_guard(
    tmp_path,
):
    binary_payload = b"header\r\nbinary\x00\xff\r\ntrailer"
    peer_a_url_holder = {"url": ""}

    async def echo_endpoint(request):
        raw_body = await request.body()
        if request.method == "HEAD":
            return Response(
                b"",
                status_code=200,
                headers={"X-Upstream-Method": "HEAD", "X-Upstream-Item": request.path_params["item"]},
            )
        payload = {
            "method": request.method,
            "item": request.path_params["item"],
            "raw_path": request.scope.get("raw_path", b"").decode("latin-1"),
            "query": request.url.query,
            "auth": request.headers.get("authorization", ""),
            "cookie": request.headers.get("cookie", ""),
            "trace": request.headers.get("x-caller-trace", ""),
            "body_utf8": raw_body.decode("utf-8", errors="replace"),
        }
        return JSONResponse(
            payload,
            status_code=201 if request.method == "PUT" else 200,
            headers={"X-Upstream-Header": "preserved-ok"},
        )

    async def gzip_binary_endpoint(_request):
        compressed = gzip.compress(binary_payload)
        return Response(
            compressed,
            status_code=200,
            media_type="application/octet-stream",
            headers={"Content-Encoding": "gzip", "X-Binary-Check": "gzipped"},
        )

    async def redirect_to_self(_request):
        return Response(
            b"",
            status_code=302,
            headers={"Location": peer_a_url_holder["url"] + "/w/work/api/sessions"},
        )

    upstream = RunningASGITestServer(
        Starlette(
            routes=[
                Route(
                    "/api/items/{item:path}",
                    echo_endpoint,
                    methods=["GET", "POST", "PUT", "DELETE", "HEAD"],
                ),
                Route("/gzip-bin", gzip_binary_endpoint, methods=["GET"]),
                Route("/ssrf-redirect", redirect_to_self, methods=["GET"]),
            ]
        )
    )
    upstream.start()
    try:
        cfg_a = tmp_path / "proxy_a" / "config.toml"
        cfg_a.parent.mkdir()
        cfg_a.write_text('model = "proxy-a"\n', encoding="utf-8")
        cfg_b = tmp_path / "proxy_b" / "config.toml"
        cfg_b.parent.mkdir()
        cfg_b.write_text('model = "proxy-b"\n', encoding="utf-8")

        app_a, _, _ = build_cli_workspace_app(tmp_path / "ws_proxy_a", cfg_a)
        app_b, _, _ = build_cli_workspace_app(tmp_path / "ws_proxy_b", cfg_b)

        gate = create_router_app("router-password")
        tunnel = create_tunnel_app(gate.state.router)
        async with serve(gate) as router_addr, serve(tunnel) as tunnel_addr:
            gate.state.router.api_port = urlsplit(router_addr).port
            target = client_target(tunnel_addr, gate.state.router.registration_key)
            async with serve_router_node(app_a, target, "proxy-node-a") as (
                client_a,
                a_url,
            ), serve_router_node(app_b, target, "proxy-node-b") as (
                client_b,
                _b_url,
            ):
                peer_a_url_holder["url"] = a_url

                # 1. PUT with query merging, raw %2F path, and caller Authorization/Cookie
                status, resp_headers, resp_body = await client_b.peer_request(
                    client_a.node_id,
                    "PUT",
                    "/api/peer-proxy",
                    headers=[
                        (
                            "X-Peer-Target-Url",
                            "{0}/api/items/ns%2Falpha?from_target=1".format(upstream.base_url),
                        ),
                        ("Authorization", "Bearer caller-custom-token"),
                        ("Cookie", "session_token=xyz; theme=dark"),
                        ("X-Caller-Trace", "trace-42"),
                        ("Content-Type", "application/json"),
                    ],
                    body=b'{"hello":"peer"}',
                    query_string=b"from_req=2",
                )
                assert status == 201
                hdr_map = {k.lower(): v for k, v in resp_headers}
                assert hdr_map.get("x-upstream-header") == "preserved-ok"
                echoed = json.loads(resp_body.decode("utf-8"))
                assert echoed["method"] == "PUT"
                assert echoed["raw_path"] == "/api/items/ns%2Falpha"
                assert echoed["query"] == "from_target=1&from_req=2"
                assert echoed["auth"] == "Bearer caller-custom-token"
                assert "session_token=xyz" in echoed["cookie"]
                assert echoed["trace"] == "trace-42"
                assert echoed["body_utf8"] == '{"hello":"peer"}'

                # 2. HEAD method returns headers without body
                head_status, head_headers, head_body = await client_b.peer_request(
                    client_a.node_id,
                    "HEAD",
                    "/api/peer-proxy",
                    headers=[
                        ("X-Peer-Target-Url", upstream.base_url),
                        ("X-Peer-Subpath", "/api/items/head-check"),
                    ],
                )
                assert head_status == 200
                assert dict(head_headers).get("x-upstream-method") == "HEAD"
                assert head_body == b""

                # 3. Gzipped binary response is decompressed once without corrupting CRLF/bytes
                gz_status, gz_headers, gz_body = await client_b.peer_request(
                    client_a.node_id,
                    "GET",
                    "/api/peer-proxy",
                    headers=[
                        ("X-Peer-Target-Url", upstream.base_url + "/gzip-bin"),
                        ("Accept-Encoding", "gzip"),
                    ],
                )
                assert gz_status == 200
                gz_hdr_map = {k.lower(): v for k, v in gz_headers}
                assert "content-encoding" not in gz_hdr_map
                assert gz_hdr_map.get("x-binary-check") == "gzipped"
                assert gz_body == binary_payload

                # 4. Self-loopback SSRF guard rejects targeting Peer A's own workspace port (direct & via redirect)
                ssrf_direct_status, _, _ = await client_b.peer_request(
                    client_a.node_id,
                    "POST",
                    "/api/peer-proxy",
                    headers=[("X-Peer-Target-Url", a_url + "/w/work/api/reload")],
                )
                assert ssrf_direct_status == 403

                ssrf_redir_status, _, _ = await client_b.peer_request(
                    client_a.node_id,
                    "GET",
                    "/api/peer-proxy",
                    headers=[("X-Peer-Target-Url", upstream.base_url + "/ssrf-redirect")],
                )
                assert ssrf_redir_status == 403
    finally:
        upstream.stop()


async def test_peer_provider_live_dotenv_and_profile_edits_without_mutating_os_environ(
    tmp_path, monkeypatch
):
    from workspace_server import peer_profiles as peer_profiles_mod

    key_env = "PYCODEX_TEST_ISOLATED_ENV_KEY"
    hdr_env = "PYCODEX_TEST_ISOLATED_ENV_HDR"
    monkeypatch.setenv(key_env, "baseline-env-key")
    monkeypatch.delenv(hdr_env, raising=False)

    codex_dir = tmp_path / "isolated_node"
    codex_dir.mkdir()
    config_path = codex_dir / "config.toml"
    dotenv_path = codex_dir / ".env"
    fast_profile_path = codex_dir / "fast.config.toml"

    config_path.write_text(
        'model = "gpt-5.4"\n'
        'model_provider = "demo"\n'
        "[model_providers.demo]\n"
        'base_url = "http://127.0.0.1:18080/v1"\n'
        'env_key = "' + key_env + '"\n'
        'env_http_headers = { "X-Env-Tag" = "' + hdr_env + '" }\n'
        "[profiles.inlinep]\n"
        'model = "gpt-5.4-mini"\n',
        encoding="utf-8",
    )
    fast_profile_path.write_text(
        "[model_providers.demo]\n"
        'http_headers = { "X-Profile-Rev" = "rev-1" }\n',
        encoding="utf-8",
    )
    dotenv_path.write_text(
        key_env + "=dotenv-key-v1\n" + hdr_env + "=tag-v1\n",
        encoding="utf-8",
    )

    catalog_loads = {"count": 0}
    orig_load_catalog = peer_profiles_mod.load_models_by_slug

    def counting_load_catalog(*args, **kwargs):
        catalog_loads["count"] += 1
        return orig_load_catalog(*args, **kwargs)

    monkeypatch.setattr(peer_profiles_mod, "load_models_by_slug", counting_load_catalog)
    provider = PeerProfileProvider(config_path)

    harness, pcfg, creds = provider._load_proxy_provider_and_credentials("named", "fast")
    assert harness == "pycodex"
    assert pcfg.base_url == "http://127.0.0.1:18080/v1"
    assert creds.native_api_key == "dotenv-key-v1"
    assert creds.http_headers == {"x-profile-rev": "rev-1", "x-env-tag": "tag-v1"}
    # Process-wide os.environ is untouched by peer requests!
    assert os.environ.get(key_env) == "baseline-env-key"
    assert hdr_env not in os.environ

    dotenv_path.write_text(
        key_env + "=dotenv-key-v2\n" + hdr_env + "=tag-v2\n", encoding="utf-8"
    )
    _, _, creds_v2 = provider._load_proxy_provider_and_credentials("named", "fast")
    assert creds_v2.native_api_key == "dotenv-key-v2"
    assert creds_v2.http_headers["x-env-tag"] == "tag-v2"

    fast_profile_path.write_text(
        "[model_providers.demo]\n"
        'http_headers = { "X-Profile-Rev" = "rev-2" }\n',
        encoding="utf-8",
    )
    _, _, creds_rev2 = provider._load_proxy_provider_and_credentials("named", "fast")
    assert creds_rev2.http_headers["x-profile-rev"] == "rev-2"
    assert catalog_loads["count"] == 1

    dotenv_path.unlink()
    _, _, creds_restored = provider._load_proxy_provider_and_credentials("named", "fast")
    assert creds_restored.native_api_key == "baseline-env-key"
    assert "x-env-tag" not in creds_restored.http_headers

    _, inline_cfg, _ = provider._load_proxy_provider_and_credentials("named", "inlinep")
    assert inline_cfg.model == "gpt-5.4-mini"
    (codex_dir / "inlinep.config.toml").write_text('model = "conflict"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous Codex profile"):
        provider._load_proxy_provider_and_credentials("named", "inlinep")


class _LegacyV072RouterClient:
    """Exact v0.7.2 RouterClient surface kept in memory across ReloadableApp.reload()."""

    def __init__(self, app, target, name, port, workspaces):
        from pycodex_router.protocol import device_id, send_event

        self._send_event = send_event
        self.app = app
        address, self._registration_key = parse_target(target)
        self.control_url = tunnel_url(address, "control")
        self.data_url = tunnel_url(address, "data")
        self._identity_key = secrets.token_hex(32)
        self.node_id = device_id(self._identity_key)
        self.name = name
        self.port = port
        self.workspaces = workspaces
        self.connected = False
        self.api_url = None
        self._task = None
        self._channels = set()
        self._requests = set()

    def _connect(self, url):
        return connect(
            url,
            extra_headers={"Authorization": "Bearer " + self._registration_key},
            open_timeout=10,
            close_timeout=2,
            ping_interval=20,
            ping_timeout=20,
            max_size=32 * 1024 * 1024,
            max_queue=4,
        )

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def close(self):
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        await asyncio.gather(*(ch.close() for ch in tuple(self._channels)))
        if self._requests:
            await asyncio.gather(*tuple(self._requests), return_exceptions=True)

    async def _run(self):
        async with self._connect(self.control_url) as channel:
            await channel.send(
                encode(
                    {
                        "type": "register",
                        "version": 2,
                        "identity_key": self._identity_key,
                        "name": self.name,
                        "port": self.port,
                        "workspaces": self.workspaces(),
                    }
                )
            )
            reply = decode(await asyncio.wait_for(channel.recv(), 10))
            assert reply.get("type") == "registered"
            self.api_url = reply.get("api_url")
            self.connected = True
            try:
                async for payload in channel:
                    message = decode(payload)
                    assert message.get("type") == "connect"
                    task = asyncio.create_task(self._serve(message["stream_id"]))
                    self._requests.add(task)
                    task.add_done_callback(self._requests.discard)
            finally:
                self.connected = False

    async def _serve(self, stream_id):
        async with self._connect(self.data_url) as channel:
            self._channels.add(channel)
            try:
                await channel.send(
                    encode({"stream_id": stream_id, "identity_key": self._identity_key})
                )
                opening = decode(await asyncio.wait_for(channel.recv(), 10))
                scope = opening["scope"]
                # v0.7.2 _serve only populated pycodex.agent_access from opening!
                scope["pycodex.agent_access"] = opening.get("agent_access") is True

                async def receive():
                    return decode(await channel.recv())

                async def send(message):
                    await self._send_event(channel.send, message)

                await self.app(scope, receive, send)
            finally:
                self._channels.discard(channel)


async def test_in_process_reload_with_legacy_v072_router_client_enables_peer_profiles(
    tmp_path, monkeypatch
):
    from workspace_server.owned_process import ReloadableApp

    async def noop_preflight():
        return None

    monkeypatch.setattr("workspace_server.owned_process.preflight_check", noop_preflight)
    monkeypatch.setattr("workspace_server.owned_process.purge_reloadable_modules", dict)

    seen_auth = []

    async def fake_responses(request):
        seen_auth.append(request.headers.get("authorization"))
        body = (
            'event: response.output_item.done\n'
            'data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"reloaded-peer-ok"}]}}\n\n'
            'event: response.completed\n'
            'data: {"type":"response.completed","response":{"id":"r1","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}\n\n'
        )
        return Response(body, media_type="text/event-stream")

    upstream = RunningASGITestServer(
        Starlette(routes=[Route("/v1/responses", fake_responses, methods=["POST"])])
    )
    upstream.start()
    try:
        cfg_a = tmp_path / "home_a" / "config.toml"
        cfg_a.parent.mkdir(parents=True)
        cfg_a.write_text('model = "local-a"\n', encoding="utf-8")

        key_env = "TEST_LEGACY_RELOAD_PEER_KEY"
        cfg_b = tmp_path / "home_b" / "config.toml"
        cfg_b.parent.mkdir(parents=True)
        (cfg_b.parent / ".env").write_text(key_env + "=secret-from-b\n", encoding="utf-8")
        cfg_b.write_text(
            'model = "gpt-5.4"\n'
            'model_provider = "up"\n'
            "[model_providers.up]\n"
            'name = "up"\n'
            'base_url = "' + upstream.base_url + '/v1"\n'
            'env_key = "' + key_env + '"\n'
            "[profiles.remote]\n"
            'model = "gpt-5.4-mini"\n',
            encoding="utf-8",
        )

        def build_a():
            app, _, _ = build_cli_workspace_app(
                tmp_path / "ws_a", cfg_a, password="pw-a"
            )
            return app

        def build_b():
            app, _, _ = build_cli_workspace_app(
                tmp_path / "ws_b", cfg_b, password="pw-b"
            )
            return app

        gate = create_router_app("router-pw", aliases_path=tmp_path / "aliases.json")
        tunnel = create_tunnel_app(gate.state.router)
        async with serve(gate) as router_addr, serve(tunnel) as tunnel_addr:
            gate.state.router.api_port = urlsplit(router_addr).port
            target = client_target(tunnel_addr, gate.state.router.registration_key)

            shell_a = ReloadableApp(build_a)
            shell_b = ReloadableApp(build_b)
            shell_a.router = _LegacyV072RouterClient(
                shell_a.current_app, target, "zhy-6", 6007, shell_a.router_inventory
            )
            shell_b.router = _LegacyV072RouterClient(
                shell_b.current_app, target, "zhy-6-step", 6008, shell_b.router_inventory
            )
            assert not hasattr(shell_a.router, "list_peers")
            assert not hasattr(shell_a.router, "loop")

            async with serve(shell_a), serve(shell_b):
                await eventually(
                    lambda: shell_a.router.connected and shell_b.router.connected
                )
                async with ip_browser(router_addr) as browser:
                    await login(browser, password="router-pw")
                    await verify_device(browser, shell_a.router.node_id)
                    await verify_device(browser, shell_b.router.node_id)
                    base_a = router_addr + "/n/" + shell_a.router.node_id
                    base_b = router_addr + "/n/" + shell_b.router.node_id
                    assert (
                        await browser.post(base_a + "/login", json={"password": "pw-a"})
                    ).status_code == 200
                    assert (
                        await browser.post(base_b + "/login", json={"password": "pw-b"})
                    ).status_code == 200

                    # Trigger in-page Reboot on both legacy-shell nodes!
                    assert (
                        await browser.post(base_a + "/w/work/api/reload", json={})
                    ).json() == {"ok": True, "status": "reloaded"}
                    assert (
                        await browser.post(base_b + "/w/work/api/reload", json={})
                    ).json() == {"ok": True, "status": "reloaded"}

                    # Active legacy identity keys were persisted for future restarts.
                    saved_keys_a = list((cfg_a.parent / "router-nodes").glob("*.key"))
                    assert len(saved_keys_a) == 1
                    assert saved_keys_a[0].read_text(encoding="ascii").strip() == shell_a.router._identity_key

                    # Re-login after reload and verify peer discovery + peer profile execution!
                    assert (
                        await browser.post(base_a + "/login", json={"password": "pw-a"})
                    ).status_code == 200
                    fast_sessions = (
                        await browser.get(base_a + "/w/work/api/sessions")
                    ).json()
                    assert fast_sessions["router_connected"] is True

                    peer_nodes = (
                        await browser.get(base_a + "/w/work/api/peer-nodes")
                    ).json()["peer_nodes"]
                    assert [n["id"] for n in peer_nodes] == [shell_b.router.node_id]

                    peer_profs = (
                        await browser.get(
                            base_a
                            + "/w/work/api/peer-nodes/"
                            + shell_b.router.node_id
                            + "/profiles"
                        )
                    ).json()
                    assert peer_profs["profiles"] == ["remote"]

                    # Browser cannot spoof pycodex.peer_access through Router.proxy on legacy shell!
                    assert (
                        await browser.get(base_b + "/api/peer-profiles")
                    ).status_code == 401
                    assert (
                        await browser.post(base_b + "/login", json={"password": "pw-b"})
                    ).status_code == 200
                    assert (
                        await browser.get(base_b + "/api/peer-profiles")
                    ).status_code == 403

                    created = await browser.post(
                        base_a + "/w/work/api/sessions",
                        json={
                            "profile": "peer:" + shell_b.router.node_id + ":remote",
                            "prompt": "ping via legacy shell",
                        },
                    )
                    assert created.status_code == 200, created.text
                    sid = created.json()["session_id"]
                    snap = await wait_async_snapshot(
                        browser,
                        base_a + "/w/work/api/session?session_id=" + sid,
                        lambda s: bool(s["turns"]) and s["turns"][-1]["status"] == "completed",
                    )
                    assert snap["turns"][-1]["response"] == "reloaded-peer-ok"
                    assert seen_auth == ["Bearer secret-from-b"]
    finally:
        upstream.stop()
