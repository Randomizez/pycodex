import asyncio
import json
import os
import re
import secrets
import signal
import socket
import sys
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx
import pytest
import uvicorn
from pycodex_router.client import RouterClient, attach_router
from pycodex_router.protocol import (
    decode, device_id, encode, normalize_api_url, parse_target, tunnel_url,
)
from pycodex_router.server import create_app, create_tunnel_app, load_registration_key
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import (
    ConnectionClosedError,
    InvalidHandshake,
    InvalidStatusCode,
)
from websockets.legacy.client import connect, unix_connect
from workspace_server import (
    WorkspaceDefinition,
    WorkspaceEntry,
    WorkspaceInteractiveSession,
    WorkspaceRegistry,
    WorkspaceSessionManager,
    create_multi_workspace_app,
)

from pycodex import (
    Agent,
    AgentRuntime,
    AssistantMessage,
    ContextConfig,
    ModelResponse,
    ToolRegistry,
)
from tests.fakes import ScriptedModelClient
from workspace_server.owned_process import run_owned_server


async def eventually(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), 5)


class TestServer(uvicorn.Server):
    __test__ = False

    def install_signal_handlers(self):
        pass

    @contextmanager
    def capture_signals(self):
        yield


@asynccontextmanager
async def serve(app, listener=None):
    if listener is None:
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = TestServer(
        uvicorn.Config(
            app,
            loop="asyncio",
            ws="websockets",
            access_log=False,
            log_level="error",
            ws_max_size=32 * 1024 * 1024,
        )
    )
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        await eventually(lambda: server.started or task.done())
        if task.done():
            task.result()
            raise AssertionError("server exited before startup")
        yield "http://127.0.0.1:{0}".format(port)
    finally:
        server.should_exit = True
        await asyncio.wait_for(asyncio.shield(task), 10)
        listener.close()


@asynccontextmanager
async def serve_router(gate, root="", listener=None):
    web = Starlette(routes=[Mount(root, app=gate)]) if root else gate
    tunnel = create_tunnel_app(gate.state.router)
    if root:
        tunnel = Starlette(routes=[Mount(root, app=tunnel)])
    async with serve(web) as address:
        gate.state.router.api_port = urlsplit(address).port
        async with serve(tunnel, listener) as target:
            yield address, client_target(target + root, gate.state.router.registration_key)


def workspace_record(name="shared"):
    return {
        "id": name,
        "work_dir": "/test",
        "state_path": "/test/workspace",
        "enable_board": True,
        "session_count": 0,
        "running": False,
        "unread": False,
    }


def node_url(address, node_id, root=""):
    return address + root + "/n/" + node_id + "/"


def client_target(address, key):
    return address.replace("://", "://" + key + "@", 1)


def tunnel_connect(target, endpoint="control", **options):
    address, key = parse_target(target)
    headers = dict(options.pop("extra_headers", {}))
    headers["Authorization"] = "Bearer " + key
    return connect(tunnel_url(address, endpoint), extra_headers=headers, **options)


def origin(url):
    parsed = urlsplit(url)
    return parsed.scheme + "://" + parsed.netloc


@asynccontextmanager
async def device(app, target, name="device", workspaces=None):
    if workspaces is None:
        workspaces = lambda: [workspace_record()]
    client = RouterClient(
        app,
        target,
        name,
        6007,
        workspaces,
        refresh_seconds=0.02,
        reconnect_seconds=0.02,
    )
    async with app.router.lifespan_context(app):
        await client.start()
        try:
            await eventually(lambda: client.connected)
            yield client
        finally:
            await client.close()


async def login(browser, root=""):
    response = await browser.post(root + "/login", json={"password": "router-password"})
    assert response.status_code == 200


def ip_browser(address, ip="127.0.0.1", timeout=5):
    return httpx.AsyncClient(
        base_url=address,
        transport=httpx.AsyncHTTPTransport(local_address=ip),
        trust_env=False,
        timeout=timeout,
    )


def workspace_app(directory, password=None, response_factory=None):
    directory.mkdir()
    state_path = directory / "workspace"
    state_path.with_suffix(".html").write_text(
        '<html><img src="image.png">Canvas {0}</html>'.format(directory.name),
        encoding="utf-8",
    )
    (directory / "image.png").write_bytes(b"\x89PNG\r\n\x1a\nimage-bytes")

    def session():
        model = ScriptedModelClient(
            response_factory=response_factory
            or (lambda prompt, count: ModelResponse([AssistantMessage("done")]))
        )
        return WorkspaceInteractiveSession(
            AgentRuntime(Agent(model, ToolRegistry(), ContextConfig()))
        )

    registry = WorkspaceRegistry(
        [
            WorkspaceEntry(
                WorkspaceDefinition("shared", state_path, directory),
                WorkspaceSessionManager(session, state_path),
            )
        ]
    )
    return create_multi_workspace_app(registry, password), registry


@asynccontextmanager
async def owned_workspace(app):
    read_fd, write_fd = os.pipe()
    task = asyncio.get_running_loop().run_in_executor(
        None, run_owned_server, app, "127.0.0.1", 0, read_fd,
    )
    try:
        await eventually(lambda: getattr(app.state, "agent_socket_path", None) is not None)
        address = app.state.local_api_url
        socket_path = app.state.agent_socket_path
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=socket_path),
            base_url=address, trust_env=False,
        ) as local:
            async def ready():
                while True:
                    if task.done():
                        task.result()
                        raise AssertionError("workspace exited before startup")
                    try:
                        await local.get("/api/workspaces")
                        return
                    except httpx.ConnectError:
                        await asyncio.sleep(0.01)
            await asyncio.wait_for(ready(), 5)
            yield address, socket_path, local
    finally:
        os.close(write_fd)
        try:
            await asyncio.wait_for(asyncio.shield(task), 10)
        finally:
            os.close(read_fd)


@pytest.mark.skipif(os.name != "posix", reason="owner-only Unix socket")
async def test_private_agent_socket_exempts_local_apis_but_not_tcp(tmp_path):
    app, registry = workspace_app(tmp_path / "local", "local-password")
    async with owned_workspace(app) as (address, socket_path, local):
        assert Path(socket_path).stat().st_mode & 0o777 == 0o600
        assert Path(socket_path).parent.stat().st_mode & 0o777 == 0o700
        inventory = await local.get("/api/workspaces")
        assert inventory.status_code == 200
        own = inventory.json()["workspaces"][0]["sessions"][0]["id"]
        response = await local.get("/w/shared/api/session", params={
            "session_id": own, "runtime_info": "true",
        })
        assert response.status_code == 200
        link = registry.get("shared").manager.get()
        assert link.snapshot()["turns"] == []
        prompt = response.json()["runtime_info"]
        assert "curl --unix-socket " + socket_path in prompt
        assert "local-password" not in prompt
        assert (await local.post("/api/workspaces", json={})).status_code == 401
        assert (await local.get("/w/shared/api/views")).status_code == 401
        assert (await local.get("/router/api/workspaces")).status_code == 503
        async with httpx.AsyncClient(base_url=address, trust_env=False) as browser:
            for headers in [
                {}, {"X-Pycodex-Agent-Access": "true"},
                {"Forwarded": "for=127.0.0.1", "X-Forwarded-For": "127.0.0.1"},
            ]:
                assert (await browser.get("/api/workspaces", headers=headers)).status_code == 401
            with pytest.raises(InvalidStatusCode):
                async with connect(address.replace("http:", "ws:") + "/w/shared/ws/session"):
                    pass
        with pytest.raises(InvalidStatusCode):
            async with unix_connect(
                socket_path, uri=address.replace("http:", "ws:") + "/w/shared/ws/session",
            ):
                pass
    assert not Path(socket_path).exists()
    assert app.state.agent_socket_path is None


@pytest.mark.skipif(os.name != "posix", reason="owner-only Unix socket")
@pytest.mark.parametrize("root", ["", "/remote"])
async def test_registered_agents_contact_password_protected_peers_and_revoke_on_disconnect(tmp_path, root):
    gate = create_app("router-password")
    source, source_registry = workspace_app(tmp_path / "source", "source-password")
    target_app, target_registry = workspace_app(tmp_path / "target", "target-password")
    async with serve_router(gate, root) as (address, target):
        gate.state.router.api_url = address + root
        attach_router(source, source_registry, target, 6007)
        attach_router(target_app, target_registry, target, 6007)
        async with serve(target_app):
            async with owned_workspace(source) as (local_address, socket_path, local):
                agent = source.state.router_client
                peer = target_app.state.router_client
                await eventually(lambda: agent.connected and peer.connected)
                agent_loop = agent._task.get_loop()
                info = await local.get("/w/shared/api/session", params={"runtime_info": "true"})
                assert info.status_code == 200
                source_link = source_registry.get("shared").manager.get()
                assert source_link.snapshot()["turns"] == []
                prompt = info.json()["runtime_info"]
                assert "GET G" + root + "/api/workspaces" in prompt
                assert "B=G+url" in prompt
                assert socket_path in prompt
                assert agent._identity_key not in prompt
                assert gate.state.router.registration_key not in prompt
                assert "source-password" not in prompt
                discovery = "/router" + root + "/api/workspaces"
                listing = await local.get(discovery)
                assert listing.status_code == 200
                peer_record = next(
                    item for item in listing.json()["workspaces"]
                    if item["node_id"] == peer.node_id
                )
                prefix = "/router" + peer_record["url"].rstrip("/")
                sessions = await local.get(prefix + "/api/sessions")
                assert sessions.status_code == 200
                own = sessions.json()["sessions"][0]["id"]
                response = await local.post(prefix + "/api/session/message", json={
                    "session_id": own, "prompt": "/queue peer request", "sender": "agent:test",
                })
                assert response.status_code == 200
                link = target_registry.get("shared").manager.get()
                await eventually(lambda: link.snapshot()["turns"] and link.snapshot()["turns"][-1]["response"] == "done")
                assert link.snapshot()["turns"][-1]["prompt"] == "peer request"
                # Privileged APIs and arbitrary URLs cannot use the private bridge.
                assert (await local.get("/router" + root + "/api/registration-key")).status_code == 403
                async with httpx.AsyncClient(base_url=address, trust_env=False) as public:
                    headers = {"Authorization": "Bearer " + agent._identity_key}
                    assert (await public.get(root + "/api/workspaces", headers=headers)).status_code == 200
                    for invalid in [
                        {},
                        {"Authorization": "Bearer " + gate.state.router.registration_key},
                        {"Authorization": "Bearer " + "0" * 64},
                        dict(headers, Origin=address),
                        dict(headers, **{"Sec-Fetch-Site": "same-origin"}),
                    ]:
                        assert (await public.get(root + "/api/workspaces", headers=invalid)).status_code == 401
                    assert (await public.get(root + "/api/registration-key", headers=headers)).status_code == 401
                    peer_url = peer_record["url"].rstrip("/") + "/api/sessions"
                    assert (await public.get(peer_url)).status_code == 401
                    await login(public, root)
                    assert (await public.get(peer_url)).status_code == 401
                    public.cookies.clear()
                    await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(
                        agent.close(), agent_loop,
                    ))
                    await eventually(lambda: agent.node_id not in gate.state.router.nodes)
                    assert (await public.get(root + "/api/workspaces", headers=headers)).status_code == 401
                    assert (await local.get(discovery)).status_code == 503
                    await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(
                        agent.start(), agent_loop,
                    ))
                    await eventually(lambda: agent.connected)
                    assert (await local.get(prefix + "/api/sessions")).status_code == 200


async def test_real_workspace_pages_files_messages_and_namespaces(tmp_path):
    gate = create_app("router-password")
    first, first_registry = workspace_app(tmp_path / "first")
    second, second_registry = workspace_app(tmp_path / "second")
    source = tmp_path / "first" / "文件 with spaces.py"
    source.write_text("print('original')\n", encoding="utf-8")
    other = tmp_path / "first" / "other.py"
    other.write_text("print('changed')\n", encoding="utf-8")

    async with serve_router(gate) as (address, target):
        attach_router(first, first_registry, target, 6007)
        attach_router(second, second_registry, target, 6008)
        first_client = first.state.router_client
        second_client = second.state.router_client
        async with first.router.lifespan_context(first):
            async with second.router.lifespan_context(second):
                await eventually(
                    lambda: first_client.connected and second_client.connected
                )
                async with ip_browser(address) as browser:
                    assert (await browser.get("/api/workspaces")).status_code == 401
                    assert (await browser.get("/")).status_code == 303
                    assert (
                        await browser.post("/login", json={"password": "incorrect"})
                    ).status_code == 401
                    await login(browser)
                    assert "Your workspaces" in (await browser.get("/")).text
                    inventory = (await browser.get("/api/workspaces")).json()
                    assert len(inventory["nodes"]) == 2
                    assert {item["name"] for item in inventory["nodes"]} == {
                        socket.gethostname() + "/127.0.0.1:6007",
                        socket.gethostname() + "/127.0.0.1:6008",
                    }
                    assert {item["node_id"] for item in inventory["workspaces"]} == {
                        first_client.node_id,
                        second_client.node_id,
                    }
                    for client, label in [
                        (first_client, "first"),
                        (second_client, "second"),
                    ]:
                        prefix = node_url(address, client.node_id) + "w/shared/"
                        assert (await browser.get(prefix)).status_code == 200
                        redirect = await browser.get(prefix.rstrip("/"))
                        assert redirect.headers["location"] == urlsplit(prefix).path
                        assert (
                            "Canvas " + label
                            in (await browser.get(prefix + "board")).text
                        )
                        image = await browser.get(prefix + "image.png")
                        assert image.content == b"\x89PNG\r\n\x1a\nimage-bytes"
                        assert image.headers["content-type"] == "image/png"
                        head = await browser.head(prefix + "image.png")
                        assert head.status_code == 200 and head.content == b""
                        assert (
                            await browser.get(prefix + "../missing")
                        ).status_code == 404

                    prefix = node_url(address, first_client.node_id) + "w/shared/"
                    asset = await browser.get(prefix + "assets/content_views.mjs")
                    assert asset.status_code == 200
                    cached = await browser.get(
                        prefix + "assets/content_views.mjs",
                        headers={"If-None-Match": asset.headers["etag"]},
                    )
                    assert cached.status_code == 304
                    opened = await browser.post(
                        prefix + "api/views/open", json={"href": str(source) + ":1"}
                    )
                    assert opened.status_code == 200
                    view_id = opened.json()["view"]["id"]
                    view = (await browser.get(prefix + "api/views/" + view_id)).json()
                    assert view["documents"][0]["text"] == "print('original')\n"
                    diff = await browser.post(
                        prefix + "api/views/open",
                        json={"href": "diff:" + quote(source.name) + ".." + other.name},
                    )
                    assert diff.status_code == 200
                    assert diff.json()["view"]["kind"] == "diff"
                    assert (
                        await browser.post(
                            prefix + "api/session/message", json={"prompt": "hello"}
                        )
                    ).status_code == 200
                    await eventually(
                        lambda: not first_registry.get("shared")
                        .manager.get()
                        .runtime.agent.is_running
                    )
                    snapshot = (await browser.get(prefix + "api/session")).json()[
                        "snapshot"
                    ]
                    assert snapshot["turns"][-1]["response"] == "done"
                    other_prefix = (
                        node_url(address, second_client.node_id) + "w/shared/"
                    )
                    assert (await browser.get(other_prefix + "api/session")).json()[
                        "snapshot"
                    ]["turns"] == []
        await eventually(lambda: not gate.state.router.nodes)


async def test_registration_uses_peer_ip_and_reconnect_keeps_assigned_id(monkeypatch):
    gate = create_app()
    source_ip = "127.0.0.2"
    async with serve_router(gate) as (address, target):
        first = RouterClient(
            Starlette(),
            target,
            "workstation",
            6007,
            lambda: [],
            reconnect_seconds=0.01,
        )
        second = RouterClient(
            Starlette(),
            target,
            "workstation",
            6007,
            lambda: [],
            reconnect_seconds=0.01,
        )
        headers = {"Authorization": "Bearer " + gate.state.router.registration_key}
        monkeypatch.setattr(
            first, "_connect",
            lambda url: connect(url, local_addr=(source_ip, 0), extra_headers=headers),
        )
        monkeypatch.setattr(
            second, "_connect",
            lambda url: connect(
                url, local_addr=("127.0.0.3", 0), extra_headers=headers
            ),
        )
        await first.start()
        await second.start()
        try:
            await eventually(lambda: first.connected and second.connected)
            assert first.api_url == address
            assert second.api_url == address
            first_id, second_id = first.node_id, second.node_id
            assert first_id != second_id
            assert (
                gate.state.router.nodes[first_id].name == "workstation/127.0.0.2:6007"
            )
            assert (
                gate.state.router.nodes[second_id].name == "workstation/127.0.0.3:6007"
            )
            original = gate.state.router.nodes[first_id]
            gate.state.router.api_url = "https://router.example/proxy"
            source_ip = "127.0.0.4"
            await original.websocket.close(code=1012)
            await eventually(
                lambda: first.connected
                and gate.state.router.nodes.get(first_id) is not None
                and gate.state.router.nodes[first_id] is not original
            )
            assert first.node_id == first_id
            assert first.api_url == "https://router.example/proxy"
            assert (
                gate.state.router.nodes[first_id].name == "workstation/127.0.0.4:6007"
            )
            assert second.node_id == second_id
        finally:
            await first.close()
            await second.close()
        assert first.api_url is None
        assert second.api_url is None


@pytest.mark.parametrize("value", [
    "https://user:secret@example.test", "https://example.test?key=secret",
    "https://example.test/#secret", "ws://example.test", "/relative",
    "https://example.test:99999", "https://example.test\n", "https://example.test\\path",
])
def test_api_url_rejects_credentials_and_invalid_addresses(value):
    with pytest.raises(ValueError) as error:
        normalize_api_url(value)
    assert value not in str(error.value)


async def test_registration_advertises_public_api_with_proxy_prefix():
    gate = create_app(api_url="https://router.example/remote/")
    async with serve_router(gate) as (_address, target):
        async with device(Starlette(), target) as client:
            assert client.api_url == "https://router.example/remote"


async def test_runtime_info_through_router_keeps_local_and_public_addresses(tmp_path):
    gate = create_app("router-password")
    app, registry = workspace_app(tmp_path / "info")
    async with serve_router(gate, "/remote") as (address, target):
        gate.state.router.api_url = address + "/remote"
        attach_router(app, registry, target, 6007)
        async with serve(app) as local_address:
            app.state.local_api_url = local_address
            await eventually(lambda: app.state.router_client.connected)
            async with httpx.AsyncClient(trust_env=False) as browser:
                response = await browser.post(
                    address + "/remote/login", json={"password": "router-password"}
                )
                assert response.status_code == 200
                records = (await browser.get(address + "/remote/api/workspaces")).json()
                prefix = address + records["workspaces"][0]["url"].rstrip("/")
                response = await browser.get(prefix + "/api/session", params={"runtime_info": "true"})
                assert response.status_code == 200
                link = registry.get("shared").manager.get()
                assert link.snapshot()["turns"] == []
                text = response.json()["runtime_info"]
                assert "L=" + local_address + "; current workspace=L/w/shared" in text
                assert "R=" + address + "/remote" in text
                assert gate.state.router.registration_key not in text
                assert app.state.router_client._identity_key not in text


@pytest.mark.parametrize("root", ["", "/remote"])
async def test_router_and_workspace_passwords_protect_remote_visitors(tmp_path, root, monkeypatch):
    monkeypatch.setattr("workspace_server.app.socket.gethostname", lambda: "test<device>")
    gate = create_app("router-password")
    app, registry = workspace_app(tmp_path / "workspace", "local-password")
    async with serve_router(gate, root) as (address, target):
        attach_router(app, registry, target, 6007)
        async with serve(app) as local_address:
            await eventually(lambda: app.state.router_client.connected)
            node_id = app.state.router_client.node_id
            base = node_url(address, node_id, root).rstrip("/")
            prefix = base + "/w/shared/"
            async with ip_browser(address) as browser:
                target = prefix + "?tab=chat"
                redirect = await browser.get(target)
                assert redirect.status_code == 303
                assert redirect.headers["location"].startswith(
                    root + "/login?"
                )
                assert (await browser.get(prefix + "api/session")).status_code == 401
                assert (
                    await browser.post(
                        base + "/login", json={"password": "local-password"}
                    )
                ).status_code == 401
                router_login = (await browser.get(root + "/login")).text
                assert "<h1>Log in to pycodex router</h1>" in router_login
                assert "<strong>test&lt;device&gt;</strong>" in router_login
                assert "<code>" + address + root + "/</code>" in router_login
                assert (
                    await browser.post(
                        root + "/login", json={"password": "router-password"}
                    )
                ).status_code == 200

                redirect = await browser.get(target)
                assert redirect.status_code == 303
                workspace_login_url = redirect.headers["location"]
                assert workspace_login_url.startswith(urlsplit(base).path + "/login?")
                workspace_login_url = origin(base) + workspace_login_url
                workspace_login_page = (await browser.get(workspace_login_url)).text
                assert "<h1>Log in to pycodex workspace</h1>" in workspace_login_page
                assert "<strong>test&lt;device&gt;</strong>" in workspace_login_page
                assert "<code>" + base + "/</code>" in workspace_login_page
                assert (await browser.get(prefix + "api/session")).status_code == 401
                websocket_url = (prefix + "ws/session").replace("http:", "ws:")
                cookies = {
                    "Cookie": browser.build_request("GET", target).headers["cookie"]
                }
                with pytest.raises(InvalidStatusCode):
                    async with connect(
                        websocket_url, extra_headers=cookies, origin=origin(base)
                    ):
                        pass
                oversized = await browser.post(
                    workspace_login_url, content=b" " * 4097,
                    headers={"Content-Type": "application/json"},
                )
                assert oversized.status_code == 413
                for _ in range(3):
                    assert (
                        await browser.post(
                            workspace_login_url, json={"password": "router-password"}
                        )
                    ).status_code == 401
                workspace_login = await browser.post(
                    workspace_login_url, json={"password": "local-password"}
                )
                assert workspace_login.status_code == 200
                assert (
                    workspace_login.json()["redirect"]
                    == urlsplit(target).path + "?tab=chat"
                )
                assert {
                    (cookie.name, cookie.path) for cookie in browser.cookies.jar
                } == {
                    ("pycodex_router_auth", root + "/"),
                    ("pycodex_ws_auth", urlsplit(base).path + "/"),
                }
                assert (await browser.get(target)).status_code == 200
                assert (await browser.get(prefix + "api/session")).status_code == 200
                cookies = {
                    "Cookie": browser.build_request("GET", target).headers["cookie"]
                }
                async with connect(
                    websocket_url,
                    extra_headers=cookies,
                    origin=origin(base),
                ) as websocket:
                    assert decode(await websocket.recv())["type"] == "hello"
                original = gate.state.router.nodes[node_id]
                await original.websocket.close(code=1012)
                await eventually(
                    lambda: app.state.router_client.connected
                    and gate.state.router.nodes.get(node_id) is not None
                    and gate.state.router.nodes[node_id] is not original
                )
                assert (await browser.get(prefix + "api/session")).status_code == 200
                browser.cookies.delete(
                    "pycodex_router_auth",
                    domain=urlsplit(address).hostname,
                    path=root + "/",
                )
                assert (await browser.get(prefix + "api/session")).status_code == 401
            async with httpx.AsyncClient(
                base_url=local_address, trust_env=False
            ) as local:
                assert (await local.get("/w/shared/api/session")).status_code == 401
                await local.post("/login", json={"password": "local-password"})
                assert (await local.get("/w/shared/api/session")).status_code == 200


@pytest.mark.parametrize("root", ["", "/remote"])
async def test_router_login_ip_lock_blocks_browser_access(root):
    gate = create_app("router-password")

    async def index(request):
        return JSONResponse({"ok": True})

    async def hello(websocket):
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()

    downstream = Starlette(routes=[Route("/", index), WebSocketRoute("/ws", hello)])
    async with serve_router(gate, root) as (address, target):
        async with device(downstream, target) as client:
            base = node_url(address, client.node_id, root)
            async with ip_browser(address, "127.0.0.2") as browser:
                assert (
                    await browser.post(
                        root + "/login", json={"password": "router-password"}
                    )
                ).status_code == 200
                assert (await browser.get(base)).status_code == 200
                websocket_url = (base + "ws").replace("http:", "ws:")
                cookies = {
                    "Cookie": browser.build_request("GET", base + "ws").headers[
                        "cookie"
                    ]
                }
                async with connect(
                    websocket_url,
                    local_addr=("127.0.0.2", 0),
                    extra_headers=cookies,
                    origin=origin(base),
                ) as websocket:
                    assert await websocket.recv() == "hello"

                for attempt, status in enumerate((401, 401, 403)):
                    response = await browser.post(
                        root + "/login",
                        json={"password": "incorrect"},
                        # This peer is not Uvicorn's trusted loopback proxy.
                        headers={"X-Forwarded-For": "198.51.100." + str(attempt + 1)},
                    )
                    assert response.status_code == status
                assert "blocked" in response.json()["error"]
                assert "restarts" in response.json()["error"]
                assert (
                    await browser.post(
                        root + "/login", json={"password": "router-password"}
                    )
                ).status_code == 403
                for path in ("/", "/login", "/api/workspaces"):
                    assert (await browser.get(root + path)).status_code == 403
                assert (await browser.get(base)).status_code == 403
                with pytest.raises(InvalidStatusCode) as rejected:
                    async with connect(
                        websocket_url,
                        local_addr=("127.0.0.2", 0),
                        extra_headers=cookies,
                        origin=origin(base),
                    ):
                        pass
                assert rejected.value.status_code == 403

            # A new connection/cookie jar cannot reset an IP's lock.
            async with ip_browser(address, "127.0.0.2") as fresh_browser:
                assert (
                    await fresh_browser.post(
                        root + "/login", json={"password": "router-password"}
                    )
                ).status_code == 403
            async with ip_browser(address, "127.0.0.3") as other_browser:
                assert (
                    await other_browser.post(
                        root + "/login", json={"password": "router-password"}
                    )
                ).status_code == 200
                assert (await other_browser.get(base)).status_code == 200


@pytest.mark.parametrize("root", ["", "/remote"])
@pytest.mark.parametrize("chunked", [False, True])
async def test_router_rejects_oversized_login_before_upload_finishes(root, chunked):
    gate = create_app("router-password")
    web = Starlette(routes=[Mount(root, app=gate)]) if root else gate
    reading = asyncio.Event()

    async def observed(scope, receive, send):
        async def observed_receive():
            message = await receive()
            if message.get("body"):
                reading.set()
            return message

        await web(scope, observed_receive, send)

    async with serve(observed) as address:
        parsed = urlsplit(address)
        reader, writer = await asyncio.open_connection(
            parsed.hostname, parsed.port, local_addr=("127.0.0.2", 0),
        )
        framing = (
            "Transfer-Encoding: chunked" if chunked else "Content-Length: 1048576"
        )
        headers = (
            "POST {0}/login HTTP/1.1\r\nHost: {1}\r\n"
            "Content-Type: application/json\r\n{2}\r\n\r\n"
        ).format(root, parsed.netloc, framing).encode("ascii")
        try:
            if chunked:
                writer.write(headers + b"800\r\n" + b" " * 2048 + b"\r\n")
                await writer.drain()
                await asyncio.wait_for(reading.wait(), 3)
                writer.write(b"801\r\n" + b" " * 2049 + b"\r\n")
            else:
                writer.write(headers + b" " * 4097)
            await writer.drain()
            # Neither upload is complete: no final chunk or remaining body is sent.
            response = await asyncio.wait_for(reader.read(), 3)
            assert response.startswith(b"HTTP/1.1 413 ")
            assert b"\r\nconnection: close\r\n" in response.lower()
            assert b"set-cookie:" not in response.lower()
        finally:
            writer.close()
            await writer.wait_closed()
        async with ip_browser(address, "127.0.0.2") as browser:
            response = await browser.post(
                root + "/login", json={"password": "router-password"},
            )
            assert response.status_code == 200


async def test_router_login_ip_lock_resets_on_success_and_restart():
    async with serve(create_app("router-password")) as address:
        async with ip_browser(address, "127.0.0.2") as browser:
            for _ in range(3):
                assert (
                    await browser.post(
                        "/login",
                        content=b"[]",
                        headers={"Content-Type": "application/json"},
                    )
                ).status_code == 422
            for _ in range(2):
                for _ in range(2):
                    assert (
                        await browser.post("/login", json={"password": "incorrect"})
                    ).status_code == 401
                await login(browser)
            for status in (401, 401, 403):
                assert (
                    await browser.post("/login", json={"password": "incorrect"})
                ).status_code == status
            assert (
                await browser.post("/login", json={"password": "router-password"})
            ).status_code == 403
    async with serve(create_app("router-password")) as address:
        async with ip_browser(address, "127.0.0.2") as browser:
            await login(browser)
            assert (await browser.get("/api/workspaces")).status_code == 200


async def test_router_login_ip_lock_counts_concurrent_failures():
    async with serve(create_app("router-password")) as address:
        async with ip_browser(address, "127.0.0.2") as browser:
            responses = await asyncio.gather(
                *(
                    browser.post("/login", json={"password": "incorrect"})
                    for _ in range(10)
                )
            )
            assert sorted(response.status_code for response in responses) == [
                401,
                401,
                403,
                403,
                403,
                403,
                403,
                403,
                403,
                403,
            ]
            assert (
                await browser.post("/login", json={"password": "router-password"})
            ).status_code == 403


async def test_router_login_ip_lock_rechecks_after_reading_request_body():
    gate = create_app("router-password")
    reading = asyncio.Event()
    release = asyncio.Event()

    async def observed(scope, receive, send):
        async def observed_receive():
            message = await receive()
            if message.get("body") == b'{"password":':
                reading.set()
            return message

        await gate(scope, observed_receive, send)

    async def body():
        yield b'{"password":'
        await release.wait()
        yield b'"router-password"}'

    async with serve(observed) as address:
        async with ip_browser(address, "127.0.0.2") as browser:
            pending = asyncio.create_task(
                browser.post(
                    "/login",
                    content=body(),
                    headers={"Content-Type": "application/json"},
                )
            )
            try:
                await asyncio.wait_for(reading.wait(), 5)
                for status in (401, 401, 403):
                    assert (
                        await browser.post("/login", json={"password": "incorrect"})
                    ).status_code == status
            finally:
                release.set()
                response = await asyncio.wait_for(pending, 5)
            assert response.status_code == 403
            assert "set-cookie" not in response.headers


async def test_workspace_passwords_and_cookies_are_isolated_between_devices(tmp_path):
    gate = create_app("router-password")
    first, first_registry = workspace_app(tmp_path / "first", "first-password")
    second, second_registry = workspace_app(tmp_path / "second", "second-password")
    async with serve_router(gate) as (address, target):
        attach_router(first, first_registry, target, 6007)
        attach_router(second, second_registry, target, 6008)
        async with first.router.lifespan_context(first):
            async with second.router.lifespan_context(second):
                await eventually(
                    lambda: first.state.router_client.connected
                    and second.state.router_client.connected
                )
                first_base = node_url(
                    address, first.state.router_client.node_id
                ).rstrip("/")
                second_base = node_url(
                    address, second.state.router_client.node_id
                ).rstrip("/")
                async with ip_browser(address) as browser:
                    await login(browser)
                    assert (
                        await browser.post(
                            first_base + "/login", json={"password": "first-password"}
                        )
                    ).status_code == 200
                    assert (
                        await browser.get(first_base + "/w/shared/api/session")
                    ).status_code == 200
                    assert (
                        await browser.get(second_base + "/w/shared/api/session")
                    ).status_code == 401
                    assert (
                        await browser.post(
                            second_base + "/login", json={"password": "first-password"}
                        )
                    ).status_code == 401
                    assert (
                        await browser.post(
                            second_base + "/login", json={"password": "second-password"}
                        )
                    ).status_code == 200
                    for base in (first_base, second_base):
                        assert (
                            await browser.get(base + "/w/shared/api/session")
                        ).status_code == 200
                    assert {
                        cookie.path
                        for cookie in browser.cookies.jar
                        if cookie.name == "pycodex_ws_auth"
                    } == {
                        urlsplit(first_base).path + "/",
                        urlsplit(second_base).path + "/",
                    }


async def test_streaming_binary_http_and_websocket_and_concurrent_requests():
    blocked = asyncio.Event()
    release = asyncio.Event()

    async def echo(request):
        return JSONResponse(
            {
                "body": (await request.body()).decode("utf-8"),
                "query": request.url.query,
                "cookie": request.headers.get("cookie", ""),
            }
        )

    async def binary(request):
        async def chunks():
            for _index in range(32):
                yield bytes(range(256)) * 256
                await asyncio.sleep(0)

        return StreamingResponse(chunks(), media_type="application/octet-stream")

    async def slow(request):
        blocked.set()
        await release.wait()
        return Response(b"released")

    async def websocket_echo(websocket):
        await websocket.accept(subprotocol="echo")
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    await websocket.send_bytes(message["bytes"])
                else:
                    await websocket.send_text(message["text"])
        except WebSocketDisconnect:
            pass

    app = Starlette(
        routes=[
            Route("/echo", echo, methods=["POST"]),
            Route("/binary", binary),
            Route("/slow", slow),
            WebSocketRoute("/ws", websocket_echo),
        ]
    )
    gate = create_app("router-password")
    async with serve_router(gate) as (address, target):
        async with device(app, target) as client:
            base = node_url(address, client.node_id)
            async with ip_browser(address) as browser:
                await login(browser)
                slow_request = asyncio.create_task(browser.get(base + "slow"))
                await asyncio.wait_for(blocked.wait(), 3)
                try:
                    payload = "中文\x00" * 100000
                    response = await browser.post(
                        base + "echo?q=%E4%B8%AD&x=2",
                        content=payload.encode("utf-8"),
                    )
                    assert response.json() == {
                        "body": payload,
                        "query": "q=%E4%B8%AD&x=2",
                        "cookie": "",
                    }
                    response = await browser.get(base + "binary")
                    assert response.content == bytes(range(256)) * 8192
                finally:
                    release.set()
                assert (await slow_request).content == b"released"
                url = (base + "ws").replace("http:", "ws:")
                with pytest.raises(InvalidStatusCode):
                    async with connect(url):
                        pass
                cookies = {
                    "Cookie": "pycodex_router_auth="
                    + browser.cookies["pycodex_router_auth"]
                }
                with pytest.raises(InvalidStatusCode):
                    async with connect(
                        url, extra_headers=cookies, origin="https://other.invalid"
                    ):
                        pass
                async with connect(
                    url,
                    extra_headers=cookies,
                    origin=origin(base),
                    subprotocols=["echo"],
                ) as websocket:
                    assert websocket.subprotocol == "echo"
                    await websocket.send("hello 中文")
                    assert await websocket.recv() == "hello 中文"
                    await websocket.send(bytes(range(256)))
                    assert await websocket.recv() == bytes(range(256))


async def test_disconnect_reconnect_does_not_replay_an_accepted_request():
    accepted = asyncio.Event()
    finish = asyncio.Event()
    calls = []

    async def once(request):
        calls.append(await request.body())
        accepted.set()
        await finish.wait()
        return Response(b"done")

    app = Starlette(routes=[Route("/once", once, methods=["POST"])])
    gate = create_app("router-password")
    async with serve_router(gate) as (address, target):
        async with device(app, target) as client:
            original = gate.state.router.nodes[client.node_id]
            async with ip_browser(address) as browser:
                await login(browser)
                pending = asyncio.create_task(
                    browser.post(
                        node_url(address, client.node_id) + "once", content=b"execute"
                    )
                )
                await asyncio.wait_for(accepted.wait(), 3)
                try:
                    await original.websocket.close(code=1012)
                    response = await pending
                    assert response.status_code == 503
                    await eventually(
                        lambda: gate.state.router.nodes.get(client.node_id) is not None
                        and gate.state.router.nodes[client.node_id] is not original
                    )
                    assert calls == [b"execute"]
                finally:
                    finish.set()
        await eventually(lambda: not gate.state.router.nodes)
        assert not gate.state.router.streams


@pytest.mark.parametrize("stage", ["registration", "session", "invalid_reply"])
async def test_refusals_then_router_failures_keep_reconnecting(
    stage, monkeypatch, caplog
):
    router = create_app().state.router
    policy_closes = 0

    async def control(websocket):
        nonlocal policy_closes
        if not policy_closes:
            await router.control(websocket)
            return
        policy_closes -= 1
        await websocket.accept()
        registration = decode(await websocket.receive_text())
        if stage == "session":
            await websocket.send_text(
                encode(
                    {
                        "type": "registered",
                        "id": device_id(registration["identity_key"]),
                    }
                )
            )
        elif stage == "invalid_reply":
            await websocket.send_text(encode({"type": "unexpected"}))
        await websocket.close(code=1008)

    gate = Starlette(routes=[WebSocketRoute("/tunnel/control", control)])
    async with serve(gate) as address:
        client = RouterClient(
            Starlette(),
            client_target(address, router.registration_key),
            "device",
            6007,
            lambda: [],
            reconnect_seconds=0,
        )
        real_connect = client._connect
        refusals = 0
        attempts = []

        @asynccontextmanager
        async def flaky_connect(url):
            nonlocal refusals
            attempts.append(client.node_id)
            if refusals:
                refusals -= 1
                raise ConnectionRefusedError("router is restarting")
            async with real_connect(url) as channel:
                yield channel

        monkeypatch.setattr(client, "_connect", flaky_connect)
        await client.start()
        try:
            await eventually(lambda: client.connected)
            original = router.nodes[client.node_id]
            refusals = 3
            policy_closes = 6
            await original.websocket.close(code=1012)
            await eventually(
                lambda: client._task.done()
                or (
                    client.connected
                    and len(attempts) >= 11
                    and router.nodes.get(client.node_id) is not None
                    and router.nodes[client.node_id] is not original
                )
            )
            assert not client._task.done(), "a router failure stopped the connector"
            assert refusals == policy_closes == 0
            assert attempts == [client.node_id] * len(attempts)
            reason = "ValueError" if stage == "invalid_reply" else "close_code=1008"
            assert reason in caplog.text
        finally:
            await client.close()


async def test_connection_cleanup_failure_keeps_reconnecting(monkeypatch):
    gate = create_app()
    async with serve(create_tunnel_app(gate.state.router)) as address:
        client = RouterClient(
            Starlette(),
            client_target(address, gate.state.router.registration_key),
            "device",
            6007,
            lambda: [],
            reconnect_seconds=0,
        )
        cleanup = client._close_channels
        node_id = client.node_id
        fail_cleanup = True

        async def close_channels():
            nonlocal fail_cleanup
            await cleanup()
            if fail_cleanup:
                fail_cleanup = False
                raise ConnectionClosedError(None, None)

        monkeypatch.setattr(client, "_close_channels", close_channels)
        await client.start()
        try:
            await eventually(lambda: client.connected)
            original = gate.state.router.nodes[client.node_id]
            await original.websocket.close(code=1012)
            await eventually(
                lambda: client._task.done()
                or (
                    client.connected
                    and gate.state.router.nodes.get(client.node_id) is not None
                    and gate.state.router.nodes[client.node_id] is not original
                )
            )
            assert not client._task.done(), "a cleanup failure stopped the connector"
            assert not fail_cleanup
            assert client.node_id == node_id
        finally:
            await client.close()
        assert client._task is None
        assert not client.connected


async def test_many_failures_before_and_after_registration_keep_the_same_id(
    monkeypatch,
):
    gate = create_app()
    app = Starlette()
    async with serve(create_tunnel_app(gate.state.router)) as address:
        client = RouterClient(
            app,
            client_target(address, gate.state.router.registration_key),
            "device",
            6007,
            lambda: [],
            reconnect_seconds=0,
        )
        real_connect = client._connect
        attempts = []
        remaining_failures = 100

        @asynccontextmanager
        async def flaky_connect(url):
            nonlocal remaining_failures
            if url == client.control_url:
                attempts.append(client.node_id)
                if remaining_failures:
                    remaining_failures -= 1
                    error = (
                        ConnectionRefusedError,
                        asyncio.TimeoutError,
                        InvalidHandshake,
                    )[remaining_failures % 3]
                    raise error("temporary network failure")
            async with real_connect(url) as channel:
                yield channel

        monkeypatch.setattr(client, "_connect", flaky_connect)
        await client.start()
        try:
            await eventually(lambda: client.connected)
            assert len(attempts) == 101
            original = gate.state.router.nodes[client.node_id]
            remaining_failures = 100
            await original.websocket.close(code=1012)
            await eventually(
                lambda: client.connected
                and gate.state.router.nodes.get(client.node_id) is not None
                and gate.state.router.nodes.get(client.node_id) is not original
            )
            assert attempts == [client.node_id] * 202
            assert list(gate.state.router.nodes) == [client.node_id]
        finally:
            await client.close()
        assert not client.connected
        assert client._task is None


async def test_offline_start_and_router_restart_preserve_workspace_and_default_id(
    tmp_path,
):
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    address = "http://127.0.0.1:{0}".format(port)
    key_path = tmp_path / "router.key"
    aliases_path = tmp_path / "router-aliases.json"
    registration_key = load_registration_key(key_path)
    app, registry = workspace_app(tmp_path / "workspace")
    attach_router(app, registry, client_target(address, registration_key), 6007)
    client = app.state.router_client
    client.reconnect_seconds = 0.01
    try:
        # The reserved port isn't listening: the local workspace must still start.
        async with serve(app) as local_address:
            assert not client.connected
            assert not client._task.done()
            session = registry.get("shared").manager.get()
            async with httpx.AsyncClient(
                base_url=local_address, trust_env=False
            ) as local:
                initial = (await local.get("/w/shared/api/session")).json()
                await local.post(
                    "/w/shared/api/session/message", json={"prompt": "work offline"}
                )
                await eventually(
                    lambda: bool(session.snapshot()["turns"])
                    and session.snapshot()["turns"][-1]["response"] == "done"
                )
                gate = create_app(
                    "router-password",
                    registration_key=load_registration_key(key_path),
                    aliases_path=aliases_path,
                )
                async with serve_router(gate, listener=listener) as (web, unused):
                    await eventually(lambda: client.connected)
                    node_id = client.node_id
                    assert node_id
                    assert list(gate.state.router.nodes) == [node_id]
                    async with ip_browser(web) as browser:
                        await login(browser)
                        renamed = await browser.post(
                            "/api/nodes/" + node_id + "/alias",
                            json={"alias": "开发机"},
                        )
                        assert renamed.status_code == 200
                await eventually(lambda: not client.connected)
                assert session.runtime.accepts_input
                assert not client._task.done()

                replacement_listener = socket.socket()
                replacement_listener.setsockopt(
                    socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
                )
                replacement_listener.bind(("127.0.0.1", port))
                restarted_gate = create_app(
                    "router-password",
                    registration_key=load_registration_key(key_path),
                    aliases_path=aliases_path,
                )
                async with serve_router(
                    restarted_gate, listener=replacement_listener
                ) as (web, unused):
                    await eventually(lambda: client.connected)
                    assert list(restarted_gate.state.router.nodes) == [node_id]
                    assert client.node_id == node_id
                    assert registry.get("shared").manager.get() is session
                    async with ip_browser(web) as browser:
                        await login(browser)
                        inventory = (await browser.get("/api/workspaces")).json()
                        assert inventory["nodes"][0]["alias"] == "开发机"
                        resumed = (
                            await browser.get(
                                node_url(web, node_id) + "w/shared/api/session"
                            )
                        ).json()
                        assert resumed["session_id"] == initial["session_id"]
                        assert resumed["snapshot"]["turns"][-1]["response"] == "done"
        assert client._task is None
        assert session.runtime.agent.is_shutdown
    finally:
        listener.close()


@pytest.mark.parametrize("password", [None, "router-password"])
async def test_registration_retries_occupied_id_without_browser_credentials(
    password, caplog
):
    app = Starlette()
    gate = create_app(password)
    async with serve_router(gate) as (address, target):
        async with device(app, target) as first:
            headers = gate.state.router.nodes[first.node_id].websocket.headers
            assert headers["authorization"] == "Bearer " + gate.state.router.registration_key
            assert "cookie" not in headers
            duplicate = RouterClient(
                app,
                target,
                "duplicate",
                6007,
                lambda: [],
                reconnect_seconds=0.01,
            )
            duplicate._identity_key = first._identity_key
            duplicate.node_id = first.node_id
            await duplicate.start()
            try:
                await eventually(
                    lambda: sum(
                        "NodeIdInUse" in record.getMessage()
                        for record in caplog.records
                    )
                    >= 3
                )
                assert not duplicate.connected
                assert not duplicate._task.done()
                assert (
                    gate.state.router.nodes[first.node_id].name
                    == "device/127.0.0.1:6007"
                )
                await first.close()
                await eventually(lambda: duplicate.connected)
                assert list(gate.state.router.nodes) == [first.node_id]
                assert (
                    gate.state.router.nodes[first.node_id].name
                    == "duplicate/127.0.0.1:6007"
                )
            finally:
                await duplicate.close()
        async with ip_browser(address) as browser:
            if password:
                await login(browser)
            response = await browser.get(node_url(address, first.node_id) + "w/shared/")
            assert response.status_code == 503
            assert "offline" in response.json()["error"]


async def test_connection_limits_timeouts_and_inventory_changes():
    inventory = [workspace_record()]
    identity_key = secrets.token_hex(32)
    node_id = device_id(identity_key)
    gate = create_app("router-password", connect_timeout=0.15, max_connections=1)
    async with serve_router(gate) as (address, target):
        async with ip_browser(address, timeout=3) as browser:
            await login(browser)
            async with tunnel_connect(target) as control:
                await control.send(
                    encode(
                        {
                            "type": "register",
                            "version": 2,
                            "identity_key": identity_key,
                            "name": "device",
                            "port": 6007,
                            "workspaces": inventory,
                        }
                    )
                )
                assert decode(await control.recv())["type"] == "registered"
                prefix = node_url(address, node_id)
                pending = asyncio.create_task(browser.get(prefix + "w/shared/"))
                assert decode(await control.recv())["type"] == "connect"
                limited = await browser.get(prefix + "w/shared/")
                assert limited.status_code == 503
                assert "too many" in limited.json()["error"]
                assert (await pending).status_code == 503
                assert not gate.state.router.streams
                await control.send(
                    encode(
                        {
                            "type": "workspaces",
                            "workspaces": [workspace_record("new-workspace")],
                        }
                    )
                )
                await eventually(
                    lambda: gate.state.router.nodes[node_id].workspaces[0]["id"]
                    == "new-workspace"
                )
                snapshot = (await browser.get("/api/workspaces")).json()
                assert (
                    snapshot["workspaces"][0]["url"]
                    == urlsplit(prefix).path + "w/new-workspace/"
                )


async def test_router_can_be_mounted_under_a_path_prefix(tmp_path):
    gate = create_app("router-password")
    app, registry = workspace_app(tmp_path / "workspace")
    async with serve_router(gate, "/remote") as (address, target):
        async with device(app, target) as client:
            async with ip_browser(address) as browser:
                response = await browser.get("/remote/")
                assert response.headers["location"].startswith("/remote/login?")
                await browser.post(
                    "/remote/login", json={"password": "router-password"}
                )
                inventory = (await browser.get("/remote/api/workspaces")).json()
                destination = node_url(address, client.node_id, "/remote") + "w/shared/"
                assert inventory["workspaces"][0]["url"] == urlsplit(destination).path
                assert (await browser.get(destination)).status_code == 200
                assert (await browser.get(destination + "image.png")).status_code == 200


@pytest.mark.parametrize("failure", [False, True])
async def test_lifespan_attachment_keeps_runtime_alive_across_reconnect(tmp_path, failure):
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def complete(prompt, count):
        entered.set()
        await finish.wait()
        if failure:
            raise RuntimeError("failed after reconnect")
        return ModelResponse([AssistantMessage("finished after reconnect")])

    app, registry = workspace_app(tmp_path / "workspace", response_factory=complete)
    gate = create_app("router-password")
    async with serve_router(gate) as (address, target):
        attach_router(app, registry, target, 6007)
        app.state.router_client.refresh_seconds = 0.02
        async with app.router.lifespan_context(app):
            await eventually(lambda: app.state.router_client.connected)
            node_id = app.state.router_client.node_id
            prefix = node_url(address, node_id) + "w/shared/"
            session = registry.get("shared").manager.get()
            original = gate.state.router.nodes[node_id]
            async with ip_browser(address) as browser:
                await login(browser)
                await browser.post(
                    prefix + "api/session/message",
                    json={"prompt": "continue"},
                )
                await asyncio.wait_for(entered.wait(), 3)
                try:
                    await original.websocket.close(code=1012)
                    await eventually(
                        lambda: gate.state.router.nodes.get(node_id) is not None
                        and gate.state.router.nodes[node_id] is not original
                    )
                    assert session.runtime.accepts_input
                    assert session.runtime.agent.is_running
                finally:
                    finish.set()
                await eventually(lambda: not session.runtime.agent.is_running)
                payload = (await browser.get(prefix + "api/session")).json()
                state = payload["snapshot"]
                if failure:
                    assert "failed after reconnect" in state["turns"][-1]["error"]
                else:
                    assert state["turns"][-1]["response"] == "finished after reconnect"
                await eventually(
                    lambda: gate.state.router.nodes[node_id].workspaces[0]["unread"]
                )
                directory = (await browser.get("/api/workspaces")).json()
                assert directory["workspaces"][0]["unread"]
                response = await browser.post(
                    prefix + "api/sessions/" + payload["session_id"] + "/read",
                    json={"final_id": state["last_final_id"]},
                )
                assert response.status_code == 200
                await eventually(
                    lambda: not gate.state.router.nodes[node_id].workspaces[0]["unread"]
                )
        assert session.runtime.agent.is_shutdown
        await eventually(lambda: not gate.state.router.nodes)


async def test_invalid_inventory_cannot_break_the_directory():
    gate = create_app("router-password")
    invalid = [
        {"work_dir": b"\x00"},
        {"state_path": "\ud800"},
        {"session_count": True},
        {"running": {}},
        {"unread": 1},
        {"extra": b"\x00"},
    ]
    async with serve_router(gate) as (address, target):
        async with device(Starlette(), target) as good:
            async with ip_browser(address) as browser:
                await login(browser)
                for update in (False, True):
                    for changes in invalid:
                        record = workspace_record()
                        record.update(changes)
                        async with tunnel_connect(target) as channel:
                            await channel.send(
                                encode(
                                    {
                                        "type": "register",
                                        "version": 2,
                                        "identity_key": secrets.token_hex(32),
                                        "name": "invalid",
                                        "port": 6007,
                                        "workspaces": [
                                            workspace_record() if update else record
                                        ],
                                    }
                                )
                            )
                            if update:
                                assert (
                                    decode(await channel.recv())["type"] == "registered"
                                )
                                await channel.send(
                                    encode(
                                        {
                                            "type": "workspaces",
                                            "workspaces": [record],
                                        }
                                    )
                                )
                            with pytest.raises(ConnectionClosedError) as rejected:
                                await channel.recv()
                            assert rejected.value.rcvd.code == 1008
                        await eventually(
                            lambda: list(gate.state.router.nodes) == [good.node_id]
                        )
                        response = await browser.get("/api/workspaces")
                        assert response.status_code == 200
                        assert response.json()["nodes"][0]["id"] == good.node_id


async def test_offline_route_requires_its_original_identity_key():
    gate = create_app("router-password")
    async with serve_router(gate) as (address, target):
        async with device(Starlette(), target) as owner:
            node_id = owner.node_id
            await owner.close()
            await eventually(lambda: node_id not in gate.state.router.nodes)
            async with tunnel_connect(target) as impostor:
                await impostor.send(
                    encode(
                        {
                            "type": "register",
                            "version": 2,
                            "id": node_id,
                            "identity_key": secrets.token_hex(32),
                            "name": "impostor",
                            "port": 6007,
                            "workspaces": [],
                        }
                    )
                )
                with pytest.raises(ConnectionClosedError):
                    await impostor.recv()
            assert node_id not in gate.state.router.nodes
            await owner.start()
            await eventually(lambda: owner.connected)
            assert owner.node_id == node_id
            async with ip_browser(address) as browser:
                await login(browser)
                response = await browser.get("/api/workspaces")
                key_leaked = owner._identity_key in response.text
                assert not key_leaked


async def test_data_channel_is_bound_to_its_device(monkeypatch):
    async def index(request):
        return Response("owner")

    gate = create_app("router-password")
    entered = asyncio.Event()
    release = asyncio.Event()
    async with serve_router(gate) as (address, target):
        async with device(Starlette(routes=[Route("/", index)]), target) as owner:
            serve_request = owner._serve

            async def delayed(stream_id):
                entered.set()
                await release.wait()
                await serve_request(stream_id)

            monkeypatch.setattr(owner, "_serve", delayed)
            async with ip_browser(address) as browser:
                await login(browser)
                pending = asyncio.create_task(
                    browser.get(node_url(address, owner.node_id))
                )
                try:
                    await asyncio.wait_for(entered.wait(), 5)
                    stream_id = next(iter(gate.state.router.streams))
                    async with tunnel_connect(target, "data") as impostor:
                        await impostor.send(
                            encode(
                                {
                                    "stream_id": stream_id,
                                    "identity_key": secrets.token_hex(32),
                                }
                            )
                        )
                        with pytest.raises(ConnectionClosedError) as rejected:
                            await impostor.recv()
                        assert rejected.value.rcvd.code == 1008
                finally:
                    release.set()
                    response = await asyncio.wait_for(pending, 5)
                assert response.status_code == 200
                assert response.text == "owner"


async def test_trusted_devices_use_ip_paths_and_keep_browser_header_checks():
    async def page(request):
        response = JSONResponse(
            {"router_cookie_received": "pycodex_router_auth" in request.cookies}
        )
        response.headers.append(
            "set-cookie", "pycodex_router_auth=poison; Path=/; Domain=router.localhost"
        )
        response.headers.append(
            "set-cookie", "pycodex_ws_auth=fixture; Path=/; Domain=router.localhost"
        )
        response.headers["service-worker-allowed"] = "/"
        response.headers["access-control-allow-origin"] = "*"
        return response

    gate = create_app("router-password")
    async with serve_router(gate) as (address, target):
        async with device(
            Starlette(routes=[Route("/", page, methods=["GET", "POST"])]), target
        ) as first:
            async with device(
                Starlette(routes=[Route("/", lambda request: Response("second"))]),
                target,
            ) as second:
                async with ip_browser(address) as browser:
                    await login(browser)
                    first_url = node_url(address, first.node_id)
                    second_url = node_url(address, second.node_id)
                    assert origin(first_url) == origin(second_url) == address
                    response = await browser.get(first_url)
                    assert response.json() == {"router_cookie_received": False}
                    assert (
                        response.headers["service-worker-allowed"]
                        == urlsplit(first_url).path
                    )
                    assert "access-control-allow-origin" not in response.headers
                    cookies = response.headers.get_list("set-cookie")
                    assert len(cookies) == 1
                    assert "Domain=" not in cookies[0]
                    assert "HttpOnly" in cookies[0]
                    inventory = await browser.get("/api/workspaces")
                    assert inventory.status_code == 200
                    assert {node["url"] for node in inventory.json()["nodes"]} == {
                        urlsplit(first_url).path,
                        urlsplit(second_url).path,
                    }
                    assert (await browser.get(second_url)).text == "second"
                    for destination in (
                        address + "/api/workspaces",
                        second_url,
                    ):
                        assert (
                            await browser.get(
                                destination, headers={"Origin": "https://other.example"}
                            )
                        ).status_code == 403
                    assert (await browser.get(urlsplit(first_url).path)).status_code == 200
                    assert (await browser.post(first_url)).status_code == 200


@pytest.mark.parametrize("root", ["", "/remote"])
async def test_device_aliases_are_authenticated_and_do_not_change_identity(
    tmp_path, root
):
    path = tmp_path / "router-aliases.json"
    gate = create_app("router-password", aliases_path=path)
    app = Starlette(routes=[Route("/", lambda request: Response("device"))])
    async with serve_router(gate, root) as (address, target):
        async with device(
            app, target, workspaces=lambda: [workspace_record()]
        ) as first:
            async with device(
                app, target, workspaces=lambda: [workspace_record()]
            ) as second:
                async with ip_browser(address) as browser:
                    endpoint = root + "/api/nodes/" + first.node_id + "/alias"
                    registered = path.read_bytes()
                    response = await browser.post(endpoint, json={"alias": "开发机"})
                    assert response.status_code == 401
                    assert path.read_bytes() == registered
                    await browser.post(
                        root + "/login", json={"password": "router-password"}
                    )
                    before = (await browser.get(root + "/api/workspaces")).json()
                    original = next(
                        node for node in before["nodes"] if node["id"] == first.node_id
                    )
                    assert (
                        original["name"] == gate.state.router.nodes[second.node_id].name
                    )

                    response = await browser.post(endpoint, json={"alias": "  开发机  "})
                    assert response.json() == {"alias": "开发机"}
                    after = (await browser.get(root + "/api/workspaces")).json()
                    after_nodes = {node["id"]: node for node in after["nodes"]}
                    assert after_nodes[first.node_id] == dict(original, alias="开发机")
                    assert after["workspaces"] == before["workspaces"]
                    assert gate.state.router.nodes[first.node_id].name == original["name"]
                    assert after_nodes[second.node_id]["alias"] == ""
                    assert (await browser.get(original["url"])).text == "device"
                    assert json.loads(path.read_text(encoding="utf-8")) == {
                        first.node_id: "开发机", second.node_id: ""
                    }
                    if os.name == "posix":
                        assert path.stat().st_mode & 0o777 == 0o600
                    saved = path.read_bytes()

                    for payload in [
                        {},
                        [],
                        {"alias": None},
                        {"alias": 1},
                        {"alias": ""},
                        {"alias": "   "},
                        {"alias": "a" * 129},
                        {"alias": "two\nlines"},
                    ]:
                        assert (
                            await browser.post(endpoint, json=payload)
                        ).status_code == 422
                    for body in [b"{", b'{"alias":"\\ud800"}']:
                        assert (
                            await browser.post(
                                endpoint,
                                content=body,
                                headers={"Content-Type": "application/json"},
                            )
                        ).status_code == 422
                    assert (
                        await browser.post(
                            endpoint,
                            json={"alias": "other"},
                            headers={"Origin": "https://other.example"},
                        )
                    ).status_code == 403
                    assert (
                        await browser.post(
                            root + "/api/nodes/missing/alias", json={"alias": "other"}
                        )
                    ).status_code == 404
                    assert gate.state.router.aliases == {
                        first.node_id: "开发机", second.node_id: ""
                    }
                    assert path.read_bytes() == saved

                    response = await browser.post(endpoint, json={"alias": "机"})
                    assert response.json() == {"alias": "机"}
                    updated = (await browser.get(root + "/api/workspaces")).json()
                    updated_nodes = {node["id"]: node for node in updated["nodes"]}
                    assert updated_nodes[first.node_id] == dict(original, alias="机")
                    assert json.loads(path.read_text(encoding="utf-8")) == {
                        first.node_id: "机", second.node_id: ""
                    }


async def test_device_aliases_cannot_be_changed_without_a_router_password():
    async with serve_router(create_app()) as (address, target):
        async with device(Starlette(), target) as client:
            async with ip_browser(address) as browser:
                response = await browser.post(
                    "/api/nodes/" + client.node_id + "/alias", json={"alias": "device"}
                )
                assert response.status_code == 401


@pytest.mark.parametrize("root", ["", "/remote"])
async def test_device_order_is_shared_persistent_and_preserves_offline_slots(
    tmp_path, root
):
    path = tmp_path / "router-aliases.json"
    gate = create_app("router-password", aliases_path=path)
    app = Starlette()
    async with serve_router(gate, root) as (address, target):
        async with device(app, target) as first:
            async with device(app, target) as second:
                async with device(app, target) as third:
                    original = [first.node_id, second.node_id, third.node_id]
                    async with ip_browser(address) as browser:
                        endpoint = root + "/api/nodes/order"
                        saved = path.read_bytes()
                        assert (await browser.post(
                            endpoint, json={"order": original[::-1]}
                        )).status_code == 401
                        assert path.read_bytes() == saved
                        await browser.post(
                            root + "/login", json={"password": "router-password"}
                        )
                        assert (await browser.post(
                            root + "/api/nodes/" + first.node_id + "/alias",
                            json={"alias": "Build server"},
                        )).status_code == 200
                        before = (await browser.get(root + "/api/workspaces")).json()
                        assert [node["id"] for node in before["nodes"]] == original
                        order = original[::-1]
                        response = await browser.post(endpoint, json={"order": order})
                        assert response.json() == {"order": order}
                        assert list(json.loads(path.read_text(encoding="utf-8"))) == order
                        if os.name == "posix":
                            assert path.stat().st_mode & 0o777 == 0o600
                        async with ip_browser(address) as other:
                            await other.post(
                                root + "/login", json={"password": "router-password"}
                            )
                            after = (await other.get(root + "/api/workspaces")).json()
                        assert after["nodes"] == before["nodes"][::-1]
                        assert after["workspaces"] == before["workspaces"][::-1]

                        saved = path.read_bytes()
                        for payload in (
                            {}, [], {"order": None}, {"order": "invalid"},
                            {"order": [first.node_id, first.node_id]},
                            {"order": [1]}, {"order": [{}]}, {"order": ["missing"]},
                        ):
                            assert (await browser.post(
                                endpoint, json=payload
                            )).status_code == 422
                        assert (await browser.post(
                            endpoint, json={"order": []}
                        )).status_code == 409
                        assert (await browser.post(
                            endpoint, json={"order": order + ["a" * 52]}
                        )).status_code == 409
                        assert (await browser.post(
                            endpoint, json={"order": order},
                            headers={"Origin": "https://other.example"},
                        )).status_code == 403
                        assert (await browser.post(
                            endpoint, content=b" " * (64 * 1024 + 1),
                            headers={"Content-Type": "application/json"},
                        )).status_code == 413
                        assert path.read_bytes() == saved

                        await second.close()
                        await eventually(lambda: second.node_id not in gate.state.router.nodes)
                        assert (await browser.post(
                            endpoint, json={"order": order}
                        )).status_code == 409
                        assert (await browser.post(
                            endpoint, json={"order": [first.node_id, third.node_id]}
                        )).status_code == 200
                        assert list(json.loads(path.read_text(encoding="utf-8"))) == original
                        await second.start()
                        await eventually(lambda: second.connected)
                        inventory = (await browser.get(root + "/api/workspaces")).json()
                        assert [node["id"] for node in inventory["nodes"]] == original

                        async with device(app, target) as newcomer:
                            inventory = (await browser.get(root + "/api/workspaces")).json()
                            assert [node["id"] for node in inventory["nodes"]] == original + [newcomer.node_id]
                        # Reload the file and reconnect in a different arrival order.
                        restarted = create_app("router-password", aliases_path=path)
                        for client in (first, second, third):
                            await client.close()
                        async with serve_router(restarted, root) as (web, new_target):
                            async with AsyncExitStack() as stack:
                                for client in (third, second, first):
                                    channel = await stack.enter_async_context(
                                        tunnel_connect(new_target)
                                    )
                                    await channel.send(encode({
                                        "type": "register", "version": 2,
                                        "identity_key": client._identity_key,
                                        "name": "device", "port": 6007,
                                        "workspaces": [workspace_record()],
                                    }))
                                    assert decode(await channel.recv())["id"] == client.node_id
                                async with ip_browser(web) as reader:
                                    await reader.post(
                                        root + "/login",
                                        json={"password": "router-password"},
                                    )
                                    inventory = (await reader.get(root + "/api/workspaces")).json()
                                    assert [node["id"] for node in inventory["nodes"]] == original
                                    assert inventory["nodes"][0]["alias"] == "Build server"
                        assert list(restarted.state.router.aliases)[:3] == original


async def test_device_order_requires_a_router_password():
    async with serve_router(create_app()) as (address, target):
        async with device(Starlette(), target) as client:
            async with ip_browser(address) as browser:
                assert (await browser.post(
                    "/api/nodes/order", json={"order": [client.node_id]}
                )).status_code == 401


@pytest.mark.parametrize(
    "contents", ["{", "[]", '{"invalid": "alias"}', '{"' + "a" * 52 + '": 42}']
)
def test_invalid_alias_files_are_not_replaced(tmp_path, contents):
    path = tmp_path / "router-aliases.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        create_app(aliases_path=path)
    assert path.read_text(encoding="utf-8") == contents


def test_alias_write_failure_keeps_previous_state(tmp_path, monkeypatch):
    path = tmp_path / "router-aliases.json"
    node_id = "a" * 52
    other_id = "b" * 52
    path.write_text(json.dumps({node_id: "original", other_id: ""}), encoding="utf-8")
    router = create_app(aliases_path=path).state.router
    before = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr("pycodex_router.server.os.replace", fail_replace)
        with pytest.raises(OSError, match="disk full"):
            router._save_aliases({other_id: "", node_id: "changed"})
    assert router.aliases == {node_id: "original", other_id: ""}
    assert list(router.aliases) == [node_id, other_id]
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("router-aliases.json.*"))


def test_registration_key_file_is_private_and_reused(tmp_path):
    path = tmp_path / "router" / "router.key"
    first = load_registration_key(path)
    assert re.fullmatch("[1-9A-HJ-NP-Za-km-z]{16,22}", first)
    assert load_registration_key(path) == first
    assert load_registration_key(tmp_path / "another.key") != first
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("contents", [b"", b"invalid", b"\xff", b"0" * 22, b"a" * 23])
def test_invalid_registration_key_file_is_not_replaced(tmp_path, contents):
    path = tmp_path / "router.key"
    path.write_bytes(contents)
    path.chmod(0o600)
    with pytest.raises(ValueError, match="invalid router registration key file"):
        load_registration_key(path)
    assert path.read_bytes() == contents


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
def test_registration_key_file_rejects_shared_permissions(tmp_path):
    path = tmp_path / "router.key"
    load_registration_key(path)
    path.chmod(0o644)
    with pytest.raises(ValueError, match="owner-only permissions"):
        load_registration_key(path)


@pytest.mark.parametrize("root", ["", "/remote"])
async def test_registration_key_requires_browser_login_and_is_not_cached(root):
    gate = create_app("router-password")
    key = gate.state.router.registration_key
    async with serve_router(gate, root) as (address, target):
        async with ip_browser(address) as browser:
            path = root + "/api/registration-key"
            assert (await browser.get(path)).status_code == 401
            assert key not in (await browser.get(root + "/login")).text
            await browser.post(root + "/login", json={"password": "router-password"})
            response = await browser.get(path)
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
            assert response.json()["key"] == key
            assert key not in (await browser.get(root + "/")).text
            assert key not in (await browser.get(root + "/api/workspaces")).text
            assert (
                await browser.get(path, headers={"Origin": "https://other.example"})
            ).status_code == 403
            for _attempt in range(3):
                await browser.post(root + "/login", json={"password": "incorrect"})
            assert (await browser.get(path)).status_code == 403


async def test_registration_key_is_not_exposed_by_an_app_without_a_password():
    async with serve_router(create_app()) as (address, target):
        async with ip_browser(address) as browser:
            assert (await browser.get("/api/registration-key")).status_code == 401


async def test_tunnels_require_registration_key_instead_of_browser_credentials():
    gate = create_app("router-password")
    async with serve_router(gate) as (address, target):
        tunnel, key = parse_target(target)
        async with ip_browser(address) as browser:
            await login(browser)
            headers = [
                {},
                {"Authorization": "Bearer incorrect"},
                {"Authorization": "Bearer router-password"},
                {"Cookie": "pycodex_router_auth=" + browser.cookies["pycodex_router_auth"]},
                {"Authorization": "Bearer " + key, "Origin": address},
            ]
            for endpoint in ("control", "data"):
                for invalid_headers in headers:
                    with pytest.raises(InvalidStatusCode) as rejected:
                        async with connect(
                            tunnel_url(tunnel, endpoint), extra_headers=invalid_headers
                        ):
                            pass
                    assert rejected.value.status_code == 403
                    assert gate.state.router.nodes == {}
                    assert gate.state.router.streams == {}
            async with device(Starlette(), target) as client:
                assert list(gate.state.router.nodes) == [client.node_id]


async def test_wrong_registration_key_retries_without_logging_credentials(caplog):
    gate = create_app("router-password", registration_key="a" * 22)
    async with serve_router(gate) as (address, target):
        tunnel, key = parse_target(target)
        wrong_key = "b" * 22
        client = RouterClient(
            Starlette(),
            client_target(tunnel, wrong_key),
            "device",
            6007,
            lambda: [],
            reconnect_seconds=0.01,
        )
        await client.start()
        try:
            await eventually(
                lambda: sum(
                    "InvalidStatusCode" in record.getMessage()
                    for record in caplog.records
                ) >= 3
            )
            assert not client.connected
            assert not client._task.done()
            assert not gate.state.router.nodes
            assert "status_code=403" in caplog.text
            assert key not in caplog.text
            assert wrong_key not in caplog.text
            assert wrong_key not in client.control_url
            assert wrong_key not in client.data_url
        finally:
            await client.close()


@pytest.mark.parametrize("password_args", [(), ("--password", "")])
async def test_router_cli_requires_a_nonempty_password(password_args):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pycodex_router",
        "--listen",
        "127.0.0.1:0",
        "--tunnel-listen",
        "localhost:0",
        *password_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        output, error = await asyncio.wait_for(process.communicate(), 5)
        assert process.returncode == 2
        assert b"--password" in error
        assert b"Uvicorn running" not in output
        assert b"pycodex-ws --listen-to" not in output
    finally:
        if process.returncode is None:
            process.terminate()
            await asyncio.wait_for(process.communicate(), 10)


@pytest.mark.parametrize("tunnel_host", ["localhost", "0.0.0.0"])
async def test_cli_starts_two_listeners_and_attaches_an_owned_workspace(
    tmp_path, tunnel_host
):
    config = tmp_path / "workspaces.json"
    config.write_text('{"workspaces": []}\n', encoding="utf-8")
    router = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pycodex_router",
        "--listen",
        "127.0.0.1:0",
        "--tunnel-listen",
        tunnel_host + ":0",
        "--password",
        "router-password",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    workspace = None
    try:

        async def addresses():
            result = []
            target = None
            while len(result) < 2 or target is None:
                line = await router.stdout.readline()
                assert line, "router exited before opening its listeners"
                match = re.search(
                    rb"http://(?:127\.0\.0\.1|localhost|0\.0\.0\.0):[0-9]+", line
                )
                if match:
                    result.append(match.group().decode("ascii"))
                prefix = b"  pycodex-ws --listen-to "
                if line.startswith(prefix):
                    target = line[len(prefix) :].decode("ascii").strip()
            return result, target

        (public, tunnel), target = await asyncio.wait_for(addresses(), 10)
        registration_key = (Path(os.environ["CODEX_HOME"]) / "router.key").read_text(
            encoding="ascii"
        ).strip()
        target_address, target_key = parse_target(target)
        assert target_key == registration_key
        assert urlsplit(target_address).port == urlsplit(tunnel).port
        assert urlsplit(target_address).scheme == "ws"
        if tunnel_host == "0.0.0.0":
            assert "ROUTER_HOST" in target
            target = target.replace("ROUTER_HOST", "127.0.0.1")
            tunnel = tunnel.replace("0.0.0.0", "127.0.0.1")
        else:
            assert urlsplit(target_address).hostname == "localhost"
        probe = RouterClient(Starlette(), target, "probe", 0, lambda: [])
        async with probe._connect(probe.control_url) as channel:
            await probe._register(channel)
            assert urlsplit(probe.api_url).port == urlsplit(public).port
            assert urlsplit(probe.api_url).hostname == urlsplit(probe.control_url).hostname
        workspace = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "workspace_server",
            "--listen",
            "0.0.0.0:0",
            "--workspace-config",
            str(config),
            "--listen-to",
            target,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        async with ip_browser(public) as browser:

            async def registered():
                while True:
                    if workspace.returncode is not None:
                        raise AssertionError(
                            (await workspace.stdout.read()).decode(
                                "utf-8", errors="replace"
                            )
                        )
                    try:
                        await login(browser)
                    except httpx.ConnectError:
                        await asyncio.sleep(0.02)
                        continue
                    inventory = (await browser.get("/api/workspaces")).json()
                    if inventory["nodes"]:
                        return inventory
                    await asyncio.sleep(0.02)

            inventory = await asyncio.wait_for(registered(), 10)
            node = inventory["nodes"][0]
            assert node["id"]
            assert node["name"].startswith(socket.gethostname() + "/127.0.0.1:")
            assert int(node["name"].rsplit(":", 1)[1]) > 0
            assert inventory["workspaces"] == []
            renamed = await browser.post(
                "/api/nodes/" + node["id"] + "/alias", json={"alias": "CLI device"}
            )
            assert renamed.status_code == 200
            aliases_file = Path(os.environ["CODEX_HOME"]) / "router-aliases.json"
            assert json.loads(aliases_file.read_text(encoding="utf-8")) == {
                probe.node_id: "",
                node["id"]: "CLI device",
            }
            assert (await browser.get(node["url"])).status_code == 200
            with pytest.raises(InvalidStatusCode):
                async with connect(tunnel_url(public, "control")):
                    pass
            async with httpx.AsyncClient(
                base_url=tunnel, trust_env=False
            ) as device_port:
                assert (await device_port.get("/login")).status_code == 404
                assert (
                    await device_port.get(
                        "/api/workspaces", headers={"Host": urlsplit(public).netloc}
                    )
                ).status_code == 404
            # Closing its owner pipe must detach the device and drain its backend.
            workspace.stdin.close()
            diagnostics, _unused = await asyncio.wait_for(workspace.communicate(), 10)
            assert workspace.returncode == 0, diagnostics.decode(
                "utf-8", errors="replace"
            )
            snapshot = (await browser.get("/api/workspaces")).json()
            assert snapshot["nodes"] == []
    finally:
        try:
            if workspace is not None and workspace.returncode is None:
                workspace.stdin.close()
                await asyncio.wait_for(workspace.communicate(), 10)
        finally:
            if router.returncode is None:
                router.terminate()
            diagnostics, _unused = await asyncio.wait_for(router.communicate(), 10)
            assert router.returncode in (0, -signal.SIGTERM), diagnostics.decode(
                "utf-8", errors="replace"
            )
            assert b"Application shutdown complete" in diagnostics


async def test_cli_can_exit_normally_while_router_is_offline(tmp_path):
    config = tmp_path / "workspaces.json"
    config.write_text('{"workspaces": []}\n', encoding="utf-8")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    target = "1" * 22 + "@127.0.0.1:{0}".format(listener.getsockname()[1])
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "workspace_server",
            "--listen",
            "127.0.0.1:0",
            "--workspace-config",
            str(config),
            "--listen-to",
            target,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        diagnostics, _unused = await asyncio.wait_for(process.communicate(), 10)
        assert process.returncode == 0, diagnostics.decode("utf-8", errors="replace")
        assert b"Application startup complete" in diagnostics
        assert b"Application shutdown complete" in diagnostics
    finally:
        listener.close()
        if process is not None and process.returncode is None:
            process.terminate()
            await asyncio.wait_for(process.communicate(), 10)


@pytest.mark.parametrize(
    "target,expected",
    [
        ("127.0.0.1:6000", "ws://127.0.0.1:6000/tunnel/control"),
        ("https://gate.example/router/", "wss://gate.example/router/tunnel/control"),
        ("ws://[::1]:6000", "ws://[::1]:6000/tunnel/control"),
    ],
)
def test_router_addresses(target, expected):
    assert tunnel_url(target, "control") == expected


@pytest.mark.parametrize(
    "target",
    [
        "ftp://example.com",
        "ws://",
        "ws://user:password@example.com",
        "ws://example.com?token=hidden",
        "ws://example.com/#fragment",
        "ws://example.com:70000",
    ],
)
def test_router_addresses_reject_credentials_and_invalid_endpoints(target):
    with pytest.raises(ValueError):
        tunnel_url(target, "control")


@pytest.mark.parametrize(
    "target,expected",
    [
        ("{key}@127.0.0.1", "ws://127.0.0.1:6000"),
        ("{key}@127.0.0.1:7000", "ws://127.0.0.1:7000"),
        ("{key}@[::1]", "ws://[::1]:6000"),
        ("ws://{key}@[::1]:7000", "ws://[::1]:7000"),
        ("ws://{key}@router.example", "ws://router.example"),
        ("wss://{key}@router.example/remote/", "wss://router.example/remote/"),
        ("https://{key}@router.example/remote", "https://router.example/remote"),
    ],
)
def test_router_target_separates_key_from_address(target, expected):
    key = "a" * 22
    address, parsed_key = parse_target(target.format(key=key))
    assert address == expected
    assert parsed_key == key
    assert key not in tunnel_url(address, "control")
    assert key not in tunnel_url(address, "data")


@pytest.mark.parametrize(
    "target",
    [
        "127.0.0.1:6000",
        "ws://127.0.0.1:6000",
        "invalid@127.0.0.1:6000",
        "{key}x@127.0.0.1:6000",
        "ws://{key}:password@127.0.0.1:6000",
        "ws://{key}@",
        "ftp://{key}@router.example",
        "ws://{key}@127.0.0.1:70000",
        "ws://{key}@127.0.0.1:{key}",
        "ws://{key}@127.0.0.1?key={key}",
        "ws://{key}@127.0.0.1/#fragment",
    ],
)
def test_invalid_router_target_errors_do_not_include_the_key(target):
    key = "a" * 22
    with pytest.raises(ValueError) as error:
        parse_target(target.format(key=key))
    assert "router-generated registration key" in str(error.value)
    assert key not in str(error.value)
