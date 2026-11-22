import asyncio
import json
import os
import stat
import subprocess
import sys
import zlib
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from test_workspace_server import make_session
from workspace_server import (
    WorkspaceDefinition,
    WorkspaceEntry,
    WorkspaceRegistry,
    WorkspaceSessionManager,
    create_app,
    create_multi_workspace_app,
)
from workspace_server.agent_access import AGENT_TOKEN_HEADER, AgentSocketAccess
from workspace_server.owned_process import ReloadableApp
from workspace_server.vscode_server import (
    BRIDGE_EXTENSION_ID, VsCodeRuntime, find_code_server, resolve_vscode_link,
)


_MOCK_CODE_SERVER_SCRIPT = """#!__PYTHON__
import asyncio
import base64
import hashlib
import json
import os
import struct
import sys
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

STATE_DIR = Path(os.environ["MOCK_VSCODE_STATE_DIR"])


def install_extension(argv):
    ext_dir = Path(argv[argv.index("--extensions-dir") + 1])
    vsix = Path(argv[argv.index("--install-extension") + 1])
    count_file = STATE_DIR / "install_count.txt"
    prev = int(count_file.read_text(encoding="utf-8")) if count_file.exists() else 0
    count_file.write_text(str(prev + 1), encoding="utf-8")
    target = ext_dir / "pycodex.vscode-bridge-0.1.0"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(vsix), "r") as zf:
        for name in zf.namelist():
            if name.startswith("extension/"):
                rel = name[len("extension/"):]
                if rel:
                    out = target / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(zf.read(name))
    (ext_dir / "extensions.json").write_text(
        json.dumps([{"identifier": {"id": "pycodex.vscode-bridge"}}]),
        encoding="utf-8",
    )


async def read_ws_frame(reader):
    head = await reader.readexactly(2)
    opcode = head[0] & 0x0F
    masked = bool(head[1] & 0x80)
    length = head[1] & 0x7F
    if length == 126:
        length = struct.unpack("!H", await reader.readexactly(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", await reader.readexactly(8))[0]
    mask = await reader.readexactly(4) if masked else b""
    payload = bytearray(await reader.readexactly(length))
    if masked:
        for idx in range(length):
            payload[idx] ^= mask[idx % 4]
    return opcode, bytes(payload)


def encode_ws_frame(opcode, payload):
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    return bytes(header) + payload


async def run_server(http_sock):
    bridge_sock = os.environ["PYCODEX_VSCODE_BRIDGE_SOCK"]
    bridge_token = os.environ["PYCODEX_VSCODE_BRIDGE_TOKEN"]

    async def bridge_loop(window_id, folder="", ignore_supersede=False):
        b_reader, b_writer = await asyncio.open_unix_connection(bridge_sock)
        hello = {"type": "hello", "token": bridge_token, "window_id": window_id}
        b_writer.write((json.dumps(hello) + "\\n").encode("utf-8"))
        await b_writer.drain()
        while True:
            line = await b_reader.readline()
            if not line:
                break
            msg = json.loads(line.decode("utf-8"))
            if msg.get("type") == "supersede":
                with (STATE_DIR / "superseded.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"window_id": window_id, "folder": folder}) + "\\n")
                if not ignore_supersede:
                    b_writer.close()
                    break
                continue
            if msg.get("type") == "open":
                spec = msg["spec"]
                with (STATE_DIR / "opened.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps({"window_id": window_id, "folder": folder, "spec": spec}) + "\\n"
                    )
                if "fail_bridge.py" in str(spec.get("title", "")):
                    ack = {"type": "ack", "id": msg["id"], "ok": False, "error": "bridge open rejected"}
                else:
                    ack = {"type": "ack", "id": msg["id"], "ok": True}
                b_writer.write((json.dumps(ack) + "\\n").encode("utf-8"))
                await b_writer.drain()

    bridge_tasks = [
        asyncio.create_task(bridge_loop(window_id))
        for window_id in json.loads(os.environ.get("MOCK_VSCODE_WINDOW_IDS", "[]"))
    ]

    async def handle_http(reader, writer):
        try:
            req_line = await reader.readline()
            if not req_line:
                return
            method, target, _proto = req_line.decode("latin-1").strip().split(" ", 2)
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\\r\\n", b"\\n", b""):
                    break
                k, _, v = line.decode("latin-1").partition(":")
                headers[k.strip().lower()] = v.strip()
            content_len = int(headers.get("content-length", "0"))
            body = await reader.readexactly(content_len) if content_len else b""

            parsed = urlsplit(target)
            if method == "GET" and parsed.path == "/":
                q = parse_qs(parsed.query, keep_blank_values=True)
                raw_payloads = q.get("payload", [])
                folder_val = (q.get("folder") or q.get("workspace") or [""])[0]
                keep_zombie = (q.get("zombie") or [""])[0] == "1"
                for raw_p in raw_payloads:
                    try:
                        entries = json.loads(raw_p)
                    except ValueError:
                        continue
                    if isinstance(entries, list):
                        for pair in entries:
                            if isinstance(pair, list) and len(pair) == 2 and pair[0] == "extensionEnvironment":
                                env_obj = json.loads(pair[1])
                                dyn_wid = env_obj.get("PYCODEX_VSCODE_WINDOW_ID")
                                if dyn_wid:
                                    bridge_tasks.append(
                                        asyncio.create_task(bridge_loop(dyn_wid, folder_val, keep_zombie))
                                    )

            if headers.get("upgrade", "").lower() == "websocket":
                if parsed.path == "/refuse-upgrade":
                    writer.write(b"HTTP/1.1 503 Service Unavailable\\r\\nContent-Length: 0\\r\\n\\r\\n")
                    await writer.drain()
                    return
                ws_key = headers["sec-websocket-key"]
                accept = base64.b64encode(
                    hashlib.sha1((ws_key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
                ).decode("ascii")
                proto_hdr = headers.get("sec-websocket-protocol", "")
                chosen = proto_hdr.split(",")[0].strip() if proto_hdr else ""
                resp = [
                    "HTTP/1.1 101 Switching Protocols\\r\\n",
                    "Upgrade: websocket\\r\\n",
                    "Connection: Upgrade\\r\\n",
                    "Sec-WebSocket-Accept: " + accept + "\\r\\n",
                ]
                if chosen:
                    resp.append("Sec-WebSocket-Protocol: " + chosen + "\\r\\n")
                resp.append("\\r\\n")
                writer.write("".join(resp).encode("latin-1"))
                await writer.drain()
                hello_info = json.dumps({
                    "target": target,
                    "x_forwarded_prefix": headers.get("x-forwarded-prefix", ""),
                    "x_forwarded_proto": headers.get("x-forwarded-proto", ""),
                    "cookie": headers.get("cookie", ""),
                    "origin": headers.get("origin", ""),
                    "subprotocol": chosen,
                }).encode("utf-8")
                writer.write(encode_ws_frame(1, hello_info))
                await writer.drain()
                while True:
                    try:
                        opcode, payload = await read_ws_frame(reader)
                    except asyncio.IncompleteReadError:
                        break
                    if opcode == 8:
                        writer.write(encode_ws_frame(8, b""))
                        await writer.drain()
                        break
                    if opcode == 1:
                        writer.write(encode_ws_frame(1, b"echo:" + payload))
                        await writer.drain()
                    elif opcode == 2:
                        writer.write(encode_ws_frame(2, b"bin:" + payload))
                        await writer.drain()
                return

            if parsed.path == "/chunked":
                writer.write(
                    b"HTTP/1.1 200 OK\\r\\n"
                    b"Content-Type: text/plain; charset=utf-8\\r\\n"
                    b"Transfer-Encoding: chunked\\r\\n"
                    b"Content-Length: 999\\r\\n"
                    b"\\r\\n"
                    b"6\\r\\nhello \\r\\n"
                    b"7\\r\\nvscode!\\r\\n"
                    b"0\\r\\n\\r\\n"
                )
                await writer.drain()
                return

            if parsed.path == "/bundle.js":
                etag = 'W/"bundle-v1"'
                if headers.get("if-none-match") == etag:
                    writer.write(
                        (
                            "HTTP/1.1 304 Not Modified\\r\\n"
                            "ETag: " + etag + "\\r\\n"
                            "Cache-Control: public, max-age=3600\\r\\n"
                            "\\r\\n"
                        ).encode("latin-1")
                    )
                    await writer.drain()
                    return
                js_body = ("export const workbenchReady = true;\\n" * 64).encode("utf-8")
                writer.write(
                    (
                        "HTTP/1.1 200 OK\\r\\n"
                        "Content-Type: text/javascript; charset=utf-8\\r\\n"
                        "Content-Length: {0}\\r\\n"
                        "ETag: {1}\\r\\n"
                        "\\r\\n"
                    ).format(len(js_body), etag).encode("latin-1")
                )
                if method != "HEAD":
                    writer.write(js_body)
                await writer.drain()
                return

            if parsed.path == "/events":
                sse_body = ("data: " + ("x" * 600) + "\\n\\n").encode("utf-8")
                writer.write(
                    (
                        "HTTP/1.1 200 OK\\r\\n"
                        "Content-Type: text/event-stream\\r\\n"
                        "Content-Length: {0}\\r\\n"
                        "\\r\\n"
                    ).format(len(sse_body)).encode("latin-1")
                )
                writer.write(sse_body)
                await writer.drain()
                return

            payload = json.dumps({
                "pid": os.getpid(),
                "method": method,
                "target": target,
                "headers": headers,
                "body": body.decode("utf-8", errors="replace"),
            }).encode("utf-8")
            head = (
                "HTTP/1.1 200 OK\\r\\n"
                "Content-Type: application/json\\r\\n"
                "Content-Length: {0}\\r\\n"
                "Service-Worker-Allowed: /\\r\\n"
                "Set-Cookie: vsc_sid=ok; Path=/\\r\\n"
                "\\r\\n"
            ).format(len(payload)).encode("latin-1")
            writer.write(head)
            if method != "HEAD":
                writer.write(payload)
            await writer.drain()
        finally:
            writer.close()

    server = await asyncio.start_unix_server(handle_http, path=http_sock)
    (STATE_DIR / "server_info.json").write_text(
        json.dumps({"pid": os.getpid(), "http_sock": http_sock}),
        encoding="utf-8",
    )
    try:
        await server.serve_forever()
    finally:
        for task in bridge_tasks:
            task.cancel()


if "--install-extension" in sys.argv:
    install_extension(sys.argv)
else:
    sock = sys.argv[sys.argv.index("--socket-path") + 1]
    asyncio.run(run_server(sock))
"""


@pytest.fixture
def mock_code_server(tmp_path, monkeypatch):
    state_dir = tmp_path / "mock-vscode-state"
    state_dir.mkdir()
    binary = tmp_path / "mock-code-server"
    binary.write_text(
        _MOCK_CODE_SERVER_SCRIPT.replace("__PYTHON__", sys.executable), encoding="utf-8"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PYCODEX_VSCODE_WEB_SERVER", str(binary))
    monkeypatch.setenv("MOCK_VSCODE_STATE_DIR", str(state_dir))
    return state_dir


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat_path = Path("/proc/{0}/stat".format(pid))
    if stat_path.is_file():
        fields = stat_path.read_text(encoding="utf-8", errors="replace").split()
        if len(fields) > 2 and fields[2] == "Z":
            return False
    return True


def _build_app(tmp_path: Path, multi: bool, password):
    def make_entry(name: str):
        work_dir = tmp_path / ("work-" + name)
        work_dir.mkdir(exist_ok=True)
        state = tmp_path / ("state-" + name)
        state.with_suffix(".html").write_text("<html>canvas</html>", encoding="utf-8")
        manager = WorkspaceSessionManager(lambda: make_session(cwd=work_dir), state)
        return WorkspaceEntry(WorkspaceDefinition(name, state, work_dir), manager)

    if multi:
        first = make_entry("first")
        other = make_entry("other")
        registry = WorkspaceRegistry(
            [first, other], config_path=tmp_path / "workspaces.json"
        )
        registry.persist_definitions()
        return (
            create_multi_workspace_app(registry, password),
            "/w/first",
            first.definition.work_dir,
            first.manager,
        )
    single = make_entry("single")
    return (
        create_app(
            single.manager,
            single.definition.state_path,
            password,
            single.definition.work_dir,
        ),
        "",
        single.definition.work_dir,
        single.manager,
    )


@pytest.mark.parametrize("multi", [False, True], ids=["single", "multi"])
@pytest.mark.parametrize("password", [None, ""], ids=["none", "empty"])
def test_vscode_disabled_without_workspace_password(
    tmp_path, mock_code_server, multi, password
):
    app, prefix, _work_dir, _manager = _build_app(tmp_path, multi, password)
    with TestClient(app) as browser:
        page = browser.get(prefix + "/")
        assert page.status_code == 200
        assert 'data-enable-vscode="false"' in page.text
        assert browser.get(prefix + "/vscode", follow_redirects=False).status_code == 403
        assert browser.get(prefix + "/vscode/").status_code == 403
        assert browser.get(prefix + "/vscode/window/win-1", follow_redirects=False).status_code == 403
        assert browser.get(prefix + "/vscode/window/win-1/", follow_redirects=False).status_code == 403
        assert (
            browser.post(prefix + "/api/vscode/open", json={"href": "a.py"}).status_code
            == 403
        )
        with pytest.raises(WebSocketDisconnect):
            with browser.websocket_connect(prefix + "/vscode/stable-mock"):
                pass
    assert not (mock_code_server / "server_info.json").exists()


@pytest.mark.parametrize("multi", [False, True], ids=["single", "multi"])
def test_vscode_permissions_admin_visitor_and_agent_access(
    tmp_path, mock_code_server, multi
):
    app, prefix, _work_dir, _manager = _build_app(tmp_path, multi, "admin-password")
    app.state.agent_auth_token = "vscode-agent-token"

    with TestClient(app) as browser:
        # Unauthenticated requests cannot reach VS Code HTTP or WebSocket endpoints.
        unauth_get = browser.get(prefix + "/vscode/", follow_redirects=False)
        assert unauth_get.status_code == 303
        assert unauth_get.headers["location"].startswith("/login?")
        unauth_win = browser.get(prefix + "/vscode/window/win-1/", follow_redirects=False)
        assert unauth_win.status_code == 303
        assert unauth_win.headers["location"].startswith("/login?")
        assert (
            browser.post(prefix + "/api/vscode/open", json={"href": "a.py"}).status_code
            == 401
        )
        with pytest.raises(WebSocketDisconnect):
            with browser.websocket_connect(prefix + "/vscode/stable-mock"):
                pass

        # Admin enables sharing, but has not opened VS Code yet (lazy startup).
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        admin_cookie = browser.cookies["pycodex_ws_auth"]
        admin_page = browser.get(prefix + "/")
        assert admin_page.status_code == 200
        assert 'data-access-role="admin"' in admin_page.text
        assert 'data-enable-vscode="true"' in admin_page.text
        assert not (mock_code_server / "server_info.json").exists()

        assert (
            browser.put(prefix + "/api/share", json={"password": "visitor-password"}).status_code
            == 200
        )
        browser.cookies.clear()

        # Visitor can use workspace chat/canvas, but VS Code is completely blocked.
        assert (
            browser.post(
                "/login",
                params={"next": prefix + "/"},
                json={"password": "visitor-password"},
            ).status_code
            == 200
        )
        visitor_page = browser.get(prefix + "/")
        assert visitor_page.status_code == 200
        assert 'data-access-role="visitor"' in visitor_page.text
        assert 'data-enable-vscode="false"' in visitor_page.text
        for path in (
            "/vscode",
            "/vscode/",
            "/vscode/window/win-1",
            "/vscode/window/win-1/",
            "/vscode/stable-mock",
        ):
            resp = browser.get(prefix + path, follow_redirects=False)
            assert resp.status_code == 403
        assert (
            browser.post(prefix + "/api/vscode/open", json={"href": "a.py"}).status_code
            == 403
        )
        with pytest.raises(WebSocketDisconnect):
            with browser.websocket_connect(prefix + "/vscode/stable-mock"):
                pass

        # Agent access requests (even carrying admin cookie) cannot enable or open VS Code.
        agent_client = TestClient(AgentSocketAccess(app), client=("127.0.0.1", 50000))
        agent_headers = {
            "cookie": "pycodex_ws_auth=" + admin_cookie,
            AGENT_TOKEN_HEADER: "vscode-agent-token",
        }
        agent_index = agent_client.get(prefix + "/", headers=agent_headers)
        assert agent_index.status_code == 200
        assert 'data-enable-vscode="false"' in agent_index.text
        assert (
            agent_client.get(prefix + "/vscode/", headers=agent_headers).status_code == 403
        )
        assert (
            agent_client.get(
                prefix + "/vscode/window/win-1/", headers=agent_headers, follow_redirects=False
            ).status_code
            == 403
        )
        assert (
            agent_client.post(
                prefix + "/api/vscode/open", headers=agent_headers, json={"href": "a.py"}
            ).status_code
            == 403
        )
        assert not (mock_code_server / "server_info.json").exists()


@pytest.mark.parametrize("multi", [False, True], ids=["single", "multi"])
def test_vscode_http_and_websocket_proxy(tmp_path, mock_code_server, multi):
    app, prefix, work_dir, _manager = _build_app(tmp_path, multi, "admin-password")
    with TestClient(app) as browser:
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200

        # Missing trailing slash redirects to /vscode/ preserving query string.
        redir = browser.get(
            prefix + "/vscode?folder=" + str(work_dir), follow_redirects=False
        )
        assert redir.status_code == 307
        assert redir.headers["location"] == prefix + "/vscode/?folder=" + str(work_dir)

        # Proxied HTTP strips pycodex_ws_auth, forwards prefix/proto and preserves raw encoding.
        resp = browser.post(
            prefix + "/vscode/vfs/dir%2Ffile%20name%231.py?mode=ro",
            headers={"cookie": "vsc_pref=dark; pycodex_ws_auth=" + browser.cookies["pycodex_ws_auth"]},
            content=b"hello-body",
        )
        assert resp.status_code == 200
        assert resp.headers["service-worker-allowed"] == prefix + "/vscode/"
        assert "Path=" + prefix + "/vscode/" in resp.headers["set-cookie"]
        data = resp.json()
        assert data["method"] == "POST"
        assert data["target"] == "/vfs/dir%2Ffile%20name%231.py?mode=ro"
        assert data["headers"]["x-forwarded-prefix"] == prefix + "/vscode"
        assert data["headers"]["x-forwarded-proto"] == "http"
        assert data["headers"].get("cookie") == "vsc_pref=dark"
        assert data["body"] == "hello-body"
        first_pid = data["pid"]

        # Chunked streaming response decodes cleanly and omits conflicting Content-Length.
        chunked = browser.get(prefix + "/vscode/chunked")
        assert chunked.status_code == 200
        assert chunked.headers.get("content-encoding") == "gzip"
        assert "content-length" not in chunked.headers
        assert chunked.text == "hello vscode!"

        # Large static JS responses negotiate streaming gzip, rewrite ETag and support 304.
        expected_js = "export const workbenchReady = true;\n" * 64
        with browser.stream(
            "GET", prefix + "/vscode/bundle.js", headers={"accept-encoding": "gzip"}
        ) as gzipped_js:
            assert gzipped_js.status_code == 200
            assert gzipped_js.headers.get("content-encoding") == "gzip"
            assert "Accept-Encoding" in gzipped_js.headers.get("vary", "")
            assert gzipped_js.headers.get("etag") == 'W/"bundle-v1-gzip"'
            assert "content-length" not in gzipped_js.headers
            raw_wire = b"".join(gzipped_js.iter_raw())
            assert 0 < len(raw_wire) < len(expected_js.encode("utf-8")) // 4
            assert zlib.decompress(raw_wire, 31).decode("utf-8") == expected_js
            etag_gzip = gzipped_js.headers["etag"]

        cached_304 = browser.get(
            prefix + "/vscode/bundle.js",
            headers={
                "accept-encoding": "gzip",
                "if-none-match": etag_gzip,
            },
        )
        assert cached_304.status_code == 304
        assert cached_304.headers.get("etag") == 'W/"bundle-v1-gzip"'
        assert cached_304.headers.get("content-encoding") == "gzip"
        assert "Accept-Encoding" in cached_304.headers.get("vary", "")
        assert cached_304.content == b""

        identity_js = browser.get(
            prefix + "/vscode/bundle.js", headers={"accept-encoding": "gzip;q=0, identity"}
        )
        assert identity_js.status_code == 200
        assert "content-encoding" not in identity_js.headers
        assert "Accept-Encoding" in identity_js.headers.get("vary", "")
        assert identity_js.headers.get("etag") == 'W/"bundle-v1"'
        assert identity_js.text == expected_js

        # Event streams are never gzip-buffered.
        sse = browser.get(prefix + "/vscode/events", headers={"accept-encoding": "gzip"})
        assert sse.status_code == 200
        assert "content-encoding" not in sse.headers
        assert sse.text.startswith("data: xxx")

        # HEAD returns headers without body.
        head_resp = browser.head(prefix + "/vscode/static/workbench.js")
        assert head_resp.status_code == 200
        assert head_resp.content == b""

        # In multi-workspace mode, second workspace reuses the single code-server instance.
        if multi:
            second = browser.get("/w/other/vscode/ping").json()
            assert second["pid"] == first_pid
            assert second["headers"]["x-forwarded-prefix"] == "/w/other/vscode"
            assert browser.get("/w/missing/vscode/").status_code == 404

        # WebSocket proxy supports subprotocols, text & binary frames, and HTTPS reverse-proxy Origin.
        with browser.websocket_connect(
            prefix + "/vscode/stable-mock?reconnectionToken=t1",
            subprotocols=["vscode.ipc"],
            headers={
                "origin": "https://proxy.example",
                "x-forwarded-proto": "https",
                "x-forwarded-host": "proxy.example:443",
            },
        ) as ws:
            hello = json.loads(ws.receive_text())
            assert hello["target"] == "/stable-mock?reconnectionToken=t1"
            assert hello["x_forwarded_prefix"] == prefix + "/vscode"
            assert hello["subprotocol"] == "vscode.ipc"
            assert "pycodex_ws_auth" not in hello["cookie"]
            ws.send_text("ping-text")
            assert ws.receive_text() == "echo:ping-text"
            ws.send_bytes(b"\x00\x01\x02")
            assert ws.receive_bytes() == b"bin:\x00\x01\x02"

        # Cross-origin WebSocket is rejected with 1008.
        with pytest.raises(WebSocketDisconnect):
            with browser.websocket_connect(
                prefix + "/vscode/stable-mock",
                headers={"origin": "https://evil.example"},
            ):
                pass

    # Lifespan close terminates code-server and cleans up temporary socket directory.
    info = json.loads((mock_code_server / "server_info.json").read_text(encoding="utf-8"))
    assert not _pid_alive(info["pid"])
    assert not Path(info["http_sock"]).parent.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["startup", "upstream-handshake"])
async def test_vscode_websocket_rejects_failed_handshake(tmp_path, mock_code_server, monkeypatch, failure):
    if failure == "startup":
        monkeypatch.setenv("PYCODEX_VSCODE_WEB_SERVER", str(tmp_path / "missing"))
    runtime = VsCodeRuntime()
    sent = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append((message["type"], message.get("code")))

    websocket = WebSocket({
        "type": "websocket", "scheme": "ws", "path": "/vscode/refuse-upgrade",
        "headers": [(b"host", b"localhost")], "query_string": b"",
    }, receive, send)
    try:
        await asyncio.wait_for(runtime.proxy_websocket(websocket, ""), timeout=5)
        assert sent == [("websocket.close", 1011)]
    finally:
        await runtime.close()


def test_vscode_mounted_under_root_path_forwards_combined_prefix(
    tmp_path, mock_code_server
):
    inner, prefix, _work_dir, _manager = _build_app(tmp_path, True, "admin-password")
    outer = Starlette(routes=[Mount("/n/node1", app=inner)], lifespan=inner.router.lifespan_context)
    with TestClient(outer) as browser:
        assert (
            browser.post("/n/node1/login", json={"password": "admin-password"}).status_code
            == 200
        )
        redir = browser.get("/n/node1" + prefix + "/vscode", follow_redirects=False)
        assert redir.status_code == 307
        assert redir.headers["location"] == "/n/node1" + prefix + "/vscode/"
        resp = browser.get("/n/node1" + prefix + "/vscode/check")
        assert resp.status_code == 200
        assert resp.json()["headers"]["x-forwarded-prefix"] == "/n/node1" + prefix + "/vscode"
        win_redir = browser.get(
            "/n/node1" + prefix + "/vscode/window/win-root/?folder=/tmp", follow_redirects=False
        )
        assert win_redir.status_code == 307
        assert win_redir.headers["location"].startswith("/n/node1" + prefix + "/vscode/window/win-root/?")
        assert browser.get(win_redir.headers["location"]).json()["headers"]["x-forwarded-prefix"] == "/n/node1" + prefix + "/vscode"


def test_vscode_open_link_resolves_files_git_diffs_and_preserves_existing_views(
    tmp_path, mock_code_server, monkeypatch
):
    app, prefix, work_dir, manager = _build_app(tmp_path, False, "admin-password")
    monkeypatch.setenv("MOCK_VSCODE_WINDOW_IDS", json.dumps(["window-a", "window-b"]))

    # Set up Git repository with tracked files, subdirectory and uncommitted edits.
    subprocess.run(["git", "init"], cwd=str(work_dir), check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(work_dir), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(work_dir), check=True)
    (work_dir / "app.py").write_text("line1\nline2\n", encoding="utf-8")
    (work_dir / "fail_bridge.py").write_text("x = 1\n", encoding="utf-8")
    (work_dir / "sub").mkdir()
    (work_dir / "sub" / "mod.py").write_text("sub1\nsub2\n", encoding="utf-8")
    (work_dir / "docs").mkdir()
    (work_dir / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(work_dir), check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(work_dir), check=True, stdout=subprocess.PIPE)
    (work_dir / "app.py").write_text("line1\nline2_modified\nline3\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("file outside the workspace\n", encoding="utf-8")

    with TestClient(app) as browser:
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        def open_link(href, window_id="window-a", **fields):
            return browser.post(
                prefix + "/api/vscode/open", json=dict(fields, href=href, window_id=window_id)
            )

        # Open one legacy view before VS Code mode; it must remain untouched afterward.
        pre_open = browser.post(prefix + "/api/views/open", json={"href": "docs/guide.md"})
        assert pre_open.status_code == 200
        views_before = manager.views.list_views()
        assert len(views_before) == 1

        # Create a session whose cwd is work_dir / "sub".
        sub_session_resp = browser.post(
            prefix + "/api/sessions", json={"title": "Sub", "prompt": "/cd sub"}
        )
        assert sub_session_resp.status_code == 200
        sub_session_id = sub_session_resp.json()["session_id"]
        assert sub_session_resp.json()["snapshot"]["cwd"] == str(work_dir / "sub")

        # 1. Local file with line selection.
        r1 = open_link("app.py#L3")
        assert r1.status_code == 200
        assert r1.json()["target"]["kind"] == "file"
        assert r1.json()["target"]["selection"] == {"document": 0, "line": 3}
        assert r1.json()["target"]["documents"][0] == {
            "scheme": "file",
            "path": str((work_dir / "app.py").resolve()),
            "label": "app.py",
        }

        # 2. Markdown relative link resolved from src_path.
        r2 = open_link("../app.py#L1", src_path=str(work_dir / "docs" / "guide.md"))
        assert r2.status_code == 200
        assert r2.json()["target"]["documents"][0]["path"] == str((work_dir / "app.py").resolve())

        # 3. Session-scoped cwd resolution.
        r3 = open_link("mod.py:2", session_id=sub_session_id)
        assert r3.status_code == 200
        assert r3.json()["target"]["documents"][0]["path"] == str(
            (work_dir / "sub" / "mod.py").resolve()
        )
        assert r3.json()["target"]["selection"] == {"document": 0, "line": 2}

        # 4. Git blob snapshot.
        r4 = open_link("git:HEAD:app.py#L2")
        assert r4.status_code == 200
        git_doc = r4.json()["target"]["documents"][0]
        assert git_doc["scheme"] == "pycodex-git"
        assert git_doc["path"] == "app.py"
        assert len(git_doc["blob"]) == 40

        # 5. Git working-tree diff and two-file diff.
        r5 = open_link("diff:HEAD:app.py#R3")
        assert r5.status_code == 200
        assert r5.json()["target"]["kind"] == "diff"
        assert r5.json()["target"]["selection"] == {"document": 1, "line": 3}
        assert r5.json()["target"]["documents"][0]["scheme"] == "pycodex-git"
        assert r5.json()["target"]["documents"][1]["scheme"] == "file"

        r6 = open_link("diff:app.py..sub/mod.py#L1")
        assert r6.status_code == 200
        assert r6.json()["target"]["kind"] == "diff"
        assert r6.json()["target"]["selection"] == {"document": 0, "line": 1}

        other = open_link(str(outside), window_id="window-b")
        assert other.status_code == 200
        assert other.json()["target"]["documents"][0]["path"] == str(outside)
        assert browser.post(prefix + "/api/vscode/open", json={"href": "app.py"}).status_code == 400
        assert open_link("missing.py").status_code == 400
        assert open_link("app.py#L99").status_code == 400
        assert open_link("app.py", session_id="").status_code == 400
        assert open_link("app.py", session_id="unknown-session").status_code == 404
        assert open_link("fail_bridge.py").status_code == 503

        # Built-in views remain completely unchanged!
        assert manager.views.list_views() == views_before
        opened_lines = (
            (mock_code_server / "opened.jsonl").read_text(encoding="utf-8").strip().splitlines()
        )
        recipients = [json.loads(line)["window_id"] for line in opened_lines]
        assert recipients == ["window-a"] * 6 + ["window-b", "window-a"]
        assert int((mock_code_server / "install_count.txt").read_text(encoding="utf-8")) == 1


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs extra Windows privileges")
def test_vscode_working_tree_diff_reads_symlink_target(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    target = "example file.txt"
    (tmp_path / target).write_text("first\nsecond\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    subprocess.run(["git", "add", "link"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "commit", "-qm", "track symbolic link"],
        cwd=tmp_path, check=True,
    )

    plain = resolve_vscode_link(tmp_path, "link#L2")
    assert plain["documents"][0]["scheme"] == "file"
    assert plain["selection"] == {"document": 0, "line": 2}
    with pytest.raises(ValueError, match="line 2 is outside"):
        resolve_vscode_link(tmp_path, "diff:HEAD:link#R2")

    (tmp_path / "directory").mkdir()
    for name in (target, "directory", "missing\nfilename"):
        link.unlink()
        link.symlink_to(name)
        line = name.count("\n") + 1
        diff = resolve_vscode_link(tmp_path, "diff:HEAD:link#R{0}".format(line))
        left, right = diff["documents"]
        committed_target = subprocess.check_output(
            ["git", "cat-file", "blob", left["blob"]], cwd=tmp_path
        )
        assert committed_target == target.encode()
        assert right["scheme"] == "pycodex-git"
        assert right["symlink_target"] == name
        assert diff["selection"] == {"document": 1, "line": line}


def test_vscode_line_selection_streams_large_files_and_git_blobs(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    path = tmp_path / "large.txt"
    with path.open("wb") as handle:
        # Split a CRLF across two 64 KiB reads. A trailing newline adds an empty line.
        handle.write(b"x" * 65535 + b"\r\n")
        block = b"x" * (1024 * 1024)
        for _ in range(64):
            handle.write(block)
        handle.write(b"\rend\n")
    subprocess.run(["git", "add", "large.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "commit", "-qm", "large text"],
        cwd=tmp_path, check=True,
    )
    for href in ("large.txt", "git:HEAD:large.txt"):
        assert resolve_vscode_link(tmp_path, href + "#L4")["selection"]["line"] == 4
        with pytest.raises(ValueError, match="line 5 is outside"):
            resolve_vscode_link(tmp_path, href + "#L5")

    with path.open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ValueError, match="requires an existing text file"):
        resolve_vscode_link(tmp_path, "large.txt#L1")


def test_vscode_restarts_on_reload_and_reuses_bridge_installation(
    tmp_path, mock_code_server, monkeypatch
):
    async def passed():
        return None

    monkeypatch.setattr("workspace_server.owned_process.preflight_check", passed)
    monkeypatch.setattr("workspace_server.owned_process.purge_reloadable_modules", dict)
    shell = ReloadableApp(lambda: _build_app(tmp_path, False, "admin-password")[0])
    prefix = ""
    work_dir = tmp_path / "work-single"
    with TestClient(shell) as browser:
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        before = browser.get(prefix + "/vscode/?folder=" + str(work_dir)).json()
        reload_resp = browser.post(prefix + "/api/reload", json={})
        assert reload_resp.status_code == 200
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        after = browser.get(prefix + "/vscode/after-reload").json()
        assert after["pid"] != before["pid"]

    # The restarted runtime reuses the installed .pycodex-ready bridge extension without reinstalling.
    assert int((mock_code_server / "install_count.txt").read_text(encoding="utf-8")) == 1
    ext_json = Path.home() / ".vscode-server" / "extensions" / "extensions.json"
    assert BRIDGE_EXTENSION_ID in ext_json.read_text(encoding="utf-8")


def test_find_code_server_discovers_serve_web_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("PYCODEX_VSCODE_WEB_SERVER", raising=False)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(FileNotFoundError) as exc_info:
        find_code_server()
    assert "Ask your agent to install VS Code Web Server (vscode-reh-web)" in str(
        exc_info.value
    )
    assert "~/.vscode-server/cli/serve-web/" in str(exc_info.value)

    serve_web = Path.home() / ".vscode-server" / "cli" / "serve-web"
    # Incomplete installation without workbench.html is ignored.
    incomplete = serve_web / "Stable-incomplete" / "bin" / "code-server"
    incomplete.parent.mkdir(parents=True)
    incomplete.write_text("#!/bin/sh\n", encoding="utf-8")
    incomplete.chmod(stat.S_IRWXU)
    with pytest.raises(FileNotFoundError):
        find_code_server()

    # Complete serve-web installation is discovered and newest mtime wins.
    for idx, name in enumerate(("Stable-old", "Stable-new")):
        root = serve_web / name
        binary = root / "bin" / "code-server"
        workbench = root / "out" / "vs" / "code" / "browser" / "workbench" / "workbench.html"
        binary.parent.mkdir(parents=True, exist_ok=True)
        workbench.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/bin/sh\n", encoding="utf-8")
        binary.chmod(stat.S_IRWXU)
        workbench.write_text("<html></html>", encoding="utf-8")
        mtime = 1700000000 + idx * 100
        os.utime(str(binary), (mtime, mtime))

    assert find_code_server() == (serve_web / "Stable-new" / "bin" / "code-server").resolve()


@pytest.mark.asyncio
async def test_vscode_open_link_times_out_when_no_window_matches(tmp_path, mock_code_server, monkeypatch):
    monkeypatch.setenv("MOCK_VSCODE_WINDOW_IDS", json.dumps(["another-window"]))
    runtime = VsCodeRuntime(startup_timeout=5.0, bridge_timeout=0.15)
    try:
        target_file = tmp_path / "note.txt"
        target_file.write_text("hi\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="VS Code window is not ready"):
            await runtime.open_link(
                "absent-window",
                {
                    "kind": "file",
                    "title": "note.txt",
                    "documents": [{"scheme": "file", "path": str(target_file), "label": "note.txt"}],
                    "selection": None,
                },
            )
    finally:
        await runtime.close()


@pytest.mark.parametrize("multi", [False, True], ids=["single", "multi"])
def test_vscode_window_route_preserves_link_binding_across_folder_switches_and_reloads(
    tmp_path, mock_code_server, multi
):
    app, prefix, work_dir, _manager = _build_app(tmp_path, multi, "admin-password")
    other_folder = tmp_path / "other-project"
    other_folder.mkdir()
    (work_dir / "first.py").write_text("one\ntwo\n", encoding="utf-8")
    (other_folder / "second.py").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    with TestClient(app) as browser:
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200

        # Missing trailing slash redirects to /vscode/window/<id>/ keeping query parameters.
        slash_redir = browser.get(
            prefix + "/vscode/window/win-nav?folder=" + str(work_dir) + "&zombie=1",
            follow_redirects=False,
        )
        assert slash_redir.status_code == 307
        assert (
            slash_redir.headers["location"]
            == prefix + "/vscode/window/win-nav/?folder=" + str(work_dir) + "&zombie=1"
        )

        # Initial load carrying PYCODEX_VSCODE_WINDOW_ID (as createVscodeWindow builds) proxies directly with 0 redirects.
        init_payload = json.dumps(
            [["extensionEnvironment", json.dumps({"PYCODEX_VSCODE_WINDOW_ID": "win-nav"})]]
        )
        load_1 = browser.get(
            prefix + "/vscode/window/win-nav/",
            params={"folder": str(work_dir), "zombie": "1", "payload": init_payload},
            follow_redirects=False,
        )
        assert load_1.status_code == 200
        upstream_1 = load_1.json()
        assert upstream_1["target"].startswith("/?folder=")
        assert upstream_1["headers"]["x-forwarded-prefix"] == prefix + "/vscode"

        open_1 = browser.post(
            prefix + "/api/vscode/open",
            json={"href": "first.py#L2", "window_id": "win-nav"},
        )
        assert open_1.status_code == 200

        # Simulate VS Code File -> Open Folder... navigating to ?folder=<other_folder> (dropping
        # extensionEnvironment while keeping non-env payload entries like openFile). Also keep the
        # first extension host socket alive (zombie=1) to simulate VS Code's 5-minute grace period.
        extra_payload = json.dumps([["openFile", "vscode-remote://localhost/init.py"]])
        switch_redir = browser.get(
            prefix + "/vscode/window/win-nav/",
            params={"folder": str(other_folder), "payload": extra_payload},
            follow_redirects=False,
        )
        assert switch_redir.status_code == 307
        loc_2 = switch_redir.headers["location"]
        assert "openFile" in loc_2
        assert "PYCODEX_VSCODE_WINDOW_ID" in loc_2

        load_2 = browser.get(loc_2, follow_redirects=False)
        assert load_2.status_code == 200
        assert load_2.json()["headers"]["x-forwarded-prefix"] == prefix + "/vscode"

        # Clicking a chat link now opens in the NEW folder's bridge client despite the zombie socket!
        open_2 = browser.post(
            prefix + "/api/vscode/open",
            json={"href": str(other_folder / "second.py") + "#L3", "window_id": "win-nav"},
        )
        assert open_2.status_code == 200

        # Simulate Developer: Reload Window (re-requesting loc_2 with matching PYCODEX_VSCODE_WINDOW_ID):
        # proxies directly with 0 redirects while superseding the previous bridge connection!
        reload_resp = browser.get(loc_2 + "&reload=1", follow_redirects=False)
        assert reload_resp.status_code == 200

        # Simulate File -> Open Folder in New Window (_blank -> Sec-Fetch-Dest: document).
        # Must strip PYCODEX_VSCODE_WINDOW_ID and redirect to /vscode/?... without stealing win-nav!
        popup_redir = browser.get(
            loc_2,
            headers={"sec-fetch-dest": "document"},
            follow_redirects=False,
        )
        assert popup_redir.status_code == 307
        popup_loc = popup_redir.headers["location"]
        assert popup_loc.startswith(prefix + "/vscode/?")
        assert "PYCODEX_VSCODE_WINDOW_ID" not in popup_loc
        assert "openFile" in popup_loc

        open_3 = browser.post(
            prefix + "/api/vscode/open",
            json={"href": "first.py#L1", "window_id": "win-nav"},
        )
        assert open_3.status_code == 200

        opened = [
            json.loads(line)
            for line in (mock_code_server / "opened.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert [item["folder"] for item in opened] == [
            str(work_dir),
            str(other_folder),
            str(other_folder),
        ]
        assert [item["window_id"] for item in opened] == ["win-nav", "win-nav", "win-nav"]
        superseded = [
            json.loads(line)
            for line in (mock_code_server / "superseded.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(superseded) >= 2
