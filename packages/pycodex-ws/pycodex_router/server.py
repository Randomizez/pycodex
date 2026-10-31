import argparse
import asyncio
import json
import os
import re
import secrets
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import HTTPConnection
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.websockets import WebSocketDisconnect, WebSocketState
from pycodex.portable.protocol import base58_encode
from pycodex.utils.session_persist import resolve_codex_home
from pycodex_ws.server_support import (
    auth_cookie_matches,
    install_auth,
    json_object_body,
    parse_listen,
)
from pycodex_ws.workspace_config import normalize_workspace_id

from .browser import BrowserAccess, require_login
from .protocol import (
    decode,
    device_id,
    encode,
    normalize_api_url,
    request_scope,
    response_headers,
    send_event,
    trusted_agent_request,
)


class TunnelUnavailable(Exception):
    pass


def load_registration_key(path):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError:
        value = path.read_bytes().strip()
    else:
        value = base58_encode(os.urandom(16)).encode("ascii")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value + b"\n")
    if not re.fullmatch(b"[1-9A-HJ-NP-Za-km-z]{1,22}", value):
        raise ValueError("invalid router registration key file")
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise ValueError("router registration key file must have owner-only permissions")
    return value.decode("ascii")


def normalize_alias(value):
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError("Device alias must be a single line of 1 to 128 characters.")
    value.encode("utf-8")
    return value.strip()


class Node:
    def __init__(self, websocket, node_id, name, workspaces):
        self.websocket = websocket
        self.id = node_id
        self.name = name
        self.workspaces = workspaces
        self.streams = {}
        self.send_lock = asyncio.Lock()


class Stream:
    def __init__(self, node_id):
        self.node_id = node_id
        self.ready = asyncio.get_running_loop().create_future()
        self.done = asyncio.Event()
        self.websocket = None


async def close_websocket(websocket, code=1000):
    if websocket.application_state == WebSocketState.CONNECTED:
        try:
            await websocket.close(code=code)
        except (WebSocketDisconnect, OSError):
            pass


async def control_message(websocket):
    payload = await websocket.receive_text()
    if not isinstance(payload, str) or len(payload.encode("utf-8")) > 1024 * 1024:
        raise ValueError("router control messages must not exceed 1 MiB")
    return decode(payload)


def validate_inventory(workspaces):
    if not isinstance(workspaces, list) or len(workspaces) > 256:
        raise ValueError("workspace inventory must be a list of at most 256 entries")
    fields = {
        "id",
        "work_dir",
        "state_path",
        "enable_board",
        "session_count",
        "running",
        "unread",
    }
    seen = set()
    for workspace in workspaces:
        if not isinstance(workspace, dict) or set(workspace) != fields:
            raise ValueError("invalid workspace inventory fields")
        workspace_id = workspace["id"]
        if (
            not isinstance(workspace_id, str)
            or not workspace_id
            or len(workspace_id) > 128
            or normalize_workspace_id(workspace_id) != workspace_id
            or workspace_id in seen
        ):
            raise ValueError("invalid or duplicate workspace id")
        for field in ("work_dir", "state_path"):
            value = workspace[field]
            if not isinstance(value, str) or not value or len(value) > 8192:
                raise ValueError("invalid workspace path")
            value.encode("utf-8")
        if (
            type(workspace["enable_board"]) is not bool
            or type(workspace["running"]) is not bool
            or type(workspace["unread"]) is not bool
            or type(workspace["session_count"]) is not int
            or not 0 <= workspace["session_count"] <= 1000000
        ):
            raise ValueError("invalid workspace inventory values")
        seen.add(workspace_id)
    return workspaces


class Router:
    def __init__(
        self,
        connect_timeout=10,
        max_connections=64,
        registration_key=None,
        aliases_path=None,
        api_url=None,
        api_port=None,
    ):
        self.api_url = normalize_api_url(api_url) if api_url is not None else None
        self.api_port = api_port
        self.registration_key = (
            base58_encode(os.urandom(16))
            if registration_key is None
            else registration_key
        )
        if not isinstance(self.registration_key, str) or not re.fullmatch(
            "[1-9A-HJ-NP-Za-km-z]{1,22}", self.registration_key
        ):
            raise ValueError("invalid router registration key")
        self.connect_timeout = connect_timeout
        self.max_connections = max_connections
        self.nodes = {}
        self.streams = {}
        self.browser_token = ""
        self.blocked_ips = set()
        self.aliases_path = Path(aliases_path) if aliases_path is not None else None
        self.aliases = {}
        if self.aliases_path is not None and self.aliases_path.exists():
            aliases = json.loads(self.aliases_path.read_text(encoding="utf-8"))
            if not isinstance(aliases, dict) or any(
                not re.fullmatch("[a-z2-7]{52}", node_id)
                or not isinstance(alias, str)
                or (alias and normalize_alias(alias) != alias)
                for node_id, alias in aliases.items()
            ):
                raise ValueError("invalid router aliases file")
            self.aliases = aliases

    def _save_aliases(self, aliases):
        if self.aliases_path is not None:
            path = self.aliases_path
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            data = json.dumps(aliases, ensure_ascii=False, indent=2).encode("utf-8")
            with tempfile.NamedTemporaryFile(
                dir=str(path.parent), prefix=path.name + ".", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                try:
                    handle.write(data + b"\n")
                    handle.close()
                    os.replace(str(temporary_path), str(path))
                finally:
                    if temporary_path.exists():
                        temporary_path.unlink()
        self.aliases = aliases

    async def reorder_devices(self, request):
        if not self.browser_token or not auth_cookie_matches(
            self.browser_token, request.cookies.get("pycodex_router_auth")
        ):
            raise HTTPException(401, "Authentication required.")
        payload = await json_object_body(request, max_bytes=64 * 1024)
        order = payload.get("order")
        if (
            not isinstance(order, list)
            or any(
                not isinstance(node_id, str)
                or not re.fullmatch("[a-z2-7]{52}", node_id)
                for node_id in order
            )
            or len(set(order)) != len(order)
        ):
            raise HTTPException(422, "Device order must be a list of unique device IDs.")
        if set(order) != set(self.nodes):
            raise HTTPException(409, "Connected devices changed. Refresh and try again.")
        online_order = iter(order)
        saved_order = [
            next(online_order) if node_id in self.nodes else node_id
            for node_id in self.aliases
        ]
        self._save_aliases({node_id: self.aliases[node_id] for node_id in saved_order})
        return JSONResponse({"order": order})

    async def rename_device(self, request):
        if not self.browser_token or not auth_cookie_matches(
            self.browser_token, request.cookies.get("pycodex_router_auth")
        ):
            raise HTTPException(401, "Authentication required.")
        node_id = request.path_params["node_id"]
        if node_id not in self.nodes:
            raise HTTPException(404, "Device is not connected.")
        payload = await json_object_body(request)
        try:
            alias = normalize_alias(payload.get("alias"))
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        aliases = dict(self.aliases)
        aliases[node_id] = alias
        self._save_aliases(aliases)
        return JSONResponse({"alias": alias})

    async def accept_tunnel(self, websocket):
        authorization = websocket.headers.get("authorization", "").encode("utf-8")
        expected = ("Bearer " + self.registration_key).encode("ascii")
        if websocket.headers.get("origin") or not secrets.compare_digest(
            authorization, expected
        ):
            await websocket.close(code=1008)
            return False
        await websocket.accept()
        return True

    def authenticated_device(self, connection):
        if "origin" in connection.headers or "sec-fetch-site" in connection.headers:
            return None
        authorization = connection.headers.get("authorization", "")
        if not authorization.startswith("Bearer "):
            return None
        try:
            node_id = device_id(authorization[7:])
        except ValueError:
            return None
        return self.nodes.get(node_id)

    async def control(self, websocket):
        if not await self.accept_tunnel(websocket):
            return
        node = None
        try:
            message = await asyncio.wait_for(control_message(websocket), 10)
            node_id = device_id(message["identity_key"])
            if (
                message.get("type") != "register"
                or message.get("version") != 2
                or "id" in message
                or not isinstance(message["name"], str)
                or not message["name"]
                or len(message["name"]) > 256
                or any(ord(char) < 32 for char in message["name"])
                or type(message["port"]) is not int
                or not 0 <= message["port"] <= 65535
            ):
                raise ValueError("invalid router registration")
            message["name"].encode("utf-8")
            name = "{0}/{1}:{2}".format(
                message["name"], websocket.client.host, message["port"]
            )
            if node_id in self.nodes:
                await websocket.send_text(encode({"type": "rejected"}))
                await close_websocket(websocket, 1008)
                return
            node = Node(
                websocket,
                node_id,
                name,
                validate_inventory(message["workspaces"]),
            )
            if node_id not in self.aliases:
                aliases = dict(self.aliases)
                aliases[node_id] = ""
                self._save_aliases(aliases)
            self.nodes[node_id] = node
            api_url = self.api_url
            if api_url is None and self.api_port is not None:
                host = websocket.url.hostname
                if ":" in host:
                    host = "[" + host + "]"
                api_url = normalize_api_url(
                    "http://{0}:{1}".format(host, self.api_port)
                )
            await websocket.send_text(encode({
                "type": "registered", "id": node_id, "api_url": api_url,
            }))
            while True:
                message = await control_message(websocket)
                if message.get("type") != "workspaces":
                    raise ValueError("unknown router control message")
                node.workspaces = validate_inventory(message["workspaces"])
        except WebSocketDisconnect:
            pass
        except (ValueError, KeyError, TypeError, asyncio.TimeoutError):
            await close_websocket(websocket, 1008)
        finally:
            if node is not None:
                self.nodes.pop(node.id, None)
                for stream in tuple(node.streams.values()):
                    if not stream.ready.done():
                        stream.ready.set_result(None)
                    if stream.websocket is not None:
                        await close_websocket(stream.websocket, 1012)
                    stream.done.set()

    async def data(self, websocket):
        if not await self.accept_tunnel(websocket):
            return
        stream = None
        try:
            message = await asyncio.wait_for(control_message(websocket), 10)
            stream = self.streams.get(message["stream_id"])
            if (
                stream is None
                or stream.ready.done()
                or device_id(message["identity_key"]) != stream.node_id
            ):
                await close_websocket(websocket, 1008)
                return
            stream.websocket = websocket
            stream.ready.set_result(websocket)
            await stream.done.wait()
        except WebSocketDisconnect:
            pass
        except (ValueError, KeyError, TypeError, asyncio.TimeoutError):
            await close_websocket(websocket, 1008)
        finally:
            await close_websocket(websocket)

    @asynccontextmanager
    async def connection(self, node_id):
        node = self.nodes.get(node_id)
        if node is None:
            raise TunnelUnavailable("Workspace device is offline.")
        if len(node.streams) >= self.max_connections:
            raise TunnelUnavailable("Workspace device has too many open requests.")
        stream_id = uuid4().hex
        stream = Stream(node_id)
        node.streams[stream_id] = stream
        self.streams[stream_id] = stream
        try:
            async with node.send_lock:
                await node.websocket.send_text(
                    encode(
                        {
                            "type": "connect",
                            "stream_id": stream_id,
                        }
                    )
                )
            websocket = await asyncio.wait_for(stream.ready, self.connect_timeout)
            if websocket is None:
                raise TunnelUnavailable("Workspace device disconnected.")
            yield websocket
        except (WebSocketDisconnect, OSError, asyncio.TimeoutError) as exc:
            raise TunnelUnavailable(
                "Connection lost. Check the session before retrying a message."
            ) from exc
        finally:
            self.streams.pop(stream_id, None)
            node.streams.pop(stream_id, None)
            stream.done.set()

    async def proxy(self, scope, receive, send):
        connection = HTTPConnection(scope)
        blocked = connection.client.host in self.blocked_ips
        agent_access = trusted_agent_request(scope)
        if blocked or not (
            agent_access
            or auth_cookie_matches(
                self.browser_token, connection.cookies.get("pycodex_router_auth")
            )
        ):
            await require_login(scope, receive, send)
            return
        started = False

        async def requests(websocket):
            while True:
                message = await receive()
                await send_event(websocket.send_text, message)
                if message["type"] in {"http.disconnect", "websocket.disconnect"}:
                    return

        async def responses(websocket):
            nonlocal started
            allowed = (
                {"http.response.start", "http.response.body"}
                if scope["type"] == "http"
                else {"websocket.accept", "websocket.send", "websocket.close"}
            )
            while True:
                message = decode(await websocket.receive_text())
                if message.get("type") not in allowed:
                    raise ValueError("invalid router response event")
                if "headers" in message:
                    message["headers"] = response_headers(
                        message["headers"],
                        scope.get("root_path", ""),
                        scope["scheme"] in {"https", "wss"},
                    )
                await send(message)
                started = True
                if (
                    message["type"] == "websocket.close"
                    or message["type"] == "http.response.body"
                    and not message.get("more_body", False)
                ):
                    return

        try:
            async with self.connection(scope["path_params"]["node_id"]) as websocket:
                await websocket.send_text(
                    encode(
                        {
                            "type": "open",
                            "scope": request_scope(scope),
                            "agent_access": agent_access,
                        }
                    )
                )
                tasks = [
                    asyncio.create_task(requests(websocket)),
                    asyncio.create_task(responses(websocket)),
                ]
                try:
                    done, _pending = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        task.result()
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except TunnelUnavailable as exc:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1013})
            elif not started:
                await JSONResponse({"error": str(exc)}, status_code=503)(
                    scope, receive, send
                )
            else:
                # Closing an incomplete HTTP response must stay visible to the caller.
                raise

    async def workspaces(self, request):
        nodes = []
        workspaces = []
        for node_id in self.aliases:
            if node_id not in self.nodes:
                continue
            node = self.nodes[node_id]
            base = request.scope.get("root_path", "") + "/n/" + node.id + "/"
            nodes.append({
                "id": node.id,
                "name": node.name,
                "alias": self.aliases.get(node.id, ""),
                "url": base,
            })
            for workspace in node.workspaces:
                item = dict(workspace)
                item.update(
                    {
                        "node_id": node.id,
                        "node_name": node.name,
                        "url": base + "w/" + workspace["id"] + "/",
                    }
                )
                workspaces.append(item)
        return JSONResponse({"nodes": nodes, "workspaces": workspaces})

    async def close(self):
        for node in tuple(self.nodes.values()):
            await close_websocket(node.websocket, 1001)


def create_app(
    password=None,
    connect_timeout=10,
    max_connections=64,
    registration_key=None,
    aliases_path=None,
    api_url=None,
    api_port=None,
):
    router = Router(
        connect_timeout, max_connections, registration_key, aliases_path, api_url, api_port
    )

    async def index(request):
        return HTMLResponse(
            Path(__file__).with_name("index.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    async def registration_key_info(request):
        if not router.browser_token or not auth_cookie_matches(
            router.browser_token, request.cookies.get("pycodex_router_auth")
        ):
            return JSONResponse({"error": "authentication required"}, status_code=401)
        return JSONResponse(
            {"key": router.registration_key},
            headers={"Cache-Control": "no-store"},
        )

    web = Starlette(
        routes=[
            Route("/", index),
            Route("/api/workspaces", router.workspaces),
            Route("/api/registration-key", registration_key_info),
            Route("/api/nodes/order", router.reorder_devices, methods=["POST"]),
            Route("/api/nodes/{node_id}/alias", router.rename_device, methods=["POST"]),
        ]
    )
    router.browser_token = install_auth(
        web,
        password,
        cookie_name="pycodex_router_auth",
        login_title="pycodex router",
        blocked_ips=router.blocked_ips,
    )

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await router.close()

    browser = Starlette(
        routes=[
            Mount("/n/{node_id}", app=router.proxy),
            Mount("/", app=web),
        ]
    )
    app = Starlette(
        routes=[Mount("/", app=BrowserAccess(router, browser))],
        lifespan=lifespan,
    )
    app.state.router = router
    return app


def create_tunnel_app(router):
    return Starlette(
        routes=[
            WebSocketRoute("/tunnel/control", router.control),
            WebSocketRoute("/tunnel/data", router.data),
        ]
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pycodex-router",
        description="Publish reverse-connected pycodex workspaces.",
        epilog=(
            "Sign in to the router page to copy its device registration key. "
            "Clients connect with --listen-to KEY@host:port; registered devices "
            "and their pages are trusted and share the router's browser origin."
        ),
    )
    parser.add_argument("--listen", default="0.0.0.0:6007", help="Browser host:port.")
    parser.add_argument(
        "--tunnel-listen", default="0.0.0.0:6000", help="Device host:port."
    )
    parser.add_argument(
        "--api-url",
        help=(
            "Public browser/API base URL advertised to devices, including any proxy "
            "path prefix. Defaults to the device connection hostname and browser "
            "HTTP port; set this for HTTPS, reverse proxies or port forwarding."
        ),
    )
    parser.add_argument(
        "--password",
        required=True,
        help="Password required for browser access to the router.",
    )
    return parser


def main(argv=None):
    import uvicorn

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.password:
        parser.error("--password must not be empty")
    if args.api_url is not None:
        try:
            args.api_url = normalize_api_url(args.api_url)
        except ValueError as exc:
            parser.error(str(exc))
    addresses = [parse_listen(args.listen), parse_listen(args.tunnel_listen)]
    if addresses[0][1] != 0 and addresses[0][1] == addresses[1][1]:
        parser.error("browser and device listeners must use different ports")
    codex_home = resolve_codex_home()
    key_path = codex_home / "router.key"
    app = create_app(
        args.password,
        registration_key=load_registration_key(key_path),
        aliases_path=codex_home / "router-aliases.json",
        api_url=args.api_url,
    )
    tunnel_app = create_tunnel_app(app.state.router)
    sockets = []

    class RouterServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if not self.started:
                return
            host = addresses[1][0]
            wildcard = host in {"0.0.0.0", "::"}
            if wildcard:
                host = "ROUTER_HOST"
            elif ":" in host:
                host = "[" + host + "]"
            target = "{0}@{1}:{2}".format(
                app.state.router.registration_key,
                host,
                sockets[1].getsockname()[1],
            )
            print(
                "Device registration key stored in {0}\n"
                "Connect a device:\n"
                "  pycodex-ws --listen-to {1}".format(key_path, target),
                flush=True,
            )
            if wildcard:
                print(
                    "Replace ROUTER_HOST with this router's reachable IP or hostname.",
                    flush=True,
                )

    async def dispatch(scope, receive, send):
        # The accepted socket's local port cannot be forged with HTTP headers.
        target = (
            app
            if (
                scope["type"] == "lifespan"
                or scope["server"][1] == sockets[0].getsockname()[1]
            )
            else tunnel_app
        )
        await target(scope, receive, send)

    try:
        config = None
        for host, port in addresses:
            listener = uvicorn.Config(
                dispatch,
                host=host,
                port=port,
                loop="asyncio",
                ws="websockets",
                ws_max_size=32 * 1024 * 1024,
            )
            sockets.append(listener.bind_socket())
            if config is None:
                config = listener
        if sockets[0].getsockname()[1] == sockets[1].getsockname()[1]:
            raise ValueError("browser and device listeners must use different ports")
        app.state.router.api_port = sockets[0].getsockname()[1]
        RouterServer(config).run(sockets=sockets)
    finally:
        for listener_socket in sockets:
            listener_socket.close()
    return 0
