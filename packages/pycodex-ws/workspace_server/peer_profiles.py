"""Peer-to-peer HTTP proxying and profile sharing over pycodex-router.

Nodes registered on the same `pycodex-router` mutually trust each other:
- `/api/peer-proxy` unconditionally proxies outbound HTTP/HTTPS requests for
  authenticated router peers, and injects a local Codex profile's upstream
  `base_url`, headers, and `.env` API key when `X-Peer-Profile` is set.
- `trusted_peer_request` keeps `/api/peer-profiles*` and `/api/peer-proxy`
  inaccessible (`403`) to direct HTTP, browser-proxied, or Agent-token callers.
"""

import asyncio
import copy
from dataclasses import dataclass, replace
from functools import partial
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import socket
import tempfile
import threading
import time
import typing
import urllib.parse

import requests
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
import uvicorn

from pycodex.bootstrap import _resolve_vllm_model
from pycodex.config import CodexConfig, _parse_harness
from pycodex.context import ContextConfig
from pycodex.model import ResponsesProviderConfig, _requests_verify_setting
from pycodex.tools import get_tools
from pycodex.utils.codex_config import list_codex_profiles, read_codex_config
from pycodex.utils.dotenv import DOTENV_FILENAME, ILLEGAL_ENV_VAR_PREFIX, parse_dotenv
from pycodex.utils.http_headers import build_http_headers
from pycodex.utils.model_metadata import load_models_by_slug
from pycodex_router.client import (
    peer_forward,
    peer_http_request,
    router_client_loop,
)
from pycodex_router.protocol import trusted_peer_request
from responses_server.config import CompatServerConfig

from .sessions import (
    ThreadedWorkspaceInteractiveSession,
    WorkspaceInteractiveSession,
    build_session_from_codex_config,
)

PEER_BRIDGE_API_KEY_ENV = "PYCODEX_PEER_BRIDGE_KEY"
PEER_BASE_SENTINEL = "@base"
PEER_NODE_ID_RE = re.compile(r"^[a-z2-7]{52}$")
PEER_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
PEER_PROFILE_ID_RE = re.compile(
    r"^peer:([a-z2-7]{52})(?:(@base)|:([A-Za-z0-9_-]+))?$"
)
PEER_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

_VALID_COMPAT_APIS = frozenset(("none", "chat_completions", "messages", "gemini"))
_MAX_PROXY_REDIRECTS = 5
_HOP_BY_HOP_HEADERS = frozenset(
    (
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    )
)
_STRIPPED_RESPONSE_HEADERS = _HOP_BY_HOP_HEADERS | frozenset(
    ("content-encoding", "content-length")
)
_REQUEST_PROTECTED_HEADERS = frozenset(
    (
        "accept",
        "x-client-request-id",
        "session_id",
        "x-codex-beta-features",
        "x-openai-internal-codex-responses-lite",
        "x-openai-subagent",
        "x-codex-turn-metadata",
    )
)
_SECRET_PROVIDER_KEYS = frozenset(
    ("base_url", "env_key", "query_params", "http_headers", "env_http_headers")
)


def is_peer_profile(profile: "typing.Optional[object]") -> bool:
    return isinstance(profile, str) and profile.startswith("peer:")


def parse_peer_profile(
    profile: "typing.Optional[str]",
) -> "typing.Tuple[str, typing.Optional[str]]":
    if not isinstance(profile, str):
        raise ValueError("invalid peer profile: {0!r}".format(profile))
    match = PEER_PROFILE_ID_RE.fullmatch(profile)
    if match is None:
        raise ValueError("invalid peer profile: {0!r}".format(profile))
    if match.group(2) is not None:
        return match.group(1), PEER_BASE_SENTINEL
    return match.group(1), match.group(3)


def format_peer_profile(
    node_id: str,
    profile_name: "typing.Optional[str]" = None,
) -> str:
    if not isinstance(node_id, str) or not PEER_NODE_ID_RE.fullmatch(node_id):
        raise ValueError("invalid peer node ID: {0!r}".format(node_id))
    if profile_name is None:
        return "peer:" + node_id
    if profile_name == PEER_BASE_SENTINEL:
        return "peer:" + node_id + PEER_BASE_SENTINEL
    if not isinstance(profile_name, str) or not PEER_PROFILE_NAME_RE.fullmatch(
        profile_name
    ):
        raise ValueError("invalid peer profile name: {0!r}".format(profile_name))
    return "peer:{0}:{1}".format(node_id, profile_name)


def node_identity_key_path(
    codex_home: "typing.Union[str, Path]",
    workspace_config_path: "typing.Union[str, Path]",
) -> Path:
    canonical_cfg = str(Path(workspace_config_path).expanduser().resolve())
    scope_digest = hashlib.sha256(canonical_cfg.encode("utf-8")).hexdigest()[:24]
    return Path(codex_home) / "router-nodes" / (scope_digest + ".key")


def selector_for_profile_name(profile_name: "typing.Optional[str]") -> str:
    if profile_name is None:
        return "default"
    if profile_name == PEER_BASE_SENTINEL:
        return "base"
    if not isinstance(profile_name, str) or not PEER_PROFILE_NAME_RE.fullmatch(
        profile_name
    ):
        raise ValueError("invalid peer profile name: {0!r}".format(profile_name))
    return "named/" + profile_name


def _parse_selector(selector: str) -> "typing.Tuple[str, typing.Optional[str]]":
    if selector == "default":
        return "default", None
    if selector == "base":
        return "base", None
    if selector.startswith("named/"):
        name = selector[len("named/") :]
        if PEER_PROFILE_NAME_RE.fullmatch(name):
            return "named", name
    raise ValueError("invalid peer profile selector")


async def call_on_router_loop(router_client, coro_fn):
    if router_client is None or not router_client.connected:
        raise OSError("router tunnel is not connected")
    loop = router_client_loop(router_client)
    if loop is None or not loop.is_running():
        raise OSError("router tunnel loop is not running")
    if asyncio.get_running_loop() is loop:
        return await coro_fn()
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro_fn(), loop))


def _connection_tokens(raw_pairs) -> "typing.Set[str]":
    tokens = set()
    for raw_name, raw_value in raw_pairs:
        name = raw_name.decode("latin-1") if isinstance(raw_name, bytes) else str(raw_name)
        if name.lower() != "connection":
            continue
        val = raw_value.decode("latin-1") if isinstance(raw_value, bytes) else str(raw_value)
        for part in val.split(","):
            token = part.strip().lower()
            if token:
                tokens.add(token)
    return tokens


def _filter_response_headers(raw_pairs) -> "typing.List[typing.Tuple[str, str]]":
    pairs_list = list(raw_pairs)
    blocked = _STRIPPED_RESPONSE_HEADERS | _connection_tokens(pairs_list)
    filtered = []
    for raw_name, raw_value in pairs_list:
        name = raw_name.decode("latin-1") if isinstance(raw_name, bytes) else str(raw_name)
        if name.lower() in blocked:
            continue
        value = raw_value.decode("latin-1") if isinstance(raw_value, bytes) else str(raw_value)
        filtered.append((name, value))
    return filtered


def _apply_response_headers(response: Response, header_pairs, default_sse_no_cache=False) -> Response:
    for name, value in header_pairs:
        response.headers.append(name, value)
    if default_sse_no_cache:
        ctype = response.headers.get("content-type", "").lower()
        if ctype.startswith("text/event-stream") and "cache-control" not in response.headers:
            response.headers["Cache-Control"] = "no-cache"
    return response


def _set_header_ci(headers: "typing.Dict[str, str]", name: str, value: str) -> None:
    lower = name.lower()
    for existing in list(headers):
        if existing.lower() == lower:
            del headers[existing]
    headers[name] = value


def _has_header_ci(headers: "typing.Mapping[str, str]", name: str) -> bool:
    lower = name.lower()
    return any(existing.lower() == lower for existing in headers)


def _parse_http_target_url(value: str) -> urllib.parse.SplitResult:
    if not isinstance(value, str) or not value or any(
        ch.isspace() or ord(ch) < 32 or ch in "\\#" for ch in value
    ):
        raise ValueError("target URL must be a valid http(s) URL")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("target URL must use http or https with a hostname")
    parsed.port
    return parsed


def _join_target_url(
    parsed: urllib.parse.SplitResult,
    subpath: str = "",
    extra_queries: "typing.Sequence[str]" = (),
) -> str:
    if subpath:
        if any(ord(ch) < 32 or ch in "\\#?" for ch in subpath):
            raise ValueError("invalid peer proxy subpath")
        prefix = parsed.path.rstrip("/")
        sep = "" if subpath.startswith("/") else "/"
        final_path = prefix + sep + subpath
    else:
        final_path = parsed.path
    query_parts = [q for q in (parsed.query, *extra_queries) if q]
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, final_path, "&".join(query_parts), "")
    )


def _error_message_from_body(body: bytes, status: int) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        text = body.decode("utf-8", errors="replace").strip()
        return text[:500] if text else "HTTP {0}".format(status)
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
        if isinstance(err, str) and err:
            return err
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return detail
    return "HTTP {0}".format(status)


def normalize_peer_manifest(
    raw: "typing.Mapping[str, typing.Any]",
) -> "typing.Dict[str, typing.Any]":
    """Validate manifest structure received from a trusted router peer."""
    if not isinstance(raw, typing.Mapping) or raw.get("version") != 2:
        raise ValueError("invalid peer profile manifest")
    resolved_profile = raw.get("resolved_profile")
    if resolved_profile is not None and not isinstance(resolved_profile, str):
        raise ValueError("invalid peer profile manifest: resolved_profile")
    selected = raw.get("selected")
    if not isinstance(selected, typing.Mapping):
        raise ValueError("invalid peer profile manifest: selected")
    harness = _parse_harness(selected)
    model = selected.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("invalid peer profile manifest: model")
    compat_api = selected.get("compat_api")
    if not isinstance(compat_api, str) or compat_api not in _VALID_COMPAT_APIS:
        raise ValueError("invalid peer profile manifest: compat_api")
    if harness == "pyagy" and compat_api != "gemini":
        raise ValueError("invalid peer profile manifest: pyagy requires gemini")
    clean_selected = json.loads(json.dumps(dict(selected)))
    return {
        "version": 2,
        "resolved_profile": resolved_profile,
        "selected": clean_selected,
    }


@dataclass(frozen=True)
class PeerWorkspaceConfig:
    config_path: Path
    launch_profile: "typing.Optional[str]"
    work_dir: Path
    toolset: "typing.Optional[typing.Sequence[str]]"
    workspace_context: "typing.Optional[str]"
    system_prompt: "typing.Optional[str]"
    timeout_seconds: float
    vllm_endpoint: "typing.Optional[str]"


class PeerManifestCache:
    """Persist peer manifests per pinned profile ID for offline session restore."""

    def __init__(self, codex_home: Path) -> None:
        self.directory = Path(codex_home) / "peer-profiles-cache"
        self._lock = threading.Lock()

    def _entry_path(self, profile_id: str) -> Path:
        parse_peer_profile(profile_id)
        digest = hashlib.sha256(profile_id.encode("utf-8")).hexdigest()[:32]
        return self.directory / (digest + ".json")

    def get(self, profile_id: str) -> "typing.Optional[typing.Dict[str, typing.Any]]":
        try:
            entry_path = self._entry_path(profile_id)
        except ValueError:
            return None
        with self._lock:
            if not entry_path.is_file():
                return None
            try:
                data = json.loads(entry_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            if not isinstance(data, dict) or data.get("profile_id") != profile_id:
                return None
            manifest = data.get("manifest")
            if not isinstance(manifest, dict):
                return None
            try:
                return normalize_peer_manifest(manifest)
            except ValueError:
                return None

    def put(self, profile_id: str, manifest: "typing.Mapping[str, typing.Any]") -> None:
        entry_path = self._entry_path(profile_id)
        clean_manifest = normalize_peer_manifest(manifest)
        raw = (
            json.dumps(
                {
                    "version": 2,
                    "profile_id": profile_id,
                    "manifest": clean_manifest,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        with self._lock:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=str(self.directory),
                prefix=entry_path.stem + ".",
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)
                try:
                    handle.write(raw)
                    handle.close()
                    os.replace(str(tmp_path), str(entry_path))
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()


async def _send_json_asgi(send, status: int, payload) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode("ascii")],
            ],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


def _parse_bridge_path(raw_path: str) -> "typing.Tuple[str, str, str]":
    if not raw_path.startswith("/p/"):
        raise ValueError("invalid peer bridge path")
    node_id, sep, rest = raw_path[3:].partition("/")
    if not PEER_NODE_ID_RE.fullmatch(node_id) or not sep:
        raise ValueError("invalid peer node ID")
    if rest == "default" or rest.startswith("default/"):
        return node_id, "default", rest[len("default") :]
    if rest == "base" or rest.startswith("base/"):
        return node_id, "base", rest[len("base") :]
    if rest.startswith("named/"):
        after_named = rest[len("named/") :]
        prof_name, slash, tail = after_named.partition("/")
        if PEER_PROFILE_NAME_RE.fullmatch(prof_name):
            return node_id, "named/" + prof_name, (slash + tail) if slash else ""
    raise ValueError("invalid peer profile selector")


class PeerLoopbackBridge:
    """Lazy 127.0.0.1 raw-ASGI server forwarding local client HTTP calls to `/api/peer-proxy`."""

    def __init__(self, router_client_getter) -> None:
        self._router_client_getter = router_client_getter
        self._lock = threading.Lock()
        self._server: "typing.Optional[uvicorn.Server]" = None
        self._thread: "typing.Optional[threading.Thread]" = None
        self._port: "typing.Optional[int]" = None
        self._default_key = secrets.token_urlsafe(32)
        os.environ.setdefault(PEER_BRIDGE_API_KEY_ENV, self._default_key)

    @property
    def router_client(self):
        return self._router_client_getter()

    @property
    def port(self) -> "typing.Optional[int]":
        return self._port

    def ensure_started(self, timeout_seconds: float = 10.0) -> str:
        os.environ.setdefault(PEER_BRIDGE_API_KEY_ENV, self._default_key)
        with self._lock:
            if self._server is not None and self._port is not None:
                return "http://127.0.0.1:{0}".format(self._port)
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            port = int(listener.getsockname()[1])
            async def asgi_app(scope, receive, send):
                await self._asgi_app(scope, receive, send)

            config = uvicorn.Config(
                asgi_app,
                host="127.0.0.1",
                port=port,
                loop="asyncio",
                interface="asgi3",
                log_level="error",
                access_log=False,
            )
            server = uvicorn.Server(config)

            def _serve():
                try:
                    server.run(sockets=[listener])
                finally:
                    listener.close()

            thread = threading.Thread(
                target=_serve, name="pycodex-peer-bridge", daemon=True
            )
            thread.start()
            deadline = time.time() + timeout_seconds
            while not server.started:
                if not thread.is_alive() or time.time() >= deadline:
                    raise RuntimeError("timed out waiting for peer loopback bridge")
                time.sleep(0.01)
            self._server = server
            self._thread = thread
            self._port = port
            return "http://127.0.0.1:{0}".format(port)

    def url_for(self, node_id: str, selector: str) -> str:
        return "{0}/p/{1}/{2}".format(self.ensure_started(), node_id, selector)

    def stop(self, timeout_seconds: float = 5.0) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
            self._port = None
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=timeout_seconds)

    def _authorized(self, raw_headers) -> bool:
        expected = os.environ.get(PEER_BRIDGE_API_KEY_ENV, "")
        if not expected:
            return False
        for raw_name, raw_value in raw_headers:
            lower = raw_name.lower()
            value = raw_value.decode("latin-1")
            if lower == b"authorization" and len(value) > 7 and value[:7].lower() == "bearer ":
                if secrets.compare_digest(value[7:], expected):
                    return True
            elif lower == b"x-goog-api-key" and value:
                if secrets.compare_digest(value, expected):
                    return True
        return False

    async def _asgi_app(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        if scope["method"] not in PEER_PROXY_METHODS:
            await _send_json_asgi(send, 405, {"error": {"message": "method not allowed"}})
            return
        raw_headers = scope.get("headers", ())
        if not self._authorized(raw_headers):
            await _send_json_asgi(
                send, 401, {"error": {"message": "invalid peer bridge authorization"}}
            )
            return
        raw_path = (
            scope.get("raw_path") or scope.get("path", "").encode("utf-8")
        ).decode("latin-1").split("?", 1)[0]
        try:
            node_id, selector, subpath = _parse_bridge_path(raw_path)
        except ValueError as exc:
            await _send_json_asgi(send, 400, {"error": {"message": str(exc)}})
            return

        router_client = self.router_client
        router_loop = router_client_loop(router_client)
        if (
            router_client is None
            or not router_client.connected
            or router_loop is None
            or not router_loop.is_running()
        ):
            await _send_json_asgi(
                send, 503, {"error": {"message": "router tunnel is not connected"}}
            )
            return

        blocked = (
            _HOP_BY_HOP_HEADERS
            | _connection_tokens(raw_headers)
            | {"host", "content-length", "authorization", "x-goog-api-key"}
        )
        forwarded_headers = []
        for raw_name, raw_value in raw_headers:
            lower = raw_name.decode("latin-1").lower()
            if lower in blocked or lower.startswith("x-peer-"):
                continue
            forwarded_headers.append([lower.encode("latin-1"), raw_value])
        forwarded_headers.append([b"x-peer-profile", selector.encode("ascii")])
        if subpath:
            forwarded_headers.append([b"x-peer-subpath", subpath.encode("latin-1")])

        peer_scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": scope["method"],
            "scheme": "http",
            "path": "/api/peer-proxy",
            "raw_path": b"/api/peer-proxy",
            "root_path": "",
            "query_string": scope.get("query_string", b""),
            "headers": forwarded_headers,
            "client": ("127.0.0.1", 0),
            "server": ("127.0.0.1", self._port or 0),
        }
        bridge_loop = asyncio.get_running_loop()
        started = False
        disconnected = False

        async def bridge_receive():
            nonlocal disconnected
            conc = asyncio.run_coroutine_threadsafe(receive(), bridge_loop)
            try:
                msg = await asyncio.wrap_future(conc)
            except asyncio.CancelledError:
                conc.cancel()
                raise
            if msg.get("type") == "http.disconnect":
                disconnected = True
            return msg

        async def bridge_send(event):
            nonlocal started
            if disconnected:
                return
            if event["type"] == "http.response.start":
                started = True
                clean = [
                    [k.encode("latin-1"), v.encode("latin-1")]
                    for k, v in _filter_response_headers(event.get("headers", ()))
                ]
                is_sse = any(
                    k == b"content-type" and v.lower().startswith(b"text/event-stream")
                    for k, v in clean
                )
                if is_sse:
                    if not any(k.lower() == b"cache-control" for k, _ in clean):
                        clean.append([b"cache-control", b"no-cache"])
                    clean.append([b"connection", b"close"])
                event = {"type": "http.response.start", "status": event["status"], "headers": clean}
            conc = asyncio.run_coroutine_threadsafe(send(event), bridge_loop)
            try:
                await asyncio.wrap_future(conc)
            except asyncio.CancelledError:
                conc.cancel()
                raise

        fut = asyncio.run_coroutine_threadsafe(
            peer_forward(router_client, node_id, peer_scope, bridge_receive, bridge_send),
            router_loop,
        )
        try:
            await asyncio.wrap_future(fut)
        except asyncio.CancelledError:
            fut.cancel()
            raise
        except Exception as exc:
            if not started and not disconnected:
                await _send_json_asgi(send, 503, {"error": {"message": str(exc)}})


def _apply_local_permissions(
    selected: "typing.Dict[str, typing.Any]", peer_config: PeerWorkspaceConfig
) -> None:
    try:
        local_selected = read_codex_config(
            peer_config.config_path, peer_config.launch_profile
        )
    except (OSError, ValueError):
        return
    for key in ("approval_policy", "sandbox_mode"):
        if key in local_selected:
            selected[key] = local_selected[key]


def build_peer_workspace_session(
    manifest: "typing.Mapping[str, typing.Any]",
    peer_config: PeerWorkspaceConfig,
    peer_bridge: PeerLoopbackBridge,
    node_id: str,
    selector: str,
    tools_factory=get_tools,
    extra_tools=(),
) -> WorkspaceInteractiveSession:
    clean = normalize_peer_manifest(manifest)
    selected = copy.deepcopy(clean["selected"])
    _apply_local_permissions(selected, peer_config)
    provider_name = selected.get("model_provider", "openai")
    providers = selected.setdefault("model_providers", {})
    provider_table = dict(providers.get(provider_name) or {})
    provider_table["base_url"] = peer_bridge.url_for(node_id, selector)
    provider_table["env_key"] = PEER_BRIDGE_API_KEY_ENV
    providers[provider_name] = provider_table

    catalog = load_models_by_slug()
    harness = _parse_harness(selected)
    provider = ResponsesProviderConfig.from_mapping(selected, catalog)
    context = ContextConfig.from_mapping(selected, peer_config.config_path, catalog)
    codex_cfg = CodexConfig(
        path=peer_config.config_path,
        profile=None,
        provider=provider,
        context=context,
        harness=harness,
    )
    return build_session_from_codex_config(
        codex_cfg,
        peer_config.work_dir,
        peer_config.toolset,
        workspace_context=peer_config.workspace_context,
        system_prompt=peer_config.system_prompt,
        tools_factory=tools_factory,
        extra_tools=extra_tools,
        config_path=peer_config.config_path,
        timeout_seconds=peer_config.timeout_seconds,
    )


def _pinned_target_for_manifest(
    node_id: str,
    requested_profile_name: "typing.Optional[str]",
    manifest: "typing.Mapping[str, typing.Any]",
) -> "typing.Tuple[str, str]":
    if requested_profile_name == PEER_BASE_SENTINEL:
        return "base", format_peer_profile(node_id, PEER_BASE_SENTINEL)
    if requested_profile_name is not None:
        return (
            selector_for_profile_name(requested_profile_name),
            format_peer_profile(node_id, requested_profile_name),
        )
    resolved_name = manifest.get("resolved_profile")
    if isinstance(resolved_name, str) and PEER_PROFILE_NAME_RE.fullmatch(resolved_name):
        return "named/" + resolved_name, format_peer_profile(node_id, resolved_name)
    return "base", format_peer_profile(node_id, PEER_BASE_SENTINEL)


async def resolve_peer_session_factory(
    profile_id: str,
    peer_config: "typing.Optional[PeerWorkspaceConfig]",
    peer_bridge: "typing.Optional[PeerLoopbackBridge]",
    cache: PeerManifestCache,
    allow_cached: bool = False,
):
    node_id, profile_name = parse_peer_profile(profile_id)
    if peer_config is None or peer_bridge is None:
        raise ValueError("peer profiles are unavailable in this workspace")
    fetch_selector = selector_for_profile_name(profile_name)
    manifest = (
        cache.get(profile_id)
        if (allow_cached and profile_name is not None)
        else None
    )
    if manifest is None:
        router_client = peer_bridge.router_client
        if router_client is None or not router_client.connected:
            raise ValueError(
                "unknown Codex profile: {0} (router is not connected)".format(
                    profile_id
                )
            )
        try:
            status, _headers, body = await call_on_router_loop(
                router_client,
                lambda: peer_http_request(
                    router_client, node_id, "GET", "/api/peer-profiles/" + fetch_selector
                ),
            )
        except OSError as exc:
            raise ValueError(
                "peer profile {0} is unreachable: {1}".format(profile_id, exc)
            ) from exc
        if status >= 400:
            raise ValueError(_error_message_from_body(body, status))
        try:
            parsed = json.loads(body.decode("utf-8"))
        except ValueError as exc:
            raise ValueError("invalid peer profile manifest") from exc
        if not isinstance(parsed, dict):
            raise ValueError("invalid peer profile manifest")
        manifest = normalize_peer_manifest(parsed)
        pinned_selector, pinned_profile_id = _pinned_target_for_manifest(
            node_id, profile_name, manifest
        )
        cache.put(pinned_profile_id, manifest)
    else:
        pinned_selector, pinned_profile_id = _pinned_target_for_manifest(
            node_id, profile_name, manifest
        )

    server_loop = asyncio.get_running_loop()
    factory = lambda: ThreadedWorkspaceInteractiveSession(
        partial(
            build_peer_workspace_session,
            manifest,
            peer_config,
            peer_bridge,
            node_id,
            pinned_selector,
        ),
        server_loop,
    )
    return factory, pinned_profile_id


def _extract_raw_socket(response: requests.Response):
    raw = getattr(response, "raw", None)
    for candidate in (
        getattr(getattr(raw, "_connection", None), "sock", None),
        getattr(getattr(raw, "_fp", None), "sock", None),
        getattr(getattr(getattr(raw, "_fp", None), "fp", None), "_sock", None),
        getattr(getattr(getattr(raw, "_fp", None), "fp", None), "raw", None),
    ):
        if candidate is not None and hasattr(candidate, "close"):
            return candidate
    return None


def _iter_raw_response_bytes(response: requests.Response) -> "typing.Iterator[bytes]":
    raw = getattr(response, "raw", None)
    if raw is not None and getattr(raw, "chunked", False):
        for item in raw.read_chunked(8192, decode_content=True):
            if item:
                yield bytes(item)
        return
    if raw is not None and hasattr(raw, "read1"):
        while True:
            item = raw.read1(8192, decode_content=True)
            if not item:
                return
            yield bytes(item)
    encoding = (response.headers.get("Content-Encoding") or "").strip().lower()
    raw_fp = getattr(raw, "_fp", None) if raw is not None else None
    if encoding in ("", "identity") and raw_fp is not None and hasattr(raw_fp, "read1"):
        while True:
            item = raw_fp.read1(8192)
            if not item:
                return
            yield bytes(item)
    for item in response.iter_content(chunk_size=512):
        if item:
            yield bytes(item)


async def _aiter_upstream_body(response: requests.Response, raw_sock):
    loop = asyncio.get_running_loop()
    iterator = _iter_raw_response_bytes(response)
    state_lock = threading.Lock()
    state = {"busy": False, "closed": False, "eof": False}

    def _next_chunk():
        try:
            return next(iterator)
        except StopIteration:
            return None
        finally:
            should_close = False
            with state_lock:
                state["busy"] = False
                if state["closed"]:
                    should_close = True
            if should_close:
                response.close()

    try:
        while True:
            with state_lock:
                state["busy"] = True
            chunk = await loop.run_in_executor(None, _next_chunk)
            if chunk is None:
                state["eof"] = True
                return
            if chunk:
                yield chunk
    finally:
        if not state["eof"] and raw_sock is not None:
            if hasattr(raw_sock, "shutdown"):
                try:
                    raw_sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            try:
                raw_sock.close()
            except OSError:
                pass
        close_now = False
        with state_lock:
            if not state["closed"]:
                state["closed"] = True
                close_now = not state["busy"]
        if close_now:
            response.close()


class _PeerUpstreamProbeError(RuntimeError):
    """Raised when probing a remote provider (such as vLLM /models) fails."""


class _PeerForbiddenTargetError(ValueError):
    """Raised when `/api/peer-proxy` targets the host's own workspace listener."""


@dataclass(frozen=True)
class _ResolvedProviderCredentials:
    compat_api_key: "typing.Optional[str]"
    native_api_key: "typing.Optional[str]"
    native_auth_error: "typing.Optional[str]"
    http_headers: "typing.Dict[str, str]"
    headers_error: "typing.Optional[str]"


def _forbidden_local_ports(request: Request) -> "typing.Set[int]":
    ports = set()
    server = request.scope.get("server")
    if isinstance(server, (tuple, list)) and len(server) >= 2 and isinstance(server[1], int) and server[1] > 0:
        ports.add(server[1])
    state = getattr(request.app, "state", None)
    if state is None:
        return ports
    local_api_url = getattr(state, "local_api_url", None)
    if isinstance(local_api_url, str) and local_api_url:
        try:
            parsed = urllib.parse.urlsplit(local_api_url)
            if parsed.port:
                ports.add(parsed.port)
        except ValueError:
            pass
    router_client = getattr(state, "router_client", None)
    if router_client is not None and isinstance(getattr(router_client, "port", None), int) and router_client.port > 0:
        ports.add(router_client.port)
    peer_bridge = getattr(state, "peer_bridge", None)
    if peer_bridge is not None and isinstance(getattr(peer_bridge, "port", None), int) and peer_bridge.port > 0:
        ports.add(peer_bridge.port)
    return ports


def _local_interface_ips() -> "typing.Set[str]":
    addrs = {"127.0.0.1", "::1", "0.0.0.0", "::"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addrs.add(info[4][0].split("%", 1)[0])
    except OSError:
        pass
    return addrs


def _check_not_self_target(parsed: urllib.parse.SplitResult, forbidden_ports: "typing.Set[int]") -> None:
    if not forbidden_ports:
        return
    effective_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if effective_port not in forbidden_ports:
        return
    host = (parsed.hostname or "").strip().lower()
    if host == "localhost" or host.endswith(".localhost"):
        raise _PeerForbiddenTargetError("peer proxy cannot target the host workspace server")
    local_ips = _local_interface_ips()
    try:
        resolved = [
            info[4][0].split("%", 1)[0]
            for info in socket.getaddrinfo(host, effective_port, type=socket.SOCK_STREAM)
        ]
    except OSError:
        resolved = [host]
    for addr in resolved:
        if addr in local_ips:
            raise _PeerForbiddenTargetError("peer proxy cannot target the host workspace server")
        try:
            ip_obj = ipaddress.ip_address(addr)
            if ip_obj.is_loopback or ip_obj.is_unspecified:
                raise _PeerForbiddenTargetError("peer proxy cannot target the host workspace server")
        except ValueError:
            continue


def _incoming_request_headers(
    request: Request, extra_blocked: "typing.AbstractSet[str]" = frozenset()
) -> "typing.Dict[str, str]":
    blocked = (
        _HOP_BY_HOP_HEADERS
        | _connection_tokens(request.headers.raw)
        | {"host", "content-length"}
        | set(extra_blocked)
    )
    merged = {}
    key_by_lower = {}
    for raw_name, raw_value in request.headers.raw:
        name = raw_name.decode("latin-1")
        lower = name.lower()
        if lower in blocked or lower.startswith("x-peer-"):
            continue
        value = raw_value.decode("latin-1")
        existing = key_by_lower.get(lower)
        if existing is None:
            key_by_lower[lower] = name
            merged[name] = value
        else:
            sep = "; " if lower == "cookie" else ", "
            merged[existing] = merged[existing] + sep + value
    return merged


class PeerProfileProvider:
    """Serve profile manifests and an unconditional `/api/peer-proxy` for router peers."""

    def __init__(
        self,
        config_path: "typing.Union[str, Path]",
        launch_profile: "typing.Optional[str]" = None,
        timeout_seconds: float = 120.0,
        vllm_endpoint: "typing.Optional[str]" = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.launch_profile = launch_profile
        self.timeout_seconds = timeout_seconds
        self.vllm_endpoint = vllm_endpoint
        self._lock = threading.Lock()
        self._model_catalog = None
        self._http_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=32)
        self._http_session.mount("http://", adapter)
        self._http_session.mount("https://", adapter)

    def close(self) -> None:
        self._http_session.close()

    @classmethod
    def from_workspace_config(cls, peer_config: PeerWorkspaceConfig) -> "PeerProfileProvider":
        return cls(
            config_path=peer_config.config_path,
            launch_profile=peer_config.launch_profile,
            timeout_seconds=peer_config.timeout_seconds,
            vllm_endpoint=peer_config.vllm_endpoint,
        )

    @classmethod
    def lazy_routes(cls, explicit_provider, peer_config_getter):
        cached = {"cfg": None, "provider": explicit_provider}

        def _get_provider() -> "typing.Optional[PeerProfileProvider]":
            if explicit_provider is not None:
                return explicit_provider
            cfg = peer_config_getter()
            if cfg is None:
                return None
            if cached["cfg"] != cfg or cached["provider"] is None:
                if cached["provider"] is not None:
                    cached["provider"].close()
                cached["cfg"] = cfg
                cached["provider"] = cls.from_workspace_config(cfg)
            return cached["provider"]

        def _close_cached() -> None:
            prov = cached.get("provider")
            if prov is not None:
                prov.close()

        async def _dispatch(request: Request, handler_attr: str) -> Response:
            if not trusted_peer_request(request.scope):
                return JSONResponse({"error": "peer tunnel access required"}, status_code=403)
            prov = _get_provider()
            if prov is None:
                return JSONResponse({"error": "peer profiles are unavailable"}, status_code=404)
            return await getattr(prov, handler_attr)(request)

        async def list_profiles(request: Request) -> Response:
            return await _dispatch(request, "_list_profiles_endpoint")

        async def default_manifest(request: Request) -> Response:
            return await _dispatch(request, "_default_manifest_endpoint")

        async def base_manifest(request: Request) -> Response:
            return await _dispatch(request, "_base_manifest_endpoint")

        async def named_manifest(request: Request) -> Response:
            return await _dispatch(request, "_named_manifest_endpoint")

        async def peer_proxy(request: Request) -> Response:
            return await _dispatch(request, "_handle_peer_proxy")

        return [
            Route("/api/peer-profiles", list_profiles, methods=["GET"]),
            Route("/api/peer-profiles/default", default_manifest, methods=["GET"]),
            Route("/api/peer-profiles/base", base_manifest, methods=["GET"]),
            Route("/api/peer-profiles/named/{profile_name}", named_manifest, methods=["GET"]),
            Route("/api/peer-proxy", peer_proxy, methods=PEER_PROXY_METHODS),
        ], _close_cached

    def _get_model_catalog(self):
        with self._lock:
            if self._model_catalog is None:
                self._model_catalog = load_models_by_slug()
            return self._model_catalog

    def _load_env_map(self) -> "typing.Dict[str, str]":
        env_map = dict(os.environ)
        dotenv_path = self.config_path.resolve().parent / DOTENV_FILENAME
        if dotenv_path.is_file():
            try:
                raw_entries = parse_dotenv(
                    dotenv_path.read_text(encoding="utf-8", errors="replace")
                )
                for key, value in raw_entries.items():
                    if not key.upper().startswith(ILLEGAL_ENV_VAR_PREFIX):
                        env_map[key] = value
            except FileNotFoundError:
                pass
        return env_map

    def _read_selected_mapping(
        self, selector_mode: str, profile_name: "typing.Optional[str]" = None
    ) -> "typing.Tuple[typing.Dict[str, typing.Any], str]":
        if selector_mode == "default":
            selected = read_codex_config(
                self.config_path, self.launch_profile, use_default_profile=True
            )
        elif selector_mode == "base":
            selected = read_codex_config(
                self.config_path, None, use_default_profile=False
            )
        elif selector_mode == "named":
            if profile_name is None or not PEER_PROFILE_NAME_RE.fullmatch(profile_name):
                raise ValueError("unknown Codex profile: {0}".format(profile_name))
            available = list_codex_profiles(self.config_path)
            if profile_name not in available:
                raise ValueError("unknown Codex profile: {0}".format(profile_name))
            selected = read_codex_config(
                self.config_path, profile_name, use_default_profile=False
            )
        else:
            raise ValueError("invalid peer profile selector: " + selector_mode)
        return selected, _parse_harness(selected)

    def _resolve_credentials(
        self, harness: str, provider_cfg: ResponsesProviderConfig, env_map: "typing.Mapping[str, str]"
    ) -> _ResolvedProviderCredentials:
        merged_raw_headers = dict(provider_cfg.http_headers)
        for name, env_var in provider_cfg.env_http_headers.items():
            val = env_map.get(env_var)
            if val is not None and val.strip():
                merged_raw_headers[name] = val
        try:
            http_headers = build_http_headers(merged_raw_headers, {})
            headers_error = None
        except (ValueError, RuntimeError) as exc:
            http_headers = {}
            headers_error = str(exc)

        if harness != "pyagy" and self.vllm_endpoint is not None:
            compat_api_key = None
        elif provider_cfg.api_key_env is None:
            compat_api_key = None
        else:
            raw_key = env_map.get(provider_cfg.api_key_env, "").strip()
            compat_api_key = raw_key or None

        if not provider_cfg.api_key_env:
            native_api_key = None
            native_auth_error = None
        else:
            raw_native = env_map.get(provider_cfg.api_key_env, "")
            if raw_native:
                native_api_key = raw_native
                native_auth_error = None
            else:
                native_api_key = None
                native_auth_error = (
                    "missing API key environment variable: " + provider_cfg.api_key_env
                )

        return _ResolvedProviderCredentials(
            compat_api_key=compat_api_key,
            native_api_key=native_api_key,
            native_auth_error=native_auth_error,
            http_headers=http_headers,
            headers_error=headers_error,
        )

    def _effective_compat_api_for(
        self, harness: str, provider_cfg: ResponsesProviderConfig
    ) -> str:
        if harness == "pyagy":
            if provider_cfg.compat_api != "gemini":
                raise ValueError("pyagy requires a provider with compat_api='gemini'")
            return "gemini"
        if self.vllm_endpoint is not None:
            if provider_cfg.compat_api not in ("none", "chat_completions"):
                raise ValueError(
                    "--vllm-endpoint cannot be combined with compat_api="
                    + provider_cfg.compat_api
                )
            return "chat_completions"
        if provider_cfg.compat_api not in _VALID_COMPAT_APIS:
            raise ValueError(
                "unsupported compat_api: {0!r}".format(provider_cfg.compat_api)
            )
        return provider_cfg.compat_api

    def _load_proxy_provider_and_credentials(
        self, selector_mode: str, profile_name: "typing.Optional[str]" = None
    ) -> "typing.Tuple[str, ResponsesProviderConfig, _ResolvedProviderCredentials]":
        env_map = self._load_env_map()
        selected, harness = self._read_selected_mapping(selector_mode, profile_name)
        provider_cfg = ResponsesProviderConfig.from_mapping(
            selected, self._get_model_catalog()
        )
        creds = self._resolve_credentials(harness, provider_cfg, env_map)
        return harness, provider_cfg, creds

    def _build_manifest_sync(
        self, selector_mode: str, profile_name: "typing.Optional[str]" = None
    ) -> "typing.Dict[str, object]":
        env_map = self._load_env_map()
        selected, harness = self._read_selected_mapping(selector_mode, profile_name)
        catalog = self._get_model_catalog()
        provider_cfg = ResponsesProviderConfig.from_mapping(selected, catalog)
        context_cfg = ContextConfig.from_mapping(selected, self.config_path, catalog)
        creds = self._resolve_credentials(harness, provider_cfg, env_map)
        effective_compat = self._effective_compat_api_for(harness, provider_cfg)

        effective_model = provider_cfg.model
        effective_provider_name = provider_cfg.provider_name
        if harness != "pyagy" and self.vllm_endpoint is not None:
            if creds.headers_error:
                raise ValueError(creds.headers_error)
            frozen_probe_provider = replace(
                provider_cfg,
                http_headers=dict(creds.http_headers),
                env_http_headers={},
            )
            try:
                effective_model = _resolve_vllm_model(
                    self.vllm_endpoint, frozen_probe_provider, self.timeout_seconds
                )
            except Exception as exc:
                raise _PeerUpstreamProbeError(str(exc)) from exc
            effective_provider_name = "vllm"

        sanitized = {
            k: copy.deepcopy(v)
            for k, v in selected.items()
            if k not in ("profiles", "openai_base_url", "approval_policy", "sandbox_mode", "model_instructions_file")
        }
        if context_cfg.base_instructions is None and context_cfg.model_instructions_file is not None:
            sanitized["base_instructions"] = context_cfg.model_instructions_file.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
        raw_provider_table = dict(
            (selected.get("model_providers") or {}).get(provider_cfg.provider_name) or {}
        )
        clean_provider_table = {
            k: copy.deepcopy(v)
            for k, v in raw_provider_table.items()
            if k not in _SECRET_PROVIDER_KEYS
        }
        sanitized["model"] = effective_model
        sanitized["model_provider"] = effective_provider_name
        sanitized["compat_api"] = effective_compat
        sanitized["harness"] = harness
        sanitized["model_providers"] = {effective_provider_name: clean_provider_table}

        return normalize_peer_manifest(
            {
                "version": 2,
                "resolved_profile": selected["profile"],
                "selected": sanitized,
            }
        )

    async def _execute_upstream_proxy(
        self,
        method: str,
        url: str,
        headers: "typing.Dict[str, str]",
        raw_body: bytes,
        timeout: "typing.Tuple[float, float]",
        allow_redirects: bool,
        forbidden_ports: "typing.Set[int]",
    ) -> Response:
        def _send_sync():
            session = self._http_session
            cur_method = method
            cur_url = url
            cur_body = raw_body if raw_body else None
            cur_headers = dict(headers)
            for _hop in range(_MAX_PROXY_REDIRECTS + 1):
                parsed = _parse_http_target_url(cur_url)
                _check_not_self_target(parsed, forbidden_ports)
                prepared = requests.PreparedRequest()
                prepared.prepare(
                    method=cur_method,
                    url=cur_url,
                    headers=cur_headers,
                    data=cur_body,
                )
                settings = session.merge_environment_settings(
                    prepared.url,
                    proxies={},
                    stream=True,
                    verify=None,
                    cert=None,
                )
                verify = _requests_verify_setting()
                if verify is not None:
                    settings["verify"] = verify
                resp = session.send(
                    prepared,
                    timeout=timeout,
                    allow_redirects=False,
                    **settings
                )
                try:
                    location = resp.headers.get("Location")
                    if (
                        allow_redirects
                        and resp.status_code in (301, 302, 303, 307, 308)
                        and location
                        and _hop < _MAX_PROXY_REDIRECTS
                    ):
                        next_url = urllib.parse.urljoin(cur_url, location)
                        next_parsed = _parse_http_target_url(next_url)
                        if (next_parsed.scheme, next_parsed.netloc) != (parsed.scheme, parsed.netloc):
                            for secret_hdr in ("authorization", "cookie", "proxy-authorization"):
                                for k in list(cur_headers):
                                    if k.lower() == secret_hdr:
                                        del cur_headers[k]
                        if resp.status_code == 303 or (
                            resp.status_code in (301, 302) and cur_method not in ("GET", "HEAD")
                        ):
                            cur_method = "GET"
                            cur_body = None
                            for k in list(cur_headers):
                                if k.lower() in ("content-type", "content-length", "transfer-encoding"):
                                    del cur_headers[k]
                        resp.close()
                        cur_url = next_url
                        continue

                    raw_obj = getattr(resp, "raw", None)
                    raw_hdrs = (
                        raw_obj.headers.items()
                        if raw_obj is not None and getattr(raw_obj, "headers", None) is not None
                        else resp.headers.items()
                    )
                    filtered_headers = _filter_response_headers(raw_hdrs)
                    if cur_method == "HEAD" or resp.status_code in (204, 304):
                        resp.close()
                        return resp.status_code, filtered_headers, b"", None
                    if resp.status_code >= 400:
                        content = resp.content
                        resp.close()
                        return resp.status_code, filtered_headers, content, None
                    raw_sock = _extract_raw_socket(resp)
                    return resp.status_code, filtered_headers, b"", (resp, raw_sock)
                except BaseException:
                    resp.close()
                    raise
            raise requests.TooManyRedirects("exceeded peer proxy redirect limit")

        try:
            status_code, filtered_headers, content, stream_pair = await run_in_threadpool(
                _send_sync
            )
        except _PeerForbiddenTargetError as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=403)
        except ValueError as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
        except requests.RequestException as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "upstream_connection_error"}},
                status_code=502,
            )

        if stream_pair is None:
            return _apply_response_headers(
                Response(content, status_code=status_code), filtered_headers
            )
        resp, raw_sock = stream_pair
        return _apply_response_headers(
            StreamingResponse(
                _aiter_upstream_body(resp, raw_sock),
                status_code=status_code,
            ),
            filtered_headers,
            default_sse_no_cache=True,
        )

    async def _handle_peer_proxy(self, request: Request) -> Response:
        raw_target_url = request.headers.get("x-peer-target-url")
        raw_profile = request.headers.get("x-peer-profile")
        raw_subpath = request.headers.get("x-peer-subpath", "")
        req_query = request.scope.get("query_string", b"").decode("latin-1")
        forbidden_ports = _forbidden_local_ports(request)

        if bool(raw_target_url) == bool(raw_profile):
            return JSONResponse(
                {
                    "error": {
                        "message": "specify exactly one of X-Peer-Target-Url or X-Peer-Profile"
                    }
                },
                status_code=400,
            )

        if raw_target_url is not None:
            try:
                parsed_target = _parse_http_target_url(raw_target_url)
                url = _join_target_url(
                    parsed_target, subpath=raw_subpath, extra_queries=(req_query,)
                )
            except ValueError as exc:
                return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
            headers = _incoming_request_headers(request)
            redir_hdr = request.headers.get("x-peer-allow-redirects", "").strip().lower()
            allow_redirects = redir_hdr not in ("0", "false", "no")
            timeout_hdr = request.headers.get("x-peer-timeout-ms", "").strip()
            if timeout_hdr:
                if not timeout_hdr.isdigit() or int(timeout_hdr) < 1:
                    return JSONResponse(
                        {"error": {"message": "invalid X-Peer-Timeout-Ms"}}, status_code=400
                    )
                read_timeout = int(timeout_hdr) / 1000.0
            else:
                read_timeout = max(self.timeout_seconds, 300.0)
            timeout = (max(self.timeout_seconds, 1.0), read_timeout)
            raw_body = bytes(await request.body())
            return await self._execute_upstream_proxy(
                request.method, url, headers, raw_body, timeout, allow_redirects, forbidden_ports
            )

        try:
            selector_mode, profile_name = _parse_selector(raw_profile or "")
            harness, provider_cfg, creds = await run_in_threadpool(
                self._load_proxy_provider_and_credentials, selector_mode, profile_name
            )
            effective_compat = self._effective_compat_api_for(harness, provider_cfg)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)

        if creds.headers_error or (effective_compat == "none" and creds.native_auth_error):
            return JSONResponse(
                {
                    "error": {
                        "message": creds.headers_error or creds.native_auth_error,
                        "type": "upstream_auth_configuration_error",
                    }
                },
                status_code=502,
            )

        if harness != "pyagy" and self.vllm_endpoint is not None:
            upstream_base = CompatServerConfig.from_base_url(
                self.vllm_endpoint, outcomming_api="chat_completions"
            ).outcomming_base_url
            provider_query = ""
        else:
            upstream_base = (
                CompatServerConfig.from_base_url(
                    provider_cfg.base_url, outcomming_api=effective_compat
                ).outcomming_base_url
                if effective_compat != "none"
                else provider_cfg.base_url
            )
            provider_query = (
                urllib.parse.urlencode(provider_cfg.query_params)
                if provider_cfg.query_params
                else ""
            )

        try:
            parsed_base = _parse_http_target_url(upstream_base)
            url = _join_target_url(
                parsed_base, subpath=raw_subpath, extra_queries=(provider_query, req_query)
            )
        except ValueError as exc:
            return JSONResponse({"error": {"message": str(exc)}}, status_code=400)

        headers = _incoming_request_headers(
            request, extra_blocked=frozenset(("authorization", "x-goog-api-key"))
        )
        for h_name, h_val in creds.http_headers.items():
            lower_h = h_name.lower()
            if harness == "pyagy" and lower_h in ("user-agent", "x-goog-api-client"):
                continue
            if lower_h in _REQUEST_PROTECTED_HEADERS and _has_header_ci(headers, h_name):
                continue
            _set_header_ci(headers, h_name, h_val)

        if effective_compat == "none":
            if creds.native_api_key is not None:
                _set_header_ci(headers, "Authorization", "Bearer " + creds.native_api_key)
        elif effective_compat == "gemini":
            if creds.compat_api_key is not None:
                _set_header_ci(headers, "x-goog-api-key", creds.compat_api_key)
        else:
            if creds.compat_api_key is not None:
                _set_header_ci(headers, "Authorization", "Bearer " + creds.compat_api_key)

        timeout = (
            max(self.timeout_seconds, 1.0),
            max(provider_cfg.effective_stream_idle_timeout_seconds(), self.timeout_seconds),
        )
        raw_body = bytes(await request.body())
        return await self._execute_upstream_proxy(
            request.method, url, headers, raw_body, timeout, False, forbidden_ports
        )

    def _list_profiles_sync(self) -> "typing.Tuple[typing.List[str], typing.Optional[str]]":
        selected = read_codex_config(self.config_path, self.launch_profile)
        profiles = list_codex_profiles(self.config_path)
        default_profile = selected["profile"]
        if default_profile is not None and default_profile not in profiles:
            raise ValueError("unknown Codex profile: {0}".format(default_profile))
        return profiles, default_profile

    async def _list_profiles_endpoint(self, _request: Request) -> JSONResponse:
        try:
            profiles, default_profile = await run_in_threadpool(self._list_profiles_sync)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(
            {"profiles": profiles, "default_profile": default_profile},
            headers={"Cache-Control": "no-store"},
        )

    async def _manifest_endpoint(
        self, selector_mode: str, profile_name: "typing.Optional[str]" = None
    ) -> JSONResponse:
        try:
            manifest = await run_in_threadpool(
                self._build_manifest_sync, selector_mode, profile_name
            )
        except _PeerUpstreamProbeError as exc:
            return JSONResponse(
                {"error": str(exc)}, status_code=502, headers={"Cache-Control": "no-store"}
            )
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            return JSONResponse(
                {"error": str(exc)}, status_code=400, headers={"Cache-Control": "no-store"}
            )
        return JSONResponse(manifest, headers={"Cache-Control": "no-store"})

    async def _default_manifest_endpoint(self, _request: Request) -> JSONResponse:
        return await self._manifest_endpoint("default")

    async def _base_manifest_endpoint(self, _request: Request) -> JSONResponse:
        return await self._manifest_endpoint("base")

    async def _named_manifest_endpoint(self, request: Request) -> JSONResponse:
        return await self._manifest_endpoint("named", request.path_params["profile_name"])

    def routes(self) -> "typing.List[Route]":
        route_list, _close = self.lazy_routes(self, lambda: None)
        return route_list
