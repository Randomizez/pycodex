"""Shared workspace routes and HTTP/WebSocket response adapters."""

import asyncio
import gzip
import hashlib
import html
import json
import mimetypes
import stat
import subprocess
import typing
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from PIL import Image
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from pycodex.utils.compat import importlib_metadata
from pycodex.utils.skill_catalog import list_skills
from pycodex_router.client import peer_http_request, peer_list_nodes
from pycodex_ws.server_support import (
    AUTH_COOKIE_NAME,
    html_response as _html_response,
    json_object_body as _json_object_body,
)

from .customization import frontend_asset_response, frontend_directory, load_workspace_template
from . import external_call
from .peer_profiles import (
    PEER_NODE_ID_RE,
    _error_message_from_body,
    call_on_router_loop,
)
from .runtime_info import agent_help_text, runtime_info_prompt
from .session_api import session_api_routes
from .vscode_server import (
    VsCodeRuntime,
    extract_vscode_window_id,
    forwarded_vscode_prefix,
    inject_vscode_window_payload,
    resolve_vscode_link,
    strip_vscode_window_payload,
    websocket_origin_allowed,
)
from .workspace_views import WorkspaceViewOpener, raster_image_type
from .workspaces import WorkspaceSessionManager, session_snapshot


@dataclass(frozen=True)
class WorkspaceRouteContext:
    manager: "WorkspaceSessionManager"
    board_path: "typing.Optional[Path]"
    work_dir: "Path"
    title: "typing.Optional[str]" = None
    workspace_id: "typing.Optional[str]" = None


_NOTIFICATION_SW_JS = """self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", event => event.waitUntil(self.clients.claim()));
self.addEventListener("notificationclick", event => {
  event.notification.close();
  const data = event.notification.data || {};
  event.waitUntil(
    self.clients.matchAll({type: "window", includeUncontrolled: true}).then(list => {
      const target = list.find(client => client.url.startsWith(self.registration.scope)) || list[0];
      if (target && "focus" in target) {
        if (data.sessionId) target.postMessage({type: "pycodex:select-session", sessionId: data.sessionId});
        return target.focus();
      }
      if (data.url && self.clients.openWindow) return self.clients.openWindow(data.url);
    })
  );
});
"""


def workspace_routes(resolve_workspace, sharing, prefix="", leading_routes=(), vscode=None):
    if vscode is None and sharing.password:
        vscode = getattr(sharing, "vscode", None)
        if vscode is None:
            vscode = VsCodeRuntime()
            sharing.vscode = vscode
    view_asset_cache = {}
    asset_types = {
        "workspace.mjs": "text/javascript", "workspace.css": "text/css",
        "workspace_dom.mjs": "text/javascript", "sessions_ui.mjs": "text/javascript",
        "conversation.mjs": "text/javascript", "markdown.mjs": "text/javascript",
        "dialogs.mjs": "text/javascript", "custom_frontend.mjs": "text/javascript",
        "banners.mjs": "text/javascript",
        "content_views.mjs": "text/javascript", "content_views.css": "text/css",
        "content_views.LICENSE.txt": "text/plain", "editor.mjs": "text/javascript",
        "editor.css": "text/css",
    }
    vscode_methods = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

    def workspace(request):
        try:
            return resolve_workspace(request)
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="workspace not found")

    def vscode_enabled(request: Request) -> bool:
        return (
            vscode is not None
            and bool(sharing.password)
            and request.scope.get("pycodex.access_role") == "admin"
            and not request.scope.get("pycodex.agent_access")
        )

    def require_vscode_http(request: Request) -> WorkspaceRouteContext:
        context = workspace(request)
        if vscode is None or not sharing.password:
            raise HTTPException(
                status_code=403, detail="VS Code requires a password-protected workspace"
            )
        if request.scope.get("pycodex.access_role") != "admin" or request.scope.get("pycodex.agent_access"):
            raise HTTPException(status_code=403, detail="administrator access required")
        return context

    def actual_workspace_prefix(context: WorkspaceRouteContext) -> str:
        return "/w/{0}".format(context.workspace_id) if prefix and context.workspace_id else ""

    async def index(request: Request) -> HTMLResponse:
        context = workspace(request)
        return _html_response(_render_workspace_shell(
            context.board_path, title=context.title, work_dir=context.work_dir,
            builtin=request.query_params.get("builtin") == "1",
            access_role=request.scope["pycodex.access_role"],
            enable_vscode=vscode_enabled(request),
        ))

    async def favicon(request: Request) -> Response:
        return Response(status_code=204)

    async def service_worker(request: Request) -> Response:
        workspace(request)
        return Response(
            b"" if request.method == "HEAD" else _NOTIFICATION_SW_JS.encode("utf-8"),
            media_type="text/javascript",
            headers={
                "Cache-Control": "no-cache",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def web_manifest(request: Request) -> Response:
        title = workspace(request).title or "pycodex"
        body = json.dumps(
            {
                "name": title,
                "short_name": title,
                "start_url": "./",
                "scope": "./",
                "display": "standalone",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        return Response(
            b"" if request.method == "HEAD" else body,
            media_type="application/manifest+json",
            headers={
                "Cache-Control": "no-cache",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def board(request: Request) -> Response:
        return _board_response(workspace(request).board_path)

    async def board_status(request: Request) -> JSONResponse:
        return _board_status_response(workspace(request).board_path)

    def views(request: Request) -> JSONResponse:
        return _views_response(workspace(request).manager)

    async def open_view(request: Request) -> JSONResponse:
        payload = await _json_object_body(request)
        context = workspace(request)
        return await run_in_threadpool(
            _open_view_response, context.manager, context.work_dir, payload
        )

    def view(request: Request) -> JSONResponse:
        return _view_response(workspace(request).manager, request.path_params["view_id"])

    def view_pdf(request: Request) -> Response:
        return _view_pdf_response(workspace(request).manager, request)

    def view_image(request: Request) -> Response:
        try:
            view = workspace(request).manager.views.get(request.path_params["view_id"])
        except KeyError:
            raise HTTPException(status_code=404, detail="view not found")
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error))
        document = view["documents"][0]
        if view["kind"] != "file" or "source" not in document:
            raise HTTPException(status_code=404, detail="local Markdown view required")
        path = Path(document["source"]["path"])
        if path.suffix.lower() not in (".md", ".markdown"):
            raise HTTPException(status_code=404, detail="local Markdown view required")
        return _image_asset_response(path, request.path_params["asset_path"])

    def local_image(request: Request) -> Response:
        manager = workspace(request).manager
        path = Path(request.query_params.get("path", ""))
        if not path.is_absolute() or path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            raise HTTPException(status_code=404, detail="local raster image required")
        try:
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("expected a regular image file")
            with path.open("rb") as handle:
                data = handle.read(manager.views.max_file_bytes + 1)
            if len(data) > manager.views.max_file_bytes:
                raise ValueError("image exceeds 10 MiB")
            media_type = raster_image_type(data, str(path))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="image not found")
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            raise HTTPException(status_code=400, detail="invalid local raster image")
        return Response(b"" if request.method == "HEAD" else data, media_type=media_type, headers={
            "Content-Length": str(len(data)),
            "Cache-Control": "private, no-cache",
            "X-Content-Type-Options": "nosniff",
        })

    def close_view(request: Request) -> JSONResponse:
        return _close_view_response(workspace(request).manager, request.path_params["view_id"])

    async def open_vscode_link(request: Request) -> JSONResponse:
        context = require_vscode_http(request)
        assert vscode is not None
        payload = await _json_object_body(request)
        window_id = payload.get("window_id")
        if not isinstance(window_id, str) or not window_id:
            raise HTTPException(status_code=400, detail="window_id must be a nonempty string")
        link_cwd = context.work_dir
        session_id = payload.get("session_id")
        if session_id is not None:
            if not isinstance(session_id, str) or not session_id:
                raise HTTPException(status_code=400, detail="session_id must be a nonempty string")
            try:
                link_cwd = Path(context.manager.get(session_id).summary()["cwd"])
            except KeyError:
                raise HTTPException(status_code=404, detail="session not found")
        try:
            spec = await run_in_threadpool(
                resolve_vscode_link, link_cwd, payload.get("href"), payload.get("src_path")
            )
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            raise HTTPException(status_code=400, detail=str(error))
        try:
            await vscode.open_link(window_id, spec)
        except (FileNotFoundError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            raise HTTPException(status_code=503, detail=str(error))
        return JSONResponse({"ok": True, "target": spec}, headers={"Cache-Control": "no-store"})

    def _require_vscode_window_id(request: Request) -> str:
        window_id = request.path_params["window_id"]
        if not window_id or any(ord(ch) < 32 or ord(ch) == 127 for ch in window_id):
            raise HTTPException(status_code=400, detail="invalid VS Code window ID")
        return window_id

    async def vscode_root(request: Request) -> Response:
        context = require_vscode_http(request)
        assert vscode is not None
        actual_prefix = actual_workspace_prefix(context)
        if request.method in ("GET", "HEAD"):
            query = ("?" + request.url.query) if request.url.query else ""
            return RedirectResponse(
                url=forwarded_vscode_prefix(
                    request.headers.get, request.scope.get("root_path", ""), actual_prefix
                )
                + "/"
                + query,
                status_code=307,
            )
        return await vscode.proxy_http(request, actual_prefix)

    async def vscode_window_root(request: Request) -> Response:
        context = require_vscode_http(request)
        assert vscode is not None
        window_id = _require_vscode_window_id(request)
        actual_prefix = actual_workspace_prefix(context)
        vscode_prefix = forwarded_vscode_prefix(
            request.headers.get, request.scope.get("root_path", ""), actual_prefix
        )
        raw_query = request.scope.get("query_string", b"").decode("latin-1")
        if request.method in ("GET", "HEAD"):
            if request.headers.get("sec-fetch-dest", "").strip().lower() == "document":
                clean_query = strip_vscode_window_payload(raw_query)
                suffix = ("?" + clean_query) if clean_query else ""
                return RedirectResponse(
                    url=vscode_prefix + "/" + suffix,
                    status_code=307,
                    headers={"Cache-Control": "no-store"},
                )
            suffix = ("?" + raw_query) if raw_query else ""
            return RedirectResponse(
                url=vscode_prefix + "/window/" + quote(window_id, safe="") + "/" + suffix,
                status_code=307,
                headers={"Cache-Control": "no-store"},
            )
        return await vscode.proxy_http(request, actual_prefix, upstream_path="/")

    async def vscode_window(request: Request) -> Response:
        context = require_vscode_http(request)
        assert vscode is not None
        window_id = _require_vscode_window_id(request)
        actual_prefix = actual_workspace_prefix(context)
        vscode_prefix = forwarded_vscode_prefix(
            request.headers.get, request.scope.get("root_path", ""), actual_prefix
        )
        raw_query = request.scope.get("query_string", b"").decode("latin-1")
        if request.method in ("GET", "HEAD"):
            dest = request.headers.get("sec-fetch-dest", "").strip().lower()
            if dest == "document":
                clean_query = strip_vscode_window_payload(raw_query)
                suffix = ("?" + clean_query) if clean_query else ""
                return RedirectResponse(
                    url=vscode_prefix + "/" + suffix,
                    status_code=307,
                    headers={"Cache-Control": "no-store"},
                )
            if request.method == "GET" and dest in ("", "iframe"):
                vscode.supersede_window(window_id)
            if extract_vscode_window_id(raw_query) != window_id:
                updated_query = inject_vscode_window_payload(raw_query, window_id)
                return RedirectResponse(
                    url=vscode_prefix + "/window/" + quote(window_id, safe="") + "/?" + updated_query,
                    status_code=307,
                    headers={"Cache-Control": "no-store"},
                )
        return await vscode.proxy_http(request, actual_prefix, upstream_path="/")

    async def vscode_proxy(request: Request) -> Response:
        context = require_vscode_http(request)
        assert vscode is not None
        return await vscode.proxy_http(request, actual_workspace_prefix(context))

    async def websocket_vscode(websocket: WebSocket) -> None:
        try:
            context = resolve_workspace(websocket)
        except (KeyError, ValueError):
            await websocket.close(code=1008)
            return
        if (
            vscode is None
            or not sharing.password
            or websocket.scope.get("pycodex.agent_access")
            or not sharing.is_admin(websocket.cookies.get(AUTH_COOKIE_NAME))
            or not websocket_origin_allowed(websocket)
        ):
            await websocket.close(code=1008)
            return
        await vscode.proxy_websocket(websocket, actual_workspace_prefix(context))

    def view_asset(request: Request) -> Response:
        workspace(request)
        name = request.url.path.rsplit("/", 1)[-1]
        return _view_asset_response(name, asset_types[name], request, view_asset_cache)

    async def websocket_backend_hint(request: Request) -> JSONResponse:
        workspace(request)
        return _websocket_backend_hint_response()

    def skills(request: Request) -> JSONResponse:
        context = workspace(request)
        return JSONResponse(
            {"skills": list_skills(context.manager.codex_home)},
            headers={"Cache-Control": "no-store"},
        )

    async def sessions(request: Request) -> JSONResponse:
        return _sessions_response(workspace(request).manager, request)

    async def peer_nodes(request: Request) -> JSONResponse:
        workspace(request)
        return await _peer_nodes_response(request)

    async def peer_node_profiles(request: Request) -> JSONResponse:
        workspace(request)
        return await _peer_node_profiles_response(
            request, request.path_params["node_id"]
        )

    async def new_session(request: Request) -> JSONResponse:
        return await _new_session_response(workspace(request).manager, request)

    def frontend_asset(request: Request) -> Response:
        workspace(request)
        return frontend_asset_response(request.path_params["asset_path"])

    async def delete_session(request: Request) -> JSONResponse:
        return await _delete_session_response(
            workspace(request).manager, request.path_params["session_id"]
        )

    async def read_session(request: Request) -> JSONResponse:
        payload = await _json_object_body(request)
        return _read_session_response(
            workspace(request).manager,
            request.path_params["session_id"],
            payload,
            request,
        )

    async def session(request: Request) -> JSONResponse:
        context = workspace(request)
        return _session_response(context, request)

    async def root_agent_help(request: Request) -> Response:
        return _agent_help_response(request, multi=bool(prefix))

    async def agent_help(request: Request) -> Response:
        context = workspace(request)
        return _agent_help_response(
            request,
            multi=bool(prefix),
            manager=context.manager,
            workspace_id=context.workspace_id,
        )

    async def message(request: Request) -> JSONResponse:
        payload = await _json_object_body(request)
        return await _message_response(workspace(request).manager, payload)

    async def websocket_session(websocket: WebSocket) -> None:
        try:
            context = resolve_workspace(websocket)
        except (KeyError, ValueError):
            await websocket.close(code=1008)
            return
        await sharing.serve_websocket(
            websocket, lambda connection: _websocket_session_handler(context.manager, connection),
        )

    async def board_asset(request: Request) -> Response:
        return _image_asset_response(
            workspace(request).board_path, request.path_params["asset_path"]
        )

    def route(path, endpoint, methods):
        name = ("workspace_" if prefix else "") + endpoint.__name__
        return Route(prefix + path, endpoint, methods=methods, name=name)

    return session_api_routes(
        lambda request: workspace(request).manager, _json_object_body, prefix,
    ) + external_call.external_call_routes(
        workspace, prefix,
    ) + list(leading_routes) + [
        *([Route("/api/help", root_agent_help, methods=["GET"])] if prefix else []),
        route("/", index, ["GET"]),
        *([] if prefix else [route("/favicon.ico", favicon, ["GET"])]),
        route("/sw.js", service_worker, ["GET", "HEAD"]),
        route("/manifest.webmanifest", web_manifest, ["GET", "HEAD"]),
        route("/board", board, ["GET", "HEAD"]),
        route("/api/board", board_status, ["GET"]),
        route("/api/views", views, ["GET"]),
        route("/api/views/open", open_view, ["POST"]),
        route("/api/images", local_image, ["GET", "HEAD"]),
        route("/api/views/{view_id}", view, ["GET"]),
        route("/api/views/{view_id}/pdf", view_pdf, ["GET", "HEAD"]),
        route("/api/views/{view_id}/images/{asset_path:path}", view_image, ["GET", "HEAD"]),
        route("/api/views/{view_id}", close_view, ["DELETE"]),
        route("/api/vscode/open", open_vscode_link, ["POST"]),
        route("/vscode", vscode_root, vscode_methods),
        route("/vscode/window/{window_id}", vscode_window_root, vscode_methods),
        route("/vscode/window/{window_id}/", vscode_window, vscode_methods),
        route("/vscode/{vscode_path:path}", vscode_proxy, vscode_methods),
        WebSocketRoute(
            prefix + "/vscode", websocket_vscode,
            name=("workspace_" if prefix else "") + "websocket_vscode_root",
        ),
        WebSocketRoute(
            prefix + "/vscode/{vscode_path:path}", websocket_vscode,
            name=("workspace_" if prefix else "") + "websocket_vscode",
        ),
        *[
            route("/assets/" + asset, view_asset, ["GET", "HEAD"])
            for asset in asset_types
        ],
        route("/ws/session", websocket_backend_hint, ["GET"]),
        route("/api/skills", skills, ["GET"]),
        route("/api/sessions", sessions, ["GET"]),
        route("/api/sessions", new_session, ["POST"]),
        route("/api/peer-nodes", peer_nodes, ["GET"]),
        route("/api/peer-nodes/{node_id}/profiles", peer_node_profiles, ["GET"]),
        route("/frontend/{asset_path:path}", frontend_asset, ["GET", "HEAD"]),
        route("/api/sessions/{session_id}", delete_session, ["DELETE"]),
        route("/api/sessions/{session_id}/read", read_session, ["POST"]),
        route("/api/help", agent_help, ["GET"]),
        route("/api/session", session, ["GET"]),
        route("/api/session/message", message, ["POST"]),
        WebSocketRoute(
            prefix + "/ws/session", websocket_session,
            name=("workspace_" if prefix else "") + "websocket_session",
        ),
        route("/{asset_path:path}", board_asset, ["GET", "HEAD"]),
    ]


def _views_response(manager: WorkspaceSessionManager) -> JSONResponse:
    try:
        views = manager.views.list_views()
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    return JSONResponse({"views": views}, headers={"Cache-Control": "no-store"})


def _open_view_response(
    manager: WorkspaceSessionManager, work_dir: Path, payload: "typing.Dict[str, object]"
) -> JSONResponse:
    try:
        session_id = payload.get("session_id")
        if session_id is not None:
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("session_id must be a nonempty string")
            work_dir = Path(manager.get(session_id).summary()["cwd"])
        view = WorkspaceViewOpener(manager.views, work_dir).open(
            payload.get("href"), payload.get("src_path")
        )
        views = manager.views.list_views()
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise HTTPException(status_code=400, detail=str(error))
    return JSONResponse(
        {"view": view, "views": views}, headers={"Cache-Control": "no-store"}
    )


def _view_response(manager: WorkspaceSessionManager, view_id: str) -> JSONResponse:
    try:
        view = manager.views.get(view_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="view not found")
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    return JSONResponse(view, headers={"Cache-Control": "no-store"})


def _view_pdf_response(manager: WorkspaceSessionManager, request: Request) -> Response:
    try:
        view = manager.views.get(request.path_params["view_id"])
    except KeyError:
        raise HTTPException(status_code=404, detail="view not found")
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    document = view["documents"][0]
    if view["kind"] != "file" or document.get("media_type") != "application/pdf":
        raise HTTPException(status_code=400, detail="view is not a local PDF")
    if document.get("error"):
        raise HTTPException(
            status_code=404 if document.get("missing") else 400,
            detail=document["error"],
        )
    disposition = "attachment" if request.query_params.get("download") == "true" else "inline"
    filename = quote(Path(document["label"]).name, safe="")
    response = FileResponse(
        document["source"]["path"],
        media_type="application/pdf",
        stat_result=Path(document["source"]["path"]).stat(),
        headers={
            "Content-Disposition": "{0}; filename*=utf-8''{1}".format(disposition, filename),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
    if request.method == "HEAD":
        return Response(headers=response.headers)
    return response


def _close_view_response(
    manager: WorkspaceSessionManager, view_id: str
) -> JSONResponse:
    try:
        manager.views.close(view_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="view not found")
    return JSONResponse({"ok": True})


def _view_asset_encoding(accept_encoding: str) -> str:
    qualities = {}
    for entry in accept_encoding.lower().split(","):
        name, *parameters = entry.strip().split(";")
        quality = 1.0
        for parameter in parameters:
            key, _, value = parameter.strip().partition("=")
            if key != "q":
                continue
            try:
                quality = float(value)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400, detail="invalid encoding quality"
                ) from exc
            if not 0 <= quality <= 1:
                raise HTTPException(status_code=400, detail="invalid encoding quality")
        qualities[name.strip()] = quality
    gzip_quality = qualities.get("gzip", qualities.get("*", 0))
    if gzip_quality > 0 and gzip_quality >= qualities.get("identity", 0):
        return "gzip"
    if qualities.get("identity", 0 if qualities.get("*") == 0 else 1) > 0:
        return "identity"
    raise HTTPException(status_code=406, detail="no acceptable asset encoding")


def _view_asset_response(asset_name: str, media_type: str, request: Request, cache: dict) -> Response:
    encoding = _view_asset_encoding(request.headers.get("accept-encoding", ""))
    path = Path(__file__).with_name(asset_name)
    stat = path.stat()
    stamp = (stat.st_mtime_ns, stat.st_size)
    key = (asset_name, encoding)
    cached = cache.get(key)
    if cached is None or cached[0] != stamp:
        body = path.read_bytes()
        if encoding == "gzip":
            body = gzip.compress(body, compresslevel=6)
        etag = '"' + hashlib.sha256(body).hexdigest() + '"'
        cached = cache[key] = (stamp, body, etag)
    _, body, etag = cached
    headers = {
        "Cache-Control": "no-cache",
        "X-Content-Type-Options": "nosniff",
        "Vary": "Accept-Encoding",
        "ETag": etag,
    }
    if encoding == "gzip":
        headers["Content-Encoding"] = "gzip"
    for candidate in request.headers.get("if-none-match", "").split(","):
        candidate = candidate.strip()
        if candidate.startswith("W/"):
            candidate = candidate[2:]
        if candidate in ("*", etag):
            return Response(status_code=304, headers=headers)
    headers["Content-Length"] = str(len(body))
    return Response(
        b"" if request.method == "HEAD" else body,
        media_type=media_type, headers=headers,
    )


def _board_status(board_path: Path) -> "typing.Dict[str, object]":
    if not board_path.is_file():
        return {"exists": False}
    stat = board_path.stat()
    return {
        "exists": True,
        "path": str(board_path),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def _board_status_response(board_path: "typing.Union[Path, None]") -> JSONResponse:
    if board_path is None:
        raise HTTPException(status_code=404, detail="Canvas is disabled")
    return JSONResponse(_board_status(board_path))


def _websocket_backend_hint_response() -> JSONResponse:
    return JSONResponse(
        {
            "error": "websocket backend is unavailable; HTTP polling is active",
        },
        status_code=426,
    )


def _sessions_response(
    manager: "WorkspaceSessionManager",
    request: "typing.Optional[Request]" = None,
) -> JSONResponse:
    router_client = (
        getattr(request.app.state, "router_client", None)
        if request is not None
        else None
    )
    router_connected = bool(router_client is not None and router_client.connected)
    local_profile_error: "typing.Optional[str]" = None
    try:
        manager.refresh_profiles()
        profiles = list(manager.profile_factories)
        default_profile = manager.default_profile
    except (OSError, ValueError) as exc:
        if not router_connected:
            raise HTTPException(status_code=400, detail=str(exc))
        profiles = []
        default_profile = None
        local_profile_error = str(exc)
    payload: "typing.Dict[str, object]" = {
        "sessions": manager.list_sessions(),
        "profiles": profiles,
        "default_profile": default_profile,
    }
    if local_profile_error is not None:
        payload["local_profile_error"] = local_profile_error
    if router_connected and not (
        request is not None and request.scope.get("pycodex.agent_access")
    ):
        payload["router_connected"] = True
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


async def _peer_nodes_response(request: Request) -> JSONResponse:
    if request.scope.get("pycodex.agent_access"):
        raise HTTPException(status_code=403, detail="browser session access required")
    router_client = getattr(request.app.state, "router_client", None)
    if router_client is None or not router_client.connected:
        return JSONResponse(
            {"error": "router tunnel is not connected"},
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )
    try:
        nodes = await call_on_router_loop(
            router_client, lambda: peer_list_nodes(router_client)
        )
    except (OSError, ValueError) as exc:
        return JSONResponse(
            {"error": str(exc)},
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(
        {"peer_nodes": nodes}, headers={"Cache-Control": "no-store"}
    )


async def _peer_node_profiles_response(
    request: Request, node_id: str
) -> JSONResponse:
    if not PEER_NODE_ID_RE.fullmatch(node_id):
        raise HTTPException(status_code=400, detail="invalid peer node ID")
    router_client = getattr(request.app.state, "router_client", None)
    if router_client is None or not router_client.connected:
        return JSONResponse(
            {"error": "router tunnel is not connected"},
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )
    try:
        status, _headers, body = await call_on_router_loop(
            router_client,
            lambda: peer_http_request(
                router_client, node_id, "GET", "/api/peer-profiles"
            ),
        )
    except (OSError, ValueError) as exc:
        return JSONResponse(
            {"error": str(exc)},
            status_code=503,
            headers={"Cache-Control": "no-store"},
        )
    if status >= 400:
        return JSONResponse(
            {"error": _error_message_from_body(body, status)},
            status_code=status,
            headers={"Cache-Control": "no-store"},
        )
    try:
        payload = json.loads(body.decode("utf-8"))
    except ValueError:
        return JSONResponse(
            {"error": "invalid peer profiles response"},
            status_code=502,
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


def _read_session_response(
    manager: "WorkspaceSessionManager",
    session_id: str,
    payload: "typing.Dict[str, object]",
    request: "typing.Optional[Request]" = None,
) -> JSONResponse:
    try:
        manager.mark_read(session_id, payload.get("final_id"))
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _sessions_response(manager, request)


def _user_message_name(payload):
    name = payload.get("name", "user")
    if not isinstance(name, str):
        raise HTTPException(status_code=422, detail="name must be a string")
    return name


def _trigger_flag(payload):
    trigger = payload.get("trigger", True)
    if not isinstance(trigger, bool):
        raise HTTPException(status_code=422, detail="trigger must be a boolean")
    return trigger


async def _new_session_response(
    manager: "WorkspaceSessionManager", request: Request,
) -> JSONResponse:
    payload = await _json_object_body(request) if await request.body() else {}
    title = payload.get("title", "")
    prompt = payload.get("prompt", "")
    profile = payload.get("profile")
    name = _user_message_name(payload)
    trigger = _trigger_flag(payload)
    if not isinstance(title, str) or not isinstance(prompt, str):
        raise HTTPException(status_code=422, detail="title and prompt must be strings")
    if profile is not None and not isinstance(profile, str):
        raise HTTPException(status_code=422, detail="profile must be a string or null")
    try:
        session_id = await manager.create_session(title=title, profile=profile)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    result = (
        await manager.get(session_id).submit(prompt, sender="web", name=name, trigger=trigger)
        if prompt else {"ok": True}
    )
    return JSONResponse(
        {
            **result,
            "session_id": session_id,
            "sessions": manager.list_sessions(),
            "snapshot": session_snapshot(manager.get(session_id)),
        },
        status_code=200 if result["ok"] else 400,
    )


async def _delete_session_response(
    manager: "WorkspaceSessionManager",
    session_id: str,
) -> JSONResponse:
    try:
        await manager.close_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return JSONResponse({"ok": True, "sessions": manager.list_sessions()})


def _session_response(
    context: WorkspaceRouteContext,
    request: Request,
) -> JSONResponse:
    manager = context.manager
    recent_only = request.query_params.get("recent") == "true"
    session_id = request.query_params.get("session_id")
    sessions = manager.list_sessions()
    if recent_only:
        sessions.sort(key=lambda item: (item["last_user_at"], item["id"]), reverse=True)
        if not session_id and sessions:
            session_id = sessions[0]["id"]
    try:
        resolved_id = manager.resolve_session_id(session_id)
        link = manager.get(resolved_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    payload = {
        "session_id": resolved_id,
        "sessions": sessions,
        "snapshot": session_snapshot(link),
    }
    if recent_only:
        payload["workspace_running"] = any(item["running"] for item in sessions)
        payload["sessions"] = sessions[:20]
        if not any(item["id"] == resolved_id for item in payload["sessions"]):
            payload["sessions"][-1] = next(item for item in sessions if item["id"] == resolved_id)
    if (
        request.scope.get("pycodex.access_role") == "admin"
        and not request.scope.get("pycodex.agent_access")
    ):
        binding = external_call.external_call_binding(request, manager, resolved_id)
        if binding is not None:
            payload["external_call"] = binding
    if request.query_params.get("workspace") == "true":
        try:
            payload["views"] = manager.views.list_views()
        except (OSError, ValueError) as error:
            payload["views_error"] = str(error)
        payload["board"] = (
            None if context.board_path is None else _board_status(context.board_path)
        )
    if request.query_params.get("runtime_info") == "true":
        state = request.app.state
        payload["runtime_info"] = runtime_info_prompt(
            resolved_id,
            context.workspace_id,
            getattr(state, "local_api_url", None),
            getattr(state, "agent_header_path", None),
        )
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


def _agent_help_response(
    request: Request,
    multi: bool,
    manager: "typing.Union[WorkspaceSessionManager, None]" = None,
    workspace_id: "typing.Union[str, None]" = None,
) -> Response:
    session_id = request.query_params.get("session_id")
    resolved_id = None
    summary = None
    if manager is not None and session_id:
        try:
            resolved_id = manager.resolve_session_id(session_id)
            summary = manager.get(resolved_id).summary()
        except KeyError:
            raise HTTPException(status_code=404, detail="session not found")
    state = request.app.state
    text = agent_help_text(
        multi,
        getattr(state, "local_api_url", None),
        getattr(state, "router_client", None),
        getattr(state, "agent_header_path", None),
        workspace_id=workspace_id,
        session_id=resolved_id,
        summary=summary,
    )
    return Response(
        text,
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


async def _message_response(
    manager: "WorkspaceSessionManager",
    payload: "typing.Dict[str, object]",
) -> JSONResponse:
    session_id = str(payload.get("session_id") or "")
    name = _user_message_name(payload)
    try:
        link = manager.get(session_id or None)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    if "request_id" in payload:
        result = await link.answer_input(payload["request_id"], payload.get("answer"))
    else:
        trigger = _trigger_flag(payload)
        result = await link.submit(
            str(payload.get("prompt") or ""),
            sender=str(payload.get("sender") or "web"),
            name=name,
            trigger=trigger,
        )
    if isinstance(result, dict):
        result.setdefault("sessions", manager.list_sessions())
    status = 200 if result.get("ok") else 400
    return JSONResponse(result, status_code=status)


async def _websocket_session_handler(
    manager: "WorkspaceSessionManager",
    websocket: WebSocket,
) -> None:
    url = websocket.url
    scheme = "https" if url.scheme == "wss" else "http"
    authority = url.netloc.lower()
    if url.port == (443 if scheme == "https" else 80):
        authority = authority.rsplit(":", 1)[0]
    # Browsers send Origin; non-browser clients may omit it.
    if websocket.headers.get("origin") not in (None, scheme + "://" + authority):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    session_id = str(websocket.query_params.get("session_id") or "")
    try:
        link = manager.get(session_id or None)
    except KeyError:
        await websocket.close(code=1008)
        return
    agent_id = websocket.query_params.get("agent_id")
    try:
        view = await link.event_view(agent_id)
        subscriber = view.subscribe(websocket.query_params.get("cursor"))
    except (KeyError, ValueError) as error:
        await websocket.send_json({"type": "error", "error": str(error)})
        await websocket.close(code=1008)
        return
    sender = asyncio.create_task(_send_ws_events(websocket, subscriber))
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
            except ValueError:
                await websocket.send_json({"type": "error", "error": "invalid json"})
                continue
            action = str(payload.get("type") or payload.get("action") or "")
            if action in {"send", "answer"}:
                if agent_id is not None:
                    await websocket.send_json({
                        "type": "error", "error": "child-agent subscriptions are read-only",
                    })
                    continue
                target_session_id = str(payload.get("session_id") or session_id or "")
                try:
                    target_link = manager.get(target_session_id or None)
                except KeyError:
                    await websocket.send_json(
                        {"type": "error", "error": "session not found"}
                    )
                    continue
                if action == "answer":
                    result = await target_link.answer_input(
                        payload.get("request_id"),
                        payload.get("answer"),
                    )
                else:
                    try:
                        name = _user_message_name(payload)
                        trigger = _trigger_flag(payload)
                    except HTTPException as error:
                        await websocket.send_json({
                            "type": "send_result", "result": {"ok": False, "error": error.detail},
                        })
                        continue
                    result = await target_link.submit(
                        str(payload.get("prompt") or ""),
                        sender=str(payload.get("sender") or "web"),
                        name=name,
                        trigger=trigger,
                    )
                await websocket.send_json({"type": "send_result", "result": result})
            elif action == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                await websocket.send_json({"type": "error", "error": "unknown action"})
    except WebSocketDisconnect:
        pass
    finally:
        view.unsubscribe(subscriber)
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)


async def _send_ws_events(websocket: WebSocket, subscriber: "asyncio.Queue") -> None:
    while True:
        event = await subscriber.get()
        if event is None:
            return
        await websocket.send_json(event)


def _board_response(board_path: "typing.Union[Path, None]") -> Response:
    if board_path is None:
        raise HTTPException(status_code=404, detail="Canvas is disabled")
    if not board_path.is_file():
        return _html_response(_render_missing_board(board_path))
    return _html_response(board_path.read_text(encoding="utf-8", errors="replace"))


def _image_asset_response(
    document_path: "typing.Union[Path, None]",
    asset_path: str,
) -> Response:
    media_type, unused_encoding = mimetypes.guess_type(str(asset_path or ""))
    del unused_encoding
    if document_path is None or not media_type or not media_type.startswith("image/"):
        raise HTTPException(status_code=404, detail="image not found")

    try:
        directory = document_path.parent.resolve()
        resolved_asset = (directory / asset_path).resolve()
        resolved_asset.relative_to(directory)
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(status_code=404, detail="image not found")
    if not resolved_asset.is_file():
        raise HTTPException(status_code=404, detail="image not found")

    return FileResponse(
        str(resolved_asset),
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _render_workspace_shell(
    board_path: "typing.Union[Path, None]",
    title: "typing.Union[str, None]" = None,
    work_dir: "typing.Union[Path, None]" = None,
    builtin: bool = False,
    access_role: str = "admin",
    enable_vscode: bool = False,
) -> str:
    board_label = str(board_path) if board_path is not None else "No board"
    cwd_label = str(work_dir or Path.cwd())
    page_title = str(title or "pycodex workspace")
    template = load_workspace_template(builtin)
    return (
        template.replace("__WORKSPACE_TITLE__", html.escape(page_title))
        .replace("__ACCESS_ROLE__", access_role)
        .replace("__BOARD_LABEL__", html.escape(board_label))
        .replace("__CWD_LABEL__", html.escape(cwd_label))
        .replace("__ENABLE_BOARD__", "true" if board_path is not None else "false")
        .replace("__ENABLE_VSCODE__", "true" if enable_vscode else "false")
        .replace("__BOARD_URL__", "board" if board_path is not None else "about:blank")
        .replace("__FRONTEND_DIRECTORY__", html.escape(str(frontend_directory())))
        .replace("__PACKAGE_DIRECTORY__", html.escape(str(Path(__file__).resolve().parent)))
        .replace("__PACKAGE_VERSION__", html.escape(importlib_metadata.version("pycodex-ws")))
    )


def _render_workspaces_manager_shell() -> str:
    return (Path(__file__).with_name("workspaces.html")).read_text(encoding="utf-8")


def _render_missing_board(board_path: Path) -> str:
    return _render_board_placeholder(
        "Canvas pending",
        "Your work will appear here",
        "Ask pycodex to create a page, a report, or a visual. "
        "This canvas updates as you work.",
        str(board_path),
    )


def _render_board_placeholder(
    title: str, heading: str, description: str, path_label: str = ""
) -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{0}</title><style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; min-height: 100dvh; display: grid; place-items: center;
  padding: 32px; background: #f8fbfd; color: #172630;
  font: 14px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width: 360px; text-align: center; }}
svg {{ width: 40px; height: 40px; color: #aebfc9; margin-bottom: 20px; }}
h1 {{ margin: 0 0 12px; font-size: 23px; font-weight: 500; line-height: 1.35;
  letter-spacing: -0.5px; }}
p {{ margin: 0; color: #60727d; }}
code {{ display: block; margin-top: 28px; font-size: 11px; color: #60727d;
  overflow-wrap: anywhere; }}
</style></head><body><main>
<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.25"
aria-hidden="true"><rect x="3" y="3" width="26" height="26" rx="5"/>
<path d="M3 11h26M11 11v18"/></svg>
<h1>{1}</h1><p>{2}</p><code>{3}</code>
</main></body></html>""".format(
        html.escape(title),
        html.escape(heading),
        html.escape(description),
        html.escape(path_label),
    )
