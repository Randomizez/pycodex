"""Shared workspace routes and HTTP/WebSocket response adapters."""

import asyncio
import gzip
import hashlib
import html
import json
import mimetypes
import os
import subprocess
import typing
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from pycodex.utils.compat import importlib_metadata

from .customization import frontend_asset_response, frontend_directory, load_workspace_template
from .runtime_info import runtime_info_prompt
from .session_api import session_api_routes
from .workspace_views import WorkspaceViewOpener
from .workspaces import WorkspaceSessionManager, session_snapshot


@dataclass(frozen=True)
class WorkspaceRouteContext:
    manager: "WorkspaceSessionManager"
    board_path: "typing.Optional[Path]"
    work_dir: "Path"
    title: "typing.Optional[str]" = None
    workspace_id: "typing.Optional[str]" = None


def workspace_routes(resolve_workspace, sharing, prefix="", leading_routes=()):
    view_asset_cache = {}

    def workspace(request):
        try:
            return resolve_workspace(request)
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="workspace not found")

    async def index(request: Request) -> HTMLResponse:
        context = workspace(request)
        return _html_response(_render_workspace_shell(
            context.board_path, title=context.title, work_dir=context.work_dir,
            builtin=request.query_params.get("builtin") == "1",
            access_role=request.scope["pycodex.access_role"],
        ))

    async def favicon(request: Request) -> Response:
        return Response(status_code=204)

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

    def close_view(request: Request) -> JSONResponse:
        return _close_view_response(workspace(request).manager, request.path_params["view_id"])

    def view_asset(request: Request) -> Response:
        workspace(request)
        return _view_asset_response(
            request.url.path.rsplit("/", 1)[-1], request, view_asset_cache
        )

    async def websocket_backend_hint(request: Request) -> JSONResponse:
        workspace(request)
        return _websocket_backend_hint_response()

    async def sessions(request: Request) -> JSONResponse:
        return _sessions_response(workspace(request).manager)

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
            workspace(request).manager, request.path_params["session_id"], payload
        )

    async def session(request: Request) -> JSONResponse:
        context = workspace(request)
        return _session_response(context.manager, request, context.workspace_id)

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
        return _board_asset_response(
            workspace(request).board_path, request.path_params["asset_path"]
        )

    def route(path, endpoint, methods):
        name = ("workspace_" if prefix else "") + endpoint.__name__
        return Route(prefix + path, endpoint, methods=methods, name=name)

    return session_api_routes(
        lambda request: workspace(request).manager, _json_object_body, prefix,
    ) + list(leading_routes) + [
        route("/", index, ["GET"]),
        *([] if prefix else [route("/favicon.ico", favicon, ["GET"])]),
        route("/board", board, ["GET", "HEAD"]),
        route("/api/board", board_status, ["GET"]),
        route("/api/views", views, ["GET"]),
        route("/api/views/open", open_view, ["POST"]),
        route("/api/views/{view_id}", view, ["GET"]),
        route("/api/views/{view_id}/pdf", view_pdf, ["GET", "HEAD"]),
        route("/api/views/{view_id}", close_view, ["DELETE"]),
        *[
            route("/assets/" + asset, view_asset, ["GET", "HEAD"])
            for asset in (
                "content_views.mjs", "content_views.css", "content_views.LICENSE.txt",
                "editor.mjs", "editor.css",
            )
        ],
        route("/ws/session", websocket_backend_hint, ["GET"]),
        route("/api/sessions", sessions, ["GET"]),
        route("/api/sessions", new_session, ["POST"]),
        route("/frontend/{asset_path:path}", frontend_asset, ["GET", "HEAD"]),
        route("/api/sessions/{session_id}", delete_session, ["DELETE"]),
        route("/api/sessions/{session_id}/read", read_session, ["POST"]),
        route("/api/session", session, ["GET"]),
        route("/api/session/message", message, ["POST"]),
        WebSocketRoute(
            prefix + "/ws/session", websocket_session,
            name=("workspace_" if prefix else "") + "websocket_session",
        ),
        route("/{asset_path:path}", board_asset, ["GET", "HEAD"]),
    ]


def _install_workspace_routes(
    app: Starlette,
    manager: WorkspaceSessionManager,
    board_path: "typing.Union[Path, None]",
    work_dir: Path,
) -> None:
    context = WorkspaceRouteContext(manager, board_path, work_dir)
    app.routes.extend(workspace_routes(
        lambda request: context, app.state.workspace_sharing,
    ))


async def _json_object_body(
    request: Request, max_bytes: "typing.Union[int, None]" = None,
) -> "typing.Dict[str, object]":
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json" and not (
        media_type.startswith("application/") and media_type.endswith("+json")
    ):
        raise HTTPException(status_code=422, detail="request body must be JSON")
    body = bytearray()
    async for chunk in request.stream():
        if max_bytes is not None and len(body) + len(chunk) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail="request body exceeds {0} bytes".format(max_bytes),
                headers={"Connection": "close"},
            )
        body.extend(chunk)
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=422, detail="invalid JSON body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="request body must be a JSON object")
    return payload


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


def _view_asset_response(asset_name: str, request: Request, cache: dict) -> Response:
    media_types = {
        "content_views.mjs": "text/javascript",
        "content_views.css": "text/css",
        "content_views.LICENSE.txt": "text/plain",
        "editor.mjs": "text/javascript",
        "editor.css": "text/css",
    }
    if asset_name not in media_types:
        raise HTTPException(status_code=404, detail="asset not found")
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
        media_type=media_types[asset_name], headers=headers,
    )


def _board_status_response(board_path: "typing.Union[Path, None]") -> JSONResponse:
    if board_path is None:
        raise HTTPException(status_code=404, detail="Canvas is disabled")
    if not board_path.is_file():
        return JSONResponse({"exists": False})
    stat = board_path.stat()
    return JSONResponse(
        {
            "exists": True,
            "path": str(board_path),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }
    )


def _websocket_backend_hint_response() -> JSONResponse:
    return JSONResponse(
        {
            "error": "websocket backend is unavailable; HTTP polling is active",
        },
        status_code=426,
    )


def _sessions_response(manager: "WorkspaceSessionManager") -> JSONResponse:
    return JSONResponse({
        "sessions": manager.list_sessions(),
        "profiles": list(manager.profile_factories),
        "default_profile": manager.default_profile,
    })


def _read_session_response(
    manager: "WorkspaceSessionManager", session_id: str, payload: "typing.Dict[str, object]"
) -> JSONResponse:
    try:
        manager.mark_read(session_id, payload.get("final_id"))
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _sessions_response(manager)


async def _new_session_response(
    manager: "WorkspaceSessionManager", request: Request,
) -> JSONResponse:
    payload = await _json_object_body(request) if await request.body() else {}
    title = payload.get("title", "")
    prompt = payload.get("prompt", "")
    profile = payload.get("profile")
    if not isinstance(title, str) or not isinstance(prompt, str):
        raise HTTPException(status_code=422, detail="title and prompt must be strings")
    if profile is not None and not isinstance(profile, str):
        raise HTTPException(status_code=422, detail="profile must be a string or null")
    if profile is not None and profile not in manager.profile_factories:
        raise HTTPException(status_code=400, detail="unknown Codex profile: " + profile)
    session_id = await manager.create_session(title=title, profile=profile)
    result = (
        await manager.get(session_id).submit(prompt, sender="web")
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
    manager: "WorkspaceSessionManager",
    request: Request,
    workspace_id: "typing.Union[str, None]" = None,
) -> JSONResponse:
    try:
        resolved_id = manager.resolve_session_id(request.query_params.get("session_id"))
        link = manager.get(resolved_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    payload = {
        "session_id": resolved_id,
        "sessions": manager.list_sessions(),
        "snapshot": session_snapshot(link),
    }
    if request.query_params.get("runtime_info") == "true":
        state = request.app.state
        payload["runtime_info"] = runtime_info_prompt(
            resolved_id, link.summary(), workspace_id,
            getattr(state, "local_api_url", None),
            getattr(state, "router_client", None),
            getattr(state, "agent_socket_path", None),
        )
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


async def _message_response(
    manager: "WorkspaceSessionManager",
    payload: "typing.Dict[str, object]",
) -> JSONResponse:
    session_id = str(payload.get("session_id") or "")
    try:
        link = manager.get(session_id or None)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    if "request_id" in payload:
        result = await link.answer_input(payload["request_id"], payload.get("answer"))
    else:
        result = await link.submit(
            str(payload.get("prompt") or ""),
            sender=str(payload.get("sender") or "web"),
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
                    result = await target_link.submit(
                        str(payload.get("prompt") or ""),
                        sender=str(payload.get("sender") or "web"),
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


def _html_response(content: str) -> HTMLResponse:
    return HTMLResponse(
        content,
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
        },
    )


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


def _board_asset_response(
    board_path: "typing.Union[Path, None]",
    asset_path: str,
) -> Response:
    media_type, unused_encoding = mimetypes.guess_type(str(asset_path or ""))
    del unused_encoding
    if board_path is None or not media_type or not media_type.startswith("image/"):
        raise HTTPException(status_code=404, detail="board image not found")

    resolved_asset = None
    try:
        board_directory = board_path.parent.resolve()
        resolved_asset = (board_directory / asset_path).resolve()
        within_board_directory = os.path.commonpath(
            [str(board_directory), str(resolved_asset)]
        ) == str(board_directory)
    except (OSError, RuntimeError, ValueError):
        within_board_directory = False

    if (
        not within_board_directory
        or resolved_asset is None
        or not resolved_asset.is_file()
    ):
        raise HTTPException(status_code=404, detail="board image not found")

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
