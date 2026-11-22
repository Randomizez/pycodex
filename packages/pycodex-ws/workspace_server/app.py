import argparse
import asyncio
import os
import secrets
import sys
import threading
from functools import partial

from contextlib import asynccontextmanager
import typing
from pathlib import Path

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    RedirectResponse,
    Response,
)
from starlette.routing import Route

from pycodex.bootstrap import configure_loguru
from pycodex.config import DEFAULT_CODEX_CONFIG_PATH
from pycodex.utils import uuid7_string
from pycodex.utils.codex_config import list_codex_profiles, read_codex_config
from pycodex_ws.server_support import (
    AUTH_COOKIE_NAME,
    auth_cookie_matches as _auth_cookie_matches,
    html_response as _html_response,
    install_auth as _install_auth,
    json_object_body as _json_object_body,
    parse_listen,
)

from .routes import (
    WorkspaceRouteContext,
    _render_workspaces_manager_shell,
    workspace_routes,
)
from .session_view import (
    JSONValue,
    SPINNER_STATUS_PREVIEW_LIMIT,
    WebSessionView,
    _event_data,
    _json_safe,
    _last_assistant_text,
    _public_turn,
)
from .sessions import (
    SessionFactory,
    ThreadedSessionFactory,
    ThreadedWorkspaceInteractiveSession,
    WorkspaceInteractiveSession,
    build_workspace_session,
)
from .sharing import WorkspaceSharing
from .vscode_server import VsCodeRuntime
from .workspaces import (
    WorkspaceDefinition,
    WorkspaceEntry,
    WorkspaceRegistry,
    WorkspaceSessionManager,
    load_workspace_definitions,
    workspace_file_path,
)


def build_parser() -> "argparse.ArgumentParser":
    parser = argparse.ArgumentParser(
        prog="pycodex-ws",
        description="Run a local pycodex workspace with a board and chat session.",
        epilog="Stdin EOF gracefully shuts down the server after accepted work finishes.",
    )
    parser.add_argument(
        "--listen",
        default="127.0.0.1:6007",
        help="Bind address as host:port, for example 0.0.0.0:6007.",
    )
    parser.add_argument(
        "--listen-to",
        help=(
            "Connect outward with the router registration key: "
            "KEY@host[:port] (default port 6000) or ws(s)://KEY@host[:port]."
        ),
    )
    parser.add_argument(
        "--workspace-config",
        default="./workspaces.json",
        help=(
            "Optional JSON file listing workspaces. Each entry needs `state_path` "
            "and `work_dir`; `id` is optional and `enable_board` defaults to true. "
            "Routes are served under /w/<id>/."
        ),
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Optional password required to open the workspace server.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CODEX_CONFIG_PATH),
        help="Path to Codex config.toml.",
    )
    parser.add_argument(
        "--profile", default=None,
        help="Select <name>.config.toml or [profiles.<name>], overriding the config default.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional base instructions override passed to the model.",
    )
    parser.add_argument(
        "--toolset",
        nargs="*",
        default=None,
        help="Builtin tool names for all sessions. An empty list disables all tools.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="HTTP timeout for one model call.",
    )
    parser.add_argument(
        "--vllm-endpoint",
        default=None,
        help="Optional base URL for a chat-completions-backed vLLM server.",
    )
    return parser


def create_app(
    session_source: "typing.Union[WorkspaceSessionManager, SessionFactory]",
    state_path: "typing.Union[Path, None]",
    password: "typing.Union[str, None]" = None,
    work_dir: "typing.Union[Path, None]" = None,
    enable_board: bool = True,
) -> Starlette:
    manager = (
        session_source
        if isinstance(session_source, WorkspaceSessionManager)
        else WorkspaceSessionManager(session_source, state_path)
    )
    board_path = workspace_file_path(state_path, ".html") if enable_board else None
    resolved_work_dir = Path(work_dir or Path.cwd()).resolve()
    vscode = VsCodeRuntime()

    async def close() -> None:
        try:
            await manager.close()
        finally:
            await vscode.close()

    app = _create_lifespan_app(manager.start, close)
    app.state.drain = partial(manager.close, True)
    sharing = WorkspaceSharing(password, multi=False)
    sharing.vscode = vscode
    _install_auth(app, password, sharing=sharing)
    _install_sharing_routes(app, sharing)
    _install_reload_routes(
        app,
        sharing,
        lambda _connection: WorkspaceRouteContext(manager, board_path, resolved_work_dir),
        "",
        (),
        vscode=vscode,
    )
    return app


def create_multi_workspace_app(
    registry: "WorkspaceRegistry",
    password: "typing.Union[str, None]" = None,
) -> Starlette:
    vscode = VsCodeRuntime()

    async def close() -> None:
        try:
            await registry.close()
        finally:
            await vscode.close()

    app = _create_lifespan_app(registry.start, close)
    app.state.drain = partial(registry.close, True)
    app.state.router_inventory = partial(_router_inventory, registry)
    sharing = WorkspaceSharing(password, multi=True)
    sharing.vscode = vscode
    _install_auth(app, password, sharing=sharing)

    async def index(request: Request) -> Response:
        return _html_response(_render_workspaces_manager_shell())

    async def favicon(request: Request) -> Response:
        return Response(status_code=204)

    async def workspaces(request: Request) -> JSONResponse:
        return JSONResponse({"workspaces": registry.list_workspaces()})

    async def add_workspace(request: Request) -> JSONResponse:
        payload = await _json_object_body(request)
        try:
            if "board" in payload:
                raise ValueError("use `state_path` and `enable_board` instead of `board`")
            entry = await registry.add_workspace(
                str(payload.get("name") or ""),
                work_dir=str(payload.get("dir") or "./"),
                state_path=payload.get("state_path"),
                enable_board=payload.get("enable_board", True),
            )
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse(
            {
                "ok": True,
                "workspace": entry.to_dict(),
                "workspaces": registry.list_workspaces(),
            }
        )

    async def delete_workspace(request: Request) -> JSONResponse:
        try:
            sharing.revoke(request.path_params["workspace_id"])
            await registry.delete_workspace(request.path_params["workspace_id"])
        except KeyError:
            raise HTTPException(status_code=404, detail="workspace not found")
        return JSONResponse({"ok": True, "workspaces": registry.list_workspaces()})

    async def workspace_index_redirect(request: Request) -> RedirectResponse:
        workspace_id = request.path_params["workspace_id"]
        _workspace_entry_or_404(registry, workspace_id)
        return RedirectResponse(
            url=request.scope.get("root_path", "") + "/w/{0}/".format(workspace_id),
            status_code=307,
        )

    def resolve_workspace(connection):
        entry = registry.get(connection.path_params["workspace_id"])
        return WorkspaceRouteContext(
            entry.manager, entry.definition.board_path, entry.definition.work_dir,
            entry.definition.workspace_id, entry.definition.workspace_id,
        )

    leading_routes = [
        Route("/", index, methods=["GET"]),
        Route("/favicon.ico", favicon, methods=["GET"]),
        Route("/api/workspaces", workspaces, methods=["GET"]),
        Route("/api/workspaces", add_workspace, methods=["POST"]),
        Route("/api/workspaces/{workspace_id}", delete_workspace, methods=["DELETE"]),
        Route("/w/{workspace_id}", workspace_index_redirect, methods=["GET", "HEAD"]),
    ]
    _install_sharing_routes(app, sharing, registry)
    _install_reload_routes(
        app,
        sharing,
        resolve_workspace,
        "/w/{workspace_id}",
        leading_routes,
        vscode=vscode,
    )
    return app


def _router_inventory(registry):
    return [
        {
            "id": workspace["id"],
            "work_dir": workspace["work_dir"],
            "state_path": workspace["state_path"],
            "enable_board": workspace["enable_board"],
            "session_count": len(workspace["sessions"]),
            "running": any(session["running"] for session in workspace["sessions"]),
            "unread": any(session["unread"] for session in workspace["sessions"]),
        }
        for workspace in registry.list_workspaces()
    ]


def _install_sharing_routes(app, sharing, registry=None):
    prefix = "/w/{workspace_id}" if sharing.multi else ""

    async def share_settings(request):
        workspace_id = request.path_params["workspace_id"] if sharing.multi else "workspace"
        if registry is not None:
            _workspace_entry_or_404(registry, workspace_id)
        if request.scope["pycodex.access_role"] != "admin":
            return JSONResponse({"error": "administrator access required"}, 403)
        if request.method == "PUT":
            payload = await _json_object_body(request, max_bytes=4096)
            password = payload.get("password")
            if not sharing.password:
                return JSONResponse({
                    "error": "Set a workspace server administrator password before enabling sharing.",
                }, 400)
            if not isinstance(password, str) or not password:
                return JSONResponse({"error": "share password must be a nonempty string"}, 422)
            if secrets.compare_digest(password.encode("utf-8"), sharing.password.encode("utf-8")):
                return JSONResponse({"error": "share password must differ from administrator password"}, 422)
            await sharing.set_password(workspace_id, password)
        elif request.method == "DELETE":
            sharing.revoke(workspace_id)
        return JSONResponse({
            "enabled": workspace_id in sharing.grants,
            "available": bool(sharing.password),
            "url": request.scope.get("root_path", "") + (
                "/w/{0}/".format(workspace_id) if sharing.multi else "/"
            ),
        }, headers={"Cache-Control": "no-store"})

    app.routes.insert(0, Route(
        prefix + "/api/share", share_settings, methods=["GET", "PUT", "DELETE"],
    ))


def _install_reload_routes(
    app,
    sharing,
    resolve_workspace,
    prefix,
    leading_routes,
    vscode=None,
):
    async def reload_pycodex(request: Request) -> JSONResponse:
        try:
            resolve_workspace(request)
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="workspace not found")
        if request.scope["pycodex.access_role"] != "admin":
            return JSONResponse({"error": "administrator access required"}, 403)
        payload = (
            await _json_object_body(request, max_bytes=4096)
            if await request.body()
            else {}
        )
        upgrade = payload.get("upgrade", False)
        if upgrade is not True and upgrade is not False and not isinstance(upgrade, str):
            return JSONResponse({"error": "upgrade must be a boolean or version string"}, 422)
        reload = getattr(request.app.state, "reload", None)
        if reload is None:
            return JSONResponse({"error": "reload requires the pycodex-ws server process"}, 503)
        requirement = None
        if upgrade is True:
            requirement = "pycodex-ws"
        elif upgrade:
            requirement = "pycodex-ws==" + upgrade
        try:
            await reload(requirement)
        except Exception as exc:
            return JSONResponse(
                {"ok": False, "error": "{0}: {1}".format(type(exc).__name__, exc)},
                status_code=400,
                headers={"Cache-Control": "no-store"},
            )
        return JSONResponse(
            {"ok": True, "status": "reloaded"},
            headers={"Cache-Control": "no-store"},
        )

    app.routes.insert(
        0, Route(prefix + "/api/reload", reload_pycodex, methods=["POST"])
    )
    app.routes.extend(
        workspace_routes(resolve_workspace, sharing, prefix, leading_routes, vscode=vscode)
    )


async def _http_exception_response(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        {"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers
    )


def _create_lifespan_app(
    start: "typing.Callable[[], typing.Awaitable[None]]",
    close: "typing.Callable[[], typing.Awaitable[None]]",
) -> Starlette:
    @asynccontextmanager
    async def lifespan(_app):
        # Python 3.7's Unix child watcher must be initialized on the main loop
        # before a session worker starts its first asyncio subprocess.
        if (
            sys.version_info < (3, 8) and os.name == "posix"
            and threading.current_thread() is threading.main_thread()
        ):
            asyncio.get_child_watcher().attach_loop(asyncio.get_running_loop())
        await start()
        try:
            yield
        finally:
            await close()

    return Starlette(
        lifespan=lifespan,
        exception_handlers={HTTPException: _http_exception_response},
    )


def _workspace_entry_or_404(
    registry: "WorkspaceRegistry",
    workspace_id: str,
) -> "WorkspaceEntry":
    try:
        return registry.get(workspace_id)
    except (KeyError, ValueError):
        raise HTTPException(status_code=404, detail="workspace not found")


def build_app(args: "argparse.Namespace") -> Starlette:
    """Build the workspace app from persisted configuration and CLI arguments."""
    definitions = load_workspace_definitions(args.workspace_config)
    entries = [_build_workspace_entry(definition, args) for definition in definitions]
    registry = WorkspaceRegistry(
        entries,
        config_path=args.workspace_config,
        entry_factory=lambda definition, persist_callback: _build_workspace_entry(
            definition,
            args,
            persist_callback,
        ),
    )
    return create_multi_workspace_app(registry, password=args.password)


def run_serve_cli(args: "argparse.Namespace") -> int:
    import importlib

    from .owned_process import run_owned_server

    host, port = parse_listen(args.listen)
    configure_loguru()
    definitions = load_workspace_definitions(args.workspace_config)
    print(
        "pycodex workspace listening on http://{0}:{1}".format(host, port),
        flush=True,
    )
    for definition in definitions:
        print(
            "workspace {0}: state_path={1} work_dir={2} url=http://{3}:{4}/w/{0}/".format(
                definition.workspace_id,
                definition.state_path,
                definition.work_dir,
                host,
                port,
            ),
            flush=True,
        )
    run_owned_server(
        lambda: importlib.import_module("workspace_server.app").build_app(args),
        host, port, sys.stdin.fileno(), args.listen_to,
    )
    return 0


def _build_workspace_entry(
    definition: "WorkspaceDefinition",
    args: "argparse.Namespace",
    persist_callback: "typing.Union[typing.Callable[[], None], None]" = None,
) -> "WorkspaceEntry":
    def session_factory(profile) -> "ThreadedWorkspaceInteractiveSession":
        return ThreadedWorkspaceInteractiveSession(
            partial(
                build_workspace_session, args.config, profile, definition.work_dir, args.toolset,
                workspace_context=_workspace_context_text(definition.board_path) if definition.board_path else None,
                system_prompt=args.system_prompt,
                timeout_seconds=args.timeout_seconds, vllm_endpoint=args.vllm_endpoint,
            ), asyncio.get_running_loop(),
        )

    def load_profiles():
        selected = read_codex_config(args.config, args.profile)
        factories = {
            name: partial(session_factory, name)
            for name in list_codex_profiles(args.config)
        }
        return factories, selected["profile"]

    return WorkspaceEntry(
        definition=definition,
        manager=WorkspaceSessionManager(
            partial(session_factory, None),
            definition.state_path,
            persist_callback=persist_callback,
            profile_loader=load_profiles,
            codex_home=Path(args.config).expanduser().resolve().parent,
        ),
    )


def _workspace_context_text(board_path: Path) -> str:
    return (
        "Current workspace Canvas file: {0}. "
        "Changes you make to this HTML file are shown to the user in real time. "
        "You can create or modify this file anytime.\n\n"
        "Replies and Canvas HTML can use extended links to open files and diffs "
        "in the board: git:REV:path, diff:REV:path, diff:BASE..HEAD:path, "
        "and diff:left..right. Append #L42 to select a file or left-side line, "
        "or #R42 for the right side of a diff. "
        "Use workspace://canvas to return to the live Canvas."
    ).format(board_path.resolve())


def main(argv: "typing.Union[typing.Sequence[str], None]" = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_serve_cli(args)
    except ValueError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        return 130
    return 0
