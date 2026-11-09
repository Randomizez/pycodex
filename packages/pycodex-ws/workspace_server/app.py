import argparse
import atexit
import asyncio
import secrets
import sys
from dataclasses import replace
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

from pycodex.bootstrap import build_model, configure_loguru
from pycodex.config import CodexConfig, DEFAULT_CODEX_CONFIG_PATH
from pycodex.events import IDLE_SLEEPING_STATUS
from pycodex.multi_agents import AgentsManager
from pycodex.utils import uuid7_string
from pycodex.utils.get_env import get_package_version
from pycodex.utils.codex_config import list_codex_profiles, read_codex_config
from pycodex_ws import register_feishu_commands
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
from .reloader import preflight_check, reload_modules_in_place, sync_reloaded_sessions
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
)
from .sharing import WorkspaceSharing
from .workspaces import (
    WorkspaceDefinition,
    WorkspaceEntry,
    WorkspaceRegistry,
    WorkspaceSessionManager,
    load_workspace_definitions,
    session_snapshot,
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
    app = _create_lifespan_app(manager.start, manager.close)
    sharing = WorkspaceSharing(password, multi=False)
    _install_auth(app, password, sharing=sharing)
    _install_sharing_routes(app, sharing)
    _install_reload_routes(
        app,
        sharing,
        lambda: [("", manager)],
        lambda _connection: WorkspaceRouteContext(manager, board_path, resolved_work_dir),
        "",
        (),
    )
    return app


def create_multi_workspace_app(
    registry: "WorkspaceRegistry",
    password: "typing.Union[str, None]" = None,
) -> Starlette:
    app = _create_lifespan_app(registry.start, registry.close)
    sharing = WorkspaceSharing(password, multi=True)
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
        lambda: [(key, registry._entries[key].manager) for key in registry._order],
        resolve_workspace,
        "/w/{workspace_id}",
        leading_routes,
    )
    return app


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
    get_managers,
    resolve_workspace,
    prefix,
    leading_routes,
):
    reloading = False

    def active_sessions() -> "typing.List[str]":
        active = []
        for workspace_id, manager in get_managers():
            for session in manager.list_sessions():
                if (
                    session.get("running")
                    and session.get("spinner") != IDLE_SLEEPING_STATUS
                ):
                    title = session.get("title") or session.get("id") or "session"
                    active.append(
                        "{0} / {1}".format(workspace_id, title)
                        if workspace_id
                        else str(title)
                    )
        return active

    async def reload_pycodex(request: Request) -> JSONResponse:
        nonlocal reloading
        try:
            target = resolve_workspace(request).manager
        except (KeyError, ValueError):
            raise HTTPException(status_code=404, detail="workspace not found")
        if request.scope["pycodex.access_role"] != "admin":
            return JSONResponse({"error": "administrator access required"}, 403)
        payload = (
            await _json_object_body(request, max_bytes=4096)
            if await request.body()
            else {}
        )
        session_id = payload.get("session_id")
        if session_id is not None and not isinstance(session_id, str):
            return JSONResponse({"error": "session_id must be a string"}, 422)
        if payload.get("check_only") is True:
            active = active_sessions()
            return JSONResponse(
                {
                    "ok": True,
                    "status": "waiting" if (reloading or active) else "ready",
                    "active_turns": len(active),
                    "running_sessions": active,
                },
                headers={"Cache-Control": "no-store"},
            )
        while reloading or active_sessions():
            await asyncio.sleep(0.05)
        reloading = True
        try:
            try:
                await preflight_check()
            except Exception as exc:
                return JSONResponse(
                    {"ok": False, "status": "preflight_failed", "error": str(exc)},
                    status_code=400,
                    headers={"Cache-Control": "no-store"},
                )
            reloaded_modules, rollback_modules = reload_modules_in_place()
            restore_views = lambda: None
            try:
                managers = [manager for _wid, manager in get_managers()]
                restore_views = await sync_reloaded_sessions(managers)
                refresh_routes()
                for manager in managers:
                    manager.refresh_profiles()
                resolved_id = target.resolve_session_id(
                    session_id if session_id in target._sessions else None
                )
                sessions_payload = target.list_sessions()
                snapshot_payload = session_snapshot(target.get(resolved_id))
            except Exception as exc:
                restore_views()
                rollback_modules()
                refresh_routes()
                return JSONResponse(
                    {
                        "ok": False,
                        "status": "reload_failed",
                        "error": "{0}: {1}".format(type(exc).__name__, exc),
                    },
                    status_code=400,
                    headers={"Cache-Control": "no-store"},
                )
            return JSONResponse(
                {
                    "ok": True,
                    "status": "reloaded",
                    "reloaded_modules": reloaded_modules,
                    "version": get_package_version(),
                    "session_id": resolved_id,
                    "sessions": sessions_payload,
                    "snapshot": snapshot_payload,
                },
                headers={"Cache-Control": "no-store"},
            )
        finally:
            reloading = False

    app.routes.insert(
        0, Route(prefix + "/api/reload", reload_pycodex, methods=["POST"])
    )
    fixed_routes = list(app.routes)

    def refresh_routes() -> None:
        from . import routes as routes_module

        app.routes[:] = fixed_routes + list(
            routes_module.workspace_routes(
                resolve_workspace, sharing, prefix, leading_routes
            )
        )

    refresh_routes()


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


def run_serve_cli(args: "argparse.Namespace") -> int:
    from .owned_process import run_owned_server

    host, port = parse_listen(args.listen)
    configure_loguru()

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
    app = create_multi_workspace_app(registry, password=args.password)
    if args.listen_to:
        from pycodex_router.client import attach_router

        attach_router(app, registry, args.listen_to, port)

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
    run_owned_server(app, host, port, sys.stdin.fileno())
    return 0


def _build_workspace_entry(
    definition: "WorkspaceDefinition",
    args: "argparse.Namespace",
    persist_callback: "typing.Union[typing.Callable[[], None], None]" = None,
) -> "WorkspaceEntry":
    def build_session(profile) -> "WorkspaceInteractiveSession":
        config = CodexConfig.from_env(
            args.config, profile, use_default_profile=False,
        )
        bootstrap = build_model(
            config.provider,
            timeout_seconds=args.timeout_seconds,
            vllm_endpoint=args.vllm_endpoint,
        )
        if bootstrap.owned_server is not None:
            atexit.register(bootstrap.owned_server.stop)
        context_config = replace(
            config.context,
            base_instructions_override=args.system_prompt,
            extra_contextual_user_messages=(
                (_workspace_context_text(definition.board_path),)
                if definition.board_path is not None
                else ()
            ),
            cwd=definition.work_dir,
        )
        runtime = AgentsManager(
            bootstrap.client, context_config, args.toolset,
        ).create_root_runtime()
        register_feishu_commands(runtime)
        return WorkspaceInteractiveSession(
            runtime,
            config_path=args.config,
        )

    def session_factory(profile) -> "ThreadedWorkspaceInteractiveSession":
        return ThreadedWorkspaceInteractiveSession(
            partial(build_session, profile), asyncio.get_running_loop()
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
