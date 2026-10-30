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
from pycodex.multi_agents import AgentsManager
from pycodex.utils import uuid7_string
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
    _install_workspace_routes,
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
)
from .sharing import WorkspaceSharing
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
    parser.add_argument(
        "--use-chat-completion",
        default=False,
        action="store_true",
        help="Start a local responses compat server for this session.",
    )
    parser.add_argument(
        "--use-messages",
        default=False,
        action="store_true",
        help="Route through a downstream /v1/messages backend.",
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
    app = _create_lifespan_app(manager.start, manager.close)
    sharing = WorkspaceSharing(password, multi=False)
    _install_auth(app, password, sharing=sharing)
    _install_workspace_routes(app, manager, board_path, Path(work_dir or Path.cwd()).resolve())
    _install_sharing_routes(app, sharing)
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

    app.routes.extend(workspace_routes(
        resolve_workspace, sharing, "/w/{workspace_id}", [
            Route("/", index, methods=["GET"]),
            Route("/favicon.ico", favicon, methods=["GET"]),
            Route("/api/workspaces", workspaces, methods=["GET"]),
            Route("/api/workspaces", add_workspace, methods=["POST"]),
            Route("/api/workspaces/{workspace_id}", delete_workspace, methods=["DELETE"]),
            Route("/w/{workspace_id}", workspace_index_redirect, methods=["GET", "HEAD"]),
        ],
    ))
    _install_sharing_routes(app, sharing, registry)
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
    selected = read_codex_config(args.config, args.profile)
    profiles = list_codex_profiles(args.config)

    def build_session(profile) -> "WorkspaceInteractiveSession":
        config = CodexConfig.from_env(
            args.config, profile, use_default_profile=False,
        )
        bootstrap = build_model(
            config.provider,
            timeout_seconds=args.timeout_seconds,
            vllm_endpoint=args.vllm_endpoint,
            use_chat_completion=args.use_chat_completion or None,
            use_messages=args.use_messages,
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

    return WorkspaceEntry(
        definition=definition,
        manager=WorkspaceSessionManager(
            partial(session_factory, None),
            definition.state_path,
            persist_callback=persist_callback,
            profile_factories={
                name: partial(session_factory, name)
                for name in profiles
            },
            default_profile=selected["profile"],
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
