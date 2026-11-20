"""Workspace sessions with host tools and a client terminal execution tool."""

import asyncio
import secrets
import shlex
import time
from functools import partial

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from pycodex.events import SessionClosedEvent, SessionStateEvent
from pycodex.tools import get_tools
from pycodex.tools.base_tool import BaseTool
from pycodex_router.protocol import route_path
from pycodex_ws.server_support import json_object_body, read_body

from .sessions import ThreadedWorkspaceInteractiveSession, build_workspace_session

TOKEN_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
EXTERNAL_CALL_CONTEXT = (
    "<external_call_context>\n"
    "This is an External Call session involving two distinct machines:\n"
    "- Host machine (宿主设备 / workspace machine): the server running this pycodex workspace. "
    "The <environment_context> above, AGENTS.md, skills, and standard tools (such as exec_command and apply_patch) "
    "all describe and operate on the host machine.\n"
    "- Connected client machine (当前设备 / 外部接入设备 / 'this machine'): the external terminal connected to this session. "
    "Use client_exec_command to run commands on the connected client machine. Its OS, shell, working directory, "
    "and files are not included in <environment_context>; inspect them with client_exec_command when needed. "
    "Each client_exec_command call runs synchronously in a fresh noninteractive shell with no automatic timeout: "
    "avoid foreground servers or indefinite waits, wrap potentially slow commands with a command-level timeout, "
    "and detach background processes with redirected stdio so the call returns.\n"
    "</external_call_context>"
)


class ClientExecCommand(BaseTool):
    name = "client_exec_command"
    exec_only = False
    supports_parallel = False
    description = (
        "Execute a noninteractive command in the linked client terminal ('this machine'), "
        "waiting for completion. Returns combined output and exit_code. "
        "Standard tools run on the host."
    )
    input_schema = {
        "type": "object",
        "properties": {"cmd": {"type": "string", "description": "Shell command to execute."}},
        "required": ["cmd"],
        "additionalProperties": False,
    }

    def __init__(self, connection):
        self.connection = connection

    async def run(self, context, args):
        cmd = args.get("cmd")
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("cmd must be a nonempty string")
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(
            self.connection.execute(cmd), self.connection.loop,
        ))


class ExternalCallSession:
    def __init__(self, on_close):
        self.loop = asyncio.get_running_loop()
        self.on_close = on_close
        self.session_id = None
        self.token = "".join(secrets.choice(TOKEN_ALPHABET) for _ in range(10))
        self.closed = False
        self.lock = asyncio.Lock()
        self.replies = asyncio.Queue()
        self.pending = None
        self.in_flight = None

    def tools_factory(self, manager, environment, toolset):
        host_tools = tuple(name for name in toolset if name != ClientExecCommand.name)
        registry = get_tools(manager, environment, host_tools)
        registry.register(ClientExecCommand(self))
        return registry

    def build_session(self, config_path, profile, cwd, host_tools, workspace_context=(), system_prompt=None):
        contextual_messages = tuple(
            msg for msg in workspace_context if msg != EXTERNAL_CALL_CONTEXT
        ) + (EXTERNAL_CALL_CONTEXT,)
        session = build_workspace_session(
            config_path, profile, cwd, host_tools,
            workspace_context=contextual_messages, system_prompt=system_prompt,
            tools_factory=self.tools_factory,
        )
        session.runtime.attach(self.observe, interactive=False)
        return session

    def observe(self, event):
        if isinstance(event, SessionClosedEvent) or (
            isinstance(event, SessionStateEvent) and event.reason == "admission"
            and not event.state["accepts_input"]
        ):
            self.loop.call_soon_threadsafe(self.close)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.on_close(self)
        if self.pending is not None and not self.pending.done():
            self.pending.set_exception(ConnectionError("external call session closed"))
        self.replies.put_nowait(None)

    async def execute(self, cmd):
        async with self.lock:
            if self.closed:
                raise ConnectionError("external call session closed")
            pending = self.loop.create_future()
            self.pending = pending
            self.replies.put_nowait((cmd, pending))
            started = time.monotonic()
            try:
                output, exit_code = await pending
                return {
                    "output": output,
                    "exit_code": exit_code,
                    "wall_time_seconds": time.monotonic() - started,
                }
            finally:
                self.pending = None
                self.in_flight = None

    async def pull(self):
        if self.closed:
            raise HTTPException(410, "external call session closed")
        previous = self.replies
        replies = self.replies = asyncio.Queue()
        if self.in_flight is not None and not self.in_flight[1].done():
            replies.put_nowait(self.in_flight)
            self.in_flight = None
        while not previous.empty():
            item = previous.get_nowait()
            if item is not None and not item[1].done():
                replies.put_nowait(item)
        previous.put_nowait(None)
        reply = await replies.get()
        if self.closed:
            raise HTTPException(410, "external call session closed")
        if replies is not self.replies or reply is None:
            if reply is not None and not reply[1].done():
                self.replies.put_nowait(reply)
            raise HTTPException(409, "terminal connection was replaced")
        self.in_flight = reply
        return reply[0]

    def submit(self, output, exit_code):
        if self.closed:
            raise HTTPException(410, "external call session closed")
        if self.in_flight is None:
            raise HTTPException(409, "no command is waiting for a result")
        _, pending = self.in_flight
        self.in_flight = None
        if not pending.done():
            pending.set_result((output, exit_code))

    def one_liners(self, url):
        linux = (
            "(u={0};while c=$(curl -fsS $u);"
            "do o=$(bash -c \"$c\" </dev/null 2>&1);"
            "curl -fsS --data-raw \"$o\" \"$u?e=$?\";"
            "printf '%s\\n' \"$c\" \"$o\";done)"
        ).format(shlex.quote(url))
        windows = (
            "&{{$u={0};while(1){{$c=irm $u -ea Stop;"
            "$o=(& (gps -Id $PID).Path -nop -noni -c {{iex $args[0]}} -args $c 2>&1|Out-String);"
            "irm \"$u?e=$LASTEXITCODE\" -Method post "
            "-Body ([Text.Encoding]::UTF8.GetBytes($o)) -ea Stop;$c;$o}}}}"
        ).format(powershell_quote(url))
        return {"linux": linux, "windows": windows}


def powershell_quote(value):
    return "'" + value.replace("'", "''") + "'"


def format_binding(request, connection):
    path = request.scope.get("root_path", "") + route_path(request.scope).rsplit("/", 2)[0]
    url = str(request.url.replace(path=path + "/b/" + connection.token, query=""))
    return {
        "session_id": connection.session_id,
        "url": url,
        "one_liners": connection.one_liners(url),
    }


def external_call_binding(request, manager, session_id):
    active = request.app.state._state.get("external_call_sessions", {})
    connection = next((
        c for (owner, _), c in active.items()
        if owner is manager and c.session_id == session_id
    ), None)
    return None if connection is None else format_binding(request, connection)


def external_call_routes(resolve_workspace, prefix=""):
    async def settings(request):
        if request.scope["pycodex.access_role"] != "admin" or request.scope.get("pycodex.agent_access"):
            raise HTTPException(403, "administrator access required")
        workspace = resolve_workspace(request)
        manager = workspace.manager
        try:
            manager.refresh_profiles()
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        active = request.app.state._state.setdefault("external_call_sessions", {})
        if request.method == "POST":
            payload = await json_object_body(request)
            profile = payload.get("profile")
            if profile is None:
                profile = manager.default_profile
            if profile is not None and (
                not isinstance(profile, str) or profile not in manager.profile_factories
            ):
                raise HTTPException(422, "unknown profile")
            source = manager.get()
            if isinstance(source, ThreadedWorkspaceInteractiveSession):
                source = source._session
            source_config = source.runtime.agent.context_manager.config
            connection = ExternalCallSession(lambda c: active.pop((manager, c.token), None))
            session = ThreadedWorkspaceInteractiveSession(partial(
                connection.build_session, source.config_path, profile, workspace.work_dir,
                source.runtime.agent.tool_registry.names(),
                source_config.extra_contextual_user_messages,
                source_config.base_instructions_override,
            ), asyncio.get_running_loop())
            try:
                connection.session_id = await manager.attach_session(
                    session, title="External Call", profile=profile,
                )
            except BaseException:
                connection.close()
                await session.close()
                raise
            if connection.closed:
                raise HTTPException(410, "external call session closed during creation")
            active[manager, connection.token] = connection
            binding = format_binding(request, connection)
        else:
            binding = external_call_binding(request, manager, request.query_params.get("session_id"))
        return JSONResponse({
            "binding": binding, "profiles": list(manager.profile_factories),
            "default_profile": manager.default_profile,
        }, headers={"Cache-Control": "no-store"})

    async def step(request):
        key = resolve_workspace(request).manager, request.path_params["token"]
        connection = request.app.state._state.get("external_call_sessions", {}).get(key)
        if connection is None:
            raise HTTPException(401, "invalid external call connection code")
        params = request.query_params
        if request.method == "GET" and "o" not in params and "e" not in params:
            command = await connection.pull()
            return PlainTextResponse(command, headers={"Cache-Control": "no-store"})
        try:
            exit_code = int(params.get("e", "0"))
        except ValueError:
            raise HTTPException(422, "invalid exit code")
        if request.method == "POST":
            body = await read_body(request, 1024 * 1024)
            if body:
                try:
                    output = bytes(body).decode("utf-8")
                except UnicodeDecodeError:
                    raise HTTPException(422, "invalid terminal output")
            else:
                output = params.get("o", "")
        else:
            output = params.get("o", "")
            if len(output.encode("utf-8")) > 1024 * 1024:
                raise HTTPException(413, "request body too large")
        connection.submit(output, exit_code)
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    return [
        Route(prefix + "/api/external-call-sessions", settings, methods=["GET", "POST"]),
        Route(prefix + "/b/{token}", step, methods=["GET", "POST"]),
    ]
