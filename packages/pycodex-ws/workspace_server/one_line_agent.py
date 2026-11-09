"""Workspace-owned Agent sessions whose only tool runs in a remote terminal."""

import asyncio
import atexit
import secrets
import shlex
import time
from dataclasses import replace
from functools import partial

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from pycodex.bootstrap import build_model
from pycodex.config import CodexConfig
from pycodex.events import SessionClosedEvent, SessionStateEvent, TurnCompletedEvent
from pycodex.multi_agents import AgentsManager
from pycodex.tools.base_tool import BaseTool, ToolRegistry
from pycodex_router.protocol import route_path
from pycodex_ws.server_support import json_object_body

from .sessions import ThreadedWorkspaceInteractiveSession, WorkspaceInteractiveSession


class RemoteExecCommand(BaseTool):
    name = "exec_command"
    exec_only = False
    supports_parallel = False
    description = (
        "Execute a noninteractive command in the connected client terminal, "
        "waiting for completion. Returns combined stdout/stderr and exit_code. "
        "Commands run in separate shells in the terminal's current directory."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "cmd": {"type": "string", "description": "Shell command to execute."},
        },
        "required": ["cmd"],
        "additionalProperties": False,
    }

    def __init__(self, connection):
        self.connection = connection

    async def run(self, context, args):
        if not isinstance(args.get("cmd"), str) or not args["cmd"].strip():
            raise ValueError("cmd must be a nonempty string")
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(
            self.connection.execute(args), self.connection.loop,
        ))


class OneLineAgent:
    def __init__(self, profile):
        self.profile = profile
        self.loop = asyncio.get_running_loop()
        self.session_id = None
        self.session = None
        self.runtime = None
        self.runtime_loop = None
        self.closed = False
        self.replies = asyncio.Queue()
        self.pending = None
        self.in_flight = None

    def context_config(self, config):
        instructions = (
            "exec_command runs on the client. "
            "The client's OS, shell and directory are not supplied; use tool results to adapt "
            "and read applicable client AGENTS.md as needed. Do not assume server files or skills "
            "are available. "
            "Execution is synchronous: until a command returns, you cannot continue or interrupt "
            "it, and there is no automatic timeout. Avoid unbounded waits and foreground services; "
            "use command-level time limits when needed. Detach background processes and redirect "
            "their standard streams so the call can return."
        )
        return replace(
            config,
            developer_instructions="\n\n".join(filter(None, (
                config.developer_instructions, instructions,
            ))),
            project_doc_max_bytes=0,
            codex_home_instructions=None,
            include_permissions_instructions=False,
            include_skills_instructions=False,
            include_environment_context=False,
            code_mode_only=False,
        )

    def build_session(self, config_path, work_dir):
        config = CodexConfig.from_env(
            config_path, self.profile, use_default_profile=False,
        )
        bootstrap = build_model(config.provider)
        if bootstrap.owned_server is not None:
            atexit.register(bootstrap.owned_server.stop)
        runtime = AgentsManager(
            bootstrap.client,
            self.context_config(replace(config.context, cwd=work_dir)),
            tools_factory=self.tools_factory,
        ).create_root_runtime()
        self.attach(runtime)
        return WorkspaceInteractiveSession(runtime, config_path)

    def tools_factory(self, manager, environment, toolset):
        registry = ToolRegistry(environment)
        registry.register(RemoteExecCommand(self))
        return registry

    def attach(self, runtime):
        self.runtime = runtime
        self.runtime_loop = asyncio.get_event_loop()
        runtime.attach(self.observe, interactive=False)

    def observe(self, event):
        if isinstance(event, SessionStateEvent):
            if event.reason == "admission" and not event.state["accepts_input"]:
                self.loop.call_soon_threadsafe(self.revoke)
        elif isinstance(event, SessionClosedEvent):
            self.loop.call_soon_threadsafe(self.revoke)
        elif isinstance(event, TurnCompletedEvent) and event.output_text:
            self.loop.call_soon_threadsafe(self.display_final, event.output_text)

    def display_final(self, text):
        if not self.closed:
            self.replies.put_nowait((echo_command(text), None))

    def revoke(self, reason="One-line Agent stopped."):
        if self.closed:
            return
        self.closed = True
        if self.runtime is not None and self.runtime.accepts_input:
            def stop_runtime():
                if self.runtime.accepts_input:
                    self.runtime.agents_manager.request_close(self.runtime.agent.agent_id)
            self.runtime_loop.call_soon_threadsafe(stop_runtime)
        if self.pending is not None and not self.pending.done():
            self.pending.set_exception(ConnectionError(reason))
        self.replies.put_nowait(None)

    async def execute(self, args):
        if self.closed:
            raise ConnectionError("One-line Agent terminal is no longer connected.")
        self.pending = self.loop.create_future()
        self.replies.put_nowait((args["cmd"], self.pending))
        started = time.monotonic()
        try:
            output, exit_code = await self.pending
            return {
                "output": output,
                "exit_code": exit_code,
                "wall_time_seconds": time.monotonic() - started,
            }
        finally:
            self.pending = None

    async def start(self, prompt):
        replies = self.replies
        result = await asyncio.shield(self.session.submit(prompt, sender="one_line_agent"))
        if not result["ok"]:
            raise HTTPException(422, result["error"])
        return await self.next_reply(replies)

    async def exchange(self, body):
        if self.in_flight is None:
            raise HTTPException(409, "no command is waiting for a result")
        try:
            output, status = body.decode("utf-8").rsplit("\n", 1)
            if not status.startswith("[exit ") or not status.endswith("]"):
                raise ValueError("exit status is required")
            exit_code = int(status[6:-1])
        except ValueError:
            raise HTTPException(422, "invalid terminal result")
        _, pending = self.in_flight
        self.in_flight = None
        if pending is not None:
            pending.set_result((output, exit_code))
        return await self.next_reply(self.replies)

    async def next_reply(self, replies):
        command = await replies.get()
        if self.closed:
            raise HTTPException(410, "One-line Agent invocation is closed")
        if replies is not self.replies:
            raise HTTPException(409, "terminal connection was replaced")
        self.in_flight = command
        return command[0]


class OneLineAgentService:
    def __init__(self, workspace, profile, on_close):
        self.manager = workspace.manager
        self.work_dir = workspace.work_dir
        source = self.manager.get()
        if isinstance(source, ThreadedWorkspaceInteractiveSession):
            source = source._session
        self.config_path = source.config_path
        self.profile = profile
        self.on_close = on_close
        self.loop = asyncio.get_running_loop()
        self.token = secrets.token_urlsafe(9)
        self.closed = False
        self.runs = {}

    def connections(self):
        return [
            connection for connection in self.runs.values()
            if connection.session_id is not None
        ]

    def authorize(self, token):
        if not secrets.compare_digest(
            token[:len(self.token)].encode("utf-8"), self.token.encode("ascii"),
        ):
            raise HTTPException(401, "invalid One-line Agent token")
        if self.closed:
            raise HTTPException(410, "One-line Agent is stopped")
        run_id = token[len(self.token):]
        if not run_id:
            raise HTTPException(422, "session suffix is required")
        return run_id

    def revoke(self, reason="One-line Agent stopped."):
        if self.closed:
            return
        self.closed = True
        for connection in self.runs.values():
            connection.revoke(reason)
        self.on_close()

    async def create_run(self, run_id):
        connection = OneLineAgent(self.profile)
        self.runs[run_id] = connection
        connection.session = ThreadedWorkspaceInteractiveSession(
            partial(connection.build_session, self.config_path, self.work_dir), self.loop,
        )
        try:
            connection.session_id = await self.manager.attach_session(
                connection.session, title=f"OneLineAgent-{run_id}", profile=self.profile,
            )
            if self.closed or connection.closed:
                raise HTTPException(410, "One-line Agent is stopped")
        except BaseException:
            connection.revoke()
            await connection.session.close()
            raise
        return connection

    async def exchange(self, run_id, body):
        if not run_id:
            raise HTTPException(422, "session is required")
        if self.closed:
            raise HTTPException(410, "One-line Agent is stopped")
        connection = self.runs.get(run_id)
        if connection is not None and connection.closed:
            connection = None
        new_task = connection is None
        if (connection is not None and connection.session_id is not None
                and connection.pending is None and connection.in_flight is None):
            snapshot = await connection.session.api_query("state", {})
            new_task = (not snapshot["state"]["busy"] and connection.pending is None
                        and connection.in_flight is None)
        if new_task:
            try:
                prompt = body.decode("utf-8")
                if not prompt.strip():
                    raise ValueError("task is required")
            except (UnicodeDecodeError, ValueError):
                raise HTTPException(422, "invalid task")
            if connection is None:
                connection = await self.create_run(run_id)
            else:
                # Hand the idle channel to this caller before admitting the new task.
                previous = connection.replies
                connection.replies = asyncio.Queue()
                previous.put_nowait(None)
            return await connection.start(prompt)
        return await connection.exchange(body)

    def one_liners(self, base_url):
        token = self.token + "1"
        task = "看一下这台机器"
        linux = (
            "(i={0};while o=$(curl -fsS -H {1} --data-raw \"$i\" {2});"
            "do i=$(bash -c \"$o\" </dev/null 2>&1);i+=$'\\n'\"[exit $?]\";"
            "printf '%s\\n' \"$o\" \"$i\";done)"
        ).format(
            shlex.quote(task), shlex.quote("T:" + token),
            shlex.quote(base_url + "/b"),
        )
        windows = (
            "&{{$i={0};while(1){{$o=irm {2} -Method Post "
            "-Headers @{{T={1}}} -Body ([Text.Encoding]::UTF8.GetBytes($i)) -ea Stop"
            ";$i=(& (gps -Id $PID).Path -NoProfile -NonInteractive -Command "
            "{{iex $args[0]}} -args $o 2>&1|Out-String);"
            "$i+=\"`n[exit $LASTEXITCODE]\";$o;$i}}}}"
        ).format(
            powershell_quote(task), powershell_quote(token),
            powershell_quote(base_url + "/b"),
        )
        return {"linux": linux, "windows": windows}


def echo_command(text):
    # PowerShell also treats typographic apostrophes as single-quote delimiters.
    # Put each apostrophe in a double-quoted segment.
    quoted = "".join(
        "'\"" + char + "\"'" if char in "'\u2018\u2019\u201a\u201b" else char
        for char in text
    )
    # A leading newline prevents Bash echo from treating the text as an option.
    return "echo '\n" + quoted + "'"


def powershell_quote(value):
    return "'" + value.replace("'", "''") + "'"


def one_line_agent_routes(resolve_workspace, prefix=""):
    async def settings(request):
        if request.scope["pycodex.access_role"] != "admin":
            raise HTTPException(403, "administrator access required")
        workspace = resolve_workspace(request)
        manager = workspace.manager
        services = request.app.state._state.setdefault("one_line_agent_endpoints", {})
        manager.refresh_profiles()
        service = services.get(manager)
        if request.method == "POST":
            payload = await json_object_body(request)
            if "profile" not in payload:
                raise HTTPException(422, "select a profile")
            profile = payload["profile"]
            if profile is None:
                profile = manager.default_profile
            if profile is not None and (
                not isinstance(profile, str) or profile not in manager.profile_factories
            ):
                raise HTTPException(422, "unknown profile")
            if service is not None and not service.closed:
                raise HTTPException(409, "stop the current One-line Agent before creating another")

            def unregister():
                if services.get(manager) is service:
                    del services[manager]

            service = OneLineAgentService(workspace, profile, unregister)
            services[manager] = service
        elif request.method == "DELETE":
            if service is not None:
                service.revoke()
        binding = None
        if service is not None:
            connections = service.connections()
            session_ids = {connection.session_id for connection in connections}
            open_sessions = [item["id"] for item in manager.list_sessions() if item["id"] in session_ids]
            binding = {
                "session_id": open_sessions[-1] if open_sessions else None,
                "session_count": len(open_sessions),
                "profile": service.profile,
                "enabled": not service.closed,
                "one_liners": service.one_liners(str(request.url.replace(
                    path=request.scope.get("root_path", "") + route_path(request.scope).rsplit("/", 2)[0],
                    query="",
                )))
                if not service.closed else None,
            }
        return JSONResponse({
            "binding": binding, "profiles": list(manager.profile_factories),
            "default_profile": manager.default_profile,
        }, headers={"Cache-Control": "no-store"})

    async def step(request):
        services = request.app.state._state.setdefault("one_line_agent_endpoints", {})
        service = services.get(resolve_workspace(request).manager)
        if service is None:
            raise HTTPException(404, "One-line Agent is not enabled")
        run_id = service.authorize(request.headers.get("t", ""))
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > 1024 * 1024:
                raise HTTPException(413, "terminal result exceeds 1 MiB")
            body.extend(chunk)
        command = await service.exchange(run_id, bytes(body))
        return PlainTextResponse(command, headers={"Cache-Control": "no-store"})

    return [
        Route(prefix + "/api/one-line-agent", settings, methods=["GET", "POST", "DELETE"]),
        Route(prefix + "/b", step, methods=["POST"]),
    ]
