import asyncio
import re
import shutil
import sys
from contextlib import AsyncExitStack

import httpx
import pytest
from starlette.exceptions import HTTPException
from starlette.testclient import TestClient

from pycodex import AssistantMessage, ModelResponse, ToolCall, UserMessage
from pycodex.bootstrap import ModelBootstrapResult
from pycodex.protocol import ToolResult
from tests.fakes import ScriptedModelClient
from workspace_server import (
    WorkspaceDefinition, WorkspaceRegistry, build_parser, create_app, create_multi_workspace_app,
)
from workspace_server.agent_access import AgentSocketAccess
from workspace_server.app import _build_workspace_entry
from workspace_server.external_call import EXTERNAL_CALL_CONTEXT, ExternalCallSession

from test_router import eventually, serve, serve_router


@pytest.fixture
def make_workspace(tmp_path, monkeypatch):
    def make(responses=(), multi=False, toolset=("exec_command",), enable_board=False, system_prompt=None):
        def build_model(provider, **kwargs):
            return ModelBootstrapResult(ScriptedModelClient(responses, model=provider.model))

        monkeypatch.setattr("workspace_server.sessions.build_model", build_model)
        config = tmp_path / "config.toml"
        config.write_text(
            'model = "base"\n[profiles.remote]\nmodel = "remote-model"\n'
            'developer_instructions = "Selected profile instructions."\n'
            'code_mode_only = false\n', encoding="utf-8",
        )
        cli_args = ["--config", str(config), "--toolset", *toolset]
        if system_prompt is not None:
            cli_args.extend(["--system-prompt", system_prompt])
        args = build_parser().parse_args(cli_args)
        entries = []
        for name in (["alpha", "beta"] if multi else ["alpha"]):
            directory = tmp_path / name
            directory.mkdir(exist_ok=True)
            (directory / "AGENTS.md").write_text("HOST_PROJECT_INSTRUCTIONS", encoding="utf-8")
            entries.append(_build_workspace_entry(
                WorkspaceDefinition(name, directory / "state", directory, enable_board), args,
            ))
        registry = WorkspaceRegistry(entries)
        manager = entries[0].manager
        app = (
            create_multi_workspace_app(registry, "owner-password") if multi else
            create_app(manager, entries[0].definition.state_path, "owner-password", entries[0].definition.work_dir, enable_board)
        )
        return app, registry, manager
    return make


def connection_code(binding):
    return re.search(r"/b/([a-z0-9]+)", binding["one_liners"]["linux"]).group(1)


@pytest.mark.parametrize("multi", [False, True])
def test_settings_create_scoped_sessions_and_revoke_on_close(make_workspace, multi):
    app, _, manager = make_workspace(multi=multi)
    prefix = "/w/alpha" if multi else ""
    path = prefix + "/api/external-call-sessions"
    with TestClient(app) as browser:
        assert browser.get(path).status_code == 401
        browser.post("/login", json={"password": "owner-password"})
        assert browser.get(path).json()["profiles"] == ["remote"]
        assert browser.post(path, json={"profile": []}).status_code == 422
        assert browser.post(path, json={"profile": "missing"}).status_code == 422
        first = browser.post(path, json={"profile": "remote"}).json()["binding"]
        second = browser.post(path, json={"profile": None}).json()["binding"]
        first_code = connection_code(first)
        second_code = connection_code(second)
        assert first["session_id"] != second["session_id"]
        assert first_code != second_code
        assert re.fullmatch(r"[a-hjkmnp-z2-9]{10}", first_code)
        assert first["url"].endswith(prefix + "/b/" + first_code)
        assert first["url"] in first["one_liners"]["linux"]
        assert "/b/" + first_code in first["one_liners"]["windows"]
        assert all("\n" not in command for command in first["one_liners"].values())
        assert len(manager.list_sessions()) == 3
        assert len(manager._state_store.load_tabs()) == 1
        assert manager.get(first["session_id"])._session.runtime.agent.model_client.model == "remote-model"
        assert browser.get(path, params={"session_id": first["session_id"]}).json()["binding"] == first
        assert browser.get(path, params={"session_id": manager.resolve_session_id()}).json()["binding"] is None
        assert browser.get(prefix + "/api/session", params={"session_id": first["session_id"]}).json()["external_call"] == first
        assert "external_call" not in browser.get(prefix + "/api/session", params={"session_id": manager.resolve_session_id()}).json()
        assert first_code not in browser.get(prefix + "/api/sessions").text
        first_step = prefix + "/b/" + first_code
        second_step = prefix + "/b/" + second_code
        if multi:
            assert browser.get("/w/beta/b/" + first_code).status_code == 401
            assert browser.post("/w/beta/b/" + first_code, content=b"bad").status_code == 401
        assert browser.delete(prefix + "/api/sessions/" + first["session_id"]).status_code == 200
        assert browser.get(path, params={"session_id": first["session_id"]}).json()["binding"] is None
        assert browser.get(first_step).status_code == 401
        assert browser.post(first_step).status_code == 401
        browser.cookies.clear()
        assert browser.get(path).status_code == 401
        assert browser.get(prefix + "/api/sessions").status_code == 401
        for value in (second["session_id"], second_code + "1", "0000"):
            assert browser.get(prefix + "/b/" + value).status_code == 401
            assert browser.post(prefix + "/b/" + value).status_code == 401
        for bad_exit in ("", "not-an-int"):
            assert browser.get(second_step, params={"e": bad_exit}).status_code == 422
            assert browser.post(second_step, params={"e": bad_exit}, content=b"ok").status_code == 422
        assert browser.post(second_step, content=b"\xff").status_code == 422
        assert browser.get(second_step, params={"e": "0"}).status_code == 409
        assert browser.get(second_step, params={"o": "ok"}).status_code == 409
        assert browser.post(second_step, content=b"ok").status_code == 409
        assert browser.post(second_step, content=b"x" * (1024 * 1024 + 1)).status_code == 413
        assert not manager.get(second["session_id"])._session.runtime.agent.model_client.prompts


def test_visitor_and_agent_access_cannot_read_session_connection_codes(make_workspace):
    app, _, _ = make_workspace(multi=True)
    app.state.agent_socket_path = "/tmp/pycodex-agent.sock"

    async def over_agent_socket(scope, receive, send):
        if scope["type"] == "http":
            scope = dict(scope, server=("/tmp/pycodex-agent.sock", None))
        await AgentSocketAccess(app)(scope, receive, send)

    with TestClient(app) as browser:
        browser.post("/login", json={"password": "owner-password"})
        path = "/w/alpha/api/external-call-sessions"
        binding = browser.post(path, json={}).json()["binding"]
        code = connection_code(binding)
        agent_client = TestClient(over_agent_socket)
        agent_session = agent_client.get("/w/alpha/api/session", params={"session_id": binding["session_id"]})
        assert agent_session.status_code == 200
        assert "external_call" not in agent_session.json()
        assert code not in agent_session.text
        assert agent_client.get(path, params={"session_id": binding["session_id"]}).status_code in {401, 403}
        assert browser.put("/w/alpha/api/share", json={"password": "visitor-password"}).status_code == 200
        browser.cookies.clear()
        login = browser.post("/login", params={"next": "/w/alpha/"}, json={"password": "visitor-password"})
        assert login.status_code == 200
        assert browser.get(path, params={"session_id": binding["session_id"]}).status_code == 403
        assert browser.post(path, json={}).status_code == 403
        visitor_session = browser.get("/w/alpha/api/session", params={"session_id": binding["session_id"]})
        assert visitor_session.status_code == 200
        assert "external_call" not in visitor_session.json()
        assert code not in visitor_session.text


def test_reload_keeps_open_binding_and_restart_revokes_it(make_workspace, monkeypatch):
    from workspace_server import reloader

    app, _, manager = make_workspace([ModelResponse([AssistantMessage("recorded")])])
    path = "/api/external-call-sessions"
    with TestClient(app) as browser:
        browser.post("/login", json={"password": "owner-password"})
        binding = browser.post(path, json={}).json()["binding"]
        session_id = binding["session_id"]
        runtime = manager.get(session_id)._session.runtime
        browser.post("/api/session/message", json={"session_id": session_id, "prompt": "Save this turn"})
        browser.portal.call(eventually, lambda: runtime.agent.model_client.call_count == 1 and not runtime.is_busy)
        rollout = runtime.agent.recorded_session_file_path
        modules = [(name, module) for name, module in reloader.loaded_reloadable_modules()
                   if name == "workspace_server.external_call"]
        monkeypatch.setattr(reloader, "loaded_reloadable_modules", lambda: modules)
        assert browser.post("/api/reload", json={}).json()["status"] == "reloaded"
        assert browser.get(path, params={"session_id": session_id}).json()["binding"] == binding
        assert len(manager._state_store.load_tabs()) == 1
    assert rollout.is_file()
    assert connection_code(binding) not in rollout.read_text(encoding="utf-8")
    restarted, _, restored = make_workspace()
    with TestClient(restarted) as browser:
        assert len(restored.list_sessions()) == 1
        assert browser.get("/b/" + connection_code(binding)).status_code == 401
        assert browser.post("/b/" + connection_code(binding)).status_code == 401


@pytest.mark.asyncio
async def test_web_task_uses_host_and_client_tools_with_correlated_results(make_workspace):
    app, _, manager = make_workspace([
        ModelResponse([ToolCall("host", "exec_command", {"cmd": "pwd"})]),
        ModelResponse([ToolCall("external", "client_exec_command", {"cmd": "pwd"})]),
        ModelResponse([ToolCall("via_get", "client_exec_command", {"cmd": "whoami"})]),
        ModelResponse([ToolCall("empty_post", "client_exec_command", {"cmd": "true"})]),
        ModelResponse([AssistantMessage("Complete")]),
    ], enable_board=True, system_prompt="CUSTOM_SYSTEM_PROMPT")
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as web:
            await web.post("/login", json={"password": "owner-password"})
            binding = (await web.post("/api/external-call-sessions", json={"profile": "remote"})).json()["binding"]
            session_id = binding["session_id"]
            step = "/b/" + connection_code(binding)
            runtime = manager.get(session_id)._session.runtime
            agent = runtime.agent
            assert set(agent.tool_registry.names()) == {"exec", "wait", "exec_command", "client_exec_command"}
            assert "client_exec_command" not in manager.get()._session.runtime.agent.tool_registry.names()
            assert EXTERNAL_CALL_CONTEXT not in manager.get()._session.runtime.agent.context_manager.config.extra_contextual_user_messages
            root_messages = agent.context_manager.config.extra_contextual_user_messages
            assert root_messages.count(EXTERNAL_CALL_CONTEXT) == 1
            assert any("Current workspace Canvas file:" in msg for msg in root_messages)
            assert agent.context_manager.config.base_instructions_override == "CUSTOM_SYSTEM_PROMPT"
            async def child_tools():
                child = runtime.agents_manager.build_agent(agent.agent_id, "/root/child")
                try:
                    return (
                        child.tool_registry.names(),
                        child.context_manager.config.extra_contextual_user_messages,
                    )
                finally:
                    await child.tool_registry.runtime_environment.close()
            child_names, child_messages = await manager.get(session_id)._call(child_tools())
            assert child_names == agent.tool_registry.names()
            assert child_messages == root_messages
            reply = await web.post("/api/session/message", json={"session_id": session_id, "prompt": "Compare directories"})
            assert reply.json()["ok"]
            pickup = await asyncio.wait_for(web.get(step), 4)
            assert pickup.text == "pwd"
            assert (await asyncio.wait_for(web.get(step), 4)).text == "pwd"
            body = b"/client's folder\n"
            assert (await web.post(step, params={"e": "7"}, content=body)).status_code == 204
            assert (await web.post(step, params={"e": "7"}, content=body)).status_code == 409
            assert (await asyncio.wait_for(web.get(step), 4)).text == "whoami"
            assert (await web.get(step, params={"o": "alice\n", "e": "3"})).status_code == 204
            assert (await web.get(step, params={"o": "alice\n"})).status_code == 409
            assert (await asyncio.wait_for(web.get(step), 4)).text == "true"
            assert (await web.post(step)).status_code == 204
            await eventually(lambda: agent.model_client.call_count == 5 and not runtime.is_busy)
            host, client, via_get, empty_post = [item for item in agent.history if isinstance(item, ToolResult)]
            assert not host.is_error, host.output
            assert str(agent.context_manager.cwd) in host.output["output"]
            assert client.call_id == "external"
            assert client.output["output"] == "/client's folder\n"
            assert client.output["exit_code"] == 7
            assert set(client.output) == {"output", "exit_code", "wall_time_seconds"}
            assert via_get.call_id == "via_get"
            assert via_get.output["output"] == "alice\n" and via_get.output["exit_code"] == 3
            assert empty_post.call_id == "empty_post"
            assert empty_post.output["output"] == "" and empty_post.output["exit_code"] == 0
            assert [item.text for item in agent.history if isinstance(item, UserMessage)] == ["Compare directories"]
            assert agent.model_client.prompts[0].base_instructions == "CUSTOM_SYSTEM_PROMPT"
            prompt = repr(agent.model_client.prompts[0].input)
            assert "Selected profile instructions." in prompt and "HOST_PROJECT_INSTRUCTIONS" in prompt
            assert "<external_call_context>" in prompt and "宿主设备" in prompt and "当前设备" in prompt
            assert "Current workspace Canvas file:" in prompt
            assert connection_code(binding) not in prompt
            assert agent.session_file_path.is_file()
            waiting = asyncio.create_task(web.get(step))
            await asyncio.sleep(0.05)
            idle = asyncio.create_task(web.get(step))
            assert (await asyncio.wait_for(waiting, 3)).status_code == 409
            assert (await web.delete("/api/sessions/" + session_id)).status_code == 200
            assert (await asyncio.wait_for(idle, 3)).status_code == 410


@pytest.mark.asyncio
async def test_parallel_client_calls_queue_and_close_releases_every_caller():
    connection = ExternalCallSession(lambda _: None)
    first = asyncio.create_task(connection.execute("first"))
    second = asyncio.create_task(connection.execute("second"))
    assert await connection.pull() == "first"
    connection.submit("first output", 0)
    first_result = await first
    assert first_result["output"] == "first output" and first_result["exit_code"] == 0
    assert await connection.pull() == "second"
    third = asyncio.create_task(connection.execute("third"))
    connection.close()
    for task in (second, third):
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(task, 2)
    with pytest.raises(HTTPException) as closed:
        await connection.pull()
    assert closed.value.status_code == 410


@pytest.mark.asyncio
async def test_close_while_client_call_is_pending_and_isolated(make_workspace):
    app, _, manager = make_workspace([ModelResponse([
        ToolCall("client", "client_exec_command", {"cmd": "pwd"}),
    ])], toolset=())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as web:
            await web.post("/login", json={"password": "owner-password"})
            bindings = [(await web.post("/api/external-call-sessions", json={})).json()["binding"]
                        for _ in range(2)]
            for binding in bindings:
                session_id = binding["session_id"]
                assert manager.get(session_id)._session.runtime.agent.tool_registry.names() == ("client_exec_command",)
                await web.post("/api/session/message", json={"session_id": session_id, "prompt": "Check the client"})
                pickup = await asyncio.wait_for(web.get("/b/" + connection_code(binding)), 3)
                assert pickup.status_code == 200 and pickup.text == "pwd"
            first, second = bindings
            assert (await asyncio.wait_for(web.delete("/api/sessions/" + first["session_id"]), 3)).status_code == 200
            assert (await web.get("/b/" + connection_code(first))).status_code == 401
            assert manager.get(second["session_id"])._session.runtime.is_busy
            assert (await asyncio.wait_for(web.delete("/api/sessions/" + second["session_id"]), 3)).status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("through_router", [False, True])
@pytest.mark.parametrize("shell", ["bash", "python", "pwsh"])
async def test_terminal_executes_commands_and_exits_on_close(make_workspace, tmp_path, through_router, shell):
    from pycodex_router.client import attach_router
    from pycodex_router.server import create_app as create_router

    executable = sys.executable if shell == "python" else shutil.which(shell)
    if executable is None or (shell == "bash" and not shutil.which("curl")):
        pytest.skip("terminal programs unavailable for " + shell)
    app, registry, manager = make_workspace([
        ModelResponse([ToolCall("client", "client_exec_command", {"cmd": "echo created > created.txt"})]),
        ModelResponse([ToolCall("second", "client_exec_command", {"cmd": "echo '终端 output'\nexit 7"})]),
        ModelResponse([AssistantMessage("done")]),
    ], multi=through_router)
    client_dir = tmp_path / "client"
    client_dir.mkdir()
    async with AsyncExitStack() as stack:
        web = await stack.enter_async_context(httpx.AsyncClient(trust_env=False))
        if through_router:
            address, target = await stack.enter_async_context(serve_router(create_router("router-password"), "/remote"))
            attach_router(app, registry, target, 6007)
            await stack.enter_async_context(serve(app))
            await eventually(lambda: app.state.router_client.connected)
            await web.post(address + "/remote/login", json={"password": "router-password"})
            node = address + "/remote/n/" + app.state.router_client.node_id[:5]
            base = node + "/w/alpha"
        else:
            node = base = await stack.enter_async_context(serve(app))
        await web.post(node + "/login", json={"password": "owner-password"})
        binding = (await web.post(base + "/api/external-call-sessions", json={})).json()["binding"]
        step_url = base + "/b/" + connection_code(binding)
        assert step_url in binding["one_liners"]["linux"]
        session_id = binding["session_id"]
        runtime = manager.get(session_id)._session.runtime
        await web.post(base + "/api/session/message", json={"session_id": session_id, "prompt": "Check the client"})
        if shell == "bash":
            args = ["-c", binding["one_liners"]["linux"]]
        elif shell == "python":
            args = [
                "-c",
                "import subprocess as s,urllib.error as e,urllib.request as r;"
                f"u={step_url!r}\n"
                "try:\n"
                " while 1:\n"
                "  c=r.urlopen(u).read().decode()\n"
                "  p=s.run(c,shell=True,capture_output=True)\n"
                "  print(c);print(p.stdout.decode(errors='replace'),end='')\n"
                "  r.urlopen(f'{u}?e={p.returncode}',p.stdout+p.stderr)\n"
                "except e.HTTPError:\n"
                " pass\n",
            ]
        else:
            args = ["-NoProfile", "-Command", binding["one_liners"]["windows"]]
        process = await asyncio.create_subprocess_exec(
            executable, *args, cwd=str(client_dir),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            await eventually(lambda: runtime.agent.model_client.call_count == 3 and not runtime.is_busy)
            first, second = [item.output for item in runtime.agent.history if isinstance(item, ToolResult)]
            assert first["exit_code"] == 0 and first["output"] == ""
            assert (client_dir / "created.txt").is_file()
            assert second["exit_code"] == 7
            assert second["output"].rstrip() == "终端 output"
            assert (await web.delete(base + "/api/sessions/" + session_id)).status_code == 200
            stdout, _ = await asyncio.wait_for(process.communicate(), 5)
            rendered = stdout.decode("utf-8", errors="replace")
            assert "echo created > created.txt" in rendered and "终端 output" in rendered
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()
