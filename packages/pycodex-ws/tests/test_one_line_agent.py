import asyncio
import inspect
import shutil
import subprocess
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient
from starlette.exceptions import HTTPException

from pycodex import AssistantMessage, ModelResponse, ToolCall, UserMessage
from pycodex.bootstrap import ModelBootstrapResult
from pycodex.protocol import ToolResult
from tests.fakes import ScriptedModelClient
from workspace_server import (
    WorkspaceDefinition, WorkspaceRegistry, build_parser,
    create_app, create_multi_workspace_app,
)
from workspace_server.app import _build_workspace_entry

from test_router import eventually, serve, serve_router


@pytest.fixture
def make_one_line_workspace(tmp_path, monkeypatch):
    def make(responses, multi=False):
        models = []

        def build_model(provider, **kwargs):
            model = ScriptedModelClient(responses, model=provider.model)
            models.append(model)
            return ModelBootstrapResult(model)

        monkeypatch.setattr("workspace_server.app.build_model", build_model)
        monkeypatch.setattr("workspace_server.one_line_agent.build_model", build_model)
        monkeypatch.setattr("pycodex.bootstrap.build_model", build_model)
        config = tmp_path / "config.toml"
        config.write_text(
            'model = "base"\n[profiles.remote]\nmodel = "remote-model"\n'
            'developer_instructions = "Selected profile instructions."\n'
            'user_instructions = "Selected profile user instructions."\n'
            'code_mode_only = true\n', encoding="utf-8",
        )
        args = build_parser().parse_args(["--config", str(config)])
        entries = []
        for name in (["alpha", "beta"] if multi else ["alpha"]):
            directory = tmp_path / name
            directory.mkdir(exist_ok=True)
            (directory / "AGENTS.md").write_text("SERVER_ONLY_INSTRUCTIONS", encoding="utf-8")
            entries.append(_build_workspace_entry(
                WorkspaceDefinition(name, directory / "state", directory, False), args,
            ))
        registry = WorkspaceRegistry(entries)
        app = (
            create_multi_workspace_app(registry, "owner-password")
            if multi else create_app(entries[0].manager, None, "owner-password")
        )
        return app, registry, entries[0].manager, models

    return make


def create_binding(browser, prefix=""):
    assert browser.post("/login", json={"password": "owner-password"}).status_code == 200
    response = browser.post(prefix + "/api/one-line-agent", json={"profile": "remote"})
    assert response.status_code == 200, response.text
    return response.json()["binding"]


def start_run(browser, service, body, run_id="1", prefix=""):
    response = browser.post(
        prefix + "/b", content=body, headers={"T": service.token + run_id},
    )
    assert response.status_code == 200, response.text
    return run_id, response


def wait_for_turn(browser, connection, calls):
    browser.portal.call(eventually, lambda: (
        connection.runtime.agent.model_client.call_count == calls
        and not connection.runtime.is_busy
    ))


def ack_final(browser, service, run_id, request, text):
    from workspace_server.one_line_agent import echo_command

    assert request.result(3) == echo_command(text)
    waiting = browser.portal.start_task_soon(
        service.exchange, run_id, (text + "\n[exit 0]").encode(),
    )
    browser.portal.call(eventually, lambda: service.runs[run_id].in_flight is None)
    assert not waiting.done()
    return waiting


def stop_waiting(browser, connection, request):
    browser.portal.call(connection.revoke)
    with pytest.raises(HTTPException) as stopped:
        request.result(3)
    assert stopped.value.status_code == 410


@pytest.mark.parametrize("multi", [False, True])
def test_one_line_agent_profile_auth_scope_and_results(make_one_line_workspace, multi):
    from workspace_server.one_line_agent import RemoteExecCommand

    app, registry, manager, models = make_one_line_workspace([
        ModelResponse([ToolCall("local", "exec_command", {"cmd": "exit 7"})]),
        ModelResponse([AssistantMessage("The command exited with 7.")]),
    ], multi)
    prefix = "/w/alpha" if multi else ""
    with TestClient(app) as browser:
        assert browser.get(prefix + "/api/one-line-agent").status_code == 401
        binding = create_binding(browser, prefix)
        service = app.state.one_line_agent_endpoints[manager]
        assert binding["session_id"] is None
        assert binding["session_count"] == 0
        assert len(manager.list_sessions()) == 1
        assert "\n" not in binding["one_liners"]["linux"]
        assert "\n" not in binding["one_liners"]["windows"]
        assert service.token in binding["one_liners"]["linux"]
        assert service.token in binding["one_liners"]["windows"]
        assert len(service.token) == 12
        assert browser.post(prefix + "/api/one-line-agent", json={"profile": "remote"}).status_code == 409
        assert "One-line Agent" in browser.get(prefix + "/").text
        if multi:
            assert browser.post("/w/beta/api/one-line-agent", json={"profile": "remote"}).status_code == 200
        browser.cookies.clear()
        endpoint = prefix + "/b"
        assert browser.post(endpoint).status_code == 401
        headers = {"T": service.token + "1"}
        # A terminal token cannot access the workspace's ordinary APIs.
        assert browser.get(prefix + "/api/sessions", headers=headers).status_code == 401
        assert browser.post(prefix + "/api/sessions", json={}, headers=headers).status_code == 401
        assert browser.get(prefix + "/api/one-line-agent", headers=headers).status_code == 401
        response = browser.get(prefix + "/b", headers=headers, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"].startswith("/login?")
        for path in ("/b/", "/p", "/api/one-line-agent/step"):
            assert browser.post(prefix + path, headers=headers).status_code == 401
        if multi:
            assert browser.post("/w/beta/b", headers=headers).status_code == 401
        run_id, first = start_run(browser, service, b"Inspect this environment", prefix=prefix)
        connection = service.runs[run_id]
        agent = connection.runtime.agent
        assert agent.tool_registry.names() == ("exec_command",)
        assert set(RemoteExecCommand.input_schema["properties"]) == {"cmd"}
        assert agent.model_client.model == "remote-model"
        assert first.text == "exit 7"
        assert browser.post(
            endpoint, content=b"invalid result", headers=headers,
        ).status_code == 422
        prompt_text = repr(agent.model_client.prompts[0].input)
        assert "SERVER_ONLY_INSTRUCTIONS" not in prompt_text
        assert "<environment_context>" not in prompt_text
        assert str(agent.context_manager.cwd) not in prompt_text
        assert set(agent.model_client.prompts[0].turn_metadata) == {"turn_id"}
        assert "Selected profile instructions." in prompt_text
        assert "Selected profile user instructions." in prompt_text
        assert "client's OS, shell and directory are not supplied" in prompt_text
        assert [item.text for item in agent.history if isinstance(item, UserMessage)] == [
            "Inspect this environment",
        ]
        assert service.token not in prompt_text
        waiting = browser.portal.start_task_soon(
            service.exchange, run_id, b"command output\n\n[exit 7]",
        )
        wait_for_turn(browser, connection, 2)
        waiting = ack_final(browser, service, run_id, waiting, "The command exited with 7.")
        assert agent.model_client.call_count == 2
        result = next(item for item in agent.history if isinstance(item, ToolResult))
        assert result.output["output"] == "command output\n"
        assert result.output["exit_code"] == 7
        assert result.call_id == "local"
        tabs = manager._state_store.load_tabs()
        assert len(manager.list_sessions()) == 2
        assert len(tabs) == 1
        assert tabs[0]["profile"] is None
        browser.post("/login", json={"password": "owner-password"})
        binding = browser.get(prefix + "/api/one-line-agent").json()["binding"]
        assert binding["enabled"]
        assert binding["session_id"] == connection.session_id
        assert binding["session_count"] == 1
        stop_waiting(browser, connection, waiting)


def test_web_followups_keep_terminal_waiting_and_preserve_history(make_one_line_workspace):
    app, registry, manager, models = make_one_line_workspace([
        ModelResponse([AssistantMessage("First answer")]),
        ModelResponse([ToolCall("followup", "exec_command", {"cmd": "pwd"})]),
        ModelResponse([AssistantMessage("Second answer")]),
        ModelResponse([AssistantMessage("Third answer")]),
    ])
    with TestClient(app) as browser:
        create_binding(browser)
        service = app.state.one_line_agent_endpoints[manager]
        run_id = "devbox"
        waiting = browser.portal.start_task_soon(service.exchange, run_id, b"First task")
        browser.portal.call(eventually, lambda: (
            run_id in service.runs and service.runs[run_id].runtime is not None
        ))
        connection = service.runs[run_id]
        wait_for_turn(browser, connection, 1)
        waiting = ack_final(browser, service, run_id, waiting, "First answer")
        session_id = connection.session_id
        agent = connection.runtime.agent
        headers = {"T": service.token + run_id}
        second = browser.post("/api/session/message", json={
            "session_id": session_id, "prompt": "Follow up on that answer",
        })
        assert second.status_code == 200
        assert waiting.result(3) == "pwd"
        # During this turn, requests still have to supply an execution result.
        assert browser.post("/b", content=b"Another task", headers=headers).status_code == 422
        waiting = browser.portal.start_task_soon(service.exchange, run_id, b"/client\n[exit 0]")
        wait_for_turn(browser, connection, 3)
        waiting = ack_final(browser, service, run_id, waiting, "Second answer")
        third = browser.post("/api/session/message", json={
            "session_id": session_id, "prompt": "Summarize both tasks",
        })
        assert third.status_code == 200
        wait_for_turn(browser, connection, 4)
        waiting = ack_final(browser, service, run_id, waiting, "Third answer")
        assert service.runs[run_id] is connection
        assert connection.session_id == session_id
        assert connection.runtime.agent is agent
        assert [item.text for item in agent.history if isinstance(item, UserMessage)] == [
            "First task", "Follow up on that answer", "Summarize both tasks",
        ]
        assert "First answer" in repr(agent.model_client.prompts[1].input)
        assert "Second answer" in repr(agent.model_client.prompts[3].input)
        assert "/client" in repr(agent.model_client.prompts[3].input)
        sessions = manager.list_sessions()
        assert len(sessions) == 2
        assert next(item for item in sessions if item["id"] == session_id)["title"] == "OneLineAgent-devbox"
        assert browser.get("/api/one-line-agent").json()["binding"]["session_count"] == 1
        assert browser.delete("/api/sessions/" + session_id).status_code == 200
        with pytest.raises(HTTPException) as stopped:
            waiting.result(3)
        assert stopped.value.status_code == 410
        assert browser.get("/api/one-line-agent").json()["binding"]["session_count"] == 0
        waiting = browser.portal.start_task_soon(service.exchange, run_id, b"Fresh task")
        browser.portal.call(eventually, lambda: (
            service.runs[run_id] is not connection and service.runs[run_id].runtime is not None
        ))
        replacement = service.runs[run_id]
        wait_for_turn(browser, replacement, 1)
        waiting = ack_final(browser, service, run_id, waiting, "First answer")
        assert replacement.session_id != session_id
        assert replacement.runtime.agent is not agent
        assert [item.text for item in replacement.runtime.agent.history if isinstance(item, UserMessage)] == [
            "Fresh task",
        ]
        assert browser.get("/api/one-line-agent").json()["binding"]["session_count"] == 1
        stop_waiting(browser, replacement, waiting)


@pytest.mark.parametrize("tool_followup", [False, True])
def test_final_ack_does_not_become_a_task_or_next_tool_result(make_one_line_workspace, tool_followup):
    from workspace_server.one_line_agent import echo_command

    responses = [ModelResponse([AssistantMessage("First answer")])]
    if tool_followup:
        responses.append(ModelResponse([ToolCall("next", "exec_command", {"cmd": "pwd"})]))
    responses.append(ModelResponse([AssistantMessage("Next answer")]))
    app, registry, manager, models = make_one_line_workspace(responses)
    with TestClient(app) as browser:
        create_binding(browser)
        service = app.state.one_line_agent_endpoints[manager]
        run_id, first = start_run(browser, service, b"First task")
        assert first.text == echo_command("First answer")
        connection = service.runs[run_id]
        wait_for_turn(browser, connection, 1)
        headers = {"T": service.token + run_id}
        assert browser.post("/b", content=b"invalid ack", headers=headers).status_code == 422
        assert browser.post("/api/session/message", json={
            "session_id": connection.session_id, "prompt": "Next task",
        }).status_code == 200
        if tool_followup:
            browser.portal.call(eventually, lambda: connection.pending is not None)
            assert not connection.pending.done()
        else:
            wait_for_turn(browser, connection, 2)
        # Even a failed echo only acknowledges the display command. The next
        # turn's tool (if any) must wait for its own result.
        next_command = browser.post("/b", content=b"echo failed\n[exit 127]", headers=headers)
        assert next_command.status_code == 200
        if tool_followup:
            assert next_command.text == "pwd"
            assert not connection.pending.done()
            next_command = browser.post("/b", content=b"/client\n[exit 0]", headers=headers)
        assert next_command.text == echo_command("Next answer")
        calls = 3 if tool_followup else 2
        wait_for_turn(browser, connection, calls)
        waiting = browser.portal.start_task_soon(service.exchange, run_id, b"echo failed again\n[exit 1]")
        browser.portal.call(eventually, lambda: connection.in_flight is None)
        assert not waiting.done()
        agent = connection.runtime.agent
        assert agent.model_client.call_count == calls
        assert [item.text for item in agent.history if isinstance(item, UserMessage)] == [
            "First task", "Next task",
        ]
        assert [item.output["output"] for item in agent.history if isinstance(item, ToolResult)] == (
            ["/client"] if tool_followup else []
        )
        stop_waiting(browser, connection, waiting)


@pytest.mark.parametrize("shell", ["bash", "powershell"])
def test_final_echo_preserves_text_without_execution(tmp_path, shell):
    from workspace_server.one_line_agent import echo_command

    executable = (
        (shutil.which("powershell.exe") or shutil.which("pwsh"))
        if shell == "powershell" else shutil.which("bash")
    )
    if executable is None:
        pytest.skip(shell + " is required for terminal execution verification")
    args = [executable, "-NoLogo", "-NoProfile", "-Command"] if shell == "powershell" else [executable, "-c"]
    texts = [
        "", "-n", "-e",
        "完成 'quoted' \"double\" $literal\n第二行",
        "$(touch final-side-effect) `touch final-side-effect`",
        "$(Set-Content final-side-effect bad); | & #",
        "'; touch final-side-effect; echo '",
        "‘; Set-Content final-side-effect bad; echo ’",
        "’‘‚‛\"“”„'\\path\\n\\c\n",
    ]
    for text in texts:
        result = subprocess.run(
            args + [echo_command(text)], cwd=str(tmp_path), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
        )
        assert result.returncode == 0, result.stderr.decode()
        output = result.stdout.decode().replace("\r\n", "\n")
        assert output == "\n" + text + "\n"
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("disconnected", [False, True])
def test_idle_terminal_resume_replaces_waiter_and_preserves_history(make_one_line_workspace, disconnected):
    app, registry, manager, models = make_one_line_workspace([
        ModelResponse([AssistantMessage("First answer")]),
        ModelResponse([ToolCall("resumed", "exec_command", {"cmd": "echo resumed"})]),
        ModelResponse([AssistantMessage("Resumed answer")]),
    ])
    with TestClient(app) as browser:
        create_binding(browser)
        service = app.state.one_line_agent_endpoints[manager]
        original = browser.portal.start_task_soon(service.exchange, "1", b"First task")
        browser.portal.call(eventually, lambda: (
            "1" in service.runs and service.runs["1"].runtime is not None
        ))
        connection = service.runs["1"]
        agent = connection.runtime.agent
        session_id = connection.session_id
        wait_for_turn(browser, connection, 1)
        original = ack_final(browser, service, "1", original, "First answer")
        headers = {"T": service.token + "1"}
        for body in (b"", b" \n", b"\xff"):
            assert browser.post("/b", content=body, headers=headers).status_code == 422
        assert not original.done()
        if disconnected:
            original.cancel()  # Only the completed turn's HTTP waiter is disconnected.
        resumed = browser.post("/b", content=b"Continue the first task", headers=headers)
        assert resumed.status_code == 200 and resumed.text == "echo resumed"
        if not disconnected:
            with pytest.raises(HTTPException) as replaced:
                original.result(3)
            assert replaced.value.status_code == 409
            assert replaced.value.detail == "terminal connection was replaced"
        assert connection.runtime.agent is agent
        assert connection.session_id == session_id
        assert "First answer" in repr(agent.model_client.prompts[1].input)
        assert [item.text for item in agent.history if isinstance(item, UserMessage)] == [
            "First task", "Continue the first task",
        ]
        waiting = browser.portal.start_task_soon(service.exchange, "1", b"resumed\n[exit 0]")
        wait_for_turn(browser, connection, 3)
        waiting = ack_final(browser, service, "1", waiting, "Resumed answer")
        assert browser.get("/api/one-line-agent").json()["binding"]["session_count"] == 1
        stop_waiting(browser, connection, waiting)


def test_idle_resume_replaces_a_request_still_awaiting_submission(make_one_line_workspace, monkeypatch):
    from concurrent.futures import Future
    from workspace_server.sessions import ThreadedWorkspaceInteractiveSession

    release = Future()
    submit = ThreadedWorkspaceInteractiveSession.submit

    async def delayed_submit(self, prompt, *args, **kwargs):
        result = await submit(self, prompt, *args, **kwargs)
        if prompt == "First task":
            await asyncio.wrap_future(release)
        return result

    monkeypatch.setattr(ThreadedWorkspaceInteractiveSession, "submit", delayed_submit)
    app, registry, manager, models = make_one_line_workspace([
        ModelResponse([AssistantMessage("First answer")]),
        ModelResponse([ToolCall("resumed", "exec_command", {"cmd": "resumed command"})]),
    ])
    with TestClient(app) as browser:
        create_binding(browser)
        service = app.state.one_line_agent_endpoints[manager]
        first = browser.portal.start_task_soon(service.exchange, "1", b"First task")
        try:
            browser.portal.call(eventually, lambda: (
                "1" in service.runs and service.runs["1"].runtime is not None
            ))
            wait_for_turn(browser, service.runs["1"], 1)
            resumed = browser.post("/b", content=b"Continue", headers={"T": service.token + "1"})
            assert resumed.status_code == 200 and resumed.text == "resumed command"
        finally:
            release.set_result(None)
        with pytest.raises(HTTPException) as replaced:
            first.result(3)
        assert replaced.value.status_code == 409


def test_terminal_resume_does_not_replace_an_active_model_request(make_one_line_workspace, monkeypatch):
    from concurrent.futures import Future

    release = Future()
    complete = ScriptedModelClient.complete

    async def delayed_complete(self, prompt, event_handler):
        await asyncio.wrap_future(release)
        return await complete(self, prompt, event_handler)

    monkeypatch.setattr(ScriptedModelClient, "complete", delayed_complete)
    app, registry, manager, models = make_one_line_workspace([
        ModelResponse([ToolCall("first", "exec_command", {"cmd": "first command"})]),
    ])
    with TestClient(app) as browser:
        create_binding(browser)
        service = app.state.one_line_agent_endpoints[manager]
        first = browser.portal.start_task_soon(service.exchange, "1", b"First task")
        try:
            browser.portal.call(eventually, lambda: (
                "1" in service.runs and service.runs["1"].session_id is not None
                and service.runs["1"].runtime.is_busy
            ))
            response = browser.post("/b", content=b"Another task", headers={"T": service.token + "1"})
            assert response.status_code == 409
            assert not first.done()
        finally:
            release.set_result(None)
        assert first.result(3) == "first command"
        assert [item.text for item in service.runs["1"].runtime.agent.history if isinstance(item, UserMessage)] == [
            "First task",
        ]


def test_web_steer_waits_for_issued_command_and_keeps_terminal_connected(make_one_line_workspace):
    app, registry, manager, models = make_one_line_workspace([
        ModelResponse([ToolCall("first", "exec_command", {"cmd": "original command"})]),
        ModelResponse([ToolCall("steered", "exec_command", {"cmd": "revised command"})]),
        ModelResponse([AssistantMessage("Revised task complete")]),
    ])
    with TestClient(app) as browser:
        create_binding(browser)
        service = app.state.one_line_agent_endpoints[manager]
        run_id, first = start_run(browser, service, b"Original task")
        assert first.text == "original command"
        connection = service.runs[run_id]
        agent = connection.runtime.agent
        steered = browser.post("/api/session/message", json={
            "session_id": connection.session_id, "prompt": "Change the plan",
        })
        assert steered.status_code == 200
        assert not connection.pending.done()
        assert agent.model_client.call_count == 1
        second = browser.post("/b", content=b"original result\n[exit 0]", headers={"T": service.token + run_id})
        assert second.status_code == 200 and second.text == "revised command"
        assert "Change the plan" in repr(agent.model_client.prompts[1].input)
        assert "original result" in repr(agent.model_client.prompts[1].input)
        waiting = browser.portal.start_task_soon(service.exchange, run_id, b"revised result\n[exit 0]")
        wait_for_turn(browser, connection, 3)
        waiting = ack_final(browser, service, run_id, waiting, "Revised task complete")
        assert [item.text for item in agent.history if isinstance(item, UserMessage)] == [
            "Original task", "Change the plan",
        ]
        assert [item.output["output"] for item in agent.history if isinstance(item, ToolResult)] == [
            "original result", "revised result",
        ]
        stop_waiting(browser, connection, waiting)


@pytest.mark.parametrize("action", ["stop", "close"])
def test_disconnect_releases_pending_tool_and_stops_runtime(make_one_line_workspace, action):
    app, registry, manager, models = make_one_line_workspace([
        ModelResponse([ToolCall("waiting", "exec_command", {"cmd": "sleep 60"})]),
    ])
    with TestClient(app) as browser:
        create_binding(browser)
        service = app.state.one_line_agent_endpoints[manager]
        headers = {"T": service.token + "1"}
        run_id, _ = start_run(browser, service, b"Run task")
        endpoint = "/b"
        connection = service.runs[run_id]
        if action == "stop":
            assert browser.delete("/api/one-line-agent").status_code == 200
        else:
            assert browser.delete("/api/sessions/" + connection.session_id).status_code == 200
        browser.portal.call(
            eventually, lambda: connection.runtime.agent.is_shutdown,
        )
        assert connection.runtime.agent.model_client.call_count == 1
        if action == "close":
            binding = browser.get("/api/one-line-agent").json()["binding"]
            assert binding["enabled"] and binding["session_count"] == 0
            assert binding["session_id"] is None
            next_id, _ = start_run(browser, service, b"New task", run_id=run_id)
            replacement = service.runs[next_id]
            assert replacement.session_id != connection.session_id
            assert [item.text for item in replacement.runtime.agent.history if isinstance(item, UserMessage)] == [
                "New task",
            ]
        else:
            assert browser.post(endpoint, content=b"output\n[exit 0]", headers=headers).status_code == 404


def test_restart_keeps_rollout_without_restoring_temporary_tab(make_one_line_workspace):
    responses = [ModelResponse([AssistantMessage("Saved answer")])]
    app, registry, manager, models = make_one_line_workspace(responses)
    with TestClient(app) as browser:
        create_binding(browser)
        service = app.state.one_line_agent_endpoints[manager]
        waiting = browser.portal.start_task_soon(service.exchange, "1", b"Task to preserve")
        browser.portal.call(eventually, lambda: "1" in service.runs and service.runs["1"].runtime is not None)
        connection = service.runs["1"]
        wait_for_turn(browser, connection, 1)
        waiting = ack_final(browser, service, "1", waiting, "Saved answer")
        rollout = Path(connection.session.rollout_path())
        assert len(manager.list_sessions()) == 2
        stop_waiting(browser, connection, waiting)
    assert "Saved answer" in rollout.read_text(encoding="utf-8")
    app, registry, manager, models = make_one_line_workspace(responses)
    with TestClient(app) as browser:
        browser.post("/login", json={"password": "owner-password"})
        assert browser.get("/api/one-line-agent").json()["binding"] is None
        assert len(manager.list_sessions()) == 1
        assert manager.get()._session.runtime.agent.model_client.call_count == 0
        assert "Saved answer" in rollout.read_text(encoding="utf-8")


def test_invalid_input_does_not_admit_a_turn(make_one_line_workspace):
    app, registry, manager, models = make_one_line_workspace([
        ModelResponse([AssistantMessage("done")]),
    ])
    with TestClient(app) as browser:
        browser.post("/login", json={"password": "owner-password"})
        for payload in ({}, {"profile": "missing"}, {"profile": []}):
            assert browser.post("/api/one-line-agent", json=payload).status_code == 422
        create_binding(browser)
        service = app.state.one_line_agent_endpoints[manager]
        for token, body, expected in [
            (service.token, b"task", 422),
            (service.token[:-1], b"task", 401),
            ("!" + service.token[1:] + "1", b"task", 401),
            (service.token + "1", b"", 422),
            (service.token + "1", b" \n ", 422),
            (service.token + "1", b"\xff", 422),
            (service.token + "1", b"x" * (1024 * 1024 + 1), 413),
        ]:
            assert browser.post(
                "/b", content=body, headers={"T": token},
            ).status_code == expected
        assert not service.runs
        assert len(manager.list_sessions()) == 1


@pytest.mark.parametrize("after_command", [False, True])
def test_turn_failure_stays_visible_in_web_without_ending_terminal(make_one_line_workspace, after_command):
    responses = [ModelResponse([ToolCall("local", "exec_command", {"cmd": "pwd"})])] if after_command else []
    app, registry, manager, models = make_one_line_workspace(responses)
    with TestClient(app) as browser:
        create_binding(browser)
        service = app.state.one_line_agent_endpoints[manager]
        body = b"Task"
        if after_command:
            start_run(browser, service, body)
            body = b"/client\n[exit 0]"
        waiting = browser.portal.start_task_soon(service.exchange, "1", body)
        browser.portal.call(eventually, lambda: "1" in service.runs and service.runs["1"].runtime is not None)
        connection = service.runs["1"]
        wait_for_turn(browser, connection, 2 if after_command else 1)
        assert not waiting.done()
        assert "scripted model ran out of responses" in repr(connection.session.snapshot())
        connection.runtime.agent.model_client._responses = iter([
            ModelResponse([ToolCall("recovered", "exec_command", {"cmd": "recovered command"})]),
        ])
        assert browser.post("/api/session/message", json={
            "session_id": connection.session_id, "prompt": "Try again",
        }).status_code == 200
        assert waiting.result(3) == "recovered command"


@pytest.mark.parametrize("command", [
    "printf '%s\\n' '引号 | $text' | cat\nexit 1",
    "Write-Error '失败 | $text'\nexit 1",
])
def test_protocol_preserves_text_without_platform_wrappers(make_one_line_workspace, command):
    app, registry, manager, models = make_one_line_workspace([
        ModelResponse([
            AssistantMessage("准备检查 'quoted' $text。", phase="commentary"),
            ToolCall("local", "exec_command", {"cmd": command}),
        ]),
        ModelResponse([
            AssistantMessage("整理检查结果。", phase="commentary"),
            AssistantMessage("完成 'quoted' $text\n第二行"),
        ]),
    ])
    with TestClient(app) as browser:
        create_binding(browser)
        service = app.state.one_line_agent_endpoints[manager]
        headers = {"t": service.token + "1"}
        run_id, initial = start_run(browser, service, "检查环境".encode())
        connection = service.runs[run_id]
        assert initial.text == command
        assert initial.headers["content-type"] == "text/plain; charset=utf-8"
        assert browser.post("/b", content=b"\xff\n[exit 1]", headers=headers).status_code == 422
        waiting = browser.portal.start_task_soon(
            service.exchange, run_id, "错误 | 'quoted' $text\r\n\n[exit 1]".encode(),
        )
        wait_for_turn(browser, connection, 2)
        waiting = ack_final(browser, service, run_id, waiting, "完成 'quoted' $text\n第二行")
        history = connection.runtime.agent.history
        messages = [item.text for item in history if isinstance(item, AssistantMessage)]
        assert "准备检查 'quoted' $text。" in messages
        assert "整理检查结果。" in messages
        assert "完成 'quoted' $text\n第二行" in messages
        result = next(item for item in connection.runtime.agent.history if isinstance(item, ToolResult))
        assert result.output["output"] == "错误 | 'quoted' $text\r\n"
        assert result.output["exit_code"] == 1
        stop_waiting(browser, connection, waiting)


@pytest.mark.asyncio
@pytest.mark.parametrize("through_router, interrupted_response", [
    (False, False), (False, True), (True, False),
])
async def test_linux_one_liner_executes_only_on_client(
    make_one_line_workspace, tmp_path, through_router, interrupted_response,
):
    from contextlib import AsyncExitStack
    from pycodex_router.client import attach_router
    from pycodex_router.server import create_app as create_router

    if shutil.which("curl") is None or shutil.which("bash") is None:
        pytest.skip("Bash and curl are required")
    commentary = "检查环境，原样显示 $(touch commentary-side-effect) 与 'quotes'。"
    app, registry, manager, models = make_one_line_workspace([
        ModelResponse([
            AssistantMessage(commentary, phase="commentary"),
            AssistantMessage("Invisible message", visible=False),
            ToolCall("first", "exec_command", {
                "cmd": "printf '@client only\\n' >> artifact; printf '失败\\n' >&2; exit 7",
            }),
        ]),
        ModelResponse([
            AssistantMessage("第一步结束，读取文件。"),
            ToolCall("second", "exec_command", {"cmd": "cat artifact"}),
        ]),
        ModelResponse([
            AssistantMessage("检查完成，整理结果。", phase="commentary"),
            AssistantMessage("完成 'quoted' $literal\nSecond line"),
        ]),
    ], multi=through_router)
    client_dir = tmp_path / "client's directory"
    client_dir.mkdir()
    interrupted = False
    requests = 0

    async def interrupt_once(scope, receive, send):
        nonlocal interrupted, requests
        is_step = scope["type"] == "http" and scope["path"].endswith("/b")
        if is_step:
            requests += 1
        if is_step and not interrupted:
            interrupted = True

            async def broken_send(message):
                if message["type"] == "http.response.start":
                    message = dict(message, headers=[
                        (name, str(int(value) + 1).encode()) if name == b"content-length"
                        else (name, value) for name, value in message["headers"]
                    ])
                await send(message)

            # An incomplete HTTP response must never execute, even if its
            # received bytes happen to contain a complete shell fragment.
            await app(scope, receive, broken_send)
        else:
            await app(scope, receive, send)

    async with AsyncExitStack() as stack:
        if through_router:
            router = create_router("router-password")
            address, target = await stack.enter_async_context(serve_router(router, "/remote"))
            attach_router(app, registry, target, 6007)
            await stack.enter_async_context(serve(app))
            await eventually(lambda: app.state.router_client.connected)
            base = address + "/remote/n/" + app.state.router_client.node_id[:5] + "/w/alpha"
        else:
            address = await stack.enter_async_context(serve(interrupt_once if interrupted_response else app))
            base = address
        browser = await stack.enter_async_context(httpx.AsyncClient(trust_env=False))
        if through_router:
            assert (await browser.post(
                address + "/remote/login", json={"password": "router-password"},
            )).status_code == 200
            # Workspace authentication is independent of router authentication.
            login = address + "/remote/n/" + app.state.router_client.node_id[:5] + "/login"
        else:
            login = base + "/login"
        assert (await browser.post(login, json={"password": "owner-password"})).status_code == 200
        response = await browser.post(base + "/api/one-line-agent", json={"profile": "remote"})
        assert response.status_code == 200
        line = response.json()["binding"]["one_liners"]["linux"]
        assert base + "/b" in line
        assert "/b?" not in line
        service = app.state.one_line_agent_endpoints[manager]
        for index in range(2):
            if index:
                client_dir = tmp_path / "second client"
                client_dir.mkdir()
            process = await asyncio.create_subprocess_exec(
                "bash", "-c", line.replace(service.token + "1", service.token + str(index + 1)),
                cwd=str(client_dir),
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            if not (interrupted_response and index == 0):
                await eventually(lambda: (
                    len(service.connections()) == index + 1
                    and service.connections()[index].runtime.agent.model_client.call_count == 3
                    and not service.connections()[index].runtime.is_busy
                    and service.connections()[index].in_flight is None
                    and service.connections()[index].replies.empty()
                ))
                assert process.returncode is None
                assert (await browser.delete(
                    base + "/api/sessions/" + service.connections()[index].session_id,
                )).status_code == 200
            stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
            assert process.returncode == 0
            assert len(service.connections()) == index + 1
            agent = service.connections()[index].runtime.agent
            prompts = [item.text for item in agent.history if isinstance(item, UserMessage)]
            assert prompts == ["看一下这台机器"]
            if interrupted_response and index == 0:
                assert b"(18)" in stderr
                assert not stdout
                assert requests == 1
                assert agent.model_client.call_count == 1
                assert not (client_dir / "artifact").exists()
                continue
            displayed = stdout.decode()
            assert commentary not in displayed
            assert "Invisible message" not in displayed
            assert "完成 'quoted' $literal\nSecond line" in displayed
            assert (
                displayed.index("printf")
                < displayed.index("[exit 7]") < displayed.index("cat artifact")
                < displayed.index("[exit 0]")
            )
            assert b"(22)" in stderr  # Closing the session ends the held HTTP request.
            assert (client_dir / "artifact").read_text() == "@client only\n"
            assert not (client_dir / "commentary-side-effect").exists()
            assert not (tmp_path / "alpha" / "artifact").exists()
            results = [item.output for item in agent.history if isinstance(item, ToolResult)]
            assert [result["exit_code"] for result in results] == [7, 0]
            assert results[0]["output"] == "失败"
            assert results[1]["output"] == "@client only"
            messages = [item.text for item in agent.history if isinstance(item, AssistantMessage)]
            assert commentary in messages
            assert "完成 'quoted' $literal\nSecond line" in messages
        assert len({item.session_id for item in service.connections()}) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("restart_terminal", [False, True])
async def test_bash_one_liner_continues_from_web_or_terminal(make_one_line_workspace, tmp_path, restart_terminal):
    if shutil.which("curl") is None or shutil.which("bash") is None:
        pytest.skip("Bash and curl are required")
    app, registry, manager, models = make_one_line_workspace([
        ModelResponse([ToolCall("first", "exec_command", {"cmd": "printf first"})]),
        ModelResponse([AssistantMessage("First task complete")]),
        ModelResponse([ToolCall("second", "exec_command", {"cmd": "printf second"})]),
        ModelResponse([AssistantMessage("Second task complete")]),
    ])
    async with serve(app) as address:
        async with httpx.AsyncClient(base_url=address, trust_env=False) as browser:
            await browser.post("/login", json={"password": "owner-password"})
            response = await browser.post("/api/one-line-agent", json={"profile": "remote"})
            line = response.json()["binding"]["one_liners"]["linux"]
            service = app.state.one_line_agent_endpoints[manager]
            process = await asyncio.create_subprocess_exec(
                "bash", "-c", line, cwd=str(tmp_path),
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            processes = [process]
            try:
                await eventually(lambda: (
                    "1" in service.runs and service.runs["1"].runtime is not None
                    and service.runs["1"].runtime.agent.model_client.call_count == 2
                    and not service.runs["1"].runtime.is_busy
                    and service.runs["1"].in_flight is None
                    and service.runs["1"].replies.empty()
                ))
                connection = service.runs["1"]
                agent = connection.runtime.agent
                assert process.returncode is None
                if restart_terminal:
                    process = await asyncio.create_subprocess_exec(
                        "bash", "-c", line.replace("看一下这台机器", "继续刚才的任务"),
                        cwd=str(tmp_path), stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    )
                    processes.append(process)
                    await asyncio.wait_for(processes[0].wait(), 5)
                else:
                    response = await browser.post("/api/session/message", json={
                        "session_id": connection.session_id, "prompt": "继续刚才的任务",
                    })
                    assert response.status_code == 200
                await eventually(lambda: (
                    agent.model_client.call_count == 4 and not connection.runtime.is_busy
                    and connection.in_flight is None and connection.replies.empty()
                ))
                assert process.returncode is None
            finally:
                await browser.delete("/api/one-line-agent")
            outputs = await asyncio.wait_for(asyncio.gather(*(
                process.communicate() for process in processes
            )), 15)
            assert all(process.returncode == 0 for process in processes)
            assert all(b"(22)" in stderr for stdout, stderr in outputs)
            stdout = b"".join(stdout for stdout, stderr in outputs)
            assert "first\n[exit 0]" in stdout.decode()
            assert "second\n[exit 0]" in stdout.decode()
            assert "First task complete" in stdout.decode()
            assert "Second task complete" in stdout.decode()
        assert len(service.connections()) == 1
        agent = service.runs["1"].runtime.agent
        assert [item.text for item in agent.history if isinstance(item, UserMessage)] == [
            "看一下这台机器", "继续刚才的任务",
        ]
        assert [item.output["output"] for item in agent.history if isinstance(item, ToolResult)] == [
            "first", "second",
        ]


@pytest.mark.asyncio
async def test_powershell_one_liner_with_real_interpreter(make_one_line_workspace, tmp_path):
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is required for terminal execution verification")
    app, registry, manager, models = make_one_line_workspace([
        ModelResponse([
            AssistantMessage("先检查命令退出码。", phase="commentary"),
            ToolCall("native", "exec_command", {
                "cmd": "exit 7",
            }),
        ]),
        ModelResponse([ToolCall("error", "exec_command", {"cmd": "throw '失败'"})]),
        ModelResponse([ToolCall("file", "exec_command", {
            "cmd": """Set-Content -LiteralPath artifact -Value '中文 | "quoted" $literal'; Get-Content artifact""",
        })]),
        ModelResponse([AssistantMessage("完成 'quoted' $literal\nSecond line")]),
    ])
    client_dir = tmp_path / "client's directory"
    client_dir.mkdir()
    async with serve(app) as address:
        async with httpx.AsyncClient(trust_env=False) as browser:
            await browser.post(address + "/login", json={"password": "owner-password"})
            response = await browser.post(address + "/api/one-line-agent", json={"profile": "remote"})
        assert response.status_code == 200
        line = response.json()["binding"]["one_liners"]["windows"]
        process = await asyncio.create_subprocess_exec(
            powershell, "-NoLogo", "-NoProfile", "-Command", line,
            cwd=str(client_dir), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        service = app.state.one_line_agent_endpoints[manager]
        await eventually(lambda: (
            "1" in service.runs and service.runs["1"].runtime is not None
            and service.runs["1"].runtime.agent.model_client.call_count == 4
            and not service.runs["1"].runtime.is_busy
            and service.runs["1"].in_flight is None
            and service.runs["1"].replies.empty()
        ))
        assert process.returncode is None
        service.revoke()
        stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
        assert process.returncode != 0  # irm -ea Stop reports the revoked connection.
        assert "完成 'quoted' $literal\nSecond line" in stdout.decode().replace("\r\n", "\n")
        assert "先检查命令退出码。" not in stdout.decode()
        assert "[exit 7]" in stdout.decode()
        assert "[exit 1]" in stdout.decode()
        assert "[exit 0]" in stdout.decode()
        assert (client_dir / "artifact").is_file()
        assert not (tmp_path / "alpha" / "artifact").exists()
        results = [
            item.output for item in service.connections()[0].runtime.agent.history
            if isinstance(item, ToolResult)
        ]
        assert [result["exit_code"] for result in results] == [7, 1, 0]
        assert "失败" in results[1]["output"]
        assert '中文 | "quoted" $literal' in results[2]["output"]


def test_reload_keeps_existing_sessions_and_zero_argument_factories(make_one_line_workspace, monkeypatch):
    import workspace_server.reloader as reloader
    from workspace_server.workspaces import WorkspaceSessionManager

    app, registry, manager, models = make_one_line_workspace([
        ModelResponse([ToolCall("remote", "exec_command", {"cmd": "pwd"})]),
        ModelResponse([AssistantMessage("Reloaded task complete")]),
    ])
    with TestClient(app) as browser:
        ordinary = manager.get()
        factory = manager._session_factory
        assert "one_line_agent" not in vars(manager)
        assert "one_line_agent" not in vars(ordinary._session)
        assert not inspect.signature(factory).parameters
        # Simulate a workspace that predates the new session-registration method.
        monkeypatch.delattr(WorkspaceSessionManager, "attach_session")
        app.routes[:] = [route for route in app.routes if route.path not in {"/api/one-line-agent", "/b"}]
        modules = [
            (name, module) for name, module in reloader.loaded_reloadable_modules()
            if name.startswith("workspace_server.")
        ]
        monkeypatch.setattr(reloader, "loaded_reloadable_modules", lambda: modules)
        browser.post("/login", json={"password": "owner-password"})
        response = browser.post("/api/reload", json={})
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "reloaded"
        assert manager.get() is ordinary
        assert manager._session_factory is factory
        assert "one_line_agent" not in vars(manager)
        assert "one_line_agent" not in vars(ordinary._session)
        assert browser.get("/api/one-line-agent").json()["binding"] is None
        create_binding(browser)
        service = app.state.one_line_agent_endpoints[manager]
        browser.cookies.clear()
        headers = {"T": service.token + "1"}
        run_id, first = start_run(browser, service, b"Run after reload")
        assert first.text == "pwd"
        waiting = browser.portal.start_task_soon(service.exchange, run_id, b"/client\n[exit 0]")
        wait_for_turn(browser, service.runs[run_id], 2)
        waiting = ack_final(browser, service, run_id, waiting, "Reloaded task complete")
        browser.post("/login", json={"password": "owner-password"})
        response = browser.post("/api/reload", json={})
        assert response.status_code == 200, response.text
        assert browser.get("/api/one-line-agent").json()["binding"]["session_id"] == service.runs[run_id].session_id
        assert app.state.one_line_agent_endpoints[manager] is service
        next_id, _ = start_run(browser, service, b"Another task", run_id="2")
        assert service.runs[next_id].session_id != service.runs[run_id].session_id
        stop_waiting(browser, service.runs[run_id], waiting)


def test_multiple_running_sessions_keep_results_separate(make_one_line_workspace):
    app, registry, manager, models = make_one_line_workspace([
        ModelResponse([ToolCall("local", "exec_command", {"cmd": "pwd"})]),
        ModelResponse([AssistantMessage("done")]),
    ])
    with TestClient(app) as browser:
        create_binding(browser)
        service = app.state.one_line_agent_endpoints[manager]
        runs = {}
        for name in ("first", "second"):
            run_id, _ = start_run(browser, service, ("Task " + name).encode(), run_id=name)
            runs[name] = run_id
        first, second = service.runs[runs["first"]], service.runs[runs["second"]]
        assert first.session_id != second.session_id
        assert first.pending is not None and second.pending is not None
        waiting = {}
        for name, output in (("second", b"second output\n[exit 5]"), ("first", b"first output\n[exit 7]")):
            waiting[name] = browser.portal.start_task_soon(service.exchange, runs[name], output)
            wait_for_turn(browser, service.runs[runs[name]], 2)
            waiting[name] = ack_final(browser, service, runs[name], waiting[name], "done")
            if name == "second":
                assert not first.pending.done()
        for connection, name, code in ((first, "first", 7), (second, "second", 5)):
            history = connection.runtime.agent.history
            assert [item.output["output"] for item in history if isinstance(item, ToolResult)] == [name + " output"]
            assert [item.output["exit_code"] for item in history if isinstance(item, ToolResult)] == [code]
            assert [item.text for item in history if isinstance(item, UserMessage)] == [
                "Task " + name,
            ]
        binding = browser.get("/api/one-line-agent").json()["binding"]
        assert binding["enabled"] and binding["session_count"] == 2
        for name, request in waiting.items():
            stop_waiting(browser, service.runs[runs[name]], request)


@pytest.mark.asyncio
async def test_stop_drains_startup(make_one_line_workspace, monkeypatch):
    from starlette.exceptions import HTTPException

    app, registry, manager, models = make_one_line_workspace([ModelResponse([AssistantMessage("done")])])
    async with serve(app) as address:
        async with httpx.AsyncClient(base_url=address, trust_env=False) as browser:
            await browser.post("/login", json={"password": "owner-password"})
            await browser.post("/api/one-line-agent", json={"profile": "remote"})
            service = app.state.one_line_agent_endpoints[manager]
            entered, release = asyncio.Event(), asyncio.Event()
            attach = manager.attach_session

            async def delayed_attach(*args, **kwargs):
                entered.set()
                await release.wait()
                return await attach(*args, **kwargs)

            monkeypatch.setattr(manager, "attach_session", delayed_attach)
            first = asyncio.create_task(service.exchange("1", b"Task"))
            await asyncio.wait_for(entered.wait(), 3)
            try:
                # The same key cannot create another session while this one starts.
                duplicate = await browser.post(
                    address + "/b", content=b"Duplicate task", headers={"T": service.token + "1"},
                )
                assert duplicate.status_code == 409
                assert len(service.runs) == 1
                assert (await browser.delete("/api/one-line-agent")).status_code == 200
            finally:
                release.set()
            with pytest.raises(HTTPException) as stopped:
                await first
            assert stopped.value.status_code == 410
            connection = service.connections()[0]
            assert connection.runtime.agent.is_shutdown
            assert connection.runtime.agent.model_client.call_count == 0
