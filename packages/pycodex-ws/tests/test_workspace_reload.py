import os
import subprocess
import sys
import asyncio
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from pycodex import AssistantMessage, ModelResponse
from tests.fakes import ScriptedModelClient
from workspace_server import (
    WorkspaceDefinition,
    WorkspaceEntry,
    WorkspaceRegistry,
    WorkspaceSessionManager,
    create_app,
    create_multi_workspace_app,
)
from workspace_server.owned_process import ReloadableApp
from test_workspace_server import make_session, wait_snapshot


@pytest.fixture
def no_module_swap(monkeypatch):
    async def passed():
        return None

    monkeypatch.setattr("workspace_server.owned_process.preflight_check", passed)
    monkeypatch.setattr("workspace_server.owned_process.purge_reloadable_modules", dict)
    monkeypatch.setattr("workspace_server.owned_process.restore_modules", lambda purged: None)


@pytest.fixture(params=[False, True], ids=["single", "multi"])
def reload_setup(request, tmp_path):
    multi = request.param
    builds = []

    def build():
        manager = WorkspaceSessionManager(lambda: make_session(cwd=tmp_path), tmp_path / "state")
        builds.append(manager)
        if multi:
            entry = WorkspaceEntry(
                WorkspaceDefinition("first", tmp_path / "state", tmp_path, False),
                manager,
            )
            return create_multi_workspace_app(WorkspaceRegistry([entry]), "admin-password")
        return create_app(manager, tmp_path / "state", "admin-password", tmp_path, False)

    return build, builds, "/w/first" if multi else ""


def test_reload_requires_admin_and_validates_payload(reload_setup, no_module_swap):
    build, _builds, prefix = reload_setup
    with TestClient(ReloadableApp(build)) as browser:
        assert browser.post(prefix + "/api/reload", json={}).status_code == 401
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        assert browser.put(prefix + "/api/share", json={"password": "visitor-password"}).status_code == 200
        assert browser.post(prefix + "/api/reload", json={"upgrade": 1}).status_code == 422

        browser.cookies.clear()
        assert browser.post(
            "/login", params={"next": prefix + "/"}, json={"password": "visitor-password"}
        ).status_code == 200
        visitor = browser.post(prefix + "/api/reload", json={})
        assert visitor.status_code == 403
        assert visitor.json()["error"] == "administrator access required"


def test_reload_is_unavailable_without_the_server_shell(reload_setup):
    build, _builds, prefix = reload_setup
    with TestClient(build()) as browser:
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        assert browser.post(prefix + "/api/reload", json={}).status_code == 503


def test_reload_restarts_app_and_restores_sessions_from_disk(reload_setup, no_module_swap):
    build, builds, prefix = reload_setup
    shell = ReloadableApp(build)
    with TestClient(shell) as browser:
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        assert browser.put(prefix + "/api/share", json={"password": "visitor-password"}).status_code == 200
        first_id = browser.get(prefix + "/api/session").json()["session_id"]
        browser.post(prefix + "/api/sessions", json={"title": "Second"})
        assert browser.post(
            prefix + "/api/session/message",
            json={"session_id": first_id, "prompt": "hello"},
        ).status_code == 200
        wait_snapshot(
            browser, lambda snap: len(snap["turns"]) == 1,
            path=prefix + "/api/session?session_id=" + first_id,
        )
        old_app = shell.app

        reloaded = browser.post(prefix + "/api/reload", json={})
        assert reloaded.json() == {"ok": True, "status": "reloaded"}
        assert shell.app is not old_app
        assert len(builds) == 2
        assert builds[0]._sessions == {}

        # Login state is process memory and is cleared like a restart.
        assert browser.get(prefix + "/api/sessions").status_code == 401
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        assert browser.get(prefix + "/api/share").json()["enabled"] is False
        sessions = browser.get(prefix + "/api/sessions").json()["sessions"]
        assert [session["title"] for session in sessions][1] == "Second"
        restored = browser.get(
            prefix + "/api/session?session_id=" + sessions[0]["id"]
        ).json()["snapshot"]
        assert restored["turns"][-1]["response"] == "done"


def test_reload_restarts_previous_code_when_new_build_fails(reload_setup, no_module_swap, monkeypatch):
    build, builds, prefix = reload_setup
    restored = []
    monkeypatch.setattr(
        "workspace_server.owned_process.restore_modules", restored.append,
    )
    failures = []

    def flaky_build():
        if len(builds) == 1 and not failures:
            failures.append(True)
            raise RuntimeError("new code is broken")
        return build()

    with TestClient(ReloadableApp(flaky_build)) as browser:
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        failed = browser.post(prefix + "/api/reload", json={})
        assert failed.status_code == 400
        assert failed.json()["error"] == "RuntimeError: new code is broken"
        assert restored == [{}]
        assert len(builds) == 2
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        assert len(browser.get(prefix + "/api/sessions").json()["sessions"]) == 1


def test_reload_preflight_failure_leaves_running_app_intact(reload_setup, monkeypatch):
    async def broken():
        raise RuntimeError("SyntaxError: invalid syntax")

    monkeypatch.setattr("workspace_server.owned_process.preflight_check", broken)
    build, builds, prefix = reload_setup
    shell = ReloadableApp(build)
    with TestClient(shell) as browser:
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        app = shell.app
        failed = browser.post(prefix + "/api/reload", json={})
        assert failed.status_code == 400
        assert "SyntaxError" in failed.json()["error"]
        assert shell.app is app and len(builds) == 1
        assert browser.get(prefix + "/api/sessions").status_code == 200


async def test_drain_finishes_running_turns_and_closes_idle_sessions(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, count):
        entered.set()
        await release.wait()
        return ModelResponse([AssistantMessage("finished")])

    sessions = []

    def session_factory():
        model = ScriptedModelClient(response_factory=respond) if sessions else None
        sessions.append(make_session(model=model, cwd=tmp_path))
        return sessions[-1]

    manager = WorkspaceSessionManager(session_factory, tmp_path / "state")
    await manager.start()
    busy_id = await manager.create_session(title="busy")
    idle, busy = manager.get(), manager.get(busy_id)
    assert (await busy.submit("work"))["ok"]
    await asyncio.wait_for(entered.wait(), 1)
    draining = asyncio.create_task(manager.close(True))
    await asyncio.sleep(0.05)
    assert idle.runtime.agent.is_shutdown
    assert not busy.runtime.agent.is_shutdown
    assert (await busy.submit("late"))["error"] == "agent is shutting down"
    with pytest.raises(ValueError, match="shutting down"):
        await manager.create_session()
    release.set()
    await asyncio.wait_for(draining, 2)
    assert busy.runtime.agent.is_shutdown
    assert busy.runtime.agent.history[-1] == AssistantMessage("finished")


def test_reload_upgrade_runs_pip_for_the_requested_version(reload_setup, no_module_swap, monkeypatch):
    calls = []

    async def upgrade(requirement):
        calls.append(requirement)

    monkeypatch.setattr("workspace_server.owned_process.upgrade_packages", upgrade)
    build, _builds, prefix = reload_setup
    with TestClient(ReloadableApp(build)) as browser:
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        assert browser.post(prefix + "/api/reload", json={"upgrade": True}).status_code == 200
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        assert browser.post(prefix + "/api/reload", json={"upgrade": "1.2.3"}).status_code == 200
    assert calls == ["pycodex-ws", "pycodex-ws==1.2.3"]


@pytest.mark.parametrize("pip_fails", [False, True])
async def test_upgrade_moves_the_locked_windows_launcher_aside(tmp_path, monkeypatch, pip_fails):
    from workspace_server import reloader

    launcher = tmp_path / "pycodex-ws.exe"
    launcher.write_bytes(b"running")

    async def pip(arguments):
        assert not launcher.exists()
        if pip_fails:
            raise RuntimeError("pip failed")
        launcher.write_bytes(b"new")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "pycodex-ws")])
    monkeypatch.setattr(reloader, "_run_python", pip)
    if pip_fails:
        with pytest.raises(RuntimeError, match="pip failed"):
            await reloader.upgrade_packages("pycodex-ws")
        assert launcher.read_bytes() == b"running"
        assert not (tmp_path / "pycodex-ws.exe.old").exists()
    else:
        await reloader.upgrade_packages("pycodex-ws")
        assert launcher.read_bytes() == b"new"
        assert (tmp_path / "pycodex-ws.exe.old").read_bytes() == b"running"


def test_reload_imports_backend_modules_again(tmp_path):
    code = """
import importlib
import sys
from starlette.testclient import TestClient
from workspace_server import reloader
from workspace_server.owned_process import ReloadableApp
import responses_server.transport

def build():
    app_module = importlib.import_module("workspace_server.app")
    args = app_module.build_parser().parse_args(sys.argv[1:])
    app = app_module.build_app(args)
    app.state.module = app_module
    return app

shell = ReloadableApp(build)
with TestClient(shell) as client:
    old_module = shell.app.state.module
    old_core = sys.modules["pycodex.runtime"]
    old_responses = sys.modules["responses_server.transport"]
    assert "responses_server" in reloader.RELOAD_PREFIXES
    response = client.post("/w/main/api/reload", json={})
    assert response.json() == {"ok": True, "status": "reloaded"}, response.text
    assert shell.app.state.module is not old_module
    assert sys.modules["pycodex.runtime"] is not old_core
    assert sys.modules.get("responses_server.transport") is not old_responses
    import responses_server.transport
    assert sys.modules["responses_server.transport"] is not old_responses

    # Also verify self-healing when an older process shell lacks "responses_server" in RELOAD_PREFIXES.
    stale_responses = sys.modules["responses_server.transport"]
    reloader.RELOAD_PREFIXES = tuple(
        p for p in reloader.RELOAD_PREFIXES if p != "responses_server"
    )
    response2 = client.post("/w/main/api/reload", json={})
    assert response2.json() == {"ok": True, "status": "reloaded"}, response2.text
    assert "responses_server" in reloader.RELOAD_PREFIXES
    import responses_server.transport
    assert sys.modules["responses_server.transport"] is not stale_responses
    assert client.get("/api/workspaces").json()["workspaces"][0]["id"] == "main"
"""
    config_path = tmp_path / "workspaces.json"
    config_path.write_text(
        '{"workspaces": [{"id": "main", "work_dir": ".", "state_path": "main", "enable_board": false}]}',
        encoding="utf-8",
    )
    model_config = tmp_path / "config.toml"
    model_config.write_text('model = "gpt-5.4"\n', encoding="utf-8")
    environment = os.environ.copy()
    environment["HOME"] = str(tmp_path)
    environment["CODEX_HOME"] = str(tmp_path / "codex")
    result = subprocess.run(
        [sys.executable, "-c", code, "--workspace-config", str(config_path),
         "--config", str(model_config)],
        cwd=str(Path(__file__).resolve().parents[3]),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout.decode("utf-8", errors="replace")
