import asyncio
import sys
import threading
from concurrent.futures import Future
from contextlib import ExitStack
import types

import pytest
from starlette.testclient import TestClient

import pycodex
from pycodex.events import IDLE_SLEEPING_STATUS
from pycodex.protocol import AssistantMessage, ModelResponse
from tests.fakes import ScriptedModelClient
from workspace_server import (
    WorkspaceDefinition,
    WorkspaceEntry,
    WorkspaceRegistry,
    WorkspaceSessionManager,
    create_app,
    create_multi_workspace_app,
)
from test_workspace_server import make_session, wait_snapshot


@pytest.fixture(params=[False, True], ids=["single", "multi"])
def reload_setup(request, tmp_path):
    multi = request.param

    def create(factory=None, password="admin-password"):
        session_factory = factory or (lambda: make_session(cwd=tmp_path))
        manager = WorkspaceSessionManager(session_factory, tmp_path / "state")
        if multi:
            entry = WorkspaceEntry(
                WorkspaceDefinition("first", tmp_path / "state", tmp_path, False),
                manager,
            )
            return create_multi_workspace_app(WorkspaceRegistry([entry]), password)
        return create_app(manager, tmp_path / "state", password, tmp_path, False)

    return create, "/w/first" if multi else ""


def test_reload_requires_admin_and_validates_payload(reload_setup):
    create, prefix = reload_setup
    with TestClient(create()) as browser:
        assert browser.post(prefix + "/api/reload", json={}).status_code == 401
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        assert browser.put(prefix + "/api/share", json={"password": "visitor-password"}).status_code == 200
        assert browser.post(prefix + "/api/reload", json={"session_id": 123}).status_code == 422

        browser.cookies.clear()
        assert browser.post(
            "/login", params={"next": prefix + "/"}, json={"password": "visitor-password"}
        ).status_code == 200
        visitor_cookie = browser.cookies["pycodex_ws_auth"]
        visitor = browser.post(prefix + "/api/reload", json={})
        assert visitor.status_code == 403
        assert visitor.json()["error"] == "administrator access required"

        # Admin reloads; share state and existing visitor cookie remain valid.
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        assert browser.post(prefix + "/api/reload", json={}).status_code == 200
        assert browser.get(prefix + "/api/share").json()["enabled"] is True
        browser.cookies.clear()
        assert browser.get(
            prefix + "/api/session", headers={"cookie": "pycodex_ws_auth=" + visitor_cookie}
        ).status_code == 200


def test_reload_applies_updated_module_in_place_and_keeps_sessions(
    reload_setup, tmp_path, monkeypatch
):
    module_file = tmp_path / "_reload_probe_mod.py"
    module_file.write_text(
        "class Greeter:\n"
        "    def reply(self, text):\n"
        "        return 'v1:' + text\n",
        encoding="utf-8",
    )
    search_paths = list(pycodex.__path__) + [str(tmp_path)]
    monkeypatch.setattr(pycodex, "__path__", search_paths)
    monkeypatch.setattr(
        pycodex.__spec__, "submodule_search_locations", search_paths
    )
    spec = __import__("importlib.util").util.spec_from_file_location(
        "pycodex._reload_probe_mod", module_file
    )
    probe_mod = __import__("importlib.util").util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "pycodex._reload_probe_mod", probe_mod)
    spec.loader.exec_module(probe_mod)
    greeter = probe_mod.Greeter()
    bound_reply = greeter.reply

    model = ScriptedModelClient(
        response_factory=lambda prompt, count: ModelResponse(
            [AssistantMessage(bound_reply(prompt.input[-1].text))]
        )
    )
    create, prefix = reload_setup
    with TestClient(create(lambda: make_session(model=model, cwd=tmp_path))) as browser:
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        first_id = browser.get(prefix + "/api/session").json()["session_id"]
        second_id = browser.post(prefix + "/api/sessions", json={"title": "Second"}).json()["session_id"]
        assert browser.post(
            prefix + "/api/session/message",
            json={"session_id": first_id, "prompt": "/effort high"},
        ).status_code == 200
        assert browser.post(
            prefix + "/api/session/message",
            json={"session_id": first_id, "prompt": "hello"},
        ).status_code == 200
        first_snap = wait_snapshot(
            browser,
            lambda snap: snap["turns"] and snap["turns"][-1]["response"] == "v1:hello",
            path=prefix + "/api/session?session_id=" + first_id,
        )
        assert any(s["id"] == first_id and s["unread"] for s in browser.get(prefix + "/api/sessions").json()["sessions"])

        module_file.write_text(
            "class Greeter:\n"
            "    def reply(self, text):\n"
            "        return 'v2:' + text\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "workspace_server.app.preflight_check",
            lambda: asyncio.sleep(0),
        )
        reloaded = browser.post(prefix + "/api/reload", json={"session_id": first_id})
        assert reloaded.status_code == 200
        payload = reloaded.json()
        assert payload["ok"] is True
        assert payload["status"] == "reloaded"
        assert payload["session_id"] == first_id
        assert [s["id"] for s in payload["sessions"]] == [first_id, second_id]
        assert payload["sessions"][0]["unread"] is True
        assert payload["snapshot"]["turns"][0]["submission_id"] == first_snap["turns"][0]["submission_id"]
        assert payload["snapshot"]["model"].endswith(":high")
        assert isinstance(greeter, probe_mod.Greeter)

        assert browser.post(
            prefix + "/api/session/message",
            json={"session_id": first_id, "prompt": "world"},
        ).status_code == 200
        updated_snap = wait_snapshot(
            browser,
            lambda snap: len(snap["turns"]) == 3 and snap["turns"][-1]["response"] == "v2:world",
            path=prefix + "/api/session?session_id=" + first_id,
        )
        assert [t["response"] for t in updated_snap["turns"][1:]] == ["v1:hello", "v2:world"]


def test_reload_preflight_failure_leaves_running_server_intact(
    reload_setup, tmp_path, monkeypatch
):
    bad_module = tmp_path / "broken_syntax_mod.py"
    bad_module.write_text("def broken(:\n", encoding="utf-8")
    dummy = types.ModuleType("pycodex.broken_syntax_mod")
    dummy.__file__ = str(bad_module)
    monkeypatch.setitem(sys.modules, "pycodex.broken_syntax_mod", dummy)

    create, prefix = reload_setup
    with TestClient(create()) as browser:
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        session_id = browser.get(prefix + "/api/session").json()["session_id"]
        response = browser.post(prefix + "/api/reload", json={"session_id": session_id})
        assert response.status_code == 400
        payload = response.json()
        assert payload["ok"] is False
        assert payload["status"] == "preflight_failed"
        assert "SyntaxError" in payload["error"]

        # Server and session continue working normally after preflight failure.
        assert browser.post(
            prefix + "/api/session/message",
            json={"session_id": session_id, "prompt": "still alive"},
        ).status_code == 200
        wait_snapshot(
            browser,
            lambda snap: snap["turns"] and snap["turns"][-1]["response"] == "done",
            path=prefix + "/api/session?session_id=" + session_id,
        )


def test_reload_waits_for_active_turn_to_finish(reload_setup, tmp_path, monkeypatch):
    release = Future()

    def finish():
        if not release.done():
            release.set_result(None)

    async def slow_response(prompt, count):
        if count == 1:
            await asyncio.wrap_future(release)
        return ModelResponse([AssistantMessage("answer-" + str(count))])

    model = ScriptedModelClient(response_factory=slow_response)
    create, prefix = reload_setup
    monkeypatch.setattr(
        "workspace_server.app.preflight_check",
        lambda: asyncio.sleep(0),
    )
    with ExitStack() as cleanup:
        browser = cleanup.enter_context(TestClient(create(lambda: make_session(model=model, cwd=tmp_path))))
        cleanup.callback(finish)
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        session_id = browser.get(prefix + "/api/session").json()["session_id"]
        assert browser.post(
            prefix + "/api/session/message",
            json={"session_id": session_id, "prompt": "busy-1"},
        ).status_code == 200
        wait_snapshot(
            browser,
            lambda snap: snap["running"] is True,
            path=prefix + "/api/session?session_id=" + session_id,
        )
        check = browser.post(
            prefix + "/api/reload", json={"session_id": session_id, "check_only": True}
        )
        assert check.status_code == 200
        assert check.json()["status"] == "waiting"
        assert check.json()["active_turns"] == 1
        assert len(check.json()["running_sessions"]) == 1

        timer = threading.Timer(0.1, release.set_result, args=(None,))
        timer.start()
        cleanup.callback(timer.join)
        cleanup.callback(timer.cancel)
        reloaded = browser.post(prefix + "/api/reload", json={"session_id": session_id})
        assert reloaded.status_code == 200
        assert reloaded.json()["ok"] is True
        assert reloaded.json()["snapshot"]["turns"][-1]["response"] == "answer-1"


def test_reload_ignores_idle_sleeping_and_rebuilds_messages_by_identity(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "workspace_server.app.preflight_check",
        lambda: asyncio.sleep(0),
    )
    manager = WorkspaceSessionManager(lambda: make_session(cwd=tmp_path), tmp_path / "state")
    app = create_app(manager, tmp_path / "state", "admin-password", tmp_path, False)
    with TestClient(app) as browser:
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        session_id = browser.get("/api/session").json()["session_id"]
        assert browser.post(
            "/api/session/message",
            json={"session_id": session_id, "prompt": "saved turn"},
        ).status_code == 200
        wait_snapshot(browser, lambda snap: len(snap["turns"]) == 1)
        session = manager.get(session_id)
        with session.view._lock:
            session.view._spinner_status = IDLE_SLEEPING_STATUS
            message = session.view._turns[0]["messages"][0]
            session.view._turns[0]["response"] = "stale cached response"
            session.view._turns[0]["messages"] = ()
        assert session.summary()["running"] is True
        assert session.summary()["spinner"] == IDLE_SLEEPING_STATUS
        check = browser.post("/api/reload", json={"session_id": session_id, "check_only": True})
        assert check.json()["status"] == "ready"
        reloaded = browser.post("/api/reload", json={"session_id": session_id})
        assert reloaded.status_code == 200
        restored_turn = reloaded.json()["snapshot"]["turns"][0]
        assert restored_turn["response"] == "done"
        assert restored_turn["kind"] == "assistant"
        assert restored_turn["messages"][0]["message_id"] == message.message_id
        assert restored_turn["messages"][0]["created_at"] == message.created_at


def test_reload_rolls_back_when_post_reload_session_sync_fails(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "workspace_server.app.preflight_check",
        lambda: asyncio.sleep(0),
    )
    module_file = tmp_path / "_rollback_probe_mod.py"
    module_file.write_text(
        "def tag(text):\n    return 'good:' + text\n",
        encoding="utf-8",
    )
    search_paths = list(pycodex.__path__) + [str(tmp_path)]
    monkeypatch.setattr(pycodex, "__path__", search_paths)
    monkeypatch.setattr(
        pycodex.__spec__, "submodule_search_locations", search_paths
    )
    spec = __import__("importlib.util").util.spec_from_file_location(
        "pycodex._rollback_probe_mod", module_file
    )
    probe_mod = __import__("importlib.util").util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "pycodex._rollback_probe_mod", probe_mod)
    spec.loader.exec_module(probe_mod)
    tag_fn = probe_mod.tag
    assert tag_fn("x") == "good:x"

    manager = WorkspaceSessionManager(lambda: make_session(cwd=tmp_path), tmp_path / "state")
    app = create_app(manager, tmp_path / "state", "admin-password", tmp_path, False)
    with TestClient(app) as browser:
        assert browser.post("/login", json={"password": "admin-password"}).status_code == 200
        session_id = browser.get("/api/session").json()["session_id"]
        assert browser.post(
            "/api/session/message",
            json={"session_id": session_id, "prompt": "before broken reload"},
        ).status_code == 200
        wait_snapshot(browser, lambda snap: len(snap["turns"]) == 1)

        module_file.write_text(
            "def tag(text):\n    return 'broken:' + text\n",
            encoding="utf-8",
        )
        session = manager.get(session_id)
        monkeypatch.setattr(
            session.view,
            "snapshot",
            lambda: (_ for _ in ()).throw(KeyError("missing_new_turn_field")),
        )
        failed = browser.post("/api/reload", json={"session_id": session_id})
        assert failed.status_code == 400
        payload = failed.json()
        assert payload["ok"] is False
        assert payload["status"] == "reload_failed"
        assert "KeyError" in payload["error"]
        # Reloaded function was rolled back in-place to its previous working code.
        assert tag_fn("x") == "good:x"
