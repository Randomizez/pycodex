"""Run with pytest after applying the workspace reference adaptation."""
import sys
import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from pycodex import AssistantMessage, ContextConfig, ModelResponse, ToolRegistry
from pycodex.utils import uuid7_string
from pycodex.utils.session_persist import rollout_path_for_session
from tests.fakes import ScriptedModelClient, make_agent, make_runtime
from workspace_server import WorkspaceInteractiveSession, create_app
from workspace_server import notifications


@pytest.mark.parametrize("quiet", [False, True])
def test_runtime_event_saves_speech_in_chat_and_restores_without_replay(tmp_path, monkeypatch, quiet):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(notifications, "quiet_hours", lambda: quiet)
    directory = tmp_path / ".pycodex"
    directory.mkdir()
    delivery_log = tmp_path / "delivery.log"
    speech = "页面检查已完成，暂时无需操作。"
    config = {
        "enabled": True,
        "prepare_command": [sys.executable, "-c", "print(" + repr(speech) + ")"],
        "deliver_command": [sys.executable, "-c",
                            "from pathlib import Path; Path(" + repr(str(delivery_log)) +
                            ").write_text('called'); print('{\"state\":\"accepted\"}')"],
    }
    (directory / "notifications.json").write_text(json.dumps(config), encoding="utf-8")
    model = ScriptedModelClient([ModelResponse([AssistantMessage("页面检查通过。")])])

    def factory():
        session_id = uuid7_string()
        return WorkspaceInteractiveSession(make_runtime(make_agent(
            model, ToolRegistry(), ContextConfig(cwd=tmp_path),
            session_file_path=rollout_path_for_session(tmp_path / "codex", session_id),
            session_id=session_id,
        )))

    prefix = tmp_path / "workspace"
    with TestClient(create_app(factory, prefix)) as browser:
        response = browser.post("/api/session/message", json={"prompt": "检查页面"})
        assert response.json()["ok"]
        deadline = time.monotonic() + 10
        while True:
            snapshot = browser.get("/api/session").json()["snapshot"]
            records = snapshot.get("notifications") or []
            if records and records[0]["state"] in ("accepted", "quiet_hours"):
                break
            assert time.monotonic() < deadline, snapshot
            time.sleep(0.02)
        assert records[0]["speech"] == speech
        assert records[0]["state"] == ("quiet_hours" if quiet else "accepted")
        assert delivery_log.exists() is not quiet
        assert len(snapshot["turns"]) == 1
        assert snapshot["turns"][0]["response"] == "页面检查通过。"
        original_id = browser.get("/api/session").json()["session_id"]

    with TestClient(create_app(factory, prefix)) as browser:
        response = browser.get("/api/session").json()
        assert response["session_id"] != original_id
        assert response["snapshot"]["notifications"] == records
        assert model.call_count == 1


def test_steer_generates_only_after_final_response(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setattr(notifications, "quiet_hours", lambda: False)
    directory = tmp_path / ".pycodex"
    directory.mkdir()
    (directory / "notifications.json").write_text(json.dumps({
        "enabled": True, "prepare_command": ["summary"], "deliver_command": ["speaker"],
    }), encoding="utf-8")
    calls = []

    async def command(self, argv, payload, timeout):
        calls.append((argv, payload))
        if argv == ["summary"]:
            return "文档已补齐并核对，你暂时无需操作。"
        return '{"state":"accepted"}'

    monkeypatch.setattr(notifications.SessionNotifications, "command", command)
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]

    async def respond(prompt, call_count):
        index = call_count - 1
        entered[index].set()
        while not release[index].is_set():
            await asyncio.sleep(0.005)
        return ModelResponse([AssistantMessage(
            "正在读取文档。" if index == 0 else "已补齐记录并核对，无需操作。")])

    model = ScriptedModelClient(response_factory=respond)

    def factory():
        return WorkspaceInteractiveSession(make_runtime(make_agent(
            model, ToolRegistry(), ContextConfig(cwd=tmp_path))))

    with TestClient(create_app(factory, tmp_path / "workspace")) as browser:
        try:
            assert browser.post("/api/session/message", json={"prompt": "填写文档"}).json()["ok"]
            assert entered[0].wait(5)
            assert browser.post("/api/session/message", json={"prompt": "参考其他记录"}).json()["ok"]
            release[0].set()
            assert entered[1].wait(5)
            snapshot = browser.get("/api/session").json()["snapshot"]
            assert snapshot["running"]
            assert snapshot["notifications"] == []
            assert calls == []
            release[1].set()
            deadline = time.monotonic() + 5
            while True:
                snapshot = browser.get("/api/session").json()["snapshot"]
                if snapshot["notifications"] and snapshot["notifications"][-1]["state"] == "accepted":
                    break
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert len(snapshot["notifications"]) == 1
            assert [argv for argv, payload in calls] == [["summary"], ["speaker"]]
            assert calls[0][1]["latest_response"] == "已补齐记录并核对，无需操作。"
        finally:
            for event in release:
                event.set()
