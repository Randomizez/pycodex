import json
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from pycodex.bootstrap import ModelBootstrapResult
from tests.fakes import ScriptedModelClient
from workspace_server.app import build_parser, run_serve_cli
from workspace_server.owned_process import run_owned_server


@pytest.mark.parametrize("legacy", [False, True])
def test_cli_reads_config_from_a_file_and_uses_stdin_ownership_by_default(
    tmp_path, monkeypatch, legacy
):
    project = tmp_path / "项目"
    project.mkdir()
    model_config = tmp_path / "model.toml"
    model_config.write_text('model = "scripted"\n', encoding="utf-8")
    monkeypatch.setattr("workspace_server.app.build_model", lambda config, **kwargs: ModelBootstrapResult(ScriptedModelClient([])))
    workspace = {"id": "vscode", "work_dir": "项目"}
    workspace.update(
        {"board": False} if legacy else
        {"state_path": "workspaces.vscode", "enable_board": False}
    )
    payload = json.dumps({"workspaces": [workspace]}, ensure_ascii=False)
    config_path = tmp_path / "workspaces.json"
    config_path.write_text(payload, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    calls = []

    with tempfile.TemporaryFile(dir=str(tmp_path)) as stream:
        stream.write(b"owner pipe\n")
        stream.seek(0)
        monkeypatch.setattr("sys.stdin", stream)

        def serve(app, host, port, input_fd):
            calls.append((host, port))
            assert input_fd == stream.fileno()
            assert os.read(input_fd, 4096) == b"owner pipe\n"
            with TestClient(app) as client:
                workspaces = client.get("/api/workspaces").json()["workspaces"]
                assert len(workspaces) == 1
                assert workspaces[0]["work_dir"] == str(project)
                assert workspaces[0]["id"] == "vscode"
                assert workspaces[0]["state_path"] == str(tmp_path / "workspaces.vscode")
                assert workspaces[0]["enable_board"] is False
                assert len(client.get("/w/vscode/api/sessions").json()["sessions"]) == 1

        monkeypatch.setattr("workspace_server.owned_process.run_owned_server", serve)
        args = build_parser().parse_args(
            ["--workspace-config", str(config_path), "--listen", "127.0.0.1:0",
             "--config", str(model_config)]
        )
        assert run_serve_cli(args) == 0
    assert calls == [("127.0.0.1", 0)]
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "workspaces": [{
            "id": "vscode", "work_dir": "项目",
            "state_path": "workspaces.vscode", "enable_board": False,
        }]
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "model.toml", "workspaces.json", "workspaces.vscode.pycodex-ws.json", "项目"
    ]


def test_cli_exits_cleanly_when_stdin_is_closed_at_startup(tmp_path):
    config_path = tmp_path / "workspaces.json"
    config_path.write_text('{"workspaces": []}\n', encoding="utf-8")
    environment = os.environ.copy()
    environment["HOME"] = str(tmp_path)
    environment["CODEX_HOME"] = str(tmp_path / "codex")
    result = subprocess.run(
        [
            sys.executable, "-m", "workspace_server",
            "--listen", "127.0.0.1:0", "--workspace-config", str(config_path),
        ],
        cwd=str(Path(__file__).resolve().parents[3]),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    diagnostics = result.stderr.decode("utf-8", errors="replace")
    assert result.returncode == 0, diagnostics
    assert "Application startup complete." in diagnostics
    assert "Application shutdown complete." in diagnostics


@pytest.mark.skipif(os.name != "posix", reason="POSIX process signals")
@pytest.mark.parametrize("shutdown_signal", ["SIGINT", "SIGTERM"])
def test_signal_shutdown_with_stdin_open_exits_cleanly(tmp_path, shutdown_signal):
    code = """
import asyncio
import os
import signal
import sys
from contextlib import asynccontextmanager
from starlette.applications import Starlette
from workspace_server.owned_process import run_owned_server

@asynccontextmanager
async def lifespan(app):
    asyncio.get_event_loop().call_later(
        0.1, os.kill, os.getpid(), getattr(signal, sys.argv[1])
    )
    yield

run_owned_server(Starlette(lifespan=lifespan), "127.0.0.1", 0, sys.stdin.fileno())
"""
    environment = os.environ.copy()
    environment["HOME"] = str(tmp_path)
    environment["CODEX_HOME"] = str(tmp_path / "codex")
    with subprocess.Popen(
        [sys.executable, "-c", code, shutdown_signal],
        cwd=str(Path(__file__).resolve().parents[3]),
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    ) as process:
        # Keep the pipe open until interpreter shutdown; communicate() closes it.
        process.wait(timeout=10)
        diagnostics = process.stderr.read().decode("utf-8", errors="replace")
    signum = getattr(signal, shutdown_signal)
    assert process.returncode in (0, -signum, 128 + signum), diagnostics
    assert "Application shutdown complete." in diagnostics
    assert "Fatal Python error" not in diagnostics


@pytest.mark.parametrize("early_eof", [True, False])
def test_owner_eof_runs_lifespan_shutdown_even_when_it_precedes_startup(early_eof):
    events = []

    @asynccontextmanager
    async def lifespan(app):
        events.append("startup")
        if not early_eof:
            writer.close()
        yield
        events.append("shutdown")

    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd, "rb") as reader, os.fdopen(write_fd, "wb") as writer:
        if early_eof:
            writer.close()
        run_owned_server(Starlette(lifespan=lifespan), "127.0.0.1", 0, reader.fileno())
    assert events == ["startup", "shutdown"]
