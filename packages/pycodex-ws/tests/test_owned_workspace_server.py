import json
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from pycodex.bootstrap import ModelBootstrapResult
from tests.fakes import ScriptedModelClient
from workspace_server.app import build_parser, run_serve_cli
from workspace_server.owned_process import _wait_for_owner_eof, run_owned_server


@pytest.mark.parametrize("terminal, inputs", [
    (True, [b"\r\n", b"\x1a\r\n"]),
    (False, [b"\x1a", b"\r\n", BrokenPipeError()]),
    (False, [b"owner data", b""]),
])
def test_windows_owner_waits_on_native_handle(monkeypatch, terminal, inputs):
    pending = list(inputs)

    def read_file(handle, size):
        assert handle == 123 and size > 0
        item = pending.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, 0

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(get_osfhandle=lambda fd: 123))
    monkeypatch.setitem(sys.modules, "_winapi", SimpleNamespace(ReadFile=read_file))
    monkeypatch.setattr(os, "isatty", lambda fd: terminal)
    _wait_for_owner_eof(42)
    assert pending == []


@pytest.mark.skipif(sys.platform != "win32", reason="requires the Windows CRT and native pipes")
def test_idle_windows_owner_pipe_keeps_its_crt_descriptor_available():
    # A subprocess bounds the test even if a regression blocks os.dup on the CRT lock.
    code = '''
import _winapi
import os
import threading
from unittest.mock import patch
from workspace_server.owned_process import _wait_for_owner_eof

reader, writer = os.pipe()
reading = threading.Event()
read_file = _winapi.ReadFile

def read(handle, size):
    reading.set()
    return read_file(handle, size)

with patch.object(_winapi, "ReadFile", read):
    watcher = threading.Thread(target=_wait_for_owner_eof, args=(reader,), daemon=True)
    watcher.start()
    assert reading.wait(5)
    os.close(os.dup(reader))
    os.close(writer)
    watcher.join(5)
    assert not watcher.is_alive()
os.close(reader)
'''
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


@pytest.mark.parametrize("legacy", [False, True])
def test_cli_reads_config_from_a_file_and_uses_stdin_ownership_by_default(
    tmp_path, monkeypatch, legacy
):
    project = tmp_path / "项目"
    project.mkdir()
    model_config = tmp_path / "model.toml"
    model_config.write_text('model = "scripted"\n', encoding="utf-8")
    monkeypatch.setattr("workspace_server.sessions.build_model", lambda config, **kwargs: ModelBootstrapResult(ScriptedModelClient([])))
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

        def serve(build, host, port, input_fd, listen_to, workspaces):
            calls.append((host, port))
            assert listen_to == "key@router"
            assert [item.workspace_id for item in workspaces] == ["vscode"]
            assert input_fd == stream.fileno()
            assert os.read(input_fd, 4096) == b"owner pipe\n"
            with TestClient(build()) as client:
                workspaces = client.get("/api/workspaces").json()["workspaces"]
                assert len(workspaces) == 1
                assert workspaces[0]["work_dir"] == str(project)
                assert workspaces[0]["id"] == "vscode"
                assert workspaces[0]["state_path"] == str(tmp_path / "workspaces.vscode")
                assert workspaces[0]["enable_board"] is False
                assert len(client.get("/w/vscode/api/sessions").json()["sessions"]) == 1

        monkeypatch.setattr("workspace_server.owned_process.run_owned_server", serve)
        args = build_parser().parse_args(
            ["--workspace-config", str(config_path), "--listen-to", "key@router",
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

run_owned_server(lambda: Starlette(lifespan=lifespan), "127.0.0.1", 0, sys.stdin.fileno())
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
        run_owned_server(lambda: Starlette(lifespan=lifespan), "127.0.0.1", 0, reader.fileno())
    assert events == ["startup", "shutdown"]
