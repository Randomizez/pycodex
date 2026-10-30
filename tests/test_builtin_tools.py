import asyncio
import os
import shlex
import sys
import typing
from pathlib import Path

import pytest

from pycodex import AgentsManager, AssistantMessage, ContextConfig, ModelResponse
from pycodex.protocol import AgentMessage, ToolCall, ToolResult, UserMessage
from pycodex.tools.runtime_services import (
    PlanStore,
    RequestPermissionsManager,
)
from pycodex.tools import (
    ApplyPatchTool,
    ClockManager,
    ClockTool,
    CodeModeManager,
    ExecCommandTool,
    ExecTool,
    GrepFilesTool,
    ListDirTool,
    ReadFileTool,
    RequestPermissionsTool,
    RequestUserInputTool,
    ShellCommandTool,
    ShellTool,
    ToolContext,
    ToolRegistry,
    UnifiedExecManager,
    UpdatePlanTool,
    ViewImageTool,
    WaitAgentTool,
    WaitTool,
    WebSearchTool,
    WriteStdinTool,
)
from tests.fakes import (
    ScriptedModelClient, make_agent, make_runtime, wait_for_runtime_idle,
)


def make_registry() -> "ToolRegistry":
    registry = ToolRegistry()
    code_mode_manager = CodeModeManager(registry)
    manager = UnifiedExecManager()
    registry.register(ShellTool())
    registry.register(ShellCommandTool())
    registry.register(ExecCommandTool(manager))
    registry.register(WriteStdinTool(manager))
    registry.register(ExecTool(code_mode_manager))
    registry.register(WaitTool(code_mode_manager))
    registry.register(WebSearchTool())
    registry.register(UpdatePlanTool(PlanStore()))
    registry.register(ApplyPatchTool())
    registry.register(ListDirTool())
    registry.register(ReadFileTool())
    registry.register(GrepFilesTool())
    registry.register(ViewImageTool())
    return registry


def make_agent_tools(tmp_path):
    manager = AgentsManager(
        ScriptedModelClient([
            ModelResponse([AssistantMessage("done one")]),
            ModelResponse([AssistantMessage("done two")]),
        ]),
        ContextConfig(cwd=tmp_path, codex_home=tmp_path),
        toolset=(
            "spawn_agent", "send_message", "followup_task", "wait_agent",
            "interrupt_agent", "list_agents",
        ),
    )
    runtime = manager.create_root_runtime()
    context = ToolContext(
        turn_id="test", history=(), cwd=tmp_path, agent_id=runtime.agent.agent_id,
        encrypted_messages=False,
    )
    return runtime, context


@pytest.mark.asyncio
async def test_shell_tool_runs_argv_command_in_target_directory(tmp_path) -> "None":
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_1",
            name="shell",
            arguments={"command": ["bash", "-lc", "pwd"]},
        ),
        ToolContext(turn_id="turn_1", history=(), cwd=tmp_path),
    )

    assert result.is_error is False
    assert "Working directory:" in result.output
    assert str(Path(tmp_path).resolve()) in result.output
    assert "Exit code: 0" in result.output


@pytest.mark.asyncio
@pytest.mark.parametrize("workdir", [False, 0, [], {}])
async def test_invalid_workdir_does_not_execute_command(tmp_path, workdir):
    registry = ToolRegistry()
    registry.register(ShellTool())
    result = await registry.execute(
        ToolCall("invalid_path", "shell", {
            "command": [sys.executable, "-c", "open('unexpected.txt', 'w').close()"],
            "workdir": workdir,
        }),
        ToolContext("invalid_path", (), tmp_path),
    )

    assert result.is_error
    assert "TypeError" in result.output["error"]
    assert not (tmp_path / "unexpected.txt").exists()


@pytest.mark.asyncio
async def test_shell_tool_reports_timeout(tmp_path) -> "None":
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_2",
            name="shell",
            arguments={"command": ["bash", "-lc", "sleep 1"], "timeout_ms": 10},
        ),
        ToolContext(turn_id="turn_2", history=(), cwd=tmp_path),
    )

    assert result.is_error is False
    assert "Timeout: exceeded 10 ms" in result.output


@pytest.mark.asyncio
async def test_shell_command_tool_runs_shell_script(tmp_path) -> "None":
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_3",
            name="shell_command",
            arguments={"command": "pwd"},
        ),
        ToolContext(turn_id="turn_3", history=(), cwd=tmp_path),
    )

    assert result.is_error is False
    assert str(Path(tmp_path).resolve()) in result.output


@pytest.mark.asyncio
async def test_shell_command_tool_default_timeout_matches_schema(
    tmp_path,
    monkeypatch,
) -> "None":
    import pycodex.tools.shell_command_tool as shell_command_tool

    captured_timeouts = []
    original_wait_for = asyncio.wait_for

    async def capture_wait_for(awaitable, timeout):
        captured_timeouts.append(timeout)
        return await original_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(shell_command_tool.asyncio, "wait_for", capture_wait_for)

    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_shell_command_default_timeout",
            name="shell_command",
            arguments={"command": "printf ok"},
        ),
        ToolContext(turn_id="turn_shell_command_default_timeout", history=(), cwd=tmp_path),
    )

    assert result.is_error is False
    assert captured_timeouts[-1] == 10.0


@pytest.mark.asyncio
async def test_list_dir_tool_lists_entries_from_absolute_path(tmp_path) -> "None":
    (tmp_path / "a.txt").write_text("a")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "b.txt").write_text("b")
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_4",
            name="list_dir",
            arguments={"dir_path": str(tmp_path), "depth": 2},
        ),
        ToolContext(turn_id="turn_4", history=(), cwd=tmp_path),
    )

    assert result.is_error is False
    assert f"Absolute path: {tmp_path}" in result.output
    assert "a.txt" in result.output
    assert "sub/" in result.output
    assert "b.txt" in result.output


@pytest.mark.asyncio
async def test_read_file_tool_reads_slice_with_line_numbers(tmp_path) -> "None":
    file_path = tmp_path / "sample.py"
    file_path.write_text("first\nsecond\nthird\n")
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_5",
            name="read_file",
            arguments={"file_path": str(file_path), "offset": 2, "limit": 2},
        ),
        ToolContext(turn_id="turn_5", history=(), cwd=tmp_path),
    )

    assert result.is_error is False
    assert result.output == "L2: second\nL3: third"


@pytest.mark.asyncio
async def test_grep_files_tool_returns_matching_paths(tmp_path) -> "None":
    matched = tmp_path / "match.py"
    matched.write_text("hello world\n")
    unmatched = tmp_path / "skip.py"
    unmatched.write_text("goodbye\n")
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_6",
            name="grep_files",
            arguments={"pattern": "hello", "path": str(tmp_path)},
        ),
        ToolContext(turn_id="turn_6", history=(), cwd=tmp_path),
    )

    assert result.is_error is False
    assert "match.py" in result.output
    assert "skip.py" not in result.output


@pytest.mark.asyncio
async def test_exec_command_tool_schema_omits_skipped_approval_fields() -> "None":
    tool = ExecCommandTool(UnifiedExecManager())
    properties = tool.input_schema["properties"]

    assert "sandbox_permissions" not in properties
    assert "justification" not in properties
    assert "prefix_rule" not in properties


@pytest.mark.asyncio
async def test_exec_command_tool_clamps_yield_time_to_schema_range() -> "None":
    class RecordingManager:
        def __init__(self) -> "None":
            self.calls = []

        async def exec_command(self, **kwargs):
            self.calls.append(kwargs)
            return "ok"

    manager = RecordingManager()
    tool = ExecCommandTool(manager)
    context = ToolContext(turn_id="turn_exec_clamp", history=(), cwd=Path.cwd())

    result = await tool.run(context, {"cmd": "true", "yield_time_ms": 1})
    assert result == "ok"
    assert manager.calls[-1]["yield_time_ms"] == 250

    result = await tool.run(context, {"cmd": "true", "yield_time_ms": 999_999})
    assert result == "ok"
    assert manager.calls[-1]["yield_time_ms"] == 30_000


@pytest.mark.asyncio
async def test_exec_command_tool_passes_optional_max_output_tokens() -> "None":
    class RecordingManager:
        def __init__(self) -> "None":
            self.calls = []

        async def exec_command(self, **kwargs):
            self.calls.append(kwargs)
            return "ok"

    manager = RecordingManager()
    tool = ExecCommandTool(manager)
    context = ToolContext(turn_id="turn_exec_max_output", history=(), cwd=Path.cwd())

    await tool.run(context, {"cmd": "true"})
    assert manager.calls[-1]["max_output_tokens"] is None

    await tool.run(context, {"cmd": "true", "max_output_tokens": 0})
    assert manager.calls[-1]["max_output_tokens"] == 0


@pytest.mark.asyncio
async def test_exec_command_tool_returns_session_for_long_running_process(
    tmp_path,
) -> "None":
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_7",
            name="exec_command",
            arguments={
                "cmd": "printf start; sleep 1; printf end",
                "yield_time_ms": 10,
            },
        ),
        ToolContext(turn_id="turn_7", history=(), cwd=tmp_path),
    )

    assert result.is_error is False
    assert result.output.startswith("Chunk ID: ")
    assert "Command: " not in result.output
    assert "Process running with session ID" in result.output
    assert "Output:" in result.output
    marker = "Process running with session ID "
    session_id_text = result.output.split(marker, 1)[1].splitlines()[0]
    session_id = int(session_id_text)

    closed = await registry.execute(
        ToolCall(
            call_id="call_7_cleanup",
            name="write_stdin",
            arguments={
                "session_id": session_id,
                "yield_time_ms": 1_200,
            },
        ),
        ToolContext(turn_id="turn_7_cleanup", history=(), cwd=tmp_path),
    )

    assert closed.is_error is False
    assert "Process exited with code 0" in closed.output
    combined_output = result.output + closed.output
    assert "start" in combined_output
    assert "end" in combined_output


@pytest.mark.asyncio
async def test_write_stdin_tool_clamps_yield_time_to_schema_range() -> "None":
    class RecordingManager:
        def __init__(self) -> "None":
            self.calls = []

        async def write_stdin(self, **kwargs):
            self.calls.append(kwargs)
            return "ok"

    manager = RecordingManager()
    tool = WriteStdinTool(manager)
    context = ToolContext(turn_id="turn_write_clamp", history=(), cwd=Path.cwd())

    result = await tool.run(
        context,
        {"session_id": 1000, "chars": "x", "yield_time_ms": 1},
    )
    assert result == "ok"
    assert manager.calls[-1]["yield_time_ms"] == 250

    result = await tool.run(context, {"session_id": 1000, "chars": "x"})
    assert result == "ok"
    assert manager.calls[-1]["yield_time_ms"] == 250

    result = await tool.run(context, {"session_id": 1000, "yield_time_ms": 1})
    assert result == "ok"
    assert manager.calls[-1]["yield_time_ms"] == 5_000

    result = await tool.run(context, {"session_id": 1000, "yield_time_ms": 999_999})
    assert result == "ok"
    assert manager.calls[-1]["yield_time_ms"] == 300_000


@pytest.mark.asyncio
async def test_write_stdin_tool_passes_optional_max_output_tokens() -> "None":
    class RecordingManager:
        def __init__(self) -> "None":
            self.calls = []

        async def write_stdin(self, **kwargs):
            self.calls.append(kwargs)
            return "ok"

    manager = RecordingManager()
    tool = WriteStdinTool(manager)
    context = ToolContext(turn_id="turn_write_max_output", history=(), cwd=Path.cwd())

    await tool.run(context, {"session_id": 1000})
    assert manager.calls[-1]["max_output_tokens"] is None

    await tool.run(context, {"session_id": 1000, "max_output_tokens": 0})
    assert manager.calls[-1]["max_output_tokens"] == 0


@pytest.mark.asyncio
async def test_exec_command_notifies_hook_when_background_session_exits(
    tmp_path,
) -> "None":
    notifications = []
    notified = asyncio.Event()

    async def on_completed(event):
        notifications.append(event)
        notified.set()

    manager = UnifiedExecManager()
    manager.set_notify_hook(on_completed)
    output = await manager.exec_command(
        "printf done",
        cwd=tmp_path,
        yield_time_ms=1,
    )

    assert "Process running with session ID " in output
    await asyncio.wait_for(notified.wait(), timeout=1.0)
    assert len(notifications) == 1
    assert notifications[0]["type"] == "exec_command_completed"
    assert notifications[0]["session_id"] == 1000
    assert notifications[0]["exit_code"] == 0


@pytest.mark.asyncio
async def test_clock_tool_sets_cancels_and_validates_period() -> "None":
    manager = ClockManager()
    registry = ToolRegistry()
    registry.register(ClockTool(manager))
    context = ToolContext(turn_id="turn_clock", history=(), cwd=Path.cwd())

    enabled = await registry.execute(
        ToolCall(
            call_id="clock_set",
            name="clock",
            arguments={"period_m": 2},
        ),
        context,
    )
    assert enabled.output == {"enabled": True, "period_m": 2.0}
    assert enabled.is_error is False

    invalid = await registry.execute(
        ToolCall(
            call_id="clock_invalid",
            name="clock",
            arguments={"period_m": 0},
        ),
        context,
    )
    assert invalid.is_error is True
    assert "positive finite number or null" in str(invalid.output)

    disabled = await registry.execute(
        ToolCall(
            call_id="clock_cancel",
            name="clock",
            arguments={"period_m": None},
        ),
        context,
    )
    assert disabled.output == {"enabled": False, "period_m": None}
    assert disabled.is_error is False


@pytest.mark.asyncio
async def test_exec_manager_reports_running_sessions(tmp_path) -> "None":
    manager = UnifiedExecManager()
    output = await manager.exec_command(
        "sleep 0.2",
        cwd=tmp_path,
        yield_time_ms=1,
    )

    assert "Process running with session ID " in output
    assert manager.running_session_count() == 1

    await manager.write_stdin(1000, yield_time_ms=1_000)

    assert manager.running_session_count() == 0


@pytest.mark.asyncio
async def test_exec_command_starts_process_in_new_session(
    tmp_path, monkeypatch
) -> "None":
    original_create_subprocess_exec = asyncio.create_subprocess_exec
    captured_kwargs = {}

    async def capture_create_subprocess_exec(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return await original_create_subprocess_exec(*args, **kwargs)

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        capture_create_subprocess_exec,
    )
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_7_session",
            name="exec_command",
            arguments={"cmd": "printf ok"},
        ),
        ToolContext(turn_id="turn_7_session", history=(), cwd=tmp_path),
    )

    assert result.is_error is False
    assert "Process exited with code 0" in result.output
    assert "ok" in result.output
    assert captured_kwargs["start_new_session"] is True


@pytest.mark.asyncio
async def test_write_stdin_tool_reuses_running_session_and_returns_exit_metadata(
    tmp_path,
) -> "None":
    registry = make_registry()
    start = await registry.execute(
        ToolCall(
            call_id="call_8",
            name="exec_command",
            arguments={
                "cmd": 'read line; printf "hello:%s" "$line"',
                "yield_time_ms": 10,
            },
        ),
        ToolContext(turn_id="turn_8", history=(), cwd=tmp_path),
    )

    assert "Process running with session ID " in start.output
    marker = "Process running with session ID "
    session_id_text = start.output.split(marker, 1)[1].splitlines()[0]
    session_id = int(session_id_text)

    finish = await registry.execute(
        ToolCall(
            call_id="call_9",
            name="write_stdin",
            arguments={
                "session_id": session_id,
                "chars": "world\n",
                "yield_time_ms": 50,
            },
        ),
        ToolContext(turn_id="turn_9", history=(), cwd=tmp_path),
    )

    assert finish.is_error is False
    assert finish.output.startswith("Chunk ID: ")
    assert "Command: " not in finish.output
    assert "Original token count: " in finish.output
    if "Process exited with code 0" not in finish.output:
        finish = await registry.execute(
            ToolCall(
                call_id="call_9b",
                name="write_stdin",
                arguments={
                    "session_id": session_id,
                    "yield_time_ms": 200,
                },
            ),
            ToolContext(turn_id="turn_9b", history=(), cwd=tmp_path),
        )

    assert "Process exited with code 0" in finish.output
    assert "hello:world" in finish.output


@pytest.mark.asyncio
async def test_write_stdin_returns_when_child_keeps_stdout_open(tmp_path) -> "None":
    manager = UnifiedExecManager()
    pid_path = tmp_path / "stdout-holder.pid"
    child_pid = None
    command = "\n".join(
        [
            "{0} - <<'PY'".format(shlex.quote(sys.executable)),
            "import subprocess",
            "import sys",
            "import time",
            "print('ready', flush=True)",
            "time.sleep(0.2)",
            "child = subprocess.Popen(",
            "    [sys.executable, '-c', 'import time; time.sleep(10)'],",
            "    stdin=subprocess.DEVNULL,",
            "    stdout=sys.stdout,",
            "    stderr=sys.stderr,",
            ")",
            "open({0!r}, 'w').write(str(child.pid))".format(str(pid_path)),
            "print('parent exiting', flush=True)",
            "PY",
        ]
    )

    try:
        start = await manager.exec_command(command, cwd=tmp_path, yield_time_ms=10)
        marker = "Process running with session ID "
        assert marker in start
        session_id = int(start.split(marker, 1)[1].splitlines()[0])

        finish = await asyncio.wait_for(
            manager.write_stdin(session_id, yield_time_ms=1_000),
            timeout=3.0,
        )

        if pid_path.is_file():
            child_pid = int(pid_path.read_text(encoding="utf-8"))
        assert "Process exited with code 0" in finish
        assert "parent exiting" in finish
        assert manager.running_session_count() == 0
        assert await manager.write_stdin(
            session_id, yield_time_ms=1
        ) == "Error: session_id {0} is not running.".format(session_id)
    finally:
        if child_pid is None and pid_path.is_file():
            child_pid = int(pid_path.read_text(encoding="utf-8"))
        if child_pid is not None:
            try:
                os.kill(child_pid, 15)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
async def test_exec_command_tool_defaults_to_upstream_truncation_budget(
    tmp_path,
) -> "None":
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_9_default_truncation",
            name="exec_command",
            arguments={
                "cmd": (
                    "python3 -c \"print('HEAD' + 'A'*25000 + 'B'*25000 + 'TAIL')\""
                ),
                "yield_time_ms": 1_000,
            },
        ),
        ToolContext(turn_id="turn_9_default_truncation", history=(), cwd=tmp_path),
    )

    assert result.is_error is False
    assert "Process exited with code 0" in result.output
    assert "Original token count: " in result.output
    body = result.output.split("Output:\n", 1)[1]
    assert body.startswith("Total output lines: 1\n\nHEAD")
    assert "tokens truncated" in body
    assert body.rstrip().endswith("TAIL")


@pytest.mark.asyncio
async def test_write_stdin_tool_defaults_to_upstream_truncation_budget(
    tmp_path,
) -> "None":
    registry = make_registry()
    start = await registry.execute(
        ToolCall(
            call_id="call_9_write_start",
            name="exec_command",
            arguments={
                "cmd": 'read line; printf "%s" "$line"',
                "yield_time_ms": 10,
            },
        ),
        ToolContext(turn_id="turn_9_write_start", history=(), cwd=tmp_path),
    )

    marker = "Process running with session ID "
    session_id_text = start.output.split(marker, 1)[1].splitlines()[0]
    session_id = int(session_id_text)
    long_line = "HEAD" + ("A" * 25_000) + ("B" * 25_000) + "TAIL\n"

    poll = await registry.execute(
        ToolCall(
            call_id="call_9_write_finish",
            name="write_stdin",
            arguments={
                "session_id": session_id,
                "chars": long_line,
                "yield_time_ms": 100,
            },
        ),
        ToolContext(turn_id="turn_9_write_finish", history=(), cwd=tmp_path),
    )

    body = poll.output.split("Output:\n", 1)[1]
    for _ in range(5):
        if "tokens truncated" in body:
            break
        if "Process exited with code 0" in poll.output:
            break
        poll = await registry.execute(
            ToolCall(
                call_id="call_9_write_finish_poll",
                name="write_stdin",
                arguments={
                    "session_id": session_id,
                    "yield_time_ms": 200,
                },
            ),
            ToolContext(turn_id="turn_9_write_finish_poll", history=(), cwd=tmp_path),
        )
        body = poll.output.split("Output:\n", 1)[1]

    assert poll.is_error is False
    assert "Original token count: " in poll.output
    assert body.startswith("Total output lines: 1\n\nHEAD")
    assert "tokens truncated" in body
    assert body.endswith("TAIL")

    if "Process exited with code 0" not in poll.output:
        closed = await registry.execute(
            ToolCall(
                call_id="call_9_write_close",
                name="write_stdin",
                arguments={
                    "session_id": session_id,
                    "yield_time_ms": 200,
                },
            ),
            ToolContext(turn_id="turn_9_write_close", history=(), cwd=tmp_path),
        )
        assert "Process exited with code 0" in closed.output


@pytest.mark.asyncio
async def test_exec_command_unread_output_preserves_head_and_tail_when_capped(
    tmp_path,
) -> "None":
    registry = make_registry()
    start = await registry.execute(
        ToolCall(
            call_id="call_9_unread_buffer_start",
            name="exec_command",
            arguments={
                "cmd": (
                    'python3 -c "import sys,time; '
                    "sys.stdout.write('READY'); sys.stdout.flush(); "
                    "time.sleep(0.5); "
                    "sys.stdout.write('HEAD' + 'A'*700000 + 'MID' + 'B'*700000 + 'TAIL'); "
                    'sys.stdout.flush()"'
                ),
                "yield_time_ms": 10,
            },
        ),
        ToolContext(turn_id="turn_9_unread_buffer_start", history=(), cwd=tmp_path),
    )

    marker = "Process running with session ID "
    session_id_text = start.output.split(marker, 1)[1].splitlines()[0]
    session_id = int(session_id_text)

    await asyncio.sleep(0.9)

    finish = await registry.execute(
        ToolCall(
            call_id="call_9_unread_buffer_finish",
            name="write_stdin",
            arguments={
                "session_id": session_id,
                "yield_time_ms": 100,
                "max_output_tokens": 400_000,
            },
        ),
        ToolContext(turn_id="turn_9_unread_buffer_finish", history=(), cwd=tmp_path),
    )

    assert finish.is_error is False
    assert "Process exited with code 0" in finish.output
    body = finish.output.split("Output:\n", 1)[1]
    assert "Total output lines:" not in body
    assert body.startswith("HEAD") or body.startswith("READYHEAD")
    assert body.endswith("TAIL")
    assert "MID" not in body


@pytest.mark.asyncio
async def test_exec_tool_runs_python_and_returns_completed_status(
    tmp_path,
) -> "None":
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_exec_1",
            name="exec",
            arguments="text('EXEC_OK')",
            tool_type="custom",
        ),
        ToolContext(turn_id="turn_exec_1", history=(), cwd=tmp_path),
    )

    assert result.is_error is False
    assert "Script completed" in result.output_text()
    assert "EXEC_OK" in result.output_text()


@pytest.mark.asyncio
async def test_exec_tool_can_yield_and_wait_for_remaining_output(tmp_path) -> "None":
    registry = make_registry()
    start = await registry.execute(
        ToolCall(
            call_id="call_exec_2",
            name="exec",
            arguments=(
                '# @exec: {"yield_time_ms": 10}\n'
                "text('start')\n"
                "await asyncio.sleep(0.05)\n"
                "text('finish')"
            ),
            tool_type="custom",
        ),
        ToolContext(turn_id="turn_exec_2", history=(), cwd=tmp_path),
    )

    assert start.is_error is False
    assert "Script running with cell ID " in start.output_text()
    cell_id = (
        start.output_text().split("Script running with cell ID ", 1)[1].splitlines()[0]
    )

    finish = await registry.execute(
        ToolCall(
            call_id="call_wait_1",
            name="wait",
            arguments={"cell_id": cell_id, "yield_time_ms": 200},
        ),
        ToolContext(turn_id="turn_wait_1", history=(), cwd=tmp_path),
    )
    if "Script completed" not in finish.output_text():
        finish = await registry.execute(
            ToolCall(
                call_id="call_wait_2",
                name="wait",
                arguments={"cell_id": cell_id, "yield_time_ms": 200},
            ),
            ToolContext(turn_id="turn_wait_2", history=(), cwd=tmp_path),
        )

    assert finish.is_error is False
    assert "Script completed" in finish.output_text()
    assert "finish" in finish.output_text()


def test_web_search_tool_serializes_as_provider_native_spec(tmp_path) -> "None":
    registry = make_registry()
    web_search = registry.get_tool("web_search")
    assert web_search is not None
    assert web_search.serialize() == {
        "type": "web_search",
        "external_web_access": True,
        "search_content_types": ["text", "image"],
    }


@pytest.mark.asyncio
async def test_update_plan_tool_returns_confirmation_and_stores_plan(
    tmp_path,
) -> "None":
    plan_store = PlanStore()
    registry = ToolRegistry()
    registry.register(UpdatePlanTool(plan_store))

    result = await registry.execute(
        ToolCall(
            call_id="call_plan",
            name="update_plan",
            arguments={
                "explanation": "Start with discovery.",
                "plan": [
                    {"step": "Inspect repo", "status": "completed"},
                    {"step": "Implement tools", "status": "in_progress"},
                ],
            },
        ),
        ToolContext(turn_id="turn_plan", history=(), cwd=tmp_path),
    )

    assert result.is_error is False
    assert result.output == "Plan updated"
    assert plan_store.snapshot() == {
        "explanation": "Start with discovery.",
        "plan": [
            {"step": "Inspect repo", "status": "completed"},
            {"step": "Implement tools", "status": "in_progress"},
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("has_handler", [False, True])
async def test_request_user_input_tool_is_unavailable(has_handler) -> "None":
    registry = ToolRegistry()
    requested = []

    async def handler(payload):
        requested.append(payload)
        return {"answers": {}}

    if has_handler:
        registry.runtime_environment.request_user_input_manager.set_handler(handler)
    registry.register(RequestUserInputTool())

    result = await registry.execute(
        ToolCall(
            call_id="call_request_user_input",
            name="request_user_input",
            arguments={
                "questions": [
                    {
                        "id": "choice",
                        "header": "Select",
                        "question": "Pick one",
                        "options": [
                            {
                                "label": "Use tool A (Recommended)",
                                "description": "Fast path",
                            },
                            {
                                "label": "Use tool B",
                                "description": "Slow path",
                            },
                        ],
                    }
                ]
            },
        ),
        ToolContext(turn_id="turn_request_user_input", history=(), cwd=Path.cwd()),
    )

    assert result.is_error is False
    assert result.success is None
    assert requested == []
    assert result.output == "request_user_input is unavailable in Default mode"
    assert result.serialize() == {
        "type": "function_call_output",
        "call_id": "call_request_user_input",
        "output": "request_user_input is unavailable in Default mode",
    }


@pytest.mark.asyncio
async def test_request_permissions_tool_returns_permission_response() -> "None":
    manager = RequestPermissionsManager()

    async def handler(payload):
        assert payload["reason"] == "Need network"
        return {
            "permissions": payload["permissions"],
            "scope": "turn",
        }

    manager.set_handler(handler)
    registry = ToolRegistry()
    registry.register(RequestPermissionsTool(manager))

    result = await registry.execute(
        ToolCall(
            call_id="call_request_permissions",
            name="request_permissions",
            arguments={
                "reason": "Need network",
                "permissions": {
                    "network": {"enabled": True},
                },
            },
        ),
        ToolContext(turn_id="turn_request_permissions", history=(), cwd=Path.cwd()),
    )

    assert result.is_error is False
    assert result.output == {
        "permissions": {
            "network": {"enabled": True},
        },
        "scope": "turn",
    }


@pytest.mark.asyncio
async def test_apply_patch_tool_applies_multiple_operations_atomically(
    tmp_path,
) -> "None":
    target = tmp_path / "modify.txt"
    doomed = tmp_path / "delete.txt"
    target.write_text("line1\nline2\n")
    doomed.write_text("obsolete\n")

    patch = "\n".join(
        [
            "*** Begin Patch",
            "*** Add File: nested/new.txt",
            "+created",
            "*** Delete File: delete.txt",
            "*** Update File: modify.txt",
            "@@",
            "-line2",
            "+changed",
            "*** End Patch",
        ]
    )
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_10",
            name="apply_patch",
            arguments=patch,
            tool_type="custom",
        ),
        ToolContext(turn_id="turn_10", history=(), cwd=tmp_path),
    )

    assert result.is_error is False
    assert result.tool_type == "custom"
    assert result.output == (
        "Exit code: 0\n"
        "Wall time: 0 seconds\n"
        "Output:\n"
        "Success: A nested/new.txt M modify.txt D delete.txt\n"
    )
    assert (tmp_path / "nested" / "new.txt").read_text() == "created\n"
    assert target.read_text() == "line1\nchanged\n"
    assert doomed.exists() is False


@pytest.mark.asyncio
async def test_apply_patch_tool_does_not_leave_partial_writes_on_failure(
    tmp_path,
) -> "None":
    patch = "\n".join(
        [
            "*** Begin Patch",
            "*** Add File: created.txt",
            "+hello",
            "*** Update File: missing.txt",
            "@@",
            "-old",
            "+new",
            "*** End Patch",
        ]
    )
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_11",
            name="apply_patch",
            arguments=patch,
            tool_type="custom",
        ),
        ToolContext(turn_id="turn_11", history=(), cwd=tmp_path),
    )

    assert result.is_error is False
    assert result.output.startswith("Exit code: 1\nWall time: 0 seconds\nOutput:\n")
    assert "apply_patch verification failed" in result.output
    assert (tmp_path / "created.txt").exists() is False


@pytest.mark.asyncio
async def test_apply_patch_tool_allows_paths_outside_workspace(tmp_path) -> "None":
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    workspace.mkdir()

    patch = "\n".join(
        [
            "*** Begin Patch",
            "*** Add File: ../outside.txt",
            "+outside",
            "*** End Patch",
        ]
    )
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_apply_outside",
            name="apply_patch",
            arguments=patch,
            tool_type="custom",
        ),
        ToolContext(turn_id="turn_apply_outside", history=(), cwd=workspace),
    )

    assert result.is_error is False
    assert result.output == (
        "Exit code: 0\n" "Wall time: 0 seconds\n" "Output:\n" f"Success: A {outside}\n"
    )
    assert outside.read_text() == "outside\n"


@pytest.mark.asyncio
async def test_apply_patch_tool_allows_absolute_paths_outside_workspace(
    tmp_path,
) -> "None":
    workspace = tmp_path / "workspace"
    outside = tmp_path / "absolute-outside.txt"
    workspace.mkdir()

    patch = "\n".join(
        [
            "*** Begin Patch",
            f"*** Add File: {outside}",
            "+absolute outside",
            "*** End Patch",
        ]
    )
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_apply_absolute_outside",
            name="apply_patch",
            arguments=patch,
            tool_type="custom",
        ),
        ToolContext(turn_id="turn_apply_absolute_outside", history=(), cwd=workspace),
    )

    assert result.is_error is False
    assert result.output == (
        "Exit code: 0\n" "Wall time: 0 seconds\n" "Output:\n" f"Success: A {outside}\n"
    )
    assert outside.read_text() == "absolute outside\n"


@pytest.mark.asyncio
async def test_view_image_tool_returns_structured_input_image_output(
    tmp_path,
) -> "None":
    target = tmp_path / "images"
    target.mkdir()
    image_path = target / "pixel.png"
    image_path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a"
            "0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63f8cfc0f01f00050001ff89993d1d"
            "0000000049454e44ae426082"
        )
    )
    registry = make_registry()
    result = await registry.execute(
        ToolCall(
            call_id="call_12",
            name="view_image",
            arguments={"path": image_path.name},
        ),
        ToolContext(turn_id="turn_12", history=(), cwd=target),
    )

    assert result.is_error is False
    assert result.content_items == (
        {
            "type": "input_image",
            "image_url": result.output["image_url"],
            "detail": "high",
        },
    )
    assert result.output["detail"] == "high"
    serialized = result.serialize()
    assert serialized["type"] == "function_call_output"
    assert serialized["output"][0]["type"] == "input_image"
    assert str(result.output["image_url"]).startswith("data:image/png;base64,")

    original = await registry.execute(
        ToolCall(
            call_id="call_12_original",
            name="view_image",
            arguments={"path": str(image_path), "detail": "original"},
        ),
        ToolContext(turn_id="turn_12_original", history=(), cwd=tmp_path),
    )

    assert original.is_error is False
    assert original.output["detail"] == "original"
    assert original.content_items == (
        {
            "type": "input_image",
            "image_url": original.output["image_url"],
            "detail": "original",
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("pillow_available", [True, False])
@pytest.mark.parametrize("detail", [None, "high", "original"])
async def test_view_image_tool_handles_oversized_images(
    tmp_path, monkeypatch, pillow_available, detail
) -> "None":
    from base64 import b64decode
    from io import BytesIO

    from PIL import Image

    from pycodex.utils.image_utils import MAX_DIMENSION

    image_path = tmp_path / "wide.png"
    Image.new("RGB", (MAX_DIMENSION * 2, MAX_DIMENSION), "red").save(image_path)

    registry = make_registry()
    arguments = {"path": str(image_path)}
    if detail is not None:
        arguments["detail"] = detail
    with monkeypatch.context() as patch:
        if not pillow_available:
            patch.setitem(sys.modules, "PIL", None)
        result = await registry.execute(
            ToolCall(
                call_id="call_resize",
                name="view_image",
                arguments=arguments,
            ),
            ToolContext(turn_id="turn_resize", history=(), cwd=tmp_path),
        )

    resized = pillow_available and detail != "original"
    expected_detail = "high" if resized else "original"
    expected_size = (
        (MAX_DIMENSION, MAX_DIMENSION // 2)
        if resized else (MAX_DIMENSION * 2, MAX_DIMENSION)
    )
    assert result.is_error is False
    assert result.output["detail"] == expected_detail
    assert result.content_items[0]["detail"] == expected_detail
    assert result.serialize()["output"][0]["detail"] == expected_detail
    prefix = "data:image/png;base64,"
    image_url = str(result.output["image_url"])
    assert image_url.startswith(prefix)
    image_bytes = b64decode(image_url[len(prefix) :])
    with Image.open(BytesIO(image_bytes)) as image:
        assert image.size == expected_size
    if not resized:
        assert image_bytes == image_path.read_bytes()


@pytest.mark.asyncio
async def test_agent_tools_deliver_messages_and_followup_tasks(tmp_path):
    root, context = make_agent_tools(tmp_path)
    registry = root.agent.tool_registry
    manager = root.agents_manager
    try:
        spawned = await registry.execute(
            ToolCall("spawn", "spawn_agent", {
                "task_name": "worker", "message": "hello", "fork_turns": "none",
            }), context,
        )
        assert spawned.output == {"task_name": "/root/worker"}
        child = manager.get_runtime(
            manager._resolve_agent_id(root.agent.agent_id, "/root/worker")
        )
        await wait_for_runtime_idle(child)
        assert child.agent.tool_registry.names() == registry.names()
        assert child.agent.model_client._session_id == child.agent.agent_id
        assert child.agent.model_client._openai_subagent == "collab_spawn"
        assert child.agent.session_file_path is None
        sent = await registry.execute(
            ToolCall("send", "send_message", {"target": "worker", "message": "note"}),
            context,
        )
        assert sent.output == ""
        await asyncio.sleep(0)
        assert child.agent.model_client.call_count == 1
        assert not child.is_busy

        followed = await registry.execute(
            ToolCall("followup", "followup_task", {
                "target": child.agent.agent_id, "message": "second task",
            }), context,
        )
        assert followed.output == ""
        await wait_for_runtime_idle(child)
        messages = [
            item for item in child.agent.history if isinstance(item, AgentMessage)
        ]
        assert [item.text.rsplit("Payload:\n", 1)[1] for item in messages] == [
            "hello", "note", "second task",
        ]
        assert all(item.author == "/root" for item in messages)
        assert all(item.recipient == "/root/worker" for item in messages)
        assert "Message Type: NEW_TASK" in messages[-1].text
        assert "Message Type: MESSAGE" in messages[1].text

        waited = await registry.execute(ToolCall("wait", "wait_agent", {}), context)
        assert waited.output == {"message": "Wait completed.", "timed_out": False}
        assert len(root._steer_queue) == 2
        listed = await registry.execute(
            ToolCall("list", "list_agents", {"path_prefix": "worker"}), context,
        )
        assert listed.output == {"agents": [{
            "agent_name": "/root/worker", "agent_status": {"completed": "done two"},
        }]}
        interrupted = await registry.execute(
            ToolCall("interrupt", "interrupt_agent", {"target": "worker"}), context,
        )
        assert interrupted.output == {"previous_status": {"completed": "done two"}}
        assert child.accepts_input
    finally:
        await root.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments, error", [
    ({}, "task_name"),
    ({"task_name": "worker"}, "message"),
    ({"task_name": "root", "message": "task"}, "reserved"),
    ({"task_name": "Bad/Name", "message": "task"}, "lowercase"),
    ({"task_name": "worker", "message": " "}, "Empty message"),
    ({"task_name": "worker", "message": "task", "fork_turns": "some"}, "fork_turns"),
    ({"task_name": "worker", "message": "task", "fork_context": True}, "unexpected"),
    ({"task_name": "worker", "message": "task", "items": []}, "unexpected"),
])
async def test_spawn_agent_rejects_invalid_tasks(tmp_path, arguments, error):
    root, context = make_agent_tools(tmp_path)
    try:
        result = await root.agent.tool_registry.execute(
            ToolCall("invalid", "spawn_agent", arguments), context,
        )
        assert result.is_error
        assert error in str(result.output)
        assert len(root.agents_manager.list_agents(root.agent.agent_id)) == 1
    finally:
        await root.close()


@pytest.mark.asyncio
async def test_wait_agent_tool_clamps_timeout_to_schema_range() -> "None":
    class RecordingManager:
        def __init__(self) -> "None":
            self.calls = []

        async def wait_for_activity(self, agent_id, timeout_ms):
            self.calls.append((agent_id, timeout_ms))
            return {"message": "Wait timed out.", "timed_out": True}

    manager = RecordingManager()
    tool = WaitAgentTool(manager)
    context = ToolContext(
        turn_id="turn_wait_agent_clamp", history=(), cwd=Path.cwd(), agent_id="root-id",
    )

    result = await tool.run(context, {"timeout_ms": 1})
    assert result["timed_out"] is True
    assert "clamped to the minimum of 10000ms" in result["message"]
    assert manager.calls[-1] == ("root-id", 10_000)

    result = await tool.run(context, {})
    assert result == {"message": "Wait timed out.", "timed_out": True}
    assert manager.calls[-1] == ("root-id", 30_000)

    for timeout in (True, 10000.0, "10000", None):
        with pytest.raises(ValueError, match="must be an integer"):
            await tool.run(context, {"timeout_ms": timeout})
    with pytest.raises(ValueError, match="at most 3600000"):
        await tool.run(context, {"timeout_ms": 3_600_001})
    with pytest.raises(ValueError, match="unexpected"):
        await tool.run(context, {"ids": ["old-agent-id"]})
    assert len(manager.calls) == 2


@pytest.mark.asyncio
async def test_spawn_agent_fork_projects_aborted_results_without_mutating_history(tmp_path):
    root, context = make_agent_tools(tmp_path)

    paired_call = ToolCall(
        call_id="call_paired",
        name="exec_command",
        arguments={"cmd": "true"},
    )
    paired_result = ToolResult(
        call_id="call_paired",
        name="exec_command",
        output="ok",
    )
    current_spawn_call = ToolCall(
        call_id="call_current_spawn",
        name="spawn_agent",
        arguments={"task_name": "worker", "message": "child"},
    )

    history = (UserMessage("parent"), paired_call, paired_result, current_spawn_call)
    context = ToolContext(
        turn_id="spawn", cwd=tmp_path, agent_id=root.agent.agent_id, history=history,
    )
    try:
        spawned = await root.agent.tool_registry.execute(current_spawn_call, context)
        manager = root.agents_manager
        child = manager.get_runtime(
            manager._resolve_agent_id(root.agent.agent_id, spawned.output["task_name"])
        )
        await wait_for_runtime_idle(child)
        assert child.agent.history[:len(history)] == history
        assert root.agent.history == ()
        results = [
            item for item in child.agent.model_client.prompts[0].input
            if isinstance(item, ToolResult)
        ]
        assert results[0] == paired_result
        assert results[1].call_id == current_spawn_call.call_id
        assert results[1].output == "aborted"
        assert results[1] not in child.agent.history
    finally:
        await root.close()


@pytest.mark.parametrize("restore_file", [False, True])
def test_resume_restores_background_tool_hooks(tmp_path, restore_file) -> "None":
    clock_manager = ClockManager()
    exec_manager = UnifiedExecManager()
    registry = ToolRegistry()
    registry.register(ClockTool(clock_manager))
    registry.register(ExecCommandTool(exec_manager))
    agent = make_agent(
        ScriptedModelClient([]),
        registry,
        ContextConfig(),
        session_file_path=tmp_path / "rollout.jsonl",
    )
    runtime = make_runtime(agent)
    if restore_file:
        agent._append_history([UserMessage("saved prompt")])
    agent.shutdown()
    assert clock_manager._notify_hook is None
    assert exec_manager._notify_hook is None

    runtime.resume(agent.session_file_path if restore_file else None)

    assert clock_manager._notify_hook == runtime.maybe_invoke
    assert exec_manager._notify_hook == runtime.maybe_invoke
    assert agent.session_file_path.exists() == restore_file
    agent.shutdown()
