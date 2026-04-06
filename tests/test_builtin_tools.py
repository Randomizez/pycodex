
import asyncio
from pathlib import Path

import pytest

from pycodex import AgentLoop, AgentRuntime, AssistantMessage, ModelResponse
from pycodex.protocol import ToolCall
from pycodex.runtime_services import (
    PlanStore,
    RequestPermissionsManager,
    RequestUserInputManager,
    SubAgentManager,
)
from pycodex.tools import (
    ApplyPatchTool,
    CloseAgentTool,
    CodeModeManager,
    ExecTool,
    ExecCommandTool,
    GrepFilesTool,
    ListDirTool,
    ReadFileTool,
    RequestPermissionsTool,
    RequestUserInputTool,
    ResumeAgentTool,
    SendInputTool,
    ShellCommandTool,
    ShellTool,
    SpawnAgentTool,
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
from tests.fakes import ScriptedModelClient
import typing


def make_registry(tmp_path) -> 'ToolRegistry':
    registry = ToolRegistry()
    code_mode_manager = CodeModeManager(registry, tmp_path)
    manager = UnifiedExecManager(tmp_path)
    registry.register(ShellTool(tmp_path))
    registry.register(ShellCommandTool(tmp_path))
    registry.register(ExecCommandTool(manager))
    registry.register(WriteStdinTool(manager))
    registry.register(ExecTool(code_mode_manager))
    registry.register(WaitTool(code_mode_manager))
    registry.register(WebSearchTool())
    registry.register(UpdatePlanTool(PlanStore()))
    registry.register(ApplyPatchTool(tmp_path))
    registry.register(ListDirTool())
    registry.register(ReadFileTool())
    registry.register(GrepFilesTool(tmp_path))
    registry.register(ViewImageTool(tmp_path))
    return registry


def make_subagent_registry(
    model_factory,
) -> 'ToolRegistry':
    manager = SubAgentManager()

    def runtime_builder(_model, _reasoning_effort, initial_history, _session_id):
        client = model_factory()
        agent = AgentLoop(
            client,
            ToolRegistry(),
            initial_history=tuple(initial_history),
        )
        return AgentRuntime(agent)

    manager.set_runtime_builder(runtime_builder)
    registry = ToolRegistry()
    registry.register(SpawnAgentTool(manager))
    registry.register(SendInputTool(manager))
    registry.register(ResumeAgentTool(manager))
    registry.register(WaitAgentTool(manager))
    registry.register(CloseAgentTool(manager))
    return registry


def make_subagent_registry_with_session_capture(
    model_factory,
    captured_session_ids: 'typing.List[str]',
) -> 'ToolRegistry':
    manager = SubAgentManager()

    def runtime_builder(_model, _reasoning_effort, initial_history, session_id):
        captured_session_ids.append(session_id)
        client = model_factory()
        agent = AgentLoop(
            client,
            ToolRegistry(),
            initial_history=tuple(initial_history),
        )
        return AgentRuntime(agent)

    manager.set_runtime_builder(runtime_builder)
    registry = ToolRegistry()
    registry.register(SpawnAgentTool(manager))
    registry.register(WaitAgentTool(manager))
    registry.register(CloseAgentTool(manager))
    return registry


@pytest.mark.asyncio
async def test_shell_tool_runs_argv_command_in_target_directory(tmp_path) -> 'None':
    registry = make_registry(tmp_path)
    result = await registry.execute(
        ToolCall(
            call_id="call_1",
            name="shell",
            arguments={"command": ["bash", "-lc", "pwd"]},
        ),
        ToolContext(turn_id="turn_1", history=()),
    )

    assert result.is_error is False
    assert "Working directory:" in result.output
    assert str(Path(tmp_path).resolve()) in result.output
    assert "Exit code: 0" in result.output


@pytest.mark.asyncio
async def test_shell_tool_reports_timeout(tmp_path) -> 'None':
    registry = make_registry(tmp_path)
    result = await registry.execute(
        ToolCall(
            call_id="call_2",
            name="shell",
            arguments={"command": ["bash", "-lc", "sleep 1"], "timeout_ms": 10},
        ),
        ToolContext(turn_id="turn_2", history=()),
    )

    assert result.is_error is False
    assert "Timeout: exceeded 10 ms" in result.output


@pytest.mark.asyncio
async def test_shell_command_tool_runs_shell_script(tmp_path) -> 'None':
    registry = make_registry(tmp_path)
    result = await registry.execute(
        ToolCall(
            call_id="call_3",
            name="shell_command",
            arguments={"command": "pwd"},
        ),
        ToolContext(turn_id="turn_3", history=()),
    )

    assert result.is_error is False
    assert str(Path(tmp_path).resolve()) in result.output


@pytest.mark.asyncio
async def test_list_dir_tool_lists_entries_from_absolute_path(tmp_path) -> 'None':
    (tmp_path / "a.txt").write_text("a")
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "b.txt").write_text("b")
    registry = make_registry(tmp_path)
    result = await registry.execute(
        ToolCall(
            call_id="call_4",
            name="list_dir",
            arguments={"dir_path": str(tmp_path), "depth": 2},
        ),
        ToolContext(turn_id="turn_4", history=()),
    )

    assert result.is_error is False
    assert f"Absolute path: {tmp_path}" in result.output
    assert "a.txt" in result.output
    assert "sub/" in result.output
    assert "b.txt" in result.output


@pytest.mark.asyncio
async def test_read_file_tool_reads_slice_with_line_numbers(tmp_path) -> 'None':
    file_path = tmp_path / "sample.py"
    file_path.write_text("first\nsecond\nthird\n")
    registry = make_registry(tmp_path)
    result = await registry.execute(
        ToolCall(
            call_id="call_5",
            name="read_file",
            arguments={"file_path": str(file_path), "offset": 2, "limit": 2},
        ),
        ToolContext(turn_id="turn_5", history=()),
    )

    assert result.is_error is False
    assert result.output == "L2: second\nL3: third"


@pytest.mark.asyncio
async def test_grep_files_tool_returns_matching_paths(tmp_path) -> 'None':
    matched = tmp_path / "match.py"
    matched.write_text("hello world\n")
    unmatched = tmp_path / "skip.py"
    unmatched.write_text("goodbye\n")
    registry = make_registry(tmp_path)
    result = await registry.execute(
        ToolCall(
            call_id="call_6",
            name="grep_files",
            arguments={"pattern": "hello", "path": str(tmp_path)},
        ),
        ToolContext(turn_id="turn_6", history=()),
    )

    assert result.is_error is False
    assert "match.py" in result.output
    assert "skip.py" not in result.output


@pytest.mark.asyncio
async def test_exec_command_tool_returns_session_for_long_running_process(tmp_path) -> 'None':
    registry = make_registry(tmp_path)
    result = await registry.execute(
        ToolCall(
            call_id="call_7",
            name="exec_command",
            arguments={
                "cmd": "printf start; sleep 1; printf end",
                "yield_time_ms": 10,
            },
        ),
        ToolContext(turn_id="turn_7", history=()),
    )

    assert result.is_error is False
    assert "Command: " in result.output
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
        ToolContext(turn_id="turn_7_cleanup", history=()),
    )

    assert closed.is_error is False
    assert "Process exited with code 0" in closed.output
    combined_output = result.output + closed.output
    assert "start" in combined_output
    assert "end" in combined_output


@pytest.mark.asyncio
async def test_write_stdin_tool_reuses_running_session_and_returns_exit_metadata(tmp_path) -> 'None':
    registry = make_registry(tmp_path)
    start = await registry.execute(
        ToolCall(
            call_id="call_8",
            name="exec_command",
            arguments={
                "cmd": "read line; printf \"hello:%s\" \"$line\"",
                "yield_time_ms": 10,
            },
        ),
        ToolContext(turn_id="turn_8", history=()),
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
        ToolContext(turn_id="turn_9", history=()),
    )

    assert finish.is_error is False
    assert "Command: " in finish.output
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
            ToolContext(turn_id="turn_9b", history=()),
        )

    assert "Process exited with code 0" in finish.output
    assert "hello:world" in finish.output


@pytest.mark.asyncio
async def test_exec_command_tool_defaults_to_upstream_truncation_budget(tmp_path) -> 'None':
    registry = make_registry(tmp_path)
    result = await registry.execute(
        ToolCall(
            call_id="call_9_default_truncation",
            name="exec_command",
            arguments={
                "cmd": (
                    'python3 -c "print(\'HEAD\' + \'A\'*25000 + \'B\'*25000 + \'TAIL\')"'
                ),
                "yield_time_ms": 1_000,
            },
        ),
        ToolContext(turn_id="turn_9_default_truncation", history=()),
    )

    assert result.is_error is False
    assert "Process exited with code 0" in result.output
    assert "Original token count: " in result.output
    body = result.output.split("Output:\n", 1)[1]
    assert body.startswith("Total output lines: 1\n\nHEAD")
    assert "tokens truncated" in body
    assert body.rstrip().endswith("TAIL")


@pytest.mark.asyncio
async def test_write_stdin_tool_defaults_to_upstream_truncation_budget(tmp_path) -> 'None':
    registry = make_registry(tmp_path)
    start = await registry.execute(
        ToolCall(
            call_id="call_9_write_start",
            name="exec_command",
            arguments={
                "cmd": 'read line; printf "%s" "$line"',
                "yield_time_ms": 10,
            },
        ),
        ToolContext(turn_id="turn_9_write_start", history=()),
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
        ToolContext(turn_id="turn_9_write_finish", history=()),
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
            ToolContext(turn_id="turn_9_write_finish_poll", history=()),
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
            ToolContext(turn_id="turn_9_write_close", history=()),
        )
        assert "Process exited with code 0" in closed.output


@pytest.mark.asyncio
async def test_exec_command_unread_output_preserves_head_and_tail_when_capped(
    tmp_path,
) -> 'None':
    registry = make_registry(tmp_path)
    start = await registry.execute(
        ToolCall(
            call_id="call_9_unread_buffer_start",
            name="exec_command",
            arguments={
                "cmd": (
                    "python3 -c \"import sys,time; "
                    "sys.stdout.write('READY'); sys.stdout.flush(); "
                    "time.sleep(0.2); "
                    "sys.stdout.write('HEAD' + 'A'*700000 + 'MID' + 'B'*700000 + 'TAIL'); "
                    "sys.stdout.flush()\""
                ),
                "yield_time_ms": 10,
            },
        ),
        ToolContext(turn_id="turn_9_unread_buffer_start", history=()),
    )

    marker = "Process running with session ID "
    session_id_text = start.output.split(marker, 1)[1].splitlines()[0]
    session_id = int(session_id_text)

    await asyncio.sleep(0.6)

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
        ToolContext(turn_id="turn_9_unread_buffer_finish", history=()),
    )

    assert finish.is_error is False
    assert "Process exited with code 0" in finish.output
    body = finish.output.split("Output:\n", 1)[1]
    assert "Total output lines:" not in body
    assert body.startswith("HEAD") or body.startswith("READYHEAD")
    assert body.endswith("TAIL")
    assert "MID" not in body


@pytest.mark.asyncio
async def test_exec_tool_runs_javascript_and_returns_completed_status(tmp_path) -> 'None':
    registry = make_registry(tmp_path)
    result = await registry.execute(
        ToolCall(
            call_id="call_exec_1",
            name="exec",
            arguments="text('EXEC_OK')",
            tool_type="custom",
        ),
        ToolContext(turn_id="turn_exec_1", history=()),
    )

    assert result.is_error is False
    assert "Script completed" in result.output_text()
    assert "EXEC_OK" in result.output_text()


@pytest.mark.asyncio
async def test_exec_tool_can_yield_and_wait_for_remaining_output(tmp_path) -> 'None':
    registry = make_registry(tmp_path)
    start = await registry.execute(
        ToolCall(
            call_id="call_exec_2",
            name="exec",
            arguments=(
                "// @exec: {\"yield_time_ms\": 10}\n"
                "text('start');\n"
                "await new Promise((resolve) => setTimeout(resolve, 50));\n"
                "text('finish');"
            ),
            tool_type="custom",
        ),
        ToolContext(turn_id="turn_exec_2", history=()),
    )

    assert start.is_error is False
    assert "Script running with cell ID " in start.output_text()
    cell_id = start.output_text().split("Script running with cell ID ", 1)[1].splitlines()[0]

    finish = await registry.execute(
        ToolCall(
            call_id="call_wait_1",
            name="wait",
            arguments={"cell_id": cell_id, "yield_time_ms": 200},
        ),
        ToolContext(turn_id="turn_wait_1", history=()),
    )
    if "Script completed" not in finish.output_text():
        finish = await registry.execute(
            ToolCall(
                call_id="call_wait_2",
                name="wait",
                arguments={"cell_id": cell_id, "yield_time_ms": 200},
            ),
            ToolContext(turn_id="turn_wait_2", history=()),
        )

    assert finish.is_error is False
    assert "Script completed" in finish.output_text()
    assert "finish" in finish.output_text()


def test_web_search_tool_serializes_as_provider_native_spec(tmp_path) -> 'None':
    registry = make_registry(tmp_path)
    web_search = registry.get_tool("web_search")
    assert web_search is not None
    assert web_search.serialize() == {
        "type": "web_search",
        "external_web_access": True,
    }


@pytest.mark.asyncio
async def test_update_plan_tool_returns_confirmation_and_stores_plan(tmp_path) -> 'None':
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
        ToolContext(turn_id="turn_plan", history=()),
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
async def test_request_user_input_tool_is_unavailable_in_default_mode() -> 'None':
    manager = RequestUserInputManager()
    registry = ToolRegistry()
    registry.register(RequestUserInputTool(manager))

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
        ToolContext(turn_id="turn_request_user_input", history=()),
    )

    assert result.is_error is False
    assert result.output == "request_user_input is unavailable in Default mode"


@pytest.mark.asyncio
async def test_request_user_input_tool_returns_structured_answers_in_plan_mode() -> 'None':
    manager = RequestUserInputManager()
    captured_payloads: 'typing.List[typing.Dict[str, object]]' = []

    async def handler(payload):
        captured_payloads.append(payload)
        return {
            "answers": {
                "choice": {
                    "answers": ["Use tool A (Recommended)"],
                }
            }
        }

    manager.set_handler(handler)
    registry = ToolRegistry()
    registry.register(RequestUserInputTool(manager))

    result = await registry.execute(
        ToolCall(
            call_id="call_request_user_input_plan",
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
        ToolContext(
            turn_id="turn_request_user_input_plan",
            history=(),
            collaboration_mode="plan",
        ),
    )

    assert result.is_error is False
    assert result.success is True
    assert result.output == (
        '{"answers":{"choice":{"answers":["Use tool A (Recommended)"]}}}'
    )
    assert captured_payloads == [
        {
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
                    "isOther": True,
                }
            ]
        }
    ]
    assert result.serialize() == {
        "type": "function_call_output",
        "call_id": "call_request_user_input_plan",
        "output": '{"answers":{"choice":{"answers":["Use tool A (Recommended)"]}}}',
        "success": True,
    }


@pytest.mark.asyncio
async def test_request_user_input_tool_requires_non_empty_options_in_plan_mode() -> 'None':
    manager = RequestUserInputManager()
    registry = ToolRegistry()
    registry.register(RequestUserInputTool(manager))

    result = await registry.execute(
        ToolCall(
            call_id="call_request_user_input_invalid",
            name="request_user_input",
            arguments={
                "questions": [
                    {
                        "id": "choice",
                        "header": "Select",
                        "question": "Pick one",
                        "options": [],
                    }
                ]
            },
        ),
        ToolContext(
            turn_id="turn_request_user_input_invalid",
            history=(),
            collaboration_mode="plan",
        ),
    )

    assert result.is_error is False
    assert (
        result.output
        == "request_user_input requires non-empty options for every question"
    )


@pytest.mark.asyncio
async def test_request_permissions_tool_returns_permission_response() -> 'None':
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
        ToolContext(turn_id="turn_request_permissions", history=()),
    )

    assert result.is_error is False
    assert result.output == {
        "permissions": {
            "network": {"enabled": True},
        },
        "scope": "turn",
    }


@pytest.mark.asyncio
async def test_apply_patch_tool_applies_multiple_operations_atomically(tmp_path) -> 'None':
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
    registry = make_registry(tmp_path)
    result = await registry.execute(
        ToolCall(
            call_id="call_10",
            name="apply_patch",
            arguments=patch,
            tool_type="custom",
        ),
        ToolContext(turn_id="turn_10", history=()),
    )

    assert result.is_error is False
    assert result.tool_type == "custom"
    assert result.output == (
        "Exit code: 0\n"
        "Wall time: 0 seconds\n"
        "Output:\n"
        "Success. Updated the following files:\n"
        "A nested/new.txt\n"
        "M modify.txt\n"
        "D delete.txt\n"
    )
    assert (tmp_path / "nested" / "new.txt").read_text() == "created\n"
    assert target.read_text() == "line1\nchanged\n"
    assert doomed.exists() is False


@pytest.mark.asyncio
async def test_apply_patch_tool_does_not_leave_partial_writes_on_failure(tmp_path) -> 'None':
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
    registry = make_registry(tmp_path)
    result = await registry.execute(
        ToolCall(
            call_id="call_11",
            name="apply_patch",
            arguments=patch,
            tool_type="custom",
        ),
        ToolContext(turn_id="turn_11", history=()),
    )

    assert result.is_error is False
    assert result.output.startswith("Exit code: 1\nWall time: 0 seconds\nOutput:\n")
    assert "apply_patch verification failed" in result.output
    assert (tmp_path / "created.txt").exists() is False


@pytest.mark.asyncio
async def test_view_image_tool_returns_structured_input_image_output(tmp_path) -> 'None':
    image_path = tmp_path / "pixel.png"
    image_path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a"
            "0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63f8cfc0f01f00050001ff89993d1d"
            "0000000049454e44ae426082"
        )
    )
    registry = make_registry(tmp_path)
    result = await registry.execute(
        ToolCall(
            call_id="call_12",
            name="view_image",
            arguments={"path": str(image_path)},
        ),
        ToolContext(turn_id="turn_12", history=()),
    )

    assert result.is_error is False
    assert result.content_items == (
        {
            "type": "input_image",
            "image_url": result.output["image_url"],
        },
    )
    serialized = result.serialize()
    assert serialized["type"] == "function_call_output"
    assert serialized["output"][0]["type"] == "input_image"
    assert str(result.output["image_url"]).startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_spawn_agent_send_input_wait_and_close_round_trip() -> 'None':
    first_client = ScriptedModelClient(
        [
            ModelResponse(items=[AssistantMessage(text="done one")]),
            ModelResponse(items=[AssistantMessage(text="done two")]),
        ]
    )
    registry = make_subagent_registry(lambda: first_client)

    spawned = await registry.execute(
        ToolCall(
            call_id="call_spawn",
            name="spawn_agent",
            arguments={"message": "hello"},
        ),
        ToolContext(turn_id="turn_spawn", history=()),
    )

    agent_id = spawned.output["agent_id"]
    assert isinstance(spawned.output["nickname"], str)
    assert spawned.output["nickname"]
    sent = await registry.execute(
        ToolCall(
            call_id="call_send",
            name="send_input",
            arguments={"id": agent_id, "message": "hello"},
        ),
        ToolContext(turn_id="turn_send", history=()),
    )
    assert sent.output["submission_id"]

    waited = await registry.execute(
        ToolCall(
            call_id="call_wait",
            name="wait_agent",
            arguments={"ids": [agent_id], "timeout_ms": 1000},
        ),
        ToolContext(turn_id="turn_wait", history=()),
    )
    if waited.output == {
        "status": {agent_id: {"completed": "done one"}},
        "timed_out": False,
    }:
        waited = await registry.execute(
            ToolCall(
                call_id="call_wait_again",
                name="wait_agent",
                arguments={"ids": [agent_id], "timeout_ms": 1000},
            ),
            ToolContext(turn_id="turn_wait_again", history=()),
        )

    assert waited.output == {
        "status": {agent_id: {"completed": "done two"}},
        "timed_out": False,
    }

    closed = await registry.execute(
        ToolCall(
            call_id="call_close",
            name="close_agent",
            arguments={"id": agent_id},
        ),
        ToolContext(turn_id="turn_close", history=()),
    )

    assert closed.output == {"status": {"completed": "done two"}}


@pytest.mark.asyncio
async def test_spawn_agent_requires_message_or_items() -> 'None':
    client = ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="unused")])])
    registry = make_subagent_registry(lambda: client)

    spawned = await registry.execute(
        ToolCall(call_id="call_spawn_empty", name="spawn_agent", arguments={}),
        ToolContext(turn_id="turn_spawn_empty", history=()),
    )

    assert spawned.is_error is False
    assert spawned.output == "Provide one of: message or items"


@pytest.mark.asyncio
async def test_spawn_agent_uses_agent_id_as_nested_session_id() -> 'None':
    captured_session_ids: 'typing.List[str]' = []
    client = ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])])
    registry = make_subagent_registry_with_session_capture(
        lambda: client,
        captured_session_ids,
    )

    spawned = await registry.execute(
        ToolCall(
            call_id="call_spawn_session",
            name="spawn_agent",
            arguments={"message": "initial"},
        ),
        ToolContext(turn_id="turn_spawn_session", history=()),
    )

    agent_id = spawned.output["agent_id"]
    assert captured_session_ids == [agent_id]

    waited = await registry.execute(
        ToolCall(
            call_id="call_wait_session",
            name="wait_agent",
            arguments={"ids": [agent_id], "timeout_ms": 1000},
        ),
        ToolContext(turn_id="turn_wait_session", history=()),
    )
    assert waited.output == {
        "status": {agent_id: {"completed": "done"}},
        "timed_out": False,
    }

    closed = await registry.execute(
        ToolCall(
            call_id="call_close_session",
            name="close_agent",
            arguments={"id": agent_id},
        ),
        ToolContext(turn_id="turn_close_session", history=()),
    )
    assert closed.output == {"status": {"completed": "done"}}


@pytest.mark.asyncio
async def test_resume_agent_restarts_closed_agent_runtime() -> 'None':
    client = ScriptedModelClient(
        [
            ModelResponse(items=[AssistantMessage(text="initial done")]),
            ModelResponse(items=[AssistantMessage(text="after resume")]),
        ]
    )
    registry = make_subagent_registry(lambda: client)

    spawned = await registry.execute(
        ToolCall(
            call_id="call_spawn_2",
            name="spawn_agent",
            arguments={"message": "initial"},
        ),
        ToolContext(turn_id="turn_spawn_2", history=()),
    )
    agent_id = spawned.output["agent_id"]

    await registry.execute(
        ToolCall(call_id="call_close_2", name="close_agent", arguments={"id": agent_id}),
        ToolContext(turn_id="turn_close_2", history=()),
    )
    resumed = await registry.execute(
        ToolCall(call_id="call_resume", name="resume_agent", arguments={"id": agent_id}),
        ToolContext(turn_id="turn_resume", history=()),
    )
    assert resumed.output == {"status": "pending_init"}

    await registry.execute(
        ToolCall(
            call_id="call_send_2",
            name="send_input",
            arguments={"id": agent_id, "message": "after"},
        ),
        ToolContext(turn_id="turn_send_2", history=()),
    )
    waited = await registry.execute(
        ToolCall(
            call_id="call_wait_2",
            name="wait_agent",
            arguments={"ids": [agent_id], "timeout_ms": 1000},
        ),
        ToolContext(turn_id="turn_wait_2", history=()),
    )
    assert waited.output == {
        "status": {agent_id: {"completed": "after resume"}},
        "timed_out": False,
    }
