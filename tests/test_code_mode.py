"""In-process exec behavior and ownership on a shared host event loop."""

import asyncio
import os
import sys
from pathlib import Path

import pytest

from pycodex import AgentsManager, AssistantMessage, ContextConfig, ModelResponse
from pycodex.protocol import ToolCall
from pycodex.tools import BaseTool, CodeModeManager, ExecTool, ToolContext, ToolRegistry
from tests.fakes import ScriptedModelClient


def make_manager(tmp_path):
    registry = ToolRegistry()
    manager = CodeModeManager(registry)
    registry.register(ExecTool(manager))
    registry.runtime_environment.register_resource("exec", manager, restart=manager.restart, close=manager.close)
    return manager, registry, ToolContext("exec-test", (), tmp_path, call_id="exec-call")


def output_items(result):
    return [item["text"] for item in result.content_items[1:] if item["type"] == "input_text"]


@pytest.mark.asyncio
async def test_exec_uses_host_process_and_explicit_workspace(tmp_path, monkeypatch):
    manager, registry, context = make_manager(tmp_path)
    original_cwd = Path.cwd()
    original_stdout = sys.stdout

    async def no_process(*args, **kwargs):
        raise AssertionError("exec must not launch a process")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_process)
    try:
        result = await registry.execute(
            ToolCall("in-process", "exec", """
import os
from pathlib import Path
text(os.getpid())
text(str(Path.cwd()))
(cwd / "written.txt").write_text("ok")
print("cell output")
""", tool_type="custom"), context,
        )
        assert not result.is_error
        assert str(os.getpid()) in result.output
        assert str(original_cwd) in result.output
        assert "cell output" in result.output
        assert (tmp_path / "written.txt").read_text() == "ok"
        assert Path.cwd() == original_cwd
        assert sys.stdout is original_stdout
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_concurrent_cells_keep_output_and_namespace_separate(tmp_path):
    left, _, context = make_manager(tmp_path)
    right, _, _ = make_manager(tmp_path)
    original_stdout = sys.stdout
    try:
        results = await asyncio.gather(
            left.exec('print("left", end=":")\nawait asyncio.sleep(0.01)\nprint("done")', context),
            right.exec('print("right", end=":")\nawait asyncio.sleep(0)\nprint("done")', context),
        )
        assert [output_items(result) for result in results] == [["left:done"], ["right:done"]]
        assert sys.stdout is original_stdout
        await left.exec('value = [1]\nstore("value", value)\nvalue.append(2)', context)
        result = await left.exec(
            'text("value" in locals())\nvalue = load("value")\nvalue.append(3)\ntext(load("value"))',
            context,
        )
        assert output_items(result) == ["false", "[1]"]
        assert output_items(await right.exec('text(load("value"))', context)) == ["null"]
    finally:
        await asyncio.gather(left.close(), right.close())


@pytest.mark.asyncio
async def test_cell_cleanup_preserves_host_tasks_and_other_cells(tmp_path):
    manager, _, context = make_manager(tmp_path)
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    host_release = asyncio.Event()
    host_task = asyncio.create_task(host_release.wait())
    try:
        await manager.exec('await yield_control()\nawait asyncio.sleep(60)', context)
        other = next(iter(manager._cells.values()))
        result = await manager.exec("""
import asyncio
async def background():
    try:
        store("started", True)
        await asyncio.sleep(60)
    finally:
        store("cleaned", True)
asyncio.ensure_future(background())
await asyncio.sleep(0)
""", context)
        assert "Script completed" in result.output
        assert manager._stored_values == {"started": True, "cleaned": True}
        assert not host_task.done()
        assert not other.done_event.is_set()
        stopped = await manager.wait(other.cell_id, 100, None, True)
        assert "Script terminated" in stopped.output
        assert not host_task.done()
        assert loop.get_task_factory() is previous_factory
    finally:
        host_release.set()
        await host_task
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["finish", "terminate", "close", "shutdown"])
async def test_issued_tool_finishes_before_cell_closes(tmp_path, ending):
    manager, registry, context = make_manager(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    cancelled = []

    class HeldTool(BaseTool):
        name = "held"
        description = "A host operation that must finish."

        async def run(self, tool_context, args):
            assert tool_context.cwd == tmp_path
            entered.set()
            try:
                await release.wait()
                finished.set()
                return "finished"
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

    registry.register(HeldTool())
    source = '# @exec: {"yield_time_ms": 1}\n'
    if ending == "finish":
        source += "asyncio.create_task(tools.held())\nawait asyncio.sleep(0)"
    else:
        source += "await tools.held()"
    pending = None
    try:
        await manager.exec(source, context)
        await asyncio.wait_for(entered.wait(), 1)
        cell = next(iter(manager._cells.values()))
        if ending == "terminate":
            pending = asyncio.create_task(manager.wait(cell.cell_id, 100, None, True))
        elif ending == "close":
            pending = asyncio.create_task(manager.close())
        elif ending == "shutdown":
            manager.shutdown()
        await asyncio.sleep(0)
        assert not cell.done_event.is_set()
        assert not finished.is_set()
        assert not cancelled
        if pending is not None:
            assert not pending.done()
        observed = await manager.wait(cell.cell_id, 1, None, False)
        assert "Script running with cell ID " in observed.output
        release.set()
        await asyncio.wait_for(asyncio.shield(cell.task), 1)
        if pending is not None:
            await pending
        assert finished.is_set()
        assert not cancelled
    finally:
        release.set()
        await manager.close()
        if pending is not None:
            await pending


@pytest.mark.asyncio
async def test_host_tool_background_tasks_are_not_cell_tasks(tmp_path):
    manager, registry, context = make_manager(tmp_path)
    release = asyncio.Event()
    tasks = []

    class BackgroundTool(BaseTool):
        name = "background"
        description = "Owns background work after returning."

        async def run(self, context, args):
            tasks.append(asyncio.create_task(release.wait()))
            return "started"

    registry.register(BackgroundTool())
    try:
        result = await manager.exec('text(await tools.background())', context)
        assert output_items(result) == ["started"]
        assert not tasks[0].done()
        await manager.close()
        assert not tasks[0].done()
    finally:
        release.set()
        await asyncio.gather(*tasks)
        await manager.close()


@pytest.mark.asyncio
async def test_cleanup_also_owns_tasks_started_by_finalizers(tmp_path):
    manager, _, context = make_manager(tmp_path)
    try:
        result = await manager.exec("""
async def child():
    try:
        await asyncio.sleep(60)
    finally:
        store("child_cleaned", True)
async def parent():
    try:
        await asyncio.sleep(60)
    finally:
        asyncio.create_task(child())
        await asyncio.sleep(0)
asyncio.create_task(parent())
await asyncio.sleep(0)
""", context)
        assert "Script completed" in result.output
        assert manager._stored_values == {"child_cleaned": True}
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_cleanup_errors_are_reported(tmp_path):
    manager, _, context = make_manager(tmp_path)
    try:
        result = await manager.exec("""
async def background():
    try:
        await asyncio.sleep(60)
    finally:
        raise ValueError("cleanup failed")
asyncio.create_task(background())
await asyncio.sleep(0)
""", context)
        assert result.success is False
        assert "ValueError: cleanup failed" in result.output
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_nested_tools_parallel_errors_and_custom_arguments(tmp_path):
    manager, registry, context = make_manager(tmp_path)
    arrived = []
    both_arrived = asyncio.Event()

    class PairTool(BaseTool):
        name = "pair"
        description = "Requires concurrent calls."
        input_schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        }

        async def run(self, context, args):
            arrived.append(args["value"])
            if len(arrived) == 2:
                both_arrived.set()
            await asyncio.wait_for(both_arrived.wait(), 1)
            return args["value"]

    class CustomTool(BaseTool):
        name = "custom"
        description = "Accepts source text."
        tool_type = "custom"

        async def run(self, context, args):
            if args == "fail":
                raise ValueError("visible tool error")
            return args

    registry.register(PairTool())
    registry.register(CustomTool())
    try:
        result = await manager.exec("""
text(await asyncio.gather(tools.pair(value=1), tools.pair(2)))
text(await tools.custom("source text"))
try:
    await tools.custom("fail")
except RuntimeError as error:
    text("visible tool error" in str(error))
text("continued")
""", context)
        assert output_items(result) == ["[1, 2]", "source text", "true", "continued"]
    finally:
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source, expected", [
    ("raise ValueError('bad cell')", "ValueError: bad cell"),
    ("if", "SyntaxError"),
    ("import sys\nsys.exit(2)", "SystemExit: 2"),
])
async def test_errors_remain_cell_results_and_restore_task_factory(tmp_path, source, expected):
    manager, registry, context = make_manager(tmp_path)
    loop = asyncio.get_running_loop()
    previous = loop.get_task_factory()
    created = []

    def factory(loop, coroutine, **kwargs):
        task = asyncio.Task(coroutine, loop=loop, **kwargs)
        created.append(task)
        return task

    loop.set_task_factory(factory)
    try:
        result = await registry.execute(
            ToolCall("failed-cell", "exec", source, tool_type="custom"), context,
        )
        assert result.is_error
        assert expected in result.output
        assert loop.get_task_factory() is factory
        assert created
        assert output_items(await manager.exec('text("next cell")', context)) == ["next cell"]
    finally:
        await manager.close()
        loop.set_task_factory(previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_prefix", ["def", "async def"])
@pytest.mark.parametrize("error,expected", [
    ("ValueError('timer failure')", "ValueError: timer failure"),
    ("SystemExit(2)", "SystemExit: 2"),
    ("KeyboardInterrupt()", "KeyboardInterrupt"),
])
async def test_timer_failures_remain_cell_errors(tmp_path, callback_prefix, error, expected):
    manager, registry, context = make_manager(tmp_path)
    source = "\n".join([
        callback_prefix + " fail():",
        "    raise " + error,
        "set_timeout(fail)",
        "await asyncio.sleep(60)",
    ])
    try:
        result = await registry.execute(
            ToolCall("failed-timer", "exec", source, tool_type="custom"), context,
        )
        assert result.is_error
        assert expected in result.output
        assert output_items(await manager.exec('text("next cell")', context)) == ["next cell"]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_timers_exit_and_notify(tmp_path):
    manager, _, context = make_manager(tmp_path)
    notifications = []
    manager.bind(None, notifications.append, None)
    try:
        result = await manager.exec("""
def stop():
    text("timer fired")
    exit()
set_timeout(stop)
await asyncio.sleep(60)
text("unreachable")
""", context)
        assert "Script completed" in result.output
        assert output_items(result) == ["timer fired"]
        result = await manager.exec("""
notify("progress")
await asyncio.sleep(0.01)
text("finished")
""", context)
        assert len(notifications) == 1
        assert notifications[0].call_id == "exec-call"
        assert notifications[0].output == "progress"
        cell = next(iter(manager._cells.values()))
        await cell.task
        assert output_items(await manager.wait(cell.cell_id, 100, None, False)) == ["finished"]
        await manager.exec('set_timeout(lambda: store("late", True), 10)', context)
        await asyncio.sleep(0.02)
        assert "late" not in manager._stored_values
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_restart_retires_old_cell_without_writing_new_session_state(tmp_path):
    manager, _, context = make_manager(tmp_path)
    try:
        await manager.exec("""
try:
    await yield_control()
    await asyncio.sleep(60)
finally:
    store("old", True)
    text("old output")
""", context)
        old = next(iter(manager._cells.values()))
        manager.restart()
        assert output_items(await manager.exec('text(load("old"))', context)) == ["null"]
        await old.task
        assert not old.output_items
        assert manager._stored_values == {}
        assert old.terminated
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_media_and_multiline_source_remain_supported(tmp_path):
    manager, _, context = make_manager(tmp_path)
    try:
        result = await manager.exec("""
value = '''first
second
'''
text(value)
image("data:image/png;base64,YWJj", detail="original")
audio("data:audio/mpeg;base64,YWJj")
""", context)
        assert output_items(result) == ["first\nsecond\n"]
        assert result.content_items[2] == {
            "type": "input_image", "image_url": "data:image/png;base64,YWJj",
            "detail": "original",
        }
        assert result.content_items[3]["type"] == "input_audio"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_cancelling_exec_observer_does_not_cancel_cell(tmp_path):
    manager, registry, context = make_manager(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()

    class HoldTool(BaseTool):
        name = "hold"
        description = "Hold the cell while its observer leaves."

        async def run(self, context, args):
            entered.set()
            await release.wait()
            return "done"

    registry.register(HoldTool())
    observer = asyncio.create_task(manager.exec("text(await tools.hold())", context))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        cell = next(iter(manager._cells.values()))
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert not cell.done_event.is_set()
        release.set()
        await cell.task
        assert not cell.terminated
        assert output_items(await manager.wait(cell.cell_id, 100, None, False)) == ["done"]
    finally:
        release.set()
        await manager.close()


@pytest.mark.asyncio
async def test_runtime_close_cleans_up_a_yielded_cell(tmp_path):
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    client = ScriptedModelClient([
        ModelResponse([ToolCall(
            "background-cell", "exec",
            'await yield_control()\nawait asyncio.sleep(60)',
            tool_type="custom",
        )]),
        ModelResponse([AssistantMessage("done")]),
    ])
    agents = AgentsManager(
        client, ContextConfig(cwd=tmp_path, codex_home=tmp_path),
        toolset=("exec", "wait"),
    )
    runtime = agents.create_root_runtime()
    try:
        await runtime.start()
        receipt = await runtime.submit_input("start a cell")
        assert (await receipt.future).output_text == "done"
        assert runtime.agent.tool_registry.runtime_environment.background_work_count(True) == 1
        await runtime.close()
        assert runtime.agent.tool_registry.runtime_environment.background_work_count(True) == 0
        assert loop.get_task_factory() is previous_factory
    finally:
        await runtime.close()
