import asyncio
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pycodex import compat
from pycodex.utils.async_bridge import run_async


def test_get_running_loop_compat_rejects_idle_loop() -> "None":
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        try:
            compat._get_running_loop_compat()
        except RuntimeError as exc:
            assert "running" in str(exc)
        else:
            raise AssertionError("idle event loop should not be treated as running")
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_run_async_with_py36_get_running_loop_polyfill(monkeypatch) -> "None":
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        monkeypatch.setattr(
            asyncio,
            "get_running_loop",
            compat._get_running_loop_compat,
        )

        async def answer():
            return "ok"

        assert run_async(answer()) == "ok"
    finally:
        asyncio.set_event_loop(None)
        loop.close()


@pytest.mark.parametrize("construction_loop", ["none", "other"])
def test_sync_construction_uses_the_execution_loop(tmp_path, construction_loop):
    source = textwrap.dedent("""
        import asyncio
        import sys
        from pycodex import (
            Agent, AgentRuntime, AssistantMessage, CodeModeManager,
            ContextConfig, ModelResponse, ToolRegistry, UnifiedExecManager,
        )
        from tests.fakes import ScriptedModelClient
        from workspace_server.app import WorkspaceInteractiveSession
        from workspace_server.workspaces import WorkspaceSessionManager

        def main():
            asyncio.run(asyncio.sleep(0))
            old_loop = None
            if sys.argv[1] == "other":
                old_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(old_loop)
            try:
                tools = ToolRegistry()
                exec_manager = UnifiedExecManager()
                code_manager = CodeModeManager(tools)
                model = ScriptedModelClient([
                    ModelResponse([AssistantMessage("parent done")]),
                ])
                runtime = AgentRuntime(Agent(model, tools, ContextConfig()))
                workspace = WorkspaceSessionManager(
                    lambda: WorkspaceInteractiveSession(runtime)
                )
            finally:
                asyncio.set_event_loop(None)
                if old_loop is not None:
                    old_loop.close()

            async def exercise():
                release_child = asyncio.Event()
                children = tools.runtime_environment.subagent_manager

                async def respond(prompt, call_count):
                    await release_child.wait()
                    return ModelResponse([AssistantMessage("child done")])

                def build_child(model, effort, history, session_id):
                    return AgentRuntime(Agent(
                        ScriptedModelClient(response_factory=respond),
                        ToolRegistry(), ContextConfig(),
                    ))

                children.set_runtime_builder(build_child)
                try:
                    await workspace.start()
                    # Let the worker block before waking it with a submission.
                    await asyncio.sleep(0)
                    receipt = await runtime.submit_input("hello")
                    assert (await receipt.future).output_text == "parent done"
                    child = await children.spawn_agent(
                        "hello", None, None, False, None, None, ()
                    )
                    agent_id = child["agent_id"]
                    waiter = asyncio.create_task(
                        children.wait_agents([agent_id], timeout_ms=1000)
                    )
                    await asyncio.sleep(0)
                    assert not waiter.done()
                    release_child.set()
                    result = await waiter
                    assert result["status"][agent_id] == {
                        "completed": "child done"
                    }
                    assert "not running" in await exec_manager.write_stdin(999)
                    assert code_manager.enabled_tools() == []
                finally:
                    release_child.set()
                    await workspace.close()

            asyncio.run(asyncio.wait_for(exercise(), 5))
            print("EXECUTION_LOOP_OK")

        main()
    """)
    result = subprocess.run(
        [sys.executable, "-c", source, construction_loop],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=dict(
            os.environ, HOME=str(tmp_path), CODEX_HOME=str(tmp_path / "codex-home")
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXECUTION_LOOP_OK" in result.stdout
    assert "Traceback" not in result.stderr
