import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("construction_loop", ["none", "other"])
def test_workspace_construction_uses_the_execution_loop(tmp_path, construction_loop):
    source = textwrap.dedent("""
        import asyncio
        import sys
        from pycodex import (
            Agent, AgentRuntime, AssistantMessage, ContextConfig,
            ModelResponse, ToolRegistry,
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
                model = ScriptedModelClient([
                    ModelResponse([AssistantMessage("done")]),
                ])
                runtime = AgentRuntime(Agent(
                    model, ToolRegistry(), ContextConfig()
                ))
                workspace = WorkspaceSessionManager(
                    lambda: WorkspaceInteractiveSession(runtime)
                )
            finally:
                asyncio.set_event_loop(None)
                if old_loop is not None:
                    old_loop.close()

            async def exercise():
                try:
                    await workspace.start()
                    await asyncio.sleep(0)
                    receipt = await runtime.submit_input("hello")
                    assert (await receipt.future).output_text == "done"
                finally:
                    await workspace.close()

            asyncio.run(asyncio.wait_for(exercise(), 5))
            print("EXECUTION_LOOP_OK")

        main()
    """)
    result = subprocess.run(
        [sys.executable, "-c", source, construction_loop],
        cwd=str(Path(__file__).resolve().parents[3]),
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
