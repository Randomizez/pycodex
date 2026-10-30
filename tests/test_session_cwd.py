import asyncio
from pathlib import Path

import pytest

from pycodex import Agent, AgentRuntime, AssistantMessage, ContextConfig, ModelResponse
from pycodex.bootstrap import get_tools
from pycodex.protocol import ToolCall, ToolResult, UserMessage
from tests.fakes import ScriptedModelClient


@pytest.mark.asyncio
async def test_cd_updates_context_tools_and_observers_without_changing_process_cwd(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    original = Path.cwd()
    destination = tmp_path / "other project"
    destination.mkdir()
    (tmp_path / "AGENTS.md").write_text("parent instructions", encoding="utf-8")
    (destination / "AGENTS.md").write_text("new directory instructions", encoding="utf-8")
    calls = [
        ModelResponse([ToolCall("pwd", "exec_command", {"cmd": "pwd", "login": False})]),
        ModelResponse([AssistantMessage("done")]),
    ]
    client = ScriptedModelClient(calls)
    runtime = AgentRuntime(Agent(client, get_tools(), ContextConfig(cwd=tmp_path)))
    other = Agent(ScriptedModelClient([]), get_tools(), ContextConfig(cwd=tmp_path))
    observers = [[], []]
    for events in observers:
        runtime.attach(events.append)
    runtime.agent.context_manager.get_turn_metadata("before")
    await runtime.start()
    try:
        read = await runtime.submit_input("/cd")
        assert (await read.future)["cwd"] == str(tmp_path)
        change = await runtime.submit_input("/cd other project")
        assert (await change.future) == {"kind": "cwd", "cwd": str(destination)}
        assert runtime.agent.context_manager._config.cwd == destination
        assert runtime.agent.context_manager._workspace_metadata_turn_id is None
        for events in observers:
            assert events[-2].state["cwd"] == str(destination)
            assert events[-1].visualize() == "Current directory: " + str(destination)
        receipt = await runtime.submit_input("where are you?")
        assert (await receipt.future).output_text == "done"
        assert "<cwd>" + str(destination) + "</cwd>" in repr(client.prompts[0].input)
        assert "new directory instructions" in repr(client.prompts[0].input)
        result = next(item for item in runtime.agent.history if isinstance(item, ToolResult))
        assert str(destination) in result.output
        assert other.context_manager.cwd == tmp_path
        assert Path.cwd() == original
        assert not any(
            item.text.startswith("/cd") for item in runtime.agent.history
            if isinstance(item, UserMessage)
        )
        previous = runtime.snapshot()
        for target, error in [("missing", FileNotFoundError), ("AGENTS.md", NotADirectoryError)]:
            receipt = await runtime.submit_input("/cd " + target)
            with pytest.raises(error):
                await receipt.future
            assert runtime.snapshot() == previous
        receipt = await runtime.submit_input("/cd ..")
        assert (await receipt.future)["cwd"] == str(tmp_path)
        monkeypatch.setenv("HOME", str(destination))
        receipt = await runtime.submit_input("/cd ~")
        assert (await receipt.future)["cwd"] == str(destination)
    finally:
        await runtime.close()
        other.shutdown()
    with pytest.raises(RuntimeError, match="closing or closed"):
        runtime.set_cwd(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [False, True])
async def test_cd_rejects_active_or_queued_work(tmp_path, active):
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, count):
        started.set()
        await release.wait()
        return ModelResponse([AssistantMessage("done")])

    runtime = AgentRuntime(Agent(
        ScriptedModelClient(response_factory=respond), get_tools(),
        ContextConfig(cwd=tmp_path),
    ))
    receipt = await runtime.submit_input("pending")
    try:
        if active:
            await runtime.start()
            await asyncio.wait_for(started.wait(), 1)
        read = await runtime.submit_input("/cd")
        assert (await read.future)["cwd"] == str(tmp_path)
        change = await runtime.submit_input("/cd ..")
        with pytest.raises(RuntimeError, match="running or queued"):
            await change.future
        assert runtime.agent.context_manager.cwd == tmp_path
    finally:
        release.set()
        await runtime.close()
        await receipt.future
