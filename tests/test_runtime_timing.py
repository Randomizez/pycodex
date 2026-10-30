import asyncio

import pytest

from pycodex import AgentsManager, AssistantMessage, ContextConfig, ModelResponse
from pycodex.runtime import SubmissionInterrupted
from tests.fakes import ScriptedModelClient


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "failed", "interrupted"])
async def test_turn_duration_freezes_at_terminal_event_and_resets_on_next_turn(
    tmp_path, monkeypatch, outcome,
):
    now = [100.0]
    monkeypatch.setattr("pycodex.runtime.monotonic", lambda: now[0])
    entered = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, count):
        entered.set()
        await release.wait()
        if outcome == "failed" and count == 1:
            raise ValueError("timed failure")
        return ModelResponse([AssistantMessage("done")])

    runtime = AgentsManager(
        ScriptedModelClient(response_factory=respond),
        ContextConfig(cwd=tmp_path, codex_home=tmp_path), toolset=(),
    ).create_root_runtime()
    assert runtime.turn_elapsed_seconds == 0
    await runtime.start()
    try:
        receipt = await runtime.submit_input("first")
        await asyncio.wait_for(entered.wait(), 2)
        assert runtime.turn_elapsed_seconds == 0
        now[0] = 190.0
        assert runtime.turn_elapsed_seconds == 90
        if outcome == "interrupted":
            runtime.stop_asap()
        now[0] = 250.0
        release.set()
        if outcome == "failed":
            with pytest.raises(ValueError, match="timed failure"):
                await receipt.future
        elif outcome == "interrupted":
            with pytest.raises(SubmissionInterrupted):
                await receipt.future
        else:
            await receipt.future
        now[0] = 1000.0
        assert runtime.turn_elapsed_seconds == 150

        entered.clear()
        release.clear()
        receipt = await runtime.submit_input("next")
        await asyncio.wait_for(entered.wait(), 2)
        assert runtime.turn_elapsed_seconds == 0
        now[0] = 1011.0
        release.set()
        await receipt.future
        now[0] = 2000.0
        assert runtime.turn_elapsed_seconds == 11
    finally:
        release.set()
        await runtime.close()
