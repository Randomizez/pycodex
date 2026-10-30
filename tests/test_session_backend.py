import asyncio
import json
import os
import subprocess
import sys

import pytest

from pycodex import (
    Agent,
    AgentRuntime,
    AssistantMessage,
    BaseTool,
    ContextConfig,
    ModelResponse,
    ResponsesModelClient,
    ResponsesProviderConfig,
    ToolRegistry,
    UserMessage,
)
from pycodex.utils import uuid7_string
from pycodex.utils.session_persist import resolve_codex_home, rollout_path_for_session
from tests.fakes import ScriptedModelClient


class ControlClient(ScriptedModelClient):
    async def list_models(self):
        return ["scripted", "changed"]


def make_queue(client=None):
    session_id = uuid7_string()
    return AgentRuntime(
        Agent(
            client or ControlClient([]),
            ToolRegistry(),
            ContextConfig(),
            session_file_path=rollout_path_for_session(
                resolve_codex_home(), session_id
            ),
            session_id=session_id,
        )
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("/data/project", "/data/project"),
        ("/tmp/", "/tmp/"),
        ("/not-created/with spaces/file.txt", "/not-created/with spaces/file.txt"),
        ("/model/config.toml", "/model/config.toml"),
        ("//server/share", "//server/share"),
        ("/", "/"),
        (" /modle ", "/modle"),
        (" /tmp", "/tmp"),
        (" /model changed", "/model changed"),
        (" /queue literal", "/queue literal"),
    ],
)
@pytest.mark.asyncio
async def test_slash_paths_and_escaped_commands_are_messages(text, expected):
    client = ControlClient([ModelResponse([AssistantMessage("done")])])
    queue = make_queue(client)
    events = []
    queue.attach(events.append)
    await queue.start()
    try:
        receipt = await queue.submit_input(text)
        assert receipt.kind == "turn"
        await receipt.future
        assert queue.agent.history == (
            UserMessage(expected), AssistantMessage("done"),
        )
        assert queue.agent.model_name == "scripted"
        queued = [event for event in events if event.kind == "input_queued"]
        assert len(queued) == 1
        assert queued[0].queue == "steer"
        assert not any(event.kind.startswith("command_") for event in events)
    finally:
        await queue.close()


@pytest.mark.parametrize("text", ["/modle", "/tmp", "/modle /data/project"])
@pytest.mark.asyncio
async def test_unknown_single_component_slash_names_remain_command_errors(text):
    queue = make_queue()
    events = []
    queue.attach(events.append)
    receipt = await queue.submit_input(text)
    assert receipt.kind == "command"
    with pytest.raises(ValueError, match="Unknown command"):
        await receipt.future
    assert not queue.agent.history
    assert events[-1].kind == "command_failed"
    assert events[-1].command == text.split()[0]


@pytest.mark.parametrize("text", ["/tmp/", " /tmp", " /model"])
@pytest.mark.asyncio
async def test_slash_text_can_answer_a_pending_question(text):
    queue = make_queue()
    observer = queue.attach(lambda event: None)
    manager = queue.agent.tool_registry.runtime_environment.request_user_input_manager
    payload = question_payload()
    payload["questions"] = payload["questions"][:1]
    request = asyncio.create_task(manager.request(payload))
    await asyncio.sleep(0)
    receipt = await queue.submit_input(text)
    assert receipt.kind == "answer"
    answer = await request
    assert answer["answers"]["first"]["answers"] == [text.strip()]
    assert queue.snapshot()["input_request"] is None
    queue.detach(observer)


@pytest.mark.parametrize("dirname", ["sessions", "archived_sessions"])
@pytest.mark.asyncio
async def test_resume_session_id_finds_older_rollouts_and_keeps_state_on_errors(dirname):
    home = resolve_codex_home()
    session_id = "abcdef01-2345-4678-89ab-0123456789ab"

    def write_rollout(directory, identity, message, modified):
        path = home / directory / "2026" / "01" / ("rollout-" + identity + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = [{"type": "session_meta", "payload": {"id": identity}}]
        if message:
            entries.extend([
                {"type": "event_msg", "payload": {"type": "user_message", "message": message}},
                {"type": "response_item", "payload": AssistantMessage("saved answer").serialize()},
            ])
        path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
        os.utime(str(path), (modified, modified))
        return path

    other = "archived_sessions" if dirname == "sessions" else "sessions"
    write_rollout(other, session_id, "outdated copy", 1000)
    path = write_rollout(dirname, session_id, "older prompt", 1500)
    for index in range(21):
        write_rollout("sessions", uuid7_string(), "recent " + str(index), 2000 + index)
    empty_id = uuid7_string()
    write_rollout("sessions", empty_id, "", 1100)
    (home / "session_index.jsonl").write_text(
        json.dumps({"id": session_id, "thread_name": "Saved title"}) + "\n",
        encoding="utf-8",
    )
    original = path.read_bytes()
    client = ControlClient([ModelResponse([AssistantMessage("continued")])])
    queue = make_queue(client)
    observers = [[], []]
    for events in observers:
        queue.attach(events.append)
    await queue.start()
    try:
        receipt = await queue.submit_input("/resume")
        sessions = (await receipt.future)["sessions"]
        assert len(sessions) == 20
        assert session_id not in [session["thread_id"] for session in sessions]

        receipt = await queue.submit_input("/resume " + session_id.upper())
        result = await receipt.future
        assert result["kind"] == "resumed"
        assert result["state"]["session_id"] == session_id
        assert queue.agent.session_file_path == path
        assert queue.title == "Saved title"
        assert queue.agent.history == (UserMessage("older prompt"), AssistantMessage("saved answer"))
        assert path.read_bytes() == original
        for events in observers:
            assert events[-1].kind == "command_completed"
            assert events[-1].result == result

        before = queue.snapshot()
        for target, error in (
            (uuid7_string(), "Session not found"),
            (empty_id, "Session not found"),
            (session_id[:8], "Usage: /resume"),
            (session_id + " extra", "Usage: /resume"),
            ("0", "Session not found"),
            ("999", "Session not found"),
        ):
            receipt = await queue.submit_input("/resume " + target)
            with pytest.raises(ValueError, match=error):
                await receipt.future
            assert queue.snapshot() == before
            assert path.read_bytes() == original

        receipt = await queue.submit_input("continue")
        await receipt.future
        assert queue.agent.history == (
            UserMessage("older prompt"), AssistantMessage("saved answer"),
            UserMessage("continue"), AssistantMessage("continued"),
        )
        assert path.read_bytes().startswith(original)
    finally:
        await queue.close()


@pytest.mark.parametrize(
    "model,default_effort", [("gpt-6-astra", "low"), ("custom-model", None)]
)
@pytest.mark.asyncio
async def test_effort_commands_update_next_request_and_broadcast(
    model, default_effort, monkeypatch
):
    client = ResponsesModelClient(
        ResponsesProviderConfig(
            model,
            "demo",
            "https://example.invalid/v1",
            None,
            reasoning_effort="medium",
        )
    )
    queue = make_queue(client)
    payloads = []

    async def respond(prompt, event_handler):
        payloads.append(client._build_payload(prompt))
        return ModelResponse([AssistantMessage("done")])

    monkeypatch.setattr(client, "complete", respond)
    observers = [[], []]
    for events in observers:
        queue.attach(events.append)
    await queue.start()
    try:
        receipt = await queue.submit_input("/effort")
        assert (await receipt.future)["reasoning_effort"] == "medium"
        for value, expected in [
            ("high", "high"),
            ("adaptive", "adaptive"),
            ("none", "none"),
            ("default", default_effort),
        ]:
            receipt = await queue.submit_input("/effort " + value)
            result = await receipt.future
            assert result == {"kind": "effort", "reasoning_effort": expected}
            for events in observers:
                assert events[-1].kind == "command_completed"
                assert events[-1].result == result
            assert queue.agent.model_client is client
            receipt = await queue.submit_input("reply")
            await receipt.future
            reasoning = payloads[-1].get("reasoning", {})
            if expected is None:
                assert "effort" not in reasoning
            else:
                assert reasoning["effort"] == expected
        assert all(
            not item.text.startswith("/effort")
            for item in queue.agent.history
            if isinstance(item, UserMessage)
        )
        receipt = await queue.submit_input("/effort high extra")
        with pytest.raises(ValueError, match="Usage: /effort"):
            await receipt.future
        assert client.reasoning_effort == default_effort
    finally:
        await queue.close()
    with pytest.raises(RuntimeError, match="shutting down"):
        await queue.submit_input("/effort high")
    assert client.reasoning_effort == default_effort


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.asyncio
async def test_effort_changes_reject_queued_or_active_work(active, monkeypatch):
    client = ResponsesModelClient(
        ResponsesProviderConfig(
            "custom-model",
            "demo",
            "https://example.invalid/v1",
            None,
            reasoning_effort="low",
        )
    )
    queue = make_queue(client)
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, event_handler):
        started.set()
        await release.wait()
        return ModelResponse([AssistantMessage("done")])

    monkeypatch.setattr(client, "complete", respond)
    receipt = await queue.submit_input("pending")
    try:
        if active:
            await queue.start()
            await asyncio.wait_for(started.wait(), 1)
        read = await queue.submit_input("/effort")
        assert (await read.future)["reasoning_effort"] == "low"
        change = await queue.submit_input("/effort high")
        with pytest.raises(RuntimeError, match="running or queued"):
            await change.future
        assert client.reasoning_effort == "low"
    finally:
        release.set()
        await queue.close()
        await receipt.future


def question_payload():
    return {
        "questions": [
            {
                "id": "first",
                "header": "First",
                "question": "Choose a path",
                "options": [
                    {"label": "Alpha", "description": "Path A"},
                    {"label": "Beta", "description": "Path B"},
                ],
            },
            {
                "id": "second",
                "header": "Second",
                "question": "Choose again",
                "options": [
                    {"label": "Gamma", "description": "Path C"},
                    {"label": "Delta", "description": "Path D"},
                ],
            },
        ]
    }


@pytest.mark.parametrize(
    "answer,scope,granted",
    [
        ("t", "turn", True),
        ("s", "session", True),
        ("n", "turn", False),
    ],
)
@pytest.mark.asyncio
async def test_permissions_share_admission_and_explicit_scope(answer, scope, granted):
    queue = make_queue()
    observer = queue.attach(lambda event: None)
    manager = queue.agent.tool_registry.runtime_environment.request_permissions_manager
    permissions = {"file_system": {"write": ["/tmp/demo"]}}
    request = asyncio.create_task(manager.request({"permissions": permissions}))
    await asyncio.sleep(0)
    await queue.submit_input(answer, "web")
    assert await request == {
        "permissions": permissions if granted else {},
        "scope": scope,
    }
    queue.detach(observer)


@pytest.mark.parametrize(
    "resolution", ["empty", "detach", "timeout", "structured", "close"]
)
@pytest.mark.asyncio
async def test_pending_question_has_bounded_resolution(resolution):
    queue = make_queue()
    events = []
    observer = queue.attach(events.append)
    manager = queue.agent.tool_registry.runtime_environment.request_user_input_manager
    payload = question_payload()
    if resolution == "timeout":
        payload["autoResolutionMs"] = 1
    request = asyncio.create_task(manager.request(payload))
    await asyncio.sleep(0)
    request_id = queue.snapshot()["input_request"].request_id
    expected = None
    if resolution == "empty":
        await queue.submit_input("")
    elif resolution == "detach":
        queue.detach(observer)
    elif resolution == "structured":
        expected = {"answers": {"first": {"answers": ["/literal"]}}}
        queue.answer_input(request_id, expected)
    elif resolution == "close":
        await queue.close()
    assert await asyncio.wait_for(request, 1) == expected
    assert queue.snapshot()["input_request"] is None
    assert sum(event.kind == "input_resolved" for event in events) == (
        0 if resolution == "detach" else 1
    )
    with pytest.raises(ValueError, match="no longer pending"):
        queue.answer_input(request_id, None)
    if resolution != "detach":
        queue.detach(observer)
    assert await manager.request(question_payload()) is None


@pytest.mark.parametrize("continue_after_fork", [False, True])
@pytest.mark.asyncio
async def test_fork_owns_identity_and_lazy_recorder(continue_after_fork):
    client = ControlClient(
        [
            ModelResponse([AssistantMessage("done")]),
            ModelResponse([AssistantMessage("continued")]),
        ]
    )
    client._session_id = "provider-created-id"
    queue = make_queue(client)
    await queue.start()
    receipt = await queue.submit_input("before")
    await receipt.future
    old_path = queue.agent.session_file_path
    old_bytes = old_path.read_bytes()
    old_id = queue.agent.session_id
    old_history = queue.agent.history
    queue.agent._last_total_usage_tokens = 123
    fork = await queue.submit_input("/fork")
    assert (await fork.future)["session_id"] == queue.agent.session_id != old_id
    assert client._session_id == queue.agent.session_id
    assert queue.agent.history == old_history
    assert queue.agent._last_total_usage_tokens == 123
    assert queue.agent.session_file_path != old_path
    assert not queue.agent.session_file_path.exists()
    if continue_after_fork:
        receipt = await queue.submit_input("after")
        await receipt.future
        entries = [
            json.loads(line)
            for line in queue.agent.session_file_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert sum(entry["type"] == "session_meta" for entry in entries) == 1
        assert entries[0]["payload"]["id"] == queue.agent.session_id
        assert [
            entry["payload"] for entry in entries if entry["type"] == "response_item"
        ] == [item.serialize() for item in queue.agent.history]
    await queue.close()
    assert queue.agent.session_file_path.exists() == continue_after_fork
    assert old_path.read_bytes() == old_bytes


@pytest.mark.parametrize("started,cleanup_fails", [(False, False), (True, True)])
@pytest.mark.asyncio
async def test_start_close_are_idempotent_and_errors_do_not_hang_shutdown(
    started, cleanup_fails
):
    hook_calls = []

    class CountingTool(BaseTool):
        name = "count_shutdown"
        description = "Records shutdown."

        def shutdown(self):
            hook_calls.append(True)

        async def run(self, context, args):
            return None

    tools = ToolRegistry()
    tools.register(CountingTool())
    queue = AgentRuntime(Agent(ControlClient([]), tools, ContextConfig()))
    if started:
        await queue.start()
        worker = queue._worker
        await queue.start()
        assert queue._worker is worker
    calls = []
    events = []
    queue.event_handler = events.append

    async def failing_close():
        calls.append(True)
        assert not hook_calls
        assert queue.is_busy
        with pytest.raises(RuntimeError, match="running or queued"):
            queue.resume()
        await asyncio.sleep(0)
        if cleanup_fails:
            raise RuntimeError("close failed")

    queue.add_close_handler(failing_close)

    async def final_close():
        calls.append("last")

    queue.add_close_handler(final_close)
    results = await asyncio.wait_for(
        asyncio.gather(queue.close(), queue.close(), return_exceptions=True),
        1,
    )
    if cleanup_fails:
        assert isinstance(results[0], RuntimeError)
        assert results[0] is results[1]
        with pytest.raises(RuntimeError, match="close failed") as repeated:
            await asyncio.wait_for(queue.close(), 1)
        assert repeated.value is results[0]
    else:
        assert results == [None, None]
        await queue.close()
    assert calls == [True, "last"]
    assert hook_calls == [True]
    assert queue.agent.is_shutdown
    assert queue.snapshot()["closed"]
    assert queue._worker.done()
    assert sum(event.kind == "session_closed" for event in events) == 1
    with pytest.raises(RuntimeError, match="resume"):
        await queue.start()


def test_backend_and_bootstrap_do_not_import_frontends():
    code = (
        "import sys; import pycodex.runtime; import pycodex.bootstrap; "
        "forbidden = ('pycodex.cli', "
        "'prompt_toolkit', 'pycodex_ws', 'workspace_server'); "
        "assert not [name for name in forbidden if name in sys.modules]"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
