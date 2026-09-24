import asyncio
import json
import subprocess
import sys
import time

import pytest

from pycodex import (
    Agent,
    AgentRuntime,
    AssistantMessage,
    BaseTool,
    ContextConfig,
    ModelResponse,
    ToolCall,
    ToolRegistry,
    UserMessage,
)
from pycodex.bootstrap import build_runtime
from pycodex.cli import CliSessionView, run_interactive_session
from pycodex.events import AssistantDeltaEvent
from pycodex.feishu_card import PycodexCard
from pycodex.feishu_link import PycodexRuntimeLink
from pycodex.runtime import SubmissionInterrupted
from pycodex.utils import uuid7_string
from pycodex.utils.session_persist import resolve_codex_home, rollout_path_for_session
from tests.fakes import ScriptedModelClient
from workspace_server.app import (
    ThreadedWorkspaceInteractiveSession,
    WorkspaceInteractiveSession,
    create_app,
)


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


class ScriptedView(CliSessionView):
    def __init__(self, inputs):
        super().__init__()
        self.inputs = iter(inputs)
        self.lines = []
        self._line_output = self.lines.append
        self.display.color_enabled = False

    async def poll_prompt(self, prompt=None):
        try:
            return next(self.inputs)
        except StopIteration:
            raise EOFError()


async def run_frontend(frontend, queue, inputs):
    if frontend == "cli":
        view = ScriptedView(inputs)
        await run_interactive_session(queue, False, view=view)
        return
    if frontend == "web":
        session = await WorkspaceInteractiveSession(queue).start()
        try:
            for text in inputs:
                result = await session.submit(text)
                assert result["ok"]
        finally:
            await session.close()
        return
    link = PycodexRuntimeLink(
        queue,
        "unused",
        loop=asyncio.get_running_loop(),
        card=PycodexCard(),
    )
    await queue.start()
    link._frontend_id = queue.attach(link._handle_runtime_event)
    try:
        for text in inputs:
            result = await link._submit({"action": "send", "prompt": text})
            assert result["toast"]["type"] == "info"
    finally:
        await queue.close()
        link.detach()


@pytest.mark.parametrize("frontend", ["cli", "web", "feishu"])
@pytest.mark.asyncio
async def test_frontends_share_all_session_commands(frontend):
    session_id = uuid7_string()
    source = Agent(
        ControlClient([ModelResponse([AssistantMessage("old answer")])]),
        ToolRegistry(),
        ContextConfig(),
        session_file_path=rollout_path_for_session(resolve_codex_home(), session_id),
        session_id=session_id,
    )
    await source.run_turn(["old prompt"])
    old_id = source.session_id
    source.shutdown()
    client = ControlClient([ModelResponse([AssistantMessage("checkpoint")])])
    queue = make_queue(client)
    events = []
    queue.event_handler = events.append

    await run_frontend(
        frontend,
        queue,
        [
            "/help",
            "/model",
            "/model changed",
            "/resume",
            "/resume 1",
            "/title Shared title",
            "/title",
            "/history",
            "/fork",
            "/compact",
            "/unknown",
            "/exit",
        ],
    )

    results = [event.result for event in events if event.kind == "command_completed"]
    assert [result["kind"] for result in results] == [
        "help",
        "models",
        "model_changed",
        "sessions",
        "resumed",
        "title_changed",
        "title",
        "history",
        "forked",
        "compacted",
        "closed",
    ]
    assert results[7]["state"]["history"] == (("old prompt", "old answer"),)
    assert queue.title == "Shared title"
    assert queue.agent.model_name == "changed"
    assert queue.agent.session_id != old_id
    assert queue.agent.is_shutdown
    assert client.call_count == 1
    assert not any(
        item.text.startswith("/")
        for item in client.prompts[0].input
        if isinstance(item, UserMessage)
    )
    assert [event.command for event in events if event.kind == "command_failed"] == [
        "/unknown"
    ]


@pytest.mark.asyncio
async def test_state_changes_reach_every_frontend_and_detach_is_not_close():
    queue = make_queue()
    web = await WorkspaceInteractiveSession(queue).start()
    cli = ScriptedView([])
    cli_id = queue.attach(cli.handle_event)
    card = PycodexCard()
    card_id = queue.attach(card.apply_event)
    try:
        await queue.submit_input("/model changed")
        await queue.submit_input("/title Shared")
        assert web.snapshot()["model"] == card.model_name == "changed"
        assert (
            web.snapshot()["title"]
            == cli.display.title
            == card.display.title
            == "Shared"
        )
        assert card.render()["header"]["title"]["content"] == "Shared"
        web.detach()
        assert queue.agent.accepts_input
        await queue.submit_input("/model scripted")
        assert card.model_name == "scripted"
        assert web.snapshot()["model"] == "changed"
    finally:
        queue.detach(cli_id)
        queue.detach(card_id)
        cli.close()
        await queue.close()


@pytest.mark.parametrize("frontend", ["cli", "web", "feishu"])
@pytest.mark.asyncio
async def test_question_answers_use_same_backend_interpretation(frontend):
    queue = make_queue()
    ready = asyncio.Event()

    def handle_event(event):
        if event.kind == "input_requested":
            ready.set()

    observer = queue.attach(handle_event)
    manager = queue.agent.tool_registry.runtime_environment.request_user_input_manager
    request = asyncio.create_task(manager.request(question_payload()))
    await asyncio.wait_for(ready.wait(), 1)
    await run_frontend(frontend, queue, ["2", "0", "Custom answer"])
    assert await request == {
        "answers": {
            "first": {"answers": ["Beta"]},
            "second": {"answers": ["Custom answer"]},
        },
    }
    assert not queue.agent.history
    assert queue.agent.model_client.call_count == 0
    queue.detach(observer)


@pytest.mark.parametrize("frontend", ["cli", "web", "feishu"])
@pytest.mark.asyncio
async def test_request_user_input_stays_unavailable_with_frontend(frontend):
    from pycodex.tools import RequestUserInputTool

    client = ControlClient(
        [
            ModelResponse(
                [ToolCall("question", "request_user_input", question_payload())]
            ),
            ModelResponse([AssistantMessage("continued")]),
        ]
    )
    queue = make_queue(client)
    queue.agent.tool_registry.register(RequestUserInputTool())
    events = []
    queue.event_handler = events.append

    await run_frontend(frontend, queue, ["ask"])

    assert client.call_count == 2
    assert not any(event.kind == "input_requested" for event in events)
    result = client.prompts[1].input[-1]
    assert result.output == "request_user_input is unavailable in Default mode"
    assert "success" not in result.serialize()


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


@pytest.mark.asyncio
async def test_busy_feishu_input_steers_and_queue_remains_ordered():
    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(prompt, call_count):
        if call_count == 1:
            started.set()
            await release.wait()
        return ModelResponse([AssistantMessage(str(call_count))])

    client = ControlClient(response_factory=respond)
    queue = make_queue(client)
    await queue.start()
    link = PycodexRuntimeLink(
        queue,
        "unused",
        loop=asyncio.get_running_loop(),
        card=PycodexCard(),
    )
    link._frontend_id = queue.attach(link._handle_runtime_event)
    try:
        first = await queue.submit_input("first", "cli")
        await started.wait()
        assert link.card.status is not None
        assert not link.card.render()["body"]["elements"][-1]["disabled"]
        assert (await link._submit({"action": "send", "prompt": "steered"}))["toast"][
            "type"
        ] == "info"
        queued = await queue.submit_input("/queue last", "web")
        rejected = await queue.submit_input("/model changed", "web")
        with pytest.raises(RuntimeError, match="running or queued"):
            await rejected.future
        release.set()
        with pytest.raises(SubmissionInterrupted):
            await first.future
        await queued.future
        assert [
            [item.text for item in prompt.input if isinstance(item, UserMessage)]
            for prompt in client.prompts
        ] == [
            ["first"],
            ["first", "steered"],
            ["first", "steered", "last"],
        ]
    finally:
        release.set()
        await queue.close()
        link.detach()


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


@pytest.mark.parametrize("cancel_caller", [False, True])
@pytest.mark.asyncio
async def test_close_waits_for_accepted_command_then_releases_connection(
    monkeypatch, cancel_caller
):
    started = asyncio.Event()
    release = asyncio.Event()
    detached = []

    class Link:
        session_key = "fake"
        message_id = "fake"

        def __init__(self, queue, target):
            pass

        async def start_async(self):
            started.set()
            await release.wait()
            return self

        def detach(self):
            detached.append(True)

    monkeypatch.setattr("pycodex.feishu_link.PycodexRuntimeLink", Link)
    queue = build_runtime(Agent(ControlClient([]), ToolRegistry(), ContextConfig()))
    await queue.start()
    connecting = asyncio.create_task(queue.submit_input("/link fake"))
    await started.wait()
    closing = asyncio.create_task(queue.close())
    await asyncio.sleep(0)
    assert not closing.done()
    with pytest.raises(RuntimeError, match="shutting down"):
        await queue.submit_input("too late")
    if cancel_caller:
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert not queue._worker.done()
    release.set()
    receipt = await connecting
    assert (await receipt.future)["kind"] == "connection"
    await asyncio.wait_for(queue.close(), 1)
    if not cancel_caller:
        await closing
    assert detached == [True]


def test_backend_and_bootstrap_do_not_import_frontends():
    code = (
        "import sys; import pycodex.runtime; import pycodex.bootstrap; "
        "forbidden = ('pycodex.cli', "
        "'prompt_toolkit', 'pycodex.feishu_link', 'workspace_server'); "
        "assert not [name for name in forbidden if name in sys.modules]"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_web_import_does_not_load_cli_frontend():
    code = (
        "import sys; import workspace_server.app; "
        "assert 'pycodex.cli' not in sys.modules; "
        "assert 'prompt_toolkit' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.asyncio
async def test_late_frontends_restore_active_stream_and_follow_completion():
    started = asyncio.Event()
    release = asyncio.Event()
    deltas = (AssistantDeltaEvent("part"), AssistantDeltaEvent("ial"))

    class StreamingClient:
        model = "streaming"

        async def complete(self, prompt, event_handler):
            event_handler(deltas[0])
            started.set()
            await release.wait()
            event_handler(deltas[1])
            return ModelResponse([AssistantMessage("partial")])

    queue = make_queue(StreamingClient())
    events = []
    queue.event_handler = events.append
    await queue.start()
    receipt = await queue.submit_input("live")
    await started.wait()
    web = await WorkspaceInteractiveSession(queue).start()
    cli = ScriptedView([])
    cli_id = queue.attach(cli.handle_event)
    card = PycodexCard()
    card_id = queue.attach(card.apply_event)
    try:
        assert web.snapshot()["running"]
        assert web.snapshot()["turns"][-1]["thinking"] == "part"
        assert cli.display.stream_buffer == card.display.stream_buffer == "part"
        release.set()
        result = await receipt.future
        assert all(
            event.turn_id == "" and event.submission_id is None for event in deltas
        )
        assert [
            (event.delta, event.turn_id, event.submission_id)
            for event in events
            if isinstance(event, AssistantDeltaEvent)
        ] == [(event.delta, result.turn_id, receipt.submission_id) for event in deltas]
        assert len(web.snapshot()["turns"]) == 1
        assert web.snapshot()["turns"][0]["response"] == "partial"
        assert card.output_text == "partial"
        assert cli.lines.count("assistant> partial") == 1
        assert queue.snapshot()["active_turn"] is None
    finally:
        release.set()
        queue.detach(cli_id)
        queue.detach(card_id)
        cli.close()
        await web.close()


@pytest.mark.parametrize("threaded", [False, True])
@pytest.mark.parametrize("transport", ["http", "websocket"])
def test_web_transports_deliver_structured_answers(threaded, transport):
    from fastapi.testclient import TestClient

    class QuestionTool(BaseTool):
        name = "question_fixture"
        description = "Exercise the runtime input transport."

        def __init__(self, manager):
            self.manager = manager

        async def run(self, context, args):
            return await self.manager.request(args)

    client = ControlClient(
        [
            ModelResponse(
                [ToolCall("question", "question_fixture", question_payload())]
            ),
            ModelResponse([AssistantMessage("answered")]),
        ]
    )

    def build_session():
        tools = ToolRegistry()
        tools.register(
            QuestionTool(tools.runtime_environment.request_user_input_manager)
        )
        return WorkspaceInteractiveSession(
            AgentRuntime(Agent(client, tools, ContextConfig()))
        )

    def build_frontend():
        if threaded:
            return ThreadedWorkspaceInteractiveSession(
                build_session, asyncio.get_running_loop()
            )
        return build_session()

    with TestClient(create_app(build_frontend, None)) as browser:
        assert (
            browser.post("/api/session/message", json={"prompt": "ask"}).status_code
            == 200
        )
        for attempt in range(100):
            state = browser.get("/api/session").json()["snapshot"]
            if state["input_request"] is not None:
                break
            time.sleep(0.01)
        assert state["input_request"] is not None
        assert "Choose a path" in state["input_request"]["text"]
        assert len(state["turns"]) == 1
        answer = {
            "request_id": state["input_request"]["request_id"],
            "answer": {
                "answers": {
                    "first": {"answers": ["Alpha"]},
                    "second": {"answers": ["Gamma"]},
                }
            },
        }
        if transport == "http":
            response = browser.post("/api/session/message", json=answer)
            assert response.status_code == 200
            assert response.json()["type"] == "answered"
        else:
            with browser.websocket_connect("/ws/session") as websocket:
                websocket.send_json(dict(answer, type="answer"))
                for attempt in range(100):
                    response = websocket.receive_json()
                    if response["type"] == "send_result":
                        break
                assert response["result"]["type"] == "answered"
        stale = browser.post("/api/session/message", json=answer)
        assert stale.status_code == 400
        assert "no longer pending" in stale.json()["error"]
        for attempt in range(100):
            state = browser.get("/api/session").json()["snapshot"]
            if state["turns"][-1]["status"] == "completed":
                break
            time.sleep(0.01)
        assert state["input_request"] is None
        assert len(state["turns"]) == 1
        assert state["turns"][-1]["response"] == "answered"
    assert client.call_count == 2


def test_ipython_bootstrap_keeps_bare_agent(monkeypatch):
    import pycodex.cli

    client = ControlClient([])
    agent = Agent(client, ToolRegistry(), ContextConfig())
    attached = []
    monkeypatch.setattr(pycodex.cli, "build_model", lambda path: client)
    monkeypatch.setattr(pycodex.cli, "build_agent", lambda **kwargs: agent)
    monkeypatch.setattr(
        "pycodex.tools.ipython_tool.attach_ipython_tool", attached.append
    )
    assert pycodex.cli.ipython_agent("unused") is agent
    assert attached == [agent]
    assert not hasattr(agent, "submission_queue")
