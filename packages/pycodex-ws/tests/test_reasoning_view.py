import asyncio
import json
import threading
import time

import pytest
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from pycodex.bootstrap import LOCAL_RESPONSES_SERVER_API_KEY_ENV
from pycodex.context import ContextConfig
from pycodex.events import (
    AssistantDeltaEvent,
    EventScope,
    ReasoningEvent,
    SessionStateEvent,
    StreamErrorEvent,
    TurnFailedEvent,
    TurnInterruptedEvent,
    TurnStartedEvent,
)
from pycodex.multi_agents import AgentsManager
from pycodex.protocol import (
    AssistantMessage,
    ModelResponse,
    ReasoningItem,
    ToolCall,
    UserMessage,
)
from pycodex.tools import BaseTool
from pycodex_ws.feishu_card import PycodexCard
from pycodex_ws.feishu_link import PycodexRuntimeLink
from tests.fakes import ScriptedModelClient
from tests.responses_server.fake_chat_completions_server import RunningASGITestServer
from workspace_server import (
    WebSessionView,
    WorkspaceDefinition,
    WorkspaceInteractiveSession,
    build_parser,
    create_app,
)
from workspace_server import app as app_module
from workspace_server.reloader import _reproject_view
from test_workspace_server import make_session


def _summary_reasoning_item(*texts: str) -> ReasoningItem:
    return ReasoningItem(
        payload={
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": text} for text in texts],
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("preview_reasoning,retry", [(False, False), (True, False), (True, True)])
async def test_late_attach_mid_turn_and_invisible_post_turn_reasoning_fallback(tmp_path, preview_reasoning, retry):
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    mixed_output = asyncio.Event()
    finish_reply = asyncio.Event()

    class HoldTool(BaseTool):
        name = "hold"
        description = "Hold execution"
        input_schema = {"type": "object", "properties": {}}

        async def run(self, context, args):
            tool_started.set()
            await release_tool.wait()
            return "done"

    class MidTurnReasoningModel(ScriptedModelClient):
        def __init__(self):
            super().__init__([])
            self.step = 0

        async def complete(self, prompt):
            self.step += 1
            if self.step == 1:
                EventScope.emit(AssistantDeltaEvent("Starting."))
                EventScope.emit(
                    ReasoningEvent("Raw step-by-step analysis before hold tool.")
                )
                return ModelResponse(
                    [
                        AssistantMessage("Starting.", phase="commentary"),
                        ReasoningItem(
                            payload={
                                "type": "reasoning",
                                "content": [
                                    {
                                        "type": "reasoning_text",
                                        "text": "Raw step-by-step analysis before hold tool.",
                                    }
                                ],
                                "summary": [
                                    {
                                        "type": "summary_text",
                                        "text": "Ignored summary when raw text exists.",
                                    }
                                ],
                            }
                        ),
                        ToolCall("c1", "hold", {}),
                    ]
                )
            if retry:
                EventScope.emit(AssistantDeltaEvent("Abandoned text."))
                EventScope.emit(ReasoningEvent("Abandoned thinking."))
                EventScope.emit(StreamErrorEvent("Retrying", 1, 2, 0.0, "timeout"))
            EventScope.emit(AssistantDeltaEvent("Turn 1"))
            EventScope.emit(ReasoningEvent("Verify"))
            EventScope.emit(ReasoningEvent("Verify result."))
            EventScope.emit(AssistantDeltaEvent(" complete."))
            mixed_output.set()
            await finish_reply.wait()
            return ModelResponse([AssistantMessage("Turn 1 complete.")])

    manager = AgentsManager(
        MidTurnReasoningModel(),
        ContextConfig(cwd=tmp_path, codex_home=tmp_path / ".codex"),
        toolset=(),
    )
    root = manager.create_root_runtime()
    root.agent.tool_registry.register(HoldTool())
    session = await WorkspaceInteractiveSession(root, preview_reasoning=preview_reasoning).start()
    try:
        card = PycodexCard(preview_reasoning=preview_reasoning)
        link = PycodexRuntimeLink(root, "unused", card=card)
        root.attach(link._handle_runtime_event, interactive=False)
        cursor = session.view.api_snapshot()["cursor"]
        receipt = await root.submit_input("run tool")
        await asyncio.wait_for(tool_started.wait(), 2)

        # A newly attached WebSessionView mid-tool receives active_turn.reasoning (preferring raw text).
        late_view = WebSessionView(preview_reasoning=preview_reasoning)
        late_view.handle_event(SessionStateEvent("attach", root.snapshot()))
        assert late_view.snapshot()["spinner"] == "working"
        raw = "Raw step-by-step analysis before hold tool."
        expected = raw if preview_reasoning else ""
        assert root.state_snapshot()["active_turn"]["reasoning"] == raw
        assert root.state_snapshot()["active_turn"]["assistant_text"] == ""
        assert late_view.snapshot()["turns"][-1]["thinking"] == expected
        expected_preview = raw if preview_reasoning else "Starting."
        assert late_view.summary()["last_assistant"] == session.view.summary()["last_assistant"] == expected_preview
        assert card.preview_output == expected_preview
        agents_payload = await session.api_query("agents", {})
        assert agents_payload["root"]["last_message"] == " ".join(expected_preview.split())

        release_tool.set()
        await asyncio.wait_for(mixed_output.wait(), 2)
        state = root.snapshot()
        assert state["active_turn"]["assistant_text"] == "Turn 1 complete."
        assert state["active_turn"]["reasoning"] == ""
        mixed = "Turn 1 complete."
        late_view.handle_event(SessionStateEvent("attach", state))
        late_card = PycodexCard(preview_reasoning=preview_reasoning)
        late_card.apply_event(SessionStateEvent("attach", state))
        assert session.view.summary()["last_assistant"] == late_view.summary()["last_assistant"] == mixed
        assert card.preview_output == late_card.preview_output == mixed
        queue = session.view.subscribe(cursor)
        replay = queue.get_nowait()
        session.view.unsubscribe(queue)
        assert not replay["reset_required"]
        updates = [item for item in replay["events"] if item.get("kind") == "reasoning"]
        assert updates[-1]["payload"]["text"] == "Verify result."
        assert updates[-1]["snapshot"]["turns"][-1]["thinking"] == (
            "Verify result." if preview_reasoning else "Turn 1"
        )
        assert updates[-1]["snapshot"]["spinner"] == "talking"
        if retry:
            error = next(item for item in replay["events"] if item.get("kind") == "stream_error")
            assert error["snapshot"]["turns"][-1]["thinking"] == raw
            assert error["snapshot"]["spinner"] == "reconnecting"
        finish_reply.set()
        await receipt.future
        assert (await session.api_query("agents", {}))["root"]["last_message"] == "Turn 1 complete."

        # Simulate an invisible system wakeup turn ending with only summary reasoning (no AssistantMessage).
        root.agent._history.extend(
            [
                UserMessage("<EPHEMERAL_MESSAGE>wakeup</EPHEMERAL_MESSAGE>", visible=False),
                _summary_reasoning_item(
                    "**Scheduled check**\n\nVerified background job finished.\n\n\n"
                ),
            ]
        )
        post_turn_view = WebSessionView(preview_reasoning=preview_reasoning)
        post_turn_view.handle_event(SessionStateEvent("attach", root.snapshot()))
        expected_preview = (
            "**Scheduled check**\n\nVerified background job finished.\n\n\n"
            if preview_reasoning else "Turn 1 complete."
        )
        assert post_turn_view.summary()["last_assistant"] == expected_preview
        # Reload reprojects the attached view from history without a new attach event.
        restore = _reproject_view(root, session.view)
        assert session.view.summary()["last_assistant"] == expected_preview
        restore()
        assert session.view.summary()["last_assistant"] == "Turn 1 complete."
    finally:
        release_tool.set()
        finish_reply.set()
        await session.close()


@pytest.mark.parametrize("preview_reasoning", [False, True])
def test_feishu_card_replaces_reasoning_with_streamed_reply(preview_reasoning) -> None:
    card = PycodexCard(preview_reasoning=preview_reasoning)
    session = make_session()
    attach = SessionStateEvent("attach", session.runtime.snapshot())
    card.apply_event(attach)
    card.apply_event(
        TurnStartedEvent("t1", (UserMessage("analyze logs", turn_id="t1"),))
    )
    card.apply_event(
        ReasoningEvent("First step.\n\n\n**Analyzing**\n\nReading recent log entries.", "t1")
    )
    assert card.preview_output == ("First step.\n\n\n**Analyzing**\n\nReading recent log entries." if preview_reasoning else "")
    assert card.status == "understood"

    card.apply_event(AssistantDeltaEvent("Found ", "t1"))
    card.apply_event(AssistantDeltaEvent("0 errors.", "t1"))
    assert card.preview_output == "Found 0 errors."
    assert card.status == "talking"

    # Mid-turn attach with reasoning and empty assistant_text populates working_output.
    reattached = PycodexCard(preview_reasoning=preview_reasoning)
    reattached.apply_event(
        SessionStateEvent(
            "attach",
            {
                "title": "Demo",
                "model": "gemini-4-carbon-high:high",
                "cwd": "/tmp",
                "busy": True,
                "closed": False,
                "accepts_input": True,
                "context_window": 256000,
                "max_context_window": 256000,
                "auto_compact_token_limit": None,
                "usage_tokens": None,
                "background_work_count": 0,
                "input_request": None,
                "history": (),
                "queued_inputs": [],
                "active_turn": {
                    "turn_id": "t2",
                    "submission_id": "s2",
                    "messages": (UserMessage("check status", turn_id="t2"),),
                    "assistant_text": "",
                    "reasoning": "Previous step.\n\n**Reattached**\n\nStill inspecting.",
                    "model_errors": [],
                },
            },
        )
    )
    assert reattached.preview_output == ("Previous step.\n\n**Reattached**\n\nStill inspecting." if preview_reasoning else "")

    # Interrupted or failed turns clear uncommitted reasoning from working_output, while keeping partial assistant text.
    interrupted_card = PycodexCard(preview_reasoning=preview_reasoning)
    interrupted_card.apply_event(attach)
    interrupted_card.apply_event(
        TurnStartedEvent("t3", (UserMessage("stop soon", turn_id="t3"),))
    )
    interrupted_card.apply_event(ReasoningEvent("**Thinking**\n\nInterrupted.", "t3"))
    assert interrupted_card.preview_output == ("**Thinking**\n\nInterrupted." if preview_reasoning else "")
    interrupted_card.apply_event(TurnInterruptedEvent("t3", 1, None, 0))
    assert interrupted_card.preview_output == ""

    failed_reasoning_card = PycodexCard(preview_reasoning=preview_reasoning)
    failed_reasoning_card.apply_event(attach)
    failed_reasoning_card.apply_event(
        TurnStartedEvent("t4", (UserMessage("crash", turn_id="t4"),))
    )
    failed_reasoning_card.apply_event(ReasoningEvent("**Thinking**\n\nBefore crash.", "t4"))
    failed_reasoning_card.apply_event(TurnFailedEvent("t4", 1, "boom", "ValueError", 0))
    assert failed_reasoning_card.preview_output == ""


def test_pyagy_subagent_and_root_reasoning_end_to_end_without_wire_pollution(
    tmp_path, monkeypatch
):
    home_dir = tmp_path / "home"
    codex_dir = home_dir / ".codex"
    codex_dir.mkdir(parents=True)
    (home_dir / ".gemini" / "antigravity-cli").mkdir(parents=True)
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("CODEX_HOME", str(codex_dir))
    monkeypatch.setenv("PYAGY_THOUGHT_TEST_KEY", "test-secret")
    monkeypatch.setenv(LOCAL_RESPONSES_SERVER_API_KEY_ENV, "dummy")

    mid_hold = threading.Event()
    subagent_entered = threading.Event()
    root_step2_entered = threading.Event()
    captured_bodies = []
    lock = threading.Lock()
    root_calls = [0]

    async def generate(request):
        body = json.loads((await request.body()).decode("utf-8"))
        sys_text = "\n".join(
            p.get("text", "")
            for p in (body.get("systemInstruction") or {}).get("parts", [])
        )
        is_subagent = "<subagent_reminder>" in sys_text
        contents = body.get("contents") or []
        has_func_resp = (
            any(
                "functionResponse" in part
                for part in (contents[-1].get("parts") or [])
            )
            if contents
            else False
        )
        with lock:
            captured_bodies.append(body)
            if not is_subagent:
                root_calls[0] += 1
                current_root_call = root_calls[0]
            else:
                current_root_call = 0

        if not is_subagent and current_root_call == 1:
            chunks = [
                [
                    {
                        "thought": True,
                        "text": "**Planning delegation**\n\nPreparing research subagent.\n\n\n",
                    }
                ],
                [
                    {
                        "thought": True,
                        "text": "**Delegating research**\n\nSpawning Explorer subagent now.\n\n\n",
                    },
                    {
                        "functionCall": {
                            "id": "call_sub_1",
                            "name": "invoke_subagent",
                            "args": {
                                "TypeName": "research",
                                "Role": "Explorer",
                                "Prompt": "Check module boundaries",
                            },
                        },
                        "thoughtSignature": "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=",
                    },
                ],
            ]
        elif not is_subagent and current_root_call == 2:
            chunks = [
                [
                    {
                        "thought": True,
                        "text": "**Waiting on Explorer**\n\nAwaiting subagent report.\n\n\n",
                    }
                ],
                [{"text": "Waiting for subagent completion."}],
            ]
        elif is_subagent and not has_func_resp:
            chunks = [
                [
                    {
                        "thought": True,
                        "text": "Preparing to inspect.\n\n\n**Subagent inspecting**\n\nScanning module boundaries.\n\n\n",
                    },
                ],
                [
                    {"text": "Opening repo."},
                    {
                        "thought": True,
                        "text": "**Subagent grep**\n\nRunning grep_search on pycodex.\n\n\n",
                    },
                    {
                        "functionCall": {
                            "id": "call_grep_1",
                            "name": "grep_search",
                            "args": {
                                "SearchPath": str(work_dir),
                                "Query": "nonexistent_token",
                            },
                        },
                        "thoughtSignature": "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=",
                    },
                ],
            ]
        elif is_subagent and has_func_resp:
            chunks = [
                [
                    {
                        "thought": True,
                        "text": "**Subagent wrapping up**\n\nSummarizing findings for parent.\n\n\n",
                    }
                ],
                [{"text": "Subagent final report."}],
            ]
        else:
            chunks = [[{"text": "Root final synthesis."}]]

        async def events():
            total = len(chunks)
            for idx, parts in enumerate(chunks):
                cand = {"content": {"role": "model", "parts": parts}}
                payload = {"candidates": [cand]}
                if idx == total - 1:
                    cand["finishReason"] = "STOP"
                    payload["usageMetadata"] = {
                        "promptTokenCount": 50,
                        "candidatesTokenCount": 20,
                        "thoughtsTokenCount": 15,
                        "totalTokenCount": 85,
                    }
                yield "data: " + json.dumps(payload) + "\n\n"
                if not is_subagent and current_root_call == 2 and idx == 0:
                    root_step2_entered.set()
                    await asyncio.get_running_loop().run_in_executor(
                        None, mid_hold.wait, 5.0
                    )
                if is_subagent and not has_func_resp and idx == 0:
                    subagent_entered.set()
                    await asyncio.get_running_loop().run_in_executor(
                        None, mid_hold.wait, 5.0
                    )

        return StreamingResponse(events(), media_type="text/event-stream")

    downstream = RunningASGITestServer(
        Starlette(
            routes=[Route("/v1beta/models/{model_path:path}", generate, methods=["POST"])]
        )
    )
    downstream.start()
    try:
        config_path = codex_dir / "config.toml"
        config_path.write_text(
            'model = "gemini-4-carbon-high"\n'
            'model_provider = "gpool"\n'
            'compat_api = "gemini"\n'
            'harness = "pyagy"\n'
            "[model_providers.gpool]\n"
            'base_url = "{}/v1beta"\n'
            'env_key = "PYAGY_THOUGHT_TEST_KEY"\n'.format(downstream.base_url),
            encoding="utf-8",
        )
        definition = WorkspaceDefinition("test", tmp_path / "state", work_dir, False)
        entry = app_module._build_workspace_entry(
            definition, build_parser().parse_args(["--config", str(config_path)])
        )
        with TestClient(create_app(entry.manager, definition.state_path)) as browser:
            session_id = browser.get("/api/sessions").json()["sessions"][0]["id"]
            posted = browser.post(
                "/api/session/message",
                json={"session_id": session_id, "prompt": "Delegate boundary check"},
            )
            assert posted.status_code == 200
            assert subagent_entered.wait(timeout=5.0)
            assert root_step2_entered.wait(timeout=5.0)

            end_mid = time.monotonic() + 3.0
            agents_mid = None
            while time.monotonic() < end_mid:
                candidate = browser.get(f"/api/sessions/{session_id}/agents").json()
                if (
                    candidate["agents"]
                    and "Subagent inspecting" in candidate["agents"][0]["last_message"]
                    and "Waiting on Explorer" in candidate["root"]["last_message"]
                ):
                    agents_mid = candidate
                    break
                time.sleep(0.02)
            assert agents_mid is not None, candidate
            assert (
                agents_mid["root"]["last_message"]
                == "**Waiting on Explorer** Awaiting subagent report."
            )
            assert (
                agents_mid["agents"][0]["last_message"]
                == "Preparing to inspect. **Subagent inspecting** Scanning module boundaries."
            )
            child_state = browser.get(
                f"/api/sessions/{session_id}/state",
                params={"agent_id": agents_mid["agents"][0]["agent_id"]},
            ).json()["state"]
            assert child_state["active_turn"]["assistant_text"] == ""
            assert child_state["active_turn"]["reasoning"].startswith("Preparing to inspect.")
            sessions_mid = browser.get("/api/sessions").json()["sessions"][0]
            assert (
                sessions_mid["last_assistant"]
                == "**Waiting on Explorer**\n\nAwaiting subagent report.\n\n\n"
            )

            mid_hold.set()

            end_at = time.monotonic() + 8.0
            final_snap = None
            while time.monotonic() < end_at:
                snap = browser.get(
                    "/api/session", params={"session_id": session_id}
                ).json()["snapshot"]
                if snap["turns"] and snap["turns"][-1]["status"] == "completed" and snap["turns"][-1]["response"] == "Root final synthesis.":
                    final_snap = snap
                    break
                time.sleep(0.05)
            assert final_snap is not None
            assert final_snap["turns"][-1]["thinking"] == ""
            assert (
                browser.get("/api/sessions").json()["sessions"][0]["last_assistant"]
                == "Root final synthesis."
            )

            # Verify zero wire pollution: no model content part has thought summary promoted to non-thought text.
            assert len(captured_bodies) >= 4
            history = browser.get(f"/api/sessions/{session_id}/history").json()["items"]
            thoughts = [
                item["payload"] for item in history
                if item["payload"].get("type") == "reasoning" and item["payload"].get("content")
            ]
            assert thoughts[0]["summary"] == []
            assert thoughts[0]["content"] == [{
                "type": "reasoning_text",
                "text": "**Planning delegation**\n\nPreparing research subagent.\n\n\n"
                        "**Delegating research**\n\nSpawning Explorer subagent now.\n\n\n",
            }]
            for req_body in captured_bodies:
                for content_item in req_body.get("contents") or []:
                    if content_item.get("role") != "model":
                        continue
                    for part in content_item.get("parts") or []:
                        if "text" in part and not part.get("thought"):
                            assert not part["text"].startswith("**")
    finally:
        mid_hold.set()
        downstream.stop()
