"""Unit and integration tests for pyagy 12+2 tools, models, config, resume, and CLI."""

import asyncio
from datetime import datetime, timezone
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import subprocess
import threading
import typing

import pytest

from pycodex.protocol import ToolCall, UserMessage
from pycodex.tools.base_tool import ToolContext

from pyagy import __version__
from pyagy.bootstrap import build_agy_session
from pyagy.cli import main as pyagy_main
from pyagy.context import (
    AGY_AUTO_COMPACTION_PROMPT,
    AGY_MANUAL_COMPACTION_PROMPT,
    SpawnedSubagentSummary,
    format_checkpoint_transcript_text,
    format_wire_context_summary,
    normalize_agy_usage,
)
from pyagy.models import (
    OFFICIAL_MODELS,
    normalize_effort_level,
    resolve_agy_model,
    resolve_subagent_model,
)
from pyagy.task_manager import cron_matches, next_cron_fire_time, parse_cron_expression
from pyagy.tools import load_official_declarations
from pyagy.transcript import (
    AGY_TOOL_OUTPUT_MAX_BYTES,
    AgyTranscriptStore,
    truncate_agy_tool_output,
)


def _write_codex_gemini_config(
    root: "Path",
    base_url: "str",
    compat_api: "str" = "gemini",
) -> "Path":
    cfg = root / "config.toml"
    cfg.write_text(
        'model = "gemini-4-carbon-high"\n'
        'model_provider = "gmock"\n\n'
        "[model_providers.gmock]\n"
        'name = "Gemini Mock"\n'
        f'base_url = "{base_url}/v1beta"\n'
        'env_key = "PYAGY_UNIT_API_KEY"\n'
        f'compat_api = "{compat_api}"\n\n'
        "[profiles.alt]\n"
        'model = "gemini-3.8-flash-low"\n'
        'model_provider = "gmock"\n',
        encoding="utf-8",
    )
    return cfg


class _LocalGeminiResponder:
    """Minimal local HTTP server for Gemini SSE and search_web tests."""

    def __init__(
        self,
        handler_fn: "typing.Callable[[str, typing.Dict[str, object]], typing.Tuple[int, bytes, bool]]",
    ) -> "None":
        self.requests: "typing.List[typing.Tuple[str, typing.Dict[str, object]]]" = []
        lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: "str", *args: "object") -> "None":
                return

            def do_POST(self) -> "None":
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                with lock:
                    outer.requests.append((self.path, body))
                    status, payload_bytes, is_gzip = handler_fn(self.path, body)
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream")
                if is_gzip:
                    self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(payload_bytes)))
                self.end_headers()
                self.wfile.write(payload_bytes)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> "str":
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def __enter__(self) -> "_LocalGeminiResponder":
        self._thread.start()
        return self

    def __exit__(self, exc_type: "object", exc: "object", tb: "object") -> "None":
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5.0)


def _sse_bytes(
    parts: "typing.Sequence[typing.Dict[str, object]]",
    extra_cand: "typing.Optional[typing.Dict[str, object]]" = None,
) -> "bytes":
    cand: "typing.Dict[str, object]" = {
        "content": {"role": "model", "parts": list(parts)},
        "finishReason": "STOP",
    }
    if extra_cand:
        cand.update(extra_cand)
    event = {
        "candidates": [cand],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
    }
    return ("data: " + json.dumps(event, separators=(",", ":")) + "\n\n").encode("utf-8")


def _make_tool_context(cwd: "Path", conv_id: "str", call_id: "str") -> "ToolContext":
    return ToolContext(
        turn_id="turn-1",
        history=(),
        cwd=cwd,
        agent_id="root",
        encrypted_messages=False,
        call_id=call_id,
        session_id=conv_id,
    )


def test_models_efforts_and_subagent_tiers() -> "None":
    assert len(OFFICIAL_MODELS) == 25
    for slug, expected_display in OFFICIAL_MODELS:
        spec = resolve_agy_model(slug)
        assert spec.display_name == expected_display
        gen_cfg = spec.generation_config()
        if "3.1-pro" in slug:
            assert spec.api_model == "gemini-3.1-pro-preview"
            assert gen_cfg["maxOutputTokens"] == 65535
            expected_budget = 1024 if slug.endswith("-low") else -1
            assert gen_cfg["thinkingConfig"] == {
                "includeThoughts": True,
                "thinkingBudget": expected_budget,
            }
        else:
            assert gen_cfg["maxOutputTokens"] == 65536
            assert gen_cfg["thinkingConfig"]["includeThoughts"] is True
            assert gen_cfg["thinkingConfig"]["thinkingLevel"] in ("LOW", "MEDIUM", "HIGH", "XHIGH")

    overridden = resolve_agy_model("gemini-4-neon-low", explicit_effort="xhigh")
    assert overridden.display_name == "Gemini 4 Neon (XHigh)"
    assert overridden.generation_config()["thinkingConfig"]["thinkingLevel"] == "XHIGH"

    max_spec = resolve_agy_model("gemini-4-carbon-low", explicit_effort="max")
    assert max_spec.display_name == "Gemini 4 Carbon (High)"
    routed_spec = resolve_agy_model(
        "gemini-4-argon:huayi-vertex-dst", config_effort="xhigh"
    )
    assert routed_spec.api_model == "gemini-4-argon:huayi-vertex-dst"
    assert routed_spec.display_name == "Gemini 4 Argon (High)"
    assert routed_spec.generation_config()["thinkingConfig"]["thinkingLevel"] == "HIGH"
    assert normalize_effort_level("minimal") == "low"
    with pytest.raises(ValueError, match="unsupported reasoning effort"):
        normalize_effort_level("turbo")

    assert resolve_subagent_model(None, "gemini-4-neon-xhigh") == "gemini-4-neon-xhigh"
    assert resolve_subagent_model("inherit", "gemini-3.1-pro-low") == "gemini-3.1-pro-low"
    assert resolve_subagent_model("flash_lite", "gemini-4-carbon-high") == "gemini-3.6-flash-low"
    assert resolve_subagent_model("flash", "gemini-4-carbon-high") == "gemini-3.8-flash-high"
    assert resolve_subagent_model("pro", "gemini-3.6-flash-low") == "gemini-4-carbon-high"
    with pytest.raises(ValueError, match="unsupported subagent Model tier"):
        resolve_subagent_model("ultra", "gemini-4-carbon-high")


def test_pyagy_config_example_loads(tmp_path: "Path") -> "None":
    from pycodex.config import CodexConfig

    repo_root = Path(__file__).resolve().parents[3]
    example_path = repo_root / "config_examples" / "pyagy.config.toml"
    assert example_path.is_file()

    standalone = CodexConfig.from_env(example_path)
    assert standalone.harness == "pyagy"
    assert standalone.provider.compat_api == "gemini"
    assert standalone.provider.model == "gemini-4-carbon-high"
    assert standalone.provider.api_key_env == "GEMINI_API_KEY"

    base_cfg = tmp_path / "config.toml"
    base_cfg.write_text('model = "gpt-5.4"\nmodel_provider = "openai"\n', encoding="utf-8")
    (tmp_path / "pyagy.config.toml").write_text(
        example_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    overlay = CodexConfig.from_env(base_cfg, profile="pyagy")
    assert overlay.profile == "pyagy"
    assert overlay.harness == "pyagy"
    assert overlay.provider.compat_api == "gemini"
    assert overlay.provider.model == "gemini-4-carbon-high"


def test_cron_parser_and_next_fire_time() -> "None":
    parsed = parse_cron_expression("*/15 9-17 * * 1-5,7")
    assert 0 in parsed.days_of_week  # Sunday 7 normalized to 0
    assert 15 in parsed.minutes and 45 in parsed.minutes

    monday_morning = datetime(2026, 11, 16, 8, 50, tzinfo=timezone.utc)
    nxt = next_cron_fire_time(parsed, monday_morning)
    assert (nxt.hour, nxt.minute) == (9, 0)
    assert cron_matches(parsed, nxt)

    with pytest.raises(ValueError, match="5 fields"):
        parse_cron_expression("* * * *")
    with pytest.raises(ValueError, match="out of bounds"):
        parse_cron_expression("61 * * * *")


@pytest.mark.asyncio
async def test_file_tools_and_placeholders(tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch") -> "None":
    monkeypatch.setenv("PYAGY_UNIT_API_KEY", "dummy")
    cfg = _write_codex_gemini_config(tmp_path, "http://127.0.0.1:9")
    app_data = tmp_path / "appdata"
    ws = tmp_path / "ws"
    ws.mkdir()

    session = build_agy_session(config_path=cfg, cwd=ws, app_data_dir=app_data)
    try:
        reg = session.root_node.tool_registry
        conv_id = session.root_node.conversation_id
        official_order, _ = load_official_declarations()
        assert "ask_question" in official_order
        assert [s.name for s in reg.model_visible_specs()] == list(official_order)
        assert len(official_order) == 14

        async def _exec(name: "str", args: "typing.Dict[str, object]", cid: "str") -> "typing.Tuple[bool, str]":
            res = await reg.execute(
                ToolCall(call_id=cid, name=name, arguments=args),
                _make_tool_context(ws, conv_id, cid),
            )
            return bool(res.is_error), str(res.output)

        # Disabled tool (`ask_question`) remains in declarations and returns disabled when called
        is_err, out = await _exec(
            "ask_question",
            {"questions": [{"question": "Pick one", "options": ["A", "B"]}]},
            "c_aq",
        )
        assert is_err is True
        assert out == "Encountered error in tool execution: ask_question is disabled"

        # Placeholders
        for ph in ("run_workflow", "swarm"):
            is_err, out = await _exec(ph, {}, f"c_{ph}")
            assert is_err is True
            assert "tool is not enabled in pyagy CLI mode" in out

        # write_file: create, overwrite guard, append, overwrite, artifact metadata
        target = ws / "sub" / "notes.txt"
        err, out = await _exec("write_file", {"Path": str(target), "Content": "a\nb\nc\n"}, "w1")
        assert not err and out.endswith("Created file.")

        err, out = await _exec("write_file", {"Path": str(target), "Content": "dup"}, "w_dup")
        assert err and "already exists" in out

        err, out = await _exec(
            "write_file",
            {"Path": str(target), "Content": "x", "Overwrite": True, "Append": True},
            "w_both",
        )
        assert err and "cannot set both" in out

        err, _ = await _exec(
            "write_file",
            {"Path": str(target), "Content": "d\ne\n", "Append": True},
            "w_app",
        )
        assert not err

        artifact_path = session.root_node.transcript_store.brain_dir / "plan.md"
        session.root_node.transcript_store.ensure_dirs()
        err, _ = await _exec(
            "write_file",
            {
                "Path": str(artifact_path),
                "Content": "# Plan\n",
                "ArtifactMetadata": {"ArtifactType": "implementation_plan", "Summary": "Plan doc"},
            },
            "w_art",
        )
        assert not err
        meta_file = (
            session.root_node.transcript_store.system_dir / "artifacts" / "plan.md.metadata.json"
        )
        assert json.loads(meta_file.read_text(encoding="utf-8"))["Summary"] == "Plan doc"

        # edit_file: multiple occurrences & missing target
        err, out = await _exec(
            "edit_file",
            {"Path": str(target), "Target": "\n", "Replacement": "!\n"},
            "e_multi",
        )
        assert not err
        assert "Made 5 replacements" in out

        err, out = await _exec(
            "edit_file",
            {"Path": str(target), "Target": "nonexistent", "Replacement": "z"},
            "e_miss",
        )
        assert err and "target string not found" in out

        # read_file: StartLine/EndLine, ByteOffset, directory error, >800 lines, >46KB single line
        err, out = await _exec(
            "read_file",
            {"Path": str(target), "StartLine": 2, "EndLine": 4},
            "r_slice",
        )
        assert not err
        assert "Showing lines 2-4 of 5\n2: b!\n3: c!\n4: d!" in out

        err, out = await _exec(
            "read_file",
            {"Path": str(target), "StartLine": 1, "EndLine": 3, "ByteOffset": 3},
            "r_byte",
        )
        assert not err
        assert "(byte offset 3)\nb!\nc!" in out

        err, out = await _exec("read_file", {"Path": str(ws)}, "r_dir")
        assert err and "is a directory" in out

        big_lines = ws / "many.txt"
        big_lines.write_text("".join(f"line-{i}\n" for i in range(1, 951)), encoding="utf-8")
        err, out = await _exec("read_file", {"Path": str(big_lines)}, "r_many")
        assert not err and "Showing lines 1-800 of 950\n" in out

        huge_single = ws / "huge.txt"
        huge_single.write_text("X" * 60000, encoding="utf-8")
        err, out = await _exec("read_file", {"Path": str(huge_single)}, "r_huge")
        assert not err and "Showing lines 1-1 of 1\n1: " in out
        assert len(out.encode("utf-8")) < 47000
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_command_schedule_and_subagent_lifecycle(
    tmp_path: "Path",
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    monkeypatch.setenv("PYAGY_UNIT_API_KEY", "dummy")

    def _responder(_path: "str", _body: "typing.Dict[str, object]") -> "typing.Tuple[int, bytes, bool]":
        return 200, _sse_bytes([{"text": "Subagent finished.", "thoughtSignature": "c3Vi"}]), False

    with _LocalGeminiResponder(_responder) as srv:
        cfg = _write_codex_gemini_config(tmp_path, srv.base_url)
        app_data = tmp_path / "appdata"
        ws = tmp_path / "repo"
        ws.mkdir()
        subprocess.run(["git", "init", str(ws)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(ws), "config", "user.email", "t@example.com"], check=True)
        subprocess.run(["git", "-C", str(ws), "config", "user.name", "Tester"], check=True)
        (ws / "seed.txt").write_text("seed\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(ws), "add", "seed.txt"], check=True)
        subprocess.run(["git", "-C", str(ws), "commit", "-m", "init"], check=True, capture_output=True)

        session = build_agy_session(config_path=cfg, cwd=ws, app_data_dir=app_data)
        try:
            await session.root_node.start()
            reg = session.root_node.tool_registry
            conv_id = session.root_node.conversation_id

            async def _exec(name: "str", args: "typing.Dict[str, object]", cid: "str") -> "typing.Tuple[bool, str]":
                res = await reg.execute(
                    ToolCall(call_id=cid, name=name, arguments=args),
                    _make_tool_context(ws, conv_id, cid),
                )
                return bool(res.is_error), str(res.output)

            # Interactive background command + status + send_input + kill
            env = reg.runtime_environment
            assert env.background_work_count(False) == 0
            assert env.background_work_snapshot() == []

            err, bg_out = await _exec(
                "run_command",
                {
                    "CommandLine": 'read line && echo "GOT:$line" && sleep 10',
                    "WaitMsBeforeAsync": 500,
                    "toolSummary": "interactive read",
                },
                "cmd_bg",
            )
            assert not err and "Tool is running as a background task with task id: " in bg_out
            bg_task_id = bg_out.splitlines()[0].split("task id: ", 1)[1].strip()
            assert env.background_work_count(False) == 1
            assert env.background_work_count(True) == 1
            bg_snap = env.background_work_snapshot()
            assert len(bg_snap) == 1
            assert bg_snap[0]["tool"] == "run_command"
            assert bg_snap[0]["kind"] == "exec"
            assert bg_snap[0]["task_id"] == bg_task_id
            assert bg_snap[0]["is_daemon"] is False
            assert bg_snap[0]["status"] == "running"
            assert bg_snap[0]["tty"] is True
            assert bg_snap[0]["exit_code"] is None
            assert "description" not in bg_snap[0]
            assert "log_path" not in bg_snap[0]
            assert isinstance(bg_snap[0]["pid"], int) and bg_snap[0]["pid"] > 0

            err, send_out = await _exec(
                "manage_task",
                {"Action": "send_input", "TaskId": bg_task_id, "Input": "hello_pty\n"},
                "mt_in",
            )
            assert not err and f'Input sent to task "{bg_task_id}".' in send_out

            await asyncio.sleep(0.25)
            err, st_out = await _exec(
                "manage_task",
                {"Action": "status", "TaskId": bg_task_id},
                "mt_st",
            )
            assert not err and "Status: RUNNING" in st_out and "GOT:hello_pty" in st_out

            err, kill_out = await _exec(
                "manage_task",
                {"Action": "kill", "TaskId": bg_task_id},
                "mt_kill",
            )
            assert not err and f'Task "{bg_task_id}" cancelled.' in kill_out
            assert env.background_work_count(False) == 0
            assert env.background_work_snapshot() == []

            # A daemon does not hold print mode open, but its exit can wake an idle agent.
            err, daemon_out = await _exec(
                "run_command",
                {
                    "CommandLine": "read line",
                    "WaitMsBeforeAsync": 500,
                    "IsDaemon": True,
                    "toolSummary": "dev daemon",
                },
                "cmd_daemon",
            )
            assert not err
            daemon_tid = daemon_out.splitlines()[0].split("task id: ", 1)[1].strip()
            assert env.background_work_count(False) == 1
            assert env.background_work_count(True) == 1
            await session.root_node.wait_until_settled(timeout_seconds=1)
            daemon_snap = env.background_work_snapshot()
            assert len(daemon_snap) == 1 and daemon_snap[0]["is_daemon"] is True
            finished = asyncio.Event()
            session.root_node.add_event_listener(
                lambda event: finished.set() if event.kind == "turn_completed" else None
            )
            err, _ = await _exec("manage_task", {"Action": "send_input", "TaskId": daemon_tid, "Input": "exit\n"}, "mt_exit")
            assert not err
            await asyncio.wait_for(finished.wait(), 5)
            assert any(
                daemon_tid + '" finished' in msg.text
                for msg in session.root_node.agent.history if isinstance(msg, UserMessage)
            )
            assert env.background_work_snapshot() == []

            # Schedule mutual exclusion & TimerCondition="any" early cancel on incoming message
            err, bad1 = await _exec(
                "schedule",
                {"DurationSeconds": 10, "MaxIterations": 2, "Prompt": "bad"},
                "sc_bad1",
            )
            assert err and "MaxIterations cannot be used with DurationSeconds" in bad1

            err, bad2 = await _exec(
                "schedule",
                {"CronExpression": "* * * * *", "TimerCondition": "any", "Prompt": "bad"},
                "sc_bad2",
            )
            assert err and "TimerCondition cannot be used with CronExpression" in bad2

            err, cond_out = await _exec(
                "schedule",
                {"DurationSeconds": 60, "TimerCondition": "any", "Prompt": "cond timer"},
                "sc_cond",
            )
            assert not err
            cond_tid = cond_out.splitlines()[0].split("task id: ", 1)[1].strip()
            assert env.background_work_count(True) == 1
            assert env.background_work_snapshot()[0]["kind"] == "schedule"
            assert "description" not in env.background_work_snapshot()[0]
            assert "log_path" not in env.background_work_snapshot()[0]
            session.root_node.deliver_system_message("peer ping", sender="peer-1", wake_if_idle=False)
            timer = session.root_node.task_manager._tasks[cond_tid]
            assert timer.status == "CANCELLED"
            assert not timer.log_path.with_suffix(".schedule.json").exists()
            assert env.background_work_count(True) == 0
            assert env.background_work_snapshot() == []

            # Waking TimerCondition message sets runtime busy BEFORE emitting background_work update
            err, wake_cond_out = await _exec(
                "schedule",
                {"DurationSeconds": 60, "TimerCondition": "any", "Prompt": "wake timer"},
                "sc_wake_cond",
            )
            assert not err
            busy_on_bg_update: "typing.List[bool]" = []
            obs_id = session.root_node.runtime.attach(
                lambda ev: busy_on_bg_update.append(session.root_node.runtime.is_busy)
                if getattr(ev, "kind", "") == "runtime_updated"
                and getattr(ev, "reason", "") == "background_work"
                else None,
                interactive=False,
            )
            try:
                session.root_node.deliver_system_message(
                    "peer wake", sender="peer-2", wake_if_idle=True
                )
                assert busy_on_bg_update == [True]
                await session.root_node.wait_until_settled(timeout_seconds=5.0)
            finally:
                session.root_node.runtime.detach(obs_id)

            # task_manager.restart() cancels running tasks and resets count/snapshot
            err, _ = await _exec(
                "schedule",
                {"DurationSeconds": 60, "Prompt": "restart timer"},
                "sc_restart",
            )
            assert not err and env.background_work_count(True) == 1
            session.root_node.task_manager.restart()
            assert env.background_work_count(True) == 0
            assert env.background_work_snapshot() == []

            # define_subagent, invoke_subagent (branch & share), send_message, manage_subagents
            err, builtin_err = await _exec(
                "define_subagent",
                {"Name": "research", "Description": "d", "SystemPrompt": "s"},
                "def_bad",
            )
            assert err and "cannot redefine built-in subagent type" in builtin_err

            err, def_out = await _exec(
                "define_subagent",
                {
                    "Name": "writer_bot",
                    "Description": "Writer",
                    "SystemPrompt": "Write code.",
                    "EnableWriteTools": True,
                    "EnableSubagentTools": True,
                },
                "def_ok",
            )
            assert not err and 'Subagent "writer_bot" defined successfully.' in def_out

            err, inv_branch = await _exec(
                "invoke_subagent",
                {
                    "TypeName": "writer_bot",
                    "Role": "BranchWorker",
                    "Prompt": "Work in branch",
                    "Model": "flash",
                    "Workspace": "branch",
                },
                "inv_b",
            )
            assert not err and '"conversationId":' in inv_branch
            branch_cid = json.loads(
                inv_branch[inv_branch.index("{") : inv_branch.rindex("}") + 1]
            )["conversationId"]
            branch_node = session.controller.get_node(branch_cid)
            assert (branch_node.cwd / "seed.txt").is_file()
            assert env.background_work_count(True) == 1
            sub_snap = env.background_work_snapshot()
            assert len(sub_snap) == 1 and sub_snap[0]["kind"] == "subagent"
            assert sub_snap[0]["conversation_id"] == branch_cid

            err, inv_share = await _exec(
                "invoke_subagent",
                {
                    "TypeName": "writer_bot",
                    "Role": "ShareWorker",
                    "Prompt": "Work in git worktree",
                    "Workspace": "share",
                },
                "inv_s",
            )
            assert not err
            share_cid = json.loads(
                inv_share[inv_share.index("{") : inv_share.rindex("}") + 1]
            )["conversationId"]
            share_node = session.controller.get_node(share_cid)
            assert share_node._git_worktree_repo == ws.resolve()

            await branch_node.wait_until_settled(timeout_seconds=5.0)
            await share_node.wait_until_settled(timeout_seconds=5.0)
            assert env.background_work_count(True) == 0
            assert env.background_work_snapshot() == []

            assert branch_node.runtime.can_close_for_capacity() is True
            await branch_node.task_manager.schedule(
                prompt="hold capacity",
                step_index=999,
                duration_seconds=60,
                timer_condition="any",
            )
            assert branch_node.runtime.can_close_for_capacity() is False
            branch_node.task_manager.kill(f"{branch_cid}/task-999")
            assert branch_node.runtime.can_close_for_capacity() is True

            branch_node.deliver_system_message("status update", wake_if_idle=False)
            assert branch_node.state == "idle"
            assert not branch_node.runtime.is_busy
            assert env.background_work_count(True) == 0

            err, sm_out = await _exec(
                "send_message",
                {"Recipient": branch_cid, "Message": "Follow-up task"},
                "sm_1",
            )
            assert not err and f'Message sent to "{branch_cid}".' in sm_out
            assert env.background_work_count(True) == 1
            await branch_node.wait_until_settled(timeout_seconds=5.0)
            assert env.background_work_count(True) == 0

            err, kill_sub = await _exec(
                "manage_subagents",
                {"Action": "kill", "ConversationIds": [branch_cid]},
                "ms_kill",
            )
            assert not err and "Killed roles: BranchWorker" in kill_sub
            assert not branch_node.cwd.exists()

            err, kill_all = await _exec(
                "manage_subagents",
                {"Action": "kill_all"},
                "ms_killall",
            )
            assert not err and "Killed roles: ShareWorker" in kill_all
            assert not share_node.cwd.exists()
        finally:
            await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", ["conversation", "rollout"])
async def test_schedule_wakes_after_resume(tmp_path, monkeypatch, resume):
    from pyagy import task_manager

    now = 1000.0
    cron_times = iter([now, now + 120])
    monkeypatch.setattr(task_manager, "time", lambda: now)
    monkeypatch.setattr(task_manager, "_next_cron_deadline", lambda expr, exact: next(cron_times))
    monkeypatch.setenv("PYAGY_UNIT_API_KEY", "dummy")
    with _LocalGeminiResponder(lambda path, body: (200, _sse_bytes([{"text": "ok"}]), False)) as srv:
        options = dict(
            config_path=_write_codex_gemini_config(tmp_path, srv.base_url), cwd=tmp_path,
            app_data_dir=tmp_path / "appdata", enable_rollout_recording=(resume == "rollout"),
        )
        session = build_agy_session(**options)
        root = session.root_node
        try:
            await root.run_user_prompt("start")
            conversation_id = root.conversation_id
            rollout = root.agent.recorded_session_file_path
            fired = asyncio.Event()
            root.add_event_listener(lambda event: fired.set() if event.kind == "turn_completed" else None)
            await root.task_manager.schedule("tick", 100, cron_expression="* * * * *", is_daemon=True, max_iterations=2)
            await asyncio.wait_for(fired.wait(), 5)
            await root.task_manager.schedule("one-shot", 101, duration_seconds=60)
            await root.task_manager.schedule("cancelled", 102, duration_seconds=60)
            root.task_manager.kill(f"{conversation_id}/task-102")
            assert root.runtime.state_snapshot()["background_work_count"] == 2
        finally:
            await session.close()

        now += 300
        if resume == "conversation":
            options["resume_conversation_id"] = conversation_id
        restored = build_agy_session(**options)
        root = restored.root_node
        try:
            await root.start()
            if resume == "rollout":
                assert root.runtime.state_snapshot()["background_work_count"] == 0
                root.runtime.resume(rollout)
            assert not root.runtime.is_busy
            assert {row["task_id"] for row in root.runtime.background_work_snapshot()} == {
                f"{conversation_id}/task-100", f"{conversation_id}/task-101",
            }
            await root.wait_until_settled(timeout_seconds=5)
            assert root.runtime.state_snapshot()["background_work_count"] == 0
            messages = [msg.text for msg in root.agent.history if isinstance(msg, UserMessage)]
            assert sum(" content=tick\n" in text for text in messages) == 2
            assert sum(" content=one-shot\n" in text for text in messages) == 1
            assert not list(root.task_manager.tasks_dir.glob("*.schedule.json"))
        finally:
            await restored.close()


@pytest.mark.asyncio
async def test_search_web_grounding_and_fallback(
    tmp_path: "Path",
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    monkeypatch.setenv("PYAGY_UNIT_API_KEY", "dummy")
    attempt = [0]

    def _responder(_path: "str", body: "typing.Dict[str, object]") -> "typing.Tuple[int, bytes, bool]":
        attempt[0] += 1
        if attempt[0] == 1:
            assert body.get("tools") == [{"googleSearch": {}}]
            sse = _sse_bytes(
                [{"text": "Python 3.13 released."}],
                extra_cand={
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": "https://docs.python.org/3/", "title": "Python Docs"}}
                        ]
                    }
                },
            )
            return 200, gzip.compress(sse), True
        if body.get("tools") == [{"googleSearch": {}}]:
            return 400, b'{"error":"googleSearch unsupported"}', False
        return 200, _sse_bytes([{"text": "Fallback search answer."}]), False

    with _LocalGeminiResponder(_responder) as srv:
        cfg = _write_codex_gemini_config(tmp_path, srv.base_url)
        session = build_agy_session(
            config_path=cfg,
            cwd=tmp_path,
            app_data_dir=tmp_path / "appdata",
        )
        try:
            reg = session.root_node.tool_registry
            cid = session.root_node.conversation_id
            r1 = await reg.execute(
                ToolCall(
                    call_id="sw1",
                    name="search_web",
                    arguments={"query": "python release", "domain": "python.org"},
                ),
                _make_tool_context(tmp_path, cid, "sw1"),
            )
            assert not r1.is_error
            assert "Python 3.13 released." in str(r1.output)
            assert "- [Python Docs](https://docs.python.org/3/)" in str(r1.output)

            r2 = await reg.execute(
                ToolCall(call_id="sw2", name="search_web", arguments={"query": "fallback test"}),
                _make_tool_context(tmp_path, cid, "sw2"),
            )
            assert not r2.is_error
            assert "Fallback search answer." in str(r2.output)
        finally:
            await session.close()


def test_config_validation_transcript_fallback_and_cli_modes(
    tmp_path: "Path",
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    monkeypatch.setenv("PYAGY_UNIT_API_KEY", "dummy")

    # Reject non-gemini provider
    bad_cfg = _write_codex_gemini_config(tmp_path, "http://127.0.0.1:9", compat_api="chat_completions")
    with pytest.raises(ValueError, match="compat_api='gemini'"):
        build_agy_session(config_path=bad_cfg, cwd=tmp_path, app_data_dir=tmp_path / "appdata")

    # CLI subcommands: models, agents, help, --version
    out = io.StringIO()
    err = io.StringIO()
    assert pyagy_main(["models"], stdout=out, stderr=err) == 0
    model_lines = out.getvalue().strip().splitlines()
    assert model_lines[0] == "Fetching available models..."
    assert len(model_lines) == 26

    out = io.StringIO()
    assert pyagy_main(["agents"], stdout=out, stderr=err) == 0
    assert "research\t" in out.getvalue() and "self\t" in out.getvalue()

    out = io.StringIO()
    assert pyagy_main(["help"], stdout=out, stderr=err) == 0
    assert "--output-format" in out.getvalue()

    out = io.StringIO()
    assert pyagy_main(["--version"], stdout=out, stderr=err) == 0
    assert __version__ in out.getvalue()

    # Transcript fallback reconstruction from transcript_full.jsonl (without pyagy_session.json)
    app_data = tmp_path / "appdata_fb"
    store = AgyTranscriptStore(app_data_dir=app_data, conversation_id="conv-fb", cwd=tmp_path)
    store.append_transcript_step(
        source="USER_EXPLICIT",
        step_type="USER_INPUT",
        content=(
            "<USER_REQUEST>\nhi\n</USER_REQUEST>\n"
            "<ADDITIONAL_METADATA>\nThe current local time is: 2026-11-20T10:00:00+08:00.\n</ADDITIONAL_METADATA>\n"
            "<USER_SETTINGS_CHANGE>\nThe user changed setting `Model Selection` from None to Gemini 4 Carbon (High). "
            "No need to comment on this change if the user doesn't ask about it. "
            "If reporting what model you are, please use a human readable name instead of the exact string.\n</USER_SETTINGS_CHANGE>"
        ),
    )
    store.append_transcript_step(
        source="MODEL",
        step_type="PLANNER_RESPONSE",
        content="hello!",
        thinking="pondering",
        model="gemini-4-carbon-high",
    )
    restored = AgyTranscriptStore.load_conversation(app_data, "conv-fb")
    assert restored.model == "gemini-4-carbon-high"
    assert restored.last_reported_model_display == "Gemini 4 Carbon (High)"
    assert len(restored.history) == 3

    # CLI json output, piped stdin, and interactive REPL with slash commands
    turn_counter = [0]

    def _cli_responder(_path: "str", _body: "typing.Dict[str, object]") -> "typing.Tuple[int, bytes, bool]":
        turn_counter[0] += 1
        return 200, _sse_bytes([{"text": f"Reply #{turn_counter[0]}", "thoughtSignature": "c2ln"}]), False

    with _LocalGeminiResponder(_cli_responder) as srv:
        cfg = _write_codex_gemini_config(tmp_path, srv.base_url)
        cli_appdata = tmp_path / "cli_appdata"

        # 1. --output-format json with --profile alt and piped prompt
        json_out = io.StringIO()
        json_err = io.StringIO()
        rc = pyagy_main(
            [
                "--config",
                str(cfg),
                "--profile",
                "alt",
                "--app-data-dir",
                str(cli_appdata),
                "--output-format",
                "json",
                "-p",
            ],
            stdin=io.StringIO("piped hello\n"),
            stdout=json_out,
            stderr=json_err,
            cwd=tmp_path,
        )
        assert rc == 0, json_err.getvalue()
        parsed_json = json.loads(json_out.getvalue())
        assert parsed_json["status"] == "SUCCESS"
        assert parsed_json["response"] == "Reply #1\n"

        # 2. interactive REPL: /help, /status, /model, /effort, normal prompt, /exit
        repl_in = io.StringIO(
            "/help\n"
            "/status\n"
            "/model gemini-3.8-flash-medium\n"
            "/effort low\n"
            "hello repl\n"
            "/exit\n"
        )
        repl_out = io.StringIO()
        repl_err = io.StringIO()
        rc = pyagy_main(
            [
                "--config",
                str(cfg),
                "--app-data-dir",
                str(cli_appdata),
            ],
            stdin=repl_in,
            stdout=repl_out,
            stderr=repl_err,
            cwd=tmp_path,
        )
        assert rc == 0, repl_err.getvalue()
        rendered = repl_out.getvalue()
        assert "Model set to gemini-3.8-flash-medium" in rendered
        assert "Reasoning effort set to low" in rendered
        assert "Reply #2" in rendered


def test_compaction_thresholds_formatting_and_repl(
    tmp_path: "Path",
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    monkeypatch.setenv("PYAGY_UNIT_API_KEY", "dummy")

    # 1. Exact prompt lengths & per-model compaction generationConfig / thresholds
    assert len(AGY_AUTO_COMPACTION_PROMPT) == 3931
    assert len(AGY_AUTO_COMPACTION_PROMPT.encode("utf-8")) == 3937
    assert len(AGY_MANUAL_COMPACTION_PROMPT) == 4051

    carbon = resolve_agy_model("gemini-4-carbon-high")
    assert carbon.compaction_generation_config()["maxOutputTokens"] == 32768
    assert carbon.resolve_auto_compact_threshold() == 512000
    assert carbon.resolve_auto_compact_threshold("50%") == 512000
    assert carbon.resolve_auto_compact_threshold("80%") == 819200
    assert carbon.resolve_auto_compact_threshold("12345") == 12345

    flash38 = resolve_agy_model("gemini-3.8-flash-medium")
    assert flash38.compaction_generation_config()["maxOutputTokens"] == 32768
    assert flash38.resolve_auto_compact_threshold() == 256000

    for slug_16k in (
        "gemini-4-hydrogen-high",
        "gemini-3.7-flash-low",
        "gemini-3.6-flash-low",
        "gemini-3.1-pro-low",
    ):
        spec_16k = resolve_agy_model(slug_16k)
        assert spec_16k.compaction_generation_config()["maxOutputTokens"] == 16384
        assert spec_16k.resolve_auto_compact_threshold() == 256000
        assert spec_16k.resolve_auto_compact_threshold("50%") == 256000

    custom = resolve_agy_model("custom-exp-model")
    assert custom.compaction_generation_config()["maxOutputTokens"] == 32768
    assert custom.resolve_auto_compact_threshold() is None
    assert custom.resolve_auto_compact_threshold("50%") == 500000

    for bad_override in ("0", "0%", "101%", "-5", "abc", "50.5%"):
        with pytest.raises(ValueError, match="invalid AGY_CLI_AUTO_COMPACT_OVERRIDE"):
            carbon.resolve_auto_compact_threshold(bad_override)

    # 2. Multi-subagent sorting, double-space formatting, and [-10:] slicing
    subs = [
        SpawnedSubagentSummary(
            type_name="self",
            role="Zeta",
            conversation_id="ffff0000-0000-0000-0000-000000000002",
            log_absolute_uri="file:///tmp/z/transcript.jsonl",
            workspace_uris=("file:///tmp/ws",),
        ),
        SpawnedSubagentSummary(
            type_name="research",
            role="Alpha",
            conversation_id="00001111-0000-0000-0000-000000000001",
            log_absolute_uri="file:///tmp/a/transcript.jsonl",
            workspace_uris=("file:///tmp/ws",),
        ),
    ]
    twelve_reqs = [f"req-{i}" for i in range(1, 13)]
    wire_sum = format_wire_context_summary(
        summary_text="<summary>ok</summary>",
        local_iso="2026-11-20T12:00:00+08:00",
        user_requests=twelve_reqs,
        subagents=subs,
        is_subagent=False,
    )
    assert "1. req-3\n" in wire_sum and "10. req-12\n" in wire_sum
    assert "req-1\n" not in wire_sum
    assert "# Subagents (2 total)\n" in wire_sum
    assert "communicate with them.\n\n{\n" in wire_sum
    assert wire_sum.index("00001111-") < wire_sum.index("ffff0000-")
    assert '"typeName": "research"' in wire_sum

    sub_ckpt = format_checkpoint_transcript_text(
        summary_text="<summary>sub</summary>",
        parent_messages=[f"pm-{i}" for i in range(1, 13)],
        is_subagent=True,
    )
    assert (
        "The following were messages from your parent agent in the truncated conversation in chronological order:\n1. pm-3"
        in sub_ckpt
    )

    # 3. Transcript CHECKPOINT fallback without pyagy_session.json
    fb_appdata = tmp_path / "ckpt_fb"
    fb_store = AgyTranscriptStore(app_data_dir=fb_appdata, conversation_id="conv-ckpt", cwd=tmp_path)
    fb_store.append_transcript_step(source="USER_EXPLICIT", step_type="USER_INPUT", content="before compact")
    fb_store.append_transcript_step(
        source="MODEL", step_type="PLANNER_RESPONSE", content="old reply", model="gemini-4-carbon-high"
    )
    fb_store.append_transcript_step(
        source="SYSTEM", step_type="CHECKPOINT", content="# Resuming from a compaction\n\n<summary>saved</summary>"
    )
    fb_store.append_transcript_step(source="USER_EXPLICIT", step_type="USER_INPUT", content="after compact")
    restored_ckpt = AgyTranscriptStore.load_conversation(fb_appdata, "conv-ckpt")
    assert restored_ckpt.user_requests == ("before compact", "after compact")
    assert len(restored_ckpt.history) == 2
    assert restored_ckpt.history[0].text.startswith("<CONTEXT_SUMMARY>")
    assert restored_ckpt.history[1].text == "after compact"

    # 4. Interactive REPL /compact commands (empty history guard, bare /compact, /compact <arg>)
    def _responder(_path: "str", body: "typing.Dict[str, object]") -> "typing.Tuple[int, bytes, bool]":
        last_txt = json.dumps(body["contents"][-1])
        if "ON-DEMAND COMPACTION" in last_txt:
            return 200, _sse_bytes([{"text": "<summary>REPL summary</summary>"}]), False
        return 200, _sse_bytes([{"text": "REPL normal reply"}]), False

    with _LocalGeminiResponder(_responder) as srv:
        cfg = _write_codex_gemini_config(tmp_path, srv.base_url)
        repl_in = io.StringIO(
            "/compact\n"
            "first turn in repl\n"
            "/compact\n"
            "/compact follow up in repl\n"
            "/exit\n"
        )
        repl_out = io.StringIO()
        repl_err = io.StringIO()
        rc = pyagy_main(
            [
                "--config",
                str(cfg),
                "--app-data-dir",
                str(tmp_path / "repl_cmp_appdata"),
            ],
            stdin=repl_in,
            stdout=repl_out,
            stderr=repl_err,
            cwd=tmp_path,
        )
        assert rc == 0, repl_err.getvalue()
        repl_rendered = repl_out.getvalue()
        assert "Nothing to compact yet.\n" in repl_rendered
        assert "Conversation compacted.\n" in repl_rendered
        assert repl_rendered.count("REPL normal reply\n") == 2


@pytest.mark.asyncio
async def test_subagent_idle_fallback_and_parent_report_suppression(
    tmp_path: "Path",
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    monkeypatch.setenv("PYAGY_UNIT_API_KEY", "dummy")
    sub_step = [0]

    def _responder(_path: "str", body: "typing.Dict[str, object]") -> "typing.Tuple[int, bytes, bool]":
        sys_txt = "".join(
            str(p.get("text", ""))
            for p in (body.get("systemInstruction") or {}).get("parts", [])
        )
        contents_dump = json.dumps(body["contents"])
        if "<subagent_reminder>" in sys_txt:
            if "SilentCheck" in contents_dump:
                return 200, _sse_bytes([{"text": ""}]), False
            sub_step[0] += 1
            if sub_step[0] == 1:
                root_id = sys_txt.split('id: "', 1)[1].split('"', 1)[0]
                return (
                    200,
                    _sse_bytes(
                        [
                            {
                                "functionCall": {
                                    "id": "sm_parent",
                                    "name": "send_message",
                                    "args": {"Recipient": root_id, "Message": "Direct report to root"},
                                },
                                "thoughtSignature": "c2lnX3Nt",
                            }
                        ]
                    ),
                    False,
                )
            return 200, _sse_bytes([{"text": "Local sub note"}]), False
        return 200, _sse_bytes([{"text": "Root ack"}]), False

    with _LocalGeminiResponder(_responder) as srv:
        cfg = _write_codex_gemini_config(tmp_path, srv.base_url)
        session = build_agy_session(
            config_path=cfg,
            cwd=tmp_path,
            app_data_dir=tmp_path / "sub_idle_appdata",
        )
        try:
            await session.root_node.start()
            # 1. Silent subagent without send_message -> emits empty-output idle fallback
            inv1 = await session.controller.invoke_subagent(
                caller_conversation_id=session.root_node.conversation_id,
                type_name="research",
                role="SilentSub",
                prompt="SilentCheck",
            )
            cid1 = json.loads(inv1[inv1.index("{") : inv1.rindex("}") + 1])["conversationId"]
            await session.controller.get_node(cid1).wait_until_settled(timeout_seconds=5.0)
            await session.root_node.wait_until_settled(timeout_seconds=5.0)

            # 2. Subagent calling send_message(Recipient=root) -> delivers message, suppresses idle notice
            inv2 = await session.controller.invoke_subagent(
                caller_conversation_id=session.root_node.conversation_id,
                type_name="self",
                role="ReportingSub",
                prompt="ReportViaSendMessage",
            )
            cid2 = json.loads(inv2[inv2.index("{") : inv2.rindex("}") + 1])["conversationId"]
            await session.controller.get_node(cid2).wait_until_settled(timeout_seconds=5.0)
            await session.root_node.wait_until_settled(timeout_seconds=5.0)

            root_texts = [
                it.text for it in session.root_node.agent.history if hasattr(it, "text")
            ]
            assert any(
                f"Subagent {cid1} went idle without reporting anything back." in txt
                for txt in root_texts
            )
            assert any("Direct report to root" in txt for txt in root_texts)
            assert not any(f"Subagent {cid2} has gone idle." in txt for txt in root_texts)
        finally:
            await session.close()


@pytest.mark.asyncio
async def test_tool_output_truncation_and_step_persistence(
    tmp_path: "Path",
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    monkeypatch.setenv("PYAGY_UNIT_API_KEY", "dummy")

    # 1. UTF-8 multibyte boundary safety and byte cap in truncate_agy_tool_output
    step_out = tmp_path / "steps" / "7" / "output.txt"
    chinese_payload = "前缀\n" + ("中文测试" * 2000) + "\n后缀"
    truncated = truncate_agy_tool_output(chinese_payload, output_path=step_out)
    assert len(truncated.encode("utf-8")) <= AGY_TOOL_OUTPUT_MAX_BYTES
    assert f"file://{step_out}" in truncated
    assert "<truncated " in truncated and " bytes from the middle of the output>" in truncated
    assert step_out.read_text(encoding="utf-8") == chinese_payload

    def _responder(_path: "str", _body: "typing.Dict[str, object]") -> "typing.Tuple[int, bytes, bool]":
        return 200, _sse_bytes([{"text": "Ack.", "thoughtSignature": "c2ln"}]), False

    with _LocalGeminiResponder(_responder) as srv:
        cfg = _write_codex_gemini_config(tmp_path, srv.base_url)
        app_data = tmp_path / "trunc_appdata"
        ws = tmp_path / "ws"
        ws.mkdir()
        session = build_agy_session(config_path=cfg, cwd=ws, app_data_dir=app_data)
        try:
            await session.root_node.start()
            reg = session.root_node.tool_registry
            conv_id = session.root_node.conversation_id
            store = session.root_node.transcript_store

            # 2. Small sync run_command (<10 KiB) persists steps/<idx>/output.txt without truncating
            r_small = await reg.execute(
                ToolCall(
                    call_id="cmd_small",
                    name="run_command",
                    arguments={
                        "CommandLine": "echo hello_small",
                        "WaitMsBeforeAsync": 5000,
                        "toolSummary": "small echo",
                    },
                ),
                _make_tool_context(ws, conv_id, "cmd_small"),
            )
            assert not r_small.is_error
            assert "<truncated " not in str(r_small.output)
            assert store.step_output_path(0).is_file()
            assert "hello_small" in store.step_output_path(0).read_text(encoding="utf-8")

            # 3. Large sync run_command (>10 KiB) truncates at <=10 KiB and writes full output.txt
            r_big_sync = await reg.execute(
                ToolCall(
                    call_id="cmd_big_sync",
                    name="run_command",
                    arguments={
                        "CommandLine": "python3 -c 'print(\"S\" * 18000)'",
                        "WaitMsBeforeAsync": 5000,
                        "toolSummary": "big sync",
                    },
                ),
                _make_tool_context(ws, conv_id, "cmd_big_sync"),
            )
            sync_txt = str(r_big_sync.output)
            assert len(sync_txt.encode("utf-8")) <= AGY_TOOL_OUTPUT_MAX_BYTES
            assert f"file://{store.step_output_path(1)}" in sync_txt
            assert len(store.step_output_path(1).read_bytes()) > 18000

            # 4. Large async background run_command (>10 KiB) truncates <SYSTEM_MESSAGE> body
            finished = asyncio.Event()
            session.root_node.add_event_listener(
                lambda ev: finished.set() if ev.kind == "turn_completed" else None
            )
            r_big_bg = await reg.execute(
                ToolCall(
                    call_id="cmd_big_bg",
                    name="run_command",
                    arguments={
                        "CommandLine": "sleep 0.7 && python3 -c 'print(\"B\" * 25000)'",
                        "WaitMsBeforeAsync": 500,
                        "toolSummary": "big bg",
                    },
                ),
                _make_tool_context(ws, conv_id, "cmd_big_bg"),
            )
            assert "Tool is running as a background task" in str(r_big_bg.output)
            await asyncio.wait_for(finished.wait(), timeout=8.0)

            sys_msgs = [
                it.text
                for it in session.root_node.agent.history
                if isinstance(it, UserMessage) and "finished with result:" in it.text
            ]
            assert len(sys_msgs) == 1
            assert len(sys_msgs[0].encode("utf-8")) < 11200
            assert f"file://{store.step_output_path(2)}" in sys_msgs[0]
            assert "<truncated " in sys_msgs[0]
            assert len(store.step_output_path(2).read_bytes()) > 25000
            assert len((store.tasks_dir / "task-2.log").read_bytes()) >= 25000

            # 5. read_file keeps its own 46,080-byte limit (not truncated at 10 KiB)
            twenty_kb = ws / "twenty_kb.txt"
            twenty_kb.write_text(("0123456789abcdef" * 4 + "\n") * 300, encoding="utf-8")
            r_read = await reg.execute(
                ToolCall(
                    call_id="read_20k",
                    name="read_file",
                    arguments={"Path": str(twenty_kb)},
                ),
                _make_tool_context(ws, conv_id, "read_20k"),
            )
            assert not r_read.is_error
            assert len(str(r_read.output).encode("utf-8")) > 19000
            assert "The tool's output was truncated" not in str(r_read.output)
        finally:
            await session.close()


@pytest.mark.asyncio
async def test_legacy_anchor_resume_and_compaction_overflow_recovery(
    tmp_path: "Path",
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    monkeypatch.setenv("PYAGY_UNIT_API_KEY", "dummy")

    # Verify cached_tokens accounting matches official agy
    norm_cached = normalize_agy_usage(
        {
            "input_tokens": 520000,
            "input_tokens_details": {"cached_tokens": 400000},
            "output_tokens": 21000,
            "output_tokens_details": {"reasoning_tokens": 20000},
            "total_tokens": 541000,
        }
    )
    assert norm_cached == {
        "input_tokens": 120000,
        "output_tokens": 21000,
        "thinking_tokens": 20000,
        "cache_read_tokens": 400000,
        "total_tokens": 141000,
    }

    overflow_err = json.dumps(
        {
            "error": {
                "code": 400,
                "message": (
                    "The input token count exceeds the maximum number of tokens allowed "
                    "(1048576); input token count: 1113435."
                ),
                "status": "INVALID_ARGUMENT",
            }
        }
    ).encode("utf-8")

    compaction_attempts = [0]

    def _responder(_path: "str", body: "typing.Dict[str, object]") -> "typing.Tuple[int, bytes, bool]":
        dumped = json.dumps(body["contents"], ensure_ascii=False)
        if "Wrap your response in <summary></summary> tags." in dumped:
            compaction_attempts[0] += 1
            # First compaction attempt has the legacy 300 KB <SYSTEM_MESSAGE> -> return Gemini 400!
            if "X" * 50000 in dumped:
                return 400, overflow_err, False
            # Second compaction attempt has truncated the oversized <SYSTEM_MESSAGE> -> succeeds!
            return 200, _sse_bytes([{"text": "<summary>Recovered Carbon summary</summary>"}]), False

        # Normal turn: return high promptTokenCount (432,476) on turn 1
        if "seed turn" in dumped and "<CONTEXT_SUMMARY>" not in dumped:
            ev = {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "Seed reply"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 432400,
                    "candidatesTokenCount": 76,
                    "totalTokenCount": 432476,
                },
            }
            return 200, ("data: " + json.dumps(ev) + "\n\n").encode("utf-8"), False

        return 200, _sse_bytes([{"text": "Post-recovery reply"}]), False

    with _LocalGeminiResponder(_responder) as srv:
        cfg = _write_codex_gemini_config(tmp_path, srv.base_url)
        app_data = tmp_path / "carbon_appdata"
        codex_home = tmp_path / "codex_home"
        monkeypatch.setenv("CODEX_HOME", str(codex_home))

        # Step 1: Run initial session with rollout recording enabled (simulating pycodex-ws Carbon)
        s1 = build_agy_session(
            config_path=cfg,
            cwd=tmp_path,
            app_data_dir=app_data,
            enable_rollout_recording=True,
        )
        try:
            await s1.root_node.run_user_prompt("seed turn")
            conv_id = s1.root_node.conversation_id
            rollout_path = s1.root_node.agent.recorded_session_file_path
            assert rollout_path is not None and rollout_path.is_file()
            assert s1.root_node.agent.anchor_tokens == 432476

            # Inject a legacy un-truncated 300 KB <SYSTEM_MESSAGE> AFTER the anchor (like task-874)
            s1.root_node.deliver_system_message(
                'Task id "legacy/task-874" finished with result:\n' + ("X" * 300000),
                sender="legacy/task-874",
                wake_if_idle=False,
            )
            # Commit pending message into rollout & pyagy_session.json, then strip anchor_history_len
            # to simulate legacy pyagy_session.json written before this fix.
            s1.root_node.agent.receive_messages(
                (s1.root_node.runtime._pending[0].messages[0],)
                if getattr(s1.root_node.runtime, "_pending", ())
                else ()
            )
            from pyagy.context import make_system_user_message

            legacy_big_msg = make_system_user_message(
                content='Task id "legacy/task-874" finished with result:\n' + ("X" * 300000),
                sender="legacy/task-874",
            )
            s1.root_node.agent._append_history([legacy_big_msg])
            s1.root_node.save_session_state()
            state_path = s1.root_node.transcript_store.session_state_path
            raw_state = json.loads(state_path.read_text(encoding="utf-8"))
            assert raw_state["anchor_history_len"] < len(raw_state["history"])
            del raw_state["anchor_history_len"]
            state_path.write_text(json.dumps(raw_state, ensure_ascii=False), encoding="utf-8")
        finally:
            await s1.close()

        # Step 2: Resume via runtime.resume(rollout_path) (exact pycodex-ws reboot path!)
        s2 = build_agy_session(
            config_path=cfg,
            cwd=tmp_path,
            app_data_dir=app_data,
            enable_rollout_recording=True,
        )
        try:
            await s2.root_node.start()
            s2.root_node.runtime.resume(rollout_path)
            assert s2.root_node.agent.anchor_tokens == 432476
            # Legacy 300 KB <SYSTEM_MESSAGE> MUST sit after anchor_history_len, pushing estimate > 512,000!
            assert s2.root_node.agent.anchor_history_len < len(s2.root_node.agent.history)
            assert (
                s2.root_node.agent.anchor_tokens + s2.root_node.agent._estimate_delta_tokens()
                >= 512000
            )

            summary = await s2.root_node.run_user_prompt("continue")
            assert summary.final_response == "Post-recovery reply"
            # Compaction hit Gemini 400 once, truncated the 300 KB <SYSTEM_MESSAGE>, and succeeded on attempt 2!
            assert compaction_attempts[0] == 2
        finally:
            await s2.close()


@pytest.mark.asyncio
async def test_sample_overflow_fallback_and_tool_pruning_compaction(
    tmp_path: "Path",
    monkeypatch: "pytest.MonkeyPatch",
) -> "None":
    monkeypatch.setenv("PYAGY_UNIT_API_KEY", "dummy")

    overflow_err = json.dumps(
        {
            "error": {
                "code": 400,
                "message": (
                    "The input token count exceeds the maximum number of tokens allowed "
                    "(1048576); input token count: 1090000."
                ),
            }
        }
    ).encode("utf-8")

    big_file_1 = tmp_path / "f1.txt"
    big_file_2 = tmp_path / "f2.txt"
    big_file_1.write_text("OLD_READ_CONTENT_" * 400, encoding="utf-8")
    big_file_2.write_text("NEW_READ_CONTENT_" * 400, encoding="utf-8")

    step = [0]

    def _responder(_path: "str", body: "typing.Dict[str, object]") -> "typing.Tuple[int, bytes, bool]":
        dumped = json.dumps(body["contents"], ensure_ascii=False)
        if "Wrap your response in <summary></summary> tags." in dumped:
            # Force pruning of the oldest read_file tool result before compaction fits!
            if "OLD_READ_CONTENT_" in dumped:
                return 400, overflow_err, False
            assert "NEW_READ_CONTENT_" in dumped
            return 200, _sse_bytes([{"text": "<summary>Pruned oldest tool and compacted</summary>"}]), False

        step[0] += 1
        if step[0] == 1:
            return (
                200,
                _sse_bytes(
                    [
                        {
                            "functionCall": {
                                "id": "rf1",
                                "name": "read_file",
                                "args": {"Path": str(big_file_1)},
                            },
                            "thoughtSignature": "c2lnMQ==",
                        },
                        {
                            "functionCall": {
                                "id": "rf2",
                                "name": "read_file",
                                "args": {"Path": str(big_file_2)},
                            },
                        },
                    ]
                ),
                False,
            )
        if step[0] == 2:
            # Simulate unexpected context overflow during normal _sample (e.g. unestimated tokens)
            return 400, overflow_err, False
        return 200, _sse_bytes([{"text": "Recovered from sample overflow"}]), False

    with _LocalGeminiResponder(_responder) as srv:
        cfg = _write_codex_gemini_config(tmp_path, srv.base_url)
        session = build_agy_session(
            config_path=cfg,
            cwd=tmp_path,
            app_data_dir=tmp_path / "overflow_appdata",
        )
        try:
            res = await session.root_node.run_user_prompt("read both files")
            assert res.final_response == "Recovered from sample overflow"
            text_items = [
                it.text for it in session.root_node.agent.history if hasattr(it, "text")
            ]
            assert "<CONTEXT_SUMMARY>" in str(text_items[0])
            assert text_items[-1] == "Recovered from sample overflow"
        finally:
            await session.close()
