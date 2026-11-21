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

from pycodex.protocol import ToolCall
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
)
from pyagy.models import (
    OFFICIAL_MODELS,
    normalize_effort_level,
    resolve_agy_model,
    resolve_subagent_model,
)
from pyagy.task_manager import cron_matches, next_cron_fire_time, parse_cron_expression
from pyagy.tools import load_official_declarations
from pyagy.transcript import AgyTranscriptStore


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
            session.root_node.deliver_system_message("peer ping", sender="peer-1", wake_if_idle=False)
            assert session.root_node.task_manager._tasks[cond_tid].status == "CANCELLED"

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

            err, sm_out = await _exec(
                "send_message",
                {"Recipient": branch_cid, "Message": "Follow-up task"},
                "sm_1",
            )
            assert not err and f'Message sent to "{branch_cid}".' in sm_out
            await branch_node.wait_until_settled(timeout_seconds=5.0)

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
    assert '"typeName":  "research"' in wire_sum

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
