"""Antigravity CLI (`agy 1.5.13`) wire-compatible command-line entrypoint."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
import typing

from pycodex.events import (
    Event,
    MessagesAddedEvent,
    ToolStartedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
)
from pycodex.protocol import UserMessage
from pycodex.utils.codex_config import DEFAULT_CODEX_CONFIG_PATH

from . import __version__
from .bootstrap import AgyBootstrappedSession, build_agy_session
from .context import (
    AGY_COMPACTION_PROMPT_NAME,
    AGY_RAW_USER_NAME,
    AGY_SYSTEM_MESSAGE_NAME,
)
from .models import OFFICIAL_MODELS, normalize_effort_level
from .multi_agents import AgyCheckpointEvent

BUILTIN_AGENTS: "typing.Tuple[typing.Tuple[str, str], ...]" = (
    ("self", "General-purpose coding agent with full workspace tools"),
    ("research", "Read-only codebase and web research subagent"),
    ("image-generator", "Image generation and visual asset preparation subagent"),
)


def parse_duration_seconds(raw_value: "typing.Optional[str]") -> "typing.Optional[float]":
    """Parse Go-style duration (`0s`, `15s`, `500ms`, `2m`, `1h`) or float seconds."""
    if raw_value is None:
        return None
    text = str(raw_value).strip().lower()
    if not text:
        return None
    multiplier = 1.0
    if text.endswith("ms"):
        multiplier = 0.001
        num_str = text[:-2]
    elif text.endswith("s"):
        multiplier = 1.0
        num_str = text[:-1]
    elif text.endswith("m"):
        multiplier = 60.0
        num_str = text[:-1]
    elif text.endswith("h"):
        multiplier = 3600.0
        num_str = text[:-1]
    else:
        num_str = text
    seconds = float(num_str) * multiplier
    if seconds < 0:
        raise ValueError(f"duration must be non-negative: {raw_value!r}")
    return None if seconds == 0.0 else seconds


def _format_trailing_newline(text: "str") -> "str":
    if not text:
        return ""
    return text if text.endswith("\n") else (text + "\n")


def _write_ndjson(stream: "typing.TextIO", obj: "typing.Dict[str, object]") -> "None":
    stream.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def _run_models_subcommand(
    argv: "typing.Sequence[str]",
    stdout: "typing.TextIO",
) -> "int":
    parser = argparse.ArgumentParser(
        prog="pyagy models",
        description="List available models",
    )
    parser.parse_args(list(argv))
    stdout.write("Fetching available models...\n")
    for slug, display_name in OFFICIAL_MODELS:
        stdout.write(f"{slug}\t{display_name}\n")
    stdout.flush()
    return 0


def _run_agents_subcommand(
    argv: "typing.Sequence[str]",
    stdout: "typing.TextIO",
) -> "int":
    parser = argparse.ArgumentParser(
        prog="pyagy agent",
        description="List available agents",
    )
    parser.parse_args(list(argv))
    for name, desc in BUILTIN_AGENTS:
        stdout.write(f"{name}\t{desc}\n")
    stdout.flush()
    return 0


def build_cli_parser() -> "argparse.ArgumentParser":
    parser = argparse.ArgumentParser(
        prog="pyagy",
        description="Antigravity CLI (agy 1.5.13) wire-compatible mode built on pycodex",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Prompt text or subcommand (models, agent, agents)",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"{__version__} (agy 1.5.13 wire-compatible)",
    )
    parser.add_argument(
        "-p",
        "--print",
        "--prompt",
        dest="print_prompt",
        nargs="?",
        const="",
        default=None,
        help="Run a single prompt non-interactively and print the response",
    )
    parser.add_argument(
        "-i",
        "--prompt-interactive",
        dest="interactive_prompt",
        nargs="?",
        const="",
        default=None,
        help="Run an initial prompt interactively and continue the session",
    )
    parser.add_argument(
        "-c",
        "--continue",
        dest="continue_latest",
        action="store_true",
        help="Continue the most recent conversation",
    )
    parser.add_argument(
        "--conversation",
        dest="conversation_id",
        default=None,
        help="Resume a previous conversation by ID",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model for the current CLI session",
    )
    parser.add_argument(
        "--effort",
        default=None,
        help="Reasoning effort for the current CLI session (low|medium|high|xhigh|max)",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CODEX_CONFIG_PATH),
        help="Path to Codex config.toml (default: ~/.codex/config.toml)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Profile name in ~/.codex/config.toml",
    )
    parser.add_argument(
        "--add-dir",
        dest="add_dirs",
        action="append",
        default=[],
        help="Add a directory to the workspace (repeatable)",
    )
    parser.add_argument(
        "--app-data-dir",
        default=None,
        help="Override AGY App Data Directory (default: ~/.gemini/antigravity-cli)",
    )
    parser.add_argument(
        "--output-format",
        choices=("text", "json", "stream-json"),
        default="text",
        help="Output format for print mode (text, json, stream-json)",
    )
    parser.add_argument(
        "--input-format",
        choices=("text", "stream-json"),
        default="text",
        help="Input format for print mode (text, stream-json)",
    )
    parser.add_argument(
        "--print-timeout",
        default="0s",
        help="Optional time limit for print mode; 0s waits until turn completes",
    )
    parser.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help="Auto-approve all tool permission requests without prompting",
    )
    parser.add_argument(
        "--mode",
        choices=("accept-edits", "auto", "plan"),
        default="auto",
        help="Agent execution mode for this session",
    )
    return parser


def _attach_stream_json_observer(
    session: "AgyBootstrappedSession",
    stdout: "typing.TextIO",
) -> "None":
    root = session.root_node
    seen_msg_ids: "typing.Set[str]" = set()
    step_seq = [root.transcript_store.step_index]

    def _next_idx() -> "int":
        val = step_seq[0]
        step_seq[0] += 1
        return val

    def _on_event(event: "Event") -> "None":
        if isinstance(event, MessagesAddedEvent):
            for msg in event.messages:
                if (
                    isinstance(msg, UserMessage)
                    and msg.message_id not in seen_msg_ids
                    and msg.name not in (AGY_RAW_USER_NAME, AGY_COMPACTION_PROMPT_NAME)
                ):
                    seen_msg_ids.add(msg.message_id)
                    stype = (
                        "system_message"
                        if msg.name == AGY_SYSTEM_MESSAGE_NAME
                        else "user_input"
                    )
                    _write_ndjson(
                        stdout,
                        {
                            "event": "step_update",
                            "step_update": {
                                "conversation_id": root.conversation_id,
                                "step_index": _next_idx(),
                                "state": "DONE",
                                "step_type": stype,
                            },
                        },
                    )
        elif isinstance(event, AgyCheckpointEvent):
            _write_ndjson(
                stdout,
                {
                    "event": "step_update",
                    "step_update": {
                        "conversation_id": root.conversation_id,
                        "step_index": _next_idx(),
                        "state": "DONE",
                        "step_type": "checkpoint",
                        "duration_seconds": event.duration_seconds,
                        "usage": dict(event.usage),
                    },
                },
            )
        elif isinstance(event, TurnCompletedEvent):
            if event.output_text is not None:
                _write_ndjson(
                    stdout,
                    {
                        "event": "step_update",
                        "step_update": {
                            "conversation_id": root.conversation_id,
                            "step_index": _next_idx(),
                            "state": "DONE",
                            "step_type": "agent_response",
                            "text_delta": _format_trailing_newline(event.output_text),
                        },
                    },
                )

    root.add_event_listener(_on_event)


async def _run_print_mode_async(
    session: "AgyBootstrappedSession",
    input_format: "str",
    output_format: "str",
    prompt_text: "typing.Optional[str]",
    timeout_seconds: "typing.Optional[float]",
    stdin: "typing.TextIO",
    stdout: "typing.TextIO",
    stderr: "typing.TextIO",
) -> "int":
    root = session.root_node
    session_start = time.monotonic()
    cumulative_turns = 0

    if output_format == "stream-json":
        _write_ndjson(
            stdout,
            {
                "event": "init",
                "conversation_id": root.conversation_id,
                "init": {
                    "model": root.agent.model_name,
                    "cwd": str(root.cwd),
                    "tools": sorted(spec.name for spec in root.tool_registry.model_visible_specs()),
                },
            },
        )
        _attach_stream_json_observer(session, stdout)

    if input_format == "stream-json":
        loop = asyncio.get_running_loop()
        while True:
            raw_line = await loop.run_in_executor(None, stdin.readline)
            if not raw_line:
                break
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                err_msg = f"failed to decode stream input: {exc}"
                stderr.write(f"error: {err_msg}\n")
                stderr.flush()
                _write_ndjson(
                    stdout,
                    {
                        "event": "result",
                        "result": {
                            "conversation_id": root.conversation_id,
                            "status": "ERROR",
                            "response": "",
                            "error": err_msg,
                            "duration_seconds": time.monotonic() - session_start,
                            "num_turns": cumulative_turns,
                        },
                    },
                )
                return 1

            if not isinstance(obj, dict) or "event" not in obj:
                err_msg = 'stream input message is missing the "event" field'
                stderr.write(f"error: {err_msg}\n")
                stderr.flush()
                _write_ndjson(
                    stdout,
                    {
                        "event": "result",
                        "result": {
                            "conversation_id": root.conversation_id,
                            "status": "ERROR",
                            "response": "",
                            "error": err_msg,
                            "duration_seconds": time.monotonic() - session_start,
                            "num_turns": cumulative_turns,
                        },
                    },
                )
                return 1

            ev_type = obj.get("event")
            if ev_type != "user":
                stderr.write(
                    f"warning: ignoring unsupported stream input message event {ev_type!r}\n"
                )
                stderr.flush()
                continue

            msg_field = obj.get("message")
            if not isinstance(msg_field, dict) or not isinstance(
                msg_field.get("content"), str
            ):
                err_msg = 'stream input "user" message is missing the "message" field'
                stderr.write(f"error: {err_msg}\n")
                stderr.flush()
                _write_ndjson(
                    stdout,
                    {
                        "event": "result",
                        "result": {
                            "conversation_id": root.conversation_id,
                            "status": "ERROR",
                            "response": "",
                            "error": err_msg,
                            "duration_seconds": time.monotonic() - session_start,
                            "num_turns": cumulative_turns,
                        },
                    },
                )
                return 1

            try:
                summary = await root.run_user_prompt(
                    msg_field["content"],
                    wait_background=True,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                err_msg = str(exc)
                stderr.write(f"error: {err_msg}\n")
                stderr.flush()
                _write_ndjson(
                    stdout,
                    {
                        "event": "result",
                        "result": {
                            "conversation_id": root.conversation_id,
                            "status": "ERROR",
                            "response": "",
                            "error": err_msg,
                            "duration_seconds": time.monotonic() - session_start,
                            "num_turns": cumulative_turns,
                        },
                    },
                )
                return 1

            cumulative_turns += summary.num_turns
            _write_ndjson(
                stdout,
                {
                    "event": "result",
                    "result": {
                        "conversation_id": summary.conversation_id,
                        "status": "SUCCESS",
                        "response": _format_trailing_newline(summary.final_response),
                        "duration_seconds": time.monotonic() - session_start,
                        "num_turns": cumulative_turns,
                        "usage": dict(root._usage_totals),
                    },
                },
            )
        return 0

    assert prompt_text is not None
    try:
        summary = await root.run_user_prompt(
            prompt_text,
            wait_background=True,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        err_msg = str(exc)
        if output_format == "json":
            stdout.write(
                json.dumps(
                    {
                        "conversation_id": root.conversation_id,
                        "status": "ERROR",
                        "response": "",
                        "error": err_msg,
                        "duration_seconds": time.monotonic() - session_start,
                        "num_turns": 0,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            stdout.flush()
        elif output_format == "stream-json":
            _write_ndjson(
                stdout,
                {
                    "event": "result",
                    "result": {
                        "conversation_id": root.conversation_id,
                        "status": "ERROR",
                        "response": "",
                        "error": err_msg,
                        "duration_seconds": time.monotonic() - session_start,
                        "num_turns": 0,
                    },
                },
            )
        else:
            stderr.write(f"Error: {err_msg}\n")
            stderr.flush()
        return 1

    formatted_resp = _format_trailing_newline(summary.final_response)
    if output_format == "text":
        if formatted_resp:
            stdout.write(formatted_resp)
            stdout.flush()
    elif output_format == "json":
        stdout.write(
            json.dumps(
                {
                    "conversation_id": summary.conversation_id,
                    "status": "SUCCESS",
                    "response": formatted_resp,
                    "duration_seconds": summary.duration_seconds,
                    "num_turns": summary.num_turns,
                    "usage": summary.usage,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        stdout.flush()
    elif output_format == "stream-json":
        _write_ndjson(
            stdout,
            {
                "event": "result",
                "result": {
                    "conversation_id": summary.conversation_id,
                    "status": "SUCCESS",
                    "response": formatted_resp,
                    "duration_seconds": summary.duration_seconds,
                    "num_turns": summary.num_turns,
                    "usage": summary.usage,
                },
            },
        )
    return 0


async def _run_interactive_repl_async(
    session: "AgyBootstrappedSession",
    initial_prompt: "typing.Optional[str]",
    stdin: "typing.TextIO",
    stdout: "typing.TextIO",
    stderr: "typing.TextIO",
) -> "int":
    root = session.root_node

    def _on_event(event: "Event") -> "None":
        if isinstance(event, ToolStartedEvent):
            stdout.write(f"  [tool] {event.call.name}\n")
            stdout.flush()
        elif isinstance(event, TurnFailedEvent):
            stderr.write(f"[turn failed] {event.error_type}: {event.error}\n")
            stderr.flush()

    root.add_event_listener(_on_event)
    await root.start()

    if initial_prompt is not None and initial_prompt.strip():
        summary = await root.run_user_prompt(initial_prompt, wait_background=True)
        if summary.final_response:
            stdout.write(_format_trailing_newline(summary.final_response))
            stdout.flush()

    loop = asyncio.get_running_loop()
    while True:
        stdout.write("agy> ")
        stdout.flush()
        raw_line = await loop.run_in_executor(None, stdin.readline)
        if not raw_line:
            stdout.write("\n")
            stdout.flush()
            break
        line = raw_line.strip()
        if not line:
            continue
        if line in ("/exit", "/quit"):
            break
        if line == "/help":
            stdout.write(
                "Commands:\n"
                "  /model [slug]    Show or switch current model\n"
                "  /effort <level>  Set reasoning effort (low|medium|high|xhigh|max)\n"
                "  /compact [msg]   Compact conversation history (with optional follow-up)\n"
                "  /status          Show active conversation & model status\n"
                "  /exit, /quit     Exit interactive session\n"
            )
            stdout.flush()
            continue
        if line == "/compact" or (line.startswith("/compact") and line[8:9].isspace()):
            if not root.agent.history:
                stdout.write("Nothing to compact yet.\n")
                stdout.flush()
                continue
            summary = await root.run_user_prompt(line, wait_background=True)
            if summary.final_response:
                stdout.write(_format_trailing_newline(summary.final_response))
            else:
                stdout.write("Conversation compacted.\n")
            stdout.flush()
            continue
        if line == "/status":
            stdout.write(
                f"Conversation ID: {root.conversation_id}\n"
                f"Model: {root.agent.model_name}\n"
                f"Effort: {root.explicit_effort or root.config_effort or 'default'}\n"
                f"Workspace: {root.cwd}\n"
            )
            stdout.flush()
            continue
        if line == "/model" or line.startswith("/model "):
            arg = line[6:].strip()
            if not arg:
                stdout.write(f"Current model: {root.agent.model_name}\nAvailable models:\n")
                for slug, display_name in OFFICIAL_MODELS:
                    stdout.write(f"  {slug}\t{display_name}\n")
            else:
                root.set_model(arg)
                stdout.write(f"Model set to {arg}\n")
            stdout.flush()
            continue
        if line == "/effort" or line.startswith("/effort "):
            arg = line[7:].strip()
            if not arg:
                stdout.write(
                    f"Current effort: {root.explicit_effort or root.config_effort or 'default'}\n"
                )
                stdout.flush()
                continue
            try:
                normalized = normalize_effort_level(arg)
            except ValueError as exc:
                stderr.write(f"Error: {exc}\n")
                stderr.flush()
                continue
            root.set_model(root.agent.model_name, explicit_effort=normalized)
            stdout.write(f"Reasoning effort set to {normalized}\n")
            stdout.flush()
            continue

        summary = await root.run_user_prompt(line, wait_background=True)
        if summary.final_response:
            stdout.write(_format_trailing_newline(summary.final_response))
            stdout.flush()

    return 0


async def _execute_session_async(
    args: "argparse.Namespace",
    is_print_mode: "bool",
    resolved_prompt: "typing.Optional[str]",
    timeout_seconds: "typing.Optional[float]",
    cwd: "Path",
    stdin: "typing.TextIO",
    stdout: "typing.TextIO",
    stderr: "typing.TextIO",
) -> "int":
    app_data_dir = (
        Path(args.app_data_dir).expanduser() if args.app_data_dir is not None else None
    )
    additional_ws = [Path(d).expanduser().resolve() for d in (args.add_dirs or [])]
    req_timeout = (
        max(30.0, timeout_seconds + 5.0) if timeout_seconds is not None else 300.0
    )
    session = build_agy_session(
        config_path=Path(args.config).expanduser(),
        profile=args.profile,
        cwd=cwd,
        model_override=args.model,
        explicit_effort=args.effort,
        app_data_dir=app_data_dir,
        additional_workspaces=additional_ws,
        continue_latest=bool(args.continue_latest),
        resume_conversation_id=args.conversation_id,
        timeout_seconds=req_timeout,
    )
    try:
        if is_print_mode:
            return await _run_print_mode_async(
                session=session,
                input_format=args.input_format,
                output_format=args.output_format,
                prompt_text=resolved_prompt,
                timeout_seconds=timeout_seconds,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
            )
        init_prompt = (
            args.interactive_prompt
            if args.interactive_prompt is not None and args.interactive_prompt != ""
            else resolved_prompt
        )
        return await _run_interactive_repl_async(
            session=session,
            initial_prompt=init_prompt,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )
    finally:
        await session.close()


def main(
    argv: "typing.Optional[typing.Sequence[str]]" = None,
    stdin: "typing.Optional[typing.TextIO]" = None,
    stdout: "typing.Optional[typing.TextIO]" = None,
    stderr: "typing.Optional[typing.TextIO]" = None,
    cwd: "typing.Optional[Path]" = None,
) -> "int":
    raw_args = list(sys.argv[1:] if argv is None else argv)
    in_stream = sys.stdin if stdin is None else stdin
    out_stream = sys.stdout if stdout is None else stdout
    err_stream = sys.stderr if stderr is None else stderr
    effective_cwd = (cwd if cwd is not None else Path.cwd()).resolve()

    if raw_args in (["-v"], ["--version"]):
        out_stream.write(f"{__version__} (agy 1.5.13 wire-compatible)\n")
        out_stream.flush()
        return 0
    if raw_args:
        first = raw_args[0]
        if first == "models":
            return _run_models_subcommand(raw_args[1:], out_stream)
        if first in ("agent", "agents"):
            return _run_agents_subcommand(raw_args[1:], out_stream)
        if first == "help":
            if len(raw_args) > 1 and raw_args[1] == "models":
                return _run_models_subcommand(["--help"], out_stream)
            if len(raw_args) > 1 and raw_args[1] in ("agent", "agents"):
                return _run_agents_subcommand(["--help"], out_stream)
            build_cli_parser().print_help(file=out_stream)
            return 0

    parser = build_cli_parser()
    args = parser.parse_args(raw_args)

    if args.input_format == "stream-json" and args.output_format != "stream-json":
        err_stream.write(
            "Error: --input-format stream-json requires --output-format stream-json\n"
        )
        err_stream.flush()
        return 2

    try:
        timeout_seconds = parse_duration_seconds(args.print_timeout)
        if args.effort is not None:
            normalize_effort_level(args.effort)
    except ValueError as exc:
        err_stream.write(f"Error: {exc}\n")
        err_stream.flush()
        return 2

    is_print_mode = (
        args.print_prompt is not None or args.input_format == "stream-json"
    )

    resolved_prompt: "typing.Optional[str]" = None
    if is_print_mode and args.input_format == "text":
        if args.print_prompt is not None and args.print_prompt != "":
            resolved_prompt = args.print_prompt
        elif args.prompt is not None and args.prompt != "":
            resolved_prompt = args.prompt
        else:
            is_tty = bool(getattr(in_stream, "isatty", lambda: False)())
            if not is_tty:
                piped = in_stream.read()
                if piped.strip():
                    resolved_prompt = piped.rstrip("\r\n")
        if resolved_prompt is None or not resolved_prompt.strip():
            err_stream.write(
                'Error: empty prompt. Usage: pyagy --print "your prompt here"\n'
            )
            err_stream.flush()
            return 2
    elif not is_print_mode:
        resolved_prompt = args.prompt

    try:
        return asyncio.run(
            _execute_session_async(
                args=args,
                is_print_mode=is_print_mode,
                resolved_prompt=resolved_prompt,
                timeout_seconds=timeout_seconds,
                cwd=effective_cwd,
                stdin=in_stream,
                stdout=out_stream,
                stderr=err_stream,
            )
        )
    except Exception as exc:
        err_stream.write(f"Error: {exc}\n")
        err_stream.flush()
        return 1
