"""Stateless presentation helpers; no event or frontend dependencies."""

import json
import re
import shlex
import typing
from dataclasses import asdict

PROMPT_CONTEXT_BASELINE_TOKENS = 12_000


def render_result(kind, result, json_mode, log):
    if kind == "turn":
        text = (
            json.dumps(asdict(result), ensure_ascii=False, indent=2)
            if json_mode
            else (result.output_text or "")
        )
        log(text)
    elif kind == "command":
        for line in format_command_result(result).splitlines():
            log(line)


def format_command_result(result: "typing.Dict[str, object]") -> "str":
    kind = result["kind"]
    if kind == "help":
        return "Extra commands: " + ", ".join("/" + name for name in result["commands"])
    if kind in {"title", "title_changed"}:
        return "Session: " + (result["title"] or "untitled")
    if kind == "cwd":
        return "Current directory: " + result["cwd"]
    if kind == "models":
        return (
            "Current model: "
            + result["model"]
            + "\nAvailable models: "
            + ", ".join(result["models"])
        )
    if kind == "model_changed":
        return "Switched model to {0}.".format(result["model"])
    if kind == "effort":
        return "Reasoning effort: {0}".format(
            result["reasoning_effort"] or "provider default"
        )
    if kind == "sessions":
        if not result["sessions"]:
            return "No resumable sessions found."
        return "\n".join(
            ["Available sessions:"]
            + [
                "[{0}] {1}".format(index, session["preview"])
                for index, session in enumerate(result["sessions"], 1)
            ]
        )
    if kind in {"history", "resumed"}:
        state = result["state"]
        lines = ["Resumed session: " + state["title"]] if kind == "resumed" else []
        if not state["history"]:
            return "\n".join(lines + ["No history yet."])
        lines.append("Session: " + (state["title"] or "untitled"))
        for index, (prompt, response) in enumerate(state["history"], 1):
            if prompt:
                lines.append("[{0}]U> {1}".format(index, prompt))
            if response:
                lines.append("[{0}]A> {1}".format(index, response))
        return "\n".join(lines)
    if kind == "compact_empty":
        return "Nothing to compact."
    if kind == "compacted":
        return compact_summary(
            result["original_item_count"],
            result["retained_item_count"],
            result["pruned_tool_results"],
        )
    return "\n".join(result["lines"])


def format_error(text, indent=True):
    if not indent:
        return "Error: " + str(text)
    lines = str(text).splitlines() or [""]
    return "\n".join(
        ["Error: " + lines[0]] + ["  " + line if line else "" for line in lines[1:]]
    )


def shorten_title(text: "str", limit: "int" = 48) -> "str":
    return truncate_text(text, limit)


def percent_of_context_window_remaining(total_tokens, context_window_tokens):
    if context_window_tokens <= PROMPT_CONTEXT_BASELINE_TOKENS:
        return 0
    effective_window = context_window_tokens - PROMPT_CONTEXT_BASELINE_TOKENS
    used = max(total_tokens - PROMPT_CONTEXT_BASELINE_TOKENS, 0)
    remaining = max(effective_window - used, 0)
    return int(round(max(0.0, min(100.0, (remaining / effective_window) * 100.0))))


def completed_history(state):
    active = state["active_turn"]
    if active is not None:
        return active["completed_history"]
    return state["history"]


def colorize_cli_message(text: "str", kind: "str", enabled: "bool") -> "str":
    if not enabled:
        return text
    if kind == "assistant":
        color = "\x1b[32m"
    elif kind == "status":
        color = "\x1b[36m"
    elif kind == "error":
        color = "\x1b[31m"
    else:
        return text
    return "\x1b[1m" + color + text + "\x1b[0m"


def colorize_tool_message(
    message: "str", enabled: "bool", tool_name: "str" = None
) -> "str":
    if not enabled:
        return message
    if message.startswith("[error]"):
        return colorize_cli_message(message, "error", enabled)
    if message.startswith("[") and message.find("]") > 1:
        tool_name = message[1 : message.find("]")]
    if message.startswith(("  [x]", "  [>]", "  [ ]")) or tool_name == "update_plan":
        color = "\x1b[36m"
    elif tool_name == "exec_command" and "session_id=" in message:
        color = "\x1b[2m"
    elif tool_name in {"exec_command", "exec"} and not message.startswith("["):
        color = ""
    elif tool_name in {"exec_command", "shell", "shell_command", "exec"}:
        color = "\x1b[33m"
    elif tool_name in {"write_stdin", "wait", "web_search"}:
        color = "\x1b[35m"
    elif tool_name in {
        "spawn_agent",
        "wait_agent",
        "send_message",
        "followup_task",
        "interrupt_agent",
        "list_agents",
    }:
        color = "\x1b[34m"
    else:
        color = "\x1b[2m"
    return "\x1b[1m" + color + message + "\x1b[0m"


def compact_summary(original: "int", retained: "int", pruned: "int") -> "str":
    text = "compact({0} {1}) -> {2} {3} + [summary]".format(
        original,
        "item" if original == 1 else "items",
        retained,
        "item" if retained == 1 else "items",
    )
    if pruned:
        text += " (dropped {0} old {1})".format(
            pruned,
            "tool response" if pruned == 1 else "tool responses",
        )
    return text


def agent_status_summary(status: "object") -> "str":
    if isinstance(status, str):
        return status
    if isinstance(status, dict):
        if "completed" in status:
            completed = status.get("completed")
            if completed is None:
                return "completed"
            return f"completed: {truncate_text(str(completed), 48)}"
        if "errored" in status:
            return f"errored: {truncate_text(str(status.get('errored', '')), 48)}"
    return truncate_text(json.dumps(status, ensure_ascii=False, separators=(",", ":")))


def truncate_text(text: "str", limit: "int" = 96) -> "str":
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def summarize_tool_result(call, result) -> "str":
    """Format call/result data without changing it or interpreting process state text."""
    name = call.name
    arguments = call.arguments if isinstance(call.arguments, dict) else {}
    output = result.output
    call_summary = ""
    result_summary = None
    if name == "exec" and isinstance(call.arguments, str):
        call_summary = call.arguments
    elif name == "exec_command":
        command = arguments.get("cmd")
        if command not in (None, ""):
            command = str(command)
            call_summary = truncate_text(command, 200)
            lines = command.splitlines()
            match = re.search(
                r"\bpython(?:\d+(?:\.\d+)?)?\s+-\s+<<[-]?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1",
                lines[0] if lines else "",
            )
            if match is not None:
                for index, line in enumerate(lines[1:], 1):
                    if line.strip() == match.group(2):
                        call_summary = "\n".join(lines[: index + 1])
                        break
    elif name == "shell_command":
        command = arguments.get("command")
        if command not in (None, ""):
            call_summary = truncate_text(str(command), 200)
    elif name == "shell":
        command = arguments.get("command")
        if isinstance(command, list) and command:
            call_summary = truncate_text(
                " ".join(shlex.quote(str(part)) for part in command), 200
            )
    elif name == "write_stdin":
        session_id = arguments.get("session_id")
        if session_id not in (None, ""):
            session_id = str(session_id)
            chars = arguments.get("chars") or ""
            call_summary = (
                f"session {session_id} <- {truncate_text(str(chars), 32)}"
                if chars
                else f"poll session {session_id}"
            )
    elif name in {"read_file", "list_dir", "view_image"}:
        key = (
            "file_path"
            if name == "read_file"
            else ("dir_path" if name == "list_dir" else "path")
        )
        path = arguments.get(key)
        if path not in (None, ""):
            call_summary = truncate_text(str(path), 200)
        if name == "view_image" and isinstance(output, list):
            result_summary = f"{len(output)} image item(s)"
    elif name == "grep_files":
        pattern, path = arguments.get("pattern"), arguments.get("path")
        if pattern not in (None, ""):
            call_summary = truncate_text(
                f"{pattern} @ {path}" if path not in (None, "") else str(pattern),
                200,
            )
    elif name == "update_plan":
        plan = arguments.get("plan")
        if isinstance(plan, list):
            total = len(plan)
            completed = 0
            in_progress = 0
            for item in plan:
                if isinstance(item, dict):
                    status = str(item.get("status", "")).strip()
                    completed += status == "completed"
                    in_progress += status == "in_progress"
            if not total:
                return "0 steps"
            if completed >= total:
                return f"Done {completed}/{total}"
            if in_progress:
                return f"Working on {completed + in_progress}/{total}"
            return f"Planned {completed}/{total}"
        if isinstance(output, dict) and isinstance(output.get("plan"), list):
            return f"{len(output['plan'])} steps"
    elif name == "spawn_agent":
        if isinstance(output, dict):
            if output.get("task_name"):
                return str(output["task_name"])
    elif name in {"send_message", "followup_task"}:
        target = arguments.get("target", "")
        message = arguments.get("message", "")
        if target or message:
            call_summary = "{0} <- {1}".format(target, truncate_text(str(message), 40))
    elif name == "wait_agent":
        if isinstance(output, dict):
            if output.get("timed_out") is True:
                return "timed out"
            if output.get("message"):
                return str(output["message"])
    elif name == "list_agents":
        if isinstance(output, dict) and isinstance(output.get("agents"), list):
            return "{0} agents".format(len(output["agents"]))
    elif name == "interrupt_agent":
        call_summary = str(arguments.get("target", ""))
        if isinstance(output, dict):
            result_summary = agent_status_summary(output.get("previous_status"))
    if result_summary is None:
        if isinstance(output, (dict, list)):
            result_summary = truncate_text(
                json.dumps(output, ensure_ascii=False, separators=(",", ":"))
            )
        else:
            lines = [
                line.strip() for line in result.output_text().splitlines()
            ]
            if "Output:" in lines:
                result_summary = next(
                    (line for line in lines[lines.index("Output:") + 1 :] if line),
                    None,
                )
            if result_summary is None:
                result_summary = next(
                    (
                        line
                        for line in lines
                        if line
                        and not line.startswith(
                            ("Exit code:", "Wall time:", "Command:")
                        )
                    ),
                    "",
                )
            result_summary = truncate_text(result_summary)
    if call_summary and result_summary:
        return f"{call_summary} -> {result_summary}"
    return call_summary or result_summary
