"""Stateless presentation helpers; no event or frontend dependencies."""

import json
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
    if kind == "models":
        return (
            "Current model: "
            + result["model"]
            + "\nAvailable models: "
            + ", ".join(result["models"])
        )
    if kind == "model_changed":
        return "Switched model to {0}.".format(result["model"])
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
    if kind == "forked":
        return "Forked session: " + result["session_id"]
    if kind == "closed":
        return ""
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
    elif tool_name == "exec_command" and not message.startswith("["):
        color = ""
    elif tool_name in {"exec_command", "shell", "shell_command", "exec"}:
        color = "\x1b[33m"
    elif tool_name in {"write_stdin", "wait", "web_search"}:
        color = "\x1b[35m"
    elif tool_name in {
        "spawn_agent",
        "send_input",
        "wait_agent",
        "resume_agent",
        "close_agent",
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


def short_id(value: "str", limit: "int" = 8) -> "str":
    compact = value.strip()
    if len(compact) <= limit + 4:
        return compact
    return f"{compact[:limit]}...{compact[-4:]}"


def truncate_text(text: "str", limit: "int" = 96) -> "str":
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
