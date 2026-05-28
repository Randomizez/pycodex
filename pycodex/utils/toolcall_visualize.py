import json
import re
import shlex
import typing

from ..protocol import ToolCall, ToolResult

ANSI_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"
ANSI_DIM = "\x1b[2m"
ANSI_RED = "\x1b[31m"
ANSI_GREEN = "\x1b[32m"
ANSI_YELLOW = "\x1b[33m"
ANSI_BLUE = "\x1b[34m"
ANSI_MAGENTA = "\x1b[35m"
ANSI_CYAN = "\x1b[36m"
TOOL_ARGUMENT_PREVIEW_LIMIT = 200
RUNNING_SESSION_MARKER = "Process running with session ID "
PYTHON_HEREDOC_RE = re.compile(
    r"\bpython(?:\d+(?:\.\d+)?)?\s+-\s+<<[-]?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1"
)
TOOL_COLOR_MAP = {
    "update_plan": ANSI_CYAN,
    "exec_command": ANSI_YELLOW,
    "write_stdin": ANSI_MAGENTA,
    "shell": ANSI_YELLOW,
    "shell_command": ANSI_YELLOW,
    "exec": ANSI_YELLOW,
    "wait": ANSI_MAGENTA,
    "web_search": ANSI_MAGENTA,
    "spawn_agent": ANSI_BLUE,
    "send_input": ANSI_BLUE,
    "wait_agent": ANSI_BLUE,
    "resume_agent": ANSI_BLUE,
    "close_agent": ANSI_BLUE,
    "read_file": ANSI_DIM,
    "list_dir": ANSI_DIM,
    "grep_files": ANSI_DIM,
    "view_image": ANSI_DIM,
}


def colorize_cli_message(text: "str", kind: "str", enabled: "bool") -> "str":
    if not enabled:
        return text
    palette = {
        "assistant": ANSI_GREEN,
        "status": ANSI_CYAN,
        "error": ANSI_RED,
    }
    color = palette.get(kind)
    return _colorize_with(text, color, enabled)


def colorize_tool_message(
    message: "str",
    enabled: "bool",
    tool_name: "typing.Union[str, None]" = None,
) -> "str":
    if message.startswith("[error]"):
        return colorize_cli_message(message, "error", enabled)
    if message.startswith(("  [x]", "  [>]", "  [ ]")):
        return _colorize_with(message, TOOL_COLOR_MAP.get("update_plan"), enabled)
    resolved_tool_name = _tool_name_from_message(message) or (tool_name or "")
    if resolved_tool_name == "exec_command" and "session_id=" in message:
        return _colorize_with(message, ANSI_DIM, enabled)
    if resolved_tool_name == "exec_command" and not message.startswith("["):
        return _colorize_with(message, "", enabled)
    return _colorize_with(
        message, TOOL_COLOR_MAP.get(resolved_tool_name, ANSI_DIM), enabled
    )


def _colorize_with(
    text: "str", color: "typing.Union[str, None]", enabled: "bool"
) -> "str":
    if not enabled or color is None:
        return text
    return f"{ANSI_BOLD}{color}{text}{ANSI_RESET}"


def _tool_name_from_message(message: "str") -> "str":
    if not message.startswith("["):
        return ""
    end = message.find("]")
    if end <= 1:
        return ""
    return message[1:end]


def tool_summary(payload: "typing.Dict[str, object]") -> "str":
    tool_name = str(payload.get("tool_name", "")).strip()
    handler = _TOOL_MESSAGE_HANDLERS.get(tool_name, _generic_tool_message)
    return handler(payload)


def _paired_payload(
    payload: "typing.Dict[str, object]",
) -> "typing.Tuple[typing.Union[ToolCall, None], typing.Union[ToolResult, None]]":
    call = payload.get("call")
    result = payload.get("result")
    if isinstance(call, ToolCall) and isinstance(result, ToolResult):
        return call, result
    return None, None


def _legacy_summary(payload: "typing.Dict[str, object]") -> "str":
    return str(payload.get("summary", "") or "").strip()


def _combine_call_and_result(
    call_summary: "typing.Union[str, None]",
    result_summary: "typing.Union[str, None]",
) -> "str":
    if call_summary and result_summary:
        return f"{call_summary} -> {result_summary}"
    if call_summary:
        return call_summary
    return result_summary or ""


def _agent_status_summary(status: "object") -> "str":
    if isinstance(status, str):
        return status
    if isinstance(status, dict):
        if "completed" in status:
            completed = status.get("completed")
            if completed is None:
                return "completed"
            return f"completed: {_truncate_text(str(completed), limit=48)}"
        if "errored" in status:
            return f"errored: {_truncate_text(str(status.get('errored', '')), limit=48)}"
    return _truncate_text(json.dumps(status, ensure_ascii=False, separators=(",", ":")))


def _result_output_summary(result: "ToolResult") -> "typing.Union[str, None]":
    if isinstance(result.output, (dict, list)):
        return _truncate_text(
            json.dumps(result.output, ensure_ascii=False, separators=(",", ":"))
        )
    return _output_text_summary(result.output_text())


def _output_text_summary(text: "str") -> "typing.Union[str, None]":
    lines = [line.strip() for line in text.splitlines()]
    if "Output:" in lines:
        output_index = lines.index("Output:")
        for line in lines[output_index + 1 :]:
            if line:
                return _truncate_text(line)
    for line in lines:
        if not line:
            continue
        if line.startswith(("Exit code:", "Wall time:", "Command:")):
            continue
        return _truncate_text(line)
    return None


def _running_exec_session_id(result: "ToolResult") -> "typing.Union[str, None]":
    for line in result.output_text().splitlines():
        line = line.strip()
        if line.startswith(RUNNING_SESSION_MARKER):
            session_id = line[len(RUNNING_SESSION_MARKER) :].strip()
            if session_id:
                return session_id
    return None


def _exec_command_preview(command: "str") -> "str":
    heredoc = _python_heredoc_preview(command)
    if heredoc:
        return heredoc
    return _truncate_text(command, limit=TOOL_ARGUMENT_PREVIEW_LIMIT)


def _python_heredoc_preview(command: "str") -> "typing.Union[str, None]":
    lines = command.splitlines()
    if not lines:
        return None

    match = PYTHON_HEREDOC_RE.search(lines[0])
    if match is None:
        return None
    delimiter = match.group(2)

    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == delimiter:
            end_index = index
            break
    if end_index is None:
        return None

    return "\n".join(lines[: end_index + 1])


def _plan_progress_summary(plan: "typing.List[object]") -> "str":
    total = len(plan)
    completed = 0
    in_progress = 0
    for item in plan:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).strip()
        if status == "completed":
            completed += 1
        elif status == "in_progress":
            in_progress += 1
    if total == 0:
        return "0 steps"
    if completed >= total:
        return f"Done {completed}/{total}"
    if in_progress:
        return f"Working on {completed + in_progress}/{total}"
    return f"Planned {completed}/{total}"


def _exec_command_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)

    command = None
    if isinstance(call.arguments, dict):
        cmd = call.arguments.get("cmd")
        if cmd not in (None, ""):
            command = _exec_command_preview(str(cmd))
    session_id = _running_exec_session_id(result)
    if session_id:
        return _combine_call_and_result(command, f"session_id={session_id}")
    return _combine_call_and_result(command, _result_output_summary(result))


def _shell_command_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)

    command = None
    if isinstance(call.arguments, dict):
        raw_command = call.arguments.get("command")
        if raw_command not in (None, ""):
            command = _truncate_text(
                str(raw_command), limit=TOOL_ARGUMENT_PREVIEW_LIMIT
            )
    return _combine_call_and_result(command, _result_output_summary(result))


def _shell_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)

    command = None
    if isinstance(call.arguments, dict):
        raw_command = call.arguments.get("command")
        if isinstance(raw_command, list) and raw_command:
            rendered = " ".join(shlex.quote(str(part)) for part in raw_command)
            command = _truncate_text(rendered, limit=TOOL_ARGUMENT_PREVIEW_LIMIT)
    return _combine_call_and_result(command, _result_output_summary(result))


def _write_stdin_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)

    command = None
    if isinstance(call.arguments, dict):
        session_id_value = call.arguments.get("session_id")
        if session_id_value not in (None, ""):
            session_id = int(session_id_value)
            chars = call.arguments.get("chars") or ""
            if not chars:
                command = f"poll session {session_id}"
            else:
                command = (
                    f"session {session_id} <- {_truncate_text(str(chars), limit=32)}"
                )
    return _combine_call_and_result(command, _result_output_summary(result))


def _exec_code_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)
    return _result_output_summary(result) or ""


def _wait_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)
    return _result_output_summary(result) or ""


def _read_file_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)

    path = None
    if isinstance(call.arguments, dict):
        raw_path = call.arguments.get("file_path")
        if raw_path not in (None, ""):
            path = _truncate_text(str(raw_path), limit=TOOL_ARGUMENT_PREVIEW_LIMIT)
    return _combine_call_and_result(path, _result_output_summary(result))


def _list_dir_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)

    path = None
    if isinstance(call.arguments, dict):
        raw_path = call.arguments.get("dir_path")
        if raw_path not in (None, ""):
            path = _truncate_text(str(raw_path), limit=TOOL_ARGUMENT_PREVIEW_LIMIT)
    return _combine_call_and_result(path, _result_output_summary(result))


def _grep_files_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)

    query = None
    if isinstance(call.arguments, dict):
        pattern = call.arguments.get("pattern")
        path = call.arguments.get("path")
        if pattern not in (None, "") and path not in (None, ""):
            query = _truncate_text(
                f"{pattern} @ {path}", limit=TOOL_ARGUMENT_PREVIEW_LIMIT
            )
        elif pattern not in (None, ""):
            query = _truncate_text(str(pattern), limit=TOOL_ARGUMENT_PREVIEW_LIMIT)
    return _combine_call_and_result(query, _result_output_summary(result))


def _view_image_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)

    path = None
    if isinstance(call.arguments, dict):
        raw_path = call.arguments.get("path")
        if raw_path not in (None, ""):
            path = _truncate_text(str(raw_path), limit=TOOL_ARGUMENT_PREVIEW_LIMIT)

    result_summary = None
    if isinstance(result.output, list):
        result_summary = f"{len(result.output)} image item(s)"
    if result_summary is None:
        result_summary = _result_output_summary(result)
    return _combine_call_and_result(path, result_summary)


def _update_plan_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)

    if isinstance(call.arguments, dict):
        plan = call.arguments.get("plan")
        if isinstance(plan, list):
            return _plan_progress_summary(plan)

    if isinstance(result.output, dict):
        plan = result.output.get("plan")
        if isinstance(plan, list):
            return f"{len(plan)} steps"
    return _result_output_summary(result) or ""


def _spawn_agent_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)

    if isinstance(result.output, dict):
        agent_id = str(result.output.get("agent_id", "")).strip()
        nickname = str(result.output.get("nickname", "")).strip()
        if nickname and agent_id:
            return f"{nickname} ({_short_id(agent_id)})"
    return _result_output_summary(result) or ""


def _send_input_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)

    command = None
    if isinstance(call.arguments, dict):
        agent_id = call.arguments.get("id")
        message = call.arguments.get("message")
        prefix = f"{_short_id(str(agent_id))} <- " if agent_id else ""
        if message not in (None, ""):
            command = f"{prefix}{_truncate_text(str(message), limit=40)}"
        elif prefix:
            command = prefix.rstrip()

    result_summary = None
    if isinstance(result.output, dict):
        submission_id = str(result.output.get("submission_id", "")).strip()
        if submission_id:
            result_summary = f"queued {_short_id(submission_id)}"
    if result_summary is None:
        result_summary = _result_output_summary(result)
    return _combine_call_and_result(command, result_summary)


def _wait_agent_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)

    if isinstance(result.output, dict):
        if result.output.get("timed_out") is True:
            return "timed out"
        status = result.output.get("status")
        if isinstance(status, dict):
            parts = []
            for agent_id, agent_status in status.items():
                if not isinstance(agent_id, str):
                    continue
                parts.append(
                    f"{_short_id(agent_id)}={_agent_status_summary(agent_status)}"
                )
            if parts:
                return _truncate_text(", ".join(parts), limit=96)
    return _result_output_summary(result) or ""


def _resume_agent_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)

    agent_id_summary = None
    if isinstance(call.arguments, dict):
        agent_id = call.arguments.get("id")
        if agent_id not in (None, ""):
            agent_id_summary = _short_id(str(agent_id))

    result_summary = None
    if isinstance(result.output, dict):
        result_summary = _agent_status_summary(result.output.get("status"))
    if result_summary is None:
        result_summary = _result_output_summary(result)
    return _combine_call_and_result(agent_id_summary, result_summary)


def _close_agent_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)

    agent_id_summary = None
    if isinstance(call.arguments, dict):
        agent_id = call.arguments.get("id")
        if agent_id not in (None, ""):
            agent_id_summary = _short_id(str(agent_id))

    result_summary = None
    if isinstance(result.output, dict):
        result_summary = _agent_status_summary(result.output.get("status"))
    if result_summary is None:
        result_summary = _result_output_summary(result)
    return _combine_call_and_result(agent_id_summary, result_summary)


def _generic_payload_summary(payload: "typing.Dict[str, object]") -> "str":
    call, result = _paired_payload(payload)
    if call is None or result is None:
        return _legacy_summary(payload)
    return _result_output_summary(result) or ""


def _update_plan_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _update_plan_summary(payload)
    if bool(payload.get("is_error")):
        return (
            f"[error] update_plan failed: {summary}"
            if summary
            else "[error] update_plan failed"
        )

    lines = [f"[update_plan] {summary}" if summary else "[update_plan] Plan updated"]
    plan_items = []
    call = payload.get("call")
    if isinstance(call, ToolCall) and isinstance(call.arguments, dict):
        raw_plan = call.arguments.get("plan")
        if isinstance(raw_plan, list):
            plan_items = raw_plan
    else:
        raw_plan_items = payload.get("plan_items")
        if isinstance(raw_plan_items, list):
            plan_items = raw_plan_items
    for item in plan_items:
        if not isinstance(item, dict):
            continue
        step = str(item.get("step", "")).strip()
        status = str(item.get("status", "")).strip()
        if not step:
            continue
        marker = {
            "completed": "[x]",
            "in_progress": "[>]",
            "pending": "[ ]",
        }.get(status, "[ ]")
        lines.append(f"  {marker} {step}")
    return "\n".join(lines)


def _exec_command_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _exec_command_summary(payload)
    if bool(payload.get("is_error")):
        return (
            f"[error] exec_command failed: {summary}"
            if summary
            else "[error] exec_command failed"
        )
    return f"[exec_command] {summary}" if summary else "[exec_command]"


def _write_stdin_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _write_stdin_summary(payload)
    if bool(payload.get("is_error")):
        return (
            f"[error] write_stdin failed: {summary}"
            if summary
            else "[error] write_stdin failed"
        )
    return f"[write_stdin] {summary}" if summary else "[write_stdin]"


def _shell_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _shell_summary(payload)
    if bool(payload.get("is_error")):
        return f"[error] shell failed: {summary}" if summary else "[error] shell failed"
    return f"[shell] {summary}" if summary else "[shell]"


def _shell_command_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _shell_command_summary(payload)
    if bool(payload.get("is_error")):
        return (
            f"[error] shell_command failed: {summary}"
            if summary
            else "[error] shell_command failed"
        )
    return f"[shell_command] {summary}" if summary else "[shell_command]"


def _exec_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _exec_code_summary(payload)
    if bool(payload.get("is_error")):
        return f"[error] exec failed: {summary}" if summary else "[error] exec failed"
    return f"[exec] {summary}" if summary else "[exec]"


def _wait_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _wait_summary(payload)
    if bool(payload.get("is_error")):
        return f"[error] wait failed: {summary}" if summary else "[error] wait failed"
    return f"[wait] {summary}" if summary else "[wait]"


def _read_file_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _read_file_summary(payload)
    if bool(payload.get("is_error")):
        return (
            f"[error] read_file failed: {summary}"
            if summary
            else "[error] read_file failed"
        )
    return f"[read_file] {summary}" if summary else "[read_file]"


def _list_dir_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _list_dir_summary(payload)
    if bool(payload.get("is_error")):
        return (
            f"[error] list_dir failed: {summary}"
            if summary
            else "[error] list_dir failed"
        )
    return f"[list_dir] {summary}" if summary else "[list_dir]"


def _grep_files_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _grep_files_summary(payload)
    if bool(payload.get("is_error")):
        return (
            f"[error] grep_files failed: {summary}"
            if summary
            else "[error] grep_files failed"
        )
    return f"[grep_files] {summary}" if summary else "[grep_files]"


def _view_image_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _view_image_summary(payload)
    if bool(payload.get("is_error")):
        return (
            f"[error] view_image failed: {summary}"
            if summary
            else "[error] view_image failed"
        )
    return f"[view_image] {summary}" if summary else "[view_image]"


def _spawn_agent_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _spawn_agent_summary(payload)
    if bool(payload.get("is_error")):
        return (
            f"[error] spawn_agent failed: {summary}"
            if summary
            else "[error] spawn_agent failed"
        )
    return f"[spawn_agent] spawned {summary}" if summary else "[spawn_agent] spawned"


def _send_input_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _send_input_summary(payload)
    if bool(payload.get("is_error")):
        return (
            f"[error] send_input failed: {summary}"
            if summary
            else "[error] send_input failed"
        )
    return f"[send_input] {summary}" if summary else "[send_input]"


def _wait_agent_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _wait_agent_summary(payload)
    if bool(payload.get("is_error")):
        return (
            f"[error] wait_agent failed: {summary}"
            if summary
            else "[error] wait_agent failed"
        )
    return f"[wait_agent] {summary}" if summary else "[wait_agent]"


def _resume_agent_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _resume_agent_summary(payload)
    if bool(payload.get("is_error")):
        return (
            f"[error] resume_agent failed: {summary}"
            if summary
            else "[error] resume_agent failed"
        )
    return f"[resume_agent] {summary}" if summary else "[resume_agent]"


def _close_agent_message(payload: "typing.Dict[str, object]") -> "str":
    summary = _close_agent_summary(payload)
    if bool(payload.get("is_error")):
        return (
            f"[error] close_agent failed: {summary}"
            if summary
            else "[error] close_agent failed"
        )
    return f"[close_agent] {summary}" if summary else "[close_agent]"


def _generic_tool_message(payload: "typing.Dict[str, object]") -> "str":
    tool_name = str(payload.get("tool_name", "")).strip()
    summary = _generic_payload_summary(payload)
    if bool(payload.get("is_error")):
        if summary:
            return f"[error] {tool_name} failed: {summary}"
        return f"[error] {tool_name} failed"
    return f"[{tool_name}] {summary}" if summary else f"[{tool_name}]"


_TOOL_MESSAGE_HANDLERS = {
    "update_plan": _update_plan_message,
    "exec_command": _exec_command_message,
    "write_stdin": _write_stdin_message,
    "shell": _shell_message,
    "shell_command": _shell_command_message,
    "exec": _exec_message,
    "wait": _wait_message,
    "read_file": _read_file_message,
    "list_dir": _list_dir_message,
    "grep_files": _grep_files_message,
    "view_image": _view_image_message,
    "spawn_agent": _spawn_agent_message,
    "send_input": _send_input_message,
    "wait_agent": _wait_agent_message,
    "resume_agent": _resume_agent_message,
    "close_agent": _close_agent_message,
}


def _short_id(value: "str", limit: "int" = 8) -> "str":
    compact = value.strip()
    if len(compact) <= limit + 4:
        return compact
    return f"{compact[:limit]}...{compact[-4:]}"


def _truncate_text(text: "str", limit: "int" = 96) -> "str":
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
