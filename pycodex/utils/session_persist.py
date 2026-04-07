import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from ..protocol import (
    AssistantMessage,
    ConversationItem,
    ReasoningItem,
    ToolCall,
    ToolResult,
    UserMessage,
)
from .get_env import get_package_version
from .visualize import shorten_title
import typing

SESSION_INDEX_FILENAME = "session_index.jsonl"
ROLLUP_SESSION_DIRNAMES = ("sessions", "archived_sessions")
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def resolve_codex_home(
    config_path: 'typing.Union[str, None]' = None,
) -> 'Path':
    if config_path:
        return Path(config_path).expanduser().resolve().parent
    codex_home = os.environ.get("CODEX_HOME", "").strip()
    if codex_home:
        return Path(codex_home).expanduser().resolve()
    return Path.home() / ".codex"


class SessionRolloutRecorder:
    def __init__(self, rollout_path: 'Path') -> 'None':
        self.rollout_path = rollout_path

    @classmethod
    def create(
        cls,
        codex_home: 'Path',
        session_id: 'str',
        cwd: 'Path',
        originator: 'str',
        model_provider: 'typing.Union[str, None]',
        base_instructions: 'str',
    ) -> 'SessionRolloutRecorder':
        recorder = cls(_rollout_path_for_session(codex_home, session_id))
        recorder.write_session_meta(
            session_id=session_id,
            cwd=cwd,
            originator=originator,
            model_provider=model_provider,
            base_instructions=base_instructions,
        )
        return recorder

    @classmethod
    def resume(
        cls,
        rollout_path: 'typing.Union[str, Path]',
    ) -> 'SessionRolloutRecorder':
        return cls(Path(rollout_path))

    def write_session_meta(
        self,
        session_id: 'str',
        cwd: 'Path',
        originator: 'str',
        model_provider: 'typing.Union[str, None]',
        base_instructions: 'str',
    ) -> 'None':
        payload = {
            "id": session_id,
            "timestamp": _timestamp_string(),
            "cwd": str(cwd),
            "originator": originator,
            "cli_version": get_package_version(),
            "source": "cli",
            "model_provider": model_provider,
            "base_instructions": {"text": base_instructions},
        }
        self._append_line("session_meta", payload)

    def append_history_items(
        self,
        items: 'typing.Iterable[ConversationItem]',
    ) -> 'None':
        for item in items:
            self.append_history_item(item)

    def append_history_item(self, item: 'ConversationItem') -> 'None':
        if isinstance(item, UserMessage):
            self._append_line("response_item", item.serialize())
            self._append_line(
                "event_msg",
                {
                    "type": "user_message",
                    "message": item.text,
                    "images": [],
                    "local_images": [],
                    "text_elements": [],
                },
            )
            return
        if isinstance(item, ToolResult):
            self._append_line("response_item", item.serialize())
            return
        serialized = item.serialize()
        if isinstance(serialized, dict):
            self._append_line("response_item", serialized)

    def _append_line(self, item_type: 'str', payload: 'typing.Dict[str, object]') -> 'None':
        self.rollout_path.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "timestamp": _timestamp_string(),
            "type": item_type,
            "payload": payload,
        }
        with self.rollout_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            handle.flush()


def list_resumable_sessions(
    codex_home: 'Path',
    limit: 'int' = 20,
) -> 'typing.Tuple[typing.Dict[str, str], ...]':
    latest_rollouts_by_id: 'typing.Dict[str, Path]' = {}
    for dirname in ROLLUP_SESSION_DIRNAMES:
        root = codex_home / dirname
        if not root.exists():
            continue
        for path in root.rglob("rollout-*.jsonl"):
            thread_id = _thread_id_from_rollout_path(path)
            if thread_id is None:
                continue
            previous = latest_rollouts_by_id.get(thread_id)
            if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
                latest_rollouts_by_id[thread_id] = path

    latest_names_by_id = _latest_thread_names_by_id(codex_home)
    ordered_paths = sorted(
        latest_rollouts_by_id.items(),
        key=lambda item: (item[1].stat().st_mtime, str(item[1])),
        reverse=True,
    )
    sessions: 'typing.List[typing.Dict[str, str]]' = []
    for thread_id, path in ordered_paths[:limit]:
        thread_name = latest_names_by_id.get(thread_id, "")
        preview = _extract_first_user_message_preview(path)
        if preview is None:
            continue
        sessions.append(
            {
                "thread_id": thread_id,
                "title": thread_name or preview,
                "preview": preview,
                "rollout_path": str(path),
            }
        )
    return tuple(sessions)


def load_resumed_session(
    codex_home: 'Path',
    resume_index_text: 'str',
) -> 'typing.Dict[str, object]':
    normalized_target = resume_index_text.strip()
    if not normalized_target.isdigit():
        raise ValueError("Usage: /resume <number>")

    sessions = list_resumable_sessions(codex_home)
    resume_index = int(normalized_target)
    if resume_index < 1 or resume_index > len(sessions):
        raise ValueError(f"Session not found: {normalized_target}")

    session = sessions[resume_index - 1]
    thread_id = session["thread_id"]
    rollout_path = Path(session["rollout_path"])
    thread_name = _latest_thread_names_by_id(codex_home).get(thread_id)
    session_id = thread_id
    history: 'typing.List[ConversationItem]' = []
    turns: 'typing.List[typing.Tuple[str, str]]' = []
    current_user_text: 'typing.Union[str, None]' = None
    current_assistant_text = ""
    saw_user_turn = False
    tool_names_by_call_id: 'typing.Dict[str, str]' = {}

    for entry in _iter_rollout_entries(rollout_path):
        item_type = str(entry.get("type", "")).strip()
        payload = entry.get("payload")

        if item_type == "session_meta" and isinstance(payload, dict):
            session_id = str(payload.get("id", "")).strip() or session_id
            continue

        if item_type == "event_msg" and isinstance(payload, dict):
            if payload.get("type") == "user_message":
                if current_user_text is not None:
                    turns.append((current_user_text, current_assistant_text))
                current_user_text = str(payload.get("message", ""))
                current_assistant_text = ""
                history.append(UserMessage(text=current_user_text))
                saw_user_turn = True
            continue

        if item_type != "response_item" or not saw_user_turn or not isinstance(payload, dict):
            continue

        response_item_type = str(payload.get("type", "")).strip()
        if response_item_type == "message":
            if str(payload.get("role", "")).strip() != "assistant":
                continue
            text = _extract_response_message_text(payload)
            history.append(AssistantMessage(text=text))
            current_assistant_text = text
            continue

        if response_item_type == "reasoning":
            history.append(ReasoningItem(payload=dict(payload)))
            continue

        if response_item_type == "function_call":
            raw_arguments = payload.get("arguments", "{}")
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments or "{}")
                except json.JSONDecodeError:
                    continue
            elif isinstance(raw_arguments, dict):
                arguments = dict(raw_arguments)
            else:
                continue
            if not isinstance(arguments, dict):
                continue
            call_id = str(payload.get("call_id", "")).strip()
            name = str(payload.get("name", "")).strip()
            if not call_id or not name:
                continue
            history.append(ToolCall(call_id=call_id, name=name, arguments=arguments))
            tool_names_by_call_id[call_id] = name
            continue

        if response_item_type == "custom_tool_call":
            call_id = str(payload.get("call_id", "")).strip()
            name = str(payload.get("name", "")).strip()
            if not call_id or not name:
                continue
            history.append(
                ToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=str(payload.get("input", "")),
                    tool_type="custom",
                )
            )
            tool_names_by_call_id[call_id] = name
            continue

        if response_item_type in {"function_call_output", "custom_tool_call_output"}:
            call_id = str(payload.get("call_id", "")).strip()
            if not call_id:
                continue
            raw_output = payload.get("output", "")
            content_items = None
            if isinstance(raw_output, list) and all(
                isinstance(item, dict) for item in raw_output
            ):
                content_items = tuple(dict(item) for item in raw_output)
                output = json.dumps(raw_output, ensure_ascii=False)
            elif isinstance(raw_output, (dict, list, str, int, float, bool)) or raw_output is None:
                output = raw_output
            else:
                output = str(raw_output)
            history.append(
                ToolResult(
                    call_id=call_id,
                    name=tool_names_by_call_id.get(call_id, ""),
                    output=output,
                    content_items=content_items,
                    success=(
                        payload.get("success")
                        if isinstance(payload.get("success"), bool)
                        else None
                    ),
                    tool_type=(
                        "custom"
                        if response_item_type == "custom_tool_call_output"
                        else "function"
                    ),
                )
            )

    if current_user_text is not None:
        turns.append((current_user_text, current_assistant_text))

    if not history:
        raise ValueError(f"No resumable history found in {rollout_path}")

    title = thread_name or (shorten_title(turns[0][0]) if turns else thread_id)
    return {
        "session_id": session_id,
        "thread_id": thread_id,
        "title": title,
        "history": tuple(history),
        "turns": tuple(turns),
        "rollout_path": rollout_path,
    }


def _latest_thread_names_by_id(codex_home: 'Path') -> 'typing.Dict[str, str]':
    index_path = codex_home / SESSION_INDEX_FILENAME
    if not index_path.exists():
        return {}

    names_by_id: 'typing.Dict[str, str]' = {}
    for raw_line in reversed(index_path.read_text().splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        thread_id = str(entry.get("id", "")).strip()
        thread_name = str(entry.get("thread_name", "")).strip()
        if thread_id and thread_name and thread_id not in names_by_id:
            names_by_id[thread_id] = thread_name
    return names_by_id


def _thread_id_from_rollout_path(path: 'Path') -> 'typing.Union[str, None]':
    stem = path.stem
    if len(stem) < 36:
        return None
    candidate = stem[-36:]
    return candidate if UUID_PATTERN.match(candidate) else None


def _extract_first_user_message_preview(rollout_path: 'Path') -> 'typing.Union[str, None]':
    for entry in _iter_rollout_entries(rollout_path):
        if entry.get("type") != "event_msg":
            continue
        payload = entry.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "user_message":
            continue
        message = str(payload.get("message", "")).strip()
        if message:
            return shorten_title(message, limit=72)
    return None


def _iter_rollout_entries(rollout_path: 'Path') -> 'typing.Iterable[typing.Dict[str, object]]':
    text = rollout_path.read_text()
    decoder = json.JSONDecoder()
    index = 0
    parsed_entries = 0
    text_length = len(text)

    while index < text_length:
        while index < text_length and text[index].isspace():
            index += 1
        if index >= text_length:
            break
        try:
            entry, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as exc:
            if parsed_entries > 0:
                break
            raise ValueError(f"failed to parse rollout file {rollout_path}: {exc}") from exc
        if isinstance(entry, dict):
            parsed_entries += 1
            yield entry

    if parsed_entries == 0:
        raise ValueError(f"no rollout entries found in {rollout_path}")


def _extract_response_message_text(payload: 'typing.Dict[str, object]') -> 'str':
    text_parts: 'typing.List[str]' = []
    for item in payload.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "output_text":
            text_parts.append(str(item.get("text", "")))
    return "".join(text_parts)


def _rollout_path_for_session(codex_home: 'Path', session_id: 'str') -> 'Path':
    now = datetime.now(timezone.utc)
    return (
        codex_home
        / "sessions"
        / now.strftime("%Y")
        / now.strftime("%m")
        / now.strftime("%d")
        / f"rollout-{now.strftime('%Y-%m-%dT%H-%M-%S')}-{session_id}.jsonl"
    )


def _timestamp_string() -> 'str':
    now = datetime.now(timezone.utc)
    milliseconds = int(now.microsecond / 1000)
    return now.strftime("%Y-%m-%dT%H:%M:%S") + f".{milliseconds:03d}Z"
