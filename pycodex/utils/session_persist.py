import json
import mmap
import os
import re
import typing
from contextlib import closing
from datetime import datetime
from pathlib import Path

from ..protocol import (
    AssistantMessage,
    ConversationItem,
    ReasoningItem,
    ToolCall,
    ToolResult,
    UserMessage,
)
from .event_helpers import shorten_title
from .get_env import get_package_version

SESSION_INDEX_FILENAME = "session_index.jsonl"
ROLLUP_SESSION_DIRNAMES = ("sessions", "archived_sessions")
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
ROLLOUT_READ_CHUNK_SIZE = 1024 * 1024
COMPACTED_RECORD_MARKER = b',"type":"compacted","payload":'
ROLLOUT_RECORD_PREFIX = b'\n{"timestamp":"'


def resolve_codex_home(
    config_path: "typing.Union[str, None]" = None,
) -> "Path":
    if config_path:
        return Path(config_path).expanduser().resolve().parent
    codex_home = os.environ.get("CODEX_HOME", "").strip()
    if codex_home:
        return Path(codex_home).expanduser().resolve()
    return Path.home() / ".codex"


class SessionRolloutRecorder:
    def __init__(self, rollout_path: "Path") -> "None":
        self.rollout_path = rollout_path
        self._session_meta: "typing.Union[typing.Dict[str, object], None]" = None

    @classmethod
    def create(
        cls,
        codex_home: "Path",
        session_id: "str",
        cwd: "Path",
        originator: "str",
        model_provider: "typing.Union[str, None]",
        base_instructions: "str",
        session_file_path: "typing.Union[str, Path, None]" = None,
    ) -> "SessionRolloutRecorder":
        path = (
            Path(session_file_path)
            if session_file_path is not None
            else _rollout_path_for_session(codex_home, session_id)
        )
        recorder = cls(path.expanduser().resolve())
        if recorder.rollout_path.exists():
            raise FileExistsError(
                "session file already exists: {0}".format(recorder.rollout_path)
            )
        recorder._session_meta = {
            "id": session_id,
            "timestamp": _timestamp_string(),
            "cwd": str(cwd),
            "originator": originator,
            "cli_version": get_package_version(),
            "source": "cli",
            "model_provider": model_provider,
            "base_instructions": {"text": base_instructions},
        }
        return recorder

    @classmethod
    def resume(
        cls,
        rollout_path: "typing.Union[str, Path]",
    ) -> "SessionRolloutRecorder":
        return cls(Path(rollout_path).expanduser().resolve())

    def append_history_items(
        self,
        items: "typing.Iterable[ConversationItem]",
        initial_history: "typing.Iterable[ConversationItem]" = (),
    ) -> "None":
        self._append_records(self._history_records(items), initial_history)

    @staticmethod
    def _history_records(
        items: "typing.Iterable[ConversationItem]",
    ) -> "typing.Iterable[typing.Tuple[str, typing.Dict[str, object]]]":
        for item in items:
            serialized = item.serialize()
            if isinstance(serialized, dict):
                yield "response_item", serialized
            if isinstance(item, UserMessage):
                yield "event_msg", {
                    "type": "user_message",
                    "message": item.text,
                    "images": [],
                    "local_images": [],
                    "text_elements": [],
                }

    def append_compacted_history(
        self,
        history: "typing.Iterable[ConversationItem]",
        initial_history: "typing.Iterable[ConversationItem]" = (),
    ) -> "None":
        serialized_items = []
        for item in history:
            serialized = item.serialize()
            if isinstance(serialized, dict):
                serialized_items.append(serialized)
        self._append_records(
            [("compacted", {"replacement_history": serialized_items})],
            initial_history,
        )

    def _append_records(
        self,
        records: "typing.Iterable[typing.Tuple[str, typing.Dict[str, object]]]",
        initial_history: "typing.Iterable[ConversationItem]" = (),
    ) -> "None":
        records = list(records)
        if not records:
            return
        mode = "a"
        if self._session_meta is not None:
            mode = "x"
            records = (
                [("session_meta", self._session_meta)]
                + list(self._history_records(initial_history))
                + records
            )
        self.rollout_path.parent.mkdir(parents=True, exist_ok=True)
        with self.rollout_path.open(mode, encoding="utf-8") as handle:
            for item_type, payload in records:
                line = {
                    "timestamp": _timestamp_string(),
                    "type": item_type,
                    "payload": payload,
                }
                handle.write(
                    json.dumps(line, ensure_ascii=False, separators=(",", ":"))
                )
                handle.write("\n")
                handle.flush()
        self._session_meta = None


def list_resumable_sessions(
    codex_home: "Path",
    limit: "int" = 20,
) -> "typing.Tuple[typing.Dict[str, str], ...]":
    latest_rollouts_by_id: "typing.Dict[str, Path]" = {}
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
    sessions: "typing.List[typing.Dict[str, str]]" = []
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


def select_resumable_session(
    codex_home: "Path",
    resume_index_text: "str",
) -> "typing.Dict[str, str]":
    normalized_target = resume_index_text.strip()
    if not normalized_target.isdigit():
        raise ValueError("Usage: /resume <number>")

    sessions = list_resumable_sessions(codex_home)
    resume_index = int(normalized_target)
    if resume_index < 1 or resume_index > len(sessions):
        raise ValueError(f"Session not found: {normalized_target}")

    return sessions[resume_index - 1]


def load_resumed_session_path(
    rollout_path: "typing.Union[str, Path]",
    thread_name: "typing.Union[str, None]" = None,
) -> "typing.Dict[str, object]":
    rollout_path = Path(rollout_path).expanduser().resolve()
    thread_id = _thread_id_from_rollout_path(rollout_path) or ""
    session_id = thread_id
    history: "typing.List[ConversationItem]" = []
    saw_user_turn = False
    tool_names_by_call_id: "typing.Dict[str, str]" = {}

    # A rollout is append-only, and a compacted entry replaces everything
    # before it for the next request.  Large tool outputs before the latest
    # checkpoint are retained for audit, but do not need to be decoded while
    # restoring the active conversation.
    compacted_offset = _find_last_compacted_offset(rollout_path)
    entry_start_offset = compacted_offset if compacted_offset is not None else 0
    if compacted_offset is not None:
        with closing(_iter_rollout_entries(rollout_path)) as entries:
            first_entry = next(entries)
        metadata = first_entry.get("payload")
        if first_entry.get("type") == "session_meta" and isinstance(metadata, dict):
            session_id = str(metadata.get("id", "")).strip() or session_id

    for entry in _iter_rollout_entries(rollout_path, entry_start_offset):
        item_type = str(entry.get("type", "")).strip()
        payload = entry.get("payload")

        if item_type == "session_meta" and isinstance(payload, dict):
            session_id = str(payload.get("id", "")).strip() or session_id
            continue

        if item_type == "compacted" and isinstance(payload, dict):
            replacement_history = payload.get("replacement_history")
            if isinstance(replacement_history, list):
                history = _deserialize_compacted_history(replacement_history)
                saw_user_turn = any(isinstance(item, UserMessage) for item in history)
                tool_names_by_call_id = {
                    item.call_id: item.name
                    for item in history
                    if isinstance(item, ToolCall)
                }
            continue

        if item_type == "event_msg" and isinstance(payload, dict):
            if payload.get("type") == "user_message":
                history.append(UserMessage(text=str(payload.get("message", ""))))
                saw_user_turn = True
            continue

        if (
            item_type != "response_item"
            or not saw_user_turn
            or not isinstance(payload, dict)
        ):
            continue

        _append_deserialized_response_item(
            history,
            payload,
            tool_names_by_call_id,
            include_user_messages=False,
        )

    if not history:
        raise ValueError(f"No resumable history found in {rollout_path}")

    history = _trim_incomplete_tool_call_tail(history)
    if not history:
        raise ValueError(f"No resumable history found in {rollout_path}")

    turns = conversation_history_to_turns(history)
    title = thread_name or (shorten_title(turns[0][0]) if turns else thread_id)
    return {
        "session_id": session_id,
        "thread_id": thread_id,
        "title": title,
        "history": tuple(history),
        "turns": tuple(turns),
        "rollout_path": rollout_path,
    }


def conversation_history_to_turns(
    history: "typing.Iterable[ConversationItem]",
) -> "typing.Tuple[typing.Tuple[str, str], ...]":
    turns: "typing.List[typing.Tuple[str, str]]" = []
    current_user_text: "typing.Union[str, None]" = None
    current_assistant_text = ""
    for item in history:
        if isinstance(item, UserMessage):
            if current_user_text is not None:
                turns.append((current_user_text, current_assistant_text))
            current_user_text = item.text
            current_assistant_text = ""
            continue
        if isinstance(item, AssistantMessage) and current_user_text is not None:
            current_assistant_text = item.text
    if current_user_text is not None:
        turns.append((current_user_text, current_assistant_text))
    return tuple(turns)


def _trim_incomplete_tool_call_tail(
    history: "typing.List[ConversationItem]",
) -> "typing.List[ConversationItem]":
    pending_call_ids: "typing.Set[str]" = set()
    call_indexes: "typing.Dict[str, int]" = {}

    for index, item in enumerate(history):
        if isinstance(item, ToolCall):
            pending_call_ids.add(item.call_id)
            call_indexes[item.call_id] = index
            continue
        if isinstance(item, ToolResult):
            pending_call_ids.discard(item.call_id)

    if not pending_call_ids:
        return history

    trim_start = min(call_indexes[call_id] for call_id in pending_call_ids)
    while trim_start > 0 and isinstance(
        history[trim_start - 1],
        (AssistantMessage, ReasoningItem, ToolCall),
    ):
        trim_start -= 1
    return history[:trim_start]


def _latest_thread_names_by_id(codex_home: "Path") -> "typing.Dict[str, str]":
    index_path = codex_home / SESSION_INDEX_FILENAME
    if not index_path.exists():
        return {}

    names_by_id: "typing.Dict[str, str]" = {}
    with index_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
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
            if thread_id and thread_name:
                names_by_id[thread_id] = thread_name
    return names_by_id


def _thread_id_from_rollout_path(path: "Path") -> "typing.Union[str, None]":
    stem = path.stem
    if len(stem) < 36:
        return None
    candidate = stem[-36:]
    return candidate if UUID_PATTERN.match(candidate) else None


def _extract_first_user_message_preview(
    rollout_path: "Path",
) -> "typing.Union[str, None]":
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


def _find_last_compacted_offset(
    rollout_path: "Path",
) -> "typing.Union[int, None]":
    """Find the latest recorder-format compact checkpoint."""

    file_size = rollout_path.stat().st_size
    if not file_size:
        return None

    with rollout_path.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            marker_offset = mapped.rfind(COMPACTED_RECORD_MARKER)
            if marker_offset < 0:
                return None
            record_start = mapped.rfind(
                ROLLOUT_RECORD_PREFIX,
                0,
                marker_offset,
            )
            if record_start < 0:
                return None
            previous = record_start - 1
            while previous >= 0 and mapped[previous] in b" \t\r\n":
                previous -= 1
            if previous >= 0 and mapped[previous] != ord("}"):
                return None
            return record_start + 1
        finally:
            mapped.close()
    return None


def _iter_rollout_entries(
    rollout_path: "Path",
    start_offset: "int" = 0,
) -> "typing.Iterable[typing.Dict[str, object]]":
    decoder = json.JSONDecoder()
    buffer = ""
    start = 0
    parsed_entries = 0

    with rollout_path.open("r", encoding="utf-8", errors="replace") as handle:
        if start_offset:
            handle.seek(start_offset)
        while True:
            chunk = handle.read(ROLLOUT_READ_CHUNK_SIZE)
            eof = not chunk
            if chunk:
                buffer += chunk

            while True:
                while start < len(buffer) and buffer[start].isspace():
                    start += 1
                if start >= len(buffer):
                    buffer = ""
                    start = 0
                    break
                try:
                    entry, start = decoder.raw_decode(buffer, start)
                except json.JSONDecodeError as exc:
                    if eof:
                        if parsed_entries > 0:
                            return
                        raise ValueError(
                            f"failed to parse rollout file {rollout_path}: {exc}"
                        ) from exc
                    if start:
                        buffer = buffer[start:]
                        start = 0
                    break
                if isinstance(entry, dict):
                    parsed_entries += 1
                    yield entry
                if start > ROLLOUT_READ_CHUNK_SIZE:
                    buffer = buffer[start:]
                    start = 0

            if eof:
                break

    if parsed_entries == 0:
        raise ValueError(f"no rollout entries found in {rollout_path}")


def _extract_response_message_text(payload: "typing.Dict[str, object]") -> "str":
    text_parts: "typing.List[str]" = []
    for item in payload.get("content") or []:
        if isinstance(item, dict) and item.get("type") in {"input_text", "output_text"}:
            text_parts.append(str(item.get("text", "")))
    return "".join(text_parts)


def _deserialize_compacted_history(
    replacement_history: "typing.Iterable[object]",
) -> "typing.List[ConversationItem]":
    history: "typing.List[ConversationItem]" = []
    tool_names_by_call_id: "typing.Dict[str, str]" = {}
    for payload in replacement_history:
        if not isinstance(payload, dict):
            continue
        _append_deserialized_response_item(
            history,
            payload,
            tool_names_by_call_id,
            include_user_messages=True,
        )
    return history


def _append_deserialized_response_item(
    history: "typing.List[ConversationItem]",
    payload: "typing.Dict[str, object]",
    tool_names_by_call_id: "typing.Dict[str, str]",
    include_user_messages: "bool",
) -> "None":
    response_item_type = str(payload.get("type", "")).strip()
    if response_item_type == "message":
        role = str(payload.get("role", "")).strip()
        if role == "assistant":
            history.append(AssistantMessage.from_response_item(payload))
            return
        if include_user_messages and role == "user":
            history.append(
                UserMessage(
                    text=_extract_response_message_text(payload), id=payload.get("id")
                )
            )
        return

    if response_item_type == "reasoning":
        history.append(ReasoningItem(payload=dict(payload)))
        return

    if response_item_type == "function_call":
        raw_arguments = payload.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError:
                return
        elif isinstance(raw_arguments, dict):
            arguments = dict(raw_arguments)
        else:
            return
        if not isinstance(arguments, dict):
            return
        call_id = str(payload.get("call_id", "")).strip()
        name = str(payload.get("name", "")).strip()
        if not call_id or not name:
            return
        history.append(
            ToolCall(
                call_id=call_id,
                name=name,
                arguments=arguments,
                id=payload.get("id"),
                raw_arguments=raw_arguments if isinstance(raw_arguments, str) else None,
                namespace=payload.get("namespace"),
            )
        )
        tool_names_by_call_id[call_id] = name
        return

    if response_item_type == "custom_tool_call":
        call_id = str(payload.get("call_id", "")).strip()
        name = str(payload.get("name", "")).strip()
        if not call_id or not name:
            return
        history.append(
            ToolCall(
                call_id=call_id,
                name=name,
                arguments=str(payload.get("input", "")),
                tool_type="custom",
                id=payload.get("id"),
                namespace=payload.get("namespace"),
                status=payload.get("status"),
            )
        )
        tool_names_by_call_id[call_id] = name
        return

    if response_item_type not in {"function_call_output", "custom_tool_call_output"}:
        return

    call_id = str(payload.get("call_id", "")).strip()
    if not call_id:
        return
    raw_output = payload.get("output", "")
    content_items = None
    if isinstance(raw_output, list) and all(
        isinstance(item, dict) for item in raw_output
    ):
        content_items = tuple(dict(item) for item in raw_output)
        output = json.dumps(raw_output, ensure_ascii=False)
    elif (
        isinstance(raw_output, (dict, list, str, int, float, bool))
        or raw_output is None
    ):
        output = raw_output
    else:
        output = str(raw_output)
    history.append(
        ToolResult(
            call_id=call_id,
            name=tool_names_by_call_id.get(call_id, ""),
            output=output,
            content_items=content_items,
            id=payload.get("id"),
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


def _rollout_path_for_session(codex_home: "Path", session_id: "str") -> "Path":
    now = datetime.now().astimezone()
    return (
        codex_home
        / "sessions"
        / now.strftime("%Y")
        / now.strftime("%m")
        / now.strftime("%d")
        / f"rollout-{now.strftime('%Y-%m-%dT%H-%M-%S')}-{session_id}.jsonl"
    )


def _timestamp_string() -> "str":
    return datetime.now().astimezone().isoformat(timespec="milliseconds")
