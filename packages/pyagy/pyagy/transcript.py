"""Conversation transcript & resume persistence under <appDataDir>/brain/<id>."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import typing
import uuid

from pycodex.protocol import (
    AssistantMessage,
    ContextMessage,
    ConversationItem,
    ReasoningItem,
    ToolCall,
    ToolResult,
    UserMessage,
)

from .context import (
    AGY_RAW_USER_NAME,
    AGY_SYSTEM_MESSAGE_NAME,
    SERVER_RESTART_NOTICE_CONTENT,
    SpawnedSubagentSummary,
    format_utc_iso,
    make_system_user_message,
    normalize_agy_usage,
)


COMPACT_TRUNCATION_LIMIT = 4096


@dataclass(frozen=True)
class RestoredConversation:
    conversation_id: "str"
    cwd: "typing.Optional[Path]"
    model: "typing.Optional[str]"
    last_reported_model_display: "typing.Optional[str]"
    step_counter: "int"
    history: "typing.Tuple[ConversationItem, ...]"
    anchored_user_messages: "typing.Dict[str, str]"
    restart_notice_message: "UserMessage"
    user_requests: "typing.Tuple[str, ...]" = ()
    parent_messages: "typing.Tuple[str, ...]" = ()
    spawned_subagents: "typing.Tuple[SpawnedSubagentSummary, ...]" = ()
    anchor_tokens: "int" = 0


def _serialize_conversation_item(item: "ConversationItem") -> "typing.Dict[str, object]":
    if isinstance(item, ContextMessage):
        return {
            "kind": "user",
            "text": item.text or "",
            "name": AGY_RAW_USER_NAME,
            "visible": False,
            "message_id": str(uuid.uuid4()),
            "created_at": None,
        }
    if isinstance(item, UserMessage):
        return {
            "kind": "user",
            "text": item.text,
            "name": item.name,
            "visible": item.visible,
            "message_id": item.message_id,
            "created_at": item.created_at,
        }
    if isinstance(item, AssistantMessage):
        return {
            "kind": "assistant",
            "text": item.text,
            "id": item.id,
            "phase": item.phase,
            "content_items": list(item.content_items) if item.content_items is not None else None,
            "visible": item.visible,
            "message_id": item.message_id,
            "created_at": item.created_at,
        }
    if isinstance(item, ReasoningItem):
        return {
            "kind": "reasoning",
            "payload": deepcopy(item.payload),
        }
    if isinstance(item, ToolCall):
        return {
            "kind": "tool_call",
            "call_id": item.call_id,
            "name": item.name,
            "arguments": deepcopy(item.arguments),
            "tool_type": item.tool_type,
            "id": item.id,
            "raw_arguments": item.raw_arguments,
            "namespace": item.namespace,
        }
    if isinstance(item, ToolResult):
        return {
            "kind": "tool_result",
            "call_id": item.call_id,
            "name": item.name,
            "output": deepcopy(item.output),
            "is_error": item.is_error,
            "tool_type": item.tool_type,
            "id": item.id,
            "success": item.success,
            "content_items": list(item.content_items) if item.content_items is not None else None,
        }
    raise TypeError(f"unsupported conversation item for session persistence: {type(item).__name__}")


def _parse_visible(
    value: "object",
    default: "typing.Union[bool, str]" = True,
) -> "typing.Union[bool, str]":
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip():
        return value
    return default


def _infer_system_message_visible(content: "str") -> "typing.Union[bool, str]":
    if "content=[Notice] " in content:
        return False
    if ' finished with result:\n' in content:
        return "Task done"
    if ' was canceled with result:\n' in content:
        return "Task canceled"
    if "content=Subagent " in content and " gone idle" in content:
        return "Subagent"
    if "/task-" in content:
        return "Wake-up"
    return "System"


def _deserialize_conversation_item(data: "typing.Dict[str, object]") -> "ConversationItem":
    kind = data.get("kind")
    if kind == "user":
        return UserMessage(
            text=str(data.get("text", "")),
            name=str(data.get("name", "user")),
            visible=_parse_visible(data.get("visible", True), True),
            message_id=str(data.get("message_id") or uuid.uuid4()),
            created_at=str(data["created_at"]) if data.get("created_at") is not None else None,
        )
    if kind == "assistant":
        raw_content = data.get("content_items")
        return AssistantMessage(
            text=str(data.get("text", "")),
            id=str(data["id"]) if data.get("id") is not None else None,
            phase=str(data["phase"]) if data.get("phase") is not None else None,
            content_items=tuple(raw_content) if isinstance(raw_content, list) else None,
            visible=bool(data.get("visible", True)),
            message_id=str(data.get("message_id") or uuid.uuid4()),
            created_at=str(data["created_at"]) if data.get("created_at") is not None else None,
        )
    if kind == "reasoning":
        payload = data.get("payload")
        if not isinstance(payload, dict):
            payload = {"type": "reasoning", "summary": []}
        return ReasoningItem(payload=deepcopy(payload))
    if kind == "tool_call":
        return ToolCall(
            call_id=str(data["call_id"]),
            name=str(data["name"]),
            arguments=deepcopy(data.get("arguments", {})),
            tool_type=str(data.get("tool_type", "function")),
            id=str(data["id"]) if data.get("id") is not None else None,
            raw_arguments=str(data["raw_arguments"]) if data.get("raw_arguments") is not None else None,
            namespace=str(data["namespace"]) if data.get("namespace") is not None else None,
        )
    if kind == "tool_result":
        raw_items = data.get("content_items")
        return ToolResult(
            call_id=str(data["call_id"]),
            name=str(data["name"]) if data.get("name") is not None else None,
            output=deepcopy(data.get("output", "")),
            is_error=bool(data.get("is_error", False)),
            tool_type=str(data.get("tool_type", "function")),
            id=str(data["id"]) if data.get("id") is not None else None,
            success=data.get("success") if isinstance(data.get("success"), bool) else None,
            content_items=tuple(raw_items) if isinstance(raw_items, list) else None,
        )
    raise ValueError(f"unknown persisted conversation item kind: {kind!r}")


def _compact_transcript_record(full_record: "typing.Dict[str, object]") -> "typing.Dict[str, object]":
    compact = deepcopy(full_record)
    truncated: "typing.List[str]" = []
    for field_name in ("content", "thinking"):
        val = compact.get(field_name)
        if isinstance(val, str) and len(val) > COMPACT_TRUNCATION_LIMIT:
            compact[field_name] = val[:COMPACT_TRUNCATION_LIMIT] + "...[truncated]"
            truncated.append(field_name)
    tool_calls = compact.get("tool_calls")
    if isinstance(tool_calls, list):
        encoded = json.dumps(tool_calls, ensure_ascii=False)
        if len(encoded) > COMPACT_TRUNCATION_LIMIT:
            compact["tool_calls"] = [{"name": tc.get("name"), "truncated": True} for tc in tool_calls if isinstance(tc, dict)]
            truncated.append("tool_calls")
    if truncated:
        compact["truncated_fields"] = truncated
    return compact


class AgyTranscriptStore:
    """Manages <appDataDir>/brain/<conversation_id>/.system_generated/ files."""

    def __init__(
        self,
        app_data_dir: "Path",
        conversation_id: "str",
        cwd: "Path",
        clock: "typing.Optional[typing.Callable[[], datetime]]" = None,
    ) -> "None":
        self.app_data_dir = app_data_dir.expanduser()
        self.conversation_id = conversation_id
        self.cwd = cwd.resolve()
        self._clock = clock
        self.brain_dir = self.app_data_dir / "brain" / self.conversation_id
        self.system_dir = self.brain_dir / ".system_generated"
        self.logs_dir = self.system_dir / "logs"
        self.tasks_dir = self.system_dir / "tasks"
        self.messages_dir = self.system_dir / "messages"
        self.transcript_path = self.logs_dir / "transcript.jsonl"
        self.transcript_full_path = self.logs_dir / "transcript_full.jsonl"
        self.session_state_path = self.system_dir / "pyagy_session.json"
        self._step_index = 0
        if self.transcript_full_path.is_file():
            existing_lines = [
                ln for ln in self.transcript_full_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.strip()
            ]
            self._step_index = len(existing_lines)

    def _now(self) -> "typing.Optional[datetime]":
        return self._clock() if self._clock is not None else None

    @property
    def step_index(self) -> "int":
        return self._step_index

    def set_step_index(self, value: "int") -> "None":
        self._step_index = max(self._step_index, int(value))

    def allocate_step_index(self) -> "int":
        idx = self._step_index
        self._step_index += 1
        return idx

    def ensure_dirs(self) -> "None":
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "chunks" / "transcript").mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "chunks" / "transcript_full").mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.messages_dir.mkdir(parents=True, exist_ok=True)

    def append_transcript_step(
        self,
        source: "str",
        step_type: "str",
        content: "str" = "",
        status: "str" = "DONE",
        model: "typing.Optional[str]" = None,
        thinking: "typing.Optional[str]" = None,
        tool_calls: "typing.Optional[typing.List[typing.Dict[str, object]]]" = None,
        usage: "typing.Optional[typing.Mapping[str, object]]" = None,
        step_index: "typing.Optional[int]" = None,
    ) -> "int":
        self.ensure_dirs()
        if step_index is None:
            step_idx = self.allocate_step_index()
        else:
            step_idx = int(step_index)
            if step_idx >= self._step_index:
                self._step_index = step_idx + 1
        record: "typing.Dict[str, object]" = {
            "step_index": step_idx,
            "source": source,
            "type": step_type,
            "status": status,
            "created_at": format_utc_iso(self._now()),
        }
        if model is not None:
            record["model"] = model
        if usage is not None:
            norm_usage = normalize_agy_usage(usage)
            record["input_tokens"] = norm_usage["input_tokens"]
            record["cache_read_tokens"] = norm_usage["cache_read_tokens"]
            record["output_tokens"] = norm_usage["output_tokens"]
        if thinking:
            record["thinking"] = thinking
        if content:
            record["content"] = content
        if tool_calls:
            record["tool_calls"] = tool_calls

        full_line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        compact_line = (
            json.dumps(_compact_transcript_record(record), ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        with self.transcript_full_path.open("a", encoding="utf-8") as fh:
            fh.write(full_line)
        with self.transcript_path.open("a", encoding="utf-8") as fh:
            fh.write(compact_line)
        with (self.logs_dir / "chunks" / "transcript_full" / "00000000.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(full_line)
        with (self.logs_dir / "chunks" / "transcript" / "00000000.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(compact_line)
        return step_idx

    def record_system_message_file(
        self,
        content: "str",
        sender: "str" = "system",
        priority: "str" = "MESSAGE_PRIORITY_LOW",
        timestamp_utc: "typing.Optional[str]" = None,
    ) -> "str":
        self.ensure_dirs()
        msg_id = str(uuid.uuid4())
        ts = timestamp_utc or format_utc_iso(self._now())
        msg_payload = {
            "id": msg_id,
            "recipient": self.conversation_id,
            "sender": sender,
            "priority": priority,
            "timestamp": ts,
            "hideFromUser": True,
            "content": content,
        }
        (self.messages_dir / f"{msg_id}.json").write_text(
            json.dumps(msg_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        read_path = self.messages_dir / "read.json"
        read_map: "typing.Dict[str, bool]" = {}
        if read_path.is_file():
            loaded = json.loads(read_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                read_map.update({str(k): bool(v) for k, v in loaded.items()})
        read_map[msg_id] = True
        read_path.write_text(json.dumps(read_map, ensure_ascii=False), encoding="utf-8")
        return msg_id

    def save_session_state(
        self,
        history: "typing.Sequence[ConversationItem]",
        model: "str",
        last_reported_model_display: "typing.Optional[str]",
        step_counter: "int",
        anchored_user_messages: "typing.Dict[str, str]",
        emitted_post_user_messages: "typing.Optional[typing.Dict[str, typing.Tuple[UserMessage, ...]]]" = None,
        user_requests: "typing.Sequence[str]" = (),
        parent_messages: "typing.Sequence[str]" = (),
        spawned_subagents: "typing.Sequence[SpawnedSubagentSummary]" = (),
        anchor_tokens: "int" = 0,
    ) -> "None":
        self.ensure_dirs()
        expanded_history: "typing.List[ConversationItem]" = []
        post_map = emitted_post_user_messages or {}
        for item in history:
            expanded_history.append(item)
            if isinstance(item, UserMessage) and item.message_id in post_map:
                expanded_history.extend(post_map[item.message_id])

        state = {
            "conversation_id": self.conversation_id,
            "cwd": str(self.cwd),
            "updated_at": format_utc_iso(self._now()),
            "model": model,
            "last_reported_model_display": last_reported_model_display,
            "step_counter": step_counter,
            "anchored_user_messages": dict(anchored_user_messages),
            "user_requests": list(user_requests),
            "parent_messages": list(parent_messages),
            "spawned_subagents": [sub.to_dict() for sub in spawned_subagents],
            "anchor_tokens": int(anchor_tokens),
            "history": [_serialize_conversation_item(it) for it in expanded_history],
        }
        self.session_state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_conversation(
        cls,
        app_data_dir: "Path",
        conversation_id: "str",
        clock: "typing.Optional[typing.Callable[[], datetime]]" = None,
    ) -> "RestoredConversation":
        resolved_app = app_data_dir.expanduser()
        brain_dir = resolved_app / "brain" / conversation_id
        system_dir = brain_dir / ".system_generated"
        session_state_path = system_dir / "pyagy_session.json"
        now_dt = clock() if clock is not None else None
        restart_ts = format_utc_iso(now_dt)
        restart_notice = SERVER_RESTART_NOTICE_CONTENT
        if any((system_dir / "tasks").glob("task-*.schedule.json")):
            restart_notice = (
                "[Notice] Your saved schedules will resume. Your subagents and shell commands "
                "have been stopped due to server restart. Check their status and restart as needed."
            )
        restart_msg = make_system_user_message(
            content=restart_notice,
            sender="system",
            priority="MESSAGE_PRIORITY_LOW",
            timestamp_utc=restart_ts,
        )

        if session_state_path.is_file():
            raw = json.loads(session_state_path.read_text(encoding="utf-8"))
            history = tuple(
                _deserialize_conversation_item(entry)
                for entry in raw.get("history", [])
                if isinstance(entry, dict)
            )
            cwd_str = raw.get("cwd")
            raw_user_reqs = raw.get("user_requests")
            raw_parent_msgs = raw.get("parent_messages")
            raw_subagents = raw.get("spawned_subagents")
            return RestoredConversation(
                conversation_id=conversation_id,
                cwd=Path(cwd_str) if isinstance(cwd_str, str) and cwd_str else None,
                model=str(raw["model"]) if raw.get("model") else None,
                last_reported_model_display=(
                    str(raw["last_reported_model_display"])
                    if raw.get("last_reported_model_display") is not None
                    else None
                ),
                step_counter=int(raw.get("step_counter", 1)),
                history=history,
                anchored_user_messages={
                    str(k): str(v)
                    for k, v in (raw.get("anchored_user_messages") or {}).items()
                },
                restart_notice_message=restart_msg,
                user_requests=(
                    tuple(str(u) for u in raw_user_reqs)
                    if isinstance(raw_user_reqs, list)
                    else ()
                ),
                parent_messages=(
                    tuple(str(m) for m in raw_parent_msgs)
                    if isinstance(raw_parent_msgs, list)
                    else ()
                ),
                spawned_subagents=(
                    tuple(
                        SpawnedSubagentSummary.from_dict(entry)
                        for entry in raw_subagents
                        if isinstance(entry, dict)
                    )
                    if isinstance(raw_subagents, list)
                    else ()
                ),
                anchor_tokens=int(raw.get("anchor_tokens", 0) or 0),
            )

        transcript_full = system_dir / "logs" / "transcript_full.jsonl"
        if not transcript_full.is_file():
            raise FileNotFoundError(f"conversation {conversation_id!r} not found under {brain_dir}")

        reconstructed: "typing.List[ConversationItem]" = []
        reconstructed_user_reqs: "typing.List[str]" = []
        last_model: "typing.Optional[str]" = None
        last_display: "typing.Optional[str]" = None
        lines = [
            ln for ln in transcript_full.read_text(encoding="utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
        for line in lines:
            step = json.loads(line)
            step_type = step.get("type")
            content = str(step.get("content", "") or "")
            if step_type == "USER_INPUT":
                if content.startswith("<USER_REQUEST>\n") and "\n</USER_REQUEST>" in content:
                    raw_req = content[len("<USER_REQUEST>\n") : content.index("\n</USER_REQUEST>")]
                    reconstructed_user_reqs.append(raw_req)
                    reconstructed.append(UserMessage(text=content, name=AGY_RAW_USER_NAME))
                else:
                    reconstructed_user_reqs.append(content)
                    reconstructed.append(UserMessage(text=content))
                marker = "The user changed setting `Model Selection` from "
                if marker in content:
                    tail = content.split(marker, 1)[1]
                    if " to " in tail:
                        after_to = tail.split(" to ", 1)[1]
                        last_display = after_to.split(". No need to comment", 1)[0]
            elif step_type == "SYSTEM_MESSAGE":
                reconstructed.append(
                    UserMessage(
                        text=content,
                        name=AGY_SYSTEM_MESSAGE_NAME,
                        visible=_infer_system_message_visible(content),
                    )
                )
            elif step_type == "CHECKPOINT":
                reconstructed.clear()
                summary_body = (
                    content
                    if content.startswith("<CONTEXT_SUMMARY>")
                    else f"<CONTEXT_SUMMARY>\n{content}\n</CONTEXT_SUMMARY>"
                )
                reconstructed.append(UserMessage(text=summary_body, name=AGY_RAW_USER_NAME))
            elif step_type == "PLANNER_RESPONSE":
                if isinstance(step.get("model"), str):
                    last_model = step["model"]
                thinking = step.get("thinking")
                if isinstance(thinking, str) and thinking:
                    reconstructed.append(
                        ReasoningItem(
                            payload={
                                "type": "reasoning",
                                "summary": [{"type": "summary_text", "text": thinking}],
                            }
                        )
                    )
                if content:
                    reconstructed.append(AssistantMessage(text=content))

        return RestoredConversation(
            conversation_id=conversation_id,
            cwd=None,
            model=last_model,
            last_reported_model_display=last_display,
            step_counter=max(1, len(lines)),
            history=tuple(reconstructed),
            anchored_user_messages={},
            restart_notice_message=restart_msg,
            user_requests=tuple(reconstructed_user_reqs),
        )


def find_latest_conversation_id(
    app_data_dir: "Path",
    cwd: "typing.Optional[Path]" = None,
) -> "str":
    brain_root = app_data_dir.expanduser() / "brain"
    if not brain_root.is_dir():
        raise FileNotFoundError(f"no conversations found in {brain_root}")

    resolved_cwd = str(cwd.resolve()) if cwd is not None else None
    matching_candidates: "typing.List[typing.Tuple[float, str]]" = []
    all_candidates: "typing.List[typing.Tuple[float, str]]" = []

    for entry in brain_root.iterdir():
        if not entry.is_dir():
            continue
        state_file = entry / ".system_generated" / "pyagy_session.json"
        transcript_file = entry / ".system_generated" / "logs" / "transcript_full.jsonl"
        if state_file.is_file():
            mtime = state_file.stat().st_mtime
            all_candidates.append((mtime, entry.name))
            if resolved_cwd is not None:
                raw = json.loads(state_file.read_text(encoding="utf-8"))
                if raw.get("cwd") == resolved_cwd:
                    matching_candidates.append((mtime, entry.name))
        elif transcript_file.is_file():
            mtime = transcript_file.stat().st_mtime
            all_candidates.append((mtime, entry.name))

    pool = matching_candidates or all_candidates
    if not pool:
        raise FileNotFoundError(f"no resumable conversations found in {brain_root}")
    pool.sort(key=lambda pair: pair[0], reverse=True)
    return pool[0][1]
