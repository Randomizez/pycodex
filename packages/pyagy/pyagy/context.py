"""AGY context manager and <USER_REQUEST> / <SYSTEM_MESSAGE> formatting."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import typing

from pycodex.context import (
    USER_INSTRUCTIONS_PREFIX,
    ContextConfig,
    ContextManager,
    _normalize_tool_calls,
)
from pycodex.protocol import (
    ContextMessage,
    ConversationItem,
    Prompt,
    ToolCall,
    ToolResult,
    ToolSpec,
    UserMessage,
)

from .models import resolve_agy_model
from .prompts.system_prompt import build_system_instruction, default_app_data_dir


AGY_SYSTEM_MESSAGE_NAME = "agy_system_message"
AGY_RAW_USER_NAME = "agy_raw_user"
AGY_COMPACTION_PROMPT_NAME = "agy_compaction_prompt"

AGY_AUTO_COMPACTION_PROMPT = (
    "You have been working on the task described above but have not yet completed\n"
    "it. Write a continuation summary that will allow you (or another instance of\n"
    "yourself) to resume work efficiently in a future context window where the\n"
    "full conversation history will NOT be available—only this summary.\n"
    "\n"
    "This summary is all that will be available to you going forward in the future\n"
    "context window. Do not call any tools, simply just provide the summary based on\n"
    "the information available in the current context window.\n"
    "\n"
    "Your summary must be structured, concise, and actionable. Optimize for enabling immediate resumption with zero redundant work.\n"
    "\n"
    "Include the following sections:\n"
    "\n"
    "1. **Task Overview**\n"
    "   - The user's core request, success criteria, and requested deliverables. Record any target IDs, URLs, or reporting recipients from the request.\n"
    "   - Scope boundaries or task-specific constraints\n"
    "   - Any ambiguities that were resolved (and how)\n"
    "\n"
    "2. **Progress**\n"
    "   - What has been completed, with concrete references (file paths,\n"
    "     test or verification commands run and their outcomes, resource identifiers, tool outputs, URLs, etc.)\n"
    "   - Key plans, design docs, or artifacts produced and their current state\n"
    "   - Skills, documentation, or reference files that were read or inspected\n"
    "   - What is in progress or only inspected/planned, and its current state. Distinguish completed edits from files that were only read or partially edited, and do not mark unverified test commands as completed or passed.\n"
    "\n"
    "3. **Key Findings**\n"
    "   - Technical constraints, requirements, or domain details uncovered\n"
    "   - Decisions made and their rationale\n"
    "\n"
    "4. **Errors & Fixes**\n"
    "   This list should be exhaustive across the entire session so that no resolved or unresolved failure is omitted:\n"
    "   - All errors, failed commands, or test/build failures encountered during the session. Include both errors that were resolved or worked around and any unresolved failures, with specific error messages and root causes.\n"
    "   - How each error was fixed or worked around, and any user feedback on the error\n"
    "   - Approaches that were tried and abandoned (and why—this prevents\n"
    "     the successor from repeating them)\n"
    "\n"
    "5. **Active Context**\n"
    "   - State of any external resources, sessions, environments, or multi-agent context, such as parent or child agents in use.\n"
    "   - Relevant intermediate results, hypotheses, or working assumptions\n"
    "   - Dependencies between components or steps\n"
    "\n"
    "6. **Next Steps**\n"
    "   The next steps should provide all information required to continue working on the task. This includes any remaining features, bugs, points of investigation, and verification items. This list should be exhaustive, so that an agent working with this list can complete the full original task end-to-end. For example:\n"
    "   - Remaining steps from any agreed plan, design doc, or active workflows and instructions, in priority order. Preserve specific uncompleted items rather than collapsing checklists into high-level summaries.\n"
    "   - Remaining files to create, modify, move, copy, or delete, with the key changes needed in each\n"
    "   - Remaining verification steps, test or build commands, or deliverables needed to complete the task\n"
    "   - Any required cleanup, such as removing temporary files, debug code, or temporary configuration changes.\n"
    "   - Known blockers, open questions, prerequisites, or risks for each step\n"
    "\n"
    "7. **Commitments & Constraints**\n"
    "   - Standing rules, personas, or response formatting requirements from user instructions, rules files, or reference documents.\n"
    '   - Promises made to the user (e.g., "I said I would do X before Y")\n'
    "   - User preferences or style requirements\n"
    "   - Any boundaries the user set on approach, tools, or scope\n"
    "\n"
    "Be concise but complete—err on the side of including concrete details needed to\n"
    "prevent duplicate work, omitted plan steps, repeated mistakes, or broken promises.\n"
    "\n"
    "Wrap your response in <summary></summary> tags.\n"
)

AGY_ON_DEMAND_COMPACTION_SUFFIX = (
    "\n\nON-DEMAND COMPACTION:\n"
    "The user triggered this compaction. Follow any instructions in their latest message, if present."
)

AGY_MANUAL_COMPACTION_PROMPT = AGY_AUTO_COMPACTION_PROMPT + AGY_ON_DEMAND_COMPACTION_SUFFIX

CONVERSATION_LOGS_SECTION = (
    "# Conversation Logs\n"
    "Start with the compact transcript; if a step has `truncated_fields`, read only that specific line in `transcript_full.jsonl`:\n"
    "- `<appDataDir>/brain/<conversation-id>/.system_generated/logs/transcript.jsonl`\n"
    "- `<appDataDir>/brain/<conversation-id>/.system_generated/logs/transcript_full.jsonl`"
)

SYSTEM_MESSAGE_PREAMBLE = (
    "The following is a <SYSTEM_MESSAGE> not actually sent by the user. "
    "It is provided by the system as important information to pay attention to."
)

SERVER_RESTART_NOTICE_CONTENT = (
    "[Notice] All your subagents and background tasks have been stopped due to server restart. "
    "If you want a subagent to continue working, it needs to be revived by sending it a new message. "
    "If resuming work, please check on status and restart as needed."
)


@dataclass(frozen=True)
class SpawnedSubagentSummary:
    type_name: "str"
    role: "str"
    conversation_id: "str"
    log_absolute_uri: "str"
    workspace_uris: "typing.Tuple[str, ...]"

    def to_dict(self) -> "typing.Dict[str, object]":
        return {
            "type_name": self.type_name,
            "role": self.role,
            "conversation_id": self.conversation_id,
            "log_absolute_uri": self.log_absolute_uri,
            "workspace_uris": list(self.workspace_uris),
        }

    @classmethod
    def from_dict(cls, data: "typing.Mapping[str, object]") -> "SpawnedSubagentSummary":
        raw_ws = data.get("workspace_uris")
        ws_tuple = (
            tuple(str(u) for u in raw_ws)
            if isinstance(raw_ws, (list, tuple))
            else ()
        )
        return cls(
            type_name=str(data.get("type_name", "")),
            role=str(data.get("role", "")),
            conversation_id=str(data.get("conversation_id", "")),
            log_absolute_uri=str(data.get("log_absolute_uri", "")),
            workspace_uris=ws_tuple,
        )


def normalize_agy_usage(
    usage: "typing.Optional[typing.Mapping[str, object]]",
) -> "typing.Dict[str, int]":
    if not isinstance(usage, typing.Mapping):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cache_read_tokens": 0,
            "total_tokens": 0,
        }
    raw_in = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    out_tok = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    cached = int(usage.get("cache_read_tokens", 0) or 0)
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details")
    has_cached_details = isinstance(details, dict) and "cached_tokens" in details
    if has_cached_details:
        assert isinstance(details, dict)
        cached = int(details.get("cached_tokens", cached) or 0)
        in_tok = max(0, raw_in - cached)
        tot_tok = in_tok + out_tok
    else:
        in_tok = raw_in
        tot_tok = int(usage.get("total_tokens", in_tok + out_tok) or 0)
    thinking = int(usage.get("thinking_tokens", 0) or 0)
    out_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details")
    if isinstance(out_details, dict):
        thinking = int(out_details.get("reasoning_tokens", thinking) or 0)
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "thinking_tokens": thinking,
        "cache_read_tokens": cached,
        "total_tokens": tot_tok,
    }


def format_subagents_section(
    subagents: "typing.Sequence[SpawnedSubagentSummary]",
) -> "str":
    by_id: "typing.Dict[str, SpawnedSubagentSummary]" = {}
    for item in subagents:
        by_id[item.conversation_id] = item
    ordered = [by_id[cid] for cid in sorted(by_id.keys())]
    blocks: "typing.List[str]" = []
    for sub in ordered:
        ws_lines = ",\n".join(
            f"      {json.dumps(uri, ensure_ascii=False)}" for uri in sub.workspace_uris
        )
        ws_block = f"[\n{ws_lines}\n    ]" if ws_lines else "[]"
        blocks.append(
            "{\n"
            '  "spec": {\n'
            f'    "typeName": {json.dumps(sub.type_name, ensure_ascii=False)},\n'
            f'    "role": {json.dumps(sub.role, ensure_ascii=False)}\n'
            "  },\n"
            '  "result": {\n'
            f'    "conversationId": {json.dumps(sub.conversation_id, ensure_ascii=False)},\n'
            f'    "logAbsoluteUri": {json.dumps(sub.log_absolute_uri, ensure_ascii=False)},\n'
            f'    "workspaceUris": {ws_block}\n'
            "  }\n"
            "}"
        )
    return (
        f"# Subagents ({len(ordered)} total)\n"
        "The following subagents were spawned during this conversation.\n"
        "View the conversation log to see the full conversation with the subagent "
        "and use send_message with the conversation_id to communicate with them.\n\n"
        + "\n".join(blocks)
    )


def format_wire_context_summary(
    summary_text: "str",
    local_iso: "str",
    user_requests: "typing.Sequence[str]" = (),
    parent_messages: "typing.Sequence[str]" = (),
    subagents: "typing.Sequence[SpawnedSubagentSummary]" = (),
    is_subagent: "bool" = False,
) -> "str":
    sections: "typing.List[str]" = [
        "The following is a summary of the conversation history that has been truncated to fit within the context window:",
        f"This summary was generated at {local_iso}.",
    ]
    if is_subagent:
        recent_parent = list(parent_messages)[-10:]
        if recent_parent:
            numbered = "\n".join(f"{idx}. {msg}" for idx, msg in enumerate(recent_parent, 1))
            sections.append(
                "# Parent Agent Messages\n"
                "The following were the most recent messages from your parent agent in chronological order:\n"
                f"{numbered}"
            )
    else:
        recent_users = list(user_requests)[-10:]
        if recent_users:
            numbered = "\n".join(f"{idx}. {req}" for idx, req in enumerate(recent_users, 1))
            sections.append(
                "# User Requests\n"
                "The following were the most recent user requests in chronological order:\n"
                f"{numbered}"
            )

    sections.append(f"# Previous Session Summary:\n{summary_text.strip()}")
    if subagents:
        sections.append(format_subagents_section(subagents))
    sections.append(CONVERSATION_LOGS_SECTION)
    return "<CONTEXT_SUMMARY>\n" + "\n\n".join(sections) + "\n</CONTEXT_SUMMARY>"


def format_checkpoint_transcript_text(
    summary_text: "str",
    user_requests: "typing.Sequence[str]" = (),
    parent_messages: "typing.Sequence[str]" = (),
    subagents: "typing.Sequence[SpawnedSubagentSummary]" = (),
    is_subagent: "bool" = False,
) -> "str":
    sections: "typing.List[str]" = [
        "# Resuming from a compaction",
        (
            "You are continuing work on the task described above, but you have lost access to "
            "the full conversation history, and need to resume work efficiently using the progress summary below:"
        ),
    ]
    if is_subagent:
        recent_parent = list(parent_messages)[-10:]
        if recent_parent:
            numbered = "\n".join(f"{idx}. {msg}" for idx, msg in enumerate(recent_parent, 1))
            sections.append(
                "# Parent Agent Messages\n"
                "The following were messages from your parent agent in the truncated conversation in chronological order:\n"
                f"{numbered}"
            )
    else:
        recent_users = list(user_requests)[-10:]
        if recent_users:
            numbered = "\n".join(f"{idx}. {req}" for idx, req in enumerate(recent_users, 1))
            sections.append(
                "# User Requests\n"
                "The following were user requests from the truncated conversation in chronological order:\n"
                f"{numbered}"
            )

    cleaned_summary = summary_text.strip()
    if cleaned_summary:
        sections.append(cleaned_summary)
    if subagents:
        sections.append(format_subagents_section(subagents))
    sections.append(CONVERSATION_LOGS_SECTION)
    return "\n\n".join(sections)


def format_local_iso(now: "typing.Optional[datetime]" = None) -> "str":
    dt = (now if now is not None else datetime.now().astimezone())
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.isoformat(timespec="seconds")


def format_utc_iso(now: "typing.Optional[datetime]" = None) -> "str":
    dt = (now if now is not None else datetime.now(timezone.utc))
    if dt.tzinfo is None:
        dt = dt.astimezone(timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def format_single_system_message(
    content: "str",
    sender: "str" = "system",
    priority: "str" = "MESSAGE_PRIORITY_HIGH",
    timestamp_utc: "typing.Optional[str]" = None,
) -> "str":
    ts = timestamp_utc or format_utc_iso()
    return (
        f"{SYSTEM_MESSAGE_PREAMBLE}\n\n"
        "<SYSTEM_MESSAGE>\n"
        f"[Message] timestamp={ts} sender={sender} priority={priority} content={content}\n"
        "</SYSTEM_MESSAGE>"
    )


def make_system_user_message(
    content: "str",
    sender: "str" = "system",
    priority: "str" = "MESSAGE_PRIORITY_HIGH",
    timestamp_utc: "typing.Optional[str]" = None,
    visible: "typing.Union[bool, str]" = False,
) -> "UserMessage":
    return UserMessage(
        text=format_single_system_message(
            content=content,
            sender=sender,
            priority=priority,
            timestamp_utc=timestamp_utc,
        ),
        name=AGY_SYSTEM_MESSAGE_NAME,
        visible=visible,
    )


def format_user_request_text(
    raw_text: "str",
    local_iso: "str",
    previous_model_display: "typing.Optional[str]",
    current_model_display: "typing.Optional[str]",
    include_settings_change: "bool",
) -> "str":
    body = (
        f"<USER_REQUEST>\n{raw_text}\n</USER_REQUEST>\n"
        f"<ADDITIONAL_METADATA>\nThe current local time is: {local_iso}.\n</ADDITIONAL_METADATA>"
    )
    if include_settings_change and current_model_display is not None:
        prev_label = "None" if previous_model_display is None else previous_model_display
        body += (
            "\n<USER_SETTINGS_CHANGE>\n"
            f"The user changed setting `Model Selection` from {prev_label} to {current_model_display}. "
            "No need to comment on this change if the user doesn't ask about it. "
            "If reporting what model you are, please use a human readable name instead of the exact string.\n"
            "</USER_SETTINGS_CHANGE>"
        )
    return body


class AgyContextManager(ContextManager):
    """ContextManager producing exact AGY systemInstruction and user turn wrappers."""

    def __init__(
        self,
        config: "ContextConfig",
        conversation_id: "str",
        app_data_dir: "typing.Optional[Path]" = None,
        additional_workspaces: "typing.Sequence[Path]" = (),
        global_config_dir: "typing.Optional[Path]" = None,
        custom_system_prompt: "typing.Optional[str]" = None,
        config_effort: "typing.Optional[str]" = None,
        explicit_effort: "typing.Optional[str]" = None,
        last_reported_model_display: "typing.Optional[str]" = None,
        anchored_user_messages: "typing.Optional[typing.Dict[str, str]]" = None,
        pending_post_user_messages: "typing.Sequence[UserMessage]" = (),
        clock: "typing.Optional[typing.Callable[[], datetime]]" = None,
    ) -> "None":
        super().__init__(
            replace(
                config,
                include_permissions_instructions=False,
                code_mode_only=False,
            )
        )
        self.conversation_id = conversation_id
        self.app_data_dir = (
            app_data_dir.expanduser() if app_data_dir is not None else default_app_data_dir()
        )
        self.additional_workspaces = tuple(Path(p).expanduser().resolve() for p in additional_workspaces)
        self.global_config_dir = global_config_dir
        self.custom_system_prompt = custom_system_prompt
        self.config_effort = config_effort
        self.explicit_effort = explicit_effort
        self._last_reported_model_display = last_reported_model_display
        self._anchored_user_messages: "typing.Dict[str, str]" = dict(anchored_user_messages or {})
        self._pending_post_user_messages: "typing.List[UserMessage]" = list(pending_post_user_messages)
        self._emitted_post_user_for_msg_id: "typing.Dict[str, typing.Tuple[UserMessage, ...]]" = {}
        self._clock = clock

    def _now(self) -> "typing.Optional[datetime]":
        return self._clock() if self._clock is not None else None

    @property
    def last_reported_model_display(self) -> "typing.Optional[str]":
        return self._last_reported_model_display

    @property
    def anchored_user_messages(self) -> "typing.Dict[str, str]":
        return dict(self._anchored_user_messages)

    @property
    def emitted_post_user_messages(self) -> "typing.Dict[str, typing.Tuple[UserMessage, ...]]":
        return dict(self._emitted_post_user_for_msg_id)

    def set_explicit_effort(self, effort: "typing.Optional[str]") -> "None":
        self.explicit_effort = effort

    def clear_anchored_messages(self) -> "None":
        self._anchored_user_messages.clear()
        self._emitted_post_user_for_msg_id.clear()

    def set_anchored_messages(
        self,
        anchored: "typing.Mapping[str, str]",
        last_reported_model_display: "typing.Optional[str]" = None,
    ) -> "None":
        self._anchored_user_messages = dict(anchored)
        self._emitted_post_user_for_msg_id.clear()
        self._last_reported_model_display = last_reported_model_display

    def _resolve_model_spec(self):
        return resolve_agy_model(
            self.config.model or "gemini-3.1-pro-high",
            config_effort=self.config_effort,
            explicit_effort=self.explicit_effort,
        )

    def resolve_auto_compact_token_limit(self) -> "typing.Optional[int]":
        env_raw = os.environ.get("AGY_CLI_AUTO_COMPACT_OVERRIDE")
        if env_raw is not None and env_raw.strip():
            return self._resolve_model_spec().resolve_auto_compact_threshold(env_raw)
        if self.config.model_auto_compact_token_limit is not None:
            return self.config.model_auto_compact_token_limit
        return self._resolve_model_spec().resolve_auto_compact_threshold()

    def resolve_model_max_context_window(self) -> "typing.Optional[int]":
        spec = self._resolve_model_spec()
        if spec.auto_compact_threshold is not None:
            return spec.auto_compact_threshold * 2
        return self.config.model_context_window

    def resolve_model_context_window(self) -> "typing.Optional[int]":
        return self.resolve_model_max_context_window()

    def resolve_code_mode_only(self) -> "bool":
        return False

    def resolve_base_instructions(self) -> "str":
        if self._base_instructions_override is not None:
            return self._base_instructions_override
        codex_skills_home = (
            self.config.codex_home
            if self._include_skills_instructions and self.config.codex_home is not None
            else None
        )
        return build_system_instruction(
            cwd=self.cwd,
            conversation_id=self.conversation_id,
            app_data_dir=self.app_data_dir,
            additional_workspaces=self.additional_workspaces,
            global_config_dir=self.global_config_dir,
            custom_system_prompt=self.custom_system_prompt,
            codex_skills_home=codex_skills_home,
        )

    def _current_model_display(self) -> "str":
        return self._resolve_model_spec().display_name

    def _build_optional_context_prefix(self) -> "typing.List[ConversationItem]":
        prefix: "typing.List[ConversationItem]" = []
        if self.config.developer_instructions is not None:
            prefix.append(
                ContextMessage(
                    role="developer",
                    text=self.config.developer_instructions,
                )
            )
        sections: "typing.List[str]" = []
        if self.config.user_instructions is not None:
            heading = (
                f"{USER_INSTRUCTIONS_PREFIX}{self.cwd}"
                if self.config.include_environment_context
                else "# User instructions"
            )
            sections.append(
                f"{heading}\n\n<INSTRUCTIONS>\n{self.config.user_instructions}\n</INSTRUCTIONS>"
            )
        if self.config.include_environment_context:
            sections.append(self._serialize_environment_context())
        sections.extend(self._extra_contextual_user_messages)
        if sections:
            prefix.append(
                ContextMessage(
                    role="user",
                    content_items=tuple(
                        {"type": "input_text", "text": section} for section in sections
                    ),
                )
            )
        return prefix

    def _format_history_item(
        self,
        item: "ConversationItem",
        capture_context: "bool",
    ) -> "typing.List[ConversationItem]":
        if not isinstance(item, UserMessage):
            return [item]
        if item.name in (
            AGY_SYSTEM_MESSAGE_NAME,
            AGY_RAW_USER_NAME,
            AGY_COMPACTION_PROMPT_NAME,
        ):
            return [item]

        anchored = self._anchored_user_messages.get(item.message_id)
        is_new_user_message = anchored is None
        if is_new_user_message:
            current_display = self._current_model_display()
            changed = current_display != self._last_reported_model_display
            anchored = format_user_request_text(
                raw_text=item.text,
                local_iso=format_local_iso(self._now()),
                previous_model_display=self._last_reported_model_display,
                current_model_display=current_display,
                include_settings_change=changed,
            )
            if capture_context:
                self._anchored_user_messages[item.message_id] = anchored
                if changed:
                    self._last_reported_model_display = current_display

        formatted_user = replace(item, text=anchored)
        result: "typing.List[ConversationItem]" = [formatted_user]

        if item.message_id in self._emitted_post_user_for_msg_id:
            result.extend(self._emitted_post_user_for_msg_id[item.message_id])
        elif is_new_user_message and self._pending_post_user_messages and capture_context:
            attached = tuple(self._pending_post_user_messages)
            self._pending_post_user_messages.clear()
            self._emitted_post_user_for_msg_id[item.message_id] = attached
            result.extend(attached)

        return result

    def build_prompt(
        self,
        history: "typing.Union[typing.Tuple[ConversationItem, ...], typing.List[ConversationItem]]",
        tools: "typing.List[ToolSpec]",
        parallel_tool_calls: "bool",
        turn_id: "typing.Union[str, None]" = None,
        reasoning_effort: "typing.Union[str, None]" = None,
        capture_context: "bool" = True,
        session_id: "typing.Union[str, None]" = None,
    ) -> "Prompt":
        turn_metadata = self.get_turn_metadata(turn_id) if turn_id is not None else None
        transformed_items: "typing.List[ConversationItem]" = self._build_optional_context_prefix()
        for item in history:
            transformed_items.extend(self._format_history_item(item, capture_context))
        normalized_items = _normalize_tool_calls(transformed_items, False)
        call_positions = {
            item.call_id: pos
            for pos, item in enumerate(normalized_items)
            if isinstance(item, ToolCall)
        }
        ordered_items: "typing.List[ConversationItem]" = []
        idx = 0
        total = len(normalized_items)
        while idx < total:
            item = normalized_items[idx]
            if isinstance(item, ToolResult):
                end = idx + 1
                while end < total and isinstance(normalized_items[end], ToolResult):
                    end += 1
                run = normalized_items[idx:end]
                run.sort(key=lambda tr: call_positions.get(tr.call_id, total))
                ordered_items.extend(run)
                idx = end
            else:
                ordered_items.append(item)
                idx += 1
        base_instructions = self.resolve_base_instructions()
        return Prompt(
            input=ordered_items,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            base_instructions=base_instructions,
            turn_id=turn_id,
            turn_metadata=turn_metadata,
            session_id=session_id,
        )
