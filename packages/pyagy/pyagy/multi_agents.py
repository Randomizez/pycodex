"""AGY conversation session nodes and multi-agent controller."""

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import typing
import uuid

from pycodex.agent import Agent
from pycodex.context import ContextConfig
from pycodex.events import (
    AutoCompactCompletedEvent,
    AutoCompactStartedEvent,
    Event,
    EventScope,
    MessagesAddedEvent,
    ModelCompletedEvent,
    RuntimeUpdatedEvent,
    TokenCountEvent,
    TurnCompletedEvent,
    TurnEvent,
    TurnFailedEvent,
    TurnInputReadyEvent,
    TurnInterrupted,
)
from pycodex.model import ContextLengthExceeded
from pycodex.multi_agents import AgentsManager
from pycodex.protocol import (
    AgentMessage,
    AssistantMessage,
    ContextMessage,
    ConversationItem,
    ModelResponse,
    ReasoningItem,
    ToolCall,
    ToolResult,
    TurnResult,
    UserMessage,
)
from pycodex.tools.base_tool import BaseTool
from pycodex.utils.compactor import CompactResult, prune_oldest_tool_response
from pycodex.utils.session_persist import resolve_codex_home, rollout_path_for_session

from .context import (
    AGY_AUTO_COMPACTION_PROMPT,
    AGY_COMPACTION_PROMPT_NAME,
    AGY_MANUAL_COMPACTION_PROMPT,
    AGY_RAW_USER_NAME,
    AGY_SYSTEM_MESSAGE_NAME,
    AgyContextManager,
    SpawnedSubagentSummary,
    format_checkpoint_transcript_text,
    format_local_iso,
    format_utc_iso,
    format_wire_context_summary,
    make_system_user_message,
    normalize_agy_usage,
)
from .gemini_wire import AgyResponsesModelClient
from .models import resolve_agy_model, resolve_subagent_model
from .prompts.system_prompt import (
    build_subagent_system_instruction,
    find_git_repo_root,
)
from .task_manager import AgyTaskManager
from .tools import (
    IMAGE_GENERATOR_SUBAGENT_TOOLS,
    RESEARCH_SUBAGENT_TOOLS,
    AgyRuntimeEnvironment,
    SearchHandler,
    build_agy_tool_registry,
    load_official_declarations,
)
from .transcript import (
    AGY_TOOL_OUTPUT_MAX_BYTES,
    AgyTranscriptStore,
    align_anchor_state_to_history,
    infer_anchor_history_len,
    truncate_agy_tool_output,
)

if typing.TYPE_CHECKING:
    from responses_server.config import CompatServerConfig
    from pycodex.agent import TurnContext
    from pycodex.runtime import AgentRuntime


@dataclass(frozen=True)
class AgyCheckpointEvent(TurnEvent):
    kind: "typing.ClassVar[str]" = "agy_checkpoint"
    turn_id: "str"
    checkpoint_text: "str"
    duration_seconds: "float"
    usage: "typing.Dict[str, int]"
    submission_id: "typing.Union[str, None]" = None


def _parse_manual_compact_messages(
    messages: "typing.Sequence[ConversationItem]",
) -> "typing.Tuple[bool, bool]":
    for item in messages:
        if isinstance(item, UserMessage) and item.name not in (
            AGY_SYSTEM_MESSAGE_NAME,
            AGY_RAW_USER_NAME,
            AGY_COMPACTION_PROMPT_NAME,
        ):
            stripped = item.text.strip()
            if stripped == "/compact":
                return True, True
            if stripped.startswith("/compact") and stripped[8:9].isspace():
                has_arg = bool(stripped[8:].strip())
                return True, not has_arg
    return False, False


class AgyAgentsManager(AgentsManager):
    """AgentsManager exposing AGY subagent tree to frontends without Codex FINAL_ANSWER injection."""

    def finish_turn(self, agent_id: "str", final_answer: "typing.Optional[str]") -> "None":
        self.notify_parents(agent_id)


def _truncate_oversized_history_for_compaction(
    items: "typing.Sequence[ConversationItem]",
) -> "typing.Tuple[typing.List[ConversationItem], int]":
    updated: "typing.List[ConversationItem]" = []
    truncated_count = 0
    for item in items:
        if (
            isinstance(item, UserMessage)
            and item.name == AGY_SYSTEM_MESSAGE_NAME
            and len(item.text.encode("utf-8")) > AGY_TOOL_OUTPUT_MAX_BYTES
        ):
            updated.append(replace(item, text=truncate_agy_tool_output(item.text)))
            truncated_count += 1
        elif (
            isinstance(item, ToolResult)
            and item.content_items is None
            and not item.is_error
            and item.name != "read_file"
            and isinstance(item.output, str)
            and len(item.output.encode("utf-8")) > AGY_TOOL_OUTPUT_MAX_BYTES
        ):
            updated.append(replace(item, output=truncate_agy_tool_output(item.output)))
            truncated_count += 1
        else:
            updated.append(item)
    return updated, truncated_count


class AgyAgent(Agent):
    """Agent subclass implementing AGY manual `/compact` and pre-invocation auto-compaction."""

    def __init__(
        self,
        model_client: "AgyResponsesModelClient",
        tool_registry: "typing.Any",
        context_config: "ContextConfig",
        session_id: "str",
        compaction_threshold_getter: "typing.Callable[[], typing.Optional[int]]",
        summary_builder: "typing.Callable[[str], typing.Tuple[str, str]]",
        parallel_tool_calls: "bool" = True,
        initial_history: "typing.Tuple[ConversationItem, ...]" = (),
        initial_anchor_tokens: "int" = 0,
        initial_anchor_history_len: "typing.Optional[int]" = None,
        session_file_path: "typing.Union[str, Path, None]" = None,
        agent_path: "str" = "/root",
        on_cwd_changed: "typing.Optional[typing.Callable[[Path], None]]" = None,
        on_resumed: "typing.Optional[typing.Callable[[str], None]]" = None,
    ) -> "None":
        super().__init__(
            model_client=model_client,
            tool_registry=tool_registry,
            context_config=context_config,
            session_id=session_id,
            parallel_tool_calls=parallel_tool_calls,
            initial_history=initial_history,
            session_file_path=session_file_path,
            agent_path=agent_path,
        )
        self._compaction_threshold_getter = compaction_threshold_getter
        self._summary_builder = summary_builder
        self._anchor_tokens = 0
        self._anchor_history_len = 0
        self.set_anchor_state(initial_anchor_tokens, initial_anchor_history_len)
        self._on_cwd_changed = on_cwd_changed
        self._on_resumed = on_resumed

    def set_cwd(self, cwd: "typing.Union[str, Path]") -> "None":
        super().set_cwd(cwd)
        if self._on_cwd_changed is not None:
            self._on_cwd_changed(self.context_manager.cwd)

    def resume(
        self, session_file_path: "typing.Union[str, Path, None]" = None
    ) -> "typing.Union[typing.Dict[str, object], None]":
        resumed = super().resume(session_file_path)
        if session_file_path is not None:
            if isinstance(self.context_manager, AgyContextManager):
                self.context_manager.clear_anchored_messages()
            self.set_anchor_state(0, 0)
            if self._on_resumed is not None:
                self._on_resumed(self.session_id)
        return resumed

    @property
    def anchor_tokens(self) -> "int":
        return self._anchor_tokens

    @property
    def anchor_history_len(self) -> "int":
        return self._anchor_history_len

    def set_anchor_state(
        self,
        anchor_tokens: "int",
        anchor_history_len: "typing.Optional[int]" = None,
    ) -> "None":
        tokens = max(0, int(anchor_tokens))
        self._anchor_tokens = tokens
        if tokens <= 0:
            self._anchor_history_len = 0
        elif anchor_history_len is None:
            self._anchor_history_len = infer_anchor_history_len(self._history)
        else:
            self._anchor_history_len = max(0, min(int(anchor_history_len), len(self._history)))

    def _estimate_delta_tokens(self) -> "int":
        delta_bytes = 0
        tail = self._history[self._anchor_history_len :]
        for item in tail:
            if isinstance(item, UserMessage):
                if isinstance(self.context_manager, AgyContextManager):
                    preview_items = self.context_manager._format_history_item(
                        item, capture_context=False
                    )
                    for p_item in preview_items:
                        if isinstance(p_item, UserMessage):
                            delta_bytes += len(p_item.text.encode("utf-8"))
                    if (
                        item.message_id not in self.context_manager._anchored_user_messages
                        and self.context_manager._pending_post_user_messages
                        and item.name
                        not in (
                            AGY_SYSTEM_MESSAGE_NAME,
                            AGY_RAW_USER_NAME,
                            AGY_COMPACTION_PROMPT_NAME,
                        )
                    ):
                        for post_msg in self.context_manager._pending_post_user_messages:
                            delta_bytes += len(post_msg.text.encode("utf-8"))
                else:
                    delta_bytes += len(item.text.encode("utf-8"))
            elif isinstance(item, ContextMessage) and item.text:
                delta_bytes += len(item.text.encode("utf-8"))
            elif isinstance(item, ToolResult):
                delta_bytes += len(str(item.output).encode("utf-8"))
            elif isinstance(item, AssistantMessage):
                delta_bytes += len(item.text.encode("utf-8"))
            elif isinstance(item, ReasoningItem):
                summary = item.payload.get("summary") if isinstance(item.payload, dict) else None
                if isinstance(summary, list):
                    for part in summary:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            delta_bytes += len(part["text"].encode("utf-8"))
            elif isinstance(item, ToolCall):
                delta_bytes += len(item.name.encode("utf-8")) + len(
                    json.dumps(item.arguments, ensure_ascii=False).encode("utf-8")
                )
        return (delta_bytes + 2) // 3 if delta_bytes > 0 else 0

    def _should_auto_compact(self) -> "bool":
        threshold = self._compaction_threshold_getter()
        if threshold is None or threshold <= 0:
            return False
        estimated = self._anchor_tokens + self._estimate_delta_tokens()
        return estimated >= threshold

    async def _run_agy_compaction(
        self,
        turn: "TurnContext",
        manual: "bool",
        prune_on_overflow: "bool" = True,
    ) -> "CompactResult":
        committed = list(self._history)
        comp_prompt_text = (
            AGY_MANUAL_COMPACTION_PROMPT if manual else AGY_AUTO_COMPACTION_PROMPT
        )
        comp_msg = UserMessage(
            text=comp_prompt_text,
            name=AGY_COMPACTION_PROMPT_NAME,
            visible=False,
        )
        history_items: "typing.List[ConversationItem]" = list(committed)
        pruned = 0
        oversized_truncated = False
        comp_start = time.monotonic()
        try:
            while True:
                working: "typing.List[ConversationItem]" = [*history_items, comp_msg]
                turn.compact_input = working
                prompt = self._build_prompt(turn.turn_id, working, capture_context=True)
                try:
                    response = await self.model_client.complete(prompt)
                    break
                except ContextLengthExceeded:
                    if not prune_on_overflow:
                        raise
                    if not oversized_truncated:
                        oversized_truncated = True
                        history_items, trunc_n = _truncate_oversized_history_for_compaction(
                            history_items
                        )
                        if trunc_n > 0:
                            pruned += trunc_n
                            continue
                    remaining_tools = sum(
                        1 for it in history_items if isinstance(it, ToolResult)
                    )
                    if remaining_tools > 0:
                        batch = (
                            1
                            if remaining_tools <= 4
                            else max(1, (remaining_tools + 3) // 4)
                        )
                        pruned_items = prune_oldest_tool_response(history_items, batch)
                        if pruned_items is not None:
                            history_items = list(pruned_items)
                            pruned += batch
                            continue
                    if len(history_items) > 1:
                        drop_n = max(1, len(history_items) // 4)
                        trimmed = list(history_items[drop_n:])
                        while len(trimmed) > 1 and isinstance(trimmed[0], ToolResult):
                            trimmed.pop(0)
                        if len(trimmed) < len(history_items):
                            pruned += len(history_items) - len(trimmed)
                            history_items = trimmed
                            continue
                    raise
        finally:
            turn.compact_input = None

        comp_duration = time.monotonic() - comp_start
        self._record_usage(turn.turn_id, response.usage, update_context=False)
        norm_usage = normalize_agy_usage(response.usage)

        summary_chunks = [
            item.text
            for item in response.items
            if isinstance(item, AssistantMessage) and item.text
        ]
        summary_text = "\n".join(summary_chunks)
        wire_summary, checkpoint_text = self._summary_builder(summary_text)

        summary_msg = ContextMessage(role="user", text=wire_summary)
        self._history = [summary_msg]
        if self._rollout_recorder is not None:
            self._rollout_recorder.append_compacted_history(self.history, committed)
        if isinstance(self.context_manager, AgyContextManager):
            self.context_manager.clear_anchored_messages()
        self.set_anchor_state(0, 0)
        self._last_total_usage_tokens = None

        EventScope.emit(
            AgyCheckpointEvent(
                turn_id=turn.turn_id,
                checkpoint_text=checkpoint_text,
                duration_seconds=comp_duration,
                usage=norm_usage,
            )
        )
        return CompactResult(self.history, len(committed), pruned)

    async def _run_compaction(
        self,
        turn: "TurnContext",
        prune_on_overflow: "bool" = True,
        record_usage: "bool" = False,
    ) -> "CompactResult":
        return await self._run_agy_compaction(
            turn,
            manual=record_usage,
            prune_on_overflow=prune_on_overflow,
        )

    async def _sample(self, turn: "TurnContext") -> "ModelResponse":
        response = await super()._sample(turn)
        if response.usage is not None:
            norm = normalize_agy_usage(response.usage)
            self._anchor_tokens = (
                norm["input_tokens"] + norm["cache_read_tokens"] + norm["output_tokens"]
            )
            self._anchor_history_len = len(self._history) + len(response.items)
        return response

    async def run_turn(
        self,
        messages: "typing.Iterable[typing.Union[UserMessage, AgentMessage, ToolResult]]",
        turn_id: "typing.Union[str, None]" = None,
    ) -> "TurnResult":
        had_prior_history = bool(self._history)
        with self._open_turn(messages, turn_id) as turn:
            is_manual_cmd, is_bare_manual = _parse_manual_compact_messages(turn.messages)
            pending_manual_compact = had_prior_history and is_manual_cmd
            last_response_items: "typing.Sequence[ConversationItem]" = ()
            while True:
                first = turn.iteration == 0
                if first:
                    self._append_history(turn.messages)
                turn.follow_up_requested = False
                if turn.deliver_messages is not None:
                    turn.deliver_messages()
                EventScope.emit(TurnInputReadyEvent(turn.turn_id))
                if turn.stop_requested:
                    raise TurnInterrupted()

                if pending_manual_compact:
                    pending_manual_compact = False
                    await self._run_agy_compaction(turn, manual=True)
                    if turn.stop_requested:
                        raise TurnInterrupted()
                    if is_bare_manual and not turn.follow_up_requested:
                        break
                elif (had_prior_history or not first) and self._should_auto_compact():
                    phase = "pre_turn" if first else "mid_turn"
                    limit = self._compaction_threshold_getter()
                    used = self._anchor_tokens + self._estimate_delta_tokens()
                    await self._auto_compact(turn, phase, used, limit)
                    if turn.stop_requested:
                        raise TurnInterrupted()

                turn.iteration += 1
                response = await self._sample(turn)
                last_response_items = response.items
                if await self._run_tools(turn, response):
                    continue
                self._append_history(response.items)
                if turn.stop_requested:
                    raise TurnInterrupted()
                if not turn.follow_up_requested:
                    break

            return TurnResult(
                turn_id=turn.turn_id,
                output_text=turn.output_text,
                iterations=turn.iteration,
                response_items=tuple(last_response_items),
                history=self.history,
            )


@dataclass(frozen=True)
class SubagentBlueprint:
    name: "str"
    description: "str"
    system_prompt: "str"
    enable_write_tools: "bool" = False
    enable_subagent_tools: "bool" = False
    enable_mcp_tools: "bool" = False


@dataclass
class SessionRunSummary:
    conversation_id: "str"
    final_response: "str"
    combined_response: "str"
    num_turns: "int"
    duration_seconds: "float"
    usage: "typing.Dict[str, int]"


class AgySubagentWorkTracker:
    """Exposes active direct AGY subagents to AgentRuntimeEnvironment."""

    def __init__(
        self,
        controller: "AgyMultiAgentController",
        conversation_id_getter: "typing.Callable[[], str]",
    ) -> "None":
        self._controller = controller
        self._conversation_id_getter = conversation_id_getter
        self._state_listener: "typing.Optional[typing.Callable[[], None]]" = None

    def bind(
        self,
        invoke: "typing.Any",
        output: "typing.Any",
        state_changed: "typing.Optional[typing.Callable[[], None]]",
    ) -> "None":
        del invoke, output
        self._state_listener = state_changed

    def set_state_listener(
        self,
        listener: "typing.Optional[typing.Callable[[], None]]",
    ) -> "None":
        self._state_listener = listener

    def shutdown(self) -> "None":
        self._state_listener = None

    def notify_changed(self) -> "None":
        if self._state_listener is not None:
            self._state_listener()

    def background_work_count(self, after_reply: "bool") -> "int":
        del after_reply
        conv_id = self._conversation_id_getter()
        return sum(
            1
            for node in self._controller.iter_direct_children(conv_id)
            if node.state == "running"
        )

    def background_work_snapshot(self) -> "typing.List[typing.Dict[str, object]]":
        conv_id = self._conversation_id_getter()
        return [
            {
                "kind": "subagent",
                "conversation_id": node.conversation_id,
                "role": node.role,
                "type_name": node.type_name,
                "status": "running",
            }
            for node in self._controller.iter_direct_children(conv_id)
            if node.state == "running"
        ]


class AgySessionNode:
    """Wraps one AGY conversation (`AgentRuntime -> Agent` + transcript + tasks)."""

    def __init__(
        self,
        conversation_id: "str",
        cwd: "Path",
        app_data_dir: "Path",
        model_client: "AgyResponsesModelClient",
        context_config: "ContextConfig",
        controller: "AgyMultiAgentController",
        compat_config: "typing.Optional[CompatServerConfig]" = None,
        parent_conversation_id: "typing.Optional[str]" = None,
        role: "str" = "root",
        type_name: "str" = "root",
        additional_workspaces: "typing.Sequence[Path]" = (),
        global_config_dir: "typing.Optional[Path]" = None,
        custom_system_prompt: "typing.Optional[str]" = None,
        toolset: "typing.Optional[typing.Sequence[str]]" = None,
        config_effort: "typing.Optional[str]" = None,
        explicit_effort: "typing.Optional[str]" = None,
        initial_history: "typing.Tuple[ConversationItem, ...]" = (),
        last_reported_model_display: "typing.Optional[str]" = None,
        anchored_user_messages: "typing.Optional[typing.Dict[str, str]]" = None,
        pending_post_user_messages: "typing.Sequence[UserMessage]" = (),
        initial_step_counter: "typing.Optional[int]" = None,
        initial_user_requests: "typing.Sequence[str]" = (),
        initial_parent_messages: "typing.Sequence[str]" = (),
        initial_spawned_subagents: "typing.Sequence[SpawnedSubagentSummary]" = (),
        initial_anchor_tokens: "int" = 0,
        initial_anchor_history_len: "typing.Optional[int]" = None,
        search_handler: "typing.Optional[SearchHandler]" = None,
        owned_workspace_dir: "typing.Optional[Path]" = None,
        git_worktree_repo: "typing.Optional[Path]" = None,
        clock: "typing.Optional[typing.Callable[[], datetime]]" = None,
        extra_tools: "typing.Sequence[BaseTool]" = (),
        enable_rollout_recording: "bool" = False,
        agents_manager: "typing.Optional[AgentsManager]" = None,
        agent_path: "str" = "/root",
    ) -> "None":
        self.conversation_id = conversation_id
        self.parent_conversation_id = parent_conversation_id
        self.role = role
        self.type_name = type_name
        self.cwd = cwd.resolve()
        self.app_data_dir = app_data_dir.expanduser()
        self.additional_workspaces = tuple(
            Path(p).expanduser().resolve() for p in additional_workspaces
        )
        self.controller = controller
        self.config_effort = config_effort
        self.explicit_effort = explicit_effort
        self._owned_workspace_dir = owned_workspace_dir
        self._git_worktree_repo = git_worktree_repo
        self._clock = clock
        self.state = "running"
        self._user_requests: "typing.List[str]" = list(initial_user_requests)
        self._parent_messages: "typing.List[str]" = list(initial_parent_messages)
        self._spawned_subagents: "typing.List[SpawnedSubagentSummary]" = list(
            initial_spawned_subagents
        )

        self.transcript_store = AgyTranscriptStore(
            app_data_dir=self.app_data_dir,
            conversation_id=self.conversation_id,
            cwd=self.cwd,
            clock=clock,
        )
        if initial_step_counter is not None:
            self.transcript_store.set_step_index(initial_step_counter)

        self.task_manager = AgyTaskManager(
            conversation_id=self.conversation_id,
            tasks_dir=self.transcript_store.tasks_dir,
            api_model_getter=self.current_api_model,
            notify_callback=self.deliver_system_message,
            transcript_store=self.transcript_store,
            clock=clock,
        )

        runtime_env = AgyRuntimeEnvironment(self.task_manager)
        self.subagent_work_tracker = AgySubagentWorkTracker(
            controller=self.controller,
            conversation_id_getter=lambda: self.conversation_id,
        )
        runtime_env.register_resource("invoke_subagent", self.subagent_work_tracker)
        self.tool_registry = build_agy_tool_registry(
            conversation_id=self.conversation_id,
            transcript_store=self.transcript_store,
            task_manager=self.task_manager,
            multi_agents=self.controller,
            compat_config=compat_config,
            api_model_getter=self.current_api_model,
            toolset=toolset,
            search_handler=search_handler,
            runtime_environment=runtime_env,
            clock=clock,
            extra_tools=extra_tools,
        )

        node_context_config = replace(
            context_config,
            cwd=self.cwd,
            model=model_client.model,
            include_permissions_instructions=False,
            code_mode_only=False,
        )
        self.context_manager = AgyContextManager(
            config=node_context_config,
            conversation_id=self.conversation_id,
            app_data_dir=self.app_data_dir,
            additional_workspaces=self.additional_workspaces,
            global_config_dir=global_config_dir,
            custom_system_prompt=custom_system_prompt,
            config_effort=config_effort,
            explicit_effort=explicit_effort,
            last_reported_model_display=last_reported_model_display,
            anchored_user_messages=anchored_user_messages,
            pending_post_user_messages=pending_post_user_messages,
            clock=clock,
        )
        if isinstance(model_client, AgyResponsesModelClient):
            model_client.set_effort_change_callback(self._on_client_effort_changed)

        from pycodex.runtime import AgentRuntime

        session_file_path = (
            rollout_path_for_session(
                node_context_config.codex_home or resolve_codex_home(),
                self.conversation_id,
            )
            if enable_rollout_recording and parent_conversation_id is None
            else None
        )
        shared_manager = (
            agents_manager
            if agents_manager is not None
            else AgyAgentsManager(model_client, node_context_config)
        )
        self.agent = AgyAgent(
            model_client=model_client,
            tool_registry=self.tool_registry,
            context_config=node_context_config,
            session_id=self.conversation_id,
            compaction_threshold_getter=self.current_auto_compact_threshold,
            summary_builder=self._build_compaction_summaries,
            parallel_tool_calls=True,
            initial_history=initial_history,
            initial_anchor_tokens=initial_anchor_tokens,
            initial_anchor_history_len=initial_anchor_history_len,
            session_file_path=session_file_path,
            agent_path=agent_path,
            on_cwd_changed=self._on_agent_cwd_changed,
            on_resumed=self._on_agent_resumed,
        )
        self.agent.context_manager = self.context_manager
        self.context_manager.agent_path = self.agent.agent_path
        self.runtime: "AgentRuntime" = AgentRuntime(self.agent, shared_manager)
        shared_manager._runtimes[self.agent.agent_id] = self.runtime

        self._logged_msg_ids: "typing.Set[str]" = {
            item.message_id for item in initial_history if isinstance(item, UserMessage)
        }
        self._activity_event: "typing.Optional[asyncio.Event]" = None
        self._turn_outputs: "typing.List[str]" = []
        self._last_turn_output = ""
        self._pending_idle_notify = False
        self._reported_to_parent = False
        self._completed_turns = 0
        self._usage_totals: "typing.Dict[str, int]" = {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cache_read_tokens": 0,
            "total_tokens": 0,
        }
        self._event_listeners: "typing.List[typing.Callable[[Event], None]]" = []
        self._frontend_id = self.runtime.attach(self._on_runtime_event, interactive=False)

    def add_event_listener(self, listener: "typing.Callable[[Event], None]") -> "None":
        self._event_listeners.append(listener)

    def current_api_model(self) -> "str":
        spec = resolve_agy_model(
            self.agent.model_name,
            config_effort=self.config_effort,
            explicit_effort=self.explicit_effort,
        )
        return spec.api_model

    def current_auto_compact_threshold(self) -> "typing.Optional[int]":
        return self.context_manager.resolve_auto_compact_token_limit()

    def record_spawned_subagent(self, summary: "SpawnedSubagentSummary") -> "None":
        self._spawned_subagents = [
            existing
            for existing in self._spawned_subagents
            if existing.conversation_id != summary.conversation_id
        ]
        self._spawned_subagents.append(summary)

    def _build_compaction_summaries(self, summary_text: "str") -> "typing.Tuple[str, str]":
        now_dt = self._clock() if self._clock is not None else None
        local_iso = format_local_iso(now_dt)
        is_sub = self.parent_conversation_id is not None
        wire_summary = format_wire_context_summary(
            summary_text=summary_text,
            local_iso=local_iso,
            user_requests=self._user_requests,
            parent_messages=self._parent_messages,
            subagents=self._spawned_subagents,
            is_subagent=is_sub,
        )
        checkpoint_text = format_checkpoint_transcript_text(
            summary_text=summary_text,
            user_requests=self._user_requests,
            parent_messages=self._parent_messages,
            subagents=self._spawned_subagents,
            is_subagent=is_sub,
        )
        return wire_summary, checkpoint_text

    def set_model(
        self,
        model: "str",
        explicit_effort: "typing.Optional[str]" = None,
    ) -> "None":
        self.runtime.set_model(model)
        if explicit_effort is not None:
            self.explicit_effort = explicit_effort
            self.context_manager.set_explicit_effort(explicit_effort)
            if isinstance(self.agent.model_client, AgyResponsesModelClient):
                self.agent.model_client.explicit_effort = explicit_effort

    def _on_client_effort_changed(self, effort: "typing.Optional[str]") -> "None":
        self.explicit_effort = effort
        self.context_manager.set_explicit_effort(effort)

    def _on_agent_cwd_changed(self, new_cwd: "Path") -> "None":
        self.cwd = new_cwd
        self.transcript_store.cwd = new_cwd

    def _on_agent_resumed(self, resumed_session_id: "str") -> "None":
        old_id = self.conversation_id
        if resumed_session_id and resumed_session_id != old_id:
            self.controller.rekey_node(old_id, resumed_session_id, self)
            self.conversation_id = resumed_session_id
            self.context_manager.conversation_id = resumed_session_id
            self.transcript_store = AgyTranscriptStore(
                app_data_dir=self.app_data_dir,
                conversation_id=resumed_session_id,
                cwd=self.cwd,
                clock=self._clock,
            )
            self.tool_registry.transcript_store = self.transcript_store
            self.task_manager.conversation_id = resumed_session_id
            self.task_manager.tasks_dir = self.transcript_store.tasks_dir
            self.task_manager.transcript_store = self.transcript_store

        self._logged_msg_ids = {
            item.message_id for item in self.agent.history if isinstance(item, UserMessage)
        }
        if self.transcript_store.session_state_path.is_file():
            restored = AgyTranscriptStore.load_conversation(
                self.app_data_dir,
                self.conversation_id,
                clock=self._clock,
            )
            self.context_manager.set_anchored_messages(
                restored.anchored_user_messages,
                last_reported_model_display=restored.last_reported_model_display,
            )
            self.transcript_store.set_step_index(restored.step_counter)
            self._user_requests = list(restored.user_requests)
            self._parent_messages = list(restored.parent_messages)
            self._spawned_subagents = list(restored.spawned_subagents)
            aligned_tokens, aligned_len = align_anchor_state_to_history(
                restored.anchor_tokens,
                restored.history,
                restored.anchor_history_len,
                self.agent.history,
            )
            self.agent.set_anchor_state(aligned_tokens, aligned_len)

    def _signal_activity(self) -> "None":
        if self._activity_event is not None:
            self._activity_event.set()

    def _on_runtime_event(self, event: "Event") -> "None":
        if isinstance(event, MessagesAddedEvent):
            for msg in event.messages:
                if isinstance(msg, UserMessage) and msg.message_id not in self._logged_msg_ids:
                    self._logged_msg_ids.add(msg.message_id)
                    if msg.name == AGY_SYSTEM_MESSAGE_NAME:
                        self.transcript_store.append_transcript_step(
                            source="SYSTEM",
                            step_type="SYSTEM_MESSAGE",
                            content=msg.text,
                        )
                    elif msg.name not in (AGY_RAW_USER_NAME, AGY_COMPACTION_PROMPT_NAME):
                        if self.parent_conversation_id is None:
                            self._user_requests.append(msg.text)
                        self.transcript_store.append_transcript_step(
                            source="USER_EXPLICIT",
                            step_type="USER_INPUT",
                            content=msg.text,
                        )
        elif isinstance(event, TokenCountEvent):
            norm = normalize_agy_usage(event.usage if isinstance(event.usage, dict) else None)
            self._usage_totals["input_tokens"] += norm["input_tokens"]
            self._usage_totals["output_tokens"] += norm["output_tokens"]
            self._usage_totals["thinking_tokens"] += norm["thinking_tokens"]
            self._usage_totals["cache_read_tokens"] += norm["cache_read_tokens"]
            self._usage_totals["total_tokens"] += norm["total_tokens"]
        elif isinstance(event, AgyCheckpointEvent):
            self.transcript_store.append_transcript_step(
                source="SYSTEM",
                step_type="CHECKPOINT",
                content=event.checkpoint_text,
                usage=event.usage,
            )
        elif isinstance(event, ModelCompletedEvent):
            self.transcript_store.append_transcript_step(
                source="MODEL",
                step_type="PLANNER_RESPONSE",
                model=self.agent.model_name,
            )
        elif isinstance(event, TurnCompletedEvent):
            self._completed_turns += 1
            out_txt = event.output_text or ""
            self._last_turn_output = out_txt
            if out_txt:
                self._turn_outputs.append(out_txt)
            self._pending_idle_notify = True
            self.save_session_state()
            self._signal_activity()
        elif isinstance(event, TurnFailedEvent):
            self._last_turn_output = f"{event.error_type}: {event.error}"
            self._pending_idle_notify = True
            self.save_session_state()
            self._signal_activity()
        elif isinstance(event, RuntimeUpdatedEvent) and event.reason == "idle":
            self._signal_activity()
            if (
                self._pending_idle_notify
                and self.parent_conversation_id is not None
                and self.state != "killed"
                and not self.has_pending_work()
            ):
                self._pending_idle_notify = False
                reported = self._reported_to_parent
                self._reported_to_parent = False
                self.state = "idle"
                self.controller.notify_parent_subagent_idle(
                    self,
                    self._last_turn_output,
                    already_reported=reported,
                )
                self.controller.notify_parent_subagent_work(self.parent_conversation_id)

        for listener in tuple(self._event_listeners):
            listener(event)

    def has_pending_work(self) -> "bool":
        return (
            self.runtime.is_busy
            or self.task_manager.has_active_non_daemon_tasks()
            or self.controller.has_active_subagents(self.conversation_id)
        )

    def deliver_system_message(
        self,
        content: "str",
        sender: "str" = "system",
        priority: "str" = "MESSAGE_PRIORITY_HIGH",
        wake_if_idle: "bool" = True,
        visible: "typing.Union[bool, str]" = "System",
    ) -> "bool":
        if self.state == "killed" or not self.runtime.accepts_input:
            return False
        ts_utc = format_utc_iso(self._clock() if self._clock is not None else None)
        if self.parent_conversation_id is not None and sender == self.parent_conversation_id:
            self._parent_messages.append(
                f"[Message] timestamp={ts_utc} sender={sender} priority={priority} content={content}"
            )
        self.transcript_store.record_system_message_file(
            content=content,
            sender=sender,
            priority=priority,
            timestamp_utc=ts_utc,
        )
        sys_msg = make_system_user_message(
            content=content,
            sender=sender,
            priority=priority,
            timestamp_utc=ts_utc,
            visible=visible,
        )
        waking = self.state == "idle" and wake_if_idle
        if waking:
            self.state = "running"
            self._reported_to_parent = False
        self.runtime.message(sys_msg)
        if wake_if_idle:
            self.runtime.trigger()
        self.task_manager.notify_incoming_message(sender)
        if waking and self.parent_conversation_id is not None:
            self.controller.notify_parent_subagent_work(self.parent_conversation_id)
        self._signal_activity()
        return True

    async def start(self) -> "None":
        await self.runtime.start()

    async def wait_until_settled(
        self,
        timeout_seconds: "typing.Optional[float]" = None,
    ) -> "None":
        if self._activity_event is None:
            self._activity_event = asyncio.Event()
        deadline = (
            time.monotonic() + timeout_seconds
            if timeout_seconds is not None and timeout_seconds > 0
            else None
        )
        while self.has_pending_work():
            self._activity_event.clear()
            if not self.has_pending_work():
                break
            wait_slice = 0.2
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("print timeout exceeded while waiting for turn completion")
                wait_slice = min(wait_slice, remaining)
            try:
                await asyncio.wait_for(self._activity_event.wait(), timeout=wait_slice)
            except asyncio.TimeoutError:
                pass
            if (
                self._pending_idle_notify
                and self.parent_conversation_id is not None
                and self.state != "killed"
                and not self.has_pending_work()
            ):
                self._pending_idle_notify = False
                reported = self._reported_to_parent
                self._reported_to_parent = False
                self.state = "idle"
                self.controller.notify_parent_subagent_idle(
                    self,
                    self._last_turn_output,
                    already_reported=reported,
                )
                self.controller.notify_parent_subagent_work(self.parent_conversation_id)

    async def run_user_prompt(
        self,
        prompt_text: "str",
        wait_background: "bool" = True,
        timeout_seconds: "typing.Optional[float]" = None,
    ) -> "SessionRunSummary":
        await self.start()
        start_ts = time.monotonic()
        before_outputs = len(self._turn_outputs)
        before_turns = self._completed_turns
        before_usage = dict(self._usage_totals)

        if timeout_seconds is not None and timeout_seconds > 0:
            await asyncio.wait_for(
                self.runtime.submit_user_turn(prompt_text),
                timeout=timeout_seconds,
            )
        else:
            await self.runtime.submit_user_turn(prompt_text)

        if wait_background:
            remaining_timeout: "typing.Optional[float]" = None
            if timeout_seconds is not None and timeout_seconds > 0:
                remaining_timeout = max(0.01, timeout_seconds - (time.monotonic() - start_ts))
            await self.wait_until_settled(timeout_seconds=remaining_timeout)

        self.save_session_state()
        new_outputs = self._turn_outputs[before_outputs:]
        final_resp = new_outputs[-1] if new_outputs else self._last_turn_output
        combined_resp = "\n".join(out.rstrip("\n") for out in new_outputs if out)
        if combined_resp:
            combined_resp += "\n"
        delta_usage = {
            k: self._usage_totals[k] - before_usage[k] for k in self._usage_totals
        }
        return SessionRunSummary(
            conversation_id=self.conversation_id,
            final_response=final_resp,
            combined_response=combined_resp,
            num_turns=max(1, self._completed_turns - before_turns),
            duration_seconds=time.monotonic() - start_ts,
            usage=delta_usage,
        )

    def save_session_state(self) -> "None":
        self.transcript_store.save_session_state(
            history=self.agent.history,
            model=self.agent.model_name,
            last_reported_model_display=self.context_manager.last_reported_model_display,
            step_counter=self.transcript_store.step_index,
            anchored_user_messages=self.context_manager.anchored_user_messages,
            emitted_post_user_messages=self.context_manager.emitted_post_user_messages,
            user_requests=self._user_requests,
            parent_messages=self._parent_messages,
            spawned_subagents=self._spawned_subagents,
            anchor_tokens=self.agent.anchor_tokens,
            anchor_history_len=self.agent.anchor_history_len,
        )

    async def close(self, delete_workspace: "bool" = False) -> "None":
        try:
            await self.runtime.close()
        finally:
            await self.task_manager.close()
            if delete_workspace and self._owned_workspace_dir is not None:
                if self._git_worktree_repo is not None:
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(self._git_worktree_repo),
                            "worktree",
                            "remove",
                            "--force",
                            str(self._owned_workspace_dir),
                        ],
                        capture_output=True,
                    )
                shutil.rmtree(self._owned_workspace_dir, ignore_errors=True)


class AgyMultiAgentController:
    """Manages root and subagent `AgySessionNode` conversations."""

    def __init__(
        self,
        app_data_dir: "Path",
        root_model_client: "AgyResponsesModelClient",
        context_config: "ContextConfig",
        compat_config: "typing.Optional[CompatServerConfig]" = None,
        global_config_dir: "typing.Optional[Path]" = None,
        config_effort: "typing.Optional[str]" = None,
        explicit_effort: "typing.Optional[str]" = None,
        search_handler: "typing.Optional[SearchHandler]" = None,
        clock: "typing.Optional[typing.Callable[[], datetime]]" = None,
    ) -> "None":
        self.app_data_dir = app_data_dir.expanduser()
        self._root_model_client = root_model_client
        self._context_config = context_config
        self._compat_config = compat_config
        self._global_config_dir = global_config_dir
        self._config_effort = config_effort
        self._explicit_effort = explicit_effort
        self._search_handler = search_handler
        self._clock = clock
        self._nodes: "typing.Dict[str, AgySessionNode]" = {}
        self._blueprints_by_conv: "typing.Dict[str, typing.Dict[str, SubagentBlueprint]]" = {}
        self._agents_manager: "typing.Optional[AgentsManager]" = None

    def register_node(self, node: "AgySessionNode") -> "None":
        self._nodes[node.conversation_id] = node
        self._blueprints_by_conv.setdefault(node.conversation_id, {})
        if node.parent_conversation_id is None and self._agents_manager is None:
            self._agents_manager = node.runtime.agents_manager

    def rekey_node(self, old_id: "str", new_id: "str", node: "AgySessionNode") -> "None":
        self._nodes.pop(old_id, None)
        self._nodes[new_id] = node
        old_bp = self._blueprints_by_conv.pop(old_id, {})
        merged_bp = dict(old_bp)
        merged_bp.update(self._blueprints_by_conv.get(new_id) or {})
        self._blueprints_by_conv[new_id] = merged_bp

    def get_node(self, conversation_id: "str") -> "AgySessionNode":
        node = self._nodes.get(conversation_id)
        if node is None:
            raise ValueError(f"conversation {conversation_id!r} not found")
        return node

    def define_subagent(
        self,
        caller_conversation_id: "str",
        name: "str",
        description: "str",
        system_prompt: "str",
        enable_write_tools: "bool" = False,
        enable_subagent_tools: "bool" = False,
        enable_mcp_tools: "bool" = False,
    ) -> "SubagentBlueprint":
        if name in ("self", "research", "image-generator"):
            raise ValueError(f"cannot redefine built-in subagent type {name!r}")
        bp = SubagentBlueprint(
            name=name,
            description=description,
            system_prompt=system_prompt,
            enable_write_tools=enable_write_tools,
            enable_subagent_tools=enable_subagent_tools,
            enable_mcp_tools=enable_mcp_tools,
        )
        self._blueprints_by_conv.setdefault(caller_conversation_id, {})[name] = bp
        return bp

    def _resolve_blueprint(
        self,
        caller_conversation_id: "str",
        type_name: "str",
    ) -> "typing.Optional[SubagentBlueprint]":
        curr: "typing.Optional[str]" = caller_conversation_id
        while curr is not None:
            table = self._blueprints_by_conv.get(curr) or {}
            if type_name in table:
                return table[type_name]
            node = self._nodes.get(curr)
            curr = node.parent_conversation_id if node is not None else None
        return None

    def _prepare_subagent_workspace(
        self,
        parent_cwd: "Path",
        sub_conv_id: "str",
        workspace_mode: "typing.Optional[str]",
    ) -> "typing.Tuple[Path, typing.Optional[Path], typing.Optional[Path]]":
        mode = (workspace_mode or "inherit").strip().lower() or "inherit"
        if mode == "inherit":
            return parent_cwd, None, None
        if mode not in ("branch", "share"):
            raise ValueError(f"unsupported Workspace mode: {workspace_mode!r}")

        target_ws = self.app_data_dir / "brain" / sub_conv_id / "workspace"
        target_ws.parent.mkdir(parents=True, exist_ok=True)
        repo_root = find_git_repo_root(parent_cwd)
        has_git = (repo_root / ".git").exists()

        if mode == "share" and has_git:
            proc = subprocess.run(
                ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(target_ws)],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                rel = parent_cwd.relative_to(repo_root)
                effective_cwd = (target_ws / rel).resolve()
                return effective_cwd, target_ws, repo_root

        shutil.copytree(
            parent_cwd,
            target_ws,
            symlinks=True,
            ignore=shutil.ignore_patterns(".git") if mode == "branch" and not has_git else None,
        )
        return target_ws.resolve(), target_ws.resolve(), None

    def _resolve_subagent_prompt_and_tools(
        self,
        caller_conversation_id: "str",
        type_name: "str",
    ) -> "typing.Tuple[typing.Optional[str], typing.Optional[typing.Sequence[str]]]":
        if type_name == "self":
            return None, None
        if type_name == "research":
            return None, RESEARCH_SUBAGENT_TOOLS
        if type_name == "image-generator":
            return None, IMAGE_GENERATOR_SUBAGENT_TOOLS

        bp = self._resolve_blueprint(caller_conversation_id, type_name)
        if bp is None:
            raise ValueError(f"unknown subagent type: {type_name!r}")

        allowed: "typing.Set[str]" = {
            "read_file",
            "search_web",
            "send_message",
            "schedule",
            "manage_task",
        }
        if bp.enable_write_tools:
            allowed.update({"write_file", "edit_file", "run_command"})
        if bp.enable_subagent_tools:
            allowed.update({"invoke_subagent", "define_subagent", "manage_subagents"})
        official_order, _ = load_official_declarations()
        ordered_tools = tuple(name for name in official_order if name in allowed)
        return bp.system_prompt, ordered_tools

    async def invoke_subagent(
        self,
        caller_conversation_id: "str",
        type_name: "str",
        role: "str",
        prompt: "str",
        model_tier: "typing.Optional[str]" = None,
        workspace_mode: "typing.Optional[str]" = None,
    ) -> "str":
        parent = self.get_node(caller_conversation_id)
        bp_prompt, toolset = self._resolve_subagent_prompt_and_tools(
            caller_conversation_id=caller_conversation_id,
            type_name=type_name,
        )
        resolved_model = resolve_subagent_model(model_tier, parent.agent.model_name)
        sub_conv_id = str(uuid.uuid4())
        norm_ws_mode = (workspace_mode or "inherit").strip().lower() or "inherit"
        sub_cwd, owned_ws, git_repo = self._prepare_subagent_workspace(
            parent_cwd=parent.cwd,
            sub_conv_id=sub_conv_id,
            workspace_mode=workspace_mode,
        )
        sub_additional_ws = parent.additional_workspaces if norm_ws_mode == "inherit" else ()
        caller_name = "parent" if parent.parent_conversation_id is None else parent.role
        sub_system_prompt = build_subagent_system_instruction(
            type_name=type_name,
            cwd=sub_cwd,
            conversation_id=sub_conv_id,
            caller_name=caller_name,
            caller_conversation_id=caller_conversation_id,
            app_data_dir=self.app_data_dir,
            additional_workspaces=sub_additional_ws,
            global_config_dir=self._global_config_dir,
            blueprint_system_prompt=bp_prompt,
        )
        sub_explicit_effort = (
            parent.explicit_effort if (model_tier or "inherit").strip().lower() in ("", "inherit") else None
        )
        sub_client = self._root_model_client.with_overrides(
            model=resolved_model,
            reasoning_effort=sub_explicit_effort,
            session_id=sub_conv_id,
        )
        sub_context_config = replace(
            self._context_config,
            model_auto_compact_token_limit=None,
            base_instructions_override=None,
            developer_instructions=None,
            user_instructions=None,
            extra_contextual_user_messages=(),
            include_permissions_instructions=False,
            include_skills_instructions=False,
            include_environment_context=False,
            code_mode_only=False,
        )
        slug = re.sub(r"[^a-z0-9_]+", "_", (role or type_name or "sub").lower()).strip("_") or "sub"
        short_id = sub_conv_id.replace("-", "")[:8]
        sub_agent_path = f"{parent.agent.agent_path}/sub_{slug[:24]}_{short_id}"
        sub_node = AgySessionNode(
            conversation_id=sub_conv_id,
            cwd=sub_cwd,
            app_data_dir=self.app_data_dir,
            model_client=sub_client,
            context_config=sub_context_config,
            controller=self,
            compat_config=self._compat_config,
            parent_conversation_id=caller_conversation_id,
            role=role,
            type_name=type_name,
            additional_workspaces=sub_additional_ws,
            global_config_dir=self._global_config_dir,
            custom_system_prompt=sub_system_prompt,
            toolset=toolset,
            config_effort=self._config_effort,
            explicit_effort=sub_explicit_effort,
            search_handler=self._search_handler,
            owned_workspace_dir=owned_ws,
            git_worktree_repo=git_repo,
            clock=self._clock,
            agents_manager=self._agents_manager,
            agent_path=sub_agent_path,
        )
        sub_node.transcript_store.ensure_dirs()
        log_uri = f"file://{sub_node.transcript_store.transcript_path}"
        ws_uri = f"file://{sub_cwd}"
        parent.record_spawned_subagent(
            SpawnedSubagentSummary(
                type_name=type_name,
                role=role,
                conversation_id=sub_conv_id,
                log_absolute_uri=log_uri,
                workspace_uris=(ws_uri,),
            )
        )
        self.register_node(sub_node)
        parent.subagent_work_tracker.notify_changed()
        if self._agents_manager is not None:
            self._agents_manager.notify_parents(sub_node.agent.agent_id)
        await sub_node.start()
        sub_node.deliver_system_message(
            content=prompt,
            sender=caller_conversation_id,
            priority="MESSAGE_PRIORITY_HIGH",
            wake_if_idle=True,
            visible="Parent",
        )
        return (
            "Created the following subagent:\n"
            "{\n"
            f'  "conversationId": {json.dumps(sub_conv_id)},\n'
            f'  "logAbsoluteUri": {json.dumps(log_uri)},\n'
            '  "workspaceUris": [\n'
            f'    {json.dumps(ws_uri)}\n'
            "  ]\n"
            "}\n"
            "The subagent will send you a message when it has completed its task or requires guidance. "
            "There is no need to poll for its response."
        )

    def iter_direct_children(
        self,
        parent_conversation_id: "str",
    ) -> "typing.List[AgySessionNode]":
        return [
            node
            for node in self._nodes.values()
            if node.parent_conversation_id == parent_conversation_id
        ]

    def notify_parent_subagent_work(
        self,
        parent_conversation_id: "typing.Optional[str]",
    ) -> "None":
        if parent_conversation_id is None:
            return
        parent = self._nodes.get(parent_conversation_id)
        if parent is not None and parent.state != "killed":
            parent.subagent_work_tracker.notify_changed()

    def has_active_subagents(self, parent_conversation_id: "str") -> "bool":
        return any(
            node.parent_conversation_id == parent_conversation_id and node.state == "running"
            for node in self._nodes.values()
        )

    def notify_parent_subagent_idle(
        self,
        sub_node: "AgySessionNode",
        final_text: "str",
        already_reported: "bool" = False,
    ) -> "None":
        if sub_node.parent_conversation_id is None:
            return
        parent = self._nodes.get(sub_node.parent_conversation_id)
        if parent is None or parent.state == "killed":
            return
        if already_reported:
            parent._signal_activity()
            return
        if final_text.strip():
            msg = (
                f"Subagent {sub_node.conversation_id} has gone idle. "
                f"This was the subagent's most recent response: {final_text}"
            )
        else:
            msg = (
                f"Subagent {sub_node.conversation_id} went idle without reporting anything back. "
                "Check its work, or send it a message at that id asking it to report back."
            )
        parent.deliver_system_message(
            content=msg,
            sender="system",
            priority="MESSAGE_PRIORITY_HIGH",
            wake_if_idle=True,
            visible="Subagent",
        )

    def _collect_descendants(self, root_sub_id: "str") -> "typing.List[AgySessionNode]":
        collected: "typing.List[AgySessionNode]" = []
        target = self._nodes.get(root_sub_id)
        if target is None:
            return collected
        collected.append(target)
        for child in list(self._nodes.values()):
            if child.parent_conversation_id == root_sub_id:
                collected.extend(self._collect_descendants(child.conversation_id))
        return collected

    async def manage_subagents(
        self,
        caller_conversation_id: "str",
        action: "str",
        conversation_ids: "typing.Optional[typing.Sequence[str]]" = None,
    ) -> "str":
        direct_children = [
            node
            for node in self._nodes.values()
            if node.parent_conversation_id == caller_conversation_id
        ]
        if action == "list":
            active_items = [
                {
                    "role": node.role,
                    "type": node.type_name,
                    "conversationId": node.conversation_id,
                    "transcript": f"file://{node.transcript_store.transcript_path}",
                    "state": "running",
                }
                for node in direct_children
                if node.state == "running"
            ]
            encoded = json.dumps(active_items, ensure_ascii=False, separators=(",", ":"))
            return f"You have {len(active_items)} active subagent(s):\n{encoded}"

        if action == "kill_all":
            targets = [node for node in direct_children if node.state != "killed"]
        elif action == "kill":
            by_id = {node.conversation_id: node for node in direct_children}
            targets = []
            for cid in conversation_ids or ():
                if cid not in by_id:
                    raise ValueError(f"direct subagent {cid!r} not found")
                targets.append(by_id[cid])
        else:
            raise ValueError(f"unsupported manage_subagents Action: {action!r}")

        to_kill: "typing.List[AgySessionNode]" = []
        seen_ids: "typing.Set[str]" = set()
        for target in targets:
            for desc in self._collect_descendants(target.conversation_id):
                if desc.conversation_id not in seen_ids and desc.state != "killed":
                    seen_ids.add(desc.conversation_id)
                    to_kill.append(desc)

        roles: "typing.List[str]" = []
        affected_parents: "typing.Set[str]" = set()
        for node in to_kill:
            if node.state == "running" and node.parent_conversation_id is not None:
                affected_parents.add(node.parent_conversation_id)
            node.state = "killed"
            roles.append(node.role)
        for pid in affected_parents:
            self.notify_parent_subagent_work(pid)
        await asyncio.gather(
            *(node.close(delete_workspace=True) for node in reversed(to_kill)),
            return_exceptions=True,
        )
        parent = self._nodes.get(caller_conversation_id)
        if parent is not None:
            parent._signal_activity()
        if not roles:
            return "Successfully killed 0 subagent(s) and their descendants.\nKilled roles: "
        return (
            f"Successfully killed {len(to_kill)} subagent(s) and their descendants.\n"
            f"Killed roles: {', '.join(roles)}"
        )

    async def send_agent_message(
        self,
        sender_conversation_id: "str",
        recipient_conversation_id: "str",
        content: "str",
    ) -> "None":
        target = self._nodes.get(recipient_conversation_id)
        if target is None or target.state == "killed":
            raise ValueError(f"recipient agent {recipient_conversation_id!r} not found or terminated")
        sender_node = self._nodes.get(sender_conversation_id)
        is_child_to_parent = (
            sender_node is not None
            and sender_node.parent_conversation_id == recipient_conversation_id
        )
        if is_child_to_parent:
            sender_node._reported_to_parent = True
            tag_label = "Subagent"
        elif target.parent_conversation_id == sender_conversation_id:
            tag_label = "Parent"
        else:
            tag_label = "Agent"
        await target.start()
        target.deliver_system_message(
            content=content,
            sender=sender_conversation_id,
            priority="MESSAGE_PRIORITY_HIGH",
            wake_if_idle=True,
            visible=tag_label,
        )

    async def close_all(self) -> "None":
        nodes = list(self._nodes.values())
        for node in nodes:
            if node.parent_conversation_id is not None:
                node.state = "killed"
        await asyncio.gather(
            *(node.close(delete_workspace=False) for node in reversed(nodes)),
            return_exceptions=True,
        )

    async def close_subagents(self) -> "None":
        sub_nodes = [
            node for node in self._nodes.values() if node.parent_conversation_id is not None
        ]
        affected_parents: "typing.Set[str]" = set()
        for node in sub_nodes:
            if node.state == "running" and node.parent_conversation_id is not None:
                affected_parents.add(node.parent_conversation_id)
            node.state = "killed"
        for pid in affected_parents:
            self.notify_parent_subagent_work(pid)
        if sub_nodes:
            await asyncio.gather(
                *(node.close(delete_workspace=False) for node in reversed(sub_nodes)),
                return_exceptions=True,
            )
