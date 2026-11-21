"""AGY 14-declaration tool registry and 11 active + 3 disabled/placeholder tools."""

import asyncio
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import typing

from pycodex.protocol import JSONDict, JSONValue, ToolCall, ToolResult
from pycodex.tools.base_tool import BaseTool, StructuredToolOutput, ToolContext, ToolRegistry
from pycodex.tools.runtime_services import AgentRuntimeEnvironment

from ..context import format_local_iso

if typing.TYPE_CHECKING:
    from responses_server.config import CompatServerConfig
    from ..multi_agents import AgyMultiAgentController
    from ..task_manager import AgyTaskManager
    from ..transcript import AgyTranscriptStore


TOOL_DECLARATIONS_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "tool_declarations.json"
)

RESEARCH_SUBAGENT_TOOLS: "typing.Tuple[str, ...]" = (
    "read_file",
    "search_web",
    "run_command",
    "send_message",
    "schedule",
    "manage_task",
)

IMAGE_GENERATOR_SUBAGENT_TOOLS: "typing.Tuple[str, ...]" = (
    "read_file",
    "write_file",
    "run_command",
    "send_message",
    "schedule",
    "manage_task",
)


class AgyToolError(Exception):
    """Raised when an AGY tool encounters a validation or execution error."""


def load_official_declarations() -> "typing.Tuple[typing.Tuple[str, ...], typing.Dict[str, JSONDict]]":
    raw_list = json.loads(TOOL_DECLARATIONS_PATH.read_text(encoding="utf-8"))
    order: "typing.List[str]" = []
    by_name: "typing.Dict[str, JSONDict]" = {}
    for entry in raw_list:
        name = str(entry["name"])
        order.append(name)
        by_name[name] = deepcopy(entry)
    return tuple(order), by_name


def format_sync_tool_header(
    created_at: "str",
    completed_at: "str",
) -> "str":
    return f"Created At: {created_at}\nCompleted At: {completed_at}\n"


class AgyBaseTool(BaseTool):
    """BaseTool wired to exact agy 1.5.13 description & parametersJsonSchema."""

    exec_only = False
    supports_parallel = True

    def __init__(
        self,
        declaration: "JSONDict",
        registry: "AgyToolRegistry",
    ) -> "None":
        self.name = str(declaration["name"])
        self.description = str(declaration["description"])
        self.input_schema = deepcopy(declaration["parametersJsonSchema"])
        self._registry = registry

    def now_local_iso(self) -> "str":
        return format_local_iso(self._registry.now())

    def step_index_for_call(self, context: "ToolContext") -> "int":
        return self._registry.step_index_for_call(context.call_id)

    @staticmethod
    def require_dict_args(args: "JSONValue") -> "JSONDict":
        if not isinstance(args, dict):
            raise AgyToolError("tool arguments must be a JSON object")
        return args


class AgyRuntimeEnvironment(AgentRuntimeEnvironment):
    """Rearm saved schedules on the session worker's event loop."""

    def __init__(self, task_manager: "AgyTaskManager") -> None:
        super().__init__()
        self.task_manager = task_manager

    def start(self) -> None:
        super().start()
        self.task_manager.restore_schedules()


class AgyToolRegistry(ToolRegistry):
    """ToolRegistry preserving exact AGY tool schemas, step indices, and error strings."""

    def __init__(
        self,
        transcript_store: "AgyTranscriptStore",
        runtime_environment: "typing.Optional[AgentRuntimeEnvironment]" = None,
        clock: "typing.Optional[typing.Callable[[], datetime]]" = None,
    ) -> "None":
        super().__init__(runtime_environment=runtime_environment)
        self.transcript_store = transcript_store
        self._clock = clock
        self._call_step_indices: "typing.Dict[str, int]" = {}

    def now(self) -> "typing.Optional[datetime]":
        return self._clock() if self._clock is not None else None

    def step_index_for_call(self, call_id: "typing.Optional[str]") -> "int":
        if call_id is not None and call_id in self._call_step_indices:
            return self._call_step_indices[call_id]
        return self.transcript_store.allocate_step_index()

    @staticmethod
    def _format_error_message(raw_msg: "str") -> "str":
        prefix = "Encountered error in tool execution: "
        return raw_msg if raw_msg.startswith(prefix) else f"{prefix}{raw_msg}"

    async def execute(self, call: "ToolCall", context: "ToolContext") -> "ToolResult":
        step_idx = self.transcript_store.allocate_step_index()
        self._call_step_indices[call.call_id] = step_idx
        tool = self._tools.get(call.name)
        if tool is None:
            self._call_step_indices.pop(call.call_id, None)
            self.runtime_environment.tool_completed()
            err_out = self._format_error_message(f"unknown tool: {call.name}")
            self.transcript_store.append_transcript_step(
                source="SYSTEM",
                step_type="TOOL_RESPONSE",
                content=err_out,
                status="ERROR",
                step_index=step_idx,
            )
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output=err_out,
                is_error=True,
                success=False,
                tool_type=call.tool_type,
            )

        try:
            raw_output = await tool.run(
                ToolContext(
                    turn_id=context.turn_id,
                    history=context.history,
                    cwd=context.cwd,
                    agent_id=context.agent_id,
                    encrypted_messages=context.encrypted_messages,
                    call_id=call.call_id,
                    session_id=context.session_id,
                ),
                call.arguments,
            )
            content_items = None
            is_err = False
            success_val: "typing.Optional[bool]" = True
            if isinstance(raw_output, StructuredToolOutput):
                content_items = raw_output.content_items
                success_val = raw_output.success
                is_err = raw_output.success is False
                output_val = raw_output.output
            else:
                output_val = raw_output
            if is_err and isinstance(output_val, str):
                output_val = self._format_error_message(output_val)
            transcript_content = (
                json.dumps(output_val, ensure_ascii=False)
                if isinstance(output_val, (dict, list))
                else str(output_val)
            )
            self.transcript_store.append_transcript_step(
                source="SYSTEM",
                step_type="TOOL_RESPONSE",
                content=transcript_content,
                status="ERROR" if is_err else "DONE",
                step_index=step_idx,
            )
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output=output_val,
                is_error=is_err,
                success=success_val,
                content_items=content_items,
                tool_type=call.tool_type,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            err_out = self._format_error_message(str(exc))
            self.transcript_store.append_transcript_step(
                source="SYSTEM",
                step_type="TOOL_RESPONSE",
                content=err_out,
                status="ERROR",
                step_index=step_idx,
            )
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output=err_out,
                is_error=True,
                success=False,
                tool_type=call.tool_type,
            )
        finally:
            self._call_step_indices.pop(call.call_id, None)
            self.runtime_environment.tool_completed()


SearchHandler = typing.Callable[
    [str, typing.Optional[str]],
    typing.Awaitable[str],
]


def build_agy_tool_registry(
    conversation_id: "str",
    transcript_store: "AgyTranscriptStore",
    task_manager: "AgyTaskManager",
    multi_agents: "AgyMultiAgentController",
    compat_config: "typing.Optional[CompatServerConfig]" = None,
    api_model_getter: "typing.Optional[typing.Callable[[], str]]" = None,
    toolset: "typing.Optional[typing.Sequence[str]]" = None,
    search_handler: "typing.Optional[SearchHandler]" = None,
    runtime_environment: "typing.Optional[AgentRuntimeEnvironment]" = None,
    clock: "typing.Optional[typing.Callable[[], datetime]]" = None,
    extra_tools: "typing.Sequence[BaseTool]" = (),
) -> "AgyToolRegistry":
    from .agent_tools import (
        DefineSubagentTool,
        InvokeSubagentTool,
        ManageSubagentsTool,
        SendMessageTool,
    )
    from .command_tools import ManageTaskTool, RunCommandTool, ScheduleTool
    from .file_tools import EditFileTool, ReadFileTool, WriteFileTool
    from .interaction_tools import (
        AskQuestionPlaceholderTool,
        RunWorkflowPlaceholderTool,
        SearchWebTool,
        SwarmPlaceholderTool,
    )

    official_order, by_name = load_official_declarations()
    registry = AgyToolRegistry(
        transcript_store=transcript_store,
        runtime_environment=runtime_environment or AgyRuntimeEnvironment(task_manager),
        clock=clock,
    )
    all_tools: "typing.Dict[str, AgyBaseTool]" = {
        "read_file": ReadFileTool(by_name["read_file"], registry),
        "run_command": RunCommandTool(by_name["run_command"], registry, task_manager),
        "manage_task": ManageTaskTool(by_name["manage_task"], registry, task_manager),
        "send_message": SendMessageTool(
            by_name["send_message"], registry, conversation_id, multi_agents
        ),
        "schedule": ScheduleTool(by_name["schedule"], registry, task_manager),
        "invoke_subagent": InvokeSubagentTool(
            by_name["invoke_subagent"], registry, conversation_id, multi_agents
        ),
        "define_subagent": DefineSubagentTool(
            by_name["define_subagent"], registry, conversation_id, multi_agents
        ),
        "manage_subagents": ManageSubagentsTool(
            by_name["manage_subagents"], registry, conversation_id, multi_agents
        ),
        "write_file": WriteFileTool(by_name["write_file"], registry),
        "edit_file": EditFileTool(by_name["edit_file"], registry),
        "search_web": SearchWebTool(
            by_name["search_web"],
            registry,
            compat_config=compat_config,
            api_model_getter=api_model_getter,
            search_handler=search_handler,
        ),
        "run_workflow": RunWorkflowPlaceholderTool(by_name["run_workflow"], registry),
        "swarm": SwarmPlaceholderTool(by_name["swarm"], registry),
        "ask_question": AskQuestionPlaceholderTool(by_name["ask_question"], registry),
    }
    selected_names = tuple(toolset) if toolset is not None else official_order
    for name in selected_names:
        if name not in all_tools:
            raise ValueError(f"unknown AGY tool name: {name!r}")
        registry.register(all_tools[name])
    for extra_tool in extra_tools:
        registry.register(extra_tool)
    registry.runtime_environment.register_resource(
        "run_command",
        task_manager,
        restart=task_manager.restart,
        close=task_manager.close,
    )
    return registry
