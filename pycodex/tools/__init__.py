"""Tool package for the Python Codex prototype.

This package groups the local tool abstractions and concrete tool
implementations that back `pycodex`.
"""

import typing

from .apply_patch_tool import ApplyPatchTool
from .base_tool import BaseTool, Registry, ToolContext, ToolRegistry
from .clock_manager import ClockManager
from .clock_tool import ClockTool
from .code_mode_manager import CodeModeManager
from .exec_command_tool import ExecCommandTool
from .exec_tool import ExecTool
from .followup_task_tool import FollowupTaskTool
from .goal_tool import CreateGoalTool, GetGoalTool, UpdateGoalTool
from .grep_files_tool import GrepFilesTool
from .interrupt_agent_tool import InterruptAgentTool
from .list_agents_tool import ListAgentsTool
from .list_dir_tool import ListDirTool
from .read_file_tool import ReadFileTool
from .request_permissions_tool import RequestPermissionsTool
from .request_user_input_tool import RequestUserInputTool
from .runtime_services import (
    AgentRuntimeEnvironment,
    create_agent_runtime_environment,
)
from .send_message_tool import SendMessageTool
from .shell_command_tool import ShellCommandTool
from .shell_tool import ShellTool
from .spawn_agent_tool import SpawnAgentTool
from .unified_exec_manager import UnifiedExecManager
from .update_plan_tool import UpdatePlanTool
from .view_image_tool import ViewImageTool
from .wait_agent_tool import WaitAgentTool
from .wait_tool import WaitTool
from .web_search_tool import WebSearchTool
from .write_stdin_tool import WriteStdinTool

if typing.TYPE_CHECKING:
    from ..multi_agents import AgentsManager


def get_tools(
    agents_manager: "AgentsManager",
    runtime_environment: "typing.Union[AgentRuntimeEnvironment, None]" = None,
    toolset: "typing.Union[typing.Iterable[str], None]" = None,
):
    runtime_environment = runtime_environment or create_agent_runtime_environment()
    registry = Registry(runtime_environment)
    code_mode_manager = CodeModeManager(registry)
    unified_exec_manager = UnifiedExecManager()
    clock_manager = runtime_environment.clock_manager
    exec_tool = ExecTool(code_mode_manager)
    wait_tool = WaitTool(code_mode_manager)
    goal_tools = (
        CreateGoalTool(runtime_environment.goal_manager),
        GetGoalTool(runtime_environment.goal_manager),
        UpdateGoalTool(runtime_environment.goal_manager),
    )
    request_user_input_tool = RequestUserInputTool()
    agent_tools = (
        SpawnAgentTool(agents_manager),
        SendMessageTool(agents_manager),
        FollowupTaskTool(agents_manager),
        WaitAgentTool(agents_manager),
        InterruptAgentTool(agents_manager),
        ListAgentsTool(agents_manager),
    )
    apply_patch_tool = ApplyPatchTool()
    exec_command_tool = ExecCommandTool(unified_exec_manager)
    write_stdin_tool = WriteStdinTool(unified_exec_manager)
    clock_tool = ClockTool(clock_manager)
    view_image_tool = ViewImageTool()
    tools = (
        exec_tool,
        wait_tool,
        exec_command_tool,
        write_stdin_tool,
        clock_tool,
        request_user_input_tool,
        apply_patch_tool,
        view_image_tool,
    ) + goal_tools + agent_tools
    if toolset is not None:
        available_tools = {tool.name: tool for tool in tools}
        available_tools.update({
            f"{tool.namespace}.{tool.name}": tool for tool in tools
            if tool.namespace not in (None, "", "functions")
        })
        toolset = tuple(toolset)
        unknown_tools = set(toolset) - set(available_tools)
        if unknown_tools:
            raise ValueError(
                "unknown toolset entries: {0}".format(", ".join(sorted(unknown_tools)))
            )
        tools = tuple(available_tools[name] for name in toolset)
    if tools:
        tools = (exec_tool, wait_tool) + tuple(
            tool for tool in tools if tool not in (exec_tool, wait_tool)
        )
    for tool in tools:
        registry.register(tool)
    for tool in registry.tools():
        if tool is exec_tool:
            runtime_environment.register_resource(
                tool.name, code_mode_manager,
                restart=code_mode_manager.restart, close=code_mode_manager.close,
            )
        elif tool is exec_command_tool:
            runtime_environment.register_resource(tool.name, unified_exec_manager)
        elif tool is clock_tool:
            runtime_environment.register_resource(
                tool.name, clock_manager, observer=clock_manager.handle_agent_event,
            )
    return registry


__all__ = [
    "ApplyPatchTool",
    "BaseTool",
    "ClockManager",
    "ClockTool",
    "CodeModeManager",
    "CreateGoalTool",
    "ExecTool",
    "FollowupTaskTool",
    "get_tools",
    "GetGoalTool",
    "ExecCommandTool",
    "GrepFilesTool",
    "InterruptAgentTool",
    "ListAgentsTool",
    "ListDirTool",
    "ReadFileTool",
    "Registry",
    "RequestPermissionsTool",
    "RequestUserInputTool",
    "SendMessageTool",
    "ShellCommandTool",
    "ShellTool",
    "SpawnAgentTool",
    "ToolContext",
    "ToolRegistry",
    "UnifiedExecManager",
    "UpdateGoalTool",
    "UpdatePlanTool",
    "ViewImageTool",
    "WaitAgentTool",
    "WaitTool",
    "WebSearchTool",
    "WriteStdinTool",
]
