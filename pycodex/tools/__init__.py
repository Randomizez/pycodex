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
    exec_mode: "bool" = False,
    toolset: "typing.Union[typing.Iterable[str], None]" = None,
):
    runtime_environment = runtime_environment or create_agent_runtime_environment()
    registry = Registry(runtime_environment)
    code_mode_manager = CodeModeManager(registry)
    unified_exec_manager = UnifiedExecManager()
    clock_manager = runtime_environment.clock_manager
    exec_tool = ExecTool(code_mode_manager)
    wait_tool = WaitTool(code_mode_manager)
    web_search_tool = WebSearchTool()
    update_plan_tool = UpdatePlanTool(runtime_environment.plan_store)
    request_user_input_tool = RequestUserInputTool()
    request_permissions_tool = RequestPermissionsTool(
        runtime_environment.request_permissions_manager
    )
    agent_tools = (
        SpawnAgentTool(agents_manager),
        SendMessageTool(agents_manager),
        FollowupTaskTool(agents_manager),
        WaitAgentTool(agents_manager),
        InterruptAgentTool(agents_manager),
        ListAgentsTool(agents_manager),
    )
    apply_patch_tool = ApplyPatchTool()
    shell_tool = ShellTool()
    shell_command_tool = ShellCommandTool()
    exec_command_tool = ExecCommandTool(unified_exec_manager)
    write_stdin_tool = WriteStdinTool(unified_exec_manager)
    clock_tool = ClockTool(clock_manager)
    grep_files_tool = GrepFilesTool()
    read_file_tool = ReadFileTool()
    list_dir_tool = ListDirTool()
    view_image_tool = ViewImageTool()
    tools = (
        shell_tool,
        shell_command_tool,
        exec_command_tool,
        write_stdin_tool,
        clock_tool,
        exec_tool,
        wait_tool,
        web_search_tool,
        update_plan_tool,
        request_user_input_tool,
        request_permissions_tool,
    ) + agent_tools + (
        apply_patch_tool,
        grep_files_tool,
        read_file_tool,
        list_dir_tool,
        view_image_tool,
    )
    if toolset is not None:
        available_tools = {tool.name: tool for tool in tools}
        toolset = tuple(toolset)
        unknown_tools = set(toolset) - set(available_tools)
        if unknown_tools:
            raise ValueError(
                "unknown toolset entries: {0}".format(", ".join(sorted(unknown_tools)))
            )
        tools = tuple(available_tools[name] for name in toolset)
    elif exec_mode:
        tools = (
            exec_command_tool,
            write_stdin_tool,
            clock_tool,
            update_plan_tool,
            request_user_input_tool,
            apply_patch_tool,
            web_search_tool,
            view_image_tool,
        ) + agent_tools
    for tool in tools:
        registry.register(tool)
    return registry


__all__ = [
    "ApplyPatchTool",
    "BaseTool",
    "ClockManager",
    "ClockTool",
    "CodeModeManager",
    "ExecTool",
    "FollowupTaskTool",
    "get_tools",
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
    "UpdatePlanTool",
    "ViewImageTool",
    "WaitAgentTool",
    "WaitTool",
    "WebSearchTool",
    "WriteStdinTool",
]
