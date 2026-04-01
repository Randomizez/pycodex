from .agent import AgentLoop
from .context import ContextConfig, ContextManager
from .model import (
    ModelClient,
    NOOP_MODEL_STREAM_EVENT_HANDLER,
    ResponsesApiError,
    ResponsesModelClient,
    ResponsesProviderConfig,
)
from .protocol import (
    AgentEvent,
    AssistantMessage,
    ContextMessage,
    ModelResponse,
    ModelStreamEvent,
    Prompt,
    ReasoningItem,
    Submission,
    ToolCall,
    ToolResult,
    ToolSpec,
    TurnResult,
    UserMessage,
)
from .runtime import AgentRuntime
from .runtime_services import (
    PlanStore,
    RequestPermissionsManager,
    RequestUserInputManager,
    SubAgentManager,
    get_runtime_environment,
)
from .tools import (
    ApplyPatchTool,
    BaseTool,
    CloseAgentTool,
    CodeModeManager,
    ExecTool,
    ExecCommandTool,
    GrepFilesTool,
    ListDirTool,
    ReadFileTool,
    Registry,
    RequestPermissionsTool,
    RequestUserInputTool,
    ResumeAgentTool,
    SendInputTool,
    ShellCommandTool,
    ShellTool,
    SpawnAgentTool,
    ToolContext,
    ToolRegistry,
    UnifiedExecManager,
    UpdatePlanTool,
    ViewImageTool,
    WaitAgentTool,
    WaitTool,
    WebSearchTool,
    WriteStdinTool,
)

def debug(stop: bool = False):

    import socket

    import debugpy

    if debugpy.is_client_connected():
        return
    try:
        host = socket.gethostbyname(socket.getfqdn(socket.gethostname()))
        port = 5000

        debugpy.listen((host, port), in_process_debug_adapter=False)
        print("Waiting for debugger...\a")
        print(f"ip: {host} port:{port}", flush=True)
        debugpy.wait_for_client()
        print("Connected.")
        if stop:
            debugpy.breakpoint()
    except Exception as e:
        import traceback

        print("\n".join(traceback.format_exception(e)))

__all__ = [
    "AgentEvent",
    "AgentLoop",
    "AgentRuntime",
    "ApplyPatchTool",
    "AssistantMessage",
    "BaseTool",
    "CloseAgentTool",
    "CodeModeManager",
    "ContextConfig",
    "ContextManager",
    "ContextMessage",
    "ExecTool",
    "ExecCommandTool",
    "GrepFilesTool",
    "ListDirTool",
    "ModelClient",
    "NOOP_MODEL_STREAM_EVENT_HANDLER",
    "Registry",
    "ReadFileTool",
    "RequestPermissionsTool",
    "RequestUserInputTool",
    "ModelResponse",
    "ModelStreamEvent",
    "PlanStore",
    "Prompt",
    "ReasoningItem",
    "RequestPermissionsManager",
    "RequestUserInputManager",
    "ResumeAgentTool",
    "ResponsesApiError",
    "ResponsesModelClient",
    "ResponsesProviderConfig",
    "SendInputTool",
    "ShellCommandTool",
    "ShellTool",
    "SpawnAgentTool",
    "Submission",
    "SubAgentManager",
    "ToolCall",
    "ToolContext",
    "ToolRegistry",
    "UnifiedExecManager",
    "UpdatePlanTool",
    "ToolResult",
    "ToolSpec",
    "TurnResult",
    "UserMessage",
    "ViewImageTool",
    "WaitAgentTool",
    "WaitTool",
    "WebSearchTool",
    "WriteStdinTool",
    "get_runtime_environment",
]
