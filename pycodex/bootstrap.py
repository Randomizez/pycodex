import atexit
import os
import sys
import typing
from dataclasses import replace
from pathlib import Path

from .agent import Agent
from .context import ContextConfig
from .model import (
    DEFAULT_CODEX_CONFIG_PATH,
    ResponsesModelClient,
    ResponsesProviderConfig,
)
from .runtime import AgentRuntime
from .tools.runtime_services import (
    AgentRuntimeEnvironment,
    create_agent_runtime_environment,
)
from .utils import get_debug_dir, load_codex_dotenv, uuid7_string
from .utils.session_persist import rollout_path_for_session

LOCAL_RESPONSES_SERVER_API_KEY_ENV = "PYCODEX_LOCAL_RESPONSES_SERVER_KEY"
CLI_ORIGINATOR = "codex-tui"


def launch_chat_completion_compat_server(*args, **kwargs):
    from responses_server import (
        launch_chat_completion_compat_server as launch_compat_server,
    )

    return launch_compat_server(*args, **kwargs)


def _resolve_vllm_model(
    endpoint: "str",
    provider_config: "ResponsesProviderConfig",
    timeout_seconds: "float",
) -> "str":
    from responses_server import CompatServerConfig

    normalized = CompatServerConfig.from_base_url(endpoint)
    probe_config = replace(
        provider_config,
        provider_name="vllm",
        base_url=normalized.outcomming_base_url,
        api_key_env=None,
        query_params={},
        responses_lite_override=False,
    )
    probe_client = ResponsesModelClient(
        probe_config,
        timeout_seconds,
        originator=CLI_ORIGINATOR,
    )
    models = probe_client.list_models_sync()
    if not models:
        raise RuntimeError(
            "vLLM endpoint returned no models from "
            f"{normalized.outcomming_models_url()}"
        )
    return models[-1]


def configure_loguru() -> "None":
    try:
        from loguru import logger
    except ImportError:  # pragma: no cover - dependency may be absent in minimal envs
        return

    logger.remove()
    debug_dir = get_debug_dir()
    if debug_dir is not None:
        logger.add(str(debug_dir / "loguru.log"), level="DEBUG")
        return

    if os.environ.get("PYCODEX_DEBUG_STDERR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        logger.add(sys.stderr, level="DEBUG")


def get_tools(
    runtime_environment: "typing.Union[AgentRuntimeEnvironment, None]" = None,
    exec_mode: "bool" = False,
    toolset: "typing.Union[typing.Iterable[str], None]" = None,
):
    from .tools import (
        ApplyPatchTool,
        ClockManager,
        ClockTool,
        CloseAgentTool,
        CodeModeManager,
        ExecCommandTool,
        ExecTool,
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
        UnifiedExecManager,
        UpdatePlanTool,
        ViewImageTool,
        WaitAgentTool,
        WaitTool,
        WebSearchTool,
        WriteStdinTool,
    )

    runtime_environment = runtime_environment or create_agent_runtime_environment()
    registry = Registry(runtime_environment)
    code_mode_manager = CodeModeManager(registry)
    unified_exec_manager = UnifiedExecManager()
    clock_manager = ClockManager()
    exec_tool = ExecTool(code_mode_manager)
    wait_tool = WaitTool(code_mode_manager)
    web_search_tool = WebSearchTool()
    update_plan_tool = UpdatePlanTool(runtime_environment.plan_store)
    request_user_input_tool = RequestUserInputTool()
    request_permissions_tool = RequestPermissionsTool(
        runtime_environment.request_permissions_manager
    )
    spawn_agent_tool = SpawnAgentTool(runtime_environment.subagent_manager)
    send_input_tool = SendInputTool(runtime_environment.subagent_manager)
    resume_agent_tool = ResumeAgentTool(runtime_environment.subagent_manager)
    wait_agent_tool = WaitAgentTool(runtime_environment.subagent_manager)
    close_agent_tool = CloseAgentTool(runtime_environment.subagent_manager)
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
        spawn_agent_tool,
        send_input_tool,
        resume_agent_tool,
        wait_agent_tool,
        close_agent_tool,
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
            spawn_agent_tool,
            send_input_tool,
            resume_agent_tool,
            wait_agent_tool,
            close_agent_tool,
        )
    for tool in tools:
        registry.register(tool)
    return registry


def get_subagent_tools(
    runtime_environment: "typing.Union[AgentRuntimeEnvironment, None]" = None,
):
    from .tools import (
        ApplyPatchTool,
        ExecCommandTool,
        Registry,
        UnifiedExecManager,
        UpdatePlanTool,
        ViewImageTool,
        WebSearchTool,
        WriteStdinTool,
    )

    runtime_environment = runtime_environment or create_agent_runtime_environment()
    registry = Registry(runtime_environment)
    unified_exec_manager = UnifiedExecManager()
    registry.register(ExecCommandTool(unified_exec_manager))
    registry.register(WriteStdinTool(unified_exec_manager))
    registry.register(UpdatePlanTool(runtime_environment.plan_store))
    registry.register(ApplyPatchTool())
    registry.register(WebSearchTool())
    registry.register(ViewImageTool())
    return registry


def build_agent(
    client,
    config_path: "typing.Union[str, Path]" = DEFAULT_CODEX_CONFIG_PATH,
    profile: "typing.Union[str, None]" = None,
    system_prompt: "typing.Union[str, None]" = None,
    extra_contextual_user_messages: "typing.Iterable[str]" = (),
    cwd: "typing.Union[str, Path, None]" = None,
    toolset: "typing.Union[typing.Iterable[str], None]" = None,
) -> "Agent":
    config_path = str(config_path)
    resolved_cwd = Path(cwd or Path.cwd()).resolve()
    context_config = replace(
        ContextConfig.from_codex_config(config_path, profile),
        base_instructions_override=system_prompt,
        extra_contextual_user_messages=tuple(extra_contextual_user_messages),
        cwd=resolved_cwd,
    )
    runtime_environment = create_agent_runtime_environment()

    def make_subagent_runtime_builder(parent_agent):
        def build_subagent_runtime(
            model_override: "typing.Union[str, None]",
            reasoning_effort_override: "typing.Union[str, None]",
            initial_history=(),
            session_id: "typing.Union[str, None]" = None,
        ) -> "AgentRuntime":
            nested_client = parent_agent.model_client.with_overrides(
                model_override,
                reasoning_effort_override,
                session_id=session_id,
                openai_subagent="collab_spawn",
            )
            subagent_agent_runtime_environment = create_agent_runtime_environment()
            inherited_cwd = parent_agent.context_manager.cwd
            sub_agent = Agent(
                nested_client,
                get_subagent_tools(subagent_agent_runtime_environment),
                replace(context_config, cwd=inherited_cwd),
                initial_history=tuple(initial_history),
                session_id=session_id,
            )
            subagent_agent_runtime_environment.subagent_manager.set_runtime_builder(
                make_subagent_runtime_builder(sub_agent)
            )
            return AgentRuntime(sub_agent)

        return build_subagent_runtime

    session_id = uuid7_string()
    agent = Agent(
        client,
        get_tools(
            runtime_environment,
            exec_mode=True,
            toolset=toolset,
        ),
        context_config,
        session_file_path=rollout_path_for_session(
            context_config.codex_home, session_id
        ),
        session_id=session_id,
    )
    runtime_environment.subagent_manager.set_runtime_builder(
        make_subagent_runtime_builder(agent)
    )
    return agent


def build_model(
    config_path: "typing.Union[str, Path]" = DEFAULT_CODEX_CONFIG_PATH,
    profile: "typing.Union[str, None]" = None,
    timeout_seconds: "float" = 120.0,
    managed_responses_base_url: "typing.Union[str, None]" = None,
    vllm_endpoint: "typing.Union[str, None]" = None,
    use_chat_completion: "typing.Union[bool, None]" = None,
    use_messages: "bool" = False,
):
    load_codex_dotenv(config_path)
    provider_config = ResponsesProviderConfig.from_codex_config(
        config_path,
        profile,
    )
    if use_chat_completion is None:
        use_chat_completion = bool(provider_config.use_chat_completion)
    if use_chat_completion and use_messages:
        raise ValueError("--use-chat-completion and --use-messages cannot be combined")
    if vllm_endpoint and use_messages:
        raise ValueError("--vllm-endpoint and --use-messages cannot be combined")
    uses_local_responses_compat = (
        managed_responses_base_url is not None
        or vllm_endpoint is not None
        or bool(use_chat_completion)
        or use_messages
    )
    if vllm_endpoint is not None:
        provider_config = replace(
            provider_config,
            model=_resolve_vllm_model(
                vllm_endpoint,
                provider_config,
                timeout_seconds,
            ),
        )
    url, key_env = provider_config.base_url, provider_config.api_key_env
    if managed_responses_base_url is not None:
        url, key_env = (
            managed_responses_base_url,
            LOCAL_RESPONSES_SERVER_API_KEY_ENV,
        )
        os.environ.setdefault(LOCAL_RESPONSES_SERVER_API_KEY_ENV, "dummy")
    elif vllm_endpoint or use_chat_completion or use_messages:
        if vllm_endpoint:
            managed_server = launch_chat_completion_compat_server(
                vllm_endpoint,
                model_provider="vllm",
            )
        else:
            managed_server = launch_chat_completion_compat_server(
                provider_config.base_url,
                provider_config.api_key_env,
                model_provider=provider_config.provider_name,
                outcomming_api=("messages" if use_messages else "chat_completions"),
            )
        atexit.register(managed_server.stop)
        url, key_env = (
            managed_server.base_url,
            LOCAL_RESPONSES_SERVER_API_KEY_ENV,
        )
        os.environ.setdefault(LOCAL_RESPONSES_SERVER_API_KEY_ENV, "dummy")

    provider_config = replace(
        provider_config,
        base_url=url,
        api_key_env=key_env,
        responses_lite_override=(
            False
            if uses_local_responses_compat
            else provider_config.responses_lite_override
        ),
    )
    return ResponsesModelClient(
        provider_config,
        timeout_seconds,
        originator=CLI_ORIGINATOR,
    )


def build_runtime(agent: "Agent") -> "AgentRuntime":
    return AgentRuntime(agent)
