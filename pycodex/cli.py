
import atexit
import argparse
import asyncio
import os
import shlex
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .agent import Agent
from .collaboration import DEFAULT_COLLABORATION_MODE, CollaborationMode
from .compat import Literal
from .context import ContextManager
from .model import DEFAULT_CODEX_CONFIG_PATH, ResponsesModelClient, ResponsesProviderConfig
from .portable import bootstrap_called_home, upload_codex_home
from .runtime import CliSubmissionQueue
from .runtime_services import AgentRuntimeEnvironment, create_agent_runtime_environment
from .utils import CliSessionView, get_debug_dir, load_codex_dotenv, uuid7_string
from .interactive_session import (
    EXTRA_COMMANDS_LINE,
    format_turn_output,
    run_interactive_session as _run_interactive_session,
    prompt_request_permissions,
    prompt_request_user_input,
)
from .utils.session_persist import (
    SessionRolloutRecorder,
    resolve_codex_home,
)
import typing

CliSessionMode = Literal["exec", "tui"]
LOCAL_RESPONSES_SERVER_API_KEY_ENV = "PYCODEX_LOCAL_RESPONSES_SERVER_KEY"
CLI_ORIGINATOR = "codex-tui"


def launch_chat_completion_compat_server(*args, **kwargs):
    from responses_server import (
        launch_chat_completion_compat_server as launch_compat_server,
    )

    return launch_compat_server(*args, **kwargs)


def configure_loguru() -> 'None':
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


def build_parser() -> 'argparse.ArgumentParser':
    parser = argparse.ArgumentParser(
        prog="pycodex",
        description="Minimal Codex-style local CLI backed by ~/.codex/config.toml.",
    )
    parser.add_argument(
        "prompt", nargs="*", help="Prompt text. If omitted, read from stdin."
    )
    parser.add_argument(
        "--put",
        default=None,
        metavar="PATH@SERVER",
        help=(
            "Upload a Codex home using `--put @host:port` or "
            "`--put /path/.codex@host:port`."
        ),
    )
    parser.add_argument(
        "--call",
        default=None,
        help=(
            "Download and use a stored Codex home via <secret>-<call_id>@<host:port>."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CODEX_CONFIG_PATH),
        help="Path to Codex config.toml.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional profile name from config.toml.",
    )
    parser.add_argument(
        "--vllm-endpoint",
        default=None,
        help=(
            "Optional base URL for a chat-completions-backed vLLM server. "
            "When set, pycodex starts a local responses compat server for this "
            "session and appends /v1 if the path is empty."
        ),
    )
    parser.add_argument(
        "--use-chat-completion",
        default=False,
        action="store_true",
        help=(
            "When set, pycodex starts a local responses compat server for this session."
        ),
    )
    parser.add_argument(
        "--use-messages",
        default=False,
        action="store_true",
        help=(
            "When set, pycodex starts a local responses compat server and routes "
            "to a downstream /v1/messages backend for this session."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional base instructions override passed to the model.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="HTTP timeout for one model call.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full TurnResult as JSON.",
    )
    return parser


def should_run_interactive(prompt_parts: 'Sequence[str]', stdin_is_tty: 'bool') -> 'bool':
    return not prompt_parts and stdin_is_tty


def resolve_prompt_text(prompt_parts: 'Sequence[str]') -> 'str':
    if prompt_parts:
        return " ".join(prompt_parts).strip()

    if not sys.stdin.isatty():
        prompt_text = sys.stdin.read().strip()
        if prompt_text:
            return prompt_text

    raise ValueError("prompt is required either as argv text or stdin")


def get_tools(
    runtime_environment: 'typing.Union[AgentRuntimeEnvironment, None]' = None,
    exec_mode: 'bool' = False,
):
    from .tools import (
        ApplyPatchTool,
        CloseAgentTool,
        CodeModeManager,
        ExecTool,
        ExecCommandTool,
        GrepFilesTool,
        ListDirTool,
        ReadFileTool,
        RequestPermissionsTool,
        RequestUserInputTool,
        ResumeAgentTool,
        Registry,
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
    registry = Registry()
    code_mode_manager = CodeModeManager(registry)
    unified_exec_manager = UnifiedExecManager()
    exec_tool = ExecTool(code_mode_manager)
    wait_tool = WaitTool(code_mode_manager)
    web_search_tool = WebSearchTool()
    update_plan_tool = UpdatePlanTool(runtime_environment.plan_store)
    request_user_input_tool = RequestUserInputTool(
        runtime_environment.request_user_input_manager
    )
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
    grep_files_tool = GrepFilesTool()
    read_file_tool = ReadFileTool()
    list_dir_tool = ListDirTool()
    view_image_tool = ViewImageTool()
    if exec_mode:
        registry.register(exec_command_tool)
        registry.register(write_stdin_tool)
        registry.register(update_plan_tool)
        registry.register(request_user_input_tool)
        registry.register(apply_patch_tool)
        registry.register(web_search_tool)
        registry.register(view_image_tool)
        registry.register(spawn_agent_tool)
        registry.register(send_input_tool)
        registry.register(resume_agent_tool)
        registry.register(wait_agent_tool)
        registry.register(close_agent_tool)
        return registry

    registry.register(shell_tool)
    registry.register(shell_command_tool)
    registry.register(exec_command_tool)
    registry.register(write_stdin_tool)
    registry.register(exec_tool)
    registry.register(wait_tool)
    registry.register(web_search_tool)
    registry.register(update_plan_tool)
    registry.register(request_user_input_tool)
    registry.register(request_permissions_tool)
    registry.register(spawn_agent_tool)
    registry.register(send_input_tool)
    registry.register(resume_agent_tool)
    registry.register(wait_agent_tool)
    registry.register(close_agent_tool)
    registry.register(apply_patch_tool)
    registry.register(grep_files_tool)
    registry.register(read_file_tool)
    registry.register(list_dir_tool)
    registry.register(view_image_tool)
    return registry


def get_subagent_tools(runtime_environment: 'typing.Union[AgentRuntimeEnvironment, None]' = None):
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
    registry = Registry()
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
    config_path: 'typing.Union[str, Path]' = DEFAULT_CODEX_CONFIG_PATH,
    profile: 'typing.Union[str, None]' = None,
    system_prompt: 'typing.Union[str, None]' = None,
    session_mode: 'CliSessionMode' = "exec",
    collaboration_mode: 'CollaborationMode' = DEFAULT_COLLABORATION_MODE,
    extra_contextual_user_messages: 'typing.Iterable[str]' = (),
) -> 'Agent':
    config_path = str(config_path)
    context_manager = ContextManager.from_codex_config(
        config_path,
        profile,
        base_instructions_override=system_prompt,
        collaboration_mode=collaboration_mode,
        include_collaboration_instructions=session_mode == "tui",
        extra_contextual_user_messages=extra_contextual_user_messages,
    )
    session_id = getattr(client, "_session_id", None) or uuid7_string()
    if hasattr(client, "_session_id"):
        client._session_id = session_id
    subagent_context_manager = ContextManager.from_codex_config(
        config_path,
        profile,
        base_instructions_override=system_prompt,
        include_collaboration_instructions=False,
        extra_contextual_user_messages=extra_contextual_user_messages,
    )
    runtime_environment = create_agent_runtime_environment()
    runtime_environment.request_user_input_manager.set_handler(None)
    runtime_environment.request_permissions_manager.set_handler(None)
    rollout_recorder = SessionRolloutRecorder.create(
        resolve_codex_home(config_path),
        session_id,
        context_manager.cwd,
        getattr(client, "_originator", CLI_ORIGINATOR),
        getattr(getattr(client, "_config", None), "provider_name", None),
        context_manager.resolve_base_instructions(),
    )

    def make_subagent_queue_builder(base_client):
        def build_subagent_queue(
            model_override: 'typing.Union[str, None]',
            reasoning_effort_override: 'typing.Union[str, None]',
            initial_history=(),
            session_id: 'typing.Union[str, None]' = None,
        ) -> 'CliSubmissionQueue':
            nested_client = base_client.with_overrides(
                model_override,
                reasoning_effort_override,
                session_id=session_id,
                openai_subagent="collab_spawn",
            )
            subagent_agent_runtime_environment = create_agent_runtime_environment()
            subagent_agent_runtime_environment.request_user_input_manager.set_handler(None)
            subagent_agent_runtime_environment.request_permissions_manager.set_handler(None)
            subagent_agent_runtime_environment.subagent_manager.set_queue_builder(
                make_subagent_queue_builder(nested_client)
            )
            sub_agent = Agent(
                nested_client,
                get_subagent_tools(subagent_agent_runtime_environment),
                subagent_context_manager,
                initial_history=tuple(initial_history),
                runtime_environment=subagent_agent_runtime_environment,
            )
            return CliSubmissionQueue(sub_agent)

        return build_subagent_queue

    runtime_environment.subagent_manager.set_queue_builder(
        make_subagent_queue_builder(client)
    )
    return Agent(
        client,
        get_tools(runtime_environment, exec_mode=True),
        context_manager,
        rollout_recorder=rollout_recorder,
        runtime_environment=runtime_environment,
    )


def build_model(
    config_path: 'typing.Union[str, Path]' = DEFAULT_CODEX_CONFIG_PATH,
    profile: 'typing.Union[str, None]' = None,
    timeout_seconds: 'float' = 120.0,
    managed_responses_base_url: 'typing.Union[str, None]' = None,
    vllm_endpoint: 'typing.Union[str, None]' = None,
    use_chat_completion: 'typing.Union[bool, None]' = None,
    use_messages: 'bool' = False,
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
                outcomming_api=(
                    "messages" if use_messages else "chat_completions"
                ),
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
    )
    return ResponsesModelClient(
        provider_config,
        timeout_seconds,
        originator=CLI_ORIGINATOR,
    )


def build_cli_queue(agent: 'Agent') -> 'CliSubmissionQueue':
    return CliSubmissionQueue(agent)


async def run_interactive_session(
    queue: 'CliSubmissionQueue',
    json_mode: 'bool',
    config_path: 'typing.Union[str, None]' = None,
) -> 'int':
    return await _run_interactive_session(
        queue,
        json_mode,
        config_path,
        view_factory=CliSessionView,
    )


async def run_cli(args: 'argparse.Namespace') -> 'int':
    queued_agent = None
    worker = None
    debug_dir = get_debug_dir()
    phase_handle = None if debug_dir is None else (debug_dir / "phase.log").open("a", encoding="utf-8")
    try:
        if args.put is not None and args.call:
            raise ValueError("--put and --call cannot be combined")
        if args.put is not None and args.prompt:
            raise ValueError("--put does not accept prompt text")
        configure_loguru()
        config_path = args.config
        if args.put is not None:
            def emit_put_log(message: 'str') -> 'None':
                print(message, flush=True)

            call_spec = upload_codex_home(args.put, event_handler=emit_put_log)
            emit_put_log(f"[put] testing call: {call_spec}")
            with tempfile.TemporaryDirectory(prefix="pycodex-put-call-test-") as tmpdir:
                config_path = bootstrap_called_home(call_spec, storage_root=tmpdir)
            emit_put_log(f"[put] call test ok: {config_path.name}")
            print("[put] one-click start:", flush=True)
            print(f"pycodex --call {shlex.quote(call_spec)}", flush=True)
            return 0
        if args.call:
            if phase_handle is not None:
                phase_handle.write("bootstrap_called_home:start\n")
                phase_handle.flush()
            config_path = bootstrap_called_home(args.call)
            if phase_handle is not None:
                phase_handle.write("bootstrap_called_home:done\n")
                phase_handle.flush()
            os.environ["CODEX_HOME"] = str(config_path.parent)
        if phase_handle is not None:
            phase_handle.write("build_model:start\n")
            phase_handle.flush()
        model = build_model(
            config_path=str(config_path),
            profile=args.profile,
            timeout_seconds=args.timeout_seconds,
            vllm_endpoint=args.vllm_endpoint,
            use_chat_completion=args.use_chat_completion or None,
            use_messages=args.use_messages,
        )
        if phase_handle is not None:
            phase_handle.write("build_model:done\n")
            phase_handle.write("build_agent:start\n")
            phase_handle.flush()
        agent = build_agent(
            model,
            config_path=str(config_path),
            profile=args.profile,
            system_prompt=args.system_prompt,
            session_mode="tui",
        )
        if phase_handle is not None:
            phase_handle.write("build_agent:done\n")
            phase_handle.write("build_cli_queue:start\n")
            phase_handle.flush()
        queued_agent = build_cli_queue(agent)
        if phase_handle is not None:
            phase_handle.write("build_cli_queue:done\n")
            phase_handle.flush()
        if should_run_interactive(args.prompt, sys.stdin.isatty()):
            return await run_interactive_session(
                queued_agent,
                args.json,
                str(config_path),
            )
        else:
            prompt_text = resolve_prompt_text(args.prompt)
            worker = asyncio.create_task(queued_agent.run_forever())
            if phase_handle is not None:
                phase_handle.write("submit_user_turn:start\n")
                phase_handle.flush()
            result = await queued_agent.submit_user_turn(prompt_text)
            if phase_handle is not None:
                phase_handle.write("submit_user_turn:done\n")
                phase_handle.flush()
            print(format_turn_output(result, args.json))
            return 0
    except Exception as exc:
        if phase_handle is not None:
            phase_handle.write("fatal_exception\n")
            phase_handle.flush()
        if debug_dir is not None:
            (debug_dir / "fatal_error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if phase_handle is not None:
            phase_handle.close()
        if queued_agent is not None and worker is not None:
            await queued_agent.shutdown()
            await worker

def ipython_agent(config_path: 'str' = DEFAULT_CODEX_CONFIG_PATH):
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    model = build_model(config_path)
    agent = build_agent(client=model, config_path=config_path)

    from pycodex.tools.ipython_tool import attach_ipython_tool

    attach_ipython_tool(agent)

    return agent

def main(argv: 'typing.Union[Sequence[str], None]' = None) -> 'int':
    raw_args = list(argv) if argv is not None else None
    if raw_args is None:
        raw_args = sys.argv[1:]

    if raw_args and raw_args[0] == "doctor":
        from .doctor import build_doctor_parser, run_doctor_cli

        parser = build_doctor_parser()
        args = parser.parse_args(raw_args[1:])
        try:
            return asyncio.run(run_doctor_cli(args))
        except ValueError as exc:
            parser.error(str(exc))
        except KeyboardInterrupt:
            return 130
        return 0

    parser = build_parser()
    args = parser.parse_args(raw_args)

    try:
        return asyncio.run(run_cli(args))
    except ValueError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
