from __future__ import annotations

import atexit
import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal, Sequence

from .agent import AgentLoop
from .collaboration import DEFAULT_COLLABORATION_MODE, CollaborationMode
from .context import ContextManager
from .model import ResponsesModelClient, ResponsesProviderConfig
from .protocol import AgentEvent
from .runtime import AgentRuntime
from .runtime_services import RuntimeEnvironment, create_runtime_environment
from .utils import CliSessionView, load_codex_dotenv
from responses_server import launch_chat_completion_compat_server

EXIT_COMMANDS = {"/exit", "/quit"}
HISTORY_COMMAND = "/history"
TITLE_COMMAND = "/title"
MODEL_COMMAND = "/model"
QUEUE_COMMAND = "/queue"
CliSessionMode = Literal["exec", "tui"]
LOCAL_RESPONSES_SERVER_API_KEY_ENV = "PYCODEX_LOCAL_RESPONSES_SERVER_KEY"
CLI_ORIGINATOR = "codex-tui"


def configure_loguru() -> None:
    try:
        from loguru import logger
    except ImportError:  # pragma: no cover - dependency may be absent in minimal envs
        return

    logger.remove()
    log_path = os.environ.get("PYCODEX_DEBUG_LOG", "").strip()
    if log_path:
        logger.add(log_path, level="DEBUG")
        return

    if os.environ.get("PYCODEX_DEBUG_STDERR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        logger.add(sys.stderr, level="DEBUG")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pycodex",
        description="Minimal Codex-style local CLI backed by ~/.codex/config.toml.",
    )
    parser.add_argument(
        "prompt", nargs="*", help="Prompt text. If omitted, read from stdin."
    )
    parser.add_argument(
        "--config",
        default=str(Path.home() / ".codex" / "config.toml"),
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


def should_run_interactive(prompt_parts: Sequence[str], stdin_is_tty: bool) -> bool:
    return not prompt_parts and stdin_is_tty


def resolve_prompt_text(prompt_parts: Sequence[str]) -> str:
    if prompt_parts:
        return " ".join(prompt_parts).strip()

    if not sys.stdin.isatty():
        prompt_text = sys.stdin.read().strip()
        if prompt_text:
            return prompt_text

    raise ValueError("prompt is required either as argv text or stdin")


def get_tools(runtime_environment: RuntimeEnvironment, exec_mode: bool = False):
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


def get_subagent_tools(runtime_environment: RuntimeEnvironment):
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

    registry = Registry()
    unified_exec_manager = UnifiedExecManager()
    registry.register(ExecCommandTool(unified_exec_manager))
    registry.register(WriteStdinTool(unified_exec_manager))
    registry.register(UpdatePlanTool(runtime_environment.plan_store))
    registry.register(ApplyPatchTool())
    registry.register(WebSearchTool())
    registry.register(ViewImageTool())
    return registry


def build_runtime(
    config_path: str,
    profile: str | None,
    system_prompt: str | None,
    client,
    session_mode: CliSessionMode = "exec",
    collaboration_mode: CollaborationMode = DEFAULT_COLLABORATION_MODE,
) -> AgentRuntime:
    use_tui_context = session_mode == "tui"
    context_manager = ContextManager.from_codex_config(
        config_path,
        profile,
        base_instructions_override=system_prompt,
        collaboration_mode=collaboration_mode,
        include_collaboration_instructions=use_tui_context,
    )
    subagent_context_manager = ContextManager.from_codex_config(
        config_path,
        profile,
        base_instructions_override=system_prompt,
        include_collaboration_instructions=False,
    )
    runtime_environment = create_runtime_environment()
    runtime_environment.request_user_input_manager.set_handler(None)
    runtime_environment.request_permissions_manager.set_handler(None)

    def make_subagent_runtime_builder(base_client):
        def build_subagent_runtime(
            model_override: str | None,
            reasoning_effort_override: str | None,
            initial_history=(),
            session_id: str | None = None,
        ) -> AgentRuntime:
            nested_client = base_client.with_overrides(
                model_override,
                reasoning_effort_override,
                session_id=session_id,
                openai_subagent="collab_spawn",
            )
            subagent_runtime_environment = create_runtime_environment()
            subagent_runtime_environment.request_user_input_manager.set_handler(None)
            subagent_runtime_environment.request_permissions_manager.set_handler(None)
            subagent_runtime_environment.subagent_manager.set_runtime_builder(
                make_subagent_runtime_builder(nested_client)
            )
            sub_agent = AgentLoop(
                nested_client,
                get_subagent_tools(subagent_runtime_environment),
                subagent_context_manager,
                initial_history=tuple(initial_history),
            )
            return AgentRuntime(
                sub_agent, runtime_environment=subagent_runtime_environment
            )

        return build_subagent_runtime

    runtime_environment.subagent_manager.set_runtime_builder(
        make_subagent_runtime_builder(client)
    )
    return AgentRuntime(
        AgentLoop(
            client, get_tools(runtime_environment, exec_mode=True), context_manager
        ),
        runtime_environment=runtime_environment,
    )


def format_turn_output(result, json_mode: bool) -> str:
    if json_mode:
        return json.dumps(asdict(result), ensure_ascii=False, indent=2)
    return result.output_text or ""


def _build_model_client(
    config_path: str,
    profile: str | None,
    timeout_seconds: float,
    managed_responses_base_url: str | None = None,
    vllm_endpoint: str | None = None,
    use_chat_completion: bool = False,
):
    load_codex_dotenv(config_path)
    provider_config = ResponsesProviderConfig.from_codex_config(
        config_path,
        profile,
    )
    url, key_env = provider_config.base_url, provider_config.api_key_env
    if managed_responses_base_url is not None:
        url, key_env = (
            managed_responses_base_url,
            LOCAL_RESPONSES_SERVER_API_KEY_ENV,
        )
        os.environ.setdefault(LOCAL_RESPONSES_SERVER_API_KEY_ENV, "dummy")
    elif vllm_endpoint or use_chat_completion:
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


async def prompt_request_user_input(
    view: CliSessionView,
    payload: dict[str, object],
) -> dict[str, object] | None:
    view.finish_stream()
    view.pause_spinner()
    view.write_line("[request_user_input] waiting for user response")
    answers: dict[str, dict[str, list[str]]] = {}
    try:
        for question in payload.get("questions", []):
            if not isinstance(question, dict):
                continue
            header = str(question.get("header", "")).strip()
            question_text = str(question.get("question", "")).strip()
            question_id = str(question.get("id", "")).strip()
            if header:
                view.write_line(f"[{header}] {question_text}")
            else:
                view.write_line(question_text)

            options = question.get("options") or []
            if isinstance(options, list):
                for index, option in enumerate(options, start=1):
                    if not isinstance(option, dict):
                        continue
                    label = str(option.get("label", "")).strip()
                    description = str(option.get("description", "")).strip()
                    view.write_line(f"  {index}. {label} - {description}")
            view.write_line("  0. Other")

            try:
                raw_answer = await view.prompt_async("answer> ")
            except EOFError:
                return None
            answer_text = raw_answer.strip()
            if not answer_text:
                return None

            selected_answer = answer_text
            if answer_text.isdigit() and isinstance(options, list):
                choice = int(answer_text)
                if 1 <= choice <= len(options):
                    option = options[choice - 1]
                    if isinstance(option, dict):
                        selected_answer = (
                            str(option.get("label", "")).strip() or answer_text
                        )
                elif choice == 0:
                    try:
                        raw_answer = await view.prompt_async("other> ")
                    except EOFError:
                        return None
                    selected_answer = raw_answer.strip()
                    if not selected_answer:
                        return None

            answers[question_id] = {"answers": [selected_answer]}

        return {"answers": answers}
    finally:
        view.resume_spinner()


async def prompt_request_permissions(
    view: CliSessionView,
    payload: dict[str, object],
) -> dict[str, object] | None:
    view.finish_stream()
    view.pause_spinner()
    view.write_line("[request_permissions] user approval required")
    reason = payload.get("reason")
    if reason:
        view.write_line(f"Reason: {reason}")
    view.write_line("Requested permissions:")
    view.write_line(
        json.dumps(payload.get("permissions", {}), ensure_ascii=False, indent=2)
    )
    view.write_line("Choose: [n] deny / [t] grant for turn / [s] grant for session")
    try:
        raw_answer = await view.prompt_async("permissions> ")
    except EOFError:
        return None
    finally:
        view.resume_spinner()

    answer = raw_answer.strip().lower()
    if answer in {"t", "turn", "y", "yes"}:
        return {
            "permissions": payload.get("permissions", {}),
            "scope": "turn",
        }
    if answer in {"s", "session"}:
        return {
            "permissions": payload.get("permissions", {}),
            "scope": "session",
        }
    return {
        "permissions": {},
        "scope": "turn",
    }


async def run_interactive_session(
    runtime: AgentRuntime,
    json_mode: bool,
) -> int:
    worker = asyncio.create_task(runtime.run_forever())
    view = CliSessionView()
    model_client = runtime._agent_loop._model_client
    runtime.set_event_handler(view.handle_event)
    pending_turn_tasks: set[asyncio.Task[None]] = set()
    runtime_environment = runtime.runtime_environment
    runtime_environment.request_user_input_manager.set_handler(
        lambda payload: prompt_request_user_input(view, payload)
    )
    runtime_environment.request_permissions_manager.set_handler(
        lambda payload: prompt_request_permissions(view, payload)
    )
    view.write_line("pycodex interactive mode. Type /exit to quit.")
    view.write_line("Extra commands: /history, /title, /model")
    try:

        def has_pending_turn_tasks() -> bool:
            pending_turn_tasks.difference_update(
                task for task in tuple(pending_turn_tasks) if task.done()
            )
            return bool(pending_turn_tasks)

        async def wait_for_turn_result(future) -> None:
            try:
                result = await future
            except Exception as exc:  # pragma: no cover - defensive surface
                if str(exc) == "submission interrupted":
                    return
                view.show_error(str(exc))
                return

            if json_mode:
                view.write_line(format_turn_output(result, True))

        while True:
            try:
                raw_line = await view.poll_prompt("pycodex> ")
            except EOFError:
                break
            if raw_line is None:
                await asyncio.sleep(0.05)
                continue

            prompt_text = raw_line.strip()
            if not prompt_text:
                continue
            if prompt_text in EXIT_COMMANDS:
                break
            if prompt_text == HISTORY_COMMAND:
                view.show_history()
                continue
            if prompt_text == TITLE_COMMAND:
                view.show_title()
                continue
            if prompt_text.startswith(f"{QUEUE_COMMAND} "):
                queued_text = prompt_text[len(QUEUE_COMMAND) :].strip()
                if not queued_text:
                    view.write_line("Usage: /queue <message>")
                    continue
                try:
                    submission_id, future = await runtime.enqueue_user_turn(
                        queued_text, queue="enqueue"
                    )
                    view.show_steer_queued(submission_id, queued_text)
                    turn_task = asyncio.create_task(wait_for_turn_result(future))
                    pending_turn_tasks.add(turn_task)
                except Exception as exc:  # pragma: no cover - defensive surface
                    view.show_error(str(exc))
                continue
            if prompt_text == MODEL_COMMAND:
                view.write_line(
                    f"Current model: {getattr(model_client, 'model', None) or 'unavailable'}"
                )
                models = await model_client.list_models()
                view.write_line(f"Available models: {', '.join(models)}")
                continue
            if prompt_text.startswith(f"{MODEL_COMMAND} "):
                if has_pending_turn_tasks():
                    view.write_line(
                        "Cannot change model while work is running or queued in steer mode."
                    )
                    continue
                model_name = prompt_text[len(MODEL_COMMAND) :].strip()
                if not model_name:
                    view.write_line("Usage: /model <model>")
                    continue

                model_client.model = model_name
                view.write_line(f"Switched model to {model_name}.")
                continue

            try:
                steered = has_pending_turn_tasks()
                submission_id, future = await runtime.enqueue_user_turn(
                    prompt_text,
                    queue="steer",
                )
                if steered:
                    view.schedule_steer_inserted(submission_id, prompt_text)
                turn_task = asyncio.create_task(wait_for_turn_result(future))
                pending_turn_tasks.add(turn_task)
                continue
            except Exception as exc:  # pragma: no cover - defensive surface
                view.show_error(str(exc))
                continue
    finally:
        runtime_environment.request_user_input_manager.set_handler(None)
        runtime_environment.request_permissions_manager.set_handler(None)
        await runtime.shutdown()
        await worker
        if pending_turn_tasks:
            await asyncio.gather(*pending_turn_tasks, return_exceptions=True)
        view.close()

    return 0


async def run_cli(args: argparse.Namespace) -> int:
    configure_loguru()
    runtime = None
    worker = None
    try:
        client = _build_model_client(
            args.config,
            args.profile,
            args.timeout_seconds,
            vllm_endpoint=args.vllm_endpoint,
            use_chat_completion=args.use_chat_completion,
        )

        runtime = build_runtime(
            args.config,
            args.profile,
            args.system_prompt,
            client,
            session_mode="tui",
        )
        if should_run_interactive(args.prompt, sys.stdin.isatty()):
            return await run_interactive_session(
                runtime,
                args.json,
            )
        else:
            prompt_text = resolve_prompt_text(args.prompt)
            worker = asyncio.create_task(runtime.run_forever())
            result = await runtime.submit_user_turn(prompt_text)
            print(format_turn_output(result, args.json))
            return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if runtime is not None and worker is not None:
            await runtime.shutdown()
            await worker


def main(argv: Sequence[str] | None = None) -> int:
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
