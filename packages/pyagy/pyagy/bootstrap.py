"""Bootstrap AGY CLI sessions from ~/.codex/config.toml."""

from dataclasses import dataclass, replace
from datetime import datetime
import os
from pathlib import Path
import typing
import uuid

from pycodex.bootstrap import LOCAL_RESPONSES_SERVER_API_KEY_ENV, configure_loguru
from pycodex.config import CodexConfig
from pycodex.context import ContextConfig
from pycodex.tools.base_tool import BaseTool
from pycodex.utils.codex_config import DEFAULT_CODEX_CONFIG_PATH
from pycodex.utils.http_headers import build_http_headers
from responses_server.config import CompatServerConfig

from .gemini_wire import AgyManagedResponseServer, AgyResponsesModelClient
from .models import normalize_effort_level
from .multi_agents import AgyMultiAgentController, AgySessionNode
from .prompts.system_prompt import default_app_data_dir
from .tools import SearchHandler
from .transcript import AgyTranscriptStore, RestoredConversation, find_latest_conversation_id


@dataclass
class AgyBootstrappedSession:
    root_node: "AgySessionNode"
    controller: "AgyMultiAgentController"
    owned_server: "AgyManagedResponseServer"
    codex_config: "CodexConfig"
    compat_config: "CompatServerConfig"

    async def close(self) -> "None":
        try:
            await self.controller.close_all()
        finally:
            self.owned_server.stop()


def build_agy_session(
    config_path: "typing.Union[str, Path]" = DEFAULT_CODEX_CONFIG_PATH,
    profile: "typing.Optional[str]" = None,
    cwd: "typing.Optional[Path]" = None,
    model_override: "typing.Optional[str]" = None,
    explicit_effort: "typing.Optional[str]" = None,
    app_data_dir: "typing.Optional[Path]" = None,
    global_config_dir: "typing.Optional[Path]" = None,
    additional_workspaces: "typing.Sequence[Path]" = (),
    conversation_id: "typing.Optional[str]" = None,
    continue_latest: "bool" = False,
    resume_conversation_id: "typing.Optional[str]" = None,
    timeout_seconds: "float" = 300.0,
    search_handler: "typing.Optional[SearchHandler]" = None,
    clock: "typing.Optional[typing.Callable[[], datetime]]" = None,
    codex_config: "typing.Optional[CodexConfig]" = None,
    context_config: "typing.Optional[ContextConfig]" = None,
    toolset: "typing.Optional[typing.Sequence[str]]" = None,
    extra_tools: "typing.Sequence[BaseTool]" = (),
    enable_rollout_recording: "bool" = False,
) -> "AgyBootstrappedSession":
    configure_loguru()
    codex_cfg = (
        codex_config
        if codex_config is not None
        else CodexConfig.from_env(config_path=config_path, profile=profile)
    )
    provider_cfg = codex_cfg.provider
    if provider_cfg.compat_api != "gemini":
        raise ValueError(
            f"pyagy requires a provider with compat_api='gemini' in {codex_cfg.path}, "
            f"got compat_api={provider_cfg.compat_api!r}"
        )
    if explicit_effort is not None:
        normalize_effort_level(explicit_effort)

    resolved_cwd = (cwd if cwd is not None else Path.cwd()).expanduser().resolve()
    resolved_app_data = (
        app_data_dir.expanduser() if app_data_dir is not None else default_app_data_dir()
    )
    config_effort = provider_cfg.reasoning_effort
    effective_context_config = (
        context_config
        if context_config is not None
        else replace(
            codex_cfg.context,
            cwd=resolved_cwd,
            model_auto_compact_token_limit=None,
            include_permissions_instructions=False,
            include_skills_instructions=False,
            include_environment_context=False,
            code_mode_only=False,
        )
    )

    restored: "typing.Optional[RestoredConversation]" = None
    if continue_latest and resume_conversation_id is not None:
        raise ValueError("cannot specify both --continue and --conversation")
    if continue_latest:
        target_conv_id = find_latest_conversation_id(resolved_app_data, cwd=resolved_cwd)
        restored = AgyTranscriptStore.load_conversation(
            resolved_app_data, target_conv_id, clock=clock
        )
    elif resume_conversation_id is not None:
        restored = AgyTranscriptStore.load_conversation(
            resolved_app_data, resume_conversation_id, clock=clock
        )

    effective_conv_id = (
        restored.conversation_id
        if restored is not None
        else (conversation_id or str(uuid.uuid4()))
    )
    effective_model = (
        model_override
        or (restored.model if restored is not None and restored.model else None)
        or provider_cfg.model
        or "gemini-3.1-pro-high"
    )

    upstream_headers = build_http_headers(
        provider_cfg.http_headers, provider_cfg.env_http_headers
    )
    compat_cfg = replace(
        CompatServerConfig.from_base_url(
            provider_cfg.base_url,
            api_key_env=provider_cfg.api_key_env,
            model_provider=provider_cfg.provider_name,
            outcomming_api="gemini",
            http_headers=upstream_headers,
        ),
        timeout_seconds=timeout_seconds,
    )
    owned_server = AgyManagedResponseServer(compat_cfg)
    owned_server.start()

    try:
        os.environ.setdefault(LOCAL_RESPONSES_SERVER_API_KEY_ENV, "dummy")
        local_provider = replace(
            provider_cfg,
            model=effective_model,
            base_url=owned_server.base_url,
            api_key_env=LOCAL_RESPONSES_SERVER_API_KEY_ENV,
            http_headers={},
            env_http_headers={},
            responses_lite_override=False,
        )
        model_client = AgyResponsesModelClient(
            config=local_provider,
            timeout_seconds=timeout_seconds,
            session_id=effective_conv_id,
            config_effort=config_effort,
            explicit_effort=explicit_effort,
        )
        controller = AgyMultiAgentController(
            app_data_dir=resolved_app_data,
            root_model_client=model_client,
            context_config=effective_context_config,
            compat_config=compat_cfg,
            global_config_dir=global_config_dir,
            config_effort=config_effort,
            explicit_effort=explicit_effort,
            search_handler=search_handler,
            clock=clock,
        )
        root_node = AgySessionNode(
            conversation_id=effective_conv_id,
            cwd=resolved_cwd,
            app_data_dir=resolved_app_data,
            model_client=model_client,
            context_config=effective_context_config,
            controller=controller,
            compat_config=compat_cfg,
            parent_conversation_id=None,
            role="root",
            type_name="root",
            additional_workspaces=additional_workspaces,
            global_config_dir=global_config_dir,
            toolset=toolset,
            config_effort=config_effort,
            explicit_effort=explicit_effort,
            initial_history=restored.history if restored is not None else (),
            last_reported_model_display=(
                restored.last_reported_model_display if restored is not None else None
            ),
            anchored_user_messages=(
                restored.anchored_user_messages if restored is not None else None
            ),
            pending_post_user_messages=(
                (restored.restart_notice_message,) if restored is not None else ()
            ),
            initial_step_counter=(
                restored.step_counter if restored is not None else None
            ),
            initial_user_requests=(
                restored.user_requests if restored is not None else ()
            ),
            initial_parent_messages=(
                restored.parent_messages if restored is not None else ()
            ),
            initial_spawned_subagents=(
                restored.spawned_subagents if restored is not None else ()
            ),
            initial_anchor_tokens=(
                restored.anchor_tokens if restored is not None else 0
            ),
            initial_anchor_history_len=(
                restored.anchor_history_len if restored is not None else None
            ),
            search_handler=search_handler,
            clock=clock,
            extra_tools=extra_tools,
            enable_rollout_recording=enable_rollout_recording,
        )
        controller.register_node(root_node)
        if restored is not None:
            root_node.transcript_store.record_system_message_file(
                content=restored.restart_notice_message.text,
                sender="system",
                priority="MESSAGE_PRIORITY_LOW",
            )
    except BaseException:
        owned_server.stop()
        raise

    return AgyBootstrappedSession(
        root_node=root_node,
        controller=controller,
        owned_server=owned_server,
        codex_config=codex_cfg,
        compat_config=compat_cfg,
    )
