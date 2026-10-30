import os
import sys
import typing
from dataclasses import dataclass, replace
from pathlib import Path

from .model import (
    DEFAULT_CODEX_CONFIG_PATH,
    ResponsesModelClient,
    ResponsesProviderConfig,
)
from .utils import get_debug_dir, load_codex_dotenv

if typing.TYPE_CHECKING:
    from responses_server import ManagedResponseServer

LOCAL_RESPONSES_SERVER_API_KEY_ENV = "PYCODEX_LOCAL_RESPONSES_SERVER_KEY"
CLI_ORIGINATOR = "codex-tui"


@dataclass(frozen=True)
class ModelBootstrapResult:
    client: "ResponsesModelClient"
    owned_server: "typing.Union[ManagedResponseServer, None]" = None

    def close(self) -> "None":
        if self.owned_server is not None:
            self.owned_server.stop()


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


def build_model(
    config_path: "typing.Union[str, Path]" = DEFAULT_CODEX_CONFIG_PATH,
    profile: "typing.Union[str, None]" = None,
    timeout_seconds: "float" = 120.0,
    managed_responses_base_url: "typing.Union[str, None]" = None,
    vllm_endpoint: "typing.Union[str, None]" = None,
    use_chat_completion: "typing.Union[bool, None]" = None,
    use_messages: "bool" = False,
    use_default_profile: "bool" = True,
) -> "ModelBootstrapResult":
    load_codex_dotenv(config_path)
    provider_config = ResponsesProviderConfig.from_codex_config(
        config_path,
        profile,
        use_default_profile,
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
    managed_server = None
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
    try:
        client = ResponsesModelClient(
            provider_config, timeout_seconds, originator=CLI_ORIGINATOR,
        )
    except BaseException:
        if managed_server is not None:
            managed_server.stop()
        raise
    return ModelBootstrapResult(client, managed_server)
