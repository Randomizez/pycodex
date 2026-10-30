import asyncio
import json
import os
import re
import threading
import typing
import urllib.parse
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import monotonic
from typing import Callable

import requests
from urllib3.exceptions import ReadTimeoutError

from .utils.compat import Protocol
from .utils.codex_config import read_codex_config

from .events import (
    AssistantDeltaEvent,
    ModelEvent,
    ModelRequestTimingEvent,
    StreamErrorEvent,
    TokenCountEvent,
    ToolCalledEvent,
)
from .utils.model_metadata import model_metadata
from .protocol import (
    AssistantMessage,
    JSONDict,
    ModelOutputItem,
    ModelResponse,
    Prompt,
    ReasoningItem,
    ToolCall,
    ToolSpec,
    qualified_tool_name,
)
from .utils import build_user_agent, uuid7_string

DEFAULT_CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"
DEFAULT_ORIGINATOR = "pycodex"
RESPONSES_LITE_HEADER = "x-openai-internal-codex-responses-lite"
ModelStreamEventHandler = Callable[[ModelEvent], None]
NOOP_MODEL_STREAM_EVENT_HANDLER: "ModelStreamEventHandler" = lambda _event: None
DEFAULT_STREAM_MAX_RETRIES = 5
DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000
INITIAL_RETRY_DELAY_SECONDS = 0.2
RETRY_BACKOFF_FACTOR = 2.0
RATE_LIMIT_RETRY_AFTER_RE = re.compile(
    r"(?i)(?:try again in|retry after)\s*(\d+(?:\.\d+)?)\s*(ms|seconds?|s)"
)


@dataclass(frozen=True)
class ModelRecordingMetadata:
    originator: "str"
    provider_name: "typing.Union[str, None]"


class ModelClient(Protocol):
    model: "str"
    reasoning_effort: "typing.Union[str, None]"
    recording_metadata: "ModelRecordingMetadata"

    def with_overrides(
        self,
        model: "typing.Union[str, None]",
        reasoning_effort: "typing.Union[str, None]",
        session_id: "typing.Union[str, None]" = None,
        openai_subagent: "typing.Union[str, None]" = None,
    ) -> "ModelClient":
        """Construct an independent client for a child Agent."""

    async def complete(
        self,
        prompt: "Prompt",
        event_handler: "ModelStreamEventHandler" = NOOP_MODEL_STREAM_EVENT_HANDLER,
    ) -> "ModelResponse":
        """Return the next batch of model output items for the current prompt."""


class ModelControl(ModelClient, Protocol):
    @property
    def endpoint(self) -> "str":
        """Return the active API base URL, safe for display."""

    async def list_models(self) -> "typing.List[str]":
        """List the models available to an interactive session."""


@dataclass(
    frozen=True,
)
class ResponsesProviderConfig:
    model: "str"
    provider_name: "str"
    base_url: "str"
    api_key_env: "typing.Union[str, None]"
    use_chat_completion: "bool" = False
    wire_api: "str" = "responses"
    query_params: "typing.Dict[str, str]" = field(default_factory=dict)
    reasoning_effort: "typing.Union[str, None]" = None
    reasoning_summary: "typing.Union[str, None]" = None
    verbosity: "typing.Union[str, None]" = None
    sandbox_mode: "typing.Union[str, None]" = None
    beta_features_header: "typing.Union[str, None]" = None
    stream_max_retries: "typing.Union[int, None]" = None
    stream_idle_timeout_ms: "typing.Union[int, None]" = None
    service_tier: "typing.Union[str, None]" = None
    responses_lite_override: "typing.Union[bool, None]" = None
    fallback_models: "typing.Tuple[str, ...]" = ()
    fallback_cooldown_seconds: "int" = 900
    stream_first_data_timeout_ms: "typing.Union[int, None]" = None

    def __post_init__(self):
        cooldown = self.fallback_cooldown_seconds
        if isinstance(cooldown, bool) or not isinstance(cooldown, int) or cooldown < 0:
            raise ValueError("fallback_cooldown_seconds must be a non-negative integer")
        value = self.stream_first_data_timeout_ms
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError("stream_first_data_timeout_ms must be a positive integer")

    @classmethod
    def from_codex_config(
        cls,
        config_path: "typing.Union[str, Path]" = DEFAULT_CODEX_CONFIG_PATH,
        profile: "typing.Union[str, None]" = None,
        use_default_profile: "bool" = True,
    ) -> "ResponsesProviderConfig":
        selected = read_codex_config(config_path, profile, use_default_profile)

        provider_name = selected.get("model_provider", "openai")
        provider = selected.get("model_providers", {}).get(provider_name)
        if provider is None:
            if provider_name != "openai":
                raise ValueError(f"unknown Codex model provider: {provider_name}")
            provider = {
                "base_url": (
                    selected.get("openai_base_url") or "https://api.openai.com/v1"
                ),
                "env_key": "OPENAI_API_KEY",
            }
        wire_api = provider.get("wire_api", "responses")
        if wire_api != "responses":
            raise ValueError(f"unsupported wire_api for Python client: {wire_api}")

        api_key_env = provider.get("env_key")

        query_params = {
            str(key): str(value)
            for key, value in provider.get("query_params", {}).items()
        }
        features = selected.get("features", {})
        beta_features: "typing.List[str]" = []
        if isinstance(features, dict) and features.get("guardian_approval") is True:
            beta_features.append("guardian_approval")
        use_chat_completion = _optional_bool(selected.get("use_chat_completion"))
        if use_chat_completion is None:
            use_chat_completion = _optional_bool(provider.get("use_chat_completion"))
        if use_chat_completion is None:
            use_chat_completion = False
        fallback_models = provider.get("fallback_models", [])
        if not isinstance(fallback_models, list) or any(
            not isinstance(model, str) or not model or any(c.isspace() for c in model)
            for model in fallback_models
        ):
            raise ValueError("fallback_models must be an array of non-empty model IDs")
        return cls(
            model=selected["model"],
            provider_name=provider_name,
            base_url=provider["base_url"],
            api_key_env=api_key_env,
            use_chat_completion=use_chat_completion,
            wire_api=wire_api,
            query_params=query_params,
            reasoning_effort=selected.get("model_reasoning_effort"),
            reasoning_summary=selected.get("model_reasoning_summary"),
            verbosity=selected.get("model_verbosity"),
            service_tier=selected.get("service_tier"),
            sandbox_mode=selected.get("sandbox_mode"),
            beta_features_header=",".join(beta_features) or None,
            stream_max_retries=_optional_int(provider.get("stream_max_retries")),
            stream_idle_timeout_ms=_optional_int(
                provider.get("stream_idle_timeout_ms")
            ),
            fallback_models=tuple(dict.fromkeys(fallback_models)),
            fallback_cooldown_seconds=provider.get("fallback_cooldown_seconds", 900),
            stream_first_data_timeout_ms=provider.get("stream_first_data_timeout_ms"),
        )

    def api_key(self) -> "typing.Union[str, None]":
        if not self.api_key_env:
            return None
        value = os.environ.get(self.api_key_env, "")
        if not value:
            raise RuntimeError(
                f"missing API key environment variable: {self.api_key_env}"
            )
        return value

    def with_overrides(
        self,
        model: "typing.Union[str, None]" = None,
        reasoning_effort: "typing.Union[str, None]" = None,
    ) -> "ResponsesProviderConfig":
        return replace(
            self,
            model=self.model if model is None else model,
            reasoning_effort=(
                self.reasoning_effort if reasoning_effort is None else reasoning_effort
            ),
        )

    def effective_stream_max_retries(self) -> "int":
        if self.stream_max_retries is None:
            return DEFAULT_STREAM_MAX_RETRIES
        return max(int(self.stream_max_retries), 0)

    def effective_stream_idle_timeout_seconds(self) -> "float":
        if self.stream_idle_timeout_ms is None:
            return DEFAULT_STREAM_IDLE_TIMEOUT_MS / 1000.0
        return max(int(self.stream_idle_timeout_ms), 1) / 1000.0

    def metadata(self) -> "typing.Union[JSONDict, None]":
        return model_metadata(self.model)

    def use_responses_lite(self) -> "bool":
        if self.responses_lite_override is not None:
            return self.responses_lite_override
        metadata = self.metadata()
        if metadata is None:
            return False
        return metadata.get("use_responses_lite") is True

    def effective_reasoning_effort(self) -> "typing.Union[str, None]":
        if self.reasoning_effort is not None:
            return str(self.reasoning_effort)
        metadata = self.metadata()
        if not _metadata_supports_reasoning(metadata):
            return None
        return _optional_metadata_string(metadata, "default_reasoning_level")

    def effective_reasoning_summary(self) -> "typing.Union[str, None]":
        summary = self.reasoning_summary
        if summary is None:
            metadata = self.metadata()
            if not _metadata_supports_reasoning(metadata):
                return None
            summary = _optional_metadata_string(metadata, "default_reasoning_summary")
        if summary is None:
            return None
        if str(summary).strip().lower() == "none":
            return None
        return str(summary)

    def effective_verbosity(self) -> "typing.Union[str, None]":
        if self.verbosity is not None:
            return str(self.verbosity)
        metadata = self.metadata()
        if metadata is None or metadata.get("support_verbosity") is not True:
            return None
        return _optional_metadata_string(metadata, "default_verbosity")

    def effective_service_tier(self) -> "typing.Union[str, None]":
        service_tier = self.service_tier
        if service_tier is None:
            return None
        service_tier = str(service_tier).strip()
        if service_tier == "fast":
            service_tier = "priority"
        if not service_tier or service_tier == "default":
            return None

        metadata = self.metadata()
        if metadata is None:
            return None
        supported = metadata.get("service_tiers")
        if not isinstance(supported, list):
            return None
        for tier in supported:
            if isinstance(tier, dict) and tier.get("id") == service_tier:
                return service_tier
        return None


def _optional_bool(
    value: "typing.Union[bool, str, int, None]",
) -> "typing.Union[bool, None]":
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean config value: {value!r}")


def _metadata_supports_reasoning(
    metadata: "typing.Union[JSONDict, None]",
) -> "bool":
    return metadata is not None and metadata.get("supports_reasoning_summaries") is True


def _optional_metadata_string(
    metadata: "typing.Union[JSONDict, None]",
    key: "str",
) -> "typing.Union[str, None]":
    if metadata is None:
        return None
    value = metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _strip_image_details(items: "typing.Iterable[object]") -> "None":
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, list):
            _strip_image_detail_from_content_items(content)
        output = item.get("output")
        if isinstance(output, list):
            _strip_image_detail_from_content_items(output)


def _strip_image_detail_from_content_items(items: "typing.Iterable[object]") -> "None":
    for item in items:
        if isinstance(item, dict) and item.get("type") == "input_image":
            item.pop("detail", None)


def _filter_audio_inputs(items, metadata):
    if metadata is None or "audio" in metadata.get("input_modalities", ()):
        return
    for item in items:
        for key in ("content", "output"):
            content = item.get(key)
            if not isinstance(content, list):
                continue
            for index, part in enumerate(content):
                if part.get("type") == "input_audio":
                    content[index] = {
                        "type": "input_text",
                        "text": "audio content omitted because you do not support audio input",
                    }


class ResponsesApiError(RuntimeError):
    pass


class ContextLengthExceeded(ResponsesApiError):
    def __init__(self, message: "str") -> "None":
        super().__init__(message)
        self.usage: "typing.Union[typing.Dict[str, int], None]" = None
        self.token_limit: "typing.Union[int, None]" = None
        requested = re.search(r"requested\s+([0-9,]+)\s+tokens", message, re.IGNORECASE)
        if requested is not None:
            total = int(requested.group(1).replace(",", ""))
            self.usage = {"total_tokens": total, "input_tokens": total}
            split = re.search(
                r"\(([0-9,]+)\s+in\s+the\s+messages,\s+([0-9,]+)\s+in\s+the\s+completion\)",
                message,
                re.IGNORECASE,
            )
            if split is not None:
                self.usage["input_tokens"] = int(split.group(1).replace(",", ""))
                self.usage["output_tokens"] = int(split.group(2).replace(",", ""))
        limit = re.search(
            r"maximum\s+context\s+length\s+is\s+([0-9,]+)\s+tokens",
            message,
            re.IGNORECASE,
        )
        if limit is not None:
            self.token_limit = int(limit.group(1).replace(",", ""))


class ResponsesIncompleteError(ResponsesApiError):
    def __init__(
        self,
        message: "str",
        partial_items: "typing.Sequence[ModelOutputItem]",
        reason: "str" = "",
    ) -> "None":
        super().__init__(message)
        self.partial_items = tuple(partial_items)
        self.reason = reason


class ResponsesRetryableError(ResponsesApiError):
    def __init__(
        self,
        message: "str",
        retry_delay_seconds: "typing.Union[float, None]" = None,
    ) -> "None":
        super().__init__(message)
        self.retry_delay_seconds = retry_delay_seconds


class ResponsesModelUnavailableError(ResponsesApiError):
    """The selected model cannot serve this request; another model may work."""


class ResponsesFirstDataTimeout(ResponsesRetryableError):
    """The configured wait for initial response data has expired."""


@dataclass
class _StreamDiagnostics:
    raw_lines_received: "int" = 0
    sse_events_received: "int" = 0
    output_items_received: "int" = 0
    last_sse_event_name: "str" = ""
    last_event_type: "str" = ""
    last_payload_excerpt: "str" = ""
    started_at: "float" = field(default_factory=monotonic)
    first_data_seconds: "typing.Union[float, None]" = None


class ResponsesModelClient:
    """Minimal OpenAI-compatible Responses API client.

    This implementation is intentionally narrow: it supports the subset needed
    by the current Agent abstraction, namely assistant text and function
    tool calls over the streaming `/responses` endpoint.
    """

    def __init__(
        self,
        config: "ResponsesProviderConfig",
        timeout_seconds: "float" = 120.0,
        session_id: "typing.Union[str, None]" = None,
        originator: "str" = DEFAULT_ORIGINATOR,
        user_agent: "typing.Union[str, None]" = None,
        openai_subagent: "typing.Union[str, None]" = None,
    ) -> "None":
        self._config = config
        self._timeout_seconds = timeout_seconds
        self._session_id = session_id or uuid7_string()
        self.recording_metadata = ModelRecordingMetadata(originator, config.provider_name)
        self._user_agent = user_agent or build_user_agent(originator)
        self._openai_subagent = openai_subagent
        self._model_cooldowns: "typing.Dict[str, float]" = {}

    @property
    def model(self) -> "str":
        return self._config.model

    @model.setter
    def model(self, model: "str") -> "None":
        self._config = replace(self._config, model=model)
        self._model_cooldowns.clear()

    @property
    def reasoning_effort(self) -> "typing.Union[str, None]":
        """Return the effective effort; assign None to restore the model default."""
        return self._config.effective_reasoning_effort()

    @reasoning_effort.setter
    def reasoning_effort(self, effort: "typing.Union[str, None]") -> "None":
        if effort is not None and (
            not effort or any(char.isspace() for char in effort)
        ):
            raise ValueError("reasoning effort must be a single non-empty value")
        self._config = replace(self._config, reasoning_effort=effort)

    @property
    def endpoint(self) -> "str":
        """Return the active API base URL without credentials, query or fragment."""
        url = urllib.parse.urlsplit(self._config.base_url)
        return urllib.parse.urlunsplit(
            (url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", "")
        )

    @classmethod
    def from_codex_config(
        cls,
        config_path: "typing.Union[str, Path]" = DEFAULT_CODEX_CONFIG_PATH,
        profile: "typing.Union[str, None]" = None,
        timeout_seconds: "float" = 120.0,
        originator: "str" = DEFAULT_ORIGINATOR,
        user_agent: "typing.Union[str, None]" = None,
    ) -> "ResponsesModelClient":
        config = ResponsesProviderConfig.from_codex_config(config_path, profile)
        return cls(
            config, timeout_seconds, originator=originator, user_agent=user_agent
        )

    def with_overrides(
        self,
        model: "typing.Union[str, None]" = None,
        reasoning_effort: "typing.Union[str, None]" = None,
        session_id: "typing.Union[str, None]" = None,
        openai_subagent: "typing.Union[str, None]" = None,
    ) -> "ResponsesModelClient":
        return ResponsesModelClient(
            self._config.with_overrides(
                model or self.model,
                reasoning_effort,
            ),
            self._timeout_seconds,
            session_id=self._session_id if session_id is None else session_id,
            originator=self.recording_metadata.originator,
            user_agent=self._user_agent,
            openai_subagent=(
                self._openai_subagent if openai_subagent is None else openai_subagent
            ),
        )

    def responses_url(self) -> "str":
        base_url = self._config.base_url.rstrip("/")
        url = f"{base_url}/responses"
        if self._config.query_params:
            return f"{url}?{urllib.parse.urlencode(self._config.query_params)}"
        return url

    def models_url(self) -> "str":
        base_url = self._config.base_url.rstrip("/")
        url = f"{base_url}/models"
        if self._config.query_params:
            return f"{url}?{urllib.parse.urlencode(self._config.query_params)}"
        return url

    async def list_models(self) -> "typing.List[str]":
        return await asyncio.to_thread(self.list_models_sync)

    def list_models_sync(self) -> "typing.List[str]":
        return self._list_models_sync()

    async def complete(
        self,
        prompt: "Prompt",
        event_handler: "ModelStreamEventHandler" = NOOP_MODEL_STREAM_EVENT_HANDLER,
    ) -> "ModelResponse":
        retries = 0
        max_retries = self._config.effective_stream_max_retries()
        models = tuple(dict.fromkeys((self.model,) + self._config.fallback_models))
        now = monotonic()
        self._model_cooldowns = {
            model: deadline
            for model, deadline in self._model_cooldowns.items()
            if model in models and deadline > now
        }
        available_models = [
            model for model in models if model not in self._model_cooldowns
        ]
        if not available_models:
            self._model_cooldowns.clear()
            available_models = list(models)
        fallback_models = iter(available_models)
        first_model = next(fallback_models)
        request_client = (
            self
            if first_model == self.model
            else self.with_overrides(model=first_model)
        )
        loop = asyncio.get_running_loop()
        active = True
        callback_lock = threading.Lock()

        def deliver(event):
            if active:
                event_handler(event)

        def receive(event):
            with callback_lock:
                if active:
                    loop.call_soon_threadsafe(deliver, event)

        try:
            while True:
                try:
                    response = await asyncio.to_thread(request_client._complete_sync, prompt, receive)
                    self._model_cooldowns.pop(request_client.model, None)
                    return response
                except (ResponsesRetryableError, ResponsesModelUnavailableError) as exc:
                    if (
                        isinstance(exc, ResponsesModelUnavailableError)
                        or (isinstance(exc, ResponsesFirstDataTimeout) and len(models) > 1)
                        or retries >= max_retries
                    ):
                        cooldown = self._config.fallback_cooldown_seconds
                        if len(models) > 1 and cooldown > 0:
                            self._model_cooldowns[request_client.model] = (
                                monotonic() + cooldown
                            )
                        fallback_model = next(fallback_models, None)
                        if fallback_model is None:
                            raise
                        event_handler(
                            StreamErrorEvent(
                                f"Falling back: {request_client.model} -> {fallback_model}",
                                0,
                                max_retries,
                                0.0,
                                str(exc),
                            )
                        )
                        request_client = self.with_overrides(model=fallback_model)
                        retries = 0
                        continue
                    retries += 1
                    delay_seconds = exc.retry_delay_seconds
                    if delay_seconds is None:
                        delay_seconds = self._retry_delay_seconds(retries)
                    event_handler(
                        StreamErrorEvent(
                            f"Reconnecting... {retries}/{max_retries}",
                            retries,
                            max_retries,
                            delay_seconds,
                            str(exc),
                        )
                    )
                    if delay_seconds > 0:
                        await asyncio.sleep(delay_seconds)
        finally:
            with callback_lock:
                active = False

    def _complete_sync(
        self,
        prompt: "Prompt",
        event_handler: "ModelStreamEventHandler",
    ) -> "ModelResponse":
        payload = self._build_payload(prompt)
        session_id = payload["prompt_cache_key"]
        body = json.dumps(payload).encode("utf-8")
        url = self.responses_url()
        prepared = requests.PreparedRequest()
        prepared.prepare(
            method="POST",
            url=url,
            headers=self._build_headers(prompt),
            data=body,
        )
        diagnostics = _StreamDiagnostics()
        completed = False
        first_data_ms = self._config.stream_first_data_timeout_ms
        first_timeout = first_data_ms / 1000.0 if first_data_ms is not None else None
        try:
            with requests.Session() as session:
                settings = session.merge_environment_settings(
                    prepared.url,
                    proxies={},
                    stream=True,
                    verify=None,
                    cert=None,
                )
                verify = _requests_verify_setting()
                if verify is not None:
                    settings["verify"] = verify
                timeout = (
                    min(max(self._timeout_seconds, 1.0), first_timeout)
                    if first_timeout is not None else max(self._timeout_seconds, 1.0),
                    first_timeout if first_timeout is not None
                    else self._config.effective_stream_idle_timeout_seconds(),
                )
                response = session.send(
                    prepared,
                    timeout=timeout,
                    allow_redirects=False,
                    **settings,
                )
                with response:
                    if response.status_code >= 400:
                        error_body = response.text
                        message = (
                            f"responses request failed with status {response.status_code}: "
                            f"{error_body[:500]}"
                        )
                        if _is_context_length_error_message(error_body):
                            raise ContextLengthExceeded(message)
                        if response.status_code >= 500 or response.status_code in (408, 429):
                            raise ResponsesRetryableError(
                                message,
                                retry_delay_seconds=self._try_parse_retry_after_seconds(
                                    "rate_limit_exceeded" if response.status_code == 429 else "",
                                    error_body,
                                ),
                            )
                        if response.status_code in (403, 404):
                            raise ResponsesModelUnavailableError(message)
                        raise ResponsesApiError(message)
                    if first_timeout is not None:
                        remaining = first_timeout - (monotonic() - diagnostics.started_at)
                        if remaining <= 0:
                            raise ResponsesFirstDataTimeout(
                                "responses first-data timeout after {:.1f}s".format(first_timeout)
                            )
                        _set_response_read_timeout(response, remaining)
                        lines = _stream_lines(self._track_stream_chunks(
                            response.iter_content(chunk_size=1, decode_unicode=False),
                            diagnostics, response,
                        ))
                    else:
                        lines = response.iter_lines(chunk_size=1, decode_unicode=False)
                    tracked_lines = self._track_stream_lines(lines, diagnostics)
                    result = self._parse_stream(
                        tracked_lines,
                        event_handler,
                        diagnostics=diagnostics,
                        session_id=session_id,
                    )
                    completed = True
                    return result
        except requests.RequestException as exc:
            error_type = ResponsesRetryableError
            if (
                first_timeout is not None and diagnostics.first_data_seconds is None
                and (
                    isinstance(exc, requests.Timeout)
                    or isinstance(exc.__context__, ReadTimeoutError)
                    or any(isinstance(arg, ReadTimeoutError) for arg in exc.args)
                )
            ):
                error_type = ResponsesFirstDataTimeout
            error = error_type(self._format_transport_error(url, exc, diagnostics, session_id))
            error.elapsed_seconds = monotonic() - diagnostics.started_at
            error.first_data_seconds = diagnostics.first_data_seconds
            raise error from exc
        except ResponsesApiError as exc:
            exc.elapsed_seconds = monotonic() - diagnostics.started_at
            exc.first_data_seconds = diagnostics.first_data_seconds
            raise
        finally:
            event_handler(ModelRequestTimingEvent(
                self.model, monotonic() - diagnostics.started_at,
                diagnostics.first_data_seconds, "completed" if completed else "failed",
            ))

    @staticmethod
    def _serialize_tools(
        tools: "typing.Iterable[ToolSpec]", use_responses_lite: "bool",
    ) -> "typing.List[JSONDict]":
        declarations = []
        namespaces = {}
        for tool in tools:
            declaration = tool.serialize()
            if not tool.namespace:
                declarations.append(declaration)
                continue
            if tool.tool_type not in ("function", "custom"):
                raise ValueError("tool namespaces only contain function/custom tools")
            if tool.namespace not in namespaces:
                namespace = {
                    "type": "namespace",
                    "name": tool.namespace,
                    "description": tool.namespace_description,
                    "tools": [],
                }
                namespaces[tool.namespace] = namespace
                declarations.append(namespace)
            namespace = namespaces[tool.namespace]
            if not namespace["description"].strip():
                namespace["description"] = tool.namespace_description
            namespace["tools"].append(declaration)
        for namespace in namespaces.values():
            namespace["tools"].sort(key=lambda tool: tool["name"])
        if not use_responses_lite:
            return declarations

        functions = {
            "type": "namespace", "name": "functions", "description": "", "tools": [],
        }
        functions_index = None
        payload = []
        for declaration in declarations:
            if declaration["type"] in ("function", "custom"):
                functions["tools"].append(declaration)
            elif (
                declaration["type"] == "namespace"
                and declaration["name"] == "functions"
            ):
                functions["tools"].extend(declaration["tools"])
                if declaration["description"].strip():
                    functions["description"] = declaration["description"]
            else:
                payload.append(declaration)
                continue
            if functions_index is None:
                functions_index = len(payload)
        if functions["tools"]:
            payload.insert(functions_index, functions)
        return payload

    def _build_payload(self, prompt: "Prompt") -> "typing.Dict[str, object]":
        session_id = self._session_id if prompt.session_id is None else prompt.session_id
        use_responses_lite = self._config.use_responses_lite()
        input_items = [item.serialize() for item in prompt.input]
        _filter_audio_inputs(input_items, model_metadata(self.model))
        if use_responses_lite:
            _strip_image_details(input_items)

        tools = self._serialize_tools(prompt.tools, use_responses_lite)
        payload: "typing.Dict[str, object]" = {
            "model": self.model,
            "input": input_items,
            "parallel_tool_calls": prompt.parallel_tool_calls
            and not use_responses_lite,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": session_id,
        }
        prefix_namespace = uuid.uuid5(uuid.NAMESPACE_OID, session_id)
        prefix: "typing.List[typing.Dict[str, object]]" = []
        if use_responses_lite:
            prefix.append(
                {
                    "type": "additional_tools",
                    "id": "at_"
                    + str(
                        uuid.uuid5(
                            prefix_namespace,
                            json.dumps(
                                tools, ensure_ascii=False, separators=(",", ":")
                            ),
                        )
                    ),
                    "role": "developer",
                    "tools": tools,
                }
            )
        if prompt.base_instructions:
            prefix.append(
                {
                    "type": "message",
                    "id": "msg_"
                    + str(uuid.uuid5(prefix_namespace, prompt.base_instructions)),
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt.base_instructions,
                        }
                    ],
                }
            )
        payload["input"] = prefix + input_items
        if not use_responses_lite:
            payload["tools"] = tools

        if prompt.tools or use_responses_lite:
            payload["tool_choice"] = "auto"

        reasoning: "typing.Dict[str, str]" = {}
        reasoning_effort = self._config.effective_reasoning_effort()
        reasoning_summary = self._config.effective_reasoning_summary()
        if reasoning_effort is not None:
            reasoning["effort"] = reasoning_effort
        if reasoning_summary is not None:
            reasoning["summary"] = reasoning_summary
        if use_responses_lite and reasoning:
            reasoning["context"] = "all_turns"
        if reasoning:
            payload["reasoning"] = reasoning

        text = None
        verbosity = self._config.effective_verbosity()
        if verbosity is not None:
            text = {"verbosity": verbosity}
        if text is not None:
            payload["text"] = text

        service_tier = self._config.effective_service_tier()
        if service_tier is not None:
            payload["service_tier"] = service_tier

        return payload

    def _list_models_sync(self) -> "typing.List[str]":
        prepared = requests.PreparedRequest()
        prepared.prepare(
            method="GET",
            url=self.models_url(),
            headers=self._build_model_list_headers(),
        )
        try:
            with requests.Session() as session:
                settings = session.merge_environment_settings(
                    prepared.url,
                    proxies={},
                    stream=False,
                    verify=None,
                    cert=None,
                )
                verify = _requests_verify_setting()
                if verify is not None:
                    settings["verify"] = verify
                response = session.send(
                    prepared,
                    timeout=self._timeout_seconds,
                    allow_redirects=False,
                    **settings,
                )
                with response:
                    if response.status_code >= 400:
                        raise ResponsesApiError(
                            f"models request failed with status {response.status_code}: "
                            f"{response.text[:500]}"
                        )
                    payload = response.json()
        except requests.RequestException as exc:
            raise ResponsesApiError(f"models request failed: {exc}") from exc

        data = payload.get("data")
        if not isinstance(data, list):
            raise ResponsesApiError("models response is missing `data` list")
        models: "typing.List[str]" = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id", "")).strip()
            if model_id:
                models.append(model_id)
        return models

    def _build_headers(self, prompt: "Prompt") -> "typing.Dict[str, str]":
        session_id = self._session_id if prompt.session_id is None else prompt.session_id
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream",
            "x-client-request-id": session_id,
            "session_id": session_id,
            "originator": self.recording_metadata.originator,
            "user-agent": self._user_agent,
        }
        api_key = self._config.api_key()
        if api_key is not None:
            headers["authorization"] = f"Bearer {api_key}"
        if self._config.beta_features_header is not None:
            headers["x-codex-beta-features"] = self._config.beta_features_header
        if self._config.use_responses_lite():
            headers[RESPONSES_LITE_HEADER] = "true"
        if self._openai_subagent is not None:
            headers["x-openai-subagent"] = self._openai_subagent
        if prompt.turn_metadata is not None:
            headers["x-codex-turn-metadata"] = json.dumps(
                prompt.turn_metadata,
                separators=(",", ":"),
            )
        return headers

    def _build_model_list_headers(self) -> "typing.Dict[str, str]":
        headers = {
            "accept": "application/json",
            "originator": self.recording_metadata.originator,
            "user-agent": self._user_agent,
        }
        api_key = self._config.api_key()
        if api_key is not None:
            headers["authorization"] = f"Bearer {api_key}"
        if self._config.beta_features_header is not None:
            headers["x-codex-beta-features"] = self._config.beta_features_header
        if self._openai_subagent is not None:
            headers["x-openai-subagent"] = self._openai_subagent
        return headers

    def _parse_stream(
        self,
        response,
        event_handler: "ModelStreamEventHandler",
        diagnostics: "typing.Union[_StreamDiagnostics, None]" = None,
        session_id: "typing.Union[str, None]" = None,
    ) -> "ModelResponse":
        items: "typing.List[typing.Union[typing.Union[AssistantMessage, ToolCall], ReasoningItem]]" = ([])
        saw_completed = False
        last_event_type = ""

        for event_name, data in self._iter_sse_events(response, diagnostics):
            if not data:
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ResponsesRetryableError(
                    self._format_invalid_event_error(event_name, data, exc, session_id)
                ) from exc
            event_type = payload.get("type", event_name)
            last_event_type = str(event_type)
            if diagnostics is not None:
                diagnostics.last_event_type = last_event_type

            if event_type == "response.output_text.delta":
                event_handler(AssistantDeltaEvent(str(payload.get("delta", ""))))
                continue

            if event_type == "response.output_item.done":
                item_payload = payload.get("item", {})
                if (
                    isinstance(item_payload, dict)
                    and item_payload.get("type") == "web_search_call"
                ):
                    action_payload = item_payload.get("action")
                    action_type = None
                    if isinstance(action_payload, dict):
                        action_type = str(action_payload.get("type", ""))
                    else:
                        action_payload = {}
                    queries = action_payload.get("queries")
                    event_handler(
                        ToolCalledEvent(
                            call_id=str(item_payload.get("id", "web_search")),
                            tool_name="web_search",
                            action_type=action_type,
                            query=(
                                str(action_payload["query"])
                                if "query" in action_payload
                                else None
                            ),
                            queries=(
                                tuple(
                                    str(query)
                                    for query in queries
                                    if str(query).strip()
                                )
                                if isinstance(queries, list)
                                else None
                            ),
                            url=(
                                str(action_payload["url"])
                                if "url" in action_payload
                                else None
                            ),
                            pattern=(
                                str(action_payload["pattern"])
                                if "pattern" in action_payload
                                else None
                            ),
                        )
                    )
                    continue

                parsed = self._parse_output_item(item_payload)
                if parsed is not None:
                    if isinstance(parsed, ToolCall):
                        event_handler(ToolCalledEvent(parsed.call_id, parsed.name))
                    items.append(parsed)
                    if diagnostics is not None:
                        diagnostics.output_items_received += 1
                continue

            if event_type == "response.completed":
                response_payload = payload.get("response")
                usage = None
                if isinstance(response_payload, dict):
                    response_usage = response_payload.get("usage")
                    if isinstance(response_usage, dict):
                        usage = dict(response_usage)
                elif isinstance(payload.get("usage"), dict):
                    usage = dict(payload["usage"])
                event_handler(TokenCountEvent(usage))
                saw_completed = True
                break

            if event_type == "response.incomplete":
                partial_items = tuple(items)
                reason = self._response_incomplete_reason(payload)
                raise ResponsesIncompleteError(
                    self._format_response_incomplete_error(
                        payload,
                        len(partial_items),
                        reason,
                        session_id,
                    ),
                    partial_items,
                    reason=reason,
                )

            if event_type == "response.failed":
                self._raise_response_failed_error(payload, session_id)

        if not saw_completed:
            raise ResponsesRetryableError(
                self._format_incomplete_stream_error(last_event_type, len(items), session_id)
            )

        return ModelResponse(items=items)

    @staticmethod
    def _parse_output_item(
        item: "typing.Dict[str, object]",
    ) -> "typing.Union[typing.Union[typing.Union[AssistantMessage, ToolCall], ReasoningItem], None]":
        item_type = item.get("type")
        if item_type == "reasoning":
            return ReasoningItem(payload=dict(item))

        if item_type == "message" and item.get("role") == "assistant":
            return AssistantMessage.from_response_item(item)

        if item_type in ("function_call", "custom_tool_call"):
            name = str(item["name"])
            if "." in name:
                raise ResponsesApiError("tool call names must not contain '.'")
            namespace = item.get("namespace")
            name = qualified_tool_name(name, namespace)

        if item_type == "function_call":
            raw_arguments = item["arguments"]
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ResponsesApiError(
                    f"function call arguments must decode to an object, got {type(arguments).__name__}"
                )
            return ToolCall(
                call_id=str(item["call_id"]),
                name=name,
                arguments=arguments,
                id=item.get("id"),
                raw_arguments=raw_arguments,
                namespace=item.get("namespace"),
                encrypted_function_args=deepcopy(item.get("encrypted_function_args")),
            )

        if item_type == "custom_tool_call":
            return ToolCall(
                call_id=str(item["call_id"]),
                name=name,
                arguments=str(item.get("input", "")),
                tool_type="custom",
                id=item.get("id"),
                namespace=item.get("namespace"),
                status=item.get("status"),
            )

        return None

    def _iter_sse_events(
        self,
        response,
        diagnostics: "typing.Union[_StreamDiagnostics, None]" = None,
    ):
        event_name: "typing.Union[str, None]" = None
        data_lines: "typing.List[str]" = []

        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == "":
                if data_lines:
                    resolved_event_name = event_name or "message"
                    payload = "\n".join(data_lines)
                    if diagnostics is not None:
                        diagnostics.sse_events_received += 1
                        diagnostics.last_sse_event_name = resolved_event_name
                        diagnostics.last_payload_excerpt = self._truncate_excerpt(
                            payload,
                            240,
                        )
                    yield resolved_event_name, payload
                event_name = None
                data_lines = []
                continue

            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].lstrip()
                continue
            if line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())

        if data_lines:
            resolved_event_name = event_name or "message"
            payload = "\n".join(data_lines)
            if diagnostics is not None:
                diagnostics.sse_events_received += 1
                diagnostics.last_sse_event_name = resolved_event_name
                diagnostics.last_payload_excerpt = self._truncate_excerpt(
                    payload,
                    240,
                )
            yield resolved_event_name, payload

    def _track_stream_lines(
        self,
        response,
        diagnostics: "_StreamDiagnostics",
    ):
        for raw_line in response:
            if diagnostics.first_data_seconds is None:
                diagnostics.first_data_seconds = monotonic() - diagnostics.started_at
            diagnostics.raw_lines_received += 1
            yield raw_line

    def _track_stream_chunks(self, chunks, diagnostics, response):
        for chunk in chunks:
            if chunk and diagnostics.first_data_seconds is None:
                diagnostics.first_data_seconds = monotonic() - diagnostics.started_at
                _set_response_read_timeout(
                    response, self._config.effective_stream_idle_timeout_seconds()
                )
            yield chunk

    def _base_error_details(
        self,
        url: "str",
        session_id: "typing.Union[str, None]" = None,
    ) -> "typing.List[typing.Tuple[str, str]]":
        return [
            ("provider", self._config.provider_name),
            ("model", self.model),
            ("request", f"POST {url}"),
            ("session_id", self._session_id if session_id is None else session_id),
        ]

    @staticmethod
    def _format_error_message(
        summary: "str",
        details: "typing.Iterable[typing.Tuple[str, str]]",
    ) -> "str":
        lines = [summary]
        for label, value in details:
            text = str(value).strip()
            if not text:
                continue
            lines.append(f"- {label}: {text}")
        return "\n".join(lines)

    def _format_transport_error(
        self,
        url: "str",
        exc: "BaseException",
        diagnostics: "typing.Union[_StreamDiagnostics, None]" = None,
        session_id: "typing.Union[str, None]" = None,
    ) -> "str":
        details = self._base_error_details(url, session_id)
        if diagnostics is not None:
            details.extend(self._transport_diagnostics_details(diagnostics))
            details.append(("elapsed_seconds", round(monotonic() - diagnostics.started_at, 3)))
            details.append(("first_data_seconds", diagnostics.first_data_seconds))
        details.append(("exception", type(exc).__name__))
        details.append(("detail", str(exc) or repr(exc)))
        details.append(
            (
                "meaning",
                "the HTTP response body ended before the SSE stream finished",
            )
        )
        details.append(
            (
                "hint",
                "the server or a proxy likely closed the connection before sending "
                "`response.completed` or `response.failed`",
            )
        )
        hostname = urllib.parse.urlparse(url).hostname or ""
        if hostname in {"127.0.0.1", "localhost"}:
            details.append(
                (
                    "hint",
                    "if this goes through local `responses_server`, inspect that "
                    "server's stderr/logs for the downstream backend failure",
                )
            )
        return self._format_error_message(
            "responses request failed while reading the HTTP stream",
            details,
        )

    def _format_response_failed_error(
        self,
        message: "str",
        code: "str" = "",
        session_id: "typing.Union[str, None]" = None,
    ) -> "str":
        details = self._base_error_details(self.responses_url(), session_id)
        details.append(("detail", message))
        if code:
            details.append(("code", code))
        details.append(
            (
                "meaning",
                "the server accepted the request but emitted a terminal "
                "`response.failed` event",
            )
        )
        return self._format_error_message(
            "responses stream failed on the server side",
            details,
        )

    def _format_response_incomplete_error(
        self,
        payload: "typing.Dict[str, object]",
        output_item_count: "int",
        reason: "typing.Union[str, None]" = None,
        session_id: "typing.Union[str, None]" = None,
    ) -> "str":
        details = self._base_error_details(self.responses_url(), session_id)
        if reason:
            details.append(("reason", reason))
        details.append(("output_items_received", str(output_item_count)))
        details.append(
            (
                "meaning",
                "the server ended the response with `response.incomplete` before "
                "a terminal `response.completed` event",
            )
        )
        return self._format_error_message(
            "responses stream ended with `response.incomplete`",
            details,
        )

    def _response_incomplete_reason(self, payload: "typing.Dict[str, object]") -> "str":
        response = payload.get("response")
        incomplete_details = (
            response.get("incomplete_details") if isinstance(response, dict) else None
        )
        if not isinstance(incomplete_details, dict):
            return ""
        return str(incomplete_details.get("reason") or "").strip()

    def _raise_response_failed_error(
        self, payload: "typing.Dict[str, object]", session_id=None,
    ) -> "None":
        response = payload.get("response")
        error = response.get("error") if isinstance(response, dict) else None
        if not isinstance(error, dict):
            raise ResponsesRetryableError(
                self._format_response_failed_error("responses stream failed", session_id=session_id)
            )

        message = str(error.get("message") or "responses stream failed")
        code = str(error.get("code") or error.get("type") or "").strip()
        formatted = self._format_response_failed_error(message, code, session_id)
        if code == "context_length_exceeded" or _is_context_length_error_message(
            message
        ):
            raise ContextLengthExceeded(formatted)
        if code in {"model_not_found", "model_not_available", "permission_denied"}:
            raise ResponsesModelUnavailableError(formatted)
        if code in {
            "insufficient_quota",
            "invalid_prompt",
            "model_output_invalid",
            "usage_not_included",
        }:
            raise ResponsesApiError(formatted)

        raise ResponsesRetryableError(
            formatted,
            retry_delay_seconds=self._try_parse_retry_after_seconds(code, message),
        )

    def _format_incomplete_stream_error(
        self,
        last_event_type: "str",
        output_item_count: "int",
        session_id: "typing.Union[str, None]" = None,
    ) -> "str":
        details = self._base_error_details(self.responses_url(), session_id)
        if last_event_type:
            details.append(("last_event", last_event_type))
        details.append(("output_items_received", str(output_item_count)))
        details.append(
            (
                "meaning",
                "the stream ended without a terminal `response.completed` event",
            )
        )
        details.append(
            (
                "hint",
                "the server should emit `response.failed` on mid-stream errors; "
                "an abrupt end usually points to a backend, proxy, or server bug",
            )
        )
        return self._format_error_message(
            "responses stream ended before `response.completed`",
            details,
        )

    def _format_invalid_event_error(
        self,
        event_name: "str",
        raw_data: "str",
        exc: "json.JSONDecodeError",
        session_id: "typing.Union[str, None]" = None,
    ) -> "str":
        details = self._base_error_details(self.responses_url(), session_id)
        details.append(("event", event_name or "message"))
        details.append(("exception", type(exc).__name__))
        details.append(("detail", str(exc)))
        excerpt = raw_data if len(raw_data) <= 240 else f"{raw_data[:240]}..."
        details.append(("data_excerpt", excerpt))
        return self._format_error_message(
            "responses stream contained an invalid JSON event",
            details,
        )

    def _transport_diagnostics_details(
        self,
        diagnostics: "_StreamDiagnostics",
    ) -> "typing.List[typing.Tuple[str, str]]":
        details: "typing.List[typing.Tuple[str, str]]" = [
            ("raw_lines_received", str(diagnostics.raw_lines_received)),
            ("sse_events_received", str(diagnostics.sse_events_received)),
            ("output_items_received", str(diagnostics.output_items_received)),
        ]
        if diagnostics.last_sse_event_name:
            details.append(("last_sse_event", diagnostics.last_sse_event_name))
        if diagnostics.last_event_type:
            details.append(("last_event_type", diagnostics.last_event_type))
        if diagnostics.last_payload_excerpt:
            details.append(("last_payload_excerpt", diagnostics.last_payload_excerpt))
        return details

    def _truncate_excerpt(self, text: "str", limit: "int") -> "str":
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."

    def _retry_delay_seconds(self, attempt: "int") -> "float":
        return INITIAL_RETRY_DELAY_SECONDS * (
            RETRY_BACKOFF_FACTOR ** max(min(attempt - 1, 10), 0)
        )  # 200s max

    def _try_parse_retry_after_seconds(
        self,
        code: "str",
        message: "str",
    ) -> "typing.Union[float, None]":
        if code != "rate_limit_exceeded":
            return None
        match = RATE_LIMIT_RETRY_AFTER_RE.search(message)
        if match is None:
            return None
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit == "ms":
            return value / 1000.0
        return value


def _set_response_read_timeout(response, seconds):
    # Keep-alive responses expose the socket on the urllib3 connection. For
    # Connection: close, http.client retains it on the response file instead.
    raw = getattr(response, "raw", None)
    connection = getattr(raw, "connection", None)
    sock = getattr(connection, "sock", None)
    if sock is None:
        fp = getattr(getattr(raw, "_fp", None), "fp", None)
        sock = getattr(getattr(fp, "raw", None), "_sock", None)
    if sock is not None:
        sock.settimeout(seconds)


def _stream_lines(chunks):
    pending = None
    for chunk in chunks:
        if pending is not None:
            chunk = pending + chunk
        lines = chunk.splitlines()
        if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]:
            pending = lines.pop()
        else:
            pending = None
        yield from lines
    if pending is not None:
        yield pending


def _optional_int(value: "object") -> "typing.Union[int, None]":
    if value is None:
        return None
    return int(value)


def _is_context_length_error_message(message: "str") -> "bool":
    lower = message.lower()
    return (
        "context_length_exceeded" in lower
        or "maximum context length" in lower
        or "exceeds the context window" in lower
        or "exceeded the context window" in lower
    )


def _requests_verify_setting() -> "typing.Union[typing.Union[str, bool], None]":
    for env_name in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return None
