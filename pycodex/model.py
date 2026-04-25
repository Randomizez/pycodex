
import asyncio
import json
import os
import re
import urllib.parse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable
from .compat import Protocol

import requests
import typing

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 path
    import tomli as tomllib

from .protocol import (
    AssistantMessage,
    ModelResponse,
    ModelStreamEvent,
    Prompt,
    ReasoningItem,
    ToolCall,
)
from .utils import build_user_agent, uuid7_string

DEFAULT_CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"
DEFAULT_ORIGINATOR = "pycodex"
ModelStreamEventHandler = Callable[[ModelStreamEvent], None]
NOOP_MODEL_STREAM_EVENT_HANDLER: 'ModelStreamEventHandler' = lambda _event: None
DEFAULT_STREAM_MAX_RETRIES = 5
DEFAULT_STREAM_IDLE_TIMEOUT_MS = 300_000
INITIAL_RETRY_DELAY_SECONDS = 0.2
RETRY_BACKOFF_FACTOR = 2.0
RATE_LIMIT_RETRY_AFTER_RE = re.compile(
    r"(?i)try again in\s*(\d+(?:\.\d+)?)\s*(s|ms|seconds?)"
)


class ModelClient(Protocol):
    async def complete(
        self,
        prompt: 'Prompt',
        event_handler: 'ModelStreamEventHandler' = NOOP_MODEL_STREAM_EVENT_HANDLER,
    ) -> 'ModelResponse':
        """Return the next batch of model output items for the current prompt."""


@dataclass(frozen=True, )
class ResponsesProviderConfig:
    model: 'str'
    provider_name: 'str'
    base_url: 'str'
    api_key_env: 'typing.Union[str, None]'
    wire_api: 'str' = "responses"
    query_params: 'typing.Dict[str, str]' = field(default_factory=dict)
    reasoning_effort: 'typing.Union[str, None]' = None
    reasoning_summary: 'typing.Union[str, None]' = None
    verbosity: 'typing.Union[str, None]' = None
    sandbox_mode: 'typing.Union[str, None]' = None
    beta_features_header: 'typing.Union[str, None]' = None
    stream_max_retries: 'typing.Union[int, None]' = None
    stream_idle_timeout_ms: 'typing.Union[int, None]' = None

    @classmethod
    def from_codex_config(
        cls,
        config_path: 'typing.Union[str, Path]' = DEFAULT_CODEX_CONFIG_PATH,
        profile: 'typing.Union[str, None]' = None,
    ) -> 'ResponsesProviderConfig':
        data = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
        selected = dict(data)
        if profile is not None:
            overrides = data.get("profiles", {}).get(profile)
            if overrides is None:
                raise ValueError(f"unknown Codex profile: {profile}")
            selected.update(overrides)

        provider_name = selected["model_provider"]
        provider = data["model_providers"][provider_name]
        wire_api = provider.get("wire_api", "responses")
        if wire_api != "responses":
            raise ValueError(f"unsupported wire_api for Python client: {wire_api}")

        api_key_env = provider.get("env_key")

        query_params = {
            str(key): str(value)
            for key, value in provider.get("query_params", {}).items()
        }
        features = selected.get("features", {})
        beta_features: 'typing.List[str]' = []
        if isinstance(features, dict) and features.get("guardian_approval") is True:
            beta_features.append("guardian_approval")
        return cls(
            model=selected["model"],
            provider_name=provider_name,
            base_url=provider["base_url"],
            api_key_env=api_key_env,
            wire_api=wire_api,
            query_params=query_params,
            reasoning_effort=selected.get("model_reasoning_effort"),
            reasoning_summary=selected.get("model_reasoning_summary"),
            verbosity=selected.get("model_verbosity"),
            sandbox_mode=selected.get("sandbox_mode"),
            beta_features_header=",".join(beta_features) or None,
            stream_max_retries=_optional_int(provider.get("stream_max_retries")),
            stream_idle_timeout_ms=_optional_int(provider.get("stream_idle_timeout_ms")),
        )

    def api_key(self) -> 'typing.Union[str, None]':
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
        model: 'typing.Union[str, None]' = None,
        reasoning_effort: 'typing.Union[str, None]' = None,
    ) -> 'ResponsesProviderConfig':
        return replace(
            self,
            model=self.model if model is None else model,
            reasoning_effort=(
                self.reasoning_effort
                if reasoning_effort is None
                else reasoning_effort
            ),
        )

    def effective_stream_max_retries(self) -> 'int':
        if self.stream_max_retries is None:
            return DEFAULT_STREAM_MAX_RETRIES
        return max(int(self.stream_max_retries), 0)

    def effective_stream_idle_timeout_seconds(self) -> 'float':
        if self.stream_idle_timeout_ms is None:
            return DEFAULT_STREAM_IDLE_TIMEOUT_MS / 1000.0
        return max(int(self.stream_idle_timeout_ms), 1) / 1000.0


class ResponsesApiError(RuntimeError):
    pass


class ResponsesRetryableError(ResponsesApiError):
    def __init__(
        self,
        message: 'str',
        retry_delay_seconds: 'typing.Union[float, None]' = None,
    ) -> 'None':
        super().__init__(message)
        self.retry_delay_seconds = retry_delay_seconds


@dataclass
class _StreamDiagnostics:
    raw_lines_received: 'int' = 0
    sse_events_received: 'int' = 0
    output_items_received: 'int' = 0
    last_sse_event_name: 'str' = ""
    last_event_type: 'str' = ""
    last_payload_excerpt: 'str' = ""


class ResponsesModelClient:
    """Minimal OpenAI-compatible Responses API client.

    This implementation is intentionally narrow: it supports the subset needed
    by the current AgentLoop abstraction, namely assistant text and function
    tool calls over the streaming `/responses` endpoint.
    """

    def __init__(
        self,
        config: 'ResponsesProviderConfig',
        timeout_seconds: 'float' = 120.0,
        session_id: 'typing.Union[str, None]' = None,
        originator: 'str' = DEFAULT_ORIGINATOR,
        user_agent: 'typing.Union[str, None]' = None,
        openai_subagent: 'typing.Union[str, None]' = None,
    ) -> 'None':
        self._config = config
        self.model = config.model
        self._timeout_seconds = timeout_seconds
        self._session_id = session_id or uuid7_string()
        self._originator = originator
        self._user_agent = user_agent or build_user_agent(originator)
        self._openai_subagent = openai_subagent

    @classmethod
    def from_codex_config(
        cls,
        config_path: 'typing.Union[str, Path]' = DEFAULT_CODEX_CONFIG_PATH,
        profile: 'typing.Union[str, None]' = None,
        timeout_seconds: 'float' = 120.0,
        originator: 'str' = DEFAULT_ORIGINATOR,
        user_agent: 'typing.Union[str, None]' = None,
    ) -> 'ResponsesModelClient':
        config = ResponsesProviderConfig.from_codex_config(config_path, profile)
        return cls(config, timeout_seconds, originator=originator, user_agent=user_agent)

    def with_overrides(
        self,
        model: 'typing.Union[str, None]' = None,
        reasoning_effort: 'typing.Union[str, None]' = None,
        session_id: 'typing.Union[str, None]' = None,
        openai_subagent: 'typing.Union[str, None]' = None,
    ) -> 'ResponsesModelClient':
        return ResponsesModelClient(
            self._config.with_overrides(
                model or self.model,
                reasoning_effort,
            ),
            self._timeout_seconds,
            session_id=self._session_id if session_id is None else session_id,
            originator=self._originator,
            user_agent=self._user_agent,
            openai_subagent=(
                self._openai_subagent
                if openai_subagent is None
                else openai_subagent
            ),
        )

    def responses_url(self) -> 'str':
        base_url = self._config.base_url.rstrip("/")
        url = f"{base_url}/responses"
        if self._config.query_params:
            return f"{url}?{urllib.parse.urlencode(self._config.query_params)}"
        return url

    def models_url(self) -> 'str':
        base_url = self._config.base_url.rstrip("/")
        url = f"{base_url}/models"
        if self._config.query_params:
            return f"{url}?{urllib.parse.urlencode(self._config.query_params)}"
        return url

    async def list_models(self) -> 'typing.List[str]':
        return await asyncio.to_thread(self._list_models_sync)

    async def complete(
        self,
        prompt: 'Prompt',
        event_handler: 'ModelStreamEventHandler' = NOOP_MODEL_STREAM_EVENT_HANDLER,
    ) -> 'ModelResponse':
        retries = 0
        max_retries = self._config.effective_stream_max_retries()
        while True:
            try:
                return await asyncio.to_thread(
                    self._complete_sync,
                    prompt,
                    event_handler,
                )
            except ResponsesRetryableError as exc:
                if retries >= max_retries:
                    raise
                retries += 1
                delay_seconds = exc.retry_delay_seconds
                if delay_seconds is None:
                    delay_seconds = self._retry_delay_seconds(retries)
                event_handler(
                    ModelStreamEvent(
                        kind="stream_error",
                        payload={
                            "message": f"Reconnecting... {retries}/{max_retries}",
                            "attempt": retries,
                            "max_retries": max_retries,
                            "delay_seconds": delay_seconds,
                            "error": str(exc),
                        },
                    )
                )
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)

    def _complete_sync(
        self,
        prompt: 'Prompt',
        event_handler: 'ModelStreamEventHandler',
    ) -> 'ModelResponse':
        payload = self._build_payload(prompt)
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
                    max(self._timeout_seconds, 1.0),
                    self._config.effective_stream_idle_timeout_seconds(),
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
                        if response.status_code >= 500:
                            raise ResponsesRetryableError(message)
                        raise ResponsesApiError(message)
                    tracked_lines = self._track_stream_lines(
                        response.iter_lines(chunk_size=1, decode_unicode=False),
                        diagnostics,
                    )
                    return self._parse_stream(
                        tracked_lines,
                        event_handler,
                        diagnostics=diagnostics,
                    )
        except requests.RequestException as exc:
            raise ResponsesRetryableError(
                self._format_transport_error(url, exc, diagnostics)
            ) from exc

    def _build_payload(self, prompt: 'Prompt') -> 'typing.Dict[str, object]':
        payload: 'typing.Dict[str, object]' = {
            "model": self.model,
            "instructions": prompt.base_instructions or "",
            "input": [item.serialize() for item in prompt.input],
            "tools": [tool.serialize() for tool in prompt.tools],
            "parallel_tool_calls": prompt.parallel_tool_calls,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": self._session_id,
        }
        if prompt.tools:
            payload["tool_choice"] = "auto"

        reasoning: 'typing.Dict[str, str]' = {}
        if self._config.reasoning_effort is not None:
            reasoning["effort"] = self._config.reasoning_effort
        if self._config.reasoning_summary is not None:
            reasoning["summary"] = self._config.reasoning_summary
        if reasoning:
            payload["reasoning"] = reasoning

        text = None
        if self._config.verbosity is not None:
            text = {"verbosity": self._config.verbosity}
        if text is not None:
            payload["text"] = text

        return payload

    def _list_models_sync(self) -> 'typing.List[str]':
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
        models: 'typing.List[str]' = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id", "")).strip()
            if model_id:
                models.append(model_id)
        return models

    def _build_headers(self, prompt: 'Prompt') -> 'typing.Dict[str, str]':
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream",
            "x-client-request-id": self._session_id,
            "session_id": self._session_id,
            "originator": self._originator,
            "user-agent": self._user_agent,
        }
        api_key = self._config.api_key()
        if api_key is not None:
            headers["authorization"] = f"Bearer {api_key}"
        if self._config.beta_features_header is not None:
            headers["x-codex-beta-features"] = self._config.beta_features_header
        if self._openai_subagent is not None:
            headers["x-openai-subagent"] = self._openai_subagent
        if prompt.turn_metadata is not None:
            headers["x-codex-turn-metadata"] = json.dumps(
                prompt.turn_metadata,
                separators=(",", ":"),
            )
        return headers

    def _build_model_list_headers(self) -> 'typing.Dict[str, str]':
        headers = {
            "accept": "application/json",
            "originator": self._originator,
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
        event_handler: 'ModelStreamEventHandler',
        diagnostics: 'typing.Union[_StreamDiagnostics, None]' = None,
    ) -> 'ModelResponse':
        items: 'typing.List[typing.Union[typing.Union[AssistantMessage, ToolCall], ReasoningItem]]' = []
        saw_completed = False
        last_event_type = ""

        for event_name, data in self._iter_sse_events(response, diagnostics):
            if not data:
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ResponsesRetryableError(
                    self._format_invalid_event_error(event_name, data, exc)
                ) from exc
            event_type = payload.get("type", event_name)
            last_event_type = str(event_type)
            if diagnostics is not None:
                diagnostics.last_event_type = last_event_type

            if event_type == "response.output_text.delta":
                event_handler(
                    ModelStreamEvent(
                        kind="assistant_delta",
                        payload={"delta": str(payload.get("delta", ""))},
                    )
                )
                continue

            if event_type == "response.output_item.done":
                item_payload = payload.get("item", {})
                if (
                    isinstance(item_payload, dict)
                    and item_payload.get("type") == "web_search_call"
                ):
                    action_payload = item_payload.get("action")
                    event_payload = {
                        "call_id": str(item_payload.get("id", "web_search")),
                        "tool_name": "web_search",
                    }
                    if isinstance(action_payload, dict):
                        event_payload["action_type"] = str(
                            action_payload.get("type", "")
                        )
                        if "query" in action_payload:
                            event_payload["query"] = str(action_payload.get("query", ""))
                        queries = action_payload.get("queries")
                        if isinstance(queries, list):
                            event_payload["queries"] = [
                                str(query) for query in queries if str(query).strip()
                            ]
                        if "url" in action_payload:
                            event_payload["url"] = str(action_payload.get("url", ""))
                        if "pattern" in action_payload:
                            event_payload["pattern"] = str(
                                action_payload.get("pattern", "")
                            )
                    event_handler(
                        ModelStreamEvent(
                            kind="tool_call",
                            payload=event_payload,
                        )
                    )
                    continue

                parsed = self._parse_output_item(item_payload)
                if parsed is not None:
                    if isinstance(parsed, ToolCall):
                        event_handler(
                            ModelStreamEvent(
                                kind="tool_call",
                                payload={
                                    "call_id": parsed.call_id,
                                    "tool_name": parsed.name,
                                },
                            )
                        )
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
                event_handler(
                    ModelStreamEvent(
                        kind="token_count",
                        payload={"usage": usage},
                    )
                )
                saw_completed = True
                break

            if event_type == "response.failed":
                self._raise_response_failed_error(payload)

        if not saw_completed:
            raise ResponsesRetryableError(
                self._format_incomplete_stream_error(last_event_type, len(items))
            )

        return ModelResponse(items=items)

    def _parse_output_item(
        self,
        item: 'typing.Dict[str, object]',
    ) -> 'typing.Union[typing.Union[typing.Union[AssistantMessage, ToolCall], ReasoningItem], None]':
        item_type = item.get("type")
        if item_type == "reasoning":
            return ReasoningItem(payload=dict(item))

        if item_type == "message" and item.get("role") == "assistant":
            content = item.get("content", [])
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text_parts.append(str(part.get("text", "")))
            return AssistantMessage(text="".join(text_parts))

        if item_type == "function_call":
            raw_arguments = str(item.get("arguments", "") or "{}")
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ResponsesApiError(
                    f"function call arguments must decode to an object, got {type(arguments).__name__}"
                )
            return ToolCall(
                call_id=str(item["call_id"]),
                name=str(item["name"]),
                arguments=arguments,
            )

        if item_type == "custom_tool_call":
            return ToolCall(
                call_id=str(item["call_id"]),
                name=str(item["name"]),
                arguments=str(item.get("input", "")),
                tool_type="custom",
            )

        return None

    def _iter_sse_events(
        self,
        response,
        diagnostics: 'typing.Union[_StreamDiagnostics, None]' = None,
    ):
        event_name: 'typing.Union[str, None]' = None
        data_lines: 'typing.List[str]' = []

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
        diagnostics: '_StreamDiagnostics',
    ):
        for raw_line in response:
            diagnostics.raw_lines_received += 1
            yield raw_line

    def _base_error_details(
        self,
        url: 'str',
    ) -> 'typing.List[typing.Tuple[str, str]]':
        return [
            ("provider", self._config.provider_name),
            ("model", self.model),
            ("request", f"POST {url}"),
            ("session_id", self._session_id),
        ]

    def _format_error_message(
        self,
        summary: 'str',
        details: 'typing.Iterable[typing.Tuple[str, str]]',
    ) -> 'str':
        lines = [summary]
        for label, value in details:
            text = str(value).strip()
            if not text:
                continue
            lines.append(f"- {label}: {text}")
        return "\n".join(lines)

    def _format_transport_error(
        self,
        url: 'str',
        exc: 'BaseException',
        diagnostics: 'typing.Union[_StreamDiagnostics, None]' = None,
    ) -> 'str':
        details = self._base_error_details(url)
        if diagnostics is not None:
            details.extend(self._transport_diagnostics_details(diagnostics))
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

    def _format_response_failed_error(self, message: 'str') -> 'str':
        details = self._base_error_details(self.responses_url())
        details.append(("detail", message))
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

    def _raise_response_failed_error(self, payload: 'typing.Dict[str, object]') -> 'None':
        response = payload.get("response")
        error = response.get("error") if isinstance(response, dict) else None
        if not isinstance(error, dict):
            raise ResponsesRetryableError(
                self._format_response_failed_error("responses stream failed")
            )

        message = str(error.get("message") or "responses stream failed")
        code = str(error.get("code") or "").strip()
        if code in {
            "context_length_exceeded",
            "insufficient_quota",
            "invalid_prompt",
            "usage_not_included",
        }:
            raise ResponsesApiError(self._format_response_failed_error(message))

        raise ResponsesRetryableError(
            self._format_response_failed_error(message),
            retry_delay_seconds=self._try_parse_retry_after_seconds(code, message),
        )

    def _format_incomplete_stream_error(
        self,
        last_event_type: 'str',
        output_item_count: 'int',
    ) -> 'str':
        details = self._base_error_details(self.responses_url())
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
        event_name: 'str',
        raw_data: 'str',
        exc: 'json.JSONDecodeError',
    ) -> 'str':
        details = self._base_error_details(self.responses_url())
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
        diagnostics: '_StreamDiagnostics',
    ) -> 'typing.List[typing.Tuple[str, str]]':
        details: 'typing.List[typing.Tuple[str, str]]' = [
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

    def _truncate_excerpt(self, text: 'str', limit: 'int') -> 'str':
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."

    def _retry_delay_seconds(self, attempt: 'int') -> 'float':
        return INITIAL_RETRY_DELAY_SECONDS * (
            RETRY_BACKOFF_FACTOR ** max(attempt - 1, 0)
        )

    def _try_parse_retry_after_seconds(
        self,
        code: 'str',
        message: 'str',
    ) -> 'typing.Union[float, None]':
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


def _optional_int(value: 'object') -> 'typing.Union[int, None]':
    if value is None:
        return None
    return int(value)


def _requests_verify_setting() -> 'typing.Union[typing.Union[str, bool], None]':
    for env_name in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return None
