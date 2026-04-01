from __future__ import annotations

import asyncio
import json
import os
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import requests

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
NOOP_MODEL_STREAM_EVENT_HANDLER: ModelStreamEventHandler = lambda _event: None


class ModelClient(Protocol):
    async def complete(
        self,
        prompt: Prompt,
        event_handler: ModelStreamEventHandler = NOOP_MODEL_STREAM_EVENT_HANDLER,
    ) -> ModelResponse:
        """Return the next batch of model output items for the current prompt."""


@dataclass(frozen=True, slots=True)
class ResponsesProviderConfig:
    model: str
    provider_name: str
    base_url: str
    api_key_env: str
    wire_api: str = "responses"
    query_params: dict[str, str] = field(default_factory=dict)
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    verbosity: str | None = None
    sandbox_mode: str | None = None
    beta_features_header: str | None = None

    @classmethod
    def from_codex_config(
        cls,
        config_path: str | Path = DEFAULT_CODEX_CONFIG_PATH,
        profile: str | None = None,
    ) -> ResponsesProviderConfig:
        data = tomllib.loads(Path(config_path).read_text())
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
        if not api_key_env:
            raise ValueError(
                f"provider {provider_name} does not define env_key in Codex config"
            )

        query_params = {
            str(key): str(value)
            for key, value in provider.get("query_params", {}).items()
        }
        features = selected.get("features", {})
        beta_features: list[str] = []
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
        )

    def api_key(self) -> str:
        value = os.environ.get(self.api_key_env, "")
        if not value:
            raise RuntimeError(
                f"missing API key environment variable: {self.api_key_env}"
            )
        return value

    def with_overrides(
        self,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> ResponsesProviderConfig:
        return replace(
            self,
            model=self.model if model is None else model,
            reasoning_effort=(
                self.reasoning_effort
                if reasoning_effort is None
                else reasoning_effort
            ),
        )


class ResponsesApiError(RuntimeError):
    pass


class ResponsesModelClient:
    """Minimal OpenAI-compatible Responses API client.

    This implementation is intentionally narrow: it supports the subset needed
    by the current AgentLoop abstraction, namely assistant text and function
    tool calls over the streaming `/responses` endpoint.
    """

    def __init__(
        self,
        config: ResponsesProviderConfig,
        timeout_seconds: float = 120.0,
        session_id: str | None = None,
        originator: str = DEFAULT_ORIGINATOR,
        user_agent: str | None = None,
        openai_subagent: str | None = None,
    ) -> None:
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
        config_path: str | Path = DEFAULT_CODEX_CONFIG_PATH,
        profile: str | None = None,
        timeout_seconds: float = 120.0,
        originator: str = DEFAULT_ORIGINATOR,
        user_agent: str | None = None,
    ) -> ResponsesModelClient:
        config = ResponsesProviderConfig.from_codex_config(config_path, profile)
        return cls(config, timeout_seconds, originator=originator, user_agent=user_agent)

    def with_overrides(
        self,
        model: str | None = None,
        reasoning_effort: str | None = None,
        session_id: str | None = None,
        openai_subagent: str | None = None,
    ) -> ResponsesModelClient:
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

    def responses_url(self) -> str:
        base_url = self._config.base_url.rstrip("/")
        url = f"{base_url}/responses"
        if self._config.query_params:
            return f"{url}?{urllib.parse.urlencode(self._config.query_params)}"
        return url

    def models_url(self) -> str:
        base_url = self._config.base_url.rstrip("/")
        url = f"{base_url}/models"
        if self._config.query_params:
            return f"{url}?{urllib.parse.urlencode(self._config.query_params)}"
        return url

    async def list_models(self) -> list[str]:
        return await asyncio.to_thread(self._list_models_sync)

    async def complete(
        self,
        prompt: Prompt,
        event_handler: ModelStreamEventHandler = NOOP_MODEL_STREAM_EVENT_HANDLER,
    ) -> ModelResponse:
        return await asyncio.to_thread(self._complete_sync, prompt, event_handler)

    def _complete_sync(
        self,
        prompt: Prompt,
        event_handler: ModelStreamEventHandler,
    ) -> ModelResponse:
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
                response = session.send(
                    prepared,
                    timeout=self._timeout_seconds,
                    allow_redirects=False,
                    **settings,
                )
                with response:
                    if response.status_code >= 400:
                        error_body = response.text
                        raise ResponsesApiError(
                            f"responses request failed with status {response.status_code}: "
                            f"{error_body[:500]}"
                        )
                    return self._parse_stream(
                        response.iter_lines(chunk_size=1, decode_unicode=False),
                        event_handler,
                    )
        except requests.RequestException as exc:
            raise ResponsesApiError(f"responses request failed: {exc}") from exc

    def _build_payload(self, prompt: Prompt) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "instructions": prompt.base_instructions or "",
            "input": [item.serialize() for item in prompt.input],
            "tools": [tool.serialize() for tool in prompt.tools],
            "tool_choice": "auto",
            "parallel_tool_calls": prompt.parallel_tool_calls,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": self._session_id,
        }

        reasoning: dict[str, str] = {}
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

    def _list_models_sync(self) -> list[str]:
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
        models: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id", "")).strip()
            if model_id:
                models.append(model_id)
        return models

    def _build_headers(self, prompt: Prompt) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream",
            "authorization": f"Bearer {self._config.api_key()}",
            "x-client-request-id": self._session_id,
            "session_id": self._session_id,
            "originator": self._originator,
            "user-agent": self._user_agent,
        }
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

    def _build_model_list_headers(self) -> dict[str, str]:
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {self._config.api_key()}",
            "originator": self._originator,
            "user-agent": self._user_agent,
        }
        if self._config.beta_features_header is not None:
            headers["x-codex-beta-features"] = self._config.beta_features_header
        if self._openai_subagent is not None:
            headers["x-openai-subagent"] = self._openai_subagent
        return headers

    def _parse_stream(
        self,
        response,
        event_handler: ModelStreamEventHandler,
    ) -> ModelResponse:
        items: list[AssistantMessage | ToolCall | ReasoningItem] = []
        saw_completed = False

        for event_name, data in self._iter_sse_events(response):
            if not data:
                continue
            payload = json.loads(data)
            event_type = payload.get("type", event_name)

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
                continue

            if event_type == "response.completed":
                saw_completed = True
                break

            if event_type == "response.failed":
                error = payload.get("response", {}).get("error") or {}
                message = error.get("message") or "responses stream failed"
                raise ResponsesApiError(message)

        if not saw_completed:
            raise ResponsesApiError("responses stream ended before response.completed")

        return ModelResponse(items=items)

    def _parse_output_item(
        self,
        item: dict[str, object],
    ) -> AssistantMessage | ToolCall | ReasoningItem | None:
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

    def _iter_sse_events(self, response):
        event_name: str | None = None
        data_lines: list[str] = []

        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == "":
                if data_lines:
                    yield event_name or "message", "\n".join(data_lines)
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
            yield event_name or "message", "\n".join(data_lines)


def _requests_verify_setting() -> str | bool | None:
    for env_name in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return None
