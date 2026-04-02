from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request

from .config import CompatServerConfig
from .session_store import StoredResponse
from .tools import WebSearchTool, collect_custom_tool_names
from .tools.custom_adapter import (
    CustomToolAdapterError,
    build_output_item as build_custom_output_item,
    build_tool_call as build_custom_tool_call,
    build_tool_definition as build_custom_tool_definition,
)
from .tools.web_search import (
    build_followup_request,
    build_output_items,
    build_tool_definition,
    hydrate_tool_call_names,
    partition_tool_calls,
)


class UnsupportedIncommingFeature(ValueError):
    pass


class OutcommingChatError(RuntimeError):
    pass


class StreamRouter:
    def __init__(self, config: CompatServerConfig) -> None:
        self._config = config
        self._mock_web_search = WebSearchTool()

    def _provider_capability(
        self,
        explicit_support: dict[str, bool],
        default: bool | None = None,
    ) -> bool:
        provider_name = str(self._config.model_provider or "").strip().lower()
        if provider_name in explicit_support:
            return explicit_support[provider_name]
        if "vllm" in explicit_support:
            return explicit_support["vllm"]
        if default is not None:
            return default
        raise KeyError("provider capability map is missing `vllm` fallback")

    def _supports_chat_reasoning(self) -> bool:
        # Unknown providers inherit the vLLM compatibility behavior unless a
        # provider is explicitly declared otherwise.
        return self._provider_capability(
            {
                "vllm": True,
                "stepfun": True,
            }
        )

    def _supports_stream_usage(self) -> bool:
        return self._provider_capability(
            {
                "vllm": True,
                "stepfun": True,
            }
        )

    def validate_incomming_request(
        self,
        incomming_request: dict[str, object],
    ) -> None:
        model = str(incomming_request.get("model", "")).strip()
        if not model:
            raise UnsupportedIncommingFeature("incomming request is missing `model`")

        stream = incomming_request.get("stream", True)
        if stream is not True:
            raise UnsupportedIncommingFeature(
                "only streaming incomming `/responses` requests are supported"
            )

        input_items = incomming_request.get("input") or []
        if not isinstance(input_items, list):
            raise UnsupportedIncommingFeature("incomming `input` must be a list")

        self.build_outcomming_request(incomming_request)

    def collect_custom_tool_names(
        self,
        incomming_request: dict[str, object],
    ) -> set[str]:
        return collect_custom_tool_names(incomming_request.get("tools") or [])

    def list_models(self) -> dict[str, object]:
        request = urllib.request.Request(
            self._config.outcomming_models_url(),
            headers=self._build_headers(accept="application/json"),
            method="GET",
        )
        return self._request_json(request)

    def build_outcomming_request(
        self,
        incomming_request: dict[str, object],
    ) -> dict[str, object]:
        model = str(incomming_request.get("model", "")).strip()
        if not model:
            raise UnsupportedIncommingFeature("incomming request is missing `model`")

        stream = incomming_request.get("stream", True)
        if stream is not True:
            raise UnsupportedIncommingFeature(
                "only streaming incomming `/responses` requests are supported"
            )

        instructions = str(incomming_request.get("instructions", "") or "")
        input_items = incomming_request.get("input") or []
        if not isinstance(input_items, list):
            raise UnsupportedIncommingFeature("incomming `input` must be a list")

        payload: dict[str, object] = {
            "model": model,
            "messages": self._responses_input_to_chat_messages(
                instructions,
                input_items,
            ),
            "stream": True,
        }
        if self._supports_stream_usage():
            payload["stream_options"] = {"include_usage": True}

        tools = incomming_request.get("tools") or []
        if tools:
            if not isinstance(tools, list):
                raise UnsupportedIncommingFeature("incomming `tools` must be a list")
            payload["tools"] = self._translate_tools(tools)

        tool_choice = incomming_request.get("tool_choice")
        if tool_choice is not None:
            payload["tool_choice"] = self._translate_tool_choice(tool_choice)

        parallel_tool_calls = incomming_request.get("parallel_tool_calls")
        if isinstance(parallel_tool_calls, bool):
            payload["parallel_tool_calls"] = parallel_tool_calls

        return payload

    def open_outcomming_stream(self, outcomming_request: dict[str, object]):
        request = urllib.request.Request(
            self._config.outcomming_chat_completions_url(),
            data=json.dumps(outcomming_request).encode("utf-8"),
            headers=self._build_headers(accept="text/event-stream"),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                context=ssl.create_default_context(),
                timeout=self._config.timeout_seconds,
            ) as response:
                for _event_name, data in self._iter_sse_events(response):
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    yield json.loads(data)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OutcommingChatError(
                f"outcomming chat request failed with status {exc.code}: {body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OutcommingChatError(
                f"outcomming chat request failed: {exc.reason}"
            ) from exc

    def route_stream(
        self,
        incomming_stream,
        stored_response: StoredResponse,
        outcomming_request: dict[str, object],
        custom_tool_names: set[str] | None = None,
    ):
        yield (
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": stored_response.response_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": stored_response.model,
                },
            },
        )

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage_totals: dict[str, object] = {}
        current_request = json.loads(json.dumps(outcomming_request))
        current_stream = incomming_stream

        while True:
            tool_calls: dict[int, dict[str, object]] = {}
            for chunk in current_stream:
                for event_name, payload in self._consume_chat_chunk(
                    chunk,
                    reasoning_parts,
                    text_parts,
                    tool_calls,
                    usage_totals,
                ):
                    yield event_name, payload

            hydrate_tool_call_names(tool_calls, current_request)
            mock_search_calls, ordinary_tool_calls = partition_tool_calls(
                self._mock_web_search,
                tool_calls,
                current_request,
            )
            if mock_search_calls and not ordinary_tool_calls:
                for item in build_output_items(mock_search_calls):
                    yield (
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "item": item,
                        },
                    )
                try:
                    current_request = build_followup_request(
                        self._mock_web_search,
                        current_request,
                        mock_search_calls,
                        reasoning_text=(
                            "".join(reasoning_parts)
                            if self._supports_chat_reasoning()
                            else None
                        ),
                    )
                except ValueError as exc:
                    raise OutcommingChatError(str(exc)) from exc
                current_stream = self.open_outcomming_stream(current_request)
                continue

            for item in self._build_output_items(
                reasoning_parts,
                text_parts,
                ordinary_tool_calls,
                custom_tool_names or set(),
            ):
                yield (
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "item": item,
                    },
                )
            for item in build_output_items(mock_search_calls):
                yield (
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "item": item,
                    },
                )
            break

        yield (
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": stored_response.response_id,
                    "output": [],
                    **(
                        {"usage": json.loads(json.dumps(usage_totals))}
                        if usage_totals
                        else {}
                    ),
                },
            },
        )

    def _responses_input_to_chat_messages(
        self,
        instructions: str,
        input_items: list[object],
    ) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []
        if instructions:
            messages.append({"role": "developer", "content": instructions})

        pending_assistant: dict[str, object] | None = None

        def flush_pending_assistant() -> None:
            nonlocal pending_assistant
            if pending_assistant is None:
                return
            if (
                "content" not in pending_assistant
                and "reasoning" not in pending_assistant
                and "tool_calls" not in pending_assistant
            ):
                pending_assistant = None
                return
            messages.append(pending_assistant)
            pending_assistant = None

        for raw_item in input_items:
            if not isinstance(raw_item, dict):
                raise UnsupportedIncommingFeature(
                    "all incomming `input` items must be objects"
                )
            item_type = raw_item.get("type")

            if item_type == "message":
                role = str(raw_item.get("role", "")).strip()
                if role not in {"developer", "user", "assistant", "system"}:
                    raise UnsupportedIncommingFeature(
                        f"unsupported incomming message role: {role or '<empty>'}"
                    )
                text = self._coalesce_content_text(raw_item.get("content"))
                if role == "assistant":
                    if pending_assistant is None:
                        pending_assistant = {"role": "assistant"}
                    if text:
                        pending_assistant["content"] = (
                            str(pending_assistant.get("content", "")) + text
                        )
                    continue
                flush_pending_assistant()
                messages.append({"role": role, "content": text})
                continue

            if item_type == "reasoning":
                if not self._supports_chat_reasoning():
                    raise UnsupportedIncommingFeature(
                        "incomming `reasoning` items are not supported by this chat backend"
                    )
                if pending_assistant is None:
                    pending_assistant = {"role": "assistant"}
                reasoning_text = self._coalesce_reasoning_text(raw_item)
                if reasoning_text:
                    pending_assistant["reasoning"] = (
                        str(pending_assistant.get("reasoning", "")) + reasoning_text
                    )
                continue

            if item_type == "function_call":
                if pending_assistant is None:
                    pending_assistant = {"role": "assistant"}
                tool_calls = pending_assistant.setdefault("tool_calls", [])
                if not isinstance(tool_calls, list):
                    raise UnsupportedIncommingFeature(
                        "assistant tool calls must be a list"
                    )
                tool_calls.append(
                    {
                        "id": str(raw_item.get("call_id", "")).strip(),
                        "type": "function",
                        "function": {
                            "name": str(raw_item.get("name", "")).strip(),
                            "arguments": str(raw_item.get("arguments", "") or "{}"),
                        },
                    }
                )
                continue

            if item_type == "function_call_output":
                flush_pending_assistant()
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(raw_item.get("call_id", "")).strip(),
                        "content": self._coalesce_tool_output_text(
                            raw_item.get("output")
                        ),
                    }
                )
                continue

            if item_type == "custom_tool_call":
                if pending_assistant is None:
                    pending_assistant = {"role": "assistant"}
                tool_calls = pending_assistant.setdefault("tool_calls", [])
                if not isinstance(tool_calls, list):
                    raise UnsupportedIncommingFeature(
                        "assistant tool calls must be a list"
                    )
                try:
                    tool_calls.append(build_custom_tool_call(raw_item))
                except CustomToolAdapterError as exc:
                    raise UnsupportedIncommingFeature(str(exc)) from exc
                continue

            if item_type == "custom_tool_call_output":
                flush_pending_assistant()
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(raw_item.get("call_id", "")).strip(),
                        "content": self._coalesce_tool_output_text(
                            raw_item.get("output")
                        ),
                    }
                )
                continue

            raise UnsupportedIncommingFeature(
                f"unsupported incomming input item type: {item_type!r}"
            )

        flush_pending_assistant()
        return messages

    def _coalesce_content_text(self, raw_content: object) -> str:
        if raw_content is None:
            return ""
        if isinstance(raw_content, str):
            return raw_content
        if not isinstance(raw_content, list):
            raise UnsupportedIncommingFeature(
                "message `content` must be a list or string"
            )

        text_parts: list[str] = []
        for part in raw_content:
            if not isinstance(part, dict):
                raise UnsupportedIncommingFeature(
                    "message content parts must be objects"
                )
            part_type = part.get("type")
            if part_type in {"input_text", "output_text"}:
                text_parts.append(str(part.get("text", "")))
                continue
            raise UnsupportedIncommingFeature(
                f"content part type `{part_type}` is not yet supported by the chat backend"
            )
        return "".join(text_parts)

    def _coalesce_tool_output_text(self, raw_output: object) -> str:
        if isinstance(raw_output, str):
            return raw_output
        if isinstance(raw_output, list):
            return self._coalesce_content_text(raw_output)
        return json.dumps(raw_output, ensure_ascii=False)

    def _coalesce_reasoning_text(self, raw_item: dict[str, object]) -> str:
        content = raw_item.get("content")
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in {"reasoning_text", "summary_text"}:
                    text_parts.append(str(part.get("text", "")))
            if text_parts:
                return "".join(text_parts)

        summary = raw_item.get("summary")
        if isinstance(summary, list):
            text_parts = []
            for part in summary:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "summary_text":
                    text_parts.append(str(part.get("text", "")))
            if text_parts:
                return "".join(text_parts)

        for key in ("reasoning", "reasoning_content", "text"):
            value = raw_item.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    def _translate_tools(self, incomming_tools: list[object]) -> list[dict[str, object]]:
        translated: list[dict[str, object]] = []
        for raw_tool in incomming_tools:
            if not isinstance(raw_tool, dict):
                raise UnsupportedIncommingFeature("tool definitions must be objects")
            tool_type = raw_tool.get("type")
            if tool_type == "function":
                name = str(raw_tool.get("name", "")).strip()
                translated.append(
                    {
                        "type": "function",
                        "name": name,
                        "function": {
                            "name": name,
                            "description": str(raw_tool.get("description", "") or ""),
                            "parameters": raw_tool.get("parameters")
                            or {"type": "object"},
                            "strict": bool(raw_tool.get("strict", False)),
                        },
                    }
                )
                continue
            if tool_type == "web_search":
                translated.append(build_tool_definition(self._mock_web_search))
                continue
            if tool_type == "custom":
                try:
                    translated.append(build_custom_tool_definition(raw_tool))
                except CustomToolAdapterError as exc:
                    raise UnsupportedIncommingFeature(str(exc)) from exc
                continue
            raise UnsupportedIncommingFeature(
                f"unsupported incomming tool type: {tool_type!r}"
            )
        return translated

    def _translate_tool_choice(self, raw_tool_choice: object) -> object:
        if isinstance(raw_tool_choice, str):
            return raw_tool_choice
        if not isinstance(raw_tool_choice, dict):
            raise UnsupportedIncommingFeature("unsupported `tool_choice` shape")

        choice_type = raw_tool_choice.get("type")
        if choice_type in {"function", "custom"}:
            name = raw_tool_choice.get("name")
            if not isinstance(name, str) or not name.strip():
                raise UnsupportedIncommingFeature(
                    f"{choice_type} tool_choice is missing `name`"
                )
            return {
                "type": "function",
                "function": {"name": name},
            }
        raise UnsupportedIncommingFeature(
            f"unsupported incomming tool_choice type: {choice_type!r}"
        )

    def _consume_chat_chunk(
        self,
        payload: dict[str, object],
        reasoning_parts: list[str],
        text_parts: list[str],
        tool_calls: dict[int, dict[str, object]],
        usage_totals: dict[str, object],
    ) -> list[tuple[str, dict[str, object]]]:
        events: list[tuple[str, dict[str, object]]] = []
        usage = payload.get("usage")
        if isinstance(usage, dict):
            self._accumulate_usage(usage_totals, usage)

        choices = payload.get("choices") or []
        if not isinstance(choices, list):
            return events

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue

            reasoning = delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                reasoning_parts.append(reasoning)

            reasoning_content = delta.get("reasoning_content")
            if isinstance(reasoning_content, str) and reasoning_content:
                reasoning_parts.append(reasoning_content)

            content = delta.get("content")
            if isinstance(content, str) and content:
                text_parts.append(content)
                events.append(
                    (
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "delta": content,
                        },
                    )
                )

            raw_tool_calls = delta.get("tool_calls") or []
            if not isinstance(raw_tool_calls, list):
                continue
            for raw_tool_call in raw_tool_calls:
                if not isinstance(raw_tool_call, dict):
                    continue
                index = int(raw_tool_call.get("index", len(tool_calls)))
                state = tool_calls.setdefault(
                    index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": "",
                        },
                    },
                )
                tool_call_id = raw_tool_call.get("id")
                if isinstance(tool_call_id, str) and tool_call_id:
                    state["id"] = tool_call_id
                tool_type = raw_tool_call.get("type")
                if isinstance(tool_type, str) and tool_type:
                    state["type"] = tool_type
                function = raw_tool_call.get("function") or {}
                if not isinstance(function, dict):
                    continue
                state_function = state["function"]
                if not isinstance(state_function, dict):
                    continue
                name = function.get("name")
                if isinstance(name, str) and name:
                    state_function["name"] = name
                arguments = function.get("arguments")
                if isinstance(arguments, str) and arguments:
                    state_function["arguments"] = (
                        str(state_function.get("arguments", "")) + arguments
                    )

        return events

    def _accumulate_usage(
        self,
        usage_totals: dict[str, object],
        usage: dict[str, object],
    ) -> None:
        scalar_mappings = (
            ("input_tokens", usage.get("input_tokens", usage.get("prompt_tokens"))),
            (
                "output_tokens",
                usage.get("output_tokens", usage.get("completion_tokens")),
            ),
            ("total_tokens", usage.get("total_tokens")),
        )
        for key, value in scalar_mappings:
            if isinstance(value, int):
                usage_totals[key] = int(usage_totals.get(key, 0)) + value

        detail_mappings = (
            (
                "input_tokens_details",
                usage.get("input_tokens_details", usage.get("prompt_tokens_details")),
            ),
            (
                "output_tokens_details",
                usage.get(
                    "output_tokens_details",
                    usage.get("completion_tokens_details"),
                ),
            ),
        )
        for key, value in detail_mappings:
            if isinstance(value, dict):
                target = usage_totals.setdefault(key, {})
                if isinstance(target, dict):
                    self._merge_usage_details(target, value)

    def _merge_usage_details(
        self,
        target: dict[str, object],
        incoming: dict[str, object],
    ) -> None:
        for key, value in incoming.items():
            if isinstance(value, int):
                target[key] = int(target.get(key, 0)) + value
                continue
            if isinstance(value, dict):
                nested = target.setdefault(key, {})
                if isinstance(nested, dict):
                    self._merge_usage_details(nested, value)

    def _build_output_items(
        self,
        reasoning_parts: list[str],
        text_parts: list[str],
        tool_calls: dict[int, dict[str, object]],
        custom_tool_names: set[str],
    ) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        reasoning_text = "".join(reasoning_parts)
        if reasoning_text:
            items.append(
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [
                        {
                            "type": "reasoning_text",
                            "text": reasoning_text,
                        }
                    ],
                }
            )
        text = "".join(text_parts)
        if text:
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                        }
                    ],
                }
            )

        for index in sorted(tool_calls):
            tool_call = tool_calls[index]
            if tool_call.get("type") != "function":
                raise OutcommingChatError(
                    f"unsupported outcomming tool call type: {tool_call.get('type')!r}"
                )
            function = tool_call.get("function") or {}
            if not isinstance(function, dict):
                raise OutcommingChatError(
                    "outcomming tool call is missing function payload"
                )
            name = str(function.get("name", "")).strip()
            if not name:
                raise OutcommingChatError(
                    "outcomming function tool call is missing `name`"
                )
            arguments = str(function.get("arguments", "") or "{}")
            if name in custom_tool_names:
                try:
                    items.append(build_custom_output_item(tool_call, index))
                except CustomToolAdapterError as exc:
                    raise OutcommingChatError(str(exc)) from exc
                continue
            items.append(
                {
                    "type": "function_call",
                    "call_id": str(tool_call.get("id", "")).strip()
                    or f"call_{index}",
                    "name": name,
                    "arguments": arguments,
                }
            )

        return items

    def _request_json(self, request: urllib.request.Request) -> dict[str, object]:
        try:
            with urllib.request.urlopen(
                request,
                context=ssl.create_default_context(),
                timeout=self._config.timeout_seconds,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OutcommingChatError(
                f"outcomming request failed with status {exc.code}: {body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OutcommingChatError(
                f"outcomming request failed: {exc.reason}"
            ) from exc

    def _build_headers(self, accept: str) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "Content-Type": "application/json",
        }
        api_key = self._config.outcomming_api_key()
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

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
