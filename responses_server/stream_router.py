"""Coordinate downstream streams, retries, mock tool follow-ups and output."""

import json
import typing

from .request_translation import ToolNameMap, coalesce_reasoning_text
from .session_store import StoredResponse
from .stream_events import ResponseOutputStream
from .tools import WebSearchTool
from .tools.custom_adapter import CustomToolAdapterError
from .tools.custom_adapter import build_output_item as build_custom_output_item
from .tools.web_search import (
    build_followup_request,
    build_output_items,
    hydrate_tool_call_names,
    partition_tool_calls,
)
from .trajectory_dump import TrajectoryDumpWriter
from .transport import OutcommingChatError, OutcommingTransport


class StreamRouter:
    def __init__(self, transport: "OutcommingTransport") -> "None":
        self.transport = transport
        self._mock_web_search = WebSearchTool()

    def route_stream(
        self,
        stored_response: "StoredResponse",
        outcomming_request: "typing.Dict[str, object]",
        tool_names: "ToolNameMap",
        trajectory_dump: "typing.Union[TrajectoryDumpWriter, None]" = None,
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

        text_parts: "typing.List[str]" = []
        reasoning_parts: "typing.List[str]" = []
        output_stream = ResponseOutputStream()
        latest_usage: "typing.Dict[str, object]" = {}
        current_request = json.loads(json.dumps(outcomming_request))
        current_stream = self._open_tracked_outcomming_stream(
            current_request,
            trajectory_dump,
        )
        retried_reasoning_only_output = False

        while True:
            tool_calls: "typing.Dict[int, typing.Dict[str, object]]" = {}
            finish_reasons: "typing.List[str]" = []
            current_usage: "typing.Dict[str, object]" = {}
            reasoning_start = len(reasoning_parts)
            text_start = len(text_parts)
            for chunk in current_stream:
                for event_name, payload in self._consume_chat_chunk(
                    chunk,
                    reasoning_parts,
                    text_parts,
                    tool_calls,
                    current_usage,
                    finish_reasons,
                ):
                    item_type = (
                        "reasoning" if event_name == "response.reasoning_text.delta"
                        else "message"
                    )
                    yield from output_stream.text_delta(item_type, payload["delta"])
            if current_usage:
                latest_usage = json.loads(json.dumps(current_usage))

            hydrate_tool_call_names(tool_calls, current_request)
            mock_search_calls, ordinary_tool_calls = partition_tool_calls(
                self._mock_web_search,
                tool_calls,
                current_request,
            )
            if mock_search_calls and not ordinary_tool_calls:
                yield from output_stream.finish_text()
                for item in build_output_items(mock_search_calls):
                    yield from output_stream.complete_tool(item)
                try:
                    current_request = build_followup_request(
                        self._mock_web_search,
                        current_request,
                        mock_search_calls,
                        reasoning_text="".join(reasoning_parts),
                    )
                except ValueError as exc:
                    raise OutcommingChatError(str(exc)) from exc
                current_stream = self._open_tracked_outcomming_stream(
                    current_request,
                    trajectory_dump,
                )
                continue

            if (
                len(reasoning_parts) > reasoning_start
                and len(text_parts) == text_start
                and not ordinary_tool_calls
            ):
                if not retried_reasoning_only_output:
                    retried_reasoning_only_output = True
                    del reasoning_parts[reasoning_start:]
                    del text_parts[text_start:]
                    output_stream.discard_text()
                    current_request = json.loads(json.dumps(current_request))
                    current_stream = self._open_tracked_outcomming_stream(
                        current_request,
                        trajectory_dump,
                    )
                    continue
                finish_reason = finish_reasons[-1] if finish_reasons else "<unknown>"
                raise OutcommingChatError(
                    "outcomming chat completion ended without assistant content "
                    "or tool calls after emitting only reasoning "
                    f"(finish_reason={finish_reason!r})",
                    error_type="model_output_invalid",
                )

            tool_items = self._build_tool_output_items(
                ordinary_tool_calls,
                tool_names.custom_tool_names,
            )
            yield from output_stream.finish_text()
            for item in tool_items:
                yield from output_stream.complete_tool(tool_names.restore_output_item(item))
            for item in build_output_items(mock_search_calls):
                yield from output_stream.complete_tool(item)
            break

        yield (
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": stored_response.response_id,
                    "object": "response",
                    "status": "completed",
                    "model": stored_response.model,
                    "output": output_stream.items,
                    **(
                        {"usage": json.loads(json.dumps(latest_usage))}
                        if latest_usage
                        else {}
                    ),
                },
            },
        )

    def _open_tracked_outcomming_stream(
        self,
        outcomming_request: "typing.Dict[str, object]",
        trajectory_dump: "typing.Union[TrajectoryDumpWriter, None]" = None,
    ):
        outcomming_stream = self.transport.open_stream(outcomming_request)
        if trajectory_dump is None:
            return outcomming_stream
        return trajectory_dump.wrap_stream(
            outcomming_stream,
            outcomming_request,
        )

    def _consume_chat_chunk(
        self,
        payload: "typing.Dict[str, object]",
        reasoning_parts: "typing.List[str]",
        text_parts: "typing.List[str]",
        tool_calls: "typing.Dict[int, typing.Dict[str, object]]",
        current_usage: "typing.Dict[str, object]",
        finish_reasons: "typing.List[str]",
    ) -> "typing.List[typing.Tuple[str, typing.Dict[str, object]]]":
        events: "typing.List[typing.Tuple[str, typing.Dict[str, object]]]" = []
        usage = payload.get("usage")
        if isinstance(usage, dict):
            self._capture_usage_snapshot(current_usage, usage)

        choices = payload.get("choices") or []
        if not isinstance(choices, list):
            return events

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason:
                finish_reasons.append(finish_reason)
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue

            reasoning = delta.get("reasoning")
            reasoning_content = delta.get("reasoning_content")
            reasoning_deltas = [reasoning]
            # Some chat providers expose the same delta under both aliases.
            if reasoning_content != reasoning:
                reasoning_deltas.append(reasoning_content)
            for reasoning_delta in reasoning_deltas:
                if isinstance(reasoning_delta, str) and reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                    events.append(
                        (
                            "response.reasoning_text.delta",
                            {
                                "type": "response.reasoning_text.delta",
                                "content_index": 0,
                                "delta": reasoning_delta,
                            },
                        )
                    )

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

    def _capture_usage_snapshot(
        self,
        current_usage: "typing.Dict[str, object]",
        usage: "typing.Dict[str, object]",
    ) -> "None":
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
                current_usage[key] = value

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
                current_usage[key] = json.loads(json.dumps(value))

    def _build_tool_output_items(
        self,
        tool_calls: "typing.Dict[int, typing.Dict[str, object]]",
        custom_tool_names: "typing.Set[str]",
    ) -> "typing.List[typing.Dict[str, object]]":
        items: "typing.List[typing.Dict[str, object]]" = []
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
            try:
                json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise OutcommingChatError(
                    f"outcomming function call arguments for {name!r} "
                    f"contain invalid JSON: {exc}",
                    error_type="server_error",
                ) from exc
            items.append(
                {
                    "type": "function_call",
                    "call_id": str(tool_call.get("id", "")).strip() or f"call_{index}",
                    "name": name,
                    "arguments": arguments,
                    "encrypted_function_args": [],
                }
            )

        return items
