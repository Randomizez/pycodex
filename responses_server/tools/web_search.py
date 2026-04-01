from __future__ import annotations

from copy import deepcopy
import json

from pycodex.protocol import JSONValue
from pycodex.tools.base_tool import BaseTool, ToolContext


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Mock web search tool for Responses compatibility. Returns empty results."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Primary search query.",
            },
            "queries": {
                "type": "array",
                "description": "Optional batch of search queries.",
                "items": {"type": "string"},
            },
        },
        "required": ["query"],
    }
    supports_parallel = False

    async def run(self, context: ToolContext, args: JSONValue) -> JSONValue:
        del context
        query, queries = extract_queries(args)
        output_payload: dict[str, object] = {
            "results": [],
            "mock": True,
        }
        if query:
            output_payload["query"] = query
        if queries:
            output_payload["queries"] = queries
        return output_payload


def build_tool_definition(tool: WebSearchTool) -> dict[str, object]:
    return {
        "type": "function",
        "name": tool.name,
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": deepcopy(tool.input_schema),
            "strict": False,
        },
    }


def partition_tool_calls(
    tool: WebSearchTool,
    tool_calls: dict[int, dict[str, object]],
    outcomming_request: dict[str, object],
) -> tuple[list[dict[str, object]], dict[int, dict[str, object]]]:
    mock_tool_names = _collect_mock_tool_names(tool, outcomming_request)
    mock_calls: list[dict[str, object]] = []
    ordinary_tool_calls: dict[int, dict[str, object]] = {}
    for index in sorted(tool_calls):
        tool_call = tool_calls[index]
        function = tool_call.get("function") or {}
        tool_name = ""
        if isinstance(function, dict):
            tool_name = str(function.get("name", "")).strip()
        if tool_name in mock_tool_names:
            mock_calls.append(tool_call)
            continue
        ordinary_tool_calls[index] = tool_call
    return mock_calls, ordinary_tool_calls


def hydrate_tool_call_names(
    tool_calls: dict[int, dict[str, object]],
    outcomming_request: dict[str, object],
) -> None:
    raw_tools = outcomming_request.get("tools") or []
    if not isinstance(raw_tools, list):
        return
    for index, tool_call in tool_calls.items():
        function = tool_call.get("function") or {}
        if not isinstance(function, dict):
            continue
        if str(function.get("name", "")).strip():
            continue
        if index >= len(raw_tools):
            continue
        raw_tool = raw_tools[index]
        if not isinstance(raw_tool, dict) or raw_tool.get("type") != "function":
            continue
        raw_function = raw_tool.get("function") or {}
        if not isinstance(raw_function, dict):
            continue
        name = str(raw_function.get("name", "")).strip()
        if name:
            function["name"] = name


def build_output_items(
    mock_search_calls: list[dict[str, object]],
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for tool_call in mock_search_calls:
        function = tool_call.get("function") or {}
        if not isinstance(function, dict):
            continue
        query, queries = extract_queries(function.get("arguments"))
        action: dict[str, object] = {"type": "search"}
        if query:
            action["query"] = query
        if queries:
            action["queries"] = queries
        items.append(
            {
                "type": "web_search_call",
                "id": str(tool_call.get("id", "")).strip() or "ws_mock",
                "action": action,
            }
        )
    return items


def build_followup_request(
    tool: WebSearchTool,
    outcomming_request: dict[str, object],
    mock_search_calls: list[dict[str, object]],
) -> dict[str, object]:
    followup_request = deepcopy(outcomming_request)
    messages = followup_request.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError("outcomming request messages must be a list")

    assistant_tool_calls: list[dict[str, object]] = []
    for tool_call in mock_search_calls:
        function = tool_call.get("function") or {}
        if not isinstance(function, dict):
            continue
        assistant_tool_calls.append(
            {
                "id": str(tool_call.get("id", "")).strip() or "ws_mock",
                "type": "function",
                "function": {
                    "name": str(function.get("name", "")).strip() or tool.name,
                    "arguments": str(function.get("arguments", "") or "{}"),
                },
            }
        )
    if assistant_tool_calls:
        messages.append(
            {
                "role": "assistant",
                "tool_calls": assistant_tool_calls,
            }
        )

    for tool_call in mock_search_calls:
        tool_output = _build_mock_output((tool_call.get("function") or {}).get("arguments"))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": str(tool_call.get("id", "")).strip() or "ws_mock",
                "content": json.dumps(tool_output, ensure_ascii=False),
            }
        )

    followup_request["messages"] = messages
    raw_tools = followup_request.get("tools") or []
    if isinstance(raw_tools, list):
        filtered_tools = [
            raw_tool for raw_tool in raw_tools if not is_mock_tool(tool, raw_tool)
        ]
        if filtered_tools:
            followup_request["tools"] = filtered_tools
        else:
            followup_request.pop("tools", None)
            followup_request.pop("tool_choice", None)
            followup_request.pop("parallel_tool_calls", None)
    return followup_request


def extract_queries(raw_arguments: JSONValue) -> tuple[str, list[str]]:
    if isinstance(raw_arguments, dict):
        parsed = raw_arguments
    else:
        parsed = None

    if isinstance(raw_arguments, str):
        raw_text = raw_arguments
    else:
        raw_text = str(raw_arguments or "")

    if parsed is None:
        try:
            parsed = json.loads(raw_text or "{}")
        except json.JSONDecodeError:
            query = raw_text.strip()
            return query, [query] if query else []

    if not isinstance(parsed, dict):
        query = raw_text.strip()
        return query, [query] if query else []

    query = str(parsed.get("query", "")).strip()
    queries_value = parsed.get("queries") or []
    queries: list[str] = []
    if isinstance(queries_value, list):
        for value in queries_value:
            normalized = str(value).strip()
            if normalized:
                queries.append(normalized)
    if not query and queries:
        query = queries[0]
    if query and query not in queries:
        queries.insert(0, query)
    return query, queries


def is_mock_tool(tool: WebSearchTool, raw_tool: object) -> bool:
    if not isinstance(raw_tool, dict) or raw_tool.get("type") != "function":
        return False
    function = raw_tool.get("function") or {}
    if not isinstance(function, dict):
        return False
    return (
        str(function.get("name", "")).strip() == tool.name
        and str(function.get("description", "")).strip() == tool.description
    )


def _collect_mock_tool_names(
    tool: WebSearchTool,
    outcomming_request: dict[str, object],
) -> set[str]:
    names: set[str] = set()
    raw_tools = outcomming_request.get("tools") or []
    if not isinstance(raw_tools, list):
        return names
    for raw_tool in raw_tools:
        if is_mock_tool(tool, raw_tool):
            names.add(tool.name)
    return names


def _build_mock_output(raw_arguments: JSONValue) -> dict[str, object]:
    query, queries = extract_queries(raw_arguments)
    output_payload: dict[str, object] = {
        "results": [],
        "mock": True,
    }
    if query:
        output_payload["query"] = query
    if queries:
        output_payload["queries"] = queries
    return output_payload
