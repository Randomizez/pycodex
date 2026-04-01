"""`web_search` tool declaration for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex provider-native `web_search` tool.

Expected behavior:
- Advertise the Responses API built-in web search tool to the model.
- Let the provider handle search execution directly instead of routing through
  the local `ToolRegistry`.
- Never expect a local tool-call round-trip result from the model.
"""

from __future__ import annotations

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, ToolContext


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Provider-native web search tool declaration."
    tool_type = "web_search"
    options = {
        "external_web_access": True,
    }
    supports_parallel = False

    async def run(self, context: ToolContext, args: JSONValue) -> JSONValue:
        del context, args
        return "Error: web_search is provider-native and should not be executed locally."
