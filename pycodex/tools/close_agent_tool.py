"""`close_agent` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `close_agent` collaboration tool.

Expected behavior:
- Shut down a spawned agent when it is no longer needed.
- Return the agent status observed at close time.
"""

from __future__ import annotations

from ..protocol import JSONDict, JSONValue
from ..runtime_services import SubAgentManager
from .agent_tool_schemas import AGENT_STATUS_SCHEMA
from .base_tool import BaseTool, ToolContext

CLOSE_AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": AGENT_STATUS_SCHEMA,
    },
    "required": ["status"],
    "additionalProperties": False,
}


class CloseAgentTool(BaseTool):
    name = "close_agent"
    description = (
        "Close an agent when it is no longer needed and return its status."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Agent id to close (from spawn_agent).",
            }
        },
        "required": ["id"],
        "additionalProperties": False,
    }
    output_schema = CLOSE_AGENT_OUTPUT_SCHEMA
    supports_parallel = False

    def __init__(self, subagent_manager: SubAgentManager) -> None:
        self._subagent_manager = subagent_manager

    async def run(self, context: ToolContext, args: JSONDict) -> JSONValue:
        del context
        agent_id = str(args.get("id", "")).strip()
        if not agent_id:
            return "Error: `id` is required."
        return await self._subagent_manager.close_agent(agent_id)
