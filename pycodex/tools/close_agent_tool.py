"""`close_agent` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `close_agent` collaboration tool.

Expected behavior:
- Shut down a spawned agent when it is no longer needed.
- Return the agent status observed before shutdown was requested.
"""

from ..protocol import JSONDict, JSONValue
from ..runtime_services import SubAgentManager
from .agent_tool_schemas import AGENT_STATUS_SCHEMA
from .base_tool import BaseTool, ToolContext

CLOSE_AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "previous_status": {
            "description": "The agent status observed before shutdown was requested.",
            "allOf": [AGENT_STATUS_SCHEMA],
        },
    },
    "required": ["previous_status"],
    "additionalProperties": False,
}


class CloseAgentTool(BaseTool):
    name = "close_agent"
    description = (
        "Close an agent and any open descendants when they are no longer "
        "needed, and return the target agent's previous status before shutdown "
        "was requested. Completed agents remain open and count toward the "
        "concurrency limit until closed. Don't keep agents open for too long "
        "if they are not needed anymore."
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

    def __init__(self, subagent_manager: 'SubAgentManager') -> 'None':
        self._subagent_manager = subagent_manager

    async def run(self, context: 'ToolContext', args: 'JSONDict') -> 'JSONValue':
        del context
        agent_id = str(args.get("id", "")).strip()
        if not agent_id:
            return "Error: `id` is required."
        return await self._subagent_manager.close_agent(agent_id)
