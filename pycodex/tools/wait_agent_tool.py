"""`wait_agent` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `wait_agent` collaboration tool.

Expected behavior:
- Wait for one or more spawned agents to reach a final state.
- Return the current status map for requested agents, or an empty map when the
  wait times out.
"""

from ..protocol import JSONDict, JSONValue
from ..runtime_services import SubAgentManager
from .agent_tool_schemas import AGENT_STATUS_SCHEMA
from .base_tool import BaseTool, ToolContext

WAIT_AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "object",
            "description": "Final statuses keyed by agent id.",
            "additionalProperties": AGENT_STATUS_SCHEMA,
        },
        "timed_out": {
            "type": "boolean",
            "description": "Whether the wait call returned due to timeout before any agent reached a final status.",
        },
    },
    "required": ["status", "timed_out"],
    "additionalProperties": False,
}


class WaitAgentTool(BaseTool):
    name = "wait_agent"
    description = (
        "Wait for agents to reach a final status. Completed statuses may "
        "include the agent's final message. Returns empty status when timed "
        "out."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Agent ids to wait on. Pass multiple ids to wait for whichever finishes first.",
            },
            "timeout_ms": {
                "type": "integer",
                "description": "Optional timeout in milliseconds.",
            },
        },
        "required": ["ids"],
        "additionalProperties": False,
    }
    output_schema = WAIT_AGENT_OUTPUT_SCHEMA
    supports_parallel = False

    def __init__(self, subagent_manager: 'SubAgentManager') -> 'None':
        self._subagent_manager = subagent_manager

    async def run(self, context: 'ToolContext', args: 'JSONDict') -> 'JSONValue':
        del context
        ids = args.get("ids")
        if not isinstance(ids, list) or not ids:
            return "Error: `ids` must be a non-empty list."
        agent_ids = [str(item).strip() for item in ids if str(item).strip()]
        if not agent_ids:
            return "Error: `ids` must include at least one non-empty id."
        timeout_ms = int(args.get("timeout_ms", 30_000))
        return await self._subagent_manager.wait_agents(agent_ids, timeout_ms)
