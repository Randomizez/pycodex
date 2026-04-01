"""`resume_agent` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `resume_agent` collaboration tool.

Expected behavior:
- Restart a previously closed local sub-agent runtime.
- Return the agent's current status payload.
"""

from __future__ import annotations

from ..protocol import JSONDict, JSONValue
from ..runtime_services import SubAgentManager
from .agent_tool_schemas import AGENT_STATUS_SCHEMA
from .base_tool import BaseTool, ToolContext

RESUME_AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": AGENT_STATUS_SCHEMA,
    },
    "required": ["status"],
    "additionalProperties": False,
}


class ResumeAgentTool(BaseTool):
    name = "resume_agent"
    description = (
        "Resume a previously closed agent by id so it can receive send_input "
        "and wait_agent calls."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Agent id to resume.",
            }
        },
        "required": ["id"],
        "additionalProperties": False,
    }
    output_schema = RESUME_AGENT_OUTPUT_SCHEMA
    supports_parallel = False

    def __init__(self, subagent_manager: SubAgentManager) -> None:
        self._subagent_manager = subagent_manager

    async def run(self, context: ToolContext, args: JSONDict) -> JSONValue:
        del context
        agent_id = str(args.get("id", "")).strip()
        if not agent_id:
            return "Error: `id` is required."
        return await self._subagent_manager.resume_agent(agent_id)
