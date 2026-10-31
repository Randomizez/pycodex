"""Request a cooperative stop while retaining the recipient agent."""

from .agent_tool_schemas import AGENT_STATUS_SCHEMA
from .base_tool import BaseTool


class InterruptAgentTool(BaseTool):
    name = "interrupt_agent"
    namespace = "collaboration"
    namespace_description = "Tools for spawning and managing sub-agents."
    exec_only = False
    description = (
        "Interrupt an agent's current turn, if any, and return its previous "
        "status. The agent remains available for messages and follow-up tasks."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Agent id or canonical task name to interrupt (from spawn_agent).",
            },
        },
        "required": ["target"],
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {
            "previous_status": {
                "description": "The agent status observed before the interrupt request was handled.",
                "allOf": [AGENT_STATUS_SCHEMA],
            },
        },
        "required": ["previous_status"],
        "additionalProperties": False,
    }
    supports_parallel = False

    def __init__(self, agents_manager):
        self._agents_manager = agents_manager

    async def run(self, context, args):
        if set(args) != {"target"}:
            raise ValueError("expected target")
        return await self._agents_manager.interrupt_agent(
            context.agent_id, args["target"],
        )
