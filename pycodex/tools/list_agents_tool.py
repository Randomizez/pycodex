"""Query live agents in the caller's MultiAgentV2 task tree."""

from .agent_tool_schemas import AGENT_STATUS_SCHEMA
from .base_tool import BaseTool


class ListAgentsTool(BaseTool):
    name = "list_agents"
    namespace = "collaboration"
    namespace_description = "Tools for spawning and managing sub-agents."
    exec_only = False
    description = (
        "List live agents in the current root thread tree. Optionally filter "
        "by task-path prefix."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path_prefix": {
                "type": "string",
                "description": "Task-path prefix filter without a trailing slash. Omit to list all live agents.",
            },
        },
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {
            "agents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Canonical task name for the agent when available, otherwise the agent id.",
                        },
                        "agent_status": {
                            "description": "Last known status of the agent.",
                            "allOf": [AGENT_STATUS_SCHEMA],
                        },
                    },
                    "required": ["agent_name", "agent_status"],
                    "additionalProperties": False,
                },
                "description": "Live agents visible in the current root thread tree.",
            },
        },
        "required": ["agents"],
        "additionalProperties": False,
    }
    supports_parallel = True

    def __init__(self, agents_manager):
        self._agents_manager = agents_manager

    async def run(self, context, args):
        if set(args) - {"path_prefix"}:
            raise ValueError("unexpected list_agents argument")
        return {"agents": [
            {"agent_name": agent["agent_path"], "agent_status": agent["status"]}
            for agent in self._agents_manager.list_agents(
                context.agent_id, args.get("path_prefix"),
            )
            if agent["status"] != "shutdown"
        ]}
