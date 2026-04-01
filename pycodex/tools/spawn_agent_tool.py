"""`spawn_agent` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `spawn_agent` collaboration tool.

Expected behavior:
- Spawn a sibling agent runtime that can work in parallel with the caller.
- Optionally seed the spawned agent with the current thread history.
- Return the new agent identifier plus any user-facing nickname.
"""

from __future__ import annotations

from ..protocol import JSONDict, JSONValue
from ..runtime_services import SubAgentManager
from .agent_tool_schemas import COLLAB_INPUT_ITEMS_SCHEMA
from .base_tool import BaseTool, ToolContext

SPAWN_AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "agent_id": {
            "type": "string",
            "description": "Thread identifier for the spawned agent.",
        },
        "nickname": {
            "type": ["string", "null"],
            "description": "User-facing nickname for the spawned agent when available.",
        },
    },
    "required": ["agent_id", "nickname"],
    "additionalProperties": False,
}


class SpawnAgentTool(BaseTool):
    name = "spawn_agent"
    description = (
        "Spawn a sub-agent for a well-scoped task. Returns the agent id (and "
        "user-facing nickname when available) to use to communicate with this "
        "agent."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Initial plain-text task for the new agent. Use either message or items.",
            },
            "items": COLLAB_INPUT_ITEMS_SCHEMA,
            "agent_type": {
                "type": "string",
                "description": "Optional type name for the new agent.",
            },
            "fork_context": {
                "type": "boolean",
                "description": "When true, fork the current thread history into the new agent before sending the initial prompt.",
            },
            "model": {
                "type": "string",
                "description": "Optional model override for the new agent.",
            },
            "reasoning_effort": {
                "type": "string",
                "description": "Optional reasoning effort override for the new agent.",
            },
        },
        "additionalProperties": False,
    }
    output_schema = SPAWN_AGENT_OUTPUT_SCHEMA
    supports_parallel = False

    def __init__(self, subagent_manager: SubAgentManager) -> None:
        self._subagent_manager = subagent_manager

    async def run(self, context: ToolContext, args: JSONDict) -> JSONValue:
        message = self._optional_string(args, "message")
        items = args.get("items")
        if items is not None and not isinstance(items, list):
            return "Error: `items` must be a list when provided."
        if message is None and not items:
            return "Provide one of: message or items"
        return await self._subagent_manager.spawn_agent(
            message=message,
            items=items,
            agent_type=self._optional_string(args, "agent_type"),
            fork_context=bool(args.get("fork_context", False)),
            model=self._optional_string(args, "model"),
            reasoning_effort=self._optional_string(args, "reasoning_effort"),
            history=context.history,
        )

    def _optional_string(self, args: JSONDict, key: str) -> str | None:
        value = args.get(key)
        if value in (None, ""):
            return None
        return str(value)
