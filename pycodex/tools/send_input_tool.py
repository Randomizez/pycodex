"""`send_input` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `send_input` collaboration tool.

Expected behavior:
- Send a follow-up message to an existing spawned agent.
- Optionally interrupt the agent's current turn before queueing the new input.
- Return the submission id for the queued input.
"""

from __future__ import annotations

from ..protocol import JSONDict, JSONValue
from ..runtime_services import SubAgentManager
from .agent_tool_schemas import COLLAB_INPUT_ITEMS_SCHEMA
from .base_tool import BaseTool, ToolContext

SEND_INPUT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "submission_id": {
            "type": "string",
            "description": "Identifier for the queued input submission.",
        }
    },
    "required": ["submission_id"],
    "additionalProperties": False,
}


class SendInputTool(BaseTool):
    name = "send_input"
    description = (
        "Send a message to an existing agent. Use interrupt=true to redirect "
        "work immediately."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Agent id to message (from spawn_agent).",
            },
            "message": {
                "type": "string",
                "description": "Legacy plain-text message to send to the agent. Use either message or items.",
            },
            "items": COLLAB_INPUT_ITEMS_SCHEMA,
            "interrupt": {
                "type": "boolean",
                "description": "When true, stop the agent's current task and handle this immediately. When false (default), queue this message.",
            },
        },
        "required": ["id"],
        "additionalProperties": False,
    }
    output_schema = SEND_INPUT_OUTPUT_SCHEMA
    supports_parallel = False

    def __init__(self, subagent_manager: SubAgentManager) -> None:
        self._subagent_manager = subagent_manager

    async def run(self, context: ToolContext, args: JSONDict) -> JSONValue:
        del context
        agent_id = str(args.get("id", "")).strip()
        if not agent_id:
            return "Error: `id` is required."
        items = args.get("items")
        if items is not None and not isinstance(items, list):
            return "Error: `items` must be a list when provided."
        prompt_text = self._compose_prompt(
            self._optional_string(args, "message"),
            items,
        )
        if not prompt_text:
            return "Error: `message` or `items` is required."
        return await self._subagent_manager.send_input(
            agent_id,
            prompt_text,
            interrupt=bool(args.get("interrupt", False)),
        )

    def _compose_prompt(
        self,
        message: str | None,
        items: list[dict[str, object]] | None,
    ) -> str:
        parts: list[str] = []
        if message:
            parts.append(message.strip())
        for item in items or []:
            item_type = str(item.get("type", ""))
            if item_type == "text":
                text = str(item.get("text", "")).strip()
                if text:
                    parts.append(text)
            else:
                parts.append(str(item))
        return "\n\n".join(part for part in parts if part)

    def _optional_string(self, args: JSONDict, key: str) -> str | None:
        value = args.get(key)
        if value in (None, ""):
            return None
        return str(value)
