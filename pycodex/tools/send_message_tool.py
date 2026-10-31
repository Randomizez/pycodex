"""Deliver a MultiAgentV2 message without starting an idle agent."""

from .base_tool import BaseTool


class SendMessageTool(BaseTool):
    name = "send_message"
    namespace = "collaboration"
    namespace_description = "Tools for spawning and managing sub-agents."
    exec_only = False
    description = (
        "Send a message to an existing agent. The message will be delivered "
        "promptly. Does not trigger a new turn."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Relative or canonical task name to message (from spawn_agent).",
            },
            "message": {
                "type": "string",
                "encrypted": True,
                "description": "Message text to queue on the target agent.",
            },
        },
        "required": ["target", "message"],
        "additionalProperties": False,
    }
    supports_parallel = False

    def __init__(self, agents_manager):
        self._agents_manager = agents_manager

    async def run(self, context, args):
        if set(args) != {"target", "message"}:
            raise ValueError("expected target and message")
        return await self._agents_manager.send_message(
            context.agent_id, args["target"], args["message"],
            encrypted=context.encrypted_messages,
        )
