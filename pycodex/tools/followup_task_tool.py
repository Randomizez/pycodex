"""Deliver a MultiAgentV2 task, starting an idle recipient when necessary."""

from .base_tool import BaseTool


class FollowupTaskTool(BaseTool):
    name = "followup_task"
    description = (
        "Send a follow-up task to an existing non-root target agent and trigger "
        "a turn if it is idle. If the target is already running, deliver the task "
        "promptly at message boundaries while sampling, or after the pending "
        "tool call completes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Agent id or canonical task name to send a follow-up task to (from spawn_agent).",
            },
            "message": {
                "type": "string",
                "encrypted": True,
                "description": "Message text to send to the target agent.",
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
            context.agent_id, args["target"], args["message"], trigger_turn=True,
            encrypted=context.encrypted_messages,
        )
