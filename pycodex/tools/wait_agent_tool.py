"""Wait for MultiAgentV2 mailbox activity without consuming messages."""

from .base_tool import BaseTool


class WaitAgentTool(BaseTool):
    name = "wait_agent"
    description = (
        "Wait for a mailbox update from any live agent, including queued messages "
        "and final-status notifications. The wait also ends early when new user "
        "input is steered into the active turn. Does not return the content; "
        "returns either a summary of which agents have updates (if any), an "
        "interruption summary for steered input, or a timeout summary if no "
        "activity arrives before the deadline."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "timeout_ms": {
                "type": "number",
                "description": "Timeout in milliseconds. Defaults to 30000, min 10000, max 3600000.",
            },
        },
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Brief wait summary without the agent's final content, including any timeout adjustment.",
            },
            "timed_out": {
                "type": "boolean",
                "description": "Whether the wait call returned because no mailbox update arrived before the timeout.",
            },
        },
        "required": ["message", "timed_out"],
        "additionalProperties": False,
    }
    supports_parallel = False

    def __init__(self, agents_manager):
        self._agents_manager = agents_manager

    async def run(self, context, args):
        if set(args) - {"timeout_ms"}:
            raise ValueError("unexpected wait_agent argument")
        requested = args.get("timeout_ms", 30000)
        if type(requested) is not int:
            raise ValueError("timeout_ms must be an integer")
        if requested > 3600000:
            raise ValueError("timeout_ms must be at most 3600000")
        timeout = max(requested, 10000)
        result = await self._agents_manager.wait_for_activity(
            context.agent_id, timeout,
        )
        if requested < timeout:
            result["message"] += (
                "\n\nRequested timeout of {0}ms was clamped to the minimum of {1}ms."
            ).format(requested, timeout)
        return result
