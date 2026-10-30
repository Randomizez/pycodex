"""Create a named task using the MultiAgentV2 contract."""

from .base_tool import BaseTool


class SpawnAgentTool(BaseTool):
    name = "spawn_agent"
    description = (
        'Spawns an agent to work on the specified task. If your current task is '
        '`/root/task1` and you spawn_agent with task_name "task_3" the agent will '
        'have canonical task name `/root/task1/task_3`.\n'
        'You are then able to refer to this agent as `task_3` or '
        '`/root/task1/task_3` interchangeably. However an agent `/root/task2/task_3` '
        'would only be able to communicate with this agent via its canonical '
        'name `/root/task1/task_3`.\n'
        'The spawned agent will have the same tools as you and the ability to '
        'spawn its own subagents.\n'
        'Spawned agents inherit your current model by default. Do not set the '
        '`model` field unless the user explicitly asks for a different model.\n'
        'It will be able to send you and other running agents messages, and its '
        'final answer will be provided to you when it finishes.\n'
        'The new agent\'s canonical task name will be provided to it along with '
        'the message.\n'
        'Note that passing `fork_turns="none"` will not pass any surrounding '
        'context to the spawned subagent, which may cause the agent to lack the '
        'context it needs to complete its task, whereas `fork_turns="all"` will '
        'provide the subagent with all surrounding context.'
    )
    input_schema = {
        "type": "object",
        "properties": {
            "task_name": {
                "type": "string",
                "description": "Task name for the new agent. Use lowercase letters, digits, and underscores.",
            },
            "message": {
                "type": "string",
                "encrypted": True,
                "description": "Initial plain-text task for the new agent.",
            },
            "fork_turns": {
                "type": "string",
                "description": "Parent history to inherit. Defaults to `all`; use `none` to start without parent history. Only `all` and `none` are supported.",
            },
            "model": {
                "type": "string",
                "description": "Model override for the new agent. Omit unless an explicit override is needed.",
            },
            "reasoning_effort": {
                "type": "string",
                "description": "Reasoning effort override for the new agent. Omit to inherit the parent effort.",
            },
        },
        "required": ["task_name", "message"],
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {
            "task_name": {
                "type": "string",
                "description": "Canonical task name for the spawned agent.",
            },
        },
        "required": ["task_name"],
        "additionalProperties": False,
    }
    supports_parallel = False

    def __init__(self, agents_manager):
        self._agents_manager = agents_manager

    async def run(self, context, args):
        if set(args) - set(self.input_schema["properties"]):
            raise ValueError("unexpected spawn_agent argument")
        agent_path = await self._agents_manager.spawn_agent(
            context.agent_id, args["task_name"], args["message"],
            args.get("fork_turns", "all"),
            args.get("model"), args.get("reasoning_effort"), context.history,
            encrypted=context.encrypted_messages,
        )
        return {"task_name": agent_path}
