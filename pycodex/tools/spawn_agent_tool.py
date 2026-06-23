"""`spawn_agent` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `spawn_agent` collaboration tool.

Expected behavior:
- Spawn a sibling agent runtime that can work in parallel with the caller.
- Optionally seed the spawned agent with the current thread history.
- Return the new agent identifier plus any user-facing nickname.
"""

from ..protocol import JSONDict, JSONValue
from ..runtime_services import SubAgentManager
from .agent_tool_schemas import COLLAB_INPUT_ITEMS_SCHEMA
from .base_tool import BaseTool, ToolContext
import typing

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
    description = """
        Spawn a sub-agent for a well-scoped task. Returns the spawned agent id plus the user-facing nickname when available. Spawned agents inherit your current model by default. Omit `model` to use that preferred default; set `model` only when an explicit override is needed.
This spawn_agent tool provides you access to sub-agents that inherit your current model by default. Do not set the `model` field unless the user explicitly asks for a different model or there is a clear task-specific reason. You should follow the rules and guidelines below to use this tool.

Do not spawn sub-agents unless the user explicitly asks for sub-agents, delegation, or parallel agent work.

### Designing delegated subtasks
- Subtasks must be concrete, well-defined, and self-contained.
- Delegated subtasks must materially advance the main task.
- Do not duplicate work between the main rollout and delegated subtasks.
- Avoid issuing multiple delegate calls on the same unresolved thread unless the new delegated task is genuinely different and necessary.
- Narrow the delegated ask to the concrete output you need next.
- For coding tasks, prefer delegating concrete code-change worker subtasks over read-only explorer analysis when the subagent can make a bounded patch in a clear write scope.
- When delegating coding work, instruct the submodel to edit files directly in its forked workspace and list the file paths it changed in the final answer.
- For code-edit subtasks, decompose work so each delegated task has a disjoint write set.

### After you delegate
- Call wait_agent very sparingly. Only call wait_agent when you need the result immediately for the next critical-path step and you are blocked until it returns.
- Do not redo delegated subagent tasks yourself; focus on integrating results or tackling non-overlapping work.
- While the subagent is running in the background, do meaningful non-overlapping work immediately.
- Do not repeatedly wait by reflex.
- When a delegated coding task returns, quickly review the uploaded changes, then integrate or refine them.

### Parallel delegation patterns
- Run multiple independent information-seeking subtasks in parallel when you have distinct questions that can be answered independently.
- Split implementation into disjoint codebase slices and spawn multiple agents in parallel when the write scopes do not overlap.
- Delegate verification only when it can run in parallel with ongoing implementation and is likely to catch a concrete risk before final integration.
- The key is to find opportunities to spawn multiple independent subtasks in parallel within the same round, while ensuring each subtask is well-defined, self-contained, and materially advances the main task."""
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
                "description": "True forks the current thread history into the new agent; false or omitted starts with only the initial prompt.",
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
        "additionalProperties": False,
    }
    output_schema = SPAWN_AGENT_OUTPUT_SCHEMA
    supports_parallel = False

    def __init__(self, subagent_manager: 'SubAgentManager') -> 'None':
        self._subagent_manager = subagent_manager

    async def run(self, context: 'ToolContext', args: 'JSONDict') -> 'JSONValue':
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

    def _optional_string(self, args: 'JSONDict', key: 'str') -> 'typing.Union[str, None]':
        value = args.get(key)
        if value in (None, ""):
            return None
        return str(value)
