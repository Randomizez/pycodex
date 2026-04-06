"""`update_plan` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `update_plan` tool.

Expected behavior:
- Give the model a structured checklist with optional explanation text.
- Validate that at most one step is `in_progress`.
- Persist the latest plan in local runtime state and return the standard
  confirmation text Codex uses.
"""

from ..protocol import JSONDict, JSONValue
from ..runtime_services import PlanItem, PlanStore
from .base_tool import BaseTool, ToolContext
import typing

VALID_PLAN_STATUSES = {"pending", "in_progress", "completed"}


class UpdatePlanTool(BaseTool):
    name = "update_plan"
    description = (
        "Updates the task plan. Provide an optional explanation and a list of "
        "plan items, each with a step and status. At most one step can be "
        "in_progress at a time."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "explanation": {"type": "string"},
            "plan": {
                "type": "array",
                "description": "The list of steps",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "string"},
                        "status": {
                            "type": "string",
                            "description": "One of: pending, in_progress, completed",
                        },
                    },
                    "required": ["step", "status"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["plan"],
        "additionalProperties": False,
    }
    supports_parallel = False

    def __init__(self, plan_store: 'PlanStore') -> 'None':
        self._plan_store = plan_store

    async def run(self, context: 'ToolContext', args: 'JSONDict') -> 'JSONValue':
        del context
        raw_plan = args.get("plan")
        if not isinstance(raw_plan, list):
            return "Error: `plan` must be a list."

        plan_items: 'typing.List[PlanItem]' = []
        for item in raw_plan:
            if not isinstance(item, dict):
                return "Error: each `plan` item must be an object."
            step = str(item.get("step", "")).strip()
            status = str(item.get("status", "")).strip()
            if not step:
                return "Error: each `plan` item must include a non-empty `step`."
            if status not in VALID_PLAN_STATUSES:
                return f"Error: invalid plan status `{status}`."
            plan_items.append(PlanItem(step=step, status=status))

        explanation_value = args.get("explanation")
        explanation = None if explanation_value in (None, "") else str(explanation_value)
        self._plan_store.update(explanation, tuple(plan_items))
        return "Plan updated"
