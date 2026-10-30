"""Built-in tools for creating, inspecting and stopping a thread goal."""

from .base_tool import BaseTool, StructuredToolOutput


class CreateGoalTool(BaseTool):
    name = "create_goal"
    description = (
        "Create a goal only when explicitly requested by the user or system/developer "
        "instructions; do not infer goals from ordinary tasks.\n"
        "Set token_budget only when an explicit token budget is requested. Fails if "
        "an unfinished goal exists; use update_goal only for status."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "objective": {
                "type": "string",
                "description": (
                    "Required. The concrete objective to start pursuing. This starts "
                    "a new active goal when no goal exists or replaces the current "
                    "goal when it is complete."
                ),
            },
            "token_budget": {
                "type": "integer",
                "description": (
                    "Positive token budget for the new goal. Omit unless explicitly requested."
                ),
            },
        },
        "required": ["objective"],
        "additionalProperties": False,
    }
    supports_parallel = False

    def __init__(self, manager):
        self._manager = manager

    async def run(self, context, args):
        try:
            return self._manager.create(
                context.session_id, args["objective"], args.get("token_budget"),
            )
        except ValueError as exc:
            return StructuredToolOutput(str(exc), success=False)


class GetGoalTool(BaseTool):
    name = "get_goal"
    description = (
        "Get the current goal for this thread, including status, budgets, token and "
        "elapsed-time usage, and remaining token budget."
    )
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, manager):
        self._manager = manager

    async def run(self, context, args):
        return self._manager.response()


class UpdateGoalTool(BaseTool):
    name = "update_goal"
    description = (
        "Update the existing goal.\n"
        "Set status to `paused` only at the user's explicit request to pause this goal, "
        "never on your own initiative. Ask if unclear; a later resume revokes that "
        "request. Report the returned status and stop goal work. Budget limits take "
        "precedence over pausing.\n"
        "Set status to `complete` only when the objective has actually been achieved "
        "and no required work remains.\n"
        "Set status to `blocked` only when the same blocking condition has repeated "
        "for at least three consecutive goal turns, counting the original/user-triggered "
        "turn and any automatic continuations, and the agent cannot make meaningful "
        "progress without user input or an external-state change.\n"
        "If the user resumes a goal that was previously marked `blocked`, treat the "
        "resumed run as a fresh blocked audit. If the same blocking condition then "
        "repeats for at least three consecutive resumed goal turns, set status to "
        "`blocked` again.\n"
        "Once the blocked threshold is satisfied, do not keep reporting that you are "
        "still blocked while leaving the goal active; set status to `blocked`.\n"
        "Do not use `blocked` merely because the work is hard, slow, uncertain, "
        "incomplete, or would benefit from clarification.\n"
        "Do not mark a goal complete merely because its budget is nearly exhausted "
        "or because you are stopping work.\n"
        "You cannot use this tool to resume, budget-limit, or usage-limit a goal; "
        "those status changes are controlled by the user or system.\n"
        "When marking a budgeted goal achieved with status `complete`, report the "
        "final token usage from the tool result to the user."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["complete", "blocked", "paused"],
                "description": (
                    "Required. `paused` requires an explicit user request. Set to "
                    "`complete` only when the objective is achieved and no required "
                    "work remains. Set to `blocked` only after the same blocking "
                    "condition has recurred for at least three consecutive goal turns "
                    "and the agent is at an impasse. After a previously blocked goal "
                    "is resumed, the resumed run starts a fresh blocked audit."
                ),
            },
        },
        "required": ["status"],
        "additionalProperties": False,
    }
    supports_parallel = False

    def __init__(self, manager):
        self._manager = manager

    async def run(self, context, args):
        try:
            return self._manager.update(args["status"])
        except ValueError as exc:
            return StructuredToolOutput(str(exc), success=False)
