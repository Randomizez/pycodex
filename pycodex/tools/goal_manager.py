"""Per-session goal state, usage accounting and continuation scheduling."""

import time

from ..context import build_goal_message
from ..events import (
    CompactStartedEvent,
    TerminalEvent,
    TokenCountEvent,
    TurnCompletedEvent,
    TurnInterruptedEvent,
    TurnStartedEvent,
)


class GoalManager:
    def __init__(self):
        self._goal = None
        self._pending_tokens = 0
        self._last_accounted_at = None
        self._count_tokens = False
        self._continuation_ready = False
        self._budget_message = None

    @staticmethod
    def _validate_objective(objective):
        if not isinstance(objective, str):
            raise ValueError("goal objective must be a string")
        if not objective:
            raise ValueError("goal objective must not be empty")
        if len(objective) > 4000:
            raise ValueError("goal objective must be at most 4000 characters")

    @staticmethod
    def _validate_budget(token_budget):
        if token_budget is None:
            return
        if type(token_budget) is not int or token_budget > 9223372036854775807:
            raise ValueError("goal token_budget must be an integer in the i64 range")
        if token_budget <= 0:
            raise ValueError("goal budgets must be positive when provided")

    def create(self, thread_id, objective, token_budget=None):
        if isinstance(objective, str):
            objective = objective.strip()
        self._validate_objective(objective)
        self._validate_budget(token_budget)
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("goal requires a session ID")
        if self._goal is not None and self._goal["status"] != "complete":
            raise ValueError(
                "cannot create a new goal because this thread has an unfinished "
                "goal; complete the existing goal first"
            )
        now = int(time.time())
        self._goal = {
            "threadId": thread_id,
            "objective": objective,
            "status": "active",
            "tokensUsed": 0,
            "timeUsedSeconds": 0,
            "createdAt": now,
            "updatedAt": now,
        }
        if token_budget is not None:
            self._goal["tokenBudget"] = token_budget
        self._pending_tokens = 0
        self._last_accounted_at = time.monotonic()
        self._count_tokens = True
        self._continuation_ready = False
        self._budget_message = None
        return self.response()

    def update(self, status):
        if status not in ("complete", "blocked", "paused"):
            raise ValueError(
                "update_goal can only mark the existing goal complete, blocked, "
                "or paused at the user's explicit request; resume, budget-limited, "
                "and usage-limited status changes are controlled by the user or system"
            )
        if self._goal is None:
            raise ValueError("cannot update goal because this thread has no goal")
        self.account(steer=status != "complete")
        if status != "paused" or self._goal["status"] != "budgetLimited":
            self._goal["status"] = status
            self._last_accounted_at = None
            self._count_tokens = False
            self._budget_message = None
        self._goal["updatedAt"] = int(time.time())
        self._continuation_ready = False
        return self.response(include_completion=status == "complete")

    def response(self, include_completion=False):
        goal = self.dump_state()
        budget = None if goal is None else goal.get("tokenBudget")
        report = None
        if include_completion and (budget is not None or goal["timeUsedSeconds"] > 0):
            report = (
                "Goal achieved. Report final usage from this tool result's structured "
                "goal fields. If `goal.tokenBudget` is present, include token usage "
                "from `goal.tokensUsed` and `goal.tokenBudget`. If "
                "`goal.timeUsedSeconds` is greater than 0, summarize elapsed time "
                "in a concise, human-friendly form appropriate to the response language."
            )
        return {
            "goal": goal,
            "remainingTokens": None if budget is None else max(0, budget - goal["tokensUsed"]),
            "completionBudgetReport": report,
        }

    def account(self, steer=True):
        if self._last_accounted_at is None:
            return
        seconds = int(time.monotonic() - self._last_accounted_at)
        if not seconds and not self._pending_tokens:
            return
        goal = self._goal
        goal["tokensUsed"] += self._pending_tokens
        goal["timeUsedSeconds"] += seconds
        goal["updatedAt"] = int(time.time())
        self._pending_tokens = 0
        self._last_accounted_at += seconds
        budget = goal.get("tokenBudget")
        if goal["status"] == "active" and budget is not None and goal["tokensUsed"] >= budget:
            goal["status"] = "budgetLimited"
            if steer:
                self._budget_message = build_goal_message(goal, budget_limited=True)

    def handle_agent_event(self, event):
        if isinstance(event, (TurnStartedEvent, CompactStartedEvent)):
            self.start()
            self._continuation_ready = False
            self._count_tokens = self._goal is not None and self._goal["status"] == "active"
        elif isinstance(event, TokenCountEvent):
            if self._count_tokens and event.usage is not None:
                usage = event.usage
                cached = usage.get("input_tokens_details", {}).get("cached_tokens", 0)
                self._pending_tokens += (
                    max(0, usage.get("input_tokens", 0) - cached)
                    + max(0, usage.get("output_tokens", 0))
                )
        elif isinstance(event, TerminalEvent):
            self.account(steer=False)
            self._count_tokens = False
            if self._goal is not None:
                if isinstance(event, TurnInterruptedEvent) and self._goal["status"] == "active":
                    self._goal["status"] = "paused"
                    self._goal["updatedAt"] = int(time.time())
                active = self._goal["status"] == "active"
                self._continuation_ready = active and isinstance(event, TurnCompletedEvent)
                if not active:
                    self._last_accounted_at = None

    def follow_up_messages(self):
        message, self._budget_message = self._budget_message, None
        return () if message is None else (message,)

    def take_continuation(self):
        if not self._continuation_ready:
            return None
        self._continuation_ready = False
        return build_goal_message(self._goal)

    def dump_state(self):
        return None if self._goal is None else dict(self._goal)

    def validate_state(self, state):
        if state is None:
            return
        self._validate_objective(state["objective"])
        self._validate_budget(state.get("tokenBudget"))
        if not isinstance(state["threadId"], str) or not state["threadId"]:
            raise ValueError("goal requires a session ID")
        if state["status"] not in (
            "active", "paused", "blocked", "usageLimited", "budgetLimited", "complete",
        ):
            raise ValueError("invalid goal status")
        for name in ("tokensUsed", "timeUsedSeconds", "createdAt", "updatedAt"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError("goal " + name + " must be a non-negative integer")

    def restore_state(self, state):
        self.validate_state(state)
        self._goal = None if state is None else dict(state)
        self._pending_tokens = 0
        self._last_accounted_at = None
        self._count_tokens = False
        self._continuation_ready = state is not None and state["status"] == "active"
        self._budget_message = None
        self.start()

    def start(self):
        if (
            self._goal is not None and self._goal["status"] == "active"
            and self._last_accounted_at is None
        ):
            self._last_accounted_at = time.monotonic()
            self._continuation_ready = True

    def shutdown(self):
        self.account(steer=False)
        self._last_accounted_at = None
        self._count_tokens = False
        self._continuation_ready = False
        self._budget_message = None
