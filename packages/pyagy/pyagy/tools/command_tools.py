"""AGY command & scheduling tools: run_command, manage_task, schedule."""

import typing

from pycodex.protocol import JSONDict, JSONValue
from pycodex.tools.base_tool import ToolContext

from . import AgyBaseTool, AgyToolError, AgyToolRegistry, format_sync_tool_header

if typing.TYPE_CHECKING:
    from ..task_manager import AgyTaskManager


class RunCommandTool(AgyBaseTool):
    """Implements AGY `run_command`."""

    def __init__(
        self,
        declaration: "JSONDict",
        registry: "AgyToolRegistry",
        task_manager: "AgyTaskManager",
    ) -> "None":
        super().__init__(declaration, registry)
        self._task_manager = task_manager

    async def run(self, context: "ToolContext", args: "JSONValue") -> "str":
        created_at = self.now_local_iso()
        params = self.require_dict_args(args)
        command_line = params.get("CommandLine")
        tool_summary = params.get("toolSummary")
        wait_ms = params.get("WaitMsBeforeAsync")
        if not isinstance(command_line, str):
            raise AgyToolError("CommandLine must be a string")
        if not isinstance(tool_summary, str) or not tool_summary.strip():
            raise AgyToolError("toolSummary must be a non-empty string")
        if isinstance(wait_ms, bool) or not isinstance(wait_ms, int):
            raise AgyToolError("WaitMsBeforeAsync must be an integer")
        is_daemon = bool(params.get("IsDaemon", False))
        step_idx = self.step_index_for_call(context)

        completed_sync, body = await self._task_manager.run_command(
            command_line=command_line,
            tool_summary=tool_summary,
            wait_ms=wait_ms,
            is_daemon=is_daemon,
            cwd=context.cwd,
            step_index=step_idx,
        )
        if completed_sync:
            completed_at = self.now_local_iso()
            return f"{format_sync_tool_header(created_at, completed_at)}{body}"
        return body


class ManageTaskTool(AgyBaseTool):
    """Implements AGY `manage_task`."""

    def __init__(
        self,
        declaration: "JSONDict",
        registry: "AgyToolRegistry",
        task_manager: "AgyTaskManager",
    ) -> "None":
        super().__init__(declaration, registry)
        self._task_manager = task_manager

    async def run(self, context: "ToolContext", args: "JSONValue") -> "str":
        created_at = self.now_local_iso()
        params = self.require_dict_args(args)
        action = params.get("Action")
        if action not in ("list", "kill", "status", "send_input"):
            raise AgyToolError(f"unsupported manage_task Action: {action!r}")

        try:
            if action == "list":
                body = self._task_manager.list_tasks()
                completed_at = self.now_local_iso()
                return f"{format_sync_tool_header(created_at, completed_at)}{body}"

            task_id = params.get("TaskId")
            if not isinstance(task_id, str) or not task_id.strip():
                raise AgyToolError(f"TaskId is required for Action={action!r}")

            if action == "status":
                body = self._task_manager.status(task_id)
                completed_at = self.now_local_iso()
                return f"{format_sync_tool_header(created_at, completed_at)}{body}"

            if action == "send_input":
                input_data = params.get("Input")
                if not isinstance(input_data, str):
                    raise AgyToolError("Input is required for Action='send_input'")
                body = self._task_manager.send_input(task_id, input_data)
                completed_at = self.now_local_iso()
                return f"{format_sync_tool_header(created_at, completed_at)}{body}"

            summary = self._task_manager.kill(task_id)
            completed_at = self.now_local_iso()
            return f"{format_sync_tool_header(created_at, completed_at)}{summary}"
        except ValueError as exc:
            raise AgyToolError(str(exc)) from exc


class ScheduleTool(AgyBaseTool):
    """Implements AGY `schedule`."""

    def __init__(
        self,
        declaration: "JSONDict",
        registry: "AgyToolRegistry",
        task_manager: "AgyTaskManager",
    ) -> "None":
        super().__init__(declaration, registry)
        self._task_manager = task_manager

    async def run(self, context: "ToolContext", args: "JSONValue") -> "str":
        params = self.require_dict_args(args)
        prompt = params.get("Prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise AgyToolError("Prompt must be a non-empty string")

        duration = params.get("DurationSeconds")
        if duration is not None and (isinstance(duration, bool) or not isinstance(duration, int)):
            raise AgyToolError("DurationSeconds must be an integer")
        cron_expr = params.get("CronExpression")
        if cron_expr is not None and not isinstance(cron_expr, str):
            raise AgyToolError("CronExpression must be a string")
        max_iters = params.get("MaxIterations")
        if max_iters is not None and (isinstance(max_iters, bool) or not isinstance(max_iters, int)):
            raise AgyToolError("MaxIterations must be an integer")
        timer_cond = params.get("TimerCondition")
        if timer_cond is not None and not isinstance(timer_cond, str):
            raise AgyToolError("TimerCondition must be a string")

        step_idx = self.step_index_for_call(context)
        try:
            return await self._task_manager.schedule(
                prompt=prompt,
                step_index=step_idx,
                duration_seconds=duration,
                cron_expression=cron_expr,
                exact_timing=bool(params.get("ExactTiming", False)),
                is_daemon=bool(params.get("IsDaemon", False)),
                max_iterations=max_iters,
                timer_condition=timer_cond,
            )
        except ValueError as exc:
            raise AgyToolError(str(exc)) from exc
