"""`wait` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex code-mode `wait` tool.

Expected behavior:
- Wait on a yielded `exec` cell for more output or completion.
- Optionally terminate the running cell.
- Return only the new output since the previous `exec` / `wait` snapshot.
"""

import math

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, ToolContext
from .code_mode_manager import DEFAULT_WAIT_YIELD_TIME_MS, CodeModeManager


class WaitTool(BaseTool):
    name = "wait"
    exec_only = False
    description = (
        "Waits on a yielded `exec` cell and returns new output or completion.\n"
        "- Use `wait` only after `exec` returns `Script running with cell ID ...`.\n"
        "- `cell_id` identifies the running `exec` cell to resume.\n"
        "- `yield_time_ms` controls how long to wait for more output before "
        "yielding again. Defaults to 10000 ms.\n"
        "- `max_tokens` limits how much new output this wait call returns. "
        "Defaults to 10000 tokens.\n"
        "- `terminate: true` stops the running cell; false or omitted waits "
        "for output.\n"
        "- `wait` returns only the new output since the last yield, or the "
        "final completion or termination result for that cell.\n"
        "- If the cell is still running, `wait` may yield again with the same "
        "`cell_id`.\n"
        "- If the cell has already finished, `wait` returns the completed "
        "result and closes the cell."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "cell_id": {
                "type": "string",
                "description": "Identifier of the running exec cell.",
            },
            "yield_time_ms": {
                "type": "number",
                "description": "Wait before yielding more output. Defaults to 10000 ms.",
            },
            "max_tokens": {
                "type": "number",
                "description": "Output token budget for this wait call. Defaults to 10000 tokens.",
            },
            "terminate": {
                "type": "boolean",
                "description": "True stops the running exec cell; false or omitted waits for output.",
            },
        },
        "required": ["cell_id"],
        "additionalProperties": False,
    }
    supports_parallel = False

    def __init__(self, manager: "CodeModeManager") -> "None":
        self._manager = manager

    async def run(self, context: "ToolContext", args: "JSONDict") -> "JSONValue":
        del context
        cell_id = args["cell_id"]
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("cell_id must be a non-empty string")
        terminate = args.get("terminate", False)
        if not isinstance(terminate, bool):
            raise ValueError("terminate must be a boolean")
        return await self._manager.wait(
            cell_id=cell_id,
            yield_time_ms=self._integer(
                args.get("yield_time_ms", DEFAULT_WAIT_YIELD_TIME_MS), "yield_time_ms",
            ),
            max_tokens=(
                self._integer(args["max_tokens"], "max_tokens")
                if "max_tokens" in args else None
            ),
            terminate=terminate,
        )

    def _integer(self, value, name):
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or int(value) != value or value < 0
        ):
            raise ValueError(name + " must be a non-negative integer")
        return int(value)
