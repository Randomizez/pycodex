"""`wait` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex code-mode `wait` tool.

Expected behavior:
- Wait on a yielded `exec` cell for more output or completion.
- Optionally terminate the running cell.
- Return only the new output since the previous `exec` / `wait` snapshot.
"""

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, ToolContext
from .code_mode_manager import DEFAULT_WAIT_YIELD_TIME_MS, CodeModeManager
import typing


class WaitTool(BaseTool):
    name = "wait"
    description = (
        "Waits on a yielded `exec` cell and returns new output or completion."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "cell_id": {
                "type": "string",
                "description": "Identifier of the running exec cell.",
            },
            "yield_time_ms": {
                "type": "integer",
                "description": "How long to wait (in milliseconds) for more output before yielding again.",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Maximum number of output tokens to return for this wait call.",
            },
            "terminate": {
                "type": "boolean",
                "description": "Whether to terminate the running exec cell.",
            },
        },
        "required": ["cell_id"],
        "additionalProperties": False,
    }
    supports_parallel = False

    def __init__(self, manager: 'CodeModeManager') -> 'None':
        self._manager = manager

    async def run(self, context: 'ToolContext', args: 'JSONDict') -> 'JSONValue':
        del context
        cell_id = str(args.get("cell_id", "")).strip()
        if not cell_id:
            return "Error: `cell_id` is required."
        return await self._manager.wait(
            cell_id=cell_id,
            yield_time_ms=int(args.get("yield_time_ms", DEFAULT_WAIT_YIELD_TIME_MS)),
            max_tokens=self._optional_int(args, "max_tokens"),
            terminate=bool(args.get("terminate", False)),
        )

    def _optional_int(self, args: 'JSONDict', key: 'str') -> 'typing.Union[int, None]':
        value = args.get(key)
        if value in (None, ""):
            return None
        return int(value)
