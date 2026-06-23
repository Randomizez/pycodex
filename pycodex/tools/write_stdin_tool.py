"""`write_stdin` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `write_stdin` tool.

Expected behavior:
- Write bytes into an existing `exec_command` session.
- Also support polling with empty input to fetch fresh output from a running
  session.
- Reuse the same `session_id` until the process exits.
"""

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, ToolContext
from .unified_exec_manager import (
    DEFAULT_WRITE_STDIN_YIELD_TIME_MS,
    UNIFIED_EXEC_OUTPUT_SCHEMA,
    UnifiedExecManager,
)
import typing

MIN_WRITE_YIELD_TIME_MS = 250
MAX_WRITE_YIELD_TIME_MS = 30_000
DEFAULT_WRITE_STDIN_POLL_YIELD_TIME_MS = 5_000
MAX_WRITE_STDIN_POLL_YIELD_TIME_MS = 300_000


class WriteStdinTool(BaseTool):
    name = "write_stdin"
    description = "Writes characters to an existing unified exec session and returns recent output."
    input_schema = {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "number",
                "description": "Identifier of the running unified exec session.",
            },
            "chars": {
                "type": "string",
                "description": "Bytes to write to stdin. Defaults to empty, which polls without writing.",
            },
            "yield_time_ms": {
                "type": "number",
                "description": "Wait before yielding output. Non-empty writes default to 250 ms and cap at 30000 ms; empty polls wait 5000-300000 ms by default.",
            },
            "max_output_tokens": {
                "type": "number",
                "description": "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy.",
            },
        },
        "required": ["session_id"],
        "additionalProperties": False,
    }
    output_schema = UNIFIED_EXEC_OUTPUT_SCHEMA
    supports_parallel = False

    def __init__(self, manager: 'UnifiedExecManager') -> 'None':
        self._manager = manager

    async def run(self, context: 'ToolContext', args: 'JSONDict') -> 'JSONValue':
        del context
        session_id = args.get("session_id")
        if session_id is None:
            return "Error: `session_id` is required."
        chars = str(args.get("chars", ""))

        return await self._manager.write_stdin(
            session_id=int(session_id),
            chars=chars,
            yield_time_ms=self._yield_time_ms(args, chars),
            max_output_tokens=self._optional_int(args, "max_output_tokens"),
        )

    def _yield_time_ms(self, args: 'JSONDict', chars: 'str') -> 'int':
        if chars:
            return self._bounded_int(
                args,
                "yield_time_ms",
                DEFAULT_WRITE_STDIN_YIELD_TIME_MS,
                MIN_WRITE_YIELD_TIME_MS,
                MAX_WRITE_YIELD_TIME_MS,
            )
        return self._bounded_int(
            args,
            "yield_time_ms",
            DEFAULT_WRITE_STDIN_POLL_YIELD_TIME_MS,
            DEFAULT_WRITE_STDIN_POLL_YIELD_TIME_MS,
            MAX_WRITE_STDIN_POLL_YIELD_TIME_MS,
        )

    def _optional_int(self, args: 'JSONDict', key: 'str') -> 'typing.Union[int, None]':
        value = args.get(key)
        if value in (None, ""):
            return None
        return int(value)

    def _bounded_int(
        self,
        args: 'JSONDict',
        key: 'str',
        default: 'int',
        minimum: 'int',
        maximum: 'int',
    ) -> 'int':
        value = int(args.get(key, default))
        return min(max(value, minimum), maximum)
