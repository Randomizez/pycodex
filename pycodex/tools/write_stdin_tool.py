"""`write_stdin` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `write_stdin` tool.

Expected behavior:
- Write bytes into an existing `exec_command` session.
- Also support polling with empty input to fetch fresh output from a running
  session.
- Reuse the same `session_id` until the process exits.
"""

from __future__ import annotations

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, ToolContext
from .unified_exec_manager import (
    DEFAULT_WRITE_STDIN_YIELD_TIME_MS,
    UNIFIED_EXEC_OUTPUT_SCHEMA,
    UnifiedExecManager,
)


class WriteStdinTool(BaseTool):
    name = "write_stdin"
    description = "Writes characters to an existing unified exec session and returns recent output."
    input_schema = {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "integer",
                "description": "Identifier of the running unified exec session.",
            },
            "chars": {
                "type": "string",
                "description": "Bytes to write to stdin (may be empty to poll).",
            },
            "yield_time_ms": {
                "type": "integer",
                "description": "How long to wait (in milliseconds) for output before yielding.",
            },
            "max_output_tokens": {
                "type": "integer",
                "description": "Maximum number of tokens to return. Excess output will be truncated.",
            },
        },
        "required": ["session_id"],
        "additionalProperties": False,
    }
    output_schema = UNIFIED_EXEC_OUTPUT_SCHEMA
    supports_parallel = False

    def __init__(self, manager: UnifiedExecManager) -> None:
        self._manager = manager

    async def run(self, context: ToolContext, args: JSONDict) -> JSONValue:
        del context
        session_id = args.get("session_id")
        if session_id is None:
            return "Error: `session_id` is required."

        return await self._manager.write_stdin(
            session_id=int(session_id),
            chars=str(args.get("chars", "")),
            yield_time_ms=int(
                args.get("yield_time_ms", DEFAULT_WRITE_STDIN_YIELD_TIME_MS)
            ),
            max_output_tokens=self._optional_int(args, "max_output_tokens"),
        )

    def _optional_int(self, args: JSONDict, key: str) -> int | None:
        value = args.get(key)
        if value in (None, ""):
            return None
        return int(value)
