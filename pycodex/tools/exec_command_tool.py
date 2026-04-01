"""`exec_command` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `exec_command` tool.

Expected behavior:
- Start a command in a session-backed execution runtime.
- Return output immediately when the process finishes during the current call.
- Otherwise return a running `session_id` that can be continued via
  `write_stdin`.
"""

from __future__ import annotations

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, ToolContext
from .unified_exec_manager import (
    DEFAULT_EXEC_YIELD_TIME_MS,
    DEFAULT_LOGIN,
    DEFAULT_TTY,
    UNIFIED_EXEC_OUTPUT_SCHEMA,
    UnifiedExecManager,
)


class ExecCommandTool(BaseTool):
    name = "exec_command"
    description = "Runs a command in a PTY, returning output or a session ID for ongoing interaction."
    input_schema = {
        "type": "object",
        "properties": {
            "cmd": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "workdir": {
                "type": "string",
                "description": "Optional working directory to run the command in; defaults to the turn cwd.",
            },
            "shell": {
                "type": "string",
                "description": "Shell binary to launch. Defaults to the user's default shell.",
            },
            "login": {
                "type": "boolean",
                "description": "Whether to run the shell with -l/-i semantics. Defaults to true.",
            },
            "tty": {
                "type": "boolean",
                "description": "Whether to allocate a TTY for the command. Defaults to false (plain pipes); set to true to open a PTY and access TTY process.",
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
        "required": ["cmd"],
        "additionalProperties": False,
    }
    output_schema = UNIFIED_EXEC_OUTPUT_SCHEMA
    supports_parallel = False

    def __init__(self, manager: UnifiedExecManager) -> None:
        self._manager = manager

    async def run(self, context: ToolContext, args: JSONDict) -> JSONValue:
        del context
        cmd = str(args.get("cmd", "")).strip()
        if not cmd:
            return "Error: `cmd` is required."

        return await self._manager.exec_command(
            cmd=cmd,
            workdir=self._optional_string(args, "workdir"),
            shell=self._optional_string(args, "shell"),
            login=bool(args.get("login", DEFAULT_LOGIN)),
            tty=bool(args.get("tty", DEFAULT_TTY)),
            yield_time_ms=int(args.get("yield_time_ms", DEFAULT_EXEC_YIELD_TIME_MS)),
            max_output_tokens=self._optional_int(args, "max_output_tokens"),
        )

    def _optional_string(self, args: JSONDict, key: str) -> str | None:
        value = args.get(key)
        if value in (None, ""):
            return None
        return str(value)

    def _optional_int(self, args: JSONDict, key: str) -> int | None:
        value = args.get(key)
        if value in (None, ""):
            return None
        return int(value)
