"""`exec_command` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `exec_command` tool.

Expected behavior:
- Start a command in a session-backed execution runtime.
- Return output immediately when the process finishes during the current call.
- Otherwise return a running `session_id` that can be continued via
  `write_stdin`.
"""

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, ToolContext
from .unified_exec_manager import (
    DEFAULT_EXEC_YIELD_TIME_MS,
    DEFAULT_LOGIN,
    DEFAULT_TTY,
    UNIFIED_EXEC_OUTPUT_SCHEMA,
    UnifiedExecManager,
)
import typing

MIN_EXEC_YIELD_TIME_MS = 250
MAX_EXEC_YIELD_TIME_MS = 30_000


class ExecCommandTool(BaseTool):
    name = "exec_command"
    description = (
        "Runs a command in a PTY, returning output or a session ID for ongoing interaction. "
        "For long tasks, you can reply first; when the task finishes, you will be "
        "invoked to continue."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "cmd": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for the command. Defaults to the turn cwd.",
            },
            "tty": {
                "type": "boolean",
                "description": "True allocates a PTY for the command; false or omitted uses plain pipes.",
            },
            "yield_time_ms": {
                "type": "number",
                "description": "Wait before yielding output. Defaults to 10000 ms; effective range is 250-30000 ms.",
            },
            "max_output_tokens": {
                "type": "number",
                "description": "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy.",
            },
            "shell": {
                "type": "string",
                "description": "Shell binary to launch. Defaults to the user's default shell.",
            },
            "login": {
                "type": "boolean",
                "description": "True runs the shell with -l/-i semantics; false disables them. Defaults to true.",
            },
        },
        "required": ["cmd"],
        "additionalProperties": False,
    }
    output_schema = UNIFIED_EXEC_OUTPUT_SCHEMA
    supports_parallel = False

    def __init__(self, manager: 'UnifiedExecManager') -> 'None':
        self._manager = manager

    async def run(self, context: 'ToolContext', args: 'JSONDict') -> 'JSONValue':
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
            yield_time_ms=self._bounded_int(
                args,
                "yield_time_ms",
                DEFAULT_EXEC_YIELD_TIME_MS,
                MIN_EXEC_YIELD_TIME_MS,
                MAX_EXEC_YIELD_TIME_MS,
            ),
            max_output_tokens=self._optional_int(args, "max_output_tokens"),
        )

    def _optional_string(self, args: 'JSONDict', key: 'str') -> 'typing.Union[str, None]':
        value = args.get(key)
        if value in (None, ""):
            return None
        return str(value)

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
