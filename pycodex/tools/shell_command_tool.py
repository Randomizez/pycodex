"""`shell_command` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `shell_command` tool.

Expected behavior:
- Execute a shell-script string in the user's shell context.
- Accept the same core input shape as Codex: `command: string`, plus `workdir`,
  `timeout_ms`, and `login`.
- Return a concise text summary including working directory, exit status,
  stdout, and stderr.
"""

import asyncio
from pathlib import Path

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, ToolContext
import typing

DEFAULT_SHELL_TIMEOUT_MS = 30_000
MAX_OUTPUT_CHARS = 12_000


class ShellCommandTool(BaseTool):
    name = "shell_command"
    description = "Runs a shell command string and returns its output."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "workdir": {"type": "string"},
            "timeout_ms": {"type": "integer"},
            "login": {"type": "boolean"},
        },
        "required": ["command"],
    }
    supports_parallel = False

    def __init__(self, cwd: 'typing.Union[typing.Union[str, Path], None]' = None) -> 'None':
        self._working_directory = Path(cwd or Path.cwd()).resolve()

    async def run(self, context: 'ToolContext', args: 'JSONDict') -> 'JSONValue':
        del context
        command = str(args.get("command", "")).strip()
        timeout_ms = int(args.get("timeout_ms", DEFAULT_SHELL_TIMEOUT_MS))
        if not command:
            return "Error: `command` is required."

        login = bool(args.get("login", True))
        workdir_arg = args.get("workdir")
        working_directory = self._resolve_workdir(workdir_arg)
        shell_args = ["bash", "-lc" if login else "-c", command]

        process = await asyncio.create_subprocess_exec(
            *shell_args,
            cwd=str(working_directory),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=max(timeout_ms, 1) / 1000.0,
            )
            timed_out = False
        except asyncio.TimeoutError:
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
            timed_out = True

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        pieces = [f"Working directory: {working_directory}"]

        if timed_out:
            pieces.append(f"Timeout: exceeded {timeout_ms} ms")
        else:
            pieces.append(f"Exit code: {process.returncode}")

        stdout = self._clip_output(stdout)
        stderr = self._clip_output(stderr)

        if stdout:
            pieces.append("Stdout:")
            pieces.append(stdout)

        if stderr:
            pieces.append("Stderr:")
            pieces.append(stderr)

        return "\n".join(pieces)

    def _resolve_workdir(self, workdir_arg) -> 'Path':
        if workdir_arg in (None, ""):
            return self._working_directory
        workdir = Path(str(workdir_arg))
        if not workdir.is_absolute():
            workdir = self._working_directory / workdir
        return workdir.resolve()

    def _clip_output(self, text: 'str') -> 'str':
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        return text[:MAX_OUTPUT_CHARS] + "\n...[truncated]..."
