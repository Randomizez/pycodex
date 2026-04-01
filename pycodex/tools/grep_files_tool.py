"""`grep_files` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `grep_files` tool.

Expected behavior:
- Search file contents with ripgrep and return only matching file paths.
- Support the original tool's `pattern`, `include`, `path`, and `limit`
  parameters rather than delegating to a shell transcript.
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
from pathlib import Path

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, ToolContext

DEFAULT_LIMIT = 100
MAX_LIMIT = 2000
COMMAND_TIMEOUT_SECONDS = 30


class GrepFilesTool(BaseTool):
    name = "grep_files"
    description = (
        "Finds files whose contents match the pattern and lists them by "
        "modification time."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "include": {"type": "string"},
            "path": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["pattern"],
    }

    def __init__(self, cwd: str | Path | None = None) -> None:
        self._working_directory = Path(cwd or Path.cwd()).resolve()

    async def run(self, context: ToolContext, args: JSONDict) -> JSONValue:
        del context
        pattern = str(args.get("pattern", "")).strip()
        include = str(args.get("include", "")).strip() or None
        limit = min(int(args.get("limit", DEFAULT_LIMIT)), MAX_LIMIT)
        path_arg = args.get("path")
        search_path = self._resolve_path(path_arg)

        if not pattern:
            return "Error: `pattern` must not be empty."
        if limit <= 0:
            return "Error: `limit` must be greater than zero."
        if not search_path.exists():
            return f"Error: unable to access `{search_path}`."

        command = [
            "rg",
            "--files-with-matches",
            "--sortr=modified",
            "--regexp",
            pattern,
            "--no-messages",
        ]
        if include is not None:
            command.extend(["--glob", include])
        command.extend(["--", str(search_path)])

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self._working_directory),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            results = self._search_with_python(pattern, include, search_path, limit)
            if not results:
                return "No matches found."
            return "\n".join(results)

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=COMMAND_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return "Error: rg timed out after 30 seconds."

        if process.returncode == 1:
            return "No matches found."
        if process.returncode != 0:
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            return f"Error: rg failed: {stderr}"

        results = [
            line
            for line in stdout_bytes.decode("utf-8", errors="replace").splitlines()
            if line.strip()
        ][:limit]
        if not results:
            return "No matches found."
        return "\n".join(results)

    def _search_with_python(
        self,
        pattern: str,
        include: str | None,
        search_path: Path,
        limit: int,
    ) -> list[str]:
        regex = re.compile(pattern)
        candidates: list[Path] = []

        if search_path.is_file():
            candidates = [search_path]
        else:
            candidates = [path for path in search_path.rglob("*") if path.is_file()]

        if include is not None:
            candidates = [
                path for path in candidates if fnmatch.fnmatch(path.name, include)
            ]

        matches = []
        for path in candidates:
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            if regex.search(text):
                matches.append(path)

        matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return [str(path) for path in matches[:limit]]

    def _resolve_path(self, path_arg) -> Path:
        if path_arg in (None, ""):
            return self._working_directory
        path = Path(str(path_arg))
        if not path.is_absolute():
            path = self._working_directory / path
        return path.resolve()
