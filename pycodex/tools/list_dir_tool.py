"""`list_dir` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `list_dir` tool.

Expected behavior:
- List directory entries from an absolute path with offset/limit/depth controls.
- Produce a stable, human-readable directory tree slice instead of using shell
  commands like `find` or `ls -R`.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, ToolContext

MAX_ENTRY_LENGTH = 500
INDENTATION_SPACES = 2


class ListDirTool(BaseTool):
    name = "list_dir"
    description = (
        "Lists entries in a local directory with 1-indexed entry numbers and "
        "simple type labels."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "dir_path": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
            "depth": {"type": "integer"},
        },
        "required": ["dir_path"],
    }

    async def run(self, context: ToolContext, args: JSONDict) -> JSONValue:
        del context
        dir_path = Path(str(args.get("dir_path", "")))
        offset = int(args.get("offset", 1))
        limit = int(args.get("limit", 25))
        depth = int(args.get("depth", 2))

        if not dir_path.is_absolute():
            return "Error: `dir_path` must be an absolute path."
        if offset <= 0:
            return "Error: `offset` must be a 1-indexed entry number."
        if limit <= 0:
            return "Error: `limit` must be greater than zero."
        if depth <= 0:
            return "Error: `depth` must be greater than zero."
        if not dir_path.exists():
            return f"Error: `{dir_path}` does not exist."
        if not dir_path.is_dir():
            return f"Error: `{dir_path}` is not a directory."

        entries = self._collect_entries(dir_path, depth)
        if not entries:
            return f"Absolute path: {dir_path}"

        start_index = offset - 1
        if start_index >= len(entries):
            return "Error: `offset` exceeds directory entry count."

        end_index = min(start_index + limit, len(entries))
        selected = entries[start_index:end_index]
        lines = [f"Absolute path: {dir_path}"]
        lines.extend(self._format_entry_line(entry) for entry in selected)
        if end_index < len(entries):
            lines.append(f"More than {len(selected)} entries found")
        return "\n".join(lines)

    def _collect_entries(self, root: Path, depth: int) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        queue = deque([(root, Path(), depth)])

        while queue:
            current_dir, prefix, remaining_depth = queue.popleft()
            dir_entries = []
            for child in current_dir.iterdir():
                relative_path = prefix / child.name if prefix.parts else Path(child.name)
                kind = self._entry_kind(child)
                dir_entries.append(
                    (
                        child,
                        relative_path,
                        {
                            "name": self._format_entry_name(relative_path),
                            "display_name": self._format_component(child.name),
                            "depth": len(prefix.parts),
                            "kind": kind,
                        },
                    )
                )

            dir_entries.sort(key=lambda item: item[2]["name"])
            for child, relative_path, entry in dir_entries:
                if entry["kind"] == "directory" and remaining_depth > 1:
                    queue.append((child, relative_path, remaining_depth - 1))
                entries.append(entry)

        entries.sort(key=lambda entry: entry["name"])
        return entries

    def _entry_kind(self, path: Path) -> str:
        if path.is_symlink():
            return "symlink"
        if path.is_dir():
            return "directory"
        if path.is_file():
            return "file"
        return "other"

    def _format_entry_name(self, path: Path) -> str:
        text = path.as_posix()
        return text[:MAX_ENTRY_LENGTH]

    def _format_component(self, name: str) -> str:
        return name[:MAX_ENTRY_LENGTH]

    def _format_entry_line(self, entry: dict[str, object]) -> str:
        indent = " " * (int(entry["depth"]) * INDENTATION_SPACES)
        name = str(entry["display_name"])
        kind = str(entry["kind"])
        if kind == "directory":
            name += "/"
        elif kind == "symlink":
            name += "@"
        elif kind == "other":
            name += "?"
        return f"{indent}{name}"
