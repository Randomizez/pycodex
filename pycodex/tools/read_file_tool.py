"""`read_file` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `read_file` tool.

Expected behavior:
- Read a local file with 1-indexed line numbers.
- Support the original tool's core slice mode and the indentation-aware block
  mode used to inspect code structure without shelling out to `sed` or `cat`.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, ToolContext

MAX_LINE_LENGTH = 500
TAB_WIDTH = 4
COMMENT_PREFIXES = ("#", "//", "--")


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Reads a local file with 1-indexed line numbers, supporting slice and "
        "indentation-aware block modes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
            "mode": {"type": "string"},
            "indentation": {
                "type": "object",
                "properties": {
                    "anchor_line": {"type": "integer"},
                    "max_levels": {"type": "integer"},
                    "include_siblings": {"type": "boolean"},
                    "include_header": {"type": "boolean"},
                    "max_lines": {"type": "integer"},
                },
            },
        },
        "required": ["file_path"],
    }

    async def run(self, context: ToolContext, args: JSONDict) -> JSONValue:
        del context
        file_path = Path(str(args.get("file_path", "")))
        offset = int(args.get("offset", 1))
        limit = int(args.get("limit", 2000))
        mode = str(args.get("mode", "slice"))
        indentation = args.get("indentation") or {}

        if not file_path.is_absolute():
            return "Error: `file_path` must be an absolute path."
        if offset <= 0:
            return "Error: `offset` must be a 1-indexed line number."
        if limit <= 0:
            return "Error: `limit` must be greater than zero."
        if not file_path.exists():
            return f"Error: `{file_path}` does not exist."
        if not file_path.is_file():
            return f"Error: `{file_path}` is not a file."

        if mode == "indentation":
            return self._read_indentation(file_path, offset, limit, indentation)
        return self._read_slice(file_path, offset, limit)

    def _read_slice(self, file_path: Path, offset: int, limit: int) -> str:
        lines = file_path.read_text(errors="replace").splitlines()
        if offset > len(lines):
            return "Error: `offset` exceeds file length."

        selected = lines[offset - 1 : offset - 1 + limit]
        return "\n".join(
            f"L{line_number}: {self._format_line(text)}"
            for line_number, text in enumerate(selected, start=offset)
        )

    def _read_indentation(
        self,
        file_path: Path,
        offset: int,
        limit: int,
        indentation: JSONDict,
    ) -> str:
        lines = self._collect_line_records(file_path)
        anchor_line = int(indentation.get("anchor_line", offset))
        max_levels = int(indentation.get("max_levels", 0))
        include_siblings = bool(indentation.get("include_siblings", False))
        include_header = bool(indentation.get("include_header", True))
        max_lines = int(indentation.get("max_lines", limit))

        if anchor_line <= 0:
            return "Error: `anchor_line` must be a 1-indexed line number."
        if anchor_line > len(lines):
            return "Error: `anchor_line` exceeds file length."
        if max_lines <= 0:
            return "Error: `max_lines` must be greater than zero."

        anchor_index = anchor_line - 1
        effective_indents = self._compute_effective_indents(lines)
        anchor_indent = effective_indents[anchor_index]
        min_indent = 0 if max_levels == 0 else max(anchor_indent - max_levels * TAB_WIDTH, 0)
        final_limit = min(limit, max_lines, len(lines))

        if final_limit == 1:
            record = lines[anchor_index]
            return f"L{record['number']}: {record['display']}"

        upper = anchor_index - 1
        lower = anchor_index + 1
        upper_min_indent_count = 0
        lower_min_indent_count = 0
        selected = deque([lines[anchor_index]])

        while len(selected) < final_limit:
            progressed = 0

            if upper >= 0:
                if effective_indents[upper] >= min_indent:
                    selected.appendleft(lines[upper])
                    progressed += 1
                    if effective_indents[upper] == min_indent and not include_siblings:
                        allow_header_comment = include_header and lines[upper]["is_comment"]
                        can_take_line = allow_header_comment or upper_min_indent_count == 0
                        if can_take_line:
                            upper_min_indent_count += 1
                        else:
                            selected.popleft()
                            progressed -= 1
                            upper = -1
                            if progressed == 0 and lower >= len(lines):
                                break
                            continue
                    upper -= 1
                else:
                    upper = -1

            if len(selected) >= final_limit:
                break

            if lower < len(lines):
                if effective_indents[lower] >= min_indent:
                    selected.append(lines[lower])
                    progressed += 1
                    if effective_indents[lower] == min_indent and not include_siblings:
                        if lower_min_indent_count > 0:
                            selected.pop()
                            progressed -= 1
                            lower = len(lines)
                            if progressed == 0 and upper < 0:
                                break
                            continue
                        lower_min_indent_count += 1
                    lower += 1
                else:
                    lower = len(lines)

            if progressed == 0:
                break

        self._trim_empty_lines(selected)
        return "\n".join(
            f"L{record['number']}: {record['display']}" for record in selected
        )

    def _collect_line_records(self, file_path: Path) -> list[dict[str, object]]:
        records = []
        for number, raw in enumerate(file_path.read_text(errors="replace").splitlines(), start=1):
            records.append(
                {
                    "number": number,
                    "raw": raw,
                    "display": self._format_line(raw),
                    "indent": self._measure_indent(raw),
                    "is_comment": raw.strip().startswith(COMMENT_PREFIXES),
                }
            )
        return records

    def _compute_effective_indents(self, records: list[dict[str, object]]) -> list[int]:
        effective = []
        previous_indent = 0
        for record in records:
            if not str(record["raw"]).strip():
                effective.append(previous_indent)
            else:
                previous_indent = int(record["indent"])
                effective.append(previous_indent)
        return effective

    def _measure_indent(self, line: str) -> int:
        total = 0
        for character in line:
            if character == " ":
                total += 1
            elif character == "\t":
                total += TAB_WIDTH
            else:
                break
        return total

    def _format_line(self, text: str) -> str:
        return text[:MAX_LINE_LENGTH]

    def _trim_empty_lines(self, records: deque[dict[str, object]]) -> None:
        while records and not str(records[0]["raw"]).strip():
            records.popleft()
        while records and not str(records[-1]["raw"]).strip():
            records.pop()
