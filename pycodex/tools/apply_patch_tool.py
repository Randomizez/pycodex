"""`apply_patch` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `apply_patch` freeform/custom tool.

Expected behavior:
- Accept the same freeform patch envelope used by Codex main.
- Verify the whole patch before mutating the filesystem so failed patches do not
  leave partial edits behind.
- Apply add/delete/update/move operations inside the workspace and return the
  same success/error text shape Codex expects.
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ..protocol import JSONValue
from .base_tool import BaseTool, ToolContext
import typing

APPLY_PATCH_LARK_GRAMMAR = """start: begin_patch hunk+ end_patch
begin_patch: \"*** Begin Patch\" LF
end_patch: \"*** End Patch\" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: \"*** Add File: \" filename LF add_line+
delete_hunk: \"*** Delete File: \" filename LF
update_hunk: \"*** Update File: \" filename LF change_move? change?

filename: /(.+)/
add_line: \"+\" /(.*)/ LF -> line

change_move: \"*** Move to: \" filename LF
change: (change_context | change_line)+ eof_line?
change_context: (\"@@\" | \"@@ \" /(.+)/) LF
change_line: (\"+\" | \"-\" | \" \" ) /(.*)/ LF
eof_line: \"*** End of File\" LF

%import common.LF
"""


class ApplyPatchError(RuntimeError):
    pass


@dataclass(frozen=True, )
class _AddFileOp:
    path: 'str'
    content: 'str'


@dataclass(frozen=True, )
class _DeleteFileOp:
    path: 'str'


@dataclass(frozen=True, )
class _UpdateSection:
    lines: 'typing.Tuple[str, ...]'
    anchor_end_of_file: 'bool' = False


@dataclass(frozen=True, )
class _UpdateFileOp:
    path: 'str'
    move_to: 'typing.Union[str, None]'
    sections: 'typing.Tuple[_UpdateSection, ...]'


class ApplyPatchTool(BaseTool):
    name = "apply_patch"
    description = (
        "Use the `apply_patch` tool to edit files. This is a FREEFORM tool, "
        "so do not wrap the patch in JSON."
    )
    tool_type = "custom"
    format = {
        "type": "grammar",
        "syntax": "lark",
        "definition": APPLY_PATCH_LARK_GRAMMAR,
    }
    supports_parallel = False

    def __init__(self, cwd: 'typing.Union[typing.Union[str, Path], None]' = None) -> 'None':
        self._workspace_root = Path(cwd or Path.cwd()).resolve()

    async def run(self, context: 'ToolContext', args: 'JSONValue') -> 'JSONValue':
        del context
        patch_text = str(args)
        logger.debug("apply_patch workspace={} bytes={}", self._workspace_root, len(patch_text))
        try:
            operations = self._parse_patch(patch_text)
            return self._format_result(self._apply_operations(operations), exit_code=0)
        except ApplyPatchError as exc:
            return self._format_result(str(exc), exit_code=1)

    def _parse_patch(self, patch_text: 'str') -> 'typing.List[typing.Union[typing.Union[_AddFileOp, _DeleteFileOp], _UpdateFileOp]]':
        lines = patch_text.splitlines()
        if not lines:
            raise ApplyPatchError("patch rejected: empty patch")
        if lines[0] != "*** Begin Patch":
            raise ApplyPatchError(
                "apply_patch verification failed: missing '*** Begin Patch' header"
            )

        operations: 'typing.List[typing.Union[typing.Union[_AddFileOp, _DeleteFileOp], _UpdateFileOp]]' = []
        index = 1
        while index < len(lines):
            line = lines[index]
            if line == "*** End Patch":
                if not operations:
                    raise ApplyPatchError("patch rejected: empty patch")
                for trailing in lines[index + 1 :]:
                    if trailing.strip():
                        raise ApplyPatchError(
                            "apply_patch verification failed: unexpected content after '*** End Patch'"
                        )
                return operations

            if line.startswith("*** Add File: "):
                path = line[len("*** Add File: ") :]
                index += 1
                content_lines: 'typing.List[str]' = []
                while index < len(lines) and not lines[index].startswith("*** "):
                    entry = lines[index]
                    if not entry.startswith("+"):
                        raise ApplyPatchError(
                            f"apply_patch verification failed: {entry!r} is not a valid add line"
                        )
                    content_lines.append(entry[1:])
                    index += 1
                if not content_lines:
                    raise ApplyPatchError(
                        f"apply_patch verification failed: add for {path} is missing file content"
                    )
                operations.append(_AddFileOp(path=path, content=self._join_lines(content_lines)))
                continue

            if line.startswith("*** Delete File: "):
                path = line[len("*** Delete File: ") :]
                operations.append(_DeleteFileOp(path=path))
                index += 1
                continue

            if line.startswith("*** Update File: "):
                path = line[len("*** Update File: ") :]
                index += 1
                move_to = None
                if index < len(lines) and lines[index].startswith("*** Move to: "):
                    move_to = lines[index][len("*** Move to: ") :]
                    index += 1

                sections: 'typing.List[_UpdateSection]' = []
                current_lines: 'typing.List[str]' = []
                saw_hunk_header = False
                anchor_end_of_file = False
                while index < len(lines):
                    entry = lines[index]
                    if entry.startswith("*** ") and entry != "*** End of File":
                        break
                    if entry == "@@" or entry.startswith("@@ "):
                        if saw_hunk_header:
                            sections.append(
                                _UpdateSection(
                                    lines=tuple(current_lines),
                                    anchor_end_of_file=anchor_end_of_file,
                                )
                            )
                            current_lines = []
                            anchor_end_of_file = False
                        saw_hunk_header = True
                        index += 1
                        continue
                    if entry == "*** End of File":
                        if not saw_hunk_header:
                            raise ApplyPatchError(
                                "apply_patch verification failed: '*** End of File' must follow a hunk"
                            )
                        anchor_end_of_file = True
                        index += 1
                        continue
                    if not saw_hunk_header:
                        raise ApplyPatchError(
                            f"apply_patch verification failed: {entry!r} is not a valid hunk header"
                        )
                    if not entry or entry[0] not in {" ", "+", "-"}:
                        raise ApplyPatchError(
                            f"apply_patch verification failed: {entry!r} is not a valid change line"
                        )
                    current_lines.append(entry)
                    index += 1

                if not saw_hunk_header:
                    raise ApplyPatchError(
                        f"apply_patch verification failed: update for {path} is missing a hunk"
                    )
                sections.append(
                    _UpdateSection(
                        lines=tuple(current_lines),
                        anchor_end_of_file=anchor_end_of_file,
                    )
                )
                operations.append(
                    _UpdateFileOp(
                        path=path,
                        move_to=move_to,
                        sections=tuple(sections),
                    )
                )
                continue

            raise ApplyPatchError(
                f"apply_patch verification failed: {line!r} is not a valid hunk header"
            )

        raise ApplyPatchError("apply_patch verification failed: missing '*** End Patch' footer")

    def _apply_operations(
        self,
        operations: 'typing.List[typing.Union[typing.Union[_AddFileOp, _DeleteFileOp], _UpdateFileOp]]',
    ) -> 'str':
        preview: 'typing.Dict[Path, typing.Union[str, None]]' = {}
        summaries: 'typing.Dict[Path, str]' = {}

        for operation in operations:
            if isinstance(operation, _AddFileOp):
                path = self._resolve_workspace_path(operation.path)
                preview[path] = operation.content
                summaries[path] = "A"
                continue

            if isinstance(operation, _DeleteFileOp):
                path = self._resolve_workspace_path(operation.path)
                self._read_preview_file(path, preview)
                preview[path] = None
                summaries[path] = "D"
                continue

            path = self._resolve_workspace_path(operation.path)
            original = self._read_preview_file(path, preview)
            updated = self._apply_update(path, original, operation.sections)
            destination = path
            if operation.move_to is not None:
                destination = self._resolve_workspace_path(operation.move_to)
                preview[path] = None
                summaries.pop(path, None)
            preview[destination] = updated
            summaries[destination] = "M"

        self._write_preview(preview)
        return self._format_success(summaries)

    def _read_preview_file(self, path: 'Path', preview: 'typing.Dict[Path, typing.Union[str, None]]') -> 'str':
        if path in preview:
            content = preview[path]
            if content is None:
                raise ApplyPatchError(
                    f"apply_patch verification failed: Failed to read {path.relative_to(self._workspace_root)}"
                )
            return content

        if not path.exists() or not path.is_file():
            raise ApplyPatchError(
                f"apply_patch verification failed: Failed to read {path.relative_to(self._workspace_root)}"
            )
        return path.read_text(encoding="utf-8", errors="replace")

    def _apply_update(
        self,
        path: 'Path',
        original_text: 'str',
        sections: 'typing.Tuple[_UpdateSection, ...]',
    ) -> 'str':
        lines = original_text.splitlines()
        cursor = 0
        for section in sections:
            old_block = [line[1:] for line in section.lines if line[:1] in {" ", "-"}]
            new_block = [line[1:] for line in section.lines if line[:1] in {" ", "+"}]
            if not old_block and not new_block:
                continue

            match_index = self._find_match(lines, old_block, cursor, section.anchor_end_of_file)
            if match_index is None:
                raise ApplyPatchError(
                    "apply_patch verification failed: Failed to find expected lines in "
                    f"{path.relative_to(self._workspace_root)}"
                )
            lines[match_index : match_index + len(old_block)] = new_block
            cursor = match_index + len(new_block)
        return self._join_lines(lines)

    def _find_match(
        self,
        lines: 'typing.List[str]',
        old_block: 'typing.List[str]',
        cursor: 'int',
        anchor_end_of_file: 'bool',
    ) -> 'typing.Union[int, None]':
        if anchor_end_of_file:
            start = len(lines) - len(old_block)
            if start >= 0 and lines[start : start + len(old_block)] == old_block:
                return start
            return None

        if not old_block:
            return cursor

        limit = len(lines) - len(old_block) + 1
        for start in range(max(cursor, 0), max(limit, 0)):
            if lines[start : start + len(old_block)] == old_block:
                return start
        for start in range(0, max(limit, 0)):
            if lines[start : start + len(old_block)] == old_block:
                return start
        return None

    def _write_preview(self, preview: 'typing.Dict[Path, typing.Union[str, None]]') -> 'None':
        for path, content in preview.items():
            if content is None:
                if path.exists():
                    path.unlink()
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def _format_success(self, summaries: 'typing.Dict[Path, str]') -> 'str':
        buckets = {"A": [], "M": [], "D": []}
        for path, status in summaries.items():
            buckets[status].append(path.relative_to(self._workspace_root).as_posix())
        lines = ["Success:"]
        for status in ("A", "M", "D"):
            for rel_path in sorted(buckets[status]):
                lines.append(f"{status} {rel_path}")
        return " ".join(lines) + "\n"

    def _format_result(self, output: 'str', exit_code: 'int') -> 'str':
        return (
            f"Exit code: {exit_code}\n"
            "Wall time: 0 seconds\n"
            "Output:\n"
            f"{output}"
        )

    def _resolve_workspace_path(self, path_text: 'str') -> 'Path':
        path = Path(path_text)
        resolved = path if path.is_absolute() else self._workspace_root / path
        resolved = resolved.resolve()
        try:
            resolved.relative_to(self._workspace_root)
        except ValueError as exc:
            raise ApplyPatchError(
                "patch rejected: writing outside of the project; rejected by user approval settings"
            ) from exc
        return resolved

    def _join_lines(self, lines: 'typing.List[str]') -> 'str':
        if not lines:
            return ""
        return "\n".join(lines) + "\n"
