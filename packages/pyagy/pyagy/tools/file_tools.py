"""AGY file tools: read_file, write_file, edit_file."""

import base64
import json
import os
from pathlib import Path
from types import MappingProxyType
import typing

from pycodex.protocol import JSONDict, JSONValue
from pycodex.tools.base_tool import StructuredToolOutput, ToolContext

from . import AgyBaseTool, AgyToolError, format_sync_tool_header

MAX_READ_LINES = 800
MAX_READ_BYTES = 46080

_IMAGE_MIME_BY_SUFFIX: "typing.Mapping[str, str]" = MappingProxyType(
    {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
)


def _require_absolute_path(raw_path: "object") -> "typing.Tuple[str, Path]":
    if not isinstance(raw_path, str) or not raw_path:
        raise AgyToolError("Path must be a non-empty string")
    if not os.path.isabs(raw_path):
        raise AgyToolError(
            f"declaring permissions: failed to parse path: {raw_path} must be an absolute path: path is not absolute"
        )
    return raw_path, Path(raw_path)


def _detect_image_mime(path: "Path", raw_bytes: "bytes") -> "typing.Optional[str]":
    suffix = path.suffix.lower()
    if suffix in _IMAGE_MIME_BY_SUFFIX:
        return _IMAGE_MIME_BY_SUFFIX[suffix]
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(raw_bytes) >= 12 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    return None


class ReadFileTool(AgyBaseTool):
    """Implements AGY `read_file`."""

    async def run(
        self,
        context: "ToolContext",
        args: "JSONValue",
    ) -> "typing.Union[str, StructuredToolOutput]":
        created_at = self.now_local_iso()
        params = self.require_dict_args(args)
        raw_path, target = _require_absolute_path(params.get("Path"))
        if not target.exists():
            raise AgyToolError("file does not exist")
        if target.is_dir():
            raise AgyToolError(f"path {raw_path} is a directory")

        raw_bytes = target.read_bytes()
        completed_at = self.now_local_iso()
        header = format_sync_tool_header(created_at, completed_at)

        mime_type = _detect_image_mime(target, raw_bytes)
        if mime_type is not None:
            msg = f"{header}Showing binary file of type {mime_type}"
            b64 = base64.b64encode(raw_bytes).decode("ascii")
            return StructuredToolOutput(
                output=msg,
                content_items=(
                    {"type": "input_text", "text": msg},
                    {"type": "input_image", "image_url": f"data:{mime_type};base64,{b64}"},
                ),
                success=True,
            )

        if len(raw_bytes) == 0:
            return f"{header}File is empty"

        text = raw_bytes.decode("utf-8", errors="replace")
        all_lines = text.splitlines()
        total_lines = len(all_lines)
        if total_lines == 0:
            return f"{header}File is empty"

        start_line = 1 if params.get("StartLine") is None else int(params["StartLine"])
        end_line = total_lines if params.get("EndLine") is None else int(params["EndLine"])
        byte_offset = 0 if params.get("ByteOffset") is None else int(params["ByteOffset"])

        if start_line < 1:
            raise AgyToolError("StartLine must be >= 1")
        if end_line < start_line:
            raise AgyToolError("EndLine must be >= StartLine")
        if byte_offset < 0:
            raise AgyToolError("ByteOffset must be >= 0")
        if start_line > total_lines:
            raise AgyToolError(
                f"StartLine {start_line} is beyond end of file ({total_lines} lines)"
            )

        capped_end = min(end_line, total_lines, start_line + MAX_READ_LINES - 1)
        candidate_slice = all_lines[start_line - 1 : capped_end]

        if byte_offset > 0:
            joined = "\n".join(candidate_slice).encode("utf-8")
            if byte_offset >= len(joined):
                raise AgyToolError(
                    f"ByteOffset {byte_offset} exceeds selected range length ({len(joined)} bytes)"
                )
            sliced_bytes = joined[byte_offset : byte_offset + MAX_READ_BYTES]
            chunk_text = sliced_bytes.decode("utf-8", errors="replace")
            return (
                f"{header}Showing lines {start_line}-{capped_end} of {total_lines} "
                f"(byte offset {byte_offset})\n{chunk_text}"
            )

        selected_formatted: "typing.List[str]" = []
        accumulated_bytes = 0
        actual_end = start_line - 1
        for offset, line in enumerate(candidate_slice):
            ln_no = start_line + offset
            entry = f"{ln_no}: {line}"
            entry_bytes = len(entry.encode("utf-8")) + (1 if selected_formatted else 0)
            if selected_formatted and accumulated_bytes + entry_bytes > MAX_READ_BYTES:
                break
            if not selected_formatted and entry_bytes > MAX_READ_BYTES:
                prefix_str = f"{ln_no}: "
                budget = max(0, MAX_READ_BYTES - len(prefix_str.encode("utf-8")))
                truncated_line = line.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
                selected_formatted.append(f"{prefix_str}{truncated_line}")
                actual_end = ln_no
                break
            selected_formatted.append(entry)
            accumulated_bytes += entry_bytes
            actual_end = ln_no

        body = "\n".join(selected_formatted)
        return f"{header}Showing lines {start_line}-{actual_end} of {total_lines}\n{body}"


class WriteFileTool(AgyBaseTool):
    """Implements AGY `write_file`."""

    async def run(
        self,
        context: "ToolContext",
        args: "JSONValue",
    ) -> "str":
        created_at = self.now_local_iso()
        params = self.require_dict_args(args)
        raw_path, target = _require_absolute_path(params.get("Path"))
        content = params.get("Content")
        if not isinstance(content, str):
            raise AgyToolError("Content must be a string")

        overwrite = bool(params.get("Overwrite", False))
        append = bool(params.get("Append", False))
        if overwrite and append:
            raise AgyToolError("cannot set both Overwrite=true and Append=true")

        if target.exists():
            if target.is_dir():
                raise AgyToolError(f"path {raw_path} is a directory")
            if not overwrite and not append:
                raise AgyToolError(
                    f"file {raw_path} already exists; set Overwrite=true or Append=true"
                )

        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8") as fh:
            fh.write(content)

        artifact_meta = params.get("ArtifactMetadata")
        if isinstance(artifact_meta, dict):
            sys_dir = target.parent / ".system_generated"
            if sys_dir.is_dir():
                meta_dir = sys_dir / "artifacts"
                meta_dir.mkdir(parents=True, exist_ok=True)
                (meta_dir / f"{target.name}.metadata.json").write_text(
                    json.dumps(artifact_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        completed_at = self.now_local_iso()
        return f"{format_sync_tool_header(created_at, completed_at)}Created file."


class EditFileTool(AgyBaseTool):
    """Implements AGY `edit_file`."""

    async def run(
        self,
        context: "ToolContext",
        args: "JSONValue",
    ) -> "str":
        created_at = self.now_local_iso()
        params = self.require_dict_args(args)
        raw_path, target = _require_absolute_path(params.get("Path"))
        target_str = params.get("Target")
        replacement_str = params.get("Replacement")
        if not isinstance(target_str, str) or target_str == "":
            raise AgyToolError("Target must be a non-empty string")
        if not isinstance(replacement_str, str):
            raise AgyToolError("Replacement must be a string")

        if not target.exists():
            raise AgyToolError("file does not exist")
        if target.is_dir():
            raise AgyToolError(f"path {raw_path} is a directory")

        original = target.read_text(encoding="utf-8")
        count = original.count(target_str)
        if count == 0:
            raise AgyToolError("target string not found in file")

        first_idx = original.index(target_str)
        start_line = original.count("\n", 0, first_idx) + 1
        removed_per_match = len(target_str.splitlines()) if target_str else 0
        added_per_match = len(replacement_str.splitlines()) if replacement_str else 0
        added = count * added_per_match
        removed = count * removed_per_match

        updated = original.replace(target_str, replacement_str)
        target.write_text(updated, encoding="utf-8")

        completed_at = self.now_local_iso()
        plural = "" if count == 1 else "s"
        return (
            f"{format_sync_tool_header(created_at, completed_at)}"
            f"Made {count} replacement{plural} in {raw_path} starting at line {start_line} "
            f"(+{added}/-{removed} lines)."
        )
