"""`view_image` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `view_image` tool.

Expected behavior:
- Load a local image file and turn it into a data URL that can be attached back
  into the next model request.
- Accept the documented `path` argument plus the optional `detail: "original"`
  hint.
- Return both the JSON object result and the structured `input_image` content
  item that Codex uses when feeding image tool output back to the model.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, StructuredToolOutput, ToolContext

VIEW_IMAGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_url": {
            "type": "string",
            "description": "Data URL for the loaded image.",
        },
        "detail": {
            "type": ["string", "null"],
            "description": "Image detail hint returned by view_image. Returns `original` when original resolution is preserved, otherwise `null`.",
        },
    },
    "required": ["image_url", "detail"],
    "additionalProperties": False,
}


class ViewImageTool(BaseTool):
    name = "view_image"
    description = (
        "View a local image from the filesystem (only use if given a full "
        "filepath by the user, and the image isn't already attached to the "
        "thread context within <image ...> tags)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Local filesystem path to an image file",
            },
            "detail": {
                "type": "string",
                "description": "Optional detail override. The only supported value is `original`; omit this field for default resized behavior. Use `original` to preserve the file's original resolution instead of resizing to fit.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    output_schema = VIEW_IMAGE_OUTPUT_SCHEMA

    def __init__(self, cwd: str | Path | None = None) -> None:
        self._workspace_root = Path(cwd or Path.cwd()).resolve()

    async def run(self, context: ToolContext, args: JSONDict) -> JSONValue:
        del context
        path_value = str(args.get("path", "")).strip()
        if not path_value:
            return "Error: `path` is required."

        detail_value = args.get("detail")
        if detail_value in (None, ""):
            detail = None
        elif detail_value == "original":
            detail = "original"
        else:
            return (
                "Error: `detail` only supports `original`; omit `detail` for default "
                f"behavior, got `{detail_value}`."
            )

        path = Path(path_value)
        if not path.is_absolute():
            path = self._workspace_root / path
        path = path.resolve()
        if not path.exists():
            return f"Error: unable to locate image at `{path}`."
        if not path.is_file():
            return f"Error: image path `{path}` is not a file."

        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type or not mime_type.startswith("image/"):
            return f"Error: `{path}` does not look like an image file."

        image_bytes = path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        image_url = f"data:{mime_type};base64,{encoded}"
        output = {
            "image_url": image_url,
            "detail": detail,
        }
        image_item: JSONDict = {
            "type": "input_image",
            "image_url": image_url,
        }
        if detail is not None:
            image_item["detail"] = detail
        return StructuredToolOutput(output=output, content_items=(image_item,))
