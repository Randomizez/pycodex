"""`view_image` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `view_image` tool.

Expected behavior:
- Load a local image file and turn it into a data URL that can be attached back
  into the next model request.
- Accept the documented `path` argument plus optional `detail: "high" |
  "original"` hint.
- Return both the JSON object result and the structured `input_image` content
  item that Codex uses when feeding image tool output back to the model.
"""

import base64
import mimetypes
from pathlib import Path

from ..protocol import JSONDict, JSONValue
from .base_tool import BaseTool, StructuredToolOutput, ToolContext
import typing

VIEW_IMAGE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "image_url": {
            "type": "string",
            "description": "Data URL for the loaded image.",
        },
        "detail": {
            "type": "string",
            "enum": ["high", "original"],
            "description": "Image detail hint returned by view_image. Returns `high` for default resized behavior or `original` when original resolution is preserved.",
        },
    },
    "required": ["image_url", "detail"],
    "additionalProperties": False,
}


class ViewImageTool(BaseTool):
    name = "view_image"
    description = (
        "View a local image file from the filesystem when visual inspection is "
        "needed. Use this for images already available on disk."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Local filesystem path to an image file.",
            },
            "detail": {
                "type": "string",
                "enum": ["high", "original"],
                "description": "Image detail level. Defaults to `high`; use `original` to preserve exact resolution.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    output_schema = VIEW_IMAGE_OUTPUT_SCHEMA

    def __init__(self, cwd: 'typing.Union[typing.Union[str, Path], None]' = None) -> 'None':
        self._workspace_root = Path(cwd or Path.cwd()).resolve()

    async def run(self, context: 'ToolContext', args: 'JSONDict') -> 'JSONValue':
        del context
        path_value = str(args.get("path", "")).strip()
        if not path_value:
            return "Error: `path` is required."

        detail_value = args.get("detail")
        if detail_value in (None, ""):
            detail = "high"
        elif detail_value in ("high", "original"):
            detail = str(detail_value)
        else:
            return (
                "Error: `detail` only supports `high` or `original`; omit "
                "`detail` for default high resized behavior, got "
                f"`{detail_value}`."
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
        image_item: 'JSONDict' = {
            "type": "input_image",
            "image_url": image_url,
            "detail": detail,
        }
        return StructuredToolOutput(output=output, content_items=(image_item,))
