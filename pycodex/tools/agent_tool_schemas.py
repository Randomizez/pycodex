"""Shared schemas for Codex-aligned collaboration tools."""

from __future__ import annotations

COLLAB_INPUT_ITEMS_SCHEMA = {
    "type": "array",
    "description": (
        "Structured input items. Use this to pass explicit mentions (for "
        "example app:// connector paths)."
    ),
    "items": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "description": "Input item type: text, image, local_image, skill, or mention.",
            },
            "text": {
                "type": "string",
                "description": "Text content when type is text.",
            },
            "image_url": {
                "type": "string",
                "description": "Image URL when type is image.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Path when type is local_image/skill, or structured mention target "
                    "such as app://<connector-id> or plugin://<plugin-name>@<marketplace-name> "
                    "when type is mention."
                ),
            },
            "name": {
                "type": "string",
                "description": "Display name when type is skill or mention.",
            },
        },
        "additionalProperties": False,
    },
}

AGENT_STATUS_SCHEMA = {
    "oneOf": [
        {
            "type": "string",
            "enum": ["pending_init", "running", "shutdown", "not_found"],
        },
        {
            "type": "object",
            "properties": {
                "completed": {
                    "type": ["string", "null"],
                }
            },
            "required": ["completed"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "errored": {
                    "type": "string",
                }
            },
            "required": ["errored"],
            "additionalProperties": False,
        },
    ]
}
