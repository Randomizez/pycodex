"""Shared schemas for Codex-aligned collaboration tools."""

AGENT_STATUS_SCHEMA = {
    "oneOf": [
        {
            "type": "string",
            "enum": ["pending_init", "running", "interrupted", "shutdown", "not_found"],
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
