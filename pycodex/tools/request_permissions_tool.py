"""`request_permissions` tool for the Python Codex prototype.

Original Codex mapping:
- Corresponds to the original Codex `request_permissions` collaboration tool.

Expected behavior:
- Ask the user to approve additional filesystem or network permissions.
- Wait for an interactive response from the client/UI layer.
- Return the granted permission profile and scope so later tool calls can use it.
"""

from ..protocol import JSONDict, JSONValue
from ..runtime_services import RequestPermissionsManager
from .base_tool import BaseTool, ToolContext

NETWORK_PERMISSIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "enabled": {
            "type": "boolean",
            "description": "Set to true to request network access.",
        }
    },
    "additionalProperties": False,
}

FILE_SYSTEM_PERMISSIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "read": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Absolute paths to grant read access to.",
        },
        "write": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Absolute paths to grant write access to.",
        },
    },
    "additionalProperties": False,
}

REQUEST_PERMISSION_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "network": NETWORK_PERMISSIONS_SCHEMA,
        "file_system": FILE_SYSTEM_PERMISSIONS_SCHEMA,
    },
    "additionalProperties": False,
}


class RequestPermissionsTool(BaseTool):
    name = "request_permissions"
    description = (
        "Request additional filesystem or network permissions from the user "
        "and wait for a response."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional short explanation for why additional permissions are needed.",
            },
            "permissions": REQUEST_PERMISSION_PROFILE_SCHEMA,
        },
        "required": ["permissions"],
        "additionalProperties": False,
    }
    supports_parallel = False

    def __init__(self, request_manager: 'RequestPermissionsManager') -> 'None':
        self._request_manager = request_manager

    async def run(self, context: 'ToolContext', args: 'JSONDict') -> 'JSONValue':
        del context
        permissions = args.get("permissions")
        if not isinstance(permissions, dict):
            return "Error: `permissions` must be an object."
        if not permissions:
            return "Error: request_permissions requires at least one permission."

        response = await self._request_manager.request(
            {
                "reason": None if args.get("reason") in (None, "") else str(args.get("reason")),
                "permissions": permissions,
            }
        )
        if response is None:
            return "Error: request_permissions was cancelled before receiving a response."
        return response
