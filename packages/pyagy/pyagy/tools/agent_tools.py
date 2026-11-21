"""AGY subagent & messaging tools: define_subagent, invoke_subagent, manage_subagents, send_message."""

import typing

from pycodex.protocol import JSONDict, JSONValue
from pycodex.tools.base_tool import ToolContext

from . import AgyBaseTool, AgyToolError, AgyToolRegistry, format_sync_tool_header

if typing.TYPE_CHECKING:
    from ..multi_agents import AgyMultiAgentController


class DefineSubagentTool(AgyBaseTool):
    """Implements AGY `define_subagent`."""

    def __init__(
        self,
        declaration: "JSONDict",
        registry: "AgyToolRegistry",
        conversation_id: "str",
        multi_agents: "AgyMultiAgentController",
    ) -> "None":
        super().__init__(declaration, registry)
        self._conversation_id = conversation_id
        self._multi_agents = multi_agents

    async def run(self, context: "ToolContext", args: "JSONValue") -> "str":
        created_at = self.now_local_iso()
        params = self.require_dict_args(args)
        name = params.get("Name")
        description = params.get("Description")
        system_prompt = params.get("SystemPrompt")
        if not isinstance(name, str) or not name.strip():
            raise AgyToolError("Name must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise AgyToolError("Description must be a non-empty string")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise AgyToolError("SystemPrompt must be a non-empty string")

        try:
            self._multi_agents.define_subagent(
                caller_conversation_id=context.session_id or self._conversation_id,
                name=name.strip(),
                description=description,
                system_prompt=system_prompt,
                enable_write_tools=bool(params.get("EnableWriteTools", False)),
                enable_subagent_tools=bool(params.get("EnableSubagentTools", False)),
                enable_mcp_tools=bool(params.get("EnableMCPTools", False)),
            )
        except ValueError as exc:
            raise AgyToolError(str(exc)) from exc

        completed_at = self.now_local_iso()
        return (
            f"{format_sync_tool_header(created_at, completed_at)}"
            f'Subagent "{name.strip()}" defined successfully. '
            "It can now be invoked via invoke_subagent."
        )


class InvokeSubagentTool(AgyBaseTool):
    """Implements AGY `invoke_subagent`."""

    def __init__(
        self,
        declaration: "JSONDict",
        registry: "AgyToolRegistry",
        conversation_id: "str",
        multi_agents: "AgyMultiAgentController",
    ) -> "None":
        super().__init__(declaration, registry)
        self._conversation_id = conversation_id
        self._multi_agents = multi_agents

    async def run(self, context: "ToolContext", args: "JSONValue") -> "str":
        created_at = self.now_local_iso()
        params = self.require_dict_args(args)
        type_name = params.get("TypeName")
        role = params.get("Role")
        prompt = params.get("Prompt")
        if not isinstance(type_name, str) or not type_name.strip():
            raise AgyToolError("TypeName must be a non-empty string")
        if not isinstance(role, str) or not role.strip():
            raise AgyToolError("Role must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise AgyToolError("Prompt must be a non-empty string")

        model_tier = params.get("Model")
        if model_tier is not None and not isinstance(model_tier, str):
            raise AgyToolError("Model must be a string")
        workspace_mode = params.get("Workspace")
        if workspace_mode is not None and not isinstance(workspace_mode, str):
            raise AgyToolError("Workspace must be a string")

        try:
            body = await self._multi_agents.invoke_subagent(
                caller_conversation_id=context.session_id or self._conversation_id,
                type_name=type_name.strip(),
                role=role.strip(),
                prompt=prompt,
                model_tier=model_tier,
                workspace_mode=workspace_mode,
            )
        except ValueError as exc:
            raise AgyToolError(str(exc)) from exc

        completed_at = self.now_local_iso()
        return f"{format_sync_tool_header(created_at, completed_at)}{body}"


class ManageSubagentsTool(AgyBaseTool):
    """Implements AGY `manage_subagents`."""

    def __init__(
        self,
        declaration: "JSONDict",
        registry: "AgyToolRegistry",
        conversation_id: "str",
        multi_agents: "AgyMultiAgentController",
    ) -> "None":
        super().__init__(declaration, registry)
        self._conversation_id = conversation_id
        self._multi_agents = multi_agents

    async def run(self, context: "ToolContext", args: "JSONValue") -> "str":
        created_at = self.now_local_iso()
        params = self.require_dict_args(args)
        action = params.get("Action")
        if action not in ("list", "kill", "kill_all"):
            raise AgyToolError(f"unsupported manage_subagents Action: {action!r}")

        conv_ids: "typing.Optional[typing.List[str]]" = None
        if action == "kill":
            raw_ids = params.get("ConversationIds")
            if not isinstance(raw_ids, list) or not raw_ids or not all(
                isinstance(item, str) and item.strip() for item in raw_ids
            ):
                raise AgyToolError("ConversationIds must be a non-empty array of strings for Action='kill'")
            conv_ids = [item.strip() for item in raw_ids]

        try:
            body = await self._multi_agents.manage_subagents(
                caller_conversation_id=context.session_id or self._conversation_id,
                action=action,
                conversation_ids=conv_ids,
            )
        except ValueError as exc:
            raise AgyToolError(str(exc)) from exc

        completed_at = self.now_local_iso()
        return f"{format_sync_tool_header(created_at, completed_at)}{body}"


class SendMessageTool(AgyBaseTool):
    """Implements AGY `send_message`."""

    def __init__(
        self,
        declaration: "JSONDict",
        registry: "AgyToolRegistry",
        conversation_id: "str",
        multi_agents: "AgyMultiAgentController",
    ) -> "None":
        super().__init__(declaration, registry)
        self._conversation_id = conversation_id
        self._multi_agents = multi_agents

    async def run(self, context: "ToolContext", args: "JSONValue") -> "str":
        created_at = self.now_local_iso()
        params = self.require_dict_args(args)
        recipient = params.get("Recipient")
        message = params.get("Message")
        if not isinstance(recipient, str) or not recipient.strip():
            raise AgyToolError("Recipient must be a non-empty string")
        if not isinstance(message, str) or not message.strip():
            raise AgyToolError("Message must be a non-empty string")

        try:
            await self._multi_agents.send_agent_message(
                sender_conversation_id=context.session_id or self._conversation_id,
                recipient_conversation_id=recipient.strip(),
                content=message,
            )
        except ValueError as exc:
            raise AgyToolError(str(exc)) from exc

        completed_at = self.now_local_iso()
        return (
            f"{format_sync_tool_header(created_at, completed_at)}"
            f'Message sent to "{recipient.strip()}".'
        )
