"""V2 coordination for all agents in one conversation tree."""

import asyncio
import re
import typing
from collections import OrderedDict
from dataclasses import replace
from typing import TYPE_CHECKING

from .agent import Agent
from .context import ContextConfig
from .events import (
    CommandStartedEvent,
    CompactStartedEvent,
    Event,
    RuntimeUpdatedEvent,
    SessionClosedEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnInterruptedEvent,
    TurnStartedEvent,
)
from .protocol import AgentMessage, ConversationItem
from .tools import get_tools
from .tools.runtime_services import create_agent_runtime_environment
from .utils import uuid7_string
from .utils.session_persist import rollout_path_for_session

if TYPE_CHECKING:
    from .model import ModelControl
    from .runtime import AgentRuntime


class AgentsManager:
    """Create and manage every agent in one root conversation tree."""

    def __init__(
        self,
        model_client: "ModelControl",
        context_config: "ContextConfig",
        toolset: "typing.Union[typing.Iterable[str], None]" = None,
        tools_factory=get_tools,
    ) -> "None":
        self._runtimes: "OrderedDict[str, AgentRuntime]" = OrderedDict()
        self._closing_children = {}
        self._spawn_lock = None
        self._root_model_client = model_client
        self._context_config = context_config
        self._root_toolset = None if toolset is None else tuple(toolset)
        self._tools_factory = tools_factory

    def build_agent(
        self,
        parent_agent_id,
        agent_path,
        model_override=None,
        reasoning_effort_override=None,
        initial_history=(),
    ) -> "Agent":
        session_id = uuid7_string()
        runtime_environment = create_agent_runtime_environment()
        if parent_agent_id is None: # root agent
            model_client = self._root_model_client
            context_config = self._context_config
            toolset = self._root_toolset
            session_file_path = rollout_path_for_session(
                context_config.codex_home, session_id,
            )
        else:
            parent_agent = self._runtimes[parent_agent_id].agent
            model_client = parent_agent.model_client.with_overrides(
                model_override,
                reasoning_effort_override,
                session_id=session_id,
                openai_subagent="collab_spawn",
            )
            context_config = replace(
                parent_agent.context_manager.config,
                cwd=parent_agent.context_manager.cwd,
                base_instructions=parent_agent.context_manager.resolve_base_instructions(),
            )
            toolset = parent_agent.tool_registry.names()
            session_file_path = None
        return Agent(
            model_client,
            self._tools_factory(self, runtime_environment, toolset),
            context_config,
            initial_history=tuple(initial_history),
            session_file_path=session_file_path,
            session_id=session_id,
            agent_path=agent_path,
        )

    def create_root_runtime(self) -> "AgentRuntime":
        if self._runtimes:
            raise ValueError("root agent already exists")
        return self._create_runtime(None, "/root", None, None, ())

    def _create_runtime(
        self, parent_agent_id, agent_path, model, reasoning_effort, history,
    ):
        from .runtime import AgentRuntime

        agent = self.build_agent(
            parent_agent_id, agent_path, model, reasoning_effort, history,
        )
        if agent.agent_id in self._runtimes:
            raise ValueError("agent is already registered: " + agent.agent_id)
        runtime = AgentRuntime(agent, self)
        self._runtimes[agent.agent_id] = runtime
        agent.context_manager.agent_path = agent.agent_path
        return runtime

    def _notify_parents(self, agent_id):
        parent_agent_id = self.get_parent_agent_id(agent_id)
        while parent_agent_id is not None:
            parent = self._runtimes[parent_agent_id]
            parent.publish_update("agents")
            parent_agent_id = self.get_parent_agent_id(parent_agent_id)

    def _resolve_agent_path(self, caller_agent_id, target):
        caller = self._runtimes[caller_agent_id]
        agent_path = (
            target if target.startswith("/") else caller.agent.agent_path + "/" + target
        )
        if (
            not re.fullmatch(r"/root(?:/[a-z0-9_]+)*", agent_path)
            or "root" in agent_path.split("/")[2:]
        ):
            raise ValueError("invalid task path: " + target)
        return agent_path

    def _resolve_agent_id(self, caller_agent_id, target):
        self._runtimes[caller_agent_id]
        if target in self._runtimes:
            return target
        agent_path = self._resolve_agent_path(caller_agent_id, target)
        for agent_id, runtime in self._runtimes.items():
            if runtime.agent.agent_path == agent_path:
                return agent_id
        raise ValueError("unknown agent: " + target)

    def list_agents(self, caller_agent_id, path_prefix=None):
        """Return agent identities and statuses from this tree."""
        self._runtimes[caller_agent_id]
        prefix = (
            self._resolve_agent_path(caller_agent_id, path_prefix)
            if path_prefix is not None else None
        )
        agents = []
        for agent_id, runtime in sorted(
            self._runtimes.items(), key=lambda item: item[1].agent.agent_path,
        ):
            agent_path = runtime.agent.agent_path
            if (
                prefix is not None
                and agent_path != prefix
                and not agent_path.startswith(prefix + "/")
            ):
                continue
            agents.append({
                "agent_id": agent_id,
                "parent_agent_id": self.get_parent_agent_id(agent_id),
                "agent_path": agent_path,
                "status": runtime.agent_status,
            })
        return agents

    def get_runtime(self, agent_id):
        return self._runtimes[agent_id]

    def get_parent_agent_id(self, agent_id):
        agent_path = self._runtimes[agent_id].agent.agent_path
        if agent_path == "/root":
            return None
        parent_path = agent_path.rsplit("/", 1)[0]
        return self._resolve_agent_id(agent_id, parent_path)

    def unregister_runtime(self, agent_id):
        parent_agent_id = self.get_parent_agent_id(agent_id)
        if parent_agent_id is None:
            return
        runtime = self._runtimes[agent_id]
        runtime.agent.tool_registry.runtime_environment.set_state_listener(None)
        del self._runtimes[agent_id]
        self._runtimes[parent_agent_id].publish_update("agents")
        self._notify_parents(parent_agent_id)

    async def _make_spawn_room(self, caller_agent_id):
        caller = self._runtimes[caller_agent_id]
        capacity = self._context_config.multi_agent_max_concurrency - 1
        while True:
            if not caller.accepts_input:
                raise RuntimeError("agent is shutting down")
            children = [
                runtime for runtime in self._runtimes.values()
                if runtime.agent.agent_path != "/root"
            ]
            if len(children) < capacity:
                return
            for child in children:
                if (
                    child.agent.agent_id == caller_agent_id
                    or not child.can_close_for_capacity()
                    or any(
                        other.agent.agent_path.startswith(child.agent.agent_path + "/")
                        for other in children
                    )
                ):
                    continue
                await child.close()
                break
            else:
                raise RuntimeError(
                    "agent capacity reached ({0} including root); "
                    "no idle child agent can be closed".format(capacity + 1)
                )

    def _message(self, sender_agent_id, recipient_agent_id, text, kind, encrypted=False):
        author = self._runtimes[sender_agent_id].agent.agent_path
        recipient = self._runtimes[recipient_agent_id].agent.agent_path
        header = "Message Type: {0}\nTask name: {1}\nSender: {2}\nPayload:\n".format(
            kind, recipient, author,
        )
        content = (
            (
                {"type": "input_text", "text": header},
                {"type": "encrypted_content", "encrypted_content": text},
            )
            if encrypted else ({"type": "input_text", "text": header + text},)
        )
        return AgentMessage(
            author, recipient, content, id="amsg_" + uuid7_string(),
        )

    async def spawn_agent(
        self,
        caller_agent_id: "str",
        task_name: "str",
        message: "str",
        fork_turns: "str" = "all",
        model: "typing.Union[str, None]" = None,
        reasoning_effort: "typing.Union[str, None]" = None,
        history: "typing.Tuple[ConversationItem, ...]" = (),
        encrypted: "bool" = False,
    ) -> "str":
        if not re.fullmatch(r"[a-z0-9_]+", task_name):
            raise ValueError("task_name must contain lowercase letters, digits, or underscores")
        if task_name == "root":
            raise ValueError("task_name `root` is reserved")
        if not message.strip():
            raise ValueError("Empty message can't be sent to an agent")
        if fork_turns not in ("all", "none"):
            raise ValueError("fork_turns must be `none` or `all`")
        if self._spawn_lock is None:
            self._spawn_lock = asyncio.Lock()
        async with self._spawn_lock:
            parent = self._runtimes[caller_agent_id]
            agent_path = parent.agent.agent_path + "/" + task_name
            if any(
                runtime.agent.agent_path == agent_path
                for runtime in self._runtimes.values()
            ):
                raise ValueError("task name already exists: " + agent_path)
            await self._make_spawn_room(caller_agent_id)

            fork_context = None
            if fork_turns == "all":
                fork_context = parent.agent.context_manager.snapshot_for_fork(
                    history,
                    parent.agent.tool_registry.model_visible_specs(
                        parent.agent.context_manager.resolve_code_mode_only(),
                    ),
                    parent.agent.model_client.reasoning_effort,
                )
            runtime = self._create_runtime(
                caller_agent_id, agent_path, model, reasoning_effort,
                fork_context.history if fork_context is not None else (),
            )
            if fork_context is not None:
                runtime.agent.context_manager.inherit_fork_context(
                    fork_context,
                    runtime.agent.tool_registry.model_visible_specs(
                        runtime.agent.context_manager.resolve_code_mode_only(),
                    ),
                    runtime.agent.model_client.reasoning_effort,
                )
            self._notify_parents(runtime.agent.agent_id)
            await runtime.start()
            await self.send_message(
                caller_agent_id, runtime.agent.agent_id, message, trigger_turn=True,
                encrypted=encrypted,
            )
            return agent_path

    async def send_message(
        self, caller_agent_id, target, message, trigger_turn=False, encrypted=False,
    ):
        if not self._runtimes[caller_agent_id].accepts_input:
            raise RuntimeError("agent is shutting down")
        if not message.strip():
            raise ValueError("Empty message can't be sent to an agent")
        recipient_agent_id = self._resolve_agent_id(caller_agent_id, target)
        recipient = self._runtimes[recipient_agent_id]
        if not recipient.accepts_input:
            raise RuntimeError("agent is shutting down")
        if trigger_turn and self.get_parent_agent_id(recipient_agent_id) is None:
            raise ValueError("followup_task cannot target the root agent")
        item = self._message(
            caller_agent_id, recipient_agent_id, message,
            "NEW_TASK" if trigger_turn else "MESSAGE", encrypted,
        )
        recipient.message(item)
        if trigger_turn:
            recipient.trigger()
        self._runtimes.move_to_end(recipient_agent_id)
        return ""

    async def interrupt_agent(self, caller_agent_id, target):
        recipient_agent_id = self._resolve_agent_id(caller_agent_id, target)
        recipient = self._runtimes[recipient_agent_id]
        if recipient.agent.agent_path == "/root":
            raise ValueError("root is not a spawned agent")
        if recipient_agent_id == caller_agent_id:
            raise ValueError(
                "an agent cannot interrupt itself; return your result and let "
                "the parent interrupt you if needed"
            )
        previous = recipient.agent_status
        if not recipient.agent.is_shutdown:
            recipient.interrupt()
        return {"previous_status": previous}

    async def wait_for_activity(self, agent_id, timeout_ms):
        return await self._runtimes[agent_id].wait_for_activity(timeout_ms)

    def request_close(self, agent_id) -> "None":
        """Signal a whole subtree before waiting for any of its operations."""
        path = self._runtimes[agent_id].agent.agent_path
        runtimes = [
            runtime for runtime in self._runtimes.values()
            if runtime.agent.agent_path == path
            or runtime.agent.agent_path.startswith(path + "/")
        ]
        for runtime in runtimes:
            if runtime.accepts_input:
                self._closing_children[runtime.agent.agent_id] = tuple(
                    child for child in runtimes
                    if child.agent.agent_path.rsplit("/", 1)[0] == runtime.agent.agent_path
                )
                runtime.request_close()

    async def shutdown_children(self, agent_id) -> "None":
        # Children may finish and unregister while their parent's turn stops.
        self._runtimes[agent_id]
        children = self._closing_children.get(agent_id, ())
        outcomes = await asyncio.gather(
            *(child.close() for child in children), return_exceptions=True,
        )
        self._closing_children.pop(agent_id, None)
        errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if errors:
            for error in errors[1:]:
                asyncio.get_running_loop().call_exception_handler(
                    {
                        "message": "Agent close failed",
                        "exception": error,
                    }
                )
            raise errors[0]

    def handle_event(self, agent_id, event: "Event") -> "None":
        self._runtimes[agent_id]
        if isinstance(event, (TurnStartedEvent, CompactStartedEvent, CommandStartedEvent)):
            self._runtimes.move_to_end(agent_id)
        text = None
        if isinstance(event, TurnCompletedEvent):
            text = event.output_text or ""
        elif isinstance(event, TurnFailedEvent):
            text = "Agent errored: {0}: {1}".format(event.error_type, event.error)
        elif isinstance(event, RuntimeUpdatedEvent):
            if event.reason == "agents":
                return
        elif not isinstance(
            event,
            (
                TurnStartedEvent,
                TurnInterruptedEvent,
                CompactStartedEvent,
                SessionClosedEvent,
            ),
        ):
            return
        parent_agent_id = self.get_parent_agent_id(agent_id)
        if parent_agent_id is not None and text is not None:
            parent = self._runtimes[parent_agent_id]
            if parent.accepts_input:
                parent.message(
                    self._message(agent_id, parent_agent_id, text, "FINAL_ANSWER"),
                )
        self._notify_parents(agent_id)
