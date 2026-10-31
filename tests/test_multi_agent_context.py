from copy import deepcopy
from dataclasses import replace
from uuid import UUID, uuid5

import pytest

from pycodex import (
    AgentMessage, AgentsManager, AssistantMessage, ContextConfig, ContextManager,
    ModelResponse, Prompt, ResponsesModelClient, ResponsesProviderConfig,
    ToolCall, ToolResult, UserMessage,
)
from pycodex.utils import uuid7_string
from pycodex.utils.session_persist import SessionRolloutRecorder, load_resumed_session_path
from tests.fakes import ScriptedModelClient, wait_for_runtime_idle


@pytest.mark.parametrize("lite", [False, True])
@pytest.mark.parametrize("marker", [None, [], ["message"]])
def test_native_message_and_function_metadata_survive_wire_resume_and_checkpoint(tmp_path, lite, marker):
    client = ResponsesModelClient(ResponsesProviderConfig(
        "fixture", "fixture", "https://example.invalid/v1", None,
        responses_lite_override=lite,
    ))
    raw = {
        "type": "agent_message", "id": "amsg_fixture",
        "author": "/root/worker", "recipient": "/root",
        "content": [
            {"type": "input_text", "text": "header"},
            {"type": "encrypted_content", "encrypted_content": "opaque-fixture"},
        ],
        "internal_chat_message_metadata_passthrough": {"fixture": {"value": 1}},
    }
    expected_message = deepcopy(raw)
    message = AgentMessage.from_response_item(raw)
    raw["content"][0]["text"] = "mutated source"
    raw["internal_chat_message_metadata_passthrough"]["fixture"]["value"] = 2
    serialized = message.serialize()
    serialized["content"][0]["text"] = "mutated serialization"
    assert message.serialize() == expected_message
    assert message.text == "header"

    raw_call = {
        "type": "function_call", "id": "fc_fixture", "call_id": "fixture",
        "name": "send_message", "namespace": "collaboration",
        "arguments": '{"target":"/root","message":"payload"}',
    }
    if marker is not None:
        raw_call["encrypted_function_args"] = list(marker)
    expected_call = deepcopy(raw_call)
    call = client._parse_output_item(raw_call)
    if marker is not None:
        raw_call["encrypted_function_args"].append("source mutation")
        call.serialize()["encrypted_function_args"].append("serialization mutation")
    assert call.serialize() == expected_call
    history = (UserMessage("start"), message, call, ToolResult("fixture", "collaboration.send_message", ""))
    payload = client._build_payload(Prompt(history, [], base_instructions="Base."))
    assert next(item for item in payload["input"] if item["type"] == "agent_message") == expected_message
    assert next(item for item in payload["input"] if item["type"] == "function_call") == expected_call

    recorder = SessionRolloutRecorder.create(
        tmp_path, uuid7_string(), tmp_path, "fixture", "fixture", "fixture",
    )
    recorder.append_history_items(history)
    assert load_resumed_session_path(recorder.rollout_path)["history"] == history
    recorder.append_compacted_history(history)
    assert load_resumed_session_path(recorder.rollout_path)["history"] == history


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", [None, [], ["message"]])
async def test_only_explicit_empty_encrypted_args_deliver_plaintext(tmp_path, monkeypatch, marker):
    def respond(prompt, count):
        if count == 1:
            return ModelResponse([ToolCall(
                "spawn", "collaboration.spawn_agent",
                {"task_name": "worker", "message": "payload", "fork_turns": "none"},
                encrypted_function_args=marker,
            )])
        return ModelResponse([AssistantMessage("parent done")])

    client = ScriptedModelClient(response_factory=respond)
    child_client = ScriptedModelClient([ModelResponse([AssistantMessage("child done")])])
    monkeypatch.setattr(client, "with_overrides", lambda *args, **kwargs: child_client)
    manager = AgentsManager(
        client, ContextConfig(cwd=tmp_path, codex_home=tmp_path), toolset=("spawn_agent",),
    )
    root = manager.create_root_runtime()
    await root.start()
    try:
        await root.submit_user_turn("start")
        child = manager.get_runtime(manager.list_agents(root.agent.agent_id)[1]["agent_id"])
        await wait_for_runtime_idle(child)
        message = next(item for item in child.agent.history if isinstance(item, AgentMessage))
        assert message.author == "/root" and message.recipient == "/root/worker"
        if marker == []:
            assert message.content_items == ({
                "type": "input_text",
                "text": "Message Type: NEW_TASK\nTask name: /root/worker\nSender: /root\nPayload:\npayload",
            },)
        else:
            assert message.content_items[-1] == {
                "type": "encrypted_content", "encrypted_content": "payload",
            }
            assert message.text.endswith("Payload:\n")
        assert message in child_client.prompts[0].input
    finally:
        await root.close()


@pytest.mark.parametrize("effort, mode", [(None, "Explicit"), ("high", "Explicit"), ("ultra", "Proactive")])
def test_role_default_and_mode_blocks_remain_separate(tmp_path, monkeypatch, effort, mode):
    metadata = {"model_messages": {
        "collaboration_modes": {"default": "Default instructions"},
        "multi_agent": {
            "role": {"root": "Root instructions", "subagent": "Child instructions"},
            "mode": {"explicit": "Explicit", "proactive": "Proactive"},
        },
    }}
    monkeypatch.setattr("pycodex.context.model_metadata", lambda model: metadata)
    config = ContextConfig(
        cwd=tmp_path, base_instructions="Base", developer_instructions="Custom",
        include_permissions_instructions=False, include_skills_instructions=False,
    )
    context = ContextManager(config)
    context.agent_path = "/root"
    history = (UserMessage("real input"),)
    prompt = context.build_prompt(history, [], False, reasoning_effort=effort)
    developer = [item.serialize() for item in prompt.input if getattr(item, "role", None) == "developer"]
    assert developer[0]["content"] == [
        {"type": "input_text", "text": "Custom"},
        {"type": "input_text", "text": "<collaboration_mode>Default instructions</collaboration_mode>"},
    ]
    assert developer[1]["content"][0]["text"].startswith("<multi_agent_role>Root instructions")
    assert "16 available concurrency slots" in developer[1]["content"][0]["text"]
    assert developer[2]["content"] == [
        {"type": "input_text", "text": "<multi_agent_mode>" + mode + "</multi_agent_mode>"}
    ]
    assert prompt.input[-1] is history[0]

    override = ContextManager(replace(
        config, multi_agent_root_instructions="Configured role", multi_agent_hint="",
    ))
    override.agent_path = "/root"
    blocks = override.build_prompt(history, [], False, reasoning_effort=effort).input
    assert blocks[1].text == "Configured role"
    assert not any("<multi_agent_mode>" in (getattr(item, "text", None) or "") for item in blocks)


@pytest.mark.parametrize("code_mode_only", [False, True])
@pytest.mark.parametrize("agent_path", ["/root", "/root/review"])
@pytest.mark.parametrize("toolset", [None, ("exec_command",)])
def test_multi_agent_tool_guidance_matches_chat_catalog(
    tmp_path, code_mode_only, agent_path, toolset,
):
    from pycodex.tools import get_tools
    from responses_server.payload_processors import post_process_outcomming_request
    from responses_server.request_translation import build_outcomming_request

    config = ContextConfig(
        model="step-5-preview", cwd=tmp_path, codex_home=tmp_path,
        code_mode_only=code_mode_only,
        include_permissions_instructions=False, include_skills_instructions=False,
    )
    client = ResponsesModelClient(ResponsesProviderConfig(
        "step-5-preview", "stepfun", "https://example.invalid/v1", None,
        responses_lite_override=False,
    ))
    context = ContextManager(config)
    context.agent_path = agent_path
    registry = get_tools(AgentsManager(client, config), toolset=toolset)
    specs = registry.model_visible_specs(context.resolve_code_mode_only())
    prompt = context.build_prompt([UserMessage("Review the task.")], specs, True)
    chat = post_process_outcomming_request(
        build_outcomming_request(client._build_payload(prompt)), "stepfun",
    )
    names = {tool["function"]["name"] for tool in chat["tools"]}
    role = context._build_multi_agent_context(specs, None)[0].text

    assert ("exec" in names) is code_mode_only
    assert ("collaboration__spawn_agent" in names) is (toolset is None)
    assert "current tool declarations" in role
    assert "functions.collaboration" not in role
    assert "functions.exec" not in role
    assert ("await collaboration." in role) is (code_mode_only and toolset is None)
    assert role in [message["content"] for message in chat["messages"]]
    assert "multi_tool_use.parallel" not in prompt.base_instructions


def test_multi_agent_config_prefers_feature_settings_and_validates_values(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("", encoding="utf-8")
    assert ContextConfig().multi_agent_max_concurrency == 16
    assert ContextConfig.from_codex_config(path).multi_agent_max_concurrency == 16
    path.write_text(
        "[agents]\nmax_concurrent_threads_per_session = 2\n"
        "[features.multi_agent_v2]\n"
        "max_concurrent_threads_per_session = 7\n"
        'root_agent_usage_hint_text = "Root"\n'
        'subagent_usage_hint_text = "Child"\n'
        'multi_agent_mode_hint_text = "Mode"\n',
        encoding="utf-8",
    )
    config = ContextConfig.from_codex_config(path)
    assert config.multi_agent_max_concurrency == 7
    assert config.multi_agent_root_instructions == "Root"
    assert config.multi_agent_subagent_instructions == "Child"
    assert config.multi_agent_hint == "Mode"
    path.write_text("[agents]\nmax_concurrent_threads_per_session = 2\n", encoding="utf-8")
    assert ContextConfig.from_codex_config(path).multi_agent_max_concurrency == 3
    for value in (0, -1, True, "4"):
        with pytest.raises(ValueError, match="positive integer"):
            ContextConfig(multi_agent_max_concurrency=value)


@pytest.mark.asyncio
async def test_full_and_nested_forks_preserve_prefix_then_append_context_updates(tmp_path):
    client = ScriptedModelClient(response_factory=lambda prompt, count: ModelResponse([
        AssistantMessage("done"),
    ]))
    manager = AgentsManager(
        client, ContextConfig(cwd=tmp_path, codex_home=tmp_path, base_instructions="Base"),
        toolset=(),
    )
    root = manager.create_root_runtime()
    root.agent._append_history([UserMessage("parent input"), AssistantMessage("parent answer")])
    parent_prompt = root.agent.context_manager.build_prompt(root.agent.history, [], False)
    try:
        path = await manager.spawn_agent(root.agent.agent_id, "worker", "task", history=root.agent.history)
        child = manager.get_runtime(manager._resolve_agent_id(root.agent.agent_id, path))
        await wait_for_runtime_idle(child)
        first = child.agent.model_client.prompts[0]
        assert list(first.input[:len(parent_prompt.input)]) == list(parent_prompt.input)
        assert first.base_instructions == parent_prompt.base_instructions
        child_dir = tmp_path / "child"
        child_dir.mkdir()
        child.agent.set_cwd(child_dir)
        child.agent.model_client.reasoning_effort = "ultra"
        updated = child.agent.context_manager.build_prompt(
            child.agent.history, [], False, reasoning_effort="ultra",
        )
        assert list(updated.input[:len(first.input)]) == list(first.input)
        assert str(child_dir) in str([item.serialize() for item in updated.input[len(first.input):]])
        path = await manager.spawn_agent(
            child.agent.agent_id, "nested", "nested task", history=child.agent.history,
        )
        nested = manager.get_runtime(manager._resolve_agent_id(root.agent.agent_id, path))
        await wait_for_runtime_idle(nested)
        nested_prompt = nested.agent.model_client.prompts[0]
        assert list(nested_prompt.input[:len(updated.input)]) == list(updated.input)
        await child.agent.compact()
        assert child.agent.context_manager._fork_context is None
        assert root.agent.history == (UserMessage("parent input"), AssistantMessage("parent answer"))
    finally:
        await root.close()


@pytest.mark.parametrize("tool_type, prefix", [("function", "fco"), ("custom", "ctco")])
def test_fork_aborted_results_are_stable_and_pruned_items_do_not_reappear(tool_type, prefix):
    parent = ContextManager(ContextConfig(base_instructions="Base"))
    parent.agent_path = "/root"
    call = ToolCall(
        "pending", "fixture", {} if tool_type == "function" else "patch",
        tool_type=tool_type, id="provider-call",
    )
    history = (UserMessage("before"), call, ToolResult("orphan", "fixture", "unused"))
    snapshot = parent.snapshot_for_fork(history, [], None)
    child = ContextManager(ContextConfig(base_instructions="Base"))
    child.agent_path = "/root/worker"
    child.inherit_fork_context(snapshot, [], None)
    projected = child.build_prompt(snapshot.history, [], False)
    results = [item for item in projected.input if isinstance(item, ToolResult)]
    assert len(results) == 1
    assert results[0].output == "aborted"
    assert results[0].id == prefix + "_" + str(uuid5(
        UUID("90d38d3e-6a5b-4d52-bfe2-2f1e634bfac4"), prefix + ":provider-call",
    ))
    assert child.build_prompt(snapshot.history, [], False).input == projected.input
    pruned = child.build_prompt(snapshot.history[:1], [], False, capture_context=False)
    assert not any(isinstance(item, (ToolCall, ToolResult)) for item in pruned.input)
