
from datetime import datetime

import pytest

from pycodex import (
    AgentLoop,
    AssistantMessage,
    ContextConfig,
    ContextManager,
    ContextMessage,
    ModelResponse,
    ToolRegistry,
    UserMessage,
)
from tests.fakes import ScriptedModelClient


def test_context_manager_resolves_base_instructions_precedence(tmp_path) -> 'None':
    instructions_file = tmp_path / "instructions.md"
    instructions_file.write_text("from model file\n")

    manager = ContextManager(
        config=ContextConfig(
            base_instructions="from config",
            model_instructions_file=instructions_file,
        ),
        base_instructions_override="from override",
    )
    assert manager.resolve_base_instructions() == "from override"

    manager = ContextManager(
        config=ContextConfig(
            base_instructions="from config",
            model_instructions_file=instructions_file,
        )
    )
    assert manager.resolve_base_instructions() == "from config"

    manager = ContextManager(config=ContextConfig(model_instructions_file=instructions_file))
    assert manager.resolve_base_instructions() == "from model file"


def test_context_manager_resolves_model_instructions_from_models_json() -> 'None':
    manager = ContextManager(
        config=ContextConfig(model="gpt-5.4", personality="pragmatic")
    )

    instructions = manager.resolve_base_instructions()

    assert instructions.startswith(
        "You are Codex, a coding agent based on GPT-5."
    )
    assert "You are a deeply pragmatic, effective software engineer." in instructions
    assert "Always use apply_patch for manual code edits." in instructions


@pytest.mark.parametrize("model", ["step-3.5-flash", "step-3.5-flash-2603"])
def test_context_manager_resolves_model_instructions_from_step_models_json_entry(model) -> 'None':
    manager = ContextManager(
        config=ContextConfig(model=model, personality="pragmatic")
    )

    instructions = manager.resolve_base_instructions()

    assert instructions.startswith(
        "You are Codex, a coding agent based on Step-3.5 Flash."
    )
    assert "GPT-5" not in instructions
    assert "You are a deeply pragmatic, effective software engineer." in instructions
    assert "Always use apply_patch for manual code edits." in instructions


def test_context_manager_builds_official_style_context_messages(
    tmp_path,
    monkeypatch,
) -> 'None':
    codex_home = tmp_path / "codex-home"
    skills_root = codex_home / "skills"
    skills_root.mkdir(parents=True)
    (codex_home / "AGENTS.md").write_text("global rules")
    (skills_root / "alpha").mkdir()
    (skills_root / "alpha" / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                'name: "alpha"',
                'description: "Alpha skill description."',
                "---",
                "",
                "# Alpha",
            ]
        )
    )
    (skills_root / ".system" / "omega").mkdir(parents=True)
    (skills_root / ".system" / "omega" / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                'name: "omega"',
                'description: "Omega skill description."',
                "---",
                "",
                "# Omega",
            ]
        )
    )
    config_path = codex_home / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'user_instructions = "config rules"',
                'developer_instructions = "developer rules"',
                'base_instructions = "base rules"',
                'project_doc_max_bytes = 1024',
                'approval_policy = "never"',
                'sandbox_mode = "danger-full-access"',
            ]
        )
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    (repo_root / "AGENTS.md").write_text("root project rules")
    nested = repo_root / "pkg"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("nested default rules")
    (nested / "AGENTS.override.md").write_text("nested override rules")
    monkeypatch.chdir(nested)
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setenv("TZ", "Asia/Hong_Kong")

    manager = ContextManager.from_codex_config(
        config_path,
        base_instructions_override="override rules",
        include_collaboration_instructions=True,
    )

    prompt = manager.build_prompt([UserMessage(text="hello")], [], True)

    assert prompt.base_instructions == "override rules"
    assert [type(item).__name__ for item in prompt.input] == [
        "ContextMessage",
        "ContextMessage",
        "UserMessage",
    ]

    developer_message = prompt.input[0]
    assert isinstance(developer_message, ContextMessage)
    assert developer_message.role == "developer"
    assert developer_message.content_items is not None
    developer_texts = [item["text"] for item in developer_message.content_items]
    assert developer_texts[0].startswith("<permissions instructions>")
    assert "Approval policy is currently never." in developer_texts[0]
    assert developer_texts[1] == "developer rules"
    assert developer_texts[2].startswith("<collaboration_mode># Collaboration Mode: Default")
    assert "Known mode names are Default and Plan." in developer_texts[2]
    assert "The `request_user_input` tool is unavailable in Default mode." in developer_texts[2]
    assert developer_texts[2].endswith(
        "Never write a multiple choice question as a textual assistant message.\n"
        "</collaboration_mode>"
    )
    assert developer_texts[3].startswith("<skills_instructions>")
    assert "- alpha: Alpha skill description." in developer_texts[3]
    assert "- omega: Omega skill description." in developer_texts[3]

    contextual_user_message = prompt.input[1]
    assert isinstance(contextual_user_message, ContextMessage)
    assert contextual_user_message.role == "user"
    assert contextual_user_message.content_items is not None
    user_texts = [item["text"] for item in contextual_user_message.content_items]
    assert len(user_texts) == 2
    assert f"# AGENTS.md instructions for {nested.resolve()}" in user_texts[0]
    assert "config rules" in user_texts[0]
    assert "global rules" in user_texts[0]
    assert "--- project-doc ---" in user_texts[0]
    assert "root project rules" in user_texts[0]
    assert "nested override rules" in user_texts[0]
    assert "nested default rules" not in user_texts[0]
    assert "<environment_context>" in user_texts[1]
    assert "<shell>zsh</shell>" in user_texts[1]
    assert (
        f"<current_date>{datetime.now().date().isoformat()}</current_date>"
        in user_texts[1]
    )
    assert "<timezone>Asia/Hong_Kong</timezone>" in user_texts[1]


def test_context_manager_builds_plan_mode_collaboration_message(tmp_path, monkeypatch) -> 'None':
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.setenv("TZ", "Asia/Hong_Kong")

    manager = ContextManager(
        base_instructions_override="base rules",
        config=ContextConfig(
            approval_policy="never",
            sandbox_mode="danger-full-access",
        ),
        collaboration_mode="plan",
        include_collaboration_instructions=True,
        include_skills_instructions=False,
    )

    prompt = manager.build_prompt([UserMessage(text="plan this")], [], True)

    developer_message = prompt.input[0]
    assert isinstance(developer_message, ContextMessage)
    developer_texts = [item["text"] for item in developer_message.content_items or ()]
    assert developer_texts[1].startswith(
        "<collaboration_mode># Plan Mode (Conversational)"
    )
    assert "Strongly prefer using the `request_user_input` tool" in developer_texts[1]
    assert developer_texts[1].endswith("</collaboration_mode>")


@pytest.mark.asyncio
async def test_agent_loop_injects_context_without_polluting_history(
    tmp_path,
    monkeypatch,
) -> 'None':
    model = ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.setenv("TZ", "Asia/Hong_Kong")
    manager = ContextManager(
        base_instructions_override="base rules",
        config=ContextConfig(
            developer_instructions="developer rules",
            user_instructions="user rules",
            approval_policy="never",
            sandbox_mode="danger-full-access",
        ),
        include_skills_instructions=False,
    )

    agent = AgentLoop(model, ToolRegistry(), manager)
    result = await agent.run_turn(["hello"])

    prompt = model.prompts[0]
    assert prompt.base_instructions == "base rules"
    assert prompt.turn_id == result.turn_id
    assert prompt.turn_metadata == {
        "turn_id": result.turn_id,
        "sandbox": "none",
    }
    assert isinstance(prompt.input[0], ContextMessage)
    assert isinstance(prompt.input[1], ContextMessage)
    assert prompt.input[2] == UserMessage(text="hello")

    developer_message = prompt.input[0]
    assert developer_message.content_items is not None
    assert len(developer_message.content_items) == 2
    assert developer_message.content_items[0]["text"].startswith(
        "<permissions instructions>"
    )
    assert developer_message.content_items[1]["text"] == "developer rules"

    contextual_user_message = prompt.input[1]
    assert contextual_user_message.content_items is not None
    assert len(contextual_user_message.content_items) == 2
    assert contextual_user_message.content_items[0]["text"].startswith(
        f"# AGENTS.md instructions for {tmp_path.resolve()}"
    )
    assert contextual_user_message.content_items[1]["text"].startswith(
        "<environment_context>"
    )

    assert result.history == (
        UserMessage(text="hello"),
        AssistantMessage(text="done"),
    )


def test_context_manager_keeps_workspaces_within_same_turn(monkeypatch, tmp_path) -> 'None':
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "pycodex.context.get_workspace_turn_metadata",
        lambda _cwd: {
            "workspaces": {
                str(tmp_path.resolve()): {
                    "latest_git_commit_hash": "abc123",
                    "associated_remote_urls": {"origin": "git@example.com/repo.git"},
                    "has_changes": True,
                }
            }
        },
    )
    manager = ContextManager(
        config=ContextConfig(
            approval_policy="never",
            sandbox_mode="danger-full-access",
        ),
        include_skills_instructions=False,
    )

    first = manager.get_turn_metadata("turn_a")
    second_same_turn = manager.get_turn_metadata("turn_a")
    third_new_turn = manager.get_turn_metadata("turn_b")

    assert list(first) == ["turn_id", "workspaces", "sandbox"]
    assert list(second_same_turn) == ["turn_id", "workspaces", "sandbox"]
    assert second_same_turn["workspaces"] == first["workspaces"]
    assert list(third_new_turn) == ["turn_id", "sandbox"]
