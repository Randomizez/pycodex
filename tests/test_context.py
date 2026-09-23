from dataclasses import replace
from datetime import datetime

import pytest

from pycodex import (
    Agent,
    AssistantMessage,
    ContextConfig,
    ContextManager,
    ContextMessage,
    ModelResponse,
    ToolRegistry,
    UserMessage,
)
from pycodex.model_metadata import model_metadata
from tests.fakes import ScriptedModelClient


def test_context_manager_resolves_base_instructions_precedence(tmp_path) -> "None":
    instructions_file = tmp_path / "instructions.md"
    instructions_file.write_text("from model file\n")

    manager = ContextManager(
        config=ContextConfig(
            base_instructions="from config",
            model_instructions_file=instructions_file,
            base_instructions_override="from override",
        ),
    )
    assert manager.resolve_base_instructions() == "from override"

    manager = ContextManager(
        config=ContextConfig(
            base_instructions="from config",
            model_instructions_file=instructions_file,
        )
    )
    assert manager.resolve_base_instructions() == "from config"

    manager = ContextManager(
        config=ContextConfig(model_instructions_file=instructions_file)
    )
    assert manager.resolve_base_instructions() == "from model file"


def test_context_manager_resolves_model_instructions_from_models_json() -> "None":
    manager = ContextManager(
        config=ContextConfig(model="gpt-5.4", personality="pragmatic")
    )

    instructions = manager.resolve_base_instructions()

    assert instructions.startswith("You are Codex, a coding agent based on GPT-5.")
    assert "You are a deeply pragmatic, effective software engineer." in instructions
    assert "Always use apply_patch for manual code edits." in instructions


def test_context_manager_resolves_gpt56_model_metadata() -> "None":
    manager = ContextManager(config=ContextConfig(model="gpt-5.6-sol"))

    instructions = manager.resolve_base_instructions()

    assert instructions.startswith("You are Codex, an agent based on GPT-5.")
    assert "curious, rich personality" in instructions
    assert manager.resolve_model_max_context_window() == 372000
    assert manager.resolve_model_context_window() == 353400


@pytest.mark.parametrize(
    "slug, expected",
    [
        ("gpt-6-astra:caishi-azure", "gpt-6-astra"),
        ("custom/gpt-6-astra:azure", "gpt-6-astra"),
        ("step-3.5-flash-2603:local", "step-3.5-flash-2603"),
        ("invalid.namespace/gpt-5.4", None),
        ("provider/nested/gpt-5.4", None),
        ("/gpt-5.4", None),
        ("unknown-model", None),
        (None, None),
    ],
)
def test_metadata_uses_upstream_longest_prefix(slug, expected):
    metadata = model_metadata(slug)
    if expected is None:
        assert metadata is None
        return
    assert metadata is model_metadata(expected)
    manager = ContextManager(ContextConfig(model=slug))
    reference = ContextManager(ContextConfig(model=expected))
    assert manager.resolve_base_instructions() == reference.resolve_base_instructions()
    assert (
        manager.resolve_model_max_context_window()
        == reference.resolve_model_max_context_window()
    )
    assert (
        manager.resolve_model_context_window()
        == reference.resolve_model_context_window()
    )
    assert (
        manager.resolve_auto_compact_token_limit()
        == reference.resolve_auto_compact_token_limit()
    )


@pytest.mark.parametrize(
    "model, guidance", [("gpt-5.4", True), ("gpt-6-astra:azure", False)]
)
def test_skills_guidance_follows_model_metadata(tmp_path, model, guidance):
    skill = tmp_path / "skills" / ".system" / "fixture" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: fixture\ndescription: Fixture skill.\n---\n")
    manager = ContextManager(ContextConfig(model=model, codex_home=tmp_path))
    text = manager._build_skills_instructions()
    assert "(file: r0/fixture/SKILL.md)" in text
    assert ("### How to use skills" in text) is guidance
    if guidance:
        assert "read its `SKILL.md` completely" in text
        assert "Do not delegate reading" in text


@pytest.mark.parametrize("implicit", ["true", "false"])
def test_skill_catalog_respects_yaml_invocation_policy(tmp_path, implicit):
    skill = tmp_path / "skills" / "fixture" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: 'fixture'\ndescription: >-\n  First line\n  second line.\n---\n"
    )
    metadata = skill.parent / "agents" / "openai.yaml"
    metadata.parent.mkdir()
    metadata.write_text("policy: {allow_implicit_invocation: " + implicit + "}\n")
    text = ContextManager(
        ContextConfig(codex_home=tmp_path)
    )._build_skills_instructions()
    if implicit == "false":
        assert text is None
    else:
        assert "fixture: First line second line." in text


@pytest.mark.parametrize(
    "sandbox", ["danger-full-access", "read-only", "workspace-write"]
)
def test_environment_filesystem_matches_default_sandbox(tmp_path, sandbox):
    from xml.etree import ElementTree

    workspace = tmp_path / """a&b<"'>"""
    workspace.mkdir()
    manager = ContextManager(ContextConfig(cwd=workspace, sandbox_mode=sandbox))
    text = manager._serialize_environment_context()
    document = ElementTree.fromstring(text)
    assert document.findtext("cwd") == str(workspace)
    assert document.findtext("filesystem/workspace_roots/root") == str(workspace)
    assert "&amp;" in text and "&apos;" in text and "&quot;" in text
    profile = document.find("filesystem/permission_profile")
    if sandbox == "danger-full-access":
        assert profile.attrib == {"type": "disabled"}
        assert profile.find("file_system").attrib == {"type": "unrestricted"}
    else:
        assert profile.attrib == {"type": "managed"}
        entries = profile.findall("file_system/entry")
        assert entries[0].attrib == {"access": "read"}
        assert entries[0].findtext("special") == ":root"
        assert len(entries) == (1 if sandbox == "read-only" else 7)
        if sandbox == "workspace-write":
            assert entries[1].findtext("path") == str(workspace)
            assert [entry.findtext("path") for entry in entries[-3:]] == [
                str(workspace / name) for name in (".git", ".agents", ".codex")
            ]


def test_context_manager_resolves_auto_compact_limit_from_config() -> "None":
    manager = ContextManager(config=ContextConfig(model_auto_compact_token_limit=12345))

    assert manager.resolve_auto_compact_token_limit() == 12345


def test_context_manager_resolves_max_length_override():
    manager = ContextManager(
        ContextConfig(model="gpt-5.6-sol", model_context_window=100000)
    )
    assert manager.resolve_model_max_context_window() == 100000
    assert manager.resolve_model_context_window() == 95000


def test_context_manager_reads_auto_compact_limit_from_codex_config(tmp_path) -> "None":
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "\n".join(
            [
                'model = "demo-model"',
                'model_provider = "demo"',
                "model_auto_compact_token_limit = 12345",
                "[model_providers.demo]",
                'base_url = "https://example.com/v1"',
            ]
        )
    )

    manager = ContextManager(
        replace(
            ContextConfig.from_codex_config(config_path),
            include_permissions_instructions=False,
            include_skills_instructions=False,
        )
    )

    assert manager.resolve_auto_compact_token_limit() == 12345


def test_context_manager_accepts_explicit_cwd(tmp_path, monkeypatch) -> "None":
    explicit_cwd = tmp_path / "workspace"
    explicit_cwd.mkdir()
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    manager = ContextManager(ContextConfig(cwd=explicit_cwd))

    assert manager.cwd == explicit_cwd.resolve()


@pytest.mark.parametrize(
    "model",
    ["step-3.5-flash", "step-3.5-flash-2603", "step-3.7-flash"],
)
def test_context_manager_resolves_model_instructions_from_step_models_json_entry(
    model,
) -> "None":
    manager = ContextManager(config=ContextConfig(model=model, personality="pragmatic"))

    instructions = manager.resolve_base_instructions()

    assert instructions.startswith("You are Codex, a coding agent based on Step-")
    assert "GPT-5" not in instructions
    assert "You are a deeply pragmatic, effective software engineer." in instructions
    assert "Always use apply_patch for manual code edits." in instructions


def test_context_manager_builds_official_style_context_messages(
    tmp_path,
    monkeypatch,
) -> "None":
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
                "project_doc_max_bytes = 1024",
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

    manager = ContextManager(
        replace(
            ContextConfig.from_codex_config(config_path),
            base_instructions_override="override rules",
        )
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
    assert developer_texts[0] == "developer rules"
    assert len(developer_texts) == 3
    assert developer_texts[1].startswith("<skills_instructions>")
    assert (
        "- alpha: Alpha skill description. (file: r0/alpha/SKILL.md)"
        in developer_texts[1]
    )
    assert (
        "- omega: Omega skill description. (file: r1/omega/SKILL.md)"
        in developer_texts[1]
    )
    assert developer_texts[1].index("- omega:") < developer_texts[1].index("- alpha:")
    assert "- `r0` = `{0}`".format(skills_root) in developer_texts[1]
    assert "- `r1` = `{0}/.system`".format(skills_root) in developer_texts[1]
    assert developer_texts[2].startswith("<permissions instructions>")
    assert "Approval policy is currently never." in developer_texts[2]

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


@pytest.mark.parametrize("developer_instructions", [None, "developer rules"])
def test_context_manager_only_includes_enabled_developer_instructions(
    developer_instructions,
):
    manager = ContextManager(
        ContextConfig(
            developer_instructions=developer_instructions,
            include_permissions_instructions=False,
            include_skills_instructions=False,
        )
    )
    prompt = manager.build_prompt([UserMessage("hello")], [], True)
    developer_messages = [
        item
        for item in prompt.input
        if isinstance(item, ContextMessage) and item.role == "developer"
    ]
    assert [
        part["text"] for item in developer_messages for part in item.content_items
    ] == ([] if developer_instructions is None else [developer_instructions])


def test_context_manager_includes_extra_contextual_user_messages() -> "None":
    manager = ContextManager(
        ContextConfig(
            extra_contextual_user_messages=(
                "",
                "Current workspace board file: ./board.html",
            ),
            include_permissions_instructions=False,
            include_skills_instructions=False,
        )
    )

    prompt = manager.build_prompt([UserMessage(text="hello")], [], True)

    contextual_user_message = prompt.input[0]
    assert isinstance(contextual_user_message, ContextMessage)
    assert contextual_user_message.content_items is not None
    texts = [item["text"] for item in contextual_user_message.content_items]
    assert texts[-1] == "Current workspace board file: ./board.html"
    assert prompt.input[-1] == UserMessage(text="hello")


@pytest.mark.asyncio
async def test_agent_injects_context_without_polluting_history(
    tmp_path,
    monkeypatch,
) -> "None":
    model = ScriptedModelClient([ModelResponse(items=[AssistantMessage(text="done")])])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.setenv("TZ", "Asia/Hong_Kong")
    config = ContextConfig(
        base_instructions_override="base rules",
        developer_instructions="developer rules",
        user_instructions="user rules",
        approval_policy="never",
        sandbox_mode="danger-full-access",
        include_skills_instructions=False,
    )

    agent = Agent(model, ToolRegistry(), config)
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
    assert developer_message.content_items[1]["text"].startswith(
        "<permissions instructions>"
    )
    assert developer_message.content_items[0]["text"] == "developer rules"

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


def test_context_manager_keeps_workspaces_within_same_turn(
    monkeypatch, tmp_path
) -> "None":
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
            include_skills_instructions=False,
        ),
    )

    first = manager.get_turn_metadata("turn_a")
    second_same_turn = manager.get_turn_metadata("turn_a")
    third_new_turn = manager.get_turn_metadata("turn_b")

    assert list(first) == ["turn_id", "workspaces", "sandbox"]
    assert list(second_same_turn) == ["turn_id", "workspaces", "sandbox"]
    assert second_same_turn["workspaces"] == first["workspaces"]
    assert list(third_new_turn) == ["turn_id", "sandbox"]
