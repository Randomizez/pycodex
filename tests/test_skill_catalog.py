from pycodex.context import ContextConfig, ContextManager
from pycodex.utils.skill_catalog import list_skills


def write_skill(home, folder, name, description="A test skill"):
    path = home / "skills" / folder / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\nname: {}\ndescription: {}\n---\nBody\n".format(name, description)
    )
    return path


def test_catalog_matches_model_context_and_reloads(tmp_path):
    write_skill(tmp_path, "work", "work-context")
    write_skill(tmp_path, ".system/work", "work-context", "System definition")
    hidden = write_skill(tmp_path, "hidden", "hidden")
    metadata = hidden.parent / "agents" / "openai.yaml"
    metadata.parent.mkdir()
    metadata.write_text("policy:\n  allow_implicit_invocation: false\n")
    (tmp_path / "skills" / "invalid").mkdir()
    (tmp_path / "skills" / "invalid" / "SKILL.md").write_text("No frontmatter")
    assert list_skills(tmp_path) == [
        {"name": "work-context", "description": "System definition"}
    ]
    context = ContextManager(ContextConfig(codex_home=tmp_path))
    assert {item.name for item in context._discover_skills()} == {"work-context"}
    write_skill(tmp_path, "new", "new-skill")
    assert [item["name"] for item in list_skills(tmp_path)] == [
        "new-skill",
        "work-context",
    ]
