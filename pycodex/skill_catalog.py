"""Skill catalog for workspace input suggestions."""

from .context import discover_skills


def list_skills(codex_home):
    """Return unique, deterministic names without exposing local filesystem paths."""
    result = []
    seen = set()
    for skill in sorted(
        discover_skills(codex_home),
        key=lambda item: (item.scope_rank, item.name, str(item.path_to_skill_md)),
    ):
        if skill.name in seen:
            continue
        seen.add(skill.name)
        result.append({"name": skill.name, "description": skill.description})
    return sorted(result, key=lambda item: item["name"].lower())
