from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 path
    import tomli as tomllib

from .collaboration import DEFAULT_COLLABORATION_MODE, CollaborationMode
from .protocol import ContextMessage, ConversationItem, JSONDict, Prompt, ToolSpec
from .utils.get_env import (
    get_sandbox_tag,
    get_shell_name,
    get_timezone_name,
    get_workspace_turn_metadata,
)

DEFAULT_BASE_INSTRUCTIONS_PATH = (
    Path(__file__).resolve().parent / "prompts" / "default_base_instructions.md"
)
DEFAULT_MODELS_PATH = Path(__file__).resolve().parent / "prompts" / "models.json"
DEFAULT_COLLABORATION_INSTRUCTIONS_PATH = (
    Path(__file__).resolve().parent / "prompts" / "collaboration_default.md"
)
PLAN_COLLABORATION_INSTRUCTIONS_PATH = (
    Path(__file__).resolve().parent / "prompts" / "collaboration_plan.md"
)
PERMISSIONS_SANDBOX_PROMPTS_PATH = (
    Path(__file__).resolve().parent / "prompts" / "permissions" / "sandbox_mode"
)
PERMISSIONS_APPROVAL_PROMPTS_PATH = (
    Path(__file__).resolve().parent / "prompts" / "permissions" / "approval_policy"
)
PROJECT_DOC_SEPARATOR = "\n\n--- project-doc ---\n\n"
DEFAULT_PROJECT_DOC_FILENAME = "AGENTS.md"
LOCAL_PROJECT_DOC_FILENAME = "AGENTS.override.md"
USER_INSTRUCTIONS_PREFIX = "# AGENTS.md instructions for "
PERMISSIONS_OPEN_TAG = "<permissions instructions>"
PERMISSIONS_CLOSE_TAG = "</permissions instructions>"
SKILLS_OPEN_TAG = "<skills_instructions>"
SKILLS_CLOSE_TAG = "</skills_instructions>"
COLLABORATION_MODE_OPEN_TAG = "<collaboration_mode>"
COLLABORATION_MODE_CLOSE_TAG = "</collaboration_mode>"
PERSONALITY_PLACEHOLDER = "{{ personality }}"
SKILLS_GUIDANCE = """- Discovery: The list above is the skills available in this session (name + description + file path). Skill bodies live on disk at the listed paths.
- Trigger rules: If the user names a skill (with `$SkillName` or plain text) OR the task clearly matches a skill's description shown above, you must use that skill for that turn. Multiple mentions mean use them all. Do not carry skills across turns unless re-mentioned.
- Missing/blocked: If a named skill isn't in the list or the path can't be read, say so briefly and continue with the best fallback.
- How to use a skill (progressive disclosure):
  1) After deciding to use a skill, open its `SKILL.md`. Read only enough to follow the workflow.
  2) When `SKILL.md` references relative paths (e.g., `scripts/foo.py`), resolve them relative to the skill directory listed above first, and only consider other paths if needed.
  3) If `SKILL.md` points to extra folders such as `references/`, load only the specific files needed for the request; don't bulk-load everything.
  4) If `scripts/` exist, prefer running or patching them instead of retyping large code blocks.
  5) If `assets/` or templates exist, reuse them instead of recreating from scratch.
- Coordination and sequencing:
  - If multiple skills apply, choose the minimal set that covers the request and state the order you'll use them.
  - Announce which skill(s) you're using and why (one short line). If you skip an obvious skill, say why.
- Context hygiene:
  - Keep context small: summarize long sections instead of pasting them; only load extra files when needed.
  - Avoid deep reference-chasing: prefer opening only files directly linked from `SKILL.md` unless you're blocked.
  - When variants exist (frameworks, providers, domains), pick only the relevant reference file(s) and note that choice.
- Safety and fallback: If a skill can't be applied cleanly (missing files, unclear instructions), state the issue, pick the next-best approach, and continue."""


@dataclass(frozen=True, slots=True)
class ContextConfig:
    base_instructions: str | None = None
    developer_instructions: str | None = None
    user_instructions: str | None = None
    codex_home_instructions: str | None = None
    model_instructions_file: Path | None = None
    codex_home: Path | None = None
    project_doc_max_bytes: int | None = None
    model: str | None = None
    personality: str | None = None
    approval_policy: str | None = None
    sandbox_mode: str | None = None

    @classmethod
    def from_codex_config(
        cls,
        config_path: str | Path,
        profile: str | None = None,
    ) -> ContextConfig:
        path = Path(config_path)
        data = tomllib.loads(path.read_text())
        selected = dict(data)
        if profile is not None:
            overrides = data.get("profiles", {}).get(profile)
            if overrides is None:
                raise ValueError(f"unknown Codex profile: {profile}")
            selected.update(overrides)

        model_instructions_file = selected.get("model_instructions_file")
        resolved_file = None
        if model_instructions_file:
            candidate = Path(str(model_instructions_file))
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            resolved_file = candidate.resolve()

        codex_home = path.parent.resolve()
        codex_home_instructions = _read_first_instruction_file(codex_home)

        return cls(
            base_instructions=_normalize_text(selected.get("base_instructions")),
            developer_instructions=_normalize_text(
                selected.get("developer_instructions")
            ),
            user_instructions=_normalize_text(selected.get("user_instructions")),
            codex_home_instructions=codex_home_instructions,
            model_instructions_file=resolved_file,
            codex_home=codex_home,
            project_doc_max_bytes=_normalize_int(selected.get("project_doc_max_bytes")),
            model=_normalize_text(selected.get("model")),
            personality=_normalize_text(selected.get("personality")),
            approval_policy=_normalize_text(selected.get("approval_policy")),
            sandbox_mode=_normalize_text(selected.get("sandbox_mode")),
        )


@dataclass(frozen=True, slots=True)
class SkillDescriptor:
    name: str
    description: str
    path_to_skill_md: Path
    scope_rank: int


class ContextManager:
    def __init__(
        self,
        base_instructions_override: str | None = None,
        config: ContextConfig | None = None,
        collaboration_mode: CollaborationMode = DEFAULT_COLLABORATION_MODE,
        collaboration_instructions: str | None = None,
        include_collaboration_instructions: bool = False,
        include_permissions_instructions: bool = True,
        include_skills_instructions: bool = True,
        network_access: str = "enabled",
    ) -> None:
        self.cwd = Path.cwd().resolve()
        self._shell = get_shell_name()
        self._current_date = datetime.now().date().isoformat()
        self._timezone_name = get_timezone_name()
        self._base_instructions_override = _normalize_text(base_instructions_override)
        self._config = config or ContextConfig()
        self._collaboration_mode = collaboration_mode
        self._collaboration_instructions = (
            collaboration_instructions
            if collaboration_instructions is not None
            else _default_collaboration_instructions(collaboration_mode)
        )
        self._include_collaboration_instructions = include_collaboration_instructions
        self._include_permissions_instructions = include_permissions_instructions
        self._include_skills_instructions = include_skills_instructions
        self._network_access = network_access
        self._default_base_instructions = DEFAULT_BASE_INSTRUCTIONS_PATH.read_text()
        self._workspace_metadata_turn_id: str | None = None
        self._workspace_metadata_cache: JSONDict | None = None

    @classmethod
    def from_codex_config(
        cls,
        config_path: str | Path,
        profile: str | None = None,
        base_instructions_override: str | None = None,
        collaboration_mode: CollaborationMode = DEFAULT_COLLABORATION_MODE,
        include_collaboration_instructions: bool = False,
        include_permissions_instructions: bool = True,
        include_skills_instructions: bool = True,
        network_access: str = "enabled",
    ) -> ContextManager:
        config = ContextConfig.from_codex_config(config_path, profile)
        return cls(
            base_instructions_override=base_instructions_override,
            config=config,
            collaboration_mode=collaboration_mode,
            include_collaboration_instructions=include_collaboration_instructions,
            include_permissions_instructions=include_permissions_instructions,
            include_skills_instructions=include_skills_instructions,
            network_access=network_access,
        )

    @property
    def collaboration_mode(self) -> CollaborationMode:
        return self._collaboration_mode

    def get_turn_metadata(self, turn_id: str) -> JSONDict:
        metadata: JSONDict = {"turn_id": turn_id}
        if self._workspace_metadata_turn_id is None:
            self._workspace_metadata_turn_id = turn_id
            self._workspace_metadata_cache = get_workspace_turn_metadata(self.cwd)
        if (
            turn_id == self._workspace_metadata_turn_id
            and self._workspace_metadata_cache is not None
        ):
            metadata.update(self._workspace_metadata_cache)
        metadata["sandbox"] = get_sandbox_tag(self._config.sandbox_mode)
        return metadata

    def build_prompt(
        self,
        history: tuple[ConversationItem, ...] | list[ConversationItem],
        tools: list[ToolSpec],
        parallel_tool_calls: bool,
        turn_id: str | None = None,
    ) -> Prompt:
        input_items: list[ConversationItem] = []
        turn_metadata = self.get_turn_metadata(turn_id) if turn_id is not None else None

        developer_message = self._build_developer_message()
        if developer_message is not None:
            input_items.append(developer_message)

        input_items.extend(self._build_contextual_user_messages())
        input_items.extend(list(history))
        return Prompt(
            input=input_items,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            base_instructions=self.resolve_base_instructions(),
            turn_id=turn_id,
            turn_metadata=turn_metadata,
        )

    def resolve_base_instructions(self) -> str:
        if self._base_instructions_override is not None:
            return self._base_instructions_override
        if self._config.base_instructions is not None:
            return self._config.base_instructions
        if self._config.model_instructions_file is not None:
            return self._config.model_instructions_file.read_text().strip()
        resolved = self._resolve_model_instructions()
        if resolved is not None:
            return resolved
        return self._default_base_instructions

    def _resolve_model_instructions(self) -> str | None:
        model_slug = self._config.model
        if model_slug is None:
            return None
        model_metadata = _load_models_by_slug().get(model_slug)
        if model_metadata is None:
            return None

        model_messages = model_metadata.get("model_messages")
        if isinstance(model_messages, dict):
            template = model_messages.get("instructions_template")
            variables = model_messages.get("instructions_variables")
            if isinstance(template, str):
                personality_message = _resolve_personality_message(
                    variables,
                    self._config.personality,
                )
                return template.replace(PERSONALITY_PLACEHOLDER, personality_message)

        base_instructions = model_metadata.get("base_instructions")
        if isinstance(base_instructions, str):
            return base_instructions
        return None

    def _build_developer_message(self) -> ContextMessage | None:
        sections: list[str] = []
        if self._include_permissions_instructions:
            permissions = self._build_permissions_instructions()
            if permissions is not None:
                sections.append(permissions)
        if self._config.developer_instructions is not None:
            sections.append(self._config.developer_instructions)
        if self._include_collaboration_instructions:
            collaboration = self._collaboration_instructions.strip()
            if collaboration:
                sections.append(
                    f"{COLLABORATION_MODE_OPEN_TAG}{collaboration}"
                    f"\n{COLLABORATION_MODE_CLOSE_TAG}"
                )
        if self._include_skills_instructions:
            skills = self._build_skills_instructions()
            if skills is not None:
                sections.append(skills)
        if not sections:
            return None
        return ContextMessage(
            role="developer",
            content_items=tuple(_input_text_item(section) for section in sections),
        )

    def _build_permissions_instructions(self) -> str | None:
        sandbox_mode = self._config.sandbox_mode or "danger-full-access"
        approval_policy = self._config.approval_policy or "never"
        sandbox_prompt_name = sandbox_mode.replace("-", "_")
        sandbox_prompt_path = (
            PERMISSIONS_SANDBOX_PROMPTS_PATH / f"{sandbox_prompt_name}.md"
        )
        approval_prompt_path = (
            PERMISSIONS_APPROVAL_PROMPTS_PATH / f"{approval_policy.replace('-', '_')}.md"
        )
        if not sandbox_prompt_path.exists() or not approval_prompt_path.exists():
            return None

        sandbox_text = (
            sandbox_prompt_path.read_text().strip().replace(
                "{network_access}", self._network_access
            )
        )
        approval_text = approval_prompt_path.read_text().strip()
        return "\n".join(
            [
                PERMISSIONS_OPEN_TAG,
                sandbox_text,
                approval_text,
                PERMISSIONS_CLOSE_TAG,
            ]
        )

    def _build_skills_instructions(self) -> str | None:
        skills = self._discover_skills()
        if not skills:
            return None

        lines = [
            "## Skills",
            "A skill is a set of local instructions to follow that is stored in a `SKILL.md` file. Below is the list of skills that can be used. Each entry includes a name, description, and file path so you can open the source for full instructions when using a specific skill.",
            "### Available skills",
        ]
        for skill in skills:
            path_str = skill.path_to_skill_md.as_posix()
            lines.append(
                f"- {skill.name}: {skill.description} (file: {path_str})"
            )
        lines.append("### How to use skills")
        lines.extend(SKILLS_GUIDANCE.splitlines())
        body = "\n".join(lines)
        return f"{SKILLS_OPEN_TAG}\n{body}\n{SKILLS_CLOSE_TAG}"

    def _discover_skills(self) -> list[SkillDescriptor]:
        codex_home = self._config.codex_home
        if codex_home is None:
            return []

        user_root = codex_home / "skills"
        system_root = user_root / ".system"
        discovered: list[SkillDescriptor] = []
        seen: set[Path] = set()

        user_paths = _discover_skill_files(user_root, excluded_root=system_root)
        system_paths = _discover_skill_files(system_root)

        for scope_rank, paths in ((0, user_paths), (1, system_paths)):
            for path in paths:
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                descriptor = _parse_skill_descriptor(path, scope_rank)
                if descriptor is not None:
                    discovered.append(descriptor)

        return sorted(
            discovered,
            key=lambda skill: (skill.scope_rank, skill.name, skill.path_to_skill_md),
        )

    def _build_contextual_user_messages(self) -> list[ContextMessage]:
        sections: list[str] = []
        user_instructions = self._merged_user_instructions()
        if user_instructions is not None:
            sections.append(
                (
                    f"{USER_INSTRUCTIONS_PREFIX}{self.cwd}\n\n"
                    f"<INSTRUCTIONS>\n{user_instructions}\n</INSTRUCTIONS>"
                )
            )
        sections.append(self._serialize_environment_context())
        if not sections:
            return []
        return [
            ContextMessage(
                role="user",
                content_items=tuple(_input_text_item(section) for section in sections),
            )
        ]

    def _merged_user_instructions(self) -> str | None:
        parts: list[str] = []
        if self._config.user_instructions is not None:
            parts.append(self._config.user_instructions)
        if self._config.codex_home_instructions is not None:
            parts.append(self._config.codex_home_instructions)

        project_doc = self._read_project_docs()
        if project_doc is not None:
            prefix = "\n\n".join(parts)
            if prefix:
                return f"{prefix}{PROJECT_DOC_SEPARATOR}{project_doc}"
            return project_doc

        return "\n\n".join(parts) or None

    def _read_project_docs(self) -> str | None:
        docs: list[str] = []
        remaining = self._config.project_doc_max_bytes
        for path in self._discover_project_doc_paths():
            text = path.read_text()
            if not text.strip():
                continue
            if remaining is None:
                docs.append(text)
                continue
            if remaining <= 0:
                break
            encoded = text.encode()
            docs.append(encoded[:remaining].decode(errors="ignore"))
            remaining -= min(len(encoded), remaining)
        if not docs:
            return None
        return "\n\n".join(docs)

    def _discover_project_doc_paths(self) -> list[Path]:
        seen: set[Path] = set()
        discovered: list[Path] = []

        search_dirs = self._project_search_dirs()
        for directory in search_dirs:
            for candidate_name in (LOCAL_PROJECT_DOC_FILENAME, DEFAULT_PROJECT_DOC_FILENAME):
                candidate = (directory / candidate_name).resolve()
                if candidate.exists() and candidate.is_file() and candidate not in seen:
                    discovered.append(candidate)
                    seen.add(candidate)
                    break
        return discovered

    def _project_search_dirs(self) -> list[Path]:
        project_root = self._find_project_root()
        directories: list[Path] = []
        current = self.cwd
        chain = [current]
        while current != project_root and current.parent != current:
            current = current.parent
            chain.append(current)
        chain.reverse()
        directories.extend(chain)
        return directories

    def _find_project_root(self) -> Path:
        for ancestor in [self.cwd, *self.cwd.parents]:
            if (ancestor / ".git").exists():
                return ancestor
        return self.cwd

    def _serialize_environment_context(self) -> str:
        lines = [
            "<environment_context>",
            f"  <cwd>{self.cwd}</cwd>",
            f"  <shell>{self._shell}</shell>",
            f"  <current_date>{self._current_date}</current_date>",
            f"  <timezone>{self._timezone_name}</timezone>",
            "</environment_context>",
        ]
        return "\n".join(lines)


def _input_text_item(text: str) -> JSONDict:
    return {"type": "input_text", "text": text}


def _normalize_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_int(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _default_collaboration_instructions(mode: CollaborationMode) -> str:
    if mode == "plan":
        return PLAN_COLLABORATION_INSTRUCTIONS_PATH.read_text()
    return DEFAULT_COLLABORATION_INSTRUCTIONS_PATH.read_text()


def _read_first_instruction_file(base: Path) -> str | None:
    for candidate_name in (LOCAL_PROJECT_DOC_FILENAME, DEFAULT_PROJECT_DOC_FILENAME):
        candidate = base / candidate_name
        try:
            contents = candidate.read_text()
        except OSError:
            continue
        trimmed = contents.strip()
        if trimmed:
            return trimmed
    return None


@lru_cache(maxsize=1)
def _load_models_by_slug() -> dict[str, JSONDict]:
    payload = json.loads(DEFAULT_MODELS_PATH.read_text())
    models = payload.get("models", [])
    by_slug: dict[str, JSONDict] = {}
    for model in models:
        slug = model.get("slug")
        if isinstance(slug, str):
            by_slug[slug] = model
    return by_slug


def _resolve_personality_message(variables, personality: str | None) -> str:
    if not isinstance(variables, dict):
        return ""
    normalized = (personality or "").strip().lower()
    if normalized == "friendly":
        key = "personality_friendly"
    elif normalized == "pragmatic":
        key = "personality_pragmatic"
    elif normalized == "none":
        return ""
    else:
        key = "personality_default"
    value = variables.get(key)
    if isinstance(value, str):
        return value
    return ""


def _discover_skill_files(
    root: Path,
    excluded_root: Path | None = None,
) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    excluded = excluded_root.resolve() if excluded_root is not None and excluded_root.exists() else None
    paths: list[Path] = []
    for path in root.glob("**/SKILL.md"):
        resolved = path.resolve()
        if excluded is not None and (resolved == excluded or excluded in resolved.parents):
            continue
        paths.append(path)
    return sorted(paths)


def _parse_skill_descriptor(path: Path, scope_rank: int) -> SkillDescriptor | None:
    text = path.read_text()
    if not text.startswith("---\n"):
        return None
    end_marker = "\n---\n"
    end_index = text.find(end_marker, 4)
    if end_index == -1:
        return None
    frontmatter = text[4:end_index]
    fields: dict[str, str] = {}
    for line in frontmatter.splitlines():
        if ":" not in line:
            continue
        key, _, raw_value = line.partition(":")
        fields[key.strip()] = _strip_yaml_string(raw_value.strip())
    name = fields.get("name")
    description = fields.get("description")
    if not name or not description:
        return None
    return SkillDescriptor(
        name=name,
        description=description,
        path_to_skill_md=path.resolve(),
        scope_rank=scope_rank,
    )


def _strip_yaml_string(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
