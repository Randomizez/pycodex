import typing
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 path
    import tomli as tomllib

from .model_metadata import model_metadata
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
DEFAULT_EFFECTIVE_CONTEXT_WINDOW_PERCENT = 95
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
PERSONALITY_PLACEHOLDER = "{{ personality }}"
SKILLS_GUIDANCE = """- Discovery: The list above is the skills available in this session (name + description + short path). Skill bodies live on disk at the listed paths after expanding the matching alias from `### Skill roots`.
- Trigger rules: If the user names a skill (with `$SkillName` or plain text) OR the task clearly matches a skill's description shown above, you must use that skill for that turn. Multiple mentions mean use them all. Do not carry skills across turns unless re-mentioned.
- Missing/blocked: If a named skill isn't in the list or the path can't be read, say so briefly and continue with the best fallback.
- How to use a skill (progressive disclosure):
  1) After deciding to use a skill, the main agent must expand the listed short `path` with the matching alias from `### Skill roots`, then open and read its `SKILL.md` completely before taking task actions. If a read is truncated or paginated, continue until EOF.
  2) When `SKILL.md` references relative paths (e.g., `scripts/foo.py`), resolve them relative to the directory containing that expanded `SKILL.md` first, and only consider other paths if needed.
  3) If `SKILL.md` points to extra folders such as `references/`, use its routing instructions to identify the files required for the task. The main agent must read each required instruction or reference file itself before acting on it. Do not delegate reading, summarizing, or interpreting skill instructions to a subagent. Subagents may still perform task work when the selected skill allows it.
  4) If `scripts/` exist, prefer running or patching them instead of retyping large code blocks.
  5) If `assets/` or templates exist, reuse them instead of recreating from scratch.
- Coordination and sequencing:
  - If multiple skills apply, choose the minimal set that covers the request and state the order you'll use them.
  - Announce which skill(s) you're using and why (one short line). If you skip an obvious skill, say why.
- Context hygiene:
  - Progressive disclosure applies to selecting relevant files, not partially reading a selected instruction file. Do not load unrelated references, scripts, or assets.
  - Avoid deep reference-chasing: prefer opening only files directly linked from `SKILL.md` unless you're blocked.
  - When variants exist (frameworks, providers, domains), pick only the relevant reference file(s) and note that choice.
- Safety and fallback: If a skill can't be applied cleanly (missing files, unclear instructions), state the issue, pick the next-best approach, and continue."""


@dataclass(
    frozen=True,
)
class ContextConfig:
    base_instructions: "typing.Union[str, None]" = None
    developer_instructions: "typing.Union[str, None]" = None
    user_instructions: "typing.Union[str, None]" = None
    codex_home_instructions: "typing.Union[str, None]" = None
    model_instructions_file: "typing.Union[Path, None]" = None
    codex_home: "typing.Union[Path, None]" = None
    project_doc_max_bytes: "typing.Union[int, None]" = None
    model: "typing.Union[str, None]" = None
    model_context_window: "typing.Union[int, None]" = None
    model_auto_compact_token_limit: "typing.Union[int, None]" = None
    personality: "typing.Union[str, None]" = None
    approval_policy: "typing.Union[str, None]" = None
    sandbox_mode: "typing.Union[str, None]" = None
    base_instructions_override: "typing.Union[str, None]" = None
    include_permissions_instructions: "bool" = True
    include_skills_instructions: "bool" = True
    network_access: "str" = "enabled"
    extra_contextual_user_messages: "typing.Tuple[str, ...]" = ()
    cwd: "typing.Union[str, Path, None]" = None

    @classmethod
    def from_codex_config(
        cls,
        config_path: "typing.Union[str, Path]",
        profile: "typing.Union[str, None]" = None,
    ) -> "ContextConfig":
        path = Path(config_path)
        data = tomllib.loads(path.read_text(encoding="utf-8"))
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
            model_context_window=_normalize_int(selected.get("model_context_window")),
            model_auto_compact_token_limit=_normalize_int(
                selected.get("model_auto_compact_token_limit")
            ),
            personality=_normalize_text(selected.get("personality")),
            approval_policy=_normalize_text(selected.get("approval_policy")),
            sandbox_mode=_normalize_text(selected.get("sandbox_mode")),
        )


@dataclass(
    frozen=True,
)
class SkillDescriptor:
    name: "str"
    description: "str"
    path_to_skill_md: "Path"
    scope_rank: "int"
    root: "Path"


class ContextManager:
    def __init__(self, config: "ContextConfig") -> "None":
        self.cwd = Path(config.cwd or Path.cwd()).resolve()
        self._shell = get_shell_name()
        self._current_date = datetime.now().date().isoformat()
        self._timezone_name = get_timezone_name()
        self._base_instructions_override = _normalize_text(
            config.base_instructions_override
        )
        self._config = config
        self._include_permissions_instructions = config.include_permissions_instructions
        self._include_skills_instructions = config.include_skills_instructions
        self._network_access = config.network_access
        self._extra_contextual_user_messages = tuple(
            text
            for text in (
                _normalize_text(message)
                for message in config.extra_contextual_user_messages
            )
            if text is not None
        )
        self._default_base_instructions = DEFAULT_BASE_INSTRUCTIONS_PATH.read_text(
            encoding="utf-8"
        )
        self._workspace_metadata_turn_id: "typing.Union[str, None]" = None
        self._workspace_metadata_cache: "typing.Union[JSONDict, None]" = None

    def get_turn_metadata(self, turn_id: "str") -> "JSONDict":
        metadata: "JSONDict" = {"turn_id": turn_id}
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
        history: "typing.Union[typing.Tuple[ConversationItem, ...], typing.List[ConversationItem]]",
        tools: "typing.List[ToolSpec]",
        parallel_tool_calls: "bool",
        turn_id: "typing.Union[str, None]" = None,
    ) -> "Prompt":
        input_items: "typing.List[ConversationItem]" = []
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

    def resolve_base_instructions(self) -> "str":
        if self._base_instructions_override is not None:
            return self._base_instructions_override
        if self._config.base_instructions is not None:
            return self._config.base_instructions
        if self._config.model_instructions_file is not None:
            return self._config.model_instructions_file.read_text(
                encoding="utf-8",
                errors="replace",
            ).strip()
        resolved = self._resolve_model_instructions()
        if resolved is not None:
            return resolved
        return self._default_base_instructions

    def resolve_model_max_context_window(self) -> "typing.Union[int, None]":
        metadata = model_metadata(self._config.model)
        context_window = self._config.model_context_window
        if context_window is None and metadata is not None:
            context_window = _normalize_int(metadata.get("context_window"))
        return context_window

    def resolve_model_context_window(self) -> "typing.Union[int, None]":
        context_window = self.resolve_model_max_context_window()
        if context_window is None:
            return None
        metadata = model_metadata(self._config.model)
        effective_percent = None
        if metadata is not None:
            effective_percent = _normalize_int(
                metadata.get("effective_context_window_percent")
            )
        if effective_percent is None:
            effective_percent = DEFAULT_EFFECTIVE_CONTEXT_WINDOW_PERCENT
        return context_window * max(effective_percent, 0) // 100

    def set_model(self, model: "str") -> "None":
        self._config = replace(self._config, model=model)

    def resolve_auto_compact_token_limit(self) -> "typing.Union[int, None]":
        if self._config.model_auto_compact_token_limit is not None:
            return self._config.model_auto_compact_token_limit

        model_slug = self._config.model
        if model_slug is None:
            return None
        metadata = model_metadata(model_slug)
        if metadata is None:
            return None
        return _normalize_int(metadata.get("auto_compact_token_limit"))

    def _resolve_model_instructions(self) -> "typing.Union[str, None]":
        model_slug = self._config.model
        if model_slug is None:
            return None
        metadata = model_metadata(model_slug)
        if metadata is None:
            return None

        model_messages = metadata.get("model_messages")
        if isinstance(model_messages, dict):
            template = model_messages.get("instructions_template")
            variables = model_messages.get("instructions_variables")
            if isinstance(template, str):
                personality_message = _resolve_personality_message(
                    variables,
                    self._config.personality,
                )
                return template.replace(PERSONALITY_PLACEHOLDER, personality_message)

        base_instructions = metadata.get("base_instructions")
        if isinstance(base_instructions, str):
            return base_instructions
        return None

    def _build_developer_message(self) -> "typing.Union[ContextMessage, None]":
        sections: "typing.List[str]" = []
        if self._config.developer_instructions is not None:
            sections.append(self._config.developer_instructions)
        if self._include_skills_instructions:
            skills = self._build_skills_instructions()
            if skills is not None:
                sections.append(skills)
        if self._include_permissions_instructions:
            permissions = self._build_permissions_instructions()
            if permissions is not None:
                sections.append(permissions)
        if not sections:
            return None
        return ContextMessage(
            role="developer",
            content_items=tuple(_input_text_item(section) for section in sections),
        )

    def _build_permissions_instructions(self) -> "typing.Union[str, None]":
        sandbox_mode = self._config.sandbox_mode or "danger-full-access"
        approval_policy = self._config.approval_policy or "never"
        sandbox_prompt_name = sandbox_mode.replace("-", "_")
        sandbox_prompt_path = (
            PERMISSIONS_SANDBOX_PROMPTS_PATH / f"{sandbox_prompt_name}.md"
        )
        approval_prompt_path = (
            PERMISSIONS_APPROVAL_PROMPTS_PATH
            / f"{approval_policy.replace('-', '_')}.md"
        )
        if not sandbox_prompt_path.exists() or not approval_prompt_path.exists():
            return None

        sandbox_text = (
            sandbox_prompt_path.read_text(encoding="utf-8")
            .strip()
            .replace("{network_access}", self._network_access)
        )
        approval_text = approval_prompt_path.read_text(encoding="utf-8").strip()
        return "\n".join(
            [
                PERMISSIONS_OPEN_TAG,
                sandbox_text,
                approval_text,
                PERMISSIONS_CLOSE_TAG,
            ]
        )

    def _build_skills_instructions(self) -> "typing.Union[str, None]":
        skills = self._discover_skills()
        if not skills:
            return None

        lines = [
            "## Skills",
            "A skill is a set of local instructions to follow that is stored in a `SKILL.md` file. Below is the list of skills that can be used. Each entry includes a name, description, and a short path that can be expanded into an absolute path using the skill roots table.",
            "### Skill roots",
        ]
        roots = {}
        for skill in skills:
            if skill.root not in roots:
                alias = "r{0}".format(len(roots))
                roots[skill.root] = alias
                lines.append("- `{0}` = `{1}`".format(alias, skill.root.as_posix()))
        lines.append("### Available skills")
        for skill in sorted(
            skills, key=lambda item: (item.scope_rank, item.name, item.path_to_skill_md)
        ):
            path_str = "{0}/{1}".format(
                roots[skill.root],
                skill.path_to_skill_md.relative_to(skill.root).as_posix(),
            )
            lines.append(f"- {skill.name}: {skill.description} (file: {path_str})")
        metadata = model_metadata(self._config.model)
        if metadata is None or metadata.get("include_skills_usage_instructions", True):
            lines.append("### How to use skills")
            lines.extend(SKILLS_GUIDANCE.splitlines())
        body = "\n".join(lines)
        return f"{SKILLS_OPEN_TAG}\n{body}\n{SKILLS_CLOSE_TAG}"

    def _discover_skills(self) -> "typing.List[SkillDescriptor]":
        codex_home = self._config.codex_home
        if codex_home is None:
            return []

        user_root = codex_home / "skills"
        system_root = user_root / ".system"
        discovered: "typing.List[SkillDescriptor]" = []
        seen: "typing.Set[Path]" = set()

        user_paths = _discover_skill_files(user_root, excluded_root=system_root)
        system_paths = _discover_skill_files(system_root)

        for scope_rank, root, paths in (
            (3, user_root, user_paths),
            (0, system_root, system_paths),
        ):
            for path in paths:
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                descriptor = _parse_skill_descriptor(path, scope_rank, root)
                if descriptor is not None:
                    discovered.append(descriptor)

        return discovered

    def _build_contextual_user_messages(self) -> "typing.List[ContextMessage]":
        sections: "typing.List[str]" = []
        user_instructions = self._merged_user_instructions()
        if user_instructions is not None:
            sections.append(
                (
                    f"{USER_INSTRUCTIONS_PREFIX}{self.cwd}\n\n"
                    f"<INSTRUCTIONS>\n{user_instructions}\n</INSTRUCTIONS>"
                )
            )
        sections.append(self._serialize_environment_context())
        sections.extend(self._extra_contextual_user_messages)
        if not sections:
            return []
        return [
            ContextMessage(
                role="user",
                content_items=tuple(_input_text_item(section) for section in sections),
            )
        ]

    def _merged_user_instructions(self) -> "typing.Union[str, None]":
        parts: "typing.List[str]" = []
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

    def _read_project_docs(self) -> "typing.Union[str, None]":
        docs: "typing.List[str]" = []
        remaining = self._config.project_doc_max_bytes
        for path in self._discover_project_doc_paths():
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            if remaining is None:
                docs.append(text)
                continue
            if remaining <= 0:
                break
            encoded = text.encode("utf-8")
            docs.append(encoded[:remaining].decode(errors="ignore"))
            remaining -= min(len(encoded), remaining)
        if not docs:
            return None
        return "\n\n".join(docs)

    def _discover_project_doc_paths(self) -> "typing.List[Path]":
        seen: "typing.Set[Path]" = set()
        discovered: "typing.List[Path]" = []

        search_dirs = self._project_search_dirs()
        for directory in search_dirs:
            for candidate_name in (
                LOCAL_PROJECT_DOC_FILENAME,
                DEFAULT_PROJECT_DOC_FILENAME,
            ):
                candidate = (directory / candidate_name).resolve()
                if candidate.exists() and candidate.is_file() and candidate not in seen:
                    discovered.append(candidate)
                    seen.add(candidate)
                    break
        return discovered

    def _project_search_dirs(self) -> "typing.List[Path]":
        project_root = self._find_project_root()
        directories: "typing.List[Path]" = []
        current = self.cwd
        chain = [current]
        while current != project_root and current.parent != current:
            current = current.parent
            chain.append(current)
        chain.reverse()
        directories.extend(chain)
        return directories

    def _find_project_root(self) -> "Path":
        for ancestor in [self.cwd, *self.cwd.parents]:
            if (ancestor / ".git").exists():
                return ancestor
        return self.cwd

    def _serialize_environment_context(self) -> "str":
        cwd = escape(str(self.cwd), {'"': "&quot;", "'": "&apos;"})
        lines = [
            "<environment_context>",
            f"  <cwd>{cwd}</cwd>",
            f"  <shell>{self._shell}</shell>",
            f"  <current_date>{self._current_date}</current_date>",
            f"  <timezone>{self._timezone_name}</timezone>",
        ]
        sandbox_mode = self._config.sandbox_mode or "danger-full-access"
        if sandbox_mode == "danger-full-access":
            permissions = (
                '<permission_profile type="disabled"><file_system type="unrestricted" />'
                "</permission_profile>"
            )
        elif sandbox_mode in {"read-only", "workspace-write"}:
            entries = ['<entry access="read"><special>:root</special></entry>']
            if sandbox_mode == "workspace-write":
                entries.extend(
                    [
                        '<entry access="write"><path>{0}</path></entry>'.format(cwd),
                        '<entry access="write"><special>:slash_tmp</special></entry>',
                        '<entry access="write"><special>:tmpdir</special></entry>',
                    ]
                )
                entries.extend(
                    '<entry access="read"><path>{0}</path></entry>'.format(
                        escape(str(self.cwd / name), {'"': "&quot;", "'": "&apos;"})
                    )
                    for name in (".git", ".agents", ".codex")
                )
            permissions = (
                '<permission_profile type="managed"><file_system type="restricted">'
                "{0}</file_system></permission_profile>".format("".join(entries))
            )
        else:
            raise ValueError("unsupported sandbox mode: {0}".format(sandbox_mode))
        lines.append(
            "  <filesystem><workspace_roots><root>{0}</root></workspace_roots>"
            "{1}</filesystem>".format(cwd, permissions)
        )
        lines.append("</environment_context>")
        return "\n".join(lines)


def _input_text_item(text: "str") -> "JSONDict":
    return {"type": "input_text", "text": text}


def _normalize_text(value) -> "typing.Union[str, None]":
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_int(value) -> "typing.Union[int, None]":
    if value is None:
        return None
    return int(value)


def _read_first_instruction_file(base: "Path") -> "typing.Union[str, None]":
    for candidate_name in (LOCAL_PROJECT_DOC_FILENAME, DEFAULT_PROJECT_DOC_FILENAME):
        candidate = base / candidate_name
        try:
            contents = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        trimmed = contents.strip()
        if trimmed:
            return trimmed
    return None


def _resolve_personality_message(
    variables, personality: "typing.Union[str, None]"
) -> "str":
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
    root: "Path",
    excluded_root: "typing.Union[Path, None]" = None,
) -> "typing.List[Path]":
    if not root.exists() or not root.is_dir():
        return []
    excluded = (
        excluded_root.resolve()
        if excluded_root is not None and excluded_root.exists()
        else None
    )
    paths: "typing.List[Path]" = []
    for path in root.glob("**/SKILL.md"):
        resolved = path.resolve()
        if excluded is not None and (
            resolved == excluded or excluded in resolved.parents
        ):
            continue
        paths.append(path)
    return sorted(paths)


def _parse_skill_descriptor(
    path: "Path", scope_rank: "int", root: "Path"
) -> "typing.Union[SkillDescriptor, None]":
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---\n"):
        return None
    end_marker = "\n---\n"
    end_index = text.find(end_marker, 4)
    if end_index == -1:
        return None
    frontmatter = text[4:end_index]
    fields = yaml.safe_load(frontmatter)
    if not isinstance(fields, dict):
        return None
    name = fields.get("name")
    description = fields.get("description")
    if not isinstance(name, str) or not isinstance(description, str):
        return None
    if not name.strip() or not description.strip():
        return None
    metadata_path = path.parent / "agents" / "openai.yaml"
    if metadata_path.is_file():
        metadata = yaml.safe_load(
            metadata_path.read_text(encoding="utf-8", errors="replace")
        )
        if (
            metadata is not None
            and metadata.get("policy", {}).get("allow_implicit_invocation") is False
        ):
            return None
    return SkillDescriptor(
        name=name.strip(),
        description=description.strip(),
        path_to_skill_md=root.resolve() / path.relative_to(root),
        scope_rank=scope_rank,
        root=root.resolve(),
    )
