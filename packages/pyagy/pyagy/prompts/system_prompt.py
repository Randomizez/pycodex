"""Exact 12-section systemInstruction builder matching agy 1.5.13."""

from dataclasses import dataclass
from pathlib import Path
import platform
import typing
import yaml


STATIC_PREFIX_PATH = Path(__file__).resolve().parent / "static_system_prefix.txt"

SKILLS_PREAMBLE = """<skills>
You can use specialized 'skills' to help you with complex tasks. Each skill has a name and a description listed below.

Skills are folders of instructions, scripts, and resources that extend your capabilities for specialized tasks. Each skill folder contains:
- **SKILL.md** (required): The main instruction file with YAML frontmatter (name, description) and detailed markdown instructions

More complex skills may include additional directories and files as needed, for example:
- **scripts/** - Helper scripts and utilities that extend your capabilities
- **examples/** - Reference implementations and usage patterns
- **resources/** - Additional files, templates, or assets the skill may reference
- **references/** - Contains additional documentation that agents can read when needed


If a skill seems relevant to your current task, you MUST read its `SKILL.md` instructions before proceeding. You may skip this step only if you are delegating the skill-related task to a subagent that will read and follow the instructions itself.

When viewing these skill files, always use the exact path provided in the "Available skills" list below.

Available skills:
"""

USER_RULES_PREAMBLE = (
    "<user_rules>\n"
    "The following are user-defined rules that you MUST ALWAYS FOLLOW WITHOUT ANY EXCEPTION. "
    "These rules take precedence over any other instructions.\n"
    "Review them carefully and always take them into account when you generate responses and code:\n"
)

RESEARCH_SUBAGENT_IDENTITY = (
    "<Identity>\n"
    "You are a research subagent. Your job is to explore the codebase, read files, and search for "
    "information on behalf of the main agent. You have read-only access and cannot make any changes "
    "to the codebase. Only use the run_command tool for read-only exploration (e.g. listing directories, "
    "reading files, inspecting VCS history); NEVER run commands that write, modify, delete, install, or "
    "otherwise change any file or state. Focus on gathering accurate, relevant information and reporting "
    "your findings clearly and concisely back to the main agent via the send_message tool.\n"
    "</Identity>"
)

IMAGE_GENERATOR_SUBAGENT_INSTRUCTIONS = (
    "<Instructions>\n"
    "You generate and edit images for the main agent with write_file and run_command.\n"
    "- Check each result against the brief. If it clearly misses, fix it; at most 3 attempts per image.\n"
    "- Finish with send_message: the absolute path of each final image, one line on what it shows.\n"
    "</Instructions>"
)

BUILTIN_MOUNTED_SKILLS: "typing.Tuple[typing.Tuple[str, str, str], ...]" = (
    (
        "agy-customizations",
        "agy-customizations",
        (
            "Comprehensive guide and reference for the Antigravity Customization System. "
            "Use to explain how customizations work, their loading priority, discovery mechanisms, "
            "and to guide the creation of skills, rules, plugins, hooks, and MCP servers."
        ),
    ),
    (
        "antigravity-guide",
        "antigravity_guide",
        (
            "Provides a comprehensive guide, quick reference, and sitemap for Google Antigravity (AGY), "
            "including the Antigravity CLI (agy), Antigravity 2.0, Antigravity IDE, Python SDK, slash commands, "
            "keybindings, and customizations (skills, rules, MCP, sidecars). Activate this skill when the user "
            "asks questions about how to use, configure, or customize Antigravity, AGY, the agy CLI, "
            "the Antigravity IDE, or Antigravity 2.0."
        ),
    ),
    (
        "run-workflow",
        "run-workflow",
        (
            "How to write and run a dynamic workflow script with the `run_workflow` tool: a Python script "
            "that orchestrates many agents with `agent`, `parallel`, `pipeline`, and `phase`. Use when a task "
            "splits into many independent or staged pieces of agent work (fan out over files, modules, bugs, "
            "or review angles, then aggregate), or when the user asks for a workflow. Don't use for a single "
            "delegated task (use a subagent) or for legacy workflow files (see migrate-workflows)."
        ),
    ),
    (
        "swarm_monitor",
        "swarm_monitor",
        (
            "ONLY for the native-swarm worker whose brief says it is the monitor. On a timer, it checks "
            "whether the whole swarm has stalled and calls on the one member who can unstick it. Coordinators "
            "and other workers: do not load this."
        ),
    ),
)


@dataclass(frozen=True)
class AgySkillEntry:
    name: "str"
    path: "Path"
    description: "str"


def default_app_data_dir(home: "typing.Optional[Path]" = None) -> "Path":
    base_home = (home if home is not None else Path.home()).expanduser()
    return base_home / ".gemini" / "antigravity-cli"


def find_git_repo_root(cwd: "Path") -> "Path":
    resolved_cwd = cwd.resolve()
    for candidate in (resolved_cwd,) + tuple(resolved_cwd.parents):
        dot_git = candidate / ".git"
        if dot_git.is_file() or (dot_git.is_dir() and (dot_git / "HEAD").exists()):
            return candidate
    return resolved_cwd


def _ancestor_chain_root_to_cwd(cwd: "Path") -> "typing.List[Path]":
    resolved_cwd = cwd.resolve()
    repo_root = find_git_repo_root(resolved_cwd)
    chain = [resolved_cwd]
    current = resolved_cwd
    while current != repo_root and current.parent != current:
        current = current.parent
        chain.append(current)
    chain.reverse()
    return chain


def _parse_skill_markdown(skill_md: "Path") -> "typing.Optional[AgySkillEntry]":
    text = skill_md.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---\n"):
        return None
    end_idx = text.find("\n---\n", 4)
    if end_idx == -1:
        return None
    frontmatter = yaml.safe_load(text[4:end_idx])
    if not isinstance(frontmatter, dict):
        return None
    if frontmatter.get("disable-model-invocation") is True:
        return None
    raw_name = frontmatter.get("name")
    raw_desc = frontmatter.get("description")
    if not isinstance(raw_name, str) or not isinstance(raw_desc, str):
        return None
    name = raw_name.strip()
    description = " ".join(line.strip() for line in raw_desc.splitlines() if line.strip())
    if not name or not description:
        return None
    return AgySkillEntry(name=name, path=skill_md, description=description)


def resolve_sorted_workspaces(
    cwd: "Path",
    additional_workspaces: "typing.Sequence[Path]" = (),
) -> "typing.List[Path]":
    unique_paths = {cwd.resolve()}
    for extra in additional_workspaces:
        unique_paths.add(Path(extra).expanduser().resolve())
    return sorted(unique_paths, key=str)


def discover_agy_skills(
    cwd: "Path",
    app_data_dir: "Path",
    global_config_dir: "typing.Optional[Path]" = None,
    additional_workspaces: "typing.Sequence[Path]" = (),
    codex_skills_home: "typing.Optional[Path]" = None,
) -> "typing.List[AgySkillEntry]":
    resolved_app_data = app_data_dir.expanduser()
    if global_config_dir is None:
        if resolved_app_data.parent.name == ".gemini":
            resolved_global = resolved_app_data.parent / "config"
        else:
            resolved_global = Path.home() / ".gemini" / "config"
    else:
        resolved_global = global_config_dir.expanduser()

    by_name: "typing.Dict[str, AgySkillEntry]" = {}

    for skill_name, dir_name, fallback_desc in BUILTIN_MOUNTED_SKILLS:
        skill_path = resolved_app_data / "builtin" / "skills" / dir_name / "SKILL.md"
        parsed = _parse_skill_markdown(skill_path) if skill_path.is_file() else None
        if parsed is not None:
            by_name[parsed.name] = AgySkillEntry(
                name=parsed.name,
                path=skill_path,
                description=parsed.description,
            )
        else:
            by_name[skill_name] = AgySkillEntry(
                name=skill_name,
                path=skill_path,
                description=fallback_desc,
            )

    if codex_skills_home is not None:
        from pycodex.context import discover_skills

        for desc in sorted(
            discover_skills(codex_skills_home),
            key=lambda item: (item.scope_rank, item.name, str(item.path_to_skill_md)),
            reverse=True,
        ):
            clean_desc = " ".join(
                line.strip() for line in desc.description.splitlines() if line.strip()
            )
            if desc.name and clean_desc:
                by_name[desc.name] = AgySkillEntry(
                    name=desc.name,
                    path=desc.path_to_skill_md,
                    description=clean_desc,
                )

    global_skills_root = resolved_global / "skills"
    if global_skills_root.is_dir():
        for skill_md in sorted(global_skills_root.glob("*/SKILL.md")):
            if skill_md.is_file():
                parsed = _parse_skill_markdown(skill_md)
                if parsed is not None:
                    by_name[parsed.name] = parsed

    for ws in resolve_sorted_workspaces(cwd, additional_workspaces):
        for directory in _ancestor_chain_root_to_cwd(ws):
            for folder_name in (".agents", ".agent", "_agents", "_agent"):
                ws_skills_root = directory / folder_name / "skills"
                if not ws_skills_root.is_dir():
                    continue
                for skill_md in sorted(ws_skills_root.glob("*/SKILL.md")):
                    if skill_md.is_file():
                        parsed = _parse_skill_markdown(skill_md)
                        if parsed is not None:
                            by_name[parsed.name] = parsed

    return sorted(by_name.values(), key=lambda entry: entry.name)


def discover_user_rules(
    cwd: "Path",
    additional_workspaces: "typing.Sequence[Path]" = (),
) -> "typing.List[typing.Tuple[Path, str]]":
    seen: "typing.Set[Path]" = set()
    rules: "typing.List[typing.Tuple[Path, str]]" = []
    for ws in resolve_sorted_workspaces(cwd, additional_workspaces):
        for directory in _ancestor_chain_root_to_cwd(ws):
            for filename in ("AGENTS.md", "GEMINI.md"):
                candidate = directory / filename
                if not candidate.is_file():
                    continue
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                raw_content = resolved.read_text(encoding="utf-8", errors="replace")
                rules.append((resolved, raw_content))
    return rules


def _os_version_label() -> "str":
    sys_name = platform.system().lower()
    if sys_name.startswith("darwin"):
        return "mac"
    if sys_name.startswith("win"):
        return "windows"
    return "linux"


def _extract_tagged_block(source: "str", tag: "str") -> "str":
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = source.index(open_tag)
    end = source.index(close_tag, start) + len(close_tag)
    return source[start:end]


def _build_user_rules_block(
    cwd: "Path",
    additional_workspaces: "typing.Sequence[Path]" = (),
) -> "str":
    rules = discover_user_rules(cwd, additional_workspaces=additional_workspaces)
    if not rules:
        return ""
    rule_blocks = "".join(
        f"<RULE[{rule_path}]>\n{raw_content}\n</RULE[{rule_path}]>\n"
        for rule_path, raw_content in rules
    )
    return f"{USER_RULES_PREAMBLE}{rule_blocks}</user_rules>"


def _build_user_information_block(
    cwd: "Path",
    conversation_id: "str",
    resolved_app_data: "Path",
    additional_workspaces: "typing.Sequence[Path]" = (),
) -> "str":
    workspaces = resolve_sorted_workspaces(cwd, additional_workspaces)
    workspace_lines = "\n".join(f"- {ws}" for ws in workspaces)
    cmd_cwd = workspaces[0]
    artifact_dir = resolved_app_data / "brain" / conversation_id
    return (
        "<user_information>\n"
        f"The USER's OS version is {_os_version_label()}.\n"
        "Active Workspaces:\n"
        f"{workspace_lines}\n"
        f"Command Working Directory: {cmd_cwd}\n"
        "Files relating to the user's requests should be written in the locations listed above. "
        "Avoid writing project files to tmp, in the .gemini dir, or directly to the Desktop and similar folders unless explicitly asked.\n"
        f"App Data Directory: {resolved_app_data}\n"
        f"Conversation ID: {conversation_id}\n"
        f"Artifact Directory Path: {artifact_dir}\n"
        "</user_information>"
    )


def build_subagent_reminder_block(caller_name: "str", caller_conversation_id: "str") -> "str":
    return (
        "<subagent_reminder>\n"
        f'You are running as a subagent, invoked by a caller agent (name: "{caller_name}", id: "{caller_conversation_id}"). '
        "You MUST use send_message to communicate all results, reports, and updates back to the caller. "
        "Your response is NOT automatically relayed — if you do not call send_message, the caller will only know that you have gone idle. "
        "Always use the caller's id as the Recipient.\n\n"
        "Text you generate outside of send_message will NOT be seen by the caller, so keep them brief. "
        "Put all important information — findings, summaries, conclusions — into your send_message calls instead. "
        "You can also share files by including their absolute paths in your message; the caller can then read them directly.\n"
        "</subagent_reminder>"
    )


def build_system_instruction(
    cwd: "Path",
    conversation_id: "str",
    app_data_dir: "typing.Optional[Path]" = None,
    additional_workspaces: "typing.Sequence[Path]" = (),
    global_config_dir: "typing.Optional[Path]" = None,
    custom_system_prompt: "typing.Optional[str]" = None,
    codex_skills_home: "typing.Optional[Path]" = None,
) -> "str":
    if custom_system_prompt is not None:
        return custom_system_prompt

    resolved_cwd = cwd.resolve()
    resolved_app_data = (
        app_data_dir.expanduser() if app_data_dir is not None else default_app_data_dir()
    )

    static_prefix = STATIC_PREFIX_PATH.read_text(encoding="utf-8")

    skills = discover_agy_skills(
        cwd=resolved_cwd,
        app_data_dir=resolved_app_data,
        global_config_dir=global_config_dir,
        additional_workspaces=additional_workspaces,
        codex_skills_home=codex_skills_home,
    )
    skill_lines = "\n".join(
        f"- {entry.name} ({entry.path}): {entry.description}" for entry in skills
    )
    skills_section = f"{SKILLS_PREAMBLE}{skill_lines}\n\n\n</skills>\n"

    rules_block = _build_user_rules_block(
        resolved_cwd, additional_workspaces=additional_workspaces
    )
    rules_section = f"{rules_block}\n" if rules_block else ""

    user_info_section = _build_user_information_block(
        cwd=resolved_cwd,
        conversation_id=conversation_id,
        resolved_app_data=resolved_app_data,
        additional_workspaces=additional_workspaces,
    )

    return f"{static_prefix}{skills_section}{rules_section}{user_info_section}"


def build_subagent_system_instruction(
    type_name: "str",
    cwd: "Path",
    conversation_id: "str",
    caller_name: "str",
    caller_conversation_id: "str",
    app_data_dir: "typing.Optional[Path]" = None,
    additional_workspaces: "typing.Sequence[Path]" = (),
    global_config_dir: "typing.Optional[Path]" = None,
    blueprint_system_prompt: "typing.Optional[str]" = None,
) -> "str":
    resolved_cwd = cwd.resolve()
    resolved_app_data = (
        app_data_dir.expanduser() if app_data_dir is not None else default_app_data_dir()
    )
    reminder = build_subagent_reminder_block(caller_name, caller_conversation_id)
    if type_name == "self":
        base = build_system_instruction(
            cwd=resolved_cwd,
            conversation_id=conversation_id,
            app_data_dir=resolved_app_data,
            additional_workspaces=additional_workspaces,
            global_config_dir=global_config_dir,
        )
        return f"{base}\n{reminder}"

    static_prefix = STATIC_PREFIX_PATH.read_text(encoding="utf-8")
    messaging_block = _extract_tagged_block(static_prefix, "messaging")
    artifacts_block = _extract_tagged_block(static_prefix, "artifacts")
    user_info_block = _build_user_information_block(
        cwd=resolved_cwd,
        conversation_id=conversation_id,
        resolved_app_data=resolved_app_data,
        additional_workspaces=additional_workspaces,
    )
    rules_block = _build_user_rules_block(
        resolved_cwd, additional_workspaces=additional_workspaces
    )

    if type_name == "research":
        sections = [
            RESEARCH_SUBAGENT_IDENTITY,
            user_info_block,
            reminder,
            messaging_block,
            artifacts_block,
        ]
        if rules_block:
            sections.append(rules_block)
        return "\n".join(sections)

    if type_name == "image-generator":
        sections = [
            IMAGE_GENERATOR_SUBAGENT_INSTRUCTIONS,
            user_info_block,
            reminder,
            messaging_block,
        ]
        if rules_block:
            sections.append(rules_block)
        return "\n".join(sections)

    guidelines_block = _extract_tagged_block(static_prefix, "guidelines")
    comm_block = _extract_tagged_block(static_prefix, "communication_style")
    agent_instr = f"<Agent System Instructions>\n{blueprint_system_prompt or ''}\n</Agent System Instructions>"
    sections = [agent_instr, user_info_block]
    if rules_block:
        sections.append(rules_block)
    sections.extend([messaging_block, artifacts_block, guidelines_block, comm_block, reminder])
    return "\n".join(sections)
