"""Workspace definitions and configuration paths shared by web and router."""

import json
import tempfile
import typing
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


def default_state_path() -> Path:
    return Path(tempfile.gettempdir()) / "pcws-{0}".format(uuid4().hex[:8])


def workspace_file_path(
    state_path: "typing.Union[Path, None]", suffix: str
) -> "typing.Union[Path, None]":
    if state_path is None:
        return None
    path = Path(state_path)
    return path.with_name(path.name + suffix)


@dataclass(frozen=True)
class WorkspaceDefinition:
    workspace_id: "str"
    state_path: "Path"
    work_dir: "Path"
    enable_board: "bool" = True

    @property
    def board_path(self) -> "typing.Union[Path, None]":
        return workspace_file_path(self.state_path, ".html") if self.enable_board else None


def load_workspace_definitions(
    config_path: "typing.Union[str, Path]",
) -> "typing.List[WorkspaceDefinition]":
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))

    if isinstance(payload, dict):
        workspaces = payload.get("workspaces")
    else:
        workspaces = payload
    if not isinstance(workspaces, list):
        raise ValueError("workspace config must contain a workspace list")

    result = []
    seen_ids = set()
    seen_states = set()
    for index, item in enumerate(workspaces, start=1):
        if not isinstance(item, dict):
            raise ValueError("workspace entry {0} must be an object".format(index))
        workspace_id = normalize_workspace_id(item.get("id"))
        if not workspace_id:
            workspace_id = next_workspace_id(seen_ids)
        if workspace_id in seen_ids:
            raise ValueError("duplicate workspace name: {0}".format(workspace_id))
        seen_ids.add(workspace_id)

        work_dir_value = item.get("work_dir")
        if work_dir_value is None:
            work_dir_value = item.get("cwd")
        if not str(work_dir_value or "").strip():
            raise ValueError(
                "workspace `{0}` is missing `work_dir`".format(workspace_id)
            )
        work_dir = resolve_workspace_path(str(work_dir_value), path.parent)
        if not work_dir.is_dir():
            raise ValueError(
                "workspace `{0}` work_dir does not exist: {1}".format(
                    workspace_id,
                    work_dir,
                )
            )

        state_value = item.get("state_path")
        enable_board = item.get("enable_board", True)
        if "board" in item:
            if "state_path" in item or "enable_board" in item:
                raise ValueError(
                    "workspace config cannot mix `board` with `state_path` or `enable_board`"
                )
            board_value = item["board"]
            enable_board = board_value is not False
            if board_value is False:
                state_value = "{0}.{1}".format(path.stem, workspace_id)
            elif isinstance(board_value, str) and board_value.strip():
                board_path = resolve_workspace_path(board_value, path.parent)
                if board_path.suffix != ".html":
                    raise ValueError(
                        "legacy workspace board must end in .html to preserve its path"
                    )
                state_value = str(board_path.with_suffix(""))
            else:
                raise ValueError("legacy workspace board must be an HTML path or false")
        if not isinstance(state_value, str) or not state_value.strip():
            raise ValueError("workspace `{0}` requires a state_path".format(workspace_id))
        if not isinstance(enable_board, bool):
            raise ValueError("enable_board must be a boolean")
        state_path = resolve_workspace_path(state_value, path.parent)
        if state_path in seen_states:
            raise ValueError("duplicate workspace state_path: {0}".format(state_path))
        seen_states.add(state_path)

        result.append(
            WorkspaceDefinition(
                workspace_id=workspace_id,
                state_path=state_path,
                work_dir=work_dir,
                enable_board=enable_board,
            )
        )
    return result


def save_workspace_definitions(
    config_path: "typing.Union[str, Path]",
    definitions: "typing.Iterable[WorkspaceDefinition]",
) -> None:
    path = Path(config_path).expanduser().resolve()
    base_dir = path.parent
    payload = {
        "workspaces": [
            _workspace_definition_to_json(definition, base_dir)
            for definition in definitions
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _workspace_definition_to_json(
    definition: "WorkspaceDefinition",
    base_dir: "Path",
) -> "typing.Dict[str, object]":
    return {
        "id": definition.workspace_id,
        "state_path": _format_path_for_workspace_config(definition.state_path, base_dir),
        "work_dir": _format_path_for_workspace_config(definition.work_dir, base_dir),
        "enable_board": definition.enable_board,
    }


def _format_path_for_workspace_config(path: "Path", base_dir: "Path") -> "str":
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(base_dir.resolve()))
    except ValueError:
        return str(resolved)


def resolve_workspace_path(value: str, base_dir: "Path") -> "Path":
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def next_workspace_id(existing: "typing.Container[str]") -> str:
    index = 1
    while True:
        candidate = "workspace-{0}".format(index)
        if candidate not in existing:
            return candidate
        index += 1


def normalize_workspace_id(workspace_id: str) -> str:
    text = str(workspace_id or "").strip()
    if not text:
        return ""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(char not in allowed for char in text):
        raise ValueError(
            "workspace id may only contain letters, numbers, underscore, and dash: {0}".format(
                text
            )
        )
    return text
