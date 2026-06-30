import asyncio
import json
import os
import tempfile
import typing
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pycodex.utils import uuid7_string


SessionFactory = typing.Callable[[], object]


def default_board_path() -> Path:
    return Path(tempfile.gettempdir()) / "pcws-{0}.html".format(uuid4().hex[:8])


class WorkspaceStateStore:
    def __init__(self, board_path: "typing.Union[Path, None]") -> None:
        self.path = (
            None if board_path is None else board_path.with_suffix(".pycodex-ws.json")
        )

    def load_tabs(self) -> "typing.List[typing.Dict[str, str]]":
        if self.path is None or not self.path.is_file():
            return []

        try:
            payload = json.loads(
                self.path.read_text(encoding="utf-8", errors="replace") or "{}"
            )
        except (OSError, ValueError):
            return []

        tabs = payload.get("tabs") if isinstance(payload, dict) else None
        if not isinstance(tabs, list):
            return []

        result = []
        for tab in tabs:
            if not isinstance(tab, dict):
                continue
            title = str(tab.get("title") or "").strip()
            rollout_path = str(tab.get("rollout_path") or "").strip()
            if title or rollout_path:
                result.append({"title": title, "rollout_path": rollout_path})
        return result

    def save_tabs(self, tabs: "typing.Iterable[typing.Dict[str, str]]") -> None:
        if self.path is None:
            return

        state_tabs = [
            {
                "title": str(tab.get("title") or ""),
                "rollout_path": str(tab.get("rollout_path") or ""),
            }
            for tab in tabs
        ]
        payload = json.dumps(
            {"version": 1, "tabs": state_tabs},
            ensure_ascii=False,
            indent=2,
        ) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(payload, encoding="utf-8")


@dataclass(frozen=True)
class WorkspaceDefinition:
    workspace_id: 'str'
    board_path: 'typing.Union[Path, None]'
    work_dir: 'Path'


def load_workspace_definitions(
    config_path: 'typing.Union[str, Path]',
) -> 'typing.List[WorkspaceDefinition]':
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        raise ValueError("unable to read workspace config: {0}".format(exc)) from exc
    except ValueError as exc:
        raise ValueError("workspace config must be valid JSON") from exc

    if isinstance(payload, dict):
        workspaces = payload.get("workspaces")
    else:
        workspaces = payload
    if not isinstance(workspaces, list):
        raise ValueError("workspace config must contain a workspace list")

    result = []
    seen_ids = set()
    seen_boards = set()
    for index, item in enumerate(workspaces, start=1):
        if not isinstance(item, dict):
            raise ValueError("workspace entry {0} must be an object".format(index))
        workspace_id = normalize_workspace_id(str(item.get("id") or "").strip())
        if not workspace_id:
            workspace_id = _next_workspace_id(seen_ids)
        if workspace_id in seen_ids:
            raise ValueError("duplicate workspace name: {0}".format(workspace_id))
        seen_ids.add(workspace_id)

        work_dir_value = item.get("work_dir")
        if work_dir_value is None:
            work_dir_value = item.get("cwd")
        if not str(work_dir_value or "").strip():
            raise ValueError("workspace `{0}` is missing `work_dir`".format(workspace_id))
        work_dir = _resolve_workspace_path(str(work_dir_value), path.parent)
        if not work_dir.is_dir():
            raise ValueError(
                "workspace `{0}` work_dir does not exist: {1}".format(
                    workspace_id,
                    work_dir,
                )
            )

        board_value = item.get("board")
        if not str(board_value or "").strip():
            raise ValueError("workspace `{0}` is missing `board`".format(workspace_id))
        board_path = _resolve_workspace_path(str(board_value), path.parent)
        if board_path in seen_boards:
            raise ValueError("duplicate workspace board: {0}".format(board_path))
        seen_boards.add(board_path)
        if not board_path.parent.is_dir():
            raise ValueError(
                "workspace `{0}` board parent directory does not exist: {1}".format(
                    workspace_id,
                    board_path.parent,
                )
            )

        result.append(
            WorkspaceDefinition(
                workspace_id=workspace_id,
                board_path=board_path,
                work_dir=work_dir,
            )
        )
    return result


def save_workspace_definitions(
    config_path: 'typing.Union[str, Path]',
    definitions: 'typing.Iterable[WorkspaceDefinition]',
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
    definition: 'WorkspaceDefinition',
    base_dir: 'Path',
) -> 'typing.Dict[str, str]':
    result = {
        "id": definition.workspace_id,
        "work_dir": _format_path_for_workspace_config(definition.work_dir, base_dir),
    }
    if definition.board_path is not None:
        result["board"] = _format_path_for_workspace_config(
            definition.board_path,
            base_dir,
        )
    return result


def _format_path_for_workspace_config(path: 'Path', base_dir: 'Path') -> 'str':
    resolved = path.resolve()
    try:
        relative = os.path.relpath(str(resolved), str(base_dir.resolve()))
    except ValueError:
        return str(resolved)
    if relative == ".":
        return "."
    if relative.startswith("..") or os.path.isabs(relative):
        return str(resolved)
    return relative


def _resolve_workspace_path(value: str, base_dir: 'Path') -> 'Path':
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _next_workspace_id(existing: 'typing.Container[str]') -> str:
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


class WorkspaceSessionManager:
    def __init__(
        self,
        session_factory: "SessionFactory",
        board_path: "typing.Union[Path, None]" = None,
        persist_callback: "typing.Union[typing.Callable[[], None], None]" = None,
    ) -> None:
        self._session_factory = session_factory
        self._sessions: "typing.Dict[str, object]" = {}
        self._session_order: "typing.List[str]" = []
        self._state_watchers: "typing.Dict[str, asyncio.Task]" = {}
        self._persisted_titles: "typing.Dict[str, str]" = {}
        self._lock = asyncio.Lock()
        self._state_store = WorkspaceStateStore(board_path)
        self._persist_callback = persist_callback

    def set_persist_callback(
        self,
        callback: "typing.Union[typing.Callable[[], None], None]",
    ) -> None:
        self._persist_callback = callback

    async def start(self) -> None:
        state_tabs = self._state_store.load_tabs()
        if not state_tabs:
            await self.create_session()
            return
        for tab in state_tabs:
            await self.create_session(
                title=str(tab.get("title") or ""),
                rollout_path=str(tab.get("rollout_path") or ""),
            )

    async def close(self) -> None:
        sessions = list(self._sessions.values())
        watchers = list(self._state_watchers.values())
        self._sessions.clear()
        self._session_order = []
        self._state_watchers.clear()
        self._persisted_titles.clear()
        for watcher in watchers:
            watcher.cancel()
        if watchers:
            await asyncio.gather(*watchers, return_exceptions=True)
        for session in sessions:
            await session.close()

    async def create_session(
        self,
        title: str = "",
        rollout_path: str = "",
    ) -> str:
        async with self._lock:
            session_id = uuid7_string()
            session = self._session_factory()
            await session.start()

            if rollout_path:
                await session.restore_from_rollout(rollout_path, title=title)

            self._sessions[session_id] = session
            self._session_order.append(session_id)
            self._persisted_titles[session_id] = str(
                session_summary(session).get("title") or ""
            )
            self._state_watchers[session_id] = asyncio.create_task(
                self._watch_session_title(session_id, session)
            )
            return session_id

    async def close_session(self, session_id: str) -> None:
        async with self._lock:
            if len(self._session_order) <= 1:
                raise ValueError("cannot close the last session")
            session = self._sessions.pop(session_id, None)
            if session is None:
                raise KeyError(session_id)
            watcher = self._state_watchers.pop(session_id, None)
            self._persisted_titles.pop(session_id, None)
            self._session_order = [
                item for item in self._session_order if item != session_id
            ]
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        await session.close()
        self.persist_workspace_state()

    async def _watch_session_title(self, session_id: str, session) -> None:
        subscriber = session.subscribe()
        try:
            while True:
                event = await subscriber.get()
                if event is None:
                    return
                if not isinstance(event, dict) or event.get("type") != "title_changed":
                    continue
                title = str(event.get("title") or "")
                if title == self._persisted_titles.get(session_id, ""):
                    continue
                self._persisted_titles[session_id] = title
                self.persist_workspace_state()
        finally:
            session.unsubscribe(subscriber)

    def persist_workspace_state(self) -> None:
        tabs = []
        for session_id in self._session_order:
            session = self._sessions.get(session_id)
            if session is None:
                continue
            summary = session_summary(session)
            title = str(summary.get("title") or "").strip()
            rollout_path = str(session.rollout_path() or "")
            if not title and not rollout_path:
                continue
            tabs.append({"title": title, "rollout_path": rollout_path})
        self._state_store.save_tabs(tabs)
        if self._persist_callback is not None:
            self._persist_callback()

    def get(self, session_id: "typing.Union[str, None]" = None) -> object:
        resolved_id = self.resolve_session_id(session_id)
        try:
            return self._sessions[resolved_id]
        except KeyError:
            raise KeyError(resolved_id)

    def resolve_session_id(self, session_id: "typing.Union[str, None]" = None) -> str:
        if session_id:
            return str(session_id)
        if not self._session_order:
            raise KeyError("no sessions")
        return self._session_order[0]

    def list_sessions(self) -> "typing.List[typing.Dict[str, object]]":
        result = []
        for session_id in self._session_order:
            session = self._sessions[session_id]
            summary = session_summary(session)
            result.append(
                {
                    "id": session_id,
                    "title": summary.get("title") or "pycodex",
                    "running": bool(summary.get("running")),
                    "spinner": summary.get("spinner") or "",
                    "turn_count": summary.get("turn_count") or 0,
                    "last_assistant": summary.get("last_assistant") or "",
                    "context_remaining_percent": summary.get(
                        "context_remaining_percent"
                    ),
                }
            )
        return result


@dataclass(frozen=True)
class WorkspaceEntry:
    definition: 'WorkspaceDefinition'
    manager: 'WorkspaceSessionManager'

    def to_dict(self) -> 'typing.Dict[str, object]':
        return {
            "id": self.definition.workspace_id,
            "board_path": (
                "" if self.definition.board_path is None
                else str(self.definition.board_path)
            ),
            "work_dir": str(self.definition.work_dir),
            "sessions": self.manager.list_sessions(),
        }


WorkspaceEntryFactory = typing.Callable[
    [WorkspaceDefinition, "typing.Union[typing.Callable[[], None], None]"],
    WorkspaceEntry,
]


class WorkspaceRegistry:
    def __init__(
        self,
        entries: 'typing.Iterable[WorkspaceEntry]',
        config_path: 'typing.Union[str, Path, None]' = None,
        entry_factory: "typing.Union[WorkspaceEntryFactory, None]" = None,
    ) -> None:
        self._entries: 'typing.Dict[str, WorkspaceEntry]' = {}
        self._name_to_key: 'typing.Dict[str, str]' = {}
        self._order: 'typing.List[str]' = []
        self._started = False
        self._config_path = (
            None if config_path is None else Path(config_path).expanduser().resolve()
        )
        self._entry_factory = entry_factory
        for entry in entries:
            self._register(entry)
        if not self._entries and self._entry_factory is None:
            raise ValueError("at least one workspace is required")

    def _register(self, entry: 'WorkspaceEntry') -> None:
        workspace_id = entry.definition.workspace_id
        key = self._key_for_definition(entry.definition)
        if key in self._entries:
            raise ValueError(
                "duplicate workspace board: {0}".format(entry.definition.board_path)
            )
        if workspace_id in self._name_to_key:
            raise ValueError("duplicate workspace name: {0}".format(workspace_id))
        self._entries[key] = entry
        self._name_to_key[workspace_id] = key
        self._order.append(key)
        entry.manager.set_persist_callback(self.persist_definitions)

    def _key_for_definition(self, definition: 'WorkspaceDefinition') -> str:
        if definition.board_path is None:
            raise ValueError("workspace `{0}` is missing board".format(definition.workspace_id))
        return str(definition.board_path.resolve())

    async def start(self) -> None:
        self._started = True
        for key in self._order:
            await self._entries[key].manager.start()

    async def close(self) -> None:
        for key in reversed(self._order):
            await self._entries[key].manager.close()
        self._started = False

    def get(self, workspace_id: str) -> 'WorkspaceEntry':
        workspace_id = normalize_workspace_id(workspace_id)
        if not workspace_id:
            raise KeyError(workspace_id)
        try:
            return self._entries[self._name_to_key[workspace_id]]
        except KeyError:
            raise KeyError(workspace_id)

    async def add_workspace(
        self,
        name: str,
        work_dir: str = "./",
        board: "typing.Union[str, None]" = None,
    ) -> 'WorkspaceEntry':
        if self._entry_factory is None:
            raise ValueError("workspace creation is unavailable")

        base_dir = self._config_path.parent if self._config_path is not None else Path.cwd()
        resolved_work_dir = _resolve_workspace_path(str(work_dir or "./"), base_dir)
        resolved_work_dir.mkdir(parents=True, exist_ok=True)

        board_path = (
            _resolve_workspace_path(str(board), base_dir)
            if str(board or "").strip()
            else default_board_path()
        )
        board_path.parent.mkdir(parents=True, exist_ok=True)
        board_key = str(board_path.resolve())
        if board_key in self._entries:
            raise ValueError("workspace board already exists: {0}".format(board_path))

        workspace_id = normalize_workspace_id(name)
        if not workspace_id:
            workspace_id = _next_workspace_id(self._name_to_key)
        if workspace_id in self._name_to_key:
            raise ValueError("workspace name already exists: {0}".format(workspace_id))

        definition = WorkspaceDefinition(
            workspace_id=workspace_id,
            board_path=board_path,
            work_dir=resolved_work_dir,
        )
        entry = self._entry_factory(definition, self.persist_definitions)
        self._register(entry)
        if self._started:
            await entry.manager.start()
        self.persist_definitions()
        return entry

    async def delete_workspace(self, name: str) -> 'WorkspaceDefinition':
        workspace_id = normalize_workspace_id(name)
        if not workspace_id:
            raise KeyError(workspace_id)
        try:
            key = self._name_to_key.pop(workspace_id)
            entry = self._entries.pop(key)
        except KeyError:
            raise KeyError(workspace_id)
        self._order = [item for item in self._order if item != key]
        await entry.manager.close()
        self.persist_definitions()
        return entry.definition

    def persist_definitions(self) -> None:
        if self._config_path is None:
            return
        save_workspace_definitions(
            self._config_path,
            [self._entries[key].definition for key in self._order],
        )

    def list_workspaces(self) -> 'typing.List[typing.Dict[str, object]]':
        return [
            self._entries[key].to_dict()
            for key in self._order
        ]


def session_snapshot(session) -> "typing.Dict[str, object]":
    return typing.cast("typing.Dict[str, object]", session.snapshot())


def session_summary(session) -> "typing.Dict[str, object]":
    summary = getattr(session, "summary", None)
    if callable(summary):
        return typing.cast("typing.Dict[str, object]", summary())
    snapshot = session_snapshot(session)
    return {
        "title": snapshot.get("title") or "",
        "running": bool(snapshot.get("running")),
        "spinner": snapshot.get("spinner") or "",
        "turn_count": len(snapshot.get("turns") or []),
        "last_assistant": _last_assistant_text(snapshot.get("turns") or []),
        "context_remaining_percent": snapshot.get("context_remaining_percent"),
    }


def _last_assistant_text(turns: "typing.Iterable[typing.Dict[str, object]]") -> str:
    for turn in reversed(list(turns)):
        if str(turn.get("kind") or "assistant") == "control":
            continue
        response = str(turn.get("response") or "").strip()
        if response:
            return response
    return ""
