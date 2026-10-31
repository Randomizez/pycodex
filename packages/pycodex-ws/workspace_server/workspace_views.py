"""Workspace view state, live file refresh and persistence."""

import base64
import io
import json
import mimetypes
import os
import stat
import tempfile
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from pycodex.utils import uuid7_string

from .view_sources import WorkspaceViewOpener


def _view_title(documents):
    return " <-> ".join(document["label"].replace("\\", "/") for document in documents)


class WorkspaceViewStore:
    """Share file-backed views and fixed Git contents across workspace sessions."""

    max_storage_bytes = 64 * 1024 * 1024
    max_file_bytes = 10 * 1024 * 1024
    max_text_lines = 20000

    def __init__(self, path=None):
        self.path = None if path is None else Path(path)
        self._lock = threading.Lock()
        self._views = []
        self._file_states = {}
        self._reveal_revision = 0
        if self.path is not None:
            try:
                with self.path.open("rb") as handle:
                    data = handle.read(self.max_storage_bytes + 1)
            except FileNotFoundError:
                return
            if len(data) > self.max_storage_bytes:
                raise ValueError("workspace view storage exceeds 64 MiB")
            payload = json.loads(data.decode("utf-8"))
            if payload["version"] != 1 or not isinstance(payload["views"], list):
                raise ValueError("invalid workspace view storage")
            self._views = payload["views"]
            self._reveal_revision = payload.get("reveal_revision", 0)
            for view in self._views:
                # Earlier snapshots have no recorded source and remain fixed.
                view.setdefault("revision", 1)
                view.setdefault("reveal_revision", 0)
                view["title"] = _view_title(view["documents"])

    def list_views(self):
        with self._lock:
            self._refresh()
            return [
                dict(
                    {key: view[key] for key in (
                        "id", "kind", "title", "created_at", "revision", "reveal_revision"
                    )},
                    live=any(document.get("source") for document in view["documents"]),
                    selection=deepcopy(view.get("selection")),
                )
                for view in self._views
            ]

    def get(self, view_id):
        with self._lock:
            self._find(view_id)
            self._refresh()
            return deepcopy(self._find(view_id))

    def add(self, snapshot, selection=None):
        with self._lock:
            existing = self._matching(snapshot)
            if existing is not None:
                return self._reveal(existing, False, selection)
            view = dict(
                deepcopy(snapshot),
                title=_view_title(snapshot["documents"]),
                id=uuid7_string(),
                created_at=datetime.now(timezone.utc).isoformat(),
                revision=1,
            )
            return self._reveal(view, True, selection)

    def select(self, reference, selection=None):
        with self._lock:
            existing = self._matching(reference)
            if existing is not None:
                return self._reveal(existing, False, selection)
            return None

    def _identity(self, view):
        references = []
        for document in view["documents"]:
            if document.get("source") is not None:
                source = document["source"]
                references.append((
                    "file", os.path.normcase(os.path.abspath(source["path"])),
                    source["working_tree"],
                ))
            elif document.get("git_source") is not None:
                source = document["git_source"]
                references.append((
                    "git", os.path.normcase(source["repository"]),
                    source["revision"], source["path"],
                ))
            else:
                return None
        return view["kind"], references

    def _matching(self, reference):
        identity = self._identity(reference)
        if identity is not None:
            for view in self._views:
                if self._identity(view) == identity:
                    return view
        return None

    def _reveal(self, view, created, selection):
        revision = self._reveal_revision + 1
        view = dict(view, reveal_revision=revision, selection=deepcopy(selection))
        views = self._views + [view] if created else [
            view if item["id"] == view["id"] else item for item in self._views
        ]
        self._save(views, revision)
        self._views = views
        self._reveal_revision = revision
        return deepcopy(view)

    def close(self, view_id):
        with self._lock:
            self._find(view_id)
            views = [view for view in self._views if view["id"] != view_id]
            self._save(views)
            self._views = views
            self._file_states = {
                key: value for key, value in self._file_states.items()
                if key[0] != view_id
            }

    def _find(self, view_id):
        for view in self._views:
            if view["id"] == view_id:
                return view
        raise KeyError(view_id)

    def _refresh(self):
        views = []
        file_states = {}
        changed = False
        for view in self._views:
            documents = []
            for index, document in enumerate(view["documents"]):
                source = document.get("source")
                if source is None:
                    documents.append(document)
                    continue
                key = (view["id"], index)
                state = None
                current = document
                try:
                    path = Path(source["path"])
                    info = path.lstat() if source["working_tree"] else path.stat()
                    state = (
                        info.st_dev, info.st_ino, info.st_size,
                        info.st_mtime_ns, info.st_ctime_ns,
                    )
                    if self._file_states.get(key) != state:
                        current = self.read_document(
                            path, document["label"], view["kind"] == "file",
                            source["working_tree"],
                        )
                except FileNotFoundError:
                    if view["kind"] == "diff":
                        current = dict(
                            label=document["label"], text="", missing=True,
                            source=source,
                        )
                    else:
                        current = dict(
                            document, missing=True,
                            error="File no longer exists: " + document["label"],
                        )
                except (OSError, ValueError, Image.DecompressionBombError) as error:
                    if isinstance(error, OSError):
                        state = None
                    current = dict(document, error=str(error))
                file_states[key] = state
                documents.append(current)
            if documents != view["documents"]:
                view = dict(view, documents=documents, revision=view["revision"] + 1)
                changed = True
            views.append(view)
        if changed:
            self._save(views)
            self._views = views
        self._file_states = file_states

    def read_document(
        self, path, label, allow_preview=False, working_tree=False, allow_missing=False
    ):
        path = Path(path)
        source = {
            "path": os.path.abspath(str(path)),
            "working_tree": working_tree,
        }
        try:
            if working_tree and path.is_symlink():
                data = os.readlink(str(path)).encode("utf-8")
            else:
                info = path.stat()
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError(
                        "only regular files can be displayed: {0}".format(label)
                    )
                with path.open("rb") as handle:
                    if allow_preview and path.suffix.lower() == ".pdf":
                        if handle.read(5) != b"%PDF-":
                            raise ValueError("invalid PDF header: {0}".format(label))
                        return {
                            "label": label,
                            "media_type": "application/pdf",
                            "source": source,
                            "file_state": [
                                info.st_dev, info.st_ino, info.st_size,
                                info.st_mtime_ns, info.st_ctime_ns,
                            ],
                        }
                    data = handle.read(self.max_file_bytes + 1)
        except FileNotFoundError:
            if not allow_missing:
                raise
            return {"label": label, "text": "", "missing": True, "source": source}
        document = self.decode_document(data, label, allow_preview)
        document["source"] = source
        return document

    def decode_document(self, data, label, allow_image=False):
        if len(data) > self.max_file_bytes:
            raise ValueError("file exceeds 10 MiB: {0}".format(label))
        media_type = mimetypes.guess_type(label)[0] or ""
        if (
            allow_image
            and media_type.startswith("image/")
            and media_type != "image/svg+xml"
        ):
            with Image.open(io.BytesIO(data)) as image:
                image_type = {
                    "PNG": "image/png",
                    "JPEG": "image/jpeg",
                    "GIF": "image/gif",
                    "WEBP": "image/webp",
                }.get(image.format)
                if image_type is None:
                    raise ValueError("unsupported image format: {0}".format(label))
                if image.width * image.height > 20000000:
                    raise ValueError("image exceeds 20 megapixels: {0}".format(label))
                image.verify()
            return {
                "label": label,
                "image_url": "data:{0};base64,{1}".format(
                    image_type, base64.b64encode(data).decode("ascii")
                ),
            }
        if b"\0" in data:
            raise ValueError(
                "binary files cannot be displayed as text: {0}".format(label)
            )
        text = data.decode("utf-8", errors="replace")
        if len(text.splitlines()) > self.max_text_lines:
            raise ValueError("text exceeds 20,000 lines: {0}".format(label))
        return {"label": label, "text": text}

    def _save(self, views, reveal_revision=None):
        data = json.dumps({
            "version": 1,
            "views": views,
            "reveal_revision": (
                self._reveal_revision if reveal_revision is None else reveal_revision
            ),
        }, ensure_ascii=False).encode("utf-8")
        if len(data) > self.max_storage_bytes:
            raise ValueError("workspace views exceed 64 MiB; close some tabs")
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=str(self.path.parent), prefix=self.path.name + ".", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            try:
                handle.write(data)
                handle.close()
                os.replace(str(temporary_path), str(self.path))
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
