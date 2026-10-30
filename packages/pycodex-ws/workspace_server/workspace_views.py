"""Read-only file and diff views opened by workspace links."""

import base64
import io
import json
import mimetypes
import os
import re
import stat
import subprocess
import tempfile
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

from PIL import Image

from pycodex.utils import uuid7_string


def _view_title(documents):
    return " <-> ".join(document["label"].replace("\\", "/") for document in documents)


class WorkspaceViewStore:
    """Share file-backed views and fixed Git contents across workspace sessions."""

    max_storage_bytes = 64 * 1024 * 1024
    max_file_bytes = 2 * 1024 * 1024
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
            raise ValueError("file exceeds 2 MiB: {0}".format(label))
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


class WorkspaceViewOpener:
    """Resolve a clicked file path or workspace URI without executing commands."""

    def __init__(self, views, cwd):
        self._views = views
        self._cwd = Path(cwd).resolve()

    def open(self, href, src_path=None):
        action, options = self._parse_link(href)
        line = options.get("line")
        if line is not None:
            if not re.fullmatch(r"[1-9][0-9]*", line):
                raise ValueError("line must be a positive integer")
            line = int(line)
        side = options.get("side", "right")
        if action == "file":
            kind = "file"
            path = options["path"]
            if src_path is not None:
                if not isinstance(src_path, str):
                    raise ValueError("src_path must be a file path")
                path = Path(src_path).parent / path
            references = [self._file_reference(path)]
        elif action == "git-file":
            kind = "file"
            references = [self._git_object_reference(options["target"])]
        elif action == "diff":
            kind = "diff"
            references = [
                self._file_reference(options[name]) for name in ("left", "right")
            ]
        else:
            kind = "diff"
            left = self._git_object_reference(options["left"])
            if "right" in options:
                right = self._git_object_reference(options["right"])
            else:
                source = left["git_source"]
                right = self._file_reference(
                    Path(source["repository"]) / source["path"],
                    source["path"], working_tree=True,
                )
            references = [left, right]
        selection = None if line is None else {
            "document": 0 if kind == "file" or side == "left" else 1,
            "line": line,
        }
        allow_missing = action == "git-diff"
        selected_document = None
        if selection is not None:
            selected_document = self._read_reference(
                references[selection["document"]], kind == "file", allow_missing
            )
            if "text" not in selected_document or selected_document.get("missing"):
                raise ValueError("line selection requires an existing text file")
            text = selected_document["text"].replace("\r\n", "\n").replace("\r", "\n")
            if selection["line"] > text.count("\n") + 1:
                raise ValueError("line {0} is outside the file".format(selection["line"]))
        existing = self._views.select(
            {"kind": kind, "documents": references}, selection
        )
        if existing is not None:
            return self._views.get(existing["id"])
        documents = [
            selected_document if selection is not None and selection["document"] == index
            else self._read_reference(reference, kind == "file", allow_missing)
            for index, reference in enumerate(references)
        ]
        if kind == "diff" and all(document.get("missing") for document in documents):
            raise ValueError("file is absent on both sides of the comparison")
        view = self._views.add({"kind": kind, "documents": documents}, selection)
        return self._views.get(view["id"])

    def _parse_link(self, href):
        if not isinstance(href, str) or not href or len(href) > 8192:
            raise ValueError("expected a file path or workspace URI (at most 8192 characters)")
        if href != href.strip() or any(
            ord(character) < 32 or ord(character) == 127 for character in href
        ):
            raise ValueError("surrounding whitespace and control characters are not allowed in workspace links")
        if re.search(r"%(?![0-9a-fA-F]{2})", href):
            raise ValueError("invalid percent encoding in workspace link")
        destination, separator, fragment = href.partition("#")
        if "?" in destination:
            raise ValueError("workspace links do not use query parameters; encode a literal ? as %3F")
        action, options = self._parse_destination(destination)
        if separator:
            pattern = r"[LR]-?[0-9]+" if action in ("diff", "git-diff") else r"L-?[0-9]+"
            if not re.fullmatch(pattern, fragment, re.IGNORECASE):
                raise ValueError("expected #L<number> or, for a diff's right side, #R<number>")
            if "line" in options:
                raise ValueError("specify the line only once")
            options["line"] = fragment[1:]
            if action in ("diff", "git-diff"):
                options["side"] = "left" if fragment[0].upper() == "L" else "right"
        options = {
            key: unquote(value, encoding="utf-8", errors="strict")
            for key, value in options.items()
        }
        if any(ord(character) < 32 or ord(character) == 127
               for value in options.values() for character in value):
            raise ValueError("control characters are not allowed in workspace links")
        return action, options

    def _parse_destination(self, destination):
        if destination.lower().startswith("git:"):
            target = destination[4:]
            if not self._is_git_target(target):
                raise ValueError("expected git:REV:path")
            return "git-file", {"target": target}
        if destination.lower().startswith("diff:"):
            expression = destination[5:]
            # Parent-directory components are paths, not comparison separators.
            separator = re.search(r"(?<![/.])\.\.(?!/)", expression)
            if separator is None:
                if not self._is_git_target(expression):
                    raise ValueError("expected diff:REV:path, diff:REV..REV:path or diff:left..right")
                return "git-diff", {"left": expression}
            left, right = expression[:separator.start()], expression[separator.end():]
            if not left or not right or re.search(r"(?<![/.])\.\.(?!/)", right):
                raise ValueError("expected two comparison endpoints; encode literal .. as %2E%2E")
            if self._is_git_target(right):
                if not self._is_git_target(left):
                    unused_revision, unused_separator, path = right.partition(":")
                    left += ":" + path
                return "git-diff", {"left": left, "right": right}
            if self._is_git_target(left) or left.startswith("//") or right.startswith("//"):
                raise ValueError("compare two local files or two Git objects")
            return "diff", {"left": left, "right": right}
        if re.match(
            r"^(?:https?|ftps?|mailto|tel|sms|javascript|vbscript|data|blob|about|file):",
            destination, re.IGNORECASE,
        ):
            raise ValueError("expected a local file path or workspace URI")
        path = destination
        options = {}
        name, separator, number = path.rpartition(":")
        if separator and re.fullmatch(r"-?[0-9]+", number):
            path, options["line"] = name, number
        if not path or path.startswith("//") or self._is_git_target(path):
            raise ValueError("expected a local file path or workspace URI")
        options["path"] = path
        return "file", options

    def _is_git_target(self, value):
        return ":" in value and not re.match(
            r"^[a-zA-Z]:(?:[/\\]|%2f|%5c)", value, re.IGNORECASE
        )

    def _file_reference(self, name, label=None, working_tree=False):
        return {
            "label": str(name if label is None else label).replace("\\", "/"),
            "source": {
                "path": os.path.abspath(str(self._cwd / name)),
                "working_tree": working_tree,
            },
        }

    def _git(self, arguments):
        result = subprocess.run(
            ["git", "--no-pager", "--literal-pathspecs"] + arguments,
            cwd=str(self._cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        if result.returncode:
            raise ValueError(
                result.stderr.decode("utf-8", errors="replace").strip()
                or "Git command failed"
            )
        return result.stdout

    def _read_reference(self, reference, allow_preview, allow_missing):
        if "source" in reference:
            source = reference["source"]
            return self._views.read_document(
                source["path"], reference["label"], allow_preview,
                source["working_tree"], allow_missing,
            )
        source = reference["git_source"]
        entry = self._git([
            "ls-tree", "--full-tree", "-z", source["revision"], "--", source["path"]
        ])
        if not entry:
            if not allow_missing:
                raise ValueError("Git file does not exist: " + reference["label"])
            return dict(reference, text="", missing=True)
        metadata, unused_path = entry.rstrip(b"\0").split(b"\t", 1)
        unused_mode, object_type, object_id = metadata.split()
        if object_type != b"blob":
            raise ValueError("Git path must identify a file: " + source["path"])
        object_id = object_id.decode("ascii")
        size = int(self._git(["cat-file", "-s", object_id]))
        if size > self._views.max_file_bytes:
            raise ValueError("file exceeds 2 MiB: " + reference["label"])
        document = self._views.decode_document(
            self._git(["cat-file", "blob", object_id]), reference["label"], allow_preview
        )
        document["git_source"] = source
        return document

    def _revision(self, revision):
        if not revision or revision.startswith("-") or ":" in revision or "\0" in revision:
            raise ValueError("expected a Git commit, branch, tag or tree")
        return (
            self._git([
                "rev-parse", "--verify", revision + "^{}"
            ])
            .decode("ascii")
            .strip()
        )

    def _git_reference(self, revision, resolved, root, path):
        label = revision if revision == resolved else "{0} ({1})".format(revision, resolved)
        return {
            "label": label + ":" + path,
            "git_source": {
                "repository": str(root),
                "revision": resolved,
                "path": path,
            },
        }

    def _git_object_reference(self, value):
        revision, separator, path = value.partition(":")
        if not separator or not revision or not path:
            raise ValueError("expected a Git revision:path")
        root = Path(
            self._git(["rev-parse", "--show-toplevel"])
            .decode("utf-8", errors="replace").strip()
        )
        resolved = self._revision(revision)
        directory = self._cwd if path.startswith(("./", "../")) else root
        absolute = Path(os.path.abspath(str(directory / path)))
        try:
            repo_path = absolute.relative_to(root).as_posix()
        except ValueError:
            raise ValueError("Git comparison path must be inside the repository")
        return self._git_reference(revision, resolved, root, repo_path)
