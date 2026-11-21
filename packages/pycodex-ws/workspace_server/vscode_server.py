"""Managed VS Code Web Server process, link bridge and HTTP/WebSocket proxy."""

import asyncio
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import time
import typing
import zipfile
import zlib
from functools import partial
from http.cookies import SimpleCookie
from io import StringIO
from pathlib import Path
from urllib.parse import quote, unquote_plus

import anyio
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from pycodex_ws.server_support import AUTH_COOKIE_NAME

from .view_sources import WorkspaceViewOpener

BRIDGE_PUBLISHER = "pycodex"
BRIDGE_NAME = "vscode-bridge"
BRIDGE_VERSION = "0.1.0"
BRIDGE_EXTENSION_ID = "{0}.{1}".format(BRIDGE_PUBLISHER, BRIDGE_NAME)
_UNIX_READER_LIMIT = 262144
_STREAM_CHUNK_SIZE = 65536
_GZIP_MIN_LENGTH = 512
_GZIP_COMPRESS_LEVEL = 3

_HOP_BY_HOP_HEADERS = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    ]
)

_COMPRESSIBLE_MEDIA_TYPES = frozenset(
    [
        "application/javascript",
        "application/x-javascript",
        "application/json",
        "application/xml",
        "application/wasm",
        "application/x-font-ttf",
        "font/ttf",
        "font/otf",
        "image/svg+xml",
    ]
)

_BRIDGE_CONTENT_TYPES_XML = """<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension=".json" ContentType="application/json"/>
  <Default Extension=".js" ContentType="application/javascript"/>
  <Default Extension=".vsixmanifest" ContentType="text/xml"/>
</Types>
"""

_BRIDGE_VSIXMANIFEST_XML = """<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011">
  <Metadata>
    <Identity Language="en-US" Id="{name}" Version="{version}" Publisher="{publisher}"/>
    <DisplayName>PyCodex Workspace Bridge</DisplayName>
    <Description>Open pycodex workspace links inside embedded VS Code Web.</Description>
    <Categories>Other</Categories>
  </Metadata>
  <Installation>
    <InstallationTarget Id="Microsoft.VisualStudio.Code"/>
  </Installation>
  <Dependencies/>
  <Assets>
    <Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json"/>
  </Assets>
</PackageManifest>
""".format(
    name=BRIDGE_NAME, version=BRIDGE_VERSION, publisher=BRIDGE_PUBLISHER
)

_BRIDGE_PACKAGE_JSON = json.dumps(
    {
        "name": BRIDGE_NAME,
        "publisher": BRIDGE_PUBLISHER,
        "version": BRIDGE_VERSION,
        "displayName": "PyCodex Workspace Bridge",
        "description": "Open pycodex workspace links inside embedded VS Code Web.",
        "engines": {"vscode": "^1.70.0"},
        "extensionKind": ["workspace"],
        "activationEvents": ["*"],
        "main": "./extension.js",
    },
    indent=2,
)

_BRIDGE_EXTENSION_JS = """"use strict";
const vscode = require("vscode");
const net = require("net");
const { execFile } = require("child_process");

function selectionRange(line) {
  const zero = Math.max(0, Number(line) - 1);
  return new vscode.Range(zero, 0, zero, 0);
}

function toUri(doc) {
  if (doc.scheme === "file") {
    return vscode.Uri.file(doc.path);
  }
  const query = Buffer.from(
    JSON.stringify({ repo: doc.repo, blob: doc.blob, symlinkTarget: doc.symlink_target })
  ).toString("base64url");
  const cleanPath = String(doc.path || doc.label || "file").replace(/^\\/+/, "");
  return vscode.Uri.from({
    scheme: "pycodex-git",
    path: "/" + cleanPath,
    query,
  });
}

async function openSpec(spec) {
  if (spec.kind === "file") {
    const uri = toUri(spec.documents[0]);
    const options = { preview: false, preserveFocus: false };
    if (spec.selection && spec.selection.line) {
      options.selection = selectionRange(spec.selection.line);
    }
    await vscode.commands.executeCommand("vscode.open", uri, options);
    return;
  }
  const leftUri = toUri(spec.documents[0]);
  const rightUri = toUri(spec.documents[1]);
  const options = { preview: false, preserveFocus: false };
  if (spec.selection && spec.selection.line && spec.selection.document === 1) {
    options.selection = selectionRange(spec.selection.line);
  }
  await vscode.commands.executeCommand("vscode.diff", leftUri, rightUri, spec.title, options);
  if (spec.selection && spec.selection.line) {
    const target = (spec.selection.document === 0 ? leftUri : rightUri).toString();
    const range = selectionRange(spec.selection.line);
    const match = vscode.window.visibleTextEditors.find(
      editor => editor.document && editor.document.uri.toString() === target
    );
    if (match) {
      match.selection = new vscode.Selection(range.start, range.end);
      match.revealRange(range, vscode.TextEditorRevealType.InCenterIfOutsideViewport);
    }
  }
}

function activate(context) {
  const sockPath = process.env.PYCODEX_VSCODE_BRIDGE_SOCK;
  const token = process.env.PYCODEX_VSCODE_BRIDGE_TOKEN || "";
  const windowId = process.env.PYCODEX_VSCODE_WINDOW_ID;
  if (!sockPath || !windowId) {
    return;
  }
  const provider = {
    provideTextDocumentContent(uri) {
      let info;
      try {
        info = JSON.parse(Buffer.from(uri.query, "base64url").toString("utf8"));
      } catch (err) {
        return Promise.reject(err);
      }
      if (!info.blob) {
        return info.symlinkTarget || "";
      }
      return new Promise((resolve, reject) => {
        execFile(
          "git",
          ["--no-pager", "cat-file", "blob", info.blob],
          { cwd: info.repo, encoding: "utf8", maxBuffer: Infinity },
          (error, stdout) => {
            if (error) reject(error);
            else resolve(stdout);
          }
        );
      });
    },
  };
  context.subscriptions.push(
    vscode.workspace.registerTextDocumentContentProvider("pycodex-git", provider)
  );

  let disposed = false;
  let socket = null;
  let reconnectTimer = null;

  function send(payload) {
    if (socket && !socket.destroyed) {
      socket.write(JSON.stringify(payload) + "\\n");
    }
  }

  function connect() {
    if (disposed) return;
    const client = net.createConnection(sockPath);
    socket = client;
    let buffer = "";
    client.on("connect", () => {
      send({ type: "hello", token, window_id: windowId });
    });
    client.on("data", chunk => {
      buffer += chunk.toString("utf8");
      let newline = buffer.indexOf("\\n");
      while (newline >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        newline = buffer.indexOf("\\n");
        if (!line) continue;
        let message;
        try {
          message = JSON.parse(line);
        } catch (_err) {
          continue;
        }
        if (message.type === "supersede") {
          disposed = true;
          if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
          }
          client.end();
          return;
        }
        if (message.type === "open") {
          Promise.resolve()
            .then(() => openSpec(message.spec))
            .then(
              () => send({ type: "ack", id: message.id, ok: true }),
              err =>
                send({
                  type: "ack",
                  id: message.id,
                  ok: false,
                  error: String((err && err.message) || err),
                })
            );
        }
      }
    });
    client.on("error", () => {});
    client.on("close", () => {
      if (socket === client) socket = null;
      if (!disposed && !reconnectTimer) {
        reconnectTimer = setTimeout(() => {
          reconnectTimer = null;
          connect();
        }, 1000);
      }
    });
  }

  connect();
  context.subscriptions.push(
    {
      dispose() {
        disposed = true;
        if (reconnectTimer) clearTimeout(reconnectTimer);
        if (socket) socket.destroy();
      },
    }
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
"""

_BRIDGE_DIGEST = hashlib.sha256(
    (_BRIDGE_PACKAGE_JSON + "\0" + _BRIDGE_EXTENSION_JS).encode("utf-8")
).hexdigest()


def find_code_server() -> Path:
    override = os.environ.get("PYCODEX_VSCODE_WEB_SERVER", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file() or not os.access(str(candidate), os.X_OK):
            raise FileNotFoundError(
                "PYCODEX_VSCODE_WEB_SERVER is not an executable file: {0}".format(
                    candidate
                )
            )
        return candidate.resolve()

    serve_web_root = Path.home() / ".vscode-server" / "cli" / "serve-web"
    if serve_web_root.is_dir():
        candidates = []
        for binary in serve_web_root.glob("*/bin/code-server"):
            workbench = (
                binary.parent.parent
                / "out"
                / "vs"
                / "code"
                / "browser"
                / "workbench"
                / "workbench.html"
            )
            if binary.is_file() and os.access(str(binary), os.X_OK) and workbench.is_file():
                try:
                    mtime = binary.stat().st_mtime
                except OSError:
                    mtime = 0.0
                candidates.append((mtime, binary))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1].resolve()

    which_binary = shutil.which("code-server")
    if which_binary:
        resolved = Path(which_binary).resolve()
        workbench = (
            resolved.parent.parent
            / "out"
            / "vs"
            / "code"
            / "browser"
            / "workbench"
            / "workbench.html"
        )
        if os.access(str(resolved), os.X_OK) and workbench.is_file():
            return resolved

    raise FileNotFoundError(
        "VS Code Web Server was not found under ~/.vscode-server/cli/serve-web "
        "(or PYCODEX_VSCODE_WEB_SERVER).\n"
        "Ask your agent to install VS Code Web Server (vscode-reh-web) into "
        "~/.vscode-server/cli/serve-web/, then click Retry."
    )


def _write_bridge_vsix(vsix_path: Path) -> None:
    with zipfile.ZipFile(str(vsix_path), "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _BRIDGE_CONTENT_TYPES_XML)
        archive.writestr("extension.vsixmanifest", _BRIDGE_VSIXMANIFEST_XML)
        archive.writestr("extension/package.json", _BRIDGE_PACKAGE_JSON)
        archive.writestr("extension/extension.js", _BRIDGE_EXTENSION_JS)


def _ensure_bridge_extension(
    code_server: Path, server_data_dir: Path, extensions_dir: Path, temp_dir: Path
) -> None:
    installed_dir = extensions_dir / "{0}-{1}".format(BRIDGE_EXTENSION_ID, BRIDGE_VERSION)
    ready_marker = installed_dir / ".pycodex-ready"
    registry_file = extensions_dir / "extensions.json"
    if (
        ready_marker.is_file()
        and registry_file.is_file()
        and ready_marker.read_text(encoding="utf-8", errors="replace").strip() == _BRIDGE_DIGEST
        and BRIDGE_EXTENSION_ID in registry_file.read_text(encoding="utf-8", errors="replace")
    ):
        return
    server_data_dir.mkdir(parents=True, exist_ok=True)
    extensions_dir.mkdir(parents=True, exist_ok=True)
    vsix_path = temp_dir / "pycodex-vscode-bridge.vsix"
    _write_bridge_vsix(vsix_path)
    result = subprocess.run(
        [
            str(code_server),
            "--server-data-dir",
            str(server_data_dir),
            "--extensions-dir",
            str(extensions_dir),
            "--install-extension",
            str(vsix_path),
            "--force",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip() or result.stdout.decode(
            "utf-8", errors="replace"
        ).strip()
        raise RuntimeError(
            "Failed to install VS Code bridge extension: {0}".format(
                detail or "exit code {0}".format(result.returncode)
            )
        )
    if installed_dir.is_dir():
        ready_marker.write_text(_BRIDGE_DIGEST, encoding="utf-8")


def _text_line_count(stream: typing.TextIO) -> int:
    line_count = 1
    for chunk in iter(partial(stream.read, _STREAM_CHUNK_SIZE), ""):
        if "\0" in chunk:
            raise ValueError("line selection requires an existing text file")
        line_count += chunk.count("\n")
    return line_count


def _check_line(line: int, line_count: int) -> None:
    if line > line_count:
        raise ValueError("line {0} is outside the file".format(line))


def _validate_local_file(
    path: Path, label: str, line: "typing.Optional[int]"
) -> "typing.Dict[str, object]":
    try:
        info = path.stat()
    except FileNotFoundError:
        raise ValueError("File does not exist: {0}".format(label))
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("only regular files can be displayed: {0}".format(label))
    if line is not None:
        with path.open(encoding="utf-8", errors="replace") as stream:
            _check_line(line, _text_line_count(stream))
    return {
        "scheme": "file",
        "path": os.path.abspath(str(path)),
        "label": label,
    }


def _resolve_git_document(
    opener: WorkspaceViewOpener,
    reference: "typing.Dict[str, object]",
    allow_missing: bool,
    line: "typing.Optional[int]",
) -> "typing.Dict[str, object]":
    source = reference["git_source"]
    entry = opener._git(
        ["ls-tree", "--full-tree", "-z", source["revision"], "--", source["path"]]
    )
    if not entry:
        if not allow_missing:
            raise ValueError("Git file does not exist: " + reference["label"])
        if line is not None:
            raise ValueError("line selection requires an existing text file")
        return {
            "scheme": "pycodex-git",
            "repo": source["repository"],
            "rev": source["revision"],
            "path": source["path"],
            "blob": "",
            "missing": True,
            "label": reference["label"],
        }
    metadata, _unused_path = entry.rstrip(b"\0").split(b"\t", 1)
    _unused_mode, object_type, object_id = metadata.split()
    if object_type != b"blob":
        raise ValueError("Git path must identify a file: " + source["path"])
    blob_id = object_id.decode("ascii")
    if line is not None:
        with tempfile.TemporaryFile() as errors, subprocess.Popen(
            ["git", "--no-pager", "cat-file", "blob", blob_id],
            cwd=source["repository"], stdout=subprocess.PIPE, stderr=errors,
            encoding="utf-8", errors="replace",
        ) as process:
            line_count = _text_line_count(process.stdout)
            if process.wait():
                errors.seek(0)
                detail = errors.read().decode("utf-8", errors="replace").strip()
                raise ValueError(detail or "Git command failed")
        _check_line(line, line_count)
    return {
        "scheme": "pycodex-git",
        "repo": source["repository"],
        "rev": source["revision"],
        "path": source["path"],
        "blob": blob_id,
        "label": reference["label"],
    }


def resolve_vscode_link(
    link_cwd: Path, href: object, src_path: object = None
) -> "typing.Dict[str, object]":
    opener = WorkspaceViewOpener(None, link_cwd)
    action, options = opener._parse_link(href)
    line_raw = options.get("line")
    line = None
    if line_raw is not None:
        if not re.fullmatch(r"[1-9][0-9]*", line_raw):
            raise ValueError("line must be a positive integer")
        line = int(line_raw)
    side = options.get("side", "right")

    if action == "file":
        path = options["path"]
        if src_path is not None:
            if not isinstance(src_path, str):
                raise ValueError("src_path must be a file path")
            path = Path(src_path).parent / path
        ref = opener._file_reference(path)
        doc = _validate_local_file(Path(ref["source"]["path"]), ref["label"], line)
        selection = {"document": 0, "line": line} if line is not None else None
        return {
            "kind": "file",
            "title": doc["label"],
            "documents": [doc],
            "selection": selection,
        }

    if action == "git-file":
        ref = opener._git_object_reference(options["target"])
        doc = _resolve_git_document(opener, ref, False, line)
        selection = {"document": 0, "line": line} if line is not None else None
        return {
            "kind": "file",
            "title": doc["label"],
            "documents": [doc],
            "selection": selection,
        }

    doc_index = 0 if side == "left" else 1
    left_line = line if line is not None and doc_index == 0 else None
    right_line = line if line is not None and doc_index == 1 else None

    if action == "diff":
        left_ref = opener._file_reference(options["left"])
        right_ref = opener._file_reference(options["right"])
        left_doc = _validate_local_file(
            Path(left_ref["source"]["path"]), left_ref["label"], left_line
        )
        right_doc = _validate_local_file(
            Path(right_ref["source"]["path"]), right_ref["label"], right_line
        )
    else:
        left_ref = opener._git_object_reference(options["left"])
        left_doc = _resolve_git_document(opener, left_ref, True, left_line)
        if "right" in options:
            right_ref = opener._git_object_reference(options["right"])
            right_doc = _resolve_git_document(opener, right_ref, True, right_line)
        else:
            source = left_ref["git_source"]
            wt_path = Path(source["repository"]) / source["path"]
            if wt_path.is_symlink():
                target = os.readlink(os.fsencode(wt_path)).decode("utf-8", errors="replace")
                if right_line is not None:
                    _check_line(right_line, _text_line_count(StringIO(target, newline=None)))
                right_doc = {
                    "scheme": "pycodex-git",
                    "repo": source["repository"],
                    "rev": "WORKTREE",
                    "path": source["path"],
                    "symlink_target": target,
                    "label": source["path"],
                }
            elif wt_path.exists():
                right_doc = _validate_local_file(wt_path, source["path"], right_line)
            else:
                if right_line is not None:
                    raise ValueError("line selection requires an existing text file")
                right_doc = {
                    "scheme": "pycodex-git",
                    "repo": source["repository"],
                    "rev": "WORKTREE",
                    "path": source["path"],
                    "blob": "",
                    "missing": True,
                    "label": source["path"],
                }
        if left_doc.get("missing") and right_doc.get("missing"):
            raise ValueError("file is absent on both sides of the comparison")

    selection = {"document": doc_index, "line": line} if line is not None else None
    title = "{0} <-> {1}".format(left_doc["label"], right_doc["label"])
    return {
        "kind": "diff",
        "title": title,
        "documents": [left_doc, right_doc],
        "selection": selection,
    }


def _strip_auth_cookie(cookie_header: str) -> str:
    kept = []
    for part in cookie_header.split(";"):
        stripped = part.strip()
        if not stripped:
            continue
        name = stripped.split("=", 1)[0].strip()
        if name == AUTH_COOKIE_NAME:
            continue
        kept.append(stripped)
    return "; ".join(kept)


def _accepts_gzip(accept_encoding: str) -> bool:
    if not accept_encoding:
        return False
    qualities = {}  # type: typing.Dict[str, float]
    for entry in accept_encoding.lower().split(","):
        parts = entry.strip().split(";")
        name = parts[0].strip()
        if not name:
            continue
        quality = 1.0
        valid = True
        for param in parts[1:]:
            key, _sep, value = param.strip().partition("=")
            if key.strip() != "q":
                continue
            try:
                quality = float(value.strip())
            except ValueError:
                valid = False
                break
            if quality < 0.0 or quality > 1.0:
                valid = False
                break
        if valid:
            qualities[name] = quality
    gzip_q = qualities.get("gzip", qualities.get("*", 0.0))
    return gzip_q > 0.0 and gzip_q >= qualities.get("identity", 0.0)


def _is_compressible_content_type(content_type: str) -> bool:
    media_type = content_type.split(";", 1)[0].strip().lower()
    if not media_type or media_type == "text/event-stream":
        return False
    if media_type.startswith("text/"):
        return True
    if media_type in _COMPRESSIBLE_MEDIA_TYPES:
        return True
    return media_type.endswith("+json") or media_type.endswith("+xml")


def _append_gzip_etag_suffix(etag_value: str) -> str:
    stripped = etag_value.strip()
    if stripped.endswith('"') and not stripped.endswith('-gzip"'):
        return stripped[:-1] + '-gzip"'
    return stripped


def _strip_gzip_etag_suffix(if_none_match: str) -> "typing.Tuple[str, bool]":
    parts = []
    changed = False
    for raw_item in if_none_match.split(","):
        item = raw_item.strip()
        if item.endswith('-gzip"'):
            parts.append(item[:-6] + '"')
            changed = True
        elif item:
            parts.append(item)
    return ", ".join(parts), changed


def _ensure_vary_accept_encoding(
    raw_headers: "typing.List[typing.Tuple[bytes, bytes]]",
) -> "typing.List[typing.Tuple[bytes, bytes]]":
    updated = []  # type: typing.List[typing.Tuple[bytes, bytes]]
    found_vary = False
    for name, value in raw_headers:
        if name.lower() == b"vary":
            found_vary = True
            text = value.decode("latin-1")
            tokens = [tok.strip().lower() for tok in text.split(",") if tok.strip()]
            if "*" not in tokens and "accept-encoding" not in tokens:
                text = (
                    (text.rstrip(", ") + ", Accept-Encoding")
                    if text.strip()
                    else "Accept-Encoding"
                )
                value = text.encode("latin-1")
        updated.append((name, value))
    if not found_vary:
        updated.append((b"vary", b"Accept-Encoding"))
    return updated


def _parse_payload_entries(
    raw_value: str,
) -> "typing.Optional[typing.List[typing.List[object]]]":
    try:
        parsed = json.loads(raw_value)
    except ValueError:
        return None
    if not isinstance(parsed, list):
        return None
    entries = []  # type: typing.List[typing.List[object]]
    for item in parsed:
        if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str):
            entries.append([item[0], item[1]])
    return entries


def extract_vscode_window_id(raw_query: str) -> "typing.Optional[str]":
    if not raw_query:
        return None
    window_id = None  # type: typing.Optional[str]
    for segment in raw_query.split("&"):
        if not segment:
            continue
        raw_key, sep, raw_val = segment.partition("=")
        if not sep or unquote_plus(raw_key) != "payload":
            continue
        entries = _parse_payload_entries(unquote_plus(raw_val))
        if entries is None:
            continue
        for key, val in entries:
            if key == "extensionEnvironment" and isinstance(val, str):
                try:
                    env_map = json.loads(val)
                except ValueError:
                    continue
                if isinstance(env_map, dict):
                    candidate = env_map.get("PYCODEX_VSCODE_WINDOW_ID")
                    if isinstance(candidate, str) and candidate:
                        window_id = candidate
                    else:
                        window_id = None
    return window_id


def inject_vscode_window_payload(raw_query: str, window_id: str) -> str:
    kept_segments = []  # type: typing.List[str]
    payload_entries = []  # type: typing.List[typing.List[object]]
    env_map = {}  # type: typing.Dict[str, object]
    if raw_query:
        for segment in raw_query.split("&"):
            if not segment:
                continue
            raw_key, sep, raw_val = segment.partition("=")
            if unquote_plus(raw_key) != "payload":
                kept_segments.append(segment)
                continue
            if not sep:
                continue
            entries = _parse_payload_entries(unquote_plus(raw_val))
            if entries is None:
                continue
            for key, val in entries:
                if key == "extensionEnvironment":
                    if isinstance(val, str):
                        try:
                            parsed_env = json.loads(val)
                        except ValueError:
                            parsed_env = None
                        if isinstance(parsed_env, dict):
                            env_map.update(parsed_env)
                else:
                    payload_entries.append([key, val])
    env_map["PYCODEX_VSCODE_WINDOW_ID"] = window_id
    payload_entries.append(
        ["extensionEnvironment", json.dumps(env_map, separators=(",", ":"))]
    )
    encoded_payload = quote(
        json.dumps(payload_entries, separators=(",", ":")), safe=""
    )
    kept_segments.append("payload=" + encoded_payload)
    return "&".join(kept_segments)


def strip_vscode_window_payload(raw_query: str) -> str:
    if not raw_query:
        return ""
    kept_segments = []  # type: typing.List[str]
    for segment in raw_query.split("&"):
        if not segment:
            continue
        raw_key, sep, raw_val = segment.partition("=")
        if unquote_plus(raw_key) != "payload":
            kept_segments.append(segment)
            continue
        if not sep:
            continue
        entries = _parse_payload_entries(unquote_plus(raw_val))
        if entries is None:
            continue
        cleaned_entries = []  # type: typing.List[typing.List[object]]
        for key, val in entries:
            if key != "extensionEnvironment":
                cleaned_entries.append([key, val])
                continue
            if not isinstance(val, str):
                continue
            try:
                parsed_env = json.loads(val)
            except ValueError:
                continue
            if isinstance(parsed_env, dict):
                parsed_env.pop("PYCODEX_VSCODE_WINDOW_ID", None)
                if parsed_env:
                    cleaned_entries.append(
                        [
                            "extensionEnvironment",
                            json.dumps(parsed_env, separators=(",", ":")),
                        ]
                    )
        if cleaned_entries:
            kept_segments.append(
                "payload="
                + quote(json.dumps(cleaned_entries, separators=(",", ":")), safe="")
            )
    return "&".join(kept_segments)


def _upstream_target(scope: dict, actual_prefix: str) -> str:
    raw_path = scope.get("raw_path") or scope["path"].encode("utf-8")
    if b"?" in raw_path:
        raw_path = raw_path.split(b"?", 1)[0]
    root_path = scope.get("root_path", "").rstrip("/")
    candidates = []
    if root_path:
        candidates.append((root_path + actual_prefix + "/vscode").encode("utf-8"))
    candidates.append((actual_prefix + "/vscode").encode("utf-8"))
    suffix = None
    for prefix_bytes in candidates:
        if raw_path == prefix_bytes:
            suffix = b"/"
            break
        if raw_path.startswith(prefix_bytes + b"/"):
            suffix = raw_path[len(prefix_bytes) :]
            break
    if suffix is None:
        path_text = scope["path"]
        short_prefix = actual_prefix + "/vscode"
        if path_text.startswith(short_prefix + "/"):
            suffix = path_text[len(short_prefix) :].encode("utf-8")
        else:
            suffix = b"/"
    if not suffix.startswith(b"/"):
        suffix = b"/" + suffix
    query_string = scope.get("query_string", b"")
    if query_string:
        suffix = suffix + b"?" + query_string
    return suffix.decode("latin-1")


def forwarded_vscode_prefix(
    headers_get: "typing.Callable[[str, str], str]", root_path: str, actual_prefix: str
) -> str:
    base = (
        root_path.rstrip("/")
        or headers_get("x-forwarded-prefix", "").split(",", 1)[0].strip().rstrip("/")
    )
    return base + actual_prefix + "/vscode"


def _normalized_authority(netloc: str, scheme: str) -> str:
    authority = netloc.strip().lower()
    default_port = ":443" if scheme == "https" else ":80"
    if authority.endswith(default_port):
        authority = authority[: -len(default_port)]
    return authority


def websocket_origin_allowed(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    if origin is None:
        return True
    allowed_schemes = {"https" if websocket.url.scheme == "wss" else "http"}
    forwarded_proto = (
        websocket.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip().lower()
    )
    if forwarded_proto in ("http", "https"):
        allowed_schemes.add(forwarded_proto)
    allowed_origins = set()
    for scheme in allowed_schemes:
        base_auth = _normalized_authority(websocket.url.netloc, scheme)
        if base_auth:
            allowed_origins.add(scheme + "://" + base_auth)
        forwarded_host = (
            websocket.headers.get("x-forwarded-host", "").split(",", 1)[0].strip()
        )
        if forwarded_host:
            fwd_auth = _normalized_authority(forwarded_host, scheme)
            if fwd_auth:
                allowed_origins.add(scheme + "://" + fwd_auth)
    return origin.strip().lower() in allowed_origins


async def _safe_close_websocket(websocket: WebSocket, code: int = 1000) -> None:
    if (
        websocket.client_state != WebSocketState.DISCONNECTED
        and websocket.application_state in (WebSocketState.CONNECTING, WebSocketState.CONNECTED)
    ):
        try:
            await websocket.close(code=code)
        except Exception:
            pass


class _BridgeClient:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer = writer
        self.window_id = None  # type: typing.Optional[str]
        self.pending = {}  # type: typing.Dict[str, asyncio.Future]


class VsCodeRuntime:
    """Lazily start and proxy a single host VS Code Web Server instance."""

    def __init__(
        self,
        startup_timeout: float = 20.0,
        bridge_timeout: float = 30.0,
    ) -> None:
        self._startup_timeout = startup_timeout
        self._bridge_timeout = bridge_timeout
        self._loop = None  # type: typing.Optional[asyncio.AbstractEventLoop]
        self._lock = None  # type: typing.Optional[asyncio.Lock]
        self._bridge_cond = None  # type: typing.Optional[asyncio.Condition]
        self._temp_dir = None  # type: typing.Optional[Path]
        self._http_sock = None  # type: typing.Optional[Path]
        self._bridge_sock = None  # type: typing.Optional[Path]
        self._bridge_token = ""
        self._bridge_server = None  # type: typing.Optional[asyncio.AbstractServer]
        self._clients = set()  # type: typing.Set[_BridgeClient]
        self._process = None  # type: typing.Optional[subprocess.Popen]
        self._log_file = None  # type: typing.Optional[typing.IO[bytes]]
        self._closed = False

    def _sync_loop_primitives(self) -> "typing.Tuple[asyncio.Lock, asyncio.Condition]":
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self._lock = asyncio.Lock()
            self._bridge_cond = asyncio.Condition()
        assert self._lock is not None and self._bridge_cond is not None
        return self._lock, self._bridge_cond

    def _supersede_client(self, client: _BridgeClient) -> None:
        client.window_id = None
        try:
            client.writer.write(b'{"type":"supersede"}\n')
        except Exception:
            pass

    def supersede_window(self, window_id: str) -> None:
        for client in list(self._clients):
            if client.window_id == window_id:
                self._supersede_client(client)

    async def ensure_started(self) -> Path:
        lock, _cond = self._sync_loop_primitives()
        async with lock:
            if self._closed:
                raise RuntimeError("VS Code runtime is closed")
            if (
                self._process is not None
                and self._process.poll() is None
                and self._http_sock is not None
                and self._http_sock.exists()
            ):
                return self._http_sock
            await self._cleanup_locked()
            code_server = find_code_server()
            temp_dir = Path(tempfile.mkdtemp(prefix="pycodex-vsc-"))
            http_sock = temp_dir / "vscode.sock"
            bridge_sock = temp_dir / "bridge.sock"
            bridge_token = secrets.token_hex(24)
            server_data_dir = Path.home() / ".vscode-server"
            extensions_dir = server_data_dir / "extensions"
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    partial(
                        _ensure_bridge_extension,
                        code_server,
                        server_data_dir,
                        extensions_dir,
                        temp_dir,
                    ),
                )
                bridge_server = await asyncio.start_unix_server(
                    self._handle_bridge_client, path=str(bridge_sock)
                )
                os.chmod(str(bridge_sock), 0o600)
                log_path = temp_dir / "server.log"
                log_handle = log_path.open("wb")
                env = dict(os.environ)
                env["PYCODEX_VSCODE_BRIDGE_SOCK"] = str(bridge_sock)
                env["PYCODEX_VSCODE_BRIDGE_TOKEN"] = bridge_token
                process = subprocess.Popen(
                    [
                        str(code_server),
                        "--socket-path",
                        str(http_sock),
                        "--without-connection-token",
                        "--accept-server-license-terms",
                        "--disable-workspace-trust",
                        "--telemetry-level",
                        "off",
                        "--server-data-dir",
                        str(server_data_dir),
                        "--extensions-dir",
                        str(extensions_dir),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                self._temp_dir = temp_dir
                self._http_sock = http_sock
                self._bridge_sock = bridge_sock
                self._bridge_token = bridge_token
                self._bridge_server = bridge_server
                self._process = process
                self._log_file = log_handle
                await self._wait_until_listening(http_sock, process, log_path)
                return http_sock
            except BaseException:
                await self._cleanup_locked()
                if temp_dir.exists():
                    shutil.rmtree(str(temp_dir), ignore_errors=True)
                raise

    async def _wait_until_listening(
        self, http_sock: Path, process: subprocess.Popen, log_path: Path
    ) -> None:
        deadline = time.monotonic() + self._startup_timeout
        while True:
            if process.poll() is not None:
                tail = ""
                if log_path.is_file():
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-1000:].strip()
                raise RuntimeError(
                    "VS Code Web Server exited unexpectedly (code {0}){1}".format(
                        process.returncode, ": " + tail if tail else ""
                    )
                )
            if http_sock.exists():
                try:
                    _reader, writer = await asyncio.open_unix_connection(str(http_sock))
                    writer.close()
                    if hasattr(writer, "wait_closed"):
                        await writer.wait_closed()
                    return
                except OSError:
                    pass
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for VS Code Web Server socket")
            await asyncio.sleep(0.015)

    async def _handle_bridge_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        _lock, cond = self._sync_loop_primitives()
        client = _BridgeClient(writer)
        self._clients.add(client)
        welcomed = False
        try:
            while True:
                try:
                    line = await reader.readline()
                except (ConnectionError, OSError):
                    break
                if not line:
                    break
                try:
                    message = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    continue
                if not isinstance(message, dict):
                    continue
                msg_type = message.get("type")
                if not welcomed:
                    welcomed = True
                    window_id = message.get("window_id")
                    if msg_type != "hello" or not isinstance(window_id, str) or not window_id:
                        break
                    token = str(message.get("token") or "")
                    if not self._bridge_token or not secrets.compare_digest(
                        token, self._bridge_token
                    ):
                        break
                    self.supersede_window(window_id)
                    client.window_id = window_id
                    async with cond:
                        cond.notify_all()
                elif client.window_id is not None and msg_type == "ack":
                    req_id = str(message.get("id") or "")
                    future = client.pending.pop(req_id, None)
                    if future is not None and not future.done():
                        future.set_result(message)
        finally:
            self._clients.discard(client)
            for future in list(client.pending.values()):
                if not future.done():
                    future.set_exception(RuntimeError("VS Code bridge disconnected"))
            client.pending.clear()
            writer.close()
            if hasattr(writer, "wait_closed"):
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

    async def open_link(self, window_id: str, spec: "typing.Dict[str, object]") -> None:
        await self.ensure_started()
        _lock, cond = self._sync_loop_primitives()

        def window_client():
            matches = [client for client in self._clients if client.window_id == window_id]
            if len(matches) > 1:
                raise RuntimeError("Multiple VS Code windows share this ID. Reload the workspace page.")
            return matches[0] if matches else None

        async with cond:
            try:
                client = await asyncio.wait_for(cond.wait_for(window_client), self._bridge_timeout)
            except asyncio.TimeoutError:
                raise RuntimeError("VS Code window is not ready")

        req_id = secrets.token_hex(8)
        future = asyncio.get_running_loop().create_future()
        client.pending[req_id] = future
        try:
            payload = json.dumps({"type": "open", "id": req_id, "spec": spec}) + "\n"
            client.writer.write(payload.encode("utf-8"))
            await client.writer.drain()
            ack = await asyncio.wait_for(future, timeout=10.0)
            if not ack.get("ok"):
                raise RuntimeError(str(ack.get("error") or "VS Code failed to open link"))
        except asyncio.TimeoutError:
            raise RuntimeError("Timed out waiting for VS Code to open the link")
        finally:
            client.pending.pop(req_id, None)

    async def proxy_http(
        self,
        request: Request,
        actual_prefix: str,
        upstream_path: "typing.Optional[str]" = None,
    ) -> Response:
        try:
            http_sock = await self.ensure_started()
        except (FileNotFoundError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        if upstream_path is not None:
            query_bytes = request.scope.get("query_string", b"")
            target = upstream_path + (
                ("?" + query_bytes.decode("latin-1")) if query_bytes else ""
            )
        else:
            target = _upstream_target(request.scope, actual_prefix)
        forwarded_prefix = forwarded_vscode_prefix(
            request.headers.get, request.scope.get("root_path", ""), actual_prefix
        )
        body = await request.body()
        host = request.headers.get("host") or request.url.netloc or "localhost"
        client_accepts_gzip = _accepts_gzip(request.headers.get("accept-encoding", ""))
        request_has_range = False
        client_sent_gzip_etag = False
        out_headers = []
        has_content_length = False
        for raw_name, raw_val in request.scope["headers"]:
            name_str = raw_name.decode("latin-1")
            lower = name_str.lower()
            if lower in _HOP_BY_HOP_HEADERS or lower in (
                "host",
                "x-forwarded-prefix",
                "x-forwarded-proto",
            ):
                continue
            val_str = raw_val.decode("latin-1")
            if lower == "cookie":
                val_str = _strip_auth_cookie(val_str)
                if not val_str:
                    continue
            elif lower == "if-none-match" and client_accepts_gzip:
                val_str, had_gzip = _strip_gzip_etag_suffix(val_str)
                if had_gzip:
                    client_sent_gzip_etag = True
            elif lower == "range":
                request_has_range = True
            if lower == "content-length":
                has_content_length = True
            out_headers.append((name_str, val_str))
        out_headers.append(("Host", host))
        out_headers.append(("Connection", "close"))
        out_headers.append(("X-Forwarded-Prefix", forwarded_prefix))
        out_headers.append(("X-Forwarded-Proto", request.url.scheme))
        if body and not has_content_length:
            out_headers.append(("Content-Length", str(len(body))))

        reader, writer = await asyncio.open_unix_connection(
            str(http_sock), limit=_UNIX_READER_LIMIT
        )
        closed = False

        async def close_upstream() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            writer.close()
            if hasattr(writer, "wait_closed"):
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

        try:
            head_lines = ["{0} {1} HTTP/1.1\r\n".format(request.method, target)]
            for h_name, h_val in out_headers:
                head_lines.append("{0}: {1}\r\n".format(h_name, h_val))
            head_lines.append("\r\n")
            writer.write("".join(head_lines).encode("latin-1"))
            if body:
                writer.write(body)
            await writer.drain()

            status_line = await reader.readline()
            if not status_line:
                await close_upstream()
                raise HTTPException(status_code=502, detail="Empty response from VS Code server")
            parts = status_line.decode("latin-1").strip().split(None, 2)
            status_code = int(parts[1])

            parsed_headers = []  # type: typing.List[typing.Tuple[str, str]]
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                if line[:1] in (b" ", b"\t") and parsed_headers:
                    prev_name, prev_val = parsed_headers[-1]
                    parsed_headers[-1] = (
                        prev_name,
                        prev_val + " " + line.decode("latin-1").strip(),
                    )
                    continue
                name_bytes, _sep, val_bytes = line.partition(b":")
                parsed_headers.append(
                    (
                        name_bytes.decode("latin-1").strip(),
                        val_bytes.decode("latin-1").strip(),
                    )
                )

            transfer_encoding = ""
            content_length = None  # type: typing.Optional[int]
            content_type = ""
            has_content_encoding = False
            response_raw_headers = []  # type: typing.List[typing.Tuple[bytes, bytes]]
            for h_name, h_val in parsed_headers:
                lower = h_name.lower()
                if lower == "transfer-encoding":
                    transfer_encoding = h_val
                    continue
                if lower in _HOP_BY_HOP_HEADERS:
                    continue
                if lower == "content-length":
                    try:
                        content_length = int(h_val)
                    except ValueError:
                        content_length = None
                elif lower == "content-type":
                    content_type = h_val
                elif lower == "content-encoding":
                    has_content_encoding = True
                if lower == "service-worker-allowed":
                    h_val = forwarded_prefix + "/"
                elif lower == "set-cookie":
                    cookies = SimpleCookie()
                    cookies.load(h_val)
                    for cookie in cookies.values():
                        if not cookie["path"] or cookie["path"] == "/":
                            cookie["path"] = forwarded_prefix + "/"
                        response_raw_headers.append(
                            (b"set-cookie", cookie.OutputString().encode("latin-1"))
                        )
                    continue
                response_raw_headers.append(
                    (h_name.encode("latin-1"), h_val.encode("latin-1"))
                )
            is_chunked = "chunked" in transfer_encoding.lower()
            if is_chunked:
                content_length = None
                response_raw_headers = [
                    pair
                    for pair in response_raw_headers
                    if pair[0].lower() != b"content-length"
                ]

            if status_code == 304 and client_sent_gzip_etag and not has_content_encoding:
                response_raw_headers = [
                    (
                        name,
                        _append_gzip_etag_suffix(val.decode("latin-1")).encode("latin-1")
                        if name.lower() == b"etag"
                        else val,
                    )
                    for name, val in response_raw_headers
                    if name.lower() != b"content-length"
                ]
                response_raw_headers = _ensure_vary_accept_encoding(response_raw_headers)
                response_raw_headers.append((b"content-encoding", b"gzip"))

            if (
                request.method == "HEAD"
                or status_code in (101, 204, 304)
                or 100 <= status_code < 200
            ):
                await close_upstream()
                empty = Response(b"", status_code=status_code)
                empty.raw_headers = response_raw_headers
                return empty

            negotiable_gzip = (
                request.method == "GET"
                and status_code == 200
                and not has_content_encoding
                and not request_has_range
                and (content_length is None or content_length >= _GZIP_MIN_LENGTH)
                and _is_compressible_content_type(content_type)
            )
            use_gzip = negotiable_gzip and client_accepts_gzip
            if negotiable_gzip:
                response_raw_headers = _ensure_vary_accept_encoding(response_raw_headers)
            if use_gzip:
                response_raw_headers = [
                    (
                        name,
                        _append_gzip_etag_suffix(val.decode("latin-1")).encode("latin-1")
                        if name.lower() == b"etag"
                        else val,
                    )
                    for name, val in response_raw_headers
                    if name.lower() != b"content-length"
                ]
                response_raw_headers.append((b"content-encoding", b"gzip"))

            is_event_stream = (
                content_type.split(";", 1)[0].strip().lower() == "text/event-stream"
            )

            async def upstream_chunks() -> "typing.AsyncIterator[bytes]":
                if is_chunked:
                    while True:
                        size_line = await reader.readline()
                        if not size_line:
                            break
                        chunk_size = int(size_line.split(b";", 1)[0].strip(), 16)
                        if chunk_size == 0:
                            while True:
                                trailer = await reader.readline()
                                if trailer in (b"\r\n", b"\n", b""):
                                    break
                            break
                        chunk = await reader.readexactly(chunk_size)
                        await reader.readexactly(2)
                        if chunk:
                            yield chunk
                elif content_length is not None:
                    remaining = content_length
                    while remaining > 0:
                        chunk = await reader.read(min(_STREAM_CHUNK_SIZE, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
                else:
                    while True:
                        chunk = await reader.read(_STREAM_CHUNK_SIZE)
                        if not chunk:
                            break
                        yield chunk

            async def body_stream() -> "typing.AsyncIterator[bytes]":
                compressor = (
                    zlib.compressobj(_GZIP_COMPRESS_LEVEL, zlib.DEFLATED, 31)
                    if use_gzip
                    else None
                )
                coalesce = use_gzip or (content_length is not None and not is_event_stream)
                out_buf = bytearray()
                try:
                    async for chunk in upstream_chunks():
                        data = compressor.compress(chunk) if compressor is not None else chunk
                        if not data:
                            continue
                        if not coalesce:
                            yield data
                            continue
                        out_buf.extend(data)
                        while len(out_buf) >= _STREAM_CHUNK_SIZE:
                            yield bytes(out_buf[:_STREAM_CHUNK_SIZE])
                            del out_buf[:_STREAM_CHUNK_SIZE]
                    if compressor is not None:
                        tail = compressor.flush()
                        if tail:
                            out_buf.extend(tail)
                    while len(out_buf) >= _STREAM_CHUNK_SIZE:
                        yield bytes(out_buf[:_STREAM_CHUNK_SIZE])
                        del out_buf[:_STREAM_CHUNK_SIZE]
                    if out_buf:
                        yield bytes(out_buf)
                finally:
                    await close_upstream()

            streaming = StreamingResponse(body_stream(), status_code=status_code)
            streaming.raw_headers = response_raw_headers
            return streaming
        except BaseException:
            await close_upstream()
            raise

    async def proxy_websocket(self, websocket: WebSocket, actual_prefix: str) -> None:
        from websockets.exceptions import ConnectionClosed
        from websockets.legacy.client import unix_connect

        try:
            http_sock = await self.ensure_started()
        except (FileNotFoundError, RuntimeError, OSError, subprocess.TimeoutExpired):
            await _safe_close_websocket(websocket, code=1011)
            return

        target = _upstream_target(websocket.scope, actual_prefix)
        forwarded_prefix = forwarded_vscode_prefix(
            websocket.headers.get, websocket.scope.get("root_path", ""), actual_prefix
        )
        host = websocket.headers.get("host") or websocket.url.netloc or "localhost"
        extra_headers = [
            ("X-Forwarded-Prefix", forwarded_prefix),
            ("X-Forwarded-Proto", "https" if websocket.url.scheme == "wss" else "http"),
        ]
        for raw_name, raw_val in websocket.scope["headers"]:
            name_str = raw_name.decode("latin-1")
            lower = name_str.lower()
            if lower in _HOP_BY_HOP_HEADERS or lower in (
                "host",
                "origin",
                "sec-websocket-key",
                "sec-websocket-version",
                "sec-websocket-extensions",
                "sec-websocket-protocol",
                "x-forwarded-prefix",
                "x-forwarded-proto",
            ):
                continue
            val_str = raw_val.decode("latin-1")
            if lower == "cookie":
                val_str = _strip_auth_cookie(val_str)
                if not val_str:
                    continue
            extra_headers.append((name_str, val_str))

        subprotocols = list(websocket.scope.get("subprotocols") or ())
        ws_uri = "ws://{0}{1}".format(host, target)
        upstream = None
        tasks = []  # type: typing.List[asyncio.Task]
        try:
            upstream = await unix_connect(
                str(http_sock),
                uri=ws_uri,
                extra_headers=extra_headers,
                origin="http://{0}".format(host),
                subprotocols=subprotocols or None,
                max_size=None,
                ping_interval=None,
                close_timeout=1.0,
                compression=None,
            )
            await websocket.accept(subprotocol=upstream.subprotocol)

            async def client_to_upstream() -> None:
                try:
                    while True:
                        message = await websocket.receive()
                        msg_type = message.get("type")
                        if msg_type == "websocket.disconnect":
                            break
                        if message.get("bytes") is not None:
                            await upstream.send(message["bytes"])
                        elif message.get("text") is not None:
                            await upstream.send(message["text"])
                except (WebSocketDisconnect, ConnectionClosed):
                    pass
                finally:
                    with anyio.CancelScope(shield=True):
                        await upstream.close()

            async def upstream_to_client() -> None:
                try:
                    async for payload in upstream:
                        if isinstance(payload, bytes):
                            await websocket.send_bytes(payload)
                        else:
                            await websocket.send_text(payload)
                except (ConnectionClosed, WebSocketDisconnect, RuntimeError):
                    pass
                finally:
                    with anyio.CancelScope(shield=True):
                        await _safe_close_websocket(websocket)

            tasks = [
                asyncio.create_task(client_to_upstream()),
                asyncio.create_task(upstream_to_client()),
            ]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            raise
        except Exception:
            with anyio.CancelScope(shield=True):
                await _safe_close_websocket(websocket, code=1011)
        finally:
            with anyio.CancelScope(shield=True):
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                if upstream is not None:
                    try:
                        await upstream.close()
                    except Exception:
                        pass

    async def close(self) -> None:
        lock, _cond = self._sync_loop_primitives()
        async with lock:
            self._closed = True
            await self._cleanup_locked()

    async def _cleanup_locked(self) -> None:
        for client in list(self._clients):
            client.writer.close()
        self._clients.clear()
        if self._bridge_server is not None:
            self._bridge_server.close()
            await self._bridge_server.wait_closed()
            self._bridge_server = None
        process = self._process
        self._process = None
        if process is not None:
            await asyncio.get_running_loop().run_in_executor(
                None, partial(_terminate_process_group, process)
            )
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        if self._temp_dir is not None:
            temp_dir = self._temp_dir
            self._temp_dir = None
            self._http_sock = None
            self._bridge_sock = None
            shutil.rmtree(str(temp_dir), ignore_errors=True)
        self._bridge_token = ""


def _terminate_process_group(process: subprocess.Popen) -> None:
    import signal

    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            try:
                process.terminate()
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                try:
                    process.kill()
                except OSError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
