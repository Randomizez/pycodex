from __future__ import annotations

import argparse
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .portable import (
    DEFAULT_STORAGE_SERVER,
    HEALTHCHECK_PATH,
    STORAGE_API_PREFIX,
    _call_id_from_payload,
)


class CodexStorageServer:
    def __init__(
        self,
        root: str | Path,
        host: str = "127.0.0.1",
        port: int = 5577,
    ) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._objects_dir = self._root / "objects"
        self._objects_dir.mkdir(parents=True, exist_ok=True)
        self._server = ThreadingHTTPServer((host, port), self._build_handler())
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return str(self._server.server_address[0])

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def server_address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def base_url(self) -> str:
        return f"http://{self.server_address}{STORAGE_API_PREFIX}"

    @property
    def root(self) -> Path:
        return self._root

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="pycodex-storage-server",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _build_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == HEALTHCHECK_PATH:
                    self._send_json(200, {"ok": True})
                    return
                if not path.startswith(f"{STORAGE_API_PREFIX}/call/"):
                    self._send_json(404, {"error": "not found"})
                    return
                call_id = unquote(path[len(f"{STORAGE_API_PREFIX}/call/") :]).strip()
                if not call_id:
                    self._send_json(400, {"error": "missing call_id"})
                    return
                object_path = server._object_path(call_id)
                if not object_path.is_file():
                    self._send_json(404, {"error": "not found"})
                    return
                payload = object_path.read_bytes()
                print(
                    "[server] call: "
                    f"client={self.client_address[0]} call_id={call_id} path={object_path}",
                    flush=True,
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Pycodex-Sha256", hashlib.sha256(payload).hexdigest())
                self.send_header("X-Pycodex-Call-Id", call_id)
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path != f"{STORAGE_API_PREFIX}/put":
                    self._send_json(404, {"error": "not found"})
                    return
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                if content_length <= 0:
                    self._send_json(400, {"error": "empty body"})
                    return
                payload = self.rfile.read(content_length)
                sha256 = hashlib.sha256(payload).hexdigest()
                expected_sha256 = self.headers.get("X-Pycodex-Sha256", "").strip().lower()
                if expected_sha256 and expected_sha256 != sha256:
                    self._send_json(400, {"error": "checksum mismatch"})
                    return
                call_id = _call_id_from_payload(payload)
                object_path = server._object_path(call_id)
                if not object_path.is_file():
                    object_path.write_bytes(payload)
                    status = "stored"
                else:
                    status = "reused"
                print(
                    "[server] put: "
                    f"client={self.client_address[0]} "
                    f"call_id={call_id} status={status} path={object_path}",
                    flush=True,
                )
                host_header = self.headers.get("Host", server.server_address).strip() or server.server_address
                self._send_json(
                    200,
                    {
                        "call_id": call_id,
                        "call": f"{call_id}@{host_header}",
                    },
                )

            def log_message(self, _format: str, *_args) -> None:
                return

            def _send_json(self, status: int, payload: dict[str, object]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def _object_path(self, call_id: str) -> Path:
        return self._objects_dir / f"{call_id}.bin"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pycodex.portable_server",
        description="Run a pycodex remote storage service for --put/--call testing.",
    )
    parser.add_argument(
        "--root",
        default=str(Path(".tmp") / "pycodex_storage"),
        help="Directory used to store uploaded encrypted bundles.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_STORAGE_SERVER.split(":", 1)[0],
        help="Host interface to bind.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(DEFAULT_STORAGE_SERVER.split(":", 1)[1]),
        help="Port to bind.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    server = CodexStorageServer(args.root, host=args.host, port=args.port)
    server.start()
    print(f"storage server listening on {server.base_url}", flush=True)
    print(f"storage root: {server.root}", flush=True)
    print(f"put current home: pycodex --put @{server.server_address}", flush=True)
    print(
        f"put custom home: pycodex --put /data/.codex/@{server.server_address}",
        flush=True,
    )
    try:
        if server._thread is not None:
            server._thread.join()
    except KeyboardInterrupt:
        return 130
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
