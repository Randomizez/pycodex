from __future__ import annotations

import os
from pathlib import Path

import pytest

from pycodex.portable import (
    DEFAULT_ENTRY_CONFIG,
    RemoteStorageError,
    bootstrap_called_home,
    resolve_put_source_dir,
    upload_codex_home,
)
from pycodex.portable_server import CodexStorageServer


def _write_codex_home(
    root: Path,
    *,
    with_model_instructions: bool = False,
) -> None:
    (root / "skills" / "demo").mkdir(parents=True)
    (root / "skills" / "demo" / "SKILL.md").write_text("# Demo\n\nStored skill.\n")
    (root / "AGENTS.md").write_text("stored agents instructions\n")
    (root / ".env").write_text('PORTABLE_API_KEY="from-storage-dotenv"\n')

    lines = [
        'model = "demo-model"',
        'model_provider = "demo"',
    ]
    if with_model_instructions:
        (root / "instructions").mkdir()
        (root / "instructions" / "base.md").write_text("custom base instructions")
        lines.append('model_instructions_file = "instructions/base.md"')
    lines.extend(
        [
            '[model_providers.demo]',
            'base_url = "https://example.com/v1"',
            'env_key = "PORTABLE_API_KEY"',
        ]
    )
    (root / DEFAULT_ENTRY_CONFIG).write_text("\n".join(lines))


def test_upload_codex_home_returns_call_spec_and_whitelist_logs(tmp_path) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    _write_codex_home(codex_home)
    (codex_home / "sessions" / "2026" / "04").mkdir(parents=True)
    (codex_home / "sessions" / "2026" / "04" / "rollout.jsonl").write_text("history")
    (codex_home / ".tmp" / "ignored.txt").parent.mkdir(parents=True)
    (codex_home / ".tmp" / "ignored.txt").write_text("ignore me")
    server = CodexStorageServer(tmp_path / "storage-server", port=0)
    server.start()
    log_lines: list[str] = []
    try:
        call_spec = upload_codex_home(
            f"{codex_home}@{server.server_address}",
            event_handler=log_lines.append,
        )
    finally:
        server.stop()

    assert call_spec.endswith(f"@{server.server_address}")
    assert "-" in call_spec.split("@", 1)[0]
    assert "[put] mode: whitelist" in log_lines
    assert "[put] file: config.toml" in log_lines
    assert "[put] file: AGENTS.md" in log_lines
    assert "[put] file: .tmp/ignored.txt" not in log_lines
    assert "[put] file: sessions/2026/04/rollout.jsonl" not in log_lines


def test_upload_codex_home_stores_ciphertext_not_plain_zip(tmp_path) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    _write_codex_home(codex_home)
    server_root = tmp_path / "storage-server"
    server = CodexStorageServer(server_root, port=0)
    server.start()
    try:
        call_spec = upload_codex_home(f"{codex_home}@{server.server_address}")
    finally:
        server.stop()

    call_id = call_spec.split("@", 1)[0].rsplit("-", 1)[1]
    stored_bytes = (server_root / "objects" / f"{call_id}.bin").read_bytes()
    assert not stored_bytes.startswith(b"PK")


def test_upload_codex_home_includes_relative_model_instructions_file(tmp_path) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    _write_codex_home(codex_home, with_model_instructions=True)
    server = CodexStorageServer(tmp_path / "storage-server", port=0)
    server.start()
    log_lines: list[str] = []
    try:
        upload_codex_home(
            f"{codex_home}@{server.server_address}",
            event_handler=log_lines.append,
        )
    finally:
        server.stop()

    assert "[put] file: instructions/base.md" in log_lines


def test_upload_codex_home_checks_server_before_packing(tmp_path, monkeypatch) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    _write_codex_home(codex_home)
    monkeypatch.setattr(
        "pycodex.portable._build_bundle_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not pack before server preflight")
        ),
    )

    with pytest.raises(RemoteStorageError, match="storage server preflight failed"):
        upload_codex_home(f"{codex_home}@127.0.0.1:1")


def test_resolve_put_source_dir_defaults_to_home_dotcodex(tmp_path, monkeypatch) -> None:
    fake_home = tmp_path / "fake-home"
    codex_home = fake_home / ".codex"
    codex_home.mkdir(parents=True)
    _write_codex_home(codex_home)
    monkeypatch.setenv("HOME", str(fake_home))

    assert resolve_put_source_dir(None) == codex_home.resolve()


def test_bootstrap_called_home_downloads_and_reuses_cache(tmp_path, monkeypatch) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    _write_codex_home(codex_home)
    server = CodexStorageServer(tmp_path / "storage-server", port=0)
    server.start()
    monkeypatch.delenv("CODEX_HOME", raising=False)
    try:
        call_spec = upload_codex_home(f"{codex_home}@{server.server_address}")
        first_config = bootstrap_called_home(call_spec, storage_root=tmp_path / "storage-cache")
    finally:
        server.stop()

    second_config = bootstrap_called_home(call_spec, storage_root=tmp_path / "storage-cache")

    assert first_config == second_config
    assert first_config.is_file()
    assert (first_config.parent / "AGENTS.md").read_text() == "stored agents instructions\n"


def test_bootstrap_called_home_rejects_invalid_call_spec() -> None:
    with pytest.raises(RemoteStorageError, match="call spec"):
        bootstrap_called_home("not-a-valid-call-spec")


def test_bootstrap_called_home_rejects_wrong_secret(tmp_path) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    _write_codex_home(codex_home)
    server = CodexStorageServer(tmp_path / "storage-server", port=0)
    server.start()
    try:
        call_spec = upload_codex_home(f"{codex_home}@{server.server_address}")
        _, call_target = call_spec.split("-", 1)
        with pytest.raises(RemoteStorageError, match="invalid or bundle is corrupted"):
            bootstrap_called_home(f"deadbeefdeadbeefdeadbeefdeadbeef-{call_target}")
    finally:
        server.stop()
