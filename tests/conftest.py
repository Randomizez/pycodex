import pytest


@pytest.fixture(autouse=True)
def isolate_session_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
