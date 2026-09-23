import os

import pytest

from tools import feishu_oauth


@pytest.mark.parametrize("existing_dotenv", [False, True])
def test_authorization_saves_only_refresh_token_file(
    tmp_path, monkeypatch, capsys, existing_dotenv
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FEISHU_REFRESH_TOKEN", "stale-refresh")
    dotenv_path = tmp_path / ".codex" / ".env"
    dotenv_content = "FEISHU_REFRESH_TOKEN=stale-refresh\nKEEP=value\n"
    if existing_dotenv:
        dotenv_path.parent.mkdir()
        dotenv_path.write_text(dotenv_content, encoding="utf-8")

    feishu_oauth.print_token_result(
        {
            "refresh_token": "authorized-refresh",
            "refresh_token_expires_in": 604800,
        }
    )

    token_path = tmp_path / ".codex" / ".feishu_refresh_token"
    assert token_path.read_text(encoding="utf-8") == "authorized-refresh\n"
    assert token_path.stat().st_mode & 0o777 == 0o600
    if existing_dotenv:
        assert dotenv_path.read_text(encoding="utf-8") == dotenv_content
    else:
        assert not dotenv_path.exists()
    assert os.environ["FEISHU_REFRESH_TOKEN"] == "stale-refresh"
    output = capsys.readouterr().out
    assert str(token_path) in output
    assert "authorized-refresh" not in output


def test_oauth_command_reads_app_credentials_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("FEISHU_APP_ID", raising=False)
    monkeypatch.delenv("FEISHU_APP_SECRET", raising=False)
    dotenv_path = tmp_path / ".codex" / ".env"
    dotenv_path.parent.mkdir()
    dotenv_content = "FEISHU_APP_ID=test-app\nFEISHU_APP_SECRET=test-secret\n"
    dotenv_path.write_text(dotenv_content, encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda prompt: "test-code")
    exchanges = []

    def exchange(api_base, app_id, app_secret, code, redirect_uri):
        exchanges.append((app_id, app_secret, code))
        return {"refresh_token": "authorized-refresh"}

    monkeypatch.setattr(feishu_oauth, "exchange_authorization_code", exchange)

    assert feishu_oauth.main([]) == 0
    assert exchanges == [("test-app", "test-secret", "test-code")]
    token_path = tmp_path / ".codex" / ".feishu_refresh_token"
    assert token_path.read_text(encoding="utf-8") == "authorized-refresh\n"
    assert dotenv_path.read_text(encoding="utf-8") == dotenv_content
