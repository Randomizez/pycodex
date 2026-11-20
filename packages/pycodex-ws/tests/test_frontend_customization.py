from pathlib import Path

import pytest
from starlette.testclient import TestClient

from workspace_server import (
    WorkspaceDefinition,
    WorkspaceEntry,
    WorkspaceRegistry,
    WorkspaceSessionManager,
    create_app,
    create_multi_workspace_app,
)
from workspace_server import customization

from test_workspace_server import make_session


@pytest.fixture(params=[False, True], ids=["single", "multi"])
def workspace_setup(request, tmp_path):
    def create(password=None):
        prefix = "/w/test" if request.param else ""
        manager = WorkspaceSessionManager(
            lambda: make_session(cwd=tmp_path), tmp_path / "state"
        )
        if request.param:
            definition = WorkspaceDefinition("test", tmp_path / "state", tmp_path, False)
            app = create_multi_workspace_app(
                WorkspaceRegistry([WorkspaceEntry(definition, manager)]), password
            )
        else:
            app = create_app(manager, tmp_path / "state", password, tmp_path, False)
        return app, prefix

    return create


def test_optional_frontend_files_and_builtin_recovery(workspace_setup):
    app, prefix = workspace_setup()
    directory = customization.frontend_directory()
    with TestClient(app) as client:
        original = client.get(prefix + "/")
        assert original.status_code == 200
        assert not directory.exists()
        assert 'data-enable-board="false"' in original.text
        assert 'src="frontend/custom.js"' not in original.text
        assert 'href="frontend/custom.css"' not in original.text
        assert 'aria-label="Settings"' in original.text

        directory.mkdir(parents=True)
        (directory / "custom.css").write_text("body { color: red; }", encoding="utf-8")
        (directory / "custom.js").write_text("document.body.dataset.custom = 'yes';", encoding="utf-8")
        page = client.get(prefix + "/")
        assert page.headers["cache-control"] == "no-store"
        assert page.text.index('href="frontend/custom.css"') > page.text.index('</style>')
        assert page.text.index('src="frontend/custom.js"') > page.text.index('<script type="module">')
        assert client.get(prefix + "/?builtin=1").text == original.text

        template = directory / "workspace.html"
        template.write_text(
            "<HTML><HEAD><title>__WORKSPACE_TITLE__</title>"
            "<script>const headExample = '</head>';</script></HEAD>"
            "<BODY>Custom page ≤ __ENABLE_BOARD__ / __BOARD_URL__ / __CWD_LABEL__"
            "<!-- </body> --><script>const bodyExample = '</body>';</script></BODY></HTML>",
            encoding="utf-8",
        )
        page = client.get(prefix + "/")
        assert "Custom page ≤ false / about:blank" in page.text
        assert 'href="frontend/custom.css"' in page.text
        assert 'src="frontend/custom.js"' in page.text
        assert "<script>const headExample = '</head>';</script>" in page.text
        assert "<script>const bodyExample = '</body>';</script>" in page.text
        assert "<!-- </body> -->" in page.text
        assert "__CWD_LABEL__" not in page.text
        assert client.get(prefix + "/?builtin=1").text == original.text
        template.unlink()
        assert "Custom page" not in client.get(prefix + "/").text
        assert 'src="frontend/custom.js"' in client.get(prefix + "/").text


def test_unreadable_override_does_not_silently_fall_back(workspace_setup, monkeypatch):
    app, prefix = workspace_setup()
    directory = customization.frontend_directory()
    directory.mkdir(parents=True)
    override = directory / "workspace.html"
    override.write_text("custom page", encoding="utf-8")
    read_text = Path.read_text

    def deny_override(path, *args, **kwargs):
        if path == override:
            raise PermissionError("test: override cannot be read")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny_override)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get(prefix + "/").status_code == 500
        assert client.get(prefix + "/?builtin=1").status_code == 200


def test_custom_assets_are_confined_and_refreshed(workspace_setup, tmp_path):
    app, prefix = workspace_setup()
    directory = customization.frontend_directory()
    assets = directory / "assets"
    assets.mkdir(parents=True)
    (directory / "custom.css").write_text('@import "./assets/theme.css";', encoding="utf-8")
    (assets / "theme.css").write_text("body { color: red; }", encoding="utf-8")
    (assets / "part.mjs").write_text("export const enabled = true;", encoding="utf-8")
    (assets / "image.png").write_bytes(b"test image")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside frontend directory", encoding="utf-8")
    (directory / "escape.txt").symlink_to(outside)
    (directory.parent / "private.txt").write_text("outside", encoding="utf-8")
    (directory / "inside.css").symlink_to(assets / "theme.css")
    with TestClient(app) as client:
        for name, content_type in [
            ("custom.css", "text/css"),
            ("assets/theme.css", "text/css"),
            ("assets/part.mjs", "text/javascript"),
            ("assets/image.png", "image/png"),
            ("inside.css", "text/css"),
        ]:
            response = client.get(prefix + "/frontend/" + name)
            assert response.status_code == 200
            assert response.headers["content-type"].startswith(content_type)
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["x-content-type-options"] == "nosniff"
            assert client.head(prefix + "/frontend/" + name).content == b""
        for name in ("escape.txt", "%2e%2e/private.txt", "assets", "missing.css"):
            assert client.get(prefix + "/frontend/" + name).status_code == 404
        (assets / "theme.css").write_text("body { color: blue; }", encoding="utf-8")
        assert "blue" in client.get(prefix + "/frontend/assets/theme.css").text
        for asset in (
            "workspace.css", "workspace.mjs", "workspace_dom.mjs", "markdown.mjs",
            "conversation.mjs", "sessions_ui.mjs", "dialogs.mjs", "custom_frontend.mjs",
            "banners.mjs", "content_views.mjs", "content_views.css", "editor.mjs", "editor.css",
        ):
            response = client.get(prefix + "/assets/" + asset)
            assert response.status_code == 200, asset
            assert response.headers["content-type"].startswith(
                "text/css" if asset.endswith(".css") else "text/javascript"
            )
            assert response.content
        assert client.get(prefix + "/api/session").status_code == 200
        if prefix:
            assert client.get("/w/missing/frontend/custom.css").status_code == 404


def test_custom_files_require_login(workspace_setup):
    app, prefix = workspace_setup(password="test-password")
    directory = customization.frontend_directory()
    directory.mkdir(parents=True)
    (directory / "custom.js").write_text("console.log('custom');", encoding="utf-8")
    with TestClient(app) as client:
        assert client.get(prefix + "/frontend/custom.js", follow_redirects=False).status_code == 303
        recovery = client.get(prefix + "/?builtin=1", follow_redirects=False)
        assert recovery.status_code == 303
        assert "builtin%3D1" in recovery.headers["location"]
        assert client.post("/login", json={"password": "test-password"}).status_code == 200
        assert len(client.get(prefix + "/api/sessions").json()["sessions"]) == 1
        assert client.get(prefix + "/frontend/custom.js").status_code == 200
        assert client.get(prefix + "/?builtin=1").status_code == 200
