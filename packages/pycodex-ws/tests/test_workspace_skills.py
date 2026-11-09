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


@pytest.mark.parametrize("multi", [False, True])
def test_skill_catalog_is_authenticated_scoped_and_live(tmp_path, multi):
    home = tmp_path / "codex"
    path = home / "skills" / "work" / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text("---\nname: work-context\ndescription: Personal notes\n---\n")

    def no_session():
        raise AssertionError("Catalog must not create a model session")

    manager = WorkspaceSessionManager(no_session, codex_home=home)
    if multi:
        other = WorkspaceSessionManager(no_session, codex_home=tmp_path / "other-home")
        registry = WorkspaceRegistry(
            [
                WorkspaceEntry(
                    WorkspaceDefinition("first", tmp_path / "first", tmp_path), manager
                ),
                WorkspaceEntry(
                    WorkspaceDefinition("second", tmp_path / "second", tmp_path), other
                ),
            ]
        )
        app = create_multi_workspace_app(registry, password="password")
        prefix = "/w/first"
    else:
        app = create_app(
            manager, tmp_path / "state", password="password", work_dir=tmp_path
        )
        prefix = ""
    browser = TestClient(app)
    try:
        assert browser.get(prefix + "/api/skills").status_code == 401
        assert browser.post("/login", json={"password": "password"}).status_code == 200
        response = browser.get(prefix + "/api/skills")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == {
            "skills": [{"name": "work-context", "description": "Personal notes"}]
        }
        assert str(home) not in response.text
        if multi:
            assert browser.get("/w/second/api/skills").json() == {"skills": []}
            assert browser.get("/w/missing/api/skills").status_code == 404
        path.unlink()
        assert browser.get(prefix + "/api/skills").json() == {"skills": []}

    finally:
        browser.close()
