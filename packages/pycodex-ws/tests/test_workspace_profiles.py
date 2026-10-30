import threading

import pytest
from starlette.testclient import TestClient

from pycodex import AssistantMessage, ModelResponse
from workspace_server import (
    WorkspaceDefinition,
    WorkspaceEntry,
    WorkspaceRegistry,
    WorkspaceSessionManager,
    WorkspaceStateStore,
    build_parser,
    create_app,
    create_multi_workspace_app,
)
from workspace_server import app as app_module
from test_workspace_server import make_session, wait_snapshot


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("default_profile", [None, "fast"])
def test_profile_creation_and_restore(tmp_path, monkeypatch, multi, default_profile):
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "base-model"\n'
        'model_reasoning_effort = "low"\n'
        'base_instructions = "Base instructions."\n'
        '[profiles.fast]\n'
        'model = "fast-model"\n'
        'model_reasoning_effort = "medium"\n'
        'base_instructions = "Fast instructions."\n'
        '[profiles.review]\n'
        'model = "review-model"\n'
        'model_provider = "review-provider"\n'
        'model_reasoning_effort = "high"\n'
        'base_instructions = "Review instructions."\n'
        'model_auto_compact_token_limit = 40000\n'
        '[model_providers.review-provider]\n'
        'base_url = "http://review.invalid/v1"\n'
        'env_key = "TEST_REVIEW_KEY"\n',
        encoding="utf-8",
    )
    definition = WorkspaceDefinition("test", tmp_path / "state", tmp_path, False)
    store = WorkspaceStateStore(tmp_path / "state.pycodex-ws.json")
    agents = []
    prompts = []
    original_build_agent = app_module.build_agent

    def build_agent(*args, **kwargs):
        assert threading.current_thread() is not threading.main_thread()
        agent = original_build_agent(*args, **kwargs)
        agents.append(agent)
        return agent

    async def complete(client, prompt, event_handler=None):
        prompts.append(prompt)
        return ModelResponse([AssistantMessage("done")])

    monkeypatch.setattr(app_module, "build_agent", build_agent)
    monkeypatch.setattr("pycodex.model.ResponsesModelClient.complete", complete)

    def build_app(profile):
        argv = ["--config", str(config), "--toolset"]
        if profile is not None:
            argv.extend(["--profile", profile])
        entry = app_module._build_workspace_entry(
            definition, build_parser().parse_args(argv)
        )
        if multi:
            return create_multi_workspace_app(WorkspaceRegistry([entry]))
        return create_app(entry.manager, definition.state_path)

    prefix = "/w/test" if multi else ""
    expected_default_model = "fast-model" if default_profile else "base-model"
    with TestClient(build_app(default_profile)) as browser:
        catalog = browser.get(prefix + "/api/sessions").json()
        assert catalog["profiles"] == ["fast", "review"]
        assert catalog["default_profile"] == default_profile
        original = catalog["sessions"][0]
        response = browser.post(prefix + "/api/sessions", json={
            "profile": "review", "title": "Review", "prompt": "Please review",
        })
        assert response.status_code == 200
        selected = response.json()["session_id"]
        assert response.json()["sessions"][-1]["profile"] == "review"
        state = wait_snapshot(
            browser, lambda item: item["turns"] and item["turns"][-1]["response"] == "done",
            prefix + "/api/session?session_id=" + selected,
        )
        assert state["model"] == "review-model:high"
        assert agents[0].model_name == expected_default_model
        assert agents[1].model_client.endpoint == "http://review.invalid/v1"
        resources = browser.get(prefix + "/api/sessions/" + selected + "/state").json()
        assert resources["state"]["auto_compact_token_limit"] == 40000
        assert prompts[0].base_instructions == "Review instructions."
        assert agents[0].context_manager is not agents[1].context_manager
        assert not agents[0].history
        assert browser.get(prefix + "/api/session", params={
            "session_id": original["id"],
        }).json()["snapshot"]["model"].startswith(expected_default_model + ":")
        # Both an omitted profile and null keep the server default.
        for payload in ({}, {"profile": None}):
            created = browser.post(prefix + "/api/sessions", json=payload).json()
            assert created["sessions"][-1]["profile"] == default_profile
        assert [tab["profile"] for tab in store.load_tabs()] == [
            default_profile, "review", default_profile, default_profile,
        ]
        # Fork events must retain the selected profile, even before a new recording.
        assert browser.post(prefix + "/api/session/message", json={
            "session_id": selected, "prompt": "/fork",
        }).status_code == 200
    assert all(agent.is_shutdown for agent in agents)
    saved = store.load_tabs()
    assert saved[1]["profile"] == "review"
    assert saved[1]["fork"] is True
    assert all(set(tab) <= {"title", "cwd", "rollout_path", "fork", "profile"} for tab in saved)
    agents.clear()
    # Changing the launch default must not change any saved tab's configuration,
    # including explicit null (base configuration).
    with TestClient(build_app("review")) as browser:
        restored = browser.get(prefix + "/api/sessions").json()["sessions"]
        assert [tab["profile"] for tab in restored] == [tab["profile"] for tab in saved]
        assert [agent.model_name for agent in agents] == [
            expected_default_model, "review-model", expected_default_model, expected_default_model,
        ]
        snapshot = browser.get(prefix + "/api/session", params={
            "session_id": restored[1]["id"],
        }).json()["snapshot"]
        assert snapshot["turns"][-1]["response"] == "done"
        assert agents[1].model_client.reasoning_effort == "high"
        created = browser.post(prefix + "/api/sessions").json()
        assert created["sessions"][-1]["profile"] == "review"
    assert all(agent.is_shutdown for agent in agents)


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("profile,status", [
    ("missing", 400), (123, 422), (False, 422), ([], 422), ({}, 422),
])
def test_invalid_profile_does_not_create_session(tmp_path, multi, profile, status):
    manager = WorkspaceSessionManager(make_session, tmp_path / "state")
    if multi:
        app = create_multi_workspace_app(WorkspaceRegistry([WorkspaceEntry(
            WorkspaceDefinition("test", tmp_path / "state", tmp_path, False), manager,
        )]))
    else:
        app = create_app(manager, tmp_path / "state")
    prefix = "/w/test" if multi else ""
    with TestClient(app) as browser:
        before = browser.get(prefix + "/api/sessions").json()
        response = browser.post(prefix + "/api/sessions", json={"profile": profile})
        assert response.status_code == status
        assert browser.get(prefix + "/api/sessions").json() == before


@pytest.mark.asyncio
async def test_legacy_tabs_use_default_and_missing_saved_profile_fails(tmp_path):
    store = WorkspaceStateStore(tmp_path / "state.pycodex-ws.json")
    store.save_tabs([{"title": "Legacy", "rollout_path": ""}])
    manager = WorkspaceSessionManager(
        make_session, tmp_path / "state",
        profile_factories={"fast": make_session}, default_profile="fast",
    )
    try:
        await manager.start()
        assert manager.list_sessions()[0]["profile"] == "fast"
    finally:
        await manager.close()
    saved = store.load_tabs()
    assert saved[0]["profile"] == "fast"
    manager = WorkspaceSessionManager(make_session, tmp_path / "state")
    try:
        with pytest.raises(ValueError, match="unknown Codex profile: fast"):
            await manager.start()
        assert not manager.list_sessions()
        assert store.load_tabs() == saved
    finally:
        await manager.close()
