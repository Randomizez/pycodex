import threading

import pytest
from starlette.testclient import TestClient

from pycodex import AssistantMessage, ModelResponse
from pycodex.multi_agents import AgentsManager
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
@pytest.mark.parametrize("profile_format", ["file", "embedded"])
def test_profiles_refresh_after_add_edit_and_delete(tmp_path, multi, profile_format):
    config = tmp_path / "config.toml"
    base = 'model = "base-model"\n'
    config.write_text(base, encoding="utf-8")
    profile_path = tmp_path / "opus.config.toml"

    def write_profile(model):
        content = 'model = "{}"\n'.format(model)
        if profile_format == "file":
            profile_path.write_text(content, encoding="utf-8")
        else:
            config.write_text(base + "[profiles.opus]\n" + content, encoding="utf-8")

    definition = WorkspaceDefinition("test", tmp_path / "state", tmp_path, False)
    entry = app_module._build_workspace_entry(
        definition, build_parser().parse_args(["--config", str(config), "--toolset"]),
    )
    app = (
        create_multi_workspace_app(WorkspaceRegistry([entry])) if multi
        else create_app(entry.manager, definition.state_path)
    )
    prefix = "/w/test" if multi else ""
    with TestClient(app) as browser:
        assert browser.get(prefix + "/api/sessions").json()["profiles"] == []
        write_profile("added-v1")
        catalog = browser.get(prefix + "/api/sessions")
        assert catalog.json()["profiles"] == ["opus"]
        assert catalog.headers["cache-control"] == "no-store"
        first = browser.post(prefix + "/api/sessions", json={"profile": "opus"})
        assert first.status_code == 200
        assert first.json()["snapshot"]["model"] == "added-v1:default"
        first_id = first.json()["session_id"]
        write_profile("added-v2")
        second = browser.post(prefix + "/api/sessions", json={"profile": "opus"})
        assert second.status_code == 200
        assert second.json()["snapshot"]["model"] == "added-v2:default"
        if profile_format == "file":
            profile_path.unlink()
        else:
            config.write_text(base, encoding="utf-8")
        # A name shown in the chooser can disappear before submission.
        rejected = browser.post(prefix + "/api/sessions", json={"profile": "opus"})
        assert rejected.status_code == 400
        catalog = browser.get(prefix + "/api/sessions").json()
        assert catalog["profiles"] == []
        assert len(catalog["sessions"]) == 3
        previous = browser.get(prefix + "/api/session", params={"session_id": first_id})
        assert previous.json()["snapshot"]["model"] == "added-v1:default"

        # A catalog refresh must surface duplicate names and recover after correction.
        profile_path.write_text('model = "file-model"\n', encoding="utf-8")
        config.write_text(base + '[profiles.opus]\nmodel = "embedded-model"\n', encoding="utf-8")
        assert browser.get(prefix + "/api/sessions").status_code == 400
        assert browser.post(prefix + "/api/sessions", json={"profile": "opus"}).status_code == 400
        assert len(entry.manager.list_sessions()) == 3
        profile_path.unlink()
        assert browser.get(prefix + "/api/sessions").json()["profiles"] == ["opus"]


@pytest.mark.parametrize("launch_profile", [None, "fast"])
def test_new_sessions_reload_config_default_but_keep_launch_override(tmp_path, launch_profile):
    config = tmp_path / "config.toml"
    base = (
        'model = "base-model"\n'
        '[profiles.fast]\nmodel = "fast-model"\n'
        '[profiles.review]\nmodel = "review-model"\n'
    )
    config.write_text(
        'profile = "fast"\nmodel = "base-model"\n[profiles.fast]\nmodel = "fast-model"\n',
        encoding="utf-8",
    )
    args = ["--config", str(config), "--toolset"]
    if launch_profile is not None:
        args.extend(["--profile", launch_profile])
    entry = app_module._build_workspace_entry(
        WorkspaceDefinition("test", None, tmp_path, False), build_parser().parse_args(args),
    )
    with TestClient(create_app(entry.manager, None)) as browser:
        original = browser.get("/api/sessions").json()["sessions"][0]["id"]
        # POST must discover the new default and name without a catalog GET.
        config.write_text('profile = "review"\n' + base, encoding="utf-8")
        expected = launch_profile or "review"
        created = browser.post("/api/sessions").json()
        assert created["snapshot"]["model"] == expected + "-model:default"
        assert created["sessions"][-1]["profile"] == expected
        assert browser.get("/api/sessions").json()["default_profile"] == expected
        config.write_text(base, encoding="utf-8")
        created = browser.post("/api/sessions", json={"profile": None}).json()
        assert created["sessions"][-1]["profile"] == launch_profile
        assert created["snapshot"]["model"] == ("fast-model:default" if launch_profile else "base-model:default")
        assert browser.get("/api/session", params={"session_id": original}).json()["snapshot"]["model"] == "fast-model:default"


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("profile_format,config_default,launch_profile", [
    ("file", None, None),
    ("file", None, "fast"),
    ("embedded", None, None),
    ("embedded", "fast", None),
    ("file", "fast", None),
    ("mixed", "review", "fast"),
])
def test_profile_creation_and_restore(
    tmp_path, monkeypatch, multi, profile_format, config_default, launch_profile,
):
    default_profile = launch_profile or config_default
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "base-model"\n'
        'model_reasoning_effort = "low"\n'
        'base_instructions = "Base instructions."\n',
        encoding="utf-8",
    )
    (tmp_path / "fast.config.toml").write_text(
        'model = "fast-model"\n'
        'model_reasoning_effort = "medium"\n'
        'base_instructions = "Fast instructions."\n',
        encoding="utf-8",
    )
    (tmp_path / "review.config.toml").write_text(
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
    base = config.read_text(encoding="utf-8")
    if config_default is not None:
        base = 'profile = "{}"\n'.format(config_default) + base
    for name in ("fast", "review"):
        if profile_format == "embedded" or (profile_format == "mixed" and name == "fast"):
            path = tmp_path / (name + ".config.toml")
            base += "[profiles.{}]\n".format(name) + path.read_text(encoding="utf-8").replace(
                "[model_providers.", "[profiles.{}.model_providers.".format(name),
            )
            path.unlink()
    config.write_text(base, encoding="utf-8")
    definition = WorkspaceDefinition("test", tmp_path / "state", tmp_path, False)
    store = WorkspaceStateStore(tmp_path / "state.pycodex-ws.json")
    agents = []
    prompts = []
    original_build_agent = AgentsManager.build_agent

    def build_agent(*args, **kwargs):
        assert threading.current_thread() is not threading.main_thread()
        agent = original_build_agent(*args, **kwargs)
        agents.append(agent)
        return agent

    async def complete(client, prompt, event_handler=None):
        prompts.append(prompt)
        return ModelResponse([AssistantMessage("done")])

    monkeypatch.setattr(AgentsManager, "build_agent", build_agent)
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
    with TestClient(build_app(launch_profile)) as browser:
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
        # A rejected command must preserve the selected profile and recording.
        assert browser.post(prefix + "/api/session/message", json={
            "session_id": selected, "prompt": "/fork",
        }).status_code == 200
    assert all(agent.is_shutdown for agent in agents)
    saved = store.load_tabs()
    assert saved[1]["profile"] == "review"
    assert saved[1]["rollout_path"]
    assert all(set(tab) <= {"title", "cwd", "rollout_path", "profile"} for tab in saved)
    agents.clear()
    # Changing the launch default must not change any saved tab's configuration,
    # including explicit null (base configuration).
    base = config.read_text(encoding="utf-8")
    if config_default is not None:
        base = base.replace('profile = "{}"\n'.format(config_default), "", 1)
    config.write_text('profile = "review"\n' + base, encoding="utf-8")
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
