import json
import os
import threading
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from pycodex import AssistantMessage, ModelResponse
from pycodex.bootstrap import LOCAL_RESPONSES_SERVER_API_KEY_ENV
from pycodex.multi_agents import AgentsManager
from pycodex.utils import model_metadata as metadata_module
from tests.responses_server.fake_chat_completions_server import RunningASGITestServer
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
from workspace_server import sessions as session_module
from test_workspace_server import make_session, wait_snapshot


@pytest.mark.parametrize("explicit_slug", [False, True])
def test_new_sessions_reload_model_metadata(tmp_path, monkeypatch, explicit_slug):
    catalog = json.loads(metadata_module.DEFAULT_MODELS_PATH.read_text(encoding="utf-8"))
    model = next(item for item in catalog["models"] if item["slug"] == "gpt-6-astra")
    old_prompt = model["model_messages"]["instructions_template"]
    old_window = model["context_window"]
    metadata_path = tmp_path / "models.json"
    metadata_path.write_text(json.dumps(catalog), encoding="utf-8")
    monkeypatch.setattr(metadata_module, "DEFAULT_MODELS_PATH", metadata_path)
    config_path = tmp_path / "config.toml"
    request_model = "deployment" if explicit_slug else "custom/gpt-6-astra-test"
    config_path.write_text(
        'model = "{}"\n'.format(request_model)
        + ('model_slug = "gpt-6-astra"\n' if explicit_slug else ""),
        encoding="utf-8",
    )
    definition = WorkspaceDefinition("test", None, tmp_path, False)
    entry = app_module._build_workspace_entry(
        definition, build_parser().parse_args(["--config", str(config_path), "--toolset"]),
    )
    with TestClient(create_app(entry.manager, None)) as browser:
        initial_id = browser.get("/api/sessions").json()["sessions"][0]["id"]
        initial = entry.manager.get(initial_id)._session.runtime.agent
        assert initial.context_manager.resolve_base_instructions() == old_prompt
        assert initial.model_client.reasoning_effort == "low"
        model["default_reasoning_level"] = "high"
        model["default_verbosity"] = "medium"
        model["use_responses_lite"] = False
        model["tool_mode"] = "standard"
        model["context_window"] = 96000
        model["model_messages"]["instructions_template"] = "Fresh model instructions."
        metadata_path.write_text(json.dumps(catalog), encoding="utf-8")

        response = browser.post("/api/sessions")
        assert response.status_code == 200
        created = response.json()
        assert created["snapshot"]["model"] == request_model + ":high"
        latest = entry.manager.get(created["session_id"])._session.runtime.agent
        assert latest.context_manager.resolve_base_instructions() == "Fresh model instructions."
        assert latest.context_manager.resolve_model_max_context_window() == 96000
        assert latest.context_manager.resolve_code_mode_only() is False
        assert latest.model_client._config.use_responses_lite() is False
        assert latest.model_client._config.effective_verbosity() == "medium"

        assert initial.context_manager.resolve_base_instructions() == old_prompt
        assert initial.context_manager.resolve_model_max_context_window() == old_window
        assert initial.context_manager.resolve_code_mode_only() is True
        assert initial.model_client.reasoning_effort == "low"
        assert initial.model_client._config.use_responses_lite() is True
        child_client = initial.model_client.with_overrides()
        assert child_client.reasoning_effort == "low"
        assert child_client._config.metadata() is initial.model_client._config.metadata()


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


@pytest.mark.parametrize("profile", [None, "fast"])
def test_new_sessions_reload_dotenv_for_gemini(tmp_path, monkeypatch, profile):
    key = "PYCODEX_TEST_WORKSPACE_RELOAD_KEY"
    monkeypatch.setenv(key, "inherited")
    monkeypatch.setenv(LOCAL_RESPONSES_SERVER_API_KEY_ENV, "dummy")
    sent_keys = []

    async def generate(request):
        sent_keys.append(request.headers["x-goog-api-key"])
        payload = {"candidates": [{
            "content": {"role": "model", "parts": [{"text": "done-" + str(len(sent_keys))}]},
            "finishReason": "STOP",
        }]}

        async def events():
            yield "data: " + json.dumps(payload) + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    downstream = RunningASGITestServer(Starlette(routes=[
        Route("/v1beta/models/{model_path}", generate, methods=["POST"]),
    ]))
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "gemini-test"\nmodel_provider = "test"\ncompat_api = "gemini"\n'
        '[model_providers.test]\nbase_url = "{}/v1beta"\nstream_max_retries = 0\n'
        'env_key = "{}"\n[profiles.fast]\nmodel = "gemini-fast"\n'.format(downstream.base_url, key),
        encoding="utf-8",
    )
    dotenv = tmp_path / ".env"
    dotenv.write_text(key + "=first\n", encoding="utf-8")
    reads = []
    read_text = Path.read_text

    def track_read(path, *args, **kwargs):
        if path == dotenv:
            reads.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", track_read)
    bootstraps = []
    build_model = session_module.build_model

    def capture_bootstrap(provider, **kwargs):
        bootstrap = build_model(provider, **kwargs)
        bootstraps.append(bootstrap)
        return bootstrap

    monkeypatch.setattr(session_module, "build_model", capture_bootstrap)
    downstream.start()
    try:
        entry = app_module._build_workspace_entry(
            WorkspaceDefinition("test", None, tmp_path, False),
            build_parser().parse_args(["--config", str(config), "--toolset"]),
        )
        with TestClient(create_app(entry.manager, None)) as browser:
            def send(session_id, expected_keys, expected_reads):
                response = browser.post("/api/session/message", json={
                    "session_id": session_id, "prompt": "Reply done.",
                })
                assert response.status_code == 200
                wait_snapshot(
                    browser,
                    lambda item: item["turns"] and item["turns"][-1]["response"] == "done-" + str(len(expected_keys)),
                    "/api/session?session_id=" + session_id,
                )
                assert sent_keys == expected_keys
                assert len(reads) == expected_reads

            original = browser.get("/api/sessions").json()["sessions"][0]["id"]
            send(original, ["first"], 1)
            dotenv.write_text(key + "=updated\n", encoding="utf-8")
            assert browser.get("/api/sessions").status_code == 200
            assert os.environ[key] == "first"
            send(original, ["first", "first"], 1)
            response = browser.post("/api/sessions", json={"profile": profile})
            assert response.status_code == 200
            assert len(reads) == 2
            created = response.json()["session_id"]
            send(created, ["first", "first", "updated"], 2)
            send(created, ["first", "first", "updated", "updated"], 2)
            send(original, ["first", "first", "updated", "updated", "updated"], 2)
    finally:
        for bootstrap in bootstraps:
            bootstrap.close()
        downstream.stop()


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

    async def complete(client, prompt):
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


def test_pyagy_harness_profile_web_end_to_end_and_optional_context(tmp_path, monkeypatch):
    import time

    home_dir = tmp_path / "home"
    codex_dir = home_dir / ".codex"
    codex_dir.mkdir(parents=True)
    app_data_dir = home_dir / ".gemini" / "antigravity-cli"
    app_data_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("CODEX_HOME", str(codex_dir))
    monkeypatch.setenv("PYAGY_WS_TEST_KEY", "test-secret")
    monkeypatch.setenv(LOCAL_RESPONSES_SERVER_API_KEY_ENV, "dummy")

    # Create a Codex home AGENTS.md (must NOT be injected into pyagy) and a Web skill (MUST be injected when enabled).
    (codex_dir / "AGENTS.md").write_text(
        "Use apply_patch for manual edits in codex_home.\n", encoding="utf-8",
    )
    skill_dir = codex_dir / "skills" / "demo-web-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-web-skill\ndescription: Demo web skill visible in pyagy.\n---\nBody\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    sub_dir = work_dir / "subdir"
    sub_dir.mkdir()
    (work_dir / "AGENTS.md").write_text("Project rule from work_dir.\n", encoding="utf-8")

    captured_requests = []
    lock = threading.Lock()

    async def generate(request):
        body = json.loads((await request.body()).decode("utf-8"))
        with lock:
            captured_requests.append({
                "path": request.url.path,
                "headers": dict(request.headers),
                "body": body,
            })

        contents = body.get("contents") or []
        last_user_texts = []
        has_func_resp = False
        func_resp_text = ""
        if contents:
            for part in contents[-1].get("parts") or []:
                if "text" in part:
                    last_user_texts.append(part["text"])
                if "functionResponse" in part:
                    has_func_resp = True
                    resp_obj = part["functionResponse"].get("response") or {}
                    func_resp_text = resp_obj.get("output") or resp_obj.get("error") or ""
        joined_last = "\n".join(last_user_texts)

        if "Wrap your response in <summary></summary> tags." in joined_last:
            parts = [{"text": "<summary>Compacted workspace session summary.</summary>"}]
        elif has_func_resp:
            parts = [{"text": "tool-done:" + func_resp_text.splitlines()[-1]}]
        elif "WRITE_BOARD_TEST" in joined_last:
            board_file = str((tmp_path / "board-on.html").resolve())
            parts = [{
                "functionCall": {
                    "id": "call_write_1",
                    "name": "write_file",
                    "args": {
                        "Path": board_file,
                        "Content": "<h1>Live Board</h1>",
                        "Overwrite": True,
                    },
                },
                "thoughtSignature": "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=",
            }]
        elif "SPAWN_SUBAGENT_TEST" in joined_last and "<subagent_reminder>" not in "\n".join(
            p.get("text", "") for p in (body.get("systemInstruction") or {}).get("parts", [])
        ):
            parts = [{
                "functionCall": {
                    "id": "call_sub_1",
                    "name": "invoke_subagent",
                    "args": {
                        "TypeName": "research",
                        "Role": "Explorer",
                        "Prompt": "Inspect repo",
                    },
                },
                "thoughtSignature": "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=",
            }]
        elif "ASK_QUESTION_TEST" in joined_last:
            parts = [{
                "functionCall": {
                    "id": "call_ask_1",
                    "name": "ask_question",
                    "args": {
                        "questions": [{
                            "question": "Pick deployment mode?",
                            "options": ["Canary", "Full"],
                        }],
                    },
                },
                "thoughtSignature": "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=",
            }]
        elif "BG_SCHEDULE_TEST" in joined_last:
            parts = [{
                "functionCall": {
                    "id": "call_sched_1",
                    "name": "schedule",
                    "args": {
                        "CronExpression": "0 0 1 1 *",
                        "IsDaemon": True,
                        "Prompt": "timer wakeup",
                    },
                },
                "thoughtSignature": "c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=",
            }]
        else:
            parts = [{"text": "agy-ok"}]

        sse_payload = {
            "candidates": [{
                "content": {"role": "model", "parts": parts},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 120,
                "candidatesTokenCount": 30,
                "totalTokenCount": 150,
            },
        }

        async def events():
            yield "data: " + json.dumps(sse_payload) + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    downstream = RunningASGITestServer(Starlette(routes=[
        Route("/v1beta/models/{model_path:path}", generate, methods=["POST"]),
    ]))
    downstream.start()
    try:
        config_path = codex_dir / "config.toml"
        config_path.write_text(
            'model = "base-codex"\n'
            'model_provider = "gpool"\n'
            'model_context_window = 272000\n'
            '[model_providers.gpool]\n'
            'base_url = "{}/v1beta"\n'
            'env_key = "PYAGY_WS_TEST_KEY"\n'
            'http_headers = {{ "user-agent" = "legacy-ua", "x-goog-api-client" = "legacy-client" }}\n'.format(
                downstream.base_url
            ),
            encoding="utf-8",
        )
        # Profile 1: gemini-pool with harness = "pyagy"
        (codex_dir / "gemini-pool.config.toml").write_text(
            'model = "gemini-4-carbon-high"\n'
            'model_provider = "gpool"\n'
            'compat_api = "gemini"\n'
            'harness = "pyagy"\n'
            'model_reasoning_effort = "high"\n'
            'developer_instructions = "Custom dev rule."\n'
            'user_instructions = "Custom user rule."\n',
            encoding="utf-8",
        )
        # Profile 2: gemini-plain WITHOUT harness = "pyagy" -> must stay native pycodex!
        (codex_dir / "gemini-plain.config.toml").write_text(
            'model = "gemini-4-carbon-high"\n'
            'model_provider = "gpool"\n'
            'compat_api = "gemini"\n',
            encoding="utf-8",
        )

        def_board_on = WorkspaceDefinition("on", tmp_path / "board-on", work_dir, True)
        def_board_off = WorkspaceDefinition("off", tmp_path / "board-off", work_dir, False)
        args = build_parser().parse_args(["--config", str(config_path)])
        entry_on = app_module._build_workspace_entry(def_board_on, args)
        entry_off = app_module._build_workspace_entry(def_board_off, args)
        app = create_multi_workspace_app(WorkspaceRegistry([entry_on, entry_off]))

        with TestClient(app) as browser:
            # 1. Plain gemini profile stays native pycodex (exec/apply_patch tools, non-AgyAgent).
            plain_created = browser.post("/w/on/api/sessions", json={"profile": "gemini-plain"}).json()
            plain_agent = entry_on.manager.get(plain_created["session_id"])._session.runtime.agent
            assert type(plain_agent).__name__ == "Agent"
            assert "apply_patch" in plain_agent.tool_registry.names()

            # 2. gemini-pool profile switches to pyagy backend with board ON.
            agy_created = browser.post("/w/on/api/sessions", json={"profile": "gemini-pool"}).json()
            agy_sid = agy_created["session_id"]
            agy_session_obj = entry_on.manager.get(agy_sid)._session
            agy_agent = agy_session_obj.runtime.agent
            assert type(agy_agent).__name__ == "AgyAgent"
            assert "edit_file" in agy_agent.tool_registry.names()
            assert "apply_patch" not in agy_agent.tool_registry.names()
            assert "ask_question" in agy_agent.tool_registry.names()

            # Verify Context Ring limits come from AGY model spec (512000 / 1024000), not base 272000.
            res_state = browser.get(f"/w/on/api/sessions/{agy_sid}/state").json()["state"]
            assert res_state["auto_compact_token_limit"] == 512000
            assert res_state["max_context_window"] == 1024000

            # Verify model catalog endpoint returns AGY official models.
            models_resp = browser.get(f"/w/on/api/sessions/{agy_sid}/models").json()
            assert "gemini-4-carbon-high" in models_resp["models"]
            assert "gemini-4-neon-xhigh" in models_resp["models"]

            # Execute write_file tool and verify Canvas + optional context injection.
            captured_requests.clear()
            assert browser.post("/w/on/api/session/message", json={
                "session_id": agy_sid, "prompt": "WRITE_BOARD_TEST",
            }).status_code == 200
            wait_snapshot(
                browser,
                lambda snap: snap["turns"] and snap["turns"][-1]["status"] == "completed",
                f"/w/on/api/session?session_id={agy_sid}",
            )
            assert (tmp_path / "board-on.html").read_text(encoding="utf-8") == "<h1>Live Board</h1>"
            req_on = captured_requests[0]
            # Ensure no duplicate/legacy user-agent overwrite.
            assert "1.79.1" in req_on["headers"]["user-agent"]
            sys_parts_on = "\n".join(
                p.get("text", "") for p in req_on["body"]["systemInstruction"]["parts"]
            )
            assert "demo-web-skill" in sys_parts_on
            assert "Project rule from work_dir." in sys_parts_on
            assert "Custom dev rule." in sys_parts_on
            first_user_parts_on = "\n".join(
                p.get("text", "") for p in req_on["body"]["contents"][0]["parts"]
            )
            assert "Current workspace Canvas file:" in first_user_parts_on
            assert "<environment_context>" in first_user_parts_on
            assert "Custom user rule." in first_user_parts_on
            assert "apply_patch for manual edits in codex_home" not in first_user_parts_on
            assert "apply_patch for manual edits in codex_home" not in sys_parts_on

            # 3. Board OFF workspace omits Canvas workspace_context!
            captured_requests.clear()
            off_created = browser.post("/w/off/api/sessions", json={
                "profile": "gemini-pool", "prompt": "Hello no board",
            }).json()
            off_sid = off_created["session_id"]
            wait_snapshot(
                browser,
                lambda snap: snap["turns"] and snap["turns"][-1]["response"] == "agy-ok",
                f"/w/off/api/session?session_id={off_sid}",
            )
            req_off = captured_requests[0]
            all_off_user_text = "\n".join(
                p.get("text", "")
                for c in req_off["body"]["contents"]
                for p in c.get("parts", [])
            )
            assert "Current workspace Canvas file:" not in all_off_user_text
            assert "<environment_context>" in all_off_user_text

            # 4. ask_question remains exposed in declarations, returns disabled immediately without popup.
            assert browser.post("/w/on/api/session/message", json={
                "session_id": agy_sid, "prompt": "ASK_QUESTION_TEST",
            }).status_code == 200
            done_snap = wait_snapshot(
                browser,
                lambda s: len(s["turns"]) >= 2 and s["turns"][-1]["status"] == "completed",
                f"/w/on/api/session?session_id={agy_sid}",
            )
            assert done_snap["input_request"] is None
            assert (
                done_snap["turns"][-1]["response"]
                == "tool-done:Encountered error in tool execution: ask_question is disabled"
            )

            def run_slash(cmd_name, arg=""):
                resp = browser.post(
                    f"/w/on/api/sessions/{agy_sid}/commands",
                    json={"command": cmd_name, "argument": arg},
                )
                assert resp.status_code == 202
                sub_id = resp.json()["submission_id"]
                for _ in range(200):
                    sub = browser.get(
                        f"/w/on/api/sessions/{agy_sid}/submissions/{sub_id}"
                    ).json()["submission"]
                    if sub["status"] in ("completed", "failed"):
                        return sub
                    time.sleep(0.01)
                raise AssertionError(f"command {cmd_name} timed out")

            # A daemon cron stays armed through /resume and subsequent user turns.
            assert browser.post("/w/on/api/session/message", json={
                "session_id": agy_sid, "prompt": "BG_SCHEDULE_TEST",
            }).status_code == 200
            sleeping_snap = wait_snapshot(
                browser,
                lambda s: len(s["turns"]) >= 3 and s["turns"][-1]["status"] == "completed",
                f"/w/on/api/session?session_id={agy_sid}",
            )
            assert sleeping_snap["spinner"] == "idle: sleeping"
            assert sleeping_snap["running"] is True
            bg_state = browser.get(f"/w/on/api/sessions/{agy_sid}/state").json()["state"]
            assert bg_state["background_work_count"] == 1
            assert bg_state["busy"] is False
            bg_list = browser.get(f"/w/on/api/sessions/{agy_sid}/background-work").json()["background_work"]
            assert len(bg_list) == 1
            assert bg_list[0]["tool"] == "run_command"
            assert bg_list[0]["kind"] == "schedule"
            assert bg_list[0]["is_daemon"] is True
            assert "description" not in bg_list[0]
            assert "log_path" not in bg_list[0]
            bg_task_id = bg_list[0]["task_id"]
            assert run_slash("resume", agy_agent.session_id)["status"] == "completed"
            assert browser.get(f"/w/on/api/sessions/{agy_sid}/background-work").json()["background_work"] == bg_list
            assert browser.get(f"/w/on/api/session?session_id={agy_sid}").json()["snapshot"]["spinner"] == "idle: sleeping"

            # 4c. Subagent spawned via invoke_subagent appears in Web Agent Tree (/agents).
            assert browser.post("/w/on/api/session/message", json={
                "session_id": agy_sid, "prompt": "SPAWN_SUBAGENT_TEST",
            }).status_code == 200
            for _ in range(200):
                agents_tree = browser.get(f"/w/on/api/sessions/{agy_sid}/agents").json()
                if agents_tree["agents"]:
                    break
                time.sleep(0.01)
            assert len(agents_tree["agents"]) == 1
            assert agents_tree["agents"][0]["agent_path"].startswith("/root/sub_explorer_")
            # Wait until root turn & idle notification settle before running idle-only slash commands.
            for _ in range(200):
                st = browser.get(f"/w/on/api/sessions/{agy_sid}/state").json()["state"]
                ag = browser.get(f"/w/on/api/sessions/{agy_sid}/agents").json()["agents"]
                if not st["busy"] and all(a["status"] != "running" for a in ag):
                    break
                time.sleep(0.02)
            # Allow any queued subagent idle wake turn to finish as well.
            time.sleep(0.05)
            for _ in range(200):
                st = browser.get(f"/w/on/api/sessions/{agy_sid}/state").json()["state"]
                if not st["busy"] and st["background_work_count"] == 1:
                    break
                time.sleep(0.02)
            assert st["background_work_count"] == 1

            # 5. Slash commands: /effort, /cd, and manual /compact!
            assert run_slash("effort", "low")["status"] == "completed"
            assert run_slash("cd", str(sub_dir))["status"] == "completed"
            sub_state = run_slash("compact")
            assert sub_state["status"] == "completed"
            assert sub_state["result"]["kind"] == "compacted"

            # Send one post-compact turn to verify <CONTEXT_SUMMARY> is sent and rollout persists!
            captured_requests.clear()
            assert browser.post("/w/on/api/session/message", json={
                "session_id": agy_sid, "prompt": "After compact turn",
            }).status_code == 200
            wait_snapshot(
                browser,
                lambda s: s["turns"] and s["turns"][-1]["response"] == "agy-ok",
                f"/w/on/api/session?session_id={agy_sid}",
            )
            post_compact_req = captured_requests[-1]["body"]
            assert post_compact_req["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "LOW"
            all_post_compact_texts = "\n".join(
                p.get("text", "")
                for c in post_compact_req["contents"]
                for p in c.get("parts", [])
            )
            assert "<CONTEXT_SUMMARY>" in all_post_compact_texts
            assert str(sub_dir.resolve()) in all_post_compact_texts

        # 6. Restart app from persisted workspace tabs -> verify rollout history & cwd restored!
        entry_on_restarted = app_module._build_workspace_entry(def_board_on, args)
        app_restarted = create_multi_workspace_app(WorkspaceRegistry([entry_on_restarted]))
        with TestClient(app_restarted) as browser2:
            sessions2 = browser2.get("/w/on/api/sessions").json()["sessions"]
            agy_tabs = [t for t in sessions2 if t["profile"] == "gemini-pool"]
            assert len(agy_tabs) == 1
            restored_sid = agy_tabs[0]["id"]
            restored_snap = browser2.get(
                f"/w/on/api/session?session_id={restored_sid}"
            ).json()["snapshot"]
            assert restored_snap["cwd"] == str(sub_dir.resolve())
            assert any(t["kind"] == "compaction" for t in restored_snap["turns"])
            assert restored_snap["turns"][-1]["response"] == "agy-ok"
            assert restored_snap["spinner"] == "idle: sleeping"
            bg_url = f"/w/on/api/sessions/{restored_sid}/background-work"
            assert browser2.get(bg_url).json()["background_work"] == bg_list

            restored_session = entry_on_restarted.manager.get(restored_sid)
            env = restored_session._session.runtime.agent.tool_registry.runtime_environment
            restored_session._worker_loop.call_soon_threadsafe(env.task_manager.kill, bg_task_id)
            cleared_snap = wait_snapshot(
                browser2,
                lambda s: s["spinner"] == "" and s["running"] is False,
                f"/w/on/api/session?session_id={restored_sid}",
            )
            assert len(cleared_snap["turns"]) == len(restored_snap["turns"])
            assert browser2.get(bg_url).json()["background_work"] == []
    finally:
        downstream.stop()
