import pytest

from pycodex.context import ContextConfig, ContextManager
from pycodex.model import ResponsesProviderConfig
from pycodex.utils.codex_config import list_codex_profiles, read_codex_config


def _write_profile(config, name, settings, embedded):
    if embedded:
        settings = "[profiles.{}]\n".format(name) + "\n".join(
            "[profiles.{}.".format(name) + line[1:] if line.startswith("[") else line
            for line in settings.splitlines()
        ) + "\n"
        config.write_text(config.read_text(encoding="utf-8") + settings, encoding="utf-8")
    else:
        (config.parent / (name + ".config.toml")).write_text(settings, encoding="utf-8")


@pytest.mark.parametrize("embedded", [False, True])
def test_profile_layers_provider_and_context_from_custom_config(tmp_path, embedded):
    config = tmp_path / "custom.toml"
    config.write_text(
        'model = "base-model"\n'
        'model_provider = "demo"\n'
        'model_reasoning_effort = "high"\n'
        'model_context_window = 32000\n'
        '[model_providers.demo]\n'
        'base_url = "https://base.invalid/v1"\n'
        'env_key = "DEMO_KEY"\n'
        'fallback_models = ["old-a", "old-b"]\n'
        '[model_providers.demo.query_params]\n'
        'version = "v1"\n'
        'region = "base"\n'
        '[features]\n'
        'guardian_approval = true\n'
        '[features.multi_agent_v2]\n'
        'max_concurrent_threads_per_session = 7\n'
        'root_agent_usage_hint_text = "Inherited hint"\n',
        encoding="utf-8",
    )
    _write_profile(config, "step",
        'model = "step-5-preview"\n'
        'model_instructions_file = "instructions/step.md"\n'
        'model_auto_compact_token_limit = 24000\n'
        '[model_providers.demo]\n'
        'base_url = "https://profile.invalid/v1"\n'
        'fallback_models = ["new"]\n'
        '[model_providers.demo.query_params]\n'
        'region = "profile"\n'
        '[features.multi_agent_v2]\n'
        'max_concurrent_threads_per_session = 3\n',
        embedded,
    )
    (tmp_path / "instructions").mkdir()
    (tmp_path / "instructions" / "step.md").write_text(
        "中文指令\n", encoding="utf-8",
    )
    provider = ResponsesProviderConfig.from_codex_config(config, "step")
    context = ContextConfig.from_codex_config(config, "step")

    assert provider.model == context.model == "step-5-preview"
    assert provider.base_url == "https://profile.invalid/v1"
    assert provider.api_key_env == "DEMO_KEY"
    assert provider.reasoning_effort == "high"
    assert provider.fallback_models == ("new",)
    assert provider.query_params == {"version": "v1", "region": "profile"}
    assert provider.beta_features_header == "guardian_approval"
    assert context.model_context_window == 32000
    assert context.model_auto_compact_token_limit == 24000
    assert context.multi_agent_max_concurrency == 3
    assert context.multi_agent_root_instructions == "Inherited hint"
    assert context.codex_home == tmp_path.resolve()
    assert ContextManager(context).resolve_base_instructions() == "中文指令"
    assert ResponsesProviderConfig.from_codex_config(config).model == "base-model"
    assert ContextConfig.from_codex_config(config).multi_agent_max_concurrency == 7


@pytest.mark.parametrize("embedded", [False, True])
def test_profile_replaces_values_with_different_types(tmp_path, embedded):
    config = tmp_path / "config.toml"
    config.write_text(
        '[features]\nmulti_agent_v2 = false\n'
        '[custom]\nitems = ["a", "b"]\n[custom.flag]\nenabled = true\n',
        encoding="utf-8",
    )
    _write_profile(config, "work",
        '[features.multi_agent_v2]\nmax_concurrent_threads_per_session = 2\n'
        '[custom]\nitems = []\nflag = false\n',
        embedded,
    )

    selected = read_codex_config(config, "work")

    assert selected["features"]["multi_agent_v2"] == {
        "max_concurrent_threads_per_session": 2,
    }
    assert selected["custom"] == {"items": [], "flag": False}


@pytest.mark.parametrize("load", [
    ResponsesProviderConfig.from_codex_config, ContextConfig.from_codex_config,
])
def test_missing_or_invalid_profile_does_not_use_base_config(tmp_path, load):
    config = tmp_path / "config.toml"
    config.write_text('model = "base"\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing.config.toml"):
        load(config, "missing")
    (tmp_path / "broken.config.toml").write_text('model = [', encoding="utf-8")
    with pytest.raises(ValueError):
        load(config, "broken")


@pytest.mark.parametrize("profile", [
    "", "../outside", "/absolute", "a/b", "a\\b", ".", "..",
    "with space", "a.b", "中文", "work\n", 123, False, [], {},
])
def test_profile_name_cannot_address_other_paths(tmp_path, profile):
    config = tmp_path / "config.toml"
    config.write_text('model = "base"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid Codex profile name"):
        read_codex_config(config, profile)


@pytest.mark.parametrize("file_contents", ['model = "new"\n', 'model = ['])
def test_duplicate_profile_is_ambiguous(tmp_path, file_contents):
    config = tmp_path / "config.toml"
    config.write_text(
        'model = "base"\n[profiles.work]\nmodel = "legacy"\n',
        encoding="utf-8",
    )
    (tmp_path / "work.config.toml").write_text(file_contents, encoding="utf-8")

    with pytest.raises(ValueError, match=r"ambiguous Codex profile 'work'"):
        read_codex_config(config, "work")
    with pytest.raises(ValueError, match=r"ambiguous Codex profile 'work'"):
        list_codex_profiles(config)
    assert read_codex_config(config)["model"] == "base"
    (tmp_path / "other.config.toml").write_text('model = "other"\n', encoding="utf-8")
    assert read_codex_config(config, "other")["model"] == "other"


@pytest.mark.parametrize("embedded", [False, True])
def test_default_profile_and_explicit_selection(tmp_path, embedded):
    config = tmp_path / "config.toml"
    config.write_text('model = "base"\nprofile = "work"\n', encoding="utf-8")
    _write_profile(config, "work", 'model = "work"\n', embedded)
    _write_profile(config, "other", 'model = "other"\n', not embedded)

    for profile, expected in [(None, "work"), ("work", "work"), ("other", "other")]:
        selected = read_codex_config(config, profile)
        assert selected["profile"] == selected["model"] == expected
        assert ResponsesProviderConfig.from_codex_config(config, profile).model == expected
        assert ContextConfig.from_codex_config(config, profile).model == expected
    base = read_codex_config(config, use_default_profile=False)
    assert base["profile"] is None
    assert base["model"] == "base"
    for load in [ResponsesProviderConfig.from_codex_config, ContextConfig.from_codex_config]:
        assert load(config, use_default_profile=False).model == "base"
        assert load(config, "other", use_default_profile=False).model == "other"


@pytest.mark.parametrize("selector,error", [
    ('"missing"', FileNotFoundError),
    ('"../outside"', ValueError),
    ("123", ValueError),
    ("false", ValueError),
    ("[]", ValueError),
])
def test_invalid_default_does_not_fall_back(tmp_path, selector, error):
    config = tmp_path / "config.toml"
    config.write_text('model = "base"\nprofile = {}\n'.format(selector), encoding="utf-8")
    with pytest.raises(error):
        read_codex_config(config)
    _write_profile(config, "work", 'model = "work"\n', False)
    assert read_codex_config(config, "work")["model"] == "work"


@pytest.mark.parametrize("embedded", [False, True])
def test_profile_cannot_select_another_profile(tmp_path, embedded):
    config = tmp_path / "config.toml"
    config.write_text('model = "base"\n', encoding="utf-8")
    _write_profile(config, "work", 'profile = "other"\n', embedded)

    with pytest.raises(ValueError, match="cannot select another profile"):
        read_codex_config(config, "work")


@pytest.mark.parametrize("settings,message", [
    ('profiles = "work"\n', "profiles .* must be a table"),
    ('[profiles]\nwork = "model"\n', r"\[profiles.work\] .* must be a table"),
    ('[profiles."../work"]\nmodel = "work"\n', "invalid Codex profile name"),
])
def test_malformed_embedded_profiles_are_rejected(tmp_path, settings, message):
    config = tmp_path / "config.toml"
    config.write_text(settings, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_codex_config(config, "work")
    with pytest.raises(ValueError, match=message):
        list_codex_profiles(config)


def test_profile_catalog_lists_embedded_and_valid_adjacent_files(tmp_path):
    config = tmp_path / "config.toml"
    for name in [
        "config.toml", "work.config.toml", "_fast-2.config.toml",
        "with space.config.toml", ".config.toml", "a.b.config.toml",
        "work.config.toml.bak",
    ]:
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "folder.config.toml").mkdir()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "hidden.config.toml").write_text("", encoding="utf-8")
    _write_profile(config, "legacy", 'model = "legacy"\n', True)

    assert list_codex_profiles(config) == ["_fast-2", "legacy", "work"]
