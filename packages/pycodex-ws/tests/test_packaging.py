from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib


def test_workspace_distribution_tracks_core_version():
    package_root = Path(__file__).resolve().parents[1]
    with (package_root.parents[1] / "pyproject.toml").open("rb") as handle:
        core = tomllib.load(handle)
    with (package_root / "pyproject.toml").open("rb") as handle:
        workspace = tomllib.load(handle)
    version = core["project"]["version"]

    assert workspace["project"]["version"] == version
    assert workspace["project"]["requires-python"] == ">=3.7"
    assert workspace["project"]["dependencies"] == [
        "python-codex=={0}".format(version),
        "pillow>=9.0.0",
        "websockets>=11,<16",
    ]
    assert workspace["project"]["scripts"] == {
        "pycodex-ws": "workspace_server:main",
        "pycodex-router": "pycodex_router.server:main",
    }
    assert workspace["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "workspace_server", "pycodex_ws", "pycodex_router",
    ]


def test_runtime_commands_follow_package_boundary():
    from pycodex import Agent, ContextConfig, ToolRegistry
    from pycodex.bootstrap import build_runtime
    from pycodex_ws import build_runtime as build_workspace_runtime
    from tests.fakes import ScriptedModelClient

    core = build_runtime(
        Agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())
    )
    workspace = build_workspace_runtime(
        Agent(ScriptedModelClient([]), ToolRegistry(), ContextConfig())
    )

    assert "link" not in core.commands()
    assert "unlink" not in core.commands()
    assert set(workspace.commands()) == set(core.commands()) | {"link", "unlink"}
