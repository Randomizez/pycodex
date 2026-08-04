from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _load_toml(path):
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_workspace_distribution_tracks_core_version() -> 'None':
    core = _load_toml(ROOT / "pyproject.toml")
    workspace = _load_toml(ROOT / "packages" / "pycodex-ws" / "pyproject.toml")
    version = core["project"]["version"]

    assert workspace["project"]["version"] == version
    assert workspace["project"]["dependencies"] == [
        "python-codex=={0}".format(version),
    ]
    assert "scripts" not in workspace["project"]
    assert core["project"]["scripts"]["pycodex-ws"] == "workspace_server:main"
