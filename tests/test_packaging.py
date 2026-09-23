from pathlib import Path

import pytest
import yaml

try:
    import tomllib
except ImportError:  # pragma: no cover - exercised on Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _load_toml(path):
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_workspace_distribution_tracks_core_version() -> "None":
    core = _load_toml(ROOT / "pyproject.toml")
    workspace = _load_toml(ROOT / "packages" / "pycodex-ws" / "pyproject.toml")
    version = core["project"]["version"]

    assert workspace["project"]["version"] == version
    assert workspace["project"]["dependencies"] == [
        "python-codex=={0}".format(version),
    ]
    assert "scripts" not in workspace["project"]
    assert core["project"]["scripts"]["pycodex-ws"] == "workspace_server:main"


@pytest.mark.parametrize("filename", ["test.yml", "publish.yml"])
def test_python36_workflows_install_yaml_dependency(filename):
    path = ROOT / ".github" / "workflows" / filename
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["pytest-py36"]["steps"]
    install = next(
        step
        for step in steps
        if step["name"] == "Install Python 3.6 compatibility dependencies"
    )
    assert '"pyyaml>=6.0"' in install["run"]
