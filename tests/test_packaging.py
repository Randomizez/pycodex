import os
import subprocess
import sys
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


def test_core_distribution_stays_independent() -> "None":
    core = _load_toml(ROOT / "pyproject.toml")
    assert not any(
        dependency.startswith((
            "pyagy", "pycodex-ws", "lark-oapi", "ipython",
            "pillow", "pycryptodomex", "fastapi", "pydantic",
        ))
        for dependency in core["project"]["dependencies"]
    )
    assert core["project"]["requires-python"] == ">=3.6.2"
    assert core["project"]["scripts"] == {"pycodex": "pycodex.cli:main"}
    assert core["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "pycodex", "responses_server",
    ]


@pytest.mark.parametrize(
    ("arguments", "exit_code", "output"),
    [
        (["--help"], 0, "usage: pycodex "),
        (["doctor", "--help"], 0, "usage: pycodex doctor "),
        (["--unknown-option"], 2, "unrecognized arguments: --unknown-option"),
    ],
)
def test_python_module_entrypoint(tmp_path, arguments, exit_code, output):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, "-m", "pycodex"] + arguments,
        cwd=str(tmp_path),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=15,
    )

    assert result.returncode == exit_code, result.stderr
    assert output in result.stdout + result.stderr


@pytest.mark.parametrize("filename", ["test.yml", "publish.yml"])
def test_python36_workflows_install_compatibility_dependencies(filename):
    path = ROOT / ".github" / "workflows" / filename
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["pytest-core-py36"]["steps"]
    install = next(
        step
        for step in steps
        if step["name"] == "Install Python 3.6 compatibility dependencies"
    )
    assert '"pyyaml>=6.0"' in install["run"]
    assert '"starlette>=0.19.1,<0.20"' in install["run"]
    assert "fastapi" not in install["run"]
