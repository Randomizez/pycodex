# python-codex

This repository is intentionally starting with a minimal placeholder release for
the `python-codex` package and a GitHub Actions-based PyPI publishing pipeline.

## What is included

- a minimal importable package: `pycodex`
- a GitHub Actions workflow at `.github/workflows/publish.yml`
- Trusted Publishing-ready jobs for both TestPyPI and PyPI

## Release flow

- `workflow_dispatch` with `repository=testpypi` publishes to TestPyPI
- `workflow_dispatch` with `repository=pypi` publishes to PyPI
- publishing a GitHub Release triggers the PyPI publish job

## Before the first publish

Configure Trusted Publishing on TestPyPI/PyPI for this GitHub repository and
workflow.
