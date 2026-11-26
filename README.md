# pycodex

[中文](README_ZH.md) · [0.7.4 release notes](https://github.com/Randomizez/pycodex/blob/v0.7.4/docs/RELEASE_0.7.4.md)

A Python implementation of the [Codex](https://github.com/openai/codex) coding agent
with a built-in browser workspace, live HTML Canvas, multi-provider support, and a
full CLI & Python SDK.

## Quick start

```bash
python3 -m pip install --upgrade pycodex-ws
pycodex-ws
```

Open `http://127.0.0.1:6007` and click **Create workspace** — everything else happens
in the browser.

- **Already using Codex CLI?** Zero setup required. `pycodex-ws` automatically inherits
  `~/.codex/config.toml`, `~/.codex/.env`, `AGENTS.md`, skills, and `~/.codex/sessions`.
- **First time without Codex?** Add `~/.codex/config.toml` with
  `model = "gpt-5.4"` and set `OPENAI_API_KEY` (in your shell or `~/.codex/.env`),
  or ask any coding agent to follow [config_examples/](config_examples/README.md) with
  your endpoint and key.
- **Terminal-only?** Run `pycodex` in your project directory (or install just
  `python-codex` for Python 3.6.2+ CLI & API without the web UI).

## Highlights

- **Drop-in Codex compatibility** — shares `~/.codex` config, `AGENTS.md`, skills, and
  saved sessions; supports mid-turn steering, `/queue`, sub-agents, and auto-compaction.
- **Multi-provider & multi-harness** — works with OpenAI Responses, Chat Completions
  (StepFun, vLLM, gateways), Claude Messages, and Gemini API, plus the
  [`pyagy`](packages/pyagy/README.md) (Antigravity CLI wire-compatible) harness;
  switch profiles per session in the browser without restarting.
- **Live HTML Canvas & code/diff viewers** — agents render interactive HTML reports,
  charts, and dashboards alongside chat; inspect files, Markdown, images, and Git diffs,
  quote code back with **Insert into chat**, or use embedded VS Code (`--password`).
- **Self-customizing UI (`Customize Agent`)** — open **Settings → Customize Agent** to
  install optional frontend features (e.g. `@/$` skill completion, file uploads, voice
  input) or tailor CSS/JS through conversation, with safe `?builtin=1` fallback.
- **Multi-device Router & Peer Profiles** — connect multiple machines into one
  `pycodex-router` directory, share workspaces by password, run tools locally while
  borrowing another verified node's model profiles (API keys never leave the peer), and
  upgrade running nodes in place via **Settings → Update & reboot**.
- **External Call, Feishu & Portable config** — bridge a remote client terminal into a
  Web session (**External Call**), chat from Feishu cards (`/link`), or transfer
  encrypted config and skills across machines (`pycodex --put` / `--call`).

## Let the agent handle advanced setup

Instead of manually editing config files or memorizing flags, open a browser session
and ask the agent:

- **Add another model / provider (Claude, Gemini, StepFun, `pyagy`, vLLM)**:
  > Follow `config_examples/README.md` to add a new profile using my endpoint and API key, then verify it with `pycodex doctor`.
- **Customize the web UI**:
  Click **Settings → Customize Agent**, browse the built-in feature catalog, and tell the agent what to enable or change.
- **Set up multi-device router, sharing, External Call, or Feishu**:
  > Follow `packages/pycodex-ws/README.md` to configure `pycodex-router` / Feishu `/link` / workspace sharing for me.

## CLI & Python API

```bash
pycodex                          # interactive terminal session
pycodex "Summarize this repo."   # single task
pycodex doctor                   # check config and connectivity
```

<details>
<summary>Minimal Python API example</summary>

```python
from contextlib import closing

from pycodex import AgentsManager, CodexConfig
from pycodex.bootstrap import build_model
from pycodex.utils.async_bridge import run_async


async def main():
    config = CodexConfig.from_env()
    with closing(build_model(config.provider)) as model:
        runtime = AgentsManager(
            model.client, config.context,
        ).create_root_runtime()
        try:
            await runtime.start()
            receipt = await runtime.submit_input("Summarize this repository.")
            print((await receipt.future).output_text)
        finally:
            await runtime.close()


run_async(main())
```

</details>

## Packages & docs

| Package | Includes | Python |
| --- | --- | --- |
| `pycodex-ws` | Browser workspace, router, optional Feishu, plus matching `python-codex` & `pyagy` ([guide](packages/pycodex-ws/README.md)) | 3.7+ |
| `pyagy` | Core plus Antigravity CLI (`agy`) wire-compatible harness ([guide](packages/pyagy/README.md)) | 3.7+ |
| `python-codex` | Core CLI, Python API, and Responses compatibility server | 3.6.2+ |

- Setup & usage: [config templates](config_examples/README.md) · [configuration & profiles](docs/CONTEXT.md#配置读取与-profile) · [Responses compat server](docs/responses_server/README.md) · [Portable transfer](docs/PORTABLE.md)
- Architecture & development: [upstream alignment](docs/ALIGNMENT.md) · [runtime contracts](docs/RUNTIME.md) · [workspace API](docs/WORKSPACE_API.md) · [AGENTS.md](AGENTS.md) · [tests](tests/TESTS.md)

```bash
env -u VIRTUAL_ENV uv sync --dev
env -u VIRTUAL_ENV uv run --dev pytest
```
