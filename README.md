# pycodex

[中文](README_ZH.md)

pycodex is a Python implementation of the [Codex](https://github.com/openai/codex)
coding agent. Use it in a terminal, in a browser, or from your own Python code.
It supports file editing, command execution, multiple agents and saved sessions.
See [upstream alignment](docs/ALIGNMENT.md) for the supported scope and differences.

## Install

| Package | Includes | Python |
| --- | --- | --- |
| `python-codex` | CLI and Python API | 3.6.2+ |
| `pycodex-ws` | Core plus browser workspace and optional Feishu integration | 3.7+ |

For the CLI and Python API:

```bash
python3 -m pip install --upgrade python-codex
```

For the browser workspace, install `pycodex-ws`; it includes the matching core version:

```bash
python3 -m pip install --upgrade pycodex-ws
```

## Configure a model

Create `~/.codex/config.toml`. For example, with OpenAI:

```toml
model = "gpt-5.4"
model_provider = "openai"
```

Supply the API key through the environment or `~/.codex/.env`:

```bash
export OPENAI_API_KEY="your-api-key"
```

For another provider, configure its `base_url` and `env_key`. Select the API with
`compat_api` in the profile or provider table:

| `compat_api` | API |
| --- | --- |
| `"none"` (default) | Responses |
| `"chat_completions"` | Chat Completions, including Step |
| `"messages"` | Claude Messages |
| `"gemini"` | Gemini API |

Legacy `use_chat_completion = true/false` remains supported as `chat_completions/none`.

Use `--profile NAME` to select `NAME.config.toml` beside the base config or an
embedded `[profiles.NAME]` table. Use `--config PATH` for a different config file,
and `pycodex doctor` to check configuration and connectivity.
The browser workspace picks up profile changes for new sessions without a restart.

See [configuration](docs/CONTEXT.md#配置读取与-profile) and the
[Gemini setup example](docs/responses_server/README.md#gemini-api).
Gemini 4 Argon includes a prompt adapted from Antigravity.
Gemini tool arguments stream as they are generated; tools run after the complete call is received.

## Use the CLI

Run these commands from your project directory:

```bash
pycodex
pycodex "Summarize this repository."
pycodex --profile my_profile "Review the latest changes."
pycodex --json "Summarize this repository."
```

The first command starts an interactive session; the others run a single task.
The agent can edit files and run commands with your user account's permissions.

| Command | Purpose |
| --- | --- |
| `/model`, `/effort` | View or change the model and reasoning effort |
| `/cd <directory>` | Change directory while idle |
| `/title`, `/history` | Manage the title and inspect conversation history |
| `/resume` | List or resume saved sessions |
| `/compact` | Summarize history to free context |
| `/queue <message>` | Submit a task after the current one finishes |

Send a message while the agent is working to redirect it, or use `/queue` to
wait. Sessions are saved under `~/.codex/sessions` by default.
Ctrl+C or Ctrl+D closes the session after issued operations finish.
Run `pycodex --help` for all options.

For a local vLLM server:

```bash
pycodex --vllm-endpoint http://127.0.0.1:18000
```

## Use the browser workspace

After installing `pycodex-ws`, create `workspaces.json`:

```json
{
  "workspaces": [
    {"id": "demo", "work_dir": ".", "state_path": "demo", "enable_board": true}
  ]
}
```

Start the server and keep the terminal open:

```bash
pycodex-ws --listen 127.0.0.1:6007 --workspace-config workspaces.json
```

Open `http://127.0.0.1:6007`. The workspace uses the same model configuration as
the CLI and provides conversations, a live HTML Canvas, and file/diff viewers.
Click file links to inspect code, then select text and use **Insert into chat**
to include it in your next message.

The settings menu provides interface customization and password-protected
workspace sharing. See the [workspace guide](packages/pycodex-ws/README.md) for
session controls, storage and Feishu setup. A separate
[VS Code extension](packages/pycodex-ws/extensions/vscode/README.md) connects to
the same backend, including over Remote SSH.

## Use the Python API

Save this as `example.py` and run `python3 example.py`. It uses the model
configuration above and the standard tools:

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

`AgentsManager` creates the runtime; `submit_input` submits work.
Await `receipt.future` for the result. The example closes the runtime and any
local compatibility server when finished. See [runtime contracts](docs/RUNTIME.md#ownership) for
custom models, tools and integrations.

## More features

- [Portable configuration](docs/PORTABLE.md): transfer encrypted configuration, instructions and skills to another machine.
- [Frontend customization](packages/pycodex-ws/README.md#customize-the-interface): change the workspace through an Agent conversation or your own CSS/JavaScript.
- [Responses compatibility server](docs/responses_server/README.md): expose a supported backend through the Responses API.
- [Retries and fallback models](docs/RUNTIME.md#responses-reliability): configure recovery from model failures.

## Development

From the repository root:

```bash
env -u VIRTUAL_ENV uv sync --dev
env -u VIRTUAL_ENV uv run --dev pytest
```

See the [development guide](AGENTS.md), [test guide](tests/TESTS.md),
[runtime architecture](docs/RUNTIME.md) and [workspace API](docs/WORKSPACE_API.md).
