# pycodex

[中文](README_ZH.md)

pycodex brings the [Codex](https://github.com/openai/codex) agent loop to Python,
with a CLI, reusable Agent API and browser workspace. It aims to follow upstream
behavior; supported features and deliberate differences are documented in
[Alignment](docs/ALIGNMENT.md).

| Package | Provides | Python |
| --- | --- | --- |
| `python-codex` | Agent, runtime, CLI, tools, Responses compatibility server and portable storage | 3.6.2+ |
| `pycodex-ws` | Browser workspace, Canvas, file/diff viewers and optional Feishu integration | 3.7+ |

The workspace package installs the matching core version. Python imports use
`pycodex`; the commands are `pycodex` and `pycodex-ws`.

## Install and run

Install the core CLI:

```bash
python3 -m pip install python-codex
pycodex
```

Or install the browser workspace, which includes core:

```bash
python3 -m pip install pycodex-ws
pycodex-ws --help
```

Use `python3 -m pip install --upgrade pycodex-ws` to upgrade both packages
together. If a command is missing from `PATH`, use `python3 -m pycodex` or
`python3 -m workspace_server` with the interpreter used for installation.

## Model configuration

pycodex reads `~/.codex/config.toml` and the adjacent `.env`. Configure `model`,
`model_provider`, and the provider's `base_url` and `env_key` as needed.
The built-in `openai` provider uses `OPENAI_API_KEY`. Authentication uses API
keys; ChatGPT sign-in credentials from `auth.json` or the OS keychain are not
loaded.

```bash
pycodex --profile my_profile "Summarize this repository."
pycodex --config /path/to/config.toml
pycodex doctor
```

Models use the streaming Responses API. For a Chat Completions backend, set
`use_chat_completion = true` in its provider table. Other backend options include:

```bash
pycodex --vllm-endpoint http://127.0.0.1:18000
pycodex --profile my_profile --use-messages
```

See [context and configuration](docs/CONTEXT.md) and the
[Responses compatibility server](docs/responses_server/README.md).

## CLI

Run `pycodex` without a prompt for an interactive session, or supply a prompt
for a single turn:

```bash
pycodex "Summarize this repository in one sentence."
printf 'Reply with exactly OK.' | pycodex
pycodex --json "Reply with exactly OK."
```

| Command | Purpose |
| --- | --- |
| `/model`, `/effort` | Show or change the model and reasoning effort |
| `/cd <directory>` | Change the session's working directory while idle |
| `/title`, `/history` | Manage the title and inspect conversation history |
| `/resume` | List saved sessions; pass an index or session UUID to resume |
| `/compact` | Summarize history to free context |
| `/fork` | Continue the current history under a new session ID |
| `/queue <message>` | Wait for the active turn to finish before submitting |

Ordinary input during a turn steers at the next safe sampling boundary.
Slash input uses Codex's command rules: `/data/project` and `/tmp/` are
ordinary messages, while an unknown single-component name such as `/modle`
or `/tmp` reports a command error. Start the input with a space to send any
slash-prefixed text literally, including known commands such as ` /model`.
These rules also apply to the browser workspace and Feishu.
Sessions are saved under `CODEX_HOME/sessions`. Set
`model_auto_compact_token_limit` in the configuration to compact automatically.

Ctrl+D or Ctrl+C closes the CLI after accepted work finishes. A second Ctrl+C
during closing exits immediately. Use `pycodex --help` for startup options.

## Browser workspace

Create `workspaces.json`:

```json
{
  "workspaces": [
    {"id": "demo", "work_dir": ".", "state_path": "demo", "enable_board": true}
  ]
}
```

```bash
pycodex-ws --listen 127.0.0.1:6007 --workspace-config workspaces.json
```

Open `http://127.0.0.1:6007`. The workspace combines conversations with a live
HTML Canvas and read-only file/diff viewers. Click file links in replies to open
them, and use **Insert into chat** to add selected code to an editable draft.
Local PDF links open in the browser's built-in reader, with **Open in new tab**
and **Download** links. PDF files are streamed separately from text snapshots.
Keep the server's stdin open: EOF initiates shutdown.
**New session** directly creates a session using the launch `--profile`, or the
base config when unset. The **+** button on its right (**Custom profile**) lets
you choose a named profile from the server's Codex config; **Default** uses the
launch setting. Each session saves its profile name and uses it again on restart; existing sessions
keep their configuration. Restart the server after adding or removing profiles.
The page loads the last selected session directly on reopening. Switching back
to a visited session immediately restores its rendered messages and scroll
position, then refreshes its content from the server.
While the selected Agent is working, the empty composer cycles through usage
hints about commands, frontend customization, workspace sharing, session collaboration and optional
features about every six seconds, starting at a random hint. Edit the short
strings in the `hints` list in `workspace.html`'s `updatePromptPlaceholder` to change the tips.
The gear button opens **Customize Agent** and **Share**, with a short description of each.
Customize Agent starts an Agent conversation for editing files in
`~/.pycodex/frontend/`. Its setup context includes an API quick
reference and the path to the full API documentation shipped with the package.
Browse the [feature catalog](packages/pycodex-ws/workspace_server/optional_features/README.md)
in the view area and tell the Agent which features to adapt. You can also ask it
to publish a local customization as a GitHub feature PR using the
[contribution format](packages/pycodex-ws/workspace_server/optional_features/CONTRIBUTING.md).
The [custom frontend API](docs/WORKSPACE_API.md) provides structured state,
command results, history pagination, event replay and Agent/background-task
inventories. Optional `custom.js` extensions can export `mount(host)` to use
the built-in page's session, draft and viewer controls.
With a workspace server administrator password configured, **Share** sets a
separate password for this workspace using the existing login page. Sharing
passwords and visitor logins live only in memory; changing the password, stopping
sharing or restarting the process invalidates them. Router login is unchanged.

See the [workspace guide](packages/pycodex-ws/README.md) for session controls,
storage and Feishu setup. The separately distributed
[VS Code extension](packages/pycodex-ws/extensions/vscode/README.md) uses the same
workspace backend, including on Remote SSH hosts.

## Python API

Use the standard model and tool setup from Python:

```python
from pycodex.bootstrap import build_agent, build_model

agent = build_agent(build_model())
try:
    result = agent.ask("Summarize this repository in one sentence.")
    print(result.output_text)
finally:
    agent.shutdown()
```

For custom integrations, `Agent` accepts a model client, `ToolRegistry` and
`ContextConfig`. `await agent.run_turn([...])` runs an asynchronous turn;
`AgentRuntime` adds input queues, commands and frontend subscriptions.
See [runtime contracts and examples](docs/RUNTIME.md).

## Portable configuration

Portable mode transfers an encrypted bundle of your configuration, instructions
and skills to another machine. Start a storage service in a separate terminal:

```bash
python3 -m pycodex.portable.server --root .tmp/pycodex_storage --host 127.0.0.1 --port 5577
```

Upload the default Codex home, or specify another directory:

```bash
pycodex --put @127.0.0.1:5577
pycodex --put /path/to/.codex/@127.0.0.1:5577
```

After verifying a download round trip, the command prints a reusable
`pycodex --call ...` command. Run it on the target machine using a storage
address reachable from that machine.

## Development and reference

```bash
uv sync --dev
uv run pycodex
uv run --dev pytest
```

- [Test environments and coverage](tests/TESTS.md)
- [Repository development rules and reading guide](AGENTS.md)
- [Architecture, package layout and runtime contracts](docs/RUNTIME.md)
- [Prompts, configuration and context](docs/CONTEXT.md)
- [Upstream alignment](docs/ALIGNMENT.md)
- [Responses compatibility server](docs/responses_server/README.md)
- [Portable configuration contracts](docs/PORTABLE.md)

Release notes and implementation details live in `docs/`.

## Markdown view modes

Markdown file views include a Show source / Show preview button. Switching
modes preserves each mode's reading position without replaying a previous line
target.

## Responses reliability

Providers can opt into ordered failover with `fallback_models = ["backup-one",
"backup-two"]` in their `[model_providers.<name>]` table. Each target uses the
same endpoint, credentials, history and tools; request settings are rebuilt for
that model, while the session's selected model stays unchanged. Transient errors
exhaust `stream_max_retries` before advancing. Unavailable-model errors and
first-data timeouts advance immediately. Existing retry notices name both models
and include the underlying error.

Failed models cool down for 900 seconds within that client. Subsequent requests
skip them, retry them after expiry, and restart the full chain when every target
is cooling. `fallback_cooldown_seconds = 0` disables cooldown; explicit model
selection clears it. New sessions and subagents have independent cooldowns.
An absent or empty list disables failover. Invalid input, context-length and
terminal incomplete errors retain their existing handling. Backups must support
the active tools and context size; compaction keeps the selected model's policy.

Responses HTTP 408/429 and server errors use the provider retry budget.
Rate-limit messages containing `try again in` or `retry after` supply the retry
delay. CLI and workspace retry notices include the underlying error details;
workspace notices survive browser refresh and active-turn reattachment, and
remain separate from the assistant reply. They are not persisted in rollouts.

## Workspace input history

The workspace message box supports Up/Down input history. Up recalls earlier
human inputs in the current conversation; Down returns to the unsent draft.
Automatic wake-up messages are excluded, and arrows retain normal cursor
movement inside multiline drafts.
