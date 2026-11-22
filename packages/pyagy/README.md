# pyagy

Antigravity CLI (`agy 1.5.14`) wire-compatible CLI harness built on top of `pycodex` and `responses_server`.

## Quick Start

Start from [config_examples/pyagy.config.toml](../../config_examples/pyagy.config.toml) and save it as `~/.codex/pyagy.config.toml` (with `GEMINI_API_KEY` set in the environment or `~/.codex/.env`):

```toml
model = "gemini-4-carbon-high"
model_provider = "gemini"
compat_api = "gemini"
harness = "pyagy"
model_reasoning_effort = "high"

[model_providers.gemini]
base_url = "https://api.example.invalid/v1beta"
env_key = "GEMINI_API_KEY"
```

Run from the CLI against any profile with `compat_api = "gemini"`:

```bash
env -u VIRTUAL_ENV PYTHONPATH=.:packages/pyagy \
  uv run python -m pyagy --profile pyagy -p "Hello"
```

## Compaction & Multi-Agent Messaging

- **Context Compaction & Tool Output Truncation (`agy 1.5.14` wire parity)**:
  - Pre-invocation auto-compaction triggers when estimated context tokens reach the model's threshold (`512000` for Gemini 4 Neon/Carbon/Argon, `256000` for Gemini 4 Hydrogen / 3.8 / 3.7 / 3.6 Flash / 3.1 Pro, or `AGY_CLI_AUTO_COMPACT_OVERRIDE` integer / `1..100%`), with automatic overflow fallback on `ContextLengthExceeded`.
  - Sync and background `run_command` outputs (as well as other non-`read_file` plain-text tool outputs exceeding `10,240` UTF-8 bytes) apply exact middle truncation and persist full output under `.system_generated/steps/<step_index>/output.txt`.
  - Manual `/compact` and `/compact <follow-up>` emit the on-demand compaction prompt (`maxOutputTokens` `32768` / `16384`), replace conversation history with `<CONTEXT_SUMMARY>`, and persist `CHECKPOINT` transcript steps across `--continue` / `--conversation`.
- **Subagent -> Root Messaging**:
  - Subagents spawned via `invoke_subagent` receive the caller's `conversation_id` and can proactively send progress or questions back to root (or any live peer) via `send_message(Recipient=<root_id>, Message=...)`, in addition to automatic idle reporting.

## Web Workspace (`pycodex-ws`) Profile Integration

Add `harness = "pyagy"` (default is `harness = "pycodex"`) to any Gemini profile (`compat_api = "gemini"`, such as `~/.codex/gemini-pool.config.toml`) to switch `pycodex-ws` Web sessions for that profile to the `pyagy` backend:

- **Optional Context Alignment**: When created from `pycodex-ws`, optional `ContextConfig` inputs (`extra_contextual_user_messages` such as Canvas board instructions when `enable_board=True` and `<external_call_context>`, `include_environment_context`, `developer_instructions`, `user_instructions`, `base_instructions_override`, and `codex_home/skills` when `include_skills_instructions=True`) are injected dynamically without altering pure CLI wire parity.
- **Disabled `ask_question` Runtime**: `ask_question` retains its official declaration in `tools` for 100% wire parity, and immediately returns `Encountered error in tool execution: ask_question is disabled` when called (without prompting).
- **Web UI Features**: Supports `/compact`, `/model`, `/effort`, `/cd`, live subagent tree (`/api/sessions/<id>/agents`), background task & subagent tracking (`idle: sleeping` status and `/api/sessions/<id>/background-work`), rollout tab persistence, and in-place Reload PyCodex.
- An idle session shows `idle: sleeping` while a background command or schedule can wake it, including daemon jobs. Print mode can exit with daemons still pending.
- Pending one-shot timers and cron jobs store their next deadline, prompt and progress under the conversation’s `.system_generated/tasks/`. Web rollout restore, `--continue` and `--conversation` rearm them. An overdue timer fires on resume. A cron delivers one overdue notification, then continues on its calendar with its remaining iteration budget. Timers created by older versions with only a task log need to be scheduled again. Shell processes end when the session closes.
- The workspace enables `preview_reasoning` for pyagy Web views and Feishu cards. Root and subagent previews show the full current reasoning text or summary, including paragraph breaks. The next assistant delta replaces it with the accumulated reply text. Recorded items retain their original roles for Gemini replay.

## Verification

See [ALIGNMENT.md](ALIGNMENT.md) for the detailed alignment status (unaligned/placeholder items vs verified wire-compatible contracts) against official `agy 1.5.14`.

```bash
env -u VIRTUAL_ENV PYTHONPATH=.:packages/pyagy \
  uv run --dev pytest packages/pyagy/tests
```
