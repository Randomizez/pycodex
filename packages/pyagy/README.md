# pyagy

Antigravity CLI (`agy 1.5.13`) wire-compatible CLI harness built on top of `pycodex` and `responses_server`.

## Quick Start

Uses existing `~/.codex/config.toml` profiles with `compat_api = "gemini"`:

```bash
env -u VIRTUAL_ENV PYTHONPATH=.:packages/pyagy \
  uv run python -m pyagy --profile gemini-pool -p "Hello"
```

## Compaction & Multi-Agent Messaging

- **Context Compaction (`agy 1.5.13` wire parity)**:
  - Pre-invocation auto-compaction triggers when estimated context tokens reach the model's threshold (`512000` for Gemini 4 Neon/Carbon/Argon, `256000` for Gemini 4 Hydrogen / 3.8 / 3.7 / 3.6 Flash / 3.1 Pro, or `AGY_CLI_AUTO_COMPACT_OVERRIDE` integer / `1..100%`).
  - Manual `/compact` and `/compact <follow-up>` emit the on-demand compaction prompt (`maxOutputTokens` `32768` / `16384`), replace conversation history with `<CONTEXT_SUMMARY>`, and persist `CHECKPOINT` transcript steps across `--continue` / `--conversation`.
- **Subagent -> Root Messaging**:
  - Subagents spawned via `invoke_subagent` receive the caller's `conversation_id` and can proactively send progress or questions back to root (or any live peer) via `send_message(Recipient=<root_id>, Message=...)`, in addition to automatic idle reporting.

## Web Workspace (`pycodex-ws`) Profile Integration

Add `harness = "pyagy"` (default is `harness = "pycodex"`) to any Gemini profile (`compat_api = "gemini"`, such as `~/.codex/gemini-pool.config.toml`) to switch `pycodex-ws` Web sessions for that profile to the `pyagy` backend:

- **Optional Context Alignment**: When created from `pycodex-ws`, optional `ContextConfig` inputs (`extra_contextual_user_messages` such as Canvas board instructions when `enable_board=True` and `<external_call_context>`, `include_environment_context`, `developer_instructions`, `user_instructions`, `base_instructions_override`, and `codex_home/skills` when `include_skills_instructions=True`) are injected dynamically without altering pure CLI wire parity.
- **Disabled `ask_question` Runtime**: `ask_question` retains its official declaration in `tools` for 100% wire parity, and immediately returns `Encountered error in tool execution: ask_question is disabled` when called (without prompting).
- **Web UI Features**: Supports `/compact`, `/model`, `/effort`, `/cd`, live subagent tree (`/api/sessions/<id>/agents`), rollout tab persistence, and in-place Reload PyCodex.

## Verification

See [ALIGNMENT.md](ALIGNMENT.md) for the detailed alignment status (unaligned/placeholder items vs verified wire-compatible contracts) against official `agy 1.5.13`.

```bash
env -u VIRTUAL_ENV PYTHONPATH=.:packages/pyagy \
  uv run --dev pytest packages/pyagy/tests
```
