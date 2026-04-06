# pycodex

English README. Chinese version: `README_ZH.md`

PyPI distribution name: `python-codex`  
Import path and CLI command remain `pycodex`.

This repository extracts the core Codex agent loop from upstream Codex
(`https://github.com/openai/codex`) into a deliberately small Python version,
while preserving the two most important layers:

- `submission_loop`: sequentially consumes submitted operations.
- `run_turn`: keeps executing `model sample -> tool call -> feed tool result
  back into the model` inside a single turn until a final answer is reached.

Relevant Rust reference points:

- `codex-rs/core/src/codex.rs` -> `submission_loop`
- `codex-rs/core/src/codex.rs` -> `run_turn`
- `codex-rs/core/src/codex.rs` -> `run_sampling_request`
- `codex-rs/core/src/tools/router.rs` -> `ToolRouter`
- `codex-rs/core/src/stream_events_utils.rs` -> `handle_output_item_done`

## Quick Start

Install dependencies first:

```bash
uv sync
```

Try the real entry points:

```bash
uv run pycodex "Reply with exactly OK."
uv run pycodex
```

## Design Tradeoffs

This is not a 1:1 port of the Rust implementation. The current goal is a
minimal reusable kernel that converges on the upstream behavior over time:

1. Use a thin `ModelClient` protocol to abstract the model side.
2. Use `ToolRegistry` to manage tool specs and executors.
3. Use `AgentLoop` to implement the core closed loop.
4. Use `AgentRuntime` to preserve the outer submission queue so it can keep
   converging toward Rust's `submission_loop` later.

Intentionally not included yet:

- TUI / streaming incremental rendering
- MCP / connectors / sandbox / approvals
- memory / compact / hooks / review mode
- a full production OpenAI adapter surface

All of those can be layered on later. For now, the project is focused on
nailing the core tool-augmented reasoning loop first.

## Layout

- `pycodex/protocol.py`: minimal conversation item / prompt / event protocol
- `pycodex/model.py`: model client protocol and Responses API adapter
- `pycodex/cli.py`: single-turn and interactive `pycodex` CLI entry points
- `pycodex/tools/base_tool.py`: `BaseTool`, `ToolRegistry`, `ToolContext`
- `pycodex/tools/`: concrete tool implementations
- `pycodex/agent.py`: inner turn loop
- `pycodex/runtime.py`: outer submission queue
- `tests/test_agent.py`: core behavior tests

## Current Alignment Status

Current progress is easiest to read in layers:

- prompt/context alignment:
  - on the non-interactive `exec` path, `instructions` and `input` already
    match upstream Codex;
  - this layer is now mainly handled by `pycodex/context.py` plus vendored
    prompt data.
- turn-loop semantic alignment:
  - `AgentLoop` no longer uses a fixed 12-iteration cap by default;
  - like upstream, it now converges naturally based on whether there is still
    follow-up work or tool handoff to do;
  - the local iteration-limit parameter is gone.
- request-level alignment:
  - the non-interactive `exec` request body is mostly aligned;
  - the default CLI non-exec first request now also follows the upstream
    `codex-tui` + `<collaboration_mode>` path;
  - the default CLI two-turn main-thread request/header behavior has also been
    captured and aligned, including omitting `workspaces` on later turns;
  - the remaining work is now more about outer behavior branches than this
    already-compared request/header path.
- tool round-trip alignment:
  - the Default-mode unavailable path for `request_user_input` is aligned to
    real upstream captures;
  - the Plan-mode happy path is also aligned at the tool/protocol layer based
    on upstream source: it forces `isOther=true`, requires non-empty `options`,
    and returns structured answers as a JSON string plus `success=true`;
  - there is now a deterministic round-trip comparison helper,
    `tests/compare_request_user_input_roundtrip.py`, built on the proxy mode in
    `tests/fake_responses_server.py`; against the locally installed
    `codex-cli 0.115.0`, the only remaining Plan-mode live-capture schema
    difference is that `pycodex` includes `success=true` in
    `function_call_output`.

See `docs/ALIGNMENT.md` for more detailed notes.

## Live Model Integration

If this machine already has a Codex CLI configuration, `pycodex` can reuse the
`model`, `model_provider`, `base_url`, and `env_key` from
`~/.codex/config.toml` directly:

```python
from pycodex import ResponsesModelClient

client = ResponsesModelClient.from_codex_config()
```

The current implementation uses the streaming OpenAI-compatible `/responses`
endpoint. This path has already been validated against the local
`~/.codex/config.toml` setup.

When launched through the CLI, `pycodex` also loads `.env` from the same
configuration directory before reading config (typically `~/.codex/.env`), so
provider keys and similar environment variables can live there. To match
upstream Codex, variables starting with `CODEX_` are not imported from `.env`.

## pycodex CLI

`pycodex` now defaults to a minimal interactive entry point. Internally it uses
`AgentRuntime` to drive the turn submission loop and reuses
`~/.codex/config.toml` by default:

```bash
pycodex
pycodex "Summarize this repo in one sentence."
printf 'Reply with exactly OK.' | pycodex
pycodex --json "Reply with exactly OK."
pycodex --profile model_proxy "Reply with exactly OK."
pycodex --vllm-endpoint http://127.0.0.1:18000 "Reply with exactly OK."
pycodex --put @127.0.0.1:5577
pycodex --put /data/.codex/@127.0.0.1:5577
pycodex --call SECRET-CALLID@127.0.0.1:5577 "Reply with exactly OK."
pycodex doctor
```

Current behavior:

- with no argv prompt and a TTY stdin, enter interactive mode
- with an argv prompt or piped stdin, run a single turn
- interactive mode supports `/exit` and `/quit`
- interactive mode shows a compact event stream for user-visible phases such as
  tool execution and model follow-up after tool results
- assistant text is printed from streaming deltas directly
- interactive mode supports `/history`, `/title`, and `/model`
- `/model <name>` switches the model used by later turns in the current
  interactive session; `/model` shows the current model and available choices
- steer is enabled by default in interactive mode: normal input goes into the
  runtime steer path, the current request stops at the next safe boundary, and
  later steer text is appended to the next model request's `input` in order;
  for explicit queueing, use `/queue <message>`, which prints
  `[steer] queued: ...` and later `[steer] inserted: ...`
- the default built-in tool subset currently exposed as local tools is:
  `shell`, `shell_command`, `exec_command`, `write_stdin`, `exec`, `wait`,
  `web_search`, `update_plan`, `request_user_input`, `request_permissions`,
  `spawn_agent`, `send_input`, `resume_agent`, `wait_agent`, `close_agent`,
  `apply_patch`, `grep_files`, `read_file`, `list_dir`, `view_image`
- `--vllm-endpoint http://host:port` automatically launches a local
  `responses_server` compatibility layer; when the URL path is empty it is
  normalized to `/v1`, and `/responses` requests are still forwarded to the
  downstream `/v1/chat/completions` endpoint. For `model_provider = "vllm"`,
  reasoning is now preserved across this path: chat chunks with `reasoning` or
  `reasoning_content` are translated back into Responses `reasoning` items, and
  historical `reasoning` items are replayed into downstream assistant messages
  via the `reasoning` field. Streaming token usage is also requested from vLLM
  and forwarded to the final `response.completed.response.usage`
- `pycodex doctor` checks config, `.env`, API keys, DNS, TCP/TLS, and an
  optional live Responses API request

Current primary uses:

- verify provider / model / auth configuration
- debug `ResponsesModelClient`
- run minimal single-turn and multi-turn smoke tests

`doctor` examples:

```bash
pycodex doctor
pycodex doctor --skip-live
pycodex doctor --json
```

## Portable Mode

`Portable Mode` is the quickest way to bring your usual `pycodex` setup into a
fresh machine, container, or debug image.

Use it like this:

```bash
pycodex --put @127.0.0.1:5577
pycodex --put /data/.codex/@127.0.0.1:5577
```

- `--put` prints a reusable `SECRET-CALLID@host:port` plus a final one-line
  `pycodex --call ...` command
- on the new environment or image, run that printed `--call` command directly
- quickly restoring your usual `config.toml`, `.env`, `AGENTS.md`, and
  `skills/` into a clean debug environment
- keeping a new image focused on the bug you are debugging instead of spending
  time rebuilding local Codex setup by hand
- bootstrapping `pycodex` even when the target environment does not already
  have a populated `~/.codex`
- bare `--put` uses the current user's `~/.codex`
- `--put /path/.codex/@host:port` lets you publish a different Codex home

## Example

```python
import asyncio

from pycodex import (
    AgentLoop,
    BaseTool,
    ContextManager,
    ResponsesModelClient,
    ToolRegistry,
)


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo the provided text."
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def run(self, context, args):
        del context
        return args["text"]


async def main() -> None:
    model = ResponsesModelClient.from_codex_config()
    context_manager = ContextManager.from_codex_config()

    tools = ToolRegistry()
    tools.register(EchoTool())

    agent = AgentLoop(model, tools, context_manager)
    result = await agent.run_turn(
        ["Call the echo tool with text=hello, then tell me what it returned."]
    )
    print(result.output_text)


asyncio.run(main())
```

## Alignment Checklist

See `docs/ALIGNMENT.md` for more detail. This section keeps a high-level
checklist for quick status scanning.

### Tool Alignment

Official upstream tools:

- [x] `shell` - run shell commands in argv form.
- [x] `shell_command` - run shell scripts in string form.
- [x] `exec_command` - start long-running commands with a session.
- [x] `write_stdin` - write stdin to an existing execution session or poll
  output.
- [x] `web_search` - expose provider-native web search capability.
- [x] `update_plan` - update the task plan and maintain step status.
- [x] `request_user_input` - ask the user structured questions and wait for an
  answer.
- [x] `request_permissions` - request extra permissions before continuing.
- [x] `spawn_agent` - create and start a sub-agent.
- [x] `send_input` - continue feeding input to an existing sub-agent.
- [x] `resume_agent` - reopen a closed sub-agent.
- [x] `wait_agent` - wait for a sub-agent to reach a terminal state.
- [x] `close_agent` - close a sub-agent that is no longer needed.
- [x] `apply_patch` - edit files precisely with a freeform patch.
- [x] `grep_files` - search file contents by pattern.
- [x] `read_file` - read file slices while preserving line-number semantics.
- [x] `list_dir` - list directory tree slices.
- [x] `view_image` - turn a local image into model-visible input.

Upstream low-frequency / special-mode tools not yet modeled separately:

- [ ] `wait_infinite` - long blocking wait for external events or later input.
- [ ] `spawn_agents_on_csv` - create sub-agent jobs in bulk from CSV.
- [ ] `report_agent_job_result` - report batch agent job results.
- [ ] `js_repl` - JavaScript REPL / code-mode primary entry point.
- [ ] `js_repl_reset` - reset `js_repl` state.
- [ ] `artifacts` - generate or manage structured artifact outputs.
- [ ] `list_mcp_resources` - list MCP resources.
- [ ] `list_mcp_resource_templates` - list MCP resource templates.
- [ ] `read_mcp_resource` - read MCP resource contents.
- [ ] `multi_tool_use.parallel` - parallel wrapper around multiple developer
  tool calls.

Repository-specific compatibility / transition tools:

- [x] `exec` - current local approximation of code mode.
- [x] `wait` - current local approximation of code-mode waiting behavior.

### Behavior Alignment

- [x] `AgentLoop` / `AgentRuntime` main loop skeleton - turn loop and submission
  queue are in place.
- [x] non-interactive `exec` `instructions` alignment - base instructions match
  upstream.
- [x] non-interactive `exec` `input` alignment - prompt input matches upstream.
- [x] developer/contextual-user message shape alignment - message/content shape
  matches upstream.
- [x] `AGENTS.md` + `<environment_context>` injection alignment - context
  assembly order matches upstream.
- [x] non-interactive `exec` tool subset alignment - the model-visible tool set
  has converged.
- [x] `include = ["reasoning.encrypted_content"]` - reasoning include field is
  aligned.
- [x] `prompt_cache_key` - request-level prompt cache key is implemented.
- [x] `x-client-request-id` - request id header is implemented.
- [x] `x-codex-turn-metadata` - turn id / sandbox header is implemented.
- [x] `originator` - mode-aware originator header is implemented.
- [x] exact `user-agent` string alignment - aligned on the non-interactive
  `exec` path.
- [x] field-by-field exec-mode tool schema alignment - currently reuses the
  upstream snapshot directly through the tool layer.
- [ ] full interactive-mode and non-`exec` behavior alignment - the non-exec
  first-turn context is now on the `codex-tui` path, but continuous REPL
  multi-turn behavior is not fully verified yet.
- [ ] sandbox / approvals / compact / memory and other outer behavior alignment
  - these systems are still in later scope.
