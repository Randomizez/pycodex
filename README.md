# pycodex

English README. Chinese version: `README_ZH.md`

0.3.0 release preparation and Python API migration: `docs/RELEASE_0.3.0.md`.

PyPI distributions:

- Primary package: `python-codex`
- Workspace install alias: `pycodex-ws`

The import path remains `pycodex`; the CLI commands are `pycodex` and
`pycodex-ws`.

`pycodex-ws` is a thin metapackage that depends on the exact matching
`python-codex` version. The implementation and console script remain owned by
`python-codex`, so the two distributions never install duplicate modules.

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

Install the full package or the workspace-oriented alias:

```bash
pip install python-codex
pip install pycodex-ws
```

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
- memory / compact / review mode
- a full production OpenAI adapter surface

All of those can be layered on later. For now, the project is focused on
nailing the core tool-augmented reasoning loop first.

## Layout

- `pycodex/protocol.py`: minimal conversation item / prompt protocol
- `pycodex/events.py`: typed events, plain-text views and stateful presentation
- `pycodex/utils/event_helpers.py`: stateless text, color, summary and result-formatting helpers
- `pycodex/utils/image_utils.py`: image loading, resizing and data-URL preparation
- `pycodex/model.py`: model client protocol and Responses API adapter
- `pycodex/cli.py`: single-turn/interactive entry points, terminal I/O executor and input loop
- `pycodex/bootstrap.py`: frontend-independent model, tools, Agent and session assembly
- `pycodex/tools/base_tool.py`: `BaseTool`, `ToolRegistry`, `ToolContext`
- `pycodex/tools/`: concrete tool implementations
- `pycodex/agent.py`: inner turn loop
- `pycodex/runtime.py`: session commands, submission queues and frontend event subscriptions
- `tests/test_agent.py`: core behavior tests

## Current Alignment Status

Current progress is easiest to read in layers:

- prompt/context alignment:
  - the 2026-09-23 audit uses Codex CLI 0.153.4: shared context matches for
    `gpt-5.4` first/resume/tool-follow-up requests and suffixed Astra first/resume
    requests, with explicit exclusions recorded in `docs/CONTEXT.md`;
  - this layer is now mainly handled by `pycodex/context.py` plus vendored
    prompt data; model lookup uses the longest matching slug prefix without
    rewriting the requested model name.
- turn-loop semantic alignment:
  - `AgentLoop` no longer uses a fixed 12-iteration cap by default;
  - like upstream, it now converges naturally based on whether there is still
    follow-up work or tool handoff to do;
  - the local iteration-limit parameter is gone.
- request-level alignment:
  - full raw request parity is **not** claimed: client-generated message/result
    IDs, the tool catalog, new upstream telemetry, and some permission profiles
    still differ;
  - the default CLI keeps the `codex-tui` client identity, but intentionally
    omits collaboration-mode developer instructions;
  - `User-Agent` and rollout `cli_version` use the fixed upstream alignment
    version `0.153.4`; session creation does not run `codex --version`;
  - `tests/compare_context_requests.py` keeps raw differences separate from
    shared-context exclusions; older interactive captures are not a fresh
    certification of 0.153.4 interactive behavior.
- structured user input:
  - `request_user_input` uses the registered input handler without mode gating;
  - it forces `isOther=true`, requires non-empty `options`, and returns
    structured answers as a JSON string in `function_call_output.output`;
    `success=true` is local execution metadata and is omitted from API requests;
  - without an input handler, or when the user cancels, it returns a cancelled
    response. CLI integration and tool-level tests cover both paths.

See `docs/ALIGNMENT.md` for more detailed notes.

`PYCODEX_DUMP` records downstream `stream_completed` and `stream_error_type`
alongside usage, finish reason, and exact token IDs. Failed attempts remain
in the dump, including failures before the first chunk. Chat SSE `error`
payloads propagate as `response.failed`, never as an empty completed response.
Chained failures also retain `stream_error_cause_type` and, for HTTP errors,
`stream_error_http_status`; exception messages, headers and URLs are not
copied into these diagnostic fields.

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
pycodex --profile opus --use-messages "Reply with exactly OK."
pycodex --vllm-endpoint http://127.0.0.1:18000 "Reply with exactly OK."
pycodex --put @127.0.0.1:5577
pycodex --put /data/.codex/@127.0.0.1:5577
pycodex --call SECRET-CALLID@127.0.0.1:5577 "Reply with exactly OK."
pycodex doctor
pycodex-ws --listen 0.0.0.0:6007 --workspace-config ./workspaces.json
pycodex-ws --listen 0.0.0.0:6007 --workspace-config ./workspaces.json --password 12345
```

Current behavior:

- with no argv prompt and a TTY stdin, enter interactive mode
- with an argv prompt or piped stdin, run a single turn
- `pycodex-ws` starts the standalone browser workspace manager and serves each
  workspace with a board pane and a pycodex session pane
- interactive mode exits through `/exit`, `/quit`, Ctrl+D on an empty prompt,
  or a single Ctrl+C; accepted work finishes before cleanup and normal exit.
  A `[closing]` message explains the wait; Ctrl+C does not cancel model/tool calls.
- interactive mode shows a compact event stream for user-visible phases such as
  tool execution and model follow-up after tool results
- assistant text is printed from streaming deltas directly
- interactive mode supports `/history`, `/title`, `/model`, `/resume`, `/compact`,
  and `/fork`
- `/model <name>` switches the model used by later turns in the current
  interactive session; `/model` shows the current model and available choices
- `/resume` with no argument lists the currently resumable sessions by their
  first user-message preview; `/resume 1` resumes the first listed session
- `/resume <number>` replaces the in-memory history with the selected recorded
  Codex rollout from `CODEX_HOME/sessions`
- `/compact` synthesizes a local handoff summary, replaces the in-memory
  conversation history with the compacted view, and appends a compacted-history
  entry to the rollout so later `/resume` sees the same state. The handoff remains
  model context, but is omitted from frontend conversation blocks and `/history`;
  real replies after compaction remain visible, including after resume or fork.
- `/fork` allocates a new Agent/provider session id and lazy rollout while
  preserving current history and the workspace tab; the original rollout stays intact
- `model_auto_compact_token_limit = <tokens>` in `config.toml` enables the same
  compaction path automatically when the latest reported usage reaches that
  threshold before a follow-up sampling request or the next user turn
- `service_tier = "fast"` enables Fast mode for models whose vendored metadata
  advertises the `priority` service tier; pycodex follows upstream Codex by
  sending `service_tier = "priority"` on the Responses request, while
  `service_tier = "default"` or unsupported tiers are omitted
- if a model request fails with `context_length_exceeded`, pycodex now treats
  the provider-reported requested token count as a failed-request usage sample,
  triggers the same compact path immediately, and retries the request once; if
  the compact request is also over the limit, it repeatedly drops the oldest
  tool response plus its matching tool call before retrying compact
- new sessions are now recorded under `CODEX_HOME/sessions/.../rollout-*.jsonl`
  with a stable session/thread id and per-item append+flush semantics so
  `/resume` reads back the same rollout format
- if `TURN_HOOK.md` exists in the workspace root and is non-empty, each
  completed turn also forks the just-finished history into a temporary,
  non-persisted follow-up session and submits the file contents as the next
  user instruction; this is intended for side-effect follow-ups such as
  Feishu notifications
- `/link <feishu-email|open_id|chat_id>` attaches the current interactive
  session to a Feishu card; multiple sessions in the same pycodex process share
  one Feishu long-connection listener and route card actions by message id
- `pycodex-ws --workspace-config workspaces.json` serves workspace boards from
  one process. The JSON file can be either a list or
  `{"workspaces": [...]}`; each entry uses `board` and `work_dir`, with optional
  display/URL `id`, and is exposed at `/w/<id>/`. Internally the resolved
  `board` path is the workspace identity, so adding the same board path twice is
  rejected even with a different name. Relative `board` and `work_dir` paths
  resolve from the config file directory. The root path `/` shows a workspaces
  management page with `add_workspace(name=None, dir="./", board=None)` and
  `delete(name)` controls; omitted names become `workspace-1`, `workspace-2`,
  etc. If `board` is omitted when adding a workspace, pycodex assigns a random
  writable `/tmp/pcws-*.html` board path. Add/delete actions and later
  session-state saves refresh the JSON file. Board HTML can reference local
  images beside the board (including nested paths) with relative URLs; only
  `image/*` files contained by the board directory are served.
  Assistant Markdown supports KaTeX formulas with `$...$`, `$$...$$`,
  `\(...\)`, and `\[...\]` delimiters.
  `--password <value>` enables a password-only login page for workspace pages,
  APIs, and websocket connections. After login, the browser returns to the
  requested workspace path with its query parameters.
  Questions and permission requests appear below the conversation only while
  waiting for input; they disappear when answered, cancelled, or timed out,
  leaving the continuing assistant output at the bottom.
  Queued enqueue/steer messages become user conversation blocks when their
  turns start, in execution order. Small capsules to the left of the status
  pill preview pending Steer (`↑`) and Queue (`◷`) text. Long text is truncated;
  hover shows the full pending content. Each disappears when that type has
  no waiting input.
- steer is enabled by default in interactive mode: normal input goes into the
  runtime steer path, the current request stops at the next safe boundary, and
  later steer text is appended to the next model request's `input` in order;
  for explicit queueing, use `/queue <message>`, which prints
  `[steer] queued: ...` and later `[steer] inserted: ...`
- the default local tool set includes the upstream-aligned subset plus the
  pycodex `clock` extension: `shell`, `shell_command`, `exec_command`,
  `write_stdin`, `clock`, `exec`, `wait`, `web_search`, `update_plan`,
  `request_user_input`, `request_permissions`, `spawn_agent`, `send_input`,
  `resume_agent`, `wait_agent`, `close_agent`, `apply_patch`, `grep_files`,
  `read_file`, `list_dir`, `view_image`
- `clock(period_m)` sets one periodic clock for the current Agent session;
  `null` cancels it. The countdown restarts after each reply and wakes the
  Agent with a `<clock_tick>` message containing the current timezone-aware
  time when it expires.
- while a background command or clock is pending, the idle status is
  `idle: sleeping`
- only the active workspace tab shows its close button
- `--vllm-endpoint http://host:port` automatically launches a local
  `responses_server` compatibility layer; when the URL path is empty it is
  normalized to `/v1`, and `/responses` requests are still forwarded to the
  downstream `/v1/chat/completions` endpoint. This local compat path always
  uses the canonical Responses request shape, even when the selected model's
  metadata enables `responses_lite`. With `--vllm-endpoint`, startup also reads
  `/v1/models` and uses the last returned model id for the downstream request.
  For `model_provider = "vllm"`, reasoning is preserved across this path:
  chat chunks with `reasoning` or `reasoning_content` are translated back into
  Responses `reasoning` items, and
  historical `reasoning` items are replayed into downstream assistant messages
  via the `reasoning` field. Streaming token usage is also requested from vLLM
  and forwarded to the final `response.completed.response.usage`. If a
  downstream chat stream terminates after emitting only reasoning, with no
  assistant content and no tool call, the compat layer discards that partial
  reasoning, retries the same downstream request once, and only then emits
  `response.failed` with `type = "model_output_invalid"` if the retry is still
  reasoning-only
- standalone `responses_server` now also supports downstream `/v1/messages`
  backends via `--outcomming-api messages`, while keeping the internal
  canonical request/route logic in chat-completions shape
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
from pathlib import Path

from pycodex import (
    Agent,
    BaseTool,
    ContextConfig,
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
    config_path = Path.home() / ".codex" / "config.toml"
    model = ResponsesModelClient.from_codex_config(config_path)
    context_config = ContextConfig.from_codex_config(config_path)

    tools = ToolRegistry()
    tools.register(EchoTool())

    agent = Agent(model, tools, context_config)
    result = await agent.run_turn(
        ["Call the echo tool with text=hello, then tell me what it returned."]
    )
    print(result.output_text)


asyncio.run(main())
```

### Turn lifecycle

`Agent` requires a model client, tool registry, and `ContextConfig`. It creates
its own `ContextManager`; cwd, instruction overrides and
extra contextual messages belong in that config. The client's model identifier
is authoritative for the Agent's context. Standalone context construction uses
`ContextManager(config)`; load files through `ContextConfig.from_codex_config`.

Collaboration modes and their prompt templates are removed. There is no mode
switch in `ContextConfig`, `ToolContext` or `build_agent`; normal interaction,
`update_plan`, structured user input and sub-agents remain independent features.

Dependencies are ordinary attributes: `model_client`, `tool_registry`,
`context_manager` and `event_handler`. The Agent has no Runtime/Queue reference;
`AgentRuntime` calls its public methods and subscribes to its events.
Pure getter/setter wrappers are removed. `history` remains a tuple snapshot,
and `model_name` and lifecycle properties remain derived values.

`session_file_path=None`, the default, means **keep history in memory without
recording**. Passing a file path creates an internal rollout recorder. Every
Agent still receives a stable UUIDv7 session id unless an explicit id is supplied.
CLI, Web and Feishu use `build_agent`, which assigns an id and a matching path
under the configured Codex home's `sessions/` directory. Sub-agents leave the path
as `None`, so their turns, compaction, forks and reopening stay in memory.

When a path is supplied, construction does not create the file or its parent
directories. The first history append or successful compaction creates it,
writing metadata and the current initial history before the new records.
An existing destination raises `FileExistsError`; exclusive creation also
prevents overwriting a file that appears before the first write.
`agent.session_id` and the read-only `agent.session_file_path` expose the identity
and optional recording path without exposing the recorder.

```python
agent = Agent(model, tools, context_config)  # In-memory session.
agent.resume("~/.codex/sessions/2026/09/22/rollout-example.jsonl")
```

`resume(path)` reads an existing file, restores history and session identity, and
continues appending to that same file. Paths accept `~`; the existing
concatenated-JSON and compact-checkpoint loader is reused. Constructing an Agent
and immediately resuming another file leaves no unused rollout behind. Initial
history is recorded only when the new recorded session first writes. `history`
is read-only; there is no public `replace_history()` interface.

An `Agent` executes one turn or manual compaction at a time.
There are two public turn interfaces:

- `agent.ask(text)` blocks and returns the turn result.
- `await agent.run_turn(texts, turn_id=None)` executes the ordinary coroutine
  and returns the turn result.

The Agent does not create, hold or return a per-turn Task. Creating a coroutine
does not start it or mark the Agent busy. A host that needs concurrent UI/input
handling can schedule the coroutine with `asyncio.create_task`; synchronous code
can use `ask` or `asyncio.run(agent.run_turn([...]))`. `start_turn` is removed.

`agent.is_running` reflects actual execution. One idle event backs both that
state and `await agent.wait_until_idle()`; `finally` releases it on exit.
Starting an overlapping operation raises `RuntimeError`; use `AgentRuntime`
for queued or steer submissions. `maybe_invoke` skips a busy or closed Agent;
when idle, it directly awaits the same `run_turn` path and propagates errors.
It returns `True` after that turn finishes, not immediately after scheduling.

`runtime.is_busy` includes active Agent work and pending submissions, including
direct/background turns that bypass the queue. CLI, web, and Feishu use this
state instead of tracking another busy flag. There is no Agent cancellation API:
`agent.cancel()`, `runtime.cancel_current()` and Task listeners are removed.

Steer asks `agent.stop_asap()` to end the current execution at a safe boundary.
The in-flight request and all issued tools finish first; the Agent records their
results, emits `turn_interrupted` and raises `TurnInterrupted`, without knowing
why the stop was requested. Runtime alone settles submission futures and starts
the next batch. Runtime-owned steer retains its logical turn id, but starts a new
`run_turn` execution with a fresh iteration count. `/queue` does not request a stop.
Checks after compaction prevent an extra sample when steer arrives during it.
Direct Agent calls never consume Runtime queues; their caller observes
`TurnInterrupted`, and pending input waits for the Runtime worker.

The core loop has one threshold-compaction entry, then samples, commits output
and executes tools or finishes. Before the first sample, compaction precedes new
user input; later it includes completed tool results. Context-overflow recovery
stays inside sampling: compact and retry once without counting another iteration
or replaying tools. Stops are checked before each request attempt and after the
issued tool batch, without a separate preparation wrapper. Network retries remain
inside the provider.

`await agent.compact()` computes a summary and commits the replacement history
only after the summary and rollout checkpoint succeed. Failed compaction leaves
the active history unchanged, including tool results pruned from retry prompts.
History persistence failures are surfaced rather than silently ignored.
Manual and automatic compaction share the same execution/observation path.

Sub-agent status comes from Agent lifecycle events and current busy/queue state,
including direct and background invocations. Bare `Agent.shutdown()` rejects new
calls and disables background hooks, but lets the current turn finish. `runtime.close()`,
`close_agent`, and workspace close wait for accepted work and child workers to
finish naturally; a stuck request can therefore keep close waiting.
`resume_agent` explicitly reopens the child. No turn-abort or synthetic
interruption-result recovery is performed.

Queue lifecycle uses only `start()` and `close()`. Closing switches off admission;
the worker exits when the queues are empty, without a special shutdown request.
Background invocations obey the same admission switch. Tool shutdown hooks run
once, after draining accepted work. Concurrent or repeated close calls share the
worker's completion and errors; cancelling a caller does not cancel its Agent turn.

CLI, web and Feishu restore through the backend's `runtime.resume(session_file_path)`,
which delegates to `agent.resume(session_file_path)`. It rejects
active or queued work, loads history, restores supported provider session
identity, switches its internal recorder, and clears stale usage in one
synchronous operation. Load failures leave the existing session unchanged.
The backend broadcasts restored history/title/identity to all attached views.
`agent.resume()` without a path reopens the same in-memory Agent and rebinds its
tool callbacks, without reading or writing a file or changing history, identity,
recorder or usage. It returns `None` and also rejects active or queued work.
Sub-agent services use this form to reopen the child's in-memory history.
There is no separate `reopen()` method.
Model switching and history replacement remain explicit operations because they
enforce state consistency, not merely assign a field.

### Shared session backend

CLI, Web and Feishu all use `AgentRuntime.submit_input`: not just `/model`
and `/resume`, but every session command, steer/queue admission, interactive
questions and permissions, state notifications and shutdown. Frontends only
adapt input and render events. Web no longer imports or runs the CLI shell.
IPython is the deliberate exception: `ipython_agent()` still returns a bare Agent.

```python
from pycodex.bootstrap import build_agent, build_model, build_runtime

async def run_session():
    runtime = build_runtime(build_agent(build_model()))
    await runtime.start()
    observer = runtime.attach(lambda event: print(event.kind))
    try:
        receipt = await runtime.submit_input("/model", sender="application")
        result = await receipt.future
        return result
    finally:
        await runtime.close()
        runtime.detach(observer)
```

Attach immediately supplies a state snapshot; subsequent events synchronize
all views. Detach does not close the backend. Plain input steers at the next
sampling boundary, `/queue` waits for its own turn, and busy Feishu cards still
accept steer or question answers. Unknown slash commands report errors instead
of silently entering model history. See [runtime contracts](docs/RUNTIME.md)
for command receipts, structured answers and session ownership.

Python observers receive typed `Event` dataclasses from `pycodex.events`, such
as `TurnStartedEvent` and `ToolCompletedEvent`, not generic payload dictionaries.
Custom model clients emit `ModelEvent` variants such as `AssistantDeltaEvent`.
Events provide reusable plain text through `event.visualize()`, or render through
`event.render(display)`. Each CLI view and Feishu card owns an `EventDisplay` with
independent stream/queue state and log/status/prompt callbacks. Events own colors,
flushing, status transitions and prompt text; frontends execute I/O or update their
display fields. Feishu disables colors and shows a bounded recent transcript plus
the live stream, rather than a separate last-answer/last-turn projection.
There is no display-handler mapping. Web retains its own projection and unchanged
event JSON format.

### Internal contracts

- Tools implement async `BaseTool.run`; synchronous implementations are not
  implicitly accepted. Tool-specific follow-up messages come from
  `BaseTool.follow_up_messages`, not tool-name checks in the Agent loop.
- Completed tool results are committed individually. Explicitly parallel tool
  batches wait for all their operations before propagating a commit failure;
  they do not cancel remaining operations or invent interruption results.
- Background tools bind their own callbacks through `BaseTool.bind_agent`.
  The Agent does not reach into specific tools' private managers.
- Event observers report failures to the asyncio exception handler without
  changing the execution result. Provider stream callbacks run on the owning
  event loop and stop delivering when their request is finished or cancelled.
- `ModelClient.model` is a required read-only model identifier, also supplied by
  test clients. `Agent.model_name` reads it directly, without `getattr` fallback.
  `ModelClient.complete` returns a `ModelResponse` containing only supported
  output item types. Context overflow is reported as `ContextLengthExceeded`;
  provider-specific text classification stays inside the Responses client.
- A terminal `response.incomplete` is not a disconnected stream and is not
  retried. For `max_output_tokens`, done assistant/reasoning items are retained
  for the next turn, but uncommitted text deltas and unresolved tool calls are not.
- Interactive clients expose the `ModelControl` interface in `pycodex.model`.
  Use `agent.set_model(name)` to update provider metadata, context instructions,
  context limits, and usage state together.
- Runtime services belong to the tool registry. Pass an explicit environment to
  `ToolRegistry(environment)` or `get_tools(environment)` when needed; the Agent
  uses `agent.tool_registry.runtime_environment`, not a forwarding Agent property.
  The global `get_agent_runtime_environment()` and
  separate `Agent(..., runtime_environment=...)` entry points are removed.
- Each sub-agent gets its own context manager and runtime services. The sub-agent
  service retains terminal event status, rather than inspecting Task results.
- Every Agent records its session, including direct and child Agents; external
  callers no longer construct or inject `SessionRolloutRecorder`. The outer queue
  remains optional and no worker starts implicitly. Empty initial history and a
  no-op event callback remain valid defaults.

See `docs/RUNTIME.md` for state ownership and failure semantics.

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

- [x] `clock` - pycodex periodic Agent wake-up extension.
- [x] `exec` - current local approximation of code mode.
- [x] `wait` - current local approximation of code-mode waiting behavior.

### Behavior Alignment

- [x] `AgentLoop` / `AgentRuntime` main loop skeleton - turn loop and submission
  queue are in place.
- [x] non-interactive `exec` `instructions` alignment - base instructions match
  upstream.
- [x] shared `exec` context content - the audited first/resume/tool-follow-up
  scenarios match after documented exclusions; raw item identity is not equal.
- [x] developer/contextual-user message shape alignment - message/content shape
  matches upstream.
- [x] `AGENTS.md` + `<environment_context>` injection alignment - context
  assembly order matches upstream.
- [ ] complete 0.153.4 tool catalog alignment - local `clock`, legacy sub-agent
  tools, and upstream goal/deferred/code-mode-v2 surfaces still differ.
- [x] `include = ["reasoning.encrypted_content"]` - reasoning include field is
  aligned.
- [x] `prompt_cache_key` - request-level prompt cache key is implemented.
- [x] `x-client-request-id` - request id header is implemented.
- [x] `x-codex-turn-metadata` - turn id / sandbox header is implemented.
- [x] `originator` - mode-aware originator header is implemented.
- [ ] current identity/telemetry header parity - local identity remains
  unchanged; the audit records upstream's newer headers and `client_metadata`.
- [x] field-by-field upstream exec-mode tool schema alignment - aligned tools
  use class-level specs; `clock` is documented separately as an extension.
- [ ] full interactive-mode and non-`exec` behavior alignment - the non-exec
  first-turn context is now on the `codex-tui` path, but continuous REPL
  multi-turn behavior is not fully verified yet.
- [ ] sandbox / approvals / compact / memory and other outer behavior alignment
  - these systems are still in later scope.
