# 0.6.0 release notes

These notes compare **0.6.0 with the tagged 0.5.0 release**, including the
improvements shipped in 0.5.1 and 0.5.2. The main changes are a new agent
collaboration model, Python orchestration tools, persistent goals and clocks,
and richer workspace history and session controls.

## Agent collaboration and lifecycle

- MultiAgentV2 uses `spawn_agent`, `send_message`, `followup_task`, `wait_agent`,
  `interrupt_agent` and `list_agents` in the `collaboration` namespace.
  Sending a message to an idle agent does not start a turn; `followup_task`
  delivers a task and starts work. Child completion and failure notifications
  reach the parent's mailbox without requiring a polling loop.
- Agents have stable IDs and task paths such as `/root/review`. Resuming changes
  the recorded session ID, while the live agent identity remains stable.
  `fork_turns="all"` inherits the parent's context; `"none"` starts fresh.
- A session allows 16 agents by default, including the root. At capacity, a
  spawn can reclaim the least recently used idle leaf with no pending or
  background work. Running agents and ancestors are not reclaimed.
- Closing a session stops admission throughout its agent subtree, discards
  queued work and waits for issued model/tool operations and resource cleanup.
  Interruptions are visible while already issued operations finish.
- Native agent messages and encrypted function-argument metadata survive
  recording, resume and context checkpoints. Providers must support these
  wire items to use native MultiAgentV2.

## Python orchestration, goals and clocks

- CodeModeOnly `exec` now accepts Python with top-level `await`. Scripts call
  ordinary tools through `tools` and agent tools through `collaboration`.
  Function arguments follow the advertised Python signatures; custom tools
  accept a source string. `wait` resumes a yielded cell.
- Cells expose `cwd`, output helpers and a per-agent `store`/`load` facility.
  Ordinary variables are local to each cell. Execution runs in the host Python
  process: imported modules and process state are shared, and stopping is
  cooperative. Use asynchronous operations or `tools.exec_command` for work
  that would otherwise block the event loop.
- `exec_command` and `write_stdin` return structured output, exit status,
  session identifiers and timing. Successful `apply_patch` returns `{}`;
  verification failures are tool errors.
- `create_goal`, `get_goal` and `update_goal` track an explicitly requested
  objective, progress state, elapsed time and optional token budget. Active
  goals can continue after a reply; budget exhaustion stops that continuation.
  Goal state is included in recorded session environment snapshots.
- Recorded sessions retain their clock period across restart and resume.
  Restoring a clock starts a new countdown and restores its idle status;
  it does not replay missed ticks.

## Workspace history, sharing and views

- Clicking an agent in the context tree opens a live transcript tab for the
  root or a child. Transcripts provide paginated history, live updates,
  collapsible records and exec/wait cell identifiers, with retained reading
  position when switching views.
- The built-in context popover shows the agent tree with running/failed totals,
  elapsed turn time and error details. The duplicate `subagent-status` optional
  feature has been removed. Returning to a visited conversation reuses its
  rendered messages, draft and reading position; reopening loads the saved
  session directly.
- Session profiles can be selected when creating a tab and are restored after
  restart. **Settings → Share** creates a separate visitor password for a
  workspace; changing or disabling sharing revokes access. Sharing remains
  in memory and ends when the server restarts.
- The custom frontend API exposes session state, command receipts, paginated
  history, descendant agents, background work and cursored WebSocket replay.
  Modules can use `mount(host)` for the page's session, draft and view controls.
  Optional feature sources cover command status, tool groups, readable recorded
  reasoning, per-conversation notes and clock ticks.
- Text and image views now accept files up to **10 MiB** each. Local PDFs stream
  to the browser's native reader with separate open/download links. Markdown
  source/preview modes retain their own reading positions, and local file/Git
  links preserve the requested path and line selection.
- Input history, collapsed reconnection diagnostics, mobile session controls
  and contextual usage hints improve navigation without replacing drafts.

## Configuration and Responses reliability

- Named profiles can live in adjacent `<name>.config.toml` files or embedded
  `[profiles.<name>]` tables. Both merge recursively over the base configuration.
  Duplicate names and missing selected profiles are reported as errors.
- HTTP 408/429 and server errors use the provider retry budget. Retry notices
  include the underlying error, and rate-limit messages can supply a delay.
- Optional `stream_first_data_timeout_ms` bounds the initial response wait;
  `stream_idle_timeout_ms` controls subsequent idle periods.
- Optional `fallback_models` tries backup models on the same provider after
  eligible failures. Failed models cool down for 900 seconds by default;
  `fallback_cooldown_seconds` controls that period. Failover is disabled unless
  configured and does not change the session's selected primary model.
- The Responses compatibility server translates namespaced function/custom
  tools and restores their identities in output and history. Streaming
  Responses include item lifecycle events for text, reasoning and tool calls.
  Custom function adapters require a JSON object containing an `input` string;
  malformed raw arguments remain visible errors.
- Chat/Messages adapters accept plaintext agent messages but reject encrypted
  content with HTTP 501. They do not decrypt messages or make an incompatible
  provider support native agent-message items.

## Python integration and tool migration

Python integrations should update along with the two packages:

- Prefer `AgentsManager(client, context_config, toolset).create_root_runtime()`.
  Direct construction requires
  `Agent(model_client, tool_registry, context_config, session_id)` and
  `AgentRuntime(agent, agents_manager)`. The old `build_agent` and workspace
  `build_runtime` helpers are removed. Register optional Feishu commands with
  `pycodex_ws.register_feishu_commands(runtime)`.
- Load startup configuration with `CodexConfig.from_env(path, profile)`, pass
  its immutable `provider` and `context` projections to their owners, and use
  `dataclasses.replace` for overrides. `build_model(provider)` returns a
  `ModelBootstrapResult`; its caller owns any local compatibility server.
- Custom model clients supply `reasoning_effort`, `recording_metadata` and
  `with_overrides` as well as `model` and `complete`. Runtime controls also
  require the model-control interface. Agent requests carry their session
  identity in `Prompt.session_id`.
- Custom tools inherit `BaseTool` and register instances. Give tools a complete
  input schema so Python exec can expose their parameters. Use qualified
  collaboration names when constructing internal `ToolCall`/`ToolResult` items.
- Stateful tools register their manager and optional event, restart and async
  cleanup callbacks with `environment.register_resource(...)`. The old
  `BaseTool.bind_agent`, `handle_agent_event`, `shutdown` and
  `background_work_count` hooks are no longer called. Supply
  `tools_factory(manager, environment, toolset)` to `AgentsManager` to construct
  each node's tools; registering an instance directly in the parent's registry
  does not define how children construct it.
- Replace the old `send_input`, `resume_agent` and `close_agent` tool workflows
  with the new message, follow-up and interrupt contracts. Session `/fork` and
  the corresponding Agent/Runtime fork APIs are removed; `fork_turns` applies
  to spawning children.
- `request_user_input` remains declared but is unavailable in Default mode.
  Generic Runtime question/permission handling remains available.
- Workspace server/router command names and HTTP/WS endpoints are retained,
  and the router tunnel protocol remains v2. Internal backend modules have
  been split by responsibility; Python monkeypatches targeting old private
  globals need updating. DOM-dependent customizations should also check the
  new agent-tree buttons and transcript views.

## Upgrade

After 0.6.0 is published, install the workspace and its matching core with:

```bash
python3 -m pip install --upgrade pycodex-ws==0.6.0
```

For a core-only installation:

```bash
python3 -m pip install --upgrade python-codex==0.6.0
```

Restart running services after upgrading, then refresh the browser. Python
support remains **3.6.2+ for core** and **3.7+ for workspace**. Both packages
remain ordinary universal wheels. Publish `python-codex` first, then
`pycodex-ws`, which depends on exactly `python-codex==0.6.0`.
The Feishu SDK is still installed manually; the VS Code extension is distributed
and versioned separately.

See the [runtime contracts](RUNTIME.md), [configuration guide](CONTEXT.md) and
[workspace API reference](../packages/pycodex-ws/workspace_server/WORKSPACE_API.md)
for the detailed interfaces.
