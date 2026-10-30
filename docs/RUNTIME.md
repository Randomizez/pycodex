# Runtime state and failure semantics

The runtime keeps state with the component that owns the work. This document
describes the Python implementation, not a claim of newly verified upstream
wire parity.

## Package layout

`python-codex` is the core distribution name; its package directory, imports
and CLI remain `pycodex`, directly under the repository root (no `src/` layout).
Keep two ordinary Python distributions: core and the exactly version-matched
`pycodex-ws` extension. Do not add extras or plugin discovery, or install a
module or command from both wheels. Publish core before workspace.
The VS Code extension remains a separate VSIX in
`packages/pycodex-ws/extensions/vscode/`, sharing the workspace composer but
excluded from both Python wheels and sdists.

The top level of `pycodex/` contains eight core modules: `agent`, `runtime`,
`model`, `context`, `protocol`, `events`, `bootstrap` and `cli`, plus
`__init__.py` and `__main__.py`. Supporting code uses these import paths:

| Module | Responsibility |
| --- | --- |
| `pycodex.tools.runtime_services` | Plan, sub-agent, user-input and permissions services |
| `pycodex.portable.client` | Codex home packing, upload, download and restoration |
| `pycodex.portable.server` | Local encrypted-bundle storage service |
| `pycodex.utils.compat` | Python and asyncio compatibility helpers |
| `pycodex.utils.doctor` | Configuration and connectivity diagnostics |
| `pycodex.utils.model_metadata` | Loading and looking up `pycodex/prompts/models.json` |
| `pycodex.utils.debug` | Debug directory and debugger helpers |

Imports of moved internal modules must use these paths; the old top-level
modules are removed. Root exports such as `from pycodex import Agent` and
`pycodex.debug()` remain available. `pycodex.portable` exports its public client
functions, constants and `RemoteStorageError`; private client helpers live in
`pycodex.portable.client`.

`pycodex.utils` loads only low-level helpers during package initialization so
the asyncio compatibility patch runs before core modules load. Compaction
helpers and constants are imported directly from `pycodex.utils.compactor`;
`compact`, `DEFAULT_COMPACT_PROMPT` and `SUMMARY_PREFIX` are no longer exported
from the `pycodex.utils` package entry point.

Portable remains entirely in core. Start its server with
`python -m pycodex.portable.server`; `pycodex --put`, `pycodex --call` and
`pycodex doctor` keep their existing command syntax.
The startup-only storage boundary, wire contract and smoke procedure are in
[Portable configuration](PORTABLE.md).

## Ownership

The main upstream mappings are `submission_loop` → `pycodex/runtime.py`,
`run_turn` / `run_sampling_request` → `pycodex/agent.py`, and `ToolRouter` →
`pycodex/tools/base_tool.py` plus the individual tools. Preserve these ownership
boundaries instead of adding another Session wrapper or `SubmissionQueue` alias.

| Component | Owns |
| --- | --- |
| `Agent` | Serial execution, local admission and cooperative stop flags, final shutdown, history, usage, context, rollout, identity and model/tool dependencies |
| `AgentRuntime` | Frontend interfaces, session commands, enqueue/mailbox/trigger state, receipts, title/state snapshots, subscriptions, interactive requests and worker lifecycle |
| `ToolRegistry` | Tool instances and their session-local runtime environment |
| `AgentRuntimeEnvironment` | Tool resource bindings, lifecycle, background work, goal continuation, plan/input services and restorable manager state |
| `AgentsManager` | Construction and registration of the whole Runtime tree, task paths, message delivery, lifecycle observation and recursive close |
| CLI / web / Feishu | Input transport and event rendering; no command interpretation or direct Agent control |

`Agent.execution_snapshot()` derives immutable running, stop, usage and session
facts from current execution state. Runtime shares common snapshot fields while
preserving distinct session-history and live-state projections. Slash parsing
is pure; each pending input request owns its question progress and answer mapping.

Construction takes four required arguments:
`Agent(model_client, tool_registry, context_config, session_id)`. `ContextConfig` includes
cwd, instruction overrides and extra contextual messages
as well as configuration-file values. The Agent constructs its own
`ContextManager` with the client's model identifier, without modifying the
caller's config. Context managers are never implicitly supplied or shared between
Agents. Standalone callers use `ContextManager(config)` and load configuration
through `ContextConfig.from_codex_config`.
`ContextManager.config` exposes the frozen configuration for child construction;
model and cwd changes still use the context operations that maintain its caches.

Agent, runtime, tool-manager and workspace-manager construction does not require
a current event loop. Async locks and conditions are created when their async
operations first use them; the runtime queue event belongs to its worker.
Each Agent turn or manual compaction creates its own completion event and releases
it when the operation ends. This supports constructing objects after
`asyncio.run()` has closed a loop, or before the eventual execution loop starts,
including on Python 3.8. Running workers and active resources still belong to
their execution loop; Web subscription queues are created in the consuming
loop when `subscribe()` is called. Unified exec uses ordinary locks only around
synchronous data updates, with no `await` inside those sections. Code-mode cells
and their task ownership belong to the event loop that executes them.

Mailbox waits finish cancelling their event waiter on timeout or caller
cancellation. Python 3.6's
`asyncio.wait_for()` returns before that cleanup completes, so this path uses
`asyncio.wait()` with explicit waiter cleanup to keep later notifications usable.
Only the waiter is cancelled; child turns continue running.
The Python 3.6 asyncio compatibility layer also supplies `all_tasks()` with the
same unfinished-task filtering used by modern Python, including during workspace
thread cleanup.

There is no collaboration-mode switching state. The model catalog's Default
developer instructions are retained when present, alongside separate MultiAgentV2
role and operating-mode blocks. CLI and workspace use the same context-building
path. `request_user_input` retains its declaration but
always returns `request_user_input is unavailable in Default mode`; registered
input handlers do not enable it. The tool has no request-manager dependency.
The generic runtime input service, sub-agents, `update_plan` and ordinary
interactive sessions remain available.
Do not restore the old collaboration-mode enum or `session_mode` parameter.
MultiAgentV2's explicit/proactive operating mode is a separate context contract;
see [Context](CONTEXT.md#multiagentv2-developer-context).

`model_client`, `tool_registry`, `context_manager` and `event_handler` are ordinary
attributes. `AgentRuntime` holds its Agent and subscribes to events, but the Agent
has no runtime/queue reference, input-source callback or submission lifecycle API.
Control flows from Runtime to Agent; events are observations, not instructions to
settle runtime receipts. There are two backend abstractions, not an additional
Session wrapper. Queue data remains internal to Runtime.

## Frontend boundary

```text
CLI / Web / Feishu -> AgentRuntime -> Agent
VS Code -> workspace_server HTTP -> AgentRuntime
```

`pycodex.bootstrap` builds the model client. `pycodex.tools.get_tools` constructs
a registry with an explicit manager and per-Agent environment.
`AgentsManager(model_client, context_config, toolset)` constructs and registers
the root through `create_root_runtime()` and all descendants through the same
pipeline. `AgentRuntime(agent, agents_manager)` provides core commands.
`pycodex_ws.register_feishu_commands(runtime)` adds `/link`, `/unlink` and
connection cleanup for workspace sessions; the Feishu adapter is imported only
when linking. `/link` first attempts to import the optional `lark_oapi` SDK.
An import failure reports that the command is unavailable and gives a manual
installation command, before reading credentials or creating a card or connection.
The same `AgentRuntime` can register
application-specific async handlers with `register_command`.

The `python-codex` distribution owns `pycodex`, `responses_server` and the
`pycodex` CLI, including all portable functionality. `pycodex-ws` owns
`workspace_server`, `pycodex_ws`, `pycodex_router`, viewer resources and the
`pycodex-ws` / `pycodex-router` commands.
Its source lives in `packages/pycodex-ws/` and it depends on the exact matching
core version. The Feishu SDK is installed manually with
`python -m pip install lark-oapi`; it is not a package dependency.
The workspace package requires Python 3.7+; core keeps Python 3.6.2 support.

Tests follow the same boundary: root `tests/` runs with the `test` dependency
group and no workspace or SDK installation; `packages/pycodex-ws/tests/` runs
with `dev` on Python 3.7+. Pure Runtime tests stay in core, while Web, Feishu
and cross-frontend integration tests belong to workspace. Both dependency
groups omit the Feishu SDK; SDK-dependent tests use fakes.
The VS Code extension in `packages/pycodex-ws/extensions/vscode/` is a workspace frontend with its
own Node tests and VSIX build, separate from both Python distributions.

Both HTTP services use Starlette and Uvicorn directly, without Pydantic.
Workspace routes parse JSON objects explicitly and return HTTP 422 with a
`detail` message for invalid bodies or content types. HTTP errors retain their
JSON `detail`, status and headers. File/Git view operations run in the thread
pool so they do not block session polling or WebSocket delivery. Startup and
shutdown continue to start and close the same runtime managers.

Portable AES-256-GCM runs through `pycodex.utils.crypto`, using the bundled pure
Python subset of pyaes and tlslite-ng. It uses no optional crypto backend or
native build. The `PCX1` layout, SHA-256 key derivation, 12-byte nonce and 16-byte
tag are unchanged. Tags are verified before ciphertext is decrypted. Core
ships a universal Python wheel and has no PyCryptodome runtime dependency;
large bundles incur the cost of Python encryption and authentication loops.
See the [vendored source notes](../pycodex/utils/_aes_gcm/README.md).

Every frontend calls `await runtime.submit_input(text, sender)`. The returned
`SubmittedInput` carries a submission id, kind and future. Normal text uses
steer; `/queue <text>` uses the ordinary queue. The backend parses `/help`,
`/history`, `/title`, `/cd`, `/model`, `/effort`, `/resume` and `/compact`.
The session `/fork` command is removed and returns an unknown-command error.
`/cd` reports the session directory; `/cd <directory>` resolves a directory
against the current one (with `~` expansion) and requires idle, accepting state.
It updates the Agent's context and broadcasts `cwd`, preserving the latest usage
so the next turn can still trigger auto-compaction. Each tool invocation receives
the current directory as the required `ToolContext.cwd`; existing exec processes and code cells
keep their original directory. New subagents inherit their parent's current
directory. No process-wide `chdir` is used. `/resume` retains the current directory.
Configure the initial directory once in `ContextConfig(cwd=...)`.
Tool constructors and `get_tools`
do not store or accept a default directory. Standalone tool calls supply
`ToolContext(turn_id, history, cwd)`, where `cwd` is an absolute `Path`;
`UnifiedExecManager.exec_command(cmd, cwd, ...)` likewise takes its directory
per invocation.
`/resume` lists recent sessions; `/resume <number>` selects from that list,
while `/resume <session_id>` resolves a full UUID independently of the list
limit, including archived sessions. Both use the same idle-only restore path.
The list includes only recordings with a real user message, excluding
metadata-only empty sessions. Ordinary recorded sessions use UUIDv7 identities
and append to `CODEX_HOME/sessions/.../rollout-*.jsonl`.
Each queued request is one private record of submission id, turn id, texts and
futures. The unused public `Submission` / `UserTurnOp` wrappers are removed;
callers submit frontend text through `submit_input`, or independent turns through `enqueue`.
The workspace keeps queued submissions out of its displayed conversation.
`TurnStartedEvent` adds the actual user texts in execution order, including
coalesced steer inputs; enqueue admission alone does not create a user block.
The composer previews pending Steer and Queue text in small capsules beside
the status pill. The preview comes from existing queued submission metadata.
Commands return structured results and never become user messages. Unknown
commands and invalid arguments produce explicit command errors. Destructive
commands reject active or queued work; a command lock serializes asynchronous
commands and lets close wait for accepted command work.

The [VS Code frontend](../packages/pycodex-ws/extensions/vscode/README.md) starts a workspace backend
per project using the existing CLI. Before startup it reads package metadata
from the selected Python interpreter. Automatic detection resolves an existing
console script's interpreter or finds Python on the workspace host. A compatible
`pycodex-ws>=0.4.0` is reused without a network check; missing or older workspace
packages are installed/upgraded through that interpreter's pip, which resolves
the matching core dependency. It launches `python -m workspace_server`
with the same interpreter, including when a selected venv initially has no Pycodex.
Installation finishes before the backend startup deadline starts. Pending
installs are shared per Python environment, and closing during preparation waits
for pip before discarding the launch. No private environment is created.
The extension stores one `workspaces.json` per canonical project path under its
host's private storage directory. It uses `state_path: "workspaces.vscode"` and
`enable_board: false`; the configuration and session index survive shutdown.
Opening the sidebar restores saved conversations; projects without saved state
stay on the welcome screen until a chat is created.
It uses the backend host's
`~/.codex/config.toml` by default; the welcome screen explains the local/remote
configuration source. The header settings menu selects project `.venv`/`venv`
interpreters or browses for another Python, and edits the Profile name.
Selections are validated without installing packages and saved to the project's
`pycodex.pythonPath` and `pycodex.profile` settings. A profile adds
`--profile <name>` to the launch. Changing either setting through the menu
offers a window reload if the backend is running; existing sessions keep their
startup settings until it restarts.
Sessions have no source-file binding or
automatically attached editor context. On request, the frontend captures the
source file and one-based cursor/selection. Nonempty selections include a bounded
excerpt of the selected text; an empty selection records only the location.
The location is either a selection range or a cursor position, with no language
field. Web context uses the view's complete title as its sole source label,
preserving both endpoints and full revisions of a diff.
Explicit Explorer/SCM file selections include bounded file contents.
Read-only Git diff documents and Explorer/SCM file selections use VS Code's
document API. Context appears as atomic tokens interleaved with ordinary text in
the composer. Insertion uses the saved cursor/selection; clicking a token edits
its content. Sending expands text and tokens in their original order into the
ordinary `prompt` field.
Commands and pending answers use
the same existing HTTP API.
Connections are keyed by workspace endpoint and session id.
Each new chat creates a session; clicking a collapsed entry
activates that chat. A stable WebviewView in the Pycodex activity-bar container
displays one expanded chat and collapsed entries for the others, preserving each
chat's draft. All opened entries retain live status while the
sidebar is visible; hiding it stops observation without closing sessions.
Model changes use typed `/model` commands; the model label is read-only.
Closing an entry uses the existing session DELETE endpoint and waits for normal
backend stop and cleanup before removing the entry. HTTP snapshot polling uses the
existing Web projection. In remote workspaces, the extension host starts Python
and makes HTTP requests on the remote machine; the webview only exchanges
messages with the extension host.
The input footer's context ring uses the existing snapshot fields
`context_remaining_percent`, `usage_tokens`, `auto_compact_token_limit` and
`max_context_window`. It shares the Web meter semantics below, including compact
thresholds, unknown usage and reset after compaction. Hover/focus shows exact
values, click pins them, and Escape, outside click or session switching closes
the details. The frontend does not estimate draft tokens or recalculate limits.
The launcher writes UTF-8 workspace JSON into a private temporary directory
and passes the filename through the existing `--workspace-config` option.
It removes that directory after backend exit or failed startup; an abrupt host
crash can leave the temporary files behind. The `pycodex-ws` CLI always
shuts down on stdin EOF. Stdin stays open for the backend's lifetime.
EOF on folder removal, deactivation,
or extension-host death starts normal server shutdown, including when EOF arrives
before startup finishes. Runtime closure interrupts pending turns and stops
active turns cooperatively across each session tree; no turn timeout or forced
cancellation is added. Deactivation waits for
all owned processes to exit. A temporary remote disconnection leaves sessions
running if their extension host remains alive.
Other CLI launchers must also keep stdin open; closed stdin (including
`/dev/null`) requests shutdown. Programmatic Starlette app construction does not
read stdin.

Web command output preserves the event's plain text, including resume-list
numbering and line breaks.

Web and VS Code restore the latest 80 display entries from the saved rollout,
including compact dividers and earlier questions/replies within that window.
Compaction adds a divider to the existing conversation list. Display restoration
reuses the rollout parser; model context still resumes from the latest checkpoint.

In the workspace's single-pane phone layout, the conversation header retains
the active session title and running status. Board/chat switches occupy the
same top-right position in their respective headers; the board switch follows
the fullscreen button in the board toolbar. Floating session controls stay below
the headers and contain only the session-list button and status dots.

In the two-pane layout, the session sidebar starts collapsed on every page load.
Hovering for 100 ms previews it; leaving schedules dismissal after 150 ms,
and returning cancels dismissal. Clicking pins it open; another click on the
toggle or empty sidebar space, or Escape, closes it. The chat and sidebar share
a fixed total width: expansion takes space from the chat without moving the
board or splitter. Dragging the splitter resizes that combined region;
double-clicking balances the visible board and chat. Both panes remain
interactive; only the phone drawer is modal. The composer follows chat-width
changes so wrapped input keeps the correct height. Replies remain readable
beside the sidebar; replies covered by the phone drawer are marked read after
it closes.
Hover timers are cleared by explicit actions, mode changes and page hiding.

The empty Web composer shows rotating usage tips while the selected session is
running and is not waiting for user input. `updatePromptPlaceholder` keeps the
editable `hints` string list in `workspace.html`; the existing snapshot refresh
advances the list about every six seconds without a separate timer. Each session
randomly selects its starting hint when work begins, then proceeds in list order.
It tracks its own start time; idle or waiting states reset it and restore the model
placeholder. Tips occupy a separate, non-editable layer over the empty composer,
so the caret starts at the beginning of the input and tips never become draft
contents or submissions. They wrap inside the composer, stay hidden behind text
or context-token drafts, and leave the input's accessible model label unchanged.
Keep each hint short and self-contained; do not split an instruction across
successive hints. The list covers commands, frontend customization, workspace sharing,
inter-session collaboration and optional features present in the packaged
catalog; optional feature hints use “在设置中添加：…” to indicate how to add them.

Status dots, activity spinners and thinking borders use the same 1.2-second
breathing cycle and easing. The workspace, workspace list and router set these
CSS animations to the document timeline origin when they start, so indicators
created or revealed later share the page's current phase. Reduced-motion
preferences disable these animations.

Assistant replies render fenced `mermaid` blocks as diagrams. The browser loads
the renderer on demand, uses strict mode and sanitizes the generated SVG.
Diagrams fit the chat column; `Expand` opens a larger view with an `Actual size`
option for scrolling through wide diagrams. The original code remains available
under `Diagram source`. Loading or syntax errors leave the source visible.
A render that finishes after its conversation view has been replaced is discarded;
completed diagrams keep the view at the bottom only if it is already there.

Closing the active workspace session tab uses the same browser transition as selecting
another tab: reset that tab's render signature, restore its draft and scroll
position, and request its snapshot immediately. A pending poll for the previous
tab is aborted; late responses or errors cannot overwrite the selected tab.
Clicking the active tab's title edits it in place; clicking an inactive tab
first selects that session. Enter or leaving the title field saves through
the existing `/title` command; Escape cancels, and an empty field leaves the
title unchanged. Polling preserves the title editor and its selection while
updating other tab state. A save targets the edited session even if another
tab is selected before its response arrives.
Selected tabs use a tinted background without a click-focus border; keyboard
focus on the title uses an underline.
The browser page title gains a `●` prefix while any session in the workspace
is running, using the same session summaries as the tab activity dots. It returns
to the original workspace title when all sessions are idle.

The workspace presents assistant replies as open text and user messages as
right-aligned bubbles. Commands and thinking use quieter secondary styling;
active thinking has a breathing left rule that respects reduced-motion settings.
The Web and VS Code inputs share the text/token composer in
`packages/pycodex-ws/workspace_server/frontend/composer.mjs`. The Web controller
bundles it; the extension's compile command copies it into `out/composer.mjs`.
Context tokens occupy one cursor position, support editing, deletion,
copy/paste and undo, and stay in each session's ordered draft until sent.
The compact composer keeps Enter to send and Shift+Enter for new lines, with
a larger context ring centered alongside multiline input. Its placeholder reads
`Message to <model>…` and follows the active session and model changes.
The Web context ring shows `1 - current / compact_limit`, using the latest
reported `total_tokens` and the same resolved limit as Agent auto-compaction.
Its integer percentage rounds up: 0% means the limit has actually been reached,
and compaction runs at the next pre-turn or tool-follow-up request boundary.
Hovering, focusing or clicking the ring reveals exact current tokens, compact
limit and maximum context length (the config override or model metadata value,
before effective-window scaling). A click pins the details; another click,
an outside click or Escape dismisses them. With auto-compaction disabled, the
ring instead uses the maximum context length and the details say `Off`.
The details also show the live Agent tree and an `n running · m failed` count
across the root and all retained descendants. Each name shows `for x min`,
the current or most recent turn's elapsed time in minutes to one decimal place.
Runtime records turn start and end using a monotonic clock; the duration freezes
on completion, failure or interruption and resets when another turn starts.
Before the first turn it is zero. Errored rows show the error from their
status in red, with wrapping and the full text available in a tooltip, even
when the Agent has an earlier assistant message.
After compaction or history restore, current usage is unknown until the next
token report; the ring resets and the details show that usage is pending.
Queue hints use neutral backgrounds and running status uses a small activity dot.
They share
the composer's outer frame in a compact top row separated from the text input.
The row also shows a compact folder indicator when the session directory differs
from the workspace default, with the full path in its tooltip. It collapses when
there is no running status, queued input or directory override.
On screens up to 760px wide, icon buttons in the existing board/session headers
switch panes without a separate navigation row or changes to the active session
and draft. Desktop resizing and per-session scroll restoration retain their
existing behavior.

Frontends subscribe with `attach(handler)` and unsubscribe with `detach(id)`.
Attach delivers an immediate `session_state`: identity, model, title, cwd, canonical
history, active turn/partial output, usage, plan, input request and admission
state. Model/history/title/directory changes are broadcast to every attached frontend.
`command_completed`, `command_failed`, `input_queued`, `input_requested`,
`input_resolved` and `session_closed` complement ordinary Agent events. Command
results are separate from model turns and are not recorded as conversation
items. Event presentation and per-subscriber display state live in `events.py`;
stateless text/color/summary helpers live in `utils/event_helpers.py`.
The terminal I/O executor, input loop and CLI startup
live together in `cli.py`. Session command
dispatch stays inside `AgentRuntime` rather than a separate command module.
The CLI renders backend history without keeping a second transcript or a
blocking question-input path. JSON receipts use future callbacks, not per-turn
tasks. Web renders each command's lines as one control message.

Generic question/permission handlers belong to the backend, not the terminal.
While an input request is pending, non-command text answers the current
question or permission request. Numbered choices, `0` followed by free text,
multiple questions, blank-answer cancellation and permission scope work
identically from all three frontends. Structured clients can instead call
`answer_input(request_id, answer)`; stale ids fail explicitly. Slash commands
remain commands during a question. A timeout, final frontend detach or shutdown
resolves a waiting request as cancelled. With no attached frontend, the tool
returns its existing cancelled/unavailable result rather than waiting forever.
The Web message endpoint also accepts `{"request_id": "...", "answer": {...}}`
(or `answer: null` to cancel), and its WebSocket accepts the same fields with
`type: "answer"`. The browser's ordinary text form uses the same input path and
allows a blank submission to cancel a pending question.

Previously recorded question answers replay as a JSON string in
`function_call_output.output`. `ToolResult.success` is local execution metadata;
serialization omits it from Responses input and new rollout response items.
Older rollout entries can still load that field, but replay never sends it to
the API.

`start()` is idempotent and owns the one queue worker. `close()` stops admission
across its Runtime subtree, interrupts pending turns, requests cooperative stop
for active turns, resolves waiting input, and runs injected connection cleanup.
It never cancels an Agent Task or an issued model/tool operation. Detach alone
does not close the backend. The CLI process and workspace session manager close sessions they
own; unlinking a Feishu card only detaches that frontend. Web submits directly
to the backend, including across its worker-thread boundary: it does not run a
CLI shell or maintain an extra prompt queue.

Workspace state stores every open session tab in order, starting when the tab is created.
An empty tab stores an empty title and rollout path, so restoring it creates a
fresh empty session without writing a rollout. Manual titles and first messages
are not prerequisites for keeping a tab across a server restart. Changes to
automatic or explicit titles and recorded paths update the same saved tab.
Startup restores the full tab list before saving it; closing a tab removes its
entry.
Each tab also stores its effective `profile` name, or `null` for the base
configuration. Restore uses that selection even if the launch default changed.
Legacy tabs without this field use the current launch default. Profile names
resolve against the current config; the state file does not copy configuration
or credentials. A removed profile fails restoration instead of silently changing
the session's provider. Resume retains the tab's profile.
Saved tabs point only to a successfully recorded or resumed rollout. An empty
titled tab saves its title with an empty rollout path. Legacy pending-fork
entries are rejected explicitly; startup leaves the state file intact.
An invalid saved source path raises an error and closes the partially created
session. Only root sessions are restored; descendant runtimes remain in memory.
`Agent.recorded_session_file_path` and the snapshot's `recorded_rollout_path`
identify a successfully written or explicitly resumed recording. A lazy
destination has no recorded path yet. The runtime broadcasts a `recording`
state change after the first successful write; an unrelated file at the
destination does not count as that write.
The workspace manager keeps its persistence subscriber attached while sessions
stop during close, then waits for the final events before releasing its state.

The queue has no separate public `shutdown()` or worker-start entry point.
Do not restore `run_forever()`, `begin_shutdown` or a separate `draining` state.
`close()` asks `AgentsManager` to signal every Runtime in the subtree before
awaiting any of them. Each Runtime switches off `accepts_input`, calls
`stop_asap()`, clears its enqueue/mailbox queues and pending trigger, resolves
discarded submission futures with `SubmissionInterrupted`, and wakes its worker.
The current submission reaches its cooperative stop boundary before cleanup;
there is no special shutdown submission. The worker's result carries
cleanup errors for concurrent and repeated close callers. Closing also works
before `start()`. A cancelled close caller does not cancel the worker.
Cleanup visits every child and connection handler even if one fails; the first
error is propagated and additional errors are reported through the event loop.
The interactive prompt treats Ctrl+C as end-of-input, just like Ctrl+D on an empty
prompt, and calls `runtime.close()`. The prompt toolkit
`c-c` and `<sigint>` bindings exit with `EOFError`, including on version 3.0.36
which has no `interrupt_exception` constructor parameter, so an input task cannot
leak `KeyboardInterrupt` into the event-loop runner. One Ctrl+C initiates normal exit;
the admission event renders a `[closing]` notice while active turns stop and
cleanup finishes. Pending turns do not start. Issued model/tool operations finish
and persist their results before the turn stops.
While closing, another Ctrl+C immediately terminates the CLI process with status
130. This also works after Ctrl+D starts closing. The CLI
installs a SIGINT handler for its interactive session and reinstalls it after
input and at the close boundary because older prompt_toolkit versions reset it.
It therefore does not depend on an inherited handler that may ignore SIGINT.
Normal completion and cleanup errors restore the
original handler. Forced exit happens at the process boundary, without cancelling
the turn or re-entering asyncio shutdown waits. It skips remaining cleanup and
does not guarantee termination of external tool processes. Runtime close itself
still waits for issued operations and cleanup.
The prompt uses `set_exception_handler=False` so prompt_toolkit does not replace
the event loop's exception handler with its blocking "Press ENTER to continue"
screen. Background notification failures (including provider rate limits) remain
visible through the existing loop handler without pausing input.
The CLI still detaches its view on cleanup
failure, and the Web adapter stops and joins its thread after session cleanup,
including on failure; startup failure closes the partially started session.

Direct Agent callers can use `ask` and `run_turn` without a submission queue.
Exec and clock wake-ups require the environment's explicit Runtime callbacks.
This does not make arbitrary dependency replacement a supported mid-turn
operation. `history` remains an immutable snapshot; model-name and admission
properties still derive their values from the owning component.

### Recording lifecycle

`session_file_path=None` is the default and means no recorder: history stays in
memory. Passing a path creates an internal recorder. Session identity is
independent of recording; callers supply the required session id.
`AgentsManager` supplies a UUIDv7 id and default recording path for ordinary frontend
sessions, using the configured Codex home. Sub-agent construction leaves the path
as `None`, including for nested children and forked context.

For recorded sessions, construction prepares metadata without creating the file
or its parent directories, even with initial history. The first nonempty history
append or successful compaction writes metadata, the Agent's current history and
the new records in order. An empty append or empty compaction does not create a
file. An explicit constructor path is a new-file destination: an existing path
is rejected at construction, and the first write still uses exclusive creation
to reject a destination created in the meantime, never an implicit load or overwrite.
The recorder does not keep a second initial-history copy, so replacing history
before the first write changes what gets recorded.
The recorder opens and closes the file for each append, so there is no persistent
file handle requiring another shutdown task.

Recorded Agents save the environment's restorable manager state alongside the
rollout in `<rollout-path>.state.json`. The environment aggregates manager-owned
data; tools have no persistence interface. The current version records the
clock's `period_m` (`null` disables it) and optional goal state, not timer Tasks,
callbacks or deadlines.
The Agent saves after committing a tool result and before tool shutdown. The
recorder skips unchanged snapshots and atomically replaces changed snapshots;
write errors propagate, and final tool cleanup still runs after a save failure.
Unused lazy recordings and unrecorded Agents, including children, create no
state file.

Explicit resume reads this small snapshot independently of model history and
compaction. A missing snapshot starts with the clock disabled; malformed or
unsupported state fails before replacing the current session. Restoring data
does not start a timer. Runtime startup, or resume in an already-running Runtime,
starts a fresh full period after binding callbacks. No-argument resume retains
the in-memory configuration. Session snapshots count current background work
from registered resources; attach/history presentation restores `idle: sleeping` for
an armed clock without requiring another reply. Resuming a session without
background work clears that status. Keep the snapshot alongside its rollout when
copying a recorded session; Portable configuration bundles exclude both.

Unrecorded sessions update only in-memory history during turns and compaction.
Compaction omits the rollout-file reference from the handoff, and runtime
snapshots expose `rollout_path=None`. No-argument resume preserves the current
recorder. Explicit `resume(path)` loads
the selected file and enables recording to that file.

The outer queue remains optional: direct `ask` / `run_turn` calls need no worker.
Empty initial history and the no-op observer remain valid defaults. Tests isolate
`CODEX_HOME` per test so recording tests do not write to the real session store.

### Turn execution and shutdown

Turn execution has two public interfaces: synchronous `ask` and ordinary async
`run_turn`, both returning `TurnResult` once execution completes. `ask` runs the
same coroutine through the existing synchronous bridge. There is no separate
`start_turn`, `_start_task`, `_run_turn`, active Task reference, Task listener or
Agent cancellation API.

Creating a `run_turn` coroutine does not start work. The host may schedule it,
but the Agent itself simply executes in its caller's coroutine. Queue submissions
await `run_turn` directly. `AgentRuntime.maybe_invoke` returns `False` while busy
or closed; otherwise it formats the notification, starts the worker, delivers a
message, triggers execution, awaits its receipt and returns `True`.
Failures propagate to the invoking caller; exec/clock notification sources
report their own failures. When steer ends a notification turn with
`SubmissionInterrupted`, those sources treat it as a normal interruption and do not
report a notification failure. Runtime still owns the pending user input, and
the clock waits for the next successful reply before arming again.
Background events are not queued or cached.

One idle event is cleared on entry and set in `finally`, covering pre-turn
compaction, sampling, tool execution and manual compaction. `is_running` derives
from that event, and `wait_until_idle` waits on it without polling or reading an
execution result. Overlapping execution is rejected; there is no pre-sampling
Task reservation or cancellation policy.

The toolcall loop reads in order: auto-compact if needed, sample the model, commit
its output, then execute tools or finish. Follow-ups and tool handoffs determine
when a turn ends; do not add a fixed iteration cap or a local iteration-limit
parameter. Threshold-triggered compaction has one
call site at the top of the loop: before the first sample it compresses previous
history and then appends the new user input (`pre_turn`); on tool follow-ups it
compresses the current history (`mid_turn`). Its `_TurnState` holds only the
current turn id, logical iteration and latest assistant text; it is local to
that call, not another admission lifecycle or queue.

Context-overflow recovery belongs to `_sample`: compact once, prepare the next
request again, and retry that sample once. A failed compact or second overflow
propagates without re-entering recovery. Network retries remain inside the
provider. Recovery keeps the same logical iteration and turn id and never
replays tools, unless an explicit cooperative stop ends that execution.
The bounded attempt loop checks for a stop before either request, including
after overflow compaction. Only its first actual attempt advances the iteration;
a stop during pre-turn or follow-up compaction does not count an unsent sample.
The other stop boundary is after model output and all issued tool results have
been committed. There is no separate pre-sampling wrapper or stop-check helper.

Runtime exposes four coordination primitives:

- `enqueue(text)` schedules an independent turn and returns `(submission_id, future)`.
- `message(str_or_agent_message)` queues mailbox input without waking an idle Agent.
  An active Agent consumes it at the next safe input boundary and samples again.
- `trigger()` requests execution and merges repeated pending triggers. It returns
  a receipt tuple, or `None` when execution is already active without a stop request.
- `stop_asap()` requests a cooperative stop and wakes a pending mailbox wait.

Frontend steer combines stop, message and trigger. The Agent records the stop
request, finishes the
in-flight model request and all tool calls already issued in that response,
persists their results, then emits `TurnInterruptedEvent` and raises
`TurnInterrupted` before another sample. It also checks after compaction, including
overflow recovery, so a stop requested during that await prevents the next sample.
This never cancels a Task, HTTP request or tool, and an idle stop does not poison
the next turn.

The Runtime worker catches the interruption, resolves the displaced submission
with `SubmissionInterrupted`, and awaits a fresh `run_turn` for the pending batch.
Steer during a runtime-owned turn retains its logical `turn_id`, while each
execution emits its own `turn_started` and resets its iteration count. Pending
steer messages merge until their batch starts; all futures in that batch share
its result. If more steer arrives during the new execution's pre-turn compact,
that execution can stop before sampling too; committed inputs remain in history.
Ordinary `/queue` input does not request a stop.

Direct `run_turn` callers receive `TurnInterrupted` if a Runtime requests a stop.
Runtime delivers messages through `TurnInputReadyEvent`, including during direct
Agent execution. This synchronous coordination propagates delivery errors to the
turn; ordinary event observers report errors to the loop without failing the
turn. The Agent does not own or complete Runtime receipts; executing
enqueued or triggered turns requires `runtime.start()`; `close()` discards
pending turns.
There are no separate
input-inserted/submission-interrupted events: presentation uses the actual
turn-started/interrupted lifecycle and the existing queued-input feedback.

The queue is used from its owning event loop. Enqueue/dequeue mutations do not
await; asynchronous session commands and final resource cleanup use the command
lock. Cross-thread callers must
schedule operations on that loop rather than mutate the queue directly.
Worker notification is scheduled after enqueue returns, allowing cross-thread
submission receipts to be delivered before model execution can block that loop.
Deferred queue notifications stop once the Agent is shut down, so a worker that
closes immediately cannot publish updates after its child Runtime is unregistered.

`runtime.is_busy` includes pending submissions, direct Agent work and closing
resource cleanup. Destructive
interactive commands such as resume/model/compact use that same state.
Exec/clock notifications require an idle Runtime with no queued work.

`Agent.accepts_input` is a local admission flag; `runtime.accepts_input` reads it
without a reverse Agent-to-Runtime dependency. `runtime.close()` clears it across
the subtree, resolves pending questions, discards pending turns and requests
cooperative stop. New submissions, mailbox messages, triggers and child spawns
are rejected; late background outputs and child completion notifications do not
enqueue work in a closing Runtime. `start()` cannot reopen a closing Runtime;
explicit `resume()` is required after shutdown.
The worker then waits for accepted command work and the command lock, closes
children, environment resources and connections, and calls `Agent.shutdown()`.
Environment shutdown runs once at that final boundary,
after accepted work can no longer rearm them. Completion comes from the worker,
not a separate shutdown future. Direct `Agent.shutdown()` disables background
hooks and rejects new calls without interrupting an active call.
Manager owns the closing child references; only Manager calls
`Runtime.request_close()` to signal a node after capturing its children. Callers
close a subtree with `Runtime.close()`. References are captured before any await, so children that
finish and unregister while their parent is stopping are still awaited and
their cleanup errors remain visible. Direct children close concurrently, each
waiting for its current operation and closing its descendants. Closing a child
only signals that child's subtree. All child closes finish before cleanup errors
are reported; this removes serial child-cleanup waits without cancelling turns.
A closed child Runtime is unregistered after publishing `SessionClosedEvent`,
including when resource cleanup or tool shutdown reports an error. Its task path
is released. The root stays registered for the session lifecycle;
`runtime.resume()` without a path reopens a registered Runtime and restarts its environment.
Runtime and workspace close wait for issued model/tool operations to finish.
There is no close timeout that cancels a turn: if a request never finishes,
close keeps waiting.

Sub-agent services observe ordinary Agent lifecycle events. They retain the last
completed/failed status, but active execution and pending inputs take precedence
when reporting busy. Direct calls and background invokes produce the same events;
status observers no longer inspect raw Task results.

### Feishu connection lifecycle

`lark_oapi.ws.client` creates a module-level asyncio `loop` during import,
and `Client.start()` always uses it. A `/link` listener owns a thread and loop;
bind the SDK global to that loop before constructing its client. `/unlink`
disconnects through `_disconnect()` and stops that loop with `loop.stop()`.
Reusing a running SDK loop across unlink/link can raise
`RuntimeError: This event loop is already running`.
Keep this SDK-specific handling inside the connection adapter. Shared rendering
and card reply boundaries are described under [Observation](#observation);
credential isolation belongs in the [workspace tests](../tests/TESTS.md#按包运行).

## Workspace session API

`GET api/sessions` returns the session list (including each tab's `profile`),
the available profile names in `profiles`, and `default_profile`. The CLI
loads the sorted profile catalog from valid adjacent `*.config.toml` files and
embedded `[profiles.NAME]` tables when building each workspace. Duplicate names
across the two formats are errors. Provider and context settings share the
[profile loader](CONTEXT.md#配置读取与-profile).
`POST api/sessions` creates a session with optional string fields `title` and
`prompt`, plus an optional `profile` name. Title and prompt default to an empty
string; requests with no body still create an empty session. An omitted or null
profile uses the launch `--profile`, then the top-level config `profile`, then
the base config. Unknown
names fail before creating a session. The selected profile is passed to both
model and Agent construction on the session's worker thread. CLI overrides
continue to apply. A nonempty prompt is submitted unchanged through the Runtime.
The response includes the new session ID, session list and snapshot. These
fields are general frontend inputs: the API does not select prompts, infer
reply languages or interpret feature-specific flags.

`WorkspaceSessionManager` accepts a base zero-argument session factory plus an
optional `profile_factories` mapping of names to zero-argument factories and a
`default_profile`. Each factory constructs a fresh session. The browser's
**New session** button directly creates a default session. Its right-hand
**+** button (`Custom profile`) opens the profile chooser; when no named profiles
exist, it shows a message without creating a session. Frontend customization
keeps using the default creation path.

The [workspace resource API](WORKSPACE_API.md) adds lightweight Runtime state,
model catalogs, asynchronous commands/results, paginated history, descendant
Agent and background-work inventories, and cursored WS replay. Queries run on
the owning session loop; rollout scans run in a thread pool. WS state and
cursors are captured together in `WebSessionView`.

`AgentRuntime.submit_input(..., wait_for_command=False)` returns an admitted
command receipt without waiting for completion. Runtime owns these command
tasks, serializes them with the same command lock and drains them on close.
The default remains synchronous command completion. These are host command
tasks, not Agent-owned turn tasks or a cancellation API.
`state_snapshot()` excludes completed history and describes queued turns,
active commands and model/effort separately. `RuntimeUpdatedEvent` signals plan,
queue and resource changes; `CommandStartedEvent` identifies an executing
command. Tool managers expose inventories through the explicit
`background_work_snapshot` / `set_state_listener` hooks.

`attach(handler, interactive=False)` observes without counting as an interactive
frontend. Child views use this mode, so inspecting a child cannot make it wait
for a permission/question response. Submission results and event replay are
bounded in-memory stores; retention and reset behavior are part of the linked
API contract.

## Frontend customization

Browser workspace pages use the server OS user's `~/.pycodex/frontend/`,
independently of `CODEX_HOME` and the workspace configuration. A
`workspace.html` in that directory replaces the packaged template. Only a
missing file selects the package default; read failures remain errors.
Optional `custom.css` and `custom.js` are inserted after the page's styles and
module, respectively, including when using the packaged template. Lazy-loaded
editor styles are inserted before `custom.css`, preserving that order when a
code view first opens; normal CSS specificity still applies. The script is an
ES module; page module variables remain private.
An optional `mount(host)` export receives the built-in page's versioned
session/draft/view interface. Side-effect-only scripts still load; full template
replacements own their initialization. See the
[host contract](../packages/pycodex-ws/workspace_server/WORKSPACE_API.md#built-in-page-extension-host).

Custom templates retain closing `head`/`body` tags and the placeholders
`__WORKSPACE_TITLE__`, `__BOARD_LABEL__`, `__CWD_LABEL__`, `__ENABLE_BOARD__` and
`__BOARD_URL__`. The page also receives `__FRONTEND_DIRECTORY__`,
`__PACKAGE_DIRECTORY__` and `__PACKAGE_VERSION__` as escaped body data attributes
for frontend code to use. API and packaged asset URLs stay relative to the workspace.
`__ACCESS_ROLE__` supplies `admin` or `visitor` for hiding management controls.
The `frontend/<path>` route serves supporting files from the customization
directory with normal CSS URL/module import resolution. Paths and symlinks
cannot escape that directory; directories are not listed. Existing
`assets/content_views.*` and `assets/editor.*` continue to serve package files.
Custom pages and resources use `Cache-Control: no-store`; refresh applies
edits without restarting the server.

`?builtin=1` on the workspace page bypasses the template override and both
automatic additions, even if custom files are unreadable or invalid. It does
not delete files or change sessions; removing the parameter restores custom
loading. The recovery page shows that customizations are disabled and links
back to the customized interface. Authentication applies to recovery pages
and custom resources just as it does to the rest of the workspace.

The gear button in `workspace.html` opens Settings, with icon rows and short descriptions
for Customize Agent and Share. Customize Agent assembles the `Customize interface` title and introductory
prompt, then passes both to the ordinary session creation API. Both settings
entries and the management navigation are hidden for workspace visitors.
The prompt and its development guidance belong to the editable frontend;
the Python customization module only loads templates and serves their resources.
The button uses the page's path metadata and `navigator.languages`
in preference order. Its context includes the installed `WORKSPACE_API.md`
path, API implementation paths, and a quick reference for the host,
state/models, commands/results, history, descendant/background inventories and
WS replay/reset. The full API reference lives beside `workspace.html` and ships
in the wheel and sdist; it does not require a source checkout. Reasoning guidance
distinguishes recorded readable content from the display `thinking` field and
does not promise live reasoning deltas. Feature reference code is adapted
into a coherent frontend, without requiring a plugin registry or runtime
composition system.
The packaged `optional_features/README.md` is a Markdown feature catalog opened in the
existing view area; users choose features by name or ID in ordinary chat text.
There are no additional selection controls or chat commands. The context links
the [feature contribution guide](../packages/pycodex-ws/workspace_server/optional_features/CONTRIBUTING.md),
which defines the target repository, per-feature source layout, README metadata,
validation evidence and GitHub PR title/body format. On a publication request, the
Agent prepares the scoped contribution in a separate checkout and submits it
subject to the user's commit/push constraints. Catalog entries describe source
actually present in the library; installed catalogs ship with the package and
do not fetch remote entries automatically.
The supplied feature panels occupy no space during initial loading, without a
selected session or when their filtered content is empty; polling continues and
reveals them when data arrives. Reasoning examples omit records without readable
text. Request failures remain visible until a successful refresh.
The Agent is instructed to explain the available changes
and wait for the user's request before editing. Replies, including the
introduction, follow explicit user language preferences first, then the user's
own conversation language, then the browser's language preferences. Time zone,
server locale and the generated English setup text are not used to infer
the user's language.
The prompt uses a `<frontend_customization>` envelope, shown as an
expandable context block in chat and preserved in history. Shared scope,
refresh instructions and a recovery link remain visible outside the block.
Creating this session does not create customization files. Normal session
persistence, tools, drafts and history apply. Each browser tab remembers its
active session per workspace in `sessionStorage`, including across refresh
and recovery navigation. Initial polling requests that session directly; a 404
for the saved session during startup selects the default, while other failures
remain visible. Visited conversations retain their rendered entry nodes, scroll
position, draft and disclosure state in page memory. Switching restores those
nodes immediately and polls for updates; unchanged messages reuse the nodes.
Closed sessions release their cached state when the session list is refreshed.
Pending Mermaid rendering can complete in detached entries without scrolling
the active conversation. Refreshing the page discards the rendered caches.

The prompt directs the Agent to read existing customizations, use event
delegation and idempotent setup for asynchronously rendered content, and
verify the affected interactions. Substantial changes need recoverable copies
outside the served directory. A first full HTML override also preserves the
original template for comparison with future upgrades.

## Workspace content views

The entire left workspace pane is the board. When `enable_board` is true, it starts
with a permanent Canvas tab for live HTML, followed by read-only file/diff tabs.
All tabs scroll together horizontally.

Session tabs, the workspace list and the router use the same activity rule:
a workspace is Working while any session is running, including background waits,
and Ready otherwise. Working indicators use the session tab's breathing animation
and honor reduced motion. The workspace list refreshes every two seconds and
preserves existing rows while updating their status and session counts.
In both lists, the existing Open link covers the entire workspace row; Delete
remains a separate action.

The router collects unread workspace shortcuts across all devices in a distinct,
tinted Unread panel above the complete device groups. Each shortcut shows its
device; the existing inventory polling removes it once all its sessions are read
and hides the panel when none remain. Device groups stay ordered by device id,
with workspaces in configuration order. They start collapsed on page load and
preserve their expanded state through inventory updates. Each device's heading
includes Manage, even when collapsed or empty. A red dot on that heading and in
the Online devices list indicates that any of the device's workspaces is unread.
Both places show `n working` while any of the device's workspaces are working.
Click the device name to rename it; the surrounding group heading toggles its
disclosure. Aliases must be nonblank, single-line text of 1 to 128 characters.

Login pages identify the service being unlocked (`pycodex router` or
`pycodex workspace`) and show its hostname and access address, including any
router or device path prefix. A device's password remains a separate login
after the router password when both services require authentication.

Workspace sharing reuses the workspace login form and existing page URLs.
The `next` workspace selects which share password to check; the administrator
password still grants full access. A process-local `WorkspaceSharing` instance
holds salted password verification values, opaque grants and revocation events.
Nothing is added to workspace definitions, saved state or rollouts.
Sharing requires an existing workspace server administrator password. JSON
`api/share` settings are administrator-only, and never disclose credentials.
HTTP and WebSocket entry points enforce the grant's workspace scope, including
Canvas, static assets and file views. Visitor redirects stay inside that scope.
Password replacement, disable and workspace deletion revoke the old grant and
close its WebSockets; already accepted Runtime work is not cancelled. Process
restart clears shares. Agent tools keep their full configured permissions.
The visitor WebSocket handler runs in its ASGI cancellation scope, with a scoped
revocation watcher. Disconnects cancel the watcher, while revocation finishes
the handler's cleanup before closing the socket.
Router login and forwarding are unchanged.

A separate red dot marks an unread final reply or failed turn, including results
produced while background work remains active. A focused, visible page marks the
selected session's latest reply or error as read only when that content enters
the chat viewport. Partial text above an error cannot mark the error as read.
Hidden pages, the mobile Canvas pane and scrolling above the result keep it unread.
Unread state is kept in workspace server memory and shared by all clients, the
workspace list and the router. Refreshing a page preserves it; restarting the
workspace server clears all unread dots. A read confirmation names the final it
displayed, so a delayed confirmation cannot clear a newer reply or error.

A workspace has one `state_path` filename prefix and an `enable_board` boolean
that defaults to true. The prefix identifies the workspace and appends
`.pycodex-ws.json`, `.pycodex-views.json` and `.html` for session state, file views
and Canvas respectively, preserving any suffix already in the prefix.
`board_path` is derived from that prefix, not a second configurable input.
The session manager receives only the prefix; each
store receives its derived JSON file path. Disabling Canvas hides its tab,
stops its polling, omits its context injection and returns 404 for its HTML,
status and image routes. File views and conversations keep the same persistence
paths across changes to the switch.
Missing state starts a new workspace; malformed saved state fails restoration
without replacing the saved index.

Configuration loading normalizes legacy `board: "<prefix>.html"` entries and
`board: false` entries to this contract. The latter preserve their
`<config-stem>.<workspace-id>` state prefix. Existing files stay in place,
and later config saves write the new fields. Mixing legacy and new fields is
rejected. Non-`.html` legacy Canvas paths require an explicit rename/config update.

The tab strips hide their native scrollbars so overflowing tabs stay vertically
aligned with the toolbar buttons while remaining scrollable.
Both content and session tab strips translate ordinary vertical mouse-wheel
input into horizontal scrolling. Horizontal trackpad gestures, Shift+wheel and
Ctrl+wheel retain their browser behavior; wheel input is consumed only when
the strip actually moves.
The file/diff toolbar beneath the tabs keeps its complete path labels in a
horizontal scroll area with no visible scrollbar. It uses the same wheel
behavior as the tab strips; the action buttons stay outside the scrolling area.
Canvas keeps its existing live HTML and image routes. Automatic refreshes replace
the iframe's current history entry, so they do not add browser Back steps.
Relative images resolve to HTTP siblings such as `/w/<workspace-id>/<image>`
(or `/<image>` in single-workspace mode), not `/board/...`. Keep image fallback
routes after concrete workspace/API routes, resolve paths relative to the Canvas
file's directory, and reject non-`image/*` content or paths escaping that directory.
Canvas and ordinary HTML previews use the same-origin browser iframe and native
resource loading, including browser-managed login cookies; do not copy cookies
into HTML or add image-fetch/data-URL preprocessing.
Clicking a Markdown file
or diff link selects the existing tab for that target, or appends a new read-only
tab when the target is not already open. Rendering a link does not open a view.
Reusing a tab keeps its ID, order and reading state. On mobile, selecting a view reveals the content pane. Switching
tabs preserves their scroll positions; closing a view removes that tab
without changing a source file, the Canvas or a conversation.

Content tabs share the session tabs' rounded styling and show a close button
only on the active content tab. They occupy the same 44-pixel toolbar as the
workspace back link and mobile pane switch. The Canvas tab's hover title labels
`Workspace`, `Current dir`, and `Canvas file` on separate lines.
Diff views start with the two file labels; the opening time, read-only status
and whether the view follows files are available in the tab's hover title.
Stored titles, API responses and hover titles retain complete paths and both Git revisions, for example
`COMMIT_A:src/A.py <-> COMMIT_B:src/A.py` for full commit IDs. A symbolic or
abbreviated revision also includes its resolved ID, such as `HEAD (COMMIT):src/A.py`,
so comparisons opened before and after a branch moves remain distinguishable.
Visible labels follow VS Code's filename/description layout: the basename is
the main label, with no leading directory. When open files share a name but
have different paths, a secondary label adds the shortest directory suffix that
distinguishes them. For example, two `src/app.py` files in different packages
can show `app.py` with `client/src` and `server/src`. Opening or closing a
duplicate updates these descriptions without changing tab identity or focus.
Git versions use seven-character commit IDs in the secondary label; a comparison
of the same path shows its filename once, followed by `abc1234 ↔ def5678` or
`abc1234 ↔ working tree`. Different comparison paths retain both filenames and
any needed directory descriptions. This is browser display formatting only;
source identity, stored titles and API responses retain their full information.
Content tabs share the session tabs' 180-pixel maximum width. Secondary text
shrinks and ends in an ellipsis before the filename loses space; a filename
that alone exceeds the available width also uses an ellipsis. The complete
title remains available on hover and as the tab's accessible label.

Workspace assembly introduces the Canvas file, explains that edits are shown
in real time and that the model can create or modify it anytime. It then describes
the extended file/diff links supported in replies and Canvas HTML, including
optional line anchors and the link back to the live Canvas.
Detailed syntax and viewer behavior are documented below. There is no `workspace_display` tool.
The assembly prompt describes available paths and links without prescribing
when the model should use Canvas or adding viewer implementation details.
Link clicks use the browser's view API, independently of Agent/Runtime, and add
no messages or tool results to the conversation. They also work with an empty
`--toolset`. Internal `board` IDs and HTTP routes keep their
existing names.

Ordinary Markdown file links accept absolute or relative paths.
Use Git-style expressions for comparisons:

| Markdown link | View |
| --- | --- |
| `[source](src/app.py)` | Working file |
| `[line](src/app.py:42)` or `[line](/data/project/app.py#L42)` | File, line 42 |
| `[document](README.md)` or `[page](report.html)` | Markdown or HTML preview |
| `[report](report.pdf)` | Local PDF in the browser's built-in reader |
| `[source](README.md:1)` or `[source](report.html#L1)` | Markdown or HTML source, line 1 |
| `[version](git:HEAD:src/app.py#L42)` | Git file at a fixed revision, line 42 |
| `[changes](diff:HEAD:src/app.py)` | HEAD versus the working file |
| `[versions](diff:HEAD~1..HEAD:src/app.py)` | Two revisions of the same file |
| `[rename](diff:HEAD~1:old.py..HEAD:new.py)` | Two Git files with different paths |
| `[files](diff:old.py..new.py)` | Two local files |
| `[files](diff:/data/project/old.py../data/project/new.py)` | Two absolute local paths |
| `[left](diff:HEAD:src/app.py#L12)` | Diff, left line 12 |
| `[right](diff:HEAD:src/app.py#R42)` | Diff, right line 42 |
| `[Canvas](workspace://canvas)` | The existing live HTML Canvas |

`..` compares the two specified snapshots directly, like `git diff A..B`.
It does not calculate a merge base. Line numbers start at 1 and select the
whole line. A diff uses `#L` for its left side and `#R` for its right side;
plain file links use `:line` or `#Lline`. Zero, negative or out-of-range lines,
images and missing target files are errors and do not change the selection.
Line targets do not affect a tab's identity or title.

These links work in chat Markdown, rendered Markdown file views and Canvas HTML
anchors. Canvas accepts absolute and relative file paths, including line targets;
relative paths resolve from the workspace working directory. Linking the
Canvas HTML file path opens a preview tab; adding a line number opens its source.
`workspace://canvas` selects the existing live page. External URLs, `mailto:` links and plain `#anchor` links keep their browser
behavior. The renderer preserves recognized view destinations through sanitizing
without permitting executable URL protocols.
Bare filenames such as `TODO.md` remain text; use `[TODO](TODO.md)` to open them.
Bare domain names are not automatically linked, so `.md` filenames are not
mistaken for websites. Explicit Markdown links, URLs with a scheme or `//`
prefix, and automatic email links still work.

Percent-encode literal special characters in paths: space as `%20`, `#` as `%23`,
`?` as `%3F`, `%` as `%25`, and a literal colon as `%3A`. For example,
`[source](src/a%20%23%25.py)` opens `src/a #%.py`. In a diff endpoint, encode
a literal `..` within a filename as `%2E%2E` so it is not a comparison separator.
Parent-directory components such as `diff:../old.py..../new.py` work directly.
Links are limited to 8192 characters; malformed percent encoding and control
characters are rejected.

File identities use normalized absolute paths, so relative/absolute spellings
of the same path reuse a tab. Comparisons use the ordered pair of their sources;
reversing the sides opens a separate view. Git identities include the repository,
resolved revision and repository-relative path. Equivalent link forms share
a tab, while distinct commits remain distinct even when their file contents
match. Without a line target, a missing file can still select its existing tab
with the last readable contents and an error. A line request reads current content to
validate the line before changing the selection.

Local paths resolve from the workspace working directory, or from the source
file's directory when clicked in a local Markdown preview; absolute paths and
percent-encoded paths with spaces are accepted. Backslashes are preserved for
Windows paths. Git `revision:path` operands follow Git's object syntax and
identify blobs; their paths start at the repository root, or at the workspace
directory when explicitly prefixed with `./` or `../`. One revision compares the
Git file with its working counterpart; two revisions compare the same repository
path in both revisions. A file missing on just one side is displayed as empty, including
an explicitly selected new working file. Missing on both sides is an error.
Other protocols, multiple comparisons and arbitrary Git flags are rejected.
Links are parsed into internal operations, never passed to a shell; Git
content is read by object ID without running diff drivers or text conversions.

Local file views follow changes on disk. Both sides of `diff:A..B` update, while
`diff:REV:file` updates only its working file. Git operands are resolved
once when opening the view, so a later commit or branch move does not change
the pinned Git contents. This also applies to comparisons between two revisions.
Opening the moved reference again resolves its new version and creates a separate
comparison when that version is not already open.

Text is decoded as UTF-8 with replacement for invalid bytes. Markdown and HTML
files default to previews. Markdown uses the existing sanitized renderer; HTML
(`.html` / `.htm`, case-insensitive) renders its markup, styles and scripts in an
ordinary same-origin iframe, just like Canvas. Canvas is the persistent iframe
for the configured live HTML file; other HTML tabs render their current file or
Git snapshot through `srcdoc`. Both use the browser's normal resource loading
and automatically send the workspace login cookie. HTML scripts have the same
access to the parent page and authenticated workspace APIs as Canvas scripts;
the login cookie remains `HttpOnly` and is never copied into the HTML.
Preview host messages only save and restore scrolling, and are checked against the
active frame and current document generation. Unchanged HTML keeps its current
frame and interactive state; content changes reload it and restore scrolling.
Switching away destroys the frame, and returning rebuilds it at the saved position.
Relative asset URLs still use the workspace HTTP base and existing Board image
routes; filesystem image paths are not implicitly converted into asset URLs.
Image `src`, `srcset`, CSS resource URLs and HTML `<base>` URLs keep their native
browser behavior. The viewer does not fetch or convert HTML images to data URLs.
The existing Board image routes continue to enforce directory and MIME boundaries.
Code and SVG files are shown as source, and
PNG/JPEG/GIF/WebP files are stored as image data URLs. Code and text diffs use
Monaco with filename-based syntax highlighting, line numbers, native
find, code folding, selection/copying and optional line wrapping. Unknown
extensions use plain text. Both editor state and DOM editing are read-only;
find has no replace controls, and diff views disable the editing gutter menu,
with no accept/reject or revert controls.
The editor uses textarea input (`editContext: false`) so `domReadOnly` sets the
native `readonly` attribute and touching code does not open the mobile keyboard.
The separate find field remains editable.
Opening Markdown or HTML with a line number switches that tab to its source
editor, so line numbers refer to the actual file. An explicit link without a
line switches the same tab back to preview. Source and preview keep separate
reading positions. Ordinary content refreshes and manual tab switches preserve
the current presentation and do not replay line requests. Presentation choices
belong to that browser page; another page's opens do not change them.
Markdown/HTML comparisons always use the code diff editor.

Local `.pdf` files open in a native browser PDF iframe with new-tab and download
links. The view stores only the source path and file metadata; the authenticated
`GET/HEAD api/views/{id}/pdf` endpoint streams the file as `application/pdf`
with inline disposition. `?download=true` selects attachment disposition.
It only serves an existing PDF view in that workspace. PDF headers and regular
file status are checked before serving; unavailable or invalid files show an error.
Metadata changes update the view revision and iframe URL. An unchanged active
PDF keeps its iframe; switching away releases it. Page and zoom state belong to
the native reader and may reset when its iframe is recreated. Mobile browsers
without embedded PDF support can use the open/download links.

The code editor's **Insert into chat** context-menu action captures the displayed
file and cursor location; a nonempty selection includes only its selected text,
with end-exclusive columns. A selection also reveals the same action in the
toolbar. Each excerpt is limited to 200 lines and 12,000 source characters, with
an explicit truncation marker. The token uses the same compact labels as view
tabs, including both short commit IDs, followed by its side and line range.
Tokens share the tabs' 180px width limit and use an ellipsis; their full labels
remain available on hover. Tokens and their details shrink with the composer,
and the details' action buttons wrap within narrow panes.
Its expanded `View` field preserves the complete title with full Git revisions;
Git context also identifies the repository, and diff context identifies the
original/modified side.
Missing or unreadable documents cannot be inserted. Insertion adds a separate
context token at the current session draft's cursor, focuses the input and
switches narrow screens to chat. Clicking a token opens its editable details;
the remove button, Backspace or Delete removes it as one element. Only submitting
the draft expands its tokens into `<editor_context>` text through the existing
message API, preserving the order of text and blocks. Captured text stays fixed when files
refresh or views change; a later insertion captures the latest displayed text.

Monaco's diff worker computes line and character changes from the two complete
documents, including leading and trailing whitespace changes.
The native diff editor keeps both sides aligned when
scrolling or opening find. Arrow buttons and `Alt+Up`/`Alt+Down` navigate
changes. A `ResizeObserver` on the content container selects split mode at
720 pixels or wider and unified mode below that, including splitter changes
without a window resize. A line request on the original (left) side keeps split
mode even below that width so the selected line remains visible. Focusing or
targeting the modified (right) editor restores automatic layout. This setting
is preserved when switching tabs. Layout changes reuse the same editors and text models.
Long documents keep their full searchable text while Monaco renders only the
visible region.

Only the active code view owns an editor (or a pair of editors for a split
diff). Switching tabs captures Monaco's serializable view state plus find and
wrapping options, then disposes the editor, text models, event subscriptions and
observer. Capture happens before hiding the pane so scroll positions remain
available. Returning restores that reading state; closing the tab also drops its
caches. Late module/editor completions
cannot replace a newly selected view. Markdown and image previews retain their
existing rendering and scroll state.
Content updates reuse that active editor and its models. Monaco's existing worker
computes text edits, which are applied without flushing the models so folding and
find decorations survive; the editor restores its reading state afterward.
Find stays open and editable. Superseded updates and updates to disposed editors
are discarded. Updates to
background tabs invalidate their cached content; selecting them fetches the
latest version. A refresh keeps the current content visible while loading and
does not select a tab or switch the mobile pane.
Explicit line requests wait for current file contents and diff computation,
then select the line, reveal it and focus the target editor. A newer selection
or a tab switch invalidates an older pending request. Ordinary content refreshes
and manual tab switches do not replay an earlier line request; another click
can explicitly reveal even the same line again.

Text and image snapshots are limited to 10 MiB each, text to 20,000 lines and raster images to
20 megapixels. View storage is limited to 64 MiB per workspace; exceeding a
limit when opening a view is reported in the browser and does not create or replace a tab.
Local PDFs stream directly and are not subject to the text/image size limit.
If a live text/image file becomes unreadable or invalid, its last readable content remains
visible with an error. A deleted text/image view also keeps its last content; a
deleted side of a diff becomes empty and is marked missing. Recreating or fixing
the file resumes updates. A storage write or total-budget failure is reported
to the browser and leaves the previous persisted views intact.

`packages/pycodex-ws/workspace_server/workspace_views.py` owns
`WorkspaceViewOpener` and `WorkspaceViewStore`.
It saves open views, explicit file/Git sources, the latest contents, optional
line targets and the monotonic reveal counter
atomically in `<state_path>.pycodex-views.json`, separately from
conversation state. Sources are checked again after a server restart.
Earlier saved snapshots without source information remain fixed; reopening
their file or diff creates a view with explicit sources. Such snapshots are not
matched by title alone. Saved titles are reconstructed from their full document
labels without guessing source paths. The store is shared across that workspace's
session threads; closing a tab is also reflected in other connected browsers.
Selected-tab state is local to each browser/workspace. A click focuses its target
only in that page; other pages can discover the tab without changing their
selection. A new page restores its saved selection or defaults to Canvas.
With Canvas disabled, it restores a file view or selects the first existing file
view; an empty workspace shows an empty content pane. Closing the last file view
returns to that empty pane, and other pages' opens do not change its selection.

`POST api/views/open` accepts `{"href": "<file path or URI>", "session_id": "<tab id>"}` and returns
`{"view": <full snapshot>, "views": <metadata list>}`. Relative local paths use the
selected session's `cwd`, or the parent of the optional `src_path` (a plain file path).
The browser supplies its active tab id. API callers omitting `session_id` use
the workspace's `work_dir`; an unknown id is rejected.
Each workspace tab persists `cwd` alongside title and rollout reference in
`<state_path>.pycodex-ws.json`, including empty tabs. Older records without `cwd`
inherit `work_dir`; invalid saved directories fail restoration visibly.
Canvas is injected into model context as an absolute path so it remains stable
across directory changes.
Single-workspace `create_app(..., work_dir=...)` defaults
to the process's working directory. The other routes use view IDs:
`GET api/views` lists metadata, including content/reveal revisions, a live/fixed
flag and optional `selection` (`document`: zero-based side, `line`: one-based line);
`GET api/views/<id>` returns the current contents and
`DELETE api/views/<id>` closes the view. These routes use the existing
workspace authentication and namespace in both single- and multi-workspace
apps. The browser polls
the list every two seconds. The store checks only recorded file paths, using
file identity, size and modification/change times to detect writes and atomic
replacements; unchanged files are not reread. No recursive scan or additional
watcher process is needed. Content/error changes advance the view revision and
invalidate browser caches. Stale poll/content responses cannot replace a
newer revision or a different selected view.
Each successful click advances the target's `reveal_revision`. Only the initiating
page uses that open response to select a tab and apply its optional line target;
polling never follows another page's reveal. Newer clicks, manual selection,
closing a tab or stopping the page invalidate an outstanding open request.
Late success/failure responses and polls cannot override the user's latest choice.
Reveals keep the current editor when possible and do not reorder tabs. The
counter survives closing its latest target and restarting the server. File
refreshes change only the content `revision` and therefore do not move focus.
The JS/CSS assets and third-party licenses ship in the wheel and are served
through concrete routes before the Board image fallback. `content_views.mjs`
and `content_views.css` contain only the lightweight tabs and preview controller.
The first code or diff view loads the separate same-origin `editor.mjs` and
`editor.css` in parallel, and waits for both before creating an editor. Canvas,
Markdown, HTML, PDF and image previews do not request either editor asset. Each
display retains its loading promise and stylesheet for later code views;
switching away during loading still invalidates the pending editor creation.

The editor bundle includes all language highlighters and its editor/diff worker,
but registers only the features used by the read-only viewer. JSON uses its
tokenizer and language configuration directly: the language-service entry point
would also import the full editor feature set even with diagnostics disabled.
The worker starts from an embedded Blob URL and the editor CSS embeds the icon
font. `MonacoEnvironment.getWorker` is the library-required global host hook;
other viewer state belongs to each content display. No editor, font, worker or
language package is fetched from a CDN. Python installation and `pycodex-ws`
startup do not require Node.

Only these static viewer assets use negotiated gzip responses. Each application
instance caches the requested identity/gzip representations, invalidating them
when the corresponding file's modification time or size changes. Responses use
`Vary: Accept-Encoding`, representation-specific ETags and `Cache-Control: no-cache`;
matching conditional requests return 304. HEAD reports the same representation
headers without a body. Compression does not buffer session traffic, and the
wheel contains one copy of each asset, without precompressed duplicates. The
package's Python requirements are unchanged. Monaco does not officially support
mobile browsers; viewport and touch checks supplement, but do not establish,
support across real mobile browsers.

Frontend sources and exact dependency versions live in
`packages/pycodex-ws/workspace_server/frontend/`, with `package-lock.json` fixing transitive
dependencies. After changing those sources, rebuild the generated assets using
Node 18+ from the repository root:

```bash
npm --prefix packages/pycodex-ws/workspace_server/frontend ci
npm --prefix packages/pycodex-ws/workspace_server/frontend run build
```

The build collects licenses and third-party notices for bundled packages and stages every output
before replacing the public JS entry. For an isolated preview, pass a staging
directory with `npm --prefix packages/pycodex-ws/workspace_server/frontend run build -- /tmp/workspace-assets`.
The wheel contains the generated assets, without frontend sources, `node_modules`
or the build toolchain.

## Observation

Runtime events are frozen dataclasses defined together in `pycodex/events.py`,
separate from the conversation and model data in `pycodex/protocol.py`.
Session commands belong in `runtime.py`; terminal startup, I/O and the input
loop belong in `cli.py`. Presentation should describe user-visible phases such
as tools starting/completing and the model reviewing results; retain `iterations`
in programmatic results instead of making it the main status label.
`Event` is the common base; each concrete class declares its fields and a
class-level wire `kind`. There is no event registry, parallel enum, or generic
`AgentEvent(kind, payload)` / `ModelStreamEvent(kind, payload)` constructor.
Backend consumers use `isinstance` and direct field access; plain-text consumers
use `event.visualize()`. Events without a text view return `""`.
CLI and Feishu presentation dispatch through `event.render(display)`. Each view owns an
`EventDisplay` with log/status/prompt executors, color capability and independent
stream, title, context and steer/nickname state. Events decide when to emit text,
flush/discard partial output, color messages and update status or input prompts.
Not calling a status executor preserves it; passing `None` explicitly clears it.
The terminal owns only input/output, locking and refresh scheduling; even spinner
frame formatting belongs to `EventDisplay`. Shared text/color/summary formatting,
non-event errors and one-shot/JSON output helpers live in `utils/event_helpers.py`.
That helper module depends only on the standard library, never events or
frontends. The compactor's event type import is type-checking-only so loading
the utils package cannot create an events/utils initialization cycle.

There is no display-handler mapping, action bus or per-tool event subclass.
Presentation never mutates an event or its nested data and is not invoked by
Agent/runtime dispatch itself. Feishu binds the same log/status/prompt callbacks
to card fields with colors disabled. Its layout adapter restores snapshot history
and tracks queued/started/completed turn boundaries for the main grey answer box.
The box shows only the latest completed reply. A new turn preserves that reply
with the `(*last turn)` prefix; completion replaces it with the new final text.
The reply is truncated to the card's 6,500-character display limit.
The card header uses the session title (or `pycodex`) instead of the fixed
`Session Connected` label. The current user's prompt is visible above the reply.
The input retains the shared `pycodex>` / `pyco(…)` context prompt plus status
or model, including the existing answer/permission prompts.
Stream, tool, retry, error and question presentation still uses the shared
`render` implementation. Active stream text stays in the green box; the latest
rendered activity, command result, error or question appears separately above
the input and does not accumulate in the answer. Starting or completing a turn
clears that activity, and resolving a question clears its prompt.
Attach/resume restores the latest completed reply and any active stream or
pending question.
Completed prompt/reply pairs are available in a native Feishu JSON 2.0
`collapsible_panel`, collapsed by default. Clicking its header expands or
collapses it; Previous/Next buttons browse one earlier turn at a time. These callbacks
only select card history; they never submit text to AgentRuntime or change the
live turn/input. History navigation remains selected while replies arrive.
Rendering one historical turn, capped at the existing 6,500-character display
limit, keeps the card payload independent of the number of stored turns.
Attach/resume reloads history from the runtime snapshot and clears the selection.
The panel schema follows the [Feishu folding-panel documentation](https://open.feishu.cn/document/feishu-cards/card-json-v2-components/containers/collapsible-panel).
Web retains its own projection. ANSI is
applied only at a terminal presentation boundary, never stored in event fields,
Web JSON or generated Feishu presentation. The module does not import terminal,
Web or Feishu implementations.

Retry events discard the failed attempt's buffered text, whereas fatal turn
errors flush it. In particular, `stream_error` must not call `finish_stream()`
and print a failed attempt's partial reply before a successful retry.
While the CLI is accepting input, phase events must not resume its spinner
over that input. Preserve `[steer] queued: <prompt>` on admission and
`[steer] inserted: <prompt>` when the queued input actually starts.
Manual compact completion/failure events only update idle
status; the command event emits their result once. Auto-compact remains inside
the active turn. Attach/history snapshots restore display state without sharing
buffers between subscribers.

Tool descriptions, command results, questions and compact messages are rendered
by their event classes. `CommandCompletedEvent.visualize()` uses
`utils.event_helpers.format_command_result(result)`, also shared by the startup
help banner and one-shot command receipts without inventing events.
Compact summaries derive from item counts and pruned-tool counts instead
of storing duplicate text; the Web adapter still supplies the same `summary`
field on the wire.

`TurnStartedEvent` carries only `user_texts`; its `visualize()` method joins the
text. Local tool events carry the existing `ToolCall` and
`ToolResult`, rather than duplicate tool names, ids and error flags.
Provider tool reports use `ToolCalledEvent`, distinct from local execution
starting with `ToolStartedEvent`.

The four `ModelEvent` variants cover assistant deltas, tool reports, usage and
stream errors. The Agent adds turn identity with `dataclasses.replace`; the
queue similarly adds submission identity without changing the concrete type or
mutating the producer's event. `Agent._emit(event)` only dispatches; terminal
events receive their background-work count explicitly at construction.
Manual compact terminal events notify lifecycle hooks, but auto-compact events
do not finish or restart the enclosing turn.

`InputRequestedEvent` also represents the active request in backend snapshots.
Commands, session state and provider-specific data keep their existing nested
data structures; this is not a second business-data schema framework.
Only the Web transport converts events to `kind/turn_id/payload` dictionaries,
including derived fields, event-rendered prompt text and the original
question/permission JSON shape. The workspace renders the pending input request
below the conversation and removes it on resolution; it is not a conversation
turn. Python observers and custom model clients must use the concrete event
classes instead of the removed generic interfaces.

Agent and submission event callbacks are observers. A callback exception is
reported through the owning asyncio loop's exception handler; it does not change
a successful turn into `turn_failed`, suppress an underlying model error, or
discard a completed tool result. Background tools observe the same lifecycle
events through `BaseTool.handle_agent_event`, not a second Task-listener channel.
No event payload is included in these observer-failure diagnostics.

The Responses client marshals worker-thread events onto the owning event loop.
When that request completes or is cancelled, its callback is deactivated; already
queued and later thread events cannot change a newer turn's usage or display.
That transport-level fence remains useful for request teardown and does not
constitute an Agent cancellation API. It does not physically stop a blocking
`requests` thread.

## Tools

Register `BaseTool` instances through `ToolRegistry.register(tool_instance)`.
Keep declarations in the tool classes and serialize through
`ToolSpec.serialize()`; do not
restore loose name/description/handler registration, prompt-level
`serialized_tools` overrides, or the removed
`pycodex/prompts/exec_tools.json` / `pycodex/prompts/subagent_tools.json` fallbacks.
Function-tool `output_schema` remains local metadata and is omitted on the wire,
matching upstream's skipped field.

`AgentsManager(..., tools_factory=get_tools)` uses the same
`factory(manager, environment, toolset) -> ToolRegistry` for root and child
construction. A custom factory reconstructs the requested selection using new
instances and registers any shared managers with that node's environment.
Direct `registry.register(tool)` only changes that node; inherited tools must be
constructible by the shared factory. Child selection is the parent's registered
tool names, including custom namespaces.

`get_tools(agents_manager, runtime_environment=None, toolset=None)` constructs
the default tools below. A nonempty selection includes `exec` and `wait`:

```text
exec wait exec_command write_stdin clock request_user_input apply_patch view_image
create_goal get_goal update_goal
collaboration.spawn_agent collaboration.send_message collaboration.followup_task
collaboration.wait_agent collaboration.interrupt_agent collaboration.list_agents
```

Except for the documented `clock` extension, each supported tool corresponds
to one upstream built-in: preserve its name, parameters, interaction and output
semantics rather than combining different tools' contracts.
Ordinary function tools and freeform/custom tools such as `apply_patch` share
the protocol; tool results can carry structured `input_image` content items.
`web_search` is a provider-native Responses declaration, not a local registry
execution. `exec` / `wait` run Python in the host process with top-level `await`,
`text`, `print`, `image`, `audio`, `generated_image`, `store`, `load`, `exit`,
`notify`, `yield_control` and `tools.<name>(...)`.
MultiAgentV2 tools share the tree's in-process `AgentsManager`. Spawn takes
`task_name`, `message` and optional `fork_turns`, `model`, `reasoning_effort`;
it returns the canonical `task_name`. Relative targets are resolved below the
caller, and canonical paths or Agent IDs address nodes elsewhere in the tree.
Children inherit their parent's visible toolset, model and current directory.
Each owns a new tool environment and model client.

`ContextConfig.multi_agent_max_concurrency` limits open Agents across the whole
tree, including the root. The default is 16; the corresponding TOML setting is
`features.multi_agent_v2.max_concurrent_threads_per_session`. Spawn serializes
capacity checks, reclamation and registration so parallel calls cannot claim the
same slot. Closing Agents keep their slot until cleanup finishes.

At capacity, spawn closes the least recently used eligible child before creating
the new one. Creation, message delivery and turn/compaction/command starts update
recency. A candidate must have completed, failed or been interrupted, with no
running/queued work, mailbox messages, commands, input request or background tool
work. Enabled clocks and running `exec` cells also prevent reclamation. The root,
the caller and nodes with live descendants are excluded: normal close is
recursive, so capacity reclamation only closes leaves. If no child qualifies,
spawn fails with a capacity error and leaves existing work running. Cleanup
errors propagate before a replacement is created.

Capacity reclamation permanently closes and unregisters the child. The node is
removed from the tree and `list_agents`; Workspace detaches its observer, drops
its child view and closes subscriptions. Old Agent IDs are no longer queryable,
and reusing the task name creates a new Agent with a new ID. Messages already
delivered to the parent remain in its conversation. There is no automatic
reopening or idle-timeout reclamation.

Unified exec has two separate output bounds: an omitted `max_output_tokens`
still applies the default `10_000`-token response limit, and unread process
output retains a bounded `1 MiB` head/tail buffer between polls.

The registry owns the registered catalog, model visibility and `exec_tools()`
metadata. Namespace qualification and Python identifiers have shared pure rules;
duplicate Python identifiers fail when the cell catalog is built.
Tools implement async `BaseTool.run`. The registry converts execution failures
to `ToolResult` errors. Existing explicitly parallel tool batches remain parallel:
the Agent waits for every operation in a batch before advancing or propagating
a commit failure. It does not cancel other operations when one fails.

Each tool result is truncated and committed as soon as that tool completes,
before its completion event. Parallel completion order can
determine result order; call ids retain the pairing. Tools in the same parallel
batch see the same history snapshot. There is no cancellation-recovery path that
invents results for unresolved calls. Persistence failures remain visible and
must be resolved before continuing from a partially recorded call/result group.

Tools may provide `follow_up_messages(output)`. These hooks receive untruncated
outputs after all tool results have been committed, so a notification cannot
split an outstanding call/result group.
This local hook does not change tool request schemas or result serialization.
Top-level and nested cell calls both use `ToolRegistry.execute`, including error
conversion and accounting. Nested calls return their output to the cell without
adding follow-up messages; outer Agent batches still own history and follow-up
delivery.

`AgentsManager` delivers child completion and failure as a native `AgentMessage`
with message type `FINAL_ANSWER`; interruption only updates status. Each terminal
event is delivered once, even when successive answers are identical. Runtime
consumes mailbox items at input boundaries after issued tool batches complete.
A message arriving during active execution requests a follow-up sample; an idle
recipient keeps it until execution is triggered.

`send_message` only delivers a message. `followup_task` also triggers an idle
non-root recipient; it does not stop active execution. `interrupt_agent` requests
a cooperative stop and returns `previous_status`, leaving the Agent available.
It rejects the root and self-targets. `list_agents` reports live task paths and
statuses, optionally filtered by a path prefix. Closed children are unregistered
and disappear from both this list and Workspace observation.

`wait_agent` takes only optional integer `timeout_ms` and returns
`{message, timed_out}` without consuming message content. It wakes on mailbox
activity or a cooperative stop. The default is 30000 ms; values below 10000 ms
are clamped with an explanatory message, and values above 3600000 ms are rejected.
It does not accept IDs or return a status map.

`BaseTool` only provides `spec`, `serialize`, `run` and `follow_up_messages`.
Factories register shared managers once with
`environment.register_resource(name, manager, restart=None, close=None, observer=None)`.
Managers implement `bind(invoke, output, state_changed)`, `shutdown`,
`set_state_listener`, `background_work_count` and `background_work_snapshot`.
Optional restart, async cleanup and event callbacks are registered explicitly.
Environment binds Runtime capabilities without passing a Runtime object to tools;
binding alone does not retire cells. Runtime construction and resume explicitly
restart the environment; CodeModeManager retains its generation isolation and
waits for retired cells during cleanup. `environment.close()` drains registered
async cleanup; `Agent.shutdown()` accounts and saves state, then stops resources.
Standalone callers can await environment close as well. Unified exec shutdown
only disconnects notifications: subprocess survival and output readers retain
their existing behavior. Goal continuation is requested through the environment.
Clock countdowns pause on turn/manual-compact start events and rearm on success,
including successful manual compaction. Terminal events retain the single
aggregate `background_work_count`.

An exiting exec session can invoke `<exec_command_completed>` through
`AgentRuntime.maybe_invoke` only while that Runtime is idle and accepting input;
busy notifications are dropped. `clock` likewise uses `maybe_invoke`, with one period per
Agent/session and a timezone-aware ISO 8601 `current_time` captured when the
timer expires. Starting a turn cancels the pending countdown; a successful
reply rearms it. `clock(null)` disables the clock; shutdown stops the timer and
unbinds callbacks while retaining the configured period for restoration.
The aggregate background count includes running exec sessions and an enabled clock after a successful
reply; frontends consume that one field, without another queue or shell sleep
loop. Running code-mode `exec` cells also report background work; a completed
cell finishes after its Python tasks are cleaned up and issued host tools settle.

### Python exec cells

`CodeModeManager` owns each cell's execution task, output events and lifecycle.
`ExecRuntime` compiles the source into an async function with a fresh namespace
and runs it on the host event loop. There is no interpreter subprocess, worker
protocol or per-cell PID. A yielded cell can keep running alongside later cells
and other Agents. `wait` returns only new output and closes completed cells.
`notify` sends an additional output through the attached Runtime with the
original exec call ID.

Plain `print()` in cell code writes to its own output buffer through the cell's
builtins mapping. `sys.stdout` is never redirected; imported modules' prints,
explicit writes to stdout/stderr and explicit `print(file=...)` use their usual
streams. Each cell exposes `cwd` as its captured workspace `Path`; use
`(cwd / "file").read_text()` for direct file access. Nested tools receive that
same directory in `ToolContext`. Python's process cwd is not changed: bare
relative paths resolve against the host cwd.

A task factory shared by active cells on the event loop records descendants
created through asyncio from owned cell tasks, including `create_task`,
`ensure_future`, `gather` and `loop.create_task`. It preserves the preceding
factory and restores it after the last active cell finishes. `set_timeout`
callbacks run as owned tasks and their timer handles belong to that cell.
Cleanup cancels and awaits only that cell's Python tasks and timers. Direct
`asyncio.Task(...)` construction, unmanaged loop callbacks, threads and
executors require explicit cleanup by the script.

Nested tools execute through the registry in separate host tasks. These tasks
and their descendants are excluded from cell cancellation. Cell completion,
`wait(terminate=True)` and manager close wait for already issued tool calls.
Cancelling an exec/wait observer does not cancel the cell. Shutdown requests
stop; async close waits for cleanup. Restart retires old cells and gives the new
session a fresh store; retired cells cannot publish output or mutate that store.

`store/load` share JSON-serializable values across cells of one manager, with
copy semantics; ordinary variables and definitions remain cell-local. Imported
modules, environment variables and other process state are shared with the host.
This is intentionally not V8-style isolation. Stop is cooperative cancellation:
blocking I/O and non-yielding CPU loops block the event loop and cannot be
independently terminated. Use async operations for orchestration and
`tools.exec_command` for blocking work.

Host session workers, notification watchers, timer cancellation and I/O cleanup
still use asyncio where appropriate. Those resource mechanisms are not per-turn
Task ownership or user-requested turn cancellation.

Use a fresh registry per Agent. To supply runtime services explicitly:

```python
environment = AgentRuntimeEnvironment()
config = ContextConfig.from_codex_config(config_path)
manager = AgentsManager(client, config)
tools = get_tools(manager, environment)
agent = Agent(client, tools, config, session_id=uuid7_string())
assert agent.tool_registry.runtime_environment is environment
```

For an operational tree, use `manager.create_root_runtime()` so the Agent is
registered before tools can address it. The example above only demonstrates
explicit standalone dependency construction.
There is no process-global default runtime environment. The former global getter
is removed; the Agent's environment comes from its registry, not a second
constructor argument or forwarding Agent property.
Context configuration may be shared as immutable data,
but child agents receive separate context managers and their own metadata cache.

## Model and history contracts

Keep `UserMessage`, `AgentMessage`, `AssistantMessage`, `ToolCall` and `ToolResult` as the basic
conversation contracts when extending the richer item/event model. The
test-only `ScriptedModelClient` belongs in `tests/fakes.py`, not the runtime
package; the model boundary must remain replaceable and testable.
Construction and first-write requirements are in
[Recording lifecycle](#recording-lifecycle); they also apply to child context
inheritance and resume.

`ModelClient` requires `model`, `reasoning_effort`, `recording_metadata`, async `complete`, and
`with_overrides` for child construction. `Agent.model_name` reads `client.model`
directly; test clients must supply the properties used by their integration.
Recording metadata is `ModelRecordingMetadata(originator, provider_name)`;
Agent does not inspect provider internals. Recorder's `recorded_path` reports
whether its lazy initial write has completed.
There is no missing-property fallback or second
model-name field. `ModelResponse` checks its item
types at construction instead of allowing the loop to silently discard invalid
internal values. Runtime/Manager clients implement `ModelControl`, which extends
`ModelClient` with writable model/effort, endpoint and model listing.
Switching through `Agent.set_model` updates both the client and
context; `ResponsesModelClient.model` uses the same provider config as metadata.

Agent passes `Prompt.session_id` explicitly for sampling and all compaction
requests. Headers, cache keys, stable prefix IDs and request diagnostics use that
identity through retry, fallback and resume. An omitted Prompt session ID uses
the client's constructor default for standalone requests; Agent never rewrites
that client field. Tool serialization and response-item decoding use pure codecs.

The CLI startup header reads the client's model, effective reasoning effort and
display-safe `endpoint` directly. `endpoint` is the active API base URL with
userinfo, query and fragment removed; locally managed compatibility transports
therefore show their local forwarding address. `EventDisplay.start` renders
`Welcome to PyCodex`, these settings and the available commands. `/help` repeats
the command listing.

`ResponsesModelClient.reasoning_effort` reads the effective effort and replaces
only the effort field of its immutable provider config on assignment. Assigning
`None` clears the override and restores the current model's metadata default;
without a metadata default the request omits effort. `/effort [<value>|default]`
uses this property directly, checks that work is neither running nor queued
before changing it, and broadcasts the effective value through the existing
`CommandCompletedEvent` rendering path. Changes also publish session state whose
existing `model` field carries the `model:effort` display label. Web, VS Code and
Feishu reuse that field without additional transport fields or formatting.
The Agent/client model identifier remains unchanged; labels show `default` when
no effort is sent.
It does not write config files or add
Agent/bootstrap parameters. Existing child clients retain their settings; newly
spawned children inherit the parent's current override unless explicitly changed.
Values remain provider-specific, with no local `adaptive` selection policy.

Only the Responses provider classifies external error text:

- Transient/disconnected streams and retryable `response.failed` events
  use `ResponsesRetryableError`.
- Context overflow uses `ContextLengthExceeded`, with optional usage and limit
  parsed from the provider's error details.
- A genuine terminal incomplete response uses `ResponsesIncompleteError` and
  does not trigger provider retries.

Custom clients must raise the corresponding exception rather than rely on
arbitrary `RuntimeError` text being reinterpreted by the Agent or compactor.

The local `responses_server` validates ordinary function-call arguments after
assembling the downstream stream. Malformed JSON produces `response.failed` with
`type = "server_error"` before any final output items are emitted. The error names
the tool and JSON parsing failure without including raw arguments.
`ResponsesModelClient` handles this through its generic provider error path and
retries only the current request under `stream_max_retries`; it does not classify
tool argument parse failures as retryable locally. Earlier completed tools stay
in history and retries do not restart the turn. Valid JSON that decodes to a
non-object still raises a terminal API error; freeform/custom tool inputs remain
plain text.

Assistant message text remains text, including content that resembles a tool
call. Tool parse errors require a structured tool item or an explicit provider
error; the client does not infer them from patterns in assistant prose.

For max-output incomplete responses, the Agent persists done assistant/reasoning
items for continuation, but not bare text deltas or tool calls without results.

When recording is enabled, the Agent's history append path invokes the rollout
recorder before extending active in-memory history. Recorder errors remain visible.
This is not a filesystem transaction: an append-only rollout may contain a partial write when
the underlying storage fails, and the existing loader tolerates incomplete tails.
If the initial write fails, the recorder remains uninitialized; it does not
silently append past a partially created file on retry. That existing file must
be handled explicitly, and errors are not suppressed.

Session restoration uses synchronous `agent.resume(session_file_path=None)`.
The Agent rejects its own active execution or manual compaction; Runtime guards
pending submissions and closing as well. CLI and workspace use `runtime.resume`
with a path: this expands `~`, loads and
validates the file, then updates the Agent's session id, supported provider
session identity, active history, internal recorder and usage, and reopens the
Agent. The returned metadata is available to the host view; callers no longer
load and assemble the recorder themselves. An unsuccessful load leaves the
previous session unchanged.

`session_file_path` is a read-only projection of the recorder's path (or `None`
when recording is disabled), not a second mutable path field. A successful
`resume(path)` does not rewrite metadata or copy
old history into a new file; later results and checkpoints append to the selected
file. Resuming a custom-named compacted file reads its initial session metadata
as well as its latest checkpoint, so identity does not depend on a filename UUID.
Constructing an Agent and immediately calling `resume(path)` leaves the original
new-session path uncreated.
The loader still accepts multiline JSON objects and tolerates incomplete tails;
resume does not truncate or repair source files.
Tool calls without matching recorded results are omitted from restored history.
An interrupted call may precede later saved turns in the same append-only file,
so restoration retains all other items, including completed sibling tools,
assistant/reasoning items and subsequent user turns. Repeated resume and append
operations therefore retain later history even while the incomplete call remains
in the source file.

Without a path, `resume()` re-enables the same Agent without reading or writing
a file or changing history, session identity, recorder or usage. It returns
`None`. `runtime.resume()` additionally rebinds background tool callbacks.
These forms are selected by the
argument, not by whether a file happens to exist;
an invalid explicit path never falls back to reopening the in-memory session.
There is no separate `reopen()` method.
Construction and `resume(path)` share one private recorder setup path
with explicit create/resume selection. It constructs the recorder before
updating session identity and does not change history, usage or admission.
`agent_id` and `agent_path` stay fixed when `session_id` changes on resume.
The session `fork()` APIs are removed. Child `fork_turns="all"` uses
`ForkContext` to inherit history and preserve the fixed-context prefix;
see [fork context](CONTEXT.md#子-agent-的-fork-context).
The old recorder constructor argument and
`restore_session`/`set_rollout_recorder`/`replace_history` interfaces are removed.
Recording is controlled solely by the path argument, never a separate enable
flag or the provider's `_openai_subagent` marker. In-memory child turns,
compaction and no-argument resume must never write a rollout; filtering
records after a write is not a substitute.
`history` is read-only; `set_model` remains an operation with consistency checks
rather than a mechanical setter wrapper.

## Compaction

`compact_history` takes history, client and context explicitly and returns a
`CompactResult`. It does not receive an Agent or mutate its active history.
Overflow retries prune paired old tool calls/results only from the local retry
history. Empty or reasoning-only summaries fail rather than replace history
with a placeholder.

The Agent commits a successful result after writing the compacted checkpoint,
then clears stale usage. Model or checkpoint-write failure leaves active history
unchanged. Manual compaction uses the same idle gate as a normal turn, preventing
background wake-ups from racing with it.

Manual, threshold-triggered and overflow-triggered compaction use the same Agent
implementation for streaming observation, checkpoint commit and terminal events.
Manual operations emit `compact_started/completed/failed`; automatic
operations retain the corresponding `auto_compact_*` names and phase metadata.
The interactive shell displays those events without manufacturing its own stream
events. `Agent.compact()` no longer takes a separate stream callback; observe
through the Agent's normal event handler. Every compact retry reuses its supplied
turn id, including the parent turn id for automatic compaction.

`model_auto_compact_token_limit` controls threshold compaction using the latest
provider `usage.total_tokens`; an absent limit disables it. A successful
`compacted` record is an append-only checkpoint whose `replacement_history`
may contain only the summary. The old transcript remains on disk for audit,
not in the next active prompt.

The local templates deliberately preserve concise excerpts of the latest real
user request in its original language, write the summary in the user's primary
language, and tell the next model to continue the task rather than merely
acknowledge the handoff. Preserve these additions
when comparing with upstream `templates/compact/{prompt.md,summary_prefix.md}`.
They are intentional differences, not stale copies to replace mechanically.

On `context_length_exceeded`, parse `requested ... tokens (... in the messages,
... in the completion)` when present and emit that real failed-request usage
as `token_count`, then compact and retry once. A generic
`Your input exceeds the context window...` error also triggers recovery but
does not justify inventing usage. If compact itself overflows, repeatedly remove
the oldest `ToolResult` and its paired `ToolCall` from the local compact input
before retrying.

`response.incomplete` is terminal, not a disconnected stream to retry five times.
Ordinary turns report it; for `reason="max_output_tokens"`, completed
assistant/reasoning items remain available for continuation as described above.
A compact request can use an already received partial assistant summary for
that same reason instead of repeatedly restarting compaction.

## Responses reliability

The client reads provider configuration from the selected Codex config file
and sends `stream = true`.
`stream_max_retries` defaults to 5; tests that need an immediate fatal stream
error must explicitly set it to 0 (see [test guidance](../tests/TESTS.md#provider-配置)).
Provider `use_chat_completion = true` persistently enables the local compat
transport; an explicit CLI `--use-chat-completion` still overrides that setting.

Do not add a model-level `model_max_output_tokens` config or
`max_output_tokens` request field for upstream parity: upstream does not send
that model limit. The similarly named tool-output limits are separate.
`service_tier = "fast"` maps to wire value `priority` only when the model's
`service_tiers` metadata explicitly advertises that tier. Omit `default` and
unsupported tiers.

An optional provider `fallback_models` list enables client-local ordered model
failover. Transient failures use the per-model retry budget; unavailable models
and first-data timeouts skip directly to the next target. Fallback preserves the
selected primary, endpoint, credentials and prompt, and rebuilds model-specific
request settings. It emits an existing `StreamErrorEvent` naming both targets
and carrying the failed attempt's diagnostic. No Runtime or frontend state is
added for routing.

`fallback_cooldown_seconds` is a nonnegative integer, defaults to 900 and can be
zero. Each client tracks monotonic cooldown deadlines; later requests skip
cooling targets and retry them in their original order after expiry. If all
targets are cooling, the next completion clears the deadlines and tries again.
Explicit model selection clears cooldowns; new clients have independent state.
Terminal request, context-length and incomplete errors do not cause failover.

Responses HTTP 408/429 and server errors use the provider retry budget.
Rate-limit messages containing `try again in` or `retry after` supply the retry
delay. CLI and workspace retry notices include the underlying error details;
workspace notices survive browser refresh and active-turn reattachment, and
remain separate from the assistant reply. They are not persisted in rollouts.

Providers may configure a positive `stream_first_data_timeout_ms` to bound
the initial wait for response body data, including waiting for headers. After
body bytes arrive, reads use `stream_idle_timeout_ms`. Omitting the option
preserves the existing timeout behavior. These are transport timeouts, not a
total generation deadline.

Each HTTP attempt emits `model_request_timing` with the model, elapsed seconds,
first-data seconds and completed/failed outcome. Without the first-data option,
the first-data measurement uses the first streamed line. Timing events are
available through workspace event records, including compaction requests.
