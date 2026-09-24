# Runtime state and failure semantics

The runtime keeps state with the component that owns the work. This document
describes the Python implementation, not a claim of newly verified upstream
wire parity.

## Ownership

| Component | Owns |
| --- | --- |
| `Agent` | Serial execution, local admission and cooperative stop flags, final shutdown, history, usage, context, rollout, identity and model/tool dependencies |
| `AgentRuntime` | Frontend interfaces, session commands, normal/steer queues, receipts, title/state snapshots, subscriptions, interactive requests and worker lifecycle |
| `ToolRegistry` | Tool instances and their session-local runtime environment |
| `AgentRuntimeEnvironment` | Plan, sub-agent, user-input and permissions services |
| `SubAgentManager` | Child runtimes, terminal event status and status waiters; each runtime owns its worker |
| CLI / web / Feishu | Input transport and event rendering; no command interpretation or direct Agent control |
| IPython | Deliberately a bare Agent, outside the frontend/queue contract |

Construction takes three required arguments:
`Agent(model_client, tool_registry, context_config)`. `ContextConfig` includes
cwd, instruction overrides and extra contextual messages
as well as configuration-file values. The Agent constructs its own
`ContextManager` with the client's model identifier, without modifying the
caller's config. Context managers are never implicitly supplied or shared between
Agents. Standalone callers use `ContextManager(config)` and load configuration
through `ContextConfig.from_codex_config`.

Agent, runtime, tool-manager and workspace-manager construction does not require
a current event loop. Async locks and conditions are created when their async
operations first use them; the runtime queue event belongs to its worker.
Each Agent turn or manual compaction creates its own completion event and releases
it when the operation ends. This supports constructing objects after
`asyncio.run()` has closed a loop, or before the eventual execution loop starts,
including on Python 3.8. Running workers and active resources still belong to
their execution loop; Web subscription queues are created in the consuming
loop when `subscribe()` is called. The exec managers use ordinary locks only around
synchronous data updates, with no `await` inside those sections.

There is no collaboration-mode state or prompt injection. CLI and workspace use
the same context-building path. `request_user_input` retains its declaration but
always returns `request_user_input is unavailable in Default mode`; registered
input handlers do not enable it. The tool has no request-manager dependency.
The generic runtime input service, sub-agents, `update_plan` and ordinary
interactive sessions remain available.

`model_client`, `tool_registry`, `context_manager` and `event_handler` are ordinary
attributes. `AgentRuntime` holds its Agent and subscribes to events, but the Agent
has no runtime/queue reference, input-source callback or submission lifecycle API.
Control flows from Runtime to Agent; events are observations, not instructions to
settle runtime receipts. There are two backend abstractions, not an additional
Session wrapper. Queue data remains internal to Runtime.

## Frontend boundary

```text
CLI / Web / Feishu -> AgentRuntime -> Agent
IPython ---------------------------> Agent
```

`pycodex.bootstrap` owns model/tool/Agent/session assembly. It imports neither
CLI nor Web nor Feishu presentation code. Optional `/link` and `/unlink`
handlers are injected by `build_runtime(agent)`; the Feishu adapter is imported
only when linking. A plain `AgentRuntime(agent)` has all core commands and
can register application-specific async handlers with `register_command`.

Every frontend calls `await runtime.submit_input(text, sender)`. The returned
`SubmittedInput` carries a submission id, kind and future. Normal text uses
steer; `/queue <text>` uses the ordinary queue. The backend parses `/help`,
`/history`, `/title`, `/model`, `/resume`, `/compact`, `/fork`, `/exit` and `/quit`.
Each queued request is one private record of submission id, turn id, texts and
futures. The unused public `Submission` / `UserTurnOp` wrappers are removed;
callers submit text through `submit_input` or `enqueue_user_turn`.
The workspace keeps queued submissions out of its displayed conversation.
`TurnStartedEvent` adds the actual user texts in execution order, including
coalesced steer inputs; enqueue admission alone does not create a user block.
The composer previews pending Steer and Queue text in small capsules beside
the status pill. The preview comes from existing queued submission metadata.
Commands return structured results and never become user messages. Unknown
commands and invalid arguments produce explicit command errors. Destructive
commands reject active or queued work; a command lock serializes asynchronous
commands and lets close wait for accepted command work.

Web command output preserves the event's plain text, including resume-list
numbering and line breaks.
Assistant replies render fenced `mermaid` blocks as diagrams. The browser loads
the renderer on demand, uses strict mode and sanitizes the generated SVG.
Diagrams fit the chat column; `Expand` opens a larger view with an `Actual size`
option for scrolling through wide diagrams. The original code remains available
under `Diagram source`. Loading or syntax errors leave the source visible.
A render that finishes after its conversation view has been replaced is discarded;
completed diagrams keep the view at the bottom only if it is already there.

Closing the active workspace tab uses the same browser transition as selecting
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

The workspace presents assistant replies as open text and user messages as
right-aligned bubbles. Commands and thinking use quieter secondary styling;
active thinking has a breathing left rule that respects reduced-motion settings.
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
After compaction or history restore, current usage is unknown until the next
token report; the ring resets and the details show that usage is pending.
Queue hints use neutral backgrounds and running status uses a small activity dot.
They share
the composer's outer frame in a compact top row separated from the text input;
the row collapses when there is no running status or queued input.
On screens up to 760px wide, icon buttons in the existing board/session headers
switch panes without a separate navigation row or changes to the active session
and draft. Desktop resizing and per-session scroll restoration retain their
existing behavior.

Frontends subscribe with `attach(handler)` and unsubscribe with `detach(id)`.
Attach delivers an immediate `session_state`: identity, model, title, canonical
history, active turn/partial output, usage, plan, input request and admission
state. Model/history/title changes are broadcast to every attached frontend.
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

`start()` is idempotent and owns the one queue worker. `close()` stops admission,
resolves waiting input, drains turns and child sessions, and runs injected
connection cleanup; it never cancels an Agent turn. Detach alone does not close
the backend. The CLI process and workspace session manager close sessions they
own; unlinking a Feishu card only detaches that frontend. Web submits directly
to the backend, including across its worker-thread boundary: it does not run a
CLI shell or maintain an extra prompt queue.

Workspace tab persistence follows recording changes as well as explicit titles.
Once a fork's first turn or successful compact creates its rollout, the saved tab
points to that file, including when the model subsequently fails. Before the
file exists, the tab stores the recorded ancestor's path with an explicit
`fork: true` marker. Restart resumes that source and calls `runtime.fork()` to
allocate a fresh identity and lazy destination; further work never appends to
the source. Repeated pending forks retain the same recorded ancestor. An empty
titled tab saves just its title until it has a recording. These are explicit
restore states: an invalid saved source path still raises an error and closes
the partially created session.
`Agent.recorded_session_file_path` and the snapshot's `recorded_rollout_path`
identify a successfully written or explicitly resumed recording. A lazy
destination has no recorded path yet. The runtime broadcasts a `recording`
state change after the first successful write; an unrelated file at the
destination does not count as that write.
The workspace manager keeps its persistence subscriber attached while sessions
drain during close, then waits for the final events before releasing its state.

The queue has no separate public `shutdown()` or worker-start entry point.
`close()` switches off `accepts_input` and wakes the worker. The worker exits when
the queues are empty; there is no special shutdown submission. Its result carries
cleanup errors for concurrent and repeated close callers. Closing also works
before `start()`. A cancelled close caller does not cancel the worker.
Cleanup visits every child and connection handler even if one fails; the first
error is propagated and additional errors are reported through the event loop.
The interactive prompt treats Ctrl+C as end-of-input, just like Ctrl+D on an empty
prompt, and follows the same `runtime.close()` path as `/exit`. The prompt toolkit
uses `interrupt_exception=EOFError` so a background input task cannot leak
`KeyboardInterrupt` into the event-loop runner. One Ctrl+C initiates normal exit;
the admission event renders a `[closing]` notice while accepted work and cleanup
finish. It does not interrupt model/tool calls or bypass their completion.
While closing, another Ctrl+C immediately terminates the CLI process with status
130. This also works after `/exit`, `/quit` or Ctrl+D starts closing. The CLI
installs a SIGINT handler for its interactive session, which prompt_toolkit
restores when the input prompt ends; it therefore does not depend on an inherited
handler that may ignore SIGINT. Normal completion and cleanup errors restore the
original handler. Forced exit happens at the process boundary, without cancelling
the turn or re-entering asyncio shutdown waits. It skips remaining cleanup and
does not guarantee termination of external tool processes. Runtime close itself
still waits for accepted work.
The prompt uses `set_exception_handler=False` so prompt_toolkit does not replace
the event loop's exception handler with its blocking "Press ENTER to continue"
screen. Background notification failures (including provider rate limits) remain
visible through the existing loop handler without pausing input.
The CLI still detaches its view on cleanup
failure, and the Web adapter stops and joins its thread after session cleanup,
including on failure; startup failure closes the partially started session.

IPython retains `ipython_agent()` returning a bare Agent with its kernel tool
attached. Direct `ask`, `run_turn` and idle-only background hooks stay available
without a submission queue.
This does not make arbitrary dependency replacement a supported mid-turn
operation. `history` remains an immutable snapshot; model-name and admission
properties still derive their values from the owning component.

`session_file_path=None` is the default and means no recorder: history stays in
memory. Passing a path creates an internal recorder. Session identity is
independent of recording; the Agent allocates a UUIDv7 id unless one is supplied.
`build_agent` supplies the id and default recording path for ordinary frontend
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

Unrecorded sessions update only in-memory history during turns and compaction.
Compaction omits the rollout-file reference from the handoff, and runtime
snapshots expose `rollout_path=None`. Fork preserves whether the session records;
no-argument resume preserves the current recorder. Explicit `resume(path)` loads
the selected file and enables recording to that file.

The outer queue remains optional: direct `ask` / `run_turn` calls need no worker.
Empty initial history and the no-op observer remain valid defaults. Tests isolate
`CODEX_HOME` per test so recording tests do not write to the real session store.

Turn execution has two public interfaces: synchronous `ask` and ordinary async
`run_turn`, both returning `TurnResult` once execution completes. `ask` runs the
same coroutine through the existing synchronous bridge. There is no separate
`start_turn`, `_start_task`, `_run_turn`, active Task reference, Task listener or
Agent cancellation API.

Creating a `run_turn` coroutine does not start work. The host may schedule it,
but the Agent itself simply executes in its caller's coroutine. Queue submissions
await `run_turn` directly. `maybe_invoke` returns `False` while busy or closed;
otherwise it formats the notification, awaits `run_turn`, and returns `True`.
Failures propagate to the invoking caller; exec/clock notification sources
report their own failures. When steer ends a notification turn with
`TurnInterrupted`, those sources treat it as a normal interruption and do not
report a notification failure. Runtime still owns the pending user input, and
the clock waits for the next successful reply before arming again.
Background events are not queued or cached.

One idle event is cleared on entry and set in `finally`, covering pre-turn
compaction, sampling, tool execution and manual compaction. `is_running` derives
from that event, and `wait_until_idle` waits on it without polling or reading an
execution result. Overlapping execution is rejected; there is no pre-sampling
Task reservation or cancellation policy.

The toolcall loop reads in order: auto-compact if needed, sample the model, commit
its output, then execute tools or finish. Threshold-triggered compaction has one
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

Steer and ordinary input remain two internal queues. Enqueuing steer calls
`agent.stop_asap()`. The Agent only records a local stop request: it finishes the
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

Direct `run_turn`/`maybe_invoke` callers receive `TurnInterrupted` if a Runtime
requests a stop. The Agent never consumes pending runtime input or completes
its futures, even when no worker is running; processing that input requires
`runtime.start()` (or a draining `close()`). There are no separate
input-inserted/submission-interrupted events: presentation uses the actual
turn-started/interrupted lifecycle and the existing queued-input feedback.

The queue is used from its owning event loop. Enqueue/dequeue mutations do not
await; asynchronous session commands and final resource cleanup use the command
lock. Cross-thread callers must
schedule operations on that loop rather than mutate the queue directly.
Worker notification is scheduled after enqueue returns, allowing cross-thread
submission receipts to be delivered before model execution can block that loop.

`runtime.is_busy` includes pending submissions, direct Agent work and closing
resource cleanup. Destructive
interactive commands such as resume/model/fork/compact use that same state.
Exec/clock notifications remain Agent-idle-only; they do not create a third queue.

`Agent.accepts_input` is a local admission flag; `runtime.accepts_input` reads it
without a reverse Agent-to-Runtime dependency. `runtime.close()` clears it,
resolves pending questions and drains existing entries. Background invocations
observe that same flag, while accepted work can still run until final shutdown.
The worker then waits for the command lock, closes children and connections, and
calls `Agent.shutdown()`. Tool shutdown hooks run once at that final boundary,
after accepted work can no longer rearm them. Completion comes from the worker,
not a separate shutdown future. Direct `Agent.shutdown()` disables background
hooks and rejects new calls without interrupting an active call.
`Agent.resume()` without a path explicitly reopens it. `close_agent` and workspace close wait
for accepted work to finish. There is no close timeout that cancels a turn:
if a request never finishes, close keeps waiting.

Sub-agent services observe ordinary Agent lifecycle events. They retain the last
completed/failed status, but active execution and pending inputs take precedence
when reporting busy. Direct calls and background invokes produce the same events;
status observers no longer inspect raw Task results.

## Observation

Runtime events are frozen dataclasses defined together in `pycodex/events.py`,
separate from the conversation and model data in `pycodex/protocol.py`.
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
Web retains its own projection; IPython keeps its tool-only printer. ANSI is
applied only at a terminal presentation boundary, never stored in event fields,
Web JSON or generated Feishu presentation. The module does not import terminal,
Web or Feishu implementations.

Retry events discard the failed attempt's buffered text, whereas fatal turn
errors flush it. Manual compact completion/failure events only update idle
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
split an outstanding call/result group. For example,
`WaitAgentTool` creates sub-agent notifications; the Agent does not decode that
tool's JSON output.
This local hook does not change tool request schemas or result serialization.

Tools with background work implement the small `BaseTool.bind_agent`,
`handle_agent_event`, `shutdown`, and `background_work_count` hooks.
Exec/clock tools wire their own managers;
the Agent does not inspect their names, concrete classes, or private attributes.
Clock countdowns pause on turn/manual-compact start events and rearm on success,
including successful manual compaction. Terminal events retain the single
aggregate `background_work_count`.

Host session workers, notification watchers, timer cancellation and I/O cleanup
still use asyncio where appropriate. Those resource mechanisms are not per-turn
Task ownership or user-requested turn cancellation.

Use a fresh registry per Agent. To supply runtime services explicitly:

```python
environment = AgentRuntimeEnvironment()
tools = get_tools(environment)
agent = Agent(client, tools, ContextConfig())
assert agent.tool_registry.runtime_environment is environment
```

There is no process-global default runtime environment. The former global getter
is removed; the Agent's environment comes from its registry, not a second
constructor argument or forwarding Agent property.
Context configuration may be shared as immutable data,
but child agents receive separate context managers and their own metadata cache.

## Model and history contracts

`ModelClient` requires both a read-only `model: str` identifier and async
`complete`. `Agent.model_name` reads `client.model` directly; test clients must
provide that property too. There is no missing-property fallback or second
model-name field. `ModelResponse` checks its item
types at construction instead of allowing the loop to silently discard invalid
internal values. Clients used for interactive model controls additionally expose
`ModelControl`. Switching through `Agent.set_model` updates both the client and
context; `ResponsesModelClient.model` uses the same provider config as metadata.

Only the Responses provider classifies external error text:

- Transient/disconnected streams use `ResponsesRetryableError`.
- Context overflow uses `ContextLengthExceeded`, with optional usage and limit
  parsed from the provider's error details.
- A genuine terminal incomplete response uses `ResponsesIncompleteError` and
  does not trigger provider retries.

Custom clients must raise the corresponding exception rather than rely on
arbitrary `RuntimeError` text being reinterpreted by the Agent or compactor.
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

Without a path, `resume()` re-enables the same in-memory Agent and rebinds tool
callbacks without reading or writing a file or changing history, session
identity, recorder or usage. It returns `None`. Sub-agent services use this form
after close to reopen their in-memory history. These forms are selected by the
argument, not by whether a file happens to exist;
an invalid explicit path never falls back to reopening the in-memory session.
There is no separate `reopen()` method.
Construction, `fork()` and `resume(path)` share one private recorder setup path
with explicit create/resume selection. It constructs the recorder before
updating session identity and does not change history, usage or admission.
`fork()` preserves history, usage and recording behavior. Recorded sessions
allocate a new path, with the file created only on first write; in-memory
sessions keep `session_file_path=None`.
The old recorder constructor argument and
`restore_session`/`set_rollout_recorder`/`replace_history` interfaces are removed.
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

The existing compact templates, append-only checkpoint format, truncation limits,
sampling-boundary steer behavior and Python 3.6 syntax requirement remain in place.
