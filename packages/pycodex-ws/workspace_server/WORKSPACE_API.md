# Workspace resource API

This reference ships with `pycodex-ws`, beside `workspace.html`. The interface
customization prompt includes its absolute path so the Agent can read the
installed API contract without a repository checkout.

Custom frontends use the same authenticated backend as the built-in workspace.
Paths below are workspace-relative: use `api/...` from its page, or prefix with
`/w/{workspace_id}/` on a multi-workspace server. Existing endpoints remain
available. Resource responses include `api_version: 1` and use
`Cache-Control: no-store`.

## Workspace sharing

`GET api/share`, `PUT api/share` and `DELETE api/share` require the server's
administrator identity. PUT takes `{"password": "..."}`; the password must be
nonempty and differ from the configured workspace server administrator password.
Sharing is unavailable when that administrator password is unset.

The response is `{enabled, available, url}`. `url` is the workspace's existing
page path including any mount prefix. GET never returns a password or token.
DELETE disables sharing; PUT rotates its password and visitor grant.
These endpoints require a JSON object for PUT and return `Cache-Control: no-store`.

`POST /login?next=<workspace-path>` reuses the normal login form. The administrator
password grants administrator access; a matching share password grants access
only to that workspace. The server checks the cookie's grant on every HTTP and
WebSocket request. Visitor login redirects are restricted to the granted
workspace. Passwords and tokens have no persistence, and password changes,
revocation or process restart invalidate old grants. Revocation closes existing
visitor WebSockets without cancelling accepted Agent work.

The rendered page has `data-access-role="admin"` or `"visitor"` on its body.
Custom frontends can use `__ACCESS_ROLE__` to hide management controls; authorization
is enforced on the server regardless of UI visibility. Sharing does not reduce
Agent execution permissions, and does not replace Router authentication.

## Creating sessions

`GET api/sessions` returns `{sessions, profiles, default_profile}`. `profiles`
contains only names from the server's Codex config; no provider configuration
or credentials are returned. The catalog is loaded when the workspace is built.
`default_profile` is the launch `--profile`, or `null` for the base configuration.
Each session list entry includes its effective `profile` name or `null`.

`POST api/sessions` accepts `{title?, prompt?, profile?}`. Title and prompt are
strings and default to empty. Profile is a string name; omission or `null` uses
the server default. An unknown name returns HTTP 400; an invalid field type
returns HTTP 422. An empty body still creates a default session.
The selected profile configures both the model and Agent instructions; CLI
overrides still apply. A nonempty prompt is submitted through Runtime.
The response contains `ok`, `session_id`, `sessions`, and `snapshot`, plus the
submission receipt when a prompt was supplied.

The saved tab stores the selected name, including an explicit `null` for the
base configuration. Restart resolves that name against the current config,
even when the launch default changed. Older tabs without a profile use the
launch default. A missing saved profile fails restoration. Existing sessions
keep their configuration; this API does not switch profiles in a running tab.

## PDF previews

`POST api/views/open` with `{"href": "report.pdf"}` opens a local PDF using the
normal file-view path resolution. Its document has `media_type: "application/pdf"`
and source metadata, without text or embedded file bytes.
`GET/HEAD api/views/{id}/pdf` streams that view's current file with
`Content-Type: application/pdf` and inline disposition; `?download=true`
selects attachment disposition. These routes require workspace authentication.
Closing the view removes access through that ID. They accept no file-path input.
PDF files do not use the text/image snapshot size limit; invalid headers are
rejected. `revision` in the iframe URL changes when file metadata changes.

## State and resources

| Method and path | Response fields |
| --- | --- |
| `GET api/sessions/{id}/state` | `state`, `cursor` |
| `GET api/sessions/{id}/models` | `model`, `models` |
| `GET api/sessions/{id}/history?limit=100&before=...` | `source`, `items`, `next_cursor` |
| `GET api/sessions/{id}/agents` | `agents` |
| `GET api/sessions/{id}/background-work` | `background_work` |
| `POST api/sessions/{id}/commands` | `submission_id`, `submission` |
| `GET api/sessions/{id}/submissions/{submission_id}` | `submission` |

`id` is the tab ID from `GET api/sessions`. It remains stable when `/fork` or
`/resume` changes the Agent identity. `state.agent_session_id` identifies the
current Agent/rollout session.

State includes:

- `model: {name, reasoning_effort}`, `title`, and `cwd`;
- `busy`, `running`, `closed`, and `accepts_input`;
- `active_turn`: IDs, user texts, accumulated assistant text and model errors,
  or `null`; completed history is excluded;
- `queued_inputs`: `submission_id`, `turn_id`, `queue` (`steer` or `enqueue`)
  and `prompts`, in execution order; merged steer inputs share one ID;
- `commands`: currently queued/running commands with their submission IDs;
- `input_request`, `plan`, and `background_work_count`;
- `context_window`, `max_context_window`, `auto_compact_token_limit`,
  and the latest `usage_tokens`.

`running` describes Agent execution; `busy` additionally includes queued turns
and draining. Command status is separate, so a custom command can run while the
Agent is idle. State/model queries do not add control messages or invoke a model
completion.
Workspace model clients implement `ModelControl`, including an explicit
`reasoning_effort` (possibly `None`) and `list_models()`. The core Agent's
minimal `ModelClient` protocol is unchanged.

`agents` is a flattened descendant tree with `agent_id`, `parent_agent_id`,
`nickname`, and the existing sub-agent `status` value. The root is identified
by `state.agent_session_id`. Descendant IDs are assigned by their manager and
remain stable across that child's fork/resume. Add `?agent_id=...` to state,
models, history, agents, background-work or commands to target an owned child.

Background records have `tool` and `kind`. Exec records include `session_id`,
`pid`, `tty`, `status` (`running` or `exited`) and `exit_code`. They omit shell
arguments, environment and process output. Entries disappear when the exec
manager releases them; this is an inventory, not an execution archive.
Clock records include `enabled`, `period_m`, `status` (`disabled`, `paused`,
or `armed`) and timezone-aware `next_fire_at` (`null` when not armed).

## Commands and results

Commands use a name without the slash and an optional string argument:

```js
const response = await fetch(`api/sessions/${sessionId}/commands`, {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({command: "effort", argument: "high"}),
});
if (!response.ok) throw new Error((await response.json()).detail);
const {submission_id} = await response.json();
const result = await fetch(
  `api/sessions/${sessionId}/submissions/${submission_id}`,
).then(response => response.json());
```

Accepted requests return HTTP **202**, including fast commands. Records contain
`submission_id`, `agent_id`, `kind`, `status`, `result`, and `error`; commands
also have `command`. Status progresses through `queued`, `running`, then
`completed` or `failed`. Failures have `error: {type, message}`. Query until
terminal or consume WS `submission` notifications. A 202 receipt does not imply
execution succeeded.

Names come from Runtime's command registry, including workspace commands.
`queue` is excluded here: send `/queue <message>` through the message endpoint.
Operations retain their idle/admission requirements. Runtime owns serialization
and draining. HTTP request completion or disconnect does not cancel an accepted
command. This adds no turn cancellation, queue editing or arbitrary tool execution.

Regular workspace message receipts are also queryable. Their successful result
contains `turn_id`, `output_text`, and `iterations`. Command events observed
from other frontends are retained as well. Retention is **all pending submissions
plus the last 500 completed submissions** in the workspace session's memory,
shared across its observed agents. Restart/deletion clears this store;
expired/unknown IDs return 404. Submission lookup does not need `agent_id`.

Malformed input returns 422, missing resources return 404, and admission rejected
because the Agent is shutting down returns 409. Execution failures after
admission appear in the submission record.

## History pagination

`limit` is between 1 and 200. The initial page contains the newest records;
each page is chronological. Pass `next_cursor` as `before` for older records;
stop at `next_cursor: null`. Cursors are opaque.

Items contain `index`, `type`, and `payload`. A `response_item` carries the
serialized conversation item, including tool calls/results and available
reasoning records. This does not decrypt reasoning or reconstruct unrecorded
output. There is currently no dedicated live reasoning-delta event; the display
snapshot's `thinking` field contains assistant text and is not a reasoning API.
A `compacted` item contains its summary in `payload.message`; it does
not duplicate the checkpoint's replacement history.

Recorded sessions use `source: "rollout"` and retain pre-compaction records.
Session metadata and duplicate display events are excluded. Reading uses bounded
page memory and scans the rollout up to the requested record; no separate
database/index is maintained. File I/O runs off the session loop.

Unrecorded sessions and forks before their first write use
`source: "active_history"`; only current in-memory history is available.
Appending preserves older cursors. Compaction, a changed session/source, or
the transition from unwritten to recorded history can invalidate a cursor;
HTTP 409 directs the client to restart pagination. Sources are append-only;
external rollout edits are not a supported pagination workflow.

## WebSocket state and replay

Connect to `ws/session?session_id=...`, optionally with `cursor=...` and
`agent_id=...`. Existing authentication and same-origin checks apply.

`hello` retains `snapshot` and `events` and adds `api_version`, `state`, `cursor`,
and `reset_required`. Subsequent events include a cursor and lightweight state.
The legacy display snapshot remains on its existing event types.

- Without a cursor, hello contains the last 200 cached events.
- A valid cursor replays events strictly after it, up to the 500-event window.
- An expired cursor, different stream or future sequence returns
  `reset_required: true`, no replay events and a fresh snapshot/cursor.
- Invalid cursor syntax produces a WS error and closes the connection.

Hello's state is authoritative at its cursor; older events fill the timeline
and should not overwrite that state. Process live events in cursor order and
save the last processed cursor for reconnecting. State snapshots, cursor
allocation, hello subscription and publication share a lock across worker/server
threads.

The stream includes structured Runtime/tool/model events, `runtime_updated`
notifications for plan, queue, command and resource changes, and `submission`
notifications containing result records. Agent/background change notifications
tell clients to re-query the inventory.

Events are memory-only and retained by count. Replay guarantees apply inside
that window; a rollout is not a persistent timing/retry/WS log. Child streams
are created when first observed and have separate cursors, excluding earlier
events. Their WS connection is read-only: `ping` works; `send`/`answer` is rejected.
Observing a child does not enable interactive requests or rollout recording.

## Built-in page extension host

An optional `mount(host)` export in `~/.pycodex/frontend/custom.js` is called
once when the built-in page initializes. Side-effect-only modules still load
normally. Initialization failures are shown in the page. `?builtin=1` disables
loading and mounting. A complete HTML replacement owns its initialization.

```js
export function mount(host) {
  const update = async sessionId => {
    if (!sessionId) return;
    const {state} = await host.request(`api/sessions/${sessionId}/state`);
    if (host.getActiveSessionId() === sessionId) {
      document.body.dataset.agentBusy = String(state.busy);
    }
  };
  update(host.getActiveSessionId()).catch(console.error);
  host.onSessionChanged(id => update(id).catch(console.error));
}
```

The frozen host has `apiVersion: 1`:

| Method | Contract |
| --- | --- |
| `getActiveSessionId()` | Current tab ID; initially may be empty |
| `selectSession(id)` | Switch to an existing tab; returns a promise |
| `onSessionChanged(handler)` | Subscribe; returns an unsubscribe function |
| `getDraft()` | Structured composer draft, including context parts |
| `setDraft(draft)` | Replace current draft; `draft.parts` must be an array |
| `openView(href, srcPath)` | Open a file/Git/workspace link for the current session |
| `request(path, options)` | Fetch workspace-relative JSON; HTTP failures reject |

Tab-change callbacks run after the synchronous draft/view switch. Check the
current session when applying asynchronous results. Host state is browser-local;
this does not add cross-device draft storage.
