# pycodex-ws

Browser conversations, a live HTML Canvas, read-only file/diff viewers and
optional Feishu integration for
[`python-codex`](https://pypi.org/project/python-codex/).
Requires Python 3.7+ and installs the matching core version.

See the [0.6.0 release notes](https://github.com/Randomizez/pycodex/blob/v0.6.0/docs/RELEASE_0.6.0.md) for changes and
integration migration from 0.5.0.

## Quick start

```bash
python3 -m pip install pycodex-ws
```

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

Open `http://127.0.0.1:6007`. Use `python3 -m workspace_server` if the command
is not on `PATH`. Keep stdin open while the server runs; EOF initiates shutdown.
Model configuration comes from `~/.codex/config.toml`.

## Conversations and views

The session sidebar starts collapsed. Hover to preview it, click to keep it
open, and click the toggle or empty space again to close it. Expansion narrows
the conversation while keeping the board fixed. On phones, use the header
buttons to switch between chat and board.

Use the expanded sidebar to create, rename or close sessions. Status dots show
running sessions and unread replies. `/cd <directory>` changes the current
session's working directory while idle; `/model`, `/resume`, `/compact` and
other core session commands are also available.

Use **New session** for the default profile, or its **+** button to choose one.
Profiles reload when you open the chooser or create a session, so config changes
apply to new sessions without restarting the server. Existing sessions keep their configuration.

Consecutive model reconnection diagnostics share one collapsed `Model retries`
row per sequence. The summary shows the number of recorded retries, the latest
attempt and its HTTP status when available. Expand it to read the original errors
in a scrollable log. New retries update the same row and preserve its open state;
refreshing starts collapsed. Other model diagnostics and final errors stay visible.

Click Markdown links to open files and comparisons, for example
`[source](app.py:42)` or `[changes](diff:HEAD:app.py)`. In a code or diff view,
select text and choose **Insert into chat** to add editable context to the draft.
Click a context block to edit or remove it before sending.
See [view links and supported formats](../../docs/RUNTIME.md#workspace-content-views).

When using a router, its directory groups workspaces by device and lists unread
workspaces at the top. Device handles reorder the list; pencil buttons edit
device names.

The runtime information shared between Agents recommends plain messages, which
steer active work at the next safe boundary. Use `/queue` explicitly for an
independent follow-up task that should wait for current work to finish.

The message API also accepts a string `name`, defaulting to `"user"`, for
identifying the source of a user message. The history API returns it, and
recorded sessions preserve it across resume. It is metadata only: the model request and frontend display do
not include it. See the [message API contract](workspace_server/WORKSPACE_API.md#commands-and-results).

The [VS Code extension](extensions/vscode/README.md) uses this backend and is
distributed separately as a VSIX.

## Customize the interface

Click the gear button and choose **Frontend customization** to start a
customization conversation with the Agent.
Custom files live in the server user's `~/.pycodex/frontend/`:

- `custom.css` and `custom.js`: optional styles and an ES module, loaded automatically.
- `workspace.html`: optional full-page override; otherwise the packaged page is used.
- Other files: available at workspace-relative `frontend/<path>` URLs.

Refresh to apply changes and keep the selected conversation. Open the workspace
with `?builtin=1` to bypass all customizations; click “Re-enable customizations”
to return. Changes apply to all workspaces for that OS user.
See the [customization contract](../../docs/RUNTIME.md#frontend-customization).
For custom interfaces, the [resource API](../../docs/WORKSPACE_API.md) exposes
state, model catalogs, asynchronous command results, paginated history, event
replay and sub-agent/background-work inventories. A `mount(host)` export in
`custom.js` can access the built-in page's session, draft and view controls.

Backend module ownership and the preserved session interfaces are documented in
the [runtime guide](../../docs/RUNTIME.md#workspace-session-api).
File/Git source resolution and view storage retain their existing public classes;
see [content views](../../docs/RUNTIME.md#workspace-content-views).

Ask the Agent to show the [feature catalog](workspace_server/optional_features/README.md)
in the view area, then choose features by name or ID in chat. The Agent adapts
their source into your frontend. To share a local change, ask it to publish a
feature PR on GitHub; the [contribution guide](workspace_server/optional_features/CONTRIBUTING.md)
specifies the source, documentation, validation and PR format.

## Share a workspace

Start the workspace server with an administrator password (`--password`), then
choose **Settings → Share** in a workspace. Set a different share password and
send the displayed workspace link and password to the recipient.

The existing login page accepts either the administrator password or the share
password for the workspace being opened. Visitors can use all its conversations,
Canvas, files, APIs and WebSocket connections, but cannot access workspace
management, another workspace or sharing settings. The settings button and
management navigation are hidden for visitors.

Passwords, verification values and visitor tokens are kept only in process
memory. Changing the password or stopping sharing invalidates existing visitor
logins and disconnects their WebSockets. Restarting the process clears all
shares. Already accepted Agent work continues.

This restricts web access; Agent tools retain their existing file and execution
permissions. An unprotected server must configure its administrator password
before enabling sharing. Router authentication is unchanged: a link through a
router still requires its normal login.

## One-line Agent

An Agent that runs directly in most shells without installation, for bootstrapping
an Agent environment or simple debugging.

Open **Settings → One-line Agent**, choose a profile and click **Enable One-line Agent**.
This enables a reusable API and displays Bash/curl and PowerShell (`irm`)
one-liners. Replace the example task `看一下这台机器` in the command, then paste
it into the target terminal. The `T` header contains the 12-character token followed
by a session suffix, initially `1`. Change only that suffix to `2`, `3`, etc. for a
separate session or another terminal. Each suffix creates its own temporary session with
independent history and command results; its tab title is initially `OneLineAgent-<suffix>`.
Leave the terminal loop running and continue the conversation or steer the Agent
from its workspace session. When the session is idle, you can also edit the task
and rerun the one-liner with the same key to continue with its existing history.
The new caller takes over any previous waiting terminal connection.
After closing that session
in the workspace, the same key starts a fresh conversation. Enabling the API alone creates no session.
Each run starts immediately without reading interactive input. The full client
loop is in that command; no launcher script, Python, Node or jq is downloaded.
Both clients use `POST /b` relative to the workspace URL. The API takes no
platform parameter: it returns the next command as plain UTF-8 text and waits
when no command is available. Only the local client chooses the interpreter.
The model and instructions come directly from the selected profile, without
workspace CLI overrides.
Host environment and Git metadata are excluded from model requests; the Agent
learns about the client environment through command results.

The Agent runs on the workspace server, but its sole `exec_command` tool executes
on the client. Commands, results and the conversation are visible in the bound
session. The executor accepts only `cmd`, waits for completion, and returns combined
stdout/stderr and the exit code. Commands run in separate, noninteractive processes
in the terminal's current directory. No cwd is collected or sent, and no directory
is selected automatically; include `cd` in a command when needed.
The prompt warns that synchronous execution blocks further commands until it
returns, and directs the Agent to avoid unbounded waits and foreground services.
The terminal displays the command, output and exit code together after execution.
The server automatically wraps each final response in a quoted `echo` command
for terminal display. Quotes and shell substitutions in the response are treated
as text. The display command's result, including failure, is consumed by the
server without another model call. Commentary and model errors remain in the
workspace session. The terminal loop continues after displaying the final response.

The workspace-scoped token can be reused with different session
suffixes. The first request for a suffix sends only the task and immediately
creates the session. While a command is waiting, requests with the same complete
key send combined output followed by `LF + [exit N]`. The request then waits for
the next command, including across completed, failed or steered turns, without
polling or empty completion responses. Further prompts use the ordinary Web session
input, or a new terminal request while the Runtime is idle with no command awaiting
a result. An idle takeover releases the old waiting request with HTTP 409.
Steer takes effect after an already-issued synchronous command finishes.
Network and HTTP errors stop the client; there is no handshake, retry or response
replay. Completing a task leaves the terminal connected. Close the session, use
**Stop**, or interrupt the local loop to disconnect it. Use **Stop** before
enabling another service. Stopping or a server restart revokes access
for every run and closes their sessions. An already running client command is not
killed by stopping the server service.

The service, token and session tabs are not restored after a restart; recorded
history is kept. Both clients send UTF-8 text, and each request is limited to 1 MiB.
Management requires workspace administrator access. The terminal token grants
access only to this execution channel, including through a password-protected
Router; it does not grant access to the normal workspace APIs.

The initial installation needs updated code and a restart on both the workspace
server and any Router serving it. Workspace Reload does not update a separate
Router or replace authentication middleware created at startup. A 401 response
with `{"error":"authentication required"}` comes from cookie authentication before
the terminal endpoint; that endpoint reports an invalid token as
`{"detail":"invalid One-line Agent token"}`.

## Storage

Paths in `workspaces.json` resolve from the configuration directory.
`state_path` is a unique filename prefix: the example saves sessions in
`demo.pycodex-ws.json`, file views in `demo.pycodex-views.json`, and Canvas in
`demo.html`. Suffixes are appended to the complete prefix, including any dots.

Set `enable_board` to `false` to disable Canvas while retaining conversations
and file views. Session directories and history survive server restarts;
changing a session's directory does not move Canvas or workspace state files.
Configuration and restoration details are in the [runtime guide](../../docs/RUNTIME.md).

## Feishu

Install the optional SDK in the workspace's Python environment, then authorize:

```bash
python3 -m pip install lark-oapi
python3 -m pycodex_ws.feishu_oauth
```

Authorization reads app credentials from `~/.codex/.env` and stores the refresh
token in `~/.codex/.feishu_refresh_token`. In a workspace session, use
`/link <feishu-email|open_id|chat_id>` to connect a card and `/unlink` to disconnect.
Other workspace features work without the SDK.

The adapter separates card rendering, refresh-token storage and SDK connections.
`PycodexCard` and `PycodexRuntimeLink` keep their existing imports and interfaces;
see [connection lifecycle](../../docs/RUNTIME.md#feishu-connection-lifecycle)
and [event presentation](../../docs/RUNTIME.md#observation).

## Development

From the repository root:

```bash
uv sync --dev
uv run pycodex-ws --help
uv run --dev pytest packages/pycodex-ws/tests
uv build packages/pycodex-ws
```

Python integrations construct a runtime through `AgentsManager.create_root_runtime()`
and call `pycodex_ws.register_feishu_commands(runtime)` to include `/link` and
`/unlink`. The context indicator shows tokens against the auto-compaction limit;
its expandable Agent tree shows each task's status and latest assistant text.
See [runtime contracts](../../docs/RUNTIME.md) and
[test documentation](../../tests/TESTS.md).
