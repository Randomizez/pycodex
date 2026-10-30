# Pycodex for VS Code

Licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).

Use pycodex's workspace frontend in VS Code's left sidebar. Start a session
and send ordinary messages, with optional code context inserted among the text
as editable tokens. The extension adds a Pycodex activity-bar icon and uses
the stable WebviewView API for its chat view.

The UI has Markdown replies, a fixed input area,
and light/dark themes with distinct panel borders and shadows. Sessions appear
as a stack in the sidebar. Opening a chat expands it and folds the previous one,
keeping its task running and its draft intact.

**The extension uses your existing `~/.codex/config.toml` by default.** It reads
the model and provider configuration on the machine running the workspace backend.
For Remote SSH, WSL and Dev Containers, that means the remote environment's
configuration. The welcome screen explains this before a backend is started.

## Python backend

The extension requires **`pycodex-ws>=0.4.0`** and Python 3.7 or newer.
On the first chat in a project,
it checks the installation on the machine that holds the project:

- Use the Python interpreter selected in **设置 → Python 环境**. If its installed
  version meets the minimum, reuse it without checking the network.
- Upgrade an older installation with its own
  `python -m pip install --upgrade "pycodex-ws>=0.4.0"`, then check again.
  Its exact core dependency is resolved in the same operation.
- With **自动检测**, reuse the interpreter behind `pycodex-ws` or `pycodex` on
  `PATH`, or find `python3` / `python` (`py -3` first on Windows).
- Install the package if missing, then run `python -m workspace_server` with
  that same interpreter. A selected venv does not need Pycodex installed in advance.

The extension uses the existing Python environment and keeps the host's HOME
and Codex configuration. It does not create a private environment or download
Python. If Python is missing, install it on the workspace host and reload the
window. If pip is missing, the environment is externally managed, or pip cannot
write to it, the error stays visible; resolve it in that environment or select
another **Python 环境**.

Installation failures show a short explanation, with the interpreter path and
sanitized pip output under **查看详情**. **重试** repeats startup; **打开设置**
opens the Python environment and Profile menu.

Installation has its own progress stage before the server startup timeout.
Concurrent projects sharing a Python environment share an in-progress install.
Closing a project during installation waits for pip to finish and prevents the
server from starting afterward.

The first chat in a project starts a dedicated `pycodex-ws` process automatically.
The extension keeps a fixed `workspaces.json` for each project's canonical path
under its `globalStorageUri/workspaces/<project-hash>/` directory on that host,
and passes it through `--workspace-config`. It sets
`state_path: "workspaces.vscode"` and `enable_board: false`, and selects an
available `127.0.0.1` port.
No server URL, workspace registration, or server password is needed.
The configuration and adjacent `workspaces.vscode.pycodex-ws.json` session index
remain after shutdown. Backend 0.4.0 adds this index for workspaces without a
Canvas. Opening the sidebar again starts the saved workspaces and loads their
conversations with history; a project used for the first time still starts from
the welcome screen.

## Python environment and Profile

The settings button in the chat header opens two actions:

- **Python 环境** lists the project's `.venv` and `venv`, the current selection,
  **自动检测**, and a file picker for another Python interpreter. For a venv, select
  `bin/python` (`Scripts/python.exe` on Windows). The selection is validated before
  saving; installation happens when you start a chat.
- **Profile** lets you edit the profile name from the workspace host's Codex
  configuration. For example, `step` selects `[profiles.step]`; leave it empty
  to use the default configuration.

Selections are saved for the displayed chat's project, or the current workspace
folder before a chat is opened. They can also be edited as VS Code settings:

```json
{
  "pycodex.pythonPath": "/absolute/path/to/project/.venv/bin/python",
  "pycodex.profile": "step"
}
```

Each folder in a multi-root workspace has its own settings and backend.
If a backend is already running, the menu offers **重新加载窗口** to apply a
change. Until then, existing sessions and new chats sharing that backend keep its
original environment and profile. Reloading waits for accepted tasks to finish.
With no running backend, the next chat uses the new selection.

For Remote SSH, WSL, or Dev Containers, install the extension in the remote
workspace and make Python available there. Version checks, pip installs and
backend startup all run on that remote host. The extension runs in the remote
workspace extension host and starts the Python process on that same machine.
Executable paths and `127.0.0.1` refer to the remote environment. The extension
runs in trusted filesystem workspaces. The webview sends messages to that
extension host; HTTP requests to Python also run on the workspace host.

## Build and install

For backend development from this checkout, run `env -u VIRTUAL_ENV uv sync --dev`
at the repository root and select its `.venv` in **Python 环境**, or set
`pycodex.pythonPath` to `.venv/bin/python` (`.venv/Scripts/python.exe` on Windows).
The required backend versions must be available on the package index used by
your users. If a release requires a new backend version, publish
`python-codex` first, then `pycodex-ws`, then the extension.
Python source distributions exclude the VS Code project; build the VSIX from
the repository checkout.

Requires Node.js 22 or newer and VS Code 1.100 or newer. From this directory:

```bash
npm ci --omit=optional --ignore-scripts
npm run package
```

Local VSIX packaging uses the JavaScript tools; optional signing and publishing
binaries and their install hooks are not needed. Compilation runs explicitly
through the package script.

### Publish to Visual Studio Marketplace

The publisher is `randomizez` and the extension ID is `randomizez.pycodex`.
Confirm the required Python backend versions can be installed from the package
index used by your users. Existing compatible releases satisfy this requirement.

Run `npm test` and `npm run package`, then upload `pycodex-0.0.4.vsix` through
the [publisher management page](https://marketplace.visualstudio.com/manage/publishers/randomizez)
as an update to the existing extension. For a first publication, use
**New extension → Visual Studio Code**. Later releases need a new version in
`package.json` and `package-lock.json`; keep the publisher and extension name
stable so existing installations receive updates.

The package includes its Apache-2.0 license, a PNG Marketplace icon, the changelog
and the bundled Markdown libraries' licenses. Packaging checks for the repository
and license without bypass flags. The `private` npm field prevents accidental npm
publication and does not make the Marketplace extension private.

### Install a VSIX locally

If you installed an earlier development build with ID `pycodex.pycodex`,
uninstall it before installing `randomizez.pycodex` to avoid duplicate commands
and sidebar views.

In VS Code, run **Extensions: Install from VSIX...** and select the generated
`pycodex-0.0.4.vsix`. To replace an earlier local build with a higher version
number, explicitly install this VSIX with:

```bash
code --install-extension ./pycodex-0.0.4.vsix --force
```

For an Extension Development Host instead:

```bash
npm run compile
code --extensionDevelopmentPath="$PWD" /absolute/path/to/your/project
```

## Use

- Click the **Pycodex** icon in the left activity bar, then **新建对话** to start
  a workspace session. A new session has no code context attached and does not
  require an open source file. Reopening the sidebar shows the selected conversation.
- To include code, place the cursor or select text, then press **Ctrl+Alt+P**
  (**Cmd+Alt+P** on macOS), run **Pycodex: Insert Code Context** from the editor
  context menu, or click the code button below the chat input. Diff editors support
  either focused side, including read-only Git revisions. You can also right-click
  one or more files in Explorer or Source Control and choose the same command.
- Each insertion appears as a file token at the message cursor, replacing any
  selected text. Tokens sit among your words like emoji. Click one to edit its
  content or remove it; Backspace and Delete remove a token as a single element.
  Long tokens are capped at 180px and show their full label on hover.
  Copy/paste within the composer preserves tokens; pasting elsewhere yields their
  expanded text. Send expands each token at its position into one ordinary prompt,
  preserving the order of text and code. If there is no session, inserting context
  creates one. It does not send a message by itself.
- **Enter** sends; **Shift+Enter** adds a line. Input-method composition does not
  send prematurely. **Send** submits through the backend's normal steer path.
  **Queue** uses `/queue`, so the message waits for the current task. Session commands such as
  `/model`, `/effort`, `/title`, `/history`, `/compact`, and `/resume` use the same
  backend command handling as the other frontends.
  The model label shows `model:effort`.
- **Pycodex: New Session** creates another backend session for the current
  or most recently focused editor's project. Without an editor, it uses the sole
  workspace folder or lets you choose one in a multi-root workspace.
  The sidebar header's **+** button uses the displayed chat's
  project. Switching to another opened session restores its draft.
  Previous chats stay visible as
  collapsed entries with live running, queued, or waiting-for-answer status.
  Click an entry to expand it.
- Code context is captured only when you insert it. Editing, switching or closing
  files does not change the draft or bind a session to a source location.
  After sending, subsequent messages have no automatically appended code context.
  Use `/resume` to restore a saved conversation.
- Pending questions and permission requests appear above the input. Reply using
  the backend's displayed choices, or use **取消问题**.
- The minus button collapses the conversation; its task continues in Python.
  Hiding the sidebar releases its polling subscriptions.
- The **×** button on an expanded or collapsed chat closes that session. It
  displays a closing state while Python drains accepted work, then removes the
  chat from the list. Other sessions keep running. A rejected close keeps the
  chat visible and displays the backend error.
- Type `/model` to list available models, or `/model <name>` to switch.
  The model name below the input is a read-only display.
  The backend rejects model changes while work is running or queued.
- The context ring in the input footer uses the same remaining percentage as the
  workspace web frontend. Hover, focus or click to see the latest model-reported
  token count, auto-compact threshold and maximum context length. With auto-compact
  enabled, 0% means its threshold has been reached; compaction runs before the next
  model request. With auto-compact off, the ring uses the maximum context length.
  It shows `—` when the limit is unknown. The ring does not estimate unsent drafts.
- **重试** repeats a failed startup or context insertion. For an existing session,
  **刷新状态** fetches its current state after a network or server error.
  Sends are never automatically retried because the server may already have
  accepted them. Failed sends keep the draft; an acknowledgement of an earlier
  send does not erase newer typing or another session's draft. Sent context blocks
  are cleared when unchanged; new or edited blocks stay in the draft. Drafts,
  context blocks and collapsed state survive hiding and switching the sidebar.

The excerpt contains only selected text. Without a selection, context records
the file and cursor location without including nearby code.
Each block records either the selection range or the cursor position, and omits
the language field.
Explicit Explorer/SCM file selections capture file contents. Each excerpt is limited to
200 lines and 12,000 source characters, with an explicit truncation marker.
Unsaved changes and the document version are included. Tools
still read and edit files on disk; save the document before asking the Agent to
apply changes to unsaved regions.

Session connections are in memory. Removing a folder or closing/reloading the
extension closes its backend's stdin pipe. The `pycodex-ws` CLI detects EOF
by default, drains accepted work and exits. Extension deactivation waits for
the processes it started,
including any already closing after a folder was removed. If the extension host
crashes, the operating system closes the pipe and triggers the same cleanup.
A temporary Remote SSH disconnection keeps sessions running while the remote
extension host is still alive. On restart, the backend reloads the saved
session index and rollouts, and the sidebar restores conversations with history
when opened. Empty sessions are not shown in the restored list. Rollouts from
older extension versions remain available through `/resume` in a new session.
When starting the CLI independently, keep stdin open for the server's lifetime.
Closed stdin, including a `/dev/null` redirect, causes a graceful exit.

## Frontend boundary

`WorkspaceProcess` starts the CLI with `--listen` and `--workspace-config <path>`.
The config file contains UTF-8 JSON; stdin stays open
for the lifetime of the project. `BackendClient`
owns a workspace endpoint and requires an explicit session id for
every session request. `SessionConnection` owns one session's snapshot, polling,
and subscribers; it has no VS Code dependency. Each `Chat` has a workspace folder
and a connection indexed by `(serverUrl, sessionId)`. `ChatView` is the sidebar
provider, rendering one expanded conversation and summaries of the others.
While visible it observes all opened chats, so folded entries retain live status.
Hiding it stops observation without closing those backend sessions.

A slow response from an old poll cannot overwrite a newer snapshot or update
another session. Webview actions carry the displayed session key, which the host
checks against its selection before sending. Acknowledgements also retain that
key so a completed send cannot erase another draft. There is no frontend prompt queue; all messages pass
through `workspace_server` to `AgentRuntime.submit_input`.

The editor helper captures a bounded excerpt and a display label for a draft
context block. VS Code supplies the document content, including virtual Git
revisions and file-tree resource URIs. Insertions keep the target session selected
when the action started, even if reading a file takes time, and wait for the
webview's ready handshake. The webview stores each draft as ordered text/token
parts with a cursor position, expands those parts in order, and passes the result
through the existing `prompt` field;
the Queue button adds `/queue`. There is no per-session source metadata or
automatic context appending. Commands and pending question answers use the
same message path, with no extra Runtime context handling.

The text/token composer is shared with the Web workspace frontend.
`npm run compile` copies its source from
`packages/pycodex-ws/workspace_server/frontend/composer.mjs` into the extension's
`out/` directory, so the installed VSIX contains everything it needs.

Closing uses `DELETE /api/sessions/<session_id>`. Polling and new sends stop for
that connection while deletion waits for backend shutdown; the ordinary HTTP
request deadline does not apply to this wait. The entry is removed only after
the server confirms completion.

The UI reads snapshots every 500 ms while active and every 2 seconds while idle,
as long as the sidebar is visible.
It uses the existing HTTP API and needs no additional Python transport dependency.
Markdown uses a bundled parser with raw HTML disabled. Images become links, and
the webview cannot make network requests. The extension host validates actions;
links can open only HTTP(S) URLs or workspace files, never command URIs.

## Validate

```bash
npm test
```

These tests exercise project process startup, stdin ownership and deactivation
waiting, remote hosting, venv selection and validation, scoped settings,
profile arguments and reload feedback, persistent project configuration,
conversation restoration and retry after a failed restore,
explicit session routing, stale polls, detach during a send, questions,
plain prompts, optional editor text, read-only Git diffs, Explorer/SCM selections,
asynchronous context capture, webview readiness, project selection,
collapsed chat status, `/model` commands,
session closing, context usage snapshots, and the webview message/CSP boundary.
They mock the VS Code API; run an Extension Development Host for a desktop UI check.

From the repository root, the backend ownership checks are:

```bash
env -u VIRTUAL_ENV uv run --dev pytest packages/pycodex-ws/tests/test_owned_workspace_server.py packages/pycodex-ws/tests/test_workspace_server.py
```

They cover UTF-8 workspace configuration files, default EOF shutdown without
an option, normal lifespan shutdown when stdin closes before or after startup,
and clean interpreter exit on SIGINT/SIGTERM while stdin remains open.
