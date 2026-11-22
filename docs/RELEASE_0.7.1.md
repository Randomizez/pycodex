# 0.7.1 release notes

Changes since 0.7.0:

- Add and publish `pyagy` alongside `python-codex` and `pycodex-ws`: an
  Antigravity CLI (`agy 1.5.14`) wire-compatible Gemini harness with standalone
  CLI (`pyagy`) and Web/Feishu profile support (`harness = "pyagy"`). Includes
  10 KiB middle truncation with persisted step output files, anchor-based
  context token accounting, automatic `ContextLengthExceeded` overflow
  compaction, streamed reasoning previews, and persistent background schedule
  tracking (`idle: sleeping`).
- Keep the router connection alive in the outer shell across graceful in-process
  workspace reloads, and assign an ephemeral loopback port when `--listen-to` is
  used without `--listen`.
- Render local workspace images in chat replies and Markdown previews, collapse
  wake-up and background notifications into compact tag dividers, share one
  four-line scrollable Markdown block between reasoning and reply previews, and
  hide the default `user` label in chat message headers.
- Fix External Call PowerShell `${u}` URL interpolation and stream oversized
  Bash/PowerShell command outputs without entering retry loops on HTTP errors.
- Recognize Gemini input-token-limit errors as `ContextLengthExceeded` in core
  without exhausting stream retries.

`python-codex`, `pyagy` and `pycodex-ws` are all version 0.7.1 and are published
in order (`python-codex -> pyagy -> pycodex-ws`). `pyagy` depends on exactly
`python-codex==0.7.1`; `pycodex-ws` depends on `python-codex==0.7.1` and
`pyagy==0.7.1`. Python support remains 3.6.2+ for core and 3.7+ for `pyagy` and
Workspace.

Upgrade Workspace, `pyagy` and core together:

```bash
python3 -m pip install --upgrade pycodex-ws==0.7.1
```

For `pyagy` and core without the browser workspace:

```bash
python3 -m pip install --upgrade pyagy==0.7.1
```

For a core-only installation:

```bash
python3 -m pip install --upgrade python-codex==0.7.1
```

Restart running services after upgrading, then refresh the browser.
