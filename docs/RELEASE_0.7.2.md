# 0.7.2 release notes

- On Windows, `pycodex-ws` can wait on an idle console or stdin pipe while a new
  session starts. The workspace server continues serving the home page and
  workspace list during session thread startup, including in `--listen-to` mode.
- Core command tools select an available Windows shell. `pyagy` runs commands
  through pipes on Windows and cleans up the command's process tree on cancellation.
- `pyagy` prevents interactive pagers and inherited output handles from keeping
  commands open indefinitely. It also wakes the parent session when a subagent
  fails, is interrupted, or finishes waiting for background work.
- The workspace sidebar lets you confirm or cancel closing a session directly
  in its row.

To upgrade the workspace and its matching `pyagy` and `python-codex` dependencies,
run the following command with the Python interpreter that runs the workspace.

```sh
python -m pip install --upgrade pycodex-ws==0.7.2
```

Restart the `pycodex-ws` process after upgrading so the new stdin watcher takes
effect.
