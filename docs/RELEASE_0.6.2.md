# 0.6.2 release notes

Changes since 0.6.1:

- Add One-line Agent sessions: enable the endpoint in Workspace settings, then
  use a Bash or PowerShell one-liner to execute the Agent's commands on the
  connected client. Sessions support continued conversations from the terminal
  or Web and display final responses in the terminal.
- Add Step Plan voice dictation with draft cleanup and optional Web skill
  completion. Display message timestamps and reply duration, and preserve
  message metadata across history, runtime and views.
- Add in-place backend reload from Workspace settings while retaining sessions.
- Shorten Router browser links to 5-character prefixes while retaining full
  device identities, existing links and registration compatibility with older
  clients. Preserve aliases when migrating from the intermediate short-ID format.
- Omit unsigned empty Gemini text parts when replaying history, preserving
  thought signatures and avoiding invalid continuation requests.
- Add configuration templates and an Agent setup guide for Responses, StepFun
  Coding Plan, Chat Completions, Claude Messages and Gemini in
  [config_examples/](../config_examples/README.md).

Both `python-codex` and `pycodex-ws` are version 0.6.2. Workspace depends on
exactly `python-codex==0.6.2`; core is published first. Python support remains
3.6.2+ for core and 3.7+ for Workspace.

Upgrade Workspace and its matching core:

```bash
python3 -m pip install --upgrade pycodex-ws==0.6.2
```

For a core-only installation:

```bash
python3 -m pip install --upgrade python-codex==0.6.2
```

Restart running services after upgrading, then refresh the browser.
