# 0.7.0 release notes

This release brings VS Code into the browser workspace and lets a Web
conversation run commands on a connected terminal. These notes cover the
changes since **0.6.0**, including 0.6.1, 0.6.2 and 0.6.3.

## Browser workspace

- **Embedded VS Code.** Start the workspace with `--password`, sign in as an
  administrator and use the VS Code button beside Canvas. The host needs a
  VS Code Web Server. Set `PYCODEX_VSCODE_WEB_SERVER` if it is outside the
  standard install locations. With the editor open, links to local files, Git
  revisions and diffs open at the requested line in the browser tab where you
  clicked. VS Code handles large files using its own editor limits.
- **External Call replaces One-line Agent.** Open **Settings → External Call**,
  select a profile and run the session's Bash or PowerShell connection command
  in the target terminal. Send tasks and read replies in the Web conversation.
  The agent can use `exec_command` on the workspace host and `client_exec_command`
  on the connected machine. The terminal returns command output and exit status.
  Closing the session revokes its connection code.
- Markdown `file:///` links open in the workspace, and links to sections of
  local HTML pages open at the anchor in preview. Conversations show message
  timestamps and reply duration. Optional features add voice dictation, file
  attachment previews and retries, skill completion, session sorting and
  spoken replies on a XiaoAI speaker.
- New sessions pick up profile, environment and model metadata changes.
  Settings can reload compatible backend changes in place. Router browser links
  use five-character device prefixes and existing links continue to work.

## Model providers and retries

- Set `compat_api` to `none` for native Responses, `chat_completions` for
  Chat Completions, `messages` for Claude Messages or `gemini` for the Gemini
  API. The Gemini adapter streams function arguments, preserves thought
  signatures across turns and carries image results back from tools. Provider
  configuration now accepts custom HTTP headers and headers supplied through
  environment variables. [Configuration examples](https://github.com/Randomizez/pycodex/tree/v0.7.0/config_examples)
- The model catalog adds GPT-6.2 metadata and Antigravity-based instructions
  for Gemini 4. Compaction preserves the sampling prefix for prompt-cache reuse.
- Use `/retry` while idle to resume the last failed model request.
  `/retry 429|502` retries either selected code indefinitely on the same model.
  `/retry *` includes every model request error, including disconnects.
  `/retry off` restores the normal retry policy.

## Python integrations and tools

- Pass an iterable of `UserMessage`, `AgentMessage` or `ToolResult` objects to
  `Agent.run_turn`. Models and tools publish events with `EventScope.emit(event)`.
  Custom model clients supply token counts in
  `ModelResponse.usage` for goal accounting and automatic compaction.
  `AgentRuntime.submit_input(..., trigger=False)` queues context for the next
  turn or sampling boundary without starting an idle agent.
  [Runtime contracts](https://github.com/Randomizez/pycodex/blob/v0.7.0/docs/RUNTIME.md#ownership)
- Use `clock.set_period` for periodic wake-ups and `clock.sleep` for a single
  wait. Background shell commands can request an idle-agent wake-up with
  `exec_command(..., wake_me_up_when_done=True)`. The inactive model-side
  `request_user_input` declaration has been removed.

## Upgrade

Install the browser workspace and its matching core on Python 3.7+.

```bash
python3 -m pip install --upgrade pycodex-ws==0.7.0
```

For a core-only installation on Python 3.6.2+, use the following command.

```bash
python3 -m pip install --upgrade python-codex==0.7.0
```

Restart running services after upgrading and refresh the browser.
