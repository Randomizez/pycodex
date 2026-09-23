# 0.3.0 release notes

Release preparation dated 2026-09-23. This document does not indicate that
0.3.0 has been pushed, tagged, or published.

Both distributions use version **0.3.0**. `python-codex` owns the implementation
and console scripts; `pycodex-ws` remains a thin metapackage depending on
`python-codex==0.3.0`. Publish the core package before the metapackage.

## Runtime and frontends

- Keep the backend at two layers: `AgentRuntime → Agent`, with no Session
  wrapper or second frontend submission queue.
- CLI, workspace and Feishu share runtime commands, steer/queue handling,
  interactive questions, subscriptions and session state.
- Centralize CLI/Feishu presentation in immutable events and per-frontend
  `EventDisplay`; keep terminal I/O in `cli.py` and common assembly in
  `bootstrap.py`.
- Preserve serial turn execution, cooperative `stop_asap()` boundaries and
  graceful shutdown. Closing drains accepted work rather than cancelling turns.
- Allocate stable session identities at construction, create rollout files
  lazily, and keep resume/fork state and recording ownership inside the Agent.
- Keep compact handoffs in model context while hiding them from frontend
  conversation blocks; preserve subsequent replies through resume and fork.
- Return password login to the requested workspace path and query parameters.

## Context and protocol

- Refresh shared model prompts against Codex CLI 0.153.4 and its matching source.
- Resolve model metadata by longest slug prefix without changing the wire model
  name, including suffixed deployment names.
- Align developer-section order, skill root aliases, YAML invocation policy,
  model-specific skill guidance and filesystem context.
- Preserve assistant IDs/phases/text parts, raw tool arguments, tool metadata
  and reasoning through history replay; use stable Responses-lite prefix IDs.
- Add a reproducible offline request audit. Shared context matches the documented
  first/resume/tool scenarios, but complete raw request parity is not claimed.
  Identity, tool-catalog, collaboration, telemetry and restricted-permission
  differences remain documented in `CONTEXT.md`.
- Include upstream downstream-SSE error propagation and trajectory finish/error
  metadata, including typed causes without copying credential-bearing messages.
  Port invalid stdin-session display handling into the new event layer.

## Python API migration

- Construct `Agent(model_client, tool_registry, context_config)` with all three
  arguments. Configure prompts/cwd through `ContextConfig`; the client must
  expose `model`. Do not inject a shared context manager or rollout recorder.
- Await `agent.run_turn(texts)` directly, or use synchronous `agent.ask(text)`.
  There is no Agent-owned turn Task or cancellation API.
- Use `AgentRuntime.start()` / `close()` and `submit_input()` for frontends.
  The old submission-queue aliases and separate interactive-session module
  are removed. IPython deliberately continues to expose a bare Agent.
- `session_file_path=None` means an in-memory session without recording.
  Frontend `build_agent` supplies the default recording path explicitly.
  Use `runtime.resume(path)` or `agent.resume(path)` for an existing rollout;
  a constructor path is only for a new file.
- Collaboration-mode configuration, prompts and tool gating are removed.
  Registered handlers now control interactive questions independently of mode.

See `RUNTIME.md` and the README API examples for the full contracts.
PyYAML is now a runtime dependency for skill metadata; both Python 3.6 CI jobs
include it in their explicit dependency lists.
