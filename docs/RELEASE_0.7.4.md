# 0.7.4 release notes

Changes since 0.7.3:

- Include `responses_server` in `workspace_server.reloader.RELOAD_PREFIXES` and
  self-heal older in-memory reloader instances on `workspace_server.app` import,
  so upgrading `pycodex-ws` via **Settings → Update & reboot** reloads
  `responses_server` alongside `pycodex` and `pyagy` without requiring an OS
  process restart (fixing `pyagy` / Gemini peer profile creation after
  in-process upgrades from 0.7.2).
- Parse non-JSON HTTP error bodies and `detail` payloads cleanly in
  `createSession()` instead of surfacing raw JSON parse errors.
- Restore breathing indicator phase synchronization (`animationstart` anchoring
  to `startTime = 0`) in the workspace page.

To upgrade the workspace and its matching `pyagy` and `python-codex`
dependencies, run the following command with the Python interpreter that runs
the workspace:

```sh
python -m pip install --upgrade pycodex-ws==0.7.4
```

Existing running `pycodex-ws` nodes can also upgrade directly from the browser
using **Settings → Update & reboot** and refreshing the page.
