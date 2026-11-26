# 0.7.5 release notes

Changes since 0.7.4:

- Require one-time device verification in `pycodex-router` (with in-place
  **Verify and unfold** / **Verify** actions), simplify the connect command,
  and refine the router directory UI.
- Store default workspace state files under `~/.pycodex/workspaces/pcws-<id>`
  instead of `/tmp` so session tabs, views, and Canvas persist across reboots.
- Improve browser completion notifications (`notify-on-done`):
  - Add a dedicated right-aligned bell toggle (`#notifyToggle`) in the status
    bar and a bell badge on armed running session tabs, and remove the `● `
    dot from `document.title`.
  - Clean Markdown from notification bodies, include the workspace title,
    surface failed turn errors and pending input requests (clearing spinner on
    `InputRequestedEvent` to match CLI), and focus/switch to the session on
    notification click.
  - Serve a minimal `sw.js` and `manifest.webmanifest` (`display: "standalone"`)
    with `ServiceWorkerRegistration.showNotification()` fallback for Android
    Edge/Chrome and iOS Safari Home Screen PWAs.
- Emit deterministic `mtime=0` gzip request bodies in `responses_server` for
  Gemini requests, eliminating cross-second retry flakiness in CI.
- Simplify the English and Chinese READMEs for one-command `pycodex-ws`
  startup, automatic `~/.codex` inheritance, and feature highlights.

To upgrade the workspace and its matching `pyagy` and `python-codex`
dependencies, run the following command with the Python interpreter that runs
the workspace:

```sh
python -m pip install --upgrade pycodex-ws==0.7.5
```

Existing running `pycodex-ws` nodes can also upgrade directly from the browser
using **Settings → Update & reboot** and refreshing the page.
