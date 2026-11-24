# 0.7.3 release notes

Changes since 0.7.2:

- Share profiles and proxy outbound HTTP/SSE requests between workspace nodes
  connected to the same `pycodex-router`. Adds `WebSocket /tunnel/peer` with
  persistent connection reuse and pre-warmed standby `/tunnel/data` channels on
  the router, plus `/api/peer-profiles*`, `/api/peer-proxy` (supporting both
  unconditional `X-Peer-Target-Url` proxying and profile-backed credential
  injection with isolated `.env` lookup), offline manifest caching, degraded
  session promotion, persistent per-config node identity keys, and an async
  **Node → Profile** chooser dialog. Existing `0.7.2` workspace nodes can
  enable peer profile sharing via in-process **Update & reboot** once the
  router is upgraded.
- Allow `pycodex-router --password` (without a value) to read the browser
  password from stdin or an interactive prompt.
- Refine the workspace UI, session sidebar/tabs, bundled offline Markdown and
  KaTeX rendering, and KaTeX preview containment.
- Preview `.svg` files as images in workspace views (`#L<line>` links open the
  source editor at that line).
- Compress outbound Gemini request bodies with gzip and automatically fall back
  when an upstream endpoint does not accept compressed requests.
- Use author names in cross-agent message examples.

To upgrade the workspace and its matching `pyagy` and `python-codex`
dependencies, run the following command with the Python interpreter that runs
the workspace:

```sh
python -m pip install --upgrade pycodex-ws==0.7.3
```

Upgrade `pycodex-router` on the router host and restart it so `/tunnel/peer`
is active, then use **Settings → Update & reboot** (or restart `pycodex-ws`) on
connected workspace nodes.
