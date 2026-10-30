# Optional frontend features

Browse this catalog in the workspace view area, then tell the customization
Agent which features to apply by name or ID. For example: "Show me the available
features" or "Apply the exec status feature using my current layout."
The Agent reads the selected source and adapts it into your frontend.

The supplied panels stay hidden until they have matching content, and hide again
when that content disappears. Unselected sessions and initial loading occupy no
space. Request failures remain visible; reasoning records need readable text.

| ID | Feature | Summary | Validation |
| --- | --- | --- | --- |
| clock-ticks | [Clock tick dividers](clock-ticks/README.md) | Quiet timestamped dividers for periodic wake-ups, with expandable trigger details. | Chromium history/live updates, preserved source, session switching, keyboard/mobile and cleanup; synthetic preview |
| consecutive-tool-groups | [Consecutive tool groups](consecutive-tool-groups/README.md) | One expandable conversation row for a run of tool calls, with running/failure counts. | Chromium fixtures, live updates, refresh, keyboard/mobile and cleanup; adapter supplied by the frontend |
| exec-status | [Exec status](exec-status/README.md) | Running and exited command sessions above the composer. | Chromium fixtures, lifecycle, desktop/mobile; live processes not tested |
| reasoning-display | [Recorded reasoning](reasoning-display/README.md) | Readable recorded reasoning summaries and text. | Chromium fixtures, opaque filtering, desktop/mobile; live model not tested |
| session-tips | [Session tips](session-tips/README.md) | Browser-local per-conversation notes in a compact corner card. | Chromium fixtures, session/rollout isolation, refresh, storage failures and desktop/mobile; synthetic sessions |
| subagent-status | [Subagent status](subagent-status/README.md) | Descendant agents, parents and completion/error details. | Chromium fixtures, lifecycle, desktop/mobile; live agents not tested |
| tool-calls | [Tool calls](tool-calls/README.md) | Recent function/custom calls with paired inputs and results. | Chromium fixtures, lifecycle, desktop/mobile; live tools not tested |

Each published row links its feature name to `<feature-id>/README.md`.
Only list features whose source and documentation are present in this library;
proposals and open PRs are not available features.

To contribute, tell the Agent: "Publish my local change as a feature PR."
The [contribution guide](CONTRIBUTING.md) defines the source layout, README and
format for pull requests on GitHub.
