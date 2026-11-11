---
title: "File attachments"
summary: "Preview, remove and retry local files in the composer before sending their contents with a message."
tags: [attachments, composer, files]
---

# File attachments

## Behavior

Select several files, paste a file, or drop files onto the composer. Each file
gets a card showing its path, size and state. Text opens in a plain-text dialog;
PNG, JPEG, GIF and WebP files have local thumbnails and an image preview.
Folder selection preserves relative paths and skips `.git`, `node_modules`,
`.venv` and `__pycache__`. Dropped directories use the folder-picker entry.

For example, selecting a readable source file, an oversized file and a file
that cannot be read produces a ready card and two independent error cards.
The readable file stays in the draft. Read/capacity errors can be retried or
removed, and cancelling leaves previously completed files in place.
Both the send action and Enter wait for unfinished reads; completion does not
send automatically. Failed files are clearly marked as excluded from sending.

This supplies a client attachment flow where the built-in page has no local
file picker. Text or base64 content enters ordinary context tokens in the
current draft and travels with the user's next message.

## Source

- [custom.js](source/custom.js) mounts the feature through the built-in host.
- [uploads.mjs](source/uploads.mjs) owns the picker, cards, preview dialog,
  input events, send guard, draft observation and resource cleanup.
- [upload_core.mjs](source/upload_core.mjs) owns encoding, queueing,
  deduplication, capacity checks, retries and session isolation.
- [custom.css](source/custom.css) contains scoped styles using the workspace
  palette, including a narrow-screen layout.
- [upload_core.test.mjs](source/upload_core.test.mjs) checks queue and draft
  behavior with controlled FileReader completion.
- [upload_browser.py](source/upload_browser.py) checks the real Composer and
  supplied sources in Chromium with local fixtures.

The UI takes inspiration from [Atrium](https://github.com/yuangpeng/atrium)'s
`web/src/chat/Composer.tsx`. The implementation uses the existing draft host
and does not require React, Atrium, an SDK or generated dependency bundles.

## Integration

Requires built-in page host API version 1: `getActiveSessionId`,
`onSessionChanged`, `getDraft` and `setDraft`. The DOM assumptions are
`#composer`, `.composer-surface`, `#prompt` and `#contextBlockDetail`.
Browser prerequisites are ES modules, FileReader, TextDecoder, Web Crypto,
MutationObserver and a native HTML dialog. Folder selection is exposed only
when the browser supports `webkitdirectory`.

1. Copy the two `.mjs` runtime modules into your configured frontend directory.
2. With no existing JavaScript customization, use the supplied `custom.js`.
   With an existing `mount(host)`, import `mountUploads` and call it there;
   retain other initialization and call its returned disposer on teardown.
3. Append the scoped rules from `custom.css` to your existing customization CSS.
4. Refresh, select two small files, inspect a card and remove one, then send
   the remaining file through the ordinary composer.

Undo by removing the mount call and scoped CSS rules, then refreshing. Preserve
other customizations and any draft you want to keep. Supporting modules do not
run without being imported. Removing this UI does not remove context tokens
already stored in the draft; those can still be removed using the composer.

The queue re-reads the latest draft before each commit, preserving typing,
other context and selection. Cards derive from the draft, so native token
deletion or draft replacement updates the shelf. An epoch rejects a picker
opened before switching A → B → A. Session changes, pagehide and disposal abort
pending reads, discard temporary File references and revoke preview URLs.

## Validation

Using Node 22, run from the repository root:

```bash
node packages/pycodex-ws/workspace_server/optional_features/file-attachments/source/upload_core.test.mjs
```

Ten behavior groups pass: UTF-8/BOM UTF-16/binary encoding, legacy tokens,
typing and selection during reads, isolated failure and retry, deduplication,
folder exclusions, cancelled/removed reads, session epochs, encoded message
and serialized draft limits, file count, draft roundtrips and raster filtering.

The manual DOM check needs Python's `httpx` and `websockets` (use the repository's
dev environment) and an existing Chromium executable. It downloads nothing.
Output must be a new canonical directory on a writable JuiceFS mount; the
writer verifies mount identity, capacity, inodes and permissions before creating
the profile, cache, log, HTML fixture and result JSON.

```bash
python packages/pycodex-ws/workspace_server/optional_features/file-attachments/source/upload_browser.py \
  --output "$ATTACHMENT_CHECK_OUTPUT" \
  --mount-source "$ATTACHMENT_CHECK_MOUNT_SOURCE" \
  --chrome "$ATTACHMENT_CHECK_CHROME"
```

Twenty-seven Chromium DOM checks pass. The fixture exercises initial mounting, Enter/button send guards, completed
and failed cards, editing during reads, plain-text escaping, mixed paste,
drop, retry, cancellation, draft restoration, session switches, stale pickers,
image decoding, URL release, native token deletion, one ordinary send,
narrow-screen overflow and teardown. It uses the real Composer and feature
sources. Its draft store and sender are in-memory fixtures; it makes no model
or production API requests. The result JSON lists each observed check.

Source syntax and whitespace checks accompany these behavior checks.
Actual cross-browser behavior, production model interpretation and server
reconnection were not checked: the feature has no server request or WS stream,
and these checks intentionally use local fixtures.

## Preview

No screenshot is included. Expect a compact paperclip/folder toolbar beneath
the editor and a scrollable shelf of file cards. Ready cards show a file icon
or image thumbnail and a remove button; failures show an explanation and a
retry button when recovery is possible. A modal displays text or a raster image.

## Limitations

- This is readable reference source for customization, with Chinese UI labels.
  Adapt labels, DOM selectors and styling when integrating a different page.
- Each file is at most 256 KiB; one message has at most 32 attachments and
  512 KiB of original file bytes. Encoded message text is at most 160,000
  characters, the serialized UTF-8 draft at most 512 KiB, and the draft at most
  500 parts. All limits include existing draft content. The pending/error queue
  holds at most 64 entries. Larger files need splitting or another workflow.
- Binary attachments travel as base64 text. Raster previews stay in the browser;
  they do not create native model image inputs. There is no server upload API,
  server file staging or executable HTML/SVG preview.
- Completed attachments use the host's draft lifecycle. Persistence after a
  page reload depends on the integrating frontend's draft storage; unfinished
  reads and retry File objects are never persisted.
- Files are appended through `setDraft`, so the host determines undo behavior.
  The context editor can alter an attachment's serialized content; malformed
  tokens remain removable but have no parsed preview.
