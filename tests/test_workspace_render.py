import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_pending_input_updates_and_disappears_before_assistant_continues():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the workspace renderer")
    source = (
        Path(__file__).resolve().parents[1] / "workspace_server" / "workspace.html"
    ).read_text(encoding="utf-8")
    script = (
        source[
            source.index("    function messageSignature(") : source.index(
                "    async function pollSession("
            )
        ]
        + source[
            source.index("    function renderSnapshot(") : source.index(
                "    async function submitPrompt("
            )
        ]
    )
    subprocess.run(
        [
            node,
            "-e",
            r"""
const assert = require("assert");
const vm = require("vm");
const script = JSON.parse(require("fs").readFileSync(0, "utf8"));
const entries = [];
const state = {lastRenderedSignature: "", inputRequest: null};
function addEntry(label, text, kind) {
  entries.push({label, text, kind});
}
const context = {
  activeSessionId: "session",
  stateForSession: () => state,
  isNearBottom: () => true,
  restoreScrollTop: null,
  suppressScrollSave: false,
  lastRenderedSignature: "",
  log: {
    scrollTop: 0,
    scrollHeight: 100,
    clientHeight: 80,
    set textContent(value) { entries.length = 0; },
  },
  addEntry,
  addMarkdownEntry: addEntry,
  setSpinner() {},
  updateContextMeter() {},
  scrollToBottom() {},
};
vm.runInNewContext(script, context);
const snapshot = {
  turns: [{prompt: "hello", thinking: "Before asking", kind: "assistant"}],
  input_request: null,
};
context.renderSnapshot(snapshot);
assert.deepStrictEqual(entries.map(entry => entry.kind), ["user", "thinking"]);

for (const request of [
  {request_id: "request", kind: "questions", text: "Choose a path"},
  {request_id: "request", kind: "questions", other: true, text: "Enter your answer"},
  {request_id: "request", kind: "questions", text: "Choose again"},
  {request_id: "permission", kind: "permissions", text: "Approve access"},
]) {
  snapshot.input_request = request;
  context.renderSnapshot(snapshot);
  assert.deepStrictEqual(
    entries.map(entry => entry.kind), ["user", "thinking", "control"],
  );
  assert.strictEqual(entries[2].text, request.text);
}

// Resolution must redraw even before the next assistant delta arrives.
snapshot.input_request = null;
context.renderSnapshot(snapshot);
assert.deepStrictEqual(entries.map(entry => entry.kind), ["user", "thinking"]);
assert.strictEqual(state.inputRequest, null);
snapshot.turns[0].thinking = "Continuing now";
context.renderSnapshot(snapshot);
assert.strictEqual(entries[entries.length - 1].kind, "thinking");
assert.strictEqual(entries[entries.length - 1].text, "Continuing now");

// An attach snapshot may contain only the pending request.
snapshot.turns = [];
snapshot.input_request = {request_id: "attached", text: "Pending on attach"};
context.renderSnapshot(snapshot);
assert.deepStrictEqual(entries.map(entry => entry.kind), ["control"]);
assert.strictEqual(entries[0].text, "Pending on attach");
""",
        ],
        input=json.dumps(script),
        universal_newlines=True,
        check=True,
    )
