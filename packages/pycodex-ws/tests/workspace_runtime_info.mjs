// Run manually: node packages/pycodex-ws/tests/workspace_runtime_info.mjs
import assert from "node:assert/strict";
import {fillRuntimeInfo} from "../workspace_server/dialogs.mjs";

globalThis.window = {location: {href: "https://workspace.test/"}};
let activeSessionId = "a";
const states = new Map([
  ["a", {}],
  ["b", {}],
  ["", {}],
]);
const stateForSession = id => states.get(id);
const notices = [];
const showToast = text => notices.push(text);
const closeContextBlock = () => {};
let draft = "unfinished draft";
const composer = {
  parts: () => [{type: "text", text: draft}],
  replace(start, end, parts) {
    assert.equal(start, draft.length);
    assert.equal(end, start);
    draft += parts.map(part => part.text).join("");
  },
};
const requests = [];
let resolve;
globalThis.fetch = async (url, options) => {
  requests.push({url, options});
  return new Promise(done => {resolve = done;});
};
const fill = () => fillRuntimeInfo({getActiveSessionId: () => activeSessionId, stateForSession, closeContextBlock, composer, showToast});
const pending = fill();
assert.equal(states.get("a").runtimeInfoLoading, true);
await fill();
assert.equal(requests.length, 1);
assert.deepEqual(requests[0], {
  url: "https://workspace.test/api/session?session_id=a&runtime_info=true", options: {cache: "no-store"},
});
draft += " typed during request";
resolve({ok: true, json: async () => ({runtime_info: "runtime information"})});
await pending;
assert.equal(draft, "unfinished draft typed during request\n\nruntime information");
assert.equal(states.get("a").runtimeInfoLoading, false);
assert.equal(notices.length, 1);

const old = fill();
activeSessionId = "b";
draft = "other session";
resolve({ok: true, json: async () => ({runtime_info: "old session information"})});
await old;
assert.equal(draft, "other session");
assert.equal(notices.length, 1);
assert.equal(states.get("a").runtimeInfoLoading, false);
// Errors leave the draft intact and allow another double-click.
const failed = fill();
resolve({ok: false, status: 400, json: async () => ({detail: "session closing"})});
await failed;
assert.match(notices[1], /session closing/);
assert.equal(draft, "other session");
assert.equal(states.get("b").runtimeInfoLoading, false);
activeSessionId = "";
await fill();
assert.equal(requests.length, 3);
assert.ok(requests.every(request => !request.options.method));
console.log("Runtime information UI checks passed.");
