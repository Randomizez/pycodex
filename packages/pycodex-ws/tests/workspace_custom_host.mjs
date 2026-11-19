// Run manually: node packages/pycodex-ws/tests/workspace_custom_host.mjs
import assert from "node:assert/strict";
import {mountCustomFrontend} from "../workspace_server/custom_frontend.mjs";

const moduleUrl = "data:text/javascript," + encodeURIComponent(
  "export let host; export function mount(value) { host = value; }",
);
const extension = await import(moduleUrl);
let activeSession = "first";
let draft = {parts: [{type: "text", text: "original"}]};
const state = {};
const events = new EventTarget();
const opened = [];
globalThis.document = {querySelector: () => ({src: moduleUrl})};
globalThis.window = {location: {href: "https://workspace.test/w/test/"}};
globalThis.fetch = async url => ({ok: true, json: async () => ({path: new URL(url).pathname})});
await mountCustomFrontend({
  getActiveSessionId: () => activeSession,
  sessionById: id => ["first", "second"].includes(id),
  selectSession: async id => { activeSession = id; },
  frontendEvents: events,
  composer: {read: () => draft, set: value => { draft = value; }},
  stateForSession: () => state, resizePrompt() {},
  contentTabs: {open: async (...args) => { opened.push(args); }},
});
const host = extension.host;
assert.equal(host.apiVersion, 1);
assert.equal(host.getActiveSessionId(), "first");
assert(Object.isFrozen(host));
await host.selectSession("second");
assert.equal(host.getActiveSessionId(), "second");
await assert.rejects(host.selectSession("missing"), /Unknown session/);
const seen = [];
const unsubscribe = host.onSessionChanged(id => seen.push(id));
const changed = new Event("sessionchange");
changed.detail = "second";
events.dispatchEvent(changed);
unsubscribe();
events.dispatchEvent(changed);
assert.deepEqual(seen, ["second"]);
host.setDraft({parts: [{type: "text", text: "edited"}]});
assert.equal(host.getDraft().parts[0].text, "edited");
assert.deepEqual(state.draft, host.getDraft());
assert.throws(() => host.setDraft({}), /parts/);
await host.openView("app.py", "src");
assert.deepEqual(opened, [["app.py", "src", "second"]]);
assert.deepEqual(await host.request("api/sessions"), {path: "/w/test/api/sessions"});
console.log("Custom frontend host contract passed.");
