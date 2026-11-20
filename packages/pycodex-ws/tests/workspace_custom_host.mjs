// Run manually: node packages/pycodex-ws/tests/workspace_custom_host.mjs
import assert from "node:assert/strict";
import {mountCustomFrontend} from "../workspace_server/custom_frontend.mjs";
import {createSessionBanners, loadEmbeddedBanners} from "../workspace_server/banners.mjs";

class FakeElement {
  constructor(tag) {
    this.tagName = tag.toUpperCase();
    this.className = "";
    this.textContent = "";
    this.hidden = false;
    this.dataset = {};
    this.attributes = new Map();
    this.listeners = new Map();
    this.children = [];
    this.parentNode = null;
  }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  addEventListener(type, fn) { this.listeners.set(type, fn); }
  click() { this.listeners.get("click")?.(); }
  focus() { this.focused = true; }
  select() { this.selected = true; }
  appendChild(child) {
    child.remove();
    child.parentNode = this;
    this.children.push(child);
    return child;
  }
  append(...nodes) { nodes.forEach(node => this.appendChild(node)); }
  insertBefore(node, ref) {
    node.remove();
    node.parentNode = this;
    const idx = ref ? this.children.indexOf(ref) : -1;
    if (idx < 0) this.children.push(node);
    else this.children.splice(idx, 0, node);
    return node;
  }
  remove() {
    if (!this.parentNode) return;
    const idx = this.parentNode.children.indexOf(this);
    if (idx >= 0) this.parentNode.children.splice(idx, 1);
    this.parentNode = null;
  }
}

const moduleUrl = "data:text/javascript," + encodeURIComponent(
  "export let host; export function mount(value) { host = value; }",
);
const extension = await import(moduleUrl);
let activeSession = "first";
let draft = {parts: [{type: "text", text: "original"}]};
const state = {};
const events = new EventTarget();
const opened = [];
const toasts = [];
let isAdmin = true;
const bannerContainer = new FakeElement("div");
bannerContainer.hidden = true;
const fakeLog = {scrollHeight: 200, scrollTop: 150, clientHeight: 100};
globalThis.document = {
  querySelector: selector => selector === "script[data-pycodex-custom-script]" ? {src: moduleUrl} : null,
  createElement: tag => new FakeElement(tag),
  createElementNS: (_ns, tag) => new FakeElement(tag),
};
globalThis.window = {location: {href: "https://workspace.test/w/test/"}};
globalThis.fetch = async url => ({ok: true, json: async () => ({path: new URL(url).pathname})});
const banners = createSessionBanners({
  container: bannerContainer,
  log: fakeLog,
  getActiveSessionId: () => activeSession,
  isAdmin: () => isAdmin,
  openView: async (...args) => { opened.push(args); },
  showToast: text => { toasts.push(text); },
});
await mountCustomFrontend({
  getActiveSessionId: () => activeSession,
  sessionById: id => ["first", "second"].includes(id),
  selectSession: async id => { activeSession = id; },
  frontendEvents: events,
  composer: {read: () => draft, set: value => { draft = value; }},
  stateForSession: () => state, resizePrompt() {},
  contentTabs: {open: async (...args) => { opened.push(args); }},
  setBanner: (id, banner) => banners.set(id, banner),
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

// Workspace-wide vs session-scoped banners, memoization, adminOnly, and cleanup.
const removeGlobal = host.setBanner("expflow", {
  icon: "info",
  label: "ExpFlow",
  text: "Run exp-042",
  details: [{label: "Config", value: "configs/train.yaml", viewHref: "configs/train.yaml"}],
});
assert.equal(bannerContainer.hidden, false);
assert.equal(bannerContainer.children.length, 1);
const globalNode = bannerContainer.children[0];
assert.equal(globalNode.dataset.bannerId, "expflow");
// Toggle details open and verify identical re-set keeps the same DOM node and expanded state.
const detailsToggle = globalNode.children[0].children[1].children[0];
detailsToggle.click();
assert.equal(globalNode.children[1].hidden, false);
const removeGlobalLatest = host.setBanner("expflow", {
  icon: "info",
  label: "ExpFlow",
  text: "Run exp-042",
  details: [{label: "Config", value: "configs/train.yaml", viewHref: "configs/train.yaml"}],
});
assert.equal(bannerContainer.children[0], globalNode);
assert.equal(globalNode.children[1].hidden, false);

const removeFirstOnly = host.setBanner("first-note", {
  label: "First only",
  sessionId: "first",
  adminOnly: true,
});
assert.equal(bannerContainer.children.map(n => n.dataset.bannerId).join(","), "expflow");
activeSession = "first";
banners.render({collapse: true});
assert.equal(globalNode.children[1].hidden, true, "switching sessions collapses expanded details");
assert.deepEqual(bannerContainer.children.map(n => n.dataset.bannerId), ["expflow", "first-note"]);
isAdmin = false;
banners.render();
assert.deepEqual(bannerContainer.children.map(n => n.dataset.bannerId), ["expflow"]);
isAdmin = true;
banners.render();
assert.deepEqual(bannerContainer.children.map(n => n.dataset.bannerId), ["expflow", "first-note"]);
removeFirstOnly();
assert.deepEqual(bannerContainer.children.map(n => n.dataset.bannerId), ["expflow"]);
removeGlobal();
assert.equal(bannerContainer.hidden, false, "superseded disposer must not remove newer banner");
removeGlobalLatest();
assert.equal(bannerContainer.hidden, true);

// Embedded declarative banners and input validation.
loadEmbeddedBanners({
  querySelectorAll: () => [{dataset: {pycodexBanner: "embedded"}, textContent: JSON.stringify({text: "Embedded info"})}],
}, banners);
assert.deepEqual(bannerContainer.children.map(n => n.dataset.bannerId), ["embedded"]);
host.setBanner("embedded", null);
assert.equal(bannerContainer.hidden, true);
assert.throws(() => host.setBanner("", {label: "Bad"}), /non-empty/);
assert.throws(() => host.setBanner("bad", {label: "Bad", actions: [{label: "Run", href: "javascript:alert(1)"}]}), /protocol/);
console.log("Custom frontend host contract passed.");
