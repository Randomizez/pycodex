// Run manually: node packages/pycodex-ws/tests/workspace_peer_dialog.mjs
import assert from "node:assert/strict";
import {installDialogs} from "../workspace_server/dialogs.mjs";

class FakeOption {
  constructor(text, value) {
    this.textContent = String(text);
    this.text = String(text);
    this.value = String(value);
  }
}

class FakeElement {
  constructor(id = "") {
    this.id = id;
    this.hidden = false;
    this.disabled = false;
    this.open = false;
    this.textContent = "";
    this.value = "";
    this.checked = false;
    this.dataset = {};
    this.options = [];
    this.listeners = new Map();
    this.attributes = new Map();
  }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  async dispatch(type, event = {}) {
    const handlers = this.listeners.get(type) || [];
    for (const fn of handlers) {
      await fn({
        target: this,
        preventDefault() {},
        ...event,
      });
    }
  }
  replaceChildren(...items) {
    this.options = items;
    this.value = items.length ? items[0].value : "";
  }
  add(option) {
    this.options.push(option);
    if (this.options.length === 1) this.value = option.value;
  }
  showModal() {
    this.open = true;
  }
  close() {
    this.open = false;
    const handlers = this.listeners.get("close") || [];
    for (const fn of handlers) fn({target: this});
  }
  focus() {}
  select() {}
  contains() {
    return false;
  }
  closest() {
    return null;
  }
  querySelector() {
    return new FakeElement("child");
  }
  querySelectorAll() {
    return [];
  }
  setAttribute(k, v) {
    this.attributes.set(k, String(v));
  }
}

globalThis.Option = FakeOption;
globalThis.window = {location: {href: "https://workspace.test/w/demo/"}};

const elements = new Map();
function getEl(id) {
  if (!elements.has(id)) elements.set(id, new FakeElement(id));
  return elements.get(id);
}

const bodyEl = getEl("body");
bodyEl.dataset = {
  accessRole: "admin",
  frontendDirectory: "/tmp/frontend",
  packageDirectory: "/tmp/pkg",
};

globalThis.document = {
  body: bodyEl,
  hidden: false,
  activeElement: null,
  getElementById: id => getEl(id),
  querySelectorAll: () => [],
  addEventListener() {},
};
globalThis.navigator = {languages: ["zh-CN", "en"]};

getEl("sessionNodeField").hidden = true;

const toasts = [];
const createdCalls = [];
let fetchHandler = async () => {
  throw new Error("unconfigured fetch");
};
globalThis.fetch = (url, opts) => fetchHandler(String(url), opts);

installDialogs({
  getSessions: () => [],
  stateForSession: () => ({}),
  createSession: async opts => {
    createdCalls.push(opts);
    return {session_id: "new-1"};
  },
  selectSession: async () => {},
  openView: async () => {},
  showToast: msg => toasts.push(msg),
});

const nodeA = "a".repeat(52);
const nodeC = "c".repeat(52);

// 1. Standalone with no named profiles -> toast, dialog stays closed.
fetchHandler = async url => {
  assert.equal(url, "https://workspace.test/w/demo/api/sessions");
  return {
    ok: true,
    json: async () => ({sessions: [], profiles: [], default_profile: null}),
  };
};
await getEl("customProfile").dispatch("click");
assert.equal(getEl("newSessionDialog").open, false);
assert.deepEqual(toasts, ["No named profiles are configured."]);

// 2. Standalone with named profiles -> Node field stays hidden!
fetchHandler = async () => ({
  ok: true,
  json: async () => ({sessions: [], profiles: ["local-pro"], default_profile: null}),
});
await getEl("customProfile").dispatch("click");
assert.equal(getEl("newSessionDialog").open, true);
assert.equal(getEl("sessionNodeField").hidden, true);
assert.deepEqual(
  getEl("sessionProfile").options.map(o => [o.text, o.value]),
  [
    ["Default", "null"],
    ["local-pro", '"local-pro"'],
  ]
);
getEl("newSessionDialog").close();

// 3. Connected to router over high-latency link -> dialog opens IMMEDIATELY with local
//    profiles while `/api/peer-nodes` loads asynchronously, then populates peer nodes!
let resolveSlowPeerNodes;
let resolveSlowNodeA;
fetchHandler = async url => {
  if (url.endsWith("/api/sessions")) {
    return {
      ok: true,
      json: async () => ({
        sessions: [],
        profiles: [],
        default_profile: null,
        local_profile_error: "duplicate local profile",
        router_connected: true,
      }),
    };
  }
  if (url.endsWith("/api/peer-nodes")) {
    return new Promise(resolve => {
      resolveSlowPeerNodes = resolve;
    });
  }
  if (url.endsWith(`/api/peer-nodes/${nodeA}/profiles`)) {
    return new Promise(resolve => {
      resolveSlowNodeA = resolve;
    });
  }
  if (url.endsWith(`/api/peer-nodes/${nodeC}/profiles`)) {
    return {
      ok: true,
      json: async () => ({profiles: ["fast", "agy"], default_profile: "fast"}),
    };
  }
  throw new Error("unexpected url: " + url);
};

await getEl("customProfile").dispatch("click");
// Modal opened IMMEDIATELY before `/api/peer-nodes` resolved!
assert.equal(getEl("newSessionDialog").open, true);
assert.equal(getEl("sessionNodeField").hidden, false);
assert.deepEqual(
  getEl("sessionNode").options.map(o => [o.text, o.value, Boolean(o.disabled)]),
  [
    ["This device", "", false],
    ["Loading peer devices…", "", true],
  ]
);
// Local profile error disables Create while "This device" is selected:
assert.equal(getEl("createSessionButton").disabled, true);
assert.equal(getEl("newSessionError").textContent, "duplicate local profile");

// Now resolve the async `/api/peer-nodes` response:
resolveSlowPeerNodes({
  ok: true,
  json: async () => ({
    peer_nodes: [
      {id: nodeA, name: "host-a/127.0.0.1:6007", alias: "GPU Box A"},
      {id: nodeC, name: "host-c/127.0.0.1:6007", alias: ""},
    ],
  }),
});
await new Promise(resolve => setTimeout(resolve, 0));
assert.deepEqual(
  getEl("sessionNode").options.map(o => [o.text, o.value]),
  [
    ["This device", ""],
    ["GPU Box A", nodeA],
    ["host-c/127.0.0.1:6007", nodeC],
  ]
);

// Select Node A (slow) then immediately switch to Node C (fast):
getEl("sessionNode").value = nodeA;
const pendingA = getEl("sessionNode").dispatch("change");
getEl("sessionNode").value = nodeC;
await getEl("sessionNode").dispatch("change");

// Late response from Node A arrives after Node C is already active -> ignored!
resolveSlowNodeA({
  ok: true,
  json: async () => ({profiles: ["stale-only"], default_profile: null}),
});
await pendingA;

assert.equal(getEl("createSessionButton").disabled, false);
assert.deepEqual(
  getEl("sessionProfile").options.map(o => [o.text, o.value]),
  [
    ["Default (fast)", JSON.stringify(`peer:${nodeC}`)],
    ["fast", JSON.stringify(`peer:${nodeC}:fast`)],
    ["agy", JSON.stringify(`peer:${nodeC}:agy`)],
  ]
);

// Selecting remote Default submits dynamic `peer:<nodeId>` rather than baking `:fast`!
getEl("sessionProfile").value = JSON.stringify(`peer:${nodeC}`);
await getEl("newSessionForm").dispatch("submit");
assert.deepEqual(createdCalls, [
  {profile: `peer:${nodeC}`},
]);
assert.equal(getEl("newSessionDialog").open, false);

// 4. In-flight submit disables Node & Create (preventing duplicate submit via node switch),
//    and closing + reopening dialog ignores stale submit completion!
let resolvePendingCreate;
installDialogs({
  getSessions: () => [],
  stateForSession: () => ({}),
  createSession: opts => new Promise(resolve => {
    createdCalls.push(opts);
    resolvePendingCreate = resolve;
  }),
  selectSession: async () => {},
  openView: async () => {},
  showToast: msg => toasts.push(msg),
});

// Reset isolated DOM elements for second install test:
elements.clear();
bodyEl.dataset = {
  accessRole: "admin",
  frontendDirectory: "/tmp/frontend",
  packageDirectory: "/tmp/pkg",
};
elements.set("body", bodyEl);
getEl("sessionNodeField").hidden = true;
createdCalls.length = 0;

installDialogs({
  getSessions: () => [],
  stateForSession: () => ({}),
  createSession: opts => new Promise(resolve => {
    createdCalls.push(opts);
    resolvePendingCreate = resolve;
  }),
  selectSession: async () => {},
  openView: async () => {},
  showToast: msg => toasts.push(msg),
});

fetchHandler = async url => {
  if (url.endsWith("/api/sessions")) {
    return {
      ok: true,
      json: async () => ({
        sessions: [],
        profiles: ["local-one"],
        default_profile: null,
        router_connected: true,
      }),
    };
  }
  if (url.endsWith("/api/peer-nodes")) {
    return {
      ok: true,
      json: async () => ({
        peer_nodes: [{id: nodeC, name: "host-c", alias: "Node C"}],
      }),
    };
  }
  if (url.endsWith(`/api/peer-nodes/${nodeC}/profiles`)) {
    return {
      ok: true,
      json: async () => ({profiles: ["fast"], default_profile: "fast"}),
    };
  }
  throw new Error("unexpected url: " + url);
};

await getEl("customProfile").dispatch("click");
await new Promise(resolve => setTimeout(resolve, 0));
getEl("sessionNode").value = nodeC;
await getEl("sessionNode").dispatch("change");
getEl("sessionProfile").value = JSON.stringify(`peer:${nodeC}:fast`);

// Start submitting without awaiting completion yet:
const submitPromise = getEl("newSessionForm").dispatch("submit");
await Promise.resolve();
assert.equal(createdCalls.length, 1);
assert.equal(getEl("sessionNode").disabled, true);
assert.equal(getEl("sessionProfile").disabled, true);
assert.equal(getEl("createSessionButton").disabled, true);

// Attempting to switch nodes or re-submit while in flight must NOT re-enable Create or double-submit!
getEl("sessionNode").value = "";
await getEl("sessionNode").dispatch("change");
assert.equal(getEl("createSessionButton").disabled, true);
await getEl("newSessionForm").dispatch("submit");
assert.equal(createdCalls.length, 1);

// Close and reopen the dialog BEFORE the first createSession resolves:
getEl("newSessionDialog").close();
await getEl("customProfile").dispatch("click");
assert.equal(getEl("newSessionDialog").open, true);

// Now resolve the stale first submission -> must NOT close the newly opened dialog!
resolvePendingCreate({session_id: "late-1"});
await submitPromise;
assert.equal(getEl("newSessionDialog").open, true);
assert.equal(getEl("createSessionButton").disabled, false);

console.log("Peer profile chooser UI checks passed.");
