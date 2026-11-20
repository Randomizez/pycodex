import assert from "node:assert/strict";
import {
  keepMobileSessionOrder,
  mountSessionTimeSort,
  sessionActivity,
  sortSessionsByActivity,
} from "./session-time-sort.mjs";

assert.equal(sessionActivity({last_user_at: 123.5, last_message_at: 999}), 123.5);
assert.equal(sessionActivity({last_user_at: 0, last_message_at: 500}), 0);
assert.equal(
  sessionActivity({last_final_id: "019ac000-1000-7000-8000-000000000001"}) >
  sessionActivity({last_final_id: "019ac000-0000-7000-8000-000000000001"}),
  true,
);
assert.equal(sessionActivity({last_final_id: "not-uuidv7"}), 0);

const replyCheck = sortSessionsByActivity([
  {id: "legacy", last_user_at: 1, last_message_at: 10},
  {id: "other", last_user_at: 2, last_message_at: 2},
]);
assert.deepEqual(
  replyCheck.visible.map(session => session.id),
  ["other", "legacy"],
  "assistant replies do not influence order",
);

const bulk = Array.from({length: 25}, (_, index) => ({
  id: `s${String(index).padStart(2, "0")}`,
  last_user_at: index,
}));
const limited = sortSessionsByActivity([...bulk, {id: "s-old", last_user_at: 0}], {
  activeId: "s-old",
  limit: 20,
});
assert.deepEqual(limited.visible.slice(0, 19), bulk.slice(6).reverse());
assert.equal(limited.visible[19].id, "s-old");
assert.equal(sortSessionsByActivity(bulk, {limit: null}).visible.length, 25);
assert.deepEqual(
  keepMobileSessionOrder(bulk.slice(0, 3).reverse(), ["s00", "gone", "s02"]),
  bulk.slice(0, 3),
  "phone dots retain their slots when the list is reordered",
);

class FakeElement {
  constructor(tag, dataset = {}) {
    this.tagName = tag;
    this.dataset = {...dataset};
    this.style = {order: ""};
    this.attributes = new Map();
    this.children = [];
    this.parentElement = null;
    this.hidden = false;
    this.textContent = "";
  }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  removeAttribute(name) { this.attributes.delete(name); }
  hasAttribute(name) { return this.attributes.has(name); }
  toggleAttribute(name, force) {
    const shouldSet = force === undefined ? !this.hasAttribute(name) : Boolean(force);
    if (shouldSet) this.setAttribute(name, "");
    else this.removeAttribute(name);
    return shouldSet;
  }
  append(...nodes) {
    for (const node of nodes) {
      node.parentElement = this;
      this.children.push(node);
    }
  }
  insertBefore(node, reference) {
    node.parentElement = this;
    const index = this.children.indexOf(reference);
    if (index === -1) this.children.push(node);
    else this.children.splice(index, 0, node);
  }
  remove() {
    if (!this.parentElement) return;
    const index = this.parentElement.children.indexOf(this);
    if (index !== -1) this.parentElement.children.splice(index, 1);
    this.parentElement = null;
  }
}

const observers = [];
globalThis.MutationObserver = class {
  constructor(callback) {
    this.callback = callback;
    this.targets = [];
    observers.push(this);
  }
  observe(target) { this.targets.push(target); }
  disconnect() { this.targets = []; }
};

const sidebar = new FakeElement("aside");
const tabs = new FakeElement("div");
const mobileDots = new FakeElement("div");
sidebar.append(tabs);
for (const id of ["a", "b", "c", "d"]) {
  tabs.append(new FakeElement("div", {sessionId: id}));
  mobileDots.append(new FakeElement("button", {sessionId: id}));
}

const docListeners = new Map();
const winListeners = new Map();
function addListener(map, type, handler, options) {
  if (!map.has(type)) map.set(type, new Set());
  map.get(type).add(handler);
  options?.signal?.addEventListener("abort", () => map.get(type)?.delete(handler));
}

const root = {
  hidden: false,
  getElementById(id) { return id === "tabs" ? tabs : null; },
  querySelector(selector) { return selector === ".mobile-session-dots" ? mobileDots : null; },
  createElement(tag) { return new FakeElement(tag); },
  addEventListener(type, handler, options) { addListener(docListeners, type, handler, options); },
};
const win = {
  addEventListener(type, handler, options) { addListener(winListeners, type, handler, options); },
};

let activeId = "a";
let failNext = false;
const sessionListeners = new Set();
let serverSessions = [
  {id: "a", last_user_at: 10, last_message_at: 100},
  {id: "b", last_user_at: 40, last_message_at: 40},
  {id: "c", last_user_at: 30, last_message_at: 90},
  {id: "d", last_user_at: 20, last_message_at: 80},
];

const host = {
  getActiveSessionId: () => activeId,
  onSessionChanged(handler) {
    sessionListeners.add(handler);
    return () => sessionListeners.delete(handler);
  },
  async request(path) {
    assert.equal(path, "api/sessions");
    if (failNext) throw new Error("offline");
    return {sessions: serverSessions};
  },
};

const cleanup = mountSessionTimeSort(host, {limit: 3, pollMs: 60_000, root, win});
await new Promise(resolve => setImmediate(resolve));

const tabById = id => tabs.children.find(node => node.dataset.sessionId === id);
const dotById = id => mobileDots.children.find(node => node.dataset.sessionId === id);
assert.equal(tabById("b").style.order, "0");
assert.equal(tabById("c").style.order, "1");
assert.equal(tabById("a").style.order, "3");
// "a" is outside top 3 (b, c, d), so "d" is hidden while active "a" stays visible.
assert.equal(tabById("a").hasAttribute("data-session-time-hidden"), false);
assert.equal(tabById("d").hasAttribute("data-session-time-hidden"), true);
// Mobile dots keep their original slot order while respecting the visibility limit.
for (const id of ["a", "b", "c", "d"]) {
  assert.equal(dotById(id).style.order, "");
}
assert.equal(dotById("a").hasAttribute("data-session-time-hidden"), false);
assert.equal(dotById("d").hasAttribute("data-session-time-hidden"), true);

activeId = "b";
failNext = true;
sessionListeners.forEach(handler => handler("b"));
await new Promise(resolve => setImmediate(resolve));
assert.equal(tabById("a").hasAttribute("data-session-time-hidden"), true);
assert.equal(tabById("d").hasAttribute("data-session-time-hidden"), false);
const errorNode = sidebar.children.find(node => node !== tabs);
assert.equal(errorNode.hidden, false);
assert.match(errorNode.textContent, /offline/);

failNext = false;
const created = new FakeElement("div", {sessionId: "e"});
tabs.append(created);
observers.forEach(observer => {
  if (observer.targets.includes(tabs)) observer.callback([]);
});
assert.equal(created.hasAttribute("data-session-time-hidden"), true);

cleanup();
for (const node of [...tabs.children, ...mobileDots.children]) {
  assert.equal(node.style.order, "");
  assert.equal(node.hasAttribute("data-session-time-hidden"), false);
}
assert.equal(sidebar.children.length, 1);
assert.equal(sessionListeners.size, 0);
