import assert from "node:assert/strict";
import {createSessionsUI} from "../workspace_server/sessions_ui.mjs";

// Record cancellation even when the simulated transport delivers a late response.
globalThis.AbortController = class {
  constructor() { this.signal = {aborted: false}; }
  abort() { this.signal.aborted = true; }
};
let logVisible = process.argv[3] === "chat";
let scrollTop = 0;
const log = {
  get textContent() { return this.children.map(node => node.textContent).join(""); },
  set textContent(value) {
    this.children = value ? [{
      textContent: value, parentElement: this,
      getBoundingClientRect: () => ({top: 0, bottom: 1000}),
    }] : [];
  },
  replaceChildren(...nodes) { this.children = nodes; },
  get scrollTop() { return logVisible ? scrollTop : 0; },
  set scrollTop(value) {
    if (logVisible) scrollTop = Math.max(0, Math.min(value, this.scrollHeight - this.clientHeight));
  },
  get scrollHeight() { return logVisible ? (this.textContent ? 1000 : 100) : 0; },
  get clientHeight() { return logVisible ? 100 : 0; },
  children: [], getBoundingClientRect: () => ({top: 0}),
};
const composer = {
  read() { return this.value; },
  set(value) { this.value = value; },
};
const cwdName = {};
const cwdLabel = {dataset: {defaultCwd: "/a"}, querySelector: () => cwdName};
const canvasTab = {};
const sessionRefresh = {hidden: false, dataset: {}, textContent: ""};
const contentTabs = {
  dataset: {boardTitle: "Old tooltip", boardPath: "/board.html"},
  querySelector: () => canvasTab,
};
globalThis.document = {
  title: "Workspace",
  getElementById: id => id === "sessionRefresh" ? sessionRefresh
    : id === "sessionCwd" ? cwdLabel : contentTabs,
};
globalThis.requestAnimationFrame = () => {};
const markVisibleFinalRead = () => {};
const activeSessionKey = "test-workspace";
globalThis.window = {sessionStorage: {getItem: () => null}, location: {href: "http://workspace.test/"}};
const resizePrompt = () => {};
const dismissSessionsDrawer = () => {};
const renderTabs = () => {};
const renderQueueHint = () => {};
const updateContextMeter = () => {};
const updatePromptPlaceholder = () => {};
const updateSpinnerNotifyUi = () => {};
const notifySessionDone = () => {};
let spinnerText = "";
const setSpinner = text => { spinnerText = text; };
const addEntry = (_role, text) => { log.textContent += text; };
const renderConversation = (_sessionId, turns) => { log.textContent = turns.map(turn => turn.response).join(""); };
const ui = createSessionsUI({
  log, composer, activeSessionKey, renderTabs, updatePromptPlaceholder, updateContextMeter,
  updateSpinnerNotifyUi, notifySessionDone, setSpinner, renderQueueHint, markVisibleFinalRead,
  renderConversation, addEntry, dismissSessionsDrawer, closeContextBlock() {}, resizePrompt,
});
const {sessionState, stateForSession, updateSessions, renderSnapshot, pollSession,
  restoreSessionScroll, saveActiveSessionUiState, switchSession, closeSession} = ui;
const tabs = [{id: "a", cwd: "/a"}, {id: "b", cwd: "/b"}];
let serverTabs = tabs.slice();
function snapshot(id) {
  return {cwd: "/" + id, turns: [{prompt: `prompt ${id}`, response: `answer ${id}`}],
          spinner: "", queued_inputs: []};
}
function response(id) {
  const payload = {session_id: id, sessions: serverTabs.slice(), snapshot: snapshot(id)};
  return {ok: true, json: async () => payload};
}
let delayOldPoll = false;
let finishOldPoll;
let delayedSignal;
globalThis.fetch = async (path, options = {}) => {
  if (options.method === "DELETE") {
    serverTabs = [tabs[0]];
    return {ok: true, json: async () => ({ok: true, sessions: serverTabs.slice()})};
  }
  assert.equal(options.signal.aborted, false);
  const id = path.endsWith("=b") ? "b" : "a";
  const reply = response(id);
  if (delayOldPoll && id === "b") {
    // Deliver a late response even if the browser has already aborted it.
    delayedSignal = options.signal;
    return new Promise((resolve, reject) => {
      finishOldPoll = () => {
        if (process.argv[2] === "error") reject(new Error("old tab is gone"));
        else resolve(reply);
      };
    });
  }
  return reply;
};
(async () => {
  updateSessions([{id: "a", running: false}, {id: "b", running: true}]);
  assert.equal(ui.activeId, "a");
  assert.equal(document.title, "● Workspace");
  updateSessions([{id: "a", running: false}, {id: "b", running: true}]);
  assert.equal(document.title, "● Workspace");
  renderSnapshot(snapshot("a"));
  assert.equal(cwdName.textContent, "a");
  assert.equal(cwdLabel.hidden, true);
  renderSnapshot({...snapshot("a"), cwd: "/a/line\nbreak"});
  renderSnapshot({...snapshot("a"), cwd: "/a/$& changed"});
  assert.equal(cwdName.textContent, "$& changed");
  assert.equal(cwdLabel.hidden, false);
  assert.equal(cwdLabel.title, "Current dir: /a/$& changed\nUse /cd <directory> to change directory");
  assert.equal(canvasTab.title, "Workspace: Workspace\nCurrent dir: /a/$& changed\nCanvas file: /board.html");
  if (!logVisible) {
    assert.equal(log.scrollTop, 0);
    assert.equal(stateForSession("a").scrollTop, null);
    renderSnapshot(snapshot("a"));
    logVisible = true;
    restoreSessionScroll();
  }
  assert.equal(log.scrollTop, 900);
  const draftA = {parts: [
    {type: "text", text: "draft a"},
    {type: "context", id: "a-code", label: "sample.py:1", text: "captured code"},
  ], selection: {start: 7, end: 7}};
  composer.set(draftA);
  log.scrollTop = 123;
  saveActiveSessionUiState();
  logVisible = false;
  renderSnapshot({turns: [{prompt: "prompt a", response: "answer a updated"}]});
  assert.equal(stateForSession("a").scrollTop, 123);
  logVisible = true;
  restoreSessionScroll();
  assert.equal(log.scrollTop, 123);
  await switchSession("b");
  assert.equal(log.textContent, "answer b");
  assert.equal(cwdName.textContent, "b");
  assert.equal(cwdLabel.hidden, false);
  assert.equal(document.title, "Workspace");
  composer.set({parts: [{type: "text", text: "draft b"}]});
  log.scrollTop = 9;
  delayOldPoll = process.argv[2] !== "none";
  const oldPoll = delayOldPoll ? pollSession() : null;
  logVisible = process.argv[3] === "chat";
  await closeSession("b");
  if (!logVisible) {
    assert.equal(stateForSession("a").scrollTop, 123);
    logVisible = true;
    restoreSessionScroll();
  }
  assert.equal(ui.activeId, "a");
  assert.equal(log.textContent, "answer a");
  assert.equal(cwdName.textContent, "a");
  assert.equal(cwdLabel.hidden, true);
  assert.deepEqual(composer.read(), draftA);
  assert.equal(log.scrollTop, 123);
  assert.equal(sessionState.has("b"), false);
  assert.equal(document.title, "Workspace");
  if (oldPoll) {
    assert.equal(delayedSignal.aborted, true);
    finishOldPoll();
    await oldPoll;
  }
  assert.equal(ui.activeId, "a");
  assert.deepEqual(ui.sessions.map(tab => tab.id), ["a"]);
  assert.equal(spinnerText, "");
  assert.equal(log.textContent, "answer a");
  await pollSession();
  assert.equal(log.textContent, "answer a");
  updateSessions([{id: "a", running: true}]);
  assert.equal(document.title, "● Workspace");
  updateSessions([]);
  assert.equal(document.title, "Workspace");
})().catch(error => { console.error(error); process.exitCode = 1; });
