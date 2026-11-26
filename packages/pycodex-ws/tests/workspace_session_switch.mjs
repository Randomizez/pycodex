import assert from "node:assert/strict";
import {createSessionsUI, formatReplyAge} from "../workspace_server/sessions_ui.mjs";

const NativeAbortController = globalThis.AbortController;
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
globalThis.window = {
  sessionStorage: {getItem: () => null}, location: {href: "http://workspace.test/"},
};
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
let bannerSyncCount = 0;
const clearedBannerSessions = [];
const ui = createSessionsUI({
  log, composer, activeSessionKey, renderTabs, updatePromptPlaceholder, updateContextMeter,
  syncSessionBanners() { bannerSyncCount += 1; },
  clearSessionBanners(id) { clearedBannerSessions.push(id); },
  updateSpinnerNotifyUi, notifySessionDone, setSpinner, renderQueueHint, markVisibleFinalRead,
  renderConversation, addEntry, dismissSessionsDrawer, closeContextBlock() {}, resizePrompt,
});
const {sessionState, stateForSession, updateSessions, renderSnapshot, pollSession,
  restoreSessionScroll, saveActiveSessionUiState, switchSession, closeSession} = ui;
const tabs = [{id: "a", cwd: "/a"}, {id: "b", cwd: "/b", title: "Second session"}];
let serverTabs = tabs.slice();
const deletions = [];
const polls = [];
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
    deletions.push(new URL(path).pathname);
    serverTabs = [tabs[0]];
    return {ok: true, json: async () => ({ok: true, sessions: serverTabs.slice()})};
  }
  polls.push(path);
  assert.equal(new URL(path).searchParams.get("recent"), null);
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
async function checkClosingActiveSession() {
  updateSessions([{id: "a", running: false, last_user_at: 1}, {id: "b", running: true}]);
  assert.equal(ui.activeId, "a");
  assert.equal(document.title, "Workspace");
  updateSessions([{id: "a", running: false}, {id: "b", running: true}]);
  assert.equal(document.title, "Workspace");
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
  assert.equal(ui.activeId, "b");
  assert.deepEqual(composer.read(), {parts: [{type: "text", text: "draft b"}]});
  delayOldPoll = process.argv[2] !== "none";
  const oldPoll = delayOldPoll ? pollSession() : null;
  logVisible = process.argv[3] === "chat";
  await closeSession("b");
  assert.deepEqual(deletions, ["/api/sessions/b"]);
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
  assert.deepEqual(clearedBannerSessions, ["b"]);
  await closeSession("a");
  assert.deepEqual(deletions, ["/api/sessions/b"]);
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
  assert.equal(stateForSession("a").customizeAgent, false);
  const syncBeforeSetup = bannerSyncCount;
  renderSnapshot({
    cwd: "/a", spinner: "", queued_inputs: [],
    turns: [{
      turn_id: "setup",
      messages: [{message_id: "m1", name: "user", text: "<frontend_customization>\ncustom\n</frontend_customization>"}],
      response: "Ready.",
    }],
  });
  assert.equal(stateForSession("a").customizeAgent, true);
  assert.equal(bannerSyncCount, syncBeforeSetup + 1);
  updateSessions([{id: "a", running: true}]);
  assert.equal(document.title, "Workspace");
  updateSessions([]);
  assert.equal(document.title, "Workspace");
}

async function checkSessionOrder() {
  updateSessions([{id: "a"}]);
  const recent = Array.from({length: 25}, (_, index) => ({
    id: String(index), last_user_at: index,
  }));
  updateSessions([...recent, {id: "a"}]);
  assert.deepEqual(ui.sessions.map(session => session.id), [...recent.map(session => session.id), "a"]);
  const now = Date.parse("2026-11-18T12:00:00+08:00");
  assert.equal(formatReplyAge({id: "legacy", turn_count: 1}, now), "");
  assert.equal(formatReplyAge({last_message_at: 0}, now), "No activity");
  assert.equal(formatReplyAge({last_message_at: (now - 20_000) / 1000}, now), "<1m ago");
  assert.equal(formatReplyAge({last_message_at: (now - 3 * 60_000) / 1000}, now), "3m ago");
  assert.equal(formatReplyAge({last_message_at: (now - 72 * 60_000) / 1000}, now), "1h 12m ago");
  assert.equal(formatReplyAge({last_message_at: (now - 60 * 60_000) / 1000}, now), "1h ago");
  assert.equal(formatReplyAge({last_message_at: (now - 27 * 3600_000) / 1000}, now), "1d 3h ago");
  serverTabs = [
    {id: "a", cwd: "/a", last_message_at: (now - 15_000) / 1000},
    {id: "b", cwd: "/b", last_message_at: (now - 3 * 60_000) / 1000},
  ];
  polls.length = 0;
  await pollSession();
  assert.equal(formatReplyAge(ui.sessionById("a"), now), "<1m ago");
  assert.equal(formatReplyAge(ui.sessionById("b"), now), "3m ago");
  await switchSession("b");
  assert.equal(ui.activeId, "b");
  assert.equal(formatReplyAge(ui.sessionById("a"), now), "<1m ago");
  assert.equal(formatReplyAge(ui.sessionById("b"), now), "3m ago");
  assert.equal(polls.length, 2);
  globalThis.AbortController = NativeAbortController;
  await import("../workspace_server/optional_features/session-time-sort/source/verify.mjs");
}

const run = process.argv[2] === "ordering" ? checkSessionOrder : checkClosingActiveSession;
run().catch(error => { console.error(error); process.exitCode = 1; });
