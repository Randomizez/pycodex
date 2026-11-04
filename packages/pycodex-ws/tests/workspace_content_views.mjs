import assert from "assert";
import {ContentTabs, SnapshotDisplay, isWorkspaceLink} from "../workspace_server/frontend/content_views.mjs";
import {editorContext} from "../workspace_server/frontend/editor_context.mjs";

// The browser supplies this API; provide it explicitly in the Node harness.
global.AbortController = class {
  constructor() { this.signal = {aborted: false}; }
  abort() { this.signal.aborted = true; }
};

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}

function fixture(selected = null, enableBoard = true) {
  let serverViews = [];
  const pending = new Map();
  const signals = new Map();
  const changes = [];
  const requests = [];
  const bodies = [];
  const selections = [];
  const tabs = new ContentTabs({
    selected,
    enableBoard,
    onSelect: id => selections.push(id),
    onChange: state => changes.push({
      id: state.activeId, snapshot: state.snapshot, selection: state.selection,
      loading: state.loading, error: state.error, displayMode: state.displayMode,
    }),
    request: async (path, options = {}) => {
      const method = options.method || "GET";
      const key = method + " " + path;
      requests.push(key);
      signals.set(key, options.signal);
      if (options.body) bodies.push(JSON.parse(options.body));
      if (pending.has(key)) return pending.get(key).promise;
      if (path === "api/views") return {views: serverViews.slice()};
      if (path === "api/views/open") {
        const id = JSON.parse(options.body).href;
        const view = serverViews.find(view => view.id === id);
        return {views: serverViews.slice(),
          view: {...view, documents: [{text: id + ":" + view.revision}]}};
      }
      const id = path.split("/").pop();
      if (method === "DELETE") {
        serverViews = serverViews.filter(view => view.id !== id);
        return {ok: true};
      }
      const view = serverViews.find(view => view.id === id);
      return {id, revision: view.revision, documents: [{text: id + ":" + view.revision}]};
    },
  });
  return {
    tabs, pending, signals, changes, requests, bodies, selections,
    setViews: ids => {
      serverViews = ids.map(id => ({id, title: "same.py", revision: 1, reveal_revision: 0}));
    },
    setRevision: (id, revision) => {
      serverViews = serverViews.map(view => view.id === id ? {...view, revision} : view);
    },
    reveal: (id, revision, selection = null) => {
      serverViews = serverViews.map(view => view.id === id ?
        {...view, reveal_revision: revision, selection} : view);
    },
  };
}

async function flush() {
  await Promise.resolve();
  await Promise.resolve();
}

async function newTabsAndRestore() {
  const state = fixture("board");
  state.setViews(["a", "b"]);
  await state.tabs.poll();
  assert.equal(state.tabs.activeId, "board");
  state.setViews(["a", "b", "c"]);
  await state.tabs.poll();
  await flush();
  assert.equal(state.tabs.activeId, "board");
  await state.tabs.open("c");
  assert.equal(state.tabs.activeId, "c");
  assert.equal(state.tabs.snapshot.id, "c");
  assert.deepEqual(state.tabs.views.map(view => view.id), ["a", "b", "c"]);
  await state.tabs.select("a");
  await state.tabs.select("c");
  assert.equal(state.requests.filter(key => key === "GET api/views/c").length, 0);
  assert.deepEqual(state.bodies, [{href: "c"}]);
  await state.tabs.close("c");
  await flush();
  assert.equal(state.tabs.activeId, "b");
  await state.tabs.poll();
  assert.deepEqual(state.tabs.views.map(view => view.id), ["a", "b"]);
  await state.tabs.close("board");
  assert.equal(state.requests.includes("DELETE api/views/board"), false);

  const restored = fixture("a");
  restored.setViews(["a", "b"]);
  assert.equal(restored.tabs.activeId, "board");
  await restored.tabs.poll();
  await flush();
  assert.equal(restored.tabs.activeId, "a");
  assert.equal(restored.tabs.snapshot.id, "a");
  assert.deepEqual(restored.selections, []);
  await restored.tabs.open("b", "/docs/a.md", "session-a");
  await restored.tabs.open("b", "/docs/a.md", "session-b");
  assert.deepEqual(restored.bodies, [
    {href: "b", src_path: "/docs/a.md", session_id: "session-a"},
    {href: "b", src_path: "/docs/a.md", session_id: "session-b"},
  ]);
}

async function disabledCanvasKeepsFileViewsAndAnEmptyState() {
  for (const selected of [null, "board", "missing", "b"]) {
    const state = fixture(selected, false);
    assert.equal(state.tabs.activeId, null);
    state.setViews(["a", "b"]);
    await state.tabs.poll();
    await flush();
    const expected = selected === "b" ? "b" : "a";
    assert.equal(state.tabs.activeId, expected);
    assert.equal(state.tabs.snapshot.id, expected);
    assert.deepEqual(state.selections, []);
    await state.tabs.select(expected);
    assert.deepEqual(state.selections, [expected]);
    await state.tabs.select("board");
    assert.equal(state.tabs.activeId, expected);
    await assert.rejects(state.tabs.open("workspace://canvas"), /Canvas is disabled/);
    assert.equal(state.tabs.activeId, expected);
    await state.tabs.close(expected);
    assert.equal(state.tabs.activeId, expected === "a" ? "b" : "a");
    await state.tabs.close(state.tabs.activeId);
    assert.equal(state.tabs.activeId, null);
    assert.equal(state.tabs.snapshot, null);
    assert.equal(state.tabs.loading, false);
    assert.equal(state.selections[state.selections.length - 1], null);
    await state.tabs.poll();
    state.setViews(["other-client"]);
    await state.tabs.poll();
    assert.equal(state.tabs.activeId, null);
    assert.equal(state.requests.includes("GET api/views/null"), false);
    assert.equal(state.requests.includes("GET api/views/board"), false);
  }
  const empty = fixture(null, false);
  await empty.tabs.poll();
  await empty.tabs.poll();
  assert.equal(empty.tabs.activeId, null);
  assert.equal(empty.tabs.error, "");
  assert.deepEqual(empty.requests, ["GET api/views", "GET api/views"]);
}

async function pollingNeverFollowsOtherClients() {
  const state = fixture("board");
  state.setViews(["a", "b"]);
  state.reveal("a", 1);
  state.reveal("b", 2);
  await state.tabs.poll();
  assert.equal(state.tabs.activeId, "board");
  await state.tabs.select("a");
  await state.tabs.select("b");
  state.reveal("a", 3);
  await state.tabs.poll();
  await flush();
  assert.equal(state.tabs.activeId, "b");
  assert.deepEqual(state.tabs.views.map(view => view.id), ["a", "b"]);
  assert.equal(state.requests.filter(key => key === "GET api/views/a").length, 1);
  const snapshot = state.tabs.snapshot;
  const selections = state.selections.length;
  state.reveal("a", 4);
  await state.tabs.poll();
  assert.equal(state.selections.length, selections);
  assert.equal(state.tabs.snapshot, snapshot);
  await state.tabs.select("b");
  await state.tabs.poll();
  assert.equal(state.tabs.activeId, "b");

  const fresh = fixture();
  fresh.setViews(["a", "b"]);
  fresh.reveal("a", 4);
  fresh.reveal("b", 2);
  await fresh.tabs.poll();
  assert.equal(fresh.tabs.activeId, "board");
}

async function revealingDuringRefreshKeepsVisibleContent() {
  const state = fixture("a");
  state.setViews(["a"]);
  await state.tabs.poll();
  await flush();
  const snapshot = state.tabs.snapshot;
  const refreshed = deferred();
  state.pending.set("GET api/views/a", refreshed);
  state.setRevision("a", 2);
  await state.tabs.poll();
  state.reveal("a", 1);
  await state.tabs.poll();
  assert.equal(state.tabs.snapshot, snapshot);
  assert.equal(state.tabs.loading, false);
  refreshed.resolve({id: "a", revision: 2, documents: [{text: "updated"}]});
  await flush();
  assert.equal(state.tabs.snapshot.revision, 2);
}

async function clickedLinesDoNotReplayOnPollOrManualSelection() {
  const state = fixture();
  state.setViews(["a"]);
  state.reveal("a", 1, {document: 0, line: 12});
  await state.tabs.poll();
  assert.equal(state.tabs.selection, null);
  await state.tabs.open("a");
  assert.equal(state.tabs.displayMode, "source");
  assert.deepEqual(state.tabs.selection, {
    document: 0, line: 12, revision: 1, contentRevision: 1,
  });
  const firstSelection = state.tabs.selection;
  const snapshot = state.tabs.snapshot;
  const selections = state.selections.length;
  await state.tabs.poll();
  assert.equal(state.tabs.selection, firstSelection);
  assert.equal(state.selections.length, selections);
  state.reveal("a", 2, {document: 0, line: 42});
  await state.tabs.poll();
  assert.equal(state.tabs.selection, firstSelection);
  await state.tabs.open("a");
  assert.equal(state.tabs.selection.line, 42);
  assert.notEqual(state.tabs.selection, firstSelection);
  assert.deepEqual(state.tabs.snapshot.documents, snapshot.documents);
  assert.equal(state.requests.filter(key => key === "GET api/views/a").length, 0);
  await state.tabs.select("board");
  await state.tabs.select("a");
  await state.tabs.poll();
  assert.equal(state.tabs.selection, null);
  assert.equal(state.tabs.displayMode, null);
  state.reveal("a", 3, {document: 0, line: 42});
  state.setRevision("a", 2);
  await state.tabs.open("a");
  assert.equal(state.tabs.selection.contentRevision, 2);
  assert.equal(state.tabs.snapshot.revision, 2);
  state.reveal("a", 4);
  await state.tabs.open("a");
  assert.equal(state.tabs.selection, null);
  assert.equal(state.tabs.displayMode, "preview");
}

async function modeToggleClearsLineTargetsAndOverridesPendingOpens() {
  const state = fixture();
  state.setViews(["a"]);
  state.reveal("a", 1, {document: 0, line: 12});
  await state.tabs.poll();
  await state.tabs.open("a");
  const snapshot = state.tabs.snapshot;
  const requests = state.requests.length;
  const selections = state.selections.length;
  state.tabs.setDisplayMode("preview");
  assert.equal(state.tabs.displayMode, "preview");
  assert.equal(state.tabs.selection, null);
  assert.equal(state.tabs.snapshot, snapshot);
  assert.equal(state.requests.length, requests);
  assert.equal(state.selections.length, selections);
  await state.tabs.poll();
  assert.equal(state.tabs.displayMode, "preview");
  state.tabs.setDisplayMode("source");
  assert.equal(state.tabs.selection, null);

  const old = deferred();
  state.pending.set("POST api/views/open", old);
  const opening = state.tabs.open("a");
  const signal = state.signals.get("POST api/views/open");
  state.tabs.setDisplayMode("preview");
  assert.equal(signal.aborted, true);
  old.resolve({view: {id: "a", revision: 99}, views: [{id: "a"}]});
  await opening;
  assert.equal(state.tabs.snapshot, snapshot);
  assert.equal(state.tabs.displayMode, "preview");
}

async function oldOpensCannotOverrideTheLatestIntent(action, fail) {
  const state = fixture("board");
  state.setViews(["a", "b"]);
  await state.tabs.poll();
  const old = deferred();
  state.pending.set("POST api/views/open", old);
  const opening = state.tabs.open("a");
  const signal = state.signals.get("POST api/views/open");
  state.pending.delete("POST api/views/open");
  if (action === "open") await state.tabs.open("b");
  if (action === "select") await state.tabs.select("b");
  if (action === "canvas") await state.tabs.open("workspace://canvas");
  if (action === "close") await state.tabs.close("a");
  if (action === "stop") state.tabs.stop();
  assert.equal(signal.aborted, true);
  if (fail) old.reject(new Error("late failure"));
  else old.resolve({view: {id: "a", revision: 99, documents: []}, views: [{id: "a"}]});
  await opening;
  assert.equal(state.tabs.activeId, ["open", "select"].includes(action) ? "b" : "board");
  assert.equal(state.tabs.error, "");
  assert.equal(state.tabs.openRequest, null);
  assert.deepEqual(state.tabs.views.map(view => view.id), action === "close" ? ["b"] : ["a", "b"]);
}

async function failedOpensPreserveTheActiveView() {
  const state = fixture("a");
  state.setViews(["a"]);
  await state.tabs.poll();
  await flush();
  const snapshot = state.tabs.snapshot;
  const failure = deferred();
  state.pending.set("POST api/views/open", failure);
  const opening = state.tabs.open("missing");
  failure.reject(new Error("File not found"));
  await assert.rejects(opening, /File not found/);
  assert.equal(state.tabs.activeId, "a");
  assert.equal(state.tabs.snapshot, snapshot);
  assert.ok(state.tabs.error.includes("File not found"));
  assert.equal(state.tabs.openRequest, null);
  state.pending.delete("POST api/views/open");
  await state.tabs.open("a");
  assert.equal(state.tabs.error, "");
}

async function oldPollCannotUndoAClick() {
  const state = fixture("board");
  state.setViews(["a"]);
  await state.tabs.poll();
  const old = deferred();
  state.pending.set("GET api/views", old);
  const polling = state.tabs.poll();
  state.setViews(["a", "b"]);
  await state.tabs.open("b");
  old.resolve({views: [{id: "a", revision: 1}]});
  await polling;
  assert.equal(state.tabs.activeId, "b");
  assert.deepEqual(state.tabs.views.map(view => view.id), ["a", "b"]);
  assert.equal(state.tabs.snapshot.id, "b");
}

function fileAndUriLinksPreserveBrowserDestinations() {
  for (const href of [
    "README.md", "README:1", "./src/app.py", "../app.py#L42", "/data/app.py:42",
    "C:/project/app.py:42", "C:%5Cproject%5Capp.py:42", "空 格.py", "space%20%23%25.py",
    "git:HEAD:app.py#L42", "diff:A.py..B.py#R42",
    "diff:HEAD:app.py", "diff:HEAD~1..HEAD:app.py#L42", "workspace://canvas",
  ]) assert.equal(isWorkspaceLink(href), true, href);
  for (const href of [
    "", null, "#anchor", "#L42", "//example.test/app.py", "https://example.test/app.py:42",
    "mailto:42", "tel:123", "app://connector", "vscode://file/path",
    "javascript:42", " JAVASCRIPT:42", "\u00a0javascript:42", "data:text/html,test",
    "java\nscript:42", "file:///data/app.py", "blob:https://example.test/id", "?query=1",
    "app.md#section",
  ]) assert.equal(isWorkspaceLink(href), false, href);
  assert.equal(isWorkspaceLink("app.py", false), false);
  assert.equal(isWorkspaceLink("diff:HEAD:app.py", false), true);
  assert.equal(isWorkspaceLink("git:HEAD:app.py", false), true);
}

async function oldContentCannotReplaceSelectedTab(fail) {
  const state = fixture("board");
  state.setViews(["a", "b"]);
  await state.tabs.poll();
  const old = deferred();
  state.pending.set("GET api/views/a", old);
  const loading = state.tabs.select("a");
  await state.tabs.select("b");
  assert.equal(state.signals.get("GET api/views/a").aborted, true);
  if (fail) old.reject(new Error("late failure"));
  else old.resolve({id: "a", documents: []});
  await loading;
  assert.equal(state.tabs.activeId, "b");
  assert.equal(state.tabs.snapshot.id, "b");
  assert.equal(state.tabs.error, "");
}

async function oldPollCannotReopenClosedTabs() {
  const state = fixture("board");
  state.setViews(["a", "b"]);
  await state.tabs.poll();
  await state.tabs.select("b");
  const old = deferred();
  state.pending.set("GET api/views", old);
  const polling = state.tabs.poll();
  await state.tabs.close("b");
  old.resolve({views: [{id: "a"}, {id: "b", reveal_revision: 99}]});
  await polling;
  assert.deepEqual(state.tabs.views.map(view => view.id), ["a"]);
  assert.equal(state.tabs.activeId, "a");
  state.pending.delete("GET api/views");
  state.reveal("a", 3);
  await state.tabs.select("board");
  await state.tabs.poll();
  assert.equal(state.tabs.activeId, "board");
}

async function closingRespectsSelectionAndFailure() {
  const state = fixture("board");
  state.setViews(["a", "b"]);
  await state.tabs.poll();
  await state.tabs.select("b");
  const pending = deferred();
  state.pending.set("DELETE api/views/b", pending);
  const closing = state.tabs.close("b");
  await state.tabs.select("board");
  pending.resolve({ok: true});
  await closing;
  assert.equal(state.tabs.activeId, "board");
  assert.deepEqual(state.tabs.views.map(view => view.id), ["a"]);

  const failure = deferred();
  state.pending.set("DELETE api/views/a", failure);
  const failedClose = state.tabs.close("a");
  failure.reject(new Error("disk is full"));
  await failedClose;
  assert.deepEqual(state.tabs.views.map(view => view.id), ["a"]);
  assert.ok(state.tabs.error.includes("disk is full"));
  assert.equal(state.tabs.closing.size, 0);
}

async function stoppingDoesNotLeaveLoadingStuck() {
  const state = fixture("board");
  state.setViews(["a"]);
  await state.tabs.poll();
  const old = deferred();
  state.pending.set("GET api/views/a", old);
  const loading = state.tabs.select("a");
  state.tabs.stop();
  state.pending.delete("GET api/views/a");
  await state.tabs.poll();
  await flush();
  assert.equal(state.tabs.snapshot.id, "a");
  assert.equal(state.tabs.loading, false);
  old.resolve({id: "stale", documents: []});
  await loading;
  assert.equal(state.tabs.snapshot.id, "a");
}

async function liveUpdatesKeepVisibleContentAndInvalidateOtherTabs() {
  const state = fixture("board");
  state.setViews(["a", "b"]);
  await state.tabs.poll();
  await state.tabs.select("b");
  await state.tabs.select("a");
  const selections = state.selections.length;
  const older = deferred();
  state.pending.set("GET api/views/a", older);
  state.setRevision("a", 2);
  state.setRevision("b", 2);
  await state.tabs.poll();
  assert.equal(state.tabs.activeId, "a");
  assert.equal(state.tabs.snapshot.revision, 1);
  assert.equal(state.tabs.loading, false);
  assert.equal(state.selections.length, selections);
  assert.equal(state.tabs.cache.has("b"), false);

  state.setRevision("a", 3);
  await state.tabs.poll();
  older.resolve({id: "a", revision: 2, documents: [{text: "outdated"}]});
  await flush();
  assert.equal(state.tabs.snapshot.revision, 1);
  state.pending.delete("GET api/views/a");
  await state.tabs.poll();
  await flush();
  assert.equal(state.tabs.snapshot.revision, 3);
  assert.equal(state.selections.length, selections);
  await state.tabs.select("b");
  assert.equal(state.tabs.snapshot.revision, 2);
  assert.equal(state.requests.filter(key => key === "GET api/views/b").length, 2);
}

async function refreshFailuresKeepContentAndRetry() {
  const state = fixture("a");
  state.setViews(["a", "b"]);
  await state.tabs.poll();
  await flush();
  const failure = deferred();
  state.pending.set("GET api/views/a", failure);
  state.setRevision("a", 2);
  await state.tabs.poll();
  failure.reject(new Error("connection lost"));
  await flush();
  assert.equal(state.tabs.snapshot.revision, 1);
  assert.equal(state.tabs.loading, false);
  assert.ok(state.tabs.error.includes("connection lost"));
  state.pending.delete("GET api/views/a");
  await state.tabs.poll();
  await flush();
  assert.equal(state.tabs.snapshot.revision, 2);
  assert.equal(state.tabs.error, "");

  const old = deferred();
  state.pending.set("GET api/views/a", old);
  state.setRevision("a", 3);
  await state.tabs.poll();
  await state.tabs.select("b");
  old.resolve({id: "a", revision: 3, documents: []});
  await flush();
  assert.equal(state.tabs.snapshot.id, "b");
}

class Node {
  constructor(document, tag) {
    this.ownerDocument = document;
    this.tagName = tag;
    this.children = [];
    this.className = "";
    this.text = "";
    this.attributes = new Map();
    this.scrollLeft = 0;
    this.scrollTop = 0;
    this.classList = {
      add: name => { this.className = (this.className + " " + name).trim(); },
      remove: name => { this.className = this.className.split(" ").filter(item => item !== name).join(" "); },
    };
  }

  appendChild(node) {
    if (node.parent) node.parent.children = node.parent.children.filter(child => child !== node);
    node.parent = this;
    this.children.push(node);
    return node;
  }

  setAttribute(name, value) { this.attributes.set(name, value); }
  getAttribute(name) { return this.attributes.has(name) ? this.attributes.get(name) : null; }

  set textContent(text) {
    this.children.forEach(node => { node.parent = null; });
    this.children = [];
    this.text = text;
  }

  get textContent() {
    return this.text + this.children.map(node => node.textContent).join("");
  }

  querySelector(selector) {
    for (const child of this.children) {
      if (child.className.split(" ").includes(selector.slice(1))) return child;
      const nested = child.querySelector(selector);
      if (nested) return nested;
    }
    return null;
  }
}

function displayFixture(loadEditor) {
  const document = {createElement: tag => new Node(document, tag)};
  const content = document.createElement("div");
  const htmlPreviews = [];
  const display = new SnapshotDisplay({
    content, loadEditor,
    renderMarkdown: (parent, text) => { parent.textContent = text; },
    createHtmlPreview: (parent, position = [0, 0]) => {
      const preview = {
        position, destroyed: false,
        update(source) { this.source = source; },
        capture() { return this.position; },
        destroy() { this.destroyed = true; },
      };
      htmlPreviews.push(preview);
      return preview;
    },
  });
  const snapshot = (id, label = id + ".py") => ({
    kind: "file", revision: 1, documents: [{label, text: id}],
  });
  return {display, content, snapshot, htmlPreviews};
}

async function htmlPreviewKeepsNativeResourcesAndReadingState() {
  const pages = [];
  const messages = [];
  const listeners = new Set();
  const document = {createElement: tag => {
    const node = new Node(document, tag);
    if (tag === "iframe") node.contentWindow = {postMessage: message => messages.push(message)};
    return node;
  }};
  document.defaultView = {
    addEventListener: (type, listener) => { listeners.add(listener); },
    removeEventListener: (type, listener) => { listeners.delete(listener); },
    DOMParser: class {
      parseFromString(source) {
        const page = {
          doctype: null, createElement: document.createElement,
          head: {prepend: script => { page.script = script; }},
          documentElement: {outerHTML: source},
        };
        pages.push(page);
        return page;
      }
    },
  };
  const content = document.createElement("div");
  const display = new SnapshotDisplay({
    content, renderMarkdown: (parent, text) => { parent.textContent = text; },
  });
  const text = '<base href="media/"><img src="plot.png" srcset="plot-2x.png 2x">';
  const snapshot = {kind: "file", revision: 1, documents: [{label: "a.html", text}]};
  await display.show("a", snapshot);
  const preview = display.htmlPreview;
  assert.equal(preview.frame.getAttribute("sandbox"), null);
  assert.equal(preview.frame.srcdoc, text);
  await display.show("a", snapshot);
  assert.equal(pages.length, 1);

  const send = (source, generation, type, position) => {
    for (const listener of listeners) listener({source, data: {generation, type, position}});
  };
  send({}, 1, "workspace-preview-scroll", [1, 2]);
  assert.deepEqual(preview.capture(), [0, 0]);
  send(preview.frame.contentWindow, 1, "workspace-preview-scroll", [10, 320]);
  assert.deepEqual(preview.capture(), [10, 320]);
  await display.show("a", {...snapshot, documents: [{label: "a.html", text: "<h1>Updated</h1>"}]});
  send(preview.frame.contentWindow, 1, "workspace-preview-scroll", [0, 0]);
  send(preview.frame.contentWindow, 2, "workspace-preview-ready");
  assert.deepEqual(messages, [{
    type: "workspace-preview-restore", generation: 2, position: [10, 320],
  }]);
  display.clear();
  assert.equal(listeners.size, 0);
}

function selectionFixture() {
  const created = [];
  const state = displayFixture(async () => ({
    createSnapshotEditor: async (parent, snapshot, saved) => {
      const editor = {
        snapshot, saved, reveals: [], destroyed: false,
        update() { return this.updating; },
        async revealLine(document, line, isCurrent) {
          await this.comparing;
          if (!isCurrent()) return false;
          this.reading = {document, line};
          this.reveals.push(this.reading);
          return true;
        },
        capture() { return this.reading; },
        destroy() { this.destroyed = true; },
      };
      created.push(editor);
      return editor;
    },
  }));
  return {...state, created};
}

async function lineSelectionReusesEditorsWithoutReplayingDuringRefresh() {
  const state = selectionFixture();
  state.display.retain(["a"]);
  const snapshot = state.snapshot("a");
  const selection = {document: 0, line: 12, revision: 1, contentRevision: 1};
  await state.display.show("a", snapshot, selection);
  const editor = state.display.editor;
  assert.deepEqual(editor.reveals, [{document: 0, line: 12}]);
  editor.reading = {document: 0, line: 6};
  await state.display.show("a", {...snapshot, revision: 2}, selection);
  assert.deepEqual(editor.reading, {document: 0, line: 6});
  assert.equal(editor.reveals.length, 1);
  await state.display.show("a", snapshot, {...selection, revision: 2});
  assert.deepEqual(editor.reading, {document: 0, line: 12});
  assert.equal(editor.reveals.length, 2);
  assert.equal(state.created.length, 1);
  await state.display.show("a", snapshot);
  assert.equal(editor.reveals.length, 2);
  state.display.clear();
  await state.display.show("a", snapshot);
  assert.equal(state.created[1].reveals.length, 0);
  assert.deepEqual(state.created[1].saved, {document: 0, line: 12});
}

async function lineSelectionWaitsForTheRequestedContentRevision() {
  const state = selectionFixture();
  const snapshot = state.snapshot("a");
  const selection = {document: 0, line: 42, revision: 1, contentRevision: 2};
  await state.display.show("a", snapshot, selection);
  assert.equal(state.display.editor.reveals.length, 0);
  await state.display.show("a", {...snapshot, revision: 2}, selection);
  assert.deepEqual(state.display.editor.reveals, [{document: 0, line: 42}]);
}

async function markupLinksChoosePreviewOrSource(extension) {
  const state = selectionFixture();
  state.display.retain(["a", "b"]);
  const snapshot = state.snapshot("a", "a." + extension);
  const html = /^html?$/i.test(extension);
  await state.display.show("a", snapshot);
  assert.equal(state.display.editor, null);
  if (html) state.display.htmlPreview.position = [0, 120];
  else state.content.scrollTop = 120;
  await state.display.show("a", snapshot, {document: 0, line: 3, revision: 1}, "source");
  assert.equal(state.content.querySelector(".snapshot-markdown"), null);
  assert.equal(state.display.htmlPreview, null);
  assert.deepEqual(state.display.editor.reveals, [{document: 0, line: 3}]);
  await state.display.show("b", state.snapshot("b", "b.md"));
  await state.display.show("a", snapshot);
  assert.ok(state.display.editor);
  assert.equal(state.display.editor.reveals.length, 0);
  assert.deepEqual(state.display.editor.saved, {document: 0, line: 3});
  const editor = state.display.editor;
  await state.display.show("a", snapshot, null, "preview");
  assert.equal(editor.destroyed, true);
  assert.equal(state.display.editor, null);
  assert.equal(state.display.sourceViews.has("a"), false);
  if (html) assert.deepEqual(state.display.htmlPreview.position, [0, 120]);
  else assert.equal(state.content.scrollTop, 120);
  await state.display.show("a", snapshot, {document: 0, line: 7, revision: 2}, "source");
  assert.deepEqual(state.display.editor.reveals, [{document: 0, line: 7}]);
  state.display.retain(["b"]);
  state.display.clear();
  await state.display.show("a", snapshot);
  assert.equal(state.display.editor, null);
  if (html) assert.equal(state.display.htmlPreview.source, "a");
  else assert.equal(state.content.querySelector(".snapshot-markdown").textContent, "a");
}

async function togglingMarkdownRestoresEachModesReadingPosition() {
  const state = selectionFixture();
  state.display.retain(["a"]);
  const snapshot = state.snapshot("a", "a.md");
  await state.display.show("a", snapshot, {document: 0, line: 12, revision: 1}, "source");
  const editor = state.display.editor;
  editor.reading = {document: 0, line: 30};
  await state.display.show("a", snapshot, null, "preview");
  assert.equal(editor.destroyed, true);
  assert.equal(state.content.querySelector(".snapshot-markdown").textContent, "a");
  state.content.scrollTop = 240;
  await state.display.show("a", snapshot, null, "source");
  assert.deepEqual(state.display.editor.saved, {document: 0, line: 30});
  assert.equal(state.display.editor.reveals.length, 0);
  await state.display.show("a", {...snapshot, revision: 2}, null, "preview");
  assert.equal(state.content.scrollTop, 240);
}

async function htmlUpdatesReuseTheFrameAndSwitchingReleasesIt() {
  const state = selectionFixture();
  state.display.retain(["a", "b"]);
  const snapshot = state.snapshot("<h1>HTML</h1>", "a.html");
  await state.display.show("a", snapshot);
  const preview = state.display.htmlPreview;
  preview.position = [25, 400];
  await state.display.show("a", {...snapshot});
  assert.equal(state.display.htmlPreview, preview);
  await state.display.show("a", {...snapshot, revision: 2,
    documents: [{label: "a.html", text: "<h1>Updated</h1>"}]});
  assert.equal(state.display.htmlPreview, preview);
  assert.equal(preview.source, "<h1>Updated</h1>");
  assert.deepEqual(preview.position, [25, 400]);
  await state.display.show("b", state.snapshot("b", "b.md"));
  assert.equal(preview.destroyed, true);
  await state.display.show("a", snapshot);
  assert.notEqual(state.display.htmlPreview, preview);
  assert.deepEqual(state.display.htmlPreview.position, [25, 400]);
  state.display.retain([]);
  state.display.clear();
  assert.equal(state.htmlPreviews[1].destroyed, true);
  assert.equal(state.display.scrollPositions.size, 0);
}

async function lateSourceLoadsCannotReplaceMarkupPreview(extension, fail) {
  const loading = deferred();
  const state = displayFixture(() => loading.promise);
  const snapshot = state.snapshot("a", "a." + extension);
  const older = state.display.show("a", snapshot, {document: 0, line: 3, revision: 1}, "source");
  await state.display.show("a", snapshot, null, "preview");
  if (fail) loading.reject(new Error("late module error"));
  else loading.resolve({createSnapshotEditor: () => { throw new Error("stale creation"); }});
  await older;
  assert.equal(state.display.editor, null);
  if (extension === "html") assert.equal(state.display.htmlPreview.source, "a");
  else assert.equal(state.content.querySelector(".snapshot-markdown").textContent, "a");
}

async function markupDiffsAlwaysUseCodeEditors() {
  const state = selectionFixture();
  for (const extension of ["md", "html"]) {
    await state.display.show(extension, {
      kind: "diff", revision: 1, documents: [
        {label: "left." + extension, text: "old"}, {label: "right." + extension, text: "new"},
      ],
    }, null, "preview");
    assert.ok(state.display.editor);
    assert.equal(state.display.htmlPreview, null);
    assert.equal(state.content.querySelector(".snapshot-markdown"), null);
  }
}

async function lateSelectionAfterAnUpdateCannotOverrideTheNewTarget() {
  const state = selectionFixture();
  const snapshot = state.snapshot("a");
  await state.display.show("a", snapshot);
  const updating = deferred();
  const editor = state.display.editor;
  editor.updating = updating.promise;
  const older = state.display.show("a", snapshot, {document: 0, line: 12, revision: 1});
  editor.updating = null;
  await state.display.show("a", snapshot, {document: 0, line: 42, revision: 2});
  updating.resolve();
  await older;
  assert.deepEqual(editor.reveals, [{document: 0, line: 42}]);
}

async function lateSelectionAfterDiffCannotOverrideTheLatestRequest(next) {
  const state = selectionFixture();
  const snapshot = state.snapshot("a");
  await state.display.show("a", snapshot);
  const comparing = deferred();
  const editor = state.display.editor;
  editor.comparing = comparing.promise;
  const older = state.display.show("a", snapshot, {document: 0, line: 12, revision: 1});
  await flush();
  editor.comparing = null;
  if (next === "close") state.display.clear();
  else await state.display.show("a", snapshot, next);
  comparing.resolve();
  await older;
  assert.deepEqual(editor.reveals, next && next !== "close" ?
    [{document: next.document, line: next.line}] : []);
}

async function newerSelectionDuringEditorConstructionWins() {
  const creating = deferred();
  const reveals = [];
  let calls = 0;
  let destroyed = false;
  const state = displayFixture(async () => ({
    createSnapshotEditor: () => ++calls === 1 ? creating.promise : {
      revealLine: async (document, line) => { reveals.push(line); return true; },
    },
  }));
  const snapshot = state.snapshot("a");
  const older = state.display.show("a", snapshot, {document: 0, line: 12, revision: 1});
  await flush();
  await state.display.show("a", snapshot, {document: 0, line: 42, revision: 2});
  creating.resolve({destroy: () => { destroyed = true; }});
  await older;
  assert.equal(destroyed, true);
  assert.deepEqual(reveals, [42]);
}

async function switchingReleasesEditorsAndRetainsReadingState() {
  const created = [];
  const state = displayFixture(async () => ({
    createSnapshotEditor: async (parent, snapshot, saved) => {
      const editor = {
        parent, snapshot, saved, destroyed: false, reading: {scroll: 135, selection: [1, 4]},
        capture() { return this.reading; },
        destroy() { this.destroyed = true; },
      };
      created.push(editor);
      return editor;
    },
  }));
  state.display.retain(["a", "b"]);
  await state.display.show("a", state.snapshot("a"));
  await state.display.show("b", state.snapshot("b"));
  assert.equal(created[0].destroyed, true);
  assert.equal(created[0].parent.parent, null);
  assert.equal(state.display.editor, created[1]);
  await state.display.show("a", state.snapshot("a"));
  assert.equal(created[1].destroyed, true);
  assert.deepEqual(created[2].saved, {scroll: 135, selection: [1, 4]});
  state.display.retain(["b"]);
  state.display.clear();
  assert.equal(created[2].destroyed, true);
  assert.equal(state.display.editorStates.has("a"), false);
  assert.equal(state.display.editor, null);
  assert.equal(state.content.children.length, 0);
}

async function liveEditorUpdatesReuseTheActiveInstance() {
  const updates = [];
  let created = 0;
  const state = displayFixture(async () => ({
    createSnapshotEditor: async () => {
      created++;
      return {
        update: snapshot => updates.push(snapshot),
        capture: () => ({}),
        destroy() { throw new Error("refresh must retain the active editor"); },
      };
    },
  }));
  state.display.retain(["a"]);
  await state.display.show("a", state.snapshot("a"));
  const editor = state.display.editor;
  const refreshed = state.snapshot("updated", "a.py");
  await state.display.show("a", refreshed);
  assert.equal(created, 1);
  assert.equal(state.display.editor, editor);
  assert.deepEqual(updates, [refreshed]);
}

async function lateEditorLoadCannotReplaceActivePreview(fail) {
  const loading = deferred();
  let constructed = false;
  const state = displayFixture(() => loading.promise);
  state.display.retain(["a", "b"]);
  const showing = state.display.show("a", state.snapshot("a"), {document: 0, line: 12, revision: 1});
  await state.display.show("b", state.snapshot("b", "b.md"));
  if (fail) loading.reject(new Error("late import failure"));
  else loading.resolve({createSnapshotEditor: () => { constructed = true; }});
  await showing;
  assert.equal(constructed, false);
  assert.equal(state.display.activeId, "b");
  assert.equal(state.content.querySelector(".snapshot-markdown").textContent, "b");
  assert.equal(state.content.textContent.includes("failure"), false);
}

async function lateEditorConstructionIsDestroyed() {
  const creating = deferred();
  let destroyed = false;
  const state = displayFixture(async () => ({createSnapshotEditor: () => creating.promise}));
  state.display.retain(["a", "b"]);
  const showing = state.display.show("a", state.snapshot("a"), {document: 0, line: 12, revision: 1});
  await flush();
  const oldParent = state.content.children[0];
  await state.display.show("b", state.snapshot("b", "b.md"));
  creating.resolve({destroy: () => { destroyed = true; }});
  await showing;
  assert.equal(destroyed, true);
  assert.equal(oldParent.parent, null);
  assert.equal(state.display.editor, null);
  assert.equal(state.display.activeId, "b");
}

async function editorFailureIsVisibleAndRetryCanSucceed() {
  let failure = true;
  const state = displayFixture(async () => {
    if (failure) throw new Error("module unavailable");
    return {createSnapshotEditor: async () => ({capture: () => ({}), destroy() {}})};
  });
  state.display.retain(["a"]);
  await state.display.show("a", state.snapshot("a"));
  assert.equal(state.content.textContent, "Unable to load code viewer: module unavailable");
  failure = false;
  await state.display.show("a", state.snapshot("a"));
  assert.ok(state.display.editor);
}

async function previewsKeepScrollAndClosedTabsReleaseCaches() {
  const state = displayFixture(() => { throw new Error("preview must not load code editor"); });
  state.display.retain(["a", "b"]);
  await state.display.show("a", state.snapshot("a", "a.md"));
  const preview = state.content.children[0];
  state.content.scrollTop = 320;
  state.content.scrollLeft = 20;
  await state.display.show("b", {kind: "file", documents: [{label: "b.png", image_url: "data:image/png;base64,test"}]});
  assert.equal(state.content.querySelector(".snapshot-image").src, "data:image/png;base64,test");
  await state.display.show("b", {kind: "file", documents: [{label: "b.png", image_url: "data:image/png;base64,updated"}]});
  assert.equal(state.content.querySelector(".snapshot-image").src, "data:image/png;base64,updated");
  await state.display.show("a", state.snapshot("a", "a.md"));
  assert.equal(state.content.children[0], preview);
  assert.equal(state.content.scrollTop, 320);
  assert.equal(state.content.scrollLeft, 20);
  await state.display.show("a", state.snapshot("updated", "a.md"));
  assert.equal(state.content.querySelector(".snapshot-markdown").textContent, "updated");
  assert.equal(state.content.scrollTop, 320);
  state.display.retain([]);
  state.display.clear();
  for (const cache of [state.display.previews, state.display.scrollPositions, state.display.editorStates]) {
    assert.equal(cache.size, 0);
  }
}

function contextModel(text) {
  const lines = text.split(/\r\n|\r|\n/);
  return {
    getLineCount: () => lines.length,
    getLineContent: line => lines[line - 1],
  };
}

function contextSelection(startLineNumber, startColumn, endLineNumber, endColumn) {
  return {startLineNumber, startColumn, endLineNumber, endColumn,
    positionLineNumber: endLineNumber, positionColumn: endColumn};
}

function contextCapturesSelectionAndCursorWithoutChangingSource() {
  const source = {label: '文件"name.py', source: {path: '/repo/文件"name.py'}, text: "unchanged"};
  const snapshot = {kind: "file", title: source.label, documents: [source]};
  const model = contextModel("before\r\n  alpha()\r\n  beta()\r\nafter\r\n");
  const selection = contextSelection(2, 3, 4, 1);
  selection.positionLineNumber = 2;
  selection.positionColumn = 3;
  const context = editorContext(snapshot, 0, model, selection);
  assert.equal(context.label, '文件"name.py · L2–3');
  assert.ok(context.text.includes('View: "文件\\"name.py"'));
  assert.ok(!/Cursor:|Language:|Label:|File:/.test(context.text));
  assert.ok(context.text.includes("Selection: 2:3–4:1 (end exclusive)"));
  assert.ok(context.text.includes("\n2: alpha()\n3:   beta()\n"));
  assert.ok(!context.text.includes("before") && !context.text.includes("after"));
  assert.equal(source.text, "unchanged");
  snapshot.title = "renamed view";
  assert.ok(context.text.includes('View: "文件\\"name.py"'));

  const unselected = contextModel("unselected text");
  unselected.getLineContent = () => { throw new Error("Unselected text must not be read."); };
  const reference = editorContext(snapshot, 0, unselected, contextSelection(15, 4, 15, 4));
  assert.ok(!/Selection:|Language:|Label:/.test(reference.text));
  assert.ok(reference.text.includes("Cursor: line 15, column 4"));
  assert.ok(reference.text.includes("cursor location"));
  assert.ok(!reference.text.includes("unselected text") && !/^\d+: /m.test(reference.text));
}

function contextPreservesGitRevisionAndDiffSide() {
  const revision = "a".repeat(40);
  const snapshot = {kind: "diff", documents: [
    {label: `HEAD~1 (${revision}):old.py`, git_source: {repository: "/repo", revision, path: "old.py"}},
    {label: "/repo/new.py", source: {path: "/repo/new.py", working_tree: true}},
  ]};
  snapshot.title = snapshot.documents.map(source => source.label).join(" <-> ");
  const left = editorContext(snapshot, 0, contextModel("old value"), contextSelection(1, 1, 1, 4));
  const right = editorContext(snapshot, 1, contextModel("new value"), contextSelection(1, 1, 1, 4));
  assert.ok(left.text.includes('View: ' + JSON.stringify(snapshot.title)));
  assert.ok(left.text.includes('Git repository: "/repo"'));
  assert.ok(left.text.includes("Diff side: left (original)\n"));
  assert.ok(left.text.includes("\n1: old\n"));
  assert.ok(right.text.includes('View: ' + JSON.stringify(snapshot.title)));
  assert.ok(right.text.includes("Diff side: right (modified)\n"));
  assert.ok(right.text.includes("\n1: new\n"));
  assert.ok(!right.text.includes("Git revision:"));
  assert.ok(left.label.endsWith(" · left L1–1") && right.label.endsWith(" · right L1–1"));
  const newer = "b".repeat(40);
  const versions = {
    ...snapshot, title: `HEAD~1 (${revision}):old.py <-> HEAD (${newer}):new.py`,
  };
  const compared = editorContext(versions, 0, contextModel("old value"), contextSelection(1, 1, 1, 4));
  assert.ok(compared.label.includes(revision.slice(0, 7) + " ↔ " + newer.slice(0, 7)));
  assert.ok(compared.text.includes('View: ' + JSON.stringify(versions.title)));
  const legacy = editorContext({kind: "file", title: "old snapshot", documents: [{label: "old snapshot"}]}, 0,
    contextModel("saved text"), contextSelection(1, 1, 1, 6));
  assert.ok(legacy.text.includes('View: "old snapshot"'));
  assert.ok(!legacy.text.includes("File:"));
}

function contextBoundsLongSelections() {
  const snapshot = {kind: "file", title: "long.py", documents: [{label: "long.py"}]};
  const lines = editorContext(snapshot, 0, contextModel("line\n".repeat(220)),
    contextSelection(1, 1, 221, 1));
  assert.ok(lines.text.includes("(excerpt truncated)"));
  assert.ok(lines.text.includes("\n200: line\n"));
  assert.ok(!lines.text.includes("\n201:"));
  const characters = editorContext(snapshot, 0, contextModel("a".repeat(12000) + "truncated tail"),
    contextSelection(1, 1, 1, 12015));
  assert.ok(characters.text.includes("(excerpt truncated)"));
  assert.ok(!characters.text.includes("truncated tail"));
  assert.equal(characters.text.split("\n").find(line => line.startsWith("1: ")).length, 12003);
}

async function onlyTheCurrentEditorCanInsertContext() {
  const actions = [];
  const inserted = [];
  const state = displayFixture(async () => ({
    createSnapshotEditor: async (parent, snapshot, saved, onInsertContext) => {
      actions.push(onInsertContext);
      return {capture: () => ({}), destroy() {}};
    },
  }));
  state.display.onInsertContext = context => inserted.push(context);
  state.display.retain(["a", "b"]);
  await state.display.show("a", state.snapshot("a"));
  actions[0]({text: "from a"});
  await state.display.show("b", state.snapshot("b"));
  actions[0]({text: "stale a"});
  actions[1]({text: "from b"});
  state.display.clear();
  actions[1]({text: "closed b"});
  assert.deepEqual(inserted, [{text: "from a"}, {text: "from b"}]);
}

async function main() {
  contextCapturesSelectionAndCursorWithoutChangingSource();
  contextPreservesGitRevisionAndDiffSide();
  contextBoundsLongSelections();
  await onlyTheCurrentEditorCanInsertContext();
  fileAndUriLinksPreserveBrowserDestinations();
  await newTabsAndRestore();
  await disabledCanvasKeepsFileViewsAndAnEmptyState();
  await pollingNeverFollowsOtherClients();
  await revealingDuringRefreshKeepsVisibleContent();
  await clickedLinesDoNotReplayOnPollOrManualSelection();
  await modeToggleClearsLineTargetsAndOverridesPendingOpens();
  for (const action of ["open", "select", "canvas", "close", "stop"]) {
    await oldOpensCannotOverrideTheLatestIntent(action, false);
    await oldOpensCannotOverrideTheLatestIntent(action, true);
  }
  await failedOpensPreserveTheActiveView();
  await oldPollCannotUndoAClick();
  await oldContentCannotReplaceSelectedTab(false);
  await oldContentCannotReplaceSelectedTab(true);
  await oldPollCannotReopenClosedTabs();
  await closingRespectsSelectionAndFailure();
  await stoppingDoesNotLeaveLoadingStuck();
  await liveUpdatesKeepVisibleContentAndInvalidateOtherTabs();
  await refreshFailuresKeepContentAndRetry();
  await switchingReleasesEditorsAndRetainsReadingState();
  await liveEditorUpdatesReuseTheActiveInstance();
  await lineSelectionReusesEditorsWithoutReplayingDuringRefresh();
  await lineSelectionWaitsForTheRequestedContentRevision();
  await togglingMarkdownRestoresEachModesReadingPosition();
  for (const extension of ["md", "markdown", "html", "htm", "HTML"]) {
    await markupLinksChoosePreviewOrSource(extension);
  }
  for (const extension of ["md", "html"]) {
    await lateSourceLoadsCannotReplaceMarkupPreview(extension, false);
    await lateSourceLoadsCannotReplaceMarkupPreview(extension, true);
  }
  await htmlUpdatesReuseTheFrameAndSwitchingReleasesIt();
  await htmlPreviewKeepsNativeResourcesAndReadingState();
  await markupDiffsAlwaysUseCodeEditors();
  await lateSelectionAfterAnUpdateCannotOverrideTheNewTarget();
  await lateSelectionAfterDiffCannotOverrideTheLatestRequest({document: 1, line: 42, revision: 2});
  await lateSelectionAfterDiffCannotOverrideTheLatestRequest(null);
  await lateSelectionAfterDiffCannotOverrideTheLatestRequest("close");
  await newerSelectionDuringEditorConstructionWins();
  await lateEditorLoadCannotReplaceActivePreview(false);
  await lateEditorLoadCannotReplaceActivePreview(true);
  await lateEditorConstructionIsDestroyed();
  await editorFailureIsVisibleAndRetryCanSucceed();
  await previewsKeepScrollAndClosedTabsReleaseCaches();
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
