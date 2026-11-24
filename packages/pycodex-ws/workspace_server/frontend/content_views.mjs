// Workspace content tabs, live updates and editor lifecycle.
import DOMPurify from "dompurify";
import markdownit from "markdown-it";
import {tabLabels} from "./view_labels.mjs";
import {reconcileChildren} from "../workspace_dom.mjs";
import {TranscriptView} from "./transcript.mjs";
export {DOMPurify, markdownit};
export {Composer} from "./composer.mjs";
export {PromptHistory} from "./prompt_history.mjs";

export function isWorkspaceLink(href, filePaths = true) {
  if (typeof href !== "string" || !href || href !== href.trim() || /[\x00-\x1f\x7f]/.test(href)) {
    return false;
  }
  if (/^(?:diff:|git:|workspace:\/\/canvas\/?$)/i.test(href)) return true;
  if (!filePaths || /^(?:https?|ftps?|mailto|tel|sms|javascript|vbscript|data|blob|about|file):/i.test(href)) {
    return false;
  }
  const hash = href.indexOf("#");
  const destination = hash < 0 ? href : href.slice(0, hash);
  if (hash >= 0 && !/\.html?$/i.test(destination) && !/^L-?\d+$/i.test(href.slice(hash + 1))) return false;
  const path = destination.replace(/:-?\d+$/, "");
  return !!path && !path.startsWith("//") && !path.includes("?") &&
    (!/^[a-z][a-z\d+.-]*:/i.test(path) || /^[a-z]:(?:[/\\]|%2f|%5c)/i.test(path));
}

function htmlSectionLink(href) {
  const match = /^([^#]*\.html?)#(.*)$/is.exec(href);
  if (!match || /^diff:/i.test(href) || /^L-?\d+$/i.test(match[2]) || !isWorkspaceLink(href)) {
    return null;
  }
  if (Array.from(href).length > 8192) {
    throw new Error("Workspace links can contain at most 8192 characters.");
  }
  const anchor = decodeURIComponent(match[2]);
  if (/[\x00-\x1f\x7f]/.test(anchor)) {
    throw new Error("Control characters are not allowed in workspace links.");
  }
  return {href: match[1], anchor};
}

export class ContentTabs {
  constructor({
    request, onChange, onSelect, selected = null, enableBoard = true, vscodeWindow = null,
  }) {
    this.request = request;
    this.onChange = onChange;
    this.onSelect = onSelect;
    this.enableBoard = enableBoard;
    this.vscodeWindow = vscodeWindow;
    this.vscodeMode = false;
    this.views = [];
    this.transcripts = new Map();
    this.activeId = enableBoard ? "board" : null;
    this.snapshot = null;
    this.selection = null;
    this.displayMode = null;
    this.loading = false;
    this.error = "";
    this.closing = new Set();
    this.cache = new Map();
    this.initialSelection = selected;
    this.initialized = false;
    this.mutation = 0;
    this.contentRequest = null;
    this.openRequest = null;
  }

  isBuiltin(id) {
    return (id === "board" && this.enableBoard) ||
      (id === "vscode" && this.vscodeMode);
  }

  stop() {
    this.cancelOpen();
    this.mutation += 1;
    if (this.contentRequest) this.contentRequest.abort();
    this.contentRequest = null;
    this.loading = false;
  }

  /** Token for a view list request; a later click, close or stop invalidates it. */
  pollToken() {
    return this.openRequest ? null : this.mutation;
  }

  applyViews(views, token, error = "") {
    if (token === null || token !== this.mutation || this.openRequest) return;
    if (error) {
      this.error = "Unable to load tabs: " + error;
      this.onChange(this);
      return;
    }
    this.updateViews(views);
    const remaining = new Set(this.views.map(view => view.id));
    this.error = "";
    let next = this.activeId;
    if (!this.initialized) {
      next = this.initialSelection || this.defaultId();
      this.initialized = true;
    }
    if (next !== null && !this.isBuiltin(next) && !remaining.has(next)) next = this.defaultId();
    if (next !== this.activeId || (next !== null && !this.isBuiltin(next) && !this.snapshot && !this.loading)) {
      this.select(next, null, null, false);
    } else {
      if (next !== null && !this.isBuiltin(next) && !this.cache.has(next) && !this.contentRequest) this.load(next);
      this.onChange(this);
    }
  }

  defaultId() {
    return this.enableBoard ? "board" : this.views.length ? this.views[0].id : null;
  }

  async setVscodeMode(enabled) {
    if (!this.vscodeWindow) return false;
    const next = Boolean(enabled);
    if (this.vscodeMode === next) {
      if (next && this.activeId !== "vscode") await this.select("vscode");
      return this.vscodeMode;
    }
    this.cancelOpen();
    this.mutation += 1;
    this.vscodeMode = next;
    if (next) {
      this.initialized = true;
      await this.select("vscode");
    } else if (this.activeId === "vscode") {
      await this.select(this.defaultId());
    } else {
      this.onChange(this);
    }
    return this.vscodeMode;
  }

  updateViews(views) {
    this.views = views.concat(Array.from(this.transcripts.values()));
    const revisions = new Map(this.views.map(view => [view.id, view.revision]));
    for (const [id, snapshot] of this.cache) {
      if (!revisions.has(id) || snapshot.revision < revisions.get(id)) this.cache.delete(id);
    }
  }

  cancelOpen() {
    if (this.openRequest) this.openRequest.abort();
    this.openRequest = null;
  }

  openTranscript(sessionId, agentId, agentPath) {
    const id = `transcript:${sessionId}:${agentId}`;
    if (!this.transcripts.has(id)) {
      const view = {
        id, kind: "transcript", title: agentPath, sessionId, agentId,
        revision: 1, documents: [],
      };
      this.transcripts.set(id, view);
      this.views.push(view);
      this.cache.set(id, view);
    }
    this.initialized = true;
    return this.select(id);
  }

  async open(href, srcPath, sessionId) {
    this.cancelOpen();
    if (/^workspace:\/\/canvas\/?$/i.test(href)) {
      if (!this.enableBoard) throw new Error("Canvas is disabled for this workspace.");
      await this.select("board");
      return;
    }
    if (this.vscodeMode) {
      const request = new AbortController();
      this.openRequest = request;
      this.mutation += 1;
      this.error = "";
      this.onChange(this);
      try {
        await this.request("api/vscode/open", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({href, src_path: srcPath, session_id: sessionId, window_id: this.vscodeWindow.id}),
          signal: request.signal,
        });
        if (this.openRequest !== request) return;
        this.mutation += 1;
        this.initialized = true;
        this.openRequest = null;
        return this.select("vscode");
      } catch (error) {
        if (this.openRequest !== request || error.name === "AbortError") return;
        this.error = "Unable to open in VS Code: " + error.message;
        throw error;
      } finally {
        if (this.openRequest === request) {
          this.openRequest = null;
          this.onChange(this);
        }
      }
    }
    const request = new AbortController();
    this.openRequest = request;
    this.mutation += 1;
    this.error = "";
    this.onChange(this);
    try {
      const section = htmlSectionLink(href);
      const payload = await this.request("api/views/open", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({href: section ? section.href : href, src_path: srcPath, session_id: sessionId}),
        signal: request.signal,
      });
      if (this.openRequest !== request) return;
      this.mutation += 1;
      this.updateViews(payload.views);
      const view = payload.view;
      this.cache.set(view.id, view);
      this.initialized = true;
      this.openRequest = null;
      const selection = section ? {anchor: section.anchor} : view.selection;
      return this.select(view.id, selection ? {
        ...selection, revision: view.reveal_revision, contentRevision: view.revision,
      } : null, selection?.line ? "source" : "preview");
    } catch (error) {
      if (this.openRequest !== request || error.name === "AbortError") return;
      this.error = "Unable to open view: " + error.message;
      throw error;
    } finally {
      if (this.openRequest === request) {
        this.openRequest = null;
        this.onChange(this);
      }
    }
  }

  async select(id, selection = null, displayMode = null, reveal = true) {
    if (id === "board" && !this.enableBoard) return;
    if (id === "vscode" && !this.vscodeMode) return;
    const builtin = this.isBuiltin(id);
    if (id !== null && !builtin && !this.views.some(view => view.id === id)) return;
    this.cancelOpen();
    const current = !builtin && id === this.activeId ? this.snapshot : null;
    if (this.contentRequest) this.contentRequest.abort();
    this.contentRequest = null;
    this.activeId = id;
    this.selection = selection;
    this.displayMode = displayMode;
    if (!this.initialized) this.initialSelection = id;
    this.snapshot = builtin ? null : (this.cache.get(id) || current);
    this.loading = id !== null && !builtin && !this.snapshot;
    this.error = "";
    this.onChange(this);
    if (reveal) this.onSelect(id);
    if (id === null || builtin || this.cache.has(id)) return;
    await this.load(id);
  }

  setDisplayMode(displayMode) {
    if (!this.snapshot || this.activeId === null || this.isBuiltin(this.activeId) ||
        !["source", "preview"].includes(displayMode)) return;
    this.cancelOpen();
    this.selection = null;
    this.displayMode = displayMode;
    this.onChange(this);
  }

  async load(id) {
    const request = new AbortController();
    this.contentRequest = request;
    try {
      const snapshot = await this.request("api/views/" + encodeURIComponent(id), {
        signal: request.signal,
      });
      if (this.contentRequest !== request || this.activeId !== id) return;
      const view = this.views.find(view => view.id === id);
      if (!view || snapshot.revision < view.revision) return;
      this.cache.set(id, snapshot);
      this.snapshot = snapshot;
      this.error = "";
    } catch (error) {
      if (this.contentRequest !== request || this.activeId !== id || error.name === "AbortError") return;
      this.error = "Unable to load view: " + error.message;
    } finally {
      if (this.contentRequest === request) {
        this.contentRequest = null;
        this.loading = false;
        this.onChange(this);
      }
    }
  }

  async close(id) {
    if (id === null || id === "board" || id === "vscode" || this.closing.has(id)) return;
    this.cancelOpen();
    this.closing.add(id);
    this.mutation += 1;
    this.onChange(this);
    try {
      if (this.transcripts.has(id)) this.transcripts.delete(id);
      else await this.request("api/views/" + encodeURIComponent(id), {method: "DELETE"});
      this.mutation += 1;
      const index = this.views.findIndex(view => view.id === id);
      this.views = this.views.filter(view => view.id !== id);
      this.cache.delete(id);
      if (this.activeId === id) {
        await this.select(
          index > 0 ? this.views[index - 1].id : (this.vscodeMode ? "vscode" : this.defaultId())
        );
      }
    } catch (error) {
      this.error = "Unable to close tab: " + error.message;
    } finally {
      this.closing.delete(id);
      this.onChange(this);
    }
  }
}

function element(document, tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

class HtmlPreview {
  constructor(parent, position = [0, 0]) {
    this.document = parent.ownerDocument;
    this.window = this.document.defaultView;
    this.position = position;
    this.generation = 0;
    this.ready = false;
    this.frame = element(this.document, "iframe", "snapshot-html");
    this.frame.title = "HTML preview";
    this.receive = event => {
      if (event.source !== this.frame.contentWindow) return;
      const message = event.data;
      if (!message || message.generation !== this.generation) return;
      if (message.type === "workspace-preview-ready") {
        this.ready = true;
        this.restorePosition();
      } else if (message.type === "workspace-preview-scroll" &&
          Array.isArray(message.position) && message.position.length === 2 &&
          message.position.every(Number.isFinite)) {
        this.position = message.position;
      }
    };
    this.window.addEventListener("message", this.receive);
    parent.appendChild(this.frame);
  }

  restorePosition() {
    this.frame.contentWindow.postMessage({
      type: "workspace-preview-restore", generation: this.generation, position: this.position,
      anchor: this.anchor,
    }, "*");
    this.anchor = undefined;
  }

  update(source, selection) {
    const reveal = selection && selection.revision !== this.selectionRevision;
    if (reveal) {
      this.selectionRevision = selection.revision;
      this.anchor = selection.anchor;
    }
    if (source === this.source) {
      if (reveal && this.ready) this.restorePosition();
      return;
    }
    this.ready = false;
    const page = new this.window.DOMParser().parseFromString(source, "text/html");
    const generation = this.generation + 1;
    const script = page.createElement("script");
    // Like Canvas, this is a same-origin page; the bridge preserves scrolling.
    script.textContent = `(() => {
      const generation = ${generation};
      let restored = false;
      function report() {
        if (restored) parent.postMessage({
          type: "workspace-preview-scroll", generation, position: [scrollX, scrollY],
        }, "*");
      }
      addEventListener("message", event => {
        const message = event.data;
        if (event.source !== parent || !message ||
            message.type !== "workspace-preview-restore" || message.generation !== generation) return;
        scrollTo(message.position[0], message.position[1]);
        if (message.anchor !== undefined) {
          const target = document.getElementById(message.anchor) ||
            Array.from(document.getElementsByName(message.anchor)).find(node => node.localName === "a");
          if (target) target.scrollIntoView();
          else if (!message.anchor || message.anchor.toLowerCase() === "top") scrollTo(0, 0);
        }
        restored = true;
        report();
      });
      addEventListener("scroll", report, {passive: true});
      addEventListener("load", () => parent.postMessage({
        type: "workspace-preview-ready", generation,
      }, "*"));
    })();`;
    page.head.prepend(script);
    const doctype = page.doctype ?
      new this.window.XMLSerializer().serializeToString(page.doctype) + "\n" : "";
    this.frame.srcdoc = doctype + page.documentElement.outerHTML;
    this.generation = generation;
    this.source = source;
  }

  capture() { return this.position; }

  destroy() {
    this.window.removeEventListener("message", this.receive);
  }
}

export class SnapshotDisplay {
  constructor({content, renderMarkdown, loadEditor = null, onInsertContext = null, onChangeMode = null,
    request = null,
    createHtmlPreview = (parent, position) => new HtmlPreview(parent, position)}) {
    this.content = content;
    this.document = content.ownerDocument;
    this.renderMarkdown = renderMarkdown;
    this.request = request;
    this.transcripts = new Map();
    this.transcript = null;
    this.onInsertContext = onInsertContext;
    this.onChangeMode = onChangeMode;
    this.loadEditor = loadEditor || (() => this.loadEditorAssets());
    this.editorAssets = null;
    this.createHtmlPreview = createHtmlPreview;
    this.activeId = null;
    this.editor = null;
    this.htmlPreview = null;
    this.pdfFrame = null;
    this.generation = 0;
    this.remaining = new Set();
    this.editorStates = new Map();
    this.previews = new Map();
    this.scrollPositions = new Map();
    this.sourceViews = new Set();
    this.pendingSelection = null;
    this.selectionRevision = 0;
  }

  loadEditorAssets() {
    if (!this.editorAssets) {
      const stylesheet = this.document.createElement("link");
      stylesheet.rel = "stylesheet";
      stylesheet.href = new URL("./editor.css", import.meta.url).href;
      const styles = new Promise((resolve, reject) => {
        stylesheet.addEventListener("load", resolve, {once: true});
        stylesheet.addEventListener("error",
          () => reject(new Error("Editor stylesheet request failed.")), {once: true});
      });
      this.document.head.insertBefore(
        stylesheet, this.document.head.querySelector("[data-pycodex-custom-style]")
      );
      this.editorAssets = Promise.all([import("./editor.mjs"), styles])
        .then(([module]) => module)
        .catch(error => {
          stylesheet.remove();
          this.editorAssets = null;
          throw error;
        });
    }
    return this.editorAssets;
  }

  retain(ids) {
    this.remaining = new Set(ids);
    for (const [id, transcript] of this.transcripts) {
      if (!this.remaining.has(id)) {
        transcript.stop();
        this.transcripts.delete(id);
      }
    }
    for (const cache of [this.editorStates, this.previews, this.scrollPositions, this.sourceViews]) {
      for (const id of cache.keys()) {
        if (!this.remaining.has(id)) cache.delete(id);
      }
    }
  }

  clear(message = "") {
    this.generation += 1;
    if (this.transcript) {
      if (this.remaining.has(this.activeId)) {
        this.scrollPositions.set(this.activeId, [this.content.scrollLeft, this.content.scrollTop]);
      }
      this.transcript.stop();
      this.transcript = null;
    }
    if (this.editor) {
      if (this.remaining.has(this.activeId)) {
        this.editorStates.set(this.activeId, this.editor.capture());
      }
      this.editor.destroy();
      this.editor = null;
    } else if (this.htmlPreview) {
      if (this.remaining.has(this.activeId)) {
        this.scrollPositions.set(this.activeId, this.htmlPreview.capture());
      }
      this.htmlPreview.destroy();
      this.htmlPreview = null;
    } else if (this.remaining.has(this.activeId) && this.previews.has(this.activeId)) {
      this.scrollPositions.set(this.activeId, [this.content.scrollLeft, this.content.scrollTop]);
    }
    this.activeId = null;
    this.pdfFrame = null;
    this.content.classList.remove("code-view");
    this.content.classList.remove("html-view");
    this.content.classList.remove("pdf-view", "transcript-view");
    this.content.textContent = "";
    if (message) {
      this.content.appendChild(element(this.document, "div", "content-placeholder", message));
    }
  }

  addModeToggle(parent, sourceMode, before = null) {
    if (!this.onChangeMode) return;
    const button = element(this.document, "button", "content-mode-toggle",
      sourceMode ? "Show preview" : "Show source");
    button.type = "button";
    button.title = sourceMode ? "Show rendered Markdown" : "Show Markdown source";
    button.setAttribute("aria-label", button.textContent);
    button.addEventListener("click", () => this.onChangeMode(sourceMode ? "preview" : "source"));
    parent.insertBefore(button, before);
  }

  async show(id, snapshot, selection = null, displayMode = null) {
    if (snapshot.kind === "transcript") {
      if (this.activeId === id && this.transcript) return;
      this.clear();
      this.activeId = id;
      this.content.classList.add("transcript-view");
      if (!this.transcripts.has(id)) {
        this.transcripts.set(id, new TranscriptView({
          viewport: this.content, request: this.request, renderMarkdown: this.renderMarkdown,
          sessionId: snapshot.sessionId, agentId: snapshot.agentId, agentPath: snapshot.title,
        }));
      }
      this.transcript = this.transcripts.get(id);
      this.content.appendChild(this.transcript.root);
      const [left, top] = this.scrollPositions.get(id) || [0, 0];
      this.content.scrollLeft = left;
      this.content.scrollTop = top;
      this.transcript.start();
      return;
    }
    if (selection && snapshot.revision < selection.contentRevision) selection = null;
    this.pendingSelection = selection;
    if (displayMode === "preview" || selection?.anchor !== undefined) this.sourceViews.delete(id);
    else if (selection?.line || displayMode === "source") this.sourceViews.add(id);
    const source = snapshot.documents[0];
    const markup = snapshot.kind === "file" && !this.sourceViews.has(id);
    const markdown = markup && /\.(md|markdown)$/i.test(source.label) && this.renderMarkdown;
    const html = markup && /\.html?$/i.test(source.label);
    const imageUrl = markup && /\.svg$/i.test(source.label)
      ? "data:image/svg+xml;charset=utf-8," + encodeURIComponent(source.text)
      : source.image_url;
    const pdf = snapshot.kind === "file" && source.media_type === "application/pdf";
    const pdfUrl = pdf ? new URL(
      `../api/views/${encodeURIComponent(id)}/pdf?revision=${snapshot.revision}`, import.meta.url,
    ).href : null;
    if (pdf && source.error) {
      this.clear(source.error);
      return;
    }
    if (this.activeId === id && this.pdfFrame && pdf) {
      if (this.pdfFrame.src !== pdfUrl) this.pdfFrame.src = pdfUrl;
      return;
    }
    if (this.activeId === id && this.htmlPreview && html) {
      return this.htmlPreview.update(source.text, selection);
    }
    if (this.activeId === id && this.editor && !imageUrl && !markdown && !html && !pdf) {
      const generation = this.generation;
      try {
        await this.editor.update(snapshot);
        if (generation === this.generation) await this.applySelection(id, selection);
      } catch (error) {
        if (generation === this.generation) throw error;
      }
      return;
    }
    this.clear();
    this.activeId = id;
    const generation = this.generation;
    if (pdf) {
      this.content.classList.add("pdf-view");
      const heading = element(this.document, "div", "snapshot-heading");
      heading.appendChild(element(this.document, "span", "snapshot-label", source.label));
      const actions = element(this.document, "div", "pdf-actions");
      const open = element(this.document, "a", "content-mode-toggle", "Open in new tab");
      open.href = pdfUrl;
      open.target = "_blank";
      open.rel = "noopener";
      const download = element(this.document, "a", "content-mode-toggle", "Download");
      download.href = pdfUrl + "&download=true";
      actions.append(open, download);
      heading.appendChild(actions);
      this.pdfFrame = element(this.document, "iframe", "snapshot-pdf");
      this.pdfFrame.title = "PDF preview: " + source.label;
      this.pdfFrame.src = pdfUrl;
      this.content.append(heading, this.pdfFrame);
      return;
    }
    if (snapshot.kind === "diff" || (!imageUrl && !markdown && !html)) {
      this.content.classList.add("code-view");
      this.content.appendChild(element(this.document, "div", "content-placeholder", "Loading code viewer…"));
      try {
        const module = await this.loadEditor();
        if (generation !== this.generation) return;
        this.content.textContent = "";
        const parent = element(this.document, "div", "workspace-editor");
        this.content.appendChild(parent);
        const onInsertContext = this.onInsertContext && (context => {
          if (generation === this.generation && this.activeId === id) this.onInsertContext(context);
        });
        const editor = await module.createSnapshotEditor(
          parent, snapshot, this.editorStates.get(id), onInsertContext,
        );
        if (generation !== this.generation) {
          editor.destroy();
          return;
        }
        this.editor = editor;
        if (this.onChangeMode && this.renderMarkdown && snapshot.kind === "file" &&
            /\.(md|markdown)$/i.test(source.label)) {
          const toolbar = parent.querySelector(".editor-toolbar");
          const actions = toolbar.querySelector(".editor-actions");
          toolbar.insertBefore(element(this.document, "span", "snapshot-stamp", "Read only"), actions);
          this.addModeToggle(toolbar, true, actions);
        }
      } catch (error) {
        if (generation !== this.generation) return;
        this.content.textContent = "";
        this.content.appendChild(element(this.document, "div", "content-placeholder",
          "Unable to load code viewer: " + error.message));
        return;
      }
      await this.applySelection(id, selection);
      return;
    }
    if (html) {
      this.content.classList.add("html-view");
      this.htmlPreview = this.createHtmlPreview(
        this.content, this.scrollPositions.get(id)
      );
      return this.htmlPreview.update(source.text, selection);
    }
    if (!this.previews.has(id)) {
      const root = element(this.document, "div", "snapshot");
      const heading = element(this.document, "div", "snapshot-heading");
      heading.appendChild(element(this.document, "span", "snapshot-label", source.label));
      heading.appendChild(element(this.document, "span", "snapshot-stamp", "Read only"));
      if (markdown && this.onChangeMode) {
        heading.classList.add("markdown-heading");
        this.addModeToggle(heading, false);
      }
      root.appendChild(heading);
      if (imageUrl) {
        const image = element(this.document, "img", "snapshot-image");
        image.alt = source.label;
        root.appendChild(image);
      } else {
        root.appendChild(element(this.document, "div", "snapshot-markdown markdown"));
      }
      this.previews.set(id, root);
    }
    this.content.appendChild(this.previews.get(id));
    const image = this.content.querySelector(".snapshot-image");
    if (image && image.getAttribute("src") !== imageUrl) image.src = imageUrl;
    if (markdown) {
      const parent = this.content.querySelector(".snapshot-markdown");
      if (source.source) parent.setAttribute("data-src-path", source.source.path);
      this.renderMarkdown(parent, source.text, source.source ? `api/views/${encodeURIComponent(id)}/images/` : null);
    }
    const [left, top] = this.scrollPositions.get(id) || [0, 0];
    this.content.scrollLeft = left;
    this.content.scrollTop = top;
  }

  async applySelection(id, selection) {
    if (!selection || selection.revision <= this.selectionRevision) return;
    const editor = this.editor;
    const isCurrent = () => this.activeId === id && this.editor === editor &&
      this.pendingSelection === selection && selection.revision > this.selectionRevision;
    if (!isCurrent()) return;
    if (await editor.revealLine(selection.document, selection.line, isCurrent)) {
      this.selectionRevision = selection.revision;
    }
  }
}

function readVscodeFrameError(targetFrame) {
  try {
    const doc = targetFrame.contentDocument;
    if (!doc || doc.contentType !== "application/json") return "";
    const raw = (doc.body ? doc.body.textContent : doc.textContent || "").trim();
    if (!raw) return "VS Code server returned an empty error response.";
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed.detail === "string" && parsed.detail) {
        return parsed.detail;
      }
    } catch (_err) {}
    return raw.slice(0, 300);
  } catch (_err) {
    return "";
  }
}

export function mountContentViews({
  tabs, frame, vscodeFrame = null, content, status, request, renderMarkdown, onInsertContext,
  enableBoard = true, vscodeWindow = null, onVscodeModeChange = null,
  onVscodeStateChange = null,
}) {
  const document = tabs.ownerDocument;
  const display = new SnapshotDisplay({
    content, renderMarkdown, onInsertContext, request,
    onChangeMode: mode => controller.setDisplayMode(mode),
  });
  let displayedId;
  let displayedSnapshot = null;
  let displayedLoading = false;
  let displayedSelection = null;
  let displayedMode = null;
  let vscodeSrcAssigned = false;
  let vscodeLoading = false;
  let vscodeLoadError = "";
  let lastVscodeMode = false;
  let lastVscodeLoading = false;
  let lastVscodeError = "";
  let noticeMessage = "";
  let noticeTimer = null;
  let controller;
  let vscodeOverlay = null;
  let vscodeSpinner = null;
  let vscodeOverlayTitle = null;
  let vscodeOverlayDetail = null;
  let vscodeRetry = null;

  function notify(message, durationMs = 4000) {
    if (noticeTimer) clearTimeout(noticeTimer);
    noticeMessage = String(message || "");
    noticeTimer = noticeMessage && durationMs > 0 ? setTimeout(() => {
      noticeTimer = null;
      noticeMessage = "";
      render(controller);
    }, durationMs) : null;
    render(controller);
  }

  function loadVscodeFrame() {
    if (!vscodeFrame || !vscodeWindow) return;
    vscodeSrcAssigned = true;
    vscodeLoading = true;
    vscodeLoadError = "";
    vscodeFrame.src = vscodeWindow.url;
  }

  if (vscodeFrame) {
    vscodeOverlay = element(document, "div", "vscode-overlay");
    vscodeOverlay.hidden = true;
    vscodeOverlay.setAttribute("role", "status");
    vscodeOverlay.setAttribute("aria-live", "polite");
    vscodeSpinner = element(document, "span", "vscode-spinner");
    vscodeSpinner.setAttribute("aria-hidden", "true");
    vscodeOverlayTitle = element(document, "div", "vscode-overlay-title");
    vscodeOverlayDetail = element(document, "div", "vscode-overlay-detail");
    vscodeRetry = element(document, "button", "vscode-retry", "Retry");
    vscodeRetry.type = "button";
    vscodeRetry.hidden = true;
    vscodeRetry.addEventListener("click", () => {
      loadVscodeFrame();
      render(controller);
    });
    vscodeOverlay.appendChild(vscodeSpinner);
    vscodeOverlay.appendChild(vscodeOverlayTitle);
    vscodeOverlay.appendChild(vscodeOverlayDetail);
    vscodeOverlay.appendChild(vscodeRetry);
    const vscodeHost = vscodeFrame.parentNode || vscodeFrame.parent;
    if (vscodeHost) vscodeHost.appendChild(vscodeOverlay);

    vscodeFrame.addEventListener("load", () => {
      if (!vscodeSrcAssigned) return;
      vscodeLoading = false;
      vscodeLoadError = readVscodeFrameError(vscodeFrame);
      render(controller);
    });
    vscodeFrame.addEventListener("error", () => {
      if (!vscodeSrcAssigned) return;
      vscodeLoading = false;
      vscodeLoadError = "Network error while loading VS Code.";
      render(controller);
    });
  }

  function focusTab(id) {
    const tab = document.getElementById("content-tab-" + id);
    if (tab) {
      tab.focus({preventScroll: true});
      tab.scrollIntoView({block: "nearest", inline: "nearest"});
    }
  }

  function updateTabOverflow() {
    const scrollLeft = tabs.scrollLeft || 0;
    const clientWidth = tabs.clientWidth || 0;
    const scrollWidth = tabs.scrollWidth || 0;
    if (!tabs.dataset) return;
    tabs.dataset.overflowLeft = String(scrollLeft > 2);
    tabs.dataset.overflowRight = String(clientWidth > 0 && scrollLeft + clientWidth < scrollWidth - 2);
  }
  if (typeof tabs.addEventListener === "function") {
    tabs.addEventListener("scroll", updateTabOverflow, {passive: true});
  }
  const hostWindow = document.defaultView;
  if (hostWindow && typeof hostWindow.addEventListener === "function") {
    hostWindow.addEventListener("resize", updateTabOverflow, {passive: true});
  }

  function render(state) {
    const modeChanged = lastVscodeMode !== state.vscodeMode;
    if (
      state.vscodeMode &&
      vscodeFrame &&
      vscodeWindow &&
      (!vscodeSrcAssigned || (modeChanged && vscodeLoadError))
    ) {
      loadVscodeFrame();
    }
    if (modeChanged) {
      lastVscodeMode = state.vscodeMode;
      if (onVscodeModeChange) onVscodeModeChange(state.vscodeMode);
    }
    if (
      onVscodeStateChange &&
      (modeChanged || lastVscodeLoading !== vscodeLoading || lastVscodeError !== vscodeLoadError)
    ) {
      lastVscodeLoading = vscodeLoading;
      lastVscodeError = vscodeLoadError;
      onVscodeStateChange({
        active: state.vscodeMode,
        loading: state.vscodeMode && vscodeLoading,
        error: state.vscodeMode ? vscodeLoadError : "",
      });
    } else {
      lastVscodeLoading = vscodeLoading;
      lastVscodeError = vscodeLoadError;
    }
    const errors = state.snapshot ? state.snapshot.documents.map(source => source.error) : [];
    const errorMessage = [state.error, ...errors].filter(Boolean).join("; ");
    const openingMessage = state.openRequest
      ? (state.vscodeMode ? "Opening in VS Code…" : "Opening view…")
      : "";
    const infoMessage = openingMessage || noticeMessage;
    status.textContent = errorMessage || infoMessage;
    status.dataset.tone = errorMessage ? "error" : (infoMessage ? "info" : "");
    status.title = status.textContent;
    status.hidden = !status.textContent;
    const existing = new Map(Array.from(tabs.children, tab => [tab.dataset.viewId, tab]));
    const descriptors = (enableBoard ? [{id: "board", title: "Canvas", kind: "board"}] : [])
      .concat(state.vscodeMode ? [{id: "vscode", title: "VS Code", kind: "vscode"}] : [])
      .concat(state.views);
    const labels = tabLabels(state.views.filter(view => view.kind !== "transcript"));
    for (const view of state.views) {
      if (view.kind === "transcript") {
        labels.set(view.id, {name: view.title.split("/").pop(), description: "Transcript"});
      }
    }
    labels.set("board", {name: "Canvas", description: ""});
    labels.set("vscode", {
      name: "VS Code",
      description: vscodeLoadError ? "Error" : (vscodeLoading ? "Loading…" : ""),
    });
    reconcileChildren(tabs, descriptors.map(view => {
      let tab = existing.get(view.id);
      if (!tab) {
        tab = element(document, "div", "tab content-tab");
        tab.dataset.viewId = view.id;
        tab.id = "content-tab-" + view.id;
        tab.setAttribute("role", "tab");
        tab.setAttribute("aria-controls",
          view.id === "board" ? frame.id :
          view.id === "vscode" && vscodeFrame ? vscodeFrame.id : content.id);
        const label = element(document, "span", "tab-label content-tab-label");
        label.appendChild(element(document, "span", "content-tab-name"));
        label.appendChild(element(document, "span", "content-tab-description"));
        tab.appendChild(label);
        tab.addEventListener("click", () => controller.select(view.id));
        if (view.id !== "board" && view.id !== "vscode") {
          tab.addEventListener("mousedown", event => {
            if (event.button === 1) event.preventDefault();
          });
          tab.addEventListener("auxclick", event => {
            if (event.button === 1) {
              event.preventDefault();
              controller.close(view.id);
            }
          });
        }
        tab.addEventListener("keydown", event => {
          if (event.target !== tab) return;
          const order = (enableBoard ? ["board"] : [])
            .concat(controller.vscodeMode ? ["vscode"] : [])
            .concat(controller.views.map(item => item.id));
          const current = order.indexOf(view.id);
          let target = null;
          if (event.key === "ArrowRight") target = order[(current + 1) % order.length];
          if (event.key === "ArrowLeft") target = order[(current + order.length - 1) % order.length];
          if (event.key === "Home") target = order[0];
          if (event.key === "End") target = order[order.length - 1];
          if (event.key === "Enter" || event.key === " ") target = view.id;
          if (event.key === "Delete" && view.id !== "board" && view.id !== "vscode") {
            event.preventDefault();
            controller.close(view.id);
          }
          if (target !== null) {
            event.preventDefault();
            controller.select(target);
            document.getElementById("content-tab-" + target).focus();
          }
        });
        if (view.id !== "board" && view.id !== "vscode") {
          const close = element(document, "button", "tab-close content-tab-close");
          close.type = "button";
          close.title = "Close tab";
          close.setAttribute("aria-label", "Close " + view.title);
          close.innerHTML = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" ' +
            'stroke-width="1.5" stroke-linecap="round" aria-hidden="true">' +
            '<path d="M4 4l8 8M12 4l-8 8"/></svg>';
          close.addEventListener("click", event => {
            event.stopPropagation();
            controller.close(view.id);
          });
          tab.appendChild(close);
        }
      }
      const label = labels.get(view.id);
      tab.querySelector(".content-tab-name").textContent = label.name;
      const description = tab.querySelector(".content-tab-description");
      description.textContent = label.description;
      description.hidden = !label.description;
      tab.className = "tab content-tab" + (view.id === state.activeId ? " active" : "");
      tab.tabIndex = view.id === state.activeId ? 0 : -1;
      tab.title = view.id === "board" ? tabs.dataset.boardTitle :
        view.id === "vscode" ? (
          vscodeLoadError ? "VS Code · " + vscodeLoadError :
          vscodeLoading ? "VS Code · Loading workbench…" : "VS Code"
        ) :
        view.kind === "transcript" ? `${view.title}\nAgent transcript` :
        view.title + (view.live ? "\nRead only · follows file changes · " : "\nRead-only snapshot · ") +
        new Date(view.created_at).toLocaleString();
      tab.setAttribute("aria-selected", String(view.id === state.activeId));
      tab.setAttribute("aria-label", view.title);
      const close = tab.querySelector(".content-tab-close");
      if (close) {
        close.disabled = state.closing.has(view.id);
        close.setAttribute("aria-label", "Close " + view.title);
      }
      return tab;
    }));
    display.retain(state.views.map(view => view.id));
    updateTabOverflow();
    const isBoard = state.activeId === "board";
    const isVscode = state.activeId === "vscode";
    const isBuiltin = isBoard || isVscode;
    const switched = displayedId !== state.activeId;
    if (switched && state.activeId) {
      const activeTab = document.getElementById("content-tab-" + state.activeId);
      if (activeTab && typeof activeTab.scrollIntoView === "function") {
        activeTab.scrollIntoView({block: "nearest", inline: "nearest"});
        updateTabOverflow();
      }
    }
    const changed = switched || displayedSnapshot !== state.snapshot ||
      displayedLoading !== state.loading || displayedSelection !== state.selection ||
      displayedMode !== state.displayMode;
    // Hidden elements report zero scroll offsets, so capture before hiding.
    if (isBuiltin && changed) display.clear();
    frame.hidden = !isBoard;
    if (vscodeFrame) vscodeFrame.hidden = !isVscode;
    if (vscodeOverlay) {
      const showOverlay = isVscode && (vscodeLoading || Boolean(vscodeLoadError));
      vscodeOverlay.hidden = !showOverlay;
      if (showOverlay) {
        const hasError = Boolean(vscodeLoadError);
        vscodeOverlay.dataset.state = hasError ? "error" : "loading";
        vscodeSpinner.hidden = hasError;
        vscodeOverlayTitle.textContent = hasError ? "Unable to load VS Code" : "Loading VS Code…";
        vscodeOverlayDetail.textContent = hasError
          ? vscodeLoadError
          : "Starting host server and loading workbench…";
        vscodeRetry.hidden = !hasError;
      }
    }
    content.hidden = isBuiltin;
    if (!changed) return;
    displayedId = state.activeId;
    displayedSnapshot = state.snapshot;
    displayedLoading = state.loading;
    displayedSelection = state.selection;
    displayedMode = state.displayMode;
    if (state.activeId) content.setAttribute("aria-labelledby", "content-tab-" + state.activeId);
    else content.removeAttribute("aria-labelledby");
    content.setAttribute("aria-busy", String(state.loading));
    if (isBuiltin) return;
    if (state.snapshot) {
      const id = state.activeId;
      const snapshot = state.snapshot;
      display.show(id, snapshot, state.selection, state.displayMode).catch(error => {
        if (displayedId !== id || displayedSnapshot !== snapshot) return;
        displayedSnapshot = null;
        state.error = "Unable to update view: " + error.message;
        status.textContent = status.title = state.error;
        status.dataset.tone = "error";
        status.hidden = false;
      });
    } else {
      display.clear(state.loading ? "Loading view…" :
        state.activeId ? "View unavailable." : "Open a file link to view it here.");
    }
  }

  controller = new ContentTabs({
    request,
    enableBoard,
    vscodeWindow,
    onChange: render,
    onSelect: id => {
      content.closest(".workspace").dataset.mobilePane = "board";
      focusTab(id);
    },
  });
  render(controller);
  return {
    start() {
      render(controller);
    },
    stop() {
      if (noticeTimer) clearTimeout(noticeTimer);
      noticeTimer = null;
      noticeMessage = "";
      controller.stop();
      display.clear();
      displayedId = null;
    },
    pollToken() { return controller.pollToken(); },
    applyViews(views, token, error) { controller.applyViews(views, token, error); },
    open(href, srcPath, sessionId) { return controller.open(href, srcPath, sessionId); },
    openTranscript(sessionId, agentId, agentPath) {
      return controller.openTranscript(sessionId, agentId, agentPath);
    },
    setVscodeMode(enabled) { return controller.setVscodeMode(enabled); },
    toggleVscodeMode() { return controller.setVscodeMode(!controller.vscodeMode); },
    isVscodeMode() { return controller.vscodeMode; },
    notify(message, durationMs) { notify(message, durationMs); },
  };
}
