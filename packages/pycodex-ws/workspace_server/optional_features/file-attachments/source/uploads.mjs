import {AttachmentQueue, attachmentPayload, draftError, draftStats, fileParts,
  formatBytes, insertText, rasterBlob, removeAttachment} from "./upload_core.mjs";

const icons = {
  attach: '<path d="m9 13 7-7a3 3 0 0 1 4 4L9 21a5 5 0 0 1-7-7L13 3a2 2 0 0 1 3 3L5 17"/>',
  folder: '<path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/>',
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8M8 17h5"/>',
  close: '<path d="m6 6 12 12M18 6 6 18"/>',
};
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  for (const [key, value] of Object.entries({viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
    "stroke-width": "1.7", "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true"})) {
    svg.setAttribute(key, value);
  }
  svg.innerHTML = icons[name];
  return svg;
}
function button(label, name, className = "pycodex-upload-button") {
  const node = element("button", className);
  node.type = "button";
  if (name) node.append(icon(name));
  if (label) node.append(element("span", "", label));
  return node;
}

export function mountUploads(host) {
  const form = document.getElementById("composer");
  const surface = form?.querySelector(".composer-surface");
  const prompt = document.getElementById("prompt");
  if (!surface || !prompt || form.querySelector("[data-pycodex-file-upload]")) return;
  const root = element("section", "pycodex-file-upload");
  root.dataset.pycodexFileUpload = "";
  root.setAttribute("aria-label", "消息附件");
  const tray = element("ul", "pycodex-attachment-tray");
  tray.setAttribute("aria-label", "已添加及待处理的附件");
  tray.hidden = true;
  const toolbar = element("div", "pycodex-upload-toolbar");
  const filesButton = button("添加附件", "attach");
  filesButton.title = "选择多个文件，也可拖拽或粘贴文件";
  const folderButton = button("", "folder");
  folderButton.title = "添加文件夹；自动跳过 .git、node_modules、.venv 和 __pycache__";
  folderButton.setAttribute("aria-label", "添加文件夹");
  const cancelButton = button("取消读取");
  const clearButton = button("清除失败项");
  const status = element("span", "pycodex-upload-status");
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  status.setAttribute("aria-atomic", "true");
  const help = element("details", "pycodex-upload-help");
  help.append(element("summary", "", "限制"), element("div", "",
    "单文件最多 256 KB；每条消息最多 32 个附件、原始文件合计 512 KB。编码后消息最多 160,000 字符，持久草稿最多 512 KB。文本以内容发送，二进制以 base64 文本发送；图片缩略图仅供本地预览。"));
  const fileInput = element("input");
  fileInput.type = "file";
  fileInput.multiple = true;
  fileInput.hidden = true;
  fileInput.setAttribute("aria-label", "选择附件");
  const folderInput = element("input");
  folderInput.type = "file";
  folderInput.multiple = true;
  folderInput.hidden = true;
  folderButton.hidden = !("webkitdirectory" in folderInput);
  folderInput.setAttribute("webkitdirectory", "");
  folderInput.setAttribute("aria-label", "选择附件文件夹");
  const dropHint = element("div", "pycodex-upload-drop-hint", "松开以添加文件");
  dropHint.setAttribute("aria-hidden", "true");
  toolbar.append(filesButton, folderButton, cancelButton, clearButton, status, help);
  root.append(tray, toolbar, fileInput, folderInput, dropHint);
  surface.insertBefore(root, document.getElementById("contextBlockDetail"));

  const dialog = element("dialog", "pycodex-upload-preview");
  dialog.setAttribute("aria-labelledby", "pycodexUploadPreviewTitle");
  const previewHeading = element("div", "pycodex-upload-preview-heading");
  const previewTitle = element("h2");
  previewTitle.id = "pycodexUploadPreviewTitle";
  const previewClose = button("", "close");
  previewClose.setAttribute("aria-label", "关闭附件预览");
  const previewMeta = element("p", "pycodex-upload-preview-meta");
  const previewBody = element("div", "pycodex-upload-preview-body");
  previewHeading.append(previewTitle, previewClose);
  dialog.append(previewHeading, previewMeta, previewBody);
  root.append(dialog);

  let disposed = false, suspended = false, scheduled = false, unsubscribe = null;
  let previewId = null, previewText = null, chooser = null, dragDepth = 0, blockedSubmit = false;
  const rows = new Map(), resources = new Map(), listeners = [];
  const queue = new AttachmentQueue(host, schedule);
  function listen(target, type, handler, options) {
    target.addEventListener(type, handler, options);
    listeners.push(() => target.removeEventListener(type, handler, options));
  }
  function schedule() {
    if (disposed || scheduled || suspended) return;
    scheduled = true;
    queueMicrotask(() => { scheduled = false; if (!disposed && !suspended) render(); });
  }
  function resource(part) {
    let item = resources.get(part.id);
    if (item?.text === part.text) return item;
    if (item?.url) URL.revokeObjectURL(item.url);
    const payload = attachmentPayload(part);
    const blob = rasterBlob(payload);
    item = {text: part.text, payload, url: blob ? URL.createObjectURL(blob) : null};
    resources.set(part.id, item);
    return item;
  }
  function release() {
    closePreview();
    for (const item of resources.values()) if (item.url) URL.revokeObjectURL(item.url);
    resources.clear();
  }
  function closePreview() {
    previewId = null;
    previewText = null;
    if (dialog.open) dialog.close();
    previewBody.replaceChildren();
  }
  function preview(part, open = true) {
    const item = resource(part);
    const payload = item.payload;
    previewId = part.id;
    previewText = part.text;
    previewTitle.textContent = payload?.path || part.fileAttachment.path || part.label;
    previewMeta.textContent = payload ? `${formatBytes(payload.bytes)} · ${payload.encoding === "base64" ?
      "二进制编码文本" : "文本内容"} · 随消息发送` : "附件内容已编辑，无法解析预览；仍可查看草稿中的上下文或移除。";
    previewBody.replaceChildren();
    if (item.url) {
      const image = element("img");
      image.src = item.url;
      image.alt = payload.name;
      previewBody.append(image);
    } else if (payload && payload.encoding !== "base64") {
      previewBody.append(element("pre", "", payload.content || "（空文件）"));
    } else {
      previewBody.append(element("p", "", payload ?
        "此类型暂不支持内容预览。文件编码内容会随消息发送。" : "可关闭此预览后点击输入框中的附件查看上下文。"));
    }
    if (open && !dialog.open) dialog.showModal();
  }
  function createRow(key) {
    const node = element("li", "pycodex-attachment");
    const open = button("", null, "pycodex-attachment-open");
    const visual = element("span", "pycodex-attachment-visual");
    visual.append(icon("file"));
    const info = element("span", "pycodex-attachment-info");
    const name = element("span", "pycodex-attachment-name");
    const detail = element("span", "pycodex-attachment-detail");
    info.append(name, detail);
    open.append(visual, info);
    const retry = button("重试");
    const remove = button("", "close");
    remove.classList.add("pycodex-attachment-remove");
    const progress = element("progress", "pycodex-attachment-progress");
    progress.max = 100;
    node.append(open, retry, remove, progress);
    const row = {node, open, visual, name, detail, retry, remove, progress, url: undefined};
    rows.set(key, row);
    return row;
  }
  function render() {
    if (!form.isConnected || !root.isConnected) { dispose(); return; }
    const sessionId = host.getActiveSessionId();
    if (queue.sessionId !== sessionId) { release(); queue.reset(sessionId); }
    const draft = host.getDraft();
    const attached = fileParts(draft), stats = draftStats(draft);
    const live = new Set(), liveResources = new Set();
    const items = [...attached.map(part => ({key: `ready:${part.id}`, part})),
      ...queue.jobs.map(job => ({key: `job:${job.id}`, job}))];
    let previous = null;
    for (const {key, part, job} of items) {
      live.add(key);
      const row = rows.get(key) || createRow(key);
      const item = part ? resource(part) : null;
      if (part) liveResources.add(part.id);
      const path = part ? item.payload?.path || part.fileAttachment.path || part.label : job.path;
      const bytes = part ? item.payload?.bytes ?? part.fileAttachment.bytes : job.bytes;
      row.name.textContent = path;
      row.name.title = path;
      row.node.dataset.state = part ? "ready" : job.state;
      row.node.dataset.attachmentId = part?.id || job.id;
      row.open.disabled = !part;
      row.open.setAttribute("aria-label", part ? `预览 ${path}` : path);
      row.open.onclick = part ? () => preview(part) : null;
      const detail = part ? `${formatBytes(bytes)} · ${!item.payload ? "已编辑" : item.payload.encoding === "base64" ? "编码附件" : "文本"} · 已就绪` :
        job.state === "error" ? job.error : `${formatBytes(bytes)} · ${job.state === "reading" ? "正在读取" : "等待读取"}`;
      row.detail.textContent = detail;
      row.detail.title = detail;
      if (row.url !== item?.url) {
        row.visual.replaceChildren();
        if (item?.url) {
          const image = element("img");
          image.src = item.url;
          image.alt = "";
          image.loading = "lazy";
          row.visual.append(image);
        } else row.visual.append(icon("file"));
        row.url = item?.url;
      }
      row.retry.hidden = !job || job.state !== "error" || !job.retryable;
      row.retry.setAttribute("aria-label", `重试 ${path}`);
      row.retry.onclick = job ? () => queue.retry(job.id) : null;
      row.remove.setAttribute("aria-label", `移除 ${path}`);
      row.remove.onclick = () => {
        const next = part ? removeAttachment(host.getDraft(), part.id) : null;
        prompt.focus({preventScroll: true});
        // setDraft restores the saved selection when the editor has focus.
        if (next) host.setDraft(next);
        else queue.remove(job.id);
        schedule();
      };
      row.progress.hidden = !job || job.state !== "reading";
      row.progress.value = job?.bytes ? Math.round(job.loaded / job.bytes * 100) : 0;
      row.progress.setAttribute("aria-label", `读取 ${path}`);
      const expected = previous ? previous.nextSibling : tray.firstChild;
      if (expected !== row.node) tray.insertBefore(row.node, expected);
      previous = row.node;
    }
    for (const [key, row] of rows) if (!live.has(key)) { row.node.remove(); rows.delete(key); }
    for (const [id, item] of resources) if (!liveResources.has(id)) {
      if (item.url) URL.revokeObjectURL(item.url);
      resources.delete(id);
    }
    if (previewId && !liveResources.has(previewId)) closePreview();
    else if (previewId) {
      const part = attached.find(item => item.id === previewId);
      if (previewText !== part.text) preview(part, false);
    }
    tray.hidden = !items.length;
    root.setAttribute("aria-busy", String(queue.busy));
    filesButton.disabled = folderButton.disabled = !sessionId;
    cancelButton.hidden = !queue.busy;
    const failures = queue.jobs.filter(job => job.state === "error").length;
    clearButton.hidden = !failures;
    const pending = queue.jobs.filter(job => ["queued", "reading"].includes(job.state)).length;
    if (!pending) blockedSubmit = false;
    const error = draftError(draft);
    status.dataset.state = error || failures ? "error" : "";
    const summary = !sessionId ? "请先选择会话" : pending ? `${pending} 个待完成 · 读取完成后可发送` :
      attached.length ? `${attached.length} 个附件 · ${formatBytes(stats.bytes)} · 随消息发送` : "拖拽或粘贴文件";
    const message = [summary, blockedSubmit && "请等待完成，或取消未完成的文件后发送", error,
      failures && `${failures} 个未添加，不会发送`, queue.notice].filter(Boolean).join("；");
    if (status.textContent !== message) status.textContent = message;
    status.title = `消息 ${stats.chars.toLocaleString()} / 160,000 字符；持久草稿 ${formatBytes(stats.draftBytes)} / 512 KB`;
  }
  function choose(input) {
    if (!host.getActiveSessionId()) return;
    chooser = queue.ticket();
    input.value = "";
    input.click();
  }
  function selection(event) {
    const files = Array.from(event.target.files || []), ticket = chooser;
    event.target.value = "";
    chooser = null;
    if (files.length && ticket) queue.add(files, ticket);
  }
  function clearDrag() { dragDepth = 0; form.classList.remove("pycodex-upload-drag"); }
  const hasFiles = event => Array.from(event.dataTransfer?.types || []).includes("Files");
  listen(filesButton, "click", () => choose(fileInput));
  listen(folderButton, "click", () => choose(folderInput));
  listen(cancelButton, "click", () => queue.cancel());
  listen(clearButton, "click", () => { queue.notice = ""; queue.clearErrors(); });
  listen(fileInput, "change", selection);
  listen(folderInput, "change", selection);
  listen(previewClose, "click", closePreview);
  listen(dialog, "click", event => { if (event.target === dialog) closePreview(); });
  listen(dialog, "close", () => { previewId = null; previewText = null; previewBody.replaceChildren(); });
  listen(form, "dragenter", event => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth++;
    form.classList.add("pycodex-upload-drag");
  }, true);
  listen(form, "dragover", event => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    event.dataTransfer.dropEffect = host.getActiveSessionId() ? "copy" : "none";
  }, true);
  listen(form, "dragleave", event => {
    if (!dragDepth) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) clearDrag();
  }, true);
  listen(form, "drop", event => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    clearDrag();
    const entries = Array.from(event.dataTransfer.items || []).filter(item => item.kind === "file")
      .map(item => ({directory: Boolean(item.webkitGetAsEntry?.()?.isDirectory), file: item.getAsFile()}));
    const files = entries.length ? entries.filter(item => !item.directory).map(item => item.file).filter(Boolean) :
      Array.from(event.dataTransfer.files || []);
    queue.add(files);
    if (entries.some(item => item.directory)) {
      queue.notice = [queue.notice, "文件夹请使用“添加文件夹”入口；其他文件已加入队列"].filter(Boolean).join("；");
      schedule();
    }
  }, true);
  listen(prompt, "paste", event => {
    let files = Array.from(event.clipboardData?.items || []).filter(item => item.kind === "file")
      .map(item => item.getAsFile()).filter(Boolean);
    if (!files.length) files = Array.from(event.clipboardData?.files || []);
    if (!files.length) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const text = event.clipboardData.getData("text/plain");
    if (text && host.getActiveSessionId()) {
      const next = insertText(host.getDraft(), text);
      const error = draftError(next);
      if (error) { queue.notice = error; schedule(); return; }
      host.setDraft(next);
    }
    queue.add(files);
  }, true);
  listen(form, "submit", event => {
    const error = draftError(host.getDraft());
    if (queue.busy || (fileParts(host.getDraft()).length && error)) {
      event.preventDefault();
      event.stopImmediatePropagation();
      blockedSubmit = queue.busy;
      if (!queue.busy) queue.notice = error;
      schedule();
    } else {
      blockedSubmit = false;
      queue.notice = "";
      queue.clearErrors();
      closePreview();
      schedule();
    }
  }, true);
  listen(prompt, "input", schedule);
  const observer = new MutationObserver(schedule);
  observer.observe(prompt, {subtree: true, childList: true, characterData: true, attributes: true,
    attributeFilter: ["data-context"]});
  observer.observe(surface, {childList: true});
  observer.observe(form.parentElement, {childList: true});
  function sessionChanged() {
    if (queue.sessionId !== host.getActiveSessionId()) {
      blockedSubmit = false;
      release();
      clearDrag();
      queue.reset();
      // Keep a chooser's original epoch, so A -> B -> A cannot accept its old result.
    }
    schedule();
  }
  function suspend() {
    suspended = true;
    blockedSubmit = false;
    chooser = null;
    unsubscribe?.();
    unsubscribe = null;
    queue.reset();
    release();
    clearDrag();
  }
  function subscribe() {
    if (disposed) return;
    suspended = false;
    if (!unsubscribe) unsubscribe = host.onSessionChanged(sessionChanged);
    sessionChanged();
    schedule();
  }
  function dispose() {
    if (disposed) return;
    suspend();
    disposed = true;
    queue.dispose();
    observer.disconnect();
    for (const remove of listeners) remove();
    root.remove();
  }
  listen(window, "pagehide", suspend);
  listen(window, "pageshow", subscribe);
  listen(window, "dragend", clearDrag);
  subscribe();
  return dispose;
}
