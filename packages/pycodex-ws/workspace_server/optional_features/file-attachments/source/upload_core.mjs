// Browser-local attachments enter the ordinary, per-session composer draft.
export const LIMITS = Object.freeze({fileBytes: 256 * 1024, totalBytes: 512 * 1024,
  files: 32, messageChars: 160000, draftBytes: 512 * 1024, parts: 500, queue: 64});
const encoder = new TextEncoder();
const uid = () => crypto.getRandomValues(new Uint32Array(4)).join("-");
export const filePath = file => file.webkitRelativePath || file.name;
export const fileKey = file => JSON.stringify([filePath(file), file.size, file.lastModified]);
export const formatBytes = bytes => bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(1)} KB`;
const size = part => part.type === "context" ? 1 : part.text.length;
export const fileParts = draft => draft.parts.filter(part => part.type === "context" && part.fileAttachment);

function fileContent(buffer) {
  const bytes = new Uint8Array(buffer);
  let encoding = "utf-8";
  if (bytes[0] === 0xff && bytes[1] === 0xfe) encoding = "utf-16le";
  if (bytes[0] === 0xfe && bytes[1] === 0xff) encoding = "utf-16be";
  try {
    const content = new TextDecoder(encoding, {fatal: true}).decode(bytes);
    if (!/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(content)) return {encoding, content};
  } catch (_) { /* Keep binary bytes intact. */ }
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 16384) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 16384));
  }
  return {encoding: "base64", content: btoa(binary)};
}

export function createAttachment(file, buffer) {
  const payload = {name: file.name, path: filePath(file), bytes: file.size, ...fileContent(buffer)};
  return {type: "context", id: `file-${uid()}`,
    label: `文件 · ${payload.path} (${formatBytes(file.size)})`,
    text: `\n\n<file_attachment>\n${JSON.stringify(payload)}\n</file_attachment>\n`,
    fileAttachment: {version: 2, key: fileKey(file), bytes: file.size,
      name: file.name, path: payload.path, mime: file.type, encoding: payload.encoding}};
}

// Parse version 1 drafts too. Edited or malformed tokens remain removable.
export function attachmentPayload(part) {
  const match = /^\s*<file_attachment>\s*([\s\S]*)\s*<\/file_attachment>\s*$/.exec(part.text);
  if (!match || part.text.length > LIMITS.messageChars) return null;
  try {
    const value = JSON.parse(match[1]);
    return value && typeof value.content === "string" && typeof value.name === "string" &&
      typeof value.path === "string" && Number.isSafeInteger(value.bytes) && value.bytes >= 0 &&
      ["utf-8", "utf-16le", "utf-16be", "base64"].includes(value.encoding) ? value : null;
  } catch (_) { return null; }
}

export function draftStats(draft) {
  const attachments = fileParts(draft);
  return {files: attachments.length,
    bytes: attachments.reduce((n, part) => n + (Number.isSafeInteger(part.fileAttachment.bytes) &&
      part.fileAttachment.bytes >= 0 ? part.fileAttachment.bytes : 0), 0),
    chars: draft.parts.reduce((n, part) => n + part.text.length, 0),
    draftBytes: encoder.encode(JSON.stringify({version: 1, draft})).byteLength};
}

export function draftError(draft) {
  const stats = draftStats(draft);
  if (stats.files > LIMITS.files) return "每条消息最多 32 个附件，请移除一些文件后重试。";
  if (stats.bytes > LIMITS.totalBytes) return "附件总大小超过 512 KB，请移除一些文件后重试。";
  if (stats.chars > LIMITS.messageChars) return "编码后的消息超过 160,000 字符，请减少文件或草稿文字后重试。";
  if (stats.draftBytes > LIMITS.draftBytes) return "持久草稿超过 512 KB，请减少文件或草稿文字后重试。";
  if (draft.parts.length > LIMITS.parts) return "草稿片段过多，请减少附件或上下文后重试。";
  return "";
}

export function removeAttachment(draft, id) {
  const removed = [];
  let at = 0;
  const parts = draft.parts.filter(part => {
    const drop = part.type === "context" && part.fileAttachment && part.id === id;
    if (drop) removed.push(at);
    at += size(part);
    return !drop;
  });
  const adjust = n => n - removed.filter(position => position < n).length;
  return {parts, ...(draft.selection ? {selection: {
    start: adjust(draft.selection.start), end: adjust(draft.selection.end)}} : {})};
}

export function insertText(draft, text) {
  const length = draft.parts.reduce((n, part) => n + size(part), 0);
  const {start, end} = draft.selection || {start: length, end: length};
  const slice = (from, to) => {
    let at = 0;
    const result = [];
    for (const part of draft.parts) {
      const next = at + size(part);
      if (from < next && to > at) result.push(part.type === "context" ? part :
        {type: "text", text: part.text.slice(Math.max(0, from - at), to - at)});
      at = next;
    }
    return result;
  };
  const cursor = start + text.length;
  return {parts: [...slice(0, start), {type: "text", text}, ...slice(end, length)],
    selection: {start: cursor, end: cursor}};
}

// Recognize raster bytes, never render arbitrary HTML/SVG from an attachment.
export function rasterBlob(payload) {
  if (!payload || payload.encoding !== "base64" || payload.bytes > LIMITS.fileBytes ||
      payload.content.length > Math.ceil(LIMITS.fileBytes / 3) * 4) return null;
  try {
    const bytes = Uint8Array.from(atob(payload.content), char => char.charCodeAt(0));
    if (bytes.length !== payload.bytes) return null;
    const prefix = Array.from(bytes.subarray(0, 12));
    const ascii = String.fromCharCode(...prefix);
    let mime = "";
    if ([137,80,78,71,13,10,26,10].every((n, i) => prefix[i] === n)) mime = "image/png";
    else if (prefix[0] === 255 && prefix[1] === 216 && prefix[2] === 255) mime = "image/jpeg";
    else if (/^GIF8[79]a/.test(ascii)) mime = "image/gif";
    else if (ascii.startsWith("RIFF") && ascii.slice(8) === "WEBP") mime = "image/webp";
    return mime ? new Blob([bytes], {type: mime}) : null;
  } catch (_) { return null; }
}

export class AttachmentQueue {
  constructor(host, changed, Reader = FileReader) {
    this.host = host;
    this.changed = changed;
    this.Reader = Reader;
    this.sessionId = host.getActiveSessionId();
    this.epoch = 0;
    this.jobs = [];
    this.current = null;
    this.draining = false;
    this.disposed = false;
    this.notice = "";
  }
  get busy() { return this.jobs.some(job => ["queued", "reading"].includes(job.state)); }
  ticket() { return {sessionId: this.sessionId, epoch: this.epoch}; }
  valid(ticket) {
    return !this.disposed && ticket.sessionId && this.sessionId === ticket.sessionId &&
      this.host.getActiveSessionId() === ticket.sessionId && this.epoch === ticket.epoch;
  }
  emit() { if (!this.disposed) this.changed(); }
  reset(sessionId = this.host.getActiveSessionId()) {
    this.epoch++;
    const reader = this.current?.reader;
    this.current = null;
    this.jobs = [];
    this.notice = "";
    this.sessionId = sessionId;
    if (reader?.readyState === 1) reader.abort();
    this.emit();
  }
  add(files, ticket = this.ticket()) {
    if (!this.valid(ticket)) {
      this.notice = "会话已切换，请重新选择文件。";
      this.emit();
      return;
    }
    const keys = new Set([...fileParts(this.host.getDraft()).map(part => part.fileAttachment.key),
      ...this.jobs.map(job => job.key)]);
    let duplicates = 0, ignored = 0, overflow = 0;
    for (const file of files) {
      const path = filePath(file);
      if (file.webkitRelativePath && path.split("/").some(segment =>
        [".git", "node_modules", ".venv", "__pycache__"].includes(segment))) { ignored++; continue; }
      const key = fileKey(file);
      if (keys.has(key)) { duplicates++; continue; }
      if (this.jobs.length >= LIMITS.queue) { overflow++; continue; }
      keys.add(key);
      const tooLarge = file.size > LIMITS.fileBytes;
      this.jobs.push({id: uid(), key, path, name: file.name, bytes: file.size,
        file: tooLarge ? null : file, loaded: 0, state: tooLarge ? "error" : "queued",
        error: tooLarge ? "超过单文件 256 KB 限制，请选择较小文件。" : "", retryable: !tooLarge});
    }
    this.notice = [duplicates && `跳过 ${duplicates} 个重复文件`,
      ignored && `跳过 ${ignored} 个依赖或缓存文件`,
      overflow && `队列已满，${overflow} 个文件未添加，请分批选择`].filter(Boolean).join("；");
    this.emit();
    void this.drain();
  }
  retry(id) {
    const job = this.jobs.find(item => item.id === id);
    if (!job || job.state !== "error" || !job.retryable || !this.valid(this.ticket())) return;
    job.state = "queued";
    job.loaded = 0;
    job.error = "";
    this.notice = "";
    this.emit();
    void this.drain();
  }
  remove(id) {
    this.jobs = this.jobs.filter(job => job.id !== id);
    if (this.current?.job.id === id && this.current.reader?.readyState === 1) this.current.reader.abort();
    this.emit();
  }
  cancel() {
    for (const job of this.jobs) {
      if (["queued", "reading"].includes(job.state)) {
        job.state = "error";
        job.error = "已取消；可重试或移除。";
      }
    }
    if (this.current?.reader?.readyState === 1) this.current.reader.abort();
    this.notice = "已取消未完成的读取，已加入草稿的附件仍会发送。";
    this.emit();
  }
  clearErrors() {
    this.jobs = this.jobs.filter(job => job.state !== "error");
    this.emit();
  }
  dispose() { this.reset(null); this.disposed = true; }
  read(job, ticket) {
    return new Promise((resolve, reject) => {
      const reader = new this.Reader();
      this.current = {job, reader};
      const finish = callback => value => {
        reader.onload = reader.onerror = reader.onabort = reader.onprogress = null;
        callback(value);
      };
      reader.onload = finish(() => resolve(reader.result));
      reader.onerror = finish(() => reject(new Error("无法读取文件，请重试。")));
      reader.onabort = finish(() => reject(new DOMException("已取消", "AbortError")));
      reader.onprogress = event => {
        if (this.valid(ticket) && job.state === "reading" && this.jobs.includes(job)) {
          job.loaded = Math.min(event.loaded, job.bytes);
          this.emit();
        }
      };
      reader.readAsArrayBuffer(job.file);
    });
  }
  async drain() {
    if (this.draining || this.disposed) return;
    this.draining = true;
    try {
      while (!this.disposed) {
        const job = this.jobs.find(item => item.state === "queued");
        if (!job || !this.valid(this.ticket())) break;
        const ticket = this.ticket();
        job.state = "reading";
        this.emit();
        try {
          // Check capacity again per file. One failure never discards other files.
          const before = draftStats(this.host.getDraft());
          if (before.files >= LIMITS.files) throw new Error("每条消息最多 32 个附件，请移除一些文件后重试。");
          if (before.bytes + job.bytes > LIMITS.totalBytes) throw new Error("附件总大小超过 512 KB，请移除一些文件后重试。");
          const buffer = await this.read(job, ticket);
          if (!this.valid(ticket) || !this.jobs.includes(job) || job.state !== "reading") continue;
          const part = createAttachment(job.file, buffer);
          const draft = this.host.getDraft();
          if (fileParts(draft).some(item => item.fileAttachment.key === job.key)) {
            this.jobs = this.jobs.filter(item => item !== job);
            continue;
          }
          const next = {...draft, parts: [...draft.parts, part]};
          const error = draftError(next);
          if (error) throw new Error(error);
          this.host.setDraft(next);
          this.jobs = this.jobs.filter(item => item !== job);
        } catch (error) {
          if (this.valid(ticket) && this.jobs.includes(job) && job.state === "reading") {
            job.state = "error";
            job.error = error.message || "文件添加失败，请重试。";
          }
        } finally {
          if (this.current?.job === job) this.current = null;
          this.emit();
        }
      }
    } finally {
      this.draining = false;
      this.emit();
    }
  }
}
