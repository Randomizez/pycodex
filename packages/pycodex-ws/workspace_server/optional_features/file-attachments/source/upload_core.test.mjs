// Run manually with Node 22. No network, browser, filesystem writes or model calls.
import assert from "node:assert/strict";
import {AttachmentQueue, LIMITS, attachmentPayload, createAttachment, draftError,
  draftStats, fileParts, insertText, rasterBlob, removeAttachment} from "./upload_core.mjs";

const clone = value => structuredClone(value);
function file(name, data = "hello", extra = {}) {
  const bytes = typeof data === "string" ? new TextEncoder().encode(data) : new Uint8Array(data);
  return {name, size: bytes.byteLength, type: "", lastModified: 1, webkitRelativePath: "",
    buffer: bytes.buffer, ...extra};
}
class Reader {
  static instances = [];
  constructor() { this.readyState = 0; Reader.instances.push(this); }
  readAsArrayBuffer(file) { this.file = file; this.readyState = 1; }
  progress(loaded) { this.onprogress?.({loaded}); }
  done() { this.readyState = 2; this.result = this.file.buffer; this.onload?.(); }
  fail() { this.readyState = 2; this.onerror?.(); }
  abort() { this.readyState = 2; this.onabort?.(); }
}
function fixture(draft = {parts: [{type: "text", text: "draft"}], selection: {start: 2, end: 2}}) {
  Reader.instances = [];
  const state = {id: "a", draft: clone(draft), writes: [], notifications: 0};
  const host = {getActiveSessionId: () => state.id, getDraft: () => clone(state.draft),
    setDraft: value => { state.draft = clone(value); state.writes.push(clone(value)); }};
  const queue = new AttachmentQueue(host, () => state.notifications++, Reader);
  return {state, queue, host};
}
const tick = async () => { for (let i = 0; i < 5; i++) await Promise.resolve(); };
const current = () => Reader.instances.at(-1);
const finish = async () => { current().done(); await tick(); };
let checks = 0;
async function check(name, run) { await run(); checks++; console.log(`PASS ${name}`); }

await check("encoding and legacy draft compatibility", () => {
  for (const input of [file("utf8.txt", "你好\ntext"), file("le.txt", [255,254,65,0]),
    file("be.txt", [254,255,0,65]), file("binary", [0,255,1,128])]) {
    const part = createAttachment(input, input.buffer), value = attachmentPayload(part);
    assert.equal(value.bytes, input.size);
    if (value.encoding === "base64") assert.deepEqual([...Buffer.from(value.content, "base64")], [...new Uint8Array(input.buffer)]);
    else assert.equal(value.content, input.name === "utf8.txt" ? "你好\ntext" : "A");
    const legacy = {...part, text: part.text.replace(JSON.stringify(value), JSON.stringify(value, null, 2)),
      fileAttachment: {key: part.fileAttachment.key, bytes: input.size}};
    assert.deepEqual(attachmentPayload(legacy), value);
  }
});
await check("latest draft, other context and selection survive reading", async () => {
  const {state, queue} = fixture();
  queue.add([file("a.txt")]);
  assert(queue.busy);
  state.draft = {parts: [{type: "context", id: "editor", label: "selection", text: "context"},
    {type: "text", text: "typed during read"}], selection: {start: 4, end: 7}};
  current().progress(3);
  assert.equal(queue.jobs[0].loaded, 3);
  await finish();
  assert.deepEqual(state.draft.parts.slice(0, 2), [{type: "context", id: "editor", label: "selection", text: "context"},
    {type: "text", text: "typed during read"}]);
  assert.deepEqual(state.draft.selection, {start: 4, end: 7});
  assert.equal(fileParts(state.draft).length, 1);
  assert(!queue.busy);
});
await check("oversize and read failures are isolated and retryable", async () => {
  const {state, queue} = fixture();
  queue.add([file("too-big", "", {size: LIMITS.fileBytes + 1}), file("unreadable"), file("ok")]);
  assert.equal(Reader.instances.length, 1);
  assert.equal(current().file.name, "unreadable");
  const failed = queue.jobs.find(job => job.name === "unreadable");
  current().fail(); await tick(); await finish();
  assert.equal(fileParts(state.draft).length, 1);
  assert.equal(queue.jobs.length, 2);
  assert(!queue.jobs.find(job => job.name === "too-big").retryable);
  queue.retry(failed.id); await finish();
  assert.equal(fileParts(state.draft).length, 2);
});
await check("dedupe includes queued and restored files; folder noise is skipped", async () => {
  const {state, queue} = fixture();
  const input = file("a");
  queue.add([input, input, file("lib", "x", {webkitRelativePath: "project/node_modules/lib"})]);
  queue.add([input]);
  assert.equal(queue.jobs.length, 1);
  await finish();
  queue.add([input]); await tick();
  assert.equal(Reader.instances.length, 1);
  assert.equal(fileParts(state.draft).length, 1);
});
await check("cancel and remove suppress late completion, retaining successes", async () => {
  const {state, queue} = fixture();
  queue.add([file("ready"), file("cancel"), file("waiting")]); await finish();
  const cancelled = queue.jobs[0], oldReader = current(), delayedLoad = oldReader.onload;
  queue.cancel(); await tick();
  oldReader.result = oldReader.file.buffer; delayedLoad(); await tick();
  assert.equal(fileParts(state.draft).length, 1);
  assert(!queue.busy);
  queue.retry(cancelled.id); await finish();
  assert.equal(fileParts(state.draft).length, 2);
  const waiting = queue.jobs[0]; queue.retry(waiting.id);
  const load = current().onload, reader = current();
  queue.remove(waiting.id); reader.result = reader.file.buffer; load(); await tick();
  assert.equal(fileParts(state.draft).length, 2);
});
await check("switch A -> B -> A rejects chooser and old reads", async () => {
  const {state, queue} = fixture(), ticket = queue.ticket();
  queue.add([file("old")]);
  const old = current(), late = old.onload;
  state.id = "b"; queue.reset();
  state.id = "a"; queue.reset();
  queue.add([file("stale chooser")], ticket);
  old.result = old.file.buffer; late(); await tick();
  assert.equal(state.writes.length, 0);
  queue.add([file("new")]); await finish();
  assert.equal(fileParts(state.draft).length, 1);
  assert.equal(attachmentPayload(fileParts(state.draft)[0]).name, "new");
});
await check("message and serialized UTF-8 draft limits are checked before commit", async () => {
  const {state, queue} = fixture({parts: [{type: "text", text: "x".repeat(100000)}]});
  const binary = file("encoded", new Uint8Array(60000));
  queue.add([binary, file("small")]); await finish(); await finish();
  assert.equal(fileParts(state.draft).length, 1);
  assert.match(queue.jobs[0].error, /160,000/);
  state.draft = {parts: []}; queue.retry(queue.jobs[0].id); await finish();
  assert.equal(fileParts(state.draft).length, 1);
  const other = fixture({parts: [{type: "text", text: "\0".repeat(40000) + "中".repeat(50000)}]});
  assert(!draftError(other.state.draft));
  other.queue.add([file("chinese", "文".repeat(45000))]); await finish();
  assert.equal(other.state.writes.length, 0);
  assert.match(other.queue.jobs[0].error, /持久草稿/);
});
await check("file count is rechecked; later files are not read when full", async () => {
  const input = file("seed");
  const initial = {parts: Array.from({length: 31}, (_, i) => createAttachment(file(`file-${i}`), input.buffer))};
  const {state, queue} = fixture(initial);
  queue.add([file("last"), file("extra")]); await finish();
  assert.equal(fileParts(state.draft).length, 32);
  assert.equal(Reader.instances.length, 1);
  assert.match(queue.jobs[0].error, /32/);
});
await check("serialized drafts and token removal preserve unrelated draft positions", () => {
  const input = file("a"), part = createAttachment(input, input.buffer);
  const draft = {parts: [{type: "text", text: "abc"}, part,
    {type: "context", id: "unrelated", label: "editor", text: "context"}, {type: "text", text: "def"}],
    selection: {start: 4, end: 8}};
  const restored = JSON.parse(JSON.stringify(draft));
  assert.deepEqual(restored, draft);
  const removed = removeAttachment(restored, part.id);
  assert.deepEqual(removed.selection, {start: 3, end: 7});
  assert.equal(removed.parts[1].id, "unrelated");
  const inserted = insertText(draft, "paste");
  assert.equal(inserted.parts[1].id, part.id);
  assert.equal(inserted.parts.at(-1).text, "paste");
});
await check("preview only recognizes raster bytes and queue remains bounded", () => {
  assert.equal(rasterBlob({encoding: "base64", content: btoa("<svg/>"), bytes: 6}), null);
  const bytes = [137,80,78,71,13,10,26,10,0];
  assert.equal(rasterBlob({encoding: "base64", content: Buffer.from(bytes).toString("base64"), bytes: bytes.length}).type, "image/png");
  const {queue} = fixture();
  queue.add(Array.from({length: 100}, (_, i) => file(`${i}`)));
  assert.equal(queue.jobs.length, LIMITS.queue);
  assert.match(queue.notice, /36/);
  queue.dispose();
});
console.log(`${checks} attachment behavior checks passed.`);
