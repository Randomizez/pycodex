"""Manual Chromium DOM checks with the real Composer and upload sources.

Use --output NEW_JUICEFS_DIRECTORY and --chrome EXISTING_BINARY. No screenshots,
external requests, production sessions, downloads or model calls are used.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import websockets

SOURCE = Path(__file__).resolve().parent
WORKSPACE = SOURCE.parents[2]


class CDP:
    def __init__(self, socket):
        self.socket = socket
        self.sequence = 0
        self.errors = []

    async def call(self, method, **params):
        self.sequence += 1
        await self.socket.send(json.dumps({"id": self.sequence, "method": method, "params": params}))
        while True:
            message = json.loads(await asyncio.wait_for(self.socket.recv(), 15))
            if message.get("method") == "Runtime.exceptionThrown":
                details = message["params"]["exceptionDetails"]
                self.errors.append(details.get("exception", {}).get("description", details.get("text")))
            if message.get("id") == self.sequence:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message["result"]

    async def evaluate(self, expression):
        result = await self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")

    async def require(self, expression):
        for _ in range(60):
            if await self.evaluate(expression):
                return
            await asyncio.sleep(0.1)
        diagnostic = await self.evaluate("({url: location.href, ready: window.ready, errors: window.errors})")
        raise AssertionError({"condition": expression, "diagnostic": diagnostic, "page_errors": self.errors})


BRIDGE = r"""
window.errors = [];
window.addEventListener('error', e => errors.push(e.message));
window.addEventListener('unhandledrejection', e => errors.push(String(e.reason)));
const stored = new Map();
const storage = {getItem: key => stored.get(key), setItem: (key, value) => stored.set(key, value), removeItem: key => stored.delete(key)};
const store = {
  save: (key, draft) => storage.setItem(key, JSON.stringify(draft)),
  load: key => {const raw = storage.getItem(key); return raw ? JSON.parse(raw) : null;},
};
window.sent = [];
let session = 'a';
const handlers = new Set();
const form = document.getElementById('composer'), prompt = document.getElementById('prompt');
const save = () => store.save(session, composer.read());
window.composer = new Composer(prompt, save, () => form.requestSubmit(), () => {}, e => errors.push(e));
window.host = {getActiveSessionId: () => session, getDraft: () => composer.read(),
  setDraft: draft => {composer.set(draft); save();}, onSessionChanged: handler => {handlers.add(handler); return () => handlers.delete(handler);}
};
window.selectSession = id => {save(); session = id; composer.set(store.load(id) || {parts: []}); for (const handler of handlers) handler(id);};
window.restore = () => composer.set(store.load(session) || {parts: []});
form.addEventListener('submit', event => {event.preventDefault(); sent.push(composer.read()); composer.set({parts: []}); save();});
const NativeReader = window.FileReader;
const failed = new Set();
window.FileReader = class {
  constructor() {this.readyState = 0; this.reader = new NativeReader();}
  readAsArrayBuffer(file) {
    this.readyState = 1;
    this.reader.onprogress = event => this.onprogress?.(event);
    this.reader.onerror = () => {this.readyState = 2; this.onerror?.();};
    this.reader.onload = () => {
      this.timer = setTimeout(() => {
        this.readyState = 2;
        if (file.name === 'fail-once.txt' && !failed.has(file.name)) {failed.add(file.name); this.onerror?.();}
        else {this.result = this.reader.result; this.onload?.();}
      }, file.name.startsWith('slow') ? 250 : 10);
    };
    this.reader.readAsArrayBuffer(file);
  }
  abort() {clearTimeout(this.timer); if (this.reader.readyState === 1) this.reader.abort(); this.readyState = 2; this.onabort?.();}
};
window.disposeUpload = mount(host);
window.addFiles = files => {
  const input = document.querySelector('input[aria-label="选择附件"]');
  // Set the ticket through the real button; avoid opening an OS dialog in this fixture.
  input.click = () => {};
  document.querySelector('.pycodex-upload-toolbar button').click();
  const data = new DataTransfer(); files.forEach(file => data.items.add(file));
  input.files = data.files; input.dispatchEvent(new Event('change', {bubbles: true}));
};
window.ready = true;
"""

CHECKS = r"""
(async () => {
  const checks = [];
  const require = (value, name) => {if (!value) throw new Error(name); checks.push(name);};
  const wait = async condition => {for (let i=0;i<200;i++) {if (condition()) return; await new Promise(r => setTimeout(r, 10));} throw new Error('DOM wait timed out');};
  const status = () => document.querySelector('.pycodex-upload-status').textContent;
  const cards = state => [...document.querySelectorAll(`.pycodex-attachment[data-state="${state}"]`)];
  const file = (name, content='hello') => new File([content], name, {lastModified:1});
  require(document.querySelectorAll('[data-pycodex-file-upload]').length === 1, 'single_mount');
  host.setDraft({parts:[{type:'text',text:'prefix suffix'}],selection:{start:7,end:7}});
  addFiles([file('slow.txt'), file('oversize',new Uint8Array(262145)), file('safe.txt','<script>danger()</script>')]);
  require(!document.getElementById('composer').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})), 'submit_blocked_while_reading');
  require(sent.length === 0, 'no_early_send');
  document.getElementById('prompt').focus();
  document.getElementById('prompt').dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true,cancelable:true}));
  require(sent.length === 0, 'enter_blocked_while_reading');
  host.setDraft({parts:[{type:'text',text:'typed during reading'}],selection:{start:3,end:5}});
  await wait(() => cards('ready').length === 2);
  require(!status().includes('请等待完成'), 'busy_notice_cleared_on_completion');
  require(host.getDraft().parts[0].text === 'typed during reading', 'typing_preserved');
  require(host.getDraft().selection.start === 3 && host.getDraft().selection.end === 5, 'selection_preserved');
  require(cards('error').length === 1 && status().includes('不会发送'), 'per_file_failure_visible');
  const safe = cards('ready').find(node => node.textContent.includes('safe.txt'));
  safe.querySelector('.pycodex-attachment-open').click();
  require(document.querySelector('dialog').open && document.querySelector('dialog pre').textContent === '<script>danger()</script>', 'text_preview_escaped');
  require(!document.querySelector('dialog script'), 'preview_has_no_injected_markup');
  document.querySelector('button[aria-label="关闭附件预览"]').click();
  addFiles([file('safe.txt','<script>danger()</script>')]);
  await new Promise(r => setTimeout(r,30));
  require(cards('ready').length === 2 && status().includes('重复'), 'dedupe_visible');
  safe.querySelector('.pycodex-attachment-remove').click();
  await wait(() => cards('ready').length === 1);
  require(host.getDraft().selection.start === 3 && host.getDraft().selection.end === 5, 'remove_keeps_text_selection');
  const data = new DataTransfer(); data.items.add(file('paste.txt')); data.setData('text/plain','pasted');
  document.getElementById('prompt').dispatchEvent(new ClipboardEvent('paste',{clipboardData:data,bubbles:true,cancelable:true}));
  await wait(() => cards('ready').length === 2);
  require(host.getDraft().parts.some(part => part.type === 'text' && part.text.includes('pasted')), 'mixed_paste_retains_text');
  addFiles([file('fail-once.txt')]);
  await wait(() => cards('error').some(node => node.textContent.includes('fail-once')));
  cards('error').find(node => node.textContent.includes('fail-once')).querySelector('button[aria-label^="重试"]').click();
  await wait(() => cards('ready').length === 3);
  require(true, 'retry_recovers_one_file');
  addFiles([file('slow-cancel.txt')]);
  document.querySelectorAll('.pycodex-upload-toolbar button')[2].click();
  await wait(() => status().includes('已取消'));
  await new Promise(r => setTimeout(r,300));
  require(cards('ready').length === 3, 'cancel_does_not_add_file');
  restore(); await wait(() => cards('ready').length === 3);
  require(true, 'draft_restore_updates_cards');
  addFiles([file('slow-switch.txt')]); selectSession('b'); selectSession('a');
  await new Promise(r => setTimeout(r,300));
  require(cards('ready').length === 3 && !host.getDraft().parts.some(part => part.label?.includes('slow-switch')), 'session_switch_discards_late_file');
  const input = document.querySelector('input[aria-label="选择附件"]');
  document.querySelector('.pycodex-upload-toolbar button').click(); selectSession('b'); selectSession('a');
  const stale = new DataTransfer(); stale.items.add(file('stale-picker'));
  input.files = stale.files; input.dispatchEvent(new Event('change',{bubbles:true}));
  await new Promise(r => setTimeout(r,30));
  require(status().includes('会话已切换') && cards('ready').length === 3, 'stale_picker_rejected');
  let blobs = 0, revoked = 0;
  const create = URL.createObjectURL.bind(URL), revoke = URL.revokeObjectURL.bind(URL);
  URL.createObjectURL = blob => {blobs++; return create(blob);};
  URL.revokeObjectURL = url => {revoked++; revoke(url);};
  const png = Uint8Array.from(atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aOt0AAAAASUVORK5CYII='),c=>c.charCodeAt(0));
  addFiles([new File([png], 'tiny.png', {type:'image/png',lastModified:1})]);
  await wait(() => cards('ready').some(node => node.querySelector('img')));
  const image = cards('ready').find(node => node.querySelector('img'));
  await wait(() => image.querySelector('img').complete && image.querySelector('img').naturalWidth === 1);
  require(blobs === 1, 'raster_thumbnail_decodes');
  image.querySelector('.pycodex-attachment-open').click();
  require(document.querySelector('dialog img'), 'image_preview_opens');
  document.querySelector('button[aria-label="关闭附件预览"]').click();
  image.querySelector('.pycodex-attachment-remove').click();
  await wait(() => revoked === 1);
  require(true, 'removed_preview_url_released');
  const drop = new DataTransfer(); drop.items.add(file('drop.txt'));
  document.getElementById('prompt').dispatchEvent(new DragEvent('drop',{dataTransfer:drop,bubbles:true,cancelable:true}));
  await wait(() => cards('ready').length === 4);
  require(true, 'file_drop_captured');
  const editable = host.getDraft().parts.find(part => part.fileAttachment);
  const token = [...document.querySelectorAll('[data-context]')].find(node => JSON.parse(node.dataset.context).id === editable.id);
  token.remove(); await wait(() => cards('ready').length === 3);
  require(true, 'native_token_deletion_updates_cards');
  document.getElementById('composer').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));
  await wait(() => cards('ready').length === 0);
  require(sent.length === 1 && sent[0].parts.filter(part => part.fileAttachment).length === 3, 'completed_attachments_sent_once');
  require(errors.length === 0, 'no_browser_errors');
  return {checks, requests:'local fixture only', screenshots:0};
})()
"""


def document():
    core = (SOURCE / "upload_core.mjs").read_text().replace("export ", "")
    ui = re.sub(r"^import .*?;\n", "", (SOURCE / "uploads.mjs").read_text(), flags=re.S).replace("export ", "")
    entry = re.sub(r"^import .*?;\n", "", (SOURCE / "custom.js").read_text(), flags=re.S).replace("export ", "")
    composer = (WORKSPACE / "frontend/composer.mjs").read_text().replace("export ", "")
    shell = (WORKSPACE / "workspace.html").read_text()
    css = re.search(r"<style>(.*?)</style>", shell, re.S).group(1)
    # Separate the real sources' lexical scopes, as modules do in production.
    scripts = "window.Composer=(()=>{" + composer + ";return Composer;})();\n"
    names = "AttachmentQueue,attachmentPayload,draftError,draftStats,fileParts,formatBytes,insertText,rasterBlob,removeAttachment"
    scripts += "window.uploadCore=(()=>{" + core + ";return {" + names + "};})();\n"
    scripts += "window.mountUploads=(()=>{const {" + names + "}=uploadCore;" + ui + ";return mountUploads;})();\n"
    scripts += "window.mount=(()=>{" + entry + ";return mount;})();\n" + BRIDGE
    return ("<!doctype html><html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        "<style>" + css + (SOURCE / "custom.css").read_text() + "</style><body>"
        "<main style='width:min(900px,100%);margin:auto'><form id='composer'><div class='composer-surface'>"
        "<div id='contextControl'><div id='contextDetails' hidden><div class='context-details-card'></div></div></div>"
        "<div class='composer-box'><div class='prompt-field'><div id='prompt' class='prompt-input empty' contenteditable='true'></div></div></div>"
        "<div id='contextBlockDetail' hidden></div></div></form></main><script>" + scripts.replace("</script", "<\\/script") + "</script></body></html>").encode()


async def run_check(root, server, result):
    for _ in range(150):
        port_file = root / "profile/DevToolsActivePort"
        if port_file.exists():
            break
        await asyncio.sleep(.1)
    port = int(port_file.read_text().splitlines()[0])
    async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
        pages = (await client.get("http://127.0.0.1:%d/json/list" % port)).json()
    endpoint = next(page["webSocketDebuggerUrl"] for page in pages if page["type"] == "page")
    async with websockets.connect(endpoint, max_size=4 * 1024 * 1024) as socket:
        cdp = CDP(socket)
        await cdp.call("Page.enable")
        await cdp.call("Runtime.enable")
        await cdp.call("Network.enable")
        await cdp.call("Network.setBlockedURLs", urls=["https://*", "http://*/*api/*"])
        await cdp.call("Page.navigate", url="http://127.0.0.1:%d/" % server.server_port)
        await cdp.require("window.ready === true")
        checked = await cdp.call("Runtime.evaluate", expression=CHECKS, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in checked:
            raise AssertionError(checked["exceptionDetails"])
        result.update(checked["result"]["value"])
        await cdp.call("Emulation.setDeviceMetricsOverride", width=375, height=812, deviceScaleFactor=1, mobile=True)
        overflow = await cdp.evaluate("document.documentElement.scrollWidth > 375")
        if overflow:
            raise AssertionError("Mobile horizontal overflow")
        result["checks"].append("mobile_no_horizontal_overflow")
        await cdp.evaluate("disposeUpload()")
        await cdp.require("!document.querySelector('[data-pycodex-file-upload]')")
        result["checks"].append("dispose_removes_controls")
        if cdp.errors:
            raise AssertionError(cdp.errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chrome", required=True)
    parser.add_argument("--mount-source", required=True, help="Expected findmnt source of the output JuiceFS mount")
    args = parser.parse_args()
    root = Path(args.output)
    if not root.is_absolute() or root.resolve() != root or root.exists():
        raise ValueError("Use a new canonical output directory")
    mount = json.loads(subprocess.check_output(["findmnt", "-J", "-T", str(root.parent)]))["filesystems"][0]
    stat = os.statvfs(root.parent)
    if (mount["source"], mount["fstype"]) != (args.mount_source, "fuse.juicefs") or stat.f_bavail * stat.f_frsize < 1024**3 or stat.f_favail < 1000 or not os.access(root.parent, os.W_OK | os.X_OK):
        raise ValueError("Writer requires writable JuiceFS with capacity and inodes")
    root.mkdir(mode=0o700)
    content = document()
    (root / "fixture.html").write_bytes(content)
    result = {"mount": mount, "output": str(root)}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = dict(os.environ, HOME=str(root), TMPDIR=str(root), XDG_CACHE_HOME=str(root / "cache"),
        XDG_CONFIG_HOME=str(root / "config"), PYTHONDONTWRITEBYTECODE="1")
    with (root / "chromium.log").open("x") as log:
        browser = subprocess.Popen([args.chrome, "--headless", "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
            "--no-zygote", "--single-process", "--disable-background-networking", "--disable-crash-reporter", "--no-first-run",
            "--remote-debugging-port=0", "--user-data-dir=" + str(root / "profile"),
            "--disk-cache-dir=" + str(root / "cache"), "about:blank"], env=env, stdout=log, stderr=log)
        try:
            asyncio.run(run_check(root, server, result))
            result["status"] = "passed"
        except Exception as error:
            result.update(status="failed", error=type(error).__name__ + ": " + str(error))
            raise
        finally:
            server.shutdown()
            server.server_close()
            browser.terminate()
            try:
                browser.wait(timeout=10)
            except subprocess.TimeoutExpired:
                browser.kill()
                browser.wait(timeout=5)
            (root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
