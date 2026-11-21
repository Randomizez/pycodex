// Run after applying workspace.patch; all sessions and network replies are synthetic.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require("playwright");

async function main() {
  const feature = path.resolve(__dirname, "..");
  const workspace = process.env.PYCODEX_NOTIFICATION_WORKSPACE || path.resolve(feature, "../..");
  const source = fs.readFileSync(path.join(workspace, "workspace.template.html"), "utf8");
  assert(fs.readFileSync(path.join(workspace, "conversation.mjs"), "utf8").includes("function renderNotification(record)"), "Apply workspace.patch first");
  const html = source.replace(/__([A-Z_]+)__/g, (_, name) => ({
    WORKSPACE_TITLE: "Speaker notifications", ENABLE_BOARD: "false",
    ACCESS_ROLE: "admin", BOARD_URL: "about:blank", CWD_LABEL: "/example",
  }[name] || ""));
  const entry = fs.readFileSync(path.join(workspace, "workspace.mjs"), "utf8") +
    "\nwindow.notificationFixture = {selectSession, renderSnapshot};";
  const sessions = ["a", "b"].map(id => ({
    id, title: id === "a" ? "页面检查" : "文档整理", cwd: "/example", running: false,
  }));
  const records = [
    {id: "sent", turn_id: "turn", state: "accepted",
      speech: "页面检查已通过，请确认显示效果。", created: 1794919200},
    {id: "quiet", turn_id: "turn", state: "quiet_hours",
      speech: "文档已整理完成，你暂时无需操作。",
      created: Date.parse("2026-11-18T03:00:00+08:00") / 1000},
  ];
  const snapshot = id => ({
    cwd: "/example", title: sessions.find(s => s.id === id).title,
    turns: [{turn_id: "turn", submission_id: "submission", final_id: "final",
      messages: [{message_id: "request", text: "检查页面和文档。", name: ""}],
      response: "页面和文档已检查。", status: "completed", kind: "turn", model_errors: []}],
    running: false, spinner: "", input_request: null, queued_inputs: [],
    notifications: id === "a" ? records : [],
  });
  const browser = await chromium.launch({
    headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
    args: ["--no-sandbox"],
  });
  try {
    const context = await browser.newContext({viewport: {width: 1100, height: 800}});
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    if (context.routeWebSocket) await context.routeWebSocket("**/*", socket => socket.close());
    await page.route("**/*", route => {
      const url = new URL(route.request().url());
      if (url.hostname !== "speaker.test") return route.fulfill({
        contentType: "text/javascript", body: url.pathname.includes("mermaid")
          ? 'export default {initialize(){},render(){return {svg:""}}};' : "",
      });
      if (url.pathname === "/") return route.fulfill({contentType: "text/html", body: html});
      if (url.pathname.startsWith("/assets/")) return route.fulfill({
        contentType: url.pathname.endsWith(".css") ? "text/css" : "text/javascript",
        body: url.pathname === "/assets/workspace.mjs" ? entry : fs.readFileSync(path.join(workspace, path.basename(url.pathname))),
      });
      if (url.pathname === "/api/session") {
        const id = url.searchParams.get("session_id") || "a";
        return route.fulfill({json: {session_id: id, sessions, snapshot: snapshot(id), views: []}});
      }
      if (url.pathname === "/api/sessions") return route.fulfill({json: {sessions, profiles: []}});
      return route.fulfill({json: {ok: true, sessions, items: [], models: [], state: {}}});
    });
    await page.goto("https://speaker.test/", {waitUntil: "domcontentloaded"});
    await page.locator(".entry.notification").first().waitFor();
    assert.equal(await page.locator(".entry.notification").count(), 2);
    assert.match(await page.locator('[data-state="accepted"]').innerText(), /已发送播报/);
    assert.match(await page.locator('[data-state="quiet_hours"]').innerText(), /免打扰未播报/);
    await page.evaluate(value => {
      notificationFixture.renderSnapshot(value);
      notificationFixture.renderSnapshot(value);
    }, snapshot("a"));
    assert.equal(await page.locator(".entry.notification").count(), 2);
    await page.screenshot({path: path.join(feature, "preview.png")});
    await page.reload({waitUntil: "domcontentloaded"});
    await page.locator(".entry.notification").first().waitFor();
    assert.equal(await page.locator(".entry.notification").count(), 2);
    await page.evaluate(() => notificationFixture.selectSession("b"));
    assert.equal(await page.locator(".entry.notification").count(), 0);
    await page.evaluate(() => notificationFixture.selectSession("a"));
    assert.equal(await page.locator(".entry.notification").count(), 2);
    const updated = snapshot("a");
    updated.notifications = [{...records[0], state: "failed", speech: "<img src=x onerror=alert(1)>"}];
    await page.evaluate(value => notificationFixture.renderSnapshot(value), updated);
    assert.match(await page.locator(".entry.notification").innerText(), /播报失败/);
    assert.equal(await page.locator(".entry.notification img").count(), 0);
    await page.evaluate(value => notificationFixture.renderSnapshot(value), snapshot("a"));
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await page.screenshot({path: path.join(feature, "preview-mobile.png")});
    assert.deepEqual(errors, []);
    console.log("PASS: sent/quiet/failure records, idempotence, refresh, session isolation, text escaping, mobile");
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
