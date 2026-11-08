// Run manually with Playwright installed; see tests/TESTS.md.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require("playwright");

const directory = path.join(__dirname, "../workspace_server");
const source = fs.readFileSync(path.join(directory, "workspace.html"), "utf8");
const html = source.replace(/__([A-Z_]+)__/g, (_, name) => ({
  WORKSPACE_TITLE: "Session cache test", ENABLE_BOARD: "true",
  ACCESS_ROLE: "admin", BOARD_URL: "about:blank", CWD_LABEL: "/test",
}[name] || "")).replace("</body>", `<script type="module">
  window.fixtureReady = true;
</script></body>`).replace("    startContentViews();", `
    window.fixture = {
      switchSession, createSession, closeSession, pollSession, restoreSessionScroll,
      renderSnapshot, updateSessions, sessionState, composer, renderMermaidInto, setSessionsOpen,
      active: () => activeSessionId,
      stop: () => {clearInterval(pollTimer); clearInterval(boardPollTimer); stopContentViews();},
    };
    startContentViews();`);

function turn(id, text) {
  const prompt = "Question " + id;
  return {turn_id: id, prompt, response: text, model_errors: [], messages: [
    {message_id: "input-" + id, name: "user", text: prompt, created_at: null},
  ]};
}

(async () => {
  const browser = await chromium.launch({headless: true});
  try {
    const errors = [];
    const page = await browser.newPage({viewport: {width: 1100, height: 850}});
    page.on("pageerror", error => errors.push(error.message));
    let tabs = ["a", "b"];
    let savedSession = "";
    const snapshots = new Map([
      ["a", {cwd: "/a", queued_inputs: [], spinner: "", turns:
        Array.from({length: 80}, (_, i) => turn("a" + i, "Long answer\n".repeat(8)))}],
      ["b", {cwd: "/b", queued_inputs: [], spinner: "", turns: [turn("b0", "Short answer")]}],
    ]);
    snapshots.get("a").turns[79].model_errors = ["Reconnecting... 1/5\nExample diagnostic"];
    const requests = [];
    const blocked = new Map();
    const list = () => tabs.map(id => ({id, title: id, cwd: "/" + id, running: false}));
    function block(key) {
      let release;
      const ready = new Promise(resolve => {release = resolve;});
      blocked.set(key, ready);
      return () => {blocked.delete(key); release();};
    }
    await page.route("**/*", async route => {
      const url = new URL(route.request().url());
      if (url.hostname !== "workspace.test") {
        // Offline dependency doubles: ordinary messages use the page's text
        // fallback; Mermaid is controlled to exercise its asynchronous lifetime.
        if (url.pathname.includes("mermaid.esm")) {
          return route.fulfill({contentType: "text/javascript", body: `
            export default {
              initialize() {},
              async render(id) {
                if (window.holdDiagram) await new Promise(resolve => {window.finishDiagram = resolve;});
                return {svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40"></svg>'};
              },
            };
          `});
        }
        return route.fulfill({body: "", contentType:
          url.pathname.endsWith(".css") ? "text/css" : "text/javascript"});
      }
      if (url.pathname === "/") return route.fulfill({
        body: html.replace("<head>", `<head><script>
          sessionStorage.setItem("pycodex.workspace.activeSession:/", ${JSON.stringify(savedSession)});
        </script>`),
        contentType: "text/html",
      });
      if (url.pathname.startsWith("/assets/")) {
        const name = url.pathname.slice("/assets/".length);
        return route.fulfill({
          body: fs.readFileSync(path.join(directory, name)),
          contentType: name.endsWith(".css") ? "text/css" : "text/javascript",
        });
      }
      if (url.pathname === "/api/views") return route.fulfill({json: {views: []}});
      if (url.pathname === "/api/board") return route.fulfill({json: {exists: false}});
      if (url.pathname === "/api/session") {
        const id = url.searchParams.get("session_id") || tabs[0];
        requests.push(id);
        if (blocked.has(id)) await blocked.get(id);
        if (id === "failed") return route.fulfill({status: 503, json: {}});
        if (!tabs.includes(id)) return route.fulfill({status: 404, json: {}});
        return route.fulfill({json: {session_id: id, sessions: list(), snapshot: snapshots.get(id)}});
      }
      if (url.pathname === "/api/sessions" && route.request().method() === "POST") {
        if (blocked.has("create")) await blocked.get("create");
        tabs.push("c");
        snapshots.set("c", {cwd: "/c", turns: [], spinner: "", queued_inputs: []});
        return route.fulfill({json: {
          ok: true, session_id: "c", sessions: list(), snapshot: snapshots.get("c"),
        }});
      }
      if (url.pathname.startsWith("/api/sessions/") && route.request().method() === "DELETE") {
        tabs = tabs.filter(id => id !== url.pathname.split("/").pop());
        return route.fulfill({json: {ok: true, sessions: list()}});
      }
      throw new Error("Unexpected fixture request: " + route.request().method() + " " + url.pathname);
    });
    async function open(saved, expected) {
      savedSession = saved;
      requests.length = 0;
      await page.goto("http://workspace.test/");
      await page.waitForFunction("window.fixtureReady");
      if (expected) {
        await page.waitForFunction(id => fixture.active() === id &&
          fixture.sessionState.get(id).lastRenderedSignature, expected);
      } else {
        await page.waitForFunction("document.getElementById('spinner').textContent.includes('503')");
      }
      await page.evaluate("fixture.stop()");
    }
    await open("b", "b");
    assert.deepEqual(requests, ["b"], "saved session should be the first and only request");
    await open("missing", "a");
    assert.deepEqual(requests, ["missing", "a"], "deleted saved session should select the default");
    await open("failed", null);
    assert.deepEqual(requests, ["failed"], "server errors must not silently change sessions");
    await open("a", "a");

    await page.evaluate(() => {
      const log = document.getElementById("log");
      window.originalEntry = log.firstElementChild;
      document.querySelector(".model-retry-group").open = true;
      log.scrollTop = 123;
      fixture.composer.set({parts: [{type: "text", text: "draft a"}]});
    });
    await page.evaluate("fixture.switchSession('b')");
    const releaseA = block("a");
    await page.evaluate(() => {window.selection = fixture.switchSession("a");});
    assert.equal(await page.evaluate(() => document.getElementById("log").firstElementChild === originalEntry), true);
    assert.equal(await page.locator(".model-retry-group").getAttribute("open"), "");
    assert.equal(await page.evaluate("document.getElementById('log').scrollTop"), 123);
    assert.equal(await page.evaluate("fixture.composer.read().parts[0].text"), "draft a");
    snapshots.get("a").turns[79].response = "Updated while inactive";
    releaseA();
    await page.evaluate("window.selection");
    assert.equal(await page.locator(".response").last().textContent(), "Updated while inactive");
    assert.equal(await page.evaluate(() => document.getElementById("log").firstElementChild === originalEntry), true);
    assert.equal(await page.locator(".model-retry-group").getAttribute("open"), "");

    await page.setViewportSize({width: 390, height: 844});
    await page.evaluate(() => {
      document.querySelector(".workspace").dataset.mobilePane = "chat";
      document.getElementById("log").scrollTop = 123;
    });
    await page.evaluate("fixture.switchSession('b')");
    await page.evaluate(() => {document.querySelector(".workspace").dataset.mobilePane = "board";});
    await page.evaluate("fixture.switchSession('a')");
    await page.evaluate(() => {
      document.querySelector(".workspace").dataset.mobilePane = "chat";
      fixture.restoreSessionScroll();
    });
    assert.equal(await page.evaluate("document.getElementById('log').scrollTop"), 123);

    const releaseCreate = block("create");
    await page.evaluate(() => {window.creation = fixture.createSession();});
    snapshots.get("a").turns.push(turn("a80", "Arrived during creation"));
    await page.evaluate("fixture.pollSession()");
    releaseCreate();
    await page.evaluate("window.creation");
    await page.evaluate("fixture.switchSession('a')");
    assert.equal(await page.locator(".response").last().textContent(), "Arrived during creation");
    await page.evaluate("fixture.closeSession('b')");
    assert.equal(await page.evaluate("fixture.sessionState.has('b')"), false);

    await page.evaluate(() => {
      window.DOMPurify = {sanitize: source => source};
      window.holdDiagram = true;
      window.diagramRoot = document.querySelector(".response .text");
      diagramRoot.innerHTML = "<pre><code class='language-mermaid'>graph TD; A-->B</code></pre>";
      window.diagramDone = fixture.renderMermaidInto(diagramRoot);
    });
    await page.waitForFunction("typeof window.finishDiagram === 'function'");
    await page.evaluate("fixture.switchSession('c')");
    await page.evaluate(() => {
      window.diagramScroll = document.getElementById("log").scrollTop;
      window.finishDiagram();
    });
    await page.evaluate("window.diagramDone");
    assert.equal(await page.evaluate("document.getElementById('log').scrollTop === diagramScroll"), true);
    await page.evaluate("fixture.switchSession('a')");
    assert.equal(await page.locator(".mermaid-diagram svg").count(), 1);
    await page.evaluate("fixture.closeSession('a')");
    assert.equal(await page.evaluate("fixture.active()"), "c");
    assert.equal(await page.evaluate("fixture.sessionState.has('a')"), false);

    await page.evaluate(() => {
      window.messageSnapshot = {model: "test", queued_inputs: [], spinner: "", turns: [{
        turn_id: "message-turn", submission_id: "first", prompt: "same\nsame",
        messages: [
          {message_id: "alice", name: "Alice", text: "same", created_at: "2026-01-01T09:00:00Z"},
          {message_id: "bob", name: "Bob", text: "same", created_at: "2026-01-01T09:01:00Z"},
        ], response: "Both inputs retained.", model_errors: [],
      }]};
      fixture.renderSnapshot(messageSnapshot);
      window.firstInput = document.querySelector(".entry.user");
    });
    assert.deepEqual(await page.locator(".message-author").allTextContents(), ["Alice", "Bob"]);
    assert.deepEqual(await page.locator(".entry.user > .text").allTextContents(), ["same", "same"]);
    assert.deepEqual(await page.locator(".message-header time").evaluateAll(nodes =>
      nodes.map(node => node.dateTime)), ["2026-01-01T09:00:00Z", "2026-01-01T09:01:00Z"]);
    await page.evaluate(() => {
      const turn = messageSnapshot.turns[0];
      turn.submission_id = "steer";
      turn.messages[0].name = "Alice updated";
      turn.messages[0].created_at = "2026-01-01T10:00:00Z";
      turn.messages.push({message_id: "legacy", text: "Unknown time stays empty.",
        name: "<img src=x onerror=alert(1)>", created_at: null});
      fixture.renderSnapshot(messageSnapshot);
    });
    assert.equal(await page.evaluate(() => firstInput === document.querySelector(".entry.user")), true);
    assert.equal(await page.locator(".message-author").first().textContent(), "Alice updated");
    assert.equal(await page.locator(".message-header time").first().getAttribute("datetime"), "2026-01-01T10:00:00Z");
    assert.equal(await page.locator(".entry.user").last().locator("time, img").count(), 0);
    await page.setViewportSize({width: 390, height: 844});
    await page.evaluate(() => {
      document.querySelector('[data-pane="chat"]').click();
      fixture.setSessionsOpen(false);
    });
    await page.locator("#sessionSidebar").waitFor({state: "hidden"});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.equal(await page.locator(".entry.user").evaluateAll(entries => entries.some(entry =>
      entry.scrollWidth > entry.clientWidth)), false);
    await page.evaluate(() => {
      window.temporaryReply = document.querySelector(".entry.response");
      Object.assign(messageSnapshot.turns[0], {
        prompt_at: "2026-01-01T09:00:00Z", response_at: "2026-01-01T09:02:00Z",
      });
    });
    for (const [phase, status, animation] of [
      ["commentary", "tool", "activity-breathe"],
      ["final", "tool", "none"],
      ["commentary", "completed", "none"],
      [null, "running", "activity-breathe"],
      [null, "completed", "none"],
    ]) {
      await page.evaluate(([phase, status]) => {
        Object.assign(messageSnapshot.turns[0], {response_phase: phase, status});
        fixture.renderSnapshot(messageSnapshot);
      }, [phase, status]);
      assert.equal(await page.locator(".entry.response").evaluate(entry =>
        getComputedStyle(entry).animationName), animation);
      assert.equal(await page.locator(".entry.response .message-time").count(),
        animation === "activity-breathe" ? 0 : 1);
      assert.equal(await page.evaluate(() =>
        document.querySelector(".entry.response") === temporaryReply), true);
    }
    assert.deepEqual(errors, []);
    console.log("Session cache, message metadata, temporary reply animation and mobile layout passed.");
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
