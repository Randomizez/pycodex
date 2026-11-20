// Run manually with Playwright, markdown-it and DOMPurify installed; see tests/TESTS.md.
const assert = require("node:assert/strict");
const path = require("node:path");
const {openWorkspaceFixture} = require("./workspace_browser_assets.cjs");
const {chromium} = require("playwright");

(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_EXECUTABLE});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await openWorkspaceFixture(page, '<div id="content"></div>');
    await page.addScriptTag({path: require.resolve("markdown-it/dist/markdown-it.min.js")});
    await page.addScriptTag({path: path.join(path.dirname(require.resolve("dompurify")), "purify.min.js")});
    await page.evaluate(async () => {
      const {createMarkdownRenderer, createViewLinkHandlers, protectMath} = await import("/assets/markdown.mjs");
      const {renderMarkdownInto} = createMarkdownRenderer(document.getElementById("content"));
      const opened = [];
      const {followMarkdownLink} = createViewLinkHandlers({
        getActiveSessionId: () => "session-a",
        openView: async (...args) => window.linkFixture.openView(...args),
        showToast(message) {throw new Error(message);},
      });
      document.addEventListener("click", followMarkdownLink);
      window.linkFixture = {
        opened,
        openView: async (href, srcPath, sessionId) => opened.push({href, srcPath, sessionId}),
        protectMath,
        render(text, srcPath) {
          opened.length = 0;
          const root = document.getElementById("content");
          root.className = srcPath ? "snapshot-markdown" : "";
          if (srcPath) root.setAttribute("data-src-path", srcPath);
          else root.removeAttribute("data-src-path");
          renderMarkdownInto(root, text);
        },
      };
    });
    const render = text => page.evaluate(value => linkFixture.render(value), text);
    const anchor = page.locator("#content a");
    for (const text of [
      "Ordinary text. ".repeat(15000),
      String.raw`Escaped \$price and \(unclosed math`,
      "`$inline code$`\n```python\nvalue = '$fenced code$'\n```",
    ]) {
      const result = await page.evaluate(text => linkFixture.protectMath(text), text);
      assert.equal(result.source, text);
      assert.deepEqual(result.expressions, []);
    }
    const formulas = await page.evaluate(() => linkFixture.protectMath(
      String.raw`Before $a_b$ then $$c_d$$ and \(e_f\) plus \[g_h\] after.`));
    assert.deepEqual(formulas.expressions.map(item => item.body), ["a_b", "c_d", "e_f", "g_h"]);
    assert.deepEqual(formulas.expressions.map(item => item.delimiter.display), [false, true, false, true]);
    assert.equal(formulas.source, "Before \uE000math0\uE001 then \uE000math1\uE001 and \uE000math2\uE001 plus \uE000math3\uE001 after.");
    const workspaceLinks = [
      "TODO.md", "./TODO.md", "/data/project/TODO.md", "TODO.md#L2", "TODO.md:2",
      "/data/project/index.html#leo-benchmarks", "report.htm#results",
      "git:HEAD:TODO.md", "diff:HEAD:TODO.md",
      "diff:/data/project/old.py../data/project/new.py#R2",
    ];
    for (const [destination, href] of [
      ...workspaceLinks.map(href => [href, href]),
      ["file:///data/pycodex/tests/test_agent.py", "/data/pycodex/tests/test_agent.py"],
      ["file:///data/pycodex/pycodex/agent.py#L411-L416", "/data/pycodex/pycodex/agent.py#L411"],
      ["file:///data/a%20%23b.py#L2", "/data/a%20%23b.py#L2"],
      ["file:///C:/project/file.py", "C:/project/file.py"],
      ["file:///data/report.html#leo-benchmarks", "/data/report.html#leo-benchmarks"],
    ]) {
      await render(`[TODO](${destination})`);
      assert.equal(await anchor.count(), 1, destination);
      assert.equal(await anchor.getAttribute("href"), href);
      assert.equal(await anchor.getAttribute("data-workspace-link"), href);
      assert.equal(await anchor.getAttribute("target"), null);
      await anchor.click();
      assert.deepEqual(await page.evaluate(() => linkFixture.opened), [
        {href, srcPath: undefined, sessionId: "session-a"},
      ]);
      assert.equal(page.url(), "http://workspace.test/");
    }
    await page.evaluate(() => linkFixture.render("[TODO](TODO.md)", "/project/docs/index.md"));
    await anchor.click();
    assert.deepEqual(await page.evaluate(() => linkFixture.opened), [
      {href: "TODO.md", srcPath: "/project/docs/index.md", sessionId: "session-a"},
    ]);
    for (const text of ["TODO.md", "README.md", "src/TODO.md"]) {
      await render(text);
      assert.equal(await anchor.count(), 0, text + " must remain plain text");
      assert.equal((await page.locator("#content").textContent()).trim(), text);
    }
    for (const [text, href] of [
      ["https://example.com", "https://example.com"],
      ["[report](https://example.com/report.html#section)", "https://example.com/report.html#section"],
      ["[website](http://TODO.md)", "http://TODO.md"],
      ["user@example.com", "mailto:user@example.com"],
    ]) {
      await render(text);
      assert.equal(await anchor.getAttribute("href"), href);
      assert.equal(await anchor.getAttribute("data-workspace-link"), null);
      assert.equal(await anchor.getAttribute("target"), "_blank");
    }
    for (const href of [
      "file://server/share", "file:////server/share", "file:///%5Cserver/share",
      "javascript:alert(1)", "data:text/html,test",
    ]) {
      await render(`[link](${href})`);
      assert.equal(await anchor.count(), 0, href);
    }
    const example = "[source](file:///data/app.py)";
    await render("`" + example + "`");
    assert.equal(await page.locator("code").textContent(), example);
    assert.equal(await anchor.count(), 0);
    await render("![image](file:///data/image.png)");
    assert.equal(await page.locator("#content img[src], #content a").count(), 0);

    const report = "/data/project/index.html";
    await page.evaluate(async report => {
      const {ContentTabs, SnapshotDisplay} = await import("/assets/content_views.mjs");
      const parent = document.body.appendChild(document.createElement("div"));
      const display = new SnapshotDisplay({content: parent});
      const snapshot = {id: "report", kind: "file", revision: 1, selection: null, documents: [{label: "index.html", text:
        '<div style="height:1500px"></div><section id="leo-benchmarks">Leo</section><div style="height:800px"></div>',
      }]};
      let revision = 0;
      const tabs = new ContentTabs({
        async request(path, options) {
          if (path === "api/views") return {views: [snapshot]};
          const body = JSON.parse(options.body);
          linkFixture.opened.push(body);
          if (path !== "api/views/open" || options.method !== "POST" || body.href !== report) {
            throw new Error("Expected a request to open the report file");
          }
          return {view: {...snapshot, reveal_revision: ++revision}, views: [snapshot]};
        },
        onChange(state) {
          if (state.snapshot) display.show(state.activeId, state.snapshot, state.selection, state.displayMode);
        },
        onSelect() {},
      });
      linkFixture.openView = (...args) => tabs.open(...args);
      linkFixture.refreshReport = () => tabs.poll();
    }, report);
    await page.evaluate(report =>
      linkFixture.render(`[Leo](${report}#leo-benchmarks)`, "/notes/index.md"), report);
    await anchor.click();
    const sectionAtTop = () => {
      const frame = document.querySelector(".snapshot-html");
      const target = frame?.contentDocument?.getElementById("leo-benchmarks");
      return target && Math.abs(target.getBoundingClientRect().top) < 2;
    };
    await page.waitForFunction(sectionAtTop);
    const body = {href: report, src_path: "/notes/index.md", session_id: "session-a"};
    assert.deepEqual(await page.evaluate(() => linkFixture.opened), [body]);
    await page.evaluate(async () => {
      document.querySelector(".snapshot-html").contentWindow.scrollTo(0, 100);
      await linkFixture.refreshReport();
    });
    assert.equal(await page.locator(".snapshot-html").evaluate(frame => frame.contentWindow.scrollY), 100);
    await anchor.click();
    await page.waitForFunction(sectionAtTop);
    assert.deepEqual(await page.evaluate(() => linkFixture.opened), [body, body]);
    assert.equal(page.context().pages().length, 1);
    assert.deepEqual(errors, []);
    console.log("Workspace Markdown math, file links, clicks and explicit URLs passed");
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
