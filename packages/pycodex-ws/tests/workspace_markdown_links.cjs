// Run manually with Playwright, markdown-it and DOMPurify installed; see tests/TESTS.md.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require("playwright");

(async () => {
  const directory = path.join(__dirname, "../workspace_server");
  const source = fs.readFileSync(path.join(directory, "workspace.html"), "utf8");
  const controller = fs.readFileSync(path.join(directory, "frontend/content_views.mjs"), "utf8");
  const recognition = controller.slice(controller.indexOf("export function isWorkspaceLink("),
    controller.indexOf("export class ContentTabs")).replace("export ", "");
  const renderer = source.slice(source.indexOf("    const markdownRenderer ="),
    source.indexOf("    const mathDelimiters ="));
  const rendering = source.slice(source.indexOf("    function renderMarkdownInto("),
    source.indexOf("    async function renderMermaidInto("));
  const clicks = source.slice(source.indexOf("    function followViewLink("),
    source.indexOf("    function followCanvasLink("));
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.setContent('<div id="content"></div>');
    await page.addScriptTag({path: require.resolve("markdown-it/dist/markdown-it.min.js")});
    await page.addScriptTag({path: path.join(path.dirname(require.resolve("dompurify")), "purify.min.js")});
    await page.addScriptTag({content: `(() => {
      ${recognition}
      ${renderer}
      const protectMath = source => ({source, expressions: []});
      const renderProtectedMath = () => {};
      const renderMermaidInto = () => {};
      ${rendering}
      const activeSessionId = "session-a";
      const opened = [];
      const contentTabs = {
        open: async (href, srcPath, sessionId) => opened.push({href, srcPath, sessionId}),
      };
      const showToast = message => {throw new Error(message);};
      ${clicks}
      document.addEventListener("click", followMarkdownLink);
      window.linkFixture = {
        opened,
        render(text, srcPath) {
          opened.length = 0;
          const root = document.getElementById("content");
          root.className = srcPath ? "snapshot-markdown" : "";
          if (srcPath) root.setAttribute("data-src-path", srcPath);
          else root.removeAttribute("data-src-path");
          renderMarkdownInto(root, text);
        },
      };
    })();`});
    const render = text => page.evaluate(value => linkFixture.render(value), text);
    const anchor = page.locator("#content a");
    for (const href of [
      "TODO.md", "./TODO.md", "/data/project/TODO.md", "TODO.md#L2", "TODO.md:2",
      "git:HEAD:TODO.md", "diff:HEAD:TODO.md",
      "diff:/data/project/old.py../data/project/new.py#R2",
    ]) {
      await render(`[TODO](${href})`);
      assert.equal(await anchor.getAttribute("href"), href);
      assert.equal(await anchor.getAttribute("data-workspace-link"), href);
      assert.equal(await anchor.getAttribute("target"), null);
      await anchor.click();
      assert.deepEqual(await page.evaluate(() => linkFixture.opened), [
        {href, srcPath: undefined, sessionId: "session-a"},
      ]);
      assert.equal(page.url(), "about:blank");
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
      ["[website](http://TODO.md)", "http://TODO.md"],
      ["user@example.com", "mailto:user@example.com"],
    ]) {
      await render(text);
      assert.equal(await anchor.getAttribute("href"), href);
      assert.equal(await anchor.getAttribute("data-workspace-link"), null);
      assert.equal(await anchor.getAttribute("target"), "_blank");
    }
    await render("[unsafe](javascript:alert(1))");
    assert.equal(await anchor.count(), 0);
    assert.deepEqual(errors, []);
    console.log("Workspace Markdown file links, clicks and explicit URLs passed");
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
