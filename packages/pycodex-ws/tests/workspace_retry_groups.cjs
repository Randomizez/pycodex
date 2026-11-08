const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require("playwright");

const source = fs.readFileSync(
  path.join(__dirname, "../workspace_server/workspace.html"), "utf8");
const helpers = source.slice(source.indexOf("    function element("),
  source.indexOf("    function renderMarkdownInto("));
const controller = source.slice(source.indexOf("    function renderConversation("),
  source.indexOf("    async function markVisibleFinalRead("));
const style = source.match(/<style>([\s\S]*?)<\/style>/)[1];

(async () => {
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage({
      viewport: {width: 1100, height: 850}, hasTouch: true,
    });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.setContent(`<style>${style}
      body {display:block;height:auto;overflow:auto}
      .log {height:800px;width:100%;box-sizing:border-box}
      </style><div id="log" class="log"></div>`);
    await page.addScriptTag({content: `
      const log = document.getElementById("log");
      let activeSessionId = "a";
      const isNearBottom = () => false;
      const scrollToBottom = () => {};
      const renderMarkdownInto = (body, text) => {body.textContent = text;};
      ${helpers}
      ${controller}
      const retry = (n, limit = "∞") => "Reconnecting... " + n + "/" + limit +
        "\\nresponses request failed with status 502: <html>" +
        "<img src=x onerror='window.retryXss=true'></html>";
      const turn = {turn_id:"turn-a", prompt:"hello", model_errors:[], messages:[
        {message_id:"input-a", name:"user", text:"hello", created_at:null},
      ]};
      window.retryFixture = {
        render(count, limit = "∞") {
          turn.model_errors = Array.from({length:count}, (_, i) => retry(i + 1, limit));
          renderConversation([turn], null);
        },
        mixed() {
          turn.model_errors = [retry(1, "5"), retry(2, "5"),
            "Falling back: primary -> secondary\\nservice unavailable", retry(1)];
          turn.error = "Final request failure";
          renderConversation([turn], null);
        },
        switchTo(id) {activeSessionId = id; renderConversation([turn], null);},
        reload() {log.replaceChildren(); renderConversation([turn], null);}
      };
    `});
    await page.evaluate(() => retryFixture.render(4));
    const groups = page.locator(".model-retry-group");
    assert.equal(await groups.count(), 1);
    assert.equal(await groups.first().getAttribute("open"), null);
    assert.equal(await groups.first().locator("summary").textContent(),
      "Model retries · 4 recorded · latest 4/∞ · HTTP 502");
    assert.equal(await page.locator(".model-retry-log").isVisible(), false);
    await groups.first().locator("summary").focus();
    await page.keyboard.press("Enter");
    assert.equal(await groups.first().getAttribute("open"), "");
    await page.evaluate(() => {
      window.originalGroup = document.querySelector(".model-retry-group");
      retryFixture.render(1000);
    });
    assert.equal(await page.evaluate(() =>
      originalGroup === document.querySelector(".model-retry-group")), true);
    assert.equal(await groups.first().getAttribute("open"), "");
    assert.match(await groups.first().locator("summary").textContent(), /1000 recorded.*1000\/∞/);
    assert.equal(await page.locator(".entry.model-error").count(), 1);
    assert.equal(await page.locator(".model-retry-log").evaluate(node =>
      node.scrollHeight > node.clientHeight && node.clientHeight <= 280), true);
    assert.equal(await page.locator(".model-retry-group img").count(), 0);
    assert.equal(await page.evaluate(() => window.retryXss), undefined);
    assert.match(await page.locator(".model-retry-log").textContent(), /Reconnecting\.\.\. 1000\/∞/);
    await groups.first().locator("summary").click();
    await page.evaluate(() => retryFixture.render(1001));
    assert.equal(await groups.first().getAttribute("open"), null);
    await page.evaluate(() => retryFixture.mixed());
    assert.equal(await groups.count(), 2);
    assert.equal(await page.locator(".entry.model-error").count(), 3);
    assert.equal(await page.getByText("Falling back: primary -> secondary", {exact:false}).isVisible(), true);
    assert.equal(await page.getByText("Final request failure", {exact:true}).isVisible(), true);
    assert.match(await groups.first().locator("summary").textContent(), /latest 2\/5/);
    await groups.first().locator("summary").click();
    await page.evaluate(() => retryFixture.switchTo("b"));
    assert.equal(await groups.first().getAttribute("open"), null);
    assert.equal(await page.locator("[data-entry-key^='a:']").count(), 0);
    await groups.first().locator("summary").click();
    await page.evaluate(() => retryFixture.reload());
    assert.equal(await groups.first().getAttribute("open"), null);
    await page.setViewportSize({width: 390, height: 844});
    await page.evaluate(() => retryFixture.render(1000));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    await groups.first().locator("summary").tap();
    assert.equal(await groups.first().getAttribute("open"), "");
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
    assert.deepEqual(errors, []);
    console.log("Retry groups: live append, diagnostics, keyboard, mobile, isolation and escaping passed");
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exit(1);});
