// Run manually with Playwright installed (see tests/TESTS.md).
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require("playwright");

(async () => {
  const browser = await chromium.launch({executablePath: process.env.CHROMIUM_EXECUTABLE});
  try {
    const page = await browser.newPage();
    const html = fs.readFileSync(path.join(__dirname, "../pycodex_router/index.html"), "utf8");
    const base = "http://router.test/remote/";
    const nodes = ["a", "b"].map(letter => ({
      id: letter.repeat(52), name: letter, alias: "", verified: false,
      url: `/remote/n/${letter.repeat(5)}/`,
    }));
    const approvals = [];
    await page.route("http://router.test/**", async route => {
      const request = route.request();
      const url = request.url();
      if (url === base) {
        return route.fulfill({contentType: "text/html", headers: {"Referrer-Policy": "no-referrer"}, body: html});
      }
      if (url === base + "api/workspaces") return route.fulfill({json: {nodes, workspaces: []}});
      for (const node of nodes) {
        if (url === `${base}api/nodes/${node.id}/verify`) {
          if (request.method() !== "POST" || await request.headerValue("origin") !== new URL(base).origin) {
            return route.fulfill({status: 403});
          }
          approvals.push(node.id);
          node.verified = true;
          return route.fulfill({json: {verified: true, url: node.url}});
        }
        if (new URL(url).pathname === node.url && node.verified) {
          return route.fulfill({contentType: "text/plain", body: node.name});
        }
      }
      return route.fulfill({status: 404});
    });

    await page.goto(base);
    await page.locator(".workspace-group summary").first().press("Enter");
    await page.waitForResponse(response => response.url() === base + "api/workspaces");
    await page.reload();
    assert.deepEqual(approvals, []);
    for (const [index, container] of [".workspace-group", ".device"].entries()) {
      const node = nodes[index];
      const row = page.locator(`${container}[data-node-id="${node.id}"]`);
      assert.match(await row.innerText(), /Unverified/);
      await row.getByRole("button", {name: `Verify and open ${node.name}`}).press("Enter");
      await page.waitForURL(new URL(node.url, base).href);
      await page.goto(base);
      await row.getByRole("link", {name: `Manage workspaces on ${node.name}`}).click();
      await page.waitForURL(new URL(node.url, base).href);
      await page.goto(base);
    }
    assert.deepEqual(approvals, nodes.map(node => node.id));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
