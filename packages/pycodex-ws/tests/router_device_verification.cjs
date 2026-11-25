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
    const hiddenWorkspaces = nodes.map(node => ({
      node_id: node.id,
      id: `ws-${node.name}`,
      work_dir: `/work/${node.name}`,
      state_path: `/state/${node.name}`,
      session_count: 1,
      running: false,
      unread: false,
      url: `${node.url}w/ws-${node.name}/`,
    }));
    const approvals = [];
    await page.route("http://router.test/**", async route => {
      const request = route.request();
      const url = request.url();
      if (url === base) {
        return route.fulfill({contentType: "text/html", headers: {"Referrer-Policy": "no-referrer"}, body: html});
      }
      if (url === base + "api/workspaces") {
        const verifiedIds = new Set(nodes.filter(n => n.verified).map(n => n.id));
        const workspaces = hiddenWorkspaces.filter(ws => verifiedIds.has(ws.node_id));
        return route.fulfill({json: {nodes, workspaces}});
      }
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

    // Collapse node a again so we verify "Verify and unfold" unfolds it in place without navigating away.
    const groupA = page.locator(`.workspace-group[data-node-id="${nodes[0].id}"]`);
    if (await groupA.evaluate(el => el.open)) {
      await groupA.locator("summary").click();
    }
    assert.equal(await groupA.evaluate(el => el.open), false);
    assert.match(await groupA.innerText(), /Unverified/);
    await groupA.getByRole("button", {name: `Verify and unfold ${nodes[0].name}`}).press("Enter");
    await groupA.getByRole("link", {name: `Manage workspaces on ${nodes[0].name}`}).waitFor();
    assert.equal(page.url(), base);
    assert.equal(await groupA.evaluate(el => el.open), true);
    assert.match(await groupA.innerText(), /ws-a/);
    await groupA.getByRole("link", {name: `Manage workspaces on ${nodes[0].name}`}).click();
    await page.waitForURL(new URL(nodes[0].url, base).href);

    // Return to directory and verify node b via the sidebar's plain "Verify" button (stays on page & keeps collapsed).
    await page.goto(base);
    const cardB = page.locator(`.device[data-node-id="${nodes[1].id}"]`);
    const groupB = page.locator(`.workspace-group[data-node-id="${nodes[1].id}"]`);
    assert.match(await cardB.innerText(), /Unverified/);
    assert.equal(await groupB.evaluate(el => el.open), false);
    await cardB.getByRole("button", {name: `Verify ${nodes[1].name}`}).press("Enter");
    await cardB.getByRole("link", {name: `Manage workspaces on ${nodes[1].name}`}).waitFor();
    assert.equal(page.url(), base);
    assert.equal(await groupB.evaluate(el => el.open), false);
    await cardB.getByRole("link", {name: `Manage workspaces on ${nodes[1].name}`}).click();
    await page.waitForURL(new URL(nodes[1].url, base).href);

    assert.deepEqual(approvals, nodes.map(node => node.id));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
