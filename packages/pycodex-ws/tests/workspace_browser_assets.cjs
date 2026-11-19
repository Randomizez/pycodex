const fs = require("node:fs");
const path = require("node:path");

async function openWorkspaceFixture(page, html) {
  const directory = path.join(__dirname, "../workspace_server");
  await page.route("http://workspace.test/**", route => {
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === "/") return route.fulfill({contentType: "text/html", body: html});
    return route.fulfill({
      contentType: pathname.endsWith(".css") ? "text/css" : "text/javascript",
      body: fs.readFileSync(path.join(directory, path.basename(pathname))),
    });
  });
  await page.goto("http://workspace.test/");
}

module.exports = {openWorkspaceFixture};
