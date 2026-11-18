// Run with NODE_PATH pointing to Playwright; optionally set CHROMIUM_EXECUTABLE.
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require("playwright");

async function main() {
  const source = fs.readFileSync(path.join(
    __dirname, "../workspace_server/frontend/transcript.mjs",
  ), "utf8");
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROMIUM_EXECUTABLE,
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage();
    await page.setContent("<main></main>");
    await page.evaluate(async url => {
      const {TranscriptView} = await import(url);
      const check = (condition, message) => {
        if (!condition) throw new Error(message);
      };
      const message = text => ({
        type: "message", role: "user", content: [{type: "input_text", text}],
      });
      let history = Array.from({length: 150}, (_, index) => message("original-" + index));
      let revision = "one";
      let replaceDuringPagination = false;
      const request = async url => {
        const params = new URL(url, "http://fixture/").searchParams;
        const before = params.get("before");
        if (replaceDuringPagination && before) {
          replaceDuringPagination = false;
          history = [message("latest restored history")];
          revision = "four";
        }
        const [cursorRevision, offset] = before ? before.split(":") : [revision, history.length];
        if (cursorRevision !== revision) throw Object.assign(new Error("History changed"), {status: 409});
        const end = Number(offset);
        const start = Math.max(0, end - 100);
        return {
          source: "active_history", revision,
          items: history.slice(start, end).map((payload, index) => ({
            index: start + index, type: "response_item", payload,
          })),
          next_cursor: start ? `${revision}:${start}` : null,
        };
      };
      const view = new TranscriptView({
        viewport: document.querySelector("main"), request,
        renderMarkdown: (node, text) => { node.textContent = text; },
        sessionId: "session", agentId: "agent", agentPath: "/root",
      });
      document.querySelector("main").append(view.root);
      view.active = true;
      try {
        await view.load();
        check(view.records.size === 100, "Initial page");
        await view.load(true);
        check(view.records.size === 150, "Earlier page");
        const unchanged = view.list.querySelector('[data-key="record:5"]');
        unchanged.open = false;
        history.push(message("appended"));
        await view.load();
        check(view.records.size === 151, "Append preserves older records");
        check(view.list.querySelector('[data-key="record:5"]') === unchanged, "Append preserves DOM");
        check(!unchanged.open, "Append preserves disclosure state");

        // A replacement before the latest page can leave every visible tail item unchanged.
        history[0] = message("replacement at original position");
        revision = "two";
        await view.load();
        check(view.records.size === 100, "Revision drops stale earlier pages");
        check(!view.list.textContent.includes("original-0"), "Removed content is gone");
        await view.load(true);
        check(view.records.size === 151, "Reload earlier current history");
        check(view.list.textContent.includes("replacement at original position"), "Replacement appears");

        // Compact replaces the old work; render the new history and its tool receipts.
        history = [
          message("Current history summary"),
          {type: "message", role: "assistant", phase: "commentary",
            content: [{type: "output_text", text: "Continuing"}]},
          {type: "function_call", name: "first_tool",
            call_id: "first", arguments: '{"value":"one"}'},
          {type: "function_call_output", call_id: "first", output: '"first result"'},
          {type: "function_call", name: "second_tool",
            call_id: "second", arguments: '{"value":"two"}'},
          {type: "function_call_output", call_id: "second", output: '"second result"'},
          {type: "message", role: "assistant", phase: "final_answer",
            content: [{type: "output_text", text: "Finished"}]},
        ];
        revision = "three";
        await view.load();
        check(view.records.size === history.length, "Uses only current history");
        check(!view.list.textContent.includes("original-"), "Pre-compact work removed");
        check(view.list.querySelectorAll('[data-kind="tool"]').length === 2, "Both call/result pairs rendered");
        check(view.list.textContent.includes("Current history summary"), "Actual summary visible");
        check(!view.list.querySelector('[data-kind="compaction"]'), "No checkpoint placeholder");
        check([...view.list.querySelectorAll(".transcript-phase[data-state]")].every(
          node => node.dataset.state === "completed",
        ), "Tools are completed");

        // A reset during a multi-page refresh must not merge histories from two revisions.
        history.push(...Array.from({length: 300}, (_, index) => message("burst-" + index)));
        replaceDuringPagination = true;
        let scheduled = 0;
        view.schedule = () => { scheduled += 1; };
        await view.load();
        check(view.pendingReset && scheduled === 1, "Invalid cursor schedules fresh pagination");
        check(view.records.size === 7, "Mixed pages were not merged");
        await view.load();
        check(view.records.size === 1, "Fresh revision replaces all cached records");
        check(view.list.textContent.includes("latest restored history"), "Restored history visible");
        check(!view.list.textContent.includes("Current history summary"), "Previous revision removed");
      } finally {
        view.stop();
      }
    }, "data:text/javascript;base64," + Buffer.from(source).toString("base64"));
    console.log("Transcript current-history, replacement, pagination and DOM checks passed.");
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
