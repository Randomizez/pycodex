import "./custom.js";

export function clockPrompt(minute = 40) {
  const wallTime = new Date(Date.UTC(2026, 0, 15, 14, minute));
  return `<clock_tick>\n${JSON.stringify({
    period_m: 5,
    current_time: wallTime.toISOString().replace(".000Z", "+08:00"),
  })}\n</clock_tick>`;
}

// The fixture preserves prompt bodies just like the conversation renderer.
export function renderEntries(log, entries) {
  const previous = new Map([...log.children].map(node => [node.dataset.entryKey, node]));
  const retained = entries.map(item => {
    let node = previous.get(item.id);
    if (!node || node.className !== `entry ${item.kind}`) {
      node = document.createElement("div");
      node.className = `entry ${item.kind}`;
      node.dataset.entryKey = item.id;
      const body = document.createElement("div");
      body.className = "text";
      node.append(body);
    }
    if (node.firstElementChild.textContent !== item.text) node.firstElementChild.textContent = item.text;
    return node;
  });
  const keep = new Set(retained);
  [...log.children].forEach(node => { if (!keep.has(node)) node.remove(); });
  retained.forEach((node, index) => {
    if (log.children[index] !== node) log.insertBefore(node, log.children[index] || null);
  });
}

const log = document.getElementById("log");
const entries = [
  {id: "a:prompt", kind: "user", text: "帮我关注这个任务，回复后等 5 分钟再检查。"},
  {id: "a:reply", kind: "response", text: "任务仍在运行。下一次唤醒后，我会继续检查进度。"},
  {id: "a:tick", kind: "user", text: clockPrompt()},
  {id: "a:result", kind: "response", text: "刚检查过，任务仍在运行，目前没有新增错误。"},
];
let active = "a";
let tickNumber = 0;
const render = () => {
  document.getElementById("session-title").textContent = active === "a" ? "示例任务 A" : "示例任务 B";
  renderEntries(log, active === "a" ? entries : [
    {id: "b:prompt", kind: "user", text: "这是另一个普通会话。"},
    {id: "b:reply", kind: "response", text: "这里没有定时唤醒记录。"},
  ]);
};
document.getElementById("new-tick").onclick = () => {
  active = "a";
  tickNumber++;
  entries.push(
    {id: `a:tick:${tickNumber}`, kind: "user", text: clockPrompt(40 + tickNumber * 5)},
    {id: `a:result:${tickNumber}`, kind: "response", text: "本轮检查完成，任务仍在运行。"},
  );
  render();
};
document.getElementById("update-reply").onclick = () => {
  active = "a";
  entries[entries.length - 1].text += " 进度已更新。";
  render();
};
document.getElementById("switch-session").onclick = () => {
  active = active === "a" ? "b" : "a";
  render();
};
document.getElementById("reload-history").onclick = () => {
  log.replaceChildren();
  render();
};
render();
if (new URL(location.href).searchParams.has("verify")) {
  const output = document.getElementById("verification");
  try {
    const {verify} = await import("./verify.mjs");
    const checks = await verify();
    output.textContent = `PASS · ${checks.length} checks\n${checks.join("\n")}`;
    output.dataset.result = "pass";
  } catch (error) {
    output.textContent = `FAIL · ${error.message}`;
    output.dataset.result = "fail";
    throw error;
  }
}
