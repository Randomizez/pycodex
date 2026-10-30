// Decorate complete clock notifications without replacing renderer-owned nodes.
const mounts = new WeakMap();

function parseTick(text) {
  const match = /^<clock_tick>\s*(\{[\s\S]*\})\s*<\/clock_tick>$/.exec(text.trim());
  if (!match) return null;
  try {
    const tick = JSON.parse(match[1]);
    if (typeof tick.period_m !== "number" || !Number.isFinite(tick.period_m) ||
        tick.period_m <= 0 || typeof tick.current_time !== "string" ||
        !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(tick.current_time) ||
        !Number.isFinite(Date.parse(tick.current_time))) return null;
    return tick;
  } catch {
    return null;
  }
}

function divider(tick) {
  const details = document.createElement("details");
  details.className = "clock-tick-divider";
  // Static markup only; notification values are assigned as text below.
  details.innerHTML = `<summary title="查看定时唤醒详情">
    <span class="clock-tick-label">
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"></circle><path d="M12 7v5l3 2"></path></svg>
      <span>定时唤醒</span><span aria-hidden="true">·</span><time></time>
    </span>
  </summary>
  <dl class="clock-tick-info">
    <dt>触发时间</dt><dd class="clock-tick-date"></dd>
    <dt>等待间隔</dt><dd class="clock-tick-period"></dd>
  </dl>`;
  const date = new Date(tick.current_time);
  const time = details.querySelector("time");
  time.dateTime = tick.current_time;
  time.textContent = date.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", hourCycle: "h23"});
  time.title = tick.current_time;
  details.querySelector(".clock-tick-date").textContent = tick.current_time.replace("T", " ");
  details.querySelector(".clock-tick-period").textContent = `回复结束后 ${tick.period_m} 分钟`;
  return details;
}

export function installClockTicks(log = document.getElementById("log")) {
  if (!log) return () => {};
  if (mounts.has(log)) return mounts.get(log);
  const decorated = new WeakMap();

  function decorate(entry) {
    if (entry.parentElement !== log || !entry.matches(".entry.user")) return;
    const body = entry.firstElementChild;
    if (!body || !body.matches(".text")) return;
    const source = body.textContent;
    const previous = decorated.get(entry);
    if (previous && previous.source === source && previous.node.parentElement === entry) return;
    const tick = parseTick(source);
    if (previous) previous.node.remove();
    decorated.delete(entry);
    if (!tick) {
      entry.removeAttribute("data-clock-tick");
      return;
    }
    const node = divider(tick);
    entry.appendChild(node);
    entry.setAttribute("data-clock-tick", "");
    decorated.set(entry, {source, node});
  }

  // Inspect only changed prompt entries, not streaming response/tool subtrees.
  const observer = new MutationObserver(records => {
    const candidates = new Set();
    for (const record of records) {
      if (record.target === log) {
        for (const node of record.addedNodes) {
          if (node.nodeType === 1 && node.matches(".entry.user")) candidates.add(node);
        }
      } else {
        const target = record.target.nodeType === 1 ? record.target : record.target.parentElement;
        const body = target?.closest(".text");
        if (body?.parentElement.matches(".entry.user")) candidates.add(body.parentElement);
      }
    }
    candidates.forEach(decorate);
  });
  observer.observe(log, {childList: true, subtree: true, characterData: true});
  log.querySelectorAll(":scope > .entry.user").forEach(decorate);
  const cleanup = () => {
    observer.disconnect();
    log.querySelectorAll(":scope > [data-clock-tick]").forEach(entry => {
      decorated.get(entry)?.node.remove();
      entry.removeAttribute("data-clock-tick");
    });
    mounts.delete(log);
  };
  mounts.set(log, cleanup);
  return cleanup;
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => installClockTicks(), {once: true});
} else {
  installClockTicks();
}
