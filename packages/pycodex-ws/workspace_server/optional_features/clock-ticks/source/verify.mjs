import {installClockTicks} from "./custom.js";

const settle = () => new Promise(resolve => requestAnimationFrame(resolve));
const tick = '<clock_tick>\n{"period_m":5,"current_time":"2026-01-15T14:40:00+08:00"}\n</clock_tick>';

export async function verify() {
  const checks = [];
  function check(condition, label) {
    if (!condition) throw new Error(label);
    checks.push(label);
  }
  const log = document.createElement("div");
  log.className = "log";
  log.style.cssText = "position:absolute;left:-10000px;top:0;width:350px";
  document.body.append(log);
  const add = (text, kind = "user") => {
    const entry = document.createElement("div");
    entry.className = `entry ${kind}`;
    const body = document.createElement("div");
    body.className = "text";
    body.textContent = text;
    entry.append(body);
    log.append(entry);
    return entry;
  };
  const first = add(tick);
  const firstBody = first.firstElementChild;
  let cleanup = installClockTicks(log);
  try {
    check(first.querySelectorAll(".clock-tick-divider").length === 1, "History present before mount");
    check(first.firstElementChild === firstBody && firstBody.textContent === tick, "Source element and text preserved");
    check(getComputedStyle(firstBody).display === "none", "Raw clock prompt hidden");
    const marker = first.querySelector("details");
    check(!marker.open && first.getBoundingClientRect().height <= 44, "Compact single row");
    check(marker.querySelector("time").dateTime === "2026-01-15T14:40:00+08:00", "Recorded timestamp preserved");
    marker.open = true;
    add("A streaming reply", "response").firstElementChild.textContent += " updated";
    await settle();
    check(first.querySelector("details") === marker && marker.open, "Streaming keeps details open");
    installClockTicks(log);
    check(first.querySelectorAll("details").length === 1, "Repeated mount is idempotent");
    const next = add(tick.replace("14:40", "14:45"));
    await settle();
    check(next.hasAttribute("data-clock-tick"), "Incoming tick decorated");
    const unchanged = [
      "Explain <clock_tick> please",
      "```xml\n" + tick + "\n```",
      '<clock_tick>{"period_m":5,"current_time":"bad"}</clock_tick>',
      tick.replace('"period_m":5', '"period_m":0'),
      tick.replace('"period_m":5', '"period_m":"5"'),
      '<clock_tick>{"period_m":5,"current_time":"<img src=x onerror=alert(1)>"}</clock_tick>',
      '<exec_command_completed>{"exit_code":0}</exec_command_completed>',
    ].map(text => add(text));
    await settle();
    check(unchanged.every(entry => !entry.hasAttribute("data-clock-tick")), "Ordinary, quoted, malformed and unrelated events unchanged");
    check(log.querySelectorAll("img").length === 0, "Payload never becomes HTML");
    firstBody.textContent = "An ordinary replacement";
    await settle();
    check(!first.querySelector("details") && getComputedStyle(firstBody).display !== "none", "Source replacement restores ordinary prompt");
    firstBody.firstChild.data = tick;
    await settle();
    check(first.hasAttribute("data-clock-tick"), "Character-data prompt update observed");
    log.replaceChildren();
    add("Another session");
    await settle();
    check(!log.querySelector("details"), "Session replacement removes prior ticks");
    add(tick);
    await settle();
    check(log.querySelectorAll("details").length === 1, "History reload decorated");
    cleanup();
    check(!log.querySelector("details") && !log.querySelector("[data-clock-tick]"), "Cleanup restores all original prompts");
    add(tick);
    await settle();
    check(!log.querySelector("details"), "Cleanup disconnects observation");
    cleanup = installClockTicks(log);
    check(log.querySelectorAll("details").length === 2, "Remount decorates restored history");
    return checks;
  } finally {
    cleanup();
    log.remove();
  }
}
