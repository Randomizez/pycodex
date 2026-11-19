import {element, frontendUrl, isNearBottom, scrollToBottom} from "./workspace_dom.mjs";

export function messageSignature(snapshot) {
  const turns = (snapshot && snapshot.turns) || [];
  return JSON.stringify([
    turns.map((item) => [
      item.submission_id || "",
      item.turn_id || "",
      item.final_id || "",
      item.messages,
      item.response || "",
      item.response_phase || "",
      item.thinking || "",
      item.status || "",
      item.error || "",
      item.kind || "",
      item.response_at || "",
      item.model_errors || [],
    ]),
    snapshot.input_request || null,
  ]);
}

export function createConversationRenderer({log, renderMarkdownInto, builtinFrontend}) {
  function addEntry(cls, text, markdown = false, shouldStick = isNearBottom(log)) {
    const entry = element("div", "entry " + cls);
    const body = element("div", markdown ? "text markdown" : "text", text || "");
    entry.appendChild(body);
    log.appendChild(entry);
    if (markdown) renderMarkdownInto(body, text || "");
    if (shouldStick) scrollToBottom(log);
    return body;
  }

  function renderConversation(activeSessionId, turns, inputRequest) {
    const previous = new Map(
      Array.from(log.children, entry => [entry.dataset.entryKey, entry])
    );
    const entries = [];
    function renderEntry(key, text, cls, markdown = false) {
      key = `${activeSessionId}:${key}`;
      let entry = previous.get(key);
      if (!entry || entry.className !== `entry ${cls}`) {
        entry = addEntry(cls, text, markdown, false).parentElement;
        entry.dataset.entryKey = key;
      } else if (entry.renderedSource !== text) {
        const body = entry.querySelector(":scope > .text");
        if (markdown) renderMarkdownInto(body, text);
        else body.textContent = text;
      }
      entry.renderedSource = text;
      entries.push(entry);
      return entry.querySelector(":scope > .text");
    }
    turns.forEach(turn => renderTurn(turn, renderEntry, activeSessionId, builtinFrontend));
    if (!turns.length && !inputRequest) {
      let empty = log.querySelector(".empty-chat");
      if (!empty) {
        log.appendChild(document.getElementById("emptyChat").content.cloneNode(true));
        empty = log.lastElementChild;
      }
      entries.push(empty);
    }
    if (inputRequest) {
      renderEntry("input-request", inputRequest.text, "control");
    }
    const retained = new Set(entries);
    Array.from(log.children).forEach(entry => {
      if (!retained.has(entry)) entry.remove();
    });
    entries.forEach((entry, index) => {
      if (log.children[index] !== entry) log.insertBefore(entry, log.children[index] || null);
    });
  }
  return {renderConversation, addEntry};
}

function renderModelErrors(messages, key, renderEntry) {
  const retryHeader = /^Reconnecting\.\.\. (\d+\/(?:\d+|∞))(?:\r?\n|$)/;
  for (let index = 0; index < messages.length;) {
    const start = index;
    if (!retryHeader.test(messages[index])) {
      renderEntry(`${key}:model-error:${index}`, messages[index], "model-error");
      index++;
      continue;
    }
    while (index < messages.length && retryHeader.test(messages[index])) index++;
    const retries = messages.slice(start, index);
    const latest = retries[retries.length - 1];
    const attempt = latest.match(retryHeader)[1];
    const status = latest.match(/responses request failed with status (\d{3})\b/);
    const label = `Model retries · ${retries.length} recorded · latest ${attempt}` +
      (status ? ` · HTTP ${status[1]}` : "");
    const body = renderEntry(`${key}:model-retries:${start}`, "", "model-error");
    let group = body.querySelector(".model-retry-group");
    if (!group) {
      group = element("details", "model-retry-group");
      group.append(element("summary"), element("pre", "model-retry-log"));
      body.replaceChildren(group);
    }
    const summary = group.firstElementChild;
    if (summary.textContent !== label) summary.textContent = label;
    const text = retries.join("\n\n");
    const output = group.lastElementChild;
    if (output.textContent !== text) output.textContent = text;
  }
}

const messageTimeFormat = new Intl.DateTimeFormat(undefined, {hour: "2-digit", minute: "2-digit"});
const messageDateFormat = new Intl.DateTimeFormat(undefined, {
  month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
});

function formatMessageTime(value) {
  const date = value ? new Date(value) : null;
  if (!date || Number.isNaN(date.getTime())) return "";
  const sameDay = date.toDateString() === new Date().toDateString();
  return (sameDay ? messageTimeFormat : messageDateFormat).format(date);
}

function formatDuration(start, end) {
  const seconds = Math.round((new Date(end) - new Date(start)) / 1000);
  if (!Number.isFinite(seconds) || seconds < 0) return "";
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

function setMessageTime(body, value, detail = "") {
  const entry = body.parentElement;
  const label = formatMessageTime(value);
  let time = entry.querySelector(".message-time");
  if (!label) {
    time?.remove();
    return;
  }
  if (!time) {
    time = element("time", "message-time");
    entry.appendChild(time);
  }
  const text = detail ? `${label} · ${detail}` : label;
  if (time.textContent !== text) time.textContent = text;
  if (time.dateTime !== value) {
    time.dateTime = value;
    time.title = new Date(value).toLocaleString();
  }
}

function renderTurn(turn, renderEntry, activeSessionId, builtinFrontend) {
  const key = `turn:${turn.turn_id}`;
  if (turn.kind === "compaction") {
    const marker = renderEntry(`${key}:compact`, "Context compacted", "compaction");
    marker.parentElement.setAttribute("role", "separator");
    marker.parentElement.setAttribute("aria-label", "Context compacted");
    return;
  }
  const isControl = turn.kind === "control";
  for (const message of turn.messages) {
    const setup = message.text.startsWith("<frontend_customization>\n") &&
      message.text.endsWith("\n</frontend_customization>");
    const body = renderEntry(`message:${message.message_id}`, message.text, setup ? "setup" : isControl ? "control" : "user");
    if (!setup && !isControl) {
      let header = body.parentElement.querySelector(":scope > .message-header");
      if (!header) {
        header = element("div", "message-header");
        header.append(element("span", "message-author"));
        body.before(header);
      }
      if (header.firstElementChild.textContent !== message.name) {
        header.firstElementChild.textContent = message.name;
      }
      setMessageTime(body, message.created_at);
      const time = body.parentElement.querySelector(".message-time");
      if (time && time.parentElement !== header) header.append(time);
    }
    if (setup && !body.firstElementChild) {
      const notice = element("p", "frontend-notice",
        "Changes affect all workspaces for this server user. Refresh to apply. ");
      const recovery = element("a", "",
        builtinFrontend ? "Re-enable customizations" : "Open built-in interface");
      recovery.href = frontendUrl(!builtinFrontend);
      notice.append(recovery);
      const details = element("details");
      details.append(
        element("summary", "", "Interface customization context"),
        element("div", "setup-source", message.text.slice(
          "<frontend_customization>\n".length, -"\n</frontend_customization>".length
        ))
      );
      body.replaceChildren(notice, details);
    }
  }
  if (turn.response) {
    const response = renderEntry(
      `${key}:response`, turn.response, isControl ? "control" : "response", !isControl
    );
    const finalId = isControl || turn.error ? "" : (turn.final_id || "");
    if (response.dataset.finalId !== finalId) response.dataset.finalId = finalId;
    if (response.dataset.sessionId !== activeSessionId) response.dataset.sessionId = activeSessionId;
    const temporary = !isControl && turn.response_phase !== "final" &&
      ["running", "responding", "tool"].includes(turn.status);
    if (response.parentElement.dataset.temporary !== String(temporary)) {
      response.parentElement.dataset.temporary = String(temporary);
    }
    if (!isControl) {
      const elapsed = turn.prompt_at && turn.response_at
        ? formatDuration(turn.prompt_at, turn.response_at) : "";
      setMessageTime(response, temporary ? null : turn.response_at, elapsed);
    }
  }
  if (turn.thinking) renderEntry(`${key}:thinking`, turn.thinking, "thinking");
  if (Array.isArray(turn.model_errors)) {
    renderModelErrors(turn.model_errors, key, renderEntry);
  }
  if (turn.error) {
    const error = renderEntry(`${key}:error`, turn.error, "error");
    error.dataset.finalId = isControl ? "" : (turn.final_id || "");
    error.dataset.sessionId = activeSessionId;
  }
}
