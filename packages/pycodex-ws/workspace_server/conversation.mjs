import {element, frontendUrl, isNearBottom, reconcileChildren, scrollToBottom} from "./workspace_dom.mjs";

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

const USER_URL_RE = /https?:\/\/[^\s<>"]+/g;

function renderUserText(body, text) {
  if (!text || !text.includes("http") || typeof document.createTextNode !== "function") {
    body.textContent = text;
    return;
  }
  USER_URL_RE.lastIndex = 0;
  let cursor = 0;
  let match;
  const nodes = [];
  while ((match = USER_URL_RE.exec(text)) !== null) {
    const trimmed = match[0].replace(/[.,;:!?)]+$/, "");
    if (!trimmed) continue;
    if (match.index > cursor) {
      nodes.push(document.createTextNode(text.slice(cursor, match.index)));
    }
    const link = element("a", "", trimmed);
    link.href = trimmed;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    nodes.push(link);
    cursor = match.index + trimmed.length;
    USER_URL_RE.lastIndex = cursor;
  }
  if (!nodes.length) {
    body.textContent = text;
    return;
  }
  if (cursor < text.length) {
    nodes.push(document.createTextNode(text.slice(cursor)));
  }
  body.replaceChildren(...nodes);
}

export function createConversationRenderer({log, renderMarkdownInto, builtinFrontend}) {
  function addEntry(cls, text, markdown = false, shouldStick = isNearBottom(log)) {
    const entry = element("div", "entry " + cls);
    const body = element("div", markdown ? "text markdown" : "text", text || "");
    entry.appendChild(body);
    log.appendChild(entry);
    if (markdown) renderMarkdownInto(body, text || "");
    else if (cls === "user") renderUserText(body, text || "");
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
        else if (cls === "user") renderUserText(body, text || "");
        else body.textContent = text;
      }
      entry.renderedSource = text;
      entries.push(entry);
      return entry.querySelector(":scope > .text");
    }
    for (let index = 0; index < turns.length; index += 1) {
      const turn = turns[index];
      if (turn.kind === "compaction") {
        let count = 1;
        while (index + 1 < turns.length && turns[index + 1].kind === "compaction") {
          count += 1;
          index += 1;
        }
        const label = count > 1 ? `Context compacted ×${count}` : "Context compacted";
        const marker = renderEntry(`turn:${turn.turn_id}:compact`, label, "compaction");
        marker.parentElement.setAttribute("role", "separator");
        marker.parentElement.setAttribute("aria-label", label);
        continue;
      }
      renderTurn(turn, renderEntry, activeSessionId, builtinFrontend);
    }
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
    reconcileChildren(log, entries);
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

const TAG_ICONS = {
  clock: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"></circle><path d="M12 7v5l3 2"></path></svg>',
  exec: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 8 4 4-4 4m6 0h6"></path></svg>',
  agent: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="7" cy="7" r="2.5"></circle><circle cx="17" cy="17" r="2.5"></circle><path d="M7 9.5v2a3.5 3.5 0 0 0 3.5 3.5h4"></path></svg>',
  event: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"></circle><circle cx="12" cy="12" r="2.4" fill="currentColor" stroke="none"></circle></svg>',
};
const EXIT_CODE_RE = /(?:^Command exited with code (-?\d+)\b|"exit_code"\s*:\s*(-?\d+))/m;

function describeTag(tag, text) {
  if (/wake|schedul|clock|timer|唤醒|定时/i.test(tag)) return {icon: "clock", tone: "clock", label: tag, exit: ""};
  if (/^task\b|exec/i.test(tag)) {
    const canceled = /cancel/i.test(tag);
    const match = canceled ? null : EXIT_CODE_RE.exec(text);
    const code = match ? (match[1] || match[2]) : "";
    const failed = Boolean(code && code !== "0");
    return {
      icon: "exec",
      tone: failed ? "warn" : canceled ? "muted" : "exec",
      label: tag,
      exit: failed ? `exit ${code}` : "",
    };
  }
  if (/agent|parent/i.test(tag)) return {icon: "agent", tone: "muted", label: tag, exit: ""};
  return {icon: "event", tone: "muted", label: tag, exit: ""};
}

function renderTagMessage(body, tag, message) {
  let details = body.querySelector(":scope > .message-tag");
  if (!details) {
    details = element("details", "message-tag");
    const summary = element("summary");
    const label = element("span", "message-tag-label");
    label.append(
      element("span", "message-tag-icon"),
      element("span", "message-tag-name"),
      element("span", "message-tag-exit"),
      element("time", "message-tag-time"),
    );
    summary.appendChild(label);
    details.append(summary, element("pre", "message-tag-source"));
    body.replaceChildren(details);
  }
  const info = describeTag(tag, message.text || "");
  if (details.dataset.tone !== info.tone) details.dataset.tone = info.tone;
  const [iconNode, nameNode, exitNode, timeNode] = details.firstElementChild.firstElementChild.children;
  if (iconNode.dataset.icon !== info.icon) {
    iconNode.dataset.icon = info.icon;
    iconNode.innerHTML = TAG_ICONS[info.icon];
  }
  if (nameNode.textContent !== info.label) nameNode.textContent = info.label;
  const hideExit = !info.exit;
  if (exitNode.hidden !== hideExit) exitNode.hidden = hideExit;
  if (info.exit && exitNode.textContent !== info.exit) exitNode.textContent = info.exit;
  const formattedTime = formatMessageTime(message.created_at);
  const hideTime = !formattedTime;
  if (timeNode.hidden !== hideTime) timeNode.hidden = hideTime;
  if (formattedTime) {
    if (timeNode.textContent !== formattedTime) timeNode.textContent = formattedTime;
    if (timeNode.dateTime !== message.created_at) {
      timeNode.dateTime = message.created_at;
      timeNode.title = new Date(message.created_at).toLocaleString();
    }
  }
  const sourceNode = details.lastElementChild;
  if (sourceNode.textContent !== message.text) sourceNode.textContent = message.text;
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
    const tag = !isControl && typeof message.visible === "string" ? message.visible.trim() : "";
    if (tag) {
      renderTagMessage(renderEntry(`message:${message.message_id}`, "", "tag"), tag, message);
      continue;
    }
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
      const authorName = message.name && message.name !== "user" ? message.name : "";
      const authorNode = header.firstElementChild;
      if (authorNode.textContent !== authorName) {
        authorNode.textContent = authorName;
      }
      const hideAuthor = !authorName;
      if (authorNode.hidden !== hideAuthor) {
        authorNode.hidden = hideAuthor;
      }
      setMessageTime(body, message.created_at);
      const time = body.parentElement.querySelector(".message-time");
      if (time && time.parentElement !== header) header.append(time);
      const hideHeader = hideAuthor && !time;
      if (header.hidden !== hideHeader) {
        header.hidden = hideHeader;
      }
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
  const temporary = !isControl && turn.response_phase !== "final" &&
    ["running", "responding", "tool"].includes(turn.status);
  const text = (temporary && turn.thinking) || turn.response || (turn.error && turn.thinking);
  if (text) {
    const response = renderEntry(
      `${key}:response`, text, isControl ? "control" : "response", !isControl
    );
    const finalId = isControl || temporary || turn.error ? "" : (turn.final_id || "");
    if (response.dataset.finalId !== finalId) response.dataset.finalId = finalId;
    if (response.dataset.sessionId !== activeSessionId) response.dataset.sessionId = activeSessionId;
    if (response.parentElement.dataset.temporary !== String(temporary)) {
      response.parentElement.dataset.temporary = String(temporary);
    }
    if (!isControl) {
      const elapsed = turn.prompt_at && turn.response_at
        ? formatDuration(turn.prompt_at, turn.response_at) : "";
      setMessageTime(response, temporary ? null : turn.response_at, elapsed);
    }
  }
  if (Array.isArray(turn.model_errors)) {
    renderModelErrors(turn.model_errors, key, renderEntry);
  }
  if (turn.error) {
    const error = renderEntry(`${key}:error`, turn.error, "error");
    error.dataset.finalId = isControl ? "" : (turn.final_id || "");
    error.dataset.sessionId = activeSessionId;
  }
}
