import { Composer } from "../out/composer.mjs";

(() => {
  const vscode = acquireVsCodeApi();
  const markdown = window.markdownit({ html: false, breaks: true, linkify: false });
  // Images are links, so model output cannot issue background network requests.
  markdown.renderer.rules.image = (tokens, index) => {
    const token = tokens[index];
    const label = markdown.utils.escapeHtml(token.content || "图片");
    const href = markdown.utils.escapeHtml(token.attrGet("src") || "");
    return `<a href="${href}">${label}</a>`;
  };
  const byId = id => document.getElementById(id);
  const stack = byId("session-stack");
  const chatElement = byId("chat");
  const draft = byId("draft");
  const conversation = byId("conversation");
  const turnsElement = byId("turns");
  const saved = vscode.getState() || {};
  const drafts = new Map(Object.entries(saved.drafts || {}));
  const pendingSends = new Map();
  const turns = new Map();
  const sessionCards = new Map();
  let sessionKey = null;
  let state;
  let actionError = "";
  let failedAction = "refresh";
  let collapsed = false;
  let editingContext;
  const composer = new Composer(
    draft, () => { persist(); controls(); }, () => send(false), editContext,
    error => { actionError = error; failedAction = "refresh"; controls(); },
  );

  function currentDraft() { return { ...composer.read(), collapsed }; }
  function persist() {
    if (sessionKey) drafts.set(sessionKey, currentDraft());
    vscode.setState({ drafts: Object.fromEntries(drafts) });
  }
  function post(type, values = {}) { vscode.postMessage({ type, sessionKey, ...values }); }
  function textElement(tag, className, text) {
    const node = document.createElement(tag);
    node.className = className;
    node.textContent = text;
    return node;
  }
  function editContext(block) {
    editingContext = block.id;
    byId("context-label").textContent = block.label;
    byId("context-editor").value = block.text;
    byId("context-editor").setAttribute("aria-label", `${block.label} 上下文`);
    byId("context-detail").hidden = false;
    byId("context-editor").focus();
    byId("composer").scrollIntoView({ block: "nearest" });
  }
  function closeContext() {
    editingContext = undefined;
    byId("context-detail").hidden = true;
  }
  function updateContextMeter(snapshot) {
    byId("context-control").hidden = !snapshot;
    if (!snapshot) return;
    const percent = snapshot.context_remaining_percent;
    const compact = snapshot.auto_compact_token_limit;
    const formatTokens = value => value == null ? "—" : value.toLocaleString("en-US");
    byId("context-meter").style.setProperty("--context-remaining", `${percent == null ? 0 : percent}%`);
    byId("context-meter-value").textContent = percent == null ? "—" : percent >= 100 ? "" : String(percent);
    const label = compact == null ? "上下文剩余" : "距自动压缩剩余";
    byId("context-meter").setAttribute("aria-label",
      percent == null ? "查看上下文用量" : `${label} ${percent}%。查看上下文用量`);
    byId("context-current").textContent = formatTokens(snapshot.usage_tokens);
    byId("context-compact").textContent = compact == null ? "关闭" : formatTokens(compact);
    byId("context-max").textContent = formatTokens(snapshot.max_context_window);
    const note = compact == null
      ? "自动压缩已关闭，圆环表示剩余上下文。"
      : "圆环在压缩阈值处降至 0%，随后在下一次模型请求前压缩。";
    byId("context-note").textContent = snapshot.usage_tokens == null ? `等待模型上报用量。${note}` : note;
  }
  function setupContextUsage() {
    const control = byId("context-control");
    const meter = byId("context-meter");
    let pinned = false;
    const show = visible => {
      byId("context-usage").hidden = !visible;
      meter.setAttribute("aria-expanded", String(visible));
    };
    const close = () => { pinned = false; show(false); };
    control.addEventListener("pointerenter", event => {
      if (event.pointerType !== "touch") show(true);
    });
    control.addEventListener("pointerleave", () => { if (!pinned) show(false); });
    meter.addEventListener("focus", () => show(true));
    control.addEventListener("focusout", event => {
      if (!pinned && !control.contains(event.relatedTarget)) show(false);
    });
    meter.addEventListener("click", () => { pinned = !pinned; show(pinned); });
    document.addEventListener("pointerdown", event => {
      if (!control.contains(event.target)) close();
    });
    document.addEventListener("keydown", event => {
      if (event.key === "Escape") close();
    });
    return close;
  }
  const closeContextUsage = setupContextUsage();
  function nearBottom() {
    return conversation.scrollHeight - conversation.scrollTop - conversation.clientHeight < 55;
  }
  function scrollToLatest() {
    conversation.scrollTop = conversation.scrollHeight;
    byId("latest").hidden = true;
  }
  function setCollapsed(value) {
    collapsed = value;
    byId("chat").classList.toggle("collapsed", value);
    byId("expanded").hidden = value;
    byId("collapsed-summary").hidden = !value;
    byId("collapse").setAttribute("aria-expanded", String(!value));
    persist();
  }
  function renderSessions(chats) {
    const ordered = [];
    const inactive = new Set();
    for (const chat of chats) {
      if (chat.sessionKey === sessionKey) {
        ordered.push(chatElement);
        continue;
      }
      inactive.add(chat.sessionKey);
      let card = sessionCards.get(chat.sessionKey);
      if (!card) {
        card = textElement("div", "session-card", "");
        card.dataset.sessionKey = chat.sessionKey;
        const summary = textElement("button", "session-summary", "");
        summary.type = "button";
        summary.dataset.sessionKey = chat.sessionKey;
        summary.setAttribute("aria-expanded", "false");
        summary.innerHTML = '<span class="summary-heading"><span class="summary-title"></span><span class="summary-id"></span><svg class="icon"><use href="#i-chevron"/></svg></span><span class="summary-detail"><span class="summary-workspace"></span><span class="summary-dot"></span><span class="summary-status"></span></span>';
        summary.addEventListener("click", () => post("activate-session", { targetSessionKey: chat.sessionKey }));
        const close = textElement("button", "icon-button session-close", "");
        close.type = "button";
        close.title = "关闭会话";
        close.setAttribute("aria-label", "关闭会话");
        close.innerHTML = '<svg class="icon"><use href="#i-close"/></svg>';
        close.addEventListener("click", () => post("close-session", { targetSessionKey: chat.sessionKey }));
        card.append(summary, close);
        sessionCards.set(chat.sessionKey, card);
      }
      card.querySelector(".summary-title").textContent = chat.title;
      card.querySelector(".summary-id").textContent = chat.sessionId.slice(-6);
      card.querySelector(".summary-workspace").textContent = chat.workspace;
      card.querySelector(".summary-status").textContent = chat.closing ? "正在关闭…" : chat.error ? "连接异常" :
        chat.closed ? "已关闭" : chat.waitingForInput ? "等待回答" :
        chat.submitting ? "正在发送" : chat.running ? "运行中" : chat.queued ? "已排队" : "空闲";
      card.classList.toggle("running", chat.running || chat.submitting);
      card.classList.toggle("waiting", chat.waitingForInput);
      card.classList.toggle("failed", Boolean(chat.error));
      card.querySelector(".session-close").disabled = chat.closing;
      card.setAttribute("aria-busy", String(chat.running || chat.submitting || chat.closing));
      card.title = `${chat.title} · ${chat.sessionId}`;
      ordered.push(card);
    }
    if (!chats.length) ordered.push(chatElement);
    for (const [key, card] of sessionCards) {
      if (!inactive.has(key)) {
        card.remove();
        sessionCards.delete(key);
      }
    }
    ordered.forEach((element, index) => {
      if (stack.children[index] !== element) stack.insertBefore(element, stack.children[index] || null);
    });
    const keys = new Set(chats.map(chat => chat.sessionKey));
    let removed = false;
    for (const key of drafts.keys()) {
      if (!keys.has(key)) {
        drafts.delete(key);
        pendingSends.delete(key);
        removed = true;
      }
    }
    if (removed) persist();
  }
  function renderTurn(turn, existing) {
    const signature = JSON.stringify(turn);
    if (existing?.signature === signature) return existing;
    const element = existing?.element || textElement("section", "turn", "");
    element.replaceChildren();
    element.classList.toggle("compaction", turn.kind === "compaction");
    if (turn.kind === "compaction") {
      element.textContent = "上下文已压缩";
      element.setAttribute("role", "separator");
      element.setAttribute("aria-label", "上下文已压缩");
      return { element, signature };
    }
    if (turn.kind !== "control" && Array.isArray(turn.messages)) {
      for (const message of turn.messages) {
        const tag = typeof message.visible === "string" ? message.visible.trim() : "";
        if (!tag) continue;
        const divider = textElement("div", "turn-tag", tag);
        divider.setAttribute("role", "separator");
        divider.setAttribute("aria-label", tag);
        element.append(divider);
      }
    }
    if (turn.kind !== "control" && turn.prompt) {
      const user = textElement("div", "user-message", "");
      user.append(textElement("div", "user-bubble", turn.prompt));
      element.append(user);
    }
    if (turn.thinking) {
      const reasoning = textElement("div", "reasoning", "");
      reasoning.append(textElement("div", "response", turn.thinking));
      element.append(reasoning);
    }
    if (turn.response) {
      const response = textElement("div", turn.kind === "control" ? "response control" : "response", "");
      if (turn.kind === "control") response.textContent = turn.response;
      else response.innerHTML = markdown.render(turn.response);
      element.append(response);
    }
    if (turn.error) element.append(textElement("div", "turn-error", turn.error));
    return { element, signature };
  }
  function renderConversation(snapshot) {
    const follow = nearBottom();
    const visible = new Set();
    let changed = false;
    const items = snapshot?.turns || [];
    for (const [index, turn] of items.entries()) {
      const previous = turns.get(turn.submission_id);
      const rendered = renderTurn(turn, previous);
      if (rendered !== previous) changed = true;
      if (turnsElement.children[index] !== rendered.element) {
        turnsElement.insertBefore(rendered.element, turnsElement.children[index] || null);
        changed = true;
      }
      turns.set(turn.submission_id, rendered);
      visible.add(turn.submission_id);
    }
    for (const [id, turn] of turns) {
      if (!visible.has(id)) {
        turn.element.remove();
        turns.delete(id);
        changed = true;
      }
    }
    byId("empty").hidden = items.length > 0;
    if (changed && follow) scrollToLatest();
    else if (changed) byId("latest").hidden = false;
  }
  function controls() {
    const snapshot = state?.snapshot;
    const unavailable = !snapshot || !snapshot.accepts_input || state.closing || Boolean(state.error);
    const sending = Boolean(pendingSends.has(sessionKey) || state?.submitting);
    const parts = composer.parts();
    byId("send").disabled = unavailable || sending || !Composer.text(parts).trim();
    byId("queue").disabled = byId("send").disabled;
    byId("queue").hidden = !snapshot?.running || Boolean(snapshot?.input_request);
    byId("cancel-input").disabled = sending || state?.closing;
    byId("insert-context").disabled = Boolean(state?.closing);
    byId("start-chat").parentElement.hidden = Boolean(sessionKey);
    byId("suggestions").hidden = !sessionKey;
    byId("empty-title").textContent = sessionKey ? "从这里开始" : "开始对话";
    byId("empty-hint").hidden = !sessionKey;
    byId("config-hint").hidden = Boolean(sessionKey);
    byId("model").textContent = snapshot?.model || "Pycodex";
    byId("model").title = snapshot?.model || "";
    byId("close-session").disabled = !sessionKey || state?.closing;
    const disabled = !sessionKey || Boolean(state?.closing);
    draft.contentEditable = String(!disabled);
    draft.setAttribute("aria-disabled", String(disabled));
    byId("context-editor").disabled = disabled;
    byId("context-remove").disabled = disabled;
    if (editingContext && !parts.some(part => part.id === editingContext)) closeContext();
    draft.dataset.placeholder = !sessionKey ? "先新建对话…" : snapshot?.input_request ? "回答上面的问题…" :
      snapshot?.running ? "补充要求，会在下一步接着处理…" :
      snapshot?.turns.length ? "继续追问，或描述下一步…" : "描述你想做的事…";
    const error = actionError || state?.error || "";
    const [summary, ...lines] = error.split("\n");
    const details = lines.join("\n").trim();
    byId("error").hidden = !error;
    byId("error-text").textContent = summary;
    if (byId("error-diagnostics").textContent !== details) byId("error-details").open = false;
    byId("error-details").hidden = !details;
    byId("error-diagnostics").textContent = details;
    const retry = actionError && failedAction !== "refresh";
    byId("refresh-label").textContent = retry ? "重试" : "刷新状态";
    byId("error-settings").hidden = !retry;
    byId("question").hidden = !snapshot?.input_request;
    byId("question-text").textContent = snapshot?.input_request?.text || "";
    const activity = !sessionKey ? "" : state.closing ? "正在关闭，等待任务收尾…" : sending ? "正在发送…" : !snapshot ? "正在连接…" :
      !snapshot.accepts_input ? "会话已关闭" : snapshot.running
        ? snapshot.status || "正在处理…" : "";
    byId("activity").hidden = !activity || Boolean(error) || Boolean(snapshot?.input_request);
    byId("activity-text").textContent = activity;
    byId("spinner").hidden = Boolean(snapshot && !snapshot.accepts_input && !state.closing);
  }
  function render(message) {
    if (sessionKey !== message.sessionKey) {
      if (sessionKey) collapsed = true;
      persist();
      sessionKey = message.sessionKey;
      const restored = drafts.get(sessionKey) || {};
      closeContext();
      closeContextUsage();
      composer.set(restored);
      setCollapsed(Boolean(restored.collapsed));
      actionError = "";
      failedAction = "refresh";
      turns.clear();
      turnsElement.replaceChildren();
      byId("latest").hidden = true;
    }
    state = message.state;
    updateContextMeter(state.snapshot);
    if (message.expand) setCollapsed(false);
    renderSessions(message.chats);
    byId("session-tag").textContent = message.sessionId.slice(-6);
    byId("session-tag").title = message.sessionId;
    byId("collapsed-title").textContent = state.snapshot?.title || "当前对话";
    renderConversation(state.snapshot);
    const queued = state.snapshot?.queued_inputs || [];
    byId("queued").hidden = queued.length === 0;
    byId("queued").replaceChildren(...queued.map(item =>
      textElement("div", "", `${item.queue === "enqueue" ? "排队" : "补充"} · ${item.prompt}`)));
    controls();
    if (message.expand) {
      composer.focus();
      chatElement.scrollIntoView({ block: "nearest" });
    }
  }
  function send(queue) {
    if (byId("send").disabled) return;
    const parts = composer.parts();
    const text = Composer.text(parts);
    const pending = { requestId: crypto.randomUUID(), parts };
    pendingSends.set(sessionKey, pending);
    actionError = "";
    controls();
    post("submit", { requestId: pending.requestId, text, queue });
  }
  function clearSent(previous, pending) {
    const text = parts => Composer.text(parts.filter(part => part.type === "text"));
    const clearText = text(previous.parts) === text(pending.parts);
    const parts = [];
    const selection = { ...previous.selection };
    let offset = 0;
    for (const part of previous.parts) {
      const size = Composer.size([part]);
      const remove = part.type === "text" ? clearText : pending.parts.some(sent =>
        sent.type === "context" && sent.id === part.id && sent.text === part.text);
      if (remove) {
        for (const key of ["start", "end"]) {
          selection[key] -= Math.min(size, Math.max(0, previous.selection[key] - offset));
        }
      } else parts.push(part);
      offset += size;
    }
    return {
      ...previous, parts, selection,
    };
  }
  window.addEventListener("message", event => {
    const message = event.data;
    if (message.type === "state") render(message);
    else if (message.type === "insert-context") {
      const current = message.sessionKey === sessionKey;
      if (current) {
        closeContext();
        composer.insert(message.blocks);
        setCollapsed(false);
        byId("composer").scrollIntoView({ block: "nearest" });
      } else {
        const previous = drafts.get(message.sessionKey) || { parts: [] };
        const size = Composer.size(previous.parts);
        const { start, end } = previous.selection || { start: size, end: size };
        const blocks = message.blocks.map(block =>
          ({ ...block, type: "context", id: crypto.randomUUID() }));
        const cursor = start + blocks.length;
        drafts.set(message.sessionKey, {
          ...previous,
          parts: [
            ...Composer.slice(previous.parts, 0, start), ...blocks,
            ...Composer.slice(previous.parts, end, size),
          ],
          selection: { start: cursor, end: cursor },
        });
      }
      persist();
      controls();
    } else if (message.type === "sent") {
      const pending = pendingSends.get(message.sessionKey);
      if (pending?.requestId !== message.requestId) return;
      if (message.sessionKey === sessionKey) {
        composer.set(clearSent(currentDraft(), pending));
      } else {
        const previous = drafts.get(message.sessionKey);
        if (previous) drafts.set(message.sessionKey, clearSent(previous, pending));
      }
      pendingSends.delete(message.sessionKey);
      persist();
      controls();
    } else if (message.type === "action-error") {
      if (message.sessionKey === sessionKey) {
        actionError = message.error;
        failedAction = message.retry;
      }
      if (pendingSends.get(message.sessionKey)?.requestId === message.requestId) {
        pendingSends.delete(message.sessionKey);
      }
      controls();
    }
  });
  byId("composer").addEventListener("submit", event => { event.preventDefault(); send(false); });
  byId("context-editor").addEventListener("input", event => {
    composer.updateContext(editingContext, event.target.value);
  });
  byId("context-editor").addEventListener("keydown", event => {
    if (event.key === "Escape") { closeContext(); composer.focus(); }
  });
  byId("context-done").addEventListener("click", () => { closeContext(); composer.focus(); });
  byId("context-remove").addEventListener("click", () => {
    composer.removeContext(editingContext);
    closeContext();
  });
  byId("queue").addEventListener("click", () => send(true));
  byId("cancel-input").addEventListener("click", () => post("cancel-input"));
  byId("close-session").addEventListener("click", () => post("close-session", { targetSessionKey: sessionKey }));
  byId("refresh").addEventListener("click", () => {
    const action = actionError ? failedAction : "refresh";
    actionError = "";
    controls();
    post(action);
  });
  for (const action of ["new-session", "insert-context", "start-chat", "settings", "error-settings"]) {
    byId(action).addEventListener("click", () => {
      actionError = "";
      controls();
      post(action === "start-chat" ? "new-session" : action === "error-settings" ? "settings" : action);
    });
  }
  byId("collapse").addEventListener("click", () => setCollapsed(!collapsed));
  byId("collapsed-summary").addEventListener("click", () => setCollapsed(false));
  byId("latest").addEventListener("click", scrollToLatest);
  conversation.addEventListener("scroll", () => {
    if (nearBottom()) byId("latest").hidden = true;
  });
  document.querySelectorAll("[data-prompt]").forEach(button => {
    button.addEventListener("click", () => {
      composer.replace(0, Composer.size(composer.parts()), [{ type: "text", text: button.dataset.prompt }]);
    });
  });
  document.addEventListener("click", event => {
    const link = event.target.closest("a");
    if (!link) return;
    event.preventDefault();
    post("open-link", { href: link.getAttribute("href") });
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) post("ready");
  });
  setCollapsed(collapsed);
  controls();
  post("ready");
})();
