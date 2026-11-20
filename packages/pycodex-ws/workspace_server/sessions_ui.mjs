import {messageSignature} from "./conversation.mjs";
import {PromptHistory} from "./content_views.mjs";
import {isNearBottom, relativeUrl, scrollToBottom} from "./workspace_dom.mjs";

function sessionActivity(session, stateById) {
  const lastUserAt = session.last_user_at ??
    stateById.get(session.id)?.legacyReplyTiming?.timing?.last_user_at;
  if (lastUserAt != null) return lastUserAt;
  const match = /^([0-9a-f]{8})-([0-9a-f]{4})-7/i.exec(session.last_final_id || "");
  return match ? Number.parseInt(match[1] + match[2], 16) / 1000 : 0;
}

function orderBySessionActivity(items, stateById) {
  const times = new Map(items.map(session => [session.id, sessionActivity(session, stateById)]));
  return [...items].sort((left, right) =>
    times.get(right.id) - times.get(left.id) ||
    String(right.id).localeCompare(String(left.id))
  );
}

function recentSessions(items, activeId, stateById) {
  const ordered = orderBySessionActivity(items, stateById);
  const visible = ordered.slice(0, 20);
  const active = ordered.find(session => session.id === activeId);
  if (active && !visible.includes(active)) visible[visible.length - 1] = active;
  return visible;
}

export function keepMobileSessionOrder(items, previousIds) {
  const remaining = new Map(items.map(session => [session.id, session]));
  const slots = previousIds.map(id => {
    const session = remaining.get(id);
    remaining.delete(id);
    return session;
  });
  const additions = remaining.values();
  return slots.map(session => session ?? additions.next().value)
    .concat(Array.from(additions)).filter(Boolean);
}

function replyTimingFromSnapshot(snapshot) {
  let firstPromptAt = null;
  let replyAt = null;
  let hasReply = false;
  let lastUserAt = 0;
  for (const turn of snapshot?.turns || []) {
    if (turn.kind === "control" || turn.kind === "compaction") continue;
    firstPromptAt ||= turn.prompt_at;
    const timestamps = [turn.prompt_at,
      ...(turn.messages || []).map(message => message.created_at)];
    for (const timestamp of timestamps) {
      const milliseconds = Date.parse(timestamp);
      if (Number.isFinite(milliseconds)) lastUserAt = Math.max(lastUserAt, milliseconds / 1000);
    }
    if (String(turn.response || "").trim()) {
      replyAt = turn.response_at;
      hasReply = true;
    }
  }
  return {has_reply: hasReply, reply_at: hasReply ? replyAt : firstPromptAt, last_user_at: lastUserAt};
}

export function formatReplyAge(session, now = Date.now()) {
  if (typeof session?.has_reply !== "boolean") return "读取回复时间…";
  if (!session.reply_at) return session.has_reply ? "回复时间未知" : "尚未回复";
  const timestamp = Date.parse(session.reply_at);
  if (!Number.isFinite(timestamp)) return "回复时间未知";
  const seconds = Math.max(0, Math.floor((now - timestamp) / 1000));
  if (seconds < 5 && session.has_reply) return "刚刚回复";
  if (seconds < 60) return `${seconds} 秒未回复`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} 分钟未回复`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时 ${minutes % 60} 分钟未回复`;
  return `${Math.floor(hours / 24)} 天 ${hours % 24} 小时未回复`;
}

function hasFrontendCustomizationTurn(turns) {
  return Array.isArray(turns) && turns.some(turn =>
    Array.isArray(turn.messages) && turn.messages.some(message =>
      typeof message.text === "string" &&
      message.text.startsWith("<frontend_customization>\n") &&
      message.text.endsWith("\n</frontend_customization>")
    )
  );
}

export function createSessionsUI({
  log, composer, activeSessionKey, renderTabs, updatePromptPlaceholder, updateContextMeter,
  syncSessionBanners = () => {}, clearSessionBanners = () => {},
  updateSpinnerNotifyUi, notifySessionDone, setSpinner,
  renderQueueHint, markVisibleFinalRead,
  renderConversation, addEntry, dismissSessionsDrawer, closeContextBlock, resizePrompt, isSidebarOpen,
}) {
  const workspaceTitle = document.title;
  let activeSessionId = window.sessionStorage.getItem(activeSessionKey) || "";
  let sessions = [];
  const sessionState = new Map();
  let sessionPoll = null;
  let legacyReplyRequests = 0;
  let sessionNeedsRefresh = true;
  let restoreScrollTop = null;
  let suppressScrollSave = false;

  function sessionById(sessionId) {
    return sessions.find((session) => (session.id || "") === sessionId) || null;
  }

  function restoreSessionScroll() {
    if (!activeSessionId || !log.clientHeight) return;
    const state = stateForSession(activeSessionId);
    if (!state.lastRenderedSignature) return;
    if (state.scrollTop === null) scrollToBottom(log);
    else log.scrollTop = state.scrollTop;
    restoreScrollTop = null;
    state.scrollTop = log.scrollTop;
  }

  function stateForSession(sessionId) {
    const key = String(sessionId || "");
    if (!sessionState.has(key)) {
      sessionState.set(key, {
        lastRenderedSignature: "",
        entries: [],
        scrollTop: null,
        draft: {parts: []},
        promptHistory: new PromptHistory(),
        notifyWhenDone: false,
        wasRunning: false,
        inputRequest: null,
        queuedInputs: [],
        hintStartedAt: null,
        hintStartIndex: 0,
        readingFinalId: "",
        externalCall: null,
        customizeAgent: false,
      });
    }
    return sessionState.get(key);
  }

  function saveActiveSessionUiState() {
    if (!activeSessionId) return;
    const state = stateForSession(activeSessionId);
    if (log.scrollHeight > log.clientHeight) {
      state.scrollTop = log.scrollTop;
    }
    state.draft = composer.read();
    state.entries = Array.from(log.children);
  }

  function sessionReplyTiming(session) {
    if (session.last_user_at != null) return session;
    if (session.turn_count === 0) return {has_reply: false};
    const state = stateForSession(session.id);
    const key = JSON.stringify([session.turn_count, session.last_final_id, session.running]);
    const previous = state.legacyReplyTiming;
    if (isSidebarOpen() && legacyReplyRequests < 3 && !previous?.pending
        && !(previous?.retryAfter > Date.now()) && previous?.key !== key) {
      const entry = state.legacyReplyTiming = {pending: true, timing: previous?.timing};
      legacyReplyRequests++;
      fetch(relativeUrl(`api/session?session_id=${encodeURIComponent(session.id)}`))
        .then(response => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return response.json();
        })
        .then(payload => {
          entry.timing = replyTimingFromSnapshot(payload.snapshot);
          entry.key = key;
        })
        .catch(() => { entry.retryAfter = Date.now() + 10000; })
        .finally(() => {
          entry.pending = false;
          legacyReplyRequests--;
          if (isSidebarOpen()) renderTabs();
        });
    }
    return previous?.timing ?? session;
  }

  function orderVisibleSessions() {
    if (sessions.every(session => session.last_user_at != null)) return sessions;
    return orderBySessionActivity(sessions, sessionState);
  }

  function updateSessions(nextSessions, workspaceRunning) {
    const incomingSessions = Array.isArray(nextSessions) ? nextSessions : sessions;
    const ordered = recentSessions(incomingSessions, activeSessionId, sessionState);
    ordered.forEach((session) => {
      const sessionId = session.id || "";
      if (!sessionId) return;
      const state = stateForSession(sessionId);
      const wasRunning = Boolean(state.wasRunning);
      const isRunning = Boolean(session.running);
      if (!wasRunning && isRunning) {
        state.notifyWhenDone = false;
      } else if (wasRunning && !isRunning) {
        notifySessionDone(sessionId, session);
      }
      state.wasRunning = isRunning;
    });
    sessions = ordered;
    let fullSessionIds = null;
    if (Array.isArray(nextSessions) && workspaceRunning === undefined) {
      fullSessionIds = new Set(incomingSessions.map(session => session.id));
      for (const id of sessionState.keys()) {
        if (id !== activeSessionId && !fullSessionIds.has(id)) {
          sessionState.delete(id);
          clearSessionBanners(id);
        }
      }
    }
    const running = workspaceRunning ?? incomingSessions.some(session => session.running);
    document.title = (running ? "● " : "") + workspaceTitle;
    if (!activeSessionId && sessions.length) {
      activeSessionId = sessions[0].id || "";
    }
    renderTabs();
    syncSessionBanners({fullSessionIds});
    updatePromptPlaceholder();
    updateSessionDirectory();
    updateContextMeter();
    updateSpinnerNotifyUi();
  }

  function updateSessionDirectory(snapshot = null) {
    const session = snapshot || sessionById(activeSessionId);
    const cwd = session ? session.cwd || "" : "";
    const label = document.getElementById("sessionCwd");
    label.hidden = !cwd || cwd === label.dataset.defaultCwd;
    label.querySelector(".session-cwd-name").textContent = cwd.split(/[\\/]/).filter(Boolean).pop() || cwd;
    label.title = `Current dir: ${cwd}\nUse /cd <directory> to change directory`;
    const tabs = document.getElementById("contentTabs");
    tabs.dataset.boardTitle = `Workspace: ${workspaceTitle}\nCurrent dir: ${cwd || label.dataset.defaultCwd}\nCanvas file: ${tabs.dataset.boardPath}`;
    const canvasTab = tabs.querySelector('[data-view-id="board"]');
    if (canvasTab) canvasTab.title = tabs.dataset.boardTitle;
  }

  function setSessionRefresh(status) {
    sessionNeedsRefresh = status !== "ready";
    const notice = document.getElementById("sessionRefresh");
    notice.hidden = !sessionNeedsRefresh;
    notice.dataset.state = status;
    notice.textContent = status === "loading" ? "Loading latest session…"
      : status === "error" ? "Unable to load latest session. Retrying…" : "";
  }

  async function pollSession() {
    const requestedSessionId = activeSessionId;
    if (sessionPoll) {
      if (sessionPoll.sessionId === requestedSessionId) return;
      sessionPoll.controller.abort();
    }
    const request = {sessionId: requestedSessionId, controller: new AbortController()};
    sessionPoll = request;
    if (sessionNeedsRefresh) setSessionRefresh("loading");
    try {
      const path = requestedSessionId
        ? `api/session?recent=true&session_id=${encodeURIComponent(requestedSessionId)}`
        : "api/session?recent=true";
      let response = await fetch(relativeUrl(path), {signal: request.controller.signal});
      if (sessionPoll !== request || requestedSessionId !== activeSessionId) return;
      // A saved session may have been closed since this page was last open.
      if (response.status === 404 && requestedSessionId && !sessions.length) {
        response = await fetch(relativeUrl("api/session?recent=true"), {signal: request.controller.signal});
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      if (sessionPoll !== request || requestedSessionId !== activeSessionId) {
        return;
      }
      if (payload.session_id && payload.session_id !== activeSessionId) {
        activeSessionId = payload.session_id;
      }
      stateForSession(activeSessionId).externalCall = payload.external_call || null;
      const active = payload.sessions?.find(session => session.id === activeSessionId);
      if (active && active.last_user_at == null) {
        Object.assign(active, replyTimingFromSnapshot(payload.snapshot));
      }
      updateSessions(payload.sessions || [], payload.workspace_running);
      renderSnapshot(payload.snapshot);
    } catch (error) {
      if (sessionPoll === request && requestedSessionId === activeSessionId) {
        if (sessionNeedsRefresh) setSessionRefresh("error");
        setSpinner(`poll error: ${error.message}`);
      }
    } finally {
      if (sessionPoll === request) sessionPoll = null;
    }
  }

  function renderSnapshot(snapshot) {
    if (!snapshot) return;
    setSessionRefresh("ready");
    updatePromptPlaceholder(snapshot);
    updateSessionDirectory(snapshot);
    const state = stateForSession(activeSessionId);
    state.inputRequest = snapshot.input_request || null;
    state.queuedInputs = snapshot.queued_inputs;
    state.promptHistory.update(snapshot.turns);
    if (!state.customizeAgent && hasFrontendCustomizationTurn(snapshot.turns)) {
      state.customizeAgent = true;
      syncSessionBanners();
    }
    renderQueueHint(snapshot.queued_inputs);
    requestAnimationFrame(markVisibleFinalRead);
    const signature = messageSignature(snapshot);
    if (signature === state.lastRenderedSignature) {
      setSpinner(snapshot.spinner || "");
      updateContextMeter(snapshot);
      return;
    }
    state.lastRenderedSignature = signature;
    const shouldStick = state.scrollTop === null || isNearBottom(log);
    const previousScrollTop = log.scrollTop;
    const viewportTop = log.getBoundingClientRect().top;
    const anchor = !shouldStick && restoreScrollTop === null
      ? Array.from(log.children).find(entry => entry.getBoundingClientRect().bottom > viewportTop)
      : null;
    const anchorTop = anchor ? anchor.getBoundingClientRect().top : 0;
    suppressScrollSave = true;
    try {
      const turns = Array.isArray(snapshot.turns) ? snapshot.turns : [];
      renderConversation(activeSessionId, turns, state.inputRequest);
      setSpinner(snapshot.spinner || "");
      updateContextMeter(snapshot);
      // Hidden mobile panes have no layout; restore their scroll when shown.
      if (!log.clientHeight) return;
      if (restoreScrollTop !== null) {
        log.scrollTop = Math.min(
          restoreScrollTop,
          Math.max(0, log.scrollHeight - log.clientHeight),
        );
        restoreScrollTop = null;
      } else if (shouldStick) {
        scrollToBottom(log);
      } else if (anchor && anchor.parentElement === log) {
        const shift = anchor.getBoundingClientRect().top - anchorTop;
        if (shift) log.scrollTop += shift;
      } else {
        log.scrollTop = Math.min(
          previousScrollTop,
          Math.max(0, log.scrollHeight - log.clientHeight),
        );
      }
    } finally {
      suppressScrollSave = false;
    }
    state.scrollTop = log.scrollTop;
  }

  async function switchSession(sessionId) {
    dismissSessionsDrawer();
    if (!sessionId || sessionId === activeSessionId) return;
    saveActiveSessionUiState();
    activeSessionId = sessionId;
    setSessionRefresh("loading");
    renderTabs();
    updatePromptPlaceholder();
    updateSessionDirectory();
    updateContextMeter();
    const state = stateForSession(sessionId);
    syncSessionBanners({collapse: true});
    closeContextBlock();
    composer.set(state.draft);
    resizePrompt();
    suppressScrollSave = true;
    log.replaceChildren(...state.entries);
    renderQueueHint(state.queuedInputs);
    suppressScrollSave = false;
    setSpinner(sessionById(sessionId)?.spinner || "", {suppressDoneNotification: true});
    restoreScrollTop = state.scrollTop;
    restoreSessionScroll();
    await pollSession();
    renderTabs();
  }

  async function createSession(options = {}) {
    const response = await fetch(relativeUrl("api/sessions"), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(options),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || `HTTP ${response.status}`);
    }
    saveActiveSessionUiState();
    activeSessionId = payload.session_id || "";
    const state = stateForSession(activeSessionId);
    state.draft = {parts: []};
    state.scrollTop = null;
    state.lastRenderedSignature = "";
    state.externalCall = null;
    syncSessionBanners({collapse: true});
    closeContextBlock();
    composer.set(state.draft);
    resizePrompt();
    restoreScrollTop = null;
    updateSessions(payload.sessions || []);
    renderSnapshot(payload.snapshot);
    dismissSessionsDrawer();
    return payload.session_id;
  }

  async function closeSession(sessionId) {
    const session = sessionById(sessionId);
    if (!session || sessions.length <= 1) return;
    if (!window.confirm(`Close session “${session.title || "pycodex"}”? It will be removed from this workspace.`)) return;
    const response = await fetch(relativeUrl(`api/sessions/${encodeURIComponent(sessionId)}`), {
      method: "DELETE",
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      addEntry("error", payload.error || `HTTP ${response.status}`);
      return;
    }
    updateSessions(payload.sessions || []);
    if (activeSessionId === sessionId) {
      await switchSession(sessions[0].id);
    }
    sessionState.delete(sessionId);
    clearSessionBanners(sessionId);
  }

  function saveScroll() {
    if (suppressScrollSave || !log.clientHeight || !activeSessionId) return;
    const state = stateForSession(activeSessionId);
    if (!state.lastRenderedSignature) return;
    state.scrollTop = log.scrollTop;
    requestAnimationFrame(markVisibleFinalRead);
  }

  return {
    get activeId() { return activeSessionId; },
    get sessions() { return sessions; },
    sessionState, sessionById, stateForSession,
    restoreSessionScroll, saveActiveSessionUiState, saveScroll, updateSessions, renderSnapshot, pollSession,
    createSession, closeSession, switchSession, sessionReplyTiming, orderVisibleSessions,
  };
}
