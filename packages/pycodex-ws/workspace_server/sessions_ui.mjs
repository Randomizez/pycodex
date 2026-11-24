import {messageSignature} from "./conversation.mjs";
import {PromptHistory} from "./content_views.mjs";
import {isNearBottom, readStorage, relativeUrl, scrollToBottom} from "./workspace_dom.mjs";

const POLL_TIMEOUT_MS = 15000;

export function formatReplyAge(session, now = Date.now()) {
  if (!Number.isFinite(session?.last_message_at)) return "";
  if (session.last_message_at <= 0) return "No activity";
  const minutes = Math.max(0, Math.floor((now - session.last_message_at * 1000) / 60000));
  if (minutes < 1) return "<1m ago";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    const remMinutes = minutes % 60;
    return remMinutes ? `${hours}h ${remMinutes}m ago` : `${hours}h ago`;
  }
  const days = Math.floor(hours / 24);
  const remHours = hours % 24;
  return remHours ? `${days}d ${remHours}h ago` : `${days}d ago`;
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
  renderConversation, addEntry, dismissSessionsDrawer, closeContextBlock, resizePrompt,
  beginWorkspacePoll = null,
}) {
  const workspaceTitle = document.title;
  let activeSessionId = readStorage("sessionStorage", activeSessionKey);
  let sessions = [];
  const sessionState = new Map();
  let sessionPoll = null;
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

  function updateSessions(nextSessions) {
    const incomingSessions = Array.isArray(nextSessions) ? nextSessions : sessions;
    incomingSessions.forEach((session) => {
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
    sessions = incomingSessions;
    let fullSessionIds = null;
    if (Array.isArray(nextSessions)) {
      fullSessionIds = new Set(sessions.map(session => session.id));
      for (const id of sessionState.keys()) {
        if (id !== activeSessionId && !fullSessionIds.has(id)) {
          sessionState.delete(id);
          clearSessionBanners(id);
        }
      }
    }
    document.title = (sessions.some(session => session.running) ? "● " : "") + workspaceTitle;
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
    let timedOut = false;
    const pollTimeout = setTimeout(() => {
      timedOut = true;
      request.controller.abort();
    }, POLL_TIMEOUT_MS);
    if (sessionNeedsRefresh) setSessionRefresh("loading");
    const finishWorkspacePoll = beginWorkspacePoll ? beginWorkspacePoll() : null;
    try {
      const sessionUrl = sessionId => {
        const params = new URLSearchParams();
        if (sessionId) params.set("session_id", sessionId);
        if (finishWorkspacePoll) params.set("workspace", "true");
        const query = params.toString();
        return relativeUrl(query ? `api/session?${query}` : "api/session");
      };
      let response = await fetch(sessionUrl(requestedSessionId), {signal: request.controller.signal});
      if (sessionPoll !== request || requestedSessionId !== activeSessionId) return;
      // A saved session may have been closed since this page was last open.
      if (response.status === 404 && requestedSessionId && !sessions.length) {
        response = await fetch(sessionUrl(""), {signal: request.controller.signal});
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      if (finishWorkspacePoll) finishWorkspacePoll(payload);
      if (sessionPoll !== request || requestedSessionId !== activeSessionId) {
        return;
      }
      if (payload.session_id && payload.session_id !== activeSessionId) {
        activeSessionId = payload.session_id;
      }
      stateForSession(activeSessionId).externalCall = payload.external_call || null;
      updateSessions(payload.sessions || []);
      renderSnapshot(payload.snapshot);
    } catch (error) {
      if (sessionPoll === request && requestedSessionId === activeSessionId) {
        if (sessionNeedsRefresh) setSessionRefresh("error");
        setSpinner(`poll error: ${timedOut ? "request timed out" : error.message}`);
      }
    } finally {
      clearTimeout(pollTimeout);
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
    if (!sessionById(sessionId) || sessions.length <= 1) return;
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
    if (sessionState.delete(sessionId)) {
      clearSessionBanners(sessionId);
    }
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
    createSession, closeSession, switchSession,
  };
}
