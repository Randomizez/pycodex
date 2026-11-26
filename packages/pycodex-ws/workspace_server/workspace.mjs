import {Composer, mountContentViews} from "./content_views.mjs";
import {
  encodeViewPath, frontendUrl, readStorage, reconcileChildren, relativeUrl, writeStorage,
} from "./workspace_dom.mjs";
import {createMarkdownRenderer, createViewLinkHandlers} from "./markdown.mjs";
import {createConversationRenderer} from "./conversation.mjs";
import {createSessionsUI, formatReplyAge} from "./sessions_ui.mjs";
import {fillRuntimeInfo, installDialogs, setupContextDetails} from "./dialogs.mjs";
import {mountCustomFrontend} from "./custom_frontend.mjs";
import {createSessionBanners, loadEmbeddedBanners} from "./banners.mjs";

const log = document.getElementById("log");
const spinner = document.getElementById("spinner");
const notifyToggle = document.getElementById("notifyToggle");
const queueHint = document.getElementById("queueHint");
const steerHint = document.getElementById("steerHint");
const enqueueHint = document.getElementById("enqueueHint");
const toast = document.getElementById("toast");
const prompt = document.getElementById("prompt");
const promptHint = document.getElementById("promptHint");
const form = document.getElementById("composer");
const contextBlockDetail = document.getElementById("contextBlockDetail");
const contextBlockLabel = document.getElementById("contextBlockLabel");
const contextBlockEditor = document.getElementById("contextBlockEditor");
const boardFrame = document.getElementById("boardFrame");
const vscodeFrame = document.getElementById("vscodeFrame");
const vscodeToggle = document.getElementById("vscodeToggle");
const enableBoard = document.body.dataset.enableBoard === "true";
const isAdmin = document.body.dataset.accessRole === "admin";
const enableVscode = isAdmin && document.body.dataset.enableVscode === "true";
const VSCODE_PASSWORD_HINT = "Start the workspace server with --password to enable VS Code.";
const workspace = document.querySelector(".workspace");
const splitter = document.getElementById("splitter");
const tabsEl = document.getElementById("tabs");
const activeSessionTitle = document.getElementById("activeSessionTitle");
const activeSessionStatus = document.getElementById("activeSessionStatus");
const sessionSidebar = document.getElementById("sessionSidebar");
const sessionsClose = document.getElementById("sessionsClose");
const sessionsBackdrop = document.getElementById("sessionsBackdrop");
const mobileSessionsMedia = window.matchMedia("(max-width: 760px)");
const mobileSessionFloat = document.getElementById("mobileSessionFloat");
const mobileSessionDots = document.querySelector(".mobile-session-dots");
const mobileSessionsToggle = document.querySelector(".mobile-sessions-toggle");
let sessionsOpen = false;
let sessionsPinned = false;
let sessionsHoverTimer = null;
let sessionCloseRequest = null;
let closingSession = false;
const contextMeter = document.getElementById("contextMeter");
const contextMeterValue = document.getElementById("contextMeterValue");
const activeSessionKey = "pycodex.workspace.activeSession:" + window.location.pathname;
const customizeSessionsKey = "pycodex.workspace.customizeSessions:" + window.location.pathname;
const builtinFrontend = new URLSearchParams(window.location.search).get("builtin") === "1";
function readStoredList(key) {
  try {
    const parsed = JSON.parse(readStorage("sessionStorage", key, "[]"));
    return Array.isArray(parsed) ? parsed : [];
  } catch (_error) {
    return [];
  }
}
const customizeSessions = new Set(readStoredList(customizeSessionsKey));
let pollTimer = null;
let lastBoardSignature = null;
let contentViewsActive = false;
let resizeStart = null;
let spinnerText = "";
let toastTimer = null;
let notificationSwPromise = null;
const frontendEvents = new EventTarget();
let lastFrontendSessionId = "";
let editingContextId = null;
const composer = new Composer(prompt, () => {
  const draft = composer.read();
  stateForSession(sessionUI.activeId).draft = draft;
  prompt.classList.toggle("empty", !draft.parts.length);
  promptHint.hidden = !prompt.classList.contains("show-hint") || draft.parts.length > 0;
  resizePrompt();
}, () => form.requestSubmit(), editContextBlock, showToast,
  (back, current) => stateForSession(sessionUI.activeId).promptHistory.recall(back, current));
prompt.addEventListener("input", () => stateForSession(sessionUI.activeId).promptHistory.reset());
const markdown = createMarkdownRenderer(log);
const {renderMarkdownInto} = markdown;
const {renderConversation, addEntry} = createConversationRenderer({
  log, renderMarkdownInto, builtinFrontend,
});
const banners = createSessionBanners({
  container: document.getElementById("sessionBanners"),
  log,
  getActiveSessionId: () => sessionUI.activeId,
  isAdmin: () => document.body.dataset.accessRole === "admin",
  openView: (...args) => contentTabs.open(...args),
  showToast,
});
const sessionUI = createSessionsUI({
  log, composer, activeSessionKey, renderTabs, updatePromptPlaceholder, updateContextMeter,
  syncSessionBanners, clearSessionBanners,
  updateSpinnerNotifyUi, notifySessionDone, setSpinner,
  renderQueueHint, markVisibleFinalRead,
  renderConversation, addEntry,
  dismissSessionsDrawer, closeContextBlock, resizePrompt,
  beginWorkspacePoll,
});
const {
  sessionById, stateForSession, restoreSessionScroll,
  updateSessions, renderSnapshot, pollSession, createSession, closeSession, switchSession,
} = sessionUI;
composer.set({parts: []});

function loadChatWidth() {
  if (mobileSessionsMedia.matches) return;
  const stored = Number(readStorage("localStorage", "pycodex.workspace.chatWidth", "0"));
  if (stored > 0) setChatWidth(stored, false);
}

function isSidebarDocked() {
  return !mobileSessionsMedia.matches && sessionsOpen && !workspace.classList.contains("sessions-overlay");
}

function syncSidebarOverlay() {
  if (!sessionsOpen || mobileSessionsMedia.matches) {
    workspace.classList.remove("sessions-overlay");
    return;
  }
  if (!sessionsPinned) {
    workspace.classList.add("sessions-overlay");
    return;
  }
  const chatWidth = document.querySelector(".chat").getBoundingClientRect().width;
  const openSize = parseFloat(getComputedStyle(workspace).getPropertyValue("--session-sidebar-size")) || 248;
  const reservedWidth = isSidebarDocked() ? openSize : sessionSidebarWidth();
  workspace.classList.toggle("sessions-overlay", chatWidth + reservedWidth - openSize < 320);
}

function centerSplit() {
  const sidebarWidth = parseFloat(getComputedStyle(workspace).getPropertyValue(
    isSidebarDocked() ? "--session-sidebar-size" : "--session-rail-width"
  ));
  setChatWidth(Math.max(0, (window.innerWidth - sidebarWidth - splitterWidth()) / 2) +
    sidebarWidth - sessionSidebarWidth());
}

function splitterWidth() {
  return parseFloat(getComputedStyle(workspace).getPropertyValue("--splitter-width")) || 0;
}

function sessionSidebarWidth() {
  return parseFloat(getComputedStyle(workspace).getPropertyValue("--session-rail-width"));
}

function clearSessionsHover() {
  clearTimeout(sessionsHoverTimer);
  sessionsHoverTimer = null;
}

function syncMobileSessionFloat() {
  const floatHidden = (sessionsOpen && mobileSessionsMedia.matches) ||
    !mobileSessionsMedia.matches ||
    workspace.dataset.mobilePane === "board";
  mobileSessionFloat.inert = floatHidden;
  mobileSessionFloat.setAttribute("aria-hidden", String(floatHidden));
}

function setSessionsOpen(open, restoreFocus = false, pinned = true) {
  clearSessionsHover();
  if (!open) dismissSessionClose();
  sessionsOpen = open;
  sessionsPinned = open && pinned;
  workspace.classList.toggle("sessions-collapsed", !open);
  syncSidebarOverlay();
  updateSessionControls();
  const modal = open && mobileSessionsMedia.matches;
  const hidden = !open && mobileSessionsMedia.matches;
  syncMobileSessionFloat();
  sessionSidebar.inert = hidden;
  sessionSidebar.setAttribute("aria-hidden", String(hidden));
  workspace.classList.toggle("sessions-drawer-open", modal);
  sessionsBackdrop.inert = !modal;
  sessionsBackdrop.setAttribute("aria-hidden", String(!modal));
  sessionSidebar.setAttribute("role", modal ? "dialog" : "complementary");
  if (modal) sessionSidebar.setAttribute("aria-modal", "true");
  else sessionSidebar.removeAttribute("aria-modal");
  for (const pane of workspace.querySelectorAll(".board-pane, .chat, .splitter")) {
    pane.inert = modal;
  }
  if (modal) sessionsClose.focus({preventScroll: true});
  else if (restoreFocus) {
    const button = mobileSessionsMedia.matches ? mobileSessionsToggle : sessionsClose;
    button.focus({preventScroll: true});
  }
  if (open) renderTabs();
  requestAnimationFrame(() => {
    restoreSessionScroll();
    if (!open) markVisibleFinalRead();
  });
}

function updateSessionControls() {
  const runningCount = sessionUI.sessions.filter(session => session.running).length;
  const status = runningCount ? ` · ${runningCount} running` : "";
  sessionsClose.title = (sessionsOpen ? "Collapse sessions" : "Expand sessions") + status;
  sessionsClose.setAttribute("aria-label", sessionsClose.title);
  sessionsClose.setAttribute("aria-expanded", String(sessionsOpen));
  sessionsClose.querySelector("path").setAttribute("d", sessionsOpen
    ? "M15 3v18M8 9l3 3-3 3"
    : "M15 3v18M18 8h1M18 12h1M18 16h1");
  mobileSessionsToggle.title = (sessionsOpen ? "Close sessions" : "Show sessions") + status;
  mobileSessionsToggle.setAttribute("aria-label", mobileSessionsToggle.title);
  mobileSessionsToggle.setAttribute("aria-expanded", String(sessionsOpen));
}

function dismissSessionsDrawer() {
  if (mobileSessionsMedia.matches && sessionsOpen) setSessionsOpen(false, true);
}

async function selectSession(sessionId) {
  workspace.dataset.mobilePane = "chat";
  if (!sessionsPinned) setSessionsOpen(false);
  await switchSession(sessionId);
  resizePrompt();
  restoreSessionScroll();
}

function setChatWidth(width, persist = true) {
  const viewportWidth = window.innerWidth || 0;
  const gutterWidth = Math.min(24, Math.max(12, Math.round(viewportWidth * 0.06)));
  const availableWidth = Math.max(0, viewportWidth - sessionSidebarWidth() - splitterWidth());
  const maxChatWidth = Math.max(gutterWidth, availableWidth - gutterWidth);
  const collapseWidth = Math.min(
    280,
    Math.max(120, Math.round(viewportWidth * 0.25)),
  );
  let nextWidth = Math.max(gutterWidth, Math.min(maxChatWidth, Math.round(width)));
  let collapsed = false;
  if (nextWidth < collapseWidth) {
    nextWidth = gutterWidth;
    collapsed = true;
  } else if (availableWidth - nextWidth < collapseWidth) {
    nextWidth = maxChatWidth;
    collapsed = true;
  }
  workspace.style.setProperty("--chat-width", `${nextWidth}px`);
  workspace.classList.toggle("collapsed", collapsed);
  syncSidebarOverlay();
  if (persist) writeStorage("localStorage", "pycodex.workspace.chatWidth", nextWidth);
}

function startResize(event) {
  if (!workspace) return;
  const openSize = parseFloat(getComputedStyle(workspace).getPropertyValue("--session-sidebar-size")) || 248;
  const chatWidth = document.querySelector(".chat").getBoundingClientRect().width +
    (isSidebarDocked() ? openSize - sessionSidebarWidth() : 0);
  resizeStart = { pointerId: event.pointerId, startX: event.clientX, startWidth: chatWidth };
  splitter.setPointerCapture(event.pointerId);
  workspace.classList.add("resizing");
  event.preventDefault();
}

function updateResize(event) {
  if (!resizeStart) return;
  const delta = resizeStart.startX - event.clientX;
  setChatWidth(resizeStart.startWidth + delta);
  event.preventDefault();
}

function endResize(event) {
  if (!resizeStart) return;
  try {
    splitter.releasePointerCapture(resizeStart.pointerId);
  } catch (_error) {
    // Pointer capture can already be released by the browser.
  }
  resizeStart = null;
  workspace.classList.remove("resizing");
  event.preventDefault();
}

function setSpinner(text, options = {}) {
  const previousText = spinnerText;
  const nextText = String(text || "").trim();
  const isError = nextText.startsWith("poll error:");
  const wasRunning = Boolean(previousText && !previousText.startsWith("poll error:"));
  const isRunning = Boolean(nextText && !isError);
  spinnerText = nextText;
  spinner.textContent = spinnerText;
  if (isError) spinner.dataset.state = "error";
  else delete spinner.dataset.state;
  if (wasRunning && !isRunning && !isError && !options.suppressDoneNotification) {
    const state = stateForSession(sessionUI.activeId);
    if (state.wasRunning) notifySessionDone(sessionUI.activeId);
  }
  updateSpinnerNotifyUi();
}

function showToast(text) {
  if (toastTimer) clearTimeout(toastTimer);
  toast.textContent = text;
  toast.classList.add("visible");
  toastTimer = setTimeout(() => {
    toastTimer = null;
    toast.classList.remove("visible");
  }, 1400);
}

function updateSpinnerNotifyUi() {
  const running = Boolean(spinnerText && !spinnerText.startsWith("poll error:"));
  const enabled = Boolean(running && stateForSession(sessionUI.activeId).notifyWhenDone);
  const title = !running ? "" : enabled
    ? "Completion notification on — click to cancel"
    : "Click to notify when done";
  spinner.title = title;
  if (!notifyToggle) return;
  notifyToggle.hidden = !running;
  notifyToggle.classList.toggle("notify-enabled", enabled);
  notifyToggle.setAttribute("aria-checked", enabled ? "true" : "false");
  notifyToggle.title = title;
}

function notificationAvailable() {
  return "Notification" in window;
}

function requestNotificationPermission() {
  try {
    const requestResult = Notification.requestPermission();
    if (requestResult && typeof requestResult.then === "function") {
      return requestResult;
    }
    return Promise.resolve(Notification.permission);
  } catch (_error) {
    return Promise.resolve(Notification.permission);
  }
}

function ensureNotificationSw() {
  if (!("serviceWorker" in navigator)) return Promise.resolve(null);
  if (!notificationSwPromise) {
    notificationSwPromise = navigator.serviceWorker.register(relativeUrl("sw.js"))
      .then(() => navigator.serviceWorker.ready)
      .catch(() => {
        notificationSwPromise = null;
        return null;
      });
  }
  return notificationSwPromise;
}

function plainTextPreview(markdownText, limit = 160) {
  const raw = String(markdownText || "").trim();
  if (!raw) return "";
  const container = document.createElement("div");
  renderMarkdownInto(container, raw);
  const text = (container.textContent || "").replace(/\s+/g, " ").trim();
  return text.length > limit ? text.slice(0, limit - 1).trimEnd() + "…" : text;
}

async function showSessionNotification(label, body, sessionId) {
  const tag = `pycodex-workspace-done-${sessionId}`;
  try {
    const notice = new Notification(label, {body, tag});
    notice.onclick = () => {
      window.focus();
      if (sessionUI.activeId !== sessionId) selectSession(sessionId);
      notice.close();
    };
    return;
  } catch (_error) {
    // Mobile Chromium (Android Edge/Chrome) and standalone PWAs require ServiceWorkerRegistration.showNotification().
  }
  const registration = await ensureNotificationSw();
  if (!registration || typeof registration.showNotification !== "function") return;
  try {
    await registration.showNotification(label, {
      body,
      tag,
      data: {sessionId, url: window.location.href},
    });
  } catch (_error) {
    // Notification display is best-effort only.
  }
}

function notifySessionDone(sessionId, sessionSummary = null) {
  if (!sessionId) return;
  const state = stateForSession(sessionId);
  if (!state.notifyWhenDone) return;
  state.notifyWhenDone = false;
  renderTabs();
  updateSpinnerNotifyUi();
  if (!notificationAvailable() || Notification.permission !== "granted") return;
  const session = sessionSummary || sessionById(sessionId);
  const title = (session && session.title) || "pycodex";
  const label = sessionUI.workspaceTitle && sessionUI.workspaceTitle !== title
    ? `${title} · ${sessionUI.workspaceTitle}`
    : title;
  const body = plainTextPreview(session && (session.last_notice || session.last_assistant)) || "Done.";
  showSessionNotification(label, body, sessionId);
}

function isSessionRunning(sessionId) {
  if (sessionId === sessionUI.activeId && spinnerText && !spinnerText.startsWith("poll error:")) return true;
  const session = sessionById(sessionId);
  if (session) return Boolean(session.running);
  return false;
}

async function toggleSpinnerNotification() {
  const sessionId = sessionUI.activeId;
  if (!sessionId || !isSessionRunning(sessionId)) {
    showToast("Only available while running");
    return;
  }
  const state = stateForSession(sessionId);
  if (state.notifyWhenDone) {
    state.notifyWhenDone = false;
    showToast("Notification canceled");
    updateSpinnerNotifyUi();
    renderTabs();
    return;
  }
  if (!notificationAvailable()) {
    showToast(window.isSecureContext === false
      ? "Notifications require HTTPS or localhost"
      : "Browser notifications unavailable");
    return;
  }
  if (Notification.permission === "denied") {
    showToast("Browser notifications are blocked");
    return;
  }
  state.notifyWhenDone = true;
  showToast("Will notify when done");
  updateSpinnerNotifyUi();
  renderTabs();
  if (Notification.permission !== "granted") {
    const permission = await requestNotificationPermission();
    if (permission !== "granted") {
      state.notifyWhenDone = false;
      showToast("Notification permission not granted");
      updateSpinnerNotifyUi();
      renderTabs();
      return;
    }
  }
  if (!isSessionRunning(sessionId)) {
    state.notifyWhenDone = false;
    showToast("Session already finished");
  } else {
    ensureNotificationSw();
  }
  updateSpinnerNotifyUi();
  renderTabs();
}

function resizePrompt() {
  if (prompt.style.height) prompt.style.height = "";
}

function renderTabs() {
  queueMicrotask(() => {
    if (sessionUI.activeId === lastFrontendSessionId) return;
    lastFrontendSessionId = sessionUI.activeId;
    frontendEvents.dispatchEvent(new CustomEvent("sessionchange", {detail: sessionUI.activeId}));
  });
  if (sessionUI.activeId) writeStorage("sessionStorage", activeSessionKey, sessionUI.activeId);
  const active = sessionById(sessionUI.activeId);
  const activeIndex = sessionUI.sessions.findIndex(session => session.id === sessionUI.activeId);
  activeSessionTitle.textContent = active ? active.title || `Session ${activeIndex + 1}` : "";
  activeSessionTitle.title = activeSessionTitle.textContent;
  activeSessionStatus.hidden = !active?.running;
  updateSessionControls();
  renderSessionList(tabsEl, false);
  renderSessionList(mobileSessionDots, true);
  if (mobileSessionDots.dataset.activeSessionId !== sessionUI.activeId && mobileSessionDots.clientWidth) {
    mobileSessionDots.querySelector(".active")?.scrollIntoView({block: "nearest", inline: "nearest"});
  }
  mobileSessionDots.dataset.activeSessionId = sessionUI.activeId;
}

function renderSessionList(container, mobile) {
  const template = document.getElementById(mobile ? "mobileSessionDot" : "sessionTab");
  const focused = document.activeElement;
  const focusedTab = !mobile && container.contains(focused) ? focused.closest(".tab") : null;
  const existing = new Map(
    Array.from(container.children, tab => [tab.dataset.sessionId, tab])
  );
  reconcileChildren(container, sessionUI.sessions.map((session, index) => {
    const sessionId = session.id || "";
    const tab = existing.get(sessionId) || template.content.firstElementChild.cloneNode(true);
    tab.dataset.sessionId = sessionId;
    const active = sessionId === sessionUI.activeId;
    tab.classList.toggle("active", active);
    tab.classList.toggle("running", Boolean(session.running));
    tab.classList.toggle("completed", Boolean(session.last_final_id));
    const title = session.title || `Session ${index + 1}`;
    const status = session.running ? "Running" : session.last_final_id ? "Finished" : "Idle";
    const notifyArmed = Boolean(session.running && stateForSession(sessionId).notifyWhenDone);
    const replyAge = formatReplyAge(session);
    const description = `${title} · ${status}${replyAge ? ` · ${replyAge}` : ""}${notifyArmed ? " · Notify when done" : ""}${session.unread ? " · Unread reply" : ""}`;
    const nextTitle = mobile ? description : title;
    if (tab.title !== nextTitle) tab.title = nextTitle;
    const selectedAttr = mobile ? "aria-pressed" : "aria-selected";
    const selectedVal = String(active);
    if (tab.getAttribute(selectedAttr) !== selectedVal) tab.setAttribute(selectedAttr, selectedVal);
    tab.querySelector(".unread-dot").hidden = !session.unread;
    if (mobile) {
      if (tab.getAttribute("aria-label") !== description) tab.setAttribute("aria-label", description);
    } else {
      const busyVal = String(Boolean(session.running));
      if (tab.getAttribute("aria-busy") !== busyVal) tab.setAttribute("aria-busy", busyVal);
      const statusButton = tab.querySelector(".tab-status");
      if (statusButton.title !== description) statusButton.title = description;
      if (statusButton.getAttribute("aria-label") !== description) {
        statusButton.setAttribute("aria-label", description);
      }
      const label = tab.querySelector(".tab-label");
      const labelText = label.querySelector(".tab-label-text");
      if (labelText.textContent !== title) labelText.textContent = title;
      const labelTitle = active ? "Click to rename session" : title;
      if (label.title !== labelTitle) label.title = labelTitle;
      const notifyIcon = tab.querySelector(".tab-notify-icon");
      if (notifyIcon) notifyIcon.hidden = !notifyArmed;
      const age = tab.querySelector(".session-reply-age");
      if (age.textContent !== replyAge) age.textContent = replyAge;
      const close = tab.querySelector(".tab-close");
      const closeLabel = `Close session: ${title}`;
      if (close.getAttribute("aria-label") !== closeLabel) {
        close.setAttribute("aria-label", closeLabel);
        const confirmation = tab.querySelector(".tab-close-confirmation");
        confirmation.setAttribute("aria-label", closeLabel);
        const promptText = confirmation.querySelector("span");
        promptText.textContent = `Close ${title}?`;
        promptText.title = closeLabel;
      }
      close.disabled = closingSession || sessionUI.sessions.length <= 1;
      if (tab === sessionCloseRequest?.tab && close.disabled) dismissSessionClose();
    }
    return tab;
  }));
  if (!mobile && sessionCloseRequest && !container.contains(sessionCloseRequest.tab)) dismissSessionClose();
  if (sessionsOpen && focusedTab && !container.contains(focusedTab)) {
    (container.querySelector(".tab.active .tab-label") || sessionsClose).focus({preventScroll: true});
  }
}

function dismissSessionClose(restoreFocus = false) {
  const tab = sessionCloseRequest?.tab;
  if (!tab) return;
  sessionCloseRequest = null;
  tab.classList.remove("confirming-close");
  tab.querySelector(".tab-close-confirmation").hidden = true;
  if (restoreFocus) tab.querySelector(".tab-close").focus({preventScroll: true});
}

function requestSessionClose(tab, trigger, event) {
  if (!sessionsOpen || trigger.disabled) return;
  dismissSessionClose();
  sessionCloseRequest = {tab, x: event.clientX, y: event.clientY};
  const confirmation = tab.querySelector(".tab-close-confirmation");
  confirmation.hidden = false;
  tab.classList.add("confirming-close");
  confirmation.querySelector(".tab-close-cancel").focus({preventScroll: true});
}

async function confirmSessionClose(tab, event) {
  const request = sessionCloseRequest;
  if (tab !== request?.tab || closingSession) return;
  // A second click at the trigger's position can land on the confirmation as the row changes.
  if (event.detail > 1 && Math.hypot(event.clientX - request.x, event.clientY - request.y) <= 6) return;
  dismissSessionClose();
  tab.querySelector(".tab-label").focus({preventScroll: true});
  closingSession = true;
  renderSessionList(tabsEl, false);
  try {
    await closeSession(tab.dataset.sessionId);
  } catch (error) {
    showToast("Unable to close session: " + error.message);
  } finally {
    closingSession = false;
    renderSessionList(tabsEl, false);
  }
}

function editSessionTitle(tab, sessionId) {
  const session = sessionById(sessionId);
  const label = tab.querySelector(".tab-label");
  const input = document.createElement("input");
  input.type = "text";
  input.className = "tab-title-input";
  input.setAttribute("aria-label", "Session title");
  input.value = session.title || "";
  input.placeholder = label.textContent;
  tab.style.width = `${tab.getBoundingClientRect().width}px`;
  label.hidden = true;
  tab.insertBefore(input, label);
  let finished = false;
  const finish = save => {
    if (finished) return;
    finished = true;
    const title = input.value.trim();
    input.remove();
    label.hidden = false;
    tab.style.width = "";
    if (save && title && title !== session.title) {
      renameSession(sessionId, title);
    }
  };
  input.addEventListener("click", event => event.stopPropagation());
  input.addEventListener("keydown", event => {
    if (event.isComposing) return;
    if (event.key === "Enter" || event.key === "Escape") {
      event.preventDefault();
      finish(event.key === "Enter");
      label.focus();
    }
  });
  input.addEventListener("blur", () => finish(true));
  input.focus();
  input.select();
}

async function renameSession(sessionId, title) {
  try {
    const response = await fetch(relativeUrl("api/session/message"), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({prompt: `/title ${title}`, sender: "web", session_id: sessionId}),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || `HTTP ${response.status}`);
    }
    const session = sessionById(sessionId);
    if (session) {
      session.title = payload.snapshot.title;
      renderTabs();
    }
  } catch (error) {
    addEntry("error", `Could not rename session: ${error.message}`);
  }
}

function updatePromptPlaceholder(snapshot = null) {
  const hints = [
    "空闲时，用 /model 列举或切换模型。",
    "空闲时，用 /effort 调整思考强度。",
    "在设置中开启定制会话，描述想要的界面。",
    "在设置的 Share 中，用密码共享工作区。",
    "用 /queue 排队，当前任务结束后执行。",
    "双击上下文圆环，填入可分享的会话信息。",
    "在设置中添加：工具调用详情。",
    "点击上下文圆环，查看 Agent 进展和错误。",
    "点击圆环弹窗中的 Agent，打开实时 transcript。",
    "Enter 发送，Shift + Enter 换行。",
    "在设置中添加：连续工具调用折叠。",
    "点击文件链接，可在旁边预览。",
    "在设置中添加：后台命令状态。",
    "用 /compact 压缩上下文。",
    "在设置中添加：会话按时间排序。",
    "在设置中添加：会话私有便笺。",
    "在设置中添加：定时唤醒时间分隔线。",
    "按 ↑ / ↓ 找回之前的输入。",
    "在设置中添加：已记录的推理摘要。",
    "点击运行状态，开启完成通知。",
  ];
  const session = snapshot || sessionById(sessionUI.activeId);
  const state = stateForSession(sessionUI.activeId);
  const label = session && session.model ? `Message to ${session.model}` : "Message";
  const waitingForInput = snapshot ? snapshot.input_request : state.inputRequest;
  const showHint = Boolean(session?.running && !waitingForInput && hints.length);
  if (!showHint) state.hintStartedAt = null;
  else if (state.hintStartedAt === null) {
    state.hintStartedAt = Date.now();
    state.hintStartIndex = Math.floor(Math.random() * hints.length);
  }
  const placeholder = showHint
    ? hints[(state.hintStartIndex + Math.floor((Date.now() - state.hintStartedAt) / 6000)) % hints.length]
    : `${label}…`;
  prompt.classList.toggle("show-hint", showHint);
  const hintHidden = !showHint || !prompt.classList.contains("empty");
  if (prompt.dataset.placeholder !== placeholder || promptHint.hidden !== hintHidden) {
    prompt.dataset.placeholder = placeholder;
    promptHint.textContent = showHint ? placeholder : "";
    promptHint.hidden = hintHidden;
    resizePrompt();
  }
  prompt.setAttribute("aria-label", label);
}

function updateContextMeter(snapshot = null) {
  const session = snapshot || sessionById(sessionUI.activeId);
  if (!session) return;
  const percent = session.context_remaining_percent;
  const compact = session.auto_compact_token_limit;
  const formatTokens = value => value == null ? "—" : value.toLocaleString("en-US");
  const remainingLabel = compact == null ? "context remaining" : "remaining until compact";
  contextMeter.style.setProperty("--context-remaining", `${percent == null ? 0 : percent}%`);
  contextMeter.dataset.full = String(percent != null && percent >= 100);
  contextMeterValue.textContent = percent == null ? "—" : percent >= 100 ? "" : String(percent);
  contextMeter.setAttribute("aria-label", percent == null
    ? "Context and agents"
    : `${percent}% ${remainingLabel}. Context and agents`);
  document.getElementById("contextCurrent").textContent = formatTokens(session.usage_tokens);
  document.getElementById("contextCompact").textContent = compact == null ? "Off" : formatTokens(compact);
}

function saveCustomizeSessions() {
  writeStorage("sessionStorage", customizeSessionsKey, JSON.stringify(Array.from(customizeSessions)));
}

function markCustomizeSession(sessionId) {
  if (!sessionId) return;
  stateForSession(sessionId).customizeAgent = true;
  if (!customizeSessions.has(sessionId)) {
    customizeSessions.add(sessionId);
    saveCustomizeSessions();
  }
  syncSessionBanners();
}

function clearSessionBanners(sessionId) {
  if (customizeSessions.delete(sessionId)) saveCustomizeSessions();
  banners.clearSession(sessionId);
}

function syncSessionBanners({collapse = false, fullSessionIds = null} = {}) {
  if (fullSessionIds) {
    let pruned = false;
    for (const id of customizeSessions) {
      if (id !== sessionUI.activeId && !fullSessionIds.has(id)) {
        customizeSessions.delete(id);
        pruned = true;
      }
    }
    if (pruned) saveCustomizeSessions();
  }
  const sessionId = sessionUI.activeId;
  const state = sessionId ? stateForSession(sessionId) : null;
  if (state && !state.customizeAgent && customizeSessions.has(sessionId)) {
    state.customizeAgent = true;
  } else if (state?.customizeAgent && !customizeSessions.has(sessionId)) {
    customizeSessions.add(sessionId);
    saveCustomizeSessions();
  }
  const binding = state ? state.externalCall : null;
  banners.setBuiltin("external-call", binding ? {
    icon: "terminal",
    label: "External Call",
    sessionId,
    adminOnly: true,
    actions: [
      {label: "Copy Bash", copyText: binding.one_liners?.linux || "", copyLabel: "Terminal command"},
      {label: "Copy PowerShell", copyText: binding.one_liners?.windows || "", copyLabel: "Terminal command"},
    ],
    details: [
      {label: "Bash", value: binding.one_liners?.linux || "", copyable: true, copyLabel: "Terminal command"},
      {label: "PowerShell", value: binding.one_liners?.windows || "", copyable: true, copyLabel: "Terminal command"},
      {label: "Endpoint", value: binding.url || "", copyable: true, copyLabel: "Endpoint URL"},
    ],
  } : null);
  const {frontendDirectory, packageDirectory} = document.body.dataset;
  banners.setBuiltin("customize-agent", state?.customizeAgent ? {
    icon: "customize",
    label: "Customize Agent",
    text: "Edits apply to all workspaces; refresh to load.",
    sessionId,
    adminOnly: true,
    actions: [
      ...(packageDirectory ? [{
        label: "Feature catalog",
        viewHref: encodeViewPath(packageDirectory + "/optional_features/README.md"),
      }] : []),
      {
        label: builtinFrontend ? "Re-enable custom" : "Built-in page",
        href: frontendUrl(!builtinFrontend),
      },
    ],
    details: [
      ...(frontendDirectory ? [{
        label: "Frontend dir",
        value: frontendDirectory,
        copyable: true,
        copyLabel: "Frontend directory",
      }] : []),
      ...(packageDirectory ? [{
        label: "Host API",
        value: packageDirectory + "/WORKSPACE_API.md",
        viewHref: encodeViewPath(packageDirectory + "/WORKSPACE_API.md"),
      }] : []),
    ],
  } : null);
  banners.render({collapse});
}

function startPolling() {
  if (pollTimer) return;
  pollSession();
  pollTimer = setInterval(pollSession, 2000);
}

function refreshBoard(board) {
  if (!board || !board.exists) {
    lastBoardSignature = null;
    return;
  }
  const signature = `${board.mtime_ns || ""}:${board.size || ""}`;
  // Automatic Canvas refreshes must not add browser Back history entries.
  if (signature !== lastBoardSignature) {
    lastBoardSignature = signature;
    boardFrame.contentWindow.location.replace(relativeUrl(`board?t=${Date.now()}`));
  }
}

/** Return a callback for this session poll's workspace fields, or null while views are paused. */
function beginWorkspacePoll() {
  if (!contentViewsActive) return null;
  const token = contentTabs.pollToken();
  return payload => {
    if (!contentViewsActive) return;
    if (enableBoard) refreshBoard(payload.board);
    contentTabs.applyViews(payload.views, token, payload.views_error);
  };
}

function renderQueueHint(inputs) {
  queueHint.hidden = inputs.length === 0;
  for (const [kind, label, badge] of [
    ["steer", "[soon]", steerHint],
    ["enqueue", "[on done]", enqueueHint],
  ]) {
    const text = inputs.filter(item => item.queue === kind).map(item => item.prompt).join("\n");
    badge.hidden = !text;
    badge.querySelector(".queue-preview").textContent = text;
    badge.title = label + "\n" + text;
  }
}

async function markVisibleFinalRead() {
  if (sessionsOpen && mobileSessionsMedia.matches) return;
  if (document.visibilityState !== "visible" || !document.hasFocus()) return;
  const sessionId = sessionUI.activeId;
  const session = sessionById(sessionId);
  if (!session || !session.unread || !session.last_final_id) return;
  const state = stateForSession(sessionId);
  if (state.readingFinalId || !log.getClientRects().length) return;
  const finalId = session.last_final_id;
  const final = Array.from(log.querySelectorAll("[data-final-id]"))
    .find(element => element.dataset.finalId === finalId && element.dataset.sessionId === sessionId);
  if (!final) return;
  const viewport = log.getBoundingClientRect();
  const rect = final.getBoundingClientRect();
  if (Math.min(rect.bottom, viewport.bottom, window.innerHeight) <= Math.max(rect.top, viewport.top, 0) ||
      Math.min(rect.right, viewport.right, window.innerWidth) <= Math.max(rect.left, viewport.left, 0)) return;
  state.readingFinalId = finalId;
  try {
    const response = await fetch(relativeUrl(`api/sessions/${encodeURIComponent(sessionId)}/read`), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({final_id: finalId}),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    const current = sessionById(sessionId);
    const saved = payload.sessions.find(item => item.id === sessionId);
    if (current && saved && current.last_final_id === saved.last_final_id) {
      current.unread = saved.unread;
      renderTabs();
    }
  } catch (error) {
    showToast("Unable to mark reply as read: " + error.message);
  } finally {
    state.readingFinalId = "";
  }
}

async function submitPrompt(text, draft = null) {
  const sessionId = sessionUI.activeId;
  const restoreDraft = () => {
    if (!draft?.parts?.length || sessionUI.activeId !== sessionId) return;
    if (!composer.read().parts.length) {
      composer.set(draft);
      stateForSession(sessionId).draft = composer.read();
      resizePrompt();
    }
  };
  try {
    const response = await fetch(relativeUrl("api/session/message"), {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({prompt: text, sender: "web", session_id: sessionId}),
    });
    const payload = await response.json();
    if (sessionUI.activeId !== sessionId) return;
    if (!response.ok || !payload.ok) {
      addEntry("error", payload.error || `HTTP ${response.status}`);
      restoreDraft();
      if (!sessionById(sessionId)?.running) setSpinner("");
      return;
    }
    if (payload.snapshot) {
      if (payload.sessions) updateSessions(payload.sessions);
      renderSnapshot(payload.snapshot);
    } else {
      setSpinner("");
    }
    startPolling();
  } catch (error) {
    if (sessionUI.activeId !== sessionId) return;
    addEntry("error", `Unable to send message: ${error.message}`);
    restoreDraft();
    if (!sessionById(sessionId)?.running) setSpinner("");
  }
}

function insertEditorContext(context) {
  if (!sessionUI.activeId) {
    showToast("Select a conversation before inserting context.");
    return;
  }
  workspace.dataset.mobilePane = "chat";
  closeContextBlock();
  composer.insert([context]);
  showToast(context.label + " added to draft");
}

function editContextBlock(block) {
  editingContextId = block.id;
  contextBlockLabel.textContent = block.label;
  contextBlockLabel.title = block.label;
  contextBlockEditor.value = block.text;
  contextBlockEditor.setAttribute("aria-label", block.label + " context");
  contextBlockDetail.hidden = false;
  contextBlockEditor.focus({preventScroll: true});
}

function closeContextBlock() {
  editingContextId = null;
  contextBlockDetail.hidden = true;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const draft = composer.read();
  const text = Composer.text(draft.parts);
  if (!text.trim() && !stateForSession(sessionUI.activeId).inputRequest) return;
  stateForSession(sessionUI.activeId).promptHistory.remember(draft);
  closeContextBlock();
  composer.set({parts: []});
  resizePrompt();
  stateForSession(sessionUI.activeId).draft = composer.read();
  submitPrompt(text, draft);
});

contextBlockEditor.addEventListener("input", () => {
  composer.updateContext(editingContextId, contextBlockEditor.value);
});
contextBlockEditor.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    closeContextBlock();
    composer.focus();
  }
});
document.getElementById("contextBlockDone").addEventListener("click", () => {
  closeContextBlock();
  composer.focus();
});
document.getElementById("contextBlockRemove").addEventListener("click", () => {
  composer.removeContext(editingContextId);
  closeContextBlock();
});

log.addEventListener("scroll", sessionUI.saveScroll);
installDialogs({
  getSessions: () => sessionUI.sessions, stateForSession, createSession, selectSession,
  openView: (...args) => contentTabs.open(...args), showToast, markCustomizeSession,
});
splitter.addEventListener("pointerdown", startResize);
splitter.addEventListener("pointermove", updateResize);
splitter.addEventListener("pointerup", endResize);
splitter.addEventListener("pointercancel", endResize);
splitter.addEventListener("dblclick", centerSplit);
spinner.addEventListener("click", toggleSpinnerNotification);
if (notifyToggle) notifyToggle.addEventListener("click", toggleSpinnerNotification);
navigator.serviceWorker?.addEventListener("message", event => {
  if (event.data?.type === "pycodex:select-session" && event.data.sessionId) {
    selectSession(event.data.sessionId);
  }
});
for (const container of [tabsEl, mobileSessionDots]) {
  container.addEventListener("click", event => {
    const tab = event.target.closest("[data-session-id]");
    if (!tab) return;
    const sessionId = tab.dataset.sessionId;
    if (container === mobileSessionDots) {
      selectSession(sessionId);
      return;
    }
    const close = event.target.closest(".tab-close");
    if (close) {
      requestSessionClose(tab, close, event);
    } else if (event.target.closest(".tab-close-confirmation")) {
      if (event.target.closest(".tab-close-confirm")) confirmSessionClose(tab, event);
      if (event.target.closest(".tab-close-cancel")) dismissSessionClose(true);
    } else if (sessionId === sessionUI.activeId && event.target.closest(".tab-label")) {
      editSessionTitle(tab, sessionId);
    } else selectSession(sessionId);
  });
}
document.addEventListener("pointerdown", event => {
  const tab = sessionCloseRequest?.tab;
  if (tab && !tab.contains(event.target)) dismissSessionClose();
});
tabsEl.addEventListener("focusout", event => {
  const confirmation = event.target.closest(".tab-close-confirmation");
  if (confirmation && confirmation.parentElement === sessionCloseRequest?.tab &&
      !confirmation.contains(event.relatedTarget)) dismissSessionClose();
});
mobileSessionsToggle.addEventListener("click", () => setSessionsOpen(!sessionsOpen));
sessionsClose.addEventListener("click", () => setSessionsOpen(!sessionsPinned, true));
sessionSidebar.addEventListener("pointerenter", event => {
  if (event.pointerType !== "mouse" || mobileSessionsMedia.matches) return;
  clearSessionsHover();
  if (!sessionsOpen) {
    sessionsHoverTimer = setTimeout(() => setSessionsOpen(true, false, false), 100);
  }
});
sessionSidebar.addEventListener("pointerleave", event => {
  if (event.pointerType !== "mouse") return;
  clearSessionsHover();
  if (sessionsOpen && !sessionsPinned) {
    sessionsHoverTimer = setTimeout(() => setSessionsOpen(false), 150);
  }
});
sessionSidebar.addEventListener("click", event => {
  if (event.target.closest("#sessionsClose, #newTab, .tab")) return;
  if (!sessionsOpen) {
    event.preventDefault();
    event.stopPropagation();
    setSessionsOpen(true, true);
  } else if (!sessionsPinned) {
    setSessionsOpen(true);
  } else if (!event.target.closest("button, input, .tab")) {
    setSessionsOpen(false, true);
  }
}, true);
sessionsBackdrop.addEventListener("click", () => setSessionsOpen(false, true));
mobileSessionsMedia.addEventListener("change", () => {
  setSessionsOpen(false, sessionSidebar.contains(document.activeElement));
});
sessionSidebar.addEventListener("keydown", event => {
  if (event.key !== "Tab" || !mobileSessionsMedia.matches || !sessionsOpen) return;
  const focusable = Array.from(sessionSidebar.querySelectorAll("button:not(:disabled), input"))
    .filter(element => element.getClientRects().length && getComputedStyle(element).visibility !== "hidden");
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && sessionsOpen && !event.defaultPrevented &&
      !event.target.closest("dialog")) {
    event.preventDefault();
    if (sessionCloseRequest) dismissSessionClose(true);
    else setSessionsOpen(false, true);
  }
});
document.querySelectorAll(".fullscreen-toggle").forEach(button => {
  button.hidden = !document.fullscreenEnabled;
  button.addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen({navigationUI: "hide"});
    } catch (error) {
      window.alert("Unable to change fullscreen: " + error.message);
    }
  });
  document.addEventListener("fullscreenchange", () => {
    const active = Boolean(document.fullscreenElement);
    button.title = active ? "Exit fullscreen" : "Enter fullscreen";
    button.setAttribute("aria-label", button.title);
    button.querySelector("path").setAttribute("d", active
      ? "M8 3v5H3m13-5v5h5M3 16h5v5m8 0v-5h5"
      : "M8 3H3v5m13-5h5v5M3 16v5h5m8 0h5v-5");
  });
});
document.querySelectorAll("[data-pane]").forEach((button) => {
  button.addEventListener("click", () => {
    workspace.dataset.mobilePane = button.dataset.pane;
    if (button.dataset.pane === "chat") {
      resizePrompt();
      restoreSessionScroll();
    }
    requestAnimationFrame(markVisibleFinalRead);
  });
});
window.addEventListener("resize", () => {
  loadChatWidth();
  syncSidebarOverlay();
  resizePrompt();
  restoreSessionScroll();
  requestAnimationFrame(markVisibleFinalRead);
});
window.addEventListener("focus", () => requestAnimationFrame(markVisibleFinalRead));
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible") {
    if (!sessionsPinned) setSessionsOpen(false);
    return;
  }
  pollSession();
  requestAnimationFrame(markVisibleFinalRead);
});

document.addEventListener("animationstart", event => {
  if (!["spinner-pulse", "activity-breathe", "agent-message-breathe"].includes(event.animationName)) return;
  // Include pseudo-elements and anchor new indicators to the page's breathing phase.
  for (const animation of event.target.getAnimations?.({subtree: true}) || []) {
    if (animation.animationName === event.animationName) animation.startTime = 0;
  }
});

const recoveryNotice = document.getElementById("frontendRecovery");
recoveryNotice.hidden = !builtinFrontend;
recoveryNotice.querySelector("a").href = frontendUrl(false);
new MutationObserver(syncMobileSessionFloat).observe(workspace, {
  attributes: true,
  attributeFilter: ["data-mobile-pane"],
});
setSessionsOpen(sessionsOpen);
requestAnimationFrame(() => requestAnimationFrame(() => workspace.classList.add("sessions-ready")));
loadChatWidth();
setupContextDetails({
  getActiveSessionId: () => sessionUI.activeId, frontendEvents,
  openTranscript: (...args) => contentTabs.openTranscript(...args),
  fillRuntimeInfo: () => fillRuntimeInfo({
    getActiveSessionId: () => sessionUI.activeId, stateForSession, closeContextBlock, composer, showToast,
  }),
});
resizePrompt();
document.querySelectorAll(".content-tabs").forEach(tabs => {
  tabs.addEventListener("wheel", event => {
    if (event.ctrlKey || event.shiftKey || event.deltaX !== 0 || event.deltaY === 0) return;
    const previous = tabs.scrollLeft;
    const unit = event.deltaMode === 1 ? tabs.clientHeight : event.deltaMode === 2 ? tabs.clientWidth : 1;
    tabs.scrollLeft += event.deltaY * unit;
    if (tabs.scrollLeft !== previous) event.preventDefault();
  }, {passive: false});
});

function createVscodeWindow() {
  const id = crypto.getRandomValues(new Uint32Array(4)).join("-");
  const url = new URL(`vscode/window/${encodeURIComponent(id)}/`, window.location.href);
  const cwd = document.body.dataset.cwd || "";
  if (cwd) url.searchParams.set("folder", cwd);
  url.searchParams.set("payload", JSON.stringify([
    ["extensionEnvironment", JSON.stringify({PYCODEX_VSCODE_WINDOW_ID: id})],
  ]));
  return {id, url: url.toString()};
}

function updateVscodeToggle({active = false, loading = false, error = ""} = {}) {
  if (!vscodeToggle) return;
  vscodeToggle.hidden = !isAdmin;
  if (enableVscode) vscodeToggle.removeAttribute("aria-disabled");
  else vscodeToggle.setAttribute("aria-disabled", "true");
  vscodeToggle.setAttribute("aria-pressed", String(Boolean(enableVscode && active)));
  if (enableVscode && active && loading) vscodeToggle.dataset.loading = "true";
  else delete vscodeToggle.dataset.loading;
  vscodeToggle.title = !enableVscode ? VSCODE_PASSWORD_HINT :
    !active ? "Open VS Code" :
    loading ? "Loading VS Code… (click to hide)" :
    error ? "VS Code failed to load (click to hide)" : "Hide VS Code";
  vscodeToggle.setAttribute("aria-label", vscodeToggle.title);
}

const contentTabs = mountContentViews({
  tabs: document.getElementById("contentTabs"),
  frame: document.getElementById("boardFrame"),
  vscodeFrame,
  content: document.getElementById("contentView"),
  status: document.getElementById("viewStatus"),
  renderMarkdown: renderMarkdownInto,
  onInsertContext: insertEditorContext,
  enableBoard,
  vscodeWindow: enableVscode ? createVscodeWindow() : null,
  onVscodeStateChange: updateVscodeToggle,
  request: async (path, options = {}) => {
    const response = await fetch(new URL(path, window.location.href), options);
    const payload = await response.json();
    if (!response.ok) {
      const error = new Error(payload.detail || "HTTP " + response.status);
      error.status = response.status;
      throw error;
    }
    return payload;
  },
});
if (vscodeToggle) {
  updateVscodeToggle();
  vscodeToggle.addEventListener("click", () => {
    if (!enableVscode) {
      contentTabs.notify(VSCODE_PASSWORD_HINT);
      showToast(VSCODE_PASSWORD_HINT);
      return;
    }
    contentTabs.toggleVscodeMode();
  });
}
const {followMarkdownLink, followCanvasLink} = createViewLinkHandlers({
  getActiveSessionId: () => sessionUI.activeId, openView: (...args) => contentTabs.open(...args), showToast,
});
let canvasDocument = null;

function bindCanvasLinks() {
  if (canvasDocument) canvasDocument.removeEventListener("click", followCanvasLink);
  canvasDocument = boardFrame.contentDocument;
  if (canvasDocument) canvasDocument.addEventListener("click", followCanvasLink);
}

function startContentViews() {
  document.addEventListener("click", followMarkdownLink);
  if (enableBoard) {
    boardFrame.addEventListener("load", bindCanvasLinks);
    bindCanvasLinks();
  }
  contentViewsActive = true;
  contentTabs.start();
}

function stopContentViews() {
  contentViewsActive = false;
  document.removeEventListener("click", followMarkdownLink);
  boardFrame.removeEventListener("load", bindCanvasLinks);
  if (canvasDocument) canvasDocument.removeEventListener("click", followCanvasLink);
  canvasDocument = null;
  contentTabs.stop();
}

startContentViews();
startPolling();
try {
  loadEmbeddedBanners(document, banners);
} catch (error) {
  showToast("Unable to load banner: " + error.message);
}
mountCustomFrontend({
  getActiveSessionId: () => sessionUI.activeId, sessionById, selectSession, frontendEvents,
  composer, stateForSession, resizePrompt, contentTabs,
  setBanner: (id, banner) => banners.set(id, banner),
}).catch(error => showToast("Unable to initialize custom frontend: " + error.message));
window.addEventListener("pagehide", clearSessionsHover);
window.addEventListener("pagehide", stopContentViews);
window.addEventListener("pageshow", startContentViews);
