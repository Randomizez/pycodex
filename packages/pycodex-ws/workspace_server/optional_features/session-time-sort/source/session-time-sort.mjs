export function sessionActivity(session) {
  if (typeof session?.last_user_at === "number" && Number.isFinite(session.last_user_at)) {
    return session.last_user_at;
  }
  const match = /^([0-9a-f]{8})-([0-9a-f]{4})-7/i.exec(session?.last_final_id || "");
  return match ? Number.parseInt(match[1] + match[2], 16) / 1000 : 0;
}

export function sortSessionsByActivity(items, {activeId = "", limit = 20} = {}) {
  const ordered = [...items].sort((left, right) =>
    sessionActivity(right) - sessionActivity(left) ||
    String(right.id).localeCompare(String(left.id))
  );
  if (!Number.isInteger(limit) || limit <= 0 || ordered.length <= limit) {
    return {ordered, visible: ordered};
  }
  const visible = ordered.slice(0, limit);
  const active = ordered.find(session => session.id === activeId);
  if (active && !visible.includes(active)) visible[visible.length - 1] = active;
  return {ordered, visible};
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

export function mountSessionTimeSort(host, {
  limit = 20,
  pollMs = 2000,
  root = document,
  win = window,
} = {}) {
  const tabs = root.getElementById ? root.getElementById("tabs") : root.querySelector("#tabs");
  const mobileDots = root.querySelector(".mobile-session-dots");
  const containers = [tabs, mobileDots].filter(Boolean);
  const status = root.createElement("p");
  status.className = "session-time-sort-error";
  status.setAttribute("role", "status");
  status.hidden = true;
  if (tabs?.parentElement) tabs.parentElement.insertBefore(status, tabs);

  const summaries = new Map();
  const lifetime = new AbortController();
  let hasLoaded = false;
  let disposed = false;
  let inFlight = null;
  let timer = null;

  function apply() {
    if (disposed || !hasLoaded) return;
    const items = new Map(summaries);
    for (const container of containers) {
      for (const node of container.children) {
        const id = node.dataset?.sessionId;
        if (id && !items.has(id)) items.set(id, {id});
      }
    }
    const activeId = host.getActiveSessionId() || "";
    const {ordered, visible} = sortSessionsByActivity(items.values(), {activeId, limit});
    const rank = new Map(ordered.map((session, index) => [String(session.id), String(index)]));
    const visibleIds = new Set(visible.map(session => String(session.id)));
    for (const container of containers) {
      const reorder = container === tabs;
      for (const node of container.children) {
        const id = node.dataset?.sessionId;
        if (!id) continue;
        if (reorder) {
          const order = rank.get(id) ?? "0";
          if (node.style.order !== order) node.style.order = order;
        }
        const hidden = !visibleIds.has(id);
        if (node.hasAttribute("data-session-time-hidden") !== hidden) {
          node.toggleAttribute("data-session-time-hidden", hidden);
        }
      }
    }
  }

  function stopPolling() {
    if (inFlight) {
      inFlight.abort();
      inFlight = null;
    }
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  }

  function schedule() {
    if (disposed || timer !== null || inFlight || root.hidden) return;
    timer = setTimeout(() => {
      timer = null;
      refresh();
    }, pollMs);
  }

  async function refresh() {
    if (disposed || root.hidden || inFlight) return;
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
    const controller = new AbortController();
    inFlight = controller;
    try {
      const payload = await host.request("api/sessions", {signal: controller.signal});
      if (disposed || inFlight !== controller || controller.signal.aborted) return;
      summaries.clear();
      for (const session of payload.sessions || []) {
        if (session?.id) summaries.set(String(session.id), session);
      }
      hasLoaded = true;
      status.hidden = true;
      status.textContent = "";
      apply();
    } catch (error) {
      if (disposed || controller.signal.aborted) return;
      status.textContent = `无法更新会话排序：${error.message}`;
      status.hidden = false;
    } finally {
      if (inFlight === controller) {
        inFlight = null;
        schedule();
      }
    }
  }

  const observer = new MutationObserver(() => apply());
  for (const container of containers) {
    observer.observe(container, {childList: true});
  }
  const unsubscribe = host.onSessionChanged(() => {
    apply();
    if (!inFlight) refresh();
  });
  root.addEventListener("visibilitychange", () => {
    if (root.hidden) stopPolling();
    else refresh();
  }, {signal: lifetime.signal});
  win.addEventListener("pagehide", stopPolling, {signal: lifetime.signal});
  win.addEventListener("pageshow", event => {
    if (event.persisted) refresh();
  }, {signal: lifetime.signal});
  refresh();

  return () => {
    disposed = true;
    stopPolling();
    observer.disconnect();
    unsubscribe();
    lifetime.abort();
    status.remove();
    for (const container of containers) {
      for (const node of container.children) {
        node.style.order = "";
        node.removeAttribute("data-session-time-hidden");
      }
    }
  };
}
