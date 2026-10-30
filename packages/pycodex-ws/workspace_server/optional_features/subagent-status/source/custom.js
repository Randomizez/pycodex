// Reference implementation: adapt this source into your own custom.js.
export function mount(host) {
  const root = document.createElement("section");
  root.hidden = true;
  root.className = "example-subagent-status";
  root.setAttribute("aria-label", "Subagents");
  const status = document.createElement("p");
  status.setAttribute("role", "status");
  const list = document.createElement("ul");
  root.append(status, list);
  document.querySelector(".composer").prepend(root);
  const lifetime = new AbortController();
  let stopSession = () => {};

  function select() {
    stopSession();
    root.hidden = true;
    list.replaceChildren();
    const sessionId = host.getActiveSessionId();
    status.textContent = "";
    if (!sessionId) return;
    const request = new AbortController();
    let timer;
    let lastRows = "";
    stopSession = () => {
      request.abort();
      clearTimeout(timer);
    };
    async function refresh() {
      try {
        const {agents} = await host.request(
          `api/sessions/${encodeURIComponent(sessionId)}/agents`,
          {signal: request.signal},
        );
        if (request.signal.aborted || host.getActiveSessionId() !== sessionId) return;
        root.hidden = agents.length === 0;
        status.textContent = `${agents.filter(agent => agent.status === "running").length} running · ${agents.length} subagents`;
        const serialized = JSON.stringify(agents);
        if (serialized !== lastRows) {
          list.replaceChildren(...agents.map(agent => {
            const item = document.createElement("li");
            const name = document.createElement("strong");
            name.textContent = agent.nickname || agent.agent_id;
            const state = document.createElement("span");
            const detail = document.createElement("pre");
            if (typeof agent.status === "string") {
              state.textContent = agent.status;
            } else if ("completed" in agent.status) {
              state.textContent = "completed";
              detail.textContent = agent.status.completed === null ? "" : agent.status.completed;
            } else {
              state.textContent = "errored";
              detail.textContent = agent.status.errored;
            }
            item.dataset.status = state.textContent;
            const info = document.createElement("details");
            const summary = document.createElement("summary");
            summary.textContent = "Agent details";
            const identity = document.createElement("p");
            identity.textContent = `ID: ${agent.agent_id} · Parent: ${agent.parent_agent_id}`;
            info.append(summary, identity, detail);
            item.append(name, state, info);
            return item;
          }));
          lastRows = serialized;
        }
      } catch (error) {
        if (request.signal.aborted) return;
        list.replaceChildren();
        lastRows = "";
        status.textContent = `Unable to read subagents: ${error.message}`;
        root.hidden = false;
      } finally {
        if (!request.signal.aborted) timer = setTimeout(refresh, 2000);
      }
    }
    refresh();
  }

  const unsubscribe = host.onSessionChanged(select);
  window.addEventListener("pagehide", () => stopSession(), {signal: lifetime.signal});
  window.addEventListener("pageshow", event => {
    if (event.persisted) select();
  }, {signal: lifetime.signal});
  select();
  return () => {
    stopSession();
    unsubscribe();
    lifetime.abort();
    root.remove();
  };
}
