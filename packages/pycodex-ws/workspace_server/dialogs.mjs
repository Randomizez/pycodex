import {Composer} from "./content_views.mjs";
import {element, relativeUrl} from "./workspace_dom.mjs";

export async function fillRuntimeInfo({
  getActiveSessionId, stateForSession, closeContextBlock, composer, showToast,
}) {
  const sessionId = getActiveSessionId();
  const state = stateForSession(sessionId);
  if (!sessionId || state.runtimeInfoLoading) return;
  state.runtimeInfoLoading = true;
  try {
    const response = await fetch(relativeUrl(
      `api/session?session_id=${encodeURIComponent(sessionId)}&runtime_info=true`
    ), {cache: "no-store"});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || payload.detail || `HTTP ${response.status}`);
    if (typeof payload.runtime_info !== "string") throw new Error("Runtime information is unavailable");
    if (getActiveSessionId() !== sessionId) return;
    closeContextBlock();
    // Read the draft now so typing during the request is preserved.
    const end = Composer.size(composer.parts());
    composer.replace(end, end, [{type: "text", text: (end ? "\n\n" : "") + payload.runtime_info}]);
    showToast("Share to enable agent communication.");
  } catch (error) {
    if (getActiveSessionId() === sessionId) showToast("Unable to load runtime information: " + error.message);
  } finally {
    state.runtimeInfoLoading = false;
  }
}

export function setupContextDetails({getActiveSessionId, frontendEvents, openTranscript, fillRuntimeInfo}) {
  const contextMeter = document.getElementById("contextMeter");
  const control = document.getElementById("contextControl");
  const details = document.getElementById("contextDetails");
  const tree = document.getElementById("contextAgents");
  const summary = document.getElementById("contextAgentsSummary");
  const status = document.getElementById("contextAgentsStatus");
  const rows = new Map();
  let pinned = false;
  let treeSessionId = "";
  let structure = "";
  let request = null;
  let timer = null;
  const stop = () => {
    request?.controller.abort();
    request = null;
    clearTimeout(timer);
    timer = null;
  };
  const reset = () => {
    treeSessionId = getActiveSessionId();
    structure = "";
    rows.clear();
    tree.replaceChildren();
    summary.hidden = true;
    status.textContent = "Loading agents…";
    status.hidden = false;
  };
  const render = payload => {
    const isActive = agent => agent.status === "running" || agent.status === "pending_init";
    const agents = [payload.root, ...payload.agents].sort(
      (left, right) => Number(isActive(right)) - Number(isActive(left))
    );
    const running = agents.filter(agent => agent.status === "running").length;
    const failed = agents.filter(agent => typeof agent.status === "object" && "errored" in agent.status).length;
    summary.textContent = `${running} running · ${failed} failed`;
    summary.hidden = false;
    const nextStructure = JSON.stringify(agents.map(agent => [agent.agent_id, agent.parent_agent_id]));
    if (nextStructure !== structure) {
      rows.clear();
      tree.replaceChildren();
      for (const agent of agents) {
        const item = document.createElement("li");
        item.className = "context-agent";
        item.dataset.agentId = agent.agent_id;
        item.innerHTML = '<button type="button" class="context-agent-row"><span class="tab-status" role="img"></span>' +
          '<span><span class="context-agent-heading"><span class="context-agent-name"></span>' +
          '<span class="context-agent-duration"></span></span><span class="context-agent-message"></span></span></button>' +
          '<ul class="context-agent-children"></ul>';
        rows.set(agent.agent_id, item);
      }
      for (const agent of agents) {
        const parent = agent.agent_id === payload.root.agent_id
          ? tree
          : rows.get(agent.parent_agent_id).querySelector(".context-agent-children");
        parent.append(rows.get(agent.agent_id));
      }
      structure = nextStructure;
    }
    for (const agent of agents) {
      const item = rows.get(agent.agent_id);
      const state = typeof agent.status === "string" ? agent.status : Object.keys(agent.status)[0];
      item.dataset.status = state;
      const dot = item.querySelector(".tab-status");
      dot.title = state.replaceAll("_", " ");
      dot.setAttribute("aria-label", dot.title);
      const name = item.querySelector(".context-agent-name");
      name.textContent = agent.agent_path.split("/").pop();
      name.title = agent.agent_path;
      item.dataset.agentPath = agent.agent_path;
      item.querySelector(".context-agent-row").setAttribute("aria-label", `Open ${agent.agent_path} transcript`);
      item.querySelector(".context-agent-duration").textContent = `for ${(agent.elapsed_seconds / 60).toFixed(1)} min`;
      const message = item.querySelector(".context-agent-message");
      const text = (state === "errored" ? agent.status.errored : agent.last_message) || "—";
      if (message.textContent !== text) message.textContent = text;
      message.title = text;
    }
    status.hidden = true;
  };
  const refresh = async () => {
    clearTimeout(timer);
    timer = null;
    if (details.hidden || document.hidden || !getActiveSessionId() || request) return;
    if (treeSessionId !== getActiveSessionId()) reset();
    const current = {sessionId: getActiveSessionId(), controller: new AbortController()};
    request = current;
    try {
      const response = await fetch(relativeUrl(
        `api/sessions/${encodeURIComponent(current.sessionId)}/agents`
      ), {signal: current.controller.signal});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      if (request !== current || getActiveSessionId() !== current.sessionId) return;
      render(payload);
    } catch (error) {
      if (request !== current || current.controller.signal.aborted) return;
      status.textContent = `Unable to load agents: ${error.message}`;
      status.hidden = false;
    } finally {
      if (request === current) {
        request = null;
        if (!details.hidden) timer = setTimeout(refresh, 2000);
      }
    }
  };
  const show = visible => {
    details.hidden = !visible;
    contextMeter.setAttribute("aria-expanded", String(visible));
    if (!visible) stop();
    else if (!request && !timer) refresh();
  };
  const close = () => {
    pinned = false;
    show(false);
  };
  tree.addEventListener("click", event => {
    const row = event.target.closest(".context-agent-row");
    if (!row || treeSessionId !== getActiveSessionId()) return;
    const agent = row.closest(".context-agent");
    openTranscript(treeSessionId, agent.dataset.agentId, agent.dataset.agentPath);
    close();
  });
  control.addEventListener("pointerenter", event => {
    if (event.pointerType !== "touch") show(true);
  });
  control.addEventListener("pointerleave", () => {
    if (!pinned) show(false);
  });
  contextMeter.addEventListener("focus", () => show(true));
  control.addEventListener("focusout", event => {
    if (!pinned && !control.contains(event.relatedTarget)) show(false);
  });
  contextMeter.addEventListener("click", () => {
    pinned = !pinned;
    show(pinned);
  });
  contextMeter.addEventListener("dblclick", event => {
    event.preventDefault();
    close();
    fillRuntimeInfo();
  });
  document.addEventListener("pointerdown", event => {
    if (!control.contains(event.target)) close();
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") {
      if (details.contains(document.activeElement)) contextMeter.focus({preventScroll: true});
      close();
    }
  });
  frontendEvents.addEventListener("sessionchange", () => {
    stop();
    reset();
    if (!details.hidden) refresh();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop();
    else if (!details.hidden) refresh();
  });
  window.addEventListener("pagehide", stop);
  window.addEventListener("pageshow", () => {
    if (!details.hidden) refresh();
  });
  document.getElementById("tabs").addEventListener("click", close);
  document.getElementById("newTab").addEventListener("click", close);
}

function frontendCustomizationPrompt() {
  const {frontendDirectory, packageDirectory} = document.body.dataset;
  return [
    "<frontend_customization>",
    "Introduce interface customization, optional features, contributing a feature PR and recovery. Wait for my specific request before editing.",
    "Use my preferred language, then the language of my messages, then my browser languages: " +
      (navigator.languages.join(", ") || "not supplied"),
    `User customization directory: ${frontendDirectory}`,
    `Packaged page and native modules: ${packageDirectory}/workspace.html and workspace.mjs`,
    `Read the host, HTTP and WebSocket contracts in ${packageDirectory}/WORKSPACE_API.md before changing behavior.`,
    `Offer the feature catalog as a clickable absolute Markdown link: ${packageDirectory}/optional_features/README.md`,
    `For a requested feature PR, follow ${packageDirectory}/optional_features/CONTRIBUTING.md in a separate checkout.`,
    "Read existing custom files first. Prefer small custom.css and custom.js overrides, using mount(host) and workspace-relative URLs.",
    "Integrate chosen features into the existing customization. Preserve unrelated edits, DOM contracts, template placeholders and Settings.",
    "If a full template override is needed, copy the packaged template only when no user copy exists. Keep installed files intact.",
    "These files apply to every workspace of this OS user. Back up substantial edits outside the served directory, including the original template for comparison.",
    "Verify affected desktop and mobile interactions and asynchronous updates. Refresh to apply; use ?builtin=1 for recovery and rename the customization directory to undo.",
    "</frontend_customization>",
  ].join("\\n");
}

export function installDialogs({
  getActiveSessionId, getSessions, createSession, selectSession, updateSessions, renderSnapshot, showToast,
}) {
  const newTabButton = document.getElementById("newTab");
  const workspace = document.querySelector(".workspace");
  const newSessionDialog = document.getElementById("newSessionDialog");
  const sessionProfile = document.getElementById("sessionProfile");
  const newSessionError = document.getElementById("newSessionError");
  const createSessionButton = document.getElementById("createSessionButton");
  newTabButton.addEventListener("click", async () => {
    newTabButton.disabled = true;
    try {
      await createSession();
    } catch (error) {
      showToast("Unable to create session: " + error.message);
    } finally {
      newTabButton.disabled = false;
    }
  });
  const customProfileButton = document.getElementById("customProfile");
  customProfileButton.addEventListener("click", async () => {
    customProfileButton.disabled = true;
    try {
      const response = await fetch(relativeUrl("api/sessions"));
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      if (!payload.profiles.length) {
        showToast("No named profiles are configured.");
        return;
      }
      const defaultLabel = payload.default_profile === null
        ? "Default" : `Default (${payload.default_profile})`;
      sessionProfile.replaceChildren(new Option(defaultLabel, "null"));
      for (const profile of payload.profiles) {
        sessionProfile.add(new Option(profile, JSON.stringify(profile)));
      }
      newSessionError.textContent = "";
      newSessionDialog.showModal();
    } catch (error) {
      showToast("Unable to create session: " + error.message);
    } finally {
      customProfileButton.disabled = false;
    }
  });
  document.getElementById("newSessionForm").addEventListener("submit", async event => {
    event.preventDefault();
    createSessionButton.disabled = true;
    newSessionError.textContent = "";
    try {
      await createSession("", "", JSON.parse(sessionProfile.value));
      newSessionDialog.close();
    } catch (error) {
      newSessionError.textContent = error.message;
    } finally {
      createSessionButton.disabled = false;
    }
  });
  const customizeButtons = document.querySelectorAll(".customize-frontend");
  const settingsDialog = document.getElementById("workspaceSettings");
  const shareDialog = document.getElementById("workspaceShare");
  const sharePassword = document.getElementById("sharePassword");
  const shareStatus = document.getElementById("shareStatus");
  const shareError = document.getElementById("shareError");
  const shareSave = document.getElementById("shareSave");
  const shareDisable = document.getElementById("shareDisable");
  const shareLink = document.getElementById("shareLink");
  const settingsError = document.getElementById("settingsError");
  const reloadButton = document.getElementById("reloadPycodex");
  const reloadHint = document.getElementById("reloadPycodexHint");
  const defaultReloadHint = reloadHint.textContent;
  let reloadAborter = null;
  const isAdministrator = document.body.dataset.accessRole === "admin";
  for (const dialog of [settingsDialog, shareDialog, newSessionDialog, document.getElementById("oneLineAgentDialog")]) {
    dialog.querySelector("[data-close-dialog]").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); });
  }
  settingsDialog.addEventListener("close", () => {
    if (reloadAborter) {
      reloadAborter.abort();
      reloadAborter = null;
      showToast("Reload canceled");
    }
    settingsError.textContent = "";
    reloadHint.textContent = defaultReloadHint;
    reloadButton.disabled = false;
  });
  shareDialog.addEventListener("close", () => { sharePassword.value = ""; });
  customizeButtons.forEach(button => {
    button.addEventListener("click", () => {
      if (isAdministrator) settingsDialog.showModal();
    });
  });
  document.getElementById("frontendCustomization").addEventListener("click", async () => {
    if (!isAdministrator) return;
    settingsDialog.close();
    customizeButtons.forEach(control => { control.disabled = true; });
    try {
      workspace.dataset.mobilePane = "chat";
      await createSession("Customize interface", frontendCustomizationPrompt());
    } catch (error) {
      showToast("Unable to start customization: " + error.message);
    } finally {
      customizeButtons.forEach(control => { control.disabled = false; });
    }
  });
  async function updateSharing(method = "GET") {
    if (!isAdministrator) return;
    shareError.textContent = "";
    shareSave.disabled = shareDisable.disabled = true;
    try {
      const options = {method};
      if (method === "PUT") {
        options.headers = {"Content-Type": "application/json"};
        options.body = JSON.stringify({password: sharePassword.value});
      }
      const response = await fetch(relativeUrl("api/share"), options);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || payload.detail || `HTTP ${response.status}`);
      shareStatus.textContent = !payload.available
        ? "Set a workspace server administrator password to enable sharing."
        : payload.enabled ? "Sharing is on. Changing the password signs out existing visitors." : "Sharing is off.";
      shareSave.disabled = !payload.available;
      shareDisable.disabled = !payload.enabled;
      shareSave.textContent = payload.enabled ? "Change password" : "Set password";
      document.getElementById("shareLinkSection").hidden = !payload.enabled;
      shareLink.value = payload.enabled ? new URL(payload.url, window.location.origin).href : "";
      if (method === "PUT") sharePassword.value = "";
    } catch (error) {
      shareError.textContent = error.message;
      shareSave.disabled = false;
      shareDisable.disabled = false;
    }
  }
  document.getElementById("shareWorkspace").addEventListener("click", () => {
    if (!isAdministrator) return;
    settingsDialog.close();
    shareStatus.textContent = "Loading…";
    document.getElementById("shareLinkSection").hidden = true;
    shareDialog.showModal();
    updateSharing();
  });
  document.getElementById("shareForm").addEventListener("submit", event => {
    event.preventDefault();
    updateSharing("PUT");
  });
  shareDisable.addEventListener("click", () => updateSharing("DELETE"));
  function installOneLineAgentSettings() {
    const dialog = document.getElementById("oneLineAgentDialog");
    const profile = document.getElementById("oneLineAgentProfile");
    const create = document.getElementById("oneLineAgentCreate");
    const stop = document.getElementById("oneLineAgentStop");
    const open = document.getElementById("oneLineAgentSession");
    const status = document.getElementById("oneLineAgentStatus");
    const error = document.getElementById("oneLineAgentError");
    const commands = document.getElementById("oneLineAgentCommands");
    let binding = null;

    async function update(method = "GET") {
      error.textContent = "";
      create.disabled = stop.disabled = true;
      try {
        const options = {method};
        if (method === "POST") {
          options.headers = {"Content-Type": "application/json"};
          options.body = JSON.stringify({profile: JSON.parse(profile.value)});
        }
        const response = await fetch(relativeUrl("api/one-line-agent"), options);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || payload.detail || `HTTP ${response.status}`);
        binding = payload.binding;
        const defaultLabel = payload.default_profile === null ? "Default" : `Default (${payload.default_profile})`;
        profile.replaceChildren(new Option(defaultLabel, "null"));
        for (const name of payload.profiles) profile.add(new Option(name, JSON.stringify(name)));
        if (binding) profile.value = JSON.stringify(binding.profile);
        const enabled = !!binding?.enabled;
        profile.disabled = enabled;
        create.disabled = enabled;
        stop.disabled = !enabled;
        open.hidden = !binding?.session_id;
        commands.hidden = !enabled;
        status.textContent = !enabled ? "One-line Agent is off." :
          `Enabled · ${binding.session_count} session${binding.session_count === 1 ? "" : "s"} open.`;
        document.getElementById("oneLineAgentLinux").value = binding?.one_liners?.linux || "";
        document.getElementById("oneLineAgentWindows").value = binding?.one_liners?.windows || "";
      } catch (failure) {
        error.textContent = failure.message;
        create.disabled = !!binding?.enabled;
        stop.disabled = !binding?.enabled;
      }
    }

    document.getElementById("oneLineAgentSettings").addEventListener("click", () => {
      if (!isAdministrator) return;
      settingsDialog.close();
      status.textContent = "Loading…";
      commands.hidden = true;
      dialog.showModal();
      update();
    });
    document.getElementById("oneLineAgentForm").addEventListener("submit", event => {
      event.preventDefault();
      update("POST");
    });
    stop.addEventListener("click", () => update("DELETE"));
    open.addEventListener("click", () => {
      dialog.close();
      selectSession(binding.session_id);
    });
    for (const button of dialog.querySelectorAll("[data-copy-one-line-agent]")) {
      button.addEventListener("click", async () => {
        const field = document.getElementById(button.dataset.copyOneLineAgent);
        try {
          await navigator.clipboard.writeText(field.value);
          showToast("Terminal command copied");
        } catch (failure) {
          field.focus();
          field.select();
          error.textContent = "Select and copy the terminal command.";
        }
      });
    }
  }
  installOneLineAgentSettings();
  reloadButton.addEventListener("click", async () => {
    if (!isAdministrator) return;
    settingsError.textContent = "";
    reloadButton.disabled = true;
    const aborter = reloadAborter = new AbortController();
    const activeLocal = getSessions().filter(s => s.running && s.spinner !== "idle: sleeping");
    reloadHint.textContent = activeLocal.length
      ? `Waiting for active turn (${activeLocal.map(s => s.title || s.id).join(", ")})…`
      : "Checking active sessions…";
    try {
      while (true) {
        const checkResp = await fetch(relativeUrl("api/reload"), {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({session_id: getActiveSessionId(), check_only: true}),
          signal: aborter.signal,
        });
        const checkPayload = await checkResp.json();
        if (!checkResp.ok || !checkPayload.ok) {
          throw new Error(checkPayload.error || checkPayload.detail || `HTTP ${checkResp.status}`);
        }
        if (checkPayload.status === "ready") break;
        const names = (checkPayload.running_sessions || []).join(", ");
        reloadHint.textContent = names
          ? `Waiting for active turn (${names})…`
          : "Waiting for active turn to finish…";
        await new Promise(resolve => setTimeout(resolve, 400));
      }
      reloadHint.textContent = "Checking syntax & reloading modules…";
      const response = await fetch(relativeUrl("api/reload"), {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({session_id: getActiveSessionId()}),
        signal: aborter.signal,
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || payload.detail || `HTTP ${response.status}`);
      }
      if (reloadAborter === aborter) reloadAborter = null;
      settingsDialog.close();
      updateSessions(payload.sessions);
      renderSnapshot(payload.snapshot);
      showToast(payload.reloaded_modules
        ? `PyCodex reloaded (${payload.reloaded_modules} modules)`
        : "PyCodex reloaded");
    } catch (error) {
      if (aborter.signal.aborted) return;
      settingsError.textContent = error.message;
      reloadHint.textContent = defaultReloadHint;
      reloadButton.disabled = false;
    } finally {
      if (reloadAborter === aborter) reloadAborter = null;
    }
  });
  document.getElementById("shareCopy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(shareLink.value);
      showToast("Share link copied");
    } catch (error) {
      shareLink.focus();
      shareLink.select();
      shareError.textContent = "Select and copy the share link.";
    }
  });
}
