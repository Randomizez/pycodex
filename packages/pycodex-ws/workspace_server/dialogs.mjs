import {Composer} from "./content_views.mjs";
import {encodeViewPath, relativeUrl} from "./workspace_dom.mjs";

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
  let hoverCloseTimer = null;
  const clearHoverClose = () => {
    clearTimeout(hoverCloseTimer);
    hoverCloseTimer = null;
  };
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
    clearHoverClose();
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
    clearHoverClose();
    if (!pinned) hoverCloseTimer = setTimeout(() => show(false), 180);
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

function frontendCustomizationPrompt(catalogPath) {
  const {frontendDirectory, packageDirectory} = document.body.dataset;
  return [
    "<frontend_customization>",
    "Help me customize the workspace interface. Make changes only when I request them.",
    "Use my preferred language, then the language of my messages, then my browser languages: " +
      (navigator.languages.join(", ") || "not supplied"),
    `User customization directory: ${frontendDirectory}`,
    `Packaged page and native modules: ${packageDirectory}/workspace.html and workspace.mjs`,
    `Read the host, HTTP and WebSocket contracts in ${packageDirectory}/WORKSPACE_API.md before changing behavior.`,
    `Use the optional feature catalog at ${catalogPath} to resolve requested feature names and IDs.`,
    `For a requested feature PR, follow ${packageDirectory}/optional_features/CONTRIBUTING.md in a separate checkout.`,
    "Read existing custom files first. Prefer small custom.css and custom.js overrides, using mount(host) and workspace-relative URLs.",
    "Integrate chosen features into the existing customization. Preserve unrelated edits, DOM contracts, template placeholders and Settings.",
    "If a full template override is needed, copy the packaged template only when no user copy exists. Keep installed files intact.",
    "These files apply to every workspace of this OS user. Back up substantial edits outside the served directory, including the original template for comparison.",
    "Verify affected desktop and mobile interactions and asynchronous updates. Refresh to apply; use ?builtin=1 for recovery and rename the customization directory to undo.",
    "</frontend_customization>",
  ].join("\n");
}

export function installDialogs({
  getSessions, stateForSession, createSession, selectSession, openView, showToast,
  markCustomizeSession = () => {},
}) {
  const newTabButton = document.getElementById("newTab");
  const newSessionDialog = document.getElementById("newSessionDialog");
  const sessionNodeField = document.getElementById("sessionNodeField");
  const sessionNode = document.getElementById("sessionNode");
  const sessionProfile = document.getElementById("sessionProfile");
  const newSessionError = document.getElementById("newSessionError");
  const createSessionButton = document.getElementById("createSessionButton");
  let newSessionRevision = 0;
  let dialogOpenRevision = 0;
  let creatingSession = false;
  let profileSelectReady = false;
  let localSessionCatalog = null;

  function syncNewSessionControls() {
    if (sessionNode) {
      sessionNode.disabled = creatingSession;
    }
    sessionProfile.disabled = creatingSession || !profileSelectReady;
    createSessionButton.disabled = creatingSession || !profileSelectReady || !sessionProfile.options.length;
  }

  function applyLocalSessionProfiles(catalog) {
    sessionProfile.replaceChildren();
    if (catalog.local_profile_error) {
      profileSelectReady = false;
      newSessionError.textContent = catalog.local_profile_error;
      syncNewSessionControls();
      return;
    }
    const defaultLabel = catalog.default_profile === null
      ? "Default" : `Default (${catalog.default_profile})`;
    sessionProfile.replaceChildren(new Option(defaultLabel, "null"));
    for (const profile of catalog.profiles) {
      sessionProfile.add(new Option(profile, JSON.stringify(profile)));
    }
    profileSelectReady = true;
    newSessionError.textContent = "";
    syncNewSessionControls();
  }

  async function loadPeerNodeProfiles(nodeId) {
    const current = ++newSessionRevision;
    newSessionError.textContent = "";
    sessionProfile.replaceChildren(new Option("Loading profiles…", ""));
    profileSelectReady = false;
    syncNewSessionControls();
    try {
      const response = await fetch(
        relativeUrl(`api/peer-nodes/${encodeURIComponent(nodeId)}/profiles`),
        {cache: "no-store"}
      );
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || payload.detail || `HTTP ${response.status}`);
      }
      if (current !== newSessionRevision || sessionNode.value !== nodeId) return;
      const defaultLabel = payload.default_profile === null
        ? "Default" : `Default (${payload.default_profile})`;
      sessionProfile.replaceChildren(new Option(defaultLabel, JSON.stringify(`peer:${nodeId}`)));
      for (const profile of payload.profiles || []) {
        sessionProfile.add(new Option(profile, JSON.stringify(`peer:${nodeId}:${profile}`)));
      }
      profileSelectReady = true;
      syncNewSessionControls();
    } catch (error) {
      if (current === newSessionRevision && sessionNode.value === nodeId) {
        sessionProfile.replaceChildren();
        profileSelectReady = false;
        syncNewSessionControls();
        newSessionError.textContent = error.message;
      }
    }
  }

  if (sessionNode) {
    sessionNode.addEventListener("change", () => {
      if (creatingSession) return;
      newSessionRevision++;
      if (!sessionNode.value) {
        if (localSessionCatalog) applyLocalSessionProfiles(localSessionCatalog);
      } else {
        loadPeerNodeProfiles(sessionNode.value);
      }
    });
  }
  newSessionDialog.addEventListener("close", () => {
    dialogOpenRevision++;
    newSessionRevision++;
    creatingSession = false;
    newSessionError.textContent = "";
    syncNewSessionControls();
  });

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
  function renderPeerNodeOptions(peers, preserveSelection = false) {
    if (!sessionNode) return;
    const previousValue = preserveSelection ? sessionNode.value : "";
    sessionNode.replaceChildren(new Option("This device", ""));
    for (const peer of peers || []) {
      sessionNode.add(new Option(peer.alias || peer.name, peer.id));
    }
    sessionNode.value = Array.from(sessionNode.options).some(o => o.value === previousValue)
      ? previousValue : "";
  }

  async function loadPeerNodesAsync(openRev) {
    try {
      const response = await fetch(relativeUrl("api/peer-nodes"), {cache: "no-store"});
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || payload.detail || `HTTP ${response.status}`);
      }
      if (openRev !== dialogOpenRevision || !newSessionDialog.open) return;
      renderPeerNodeOptions(payload.peer_nodes, true);
    } catch (_error) {
      if (openRev !== dialogOpenRevision || !newSessionDialog.open) return;
      renderPeerNodeOptions([], true);
    }
  }

  const customProfileButton = document.getElementById("customProfile");
  customProfileButton.addEventListener("click", async () => {
    customProfileButton.disabled = true;
    try {
      const response = await fetch(relativeUrl("api/sessions"), {cache: "no-store"});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || payload.detail || `HTTP ${response.status}`);
      const openRev = ++dialogOpenRevision;
      newSessionRevision++;
      creatingSession = false;
      localSessionCatalog = payload;
      const routerConnected = Boolean(payload.router_connected);
      if (!routerConnected && !payload.profiles.length) {
        showToast("No named profiles are configured.");
        return;
      }
      if (sessionNodeField && sessionNode) {
        sessionNodeField.hidden = !routerConnected;
        if (routerConnected) {
          const loadingOpt = new Option("Loading peer devices…", "");
          loadingOpt.disabled = true;
          sessionNode.replaceChildren(new Option("This device", ""), loadingOpt);
          sessionNode.value = "";
        } else {
          sessionNode.replaceChildren();
        }
      }
      applyLocalSessionProfiles(payload);
      newSessionDialog.showModal();
      if (routerConnected) {
        loadPeerNodesAsync(openRev);
      }
    } catch (error) {
      showToast("Unable to create session: " + error.message);
    } finally {
      customProfileButton.disabled = false;
    }
  });
  document.getElementById("newSessionForm").addEventListener("submit", async event => {
    event.preventDefault();
    if (creatingSession || createSessionButton.disabled || !profileSelectReady || !sessionProfile.options.length) return;
    const submitDialogRevision = dialogOpenRevision;
    const selectedProfile = JSON.parse(sessionProfile.value);
    creatingSession = true;
    newSessionError.textContent = "";
    syncNewSessionControls();
    try {
      await createSession({profile: selectedProfile});
      if (submitDialogRevision === dialogOpenRevision && newSessionDialog.open) {
        newSessionDialog.close();
      }
    } catch (error) {
      if (submitDialogRevision === dialogOpenRevision && newSessionDialog.open) {
        newSessionError.textContent = error.message;
      }
    } finally {
      if (submitDialogRevision === dialogOpenRevision) {
        creatingSession = false;
        syncNewSessionControls();
      }
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
  const rebootDialog = document.getElementById("rebootDialog");
  const rebootRunning = document.getElementById("rebootRunning");
  const rebootError = document.getElementById("rebootError");
  const rebootButtons = [document.getElementById("rebootOnly"), document.getElementById("rebootUpdate")];
  const isAdministrator = document.body.dataset.accessRole === "admin";
  for (const dialog of [settingsDialog, shareDialog, rebootDialog, newSessionDialog, document.getElementById("externalCallDialog")]) {
    dialog.querySelector("[data-close-dialog]").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); });
  }
  settingsDialog.addEventListener("close", () => {
    settingsError.textContent = "";
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
    const catalogPath = document.body.dataset.packageDirectory + "/optional_features/README.md";
    let sessionId;
    try {
      sessionId = await createSession({
        title: "Customize interface",
        prompt: frontendCustomizationPrompt(catalogPath),
        trigger: false,
      });
      markCustomizeSession(sessionId);
      await openView(encodeViewPath(catalogPath), null, sessionId);
    } catch (error) {
      showToast((sessionId ? "Unable to open the feature catalog: " : "Unable to start customization: ") + error.message);
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
  function installExternalCallSettings() {
    const dialog = document.getElementById("externalCallDialog");
    const profile = document.getElementById("externalCallProfile");
    const create = document.getElementById("externalCallCreate");
    const error = document.getElementById("externalCallError");
    let revision = 0;

    async function loadProfiles() {
      const current = ++revision;
      error.textContent = "";
      create.disabled = true;
      try {
        const response = await fetch(relativeUrl("api/external-call-sessions"));
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || payload.detail || `HTTP ${response.status}`);
        if (current !== revision) return;
        const label = payload.default_profile === null ? "Default" : `Default (${payload.default_profile})`;
        profile.replaceChildren(new Option(label, "null"));
        for (const name of payload.profiles) profile.add(new Option(name, JSON.stringify(name)));
      } catch (failure) {
        if (current === revision) error.textContent = failure.message;
      } finally {
        if (current === revision) create.disabled = !profile.options.length;
      }
    }

    async function createExternalSession() {
      const current = ++revision;
      error.textContent = "";
      create.disabled = true;
      try {
        const response = await fetch(relativeUrl("api/external-call-sessions"), {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({profile: JSON.parse(profile.value)}),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || payload.detail || `HTTP ${response.status}`);
        if (current !== revision) return;
        const binding = payload.binding;
        stateForSession(binding.session_id).externalCall = binding;
        dialog.close();
        await selectSession(binding.session_id);
      } catch (failure) {
        if (current === revision) error.textContent = failure.message;
      } finally {
        if (current === revision) create.disabled = !profile.options.length;
      }
    }

    dialog.addEventListener("close", () => {
      revision++;
      error.textContent = "";
    });
    document.getElementById("externalCallSettings").addEventListener("click", () => {
      if (!isAdministrator) return;
      settingsDialog.close();
      profile.replaceChildren();
      dialog.showModal();
      loadProfiles();
    });
    document.getElementById("externalCallForm").addEventListener("submit", event => {
      event.preventDefault();
      if (!create.disabled) createExternalSession();
    });
  }
  installExternalCallSettings();
  reloadButton.addEventListener("click", () => {
    if (!isAdministrator) return;
    const running = getSessions().filter(session => session.running).map(session => session.title || "Untitled");
    rebootRunning.textContent = running.length
      ? `Waiting for ${running.length} running session(s): ${running.join(", ")}`
      : "No session is running.";
    rebootError.textContent = "";
    rebootButtons.forEach(button => { button.disabled = false; });
    settingsDialog.close();
    rebootDialog.showModal();
  });
  rebootButtons.forEach((button, index) => button.addEventListener("click", async () => {
    rebootError.textContent = "";
    rebootButtons.forEach(item => { item.disabled = true; });
    rebootRunning.textContent = index
      ? "Updating, then waiting for running turns to finish…"
      : "Waiting for running turns to finish…";
    try {
      const response = await fetch(relativeUrl("api/reload"), {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({upgrade: index === 1}),
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || payload.detail || `HTTP ${response.status}`);
      }
      rebootRunning.textContent = "PyCodex restarted. Refresh the page and log in again.";
    } catch (error) {
      rebootError.textContent = error.message;
      rebootButtons.forEach(item => { item.disabled = false; });
    }
  }));
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
