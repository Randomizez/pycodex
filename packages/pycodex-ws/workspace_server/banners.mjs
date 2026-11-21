import {element, isNearBottom, reconcileChildren, scrollToBottom} from "./workspace_dom.mjs";

const SVG_NS = "http://www.w3.org/2000/svg";
const ICON_SHAPES = {
  terminal: [
    ["rect", {x: "3", y: "4", width: "18", height: "16", rx: "2"}],
    ["path", {d: "m7 9 3 3-3 3m6 0h4"}],
  ],
  customize: [
    ["path", {d: "m15 4 5 5M4 20l4-1L20 7a2.8 2.8 0 0 0-4-4L4 15l-1 6M4 4v4M2 6h4"}],
  ],
  info: [
    ["circle", {cx: "12", cy: "12", r: "9"}],
    ["path", {d: "M12 16v-4M12 8h.01"}],
  ],
};

function createIcon(name) {
  const shapes = ICON_SHAPES[name];
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.6");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  for (const [tag, attrs] of shapes) {
    const child = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) child.setAttribute(key, value);
    svg.appendChild(child);
  }
  return svg;
}

function safeHref(href) {
  if (typeof href !== "string" || !href.trim()) {
    throw new TypeError("Banner href must be a non-empty string");
  }
  const parsed = new URL(href, window.location.href);
  if (!["http:", "https:", "mailto:"].includes(parsed.protocol)) {
    throw new TypeError("Unsupported banner href protocol: " + parsed.protocol);
  }
  return href;
}

function normalizeAction(action) {
  if (!action || typeof action !== "object" || Array.isArray(action)) {
    throw new TypeError("Banner action must be an object");
  }
  const label = action.label == null ? "" : String(action.label).trim();
  if (!label) throw new TypeError("Banner action requires a label");
  const onClick = action.onClick == null ? null : action.onClick;
  if (onClick !== null && typeof onClick !== "function") {
    throw new TypeError("Banner action onClick must be a function");
  }
  return {
    label,
    href: action.href == null ? null : safeHref(action.href),
    target: action.target === "_blank" ? "_blank" : "",
    copyText: action.copyText == null ? null : String(action.copyText),
    copyLabel: action.copyLabel == null ? label : String(action.copyLabel),
    viewHref: action.viewHref == null ? null : String(action.viewHref),
    srcPath: action.srcPath == null ? null : String(action.srcPath),
    onClick,
  };
}

function normalizeDetails(details) {
  if (details == null || details === "") return {details: null, hasCallback: false};
  if (typeof details === "string") {
    return {details: {kind: "text", text: details}, hasCallback: false};
  }
  if (!Array.isArray(details)) {
    throw new TypeError("Banner details must be a string or array");
  }
  if (!details.length) return {details: null, hasCallback: false};
  let hasCallback = false;
  const rows = details.map(row => {
    if (!row || typeof row !== "object" || Array.isArray(row)) {
      throw new TypeError("Banner detail row must be an object");
    }
    const label = row.label == null ? "" : String(row.label).trim();
    if (!label) throw new TypeError("Banner detail row requires a label");
    const onClick = row.onClick == null ? null : row.onClick;
    if (onClick !== null && typeof onClick !== "function") {
      throw new TypeError("Banner detail row onClick must be a function");
    }
    if (onClick) hasCallback = true;
    return {
      label,
      value: row.value == null ? "" : String(row.value),
      copyable: Boolean(row.copyable),
      copyLabel: row.copyLabel == null ? label : String(row.copyLabel),
      actionLabel: row.actionLabel == null ? "" : String(row.actionLabel),
      href: row.href == null ? null : safeHref(row.href),
      target: row.target === "_blank" ? "_blank" : "",
      viewHref: row.viewHref == null ? null : String(row.viewHref),
      srcPath: row.srcPath == null ? null : String(row.srcPath),
      onClick,
    };
  });
  return {details: {kind: "rows", rows}, hasCallback};
}

function normalizeBanner(id, config) {
  if (typeof id !== "string" || !id.trim()) {
    throw new TypeError("Banner id must be a non-empty string");
  }
  if (!config || typeof config !== "object" || Array.isArray(config)) {
    throw new TypeError("Banner must be an object or null");
  }
  const label = config.label == null ? "" : String(config.label).trim();
  const text = config.text == null ? "" : String(config.text).trim();
  if (!label && !text) {
    throw new TypeError("Banner requires a label or text");
  }
  const icon = config.icon == null || config.icon === "" ? "" : String(config.icon);
  if (icon && !ICON_SHAPES[icon]) {
    throw new TypeError("Unknown banner icon: " + icon);
  }
  if (config.actions != null && !Array.isArray(config.actions)) {
    throw new TypeError("Banner actions must be an array");
  }
  const actions = (config.actions || []).map(normalizeAction);
  const {details, hasCallback: detailsCallback} = normalizeDetails(config.details);
  const hasCallback = detailsCallback || actions.some(action => action.onClick !== null);
  return {
    id,
    label,
    text,
    icon,
    sessionId: config.sessionId == null || config.sessionId === "" ? null : String(config.sessionId),
    adminOnly: Boolean(config.adminOnly),
    actions,
    details,
    hasCallback,
  };
}

export function createSessionBanners({
  container, log, getActiveSessionId, isAdmin = () => true, openView = async () => {}, showToast = () => {},
}) {
  const entries = new Map();
  let renderedSessionId = "";
  let nextElementId = 0;

  function buildActionControl(spec, copyValue, fallbackInput = null) {
    if (spec.href !== null) {
      const link = element("a", "session-banner-btn", spec.label);
      link.href = spec.href;
      if (spec.target === "_blank") {
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      }
      return link;
    }
    const button = element("button", "session-banner-btn", spec.label);
    button.addEventListener("click", () => {
      if (spec.copyText !== null) {
        copyValue(spec.copyText, spec.copyLabel, fallbackInput);
      } else if (spec.viewHref !== null) {
        Promise.resolve(openView(spec.viewHref, spec.srcPath, getActiveSessionId()))
          .catch(error => showToast("Unable to open view: " + error.message));
      } else if (spec.onClick) {
        spec.onClick();
      }
    });
    return button;
  }

  function buildNode(entry) {
    const {config} = entry;
    const root = element("section", "session-banner");
    root.dataset.bannerId = config.id;
    if (config.adminOnly) root.dataset.adminOnly = "true";
    root.setAttribute("aria-label", config.label || config.text);

    const detailInputs = new Map();
    let detailsEl = null;
    let toggleBtn = null;
    const setExpanded = open => {
      entry.expanded = open;
      if (!detailsEl || !toggleBtn) return;
      const stick = open && detailsEl.hidden ? isNearBottom(log) : false;
      detailsEl.hidden = !open;
      toggleBtn.setAttribute("aria-expanded", String(open));
      if (open && stick && log.clientHeight) scrollToBottom(log);
    };
    const copyValue = async (value, label, fallbackInput = null) => {
      try {
        await navigator.clipboard.writeText(value);
        showToast(`${label} copied`);
      } catch (_error) {
        if (fallbackInput && detailsEl) {
          setExpanded(true);
          fallbackInput.focus();
          fallbackInput.select();
          showToast(`Select and copy the ${label.toLowerCase()}.`);
        } else {
          showToast(`Unable to copy ${label.toLowerCase()}.`);
        }
      }
    };

    if (config.details) {
      detailsEl = element("div", "session-banner-details");
      const detailsId = `sessionBannerDetails-${++nextElementId}`;
      detailsEl.id = detailsId;
      detailsEl.hidden = !entry.expanded;
      if (config.details.kind === "text") {
        detailsEl.appendChild(element("p", "session-banner-note", config.details.text));
      } else {
        for (const row of config.details.rows) {
          const rowEl = element("div", "session-banner-row");
          const inputId = `sessionBannerField-${++nextElementId}`;
          const labelEl = element("label", "", row.label);
          labelEl.htmlFor = inputId;
          const inputEl = element("input");
          inputEl.id = inputId;
          inputEl.readOnly = true;
          inputEl.spellcheck = false;
          inputEl.value = row.value;
          if (!detailInputs.has(row.value)) detailInputs.set(row.value, inputEl);
          rowEl.append(labelEl, inputEl);
          if (row.href !== null || row.viewHref !== null || row.onClick || row.copyable) {
            const rowAction = {
              label: row.actionLabel || (row.copyable && row.href === null && row.viewHref === null && !row.onClick ? "Copy" : "Open"),
              href: row.href,
              target: row.target,
              copyText: row.copyable && row.href === null && row.viewHref === null && !row.onClick ? row.value : null,
              copyLabel: row.copyLabel,
              viewHref: row.viewHref,
              srcPath: row.srcPath,
              onClick: row.onClick,
            };
            rowEl.appendChild(buildActionControl(rowAction, copyValue, inputEl));
          }
          detailsEl.appendChild(rowEl);
        }
      }
      toggleBtn = element("button", "session-banner-btn session-banner-toggle", "Details");
      toggleBtn.setAttribute("aria-expanded", String(entry.expanded));
      toggleBtn.setAttribute("aria-controls", detailsId);
      toggleBtn.addEventListener("click", () => setExpanded(detailsEl.hidden));
    }

    const bar = element("div", "session-banner-bar");
    const main = element("div", "session-banner-main");
    if (config.label) {
      const badge = element("span", "session-banner-badge");
      if (config.icon) badge.appendChild(createIcon(config.icon));
      badge.appendChild(element("span", "", config.label));
      main.appendChild(badge);
    }
    if (config.text) {
      main.appendChild(element("span", "session-banner-text", config.text));
    }
    bar.appendChild(main);

    if (config.actions.length || toggleBtn) {
      const actionsEl = element("div", "session-banner-actions");
      for (const action of config.actions) {
        const fallback = action.copyText !== null ? detailInputs.get(action.copyText) || null : null;
        actionsEl.appendChild(buildActionControl(action, copyValue, fallback));
      }
      if (toggleBtn) actionsEl.appendChild(toggleBtn);
      bar.appendChild(actionsEl);
    }

    root.appendChild(bar);
    if (detailsEl) root.appendChild(detailsEl);
    entry.setExpanded = setExpanded;
    return root;
  }

  function render({collapse = false} = {}) {
    const sessionId = getActiveSessionId();
    const switched = collapse || sessionId !== renderedSessionId;
    renderedSessionId = sessionId;
    if (switched) {
      for (const entry of entries.values()) {
        if (entry.expanded) entry.setExpanded(false);
      }
    }
    const visible = [];
    for (const entry of entries.values()) {
      if (entry.config.adminOnly && !isAdmin()) continue;
      if (entry.config.sessionId !== null && (!sessionId || entry.config.sessionId !== sessionId)) continue;
      if (!entry.node) entry.node = buildNode(entry);
      visible.push(entry.node);
    }
    reconcileChildren(container, visible);
    container.hidden = visible.length === 0;
  }

  function upsert(scope, id, rawConfig) {
    if (typeof id !== "string" || !id.trim()) {
      throw new TypeError("Banner id must be a non-empty string");
    }
    const key = `${scope}:${id}`;
    if (rawConfig === null || rawConfig === undefined) {
      if (entries.delete(key)) render();
      return () => {};
    }
    const config = normalizeBanner(id, rawConfig);
    const signature = config.hasCallback ? null : JSON.stringify(config);
    const previous = entries.get(key);
    const token = {};
    if (previous && previous.signature !== null && previous.signature === signature) {
      previous.token = token;
      return () => {
        if (entries.get(key)?.token === token && entries.delete(key)) render();
      };
    }
    const expanded = Boolean(previous && previous.config.sessionId === config.sessionId && previous.expanded);
    const entry = {key, config, signature, expanded, token, node: null, setExpanded() {}};
    entries.set(key, entry);
    render();
    return () => {
      if (entries.get(key)?.token === token && entries.delete(key)) render();
    };
  }

  function clearSession(sessionId) {
    if (!sessionId) return;
    let removed = false;
    for (const [key, entry] of entries.entries()) {
      if (entry.config.sessionId === sessionId) {
        entries.delete(key);
        removed = true;
      }
    }
    if (removed) render();
  }

  return {
    set: (id, config) => upsert("custom", id, config),
    setBuiltin: (id, config) => upsert("builtin", id, config),
    clearSession,
    render,
  };
}

export function loadEmbeddedBanners(root, banners) {
  const nodes = root.querySelectorAll("script[data-pycodex-banner]");
  nodes.forEach((node, index) => {
    const id = node.dataset.pycodexBanner || node.id || `embedded-${index + 1}`;
    banners.set(id, JSON.parse(node.textContent));
  });
}
