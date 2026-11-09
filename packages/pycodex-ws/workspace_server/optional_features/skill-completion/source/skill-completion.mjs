/** Draft offsets count each context chip as one position. */
export function skillQuery(parts, selection) {
  if (selection.start !== selection.end) return null;
  const text = parts.map(part => part.type === "text" ? part.text : "\uFFFC").join("");
  const match = /(?:^|\s)([@$][A-Za-z0-9_-]*)$/.exec(text.slice(0, selection.start));
  if (!match || /^[A-Za-z0-9_-]/.test(text.slice(selection.end))) return null;
  return {start: selection.start - match[1].length, end: selection.end,
    prefix: match[1].slice(1), key: `${selection.start}:${match[1]}`};
}

export function matchingItems(items, prefix) {
  return items.filter(item => item.name.toLowerCase().startsWith(prefix.toLowerCase()));
}

/** Optional mount(host) adapter; importing this file alone does not install anything. */
export function mountSkillCompletion(host, options = {}) {
  const prompt = options.element || document.getElementById("prompt");
  if (!prompt) throw new Error("Skill completion requires a prompt input");
  const queryDraft = options.query || skillQuery;
  const loadItems = options.loadItems || (async signal =>
    (await host.request("api/skills", {signal})).skills);
  const insertText = options.insertText || (item => `$${item.name} `);
  const menu = document.createElement("div");
  menu.className = "skill-completion-menu";
  menu.id = `${prompt.id || "prompt"}-skill-completion`;
  menu.setAttribute("role", "listbox");
  menu.setAttribute("aria-label", "Completion candidates");
  menu.hidden = true;
  const anchor = prompt.parentElement;
  const hadAnchorClass = anchor.classList.contains("skill-completion-anchor");
  anchor.classList.add("skill-completion-anchor");
  anchor.append(menu);
  const attributes = ["aria-controls", "aria-autocomplete", "aria-expanded", "aria-activedescendant"];
  const originalAttributes = attributes.map(name => [name, prompt.getAttribute(name)]);
  prompt.setAttribute("aria-controls", menu.id);
  prompt.setAttribute("aria-autocomplete", "list");
  const lifetime = new AbortController();
  let items = null, matches = [], query = null, index = 0, dismissed = null;
  let composing = false, suspended = false, disposed = false, error = "";
  let request = null, generation = 0;

  function hide() {
    menu.hidden = true;
    prompt.setAttribute("aria-expanded", "false");
    prompt.removeAttribute("aria-activedescendant");
  }
  function invalidate() {
    generation++;
    request?.abort();
    request = null;
    items = null;
    error = "";
    query = null;
    dismissed = null;
    hide();
  }
  function currentQuery() {
    const current = host.getDraft();
    const value = queryDraft(current.parts, current.selection);
    return value ? {...value, key: `${host.getActiveSessionId()}:${value.key ?? `${value.start}:${value.end}:${value.prefix}`}`} : null;
  }
  function refresh() {
    if (request || items !== null) return;
    const controller = new AbortController(), version = ++generation;
    request = controller;
    Promise.resolve().then(() => loadItems(controller.signal)).then(data => {
      if (disposed || controller.signal.aborted || generation !== version) return;
      if (!Array.isArray(data) || data.some(item => typeof item?.name !== "string")) {
        throw new Error("Invalid completion catalog");
      }
      items = data;
      error = "";
    }).catch(failure => {
      if (disposed || controller.signal.aborted || generation !== version) return;
      items = [];
      error = `Unable to load completion candidates: ${failure.message}`;
    }).finally(() => {
      if (disposed || generation !== version) return;
      request = null;
      update();
    });
  }
  function render() {
    menu.replaceChildren();
    if (!matches.length) {
      if (!error) { hide(); return; }
      const status = document.createElement("p");
      status.className = "skill-completion-status";
      status.setAttribute("role", "status");
      status.textContent = error;
      menu.append(status);
    }
    matches.forEach((item, number) => {
      const option = document.createElement("button");
      option.type = "button";
      option.tabIndex = -1;
      option.id = `${menu.id}-${number}`;
      option.className = "skill-completion-option";
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", String(number === index));
      const name = document.createElement("strong"), description = document.createElement("span");
      name.textContent = item.name;
      description.textContent = item.description || "";
      option.append(name, description);
      option.title = item.description || "";
      option.addEventListener("pointerdown", event => event.preventDefault());
      option.addEventListener("click", () => accept(number));
      menu.append(option);
    });
    menu.hidden = false;
    prompt.setAttribute("aria-expanded", "true");
    if (matches.length) {
      prompt.setAttribute("aria-activedescendant", `${menu.id}-${index}`);
      menu.children[index].scrollIntoView({block: "nearest"});
    } else prompt.removeAttribute("aria-activedescendant");
  }
  function update() {
    if (disposed || suspended || composing || document.activeElement !== prompt ||
        !host.getActiveSessionId()) { hide(); return; }
    const value = currentQuery();
    if (!value) { query = null; dismissed = null; hide(); return; }
    if (value.key === dismissed) { hide(); return; }
    if (value.key !== query?.key) index = 0;
    query = value;
    refresh();
    matches = matchingItems(items || [], value.prefix);
    index = Math.min(index, Math.max(0, matches.length - 1));
    render();
  }
  function accept(number = index) {
    if (disposed || suspended) return;
    const value = currentQuery();
    if (!value || value.key !== query?.key) { hide(); return; }
    const text = insertText(matches[number]);
    hide();
    host.replaceDraftText(value.start, value.end, text);
  }
  prompt.addEventListener("keydown", event => {
    if (event.defaultPrevented || event.isComposing || composing || event.keyCode === 229 ||
        event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
    update();
    if (menu.hidden) return;
    if (event.key === "Escape") { dismissed = query.key; hide(); }
    else if (matches.length && ["ArrowUp", "ArrowDown"].includes(event.key)) {
      index = (index + (event.key === "ArrowDown" ? 1 : -1) + matches.length) % matches.length;
      render();
    } else if (matches.length && ["Enter", "Tab"].includes(event.key)) accept();
    else return;
    event.preventDefault();
    event.stopImmediatePropagation();
  }, {capture: true, signal: lifetime.signal});
  prompt.addEventListener("input", update, {signal: lifetime.signal});
  prompt.addEventListener("focus", update, {signal: lifetime.signal});
  prompt.addEventListener("blur", hide, {signal: lifetime.signal});
  prompt.addEventListener("compositionstart", () => { composing = true; hide(); }, {signal: lifetime.signal});
  prompt.addEventListener("compositionend", () => { composing = false; update(); }, {signal: lifetime.signal});
  document.addEventListener("selectionchange", update, {signal: lifetime.signal});
  window.addEventListener("pagehide", () => { suspended = true; invalidate(); }, {signal: lifetime.signal});
  window.addEventListener("pageshow", () => { suspended = false; update(); }, {signal: lifetime.signal});
  const unsubscribe = host.onSessionChanged(() => { invalidate(); update(); });
  update();
  return () => {
    if (disposed) return;
    disposed = true;
    invalidate();
    lifetime.abort();
    unsubscribe();
    menu.remove();
    if (!hadAnchorClass) anchor.classList.remove("skill-completion-anchor");
    for (const [name, value] of originalAttributes) {
      if (value === null) prompt.removeAttribute(name);
      else prompt.setAttribute(name, value);
    }
  };
}
