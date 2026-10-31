function text(value) {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value.map(part => part.text || part.refusal || `[${part.type}]`).join("\n");
  }
  return JSON.stringify(value, null, 2);
}

function transcriptBlocks(records) {
  const blocks = [];
  const calls = new Map();
  for (const record of records) {
    const item = record.payload;
    const key = `record:${record.index}`;
    if (record.type === "compacted") {
      blocks.push({key, kind: "compaction", label: "Compaction", body: item.message});
    } else if (item.type === "function_call" || item.type === "custom_tool_call") {
      const block = {
        key: `call:${item.call_id}`, kind: "tool", callId: item.call_id,
        toolName: item.name,
        label: [item.namespace, item.name].filter(Boolean).join("."),
        input: text(item.type === "custom_tool_call" ? item.input : item.arguments),
      };
      calls.set(item.call_id, block);
      blocks.push(block);
    } else if (item.type === "function_call_output" || item.type === "custom_tool_call_output") {
      let block = calls.get(item.call_id);
      if (!block) {
        block = {key: `call:${item.call_id}`, kind: "tool", callId: item.call_id,
          toolName: item.name, label: item.name || "Tool"};
        calls.set(item.call_id, block);
        blocks.push(block);
      }
      block.output = text(item.output);
    } else if (item.type === "message" || item.type === "agent_message") {
      blocks.push({
        key, kind: item.role || "agent_message",
        label: item.type === "agent_message" ? `${item.author} → ${item.recipient}` : item.role,
        phase: item.phase, body: text(item.content),
      });
    } else if (item.type === "reasoning") {
      const body = [...(item.summary || []), ...(item.content || [])]
        .map(part => part.text || "").filter(Boolean).join("\n\n") || "No readable reasoning was recorded.";
      blocks.push({key, kind: "reasoning", label: "Reasoning", body});
    } else {
      blocks.push({key, kind: "raw", label: item.type || record.type, body: text(item)});
    }
  }
  return blocks;
}

function toolStatus(block) {
  const status = {state: block.output === undefined ? "pending" : "completed"};
  if (block.toolName !== "exec" && block.toolName !== "wait") return status;
  if (block.toolName === "wait" && block.input !== undefined) {
    let args;
    try {
      args = JSON.parse(block.input);
    } catch (error) {
      // Invalid arguments are still valid history to display.
      if (!(error instanceof SyntaxError)) throw error;
    }
    if (typeof args?.cell_id === "string") status.cellId = args.cell_id;
  }
  // Read only the exec/wait protocol header, never text printed by the script.
  const result = /^Script (running with cell ID (\S+)|completed|failed|terminated)\nWall time [\d.]+ seconds\nOutput:\n/.exec(block.output);
  if (result) {
    status.state = result[2] ? "running" : result[1];
    if (result[2]) status.cellId = result[2];
  }
  return status;
}

export class TranscriptView {
  constructor({viewport, request, renderMarkdown, sessionId, agentId, agentPath}) {
    this.viewport = viewport;
    this.document = viewport.ownerDocument;
    this.window = this.document.defaultView;
    this.request = request;
    this.renderMarkdown = renderMarkdown;
    this.position = [0, 0];
    this.sessionId = sessionId;
    this.agentId = agentId;
    this.records = new Map();
    this.source = null;
    this.nextCursor = null;
    this.agentSessionId = null;
    this.active = false;
    this.socket = null;
    this.requestController = null;
    this.reloadTimer = null;
    this.retryTimer = null;
    this.pendingRefresh = false;
    this.pendingReset = false;
    this.lastLoad = 0;
    this.root = this.element("section", "transcript");
    this.root.dataset.sessionId = sessionId;
    const heading = this.element("div", "snapshot-heading transcript-heading");
    const label = this.element("span", "snapshot-label", agentPath);
    label.title = `Agent: ${agentId}\nSession: ${sessionId}`;
    this.status = this.element("span", "snapshot-stamp", "Loading…");
    this.refresh = this.element("button", "content-mode-toggle", "Refresh");
    this.refresh.type = "button";
    this.refresh.addEventListener("click", () => this.load());
    this.error = this.element("div", "transcript-error");
    this.error.setAttribute("role", "alert");
    this.error.hidden = true;
    this.older = this.element("button", "content-mode-toggle", "Load earlier");
    this.older.type = "button";
    this.older.hidden = true;
    this.older.addEventListener("click", () => this.load(true));
    heading.append(label, this.status, this.older, this.refresh);
    this.scroll = this.element("div", "transcript-scroll");
    this.scroll.tabIndex = 0;
    this.scroll.setAttribute("aria-label", agentPath + " transcript");
    this.list = this.element("div", "transcript-list", "Loading transcript…");
    this.scroll.appendChild(this.list);
    this.root.append(heading, this.error, this.scroll);
    this.visibilityChanged = () => {
      if (this.document.hidden) this.disconnect();
      else {
        this.load();
        this.connect();
      }
    };
  }

  element(tag, className, value) {
    const node = this.document.createElement(tag);
    node.className = className;
    if (value !== undefined) node.textContent = value;
    return node;
  }

  start() {
    if (this.active) return;
    this.active = true;
    [this.scroll.scrollLeft, this.scroll.scrollTop] = this.position;
    this.document.addEventListener("visibilitychange", this.visibilityChanged);
    if (!this.document.hidden) {
      this.load();
      this.connect();
    }
  }

  disconnect() {
    clearTimeout(this.reloadTimer);
    clearTimeout(this.retryTimer);
    this.reloadTimer = this.retryTimer = null;
    this.requestController?.abort();
    this.requestController = null;
    const socket = this.socket;
    this.socket = null;
    if (socket) socket.close();
  }

  stop() {
    this.active = false;
    this.document.removeEventListener("visibilitychange", this.visibilityChanged);
    this.disconnect();
    this.position = [this.scroll.scrollLeft, this.scroll.scrollTop];
  }

  schedule(reset = false) {
    this.pendingReset ||= reset;
    if (this.reloadTimer !== null || !this.active || this.document.hidden) return;
    // Read history when records change, coalescing bursts from the same model/tool operation.
    this.reloadTimer = setTimeout(() => {
      this.reloadTimer = null;
      this.load();
    }, Math.max(150, 1000 - (Date.now() - this.lastLoad)));
  }

  connect() {
    if (this.socket || !this.active || this.document.hidden) return;
    const url = new URL("ws/session", this.window.location.href);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    url.searchParams.set("session_id", this.sessionId);
    url.searchParams.set("agent_id", this.agentId);
    const socket = new this.window.WebSocket(url);
    this.socket = socket;
    socket.onmessage = event => {
      if (this.socket !== socket) return;
      const message = JSON.parse(event.data);
      if (message.type === "error") {
        this.showError(message.error);
        return;
      }
      const state = message.state;
      if (state) {
        if (this.agentSessionId !== null && this.agentSessionId !== state.agent_session_id) {
          this.pendingReset = true;
        }
        this.agentSessionId = state.agent_session_id;
        this.status.textContent = state.running ? "Running" : "Live";
      }
      if (message.type === "hello" || [
        "model_called", "model_completed", "tool_started", "tool_completed",
        "turn_completed", "turn_failed", "turn_interrupted", "session_state",
        "compact_completed", "auto_compact_completed",
      ].includes(message.kind)) {
        this.schedule(message.kind === "compact_completed" || message.kind === "auto_compact_completed");
      }
    };
    socket.onclose = event => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.status.textContent = "Disconnected";
      if (event.code !== 1000 && event.code !== 1008) {
        this.retryTimer = setTimeout(() => {
          this.retryTimer = null;
          this.connect();
        }, 2000);
      }
    };
  }

  showError(message) {
    this.error.textContent = message;
    this.error.hidden = !message;
  }

  async load(older = false) {
    if (!this.active || this.document.hidden || (older && !this.nextCursor)) return;
    if (this.requestController) {
      if (!older) this.pendingRefresh = true;
      return;
    }
    if (this.pendingReset) older = false;
    const controller = new AbortController();
    this.requestController = controller;
    const reset = this.pendingReset;
    this.pendingReset = false;
    this.pendingRefresh = false;
    this.older.disabled = true;
    const page = before => {
      const params = new URLSearchParams({agent_id: this.agentId, limit: "100"});
      if (before !== null) params.set("before", before);
      return this.request(`api/sessions/${encodeURIComponent(this.sessionId)}/history?${params}`, {
        signal: controller.signal,
      });
    };
    try {
      const newest = await page(older ? this.nextCursor : null);
      let records = newest.items;
      const previous = Array.from(this.records.values());
      const lastIndex = previous.length ? previous[previous.length - 1].index : -1;
      let cursor = newest.next_cursor;
      // A burst can exceed one page; fill the gap before merging with loaded history.
      if (!older && !reset && this.source === newest.source && previous.length) {
        while (records.length && records[0].index > lastIndex && cursor !== null) {
          const earlier = await page(cursor);
          if (earlier.source !== newest.source) throw Object.assign(new Error("History changed"), {status: 409});
          records = earlier.items.concat(records);
          cursor = earlier.next_cursor;
        }
      }
      if (this.requestController !== controller) return;
      const overlap = records.filter(item => this.records.has(item.index));
      const changed = reset || this.source !== newest.source || (!older && previous.length && (
        !overlap.length || overlap.some(item =>
          JSON.stringify(item) !== JSON.stringify(this.records.get(item.index)))
      ));
      if (changed) {
        this.records.clear();
        records = newest.items;
      }
      for (const item of records) this.records.set(item.index, item);
      this.records = new Map(Array.from(this.records).sort((left, right) => left[0] - right[0]));
      if (older || changed || !previous.length) this.nextCursor = newest.next_cursor;
      this.source = newest.source;
      this.showError("");
      this.render();
    } catch (error) {
      if (this.requestController !== controller || error.name === "AbortError") return;
      if (error.status === 409) {
        this.pendingReset = true;
        this.pendingRefresh = true;
      } else {
        this.showError("Unable to load transcript: " + error.message);
        if ([401, 403, 404].includes(error.status)) this.disconnect();
      }
    } finally {
      if (this.requestController === controller) {
        this.requestController = null;
        this.lastLoad = Date.now();
        this.older.disabled = false;
        if (this.pendingRefresh || this.pendingReset) this.schedule();
      }
    }
  }

  renderBlock(block) {
    const tool = block.kind === "tool";
    const node = this.element("details", "transcript-record");
    node.dataset.key = block.key;
    node.dataset.kind = block.kind;
    if (block.toolName) node.dataset.tool = block.toolName;
    node.open = block.kind !== "reasoning" && block.kind !== "compaction";
    const heading = this.element("summary", "transcript-marker");
    const line = this.element("span", "transcript-marker-line");
    const label = this.element("span", "transcript-label");
    label.appendChild(this.element("span", "transcript-role", block.label));
    line.appendChild(label);
    heading.appendChild(line);
    if (tool) {
      heading.title = "Call: " + block.callId;
      const status = toolStatus(block);
      if (status.cellId) {
        const cell = this.element("span", "transcript-cell", status.cellId);
        cell.title = "Cell: " + status.cellId;
        let hue = 0;
        // Association colors stay blue/violet, separate from execution and status colors.
        for (const char of status.cellId) hue = (hue * 31 + char.charCodeAt(0)) % 80;
        cell.style.setProperty("--transcript-cell-hue", 200 + hue);
        label.appendChild(cell);
      }
      const phase = this.element("span", "transcript-phase", {
        pending: "Awaiting result", running: "Running", completed: "Completed",
        failed: "Failed", terminated: "Terminated",
      }[status.state]);
      phase.dataset.state = status.state;
      line.appendChild(phase);
      node.append(
        heading,
        block.input === undefined
          ? this.element("p", "transcript-note", "Load earlier records to see this call's input.")
          : this.element("pre", "transcript-code", block.input),
      );
      if (block.output !== undefined) {
        const output = this.element("details", "transcript-output");
        output.append(
          this.element("summary", "", "Output"),
          this.element("pre", "transcript-code", block.output),
        );
        node.appendChild(output);
      }
    } else {
      if (block.phase) line.appendChild(this.element("span", "transcript-phase", block.phase));
      const body = this.element("div", "transcript-body markdown");
      if (block.kind !== "raw" && this.renderMarkdown) this.renderMarkdown(body, block.body || "");
      else body.appendChild(this.element("pre", "transcript-code", block.body));
      node.append(heading, body);
    }
    node.signature = JSON.stringify(block);
    return node;
  }

  render() {
    const atBottom = this.scroll.scrollHeight - this.scroll.scrollTop - this.scroll.clientHeight < 40;
    const top = this.scroll.getBoundingClientRect().top;
    const anchor = Array.from(this.list.children).find(node => node.getBoundingClientRect().bottom > top);
    const anchorTop = anchor?.getBoundingClientRect().top;
    const existing = new Map(Array.from(this.list.children, node => [node.dataset.key, node]));
    const blocks = transcriptBlocks(Array.from(this.records.values()));
    const nodes = blocks.map(block => {
      const signature = JSON.stringify(block);
      const old = existing.get(block.key);
      if (old?.signature === signature) return old;
      const node = this.renderBlock(block);
      if (old) {
        node.open = old.open;
        const oldOutput = old.querySelector(".transcript-output");
        const output = node.querySelector(".transcript-output");
        if (oldOutput && output) output.open = oldOutput.open;
      }
      return node;
    });
    if (!this.list.children.length) this.list.textContent = "";
    const retained = new Set(nodes);
    for (const node of Array.from(this.list.children)) {
      if (!retained.has(node)) node.remove();
    }
    nodes.forEach((node, index) => {
      if (this.list.children[index] !== node) this.list.insertBefore(node, this.list.children[index] || null);
    });
    if (!blocks.length) this.list.textContent = "No messages yet.";
    this.older.hidden = this.nextCursor === null;
    if (atBottom) this.scroll.scrollTop = this.scroll.scrollHeight;
    else if (anchor) {
      const next = nodes.find(node => node.dataset.key === anchor.dataset.key);
      if (next) this.scroll.scrollTop += next.getBoundingClientRect().top - anchorTop;
    }
  }
}
