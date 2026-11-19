import {Composer} from "./content_views.mjs";
import {relativeUrl} from "./workspace_dom.mjs";

export async function mountCustomFrontend({
  getActiveSessionId, sessionById, selectSession, frontendEvents,
  composer, stateForSession, resizePrompt, contentTabs,
}) {
  const script = document.querySelector("script[data-pycodex-custom-script]");
  if (!script) return;
  const extension = await import(script.src);
  if (typeof extension.mount !== "function") return;
  return extension.mount(Object.freeze({
    apiVersion: 1,
    getActiveSessionId,
    selectSession: async sessionId => {
      if (!sessionById(sessionId)) throw new Error("Unknown session: " + sessionId);
      await selectSession(sessionId);
    },
    onSessionChanged: handler => {
      const listener = event => handler(event.detail);
      frontendEvents.addEventListener("sessionchange", listener);
      return () => frontendEvents.removeEventListener("sessionchange", listener);
    },
    getDraft: () => composer.read(),
    replaceDraftText: (start, end, text) => {
      if (typeof text !== "string" || !Number.isInteger(start) || !Number.isInteger(end) ||
          start < 0 || end < start || end > Composer.size(composer.parts())) {
        throw new TypeError("Invalid draft text replacement");
      }
      composer.replace(start, end, [{type: "text", text}]);
    },
    setDraft: draft => {
      if (!draft || !Array.isArray(draft.parts)) throw new TypeError("draft.parts must be an array");
      composer.set(draft);
      stateForSession(getActiveSessionId()).draft = composer.read();
      resizePrompt();
    },
    openView: (href, srcPath) => contentTabs.open(href, srcPath, getActiveSessionId()),
    request: async (path, options = {}) => {
      const response = await fetch(relativeUrl(path), options);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
      return payload;
    },
  }));
}
