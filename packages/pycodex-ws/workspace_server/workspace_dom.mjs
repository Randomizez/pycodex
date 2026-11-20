export function relativeUrl(path) {
  return new URL(path, window.location.href).toString();
}

export function frontendUrl(builtin) {
  const url = new URL(window.location.href);
  if (builtin) url.searchParams.set("builtin", "1");
  else url.searchParams.delete("builtin");
  return url.toString();
}

export function encodeViewPath(path) {
  return String(path).split("/").map(encodeURIComponent).join("/");
}

export function element(tag, className = "", text = "") {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  if (tag === "button") node.type = "button";
  return node;
}

export function isNearBottom(log) {
  return log.scrollHeight - log.scrollTop - log.clientHeight < 80;
}

export function scrollToBottom(log) {
  log.scrollTop = log.scrollHeight;
}
