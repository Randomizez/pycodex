import {
  DOMPurify as bundledDOMPurify,
  isWorkspaceLink,
  markdownit as bundledMarkdownit,
} from "./content_views.mjs";
import {element, isNearBottom, scrollToBottom} from "./workspace_dom.mjs";

const mathDelimiters = [
  {left: "$$", right: "$$", display: true},
  {left: "\\[", right: "\\]", display: true},
  {left: "\\(", right: "\\)", display: false},
  {left: "$", right: "$", display: false},
];
const mathTokenPrefix = "\uE000math";
const mathTokenSuffix = "\uE001";
const KATEX_CDN_BASE = "https://cdn.jsdelivr.net/npm/katex@0.16.22/dist";
let katexPromise = null;
let katexRetry = 0;

function loadKatex() {
  if (typeof window === "undefined") return Promise.resolve(null);
  if (window.katex?.render) return Promise.resolve(window.katex);
  if (!katexPromise) {
    const cssHref = `${KATEX_CDN_BASE}/katex.min.css`;
    if (!document.querySelector(`link[href="${cssHref}"]`)) {
      document.head.appendChild(Object.assign(document.createElement("link"), {
        rel: "stylesheet", href: cssHref,
      }));
    }
    const url = `${KATEX_CDN_BASE}/katex.mjs${katexRetry ? `?retry=${katexRetry}` : ""}`;
    katexPromise = import(url)
      .then(mod => {
        window.katex = mod.default || mod;
        return window.katex;
      })
      .catch(error => {
        katexPromise = null;
        katexRetry += 1;
        throw error;
      });
  }
  return katexPromise;
}

function renderMathNode(katex, node, {delimiter, body}) {
  if (katex?.render) {
    katex.render(body, node, {displayMode: delimiter.display, throwOnError: false, trust: false});
  } else if (window.renderMathInElement) {
    node.textContent = delimiter.left + body + delimiter.right;
    window.renderMathInElement(node, {delimiters: [{...delimiter}], throwOnError: false, trust: false});
  }
}

const renderMermaidSvg = (() => {
  let rendererPromise = null;
  let diagramId = 0;
  return async (source) => {
    if (!rendererPromise) {
      rendererPromise = import(
        "https://cdn.jsdelivr.net/npm/mermaid@11.12.0/dist/mermaid.esm.min.mjs"
      ).then(({default: renderer}) => {
        renderer.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          suppressErrorRendering: true,
          htmlLabels: false,
          flowchart: {htmlLabels: false},
          theme: "neutral",
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        });
        return renderer;
      }).catch(error => {
        rendererPromise = null;
        throw error;
      });
    }
    const renderer = await rendererPromise;
    const measure = document.createElement("div");
    measure.style.cssText =
      "position:fixed;left:0;top:0;visibility:hidden;pointer-events:none";
    document.body.appendChild(measure);
    try {
      diagramId += 1;
      return await renderer.render("workspace-diagram-" + diagramId, source, measure);
    } finally {
      measure.remove();
    }
  };
})();

function isEscapedAt(source, index) {
  let slashCount = 0;
  for (let cursor = index - 1; cursor >= 0 && source[cursor] === "\\"; cursor -= 1) {
    slashCount += 1;
  }
  return slashCount % 2 === 1;
}

function countBackticks(source, index) {
  let count = 0;
  while (source[index + count] === "`") count += 1;
  return count;
}

function findMathEnd(source, start, delimiter) {
  let searchFrom = start + delimiter.left.length;
  while (searchFrom < source.length) {
    const end = source.indexOf(delimiter.right, searchFrom);
    if (end < 0) return -1;
    if (!isEscapedAt(source, end)) {
      if (
        delimiter.left === "$" &&
        (source[end - 1] === "$" || source[end + 1] === "$")
      ) {
        searchFrom = end + delimiter.right.length;
        continue;
      }
      return end;
    }
    searchFrom = end + delimiter.right.length;
  }
  return -1;
}

export function protectMath(source) {
  const expressions = [];
  let protectedSource = "";
  let index = 0;
  while (index < source.length) {
    if (source[index] === "`") {
      const tickCount = countBackticks(source, index);
      const fence = "`".repeat(tickCount);
      const close = source.indexOf(fence, index + tickCount);
      if (close >= 0) {
        const end = close + tickCount;
        protectedSource += source.slice(index, end);
        index = end;
        continue;
      }
    }

    let matched = null;
    if (!isEscapedAt(source, index)) {
      for (const delimiter of mathDelimiters) {
        if (
          !source.startsWith(delimiter.left, index) ||
          (delimiter.left === "$" && source[index + 1] === "$")
        ) {
          continue;
        }
        const end = findMathEnd(source, index, delimiter);
        if (end > index + delimiter.left.length) {
          matched = {
            delimiter,
            start: index,
            end,
          };
          break;
        }
      }
    }

    if (matched) {
      const expressionIndex = expressions.length;
      expressions.push({
        body: source.slice(
          matched.start + matched.delimiter.left.length,
          matched.end,
        ),
        delimiter: matched.delimiter,
      });
      protectedSource += mathTokenPrefix + expressionIndex + mathTokenSuffix;
      index = matched.end + matched.delimiter.right.length;
      continue;
    }

    protectedSource += source[index];
    index += 1;
  }
  return {
    source: protectedSource,
    expressions,
  };
}

function renderProtectedMath(root, expressions, log) {
  if (!expressions.length) return;
  const expressionByToken = new Map(expressions.map((expression, index) => [
    mathTokenPrefix + index + mathTokenSuffix, expression,
  ]));
  const pending = [];
  const textWalker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  let textNode = textWalker.nextNode();
  while (textNode) {
    textNodes.push(textNode);
    textNode = textWalker.nextNode();
  }
  for (const textNode of textNodes) {
    const value = textNode.nodeValue || "";
    let cursor = 0;
    let tokenIndex = value.indexOf(mathTokenPrefix, cursor);
    if (tokenIndex < 0) continue;
    const fragment = document.createDocumentFragment();
    while (tokenIndex >= 0) {
      const tokenEnd = value.indexOf(mathTokenSuffix, tokenIndex);
      if (tokenEnd < 0) break;
      const token = value.slice(tokenIndex, tokenEnd + mathTokenSuffix.length);
      const expression = expressionByToken.get(token);
      if (!expression) break;
      if (tokenIndex > cursor) {
        fragment.appendChild(
          document.createTextNode(value.slice(cursor, tokenIndex)),
        );
      }
      const mathNode = document.createElement("span");
      if (window.katex?.render || window.renderMathInElement) {
        renderMathNode(window.katex, mathNode, expression);
      } else {
        mathNode.textContent = expression.body;
        pending.push({node: mathNode, expression});
      }
      fragment.appendChild(mathNode);
      cursor = tokenEnd + mathTokenSuffix.length;
      tokenIndex = value.indexOf(mathTokenPrefix, cursor);
    }
    if (cursor > 0) {
      if (cursor < value.length) {
        fragment.appendChild(document.createTextNode(value.slice(cursor)));
      }
      if (textNode.parentNode) {
        textNode.parentNode.replaceChild(fragment, textNode);
      }
    }
  }
  restoreUnrenderedMathTokens(root, expressionByToken);
  if (pending.length) {
    loadKatex().then(katex => {
      if (!katex?.render) return;
      const stickToBottom = log && root.isConnected && isNearBottom(log);
      let rendered = false;
      for (const {node, expression} of pending) {
        if (!node.isConnected) continue;
        renderMathNode(katex, node, expression);
        rendered = true;
      }
      if (rendered && stickToBottom) scrollToBottom(log);
    }).catch(() => {});
  }
}

function restoreUnrenderedMathTokens(root, expressionByToken) {
  const textWalker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let textNode = textWalker.nextNode();
  while (textNode) {
    let value = textNode.nodeValue || "";
    expressionByToken.forEach((expression, token) => {
      const {delimiter, body} = expression;
      value = value.split(token).join(delimiter.left + body + delimiter.right);
    });
    textNode.nodeValue = value;
    textNode = textWalker.nextNode();
  }
}

function showMermaidDiagram(svg) {
  const copy = svg.cloneNode(true);
  copy.style.removeProperty("width");
  copy.style.removeProperty("max-width");
  copy.setAttribute("width", svg.viewBox.baseVal.width);
  copy.setAttribute("height", svg.viewBox.baseVal.height);
  const imageUrl = URL.createObjectURL(new Blob(
    [new XMLSerializer().serializeToString(copy)],
    {type: "image/svg+xml"},
  ));
  const dialog = element("dialog", "mermaid-viewer");
  dialog.setAttribute("aria-label", "Expanded diagram");
  const header = element("div", "mermaid-viewer-header");
  const zoom = element("button", "", "Actual size");
  const close = element("button", "", "Close");
  const content = element("div", "mermaid-viewer-content");
  const image = document.createElement("img");
  image.src = imageUrl;
  image.alt = "Diagram";
  content.appendChild(image);
  header.append(element("strong", "", "Diagram"), zoom, close);
  dialog.append(header, content);
  zoom.addEventListener("click", () => {
    const actualSize = content.classList.toggle("actual-size");
    zoom.textContent = actualSize ? "Fit width" : "Actual size";
  });
  close.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => {
    if (event.target === dialog) dialog.close();
  });
  dialog.addEventListener("close", () => {
    URL.revokeObjectURL(imageUrl);
    dialog.remove();
  }, {once: true});
  document.body.appendChild(dialog);
  dialog.showModal();
}

export function createMarkdownRenderer(log) {
  const markdownFactory = (typeof window !== "undefined" && window.markdownit) || bundledMarkdownit || null;
  const getPurifier = () => (typeof window !== "undefined" && window.DOMPurify) || bundledDOMPurify || null;
  const markdownRenderer = markdownFactory
    ? markdownFactory({html: false, linkify: true, breaks: false})
    : null;
  if (markdownRenderer) {
    // Bare filenames such as TODO.md must not become http://TODO.md.
    markdownRenderer.linkify.set({fuzzyLink: false});
    function workspaceFilePath(href) {
      if (!/^file:\/\/\/(?!\/|%2f|%5c)/i.test(href)) return null;
      const path = href.slice("file://".length)
        .replace(/^\/([a-z]:\/)/i, "$1")
        .replace(/#(L[1-9]\d*)-L[1-9]\d*$/i, "#$1");
      return isWorkspaceLink(path) ? path : null;
    }
    const validateLink = markdownRenderer.validateLink;
    markdownRenderer.validateLink = href => validateLink(href) || workspaceFilePath(href) !== null;
    const renderImage = markdownRenderer.renderer.rules.image;
    markdownRenderer.renderer.rules.image = (tokens, index, options, env, renderer) => {
      const token = tokens[index];
      const src = token.attrGet("src");
      if (/^(?:\/(?![/\\])|[a-z]:(?:[\\/]|%5c))/i.test(src) && /\.(?:png|jpe?g|gif|webp)$/i.test(src)) {
        // markdown-it already percent-encoded the path. Escape query delimiters without encoding it twice.
        token.attrSet("src", "api/images?path=" + src.replace(/[&+#]/g, encodeURIComponent));
      } else if (env.imageBase && src && !/^(?:[a-z][a-z\d+.-]*:|[/\\?#])/i.test(src)) {
        token.attrSet("src", env.imageBase + src);
      }
      return renderImage(tokens, index, options, env, renderer);
    };
    markdownRenderer.renderer.rules.link_open = (tokens, index, options, env, renderer) => {
      const token = tokens[index];
      const href = token.attrGet("href");
      const destination = workspaceFilePath(href) || href;
      if (isWorkspaceLink(destination)) {
        token.attrSet("data-workspace-link", destination);
        // Preserve file/URI destinations without widening DOMPurify's protocol allowlist.
        token.attrSet("href", "#");
      }
      return renderer.renderToken(tokens, index, options);
    };
  }

  function renderMarkdownInto(root, source, imageBase = null) {
    const purifier = getPurifier();
    if (!markdownRenderer || !purifier) {
      root.dataset.plainText = "true";
      root.textContent = source || "";
      return;
    }
    delete root.dataset.plainText;
    const protectedMath = protectMath(String(source || ""));
    const dirtyHtml = markdownRenderer.render(protectedMath.source, {imageBase});
    root.innerHTML = purifier.sanitize(dirtyHtml, {
      USE_PROFILES: {html: true},
      ADD_ATTR: ["target", "rel"],
    });
    renderProtectedMath(root, protectedMath.expressions, log);
    root.querySelectorAll("a").forEach((anchor) => {
      const link = anchor.getAttribute("data-workspace-link");
      if (isWorkspaceLink(link)) {
        anchor.setAttribute("href", link);
      } else if (!(anchor.getAttribute("href") || "").startsWith("#")) {
        anchor.target = "_blank";
        anchor.rel = "noopener noreferrer";
      }
    });
    renderMermaidInto(root);
  }

  async function renderMermaidInto(root) {
    for (const code of root.querySelectorAll("pre > code.language-mermaid")) {
      if (!root.contains(code)) return;
      const source = code.textContent;
      const pre = code.parentElement;
      const figure = element("figure", "mermaid-figure");
      const diagram = element("div", "mermaid-diagram", "Rendering diagram…");
      const details = element("details", "mermaid-source");
      details.open = true;
      details.appendChild(element("summary", "", "Diagram source"));
      pre.replaceWith(figure);
      details.appendChild(pre);
      const controls = element("div", "mermaid-controls");
      controls.appendChild(details);
      figure.append(diagram, controls);
      try {
        const result = await renderMermaidSvg(source);
        // Cached conversations may finish rendering while detached.
        if (!root.contains(figure)) return;
        const stickToBottom = root.isConnected && isNearBottom(log);
        const purifier = getPurifier();
        if (!purifier) throw new Error("DOMPurify is unavailable");
        diagram.innerHTML = purifier.sanitize(result.svg, {
          USE_PROFILES: {svg: true, svgFilters: true},
        });
        const svg = diagram.querySelector("svg");
        svg.style.width = svg.viewBox.baseVal.width + "px";
        svg.style.maxWidth = "100%";
        const expand = element("button", "", "Expand");
        expand.addEventListener("click", () => showMermaidDiagram(svg));
        controls.appendChild(expand);
        details.open = false;
        if (stickToBottom) scrollToBottom(log);
      } catch (error) {
        if (!root.contains(figure)) return;
        diagram.textContent = "Diagram could not be rendered. Source is shown below.";
        details.open = true;
        console.warn("Mermaid diagram rendering failed", error);
      }
    }
  }
  return {renderMarkdownInto, renderMermaidInto};
}

export function createViewLinkHandlers({getActiveSessionId, openView, showToast}) {
  function followViewLink(event, href) {
    if (event.defaultPrevented || event.button !== 0) return;
    event.preventDefault();
    const source = event.target.closest(".snapshot-markdown[data-src-path]");
    const srcPath = source ? source.getAttribute("data-src-path") : undefined;
    const sessionId = event.target.closest(".transcript")?.dataset.sessionId || getActiveSessionId();
    openView(href, srcPath, sessionId).catch(error => showToast("Unable to open view: " + error.message));
  }

  function followMarkdownLink(event) {
    const anchor = event.target.closest("a[data-workspace-link]");
    if (anchor) followViewLink(event, anchor.getAttribute("data-workspace-link"));
  }

  function followCanvasLink(event) {
    const anchor = event.target.closest("a[href]");
    if (anchor && isWorkspaceLink(anchor.getAttribute("href"))) {
      followViewLink(event, anchor.getAttribute("href"));
    }
  }
  return {followMarkdownLink, followCanvasLink};
}
