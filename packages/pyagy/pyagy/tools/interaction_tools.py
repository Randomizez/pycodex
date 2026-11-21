"""AGY interaction & placeholder tools: search_web, ask_question, run_workflow, swarm."""

import asyncio
import gzip
import json
import ssl
import typing
import urllib.error
import urllib.request

from pycodex.protocol import JSONDict, JSONValue
from pycodex.tools.base_tool import ToolContext

from ..models import AGY_USER_AGENT
from . import (
    AgyBaseTool,
    AgyToolError,
    AgyToolRegistry,
    SearchHandler,
    format_sync_tool_header,
)

if typing.TYPE_CHECKING:
    from responses_server.config import CompatServerConfig


def _execute_gemini_web_search_sync(
    compat_config: "CompatServerConfig",
    api_model: "str",
    query: "str",
    domain: "typing.Optional[str]",
) -> "str":
    search_prompt = (
        f"Search the web and provide a concise, well-cited answer for: {query}"
        if not domain
        else f"Search the web (prioritizing site:{domain}) and provide a concise, well-cited answer for: {query}"
    )
    url = compat_config.outcomming_gemini_url(api_model)
    headers: "typing.Dict[str, str]" = {
        "Content-Type": "application/json",
        "User-agent": AGY_USER_AGENT,
        "x-goog-api-client": AGY_USER_AGENT,
        "Accept-Encoding": "gzip",
    }
    for k, v in compat_config.outcomming_http_headers.items():
        if k.lower() in ("user-agent", "x-goog-api-client"):
            continue
        headers[k] = v
    api_key = compat_config.outcomming_api_key()
    if api_key is not None:
        headers["x-goog-api-key"] = api_key

    def _call_once(use_google_search: "bool") -> "str":
        payload: "typing.Dict[str, object]" = {
            "contents": [{"role": "user", "parts": [{"text": search_prompt}]}],
            "generationConfig": {"maxOutputTokens": 4096},
        }
        if use_google_search:
            payload["tools"] = [{"googleSearch": {}}]
        body_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(
            req,
            context=ssl.create_default_context(),
            timeout=compat_config.timeout_seconds,
        ) as resp:
            raw = resp.read()
            if (resp.headers.get("Content-Encoding") or "").strip().lower() == "gzip":
                raw = gzip.decompress(raw)
            text_chunks: "typing.List[str]" = []
            sources: "typing.List[str]" = []
            seen_uris: "typing.Set[str]" = set()
            for line in raw.decode("utf-8", errors="replace").splitlines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                chunk = json.loads(data_str)
                for cand in chunk.get("candidates") or []:
                    for part in (cand.get("content") or {}).get("parts") or []:
                        if not part.get("thought") and isinstance(part.get("text"), str):
                            text_chunks.append(part["text"])
                    grounding = cand.get("groundingMetadata") or {}
                    for gchunk in grounding.get("groundingChunks") or []:
                        web = gchunk.get("web") if isinstance(gchunk, dict) else None
                        if isinstance(web, dict):
                            uri = str(web.get("uri") or "").strip()
                            title = str(web.get("title") or uri).strip()
                            if uri and uri not in seen_uris:
                                seen_uris.add(uri)
                                sources.append(f"- [{title}]({uri})")
            summary = "".join(text_chunks).strip()
            if sources:
                summary = f"{summary}\n\nSources:\n" + "\n".join(sources)
            return summary or "No search results returned."

    try:
        return _call_once(True)
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 501):
            return _call_once(False)
        err_raw = exc.read()
        if (exc.headers.get("Content-Encoding") or "").strip().lower() == "gzip":
            err_raw = gzip.decompress(err_raw)
        raise AgyToolError(
            f"search_web HTTP {exc.code}: {err_raw.decode('utf-8', errors='replace')[:300]}"
        ) from exc
    except Exception as exc:
        raise AgyToolError(f"search_web failed: {exc}") from exc


class SearchWebTool(AgyBaseTool):
    """Implements AGY `search_web`."""

    def __init__(
        self,
        declaration: "JSONDict",
        registry: "AgyToolRegistry",
        compat_config: "typing.Optional[CompatServerConfig]" = None,
        api_model_getter: "typing.Optional[typing.Callable[[], str]]" = None,
        search_handler: "typing.Optional[SearchHandler]" = None,
    ) -> "None":
        super().__init__(declaration, registry)
        self._compat_config = compat_config
        self._api_model_getter = api_model_getter
        self._search_handler = search_handler

    async def run(self, context: "ToolContext", args: "JSONValue") -> "str":
        created_at = self.now_local_iso()
        params = self.require_dict_args(args)
        query = params.get("query")
        if not isinstance(query, str) or not query.strip():
            raise AgyToolError("query must be a non-empty string")
        domain = params.get("domain")
        if domain is not None and not isinstance(domain, str):
            raise AgyToolError("domain must be a string")
        clean_domain = domain.strip() if isinstance(domain, str) and domain.strip() else None

        if self._search_handler is not None:
            summary = await self._search_handler(query.strip(), clean_domain)
        elif self._compat_config is not None:
            api_model = (
                self._api_model_getter()
                if self._api_model_getter is not None
                else "gemini-3.8-flash"
            )
            loop = asyncio.get_running_loop()
            summary = await loop.run_in_executor(
                None,
                _execute_gemini_web_search_sync,
                self._compat_config,
                api_model,
                query.strip(),
                clean_domain,
            )
        else:
            raise AgyToolError("search_web backend is not configured")

        completed_at = self.now_local_iso()
        return f"{format_sync_tool_header(created_at, completed_at)}{summary}"


class AskQuestionPlaceholderTool(AgyBaseTool):
    """Schema-preserving placeholder for disabled `ask_question`."""

    async def run(self, context: "ToolContext", args: "JSONValue") -> "str":
        raise AgyToolError("ask_question is disabled")


class RunWorkflowPlaceholderTool(AgyBaseTool):
    """Schema-preserving placeholder for `run_workflow` in CLI mode."""

    async def run(self, context: "ToolContext", args: "JSONValue") -> "str":
        raise AgyToolError("tool is not enabled in pyagy CLI mode")


class SwarmPlaceholderTool(AgyBaseTool):
    """Schema-preserving placeholder for `swarm` in CLI mode."""

    async def run(self, context: "ToolContext", args: "JSONValue") -> "str":
        raise AgyToolError("tool is not enabled in pyagy CLI mode")
