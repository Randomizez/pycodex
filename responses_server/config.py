from __future__ import annotations

import os
from dataclasses import dataclass
import urllib.parse


@dataclass(frozen=True, slots=True)
class CompatServerConfig:
    host: str = "127.0.0.1"
    port: int = 0
    outcomming_base_url: str = "http://127.0.0.1:8000/v1"
    outcomming_api_key_env: str | None = None
    timeout_seconds: float = 120.0

    def outcomming_api_key(self) -> str | None:
        if self.outcomming_api_key_env is None:
            return None
        value = os.environ.get(self.outcomming_api_key_env, "").strip()
        return value or None

    def outcomming_chat_completions_url(self) -> str:
        base = self.outcomming_base_url.rstrip("/")
        return f"{base}/chat/completions"

    def outcomming_models_url(self) -> str:
        base = self.outcomming_base_url.rstrip("/")
        return f"{base}/models"

    def with_ephemeral_port(self) -> CompatServerConfig:
        return CompatServerConfig(
            host=self.host,
            port=0,
            outcomming_base_url=self.outcomming_base_url,
            outcomming_api_key_env=self.outcomming_api_key_env,
            timeout_seconds=self.timeout_seconds,
        )

    @classmethod
    def from_base_url(
        cls,
        outcomming_base_url: str,
        api_key_env: str | None = None,
    ) -> CompatServerConfig:
        parsed = urllib.parse.urlparse(outcomming_base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"invalid outcomming base url: {outcomming_base_url}")
        normalized_path = parsed.path.rstrip("/")
        if normalized_path in {"", "/"}:
            parsed = parsed._replace(path="/v1")
            outcomming_base_url = urllib.parse.urlunparse(parsed)
        else:
            outcomming_base_url = urllib.parse.urlunparse(
                parsed._replace(path=normalized_path)
            )
        return cls(
            outcomming_base_url=outcomming_base_url,
            outcomming_api_key_env=api_key_env,
        )
