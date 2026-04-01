from __future__ import annotations

import asyncio
import json
import socket
import ssl
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

from .model import ResponsesModelClient, ResponsesProviderConfig
from .protocol import AssistantMessage, Prompt, UserMessage
from .utils.dotenv import DOTENV_FILENAME, load_codex_dotenv


@dataclass(slots=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


@dataclass(slots=True)
class DoctorReport:
    ok: bool
    config_path: str
    dotenv_path: str
    profile: str | None
    provider_name: str | None = None
    model: str | None = None
    base_url: str | None = None
    responses_url: str | None = None
    api_key_env: str | None = None
    api_key_loaded: bool = False
    checks: list[DoctorCheck] = field(default_factory=list)
    live_output_text: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_doctor_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="pycodex doctor",
        description="Diagnose pycodex config, auth, and provider connectivity.",
    )
    parser.add_argument(
        "--config",
        default=str(Path.home() / ".codex" / "config.toml"),
        help="Path to Codex config.toml.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional profile name from config.toml.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout used for network and live model checks.",
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip the live Responses API request and only run static/network checks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full doctor report as JSON.",
    )
    return parser


async def collect_doctor_report(
    config_path: str | Path,
    profile: str | None = None,
    timeout_seconds: float = 120.0,
    skip_live: bool = False,
) -> DoctorReport:
    config_file = Path(config_path).expanduser().resolve()
    dotenv_file = config_file.parent / DOTENV_FILENAME
    report = DoctorReport(
        ok=False,
        config_path=str(config_file),
        dotenv_path=str(dotenv_file),
        profile=profile,
    )

    checks = report.checks
    checks.append(
        DoctorCheck(
            "config",
            config_file.is_file(),
            str(config_file) if config_file.is_file() else f"missing: {config_file}",
        )
    )
    checks.append(
        DoctorCheck(
            "dotenv",
            True,
            (
                str(dotenv_file)
                if dotenv_file.is_file()
                else f"missing (optional): {dotenv_file}"
            ),
        )
    )
    if not config_file.is_file():
        return report

    load_codex_dotenv(config_file)

    try:
        provider_config = ResponsesProviderConfig.from_codex_config(config_file, profile)
    except Exception as exc:
        checks.append(DoctorCheck("config_parse", False, f"{type(exc).__name__}: {exc}"))
        return report

    report.provider_name = provider_config.provider_name
    report.model = provider_config.model
    report.base_url = provider_config.base_url
    client = ResponsesModelClient(provider_config)
    report.responses_url = client.responses_url()
    report.api_key_env = provider_config.api_key_env
    report.api_key_loaded = bool(provider_config.api_key_env and _loaded_api_key(provider_config))

    checks.append(
        DoctorCheck(
            "provider",
            True,
            (
                f"provider={provider_config.provider_name} "
                f"model={provider_config.model} "
                f"wire_api={provider_config.wire_api}"
            ),
        )
    )
    checks.append(
        DoctorCheck(
            "api_key",
            report.api_key_loaded,
            (
                f"{provider_config.api_key_env} loaded"
                if report.api_key_loaded
                else f"{provider_config.api_key_env} is missing"
            ),
        )
    )

    parsed_url = urllib.parse.urlsplit(client.responses_url())
    host = parsed_url.hostname or ""
    port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
    proxies = requests.utils.get_environ_proxies(client.responses_url())

    checks.append(
        DoctorCheck(
            "proxy",
            True,
            _proxy_detail(proxies),
        )
    )

    try:
        addresses = await asyncio.to_thread(
            socket.getaddrinfo,
            host,
            port,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        checks.append(DoctorCheck("dns", False, f"{host}:{port} -> {exc}"))
        return _finalize_report(report)

    resolved_addresses = sorted(
        {
            result[4][0]
            for result in addresses
            if len(result) >= 5 and isinstance(result[4], tuple) and result[4]
        }
    )
    checks.append(
        DoctorCheck(
            "dns",
            True,
            f"{host}:{port} -> {', '.join(resolved_addresses) or 'resolved'}",
        )
    )

    if proxies:
        checks.append(
            DoctorCheck(
                "transport",
                True,
                "skipped direct probe because requests will use environment proxy settings",
            )
        )
    else:
        tcp_ok, tcp_detail = await asyncio.to_thread(
            _probe_transport,
            parsed_url.scheme,
            host,
            port,
            timeout_seconds,
        )
        checks.append(DoctorCheck("transport", tcp_ok, tcp_detail))

    if skip_live:
        return _finalize_report(report)
    if not report.api_key_loaded:
        checks.append(DoctorCheck("live", False, "skipped: missing API key"))
        return _finalize_report(report)

    live_ok, live_detail, live_output_text = await _run_live_check(
        config_file,
        profile,
        timeout_seconds,
    )
    report.live_output_text = live_output_text
    checks.append(DoctorCheck("live", live_ok, live_detail))
    return _finalize_report(report)


async def run_doctor_cli(args) -> int:
    report = await collect_doctor_report(
        args.config,
        args.profile,
        args.timeout_seconds,
        args.skip_live,
    )

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_doctor_report(report))

    return 0 if report.ok else 1


def format_doctor_report(report: DoctorReport) -> str:
    lines = [
        f"config: {report.config_path}",
        f"dotenv: {report.dotenv_path}",
    ]
    if report.profile is not None:
        lines.append(f"profile: {report.profile}")
    if report.provider_name is not None:
        lines.append(f"provider: {report.provider_name}")
    if report.model is not None:
        lines.append(f"model: {report.model}")
    if report.responses_url is not None:
        lines.append(f"responses_url: {report.responses_url}")
    if report.api_key_env is not None:
        lines.append(f"api_key_env: {report.api_key_env}")
    if report.live_output_text is not None:
        lines.append(f"live_output: {report.live_output_text}")
    for check in report.checks:
        status = "ok" if check.ok else "fail"
        lines.append(f"{check.name}: {status} - {check.detail}")
    lines.append(f"overall: {'ok' if report.ok else 'fail'}")
    return "\n".join(lines)


def _loaded_api_key(provider_config: ResponsesProviderConfig) -> str:
    try:
        return provider_config.api_key()
    except Exception:
        return ""


def _probe_transport(
    scheme: str,
    host: str,
    port: int,
    timeout_seconds: float,
) -> tuple[bool, str]:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds) as sock:
            if scheme == "https":
                with ssl.create_default_context().wrap_socket(
                    sock,
                    server_hostname=host,
                ):
                    pass
    except OSError as exc:
        elapsed = time.perf_counter() - started
        return False, f"{scheme.upper()} {host}:{port} failed after {elapsed:.2f}s: {exc}"

    elapsed = time.perf_counter() - started
    label = "tls" if scheme == "https" else "tcp"
    return True, f"{label} {host}:{port} connected in {elapsed:.2f}s"


def _proxy_detail(proxies: dict[str, str]) -> str:
    if not proxies:
        return "not configured"
    return ", ".join(
        f"{key}={_redact_proxy_url(value)}" for key, value in sorted(proxies.items())
    )


def _redact_proxy_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    netloc = f"{host}{port}"
    return urllib.parse.urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


async def _run_live_check(
    config_path: Path,
    profile: str | None,
    timeout_seconds: float,
) -> tuple[bool, str, str | None]:
    client = ResponsesModelClient.from_codex_config(
        config_path,
        profile,
        timeout_seconds,
        originator="pycodex_doctor",
    )
    prompt = Prompt(
        input=[UserMessage(text="Reply with exactly OK.")],
        tools=[],
        parallel_tool_calls=True,
        base_instructions="Reply with exactly OK.",
    )

    started = time.perf_counter()
    try:
        response = await client.complete(prompt)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return False, f"failed after {elapsed:.2f}s: {type(exc).__name__}: {exc}", None

    elapsed = time.perf_counter() - started
    output_text = next(
        (
            item.text
            for item in response.items
            if isinstance(item, AssistantMessage)
        ),
        None,
    )
    return True, f"completed in {elapsed:.2f}s", output_text


def _finalize_report(report: DoctorReport) -> DoctorReport:
    report.ok = all(check.ok for check in report.checks)
    return report
