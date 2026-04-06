
import threading

import pytest

from pycodex.http_compat import ThreadingHTTPServer
from pycodex.cli import main
from pycodex.doctor import collect_doctor_report
from tests.fake_responses_server import CaptureStore, build_handler


def _write_config(config_path, base_url: 'str', env_key: 'str' = "DOCTOR_KEY") -> 'None':
    config_path.write_text(
        "\n".join(
            [
                'model = "gpt-5.4"',
                'model_provider = "neo"',
                'model_reasoning_summary = "auto"',
                'model_reasoning_effort = "medium"',
                'model_verbosity = "medium"',
                '',
                '[model_providers.neo]',
                f'base_url = "{base_url}"',
                f'env_key = "{env_key}"',
                'wire_api = "responses"',
            ]
        )
    )


@pytest.mark.asyncio
async def test_collect_doctor_report_succeeds_with_live_check(
    tmp_path,
    monkeypatch,
) -> 'None':
    capture_store = CaptureStore(tmp_path / "capture")
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        build_handler(capture_store, "gpt-5.4", "OK"),
    )
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    config_path = tmp_path / "config.toml"
    _write_config(config_path, f"http://127.0.0.1:{httpd.server_port}/v1")
    monkeypatch.setenv("DOCTOR_KEY", "dummy-key")

    try:
        report = await collect_doctor_report(
            config_path,
            timeout_seconds=5.0,
            skip_live=False,
        )
    finally:
        httpd.shutdown()
        server_thread.join(timeout=5)
        httpd.server_close()

    assert report.ok is True
    assert report.provider_name == "neo"
    assert report.api_key_env == "DOCTOR_KEY"
    assert report.api_key_loaded is True
    assert report.live_output_text == "OK"
    assert {check.name for check in report.checks} == {
        "config",
        "dotenv",
        "provider",
        "api_key",
        "proxy",
        "dns",
        "transport",
        "live",
    }
    assert all(check.ok for check in report.checks)


@pytest.mark.asyncio
async def test_collect_doctor_report_fails_on_transport_error(
    tmp_path,
    monkeypatch,
) -> 'None':
    config_path = tmp_path / "config.toml"
    _write_config(config_path, "http://127.0.0.1:9/v1")
    monkeypatch.setenv("DOCTOR_KEY", "dummy-key")

    report = await collect_doctor_report(
        config_path,
        timeout_seconds=0.5,
        skip_live=True,
    )

    assert report.ok is False
    transport_checks = [check for check in report.checks if check.name == "transport"]
    assert len(transport_checks) == 1
    assert transport_checks[0].ok is False


@pytest.mark.asyncio
async def test_collect_doctor_report_skips_direct_transport_when_proxy_env_present(
    tmp_path,
    monkeypatch,
) -> 'None':
    config_path = tmp_path / "config.toml"
    _write_config(config_path, "https://example.com/v1")
    monkeypatch.setenv("DOCTOR_KEY", "dummy-key")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:3128")

    report = await collect_doctor_report(
        config_path,
        timeout_seconds=0.5,
        skip_live=True,
    )

    proxy_checks = [check for check in report.checks if check.name == "proxy"]
    transport_checks = [check for check in report.checks if check.name == "transport"]
    assert len(proxy_checks) == 1
    assert proxy_checks[0].ok is True
    assert "https=http://127.0.0.1:3128" in proxy_checks[0].detail
    assert len(transport_checks) == 1
    assert transport_checks[0].ok is True
    assert "skipped direct probe" in transport_checks[0].detail


@pytest.mark.asyncio
async def test_collect_doctor_report_redacts_proxy_credentials(
    tmp_path,
    monkeypatch,
) -> 'None':
    config_path = tmp_path / "config.toml"
    _write_config(config_path, "https://example.com/v1")
    monkeypatch.setenv("DOCTOR_KEY", "dummy-key")
    monkeypatch.setenv("https_proxy", "http://user:secret@127.0.0.1:3128")

    report = await collect_doctor_report(
        config_path,
        timeout_seconds=0.5,
        skip_live=True,
    )

    proxy_check = next(check for check in report.checks if check.name == "proxy")
    assert "user" not in proxy_check.detail
    assert "secret" not in proxy_check.detail
    assert "https=http://127.0.0.1:3128" in proxy_check.detail


def test_main_dispatches_doctor_subcommand(monkeypatch: 'pytest.MonkeyPatch') -> 'None':
    async def fake_run_doctor_cli(args) -> 'int':
        assert args.skip_live is True
        return 7

    monkeypatch.setattr("pycodex.doctor.run_doctor_cli", fake_run_doctor_cli)

    assert main(["doctor", "--skip-live"]) == 7
