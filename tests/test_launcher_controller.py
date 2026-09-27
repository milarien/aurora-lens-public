from __future__ import annotations

import socket
from pathlib import Path

import httpx
import pytest

from aurora_lens.cli import main as cli_main
from aurora_lens.launcher import controller
from aurora_lens.launcher.paths import resolve_config_path, resolve_runtime_paths
from aurora_lens.launcher.state import ProxyStartResult
from aurora_lens.launcher.lifecycle import RuntimeStatusCode
from aurora_lens.proxy.config import ProxyConfig


def test_runtime_root_prefers_localappdata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    paths = resolve_runtime_paths()
    assert paths.runtime_root == (tmp_path / "Aurora-Lens").resolve()


def test_resolve_config_path_prefers_cli_argument(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    paths = resolve_runtime_paths()
    chosen = resolve_config_path("relative-config.yaml", paths)
    assert chosen == (Path.cwd() / "relative-config.yaml").resolve()
    assert chosen != paths.default_config_path


def test_start_fails_when_config_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    rc = controller.main(["start"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Resolved config path:" in out
    assert "Config file not found:" in out


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def test_spacy_setup_when_already_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(controller, "_spacy_installed", lambda: True)
    monkeypatch.setattr(controller, "_spacy_model_available", lambda model: True)  # noqa: ARG005
    monkeypatch.setattr(controller, "_run_setup_command", lambda cmd: (calls.append(cmd) or (True, "")))
    ok, detail = controller._ensure_spacy_dependencies("en_core_web_sm")
    assert ok is True
    assert detail == ""
    assert calls == []


def test_spacy_setup_installs_package_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    spacy_checks = iter([False, True, True])
    model_checks = iter([True, True])
    calls: list[list[str]] = []
    monkeypatch.setattr(controller, "_spacy_installed", lambda: next(spacy_checks))
    monkeypatch.setattr(controller, "_spacy_model_available", lambda model: next(model_checks))  # noqa: ARG005
    monkeypatch.setattr(controller, "_run_setup_command", lambda cmd: (calls.append(cmd) or (True, "")))
    ok, detail = controller._ensure_spacy_dependencies("en_core_web_sm")
    assert ok is True
    assert detail == ""
    assert calls == [[controller.sys.executable, "-m", "pip", "install", "aurora-lens[spacy]"]]


def test_spacy_setup_installs_model_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    model_checks = iter([False, True, True])
    calls: list[list[str]] = []
    monkeypatch.setattr(controller, "_spacy_installed", lambda: True)
    monkeypatch.setattr(controller, "_spacy_model_available", lambda model: next(model_checks))  # noqa: ARG005
    monkeypatch.setattr(controller, "_run_setup_command", lambda cmd: (calls.append(cmd) or (True, "")))
    ok, detail = controller._ensure_spacy_dependencies("en_core_web_sm")
    assert ok is True
    assert detail == ""
    assert calls == [[controller.sys.executable, "-m", "spacy", "download", "en_core_web_sm"]]


def test_start_aborts_cleanly_when_spacy_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert cli_main(["init-config"]) == 0

    class FakeLauncherState:
        def __init__(self, paths) -> None:  # noqa: ANN001
            self.paths = paths
            self.proxy_log_path = paths.logs_dir / "proxy.log"
            self.proxy_pid_path = paths.state_dir / "aurora-lens.proxy.pid"

        def log(self, message: str) -> None:  # noqa: ARG002
            return None

        def start_proxy(self, config_path: Path, runtime_env: dict[str, str]) -> ProxyStartResult:  # noqa: ARG002
            raise AssertionError("start_proxy must not run when setup fails")

    monkeypatch.setattr(controller, "LauncherState", FakeLauncherState)
    monkeypatch.setattr(
        controller,
        "_ensure_spacy_dependencies",
        lambda model: (False, f"Failed setup for {model}"),  # noqa: ARG005
    )
    rc = cli_main(["start"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Failed setup for en_core_web_sm" in out
    assert "Start aborted. Proxy remains stopped." in out


def test_init_config_start_status_and_governed_request_without_llm_extractor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert cli_main(["init-config"]) == 0
    paths = resolve_runtime_paths()
    config_path = paths.default_config_path
    port = _free_tcp_port()
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("  port: 8081\n", f"  port: {port}\n"),
        encoding="utf-8",
    )

    cfg = ProxyConfig.from_yaml(config_path)
    assert cfg.upstream.provider == "mock"
    assert cfg.extraction.backend == "spacy"

    try:
        assert cli_main(["start"]) == 0
        start_out = capsys.readouterr().out
        assert "Start result: RUNNING_HEALTHY" in start_out

        assert cli_main(["status"]) == 0
        status_out = capsys.readouterr().out
        assert "Status: Running and healthy (RUNNING_HEALTHY)" in status_out

        health = httpx.get(f"http://127.0.0.1:{port}/health", timeout=10.0)
        assert health.status_code == 200
        health_body = health.json()
        assert health_body.get("extraction_backend") == "spacy"

        response = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={
                "model": "mock",
                "messages": [{"role": "user", "content": "What is 2 + 2?"}],
                "temperature": 0,
            },
            timeout=20.0,
        )
        assert response.status_code == 200

        audit_path = Path(cfg.governance.audit_log or "")
        assert audit_path.exists()
        audit_text = audit_path.read_text(encoding="utf-8")
        assert audit_text.strip()
        assert '"extractor_backend":"llm"' not in audit_text
    finally:
        cli_main(["stop"])
