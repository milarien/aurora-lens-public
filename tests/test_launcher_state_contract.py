from __future__ import annotations

from pathlib import Path
import json

import pytest

from aurora_lens.launcher.lifecycle import LifecycleResultCode, RuntimeStatusCode
from aurora_lens.launcher.paths import RuntimePaths
from aurora_lens.launcher import state as launcher_state
from aurora_lens.launcher.state import LauncherState


def _paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(
        runtime_root=tmp_path,
        config_dir=tmp_path / "config",
        state_dir=tmp_path / "state",
        logs_dir=tmp_path / "logs",
        support_dir=tmp_path / "support",
        temp_dir=tmp_path / "tmp",
    )


def _write_config(path: Path, *, port: int = 8081) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "upstream:\n"
            "  provider: openai\n"
            "  model: gpt-4o-mini\n"
            "  api_key_env: OPENAI_API_KEY\n"
            "listen:\n"
            "  host: 127.0.0.1\n"
            f"  port: {port}\n"
            "governance:\n"
            "  default_policy: strict\n"
            "  mode: public\n"
            "  audit_log: ./audit.jsonl\n"
        ),
        encoding="utf-8",
    )


def test_status_stopped_when_no_pid_and_port_free(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    for d in (paths.config_dir, paths.state_dir, paths.logs_dir, paths.support_dir, paths.temp_dir):
        d.mkdir(parents=True, exist_ok=True)
    config_path = paths.default_config_path
    _write_config(config_path)
    state = LauncherState(paths=paths)
    status = state.proxy_runtime_status(config_path)
    assert status.code == RuntimeStatusCode.STOPPED


def test_status_port_occupied_without_owned_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    for d in (paths.config_dir, paths.state_dir, paths.logs_dir, paths.support_dir, paths.temp_dir):
        d.mkdir(parents=True, exist_ok=True)
    config_path = paths.default_config_path
    _write_config(config_path)
    state = LauncherState(paths=paths)
    monkeypatch.setattr("aurora_lens.launcher.state._port_is_free", lambda port, host="127.0.0.1": False)
    monkeypatch.setattr(state, "_probe_health", lambda port: (False, "no health"))  # noqa: ARG005
    status = state.proxy_runtime_status(config_path)
    assert status.code == RuntimeStatusCode.PORT_OCCUPIED


def test_stop_already_stopped_is_success(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    for d in (paths.config_dir, paths.state_dir, paths.logs_dir, paths.support_dir, paths.temp_dir):
        d.mkdir(parents=True, exist_ok=True)
    config_path = paths.default_config_path
    _write_config(config_path)
    state = LauncherState(paths=paths)
    state._probe_health = lambda port: (False, "no health")  # type: ignore[method-assign]
    result = state.stop_proxy_lifecycle(config_path)
    assert result.code == LifecycleResultCode.ALREADY_STOPPED


def test_tasklist_access_denied_means_pid_status_unknown() -> None:
    status = launcher_state._classify_windows_tasklist_result(
        4321,
        returncode=1,
        stdout="",
        stderr="ERROR: Access denied",
    )
    assert status == launcher_state._PidStatus.UNKNOWN


def test_unknown_pid_with_healthy_proxy_reports_healthy_without_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _paths(tmp_path)
    for directory in (
        paths.config_dir,
        paths.state_dir,
        paths.logs_dir,
        paths.support_dir,
        paths.temp_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    config_path = paths.default_config_path
    _write_config(config_path)
    state = LauncherState(paths=paths)
    state.proxy_pid_path.write_text(
        json.dumps(
            {
                "pid": 4321,
                "port": 8081,
                "phase": "running",
                "config": str(config_path),
                "log_path": str(state.proxy_log_path),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        launcher_state,
        "_pid_status",
        lambda pid: launcher_state._PidStatus.UNKNOWN,
    )
    monkeypatch.setattr(state, "_probe_health", lambda port: (True, "ok"))

    status = state.proxy_runtime_status(config_path)

    assert status.code == RuntimeStatusCode.RUNNING_HEALTHY
    assert status.details["pid_status"] == "unknown"
    assert state.proxy_pid_path.exists()
    assert list(paths.state_dir.glob("*.quarantine.*.json")) == []
