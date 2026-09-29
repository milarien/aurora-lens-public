from __future__ import annotations

import json
from pathlib import Path

from aurora_lens.launcher.state import LauncherState


def test_stale_proxy_pid_file_is_cleaned(tmp_path: Path) -> None:
    state = LauncherState(home=tmp_path)
    stale_path = tmp_path / "aurora-lens.proxy.pid"
    stale_path.write_text(json.dumps({"pid": 999999, "port": 8000}), encoding="utf-8")
    assert state.proxy_pid() is None
    assert not stale_path.exists()


def test_stop_proxy_when_not_running_returns_false(tmp_path: Path) -> None:
    state = LauncherState(home=tmp_path)
    assert state.stop_proxy() is False


def test_invalid_proxy_pid_file_is_removed(tmp_path: Path) -> None:
    state = LauncherState(home=tmp_path)
    p = tmp_path / "aurora-lens.proxy.pid"
    p.write_text("\ufeffnot-json", encoding="utf-8")
    assert state.proxy_pid() is None
    assert not p.exists()

