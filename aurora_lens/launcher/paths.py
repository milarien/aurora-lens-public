from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


APP_DIR_NAME = "Aurora-Lens"
CONFIG_FILE_NAME = "aurora-lens.yaml"


@dataclass(frozen=True)
class RuntimePaths:
    runtime_root: Path
    config_dir: Path
    state_dir: Path
    logs_dir: Path
    support_dir: Path
    temp_dir: Path

    @property
    def default_config_path(self) -> Path:
        return self.config_dir / CONFIG_FILE_NAME

    @property
    def proxy_pid_path(self) -> Path:
        return self.state_dir / "aurora-lens.proxy.pid"

    @property
    def launcher_pid_path(self) -> Path:
        return self.state_dir / "aurora-lens.launcher.pid"

    @property
    def launcher_log_path(self) -> Path:
        return self.logs_dir / "launcher.log"

    @property
    def proxy_log_path(self) -> Path:
        return self.logs_dir / "proxy.log"


def _default_runtime_root() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    if local_appdata:
        return Path(local_appdata) / APP_DIR_NAME
    xdg_state = os.environ.get("XDG_STATE_HOME", "").strip()
    if xdg_state:
        return Path(xdg_state) / "aurora-lens"
    return Path.home() / ".local" / "state" / "aurora-lens"


def resolve_runtime_paths() -> RuntimePaths:
    runtime_root = _default_runtime_root().resolve()
    return RuntimePaths(
        runtime_root=runtime_root,
        config_dir=runtime_root / "config",
        state_dir=runtime_root / "state",
        logs_dir=runtime_root / "logs",
        support_dir=runtime_root / "support",
        temp_dir=runtime_root / "tmp",
    )


def resolve_config_path(config_arg: str | None, paths: RuntimePaths) -> Path:
    if config_arg:
        return Path(config_arg).expanduser().resolve()
    return paths.default_config_path


def ensure_runtime_dirs(paths: RuntimePaths) -> None:
    paths.runtime_root.mkdir(parents=True, exist_ok=True)
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.support_dir.mkdir(parents=True, exist_ok=True)
    paths.temp_dir.mkdir(parents=True, exist_ok=True)


def runtime_paths_from_setup_home(home: Path) -> RuntimePaths:
    """Layout used by the setup wizard and legacy launcher tests (PID files under home)."""

    resolved = home.resolve()
    return RuntimePaths(
        runtime_root=resolved,
        config_dir=resolved / "config",
        state_dir=resolved,
        logs_dir=resolved / "logs",
        support_dir=resolved / "support",
        temp_dir=resolved / "tmp",
    )
