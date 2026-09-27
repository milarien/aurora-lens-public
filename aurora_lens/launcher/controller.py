from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from importlib import util as importlib_util
from importlib import metadata
from pathlib import Path

from aurora_lens.launcher.lifecycle import LifecycleResultCode, RuntimeStatusCode
from aurora_lens.launcher.paths import (
    ensure_runtime_dirs,
    resolve_config_path,
    resolve_runtime_paths,
)
from aurora_lens.launcher.state import LauncherState
from aurora_lens.proxy.config import ProxyConfig


def _editable_install_warning(runtime_root: Path) -> str | None:
    try:
        dist = metadata.distribution("aurora-lens")
    except metadata.PackageNotFoundError:
        return None
    direct_url_path = Path(dist._path) / "direct_url.json"  # type: ignore[attr-defined]
    if not direct_url_path.exists():
        return None
    try:
        payload = json.loads(direct_url_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not bool((payload.get("dir_info") or {}).get("editable")):
        return None
    raw_url = str(payload.get("url") or "").strip()
    if raw_url.startswith("file:///"):
        source_hint = raw_url.removeprefix("file:///").replace("/", os.sep)
    else:
        source_hint = raw_url or "unknown location"
    root_resolved = runtime_root.resolve()
    try:
        source_path = Path(source_hint).resolve()
        if source_path == root_resolved:
            return None
    except Exception:
        pass
    return (
        "Detected editable aurora-lens install pointing to "
        f"{source_hint}. This can break release import isolation. "
        "Reinstall Aurora-Lens from the release tree with a non-editable install."
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aurora-lens-launcher-controller",
        description="Aurora-Lens lifecycle controller (start/status/stop).",
    )
    parser.add_argument("action", choices=("start", "status", "stop"))
    parser.add_argument("--config", default=None, help="Absolute or relative config path")
    return parser.parse_args(argv)


def _print_header(config_path: Path, state: LauncherState) -> None:
    print(f"Resolved config path: {config_path}")
    print(f"Runtime root: {state.paths.runtime_root}")
    print(f"State directory: {state.paths.state_dir}")
    print(f"Support directory: {state.paths.support_dir}")
    print(f"Temp directory: {state.paths.temp_dir}")
    print(f"Proxy log path: {state.proxy_log_path}")
    print(f"PID path: {state.proxy_pid_path}")


def _validate_config_or_fail(config_path: Path) -> tuple[bool, str, ProxyConfig | None]:
    if not config_path.exists():
        return False, f"Config file not found: {config_path}", None
    try:
        cfg = ProxyConfig.from_yaml(config_path)
        cfg.validate()
    except Exception as exc:
        return False, f"Invalid config: {exc}", None
    return True, "", cfg


def _spacy_installed() -> bool:
    return importlib_util.find_spec("spacy") is not None


def _spacy_model_available(model: str) -> bool:
    try:
        import spacy
    except Exception:
        return False
    try:
        spacy.load(model)
        return True
    except Exception:
        return False


def _run_setup_command(command: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output.strip()


def _ensure_spacy_dependencies(model: str) -> tuple[bool, str]:
    if not _spacy_installed():
        print('Setup: spaCy not found. Installing with "aurora-lens[spacy]" ...')
        ok, output = _run_setup_command([sys.executable, "-m", "pip", "install", "aurora-lens[spacy]"])
        if not ok:
            return (
                False,
                "Failed to install spaCy dependencies automatically.\n"
                f"Command: {sys.executable} -m pip install \"aurora-lens[spacy]\"\n"
                f"Output:\n{output}",
            )

    if not _spacy_model_available(model):
        print(f"Setup: spaCy model '{model}' not found. Downloading ...")
        ok, output = _run_setup_command([sys.executable, "-m", "spacy", "download", model])
        if not ok:
            return (
                False,
                "Failed to install spaCy language model automatically.\n"
                f"Command: {sys.executable} -m spacy download {model}\n"
                f"Output:\n{output}",
            )

    if not _spacy_installed() or not _spacy_model_available(model):
        return (
            False,
            "spaCy setup did not complete successfully; proxy startup aborted.\n"
            f"Expected model: {model}",
        )
    return True, ""


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    paths = resolve_runtime_paths()
    ensure_runtime_dirs(paths)
    config_path = resolve_config_path(args.config, paths)
    state = LauncherState(paths=paths)
    _print_header(config_path, state)
    warning = _editable_install_warning(paths.runtime_root)
    if warning:
        print(f"WARNING: {warning}")
        state.log(f"import_isolation_warning detail={warning}")

    if args.action == "stop":
        result = state.stop_proxy_lifecycle(config_path=config_path)
        print(f"Stop result: {result.state_label} ({result.code.value})")
        print(result.message)
        return 0 if result.code in {
            LifecycleResultCode.STOPPED,
            LifecycleResultCode.ALREADY_STOPPED,
            LifecycleResultCode.STALE_STATE_REPAIRED,
        } else 1

    if args.action == "status":
        status = state.proxy_runtime_status(config_path=config_path)
        print(f"Status: {status.state_label} ({status.code.value})")
        print(status.message)
        details = status.details
        if details.get("proxy_url"):
            print(f"Proxy URL: {details['proxy_url']}")
        if details.get("health_url"):
            print(f"Health URL: {details['health_url']}")
        return 0 if status.code not in {
            RuntimeStatusCode.OWNERSHIP_UNVERIFIED,
            RuntimeStatusCode.PORT_OCCUPIED,
        } else 1

    ok, err, cfg = _validate_config_or_fail(config_path)
    if not ok:
        print(err)
        print(f"Log path: {state.proxy_log_path}")
        return 1
    assert cfg is not None
    print(f"Provider: {cfg.upstream.provider}")
    print(f"Model: {cfg.upstream.model}")
    print(f"Governance: policy={cfg.governance.default_policy} mode={cfg.governance.mode}")
    print(f"Endpoint: {cfg.upstream.base_url or '(provider default)'}")
    if cfg.extraction.backend == "spacy":
        ready, detail = _ensure_spacy_dependencies(cfg.extraction.spacy_model)
        if not ready:
            print(detail)
            print("Start aborted. Proxy remains stopped.")
            print(f"Log path: {state.proxy_log_path}")
            return 1
    result = state.start_proxy(config_path=config_path, runtime_env={})
    print(f"Start result: {result.code.value}")
    print(result.message)
    if result.proxy_url:
        print(f"Proxy URL: {result.proxy_url}")
    if result.health_url:
        print(f"Health URL: {result.health_url}")
    if result.log_path:
        print(f"Log path: {result.log_path}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

