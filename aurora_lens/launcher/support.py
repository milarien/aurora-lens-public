from __future__ import annotations

import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from aurora_lens.build_info import load_build_info, public_release_metadata
from aurora_lens.launcher.paths import RuntimePaths


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base_dir / path
    return path


def export_support_bundle(paths: RuntimePaths, config_path: Path) -> dict[str, Any]:
    runtime_files: list[Path] = [
        paths.proxy_pid_path,
        paths.launcher_pid_path,
        paths.launcher_log_path,
        paths.proxy_log_path,
    ]
    if config_path.exists():
        runtime_files.append(config_path)
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            gov = payload.get("governance") or {}
            audit_path = str(gov.get("audit_log") or "").strip()
            if audit_path:
                runtime_files.append(_resolve_path(audit_path, config_path.parent))
        except Exception:
            pass
    archive_path = paths.support_dir / f"aurora-lens-support-{_utc_stamp()}.zip"
    included_files: list[str] = []
    seen: set[Path] = set()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file_path in runtime_files:
            if not file_path.exists() or not file_path.is_file():
                continue
            resolved = file_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                arcname = file_path.relative_to(paths.runtime_root).as_posix()
            except ValueError:
                arcname = file_path.name
            bundle.write(file_path, arcname=arcname)
            included_files.append(arcname)
        manifest = {
            "created_utc": datetime.now(UTC).isoformat(),
            "runtime_root": str(paths.runtime_root),
            "config_path": str(config_path),
            "included_files": included_files,
            "release": public_release_metadata(load_build_info()),
        }
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    return {"archive_path": str(archive_path), "included_files": included_files}
