from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_BUILD_INFO: dict[str, str] = {
    "version": "unofficial",
    "source_commit": "unofficial",
    "built_at": "unofficial",
    "release_tag": "unofficial",
}

_REQUIRED_KEYS: tuple[str, ...] = (
    "version",
    "source_commit",
    "built_at",
    "release_tag",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _in_package_build_info() -> Path:
    return Path(__file__).resolve().parent / "BUILD_INFO.json"


def _normalize_payload(payload: dict[str, Any]) -> dict[str, str]:
    out = dict(DEFAULT_BUILD_INFO)
    for key in _REQUIRED_KEYS:
        raw = payload.get(key)
        if raw is None:
            continue
        value = str(raw).strip()
        out[key] = value if value else DEFAULT_BUILD_INFO[key]
    return out


def _candidate_build_files(repo_root: Path | None) -> list[Path]:
    """Ordered candidate locations for BUILD_INFO.json.

    1. Installed wheel: co-located inside the aurora_lens package.
    2. Editable dev install: repo root (one level above the package).
    3. Explicit override: the ``repo_root`` argument, if provided.
    """
    candidates: list[Path] = []
    candidates.append(_in_package_build_info())
    candidates.append(_repo_root() / "BUILD_INFO.json")
    if repo_root is not None:
        override = repo_root / "BUILD_INFO.json"
        if override not in candidates:
            candidates.insert(0, override)
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        try:
            key = path.resolve()
        except Exception:
            key = path
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def load_build_info(repo_root: Path | None = None) -> dict[str, str]:
    for build_file in _candidate_build_files(repo_root):
        if not build_file.exists():
            continue
        try:
            raw = json.loads(build_file.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        return _normalize_payload(raw)
    return dict(DEFAULT_BUILD_INFO)


def public_release_metadata(info: dict[str, str] | None = None) -> dict[str, str]:
    """Ordinary public release fields only (no distribution fingerprinting)."""
    data = info if info is not None else load_build_info()
    return {
        "version": data["version"],
        "source_commit": data["source_commit"],
        "built_at": data["built_at"],
        "release_tag": data["release_tag"],
    }
