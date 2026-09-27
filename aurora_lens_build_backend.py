from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from setuptools import build_meta as _setuptools_build_meta

from release_guard import assert_clean_release_tree

_ROOT = Path(__file__).resolve().parent
_WHEEL_LICENSE_FILES = ("LICENSE", "NOTICE", "LICENSING.md", "COMMERCIAL-LICENCE.md")


def _read_project_version() -> str:
    """Best-effort read of [project].version from pyproject.toml."""
    pyproject = _ROOT / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except Exception:
        return "unofficial"
    try:
        # Python 3.11+ ships tomllib
        import tomllib  # type: ignore
        data = tomllib.loads(text)
        version = str(((data.get("project") or {}).get("version") or "")).strip()
        return version or "unofficial"
    except Exception:
        # Simple fallback line-scan
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("version") and "=" in stripped:
                _, _, value = stripped.partition("=")
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
        return "unofficial"


def _git_head_commit() -> str:
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
        )
        out = (cp.stdout or "").strip()
        if cp.returncode == 0 and out:
            return out
    except Exception:
        pass
    return "unofficial"


def _git_release_tag() -> str:
    try:
        cp = subprocess.run(
            ["git", "describe", "--tags", "--exact-match", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(_ROOT),
        )
        out = (cp.stdout or "").strip()
        if cp.returncode == 0 and out:
            return out
    except Exception:
        pass
    return "untagged"


def _utc_iso_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_build_info() -> None:
    """Write aurora_lens/BUILD_INFO.json with truthful release identity.

    When building from an sdist in a gitless temp directory, pip invokes this
    backend a second time.  If a truthful BUILD_INFO.json was already baked
    into the source tree (source_commit is a real SHA, not "unofficial"), skip
    the write so the original values are preserved in the output wheel.
    """
    target_dir = _ROOT / "aurora_lens"
    target = target_dir / "BUILD_INFO.json"
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
            commit = existing.get("source_commit", "unofficial")
            if commit != "unofficial" and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit):
                return  # preserve the truthful values baked in at sdist-build time
        except Exception:
            pass
    payload = {
        "version": _read_project_version(),
        "source_commit": _git_head_commit(),
        "built_at": _utc_iso_now(),
        "release_tag": _git_release_tag(),
        "product": "Aurora-Lens",
    }
    target_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _guard() -> None:
    _write_build_info()
    assert_clean_release_tree(_ROOT)


def _ensure_wheel_license_files(wheel_path: Path) -> None:
    """Ensure PEP 639 license-files are present in the wheel licenses/ directory.

    PEP 517 isolated builds use setuptools' vendored bdist_wheel, which only
    copies LICENSE/LICENCE/NOTICE via legacy glob patterns and can omit
    LICENSING.md even when declared in project.license-files.
    """
    with zipfile.ZipFile(wheel_path, "r") as src:
        names = set(src.namelist())
        dist_info_prefix = next(
            (name.split("/")[0] for name in names if name.endswith(".dist-info/METADATA")),
            None,
        )
        if dist_info_prefix is None:
            return
        missing = [
            name
            for name in _WHEEL_LICENSE_FILES
            if f"{dist_info_prefix}/licenses/{name}" not in names and (_ROOT / name).is_file()
        ]
        if not missing:
            return
        fd, tmp_name = tempfile.mkstemp(suffix=".whl")
        try:
            os.close(fd)
            with zipfile.ZipFile(wheel_path, "r") as src, zipfile.ZipFile(tmp_name, "w") as dst:
                for item in src.infolist():
                    dst.writestr(item, src.read(item.filename))
                for name in missing:
                    target = f"{dist_info_prefix}/licenses/{name}"
                    dst.writestr(target, (_ROOT / name).read_bytes())
            shutil.move(tmp_name, wheel_path)
        finally:
            tmp_path = Path(tmp_name)
            if tmp_path.exists():
                tmp_path.unlink()


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, str] | None = None,
    metadata_directory: str | None = None,
) -> str:
    _guard()
    wheel_name = _setuptools_build_meta.build_wheel(
        wheel_directory=wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )
    _ensure_wheel_license_files(Path(wheel_directory) / wheel_name)
    return wheel_name


def build_sdist(sdist_directory: str, config_settings: dict[str, str] | None = None) -> str:
    _guard()
    return _setuptools_build_meta.build_sdist(
        sdist_directory=sdist_directory,
        config_settings=config_settings,
    )


def get_requires_for_build_wheel(config_settings: dict[str, str] | None = None) -> list[str]:
    _guard()
    return _setuptools_build_meta.get_requires_for_build_wheel(config_settings=config_settings)


def get_requires_for_build_sdist(config_settings: dict[str, str] | None = None) -> list[str]:
    _guard()
    return _setuptools_build_meta.get_requires_for_build_sdist(config_settings=config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, str] | None = None,
) -> str:
    _guard()
    return _setuptools_build_meta.prepare_metadata_for_build_wheel(
        metadata_directory=metadata_directory,
        config_settings=config_settings,
    )
