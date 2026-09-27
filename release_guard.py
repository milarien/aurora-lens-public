from __future__ import annotations

import re
import tarfile
import zipfile
from pathlib import Path


_FORBIDDEN_GLOBS: tuple[str, ...] = (
    "audit.jsonl",
    "audit*.jsonl",
    "audit*.jsonl.pre_*",
    "*.ledger",
    "*.ledger.*",
    "*.pid",
    "*.quarantine.*.json",
    "launcher.log",
    "proxy.log",
    "support*.zip",
    "*.support*.zip",
)

_FORBIDDEN_DIR_NAMES: tuple[str, ...] = (
    "state",
    "logs",
    "support",
    "tmp",
    "sessions",
    "evidence",
)

_ALLOWLIST_ROOTS: tuple[str, ...] = ("tests", "examples")
_IGNORE_ROOTS: tuple[str, ...] = (".git", "build", "dist", ".pytest_cache", "__pycache__", "aurora_lens.egg-info")

_CREDENTIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9]{12,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
)
_GENERIC_ABS_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9_. -]+\\"),
    re.compile(r"[A-Za-z]:\\(?:Temp|tmp)\\"),
    re.compile(r"/Users/[A-Za-z0-9_.-]+/"),
    re.compile(r"/home/[A-Za-z0-9_.-]+/"),
)


def _is_allowlisted(rel: Path) -> bool:
    parts = rel.parts
    return bool(parts) and parts[0] in _ALLOWLIST_ROOTS


def _is_ignored(rel: Path) -> bool:
    parts = rel.parts
    return bool(parts) and parts[0] in _IGNORE_ROOTS


def scan_release_tree(root: Path) -> list[str]:
    problems: list[str] = []
    root = root.resolve()
    for pattern in _FORBIDDEN_GLOBS:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if _is_ignored(rel):
                continue
            if _is_allowlisted(rel):
                continue
            if path.stat().st_size > 0:
                problems.append(f"forbidden runtime artifact: {rel}")
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        rel = path.relative_to(root)
        if _is_ignored(rel):
            continue
        if _is_allowlisted(rel):
            continue
        if path.name.lower().startswith("exports"):
            if any(p.is_file() for p in path.rglob("*")):
                problems.append(f"forbidden runtime directory with files: {rel}")
        if path.name.lower() in _FORBIDDEN_DIR_NAMES or path.name.lower().endswith(".evidence"):
            if any(p.is_file() for p in path.rglob("*")):
                problems.append(f"forbidden runtime directory with files: {rel}")
    return sorted(set(problems))


def assert_clean_release_tree(root: Path) -> None:
    issues = scan_release_tree(root)
    if issues:
        formatted = "\n".join(f"- {i}" for i in issues)
        raise RuntimeError(f"Clean-audit packaging guard failed:\n{formatted}")


def _scan_archive_names(names: list[str]) -> list[str]:
    problems: list[str] = []
    for name in names:
        lower = name.lower()
        if "/tests/" in lower or lower.startswith("tests/"):
            continue
        if "/examples/" in lower or lower.startswith("examples/"):
            continue
        if any(lower.endswith(glob.replace("*", "")) for glob in ("audit.jsonl", "launcher.log", "proxy.log")):
            problems.append(f"archive contains forbidden runtime file: {name}")
        if ".pid" in lower or ".quarantine." in lower or ".evidence" in lower:
            problems.append(f"archive contains forbidden runtime state/evidence: {name}")
        if "support" in lower and lower.endswith(".zip"):
            problems.append(f"archive contains support bundle: {name}")
    return problems


def _make_path_markers() -> list[str]:
    markers: list[str] = []
    roots = [
        Path.cwd(),
        Path.cwd().parent,
        Path.home(),
        Path.home() / "AppData" / "Local" / "Temp",
    ]
    for candidate in roots:
        raw = str(candidate)
        if raw and raw not in markers:
            markers.append(raw)
    return markers


def _scan_member_payload(name: str, payload: bytes, path_markers: list[str]) -> list[str]:
    if len(payload) > 1_000_000:
        return []
    text = payload.decode("utf-8", errors="ignore")
    problems: list[str] = []
    lower_text = text.lower()
    for marker in path_markers:
        marker_norm = marker.replace("/", "\\").lower()
        marker_posix = marker.replace("\\", "/").lower()
        if marker_norm and marker_norm in lower_text:
            problems.append(f"archive member leaks local path marker '{marker}': {name}")
            break
        if marker_posix and marker_posix in lower_text:
            problems.append(f"archive member leaks local path marker '{marker}': {name}")
            break
    if not any("local path marker" in issue for issue in problems):
        for path_pattern in _GENERIC_ABS_PATH_PATTERNS:
            if path_pattern.search(text):
                problems.append(f"archive member contains absolute path text: {name}")
                break
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            problems.append(f"archive member appears to contain credential material: {name}")
            break
    return problems


def inspect_distribution_archive(path: Path) -> list[str]:
    path = path.resolve()
    problems: list[str] = []
    path_markers = _make_path_markers()
    if path.suffix == ".whl" or path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            problems.extend(_scan_archive_names(names))
            for name in names:
                if name.endswith(("/", ".png", ".jpg", ".jpeg", ".gif", ".whl")):
                    continue
                lower_name = name.lower()
                if "/tests/" in lower_name or lower_name.startswith("tests/"):
                    continue
                if "/examples/" in lower_name or lower_name.startswith("examples/"):
                    continue
                try:
                    payload = zf.read(name)
                except Exception:
                    continue
                problems.extend(_scan_member_payload(name, payload, path_markers))
        return sorted(set(problems))
    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        with tarfile.open(path, "r:gz") as tf:
            names = tf.getnames()
            problems.extend(_scan_archive_names(names))
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                lower_name = member.name.lower()
                if "/tests/" in lower_name or lower_name.startswith("tests/"):
                    continue
                if "/examples/" in lower_name or lower_name.startswith("examples/"):
                    continue
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                payload = extracted.read()
                problems.extend(_scan_member_payload(member.name, payload, path_markers))
        return sorted(set(problems))
    return [f"unsupported archive format: {path.name}"]
