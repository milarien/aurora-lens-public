"""Recursive folder scan and ingest reporting for operator corpus ingest."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import re

from aurora_lens.corpus.ingest import ingest_supported_suffix, planned_ingest_suffix

SkipReason = Literal[
    "unsupported_type",
    "planned_type",
    "docx_dependency_missing",
    "hidden_file",
    "ingest_error",
    "empty_text",
]


@dataclass(frozen=True)
class SkippedFile:
    path: str
    relative_path: str
    suffix: str
    reason: SkipReason
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "relative_path": self.relative_path,
            "suffix": self.suffix,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass
class FolderScanReport:
    folder_selected: str
    recursive: bool
    supported_types: list[str]
    planned_types: list[str]
    files_found: int = 0
    files_supported: int = 0
    files_ingested: int = 0
    files_unchanged: int = 0
    files_skipped: int = 0
    skipped_by_type: dict[str, int] = field(default_factory=dict)
    skipped_folders: list[str] = field(default_factory=list)
    skipped_files: list[SkippedFile] = field(default_factory=list)
    supported_files: list[Path] = field(default_factory=list)
    ingest_results: list[dict[str, Any]] = field(default_factory=list)
    ingest_errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "folder_selected": self.folder_selected,
            "recursive": self.recursive,
            "supported_types": self.supported_types,
            "planned_types": self.planned_types,
            "files_found": self.files_found,
            "files_supported": self.files_supported,
            "files_ingested": self.files_ingested,
            "files_unchanged": self.files_unchanged,
            "files_skipped": self.files_skipped,
            "skipped_by_type": dict(sorted(self.skipped_by_type.items())),
            "skipped_folders": self.skipped_folders,
            "skipped_files": [s.to_dict() for s in self.skipped_files],
            "ingest_results": self.ingest_results,
            "ingest_errors": self.ingest_errors,
            "ok": self.files_ingested > 0 or self.files_unchanged > 0,
            "partial": self.files_skipped > 0 and (self.files_ingested + self.files_unchanged) > 0,
            "failed": (self.files_ingested + self.files_unchanged) == 0 and self.files_found > 0,
        }


def supported_type_labels() -> list[str]:
    return sorted(ingest_supported_suffix())


def planned_type_labels() -> list[str]:
    return sorted(planned_ingest_suffix())


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _suffix_key(path: Path) -> str:
    suffix = path.suffix.lower()
    return suffix if suffix else "(no extension)"


def _classify_file(path: Path) -> tuple[str, SkipReason | None, str]:
    """Return (suffix_key, skip_reason or None, detail)."""
    suffix = path.suffix.lower()
    if _is_hidden(path):
        return _suffix_key(path), "hidden_file", "hidden or dotfile"
    if suffix in ingest_supported_suffix():
        if suffix == ".docx":
            try:
                import docx  # noqa: F401
            except ImportError:
                return suffix, "docx_dependency_missing", "install python-docx to ingest .docx"
        return suffix, None, ""
    if suffix in planned_ingest_suffix():
        return suffix, "planned_type", "type planned for a later release (.xlsx, .pptx, .eml, .msg)"
    return _suffix_key(path), "unsupported_type", f"not in supported types: {', '.join(supported_type_labels())}"


def iter_folder_files(folder: Path, *, recursive: bool) -> list[Path]:
    root = folder.resolve()
    if recursive:
        candidates = sorted(p for p in root.rglob("*") if p.is_file())
    else:
        candidates = sorted(p for p in root.iterdir() if p.is_file())
    return candidates


def scan_path(path: Path, *, recursive: bool = True) -> FolderScanReport:
    """Scan a file or directory for ingest (no writes)."""
    target = path.expanduser().resolve()
    if target.is_file():
        report = FolderScanReport(
            folder_selected=str(target.parent),
            recursive=False,
            supported_types=supported_type_labels(),
            planned_types=planned_type_labels(),
            files_found=1,
        )
        suffix_key, skip_reason, detail = _classify_file(target)
        if skip_reason:
            report.files_skipped = 1
            report.skipped_by_type[suffix_key] = 1
            report.skipped_files.append(
                SkippedFile(
                    path=str(target),
                    relative_path=target.name,
                    suffix=suffix_key,
                    reason=skip_reason,
                    detail=detail,
                )
            )
        else:
            report.files_supported = 1
            report.supported_files = [target]
        return report
    if target.is_dir():
        return scan_folder(target, recursive=recursive)
    raise FileNotFoundError(f"path not found: {target}")


def scan_folder(folder: Path, *, recursive: bool = True) -> FolderScanReport:
    """Discover files under ``folder`` and classify for ingest (no writes)."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    report = FolderScanReport(
        folder_selected=str(root),
        recursive=recursive,
        supported_types=supported_type_labels(),
        planned_types=planned_type_labels(),
    )
    ingested_dirs: set[str] = set()
    skipped_dirs: set[str] = set()

    for file_path in iter_folder_files(root, recursive=recursive):
        report.files_found += 1
        rel = str(file_path.relative_to(root))
        parent_rel = str(file_path.parent.relative_to(root)) if file_path.parent != root else "."
        suffix_key, skip_reason, detail = _classify_file(file_path)

        if skip_reason is None:
            report.files_supported += 1
            report.supported_files.append(file_path)
            ingested_dirs.add(parent_rel)
            continue

        report.files_skipped += 1
        report.skipped_by_type[suffix_key] = report.skipped_by_type.get(suffix_key, 0) + 1
        report.skipped_files.append(
            SkippedFile(
                path=str(file_path),
                relative_path=rel,
                suffix=suffix_key,
                reason=skip_reason,
                detail=detail,
            )
        )
        skipped_dirs.add(parent_rel)

    for dir_rel in sorted(skipped_dirs):
        if dir_rel not in ingested_dirs:
            report.skipped_folders.append("." if dir_rel == "." else dir_rel.replace("\\", "/"))

    return report


def record_id_for_file(file_path: Path, root: Path, explicit: str | None = None) -> str:
    if explicit:
        rid = explicit.strip()
        if not rid:
            raise ValueError("record_id must be non-empty")
        return rid
    rel = file_path.relative_to(root)
    stem_slug = _slug_record_id(str(rel.with_suffix("")))
    return stem_slug or _slug_record_id(file_path.stem)


def _slug_record_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "document"
