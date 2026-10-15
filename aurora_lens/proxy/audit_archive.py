"""Read-only exports of retained audit bytes, including restart archives.

Exports preserve malformed rows and hash-only records exactly as stored. A
snapshot is bounded at each file's open size; it is not a global transaction.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask


def retained_files(current: Path) -> list[Path]:
    parent = current.parent.resolve()
    archive = re.compile(re.escape(current.stem) + r"\.archive\.[0-9TZ]+" + re.escape(current.suffix))
    restart = re.compile(re.escape(current.name) + r"\.pre_restart\.[0-9TZ]+")
    return sorted((p for p in parent.iterdir() if
                   (p.name == current.name or archive.fullmatch(p.name) or restart.fullmatch(p.name))
                   and not p.is_symlink() and p.is_file() and p.resolve().parent == parent),
                  key=lambda p: p.name)


def file_id(path: Path) -> str:
    stat = path.stat()
    # A rotated/replaced current file must not impersonate the selected file.
    return hashlib.sha256(f"{path.name}:{stat.st_dev}:{stat.st_ino}".encode()).hexdigest()


def _copy_snapshot(path: Path, target, forbidden: bytes) -> dict:
    digest = hashlib.sha256()
    lines = 0
    last = b""
    overlap = b""
    with path.open("rb") as source:
        stat = os.fstat(source.fileno())
        remaining = stat.st_size
        while remaining:
            chunk = source.read(min(65536, remaining))
            if not chunk:
                raise OSError("Audit file truncated during export")
            if forbidden and forbidden in overlap + chunk:
                raise ValueError("Credential present in audit file; export withheld")
            overlap = (overlap + chunk)[-max(len(forbidden) - 1, 0):] if forbidden else b""
            target.write(chunk)
            digest.update(chunk)
            lines += chunk.count(b"\n")
            last = chunk[-1:]
            remaining -= len(chunk)
    return {"name": path.name, "bytes": stat.st_size, "sha256": digest.hexdigest(),
            "physical_lines": lines + int(bool(last and last != b"\n"))}


def _stream_file(spool):
    try:
        while chunk := spool.read(65536):
            yield chunk
    finally:
        spool.close()


def archive_response(current: Path, selected: str | None, download: bool, forbidden: str):
    """Called only behind the existing diagnostic-authentication middleware."""
    headers = {"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"}
    spool = None
    try:
        files = retained_files(current)
        if selected is None and not download:
            return JSONResponse({"files": [{"id": file_id(p), "name": p.name,
                "bytes": p.stat().st_size, "current": p.name == current.name} for p in files],
                "scope": "Retained configured audit file and its restart/dated archives. No row limit.",
                "capture_limit": "Missing plaintext cannot be recovered from hashes."}, headers=headers)
        spool = tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b")
        if selected is not None:
            path = next((p for p in files if file_id(p) == selected), None)
            if path is None:
                spool.close()
                return JSONResponse({"error": "Audit file no longer selected; refresh the file list."},
                                    status_code=404, headers=headers)
            info = _copy_snapshot(path, spool, forbidden.encode())
            headers["X-Audit-SHA256"] = info["sha256"]
            headers["Content-Disposition"] = f'{"attachment" if download else "inline"}; filename="{path.name}"'
            media_type = "text/plain; charset=utf-8"
        else:
            manifest = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "scope": "All retained audit files; per-file byte snapshots, not an atomic global snapshot.",
                        "capture_limit": "Bytes preserved as stored, including malformed and hash-only rows.",
                        "files": []}
            with zipfile.ZipFile(spool, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in files:
                    with bundle.open(path.name, "w") as target:
                        manifest["files"].append(_copy_snapshot(path, target, forbidden.encode()))
                bundle.writestr("manifest.json", json.dumps(manifest, indent=2))
            headers["Content-Disposition"] = 'attachment; filename="aurora-lens-full-audit.zip"'
            media_type = "application/zip"
        headers["Content-Length"] = str(spool.tell())
        spool.seek(0)
        return StreamingResponse(_stream_file(spool), media_type=media_type, headers=headers,
                                 background=BackgroundTask(spool.close))
    except (OSError, ValueError):
        if spool is not None:
            spool.close()
        return JSONResponse({"error": "Audit export unavailable; no records were changed."},
                            status_code=503, headers=headers)
