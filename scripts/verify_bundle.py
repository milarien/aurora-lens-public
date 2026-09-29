#!/usr/bin/env python3
"""Verify an aurora-lens evidence bundle against its manifest.

Stdlib only. Checks each file listed in manifest.json for presence, SHA-256,
and byte size match.

Usage:
    python scripts/verify_bundle.py <bundle_dir>

Exit codes:
    0  All listed files match the manifest.
    1  One or more listed files missing, wrong size, or hash mismatch.
    2  bundle_dir invalid, or manifest.json not found/unreadable invalid.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest_entry(bundle_dir: Path, name: str, entry: dict) -> tuple[bool, list[str]]:
    """Verify one manifest row. Returns (ok, messages for diagnostics)."""
    fp = bundle_dir / name
    msgs: list[str] = []
    if not fp.exists():
        return False, [f"MISSING: {name}"]
    actual_sha256 = _file_sha256(fp)
    expected_sha256 = entry.get("sha256", "")
    if actual_sha256 != expected_sha256:
        msgs.append(
            f"HASH_MISMATCH: {name}\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual_sha256}"
        )
        return False, msgs
    actual_size = fp.stat().st_size
    expected_size = entry.get("size_bytes")
    if expected_size is not None and actual_size != expected_size:
        msgs.append(
            f"SIZE_MISMATCH: {name} expected_bytes={expected_size} actual_bytes={actual_size}"
        )
        return False, msgs
    return True, []


def verify_bundle(
    bundle_dir: Path,
) -> tuple[bool, list[str], list[tuple[str, bool]]]:
    """Verify manifest-listed files under ``bundle_dir``. Only listed files are checked.

    Returns ``(overall_ok, issues, per_file)`` where ``per_file`` is
    ``(filename, ok)`` rows in manifest order (well-formed entries only).
    """
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        return False, [f"manifest.json not found in {bundle_dir}"], []

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return False, [f"manifest.json unreadable: {e}"], []

    issues: list[str] = []
    file_rows = manifest.get("files")
    if file_rows is None:
        file_rows = []
    if not isinstance(file_rows, list):
        return False, ["manifest.files must be a list"], []

    per_file: list[tuple[str, bool]] = []

    for row in file_rows:
        if not isinstance(row, dict) or "name" not in row:
            issues.append(f"Malformed manifest.files entry: {row!r}")
            continue
        name = row["name"]
        ok_ent, msgs = verify_manifest_entry(bundle_dir, name, row)
        per_file.append((name, ok_ent))
        if not ok_ent:
            issues.extend(msgs)

    return len(issues) == 0, issues, per_file


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: verify_bundle.py <bundle_dir>", file=sys.stderr)
        return 2

    bundle_dir = Path(sys.argv[1])
    if not bundle_dir.is_dir():
        print(f"Error: {bundle_dir} is not a directory", file=sys.stderr)
        return 2

    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: manifest.json not found under {bundle_dir}", file=sys.stderr)
        return 2

    try:
        json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error: manifest.json unreadable: {e}", file=sys.stderr)
        return 2

    ok, issues, per_file = verify_bundle(bundle_dir)

    for name, ent_ok in per_file:
        if ent_ok:
            print(f"PASS {name}")
        else:
            print(f"FAIL {name}")

    for msg in issues:
        print(msg, file=sys.stderr)

    if ok:
        print("OK: all manifest-listed files verified.")
        return 0

    print("Bundle verification FAILED.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
