#!/usr/bin/env python3
"""Export an aurora-lens audit log with a custody record.

Computes SHA-256 of the full log file, counts entries, and writes a custody
record JSON to stdout (or --output). This record may be used to verify that a
log file has not been altered during transfer.

Usage:
    python scripts/export_log.py --log ./audit.jsonl
    python scripts/export_log.py --log ./audit.jsonl --output custody.json
    python scripts/export_log.py --log ./audit.jsonl --exported-by "Acme Corp"
    python scripts/export_log.py --log ./audit.jsonl --operator "Acme Corp"   # alias for --exported-by

Environment (optional custody label):
    AURORA_LENS_EXPORTED_BY  Operator or system label recorded as ``exported_by`` in the JSON.

Output format (JSON):
    {
      "file": "audit.jsonl",
      "sha256": "<hex>",
      "size_bytes": <int>,
      "entry_count": <int>,
      "exported_at": "<UTC ISO-8601>",
      "exported_by": "<optional; from env or --exported-by / --operator>",
      "sequence_available": <bool>,
      "sequence_label": "seq <first>-<last> (excerpt|full)"   # when sequence_available=true
    }

Exit codes:
    0  Success.
    1  File not found or unreadable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_entries(path: Path) -> int:
    """Count non-empty JSONL lines."""
    try:
        return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
    except OSError:
        return 0


def _iter_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    except OSError:
        return []
    return rows


def _collapse_missing_ranges(values: list[int]) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    start = values[0]
    prev = values[0]
    for cur in values[1:]:
        if cur == prev + 1:
            prev = cur
            continue
        out.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = cur
    out.append(f"{start}-{prev}" if start != prev else str(start))
    return out


def _sequence_metadata(path: Path) -> dict:
    rows = _iter_jsonl_rows(path)
    seq_values = sorted({
        int(row["seq"])
        for row in rows
        if isinstance(row.get("seq"), int)
    })
    if not seq_values:
        return {"sequence_available": False}

    first = seq_values[0]
    last = seq_values[-1]
    full = list(range(first, last + 1))
    missing = sorted(set(full) - set(seq_values))
    contiguous = len(missing) == 0
    excerpt = first > 1 or not contiguous
    label = f"seq {first}-{last} ({'excerpt' if excerpt else 'full'})"

    meta: dict = {
        "sequence_available": True,
        "sequence_first": first,
        "sequence_last": last,
        "sequence_count": len(seq_values),
        "sequence_contiguous": contiguous,
        "sequence_excerpt": excerpt,
        "sequence_label": label,
    }
    if missing:
        meta["sequence_missing_ranges"] = _collapse_missing_ranges(missing)
    return meta


def build_custody_record(
    log_path: Path,
    *,
    exported_at: datetime | None = None,
    exported_by: str = "",
) -> dict:
    """Build the custody-record dict for *log_path* (file must exist). For tests and reuse."""
    exported_at = exported_at if exported_at is not None else datetime.now(timezone.utc)
    sha256 = _file_sha256(log_path)
    size_bytes = log_path.stat().st_size
    entry_count = _count_entries(log_path)
    record: dict = {
        "file": log_path.name,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "entry_count": entry_count,
        "exported_at": exported_at.isoformat(),
    }
    if exported_by:
        record["exported_by"] = exported_by
    record.update(_sequence_metadata(log_path))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export an audit log with a chain-of-custody record.",
    )
    parser.add_argument("--log", required=True, help="Path to audit JSONL log file")
    parser.add_argument("--output", default=None, help="Write custody record to this file (default: stdout)")
    parser.add_argument(
        "--exported-by",
        default="",
        help="Optional label stored as exported_by in the custody record",
    )
    parser.add_argument(
        "--operator",
        default="",
        help="Alias for --exported-by (backward compatibility)",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: {log_path} not found", file=sys.stderr)
        return 1

    env_label = os.environ.get("AURORA_LENS_EXPORTED_BY", "").strip()
    cli_label = (args.exported_by or args.operator or "").strip()
    exported_by = env_label or cli_label

    record = build_custody_record(log_path, exported_by=exported_by)

    out = json.dumps(record, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(out + "\n", encoding="utf-8")
        print(f"Custody record written to {args.output}")
        print(f"Log SHA-256: {record['sha256']}")
    else:
        print(out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
