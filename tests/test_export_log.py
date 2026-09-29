"""Tests for scripts/export_log.py (Stage G custody export)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def test_export_log_subprocess_custody_record(tmp_path, monkeypatch):
    """export_log writes correct sha256, size_bytes, entry_count, exported_at (and optional exported_by)."""
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "export_log.py"
    log = tmp_path / "audit.jsonl"
    file_bytes = b'{"row":1}\n\n{"row":2}\n'
    log.write_bytes(file_bytes)

    monkeypatch.setenv("AURORA_LENS_EXPORTED_BY", "stage-g-test-exporter")

    before = datetime.now(timezone.utc)
    proc = subprocess.run(
        [sys.executable, str(script), "--log", str(log)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    after = datetime.now(timezone.utc) + timedelta(seconds=2)

    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)

    assert data["file"] == "audit.jsonl"
    assert data["sha256"] == hashlib.sha256(file_bytes).hexdigest()
    assert data["size_bytes"] == len(file_bytes)
    assert data["entry_count"] == 2
    assert data["exported_by"] == "stage-g-test-exporter"

    exported_at = datetime.fromisoformat(data["exported_at"])
    assert exported_at.tzinfo is not None
    assert exported_at.utcoffset() == timedelta(0)
    assert before <= exported_at <= after


def test_export_log_sequence_excerpt_label(tmp_path, monkeypatch):
    """When seq starts above 1, export labels sequence as excerpt."""
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "export_log.py"
    log = tmp_path / "audit.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"seq": 6, "kind": "aurora.event"}),
                json.dumps({"seq": 7, "kind": "aurora.event"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("AURORA_LENS_EXPORTED_BY", raising=False)
    proc = subprocess.run(
        [sys.executable, str(script), "--log", str(log)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["sequence_available"] is True
    assert data["sequence_first"] == 6
    assert data["sequence_last"] == 7
    assert data["sequence_contiguous"] is True
    assert data["sequence_excerpt"] is True
    assert data["sequence_label"] == "seq 6-7 (excerpt)"


def test_export_log_sequence_full_label(tmp_path, monkeypatch):
    """When seq starts at 1 and has no gaps, export labels sequence as full."""
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "scripts" / "export_log.py"
    log = tmp_path / "audit.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"seq": 1, "kind": "aurora.event"}),
                json.dumps({"seq": 2, "kind": "aurora.event"}),
                json.dumps({"seq": 3, "kind": "aurora.event"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("AURORA_LENS_EXPORTED_BY", raising=False)
    proc = subprocess.run(
        [sys.executable, str(script), "--log", str(log)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["sequence_available"] is True
    assert data["sequence_first"] == 1
    assert data["sequence_last"] == 3
    assert data["sequence_contiguous"] is True
    assert data["sequence_excerpt"] is False
    assert data["sequence_label"] == "seq 1-3 (full)"
