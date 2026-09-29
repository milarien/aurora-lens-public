"""Tests for recursive folder scan and ingest reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from aurora_lens.corpus.folder_scan import scan_folder, scan_path
from aurora_lens.corpus.operator_api import api_ingest_path, api_scan_folder


def test_scan_folder_recursive_counts(tmp_path: Path):
    root = tmp_path / "patents"
    sub = root / "Actual patent applications"
    wipo = root / "WIPO"
    sub.mkdir(parents=True)
    wipo.mkdir(parents=True)
    (sub / "a.pdf").write_bytes(b"%PDF-1.4\n")  # not real pdf but extension ok for scan
    (sub / "b.md").write_text("# B\n\nBody\n", encoding="utf-8")
    (wipo / "receipt.xlsx").write_bytes(b"PK")  # planned type only in WIPO
    (root / "skip.png").write_bytes(b"\x89PNG")
    (root / "top.txt").write_text("top level\n", encoding="utf-8")

    report = scan_folder(root, recursive=True)
    assert report.files_found == 5
    assert report.files_supported >= 3  # pdf, md, txt (+ docx if python-docx installed)
    assert report.files_skipped >= 1
    assert ".png" in report.skipped_by_type
    assert "WIPO" in report.skipped_folders or any("WIPO" in f for f in report.skipped_folders)


def test_scan_non_recursive_only_top_level(tmp_path: Path):
    root = tmp_path / "docs"
    sub = root / "nested"
    sub.mkdir(parents=True)
    (root / "top.md").write_text("# Top\n", encoding="utf-8")
    (sub / "nested.md").write_text("# Nested\n", encoding="utf-8")

    flat = scan_folder(root, recursive=False)
    assert flat.files_found == 1
    assert flat.supported_files[0].name == "top.md"

    deep = scan_folder(root, recursive=True)
    assert deep.files_found == 2


def test_api_scan_folder_preview(tmp_path: Path):
    root = tmp_path / "batch"
    root.mkdir()
    (root / "one.txt").write_text("hello", encoding="utf-8")
    (root / "two.xyz").write_text("nope", encoding="utf-8")

    out = api_scan_folder(path=root, recursive=True)
    assert out["files_found"] == 2
    assert out["files_supported"] == 1
    assert out["files_skipped"] == 1


def test_api_ingest_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    op_root = tmp_path / "operator"
    monkeypatch.setenv("AURORA_LENS_OPERATOR_CORPUS_ROOT", str(op_root))
    root = tmp_path / "src"
    root.mkdir()
    (root / "doc.md").write_text("# Doc\n\nText\n", encoding="utf-8")

    preview = api_ingest_path(path=root, dry_run=True)
    assert preview["dry_run"] is True
    assert preview["files_supported"] == 1
    assert api_ingest_path(path=root)["files_ingested"] == 1
