"""Tests for aurora-lens corpus CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aurora_lens.cli import main
from aurora_lens.corpus.cli_support import cmd_ingest, cmd_list
from aurora_lens.corpus.models import CorpusRecord


def test_cli_corpus_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["corpus", "--help"])
    assert exc.value.code == 0


def test_cli_corpus_ingest_help():
    with pytest.raises(SystemExit) as exc:
        main(["corpus", "ingest", "--help"])
    assert exc.value.code == 0


def test_cmd_list_empty_registry(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert cmd_list(corpus_root=tmp_path) == 0
    assert "No corpus records" in capsys.readouterr().out


def test_cmd_list_shows_records(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    tmp_path.mkdir(parents=True, exist_ok=True)
    registry_path = tmp_path / "registry.json"
    rec = CorpusRecord(
        record_id="demo",
        title="Demo Doc",
        source_path="data/demo.md",
        sha256="abc",
        chunk_count=3,
        ingested_at="2026-06-26T00:00:00Z",
    )
    registry_path.write_text(
        json.dumps({"version": 1, "records": [rec.to_dict()]}),
        encoding="utf-8",
    )
    assert cmd_list(corpus_root=tmp_path) == 0
    out = capsys.readouterr().out
    assert "demo" in out
    assert "Demo Doc" in out


def test_cmd_ingest_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    source = tmp_path / "sample.md"
    source.write_text("# Title\n\nBody paragraph for ingest test.\n", encoding="utf-8")
    assert (
        cmd_ingest(
            record_id="sample-md",
            file_path=source,
            corpus_root=tmp_path / "corpus",
            title="Sample",
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "ingested" in out.lower() or "unchanged" in out.lower()
    assert cmd_list(corpus_root=tmp_path / "corpus") == 0


def test_cli_corpus_list_integration(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    source = tmp_path / "doc.md"
    source.write_text("Hello corpus CLI.\n", encoding="utf-8")
    corpus_root = tmp_path / "corpus"
    cmd_ingest(record_id="cli-doc", file_path=source, corpus_root=corpus_root)
    capsys.readouterr()
    with patch("sys.argv", ["aurora-lens", "corpus", "list", "--corpus-root", str(corpus_root)]):
        # main reads argv — invoke via direct call
        pass
    assert main(["corpus", "list", "--corpus-root", str(corpus_root)]) == 0
    assert "cli-doc" in capsys.readouterr().out
