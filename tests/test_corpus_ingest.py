"""Tests for aurora_lens.corpus ingest and review selection (Phase 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aurora_lens.corpus.document_ingestion import ingest_file
from aurora_lens.corpus.ingest import (
    build_chunks,
    chunk_source_text,
    read_source_text,
    segment_text_by_headings,
    sha256_file,
)
from aurora_lens.corpus.registry import CorpusRegistry
from aurora_lens.corpus.review import format_corpus_for_review, select_chunks


_SAMPLE_MD = """\
# Title

Intro paragraph.

### Section 1
Alpha body line one.
Alpha body line two.

### Section 2
Beta body only here.
"""


class TestSegmentTextByHeadings:
    def test_markdown_sections(self):
        segments = segment_text_by_headings(_SAMPLE_MD)
        assert len(segments) == 2
        assert segments[0][0].startswith("### Section 1")
        assert "Alpha body" in segments[0][1]
        assert "Beta body" in segments[1][1]


class TestChunkSourceText:
    def test_prefers_headings(self):
        segments = chunk_source_text(_SAMPLE_MD, max_chars=4000)
        assert len(segments) >= 2
        chunks = build_chunks("doc-1", segments)
        assert chunks[0].chunk_id == "doc-1--chunk-0000"
        assert chunks[0].record_id == "doc-1"

    def test_size_fallback(self):
        plain = "word " * 5000
        segments = chunk_source_text(plain, max_chars=1000, overlap=50)
        assert len(segments) > 1
        assert segments[0][0] == "Block 1"


class TestIngestFile:
    def test_ingest_markdown_round_trip(self, tmp_path: Path):
        src = tmp_path / "sample.md"
        src.write_text(_SAMPLE_MD, encoding="utf-8")
        reg = CorpusRegistry(tmp_path / "corpus")
        record, chunks, wrote = ingest_file(
            reg,
            record_id="sample-doc",
            source_path=src,
            title="Sample",
        )
        assert wrote is True
        assert record.chunk_count == len(chunks)
        assert record.sha256 == sha256_file(src)
        assert reg.get_record("sample-doc") == record
        assert reg.load_chunks("sample-doc") == chunks
        meta = reg.record_dir("sample-doc") / "source.meta.json"
        assert meta.is_file()

    def test_skips_when_sha256_unchanged(self, tmp_path: Path):
        src = tmp_path / "sample.md"
        src.write_text(_SAMPLE_MD, encoding="utf-8")
        reg = CorpusRegistry(tmp_path / "corpus")
        ingest_file(reg, record_id="sample-doc", source_path=src)
        record2, chunks2, wrote2 = ingest_file(
            reg, record_id="sample-doc", source_path=src
        )
        assert wrote2 is False
        assert record2.chunk_count == len(chunks2)

    def test_force_reingest(self, tmp_path: Path):
        src = tmp_path / "sample.md"
        src.write_text(_SAMPLE_MD, encoding="utf-8")
        reg = CorpusRegistry(tmp_path / "corpus")
        ingest_file(reg, record_id="sample-doc", source_path=src)
        src.write_text(_SAMPLE_MD + "\n\n### Section 3\nExtra.\n", encoding="utf-8")
        _, chunks, wrote = ingest_file(
            reg, record_id="sample-doc", source_path=src, force=True
        )
        assert wrote is True
        assert len(chunks) == 3


class TestReadSourceText:
    def test_reads_markdown(self, tmp_path: Path):
        p = tmp_path / "x.md"
        p.write_text("hello", encoding="utf-8")
        text, meta = read_source_text(p)
        assert text == "hello"
        assert meta["extractor"] == "text"


class TestReviewSelection:
    def test_select_by_ordinal(self):
        from aurora_lens.corpus.models import CorpusChunk

        chunks = [
            CorpusChunk.build(record_id="r", ordinal=0, text="aaa", locator="A"),
            CorpusChunk.build(record_id="r", ordinal=1, text="bbb", locator="B"),
            CorpusChunk.build(record_id="r", ordinal=2, text="ccc", locator="C"),
        ]
        sel = select_chunks(chunks, chunk_ordinals={0, 2})
        assert [c.ordinal for c in sel] == [0, 2]

    def test_max_chars_budget(self):
        from aurora_lens.corpus.models import CorpusChunk

        chunks = [
            CorpusChunk.build(record_id="r", ordinal=0, text="a" * 100, locator="A"),
            CorpusChunk.build(record_id="r", ordinal=1, text="b" * 100, locator="B"),
            CorpusChunk.build(record_id="r", ordinal=2, text="c" * 100, locator="C"),
        ]
        sel = select_chunks(chunks, max_chars=150)
        assert len(sel) == 1
        assert sel[0].ordinal == 0

    def test_format_corpus_for_review(self):
        from aurora_lens.corpus.models import CorpusChunk

        ch = CorpusChunk.build(record_id="r", ordinal=0, text="body", locator="Loc 1")
        out = format_corpus_for_review([ch])
        assert "### Loc 1" in out
        assert "body" in out

    def test_select_empty_raises(self):
        from aurora_lens.corpus.models import CorpusChunk

        chunks = [CorpusChunk.build(record_id="r", ordinal=0, text="x", locator="A")]
        with pytest.raises(ValueError, match="no chunks"):
            select_chunks(chunks, chunk_ordinals={9})
