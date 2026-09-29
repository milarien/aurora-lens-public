"""Tests for aurora_lens.corpus registry and models (Phase 0)."""

from __future__ import annotations

import json

import pytest

from aurora_lens.corpus import (
    CorpusChunk,
    CorpusRecord,
    CorpusRegistry,
    make_chunk_id,
)
from aurora_lens.corpus.models import utc_now_iso


class TestMakeChunkId:
    def test_stable_format(self):
        assert make_chunk_id("test-doc-v1", 0) == "test-doc-v1--chunk-0000"
        assert make_chunk_id("test-doc-v1", 12) == "test-doc-v1--chunk-0012"

    def test_rejects_empty_record_id(self):
        with pytest.raises(ValueError, match="record_id"):
            make_chunk_id("", 0)

    def test_rejects_negative_ordinal(self):
        with pytest.raises(ValueError, match="ordinal"):
            make_chunk_id("doc", -1)


class TestCorpusRecordWire:
    def test_round_trip(self):
        rec = CorpusRecord(
            record_id="test-doc-v1",
            title="Test Document",
            source_path="data/test-doc.pdf",
            sha256="abc123",
            ingested_at=utc_now_iso(),
            chunk_count=3,
        )
        restored = CorpusRecord.from_dict(rec.to_dict())
        assert restored == rec

    def test_from_dict_back_compat_without_governance_metadata(self):
        legacy = {
            "record_id": "legacy-policy",
            "title": "Legacy Policy",
            "source_path": "docs/legacy.md",
            "sha256": "abc123",
            "ingested_at": "2026-06-29T00:00:00+00:00",
            "chunk_count": 1,
        }
        rec = CorpusRecord.from_dict(legacy)
        assert rec.authority_class is None
        assert rec.status is None
        assert rec.supersedes == []
        assert rec.superseded_by == []
        assert rec.allowed_roles == []
        assert rec.source_ref is None

    def test_round_trip_with_governance_metadata(self):
        rec = CorpusRecord(
            record_id="policy-v2",
            title="Policy V2",
            source_path="docs/policy-v2.md",
            sha256="feedbeef",
            ingested_at=utc_now_iso(),
            chunk_count=3,
            authority_class="corporate_policy",
            status="approved",
            effective_from="2026-01-01",
            effective_until="2026-12-31",
            review_due="2026-10-01",
            jurisdiction="AU",
            department="Finance",
            scope="travel",
            version="2.0",
            supersedes=["policy-v1"],
            superseded_by=[],
            confidentiality="internal",
            allowed_roles=["manager", "finance"],
            parser="text",
            parser_status="ok",
            source_ref="policy://finance/travel/v2",
        )
        restored = CorpusRecord.from_dict(rec.to_dict())
        assert restored == rec

    def test_list_defaults_are_distinct_and_safe(self):
        rec1 = CorpusRecord(
            record_id="a",
            title="A",
            source_path="a.md",
            sha256="a",
            ingested_at=utc_now_iso(),
            chunk_count=0,
        )
        rec2 = CorpusRecord(
            record_id="b",
            title="B",
            source_path="b.md",
            sha256="b",
            ingested_at=utc_now_iso(),
            chunk_count=0,
        )
        assert rec1.supersedes == []
        assert rec2.supersedes == []
        assert rec1.supersedes is not rec2.supersedes
        assert rec1.allowed_roles is not rec2.allowed_roles


class TestCorpusChunkWire:
    def test_round_trip(self):
        ch = CorpusChunk.build(
            record_id="test-doc-v1",
            ordinal=1,
            locator="Section 2",
            text="Environment captures pressure.",
        )
        restored = CorpusChunk.from_dict(ch.to_dict())
        assert restored == ch

    def test_build_assigns_chunk_id(self):
        ch = CorpusChunk.build(record_id="doc-a", ordinal=0, text="body")
        assert ch.chunk_id == "doc-a--chunk-0000"

    def test_from_dict_rejects_chunk_id_drift(self):
        data = CorpusChunk.build(record_id="doc-a", ordinal=0, text="x").to_dict()
        data["chunk_id"] = "wrong-id"
        with pytest.raises(ValueError, match="chunk_id"):
            CorpusChunk.from_dict(data)


class TestCorpusRegistry:
    def test_empty_registry(self, tmp_path):
        reg = CorpusRegistry(tmp_path)
        assert reg.load_records() == {}
        assert reg.load_chunks("missing") == []

    def test_upsert_and_load_record(self, tmp_path):
        reg = CorpusRegistry(tmp_path)
        rec = CorpusRecord(
            record_id="test-doc-v1",
            title="Test Document",
            source_path="data/doc.pdf",
            sha256="deadbeef",
            ingested_at="2026-05-19T12:00:00+00:00",
            chunk_count=2,
        )
        reg.upsert_record(rec)
        assert reg.get_record("test-doc-v1") == rec
        raw = json.loads(reg.registry_path.read_text(encoding="utf-8"))
        assert raw["version"] == 1
        assert len(raw["records"]) == 1

    def test_save_and_load_chunks(self, tmp_path):
        reg = CorpusRegistry(tmp_path)
        chunks = [
            CorpusChunk.build(
                record_id="doc-1",
                ordinal=0,
                locator="Section 1",
                text="First chunk.",
            ),
            CorpusChunk.build(
                record_id="doc-1",
                ordinal=1,
                locator="Section 2",
                text="Second chunk.",
            ),
        ]
        reg.save_chunks("doc-1", chunks)
        loaded = reg.load_chunks("doc-1")
        assert loaded == chunks
        assert reg.chunks_path("doc-1").is_file()

    def test_chunk_ordinals_must_be_contiguous_on_save(self, tmp_path):
        reg = CorpusRegistry(tmp_path)
        bad = [
            CorpusChunk.build(record_id="doc-1", ordinal=0, text="a"),
            CorpusChunk.build(record_id="doc-1", ordinal=2, text="c"),
        ]
        with pytest.raises(ValueError, match="contiguous"):
            reg.save_chunks("doc-1", bad)

    def test_chunk_record_id_mismatch_on_save(self, tmp_path):
        reg = CorpusRegistry(tmp_path)
        ch = CorpusChunk.build(record_id="other", ordinal=0, text="a")
        with pytest.raises(ValueError, match="record_id"):
            reg.save_chunks("doc-1", [ch])

    def test_delete_record(self, tmp_path):
        reg = CorpusRegistry(tmp_path)
        rec = CorpusRecord(
            record_id="x",
            title="x",
            source_path="",
            sha256="",
            ingested_at=utc_now_iso(),
            chunk_count=0,
        )
        reg.upsert_record(rec)
        assert reg.delete_record("x") is True
        assert reg.get_record("x") is None
        assert reg.delete_record("x") is False

    def test_registry_sorted_on_save(self, tmp_path):
        reg = CorpusRegistry(tmp_path)
        reg.upsert_record(
            CorpusRecord(
                record_id="b-record",
                title="B",
                source_path="",
                sha256="1",
                ingested_at=utc_now_iso(),
                chunk_count=0,
            )
        )
        reg.upsert_record(
            CorpusRecord(
                record_id="a-record",
                title="A",
                source_path="",
                sha256="2",
                ingested_at=utc_now_iso(),
                chunk_count=0,
            )
        )
        ids = [r["record_id"] for r in json.loads(reg.registry_path.read_text())["records"]]
        assert ids == ["a-record", "b-record"]
