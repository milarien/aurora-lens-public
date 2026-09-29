"""Tests for aurora_lens.corpus retrieve and RAG message assembly (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aurora_lens.corpus.document_ingestion import ingest_file
from aurora_lens.corpus.models import CorpusChunk
from aurora_lens.corpus.registry import CorpusRegistry
from aurora_lens.corpus.retrieve import (
    assemble_rag_message,
    assemble_request_metadata,
    rank_chunks_by_query,
    retrieve_chunks,
)
from aurora_lens.lens import blocked_act_request_scan_text, split_rag_context_question
from aurora_lens.verify.blocked_request_policy import evaluate_blocked_act_request

_SAMPLE_MD = """\
### Section 1
Alpha about regulators and environment pressure.

### Section 2
Beta about time amplifier and structural load.
"""


@pytest.fixture
def ingested_registry(tmp_path: Path) -> CorpusRegistry:
    src = tmp_path / "doc.md"
    src.write_text(_SAMPLE_MD, encoding="utf-8")
    reg = CorpusRegistry(tmp_path / "corpus")
    ingest_file(reg, record_id="doc-1", source_path=src)
    return reg


class TestRankChunksByQuery:
    def test_prefers_matching_chunk(self):
        chunks = [
            CorpusChunk.build(record_id="r", ordinal=0, text="alpha plain", locator="A"),
            CorpusChunk.build(
                record_id="r",
                ordinal=1,
                text="regulators environment structural",
                locator="B",
            ),
        ]
        ranked = rank_chunks_by_query(chunks, "What are the five regulators?")
        assert ranked[0].ordinal == 1

    def test_phrase_boost_prefers_personal_baseline_chunk(self):
        chunks = [
            CorpusChunk.build(
                record_id="r",
                ordinal=0,
                text="Document scope and operational overview.",
                locator="early",
            ),
            CorpusChunk.build(
                record_id="r",
                ordinal=40,
                text=(
                    "The assessment adjusts all tier thresholds relative to the "
                    "individual's established personal baseline. Delta = PB - 50."
                ),
                locator="12.5 baseline",
            ),
        ]
        ranked = rank_chunks_by_query(
            chunks,
            "How does Personal Baseline change operational mode thresholds?",
        )
        assert ranked[0].ordinal == 40


class TestRetrieveChunks:
    def test_loads_with_max_chars(self, ingested_registry: CorpusRegistry):
        chunks = retrieve_chunks(
            ingested_registry,
            "doc-1",
            max_chars=5000,
            query="regulators",
        )
        assert chunks
        assert all(c.record_id == "doc-1" for c in chunks)

    def test_query_ranking_preserves_relevance_order(self, ingested_registry: CorpusRegistry):
        chunks = retrieve_chunks(
            ingested_registry,
            "doc-1",
            max_chars=5000,
            query="time amplifier structural load",
        )
        assert chunks[0].ordinal == 1

    def test_missing_record(self, ingested_registry: CorpusRegistry):
        with pytest.raises(KeyError):
            retrieve_chunks(ingested_registry, "missing")


class TestAssembleRagMessage:
    def test_split_rag_context_question(self, ingested_registry: CorpusRegistry):
        chunks = retrieve_chunks(ingested_registry, "doc-1", max_chars=5000)
        question = "What are the five regulators?"
        msg = assemble_rag_message(chunks, question)
        split = split_rag_context_question(msg)
        assert split is not None
        context_body, question_line = split
        assert question_line == question
        assert "### Section" in context_body
        assert "Insufficient context in retrieved chunks." in context_body

    def test_grounding_rules_in_context(self, ingested_registry: CorpusRegistry):
        chunks = retrieve_chunks(ingested_registry, "doc-1", max_chars=5000)
        msg = assemble_rag_message(chunks, "What are the five regulators?")
        assert msg.startswith("Context:\n")
        assert "Quote verbatim phrases" in msg

    def test_blocked_act_scans_question_only(self, ingested_registry: CorpusRegistry):
        chunks = retrieve_chunks(ingested_registry, "doc-1", max_chars=5000)
        question = "What are the five regulators?"
        msg = assemble_rag_message(chunks, question)
        assert not evaluate_blocked_act_request(blocked_act_request_scan_text(msg))

    def test_rejects_empty_question(self, ingested_registry: CorpusRegistry):
        chunks = retrieve_chunks(ingested_registry, "doc-1", max_chars=5000)
        with pytest.raises(ValueError, match="question"):
            assemble_rag_message(chunks, "  ")


class TestAssembleRequestMetadata:
    def test_record_ids_and_scope(self):
        meta = assemble_request_metadata(
            "test-doc-v1",
            source_scope=("corpus",),
            policy_profile="enterprise_strict",
        )
        assert meta["record_ids"] == ["test-doc-v1"]
        assert meta["source_scope"] == ["corpus"]
        assert meta["policy_profile"] == "enterprise_strict"
