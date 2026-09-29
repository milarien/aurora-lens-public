"""Product Phase 3 document ingestion — proposal-only entrypoint tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aurora_lens.corpus.document_ingestion import (
    corpus_chunks_from_report,
    ingest_file,
    propose_document_ingestion,
)
from aurora_lens.corpus.ingestion_report import (
    ADMISSIBILITY_DECISION_NONE,
    CANDIDATE_STATUS_PROPOSED,
    PROPOSAL_STATUS_PROPOSED_ONLY,
    IngestionReport,
)
from aurora_lens.corpus.ingest_metadata import IngestMetadata
from aurora_lens.corpus.registry import CorpusRegistry

_SAMPLE_MD = """\
# Title

Intro paragraph.

### Section 1
Alpha body line one.

### Section 2
Beta body only here.
"""


class TestProposeDocumentIngestion:
    def test_returns_provenance_carrying_report(self, tmp_path: Path):
        src = tmp_path / "sample.md"
        src.write_text(_SAMPLE_MD, encoding="utf-8")
        report = propose_document_ingestion(src, record_id="sample-doc", title="Sample")
        assert report.proposal_status == PROPOSAL_STATUS_PROPOSED_ONLY
        assert report.establishes_truth is False
        assert report.admissibility_decision == ADMISSIBILITY_DECISION_NONE
        assert report.candidates
        assert report.report_id.startswith("ingestion-report:sample-doc:")
        for candidate in report.candidates:
            assert candidate.status == CANDIDATE_STATUS_PROPOSED
            assert candidate.provenance.source_sha256 == report.source_sha256
            assert candidate.provenance.chunk_id
            assert candidate.provenance.record_id == "sample-doc"
            assert candidate.text.strip()

    def test_report_round_trip_json(self, tmp_path: Path):
        src = tmp_path / "sample.md"
        src.write_text(_SAMPLE_MD, encoding="utf-8")
        report = propose_document_ingestion(src, record_id="sample-doc")
        restored = IngestionReport.from_dict(json.loads(json.dumps(report.to_dict())))
        assert restored.report_id == report.report_id
        assert len(restored.candidates) == len(report.candidates)
        assert restored.establishes_truth is False

    def test_candidates_map_to_corpus_chunks_without_pef(self, tmp_path: Path):
        src = tmp_path / "sample.md"
        src.write_text(_SAMPLE_MD, encoding="utf-8")
        report = propose_document_ingestion(src, record_id="sample-doc")
        chunks = corpus_chunks_from_report(report)
        assert len(chunks) == len(report.candidates)
        assert chunks[0].chunk_id == report.candidates[0].provenance.chunk_id

    def test_module_does_not_import_lens_or_pef(self):
        import aurora_lens.corpus.document_ingestion as mod

        source = open(mod.__file__, encoding="utf-8").read()
        assert "aurora_lens.lens" not in source
        assert "aurora_lens.pef" not in source

    def test_ingest_file_still_persists_via_proposal_path(self, tmp_path: Path):
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
        report = propose_document_ingestion(src, record_id="sample-doc", title="Sample")
        assert record.chunk_count == len(report.candidates)

    def test_ingest_file_attaches_valid_ingest_metadata(self, tmp_path: Path):
        src = tmp_path / "sample.md"
        src.write_text(_SAMPLE_MD, encoding="utf-8")
        reg = CorpusRegistry(tmp_path / "corpus")
        metadata = IngestMetadata.from_dict(
            {
                "record_id": "sample-doc",
                "status": "approved",
                "authority": "corporate_policy",
                "scope": "travel",
                "effective_from": "2026-01-01",
                "effective_to": "2026-12-31",
                "version": "2.0",
                "source_ref": "policy://travel/v2",
            }
        )
        record, _chunks, _wrote = ingest_file(
            reg,
            record_id="sample-doc",
            source_path=src,
            title="Sample",
            ingest_metadata=metadata,
        )
        assert record.status == "approved"
        assert record.authority_class == "corporate_policy"
        assert record.scope == "travel"
        assert record.effective_from == "2026-01-01"
        assert record.effective_until == "2026-12-31"
        assert record.version == "2.0"
        assert record.source_ref == "policy://travel/v2"

    def test_ingest_file_rejects_mismatched_ingest_metadata_record(self, tmp_path: Path):
        src = tmp_path / "sample.md"
        src.write_text(_SAMPLE_MD, encoding="utf-8")
        reg = CorpusRegistry(tmp_path / "corpus")
        metadata = IngestMetadata.from_dict(
            {
                "record_id": "wrong-doc",
                "status": "approved",
                "authority": "corporate_policy",
                "scope": "travel",
                "effective_from": "2026-01-01",
                "effective_to": "2026-12-31",
                "version": "2.0",
                "source_ref": "policy://travel/v2",
            }
        )
        with pytest.raises(ValueError, match="record_id"):
            ingest_file(
                reg,
                record_id="sample-doc",
                source_path=src,
                title="Sample",
                ingest_metadata=metadata,
            )

    def test_from_dict_rejects_truth_establishment(self, tmp_path: Path):
        src = tmp_path / "sample.md"
        src.write_text("x", encoding="utf-8")
        report = propose_document_ingestion(src, record_id="doc")
        payload = report.to_dict()
        payload["establishes_truth"] = True
        with pytest.raises(ValueError, match="establish truth"):
            IngestionReport.from_dict(payload)
