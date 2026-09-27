"""Product Phase 3 document ingestion — propose candidate records only.

Ingestion proposes. Lens disposes.

This module converts a source document into a provenance-carrying ``IngestionReport``.
It does not mutate PEF, import Lens, or decide admissibility.
"""

from __future__ import annotations

from pathlib import Path

from aurora_lens.corpus.ingest_metadata import IngestMetadata
from aurora_lens.corpus.ingest import (
    build_chunks,
    chunk_source_text,
    read_source_text,
    sha256_file,
    write_source_meta,
)
from aurora_lens.corpus.ingestion_report import (
    ADMISSIBILITY_DECISION_NONE,
    CANDIDATE_KIND_EVIDENCE_SPAN,
    CANDIDATE_STATUS_PROPOSED,
    PROPOSAL_STATUS_PROPOSED_ONLY,
    CandidateRecord,
    EvidenceProvenance,
    IngestionReport,
    make_candidate_id,
    make_ingestion_report_id,
)
from aurora_lens.corpus.models import CorpusChunk, CorpusRecord, make_chunk_id, utc_now_iso
from aurora_lens.corpus.registry import CorpusRegistry

_DEFAULT_MAX_CHUNK_CHARS = 4000
_DEFAULT_OVERLAP_CHARS = 200


def _candidate_from_chunk(
    chunk: CorpusChunk,
    *,
    source_path: str,
    source_sha256: str,
    extracted_at: str,
    extractor: str,
) -> CandidateRecord:
    provenance = EvidenceProvenance(
        record_id=chunk.record_id,
        source_path=source_path,
        source_sha256=source_sha256,
        chunk_id=chunk.chunk_id,
        locator=chunk.locator,
        extracted_at=extracted_at,
        extractor=extractor,
    )
    return CandidateRecord(
        candidate_id=make_candidate_id(chunk.record_id, chunk.ordinal),
        kind=CANDIDATE_KIND_EVIDENCE_SPAN,
        status=CANDIDATE_STATUS_PROPOSED,
        text=chunk.text,
        provenance=provenance,
        bears_on=(),
        meta={"ordinal": chunk.ordinal},
    )


def propose_document_ingestion(
    source_path: Path,
    *,
    record_id: str,
    title: str | None = None,
    max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
    overlap: int = _DEFAULT_OVERLAP_CHARS,
) -> IngestionReport:
    """Convert a document into a provenance-carrying ingestion report (proposals only)."""
    rid = record_id.strip()
    if not rid:
        raise ValueError("record_id must be non-empty")
    src = source_path.resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    body, meta = read_source_text(src)
    digest = sha256_file(src)
    extracted_at = utc_now_iso()
    extractor = str(meta.get("extractor") or "unknown")
    segments = chunk_source_text(body, max_chars=max_chunk_chars, overlap=overlap)
    chunks = build_chunks(rid, segments)
    candidates = tuple(
        _candidate_from_chunk(
            chunk,
            source_path=str(src),
            source_sha256=digest,
            extracted_at=extracted_at,
            extractor=extractor,
        )
        for chunk in chunks
    )
    return IngestionReport(
        report_id=make_ingestion_report_id(rid, digest),
        record_id=rid,
        title=(title or rid).strip(),
        source_path=str(src),
        source_sha256=digest,
        ingested_at=extracted_at,
        extractor=extractor,
        proposal_status=PROPOSAL_STATUS_PROPOSED_ONLY,
        establishes_truth=False,
        admissibility_decision=ADMISSIBILITY_DECISION_NONE,
        candidates=candidates,
        source_meta=dict(meta),
    )


def corpus_record_from_report(report: IngestionReport) -> CorpusRecord:
    """Map an ingestion report to corpus storage metadata (persistence layer)."""
    return CorpusRecord(
        record_id=report.record_id,
        title=report.title,
        source_path=report.source_path,
        sha256=report.source_sha256,
        ingested_at=report.ingested_at,
        chunk_count=len(report.candidates),
    )


def corpus_chunks_from_report(report: IngestionReport) -> list[CorpusChunk]:
    """Map proposed candidates to corpus chunks for optional registry persistence."""
    chunks: list[CorpusChunk] = []
    for candidate in report.candidates:
        ordinal = int(candidate.meta.get("ordinal", len(chunks)))
        chunk_id = candidate.provenance.chunk_id
        expected = make_chunk_id(report.record_id, ordinal)
        if chunk_id != expected:
            chunk_id = expected
        chunks.append(
            CorpusChunk(
                chunk_id=chunk_id,
                record_id=report.record_id,
                locator=candidate.provenance.locator,
                text=candidate.text,
                ordinal=ordinal,
            )
        )
    return chunks


def persist_ingestion_report(
    registry: CorpusRegistry,
    report: IngestionReport,
    *,
    source_path: Path,
    ingest_metadata: IngestMetadata | None = None,
    force: bool = False,
) -> tuple[CorpusRecord, list[CorpusChunk], bool]:
    """Optional operator step: persist proposed spans to the corpus registry."""
    existing = registry.get_record(report.record_id)
    if existing is not None and existing.sha256 == report.source_sha256 and not force:
        return existing, registry.load_chunks(report.record_id), False

    record = corpus_record_from_report(report)
    if ingest_metadata is not None:
        record = ingest_metadata.apply_to_record(record)
    chunks = corpus_chunks_from_report(report)
    registry.upsert_record(record)
    registry.save_chunks(report.record_id, chunks)
    write_source_meta(
        registry.record_dir(report.record_id),
        source_path=source_path.resolve(),
        sha256=report.source_sha256,
        meta=report.source_meta,
    )
    return record, chunks, True


def ingest_file(
    registry: CorpusRegistry,
    *,
    record_id: str,
    source_path: Path,
    title: str | None = None,
    ingest_metadata: IngestMetadata | None = None,
    force: bool = False,
    max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
    overlap: int = _DEFAULT_OVERLAP_CHARS,
) -> tuple[CorpusRecord, list[CorpusChunk], bool]:
    """Propose document ingestion, then optionally persist to corpus registry."""
    src = source_path.resolve()
    report = propose_document_ingestion(
        src,
        record_id=record_id,
        title=title,
        max_chunk_chars=max_chunk_chars,
        overlap=overlap,
    )
    return persist_ingestion_report(
        registry,
        report,
        source_path=src,
        ingest_metadata=ingest_metadata,
        force=force,
    )
