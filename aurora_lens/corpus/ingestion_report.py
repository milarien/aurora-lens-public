"""Product Phase 3 — provenance-carrying ingestion report models.

Ingestion proposes candidate records. Nothing here establishes truth or admissibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PROPOSAL_STATUS_PROPOSED_ONLY = "proposed_only"
CANDIDATE_STATUS_PROPOSED = "proposed"
CANDIDATE_KIND_EVIDENCE_SPAN = "evidence_span"
ADMISSIBILITY_DECISION_NONE = "none"


def make_candidate_id(record_id: str, ordinal: int) -> str:
    rid = (record_id or "").strip()
    if not rid:
        raise ValueError("record_id must be non-empty")
    if ordinal < 0:
        raise ValueError("ordinal must be >= 0")
    return f"{rid}--candidate-{ordinal:04d}"


def make_ingestion_report_id(record_id: str, source_sha256: str) -> str:
    rid = (record_id or "").strip()
    digest = (source_sha256 or "").strip().lower()
    if not rid or not digest:
        raise ValueError("record_id and source_sha256 are required")
    return f"ingestion-report:{rid}:{digest[:16]}"


@dataclass(frozen=True)
class EvidenceProvenance:
    """Provenance chain for one proposed evidence span."""

    record_id: str
    source_path: str
    source_sha256: str
    chunk_id: str
    locator: str
    extracted_at: str
    extractor: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "chunk_id": self.chunk_id,
            "locator": self.locator,
            "extracted_at": self.extracted_at,
            "extractor": self.extractor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceProvenance:
        record_id = str(data.get("record_id", "")).strip()
        source_path = str(data.get("source_path", "")).strip()
        source_sha256 = str(data.get("source_sha256", "")).strip().lower()
        chunk_id = str(data.get("chunk_id", "")).strip()
        locator = str(data.get("locator", "")).strip()
        extracted_at = str(data.get("extracted_at", "")).strip()
        extractor = str(data.get("extractor", "")).strip()
        if not all([record_id, source_sha256, chunk_id, extracted_at, extractor]):
            raise ValueError("EvidenceProvenance requires record_id, source_sha256, chunk_id, extracted_at, extractor")
        return cls(
            record_id=record_id,
            source_path=source_path,
            source_sha256=source_sha256,
            chunk_id=chunk_id,
            locator=locator,
            extracted_at=extracted_at,
            extractor=extractor,
        )


@dataclass(frozen=True)
class CandidateRecord:
    """One proposed record — not committed truth."""

    candidate_id: str
    kind: str
    status: str
    text: str
    provenance: EvidenceProvenance
    bears_on: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "status": self.status,
            "text": self.text,
            "provenance": self.provenance.to_dict(),
            "bears_on": list(self.bears_on),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateRecord:
        candidate_id = str(data.get("candidate_id", "")).strip()
        kind = str(data.get("kind", "")).strip()
        status = str(data.get("status", "")).strip()
        text = data.get("text", "")
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if not candidate_id or not kind or not status:
            raise ValueError("candidate_id, kind, and status are required")
        provenance = EvidenceProvenance.from_dict(data.get("provenance") or {})
        bears_raw = data.get("bears_on") or []
        if not isinstance(bears_raw, list):
            raise ValueError("bears_on must be a list")
        meta = data.get("meta") or {}
        if not isinstance(meta, dict):
            raise ValueError("meta must be a dict")
        return cls(
            candidate_id=candidate_id,
            kind=kind,
            status=status,
            text=text,
            provenance=provenance,
            bears_on=tuple(str(b).strip() for b in bears_raw if str(b).strip()),
            meta=dict(meta),
        )


@dataclass(frozen=True)
class IngestionReport:
    """Provenance-carrying ingestion output — proposals only."""

    report_id: str
    record_id: str
    title: str
    source_path: str
    source_sha256: str
    ingested_at: str
    extractor: str
    proposal_status: str
    establishes_truth: bool
    admissibility_decision: str
    candidates: tuple[CandidateRecord, ...]
    source_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "record_id": self.record_id,
            "title": self.title,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "ingested_at": self.ingested_at,
            "extractor": self.extractor,
            "proposal_status": self.proposal_status,
            "establishes_truth": self.establishes_truth,
            "admissibility_decision": self.admissibility_decision,
            "candidates": [c.to_dict() for c in self.candidates],
            "source_meta": dict(self.source_meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestionReport:
        report_id = str(data.get("report_id", "")).strip()
        record_id = str(data.get("record_id", "")).strip()
        if not report_id or not record_id:
            raise ValueError("report_id and record_id are required")
        candidates_raw = data.get("candidates") or []
        if not isinstance(candidates_raw, list):
            raise ValueError("candidates must be a list")
        candidates = tuple(CandidateRecord.from_dict(c) for c in candidates_raw)
        source_meta = data.get("source_meta") or {}
        if not isinstance(source_meta, dict):
            raise ValueError("source_meta must be a dict")
        establishes_truth = data.get("establishes_truth", False)
        if establishes_truth is not False:
            raise ValueError("IngestionReport must not establish truth")
        admissibility = str(data.get("admissibility_decision", "")).strip()
        if admissibility != ADMISSIBILITY_DECISION_NONE:
            raise ValueError("IngestionReport must not carry an admissibility decision")
        return cls(
            report_id=report_id,
            record_id=record_id,
            title=str(data.get("title", "")).strip() or record_id,
            source_path=str(data.get("source_path", "")).strip(),
            source_sha256=str(data.get("source_sha256", "")).strip().lower(),
            ingested_at=str(data.get("ingested_at", "")).strip(),
            extractor=str(data.get("extractor", "")).strip(),
            proposal_status=str(data.get("proposal_status", "")).strip(),
            establishes_truth=False,
            admissibility_decision=ADMISSIBILITY_DECISION_NONE,
            candidates=candidates,
            source_meta=dict(source_meta),
        )
