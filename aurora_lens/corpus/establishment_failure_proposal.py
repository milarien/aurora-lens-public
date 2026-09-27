"""Product Phase 3 — establishment-failure intake proposal models.

Intake proposes structured records for analyser/gate evaluation.
Proposals do not establish truth or decide admissibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from aurora_lens.corpus.ingestion_report import (
    ADMISSIBILITY_DECISION_NONE,
    EvidenceProvenance,
)
from aurora_lens.pef.uncertainty_analysis import EXTERNAL_ESTABLISHMENT_FAILURE_KINDS

PROPOSAL_STATUS_PROPOSED = "proposed"
PROPOSAL_STATUS_ADMITTED_FOR_EVALUATION = "admitted_for_evaluation"

ORIGIN_CANDIDATE_RECORD = "candidate_record"
ORIGIN_PLAIN_ENGLISH_INTAKE = "plain_english_intake"
ORIGIN_STRUCTURED_INTAKE = "structured_intake"


def make_proposal_id(prefix: str = "establishment-proposal") -> str:
    return f"{prefix}:{uuid4().hex[:16]}"


def normalize_establishment_failure_kind(raw: object) -> str | None:
    kind = str(raw or "").strip().lower()
    if kind in EXTERNAL_ESTABLISHMENT_FAILURE_KINDS:
        return kind
    return None


def normalize_bears_on(raw: object) -> tuple[str, ...]:
    if isinstance(raw, str):
        token = raw.strip()
        return (token,) if token else ()
    if isinstance(raw, (list, tuple)):
        out: list[str] = []
        for item in raw:
            token = str(item or "").strip()
            if token and token not in out:
                out.append(token)
        return tuple(out)
    return ()


@dataclass(frozen=True)
class EstablishmentFailureProposal:
    """Structured establishment-failure record proposed for downstream evaluation."""

    proposal_id: str
    kind: str
    description: str
    bears_on: tuple[str, ...]
    status: str
    origin: str
    establishes_truth: bool
    admissibility_decision: str
    meta: dict[str, Any] = field(default_factory=dict)
    source_candidate_id: str | None = None
    provenance: EvidenceProvenance | None = None
    intake_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "proposal_id": self.proposal_id,
            "kind": self.kind,
            "description": self.description,
            "bears_on": list(self.bears_on),
            "status": self.status,
            "origin": self.origin,
            "establishes_truth": self.establishes_truth,
            "admissibility_decision": self.admissibility_decision,
            "meta": dict(self.meta),
        }
        if self.source_candidate_id:
            payload["source_candidate_id"] = self.source_candidate_id
        if self.provenance is not None:
            payload["provenance"] = self.provenance.to_dict()
        if self.intake_ref:
            payload["intake_ref"] = self.intake_ref
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EstablishmentFailureProposal:
        kind = normalize_establishment_failure_kind(data.get("kind"))
        if kind is None:
            raise ValueError("kind must be a valid establishment-failure kind")
        description = str(data.get("description") or "").strip()
        if not description:
            raise ValueError("description is required")
        bears_on = normalize_bears_on(data.get("bears_on"))
        if not bears_on:
            raise ValueError("bears_on is required")
        proposal_id = str(data.get("proposal_id") or "").strip() or make_proposal_id()
        status = str(data.get("status") or "").strip()
        if status not in {PROPOSAL_STATUS_PROPOSED, PROPOSAL_STATUS_ADMITTED_FOR_EVALUATION}:
            raise ValueError("invalid proposal status")
        origin = str(data.get("origin") or "").strip()
        if not origin:
            raise ValueError("origin is required")
        if data.get("establishes_truth") is not False:
            raise ValueError("EstablishmentFailureProposal must not establish truth")
        admissibility = str(data.get("admissibility_decision") or "").strip()
        if admissibility != ADMISSIBILITY_DECISION_NONE:
            raise ValueError("EstablishmentFailureProposal must not carry admissibility decision")
        meta = data.get("meta") or {}
        if not isinstance(meta, dict):
            raise ValueError("meta must be a dict")
        provenance_raw = data.get("provenance")
        provenance = (
            EvidenceProvenance.from_dict(provenance_raw)
            if isinstance(provenance_raw, dict)
            else None
        )
        return cls(
            proposal_id=proposal_id,
            kind=kind,
            description=description,
            bears_on=bears_on,
            status=status,
            origin=origin,
            establishes_truth=False,
            admissibility_decision=ADMISSIBILITY_DECISION_NONE,
            meta=dict(meta),
            source_candidate_id=str(data.get("source_candidate_id") or "").strip() or None,
            provenance=provenance,
            intake_ref=str(data.get("intake_ref") or "").strip() or None,
        )

    def with_admitted_for_evaluation(self) -> EstablishmentFailureProposal:
        """Operator step: mark proposal ready for gate/analyser preview."""
        return EstablishmentFailureProposal(
            proposal_id=self.proposal_id,
            kind=self.kind,
            description=self.description,
            bears_on=self.bears_on,
            status=PROPOSAL_STATUS_ADMITTED_FOR_EVALUATION,
            origin=self.origin,
            establishes_truth=False,
            admissibility_decision=ADMISSIBILITY_DECISION_NONE,
            meta=dict(self.meta),
            source_candidate_id=self.source_candidate_id,
            provenance=self.provenance,
            intake_ref=self.intake_ref,
        )

    def to_open_epistemic_uncertainty(self) -> dict[str, Any]:
        """Gate/analyser payload — only when admitted for evaluation."""
        if self.status != PROPOSAL_STATUS_ADMITTED_FOR_EVALUATION:
            raise ValueError("proposal must be admitted_for_evaluation before gate payload")
        record: dict[str, Any] = {
            "id": self.proposal_id,
            "kind": self.kind,
            "description": self.description,
            "bears_on": list(self.bears_on),
            "status": "open",
        }
        if self.meta:
            record["meta"] = dict(self.meta)
        return record
