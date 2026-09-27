"""Product Phase 3 intake translator — proposals for analyser/gate evaluation.

Bridge:

  CandidateRecord / plain-English / structured intake
          ↓
  EstablishmentFailureProposal
          ↓
  existing analyser / gates (after explicit admission)

Intake proposes. Lens disposes. No PEF mutation in this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from aurora_lens.corpus.establishment_failure_proposal import (
    ORIGIN_CANDIDATE_RECORD,
    ORIGIN_PLAIN_ENGLISH_INTAKE,
    ORIGIN_STRUCTURED_INTAKE,
    PROPOSAL_STATUS_PROPOSED,
    EstablishmentFailureProposal,
    make_proposal_id,
    normalize_bears_on,
    normalize_establishment_failure_kind,
)
from aurora_lens.corpus.ingestion_report import ADMISSIBILITY_DECISION_NONE, CandidateRecord
from aurora_lens.govern.epistemic_uncertainty_gate import (
    EpistemicUncertaintyGateResult,
    evaluate_epistemic_uncertainty_gate,
    parse_open_epistemic_uncertainties,
)

_LABELED_ASSERTION_RE = re.compile(
    r"^(?P<kind>[a-z_]+)\s*:\s*(?P<description>.+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IntakeTranslationBatch:
    """One intake translation pass — proposals only."""

    proposals: tuple[EstablishmentFailureProposal, ...]
    establishes_truth: bool = False
    admissibility_decision: str = ADMISSIBILITY_DECISION_NONE


def _base_proposal(
    *,
    kind: str,
    description: str,
    bears_on: tuple[str, ...],
    origin: str,
    meta: dict[str, Any] | None = None,
    source_candidate_id: str | None = None,
    provenance: Any = None,
    intake_ref: str | None = None,
    proposal_id: str | None = None,
) -> EstablishmentFailureProposal:
    normalized_kind = normalize_establishment_failure_kind(kind)
    if normalized_kind is None:
        raise ValueError(f"invalid establishment-failure kind: {kind!r}")
    desc = str(description or "").strip()
    if not desc:
        raise ValueError("description is required")
    targets = normalize_bears_on(bears_on)
    if not targets:
        raise ValueError("bears_on is required")
    return EstablishmentFailureProposal(
        proposal_id=proposal_id or make_proposal_id(),
        kind=normalized_kind,
        description=desc,
        bears_on=targets,
        status=PROPOSAL_STATUS_PROPOSED,
        origin=origin,
        establishes_truth=False,
        admissibility_decision=ADMISSIBILITY_DECISION_NONE,
        meta=dict(meta or {}),
        source_candidate_id=source_candidate_id,
        provenance=provenance,
        intake_ref=intake_ref,
    )


def translate_structured_intake(item: dict[str, Any]) -> EstablishmentFailureProposal:
    """Translate explicit structured intake dict to a proposal."""
    if not isinstance(item, dict):
        raise ValueError("structured intake item must be a dict")
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    return _base_proposal(
        kind=str(item.get("kind") or ""),
        description=str(item.get("description") or ""),
        bears_on=normalize_bears_on(item.get("bears_on")),
        origin=ORIGIN_STRUCTURED_INTAKE,
        meta=meta,
        intake_ref=str(item.get("intake_ref") or "").strip() or None,
        proposal_id=str(item.get("proposal_id") or "").strip() or None,
    )


def parse_labeled_plain_english_assertion(text: str) -> tuple[str, str]:
    """Parse ``kind: description`` when kind is an exact taxonomy token (lexical only)."""
    raw = str(text or "").strip()
    match = _LABELED_ASSERTION_RE.match(raw)
    if not match:
        raise ValueError("plain-English intake must use labeled form: kind: description")
    kind = normalize_establishment_failure_kind(match.group("kind"))
    if kind is None:
        raise ValueError(f"unknown establishment-failure kind label: {match.group('kind')!r}")
    description = str(match.group("description") or "").strip()
    if not description:
        raise ValueError("description is required after kind label")
    return kind, description


def translate_plain_english_intake(
    assertion: str,
    *,
    bears_on: list[str] | tuple[str, ...],
    meta: dict[str, Any] | None = None,
    intake_ref: str | None = None,
) -> EstablishmentFailureProposal:
    """Translate labeled plain-English assertion to a proposal (no semantic inference)."""
    kind, description = parse_labeled_plain_english_assertion(assertion)
    return _base_proposal(
        kind=kind,
        description=description,
        bears_on=normalize_bears_on(bears_on),
        origin=ORIGIN_PLAIN_ENGLISH_INTAKE,
        meta=meta,
        intake_ref=intake_ref,
    )


def translate_candidate_record(
    candidate: CandidateRecord,
    *,
    kind: str | None = None,
    description: str | None = None,
    bears_on: list[str] | tuple[str, ...] | None = None,
    meta: dict[str, Any] | None = None,
) -> EstablishmentFailureProposal:
    """Translate selected ``CandidateRecord`` plus explicit intake fields to a proposal."""
    if candidate.status != "proposed":
        raise ValueError("candidate must remain proposed; ingestion does not establish truth")

    cm = candidate.meta if isinstance(candidate.meta, dict) else {}
    resolved_kind = kind or cm.get("intake_kind") or cm.get("establishment_failure_kind")
    resolved_description = (
        description or cm.get("intake_description") or cm.get("establishment_failure_description")
    )
    resolved_bears_on = bears_on if bears_on is not None else cm.get("intake_bears_on") or candidate.bears_on
    merged_meta = dict(cm)
    if meta:
        merged_meta.update(meta)
    merged_meta.setdefault("source_chunk_id", candidate.provenance.chunk_id)
    merged_meta.setdefault("source_locator", candidate.provenance.locator)

    return _base_proposal(
        kind=str(resolved_kind or ""),
        description=str(resolved_description or ""),
        bears_on=normalize_bears_on(resolved_bears_on),
        origin=ORIGIN_CANDIDATE_RECORD,
        meta=merged_meta,
        source_candidate_id=candidate.candidate_id,
        provenance=candidate.provenance,
    )


def translate_intake_batch(
    *,
    structured: list[dict[str, Any]] | None = None,
    plain_english: list[dict[str, Any]] | None = None,
    candidates: list[tuple[CandidateRecord, dict[str, Any] | None]] | None = None,
) -> IntakeTranslationBatch:
    """Translate multiple intake sources into one proposal batch."""
    proposals: list[EstablishmentFailureProposal] = []
    for item in structured or []:
        proposals.append(translate_structured_intake(item))
    for item in plain_english or []:
        proposals.append(
            translate_plain_english_intake(
                str(item.get("assertion") or ""),
                bears_on=item.get("bears_on") or [],
                meta=item.get("meta") if isinstance(item.get("meta"), dict) else None,
                intake_ref=str(item.get("intake_ref") or "").strip() or None,
            )
        )
    for entry in candidates or []:
        candidate, overrides = entry
        ov = overrides or {}
        proposals.append(
            translate_candidate_record(
                candidate,
                kind=ov.get("kind"),
                description=ov.get("description"),
                bears_on=ov.get("bears_on"),
                meta=ov.get("meta") if isinstance(ov.get("meta"), dict) else None,
            )
        )
    return IntakeTranslationBatch(proposals=tuple(proposals))


def admit_proposals_for_evaluation(
    proposals: list[EstablishmentFailureProposal] | tuple[EstablishmentFailureProposal, ...],
) -> tuple[EstablishmentFailureProposal, ...]:
    """Operator step: mark proposals admitted for gate/analyser preview."""
    return tuple(p.with_admitted_for_evaluation() for p in proposals)


def proposals_to_gate_uncertainties(
    proposals: list[EstablishmentFailureProposal] | tuple[EstablishmentFailureProposal, ...],
) -> list[dict[str, Any]]:
    """Convert admitted proposals to gate-ready open epistemic uncertainty dicts."""
    raw = [p.to_open_epistemic_uncertainty() for p in proposals]
    return parse_open_epistemic_uncertainties(raw)


def preview_gate_evaluation(
    proposals: list[EstablishmentFailureProposal] | tuple[EstablishmentFailureProposal, ...],
    user_text: str,
    *,
    admit: bool = True,
) -> EpistemicUncertaintyGateResult | None:
    """Preview analyser/gate effect for admitted proposals (no PEF write)."""
    admitted = admit_proposals_for_evaluation(proposals) if admit else tuple(proposals)
    uncertainties = proposals_to_gate_uncertainties(admitted)
    return evaluate_epistemic_uncertainty_gate(uncertainties, user_text)
