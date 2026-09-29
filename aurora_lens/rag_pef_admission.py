"""Retrieved evidence → lawful PEF admission (RAG Context: block only).

This is not “make RAG work.” It tags facts grounded in the retrieved context
block with explicit provenance, drops unsupported causal extraction (BECAUSE),
detects simple IS-slot literal conflicts so PEF does not silently pick a winner,
records **temporal_arrival_incompatible** when the same narrow C1 pattern appears
in context claims (calendar month+day and *last week* in one blob — see
:mod:`aurora_lens.pef.arrival_incompatible`), and leaves unresolved entries on
:attr:`PEFState.retrieval_unresolved` for governance / summaries. Non-RAG turns
never call this module.
"""

from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import TYPE_CHECKING

from aurora_lens.corpus.evidence_consistency import detect_consistency_conflicts
from aurora_lens.corpus.evidence_admissibility import evaluate_retrieved_evidence
from aurora_lens.corpus.models import CorpusChunk, CorpusRecord
from aurora_lens.interpret.pef_admission import (
    PEFAdmissionDecision,
    PEFAdmissionEvidence,
)
from aurora_lens.interpret.pef_admission import PEFAdmissionResult
from aurora_lens.interpret.pef_updater import update_pef
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.pef.arrival_incompatible import incompatible_arrival_time_literal_strings
from aurora_lens.pef.state import PEFState, canonicalize_relation

if TYPE_CHECKING:
    from aurora_lens.request_metadata import RequestMetadata

PROVENANCE_RETRIEVED_CONTEXT = "retrieved_context"

# ── Retrieval-unresolved kind taxonomy ────────────────────────────────
# These live on PEFState.retrieval_unresolved, not open_epistemic_uncertainties.
# They are admission-layer artefacts consumed directly by checker.py.
# They carry no bears_on field and are structurally distinct from
# CORE_UNCERTAINTY_KINDS in aurora_lens.pef.uncertainty_analysis.
RETRIEVAL_UNRESOLVED_KINDS: frozenset[str] = frozenset({
    "literal_conflict",              # conflicting IS-slot literals; no winner admitted
    "temporal_arrival_incompatible", # calendar-date vs relative-time in same blob
    "evidence_inadmissible",         # record/chunk metadata failed admissibility checks
    "evidence_unresolved",           # admissibility requires clarification
    "threshold_conflict",            # admissible same-scope threshold values conflict
    "effective_date_conflict",       # admissible evidence has incompatible effective metadata
})

# Single-slot relations where at most one literal object should be committed per subject.
_IS_SLOT_RELATIONS = frozenset({"IS"})
_JSON_OBJECT_BLEED_MARKERS = (".json", "file:", "\n", "[", "]", "{", "}")


def _document_id_for_retrieval(metadata: RequestMetadata | None) -> str | None:
    if metadata is None:
        return None
    if metadata.record_ids:
        return str(metadata.record_ids[0])
    return None


def _document_locator_from_context_block(context_block: str) -> str | None:
    """First heading-like line (### … or Section …) as a cheap section marker."""
    for line in context_block.strip().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("###"):
            return s[:240]
        low = s.lower()
        if low.startswith("section ") or low.startswith("section:"):
            return s[:240]
    return None


def _partition_temporal_arrival_incompatibility(
    claims: list[ExtractedClaim],
) -> tuple[list[ExtractedClaim], list[dict]]:
    """Relation-agnostic: per subject, join claim surfaces; same narrow C1 test as verify.

    When incompatible, no claims for that subject are admitted (calendar vs *last week*).
    """
    groups: dict[str, list[ExtractedClaim]] = defaultdict(list)
    for c in claims:
        groups[c.subject.strip().lower()].append(c)

    temporal_entries: list[dict] = []
    blocked_subjects: set[str] = set()

    for subj, group in groups.items():
        pieces = [f"{c.obj} {(c.evidence or '')}".strip() for c in group]
        pieces = [p for p in pieces if p]
        if not pieces:
            continue
        if not incompatible_arrival_time_literal_strings(pieces):
            continue
        blocked_subjects.add(subj)
        temporal_entries.append({
            "kind": "temporal_arrival_incompatible",
            "subject": subj,
            "evidence": [c.evidence for c in group if c.evidence],
        })

    kept = [c for c in claims if c.subject.strip().lower() not in blocked_subjects]
    return kept, temporal_entries


def _drop_causal_claims(claims: list[ExtractedClaim]) -> list[ExtractedClaim]:
    """Do not admit unsupported causal links from retrieval into committed PEF."""
    out: list[ExtractedClaim] = []
    for c in claims:
        if canonicalize_relation(c.relation) == "BECAUSE":
            continue
        out.append(c)
    return out


def _normalise_surface(s: str) -> str:
    t = re.sub(r"\s+", " ", (s or "").strip().lower())
    return t.strip(" \"'.,;:!?")


def _claim_under_evaluation_literals(context_block: str) -> set[str]:
    """Extract proposition strings from JSON ``claim_under_evaluation`` values."""
    out: set[str] = set()
    for m in re.finditer(
        r'"claim_under_evaluation"\s*:\s*"((?:[^"\\]|\\.)*)"',
        context_block,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        raw = m.group(1)
        try:
            value = json.loads(f"\"{raw}\"")
        except Exception:
            value = raw
        norm = _normalise_surface(str(value))
        if norm:
            out.add(norm)
    return out


def _is_json_object_bleed_claim(claim: ExtractedClaim) -> bool:
    obj = str(claim.obj or "")
    if len(obj) < 140:
        return False
    low = obj.lower()
    return any(marker in low for marker in _JSON_OBJECT_BLEED_MARKERS)


def _drop_non_assertive_or_bleed_claims(
    claims: list[ExtractedClaim],
    *,
    context_block: str,
) -> list[ExtractedClaim]:
    """Drop proposition-under-evaluation claims and JSON object-bleed artifacts."""
    claim_eval = _claim_under_evaluation_literals(context_block)
    out: list[ExtractedClaim] = []
    for c in claims:
        if _is_json_object_bleed_claim(c):
            continue
        ev_norm = _normalise_surface(c.evidence or "")
        obj_norm = _normalise_surface(str(c.obj or ""))
        if claim_eval:
            matched_eval = False
            for proposition in claim_eval:
                if proposition and (
                    ev_norm == proposition
                    or obj_norm == proposition
                    or proposition in ev_norm
                    or proposition in obj_norm
                    or (obj_norm and obj_norm in proposition)
                ):
                    matched_eval = True
                    break
            if matched_eval:
                continue
        out.append(c)
    return out


def _is_literal_conflict_group(relation: str, claims: list[ExtractedClaim]) -> bool:
    rel_c = canonicalize_relation(relation)
    if rel_c not in _IS_SLOT_RELATIONS:
        return False
    literals: set[str] = set()
    for c in claims:
        lit = str(c.obj).strip().lower()
        if lit:
            literals.add(lit)
    return len(literals) > 1


def _partition_is_literal_conflicts(
    claims: list[ExtractedClaim],
) -> tuple[list[ExtractedClaim], list[dict]]:
    """Split claims; groups with multiple incompatible IS literals become conflicts (none admitted)."""
    groups: dict[tuple[str, str], list[ExtractedClaim]] = defaultdict(list)
    for c in claims:
        subj = c.subject.strip().lower()
        rel = canonicalize_relation(c.relation)
        groups[(subj, rel)].append(c)

    accepted: list[ExtractedClaim] = []
    conflicts: list[dict] = []

    for (subj, rel), group in groups.items():
        if _is_literal_conflict_group(rel, group):
            literals = sorted({str(c.obj).strip() for c in group if str(c.obj).strip()})
            evidence = [c.evidence for c in group if c.evidence]
            conflicts.append({
                "kind": "literal_conflict",
                "subject": subj,
                "relation": rel,
                "literals": literals,
                "evidence": evidence,
            })
            continue
        # Deduplicate identical claims in the same slot
        seen: set[tuple[str, str, str, bool]] = set()
        for c in group:
            key = (
                c.subject.strip().lower(),
                rel,
                str(c.obj).strip().lower(),
                c.negated,
            )
            if key in seen:
                continue
            seen.add(key)
            accepted.append(c)

    return accepted, conflicts


def _tag_retrieval_claims(
    claims: list[ExtractedClaim],
    *,
    document_id: str | None,
    document_locator: str | None,
) -> list[ExtractedClaim]:
    out: list[ExtractedClaim] = []
    for c in claims:
        out.append(
            ExtractedClaim(
                subject=c.subject,
                relation=c.relation,
                obj=c.obj,
                span=c.span,
                negated=c.negated,
                evidence=c.evidence,
                provenance=PROVENANCE_RETRIEVED_CONTEXT,
                extractor_backend=c.extractor_backend,
                document_id=document_id,
                document_locator=document_locator,
            )
        )
    return out


def admit_retrieved_context_to_pef(
    ext_ctx: ExtractionResult,
    pef: PEFState,
    *,
    context_block: str,
    request_metadata: RequestMetadata | None,
    provenance_document_id: str | None = None,
    provenance_document_locator: str | None = None,
    evidence_chunks: list[CorpusChunk] | None = None,
    evidence_record: CorpusRecord | None = None,
    evidence_records: list[CorpusRecord] | None = None,
    request_scope: str | None = None,
    require_authority: bool = True,
) -> PEFAdmissionResult:
    """Apply extraction from the retrieved ``Context:`` body with RAG-specific PEF rules.

    Mutates ``pef`` via :func:`update_pef` (no ``user_text`` — QUERY classification
    does not apply to corpus seeding). Replaces ``pef.retrieval_unresolved`` for
    this admission pass with any conflicts detected in the current context batch.

    Returns the bounded mutator :class:`~aurora_lens.interpret.pef_admission.PEFAdmissionResult`
    from the final ``update_pef`` call (for operator-plane telemetry only).
    """
    pef.retrieval_unresolved = []

    use_single_record_admissibility = (evidence_records is None) and (
        evidence_chunks is not None or evidence_record is not None
    )
    if use_single_record_admissibility:
        admissibility = evaluate_retrieved_evidence(
            chunks=evidence_chunks or [],
            record=evidence_record,
            request_scope=request_scope,
            require_authority=require_authority,
        )
        if admissibility.status != "admissible":
            kind = (
                "evidence_inadmissible"
                if admissibility.status == "inadmissible"
                else "evidence_unresolved"
            )
            pef.retrieval_unresolved = [
                {
                    "kind": kind,
                    "failure_kind": admissibility.failure_kind,
                    "failure_meta": dict(admissibility.failure_meta),
                    "authority_state": admissibility.authority_state,
                    "freshness_status": admissibility.freshness_status,
                    "authority_status": admissibility.authority_status,
                    "demotion_reason": admissibility.demotion_reason,
                    "record_id": admissibility.record_id,
                    "chunk_ids": list(admissibility.chunk_ids),
                    "reasons": list(admissibility.reasons),
                }
            ]
            decision = (
                PEFAdmissionDecision.REJECT
                if admissibility.status == "inadmissible"
                else PEFAdmissionDecision.HOLD
            )
            return PEFAdmissionResult(
                decision=decision,
                write_intent=False,
                planned_mutation_slices=len(ext_ctx.claims),
                world_state_mutation_count=0,
                continuation_state_mutation_count=0,
                mutation_count=0,
                mutation_summary={
                    "blocked_by_evidence_admissibility": True,
                    "admissibility_status": admissibility.status,
                    "failure_kind": admissibility.failure_kind,
                    "failure_meta": dict(admissibility.failure_meta),
                    "authority_state": admissibility.authority_state,
                    "freshness_status": admissibility.freshness_status,
                    "authority_status": admissibility.authority_status,
                    "demotion_reason": admissibility.demotion_reason,
                    "record_id": admissibility.record_id,
                    "chunk_ids": list(admissibility.chunk_ids),
                },
                evidence=[
                    PEFAdmissionEvidence(
                        veto_kind=kind,
                        scope="retrieved_context",
                        held_reason=admissibility.failure_kind,
                        detail="; ".join(admissibility.reasons),
                        slice_index=None,
                    )
                ],
            )

    doc_id = provenance_document_id or _document_id_for_retrieval(request_metadata)
    locator = provenance_document_locator or _document_locator_from_context_block(context_block)

    filtered = _drop_causal_claims(list(ext_ctx.claims))
    filtered = _drop_non_assertive_or_bleed_claims(filtered, context_block=context_block)
    after_temporal, temporal_unresolved = _partition_temporal_arrival_incompatibility(filtered)
    admitted, literal_conflicts = _partition_is_literal_conflicts(after_temporal)
    consistency_conflicts: list[dict] = []
    if evidence_chunks:
        _records = list(evidence_records or [])
        if not _records and evidence_record is not None:
            _records = [evidence_record]
        if _records:
            consistency_conflicts = detect_consistency_conflicts(
                chunks=list(evidence_chunks),
                records=_records,
                commitment_scope=request_scope,
            )
    pef.retrieval_unresolved = temporal_unresolved + literal_conflicts + consistency_conflicts

    tagged = _tag_retrieval_claims(
        admitted,
        document_id=doc_id,
        document_locator=locator,
    )

    result = ExtractionResult(
        claims=tagged,
        entity_mentions=list(ext_ctx.entity_mentions),
        span=ext_ctx.span,
        pronoun_candidates=dict(ext_ctx.pronoun_candidates),
        ambiguous_referents=list(ext_ctx.ambiguous_referents),
        comparative_ambiguities=list(ext_ctx.comparative_ambiguities),
        extraction_error=ext_ctx.extraction_error,
    )
    return update_pef(result, pef, user_text=None)
