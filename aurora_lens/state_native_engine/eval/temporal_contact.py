"""Temporal-contact query: when should I contact a resolved-state entity?

Invariant: if the entity is in PEF, timing questions about that entity
must not fall through to LLM. The LLM has no temporal policy; it will
generate outside the verified world.

If the entity is not in PEF → pass through to LLM via ``delegation … None``.
Otherwise resolves from :class:`~aurora_lens.pef.temporal_commitment.PresentTemporalCommitment`
rows and the **Temporal Outcome Contract** via :class:`~aurora_lens.state_native_engine.temporal_eval_result.PresentBoundTemporalEvalResult`.

**Ontology:** present-bound semantics per ``docs/pef_temporality_present_bound_ontology.md``.
``TEMPORAL_MISSING`` ≠ ``TEMPORAL_UNKNOWN``; neither is aliased to
:class:`~aurora_lens.state_native_engine.epistemic.EpistemicResult.UNKNOWN`.
"""

from __future__ import annotations

from typing import Sequence

from aurora_lens.pef.state import PEFState
from aurora_lens.pef.temporal_commitment import PresentTemporalCommitment
from aurora_lens.state_native_engine.bind import resolve_entity, BindingKind
from aurora_lens.state_native_engine.contracts import (
    StateNativeDelegationResult,
    StateNativeOutcome,
    StateNativeSolverFamily,
)
from aurora_lens.state_native_engine.epistemic import EpistemicResult
from aurora_lens.state_native_engine.temporal_eval_result import (
    PresentBoundTemporalEvalResult,
    TemporalGovernanceCue,
)
from aurora_lens.state_native_engine.temporal_outcome_contract import TemporalOutcomeContract


def _projection_surface(row: PresentTemporalCommitment) -> str:
    p = row.continuity_payload
    surf = (
        str(p.get("surface") or p.get("projection_surface") or "").strip()
        or "(unspecified_projection)"
    )
    return surf


def _sorted_rows_for_subject(
    pef: PEFState,
    subject_id: str,
) -> list[PresentTemporalCommitment]:
    rows = [
        r
        for r in pef.temporal_commitments
        if r.subject_entity_id == subject_id
    ]
    return sorted(rows, key=lambda r: (r.source_turn, r.commitment_id), reverse=True)


def _present_rows_at_latest_turn(rows: Sequence[PresentTemporalCommitment]) -> list[PresentTemporalCommitment]:
    if not rows:
        return []
    latest = max(r.source_turn for r in rows)
    return [r for r in rows if r.source_turn == latest]


def evaluate_temporal_contact(
    pef: PEFState,
    entity_phrase: str,
) -> PresentBoundTemporalEvalResult | None:
    """Pure evaluation: resolves binding and present temporal commitments only.

    Returns ``None`` when the entity phrase does not bind to PEF (caller delegates
    to LLM). Otherwise returns :class:`PresentBoundTemporalEvalResult`.

    Legacy quartet tuples are intentionally not emitted from the evaluator core.
    """
    binding = resolve_entity(entity_phrase, pef)
    if binding.kind == BindingKind.MISSING or not binding.entities:
        return None

    subject = binding.entities[0]
    sid = subject.id
    display = subject.name
    rows = _sorted_rows_for_subject(pef, sid)

    payload_base: dict[str, str | object] = {"subject_display": display}

    if not rows:
        return PresentBoundTemporalEvalResult(
            outcome_kind=TemporalOutcomeContract.TEMPORAL_MISSING,
            governance_cue=TemporalGovernanceCue.ASK_TEMPORAL_SCOPE,
            rationale=(
                "No governing temporal anchor for this contact question is recorded "
                "in committed state yet."
            ),
            temporal_anchor_surface=None,
            temporal_span_label=None,
            evidence_payload=dict(
                payload_base,
                structural_note="no_temporal_commitment_rows_for_subject",
            ),
        )

    slice_latest = _present_rows_at_latest_turn(rows)

    ambiguous_markers = [
        r for r in slice_latest if r.outcome_kind == TemporalOutcomeContract.TEMPORAL_AMBIGUOUS
    ]
    if ambiguous_markers:
        cand = ambiguous_markers[0].continuity_payload.get("candidate_anchors") or []
        return PresentBoundTemporalEvalResult(
            outcome_kind=TemporalOutcomeContract.TEMPORAL_AMBIGUOUS,
            governance_cue=TemporalGovernanceCue.ASK_TEMPORAL_SCOPE,
            rationale=(
                "More than one admissible temporal scope is present in committed state "
                "for this contact question."
            ),
            temporal_anchor_surface=None,
            temporal_span_label="multiple_scopes",
            source_commitment_id=ambiguous_markers[0].commitment_id,
            evidence_payload=dict(
                payload_base,
                candidate_anchors=list(cand) if isinstance(cand, list) else [],
                structural_note="committed_ambiguity_marker",
            ),
        )

    future_rows = [
        r for r in slice_latest if r.outcome_kind == TemporalOutcomeContract.FUTURE_PROJECTION
    ]
    surfaces = {_projection_surface(r) for r in future_rows}
    if len(surfaces) > 1:
        return PresentBoundTemporalEvalResult(
            outcome_kind=TemporalOutcomeContract.TEMPORAL_AMBIGUOUS,
            governance_cue=TemporalGovernanceCue.ASK_TEMPORAL_SCOPE,
            rationale=(
                "Committed present projections specify more than one contact timing posture; "
                "a single governing temporal scope must be chosen."
            ),
            temporal_anchor_surface=None,
            temporal_span_label="competing_projections",
            evidence_payload=dict(
                payload_base,
                projection_surfaces=sorted(surfaces),
                structural_note="multiple_projection_surfaces_same_turn",
            ),
        )

    superseded_rows = [
        r for r in slice_latest if r.outcome_kind == TemporalOutcomeContract.SUPERSEDED
    ]
    if superseded_rows:
        row = superseded_rows[0]
        prior = row.continuity_payload.get("supersedes_commitment_id")
        return PresentBoundTemporalEvalResult(
            outcome_kind=TemporalOutcomeContract.SUPERSEDED,
            governance_cue=TemporalGovernanceCue.PRESENT_COMMITTED_ANSWER,
            rationale=(
                "A prior present temporal commitment was displaced by newer committed continuity; "
                "answer reflects the superseding present record (not timeline-false)."
            ),
            temporal_anchor_surface=str(
                row.continuity_payload.get("current_projection_surface") or ""
            ).strip()
            or None,
            temporal_span_label="superseded_chain",
            source_commitment_id=row.commitment_id,
            evidence_payload=dict(
                payload_base,
                supersedes_commitment_id=str(prior) if prior else None,
                structural_note="supersedes_present_commitment",
            ),
        )

    if len(future_rows) == 1:
        row = future_rows[0]
        surf = _projection_surface(row)
        return PresentBoundTemporalEvalResult(
            outcome_kind=TemporalOutcomeContract.FUTURE_PROJECTION,
            governance_cue=TemporalGovernanceCue.PRESENT_COMMITTED_ANSWER,
            rationale=(
                "Present commitment records a forward-oriented contact posture (projection), "
                "not an accomplished future fact."
            ),
            temporal_anchor_surface=surf,
            temporal_span_label="present_projection",
            source_commitment_id=row.commitment_id,
            evidence_payload=dict(payload_base, structural_note="single_projection_ack"),
        )

    recon_rows = [
        r
        for r in slice_latest
        if r.outcome_kind == TemporalOutcomeContract.PAST_RECONSTRUCTION
    ]
    if len(recon_rows) == 1:
        row = recon_rows[0]
        basis = str(row.continuity_payload.get("basis_relation") or "continuity").strip()
        return PresentBoundTemporalEvalResult(
            outcome_kind=TemporalOutcomeContract.PAST_RECONSTRUCTION,
            governance_cue=TemporalGovernanceCue.PRESENT_COMMITTED_ANSWER,
            rationale=(
                "Past-oriented wording is justified only via surviving present continuity "
                "(reconstruction), not direct archival replay."
            ),
            temporal_anchor_surface=str(row.continuity_payload.get("reconstruction_surface") or "")
            .strip()
            or None,
            temporal_span_label="present_reconstruction",
            source_commitment_id=row.commitment_id,
            evidence_payload=dict(
                payload_base,
                basis_relation=basis,
                structural_note="past_reconstruction_ack",
            ),
        )

    unknown_markers = [
        r for r in slice_latest if r.outcome_kind == TemporalOutcomeContract.TEMPORAL_UNKNOWN
    ]
    if unknown_markers:
        row = unknown_markers[0]
        anchor_s = (
            str(row.continuity_payload.get("anchor_surface") or "").strip() or None
        )
        return PresentBoundTemporalEvalResult(
            outcome_kind=TemporalOutcomeContract.TEMPORAL_UNKNOWN,
            governance_cue=TemporalGovernanceCue.GOVERNED_NON_ANSWER,
            rationale=(
                "The governing temporal anchor is located in committed state, but "
                "the requested contact relation cannot be determined from present continuity."
            ),
            temporal_anchor_surface=anchor_s,
            temporal_span_label="located_anchor_indeterminate_relation",
            source_commitment_id=row.commitment_id,
            evidence_payload=dict(payload_base, structural_note="explicit_temporal_unknown_row"),
        )

    missing_marker = [
        r for r in slice_latest if r.outcome_kind == TemporalOutcomeContract.TEMPORAL_MISSING
    ]
    if missing_marker:
        return PresentBoundTemporalEvalResult(
            outcome_kind=TemporalOutcomeContract.TEMPORAL_MISSING,
            governance_cue=TemporalGovernanceCue.ASK_TEMPORAL_SCOPE,
            rationale=(
                "Committed state explicitly records absence of a governing temporal anchor "
                "for this query shape."
            ),
            temporal_anchor_surface=None,
            evidence_payload=dict(payload_base, structural_note="explicit_temporal_missing_row"),
            source_commitment_id=missing_marker[0].commitment_id,
        )

    # Non-temporal semantic rows only (unexpected); treat conservatively as missing anchor.
    return PresentBoundTemporalEvalResult(
        outcome_kind=TemporalOutcomeContract.TEMPORAL_MISSING,
        governance_cue=TemporalGovernanceCue.ASK_TEMPORAL_SCOPE,
        rationale="Committed temporal rows do not license a governing anchor for contact timing.",
        temporal_anchor_surface=None,
        evidence_payload=dict(payload_base, structural_note="unusable_temporal_rows"),
    )


def _user_visible_temporal_answer(display: str, ev: PresentBoundTemporalEvalResult) -> str:
    # Unclassified temporal shapes return UNKNOWN (fail-closed) — docs/closed-decisions.md.
    kind = ev.outcome_kind
    if kind == TemporalOutcomeContract.TEMPORAL_MISSING:
        return (
            f"Committed state includes {display}, but it does not yet specify "
            "a governing temporal anchor for when to contact them. "
            "What time frame, schedule, or constraint should determine that contact?"
        )
    if kind == TemporalOutcomeContract.TEMPORAL_AMBIGUOUS:
        return (
            f"Committed state admits more than one temporal scope for contacting {display}. "
            "Which temporal window should govern this contact decision?"
        )
    if kind == TemporalOutcomeContract.TEMPORAL_UNKNOWN:
        anchor = (
            ev.temporal_anchor_surface or "recorded temporal anchor"
        )
        return (
            f"A temporal anchor ({anchor}) is fixed for contacting {display}, but "
            "the requested relation cannot be determined from present continuity."
        )
    if kind == TemporalOutcomeContract.FUTURE_PROJECTION:
        surf = ev.temporal_anchor_surface or "recorded projection"
        return (
            f"Committed state records contact timing only as present projection (not accomplished fact): "
            f"use {surf}."
        ).replace("  ", " ")
    if kind == TemporalOutcomeContract.PAST_RECONSTRUCTION:
        basis = (
            str(ev.evidence_payload.get("basis_relation") or "continuity").strip()
        )
        surface = ev.temporal_anchor_surface or "(from present surviving continuity)"
        return (
            f"From surviving present continuity (reconstruction, basis={basis}), "
            f"timing context for contacting {display} is summarized as {surface}. "
            "Not an archival replay of objective past facts."
        )
    if kind == TemporalOutcomeContract.SUPERSEDED:
        anchor = (
            ev.temporal_anchor_surface or "the superseding committed posture"
        )
        tail = ""
        supers = ev.evidence_payload.get("supersedes_commitment_id")
        if supers:
            tail = f" (supersedes present commitment reference {supers})."
        return (
            f"Present temporal commitments were updated.{tail} Follow {anchor} for current contact posture."
        )
    raise AssertionError(f"unsupported temporal outcomeKind {kind}")


def present_bound_temporal_eval_to_delegation(
    eval_res: PresentBoundTemporalEvalResult,
) -> StateNativeDelegationResult:
    """Translate evaluator result into :class:`StateNativeDelegationResult` (no UNKNOWN smuggling).

    Temporal deficits use CLARIFY (``EpistemicResult.AMBIGUOUS``) or STOP with
    ``epistemic_result=None`` so :func:`EpistemicResult.UNKNOWN` never encodes temporal contract rows.
    """
    display = str(eval_res.evidence_payload.get("subject_display") or "").strip()
    if not display:
        display = "this subject"

    text = _user_visible_temporal_answer(display, eval_res)
    cue = eval_res.governance_cue

    # Contract enum values look like TEMPORAL_UNKNOWN; strip the TEMPORAL_ prefix once
    # so stop codes remain ``state_native_temporal_<missing|unknown|…>``.
    ov = eval_res.outcome_kind.value.lower()
    stop_reason_tail = ov.removeprefix("temporal_")

    if cue == TemporalGovernanceCue.GOVERNED_NON_ANSWER:
        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.STOP,
            user_visible_text=text,
            clarify_context=None,
            stop_reason_code=f"state_native_temporal_{stop_reason_tail}",
            solver_family=StateNativeSolverFamily.COMMITTED_TEMPORAL_CONTACT_READ,
            epistemic_result=None,
            temporal_eval_result=eval_res,
        )

    if cue == TemporalGovernanceCue.ASK_TEMPORAL_SCOPE:
        if eval_res.outcome_kind == TemporalOutcomeContract.TEMPORAL_AMBIGUOUS:
            fc = "STATE_NATIVE_TEMPORAL_SCOPE_AMBIGUITY"
        else:
            fc = "STATE_NATIVE_TEMPORAL_ANCHOR_MISSING"
        cand = eval_res.evidence_payload.get("candidate_anchors") or []
        extra_anchors = eval_res.evidence_payload.get("projection_surfaces") or []
        clarify: dict[str, object] = {
            "failed_constraint": fc,
            "subject_phrase": display,
            "temporal_outcome_contract": eval_res.outcome_kind.value,
        }
        if isinstance(cand, list) and cand:
            clarify["candidate_entities"] = [str(x) for x in cand]
        if isinstance(extra_anchors, list) and extra_anchors:
            clarify["temporal_candidate_surfaces"] = [str(x) for x in extra_anchors]

        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.CLARIFY,
            user_visible_text=text,
            clarify_context=clarify,
            stop_reason_code=None,
            solver_family=StateNativeSolverFamily.COMMITTED_TEMPORAL_CONTACT_READ,
            epistemic_result=EpistemicResult.AMBIGUOUS,
            temporal_eval_result=eval_res,
        )

    if cue == TemporalGovernanceCue.PRESENT_COMMITTED_ANSWER:
        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.ANSWER,
            user_visible_text=text,
            clarify_context=None,
            stop_reason_code=None,
            solver_family=StateNativeSolverFamily.COMMITTED_TEMPORAL_CONTACT_READ,
            epistemic_result=None,
            temporal_eval_result=eval_res,
        )

    raise ValueError(f"unsupported TemporalGovernanceCue: {cue!r}")


def delegation_result_for_temporal_contact_query(
    pef: PEFState,
    entity_phrase: str,
) -> StateNativeDelegationResult | None:
    """Return handled state-native result, or ``None`` when temporal query binds no PEF entity."""
    ev = evaluate_temporal_contact(pef, entity_phrase)
    if ev is None:
        return None
    return present_bound_temporal_eval_to_delegation(ev)


def temporal_contact_legacy_quartet(
    pef: PEFState,
    entity_phrase: str,
) -> tuple[str | None, str, dict | None, str | None]:
    """Backward-compatible adapter (narrow): synthesize quartet from Phase 5 pipeline.

    **Tests / external tooling only** — Prefer :func:`delegation_result_for_temporal_contact_query`
    in new code.

    Quartet shape mirrors pre-Phase-5 callers: ``out_s``, ``text``, ``clarify``, ``stop_code``.
    Never fabricates EpistemicResult at this boundary; STOP rows leave ``stop_code`` prefixed.
    """
    ev = evaluate_temporal_contact(pef, entity_phrase)
    if ev is None:
        return None, "", None, None
    delegated = present_bound_temporal_eval_to_delegation(ev)
    return (
        delegated.outcome.value if delegated.outcome else None,
        delegated.user_visible_text,
        delegated.clarify_context,
        delegated.stop_reason_code,
    )
