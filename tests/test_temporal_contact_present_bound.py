"""Narrow Phase 5 tests: temporal_contact present-bound evaluator + delegation helpers."""

from __future__ import annotations

from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.pef.temporal_commitment import PresentTemporalCommitment
from aurora_lens.state_native_engine.contracts import StateNativeOutcome
from aurora_lens.state_native_engine.epistemic import EpistemicResult
from aurora_lens.state_native_engine.eval.temporal_contact import (
    delegation_result_for_temporal_contact_query,
    evaluate_temporal_contact,
    present_bound_temporal_eval_to_delegation,
)
from aurora_lens.state_native_engine.temporal_eval_result import TemporalGovernanceCue
from aurora_lens.state_native_engine.temporal_outcome_contract import TemporalOutcomeContract
from aurora_lens.verify.flags import FlagType


def _pef_with_subject_and_rows(*rows: PresentTemporalCommitment) -> tuple[PEFState, Entity]:
    pef = PEFState(session_id="t")
    e = Entity.create("Rae", turn=0, session_id="t")
    pef.add_entity(e)
    pef.add_relationship(
        Relationship(
            subject_id=e.id,
            relation="AT",
            object_entity_id=None,
            object_literal="clinic",
            span=Span.PRESENT,
            source_turn=1,
            evidence="fixture",
        )
    )
    for r in rows:
        pef.add_present_temporal_commitment(
            PresentTemporalCommitment(
                outcome_kind=r.outcome_kind,
                source_turn=r.source_turn,
                evidence=r.evidence,
                subject_entity_id=r.subject_entity_id,
                commitment_id=r.commitment_id,
                continuity_payload=dict(r.continuity_payload),
            )
        )
    return pef, e


def test_missing_anchor_returns_temporal_missing_and_asks_scope() -> None:
    pef, _ = _pef_with_subject_and_rows()
    ev = evaluate_temporal_contact(pef, "Rae")
    assert ev is not None
    assert ev.outcome_kind == TemporalOutcomeContract.TEMPORAL_MISSING
    assert ev.governance_cue == TemporalGovernanceCue.ASK_TEMPORAL_SCOPE
    assert ev.temporal_anchor_surface is None
    sn = present_bound_temporal_eval_to_delegation(ev)
    assert sn.outcome == StateNativeOutcome.CLARIFY
    assert sn.epistemic_result == EpistemicResult.AMBIGUOUS
    assert sn.clarify_context is not None
    assert (
        sn.clarify_context.get("failed_constraint")
        == FlagType.STATE_NATIVE_TEMPORAL_ANCHOR_MISSING.name
    )


def test_located_anchor_unknown_returns_temporal_unknown_stop_without_epistemic_unknown() -> None:
    pef = PEFState(session_id="t")
    e = Entity.create("Kim", turn=0, session_id="t")
    pef.add_entity(e)
    pef.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.TEMPORAL_UNKNOWN,
            source_turn=2,
            evidence="fixture",
            subject_entity_id=e.id,
            commitment_id="fixed-unknown-row",
            continuity_payload={"anchor_surface": "after morning rounds"},
        )
    )
    ev = evaluate_temporal_contact(pef, "Kim")
    assert ev is not None
    assert ev.outcome_kind == TemporalOutcomeContract.TEMPORAL_UNKNOWN
    assert ev.governance_cue == TemporalGovernanceCue.GOVERNED_NON_ANSWER
    assert ev.temporal_anchor_surface == "after morning rounds"
    sn = present_bound_temporal_eval_to_delegation(ev)
    assert sn.outcome == StateNativeOutcome.STOP
    assert sn.epistemic_result is None
    assert sn.stop_reason_code == "state_native_temporal_unknown"


def test_ambiguous_explicit_marker_returns_ambiguous_scope() -> None:
    pef = PEFState(session_id="t")
    e = Entity.create("Jay", turn=0, session_id="t")
    pef.add_entity(e)
    pef.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.TEMPORAL_AMBIGUOUS,
            source_turn=1,
            evidence="fixture",
            subject_entity_id=e.id,
            continuity_payload={"candidate_anchors": ["week of 5 May", "post-discharge window"]},
        )
    )
    ev = evaluate_temporal_contact(pef, "Jay")
    assert ev is not None and ev.outcome_kind == TemporalOutcomeContract.TEMPORAL_AMBIGUOUS
    sn = present_bound_temporal_eval_to_delegation(ev)
    assert sn.clarify_context is not None
    assert (
        sn.clarify_context.get("failed_constraint")
        == FlagType.STATE_NATIVE_TEMPORAL_SCOPE_AMBIGUITY.name
    )


def test_projection_surfaces_projection_not_future_fact() -> None:
    pef = PEFState(session_id="t")
    e = Entity.create("Lee", turn=0, session_id="t")
    pef.add_entity(e)
    pef.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.FUTURE_PROJECTION,
            source_turn=1,
            evidence="committed intent",
            subject_entity_id=e.id,
            continuity_payload={"surface": "after Tuesday lab values"},
        )
    )
    ev = evaluate_temporal_contact(pef, "Lee")
    assert ev is not None and ev.outcome_kind == TemporalOutcomeContract.FUTURE_PROJECTION
    sn = delegation_result_for_temporal_contact_query(pef, "Lee")
    assert sn is not None and sn.outcome == StateNativeOutcome.ANSWER
    lower = sn.user_visible_text.lower()
    assert "present projection" in lower
    assert "not accomplished fact" in lower


def test_superseded_projection_not_false_via_outcome_contract() -> None:
    pef = PEFState(session_id="t")
    e = Entity.create("Morgan", turn=0, session_id="t")
    pef.add_entity(e)
    pef.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.SUPERSEDED,
            source_turn=3,
            evidence="replacement committed",
            subject_entity_id=e.id,
            continuity_payload={
                "current_projection_surface": "after wards round",
                "supersedes_commitment_id": "old-proj-id",
            },
        )
    )
    ev = evaluate_temporal_contact(pef, "Morgan")
    assert ev is not None and ev.outcome_kind == TemporalOutcomeContract.SUPERSEDED
    sn = present_bound_temporal_eval_to_delegation(ev)
    assert sn.outcome == StateNativeOutcome.ANSWER
    assert "updated" in sn.user_visible_text.lower()
    assert "not timeline-false" in ev.rationale.lower()


def test_temporal_missing_and_unknown_are_distinct_outcomes() -> None:
    pef_miss, e_miss = _pef_with_subject_and_rows()
    pef_known = PEFState(session_id="t")
    eu = Entity.create("Pat", turn=0, session_id="t")
    pef_known.add_entity(eu)
    pef_known.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.TEMPORAL_UNKNOWN,
            source_turn=1,
            evidence="fixture",
            subject_entity_id=eu.id,
            continuity_payload={"anchor_surface": "slot A"},
        )
    )

    missing = evaluate_temporal_contact(pef_miss, e_miss.name)
    located_unknown = evaluate_temporal_contact(pef_known, "Pat")
    assert missing is not None and missing.outcome_kind == TemporalOutcomeContract.TEMPORAL_MISSING
    assert (
        located_unknown is not None
        and located_unknown.outcome_kind == TemporalOutcomeContract.TEMPORAL_UNKNOWN
    )