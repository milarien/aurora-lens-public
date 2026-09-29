"""Phase 4: evaluator-facing temporal result type (narrow)."""

from __future__ import annotations

import pytest

from aurora_lens.state_native_engine.epistemic import EpistemicResult
from aurora_lens.state_native_engine.temporal_eval_result import (
    PresentBoundTemporalEvalResult,
    TemporalGovernanceCue,
)
from aurora_lens.state_native_engine.temporal_outcome_contract import TemporalOutcomeContract


def _sample_result(kind: TemporalOutcomeContract) -> PresentBoundTemporalEvalResult:
    if kind == TemporalOutcomeContract.TEMPORAL_MISSING:
        return PresentBoundTemporalEvalResult(
            outcome_kind=kind,
            governance_cue=TemporalGovernanceCue.ASK_TEMPORAL_SCOPE,
            rationale="No governing temporal anchor located for this query in committed state.",
            temporal_anchor_surface=None,
            temporal_span_label=None,
            source_commitment_id=None,
            evidence_payload={"structural_note": "anchor_absent"},
        )
    if kind == TemporalOutcomeContract.TEMPORAL_UNKNOWN:
        return PresentBoundTemporalEvalResult(
            outcome_kind=kind,
            governance_cue=TemporalGovernanceCue.GOVERNED_NON_ANSWER,
            rationale=(
                "Governing temporal anchor is located; requested relation "
                "is indeterminate from committed continuity."
            ),
            temporal_anchor_surface="after morning rounds",
            temporal_span_label="contact_window",
            source_commitment_id="commit-anchor-7",
            evidence_payload={"structural_note": "anchor_present_relation_indeterminate"},
        )
    if kind == TemporalOutcomeContract.TEMPORAL_AMBIGUOUS:
        return PresentBoundTemporalEvalResult(
            outcome_kind=kind,
            governance_cue=TemporalGovernanceCue.ASK_TEMPORAL_SCOPE,
            rationale="Multiple admissible temporal anchors; scope must be narrowed.",
            temporal_anchor_surface=None,
            temporal_span_label="either_window_A_or_window_B",
            evidence_payload={"candidates": ["week of 3 Feb", "after discharge"]},
        )
    if kind == TemporalOutcomeContract.SUPERSEDED:
        return PresentBoundTemporalEvalResult(
            outcome_kind=kind,
            governance_cue=TemporalGovernanceCue.PRESENT_COMMITTED_ANSWER,
            rationale="Earlier present commitment superseded by later committed continuity.",
            source_commitment_id="commit-supersedes-12",
            evidence_payload={"supersedes": "prior-id"},
        )
    if kind == TemporalOutcomeContract.FUTURE_PROJECTION:
        return PresentBoundTemporalEvalResult(
            outcome_kind=kind,
            governance_cue=TemporalGovernanceCue.PRESENT_COMMITTED_ANSWER,
            rationale="Present commitment capturing forward-oriented plan/deadline (not accomplished fact).",
            temporal_anchor_surface="planned: Tuesday review",
            evidence_payload={"projection_surface": "plan_not_fact"},
        )
    if kind == TemporalOutcomeContract.PAST_RECONSTRUCTION:
        return PresentBoundTemporalEvalResult(
            outcome_kind=kind,
            governance_cue=TemporalGovernanceCue.PRESENT_COMMITTED_ANSWER,
            rationale="Past-oriented reading licensed only via surviving present continuity.",
            evidence_payload={"reconstruction_basis": "HAS_snapshots"},
        )
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", list(TemporalOutcomeContract))
def test_one_eval_result_per_outcome_contract(kind: TemporalOutcomeContract) -> None:
    r = _sample_result(kind)
    assert isinstance(r, PresentBoundTemporalEvalResult)
    assert r.outcome_kind == kind
    assert r.rationale.strip()
    assert isinstance(r.evidence_payload, dict)


def test_temporal_missing_vs_unknown_structurally_distinct() -> None:
    m = _sample_result(TemporalOutcomeContract.TEMPORAL_MISSING)
    u = _sample_result(TemporalOutcomeContract.TEMPORAL_UNKNOWN)
    assert m.outcome_kind != u.outcome_kind
    assert m.governance_cue != u.governance_cue
    assert m.temporal_anchor_surface is None
    assert u.temporal_anchor_surface is not None
    assert "No governing temporal anchor located" in m.rationale
    assert "Governing temporal anchor is located" in u.rationale
    assert "anchor_absent" in m.evidence_payload.get("structural_note", "")
    assert "anchor_present" in u.evidence_payload.get("structural_note", "")


@pytest.mark.parametrize("kind", list(TemporalOutcomeContract))
def test_no_contract_value_equals_generic_epistemic_unknown(kind: TemporalOutcomeContract) -> None:
    assert kind is not EpistemicResult.UNKNOWN
    assert kind.value != EpistemicResult.UNKNOWN.value
