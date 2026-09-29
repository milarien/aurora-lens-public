"""Phase 3: PresentTemporalCommitment and PEF storage (narrow)."""

from __future__ import annotations

from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.pef.temporal_commitment import (
    PresentTemporalCommitment,
    commitment_from_wire_dict,
    commitment_to_wire_dict,
)
from aurora_lens.state_native_engine.epistemic import EpistemicResult
from aurora_lens.state_native_engine.temporal_outcome_contract import TemporalOutcomeContract


def _fixture_commitment(kind: TemporalOutcomeContract) -> PresentTemporalCommitment:
    """Synthetic payloads distinguishing modalities (present commitment language only)."""
    subject = "e-test"
    if kind == TemporalOutcomeContract.FUTURE_PROJECTION:
        return PresentTemporalCommitment(
            outcome_kind=kind,
            source_turn=1,
            evidence="intent: schedule follow-up",
            subject_entity_id=subject,
            commitment_id=f"id-{kind.value}",
            continuity_payload={
                "present_commitment": "projection",
                "surface": "after morning rounds",
            },
        )
    if kind == TemporalOutcomeContract.PAST_RECONSTRUCTION:
        return PresentTemporalCommitment(
            outcome_kind=kind,
            source_turn=1,
            evidence="derived from surviving HAS/AT continuity",
            subject_entity_id=subject,
            commitment_id=f"id-{kind.value}",
            continuity_payload={
                "present_commitment": "reconstruction",
                "basis_relation": "HAS",
            },
        )
    if kind == TemporalOutcomeContract.SUPERSEDED:
        return PresentTemporalCommitment(
            outcome_kind=kind,
            source_turn=2,
            evidence="later turn displaces earlier projection record",
            subject_entity_id=subject,
            commitment_id=f"id-{kind.value}",
            continuity_payload={
                "present_commitment": "supersession",
                "supersedes_commitment_id": "id-prior",
            },
        )
    if kind == TemporalOutcomeContract.TEMPORAL_AMBIGUOUS:
        return PresentTemporalCommitment(
            outcome_kind=kind,
            source_turn=1,
            evidence="two admissible windows",
            subject_entity_id=subject,
            commitment_id=f"id-{kind.value}",
            continuity_payload={
                "present_commitment": "ambiguity",
                "candidate_anchors": ["week of 3 Feb", "after discharge"],
            },
        )
    if kind == TemporalOutcomeContract.TEMPORAL_MISSING:
        return PresentTemporalCommitment(
            outcome_kind=kind,
            source_turn=1,
            evidence="temporal query with no governing anchor in PEF",
            subject_entity_id=subject,
            commitment_id=f"id-{kind.value}",
            continuity_payload={
                "deficit_kind": "no_governing_temporal_anchor_located",
            },
        )
    if kind == TemporalOutcomeContract.TEMPORAL_UNKNOWN:
        return PresentTemporalCommitment(
            outcome_kind=kind,
            source_turn=1,
            evidence="anchor fixed; relation indeterminate from committed state",
            subject_entity_id=subject,
            commitment_id=f"id-{kind.value}",
            continuity_payload={
                "deficit_kind": "relation_indeterminate_within_located_anchor",
                "anchor_surface": "after morning rounds",
            },
        )
    raise AssertionError(f"unhandled kind {kind!r}")


def test_all_six_temporal_outcome_contract_rows_instantiable() -> None:
    for kind in TemporalOutcomeContract:
        c = _fixture_commitment(kind)
        assert c.outcome_kind == kind
        wire = commitment_to_wire_dict(c)
        back = commitment_from_wire_dict(wire)
        assert back.outcome_kind == kind
        assert back.source_turn == c.source_turn


def test_temporal_missing_vs_unknown_structurally_different_records() -> None:
    missing = _fixture_commitment(TemporalOutcomeContract.TEMPORAL_MISSING)
    unknown = _fixture_commitment(TemporalOutcomeContract.TEMPORAL_UNKNOWN)
    assert missing.outcome_kind != unknown.outcome_kind
    assert missing.continuity_payload["deficit_kind"] != unknown.continuity_payload["deficit_kind"]
    assert "anchor_surface" not in missing.continuity_payload
    assert unknown.continuity_payload.get("anchor_surface")
    wm = commitment_to_wire_dict(missing)
    wu = commitment_to_wire_dict(unknown)
    assert wm["outcome_kind"] != wu["outcome_kind"]
    assert wm["continuity_payload"] != wu["continuity_payload"]


def test_no_epistemic_unknown_alias_on_wire_values() -> None:
    for kind in TemporalOutcomeContract:
        c = _fixture_commitment(kind)
        w = commitment_to_wire_dict(c)
        assert w["outcome_kind"] != EpistemicResult.UNKNOWN.value


def test_pef_round_trip_preserves_temporal_commitments() -> None:
    pef = PEFState(session_id="t")
    e = Entity.create("Ada", turn=0, session_id="t")
    pef.add_entity(e)
    pef.add_relationship(
        Relationship(
            subject_id=e.id,
            relation="HAS",
            object_entity_id=None,
            object_literal="chart",
            span=Span.PRESENT,
            source_turn=0,
            evidence="fixture",
        )
    )
    for kind in TemporalOutcomeContract:
        pef.add_present_temporal_commitment(_fixture_commitment(kind))
    d = pef.to_dict()
    restored = PEFState.from_dict(d)
    assert len(restored.temporal_commitments) == len(TemporalOutcomeContract)
    kinds = {c.outcome_kind for c in restored.temporal_commitments}
    assert kinds == set(TemporalOutcomeContract)
