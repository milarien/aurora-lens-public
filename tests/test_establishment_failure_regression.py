"""Track A sign-off regression — establishment-failure generator track.

Confirms implementation posture, anti-soup boundaries, and gate parity for all six
Phase 2 kinds. Five kinds are deterministic-generated; ``source_untrusted`` remains
external-declared only until trust-registry machinery exists.

Full suite (run before Track A sign-off):

  pytest tests/test_establishment_failure_regression.py \\
         tests/test_pef_uncertainty_analysis.py \\
         tests/test_epistemic_uncertainty_gate.py \\
         tests/test_stale_evidence_generation.py \\
         tests/test_threshold_not_met_generation.py \\
         tests/test_inferential_gap_generation.py \\
         tests/test_scope_mismatch_generation.py \\
         tests/test_unresolved_referent_registry.py -q
"""

from __future__ import annotations

import pytest

from aurora_lens.govern.epistemic_uncertainty_gate import evaluate_epistemic_uncertainty_gate
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.pef.span import Span
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.uncertainty_analysis import (
    EXTERNAL_ESTABLISHMENT_FAILURE_KINDS,
    PHASE2_ESTABLISHMENT_FAILURE_IMPLEMENTATION,
    pef_uncertainty_analysis,
)

TRACK_A_DETERMINISTIC_KINDS: frozenset[str] = frozenset({
    "identity_or_referent_unresolved",
    "stale_evidence",
    "scope_mismatch",
    "threshold_not_met",
    "inferential_gap",
})

TRACK_A_EXTERNAL_DECLARED_ONLY: frozenset[str] = frozenset({"source_untrusted"})


class TestTrackAImplementationPosture:
    def test_five_kinds_are_deterministic_generated(self):
        for kind in TRACK_A_DETERMINISTIC_KINDS:
            assert (
                PHASE2_ESTABLISHMENT_FAILURE_IMPLEMENTATION[kind]
                == "deterministic_generated"
            )

    def test_source_untrusted_is_external_declared_only(self):
        assert (
            PHASE2_ESTABLISHMENT_FAILURE_IMPLEMENTATION["source_untrusted"]
            == "external_declared_only"
        )

    def test_all_six_kinds_in_taxonomy(self):
        assert TRACK_A_DETERMINISTIC_KINDS | TRACK_A_EXTERNAL_DECLARED_ONLY == (
            EXTERNAL_ESTABLISHMENT_FAILURE_KINDS
        )


class TestTrackAAntiSoup:
    def test_empty_pef_emits_no_establishment_failure_kinds(self):
        kinds = {r["kind"] for r in pef_uncertainty_analysis(PEFState())}
        assert kinds.isdisjoint(EXTERNAL_ESTABLISHMENT_FAILURE_KINDS)

    def test_informal_metadata_does_not_generate_source_untrusted(self):
        pef = PEFState()
        sid = Entity.create("claim", turn=1)
        pef.entities[sid.id] = sid
        pef.relationships.append(
            Relationship(
                subject_id=sid.id,
                relation="IS",
                object_entity_id=None,
                object_literal="trusted",
                span=Span.PRESENT,
                source_turn=1,
                evidence="note",
                relation_metadata={
                    "trust": {"notes": "sounds untrusted", "source_class": "unknown"},
                },
            )
        )
        kinds = {r["kind"] for r in pef_uncertainty_analysis(pef)}
        assert "source_untrusted" not in kinds


class TestTrackAGateParity:
    """Generated and externally declared records must both block decision-seeking turns."""

    @pytest.mark.parametrize("kind", sorted(EXTERNAL_ESTABLISHMENT_FAILURE_KINDS))
    def test_open_uncertainty_blocks_decision_turn(self, kind: str):
        result = evaluate_epistemic_uncertainty_gate(
            [
                {
                    "id": f"regression_{kind}",
                    "kind": kind,
                    "description": f"Regression declaration for {kind}.",
                    "bears_on": ["test.claim"],
                    "status": "open",
                }
            ],
            "Which option should we choose?",
        )
        assert result is not None
        assert result.blocks is True

    def test_resolved_uncertainty_does_not_block(self):
        result = evaluate_epistemic_uncertainty_gate(
            [
                {
                    "id": "regression_resolved",
                    "kind": "scope_mismatch",
                    "description": "Resolved scope mismatch.",
                    "bears_on": ["test.claim"],
                    "status": "resolved",
                }
            ],
            "Which option should we choose?",
        )
        assert result is None

    def test_non_decision_turn_does_not_block(self):
        result = evaluate_epistemic_uncertainty_gate(
            [
                {
                    "id": "regression_open",
                    "kind": "threshold_not_met",
                    "description": "Open threshold failure.",
                    "bears_on": ["test.claim"],
                    "status": "open",
                }
            ],
            "Summarise the weather bulletin.",
        )
        assert result is None
