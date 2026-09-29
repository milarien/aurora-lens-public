"""PEF admission envelope returned by ``update_pef`` (`PEFAdmissionResult`)."""

from __future__ import annotations

import pytest

from aurora_lens.interpret.pef_admission import PEFAdmissionDecision, finalized_mutation_aggregate, pef_admission_result_wire_dict
from aurora_lens.interpret.pef_updater import update_pef
from aurora_lens.interpret.schema import ComparativeAmbiguity, ExtractedClaim, ExtractionResult
from aurora_lens.interpret.turn_act import TurnAct
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState


def test_u_q0_admit_zeros_not_none() -> None:
    """Read-only QUERY path sets explicit mutation zeros (never None once envelope ran)."""
    pef = PEFState()
    r = update_pef(
        ExtractionResult(claims=[], span=Span.PRESENT),
        pef,
        user_text="What colour is Emma's book?",
        turn_act=TurnAct.QUERY,
    )
    assert r.decision == PEFAdmissionDecision.ADMIT
    assert r.write_intent is False
    assert r.mutation_count == 0
    assert r.world_state_mutation_count == 0
    assert r.continuation_state_mutation_count == 0
    assert len(r.evidence) == 1 and r.evidence[0].veto_kind == "U-Q0"


def test_finalize_mutation_aggregate_sum() -> None:
    w, c, total, summary = finalized_mutation_aggregate(2, 1)
    assert total == 3
    assert w == 2 and c == 1
    assert summary.get("ontology") == "pef_admission_xor_v1"


@pytest.mark.parametrize(
    "subject,ambiguous,comps,held_veto",
    [
        ("he", ["he"], [], "U-TX3"),
        ("his wallet", ["his"], [], "U-TX2"),
        ("Richard", [], [ComparativeAmbiguity(adjective="big", noun="stick", candidates=["Richard", "Lucy"])], "U-TX1"),
    ],
)
def test_semantic_hold_appends_evidence(
    subject: str,
    ambiguous: list[str],
    comps: list[ComparativeAmbiguity],
    held_veto: str,
) -> None:
    pef = PEFState()
    claim_obj = "big" if held_veto == "U-TX1" else ("wallet" if held_veto == "U-TX2" else "tall")
    ext = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject=subject,
                relation="IS",
                obj=claim_obj,
                span=Span.PRESENT,
                negated=False,
                evidence="t",
            )
        ],
        ambiguous_referents=ambiguous,
        comparative_ambiguities=comps,
    )
    r = update_pef(ext, pef, user_text=None)
    assert any(e.veto_kind == held_veto for e in r.evidence)
    assert r.world_state_mutation_count == 0
    assert r.continuation_state_mutation_count == 0


def test_pronoun_candidate_binding_counts_continuation_writes() -> None:
    """Each ``resolve_pronoun`` increments continuation_state_mutation_count (xor ontology)."""
    pef = PEFState()
    emma, _ = pef.get_or_create_entity("Emma")
    emma.resolved = True
    ext = ExtractionResult(
        claims=[],
        pronoun_candidates={"she@1": "Emma"},
        span=Span.PRESENT,
    )
    r = update_pef(ext, pef, user_text=None)
    assert r.continuation_state_mutation_count == 1
    assert r.world_state_mutation_count == 0


def test_pef_admission_result_wire_dict_projector() -> None:
    """Wire dict mirrors dataclass surface for HTTP operator payloads."""
    pef = PEFState()
    r = update_pef(
        ExtractionResult(claims=[], span=Span.PRESENT),
        pef,
        user_text="What colour is Emma's book?",
        turn_act=TurnAct.QUERY,
    )
    wire = pef_admission_result_wire_dict(r)
    assert wire["decision"] == "ADMIT"
    assert "mutation_summary" in wire and wire["mutation_summary"]["ontology"] == "pef_admission_xor_v1"
    assert isinstance(wire["evidence"], list)
