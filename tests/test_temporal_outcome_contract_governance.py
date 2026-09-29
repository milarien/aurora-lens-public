"""Phase 6: present-bound Temporal Outcome Contract — governance + Lens regressions.

Proves MISSING / UNKNOWN never route through generic UNKNOWN epistemics or ANSWER/PASS where
disallowed, and ambiguity / projection / reconstruction / superseded retain contract semantics.

Does not introduce temporal evaluator features beyond assertions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aurora_lens.adapters.base import AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.govern.state_native_mapping import governance_decision_from_state_native
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractionResult
from aurora_lens.lens import Lens
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.pef.temporal_commitment import PresentTemporalCommitment
from aurora_lens.state_native_engine.contracts import StateNativeOutcome
from aurora_lens.state_native_engine.contracts import (
    StateNativeDelegationResult,
    StateNativeRequest,
    StateNativeSolverFamily,
)
from aurora_lens.state_native_engine.epistemic import EpistemicResult
from aurora_lens.state_native_engine.eval.temporal_contact import (
    delegation_result_for_temporal_contact_query,
    evaluate_temporal_contact,
    present_bound_temporal_eval_to_delegation,
)
from aurora_lens.state_native_engine.temporal_eval_result import (
    PresentBoundTemporalEvalResult,
    TemporalGovernanceCue,
)
from aurora_lens.state_native_engine.temporal_outcome_contract import (
    TEMPORAL_OUTCOME_CONTRACT_VALUES,
    TemporalOutcomeContract,
)
from aurora_lens.verify.flags import FlagType


class _EmptyExtractBackend(ExtractionBackend):
    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:  # noqa: ARG002
        return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)


class _RecordingAdapter:
    def __init__(self) -> None:
        self.generate = AsyncMock(
            return_value=AdapterResponse(text="MODEL_SHOULD_NOT_RUN", model="mock"),
        )


class _FixedStateNativeEngine:
    def __init__(self, result: StateNativeDelegationResult) -> None:
        self._result = result

    def evaluate(self, req: StateNativeRequest) -> StateNativeDelegationResult:  # noqa: ARG002
        return self._result


def _pef_subject_named(name: str) -> tuple[PEFState, Entity]:
    p = PEFState(session_id="t")
    e = Entity.create(name, 0, session_id="t")
    p.add_entity(e)
    p.add_relationship(
        Relationship(
            subject_id=e.id,
            relation="AT",
            object_entity_id=None,
            object_literal="fixture loc",
            span=Span.PRESENT,
            source_turn=1,
            evidence="fixture",
        )
    )
    return p, e


def test_guard_temporal_contract_wire_values_are_not_generic_unknown() -> None:
    unk = EpistemicResult.UNKNOWN.value
    assert unk == "unknown"
    assert unk not in TEMPORAL_OUTCOME_CONTRACT_VALUES
    for m in TemporalOutcomeContract:
        assert m.value.lower() != unk
        assert m.value != unk
    assert "unknown" not in {m.value for m in TemporalOutcomeContract}


def test_temporal_missing_contract_and_no_unknown_epistemic_or_answer_pass_path() -> None:
    p, e = _pef_subject_named("Rae")
    ev = evaluate_temporal_contact(p, e.name)
    assert ev.outcome_kind == TemporalOutcomeContract.TEMPORAL_MISSING
    assert ev.governance_cue == TemporalGovernanceCue.ASK_TEMPORAL_SCOPE
    sn = present_bound_temporal_eval_to_delegation(ev)
    assert sn.handled is True
    assert sn.outcome == StateNativeOutcome.CLARIFY
    assert sn.epistemic_result != EpistemicResult.UNKNOWN
    assert sn.epistemic_result != EpistemicResult.VALUE
    assert sn.outcome != StateNativeOutcome.ANSWER
    gd = governance_decision_from_state_native(sn)
    assert gd.action == InterventionAction.CONTAIN
    assert gd.action != InterventionAction.PASS


@pytest.mark.asyncio
async def test_temporal_missing_lens_contain_never_answer_pass_unknown_epistemic() -> None:
    adapter = _RecordingAdapter()
    p, ent = _pef_subject_named("Blair")
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=p,
        session_id="t",
    )
    r = await lens.process(f"When should I contact {ent.name}?")
    assert r.action == InterventionAction.CONTAIN
    assert r.decision is not None
    assert any(
        f.flag_type == FlagType.STATE_NATIVE_TEMPORAL_ANCHOR_MISSING
        for f in r.decision.flags
    )
    assert r.action != InterventionAction.PASS
    assert r.decision.rationale.startswith("state_native_clarify:")
    assert ":unknown" not in r.decision.rationale
    adapter.generate.assert_not_called()


def test_temporal_unknown_contract_stop_no_answer_pass_or_value_epistemic() -> None:
    p, ent = _pef_subject_named("Kim")
    p.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.TEMPORAL_UNKNOWN,
            source_turn=3,
            evidence="fixture",
            subject_entity_id=ent.id,
            continuity_payload={"anchor_surface": "after discharge planning"},
        )
    )
    ev = evaluate_temporal_contact(p, ent.name)
    assert ev.outcome_kind == TemporalOutcomeContract.TEMPORAL_UNKNOWN
    assert ev.governance_cue == TemporalGovernanceCue.GOVERNED_NON_ANSWER
    assert ev.temporal_anchor_surface == "after discharge planning"
    sn = present_bound_temporal_eval_to_delegation(ev)
    assert sn.outcome == StateNativeOutcome.STOP
    assert sn.epistemic_result is None
    assert sn.epistemic_result is not EpistemicResult.UNKNOWN
    assert sn.outcome != StateNativeOutcome.ANSWER
    gd = governance_decision_from_state_native(sn)
    assert gd.action == InterventionAction.HARD_STOP
    assert gd.action != InterventionAction.PASS
    assert gd.rationale.startswith("state_native_stop:state_native_temporal_unknown")
    assert not gd.rationale.endswith(f":{EpistemicResult.UNKNOWN.value}")


@pytest.mark.asyncio
async def test_temporal_unknown_lens_hard_stop_not_pass() -> None:
    adapter = _RecordingAdapter()
    p, ent = _pef_subject_named("Quinn")
    p.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.TEMPORAL_UNKNOWN,
            source_turn=2,
            evidence="fixture",
            subject_entity_id=ent.id,
            continuity_payload={"anchor_surface": "post-handoff window"},
        )
    )
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=p,
        session_id="t",
    )
    r = await lens.process(f"When should I contact {ent.name}?")
    assert r.action == InterventionAction.HARD_STOP
    assert r.action != InterventionAction.PASS
    assert r.decision is not None
    assert "state_native_temporal_unknown" in (r.decision.rationale or "")
    assert r.decision.rationale is not None and not r.decision.rationale.endswith(
        f":{EpistemicResult.UNKNOWN.value}"
    )
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_temporal_ambiguous_lens_routes_to_contain_without_adapter() -> None:
    adapter = _RecordingAdapter()
    p, ent = _pef_subject_named("Avery")
    p.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.TEMPORAL_AMBIGUOUS,
            source_turn=4,
            evidence="fixture",
            subject_entity_id=ent.id,
            continuity_payload={"candidate_anchors": ["Tuesday block", "Friday handoff"]},
        )
    )
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=p,
        session_id="t",
    )
    r = await lens.process(f"When should I contact {ent.name}?")
    assert r.action == InterventionAction.CONTAIN
    assert r.decision is not None
    assert r.decision.rationale.startswith("state_native_clarify:")
    adapter.generate.assert_not_called()


def test_temporal_ambiguous_explicit_marker_no_single_anchor_lens_asks_scope() -> None:
    p, ent = _pef_subject_named("Jay")
    p.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.TEMPORAL_AMBIGUOUS,
            source_turn=4,
            evidence="fixture",
            subject_entity_id=ent.id,
            continuity_payload={
                "candidate_anchors": ["Tuesday block", "post-clinic follow-up"],
            },
        )
    )
    ev = evaluate_temporal_contact(p, ent.name)
    assert ev.outcome_kind == TemporalOutcomeContract.TEMPORAL_AMBIGUOUS
    assert ev.temporal_anchor_surface is None
    sn = delegation_result_for_temporal_contact_query(p, ent.name)
    assert sn is not None and sn.outcome == StateNativeOutcome.CLARIFY
    assert sn.epistemic_result == EpistemicResult.AMBIGUOUS
    assert sn.clarify_context is not None
    cand = sn.clarify_context.get("candidate_entities") or []
    assert len(cand) >= 2
    gd = governance_decision_from_state_native(sn)
    assert gd.action == InterventionAction.CONTAIN


def test_temporal_ambiguous_competing_projections_surfaces_both_not_collapsed() -> None:
    p, ent = _pef_subject_named("Rio")
    t = 7
    for surf in ("after lab draw", "once nursing signs off"):
        p.add_present_temporal_commitment(
            PresentTemporalCommitment(
                outcome_kind=TemporalOutcomeContract.FUTURE_PROJECTION,
                source_turn=t,
                evidence="fixture",
                subject_entity_id=ent.id,
                continuity_payload={"surface": surf},
            )
        )
    ev = evaluate_temporal_contact(p, ent.name)
    assert ev.outcome_kind == TemporalOutcomeContract.TEMPORAL_AMBIGUOUS
    assert ev.temporal_anchor_surface is None
    surfs = ev.evidence_payload.get("projection_surfaces") or []
    assert isinstance(surfs, list) and len(surfs) >= 2
    sn = present_bound_temporal_eval_to_delegation(ev)
    assert sn.clarify_context is not None
    extras = sn.clarify_context.get("temporal_candidate_surfaces") or []
    assert len(extras) >= 2


def test_future_projection_contract_answer_path_not_accomplished_future_fact() -> None:
    p, ent = _pef_subject_named("Lee")
    p.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.FUTURE_PROJECTION,
            source_turn=5,
            evidence="committed intent / plan",
            subject_entity_id=ent.id,
            continuity_payload={"surface": "after rounding completes"},
        )
    )
    ev = evaluate_temporal_contact(p, ent.name)
    assert ev.outcome_kind == TemporalOutcomeContract.FUTURE_PROJECTION
    sn = delegation_result_for_temporal_contact_query(p, ent.name)
    assert sn is not None
    assert sn.outcome == StateNativeOutcome.ANSWER
    txt = sn.user_visible_text.lower()
    assert "projection" in txt
    assert "not accomplished fact" in txt
    assert "present projection" in txt
    gd = governance_decision_from_state_native(sn)
    assert gd.action == InterventionAction.PASS


def test_past_reconstruction_is_present_continuity_not_timeline_storage() -> None:
    p, ent = _pef_subject_named("Sage")
    p.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.PAST_RECONSTRUCTION,
            source_turn=2,
            evidence="continuity fixture",
            subject_entity_id=ent.id,
            continuity_payload={
                "reconstruction_surface": "post-admission handoff summary",
                "basis_relation": "TELL",
            },
        )
    )
    ev = evaluate_temporal_contact(p, ent.name)
    assert ev.outcome_kind == TemporalOutcomeContract.PAST_RECONSTRUCTION
    assert ev.governance_cue == TemporalGovernanceCue.PRESENT_COMMITTED_ANSWER
    assert "surviving present continuity" in ev.rationale.lower()
    assert "reconstruction" in ev.rationale.lower()
    sn = present_bound_temporal_eval_to_delegation(ev)
    assert sn.outcome == StateNativeOutcome.ANSWER
    lowered = sn.user_visible_text.lower()
    assert "reconstruction" in lowered or "surviving" in lowered
    assert "not an archival replay" in lowered
    gd = governance_decision_from_state_native(sn)
    assert gd.action == InterventionAction.PASS


@pytest.mark.asyncio
async def test_future_projection_lens_routes_to_pass_without_adapter() -> None:
    adapter = _RecordingAdapter()
    p, ent = _pef_subject_named("Lee")
    p.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.FUTURE_PROJECTION,
            source_turn=5,
            evidence="committed intent / plan",
            subject_entity_id=ent.id,
            continuity_payload={"surface": "after rounding completes"},
        )
    )
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=p,
        session_id="t",
    )
    r = await lens.process(f"When should I contact {ent.name}?")
    assert r.action == InterventionAction.PASS
    assert r.decision is not None
    assert r.decision.rationale.startswith("state_native_answer")
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_past_reconstruction_lens_routes_to_pass_without_adapter() -> None:
    adapter = _RecordingAdapter()
    p, ent = _pef_subject_named("Sage")
    p.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.PAST_RECONSTRUCTION,
            source_turn=2,
            evidence="continuity fixture",
            subject_entity_id=ent.id,
            continuity_payload={
                "reconstruction_surface": "post-admission handoff summary",
                "basis_relation": "TELL",
            },
        )
    )
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=p,
        session_id="t",
    )
    r = await lens.process(f"When should I contact {ent.name}?")
    assert r.action == InterventionAction.PASS
    assert r.decision is not None
    assert r.decision.rationale.startswith("state_native_answer")
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_lens_uses_present_bound_temporal_cue_to_route_ask_when_available() -> None:
    """Temporal eval cue is consumed by Lens when present, even if outcome shape drifts."""
    adapter = _RecordingAdapter()
    temporal_eval = PresentBoundTemporalEvalResult(
        outcome_kind=TemporalOutcomeContract.TEMPORAL_MISSING,
        governance_cue=TemporalGovernanceCue.ASK_TEMPORAL_SCOPE,
        rationale="fixture temporal missing anchor",
        evidence_payload={"subject_display": "Blair"},
    )
    forced_answer_shape = StateNativeDelegationResult(
        handled=True,
        outcome=StateNativeOutcome.ANSWER,
        user_visible_text="Need temporal scope.",
        clarify_context={
            "failed_constraint": "STATE_NATIVE_TEMPORAL_ANCHOR_MISSING",
            "subject_phrase": "Blair",
            "temporal_outcome_contract": TemporalOutcomeContract.TEMPORAL_MISSING.value,
        },
        stop_reason_code=None,
        solver_family=StateNativeSolverFamily.COMMITTED_TEMPORAL_CONTACT_READ,
        temporal_eval_result=temporal_eval,
    )
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
            state_native_engine=_FixedStateNativeEngine(forced_answer_shape),
        ),
        initial_pef=PEFState(session_id="t"),
        session_id="t",
    )
    r = await lens.process("When should I contact Blair?")
    assert r.action == InterventionAction.CONTAIN
    assert r.decision is not None
    assert r.decision.pathway_id == "P_ASK_MISSING_FACT"
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
async def test_lens_preserves_existing_state_native_mapping_without_temporal_eval() -> None:
    """No temporal eval payload -> preserve existing ANSWER/PASS behavior."""
    adapter = _RecordingAdapter()
    answer_only = StateNativeDelegationResult(
        handled=True,
        outcome=StateNativeOutcome.ANSWER,
        user_visible_text="Contact Blair after rounds.",
        clarify_context=None,
        stop_reason_code=None,
        solver_family=StateNativeSolverFamily.COMMITTED_TEMPORAL_CONTACT_READ,
        temporal_eval_result=None,
    )
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
            state_native_engine=_FixedStateNativeEngine(answer_only),
        ),
        initial_pef=PEFState(session_id="t"),
        session_id="t",
    )
    r = await lens.process("When should I contact Blair?")
    assert r.action == InterventionAction.PASS
    adapter.generate.assert_not_called()


def test_superseded_not_simple_boolean_false_under_governance() -> None:
    p, ent = _pef_subject_named("Morgan")
    p.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.SUPERSEDED,
            source_turn=9,
            evidence="replacement posture",
            subject_entity_id=ent.id,
            continuity_payload={
                "current_projection_surface": "after handoff checklist",
                "supersedes_commitment_id": "prior-proj-row",
            },
        )
    )
    ev = evaluate_temporal_contact(p, ent.name)
    assert ev.outcome_kind == TemporalOutcomeContract.SUPERSEDED
    sn = present_bound_temporal_eval_to_delegation(ev)
    assert sn.outcome == StateNativeOutcome.ANSWER
    assert sn.epistemic_result is None
    assert sn.epistemic_result is not EpistemicResult.FALSE
    low = sn.user_visible_text.lower()
    assert ("false" in low or "incorrect" in low) is False
    gd = governance_decision_from_state_native(sn)
    assert gd.action == InterventionAction.PASS


@pytest.mark.asyncio
async def test_superseded_lens_routes_to_pass_without_adapter() -> None:
    adapter = _RecordingAdapter()
    p, ent = _pef_subject_named("Morgan")
    p.add_present_temporal_commitment(
        PresentTemporalCommitment(
            outcome_kind=TemporalOutcomeContract.SUPERSEDED,
            source_turn=9,
            evidence="replacement posture",
            subject_entity_id=ent.id,
            continuity_payload={
                "current_projection_surface": "after handoff checklist",
                "supersedes_commitment_id": "prior-proj-row",
            },
        )
    )
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            auto_verify=False,
        ),
        initial_pef=p,
        session_id="t",
    )
    r = await lens.process(f"When should I contact {ent.name}?")
    assert r.action == InterventionAction.PASS
    assert r.decision is not None
    assert r.decision.rationale.startswith("state_native_answer")
    adapter.generate.assert_not_called()
