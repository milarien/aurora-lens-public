"""Epistemic adjudication regression tests — state-native query layer.

Five regression cases specified in the epistemic adjudication design:
1. Did Dr Patel order the medication change? → FALSE → ANSWER(No.)
2. Who ordered the medication change? → VALUE(Emma) → ANSWER
3. Where is Emma's sister? (IS overseas in PEF) → ANSWER(overseas) → PASS
4. Where is Rachel? (no committed state) → STOP → HARD_STOP
5. Ambiguous actor referent → CLARIFY / AMBIGUOUS → CONTAIN
Plus CONTRADICTED epistemic override → FORCE_REVISE.
"""

from __future__ import annotations

import copy

import pytest

from aurora_lens.govern.decision import InterventionAction
from aurora_lens.govern.state_native_mapping import governance_decision_from_state_native
from aurora_lens.interpret.turn_act import TurnAct
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.state_native_engine.contracts import (
    StateNativeDelegationResult,
    StateNativeOutcome,
    StateNativeRequest,
    StateNativeSolverFamily,
)
from aurora_lens.state_native_engine.default_engine import DefaultStateNativeEngine
from aurora_lens.state_native_engine.epistemic import EpistemicResult


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _engine() -> DefaultStateNativeEngine:
    return DefaultStateNativeEngine()


def _req(user_text: str, pef: PEFState) -> StateNativeRequest:
    return StateNativeRequest(
        user_text=user_text,
        pef=pef,
        turn_act=TurnAct.QUERY,
        detected_span=Span.PRESENT,
    )


# ─── PEF builders ────────────────────────────────────────────────────────────

def _pef_emma_ordered_no_patel() -> PEFState:
    """Emma has ORDER(medication change); Dr Patel is NOT in PEF."""
    p = PEFState(session_id="t")
    emma = Entity.create("Emma", 0, session_id="t")
    p.add_entity(emma)
    p.add_relationship(
        Relationship(
            subject_id=emma.id,
            relation="ORDER",
            object_literal="medication change",
            object_entity_id=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="Emma ordered the medication change.",
        )
    )
    return p


def _pef_emma_ordered_patel_present() -> PEFState:
    """Emma has ORDER(medication change); Dr Patel IS in PEF but has no order."""
    p = PEFState(session_id="t")
    emma = Entity.create("Emma", 0, session_id="t")
    patel = Entity.create("Dr Patel", 0, session_id="t")
    p.add_entity(emma)
    p.add_entity(patel)
    p.add_relationship(
        Relationship(
            subject_id=emma.id,
            relation="ORDER",
            object_literal="medication change",
            object_entity_id=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="Emma ordered the medication change.",
        )
    )
    return p


def _pef_emma_sister_is_overseas() -> PEFState:
    """Entity 'Emma's sister' has IS(overseas) — no AT relation."""
    p = PEFState(session_id="t")
    emma = Entity.create("Emma", 0, session_id="t")
    sister = Entity.create("Emma's sister", 0, session_id="t")
    p.add_entity(emma)
    p.add_entity(sister)
    p.add_relationship(
        Relationship(
            subject_id=sister.id,
            relation="IS",
            object_literal="overseas",
            object_entity_id=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="Emma's sister is overseas.",
        )
    )
    return p


def _pef_empty() -> PEFState:
    return PEFState(session_id="t")


def _pef_two_patels_both_ordered() -> PEFState:
    """Two 'Dr Patel' variants — both ordered something, creating ambiguity."""
    p = PEFState(session_id="t")
    p1 = Entity.create("Dr Patel Singh", 0, session_id="t")
    p2 = Entity.create("Dr Patel Kumar", 0, session_id="t")
    p.add_entity(p1)
    p.add_entity(p2)
    for ent in (p1, p2):
        p.add_relationship(
            Relationship(
                subject_id=ent.id,
                relation="ORDER",
                object_literal="medication change",
                object_entity_id=None,
                span=Span.PRESENT,
                source_turn=1,
                evidence=f"{ent.name} ordered the medication change.",
            )
        )
    return p


# ─── Case 1: Existence FALSE ──────────────────────────────────────────────────

class TestExistenceQueryFalse:
    """Did Dr Patel order the medication change? → FALSE → No."""

    def test_false_when_actor_absent_from_pef(self) -> None:
        result = _engine().evaluate(
            _req("Did Dr Patel order the medication change?", _pef_emma_ordered_no_patel())
        )
        assert result.handled
        assert result.outcome == StateNativeOutcome.ANSWER
        assert result.epistemic_result == EpistemicResult.FALSE
        assert result.user_visible_text.startswith("No.")

    def test_false_when_actor_in_pef_but_did_not_act(self) -> None:
        result = _engine().evaluate(
            _req(
                "Did Dr Patel order the medication change?",
                _pef_emma_ordered_patel_present(),
            )
        )
        assert result.handled
        assert result.outcome == StateNativeOutcome.ANSWER
        assert result.epistemic_result == EpistemicResult.FALSE
        assert result.user_visible_text.startswith("No.")

    def test_governance_maps_false_to_pass_with_epistemic_suffix(self) -> None:
        result = _engine().evaluate(
            _req("Did Dr Patel order the medication change?", _pef_emma_ordered_no_patel())
        )
        decision = governance_decision_from_state_native(result)
        assert decision.action == InterventionAction.PASS
        assert ":false" in decision.rationale


# ─── Case 2: Actor-value VALUE ────────────────────────────────────────────────

class TestActorValueQuery:
    """Who ordered the medication change? → VALUE(Emma) → PASS."""

    def test_value_epistemic_for_actor_query(self) -> None:
        result = _engine().evaluate(
            _req("Who ordered the medication change?", _pef_emma_ordered_no_patel())
        )
        assert result.handled
        assert result.outcome == StateNativeOutcome.ANSWER
        assert result.epistemic_result == EpistemicResult.VALUE
        assert result.user_visible_text == "Emma ordered the medication change."

    def test_governance_maps_value_to_pass_with_epistemic_suffix(self) -> None:
        result = _engine().evaluate(
            _req("Who ordered the medication change?", _pef_emma_ordered_no_patel())
        )
        decision = governance_decision_from_state_native(result)
        assert decision.action == InterventionAction.PASS
        assert ":value" in decision.rationale


# ─── Case 3: Location IS-fallback → VALUE → PASS ─────────────────────────────

class TestLocationISFallback:
    """Where is Emma's sister? (IS overseas) → ANSWER(overseas) → PASS."""

    def test_is_predicate_serves_location_answer(self) -> None:
        result = _engine().evaluate(
            _req("Where is Emma's sister?", _pef_emma_sister_is_overseas())
        )
        assert result.handled
        assert result.outcome == StateNativeOutcome.ANSWER
        assert "overseas" in result.user_visible_text

    def test_governance_maps_location_answer_to_pass(self) -> None:
        result = _engine().evaluate(
            _req("Where is Emma's sister?", _pef_emma_sister_is_overseas())
        )
        decision = governance_decision_from_state_native(result)
        assert decision.action == InterventionAction.PASS


# ─── Case 4: Location UNKNOWN → HARD_STOP ────────────────────────────────────

class TestLocationUnknownStop:
    """Where is Rachel? (no committed state) → STOP → HARD_STOP."""

    def test_no_committed_entity_yields_stop(self) -> None:
        result = _engine().evaluate(_req("Where is Rachel?", _pef_empty()))
        assert result.handled
        assert result.outcome == StateNativeOutcome.STOP

    def test_governance_maps_unknown_location_to_hard_stop(self) -> None:
        result = _engine().evaluate(_req("Where is Rachel?", _pef_empty()))
        decision = governance_decision_from_state_native(result)
        assert decision.action == InterventionAction.HARD_STOP
        assert decision.interaction_open is True
        assert decision.pathway_id == "P_STOP_REFUSE_CLEAN"


# ─── Case 5: Ambiguous actor → AMBIGUOUS → CONTAIN ───────────────────────────

class TestAmbiguousActorReferent:
    """Ambiguous 'Dr Patel' phrase → CLARIFY / AMBIGUOUS → CONTAIN."""

    def test_ambiguous_actor_yields_clarify_with_ambiguous_epistemic(self) -> None:
        result = _engine().evaluate(
            _req(
                "Did Dr Patel order the medication change?",
                _pef_two_patels_both_ordered(),
            )
        )
        assert result.handled
        assert result.outcome == StateNativeOutcome.CLARIFY
        assert result.epistemic_result == EpistemicResult.AMBIGUOUS

    def test_governance_maps_ambiguous_to_contain(self) -> None:
        result = _engine().evaluate(
            _req(
                "Did Dr Patel order the medication change?",
                _pef_two_patels_both_ordered(),
            )
        )
        decision = governance_decision_from_state_native(result)
        assert decision.action == InterventionAction.CONTAIN


# ─── CONTRADICTED override → FORCE_REVISE ────────────────────────────────────

class TestContradictedEpistemicOverride:
    """CONTRADICTED epistemic_result overrides ANSWER outcome → FORCE_REVISE."""

    def test_contradicted_overrides_answer_to_force_revise(self) -> None:
        result = StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.ANSWER,
            user_visible_text="PEF state contradicts this claim.",
            solver_family=StateNativeSolverFamily.COMMITTED_ACTION_AGENT_READ,
            epistemic_result=EpistemicResult.CONTRADICTED,
        )
        decision = governance_decision_from_state_native(result)
        assert decision.action == InterventionAction.FORCE_REVISE

    def test_contradicted_rationale_contains_epistemic_marker(self) -> None:
        result = StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.ANSWER,
            user_visible_text="Contradiction detected.",
            solver_family=StateNativeSolverFamily.COMMITTED_LOCATION_READ,
            epistemic_result=EpistemicResult.CONTRADICTED,
        )
        decision = governance_decision_from_state_native(result)
        assert "contradicted" in decision.rationale


class TestLane1GhostContinuity:
    """SNC-1: state-native must read from pre-commit PEF snapshot, not post-commit self._pef.

    Regression test for the constitutional boundary condition:
    a same-turn claim committed to PEF must not become query substrate before
    governance completion (admissibility-before-consequence invariant).
    """

    def test_state_native_does_not_answer_from_same_turn_at_claim(self) -> None:
        """Mixed turn: 'Alice went to the store. Where is Alice?'

        State-native receives the pre-commit snapshot (Alice has no AT relation).
        It must return STOP/UNKNOWN, not the same-turn AT(store) claim.
        """
        pef_start = PEFState(session_id="snc1")
        alice = Entity.create("Alice", 0, session_id="snc1")
        pef_start.add_entity(alice)

        # Capture pre-commit snapshot (Alice exists, no AT relation yet).
        pef_snapshot = copy.deepcopy(pef_start)

        # Simulate _commit_extraction_if_admissible adding AT(Alice, store) to live PEF.
        pef_start.add_relationship(
            Relationship(
                subject_id=alice.id,
                relation="AT",
                object_literal="the store",
                object_entity_id=None,
                span=Span.PRESENT,
                source_turn=1,
                evidence="Alice went to the store.",
            )
        )

        # State-native must use the snapshot (pre-commit), not pef_start (post-commit).
        req = StateNativeRequest(
            user_text="Where is Alice?",
            pef=pef_snapshot,
            turn_act=TurnAct.QUERY,
            detected_span=Span.PRESENT,
        )
        result = _engine().evaluate(req)

        assert result.handled is True
        assert result.outcome == StateNativeOutcome.STOP
        assert result.epistemic_result == EpistemicResult.UNKNOWN
        assert "the store" not in (result.user_visible_text or "").lower()

    def test_state_native_answers_from_prior_turn_at_claim(self) -> None:
        """When AT(store) was committed in a prior turn, state-native correctly answers."""
        pef_prior = PEFState(session_id="snc1b")
        alice = Entity.create("Alice", 0, session_id="snc1b")
        pef_prior.add_entity(alice)
        pef_prior.add_relationship(
            Relationship(
                subject_id=alice.id,
                relation="AT",
                object_literal="the store",
                object_entity_id=None,
                span=Span.PRESENT,
                source_turn=1,
                evidence="Alice is at the store.",
            )
        )

        req = StateNativeRequest(
            user_text="Where is Alice?",
            pef=pef_prior,
            turn_act=TurnAct.QUERY,
            detected_span=Span.PRESENT,
        )
        result = _engine().evaluate(req)

        assert result.handled is True
        assert result.outcome == StateNativeOutcome.ANSWER
        assert result.epistemic_result == EpistemicResult.VALUE
        assert "store" in result.user_visible_text.lower()
