"""Contract: tuple-path booleans never infer epistemics from ANSWER alone.

Regression coverage for inventory ``Does … have …?``, ``What does X have?``,
location ``Is … still in …?``, UNKNOWN stops, VALUE quantity/where queries,
and governance mapping for FALSE answers.

``evaluate_temporal_contact`` is **out of scope** here — PEF temporal semantics
need a separate audit before tightening epistemics.
"""

from __future__ import annotations

from aurora_lens.govern.decision import InterventionAction
from aurora_lens.govern.state_native_mapping import governance_decision_from_state_native
from aurora_lens.interpret.turn_act import TurnAct
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.state_native_engine.contracts import (
    StateNativeOutcome,
    StateNativeRequest,
)
from aurora_lens.state_native_engine.default_engine import DefaultStateNativeEngine
from aurora_lens.state_native_engine.epistemic import EpistemicResult


def _engine() -> DefaultStateNativeEngine:
    return DefaultStateNativeEngine()


def _req(user_text: str, pef: PEFState) -> StateNativeRequest:
    return StateNativeRequest(
        user_text=user_text,
        pef=pef,
        turn_act=TurnAct.QUERY,
        detected_span=Span.PRESENT,
    )


def _pef_richard_entity_only() -> PEFState:
    p = PEFState(session_id="t")
    r = Entity.create("Richard", 0, session_id="t")
    p.add_entity(r)
    return p


def _pef_richard_negated_six_eggs() -> PEFState:
    p = PEFState(session_id="t")
    r = Entity.create("Richard", 0, session_id="t")
    p.add_entity(r)
    p.add_relationship(
        Relationship(
            subject_id=r.id,
            relation="HAS",
            object_literal="six eggs",
            object_entity_id=None,
            span=Span.PRESENT,
            source_turn=2,
            evidence="user",
            negated=True,
        )
    )
    return p


def _pef_richard_positive_six_eggs() -> PEFState:
    p = PEFState(session_id="t")
    r = Entity.create("Richard", 0, session_id="t")
    p.add_entity(r)
    p.add_relationship(
        Relationship(
            subject_id=r.id,
            relation="HAS",
            object_literal="6 eggs",
            object_entity_id=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="user",
            negated=False,
        )
    )
    return p


def _pef_key_at_safe() -> PEFState:
    p = PEFState(session_id="t")
    key = Entity.create("gold key", 0, session_id="t")
    safe = Entity.create("safe", 0, session_id="t")
    p.add_entity(key)
    p.add_entity(safe)
    p.add_relationship(
        Relationship(
            subject_id=key.id,
            relation="AT",
            object_entity_id=safe.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="in the safe",
        )
    )
    return p


def _pef_bob_negated_book_alice_holder() -> PEFState:
    """Bob's latest HAS for the book is negated; Alice holds the book."""
    p = PEFState(session_id="t")
    bob = Entity.create("Bob", 0, session_id="t")
    alice = Entity.create("Alice", 0, session_id="t")
    p.add_entity(bob)
    p.add_entity(alice)
    p.add_relationship(
        Relationship(
            subject_id=bob.id,
            relation="HAS",
            object_literal="the book",
            object_entity_id=None,
            span=Span.PRESENT,
            source_turn=2,
            evidence="transfer",
            negated=True,
        )
    )
    p.add_relationship(
        Relationship(
            subject_id=alice.id,
            relation="HAS",
            object_literal="the book",
            object_entity_id=None,
            span=Span.PRESENT,
            source_turn=3,
            evidence="received",
            negated=False,
        )
    )
    return p


class TestInventoryForSubjectEpistemics:
    """``What does X have?`` — explicit VALUE / FALSE / UNKNOWN."""

    def test_listed_holdings_are_value_not_inferred_ambiguously(self) -> None:
        result = _engine().evaluate(
            _req("What does Richard have?", _pef_richard_positive_six_eggs())
        )
        assert result.handled
        assert result.outcome == StateNativeOutcome.ANSWER
        assert result.epistemic_result == EpistemicResult.VALUE
        assert "Richard" in result.user_visible_text
        assert "egg" in result.user_visible_text.lower()

    def test_confirmed_transfer_absence_is_false(self) -> None:
        result = _engine().evaluate(
            _req("What does Bob have?", _pef_bob_negated_book_alice_holder())
        )
        assert result.handled
        assert result.outcome == StateNativeOutcome.ANSWER
        assert result.epistemic_result == EpistemicResult.FALSE
        assert "does not have" in result.user_visible_text.lower()
        decision = governance_decision_from_state_native(result)
        assert decision.action == InterventionAction.PASS
        assert ":false" in decision.rationale

    def test_no_inventory_evidence_is_unknown(self) -> None:
        result = _engine().evaluate(
            _req("What does Richard have?", _pef_richard_entity_only())
        )
        assert result.handled
        assert result.outcome == StateNativeOutcome.STOP
        assert result.epistemic_result == EpistemicResult.UNKNOWN
        assert result.stop_reason_code == "state_native_no_inventory"


class TestBooleanEpistemicContract:

    def test_inventory_negative_is_false_not_value(self) -> None:
        result = _engine().evaluate(
            _req("Does Richard have six eggs?", _pef_richard_negated_six_eggs())
        )
        assert result.handled
        assert result.outcome == StateNativeOutcome.ANSWER
        assert result.epistemic_result == EpistemicResult.FALSE
        assert result.user_visible_text.startswith("No, Richard")

    def test_location_negative_still_in_is_false_not_value(self) -> None:
        result = _engine().evaluate(
            _req(
                "Is gold key still in the desk?",
                _pef_key_at_safe(),
            )
        )
        assert result.handled
        assert result.outcome == StateNativeOutcome.ANSWER
        assert result.epistemic_result == EpistemicResult.FALSE
        # Do not pin fixture literals (place names / item labels) inside user-visible text;
        # epistemic FALSE vs VALUE is the contract here.

    def test_unknown_inventory_and_location(self) -> None:
        inv = _engine().evaluate(
            _req("Does Richard have six eggs?", _pef_richard_entity_only())
        )
        assert inv.handled
        assert inv.outcome == StateNativeOutcome.STOP
        assert inv.epistemic_result == EpistemicResult.UNKNOWN
        assert inv.stop_reason_code == "state_native_no_inventory"

        loc = _engine().evaluate(
            _req("Where is Zephyr Alpha?", _pef_richard_entity_only())
        )
        assert loc.handled
        assert loc.outcome == StateNativeOutcome.STOP
        assert loc.epistemic_result == EpistemicResult.UNKNOWN
        assert loc.stop_reason_code == "state_native_unknown_entity"

    def test_value_queries_remain_value(self) -> None:
        how = _engine().evaluate(
            _req(
                "How many eggs does Richard have?",
                _pef_richard_positive_six_eggs(),
            )
        )
        assert how.handled
        assert how.outcome == StateNativeOutcome.ANSWER
        assert how.epistemic_result == EpistemicResult.VALUE

        where = _engine().evaluate(
            _req("Where is gold key?", _pef_key_at_safe())
        )
        assert where.handled
        assert where.outcome == StateNativeOutcome.ANSWER
        assert where.epistemic_result == EpistemicResult.VALUE

    def test_location_positive_still_in_is_true(self) -> None:
        result = _engine().evaluate(
            _req("Is gold key still in the safe?", _pef_key_at_safe())
        )
        assert result.handled
        assert result.outcome == StateNativeOutcome.ANSWER
        assert result.epistemic_result == EpistemicResult.TRUE
        assert result.user_visible_text.lower().startswith("yes:")

    def test_governance_false_answers_still_pass_with_false_suffix(self) -> None:
        inv = _engine().evaluate(
            _req("Does Richard have six eggs?", _pef_richard_negated_six_eggs())
        )
        loc = _engine().evaluate(
            _req("Is gold key still in the desk?", _pef_key_at_safe())
        )
        for sn in (inv, loc):
            d = governance_decision_from_state_native(sn)
            assert d.action == InterventionAction.PASS
            assert ":false" in d.rationale
