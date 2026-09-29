"""Phase-3 state-native COMPARE query invariant tests.

Laws tested:
  C1. State-native is the sole authority for COMPARE answers (handled=True).
  C2. COMPARE relation_metadata stores both adjective and noun.
  C3. Zero matches → STOP with governed response.
  C4. No LLM fallback when COMPARE answers (handled=True means engine short-circuits).
"""

from __future__ import annotations

import pytest

from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.state_native_engine.contracts import (
    StateNativeDelegationResult,
    StateNativeOutcome,
    StateNativeSolverFamily,
)
from aurora_lens.state_native_engine.eval.compare import evaluate_comparative_query
from aurora_lens.state_native_engine.parse.query_surface import parse_comparative_question


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pef_with_compare(
    subject_name: str,
    comparand_name: str,
    adjective: str,
    noun: str,
    *,
    source_turn: int = 2,
    use_entity_ref: bool = True,
) -> PEFState:
    """PEF with two entities and a committed COMPARE relation."""
    pef = PEFState()
    subj_ent = Entity.create(subject_name, turn=1, session_id=pef.session_id)
    pef.add_entity(subj_ent)
    if use_entity_ref:
        comp_ent = Entity.create(comparand_name, turn=1, session_id=pef.session_id)
        pef.add_entity(comp_ent)
        comp_rel = Relationship(
            subject_id=subj_ent.id,
            relation="COMPARE",
            object_entity_id=comp_ent.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=source_turn,
            evidence=f"{subject_name}'s {noun} was {adjective} than {comparand_name}'s.",
            relation_metadata={"adjective": adjective, "noun": noun},
        )
    else:
        comp_rel = Relationship(
            subject_id=subj_ent.id,
            relation="COMPARE",
            object_entity_id=None,
            object_literal=comparand_name,
            span=Span.PRESENT,
            source_turn=source_turn,
            evidence=f"{subject_name}'s {noun} was {adjective} than {comparand_name}.",
            relation_metadata={"adjective": adjective, "noun": noun},
        )
    pef.add_relationship(comp_rel)
    pef.current_turn = source_turn + 1
    return pef


# ── Parser tests ──────────────────────────────────────────────────────────────


def test_parse_whose_dog_was_bigger():
    assert parse_comparative_question("Whose dog was bigger?") == ("dog", "bigger")


def test_parse_which_dog_was_bigger():
    assert parse_comparative_question("Which dog was bigger?") == ("dog", "bigger")


def test_parse_whose_without_question_mark():
    assert parse_comparative_question("Whose dog was bigger") == ("dog", "bigger")


def test_parse_multiword_noun():
    assert parse_comparative_question("Whose leather wallet was red?") == ("leather wallet", "red")


def test_parse_non_comparative_returns_none():
    assert parse_comparative_question("Where is James?") is None


def test_parse_inventory_question_returns_none():
    assert parse_comparative_question("Who has the wallet?") is None


def test_parse_what_question_returns_none():
    assert parse_comparative_question("What did James do?") is None


def test_parse_multi_word_adjective_returns_none():
    # "more powerful" has a space — deferred to Phase 4
    assert parse_comparative_question("Whose dog was more powerful?") is None


def test_parse_empty_returns_none():
    assert parse_comparative_question("") is None


# ── Evaluator tests ───────────────────────────────────────────────────────────


def test_whose_dog_was_bigger_reads_compare_relation():
    """Law C1: evaluate_comparative_query returns ANSWER from committed COMPARE."""
    from aurora_lens.state_native_engine.epistemic import EpistemicResult
    pef = _pef_with_compare("James", "Richard", "bigger", "dog")
    result = evaluate_comparative_query(pef, "dog", "bigger")
    assert result.outcome == StateNativeOutcome.ANSWER
    assert result.epistemic_result == EpistemicResult.VALUE
    assert "James" in result.user_visible_text
    assert "bigger" in result.user_visible_text
    assert "Richard" in result.user_visible_text
    assert result.clarify_context is None
    assert result.stop_reason_code is None


def test_compare_zero_matches_gives_stop():
    """Law C3: No COMPARE in PEF → STOP with governed response."""
    from aurora_lens.state_native_engine.epistemic import EpistemicResult
    pef = PEFState()
    result = evaluate_comparative_query(pef, "dog", "bigger")
    assert result.outcome == StateNativeOutcome.STOP
    assert result.epistemic_result == EpistemicResult.UNKNOWN
    assert result.stop_reason_code == "state_native_no_compare"
    assert "dog" in result.user_visible_text


def test_compare_adjective_mismatch_gives_stop():
    """Wrong adjective → STOP."""
    pef = _pef_with_compare("James", "Richard", "bigger", "dog")
    result = evaluate_comparative_query(pef, "dog", "longer")
    assert result.outcome == StateNativeOutcome.STOP
    assert result.stop_reason_code == "state_native_no_compare"


def test_compare_noun_mismatch_gives_stop():
    """Wrong noun → STOP."""
    pef = _pef_with_compare("James", "Richard", "bigger", "dog")
    result = evaluate_comparative_query(pef, "cat", "bigger")
    assert result.outcome == StateNativeOutcome.STOP
    assert result.stop_reason_code == "state_native_no_compare"


def test_compare_noun_key_normalization():
    """item_key normalization: COMPARE noun='dogs' matches query noun='dog'."""
    pef = _pef_with_compare("James", "Richard", "bigger", "dogs")
    result = evaluate_comparative_query(pef, "dog", "bigger")
    assert result.outcome == StateNativeOutcome.ANSWER


def test_compare_literal_comparand():
    """COMPARE with object_literal (no entity ref): response still includes comparand name."""
    pef = _pef_with_compare("James", "Richard", "bigger", "dog", use_entity_ref=False)
    result = evaluate_comparative_query(pef, "dog", "bigger")
    assert result.outcome == StateNativeOutcome.ANSWER
    assert "Richard" in result.user_visible_text


def test_compare_latest_turn_wins():
    """When two COMPARE relations match, most recent source_turn is used."""
    pef = PEFState()
    james = Entity.create("James", turn=1, session_id=pef.session_id)
    carol = Entity.create("Carol", turn=1, session_id=pef.session_id)
    richard = Entity.create("Richard", turn=1, session_id=pef.session_id)
    pef.add_entity(james)
    pef.add_entity(carol)
    pef.add_entity(richard)

    # Older relation: James bigger than Richard (turn 2)
    pef.add_relationship(Relationship(
        subject_id=james.id,
        relation="COMPARE",
        object_entity_id=richard.id,
        object_literal=None,
        span=Span.PRESENT,
        source_turn=2,
        evidence="older",
        relation_metadata={"adjective": "bigger", "noun": "dog"},
    ))
    # Newer relation: Carol bigger than Richard (turn 3)
    pef.add_relationship(Relationship(
        subject_id=carol.id,
        relation="COMPARE",
        object_entity_id=richard.id,
        object_literal=None,
        span=Span.PRESENT,
        source_turn=3,
        evidence="newer",
        relation_metadata={"adjective": "bigger", "noun": "dog"},
    ))
    pef.current_turn = 4

    result = evaluate_comparative_query(pef, "dog", "bigger")
    assert result.outcome == StateNativeOutcome.ANSWER
    assert "Carol" in result.user_visible_text   # latest turn wins


# ── Engine integration ────────────────────────────────────────────────────────


def test_compare_engine_integration_handled_true():
    """Law C1: DefaultStateNativeEngine returns handled=True for comparative question."""
    from aurora_lens.interpret.turn_act import TurnAct
    from aurora_lens.state_native_engine.contracts import StateNativeRequest
    from aurora_lens.state_native_engine.default_engine import DefaultStateNativeEngine

    pef = _pef_with_compare("James", "Richard", "bigger", "dog")
    req = StateNativeRequest(
        user_text="Whose dog was bigger?",
        pef=pef,
        turn_act=TurnAct.QUERY,
        detected_span=Span.PRESENT,
    )
    engine = DefaultStateNativeEngine()
    result = engine.evaluate(req)

    assert result.handled is True
    assert result.outcome == StateNativeOutcome.ANSWER
    assert result.solver_family == StateNativeSolverFamily.COMMITTED_COMPARE_READ
    assert "James" in result.user_visible_text
    assert "bigger" in result.user_visible_text


def test_compare_engine_stop_when_no_compare_in_pef():
    """Law C3: Engine returns handled=True, outcome=STOP when no COMPARE matches."""
    from aurora_lens.interpret.turn_act import TurnAct
    from aurora_lens.state_native_engine.contracts import StateNativeRequest
    from aurora_lens.state_native_engine.default_engine import DefaultStateNativeEngine

    pef = PEFState()
    req = StateNativeRequest(
        user_text="Whose dog was bigger?",
        pef=pef,
        turn_act=TurnAct.QUERY,
        detected_span=Span.PRESENT,
    )
    result = DefaultStateNativeEngine().evaluate(req)
    assert result.handled is True
    assert result.outcome == StateNativeOutcome.STOP
    assert result.stop_reason_code == "state_native_no_compare"


def test_compare_solver_family_correct():
    """Law C4: solver_family is COMMITTED_COMPARE_READ (not inventory, not location)."""
    from aurora_lens.interpret.turn_act import TurnAct
    from aurora_lens.state_native_engine.contracts import StateNativeRequest
    from aurora_lens.state_native_engine.default_engine import DefaultStateNativeEngine

    pef = _pef_with_compare("James", "Richard", "bigger", "dog")
    req = StateNativeRequest(
        user_text="Whose dog was bigger?",
        pef=pef,
        turn_act=TurnAct.QUERY,
        detected_span=Span.PRESENT,
    )
    result = DefaultStateNativeEngine().evaluate(req)
    assert result.solver_family == StateNativeSolverFamily.COMMITTED_COMPARE_READ


def test_compare_non_query_turn_act_not_handled():
    """Engine must only route COMPARE queries when turn_act is QUERY."""
    from aurora_lens.interpret.turn_act import TurnAct
    from aurora_lens.state_native_engine.contracts import StateNativeRequest
    from aurora_lens.state_native_engine.default_engine import DefaultStateNativeEngine

    pef = _pef_with_compare("James", "Richard", "bigger", "dog")
    req = StateNativeRequest(
        user_text="Whose dog was bigger?",
        pef=pef,
        turn_act=TurnAct.ASSERT,   # not a query
        detected_span=Span.PRESENT,
    )
    result = DefaultStateNativeEngine().evaluate(req)
    assert result.handled is False
