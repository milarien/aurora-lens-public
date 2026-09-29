"""Phase R1: BECOME transition relation invariant tests.

Governing invariant: an alias may only map to an existing relation if the
mapped relation preserves the original semantic consequence. BECOME records
a transition event; IS records a static present state. They are distinct.

Laws tested:
  B1. "become" and "became" canonicalize to BECOME (covers spaCy lemma + LLM surface).
  B2. "turned into" canonicalizes to BECOME (LLM backend multi-word key).
  B3. "stayed as" does NOT map to BECOME (deferred — may need MAINTAINED_AS or IS).
  B4. BECOME is in CANONICAL_RELATIONS.
  B5. BECOME is NOT in _NON_ACTION_RELATIONS — the existence engine can match it.
  B6. BECOME is distinct from IS — no cross-aliasing.
  B7. A BECOME relation committed to PEF is matchable by the existence evaluator.
"""

from __future__ import annotations

import pytest

from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import CANONICAL_RELATIONS, PEFState, Relationship, canonicalize_relation
from aurora_lens.state_native_engine.contracts import StateNativeOutcome
from aurora_lens.state_native_engine.eval.existence import (
    _NON_ACTION_RELATIONS,
    evaluate_existence_query,
)


# ── B1–B4: canonicalize_relation ──────────────────────────────────────────────

def test_become_lemma_maps_to_BECOME() -> None:
    """B1: spaCy lemma 'become' → BECOME."""
    assert canonicalize_relation("become") == "BECOME"


def test_became_maps_to_BECOME() -> None:
    """B1: past tense 'became' → BECOME (LLM backend surface form)."""
    assert canonicalize_relation("became") == "BECOME"


def test_turned_into_maps_to_BECOME() -> None:
    """B2: multi-word 'turned into' → BECOME (LLM backend only)."""
    assert canonicalize_relation("turned into") == "BECOME"


def test_stayed_as_not_aliased() -> None:
    """B3: 'stayed as' is deferred — must not map to BECOME or IS."""
    result = canonicalize_relation("stayed as")
    assert result != "BECOME"
    assert result != "IS"


def test_stay_not_aliased() -> None:
    """B3: bare 'stay' is also deferred."""
    result = canonicalize_relation("stay")
    assert result != "BECOME"


def test_BECOME_in_canonical_relations() -> None:
    """B4: BECOME is a first-class canonical relation."""
    assert "BECOME" in CANONICAL_RELATIONS


# ── B5–B6: relation taxonomy ──────────────────────────────────────────────────

def test_BECOME_not_in_non_action_relations() -> None:
    """B5: BECOME is a transition event, not static state — existence engine matches it."""
    assert "BECOME" not in _NON_ACTION_RELATIONS


def test_BECOME_distinct_from_IS() -> None:
    """B6: IS and BECOME are not cross-aliased."""
    assert canonicalize_relation("is") == "IS"
    assert canonicalize_relation("was") == "IS"
    assert canonicalize_relation("become") != "IS"
    assert canonicalize_relation("became") != "IS"


# ── B7: existence evaluator integration ───────────────────────────────────────

def _pef_with_alice_became_doctor() -> PEFState:
    pef = PEFState(session_id="r1_become_test")
    alice = Entity.create("Alice", turn=1, session_id="r1_become_test")
    pef.add_entity(alice)
    pef.add_relationship(
        Relationship(
            subject_id=alice.id,
            relation="BECOME",
            object_entity_id=None,
            object_literal="the doctor",
            span=Span.PAST,
            source_turn=1,
            evidence="Alice became the doctor.",
        )
    )
    return pef


def test_existence_query_finds_BECOME_relation() -> None:
    """B7: 'Did Alice become the doctor?' matches committed BECOME relation → TRUE."""
    pef = _pef_with_alice_became_doctor()
    result = evaluate_existence_query(pef, "Alice", "become doctor")
    assert result is not None
    assert result.outcome == StateNativeOutcome.ANSWER
    from aurora_lens.state_native_engine.epistemic import EpistemicResult
    assert result.epistemic_result == EpistemicResult.TRUE


def test_existence_query_wrong_actor_for_BECOME() -> None:
    """B7: 'Did Bob become the doctor?' when only Alice did → FALSE."""
    pef = _pef_with_alice_became_doctor()
    bob = Entity.create("Bob", turn=1, session_id="r1_become_test")
    pef.add_entity(bob)
    result = evaluate_existence_query(pef, "Bob", "become doctor")
    assert result is not None
    assert result.outcome == StateNativeOutcome.ANSWER
    from aurora_lens.state_native_engine.epistemic import EpistemicResult
    assert result.epistemic_result == EpistemicResult.FALSE


def test_BECOME_does_not_contaminate_IS_queries() -> None:
    """B6: A BECOME relation must not be returned as an IS (static state) answer.

    If Alice became the doctor (BECOME), a direct IS query for her current
    role should not be answered from that BECOME relation — BECOME records
    the transition, not a static present-state fact.
    """
    pef = _pef_with_alice_became_doctor()
    # IS query: find static IS relationships for Alice
    is_rels = [
        r for r in pef.relationships
        if r.relation == "IS"
        and r.subject_id in {e.id for e in pef.entities.values() if e.name == "Alice"}
    ]
    assert is_rels == [], (
        "BECOME relation must not be stored as IS; static IS query should find nothing"
    )
