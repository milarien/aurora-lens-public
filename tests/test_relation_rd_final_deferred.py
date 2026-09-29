"""Final deferred relation resolution: REMAIN, OUTSIDE, brought→GIVE.

Three items that had been held in the deferred comment block are now resolved:

  RD1 — REMAIN (new relation)
    "stayed as" / "stay as" record state persistence — the subject's identity
    or role continued across a possible transition boundary. Distinct from
    BECOME (which records the transition event) and IS (which is a static
    present-state snapshot). In _NON_ACTION_RELATIONS.

  RD2 — OUTSIDE (new relation)
    Exterior / negated-containment spatial state. Ontologically distinct from
    AT (co-location), NEAR (proximity), and INSIDE→AT (containment). Does not
    imply AT or NEAR. Records topology without collapsing distance semantics.
    In _NON_ACTION_RELATIONS.

  RD3 — plain "brought" / "bring" / "brings" → GIVE
    Transfer/delivery: "Alice brought Bob coffee" → Bob has coffee.
    Possession-transfer consequence is identical to GIVE semantics.
    "brought back" / "bring back" remain → RETURN (multi-word key takes
    priority over single-word key in dict lookup).

Laws tested:
  RD1a. "stayed as" / "stay as" canonicalize to REMAIN.
  RD1b. REMAIN ≠ BECOME (no cross-aliasing of continuity and transition).
  RD1c. REMAIN ≠ IS (persistence assertion ≠ static snapshot).
  RD1d. REMAIN in CANONICAL_RELATIONS.
  RD1e. REMAIN in _NON_ACTION_RELATIONS.
  RD1f. Existence evaluator returns None for REMAIN (static state).
  RD2a. "outside" / "outside of" canonicalize to OUTSIDE.
  RD2b. OUTSIDE ≠ NEAR, ≠ AT.
  RD2c. OUTSIDE in CANONICAL_RELATIONS.
  RD2d. OUTSIDE in _NON_ACTION_RELATIONS.
  RD2e. Existence evaluator returns None for OUTSIDE (static state).
  RD3a. "brought" / "bring" / "brings" canonicalize to GIVE.
  RD3b. "brought back" still canonicalizes to RETURN (multi-word precedence).
  RD3c. "bring back" still canonicalizes to RETURN.
"""

from __future__ import annotations

import pytest

from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import CANONICAL_RELATIONS, PEFState, Relationship, canonicalize_relation
from aurora_lens.state_native_engine.eval.existence import (
    _NON_ACTION_RELATIONS,
    evaluate_existence_query,
)


# ── RD1: REMAIN ───────────────────────────────────────────────────────────────

def test_stayed_as_maps_to_REMAIN() -> None:
    """RD1a: 'stayed as' → REMAIN."""
    assert canonicalize_relation("stayed as") == "REMAIN"


def test_stay_as_maps_to_REMAIN() -> None:
    """RD1a: lemma/present form 'stay as' → REMAIN."""
    assert canonicalize_relation("stay as") == "REMAIN"


def test_REMAIN_distinct_from_BECOME() -> None:
    """RD1b: continuity (REMAIN) must not cross-alias with transition (BECOME)."""
    assert canonicalize_relation("stayed as") != "BECOME"
    assert canonicalize_relation("become") != "REMAIN"


def test_REMAIN_distinct_from_IS() -> None:
    """RD1c: persistence assertion ≠ static snapshot."""
    assert canonicalize_relation("stayed as") != "IS"
    assert canonicalize_relation("is") != "REMAIN"


def test_REMAIN_in_canonical_relations() -> None:
    """RD1d: REMAIN is a first-class canonical relation."""
    assert "REMAIN" in CANONICAL_RELATIONS


def test_REMAIN_in_non_action_relations() -> None:
    """RD1e: REMAIN is a static continuity state — existence engine must not match as action."""
    assert "REMAIN" in _NON_ACTION_RELATIONS


def test_existence_evaluator_skips_REMAIN() -> None:
    """RD1f: REMAIN in _NON_ACTION_RELATIONS → evaluator returns None (falls through)."""
    pef = PEFState(session_id="rd_remain_test")
    alice = Entity.create("Alice", turn=1, session_id="rd_remain_test")
    pef.add_entity(alice)
    pef.add_relationship(Relationship(
        subject_id=alice.id,
        relation="REMAIN",
        object_entity_id=None,
        object_literal="CEO",
        span=Span.PAST,
        source_turn=1,
        evidence="Alice stayed as CEO.",
    ))
    result = evaluate_existence_query(pef, "Alice", "stay CEO")
    assert result is None


# ── RD2: OUTSIDE ──────────────────────────────────────────────────────────────

def test_outside_maps_to_OUTSIDE() -> None:
    """RD2a: 'outside' → OUTSIDE."""
    assert canonicalize_relation("outside") == "OUTSIDE"


def test_outside_of_maps_to_OUTSIDE() -> None:
    """RD2a: multi-word 'outside of' → OUTSIDE (LLM backend)."""
    assert canonicalize_relation("outside of") == "OUTSIDE"


def test_OUTSIDE_distinct_from_NEAR() -> None:
    """RD2b: exterior topology ≠ proximity."""
    assert canonicalize_relation("outside") != "NEAR"


def test_OUTSIDE_distinct_from_AT() -> None:
    """RD2b: exterior position ≠ co-location."""
    assert canonicalize_relation("outside") != "AT"


def test_OUTSIDE_in_canonical_relations() -> None:
    """RD2c: OUTSIDE is a first-class canonical relation."""
    assert "OUTSIDE" in CANONICAL_RELATIONS


def test_OUTSIDE_in_non_action_relations() -> None:
    """RD2d: OUTSIDE is a static spatial state — existence engine must not match as action."""
    assert "OUTSIDE" in _NON_ACTION_RELATIONS


def test_existence_evaluator_skips_OUTSIDE() -> None:
    """RD2e: OUTSIDE in _NON_ACTION_RELATIONS → evaluator returns None (falls through)."""
    pef = PEFState(session_id="rd_outside_test")
    bob = Entity.create("Bob", turn=1, session_id="rd_outside_test")
    pef.add_entity(bob)
    pef.add_relationship(Relationship(
        subject_id=bob.id,
        relation="OUTSIDE",
        object_entity_id=None,
        object_literal="hospital",
        span=Span.PRESENT,
        source_turn=1,
        evidence="Bob is outside the hospital.",
    ))
    result = evaluate_existence_query(pef, "Bob", "outside hospital")
    assert result is None


# ── RD3: brought / bring / brings → GIVE ─────────────────────────────────────

def test_brought_maps_to_GIVE() -> None:
    """RD3a: plain 'brought' → GIVE (transfer/delivery consequence)."""
    assert canonicalize_relation("brought") == "GIVE"


def test_bring_maps_to_GIVE() -> None:
    """RD3a: lemma 'bring' → GIVE."""
    assert canonicalize_relation("bring") == "GIVE"


def test_brings_maps_to_GIVE() -> None:
    """RD3a: 'brings' → GIVE."""
    assert canonicalize_relation("brings") == "GIVE"


def test_brought_back_still_maps_to_RETURN() -> None:
    """RD3b: 'brought back' multi-word key overrides single-word 'brought' → RETURN."""
    assert canonicalize_relation("brought back") == "RETURN"


def test_bring_back_still_maps_to_RETURN() -> None:
    """RD3c: 'bring back' multi-word key overrides single-word 'bring' → RETURN."""
    assert canonicalize_relation("bring back") == "RETURN"
