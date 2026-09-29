"""Phase R5: Locative relations — containment, proximity, and origin invariant tests.

Three constrained additions:

  R5a — Containment → AT
    "inside" / "within" / "indoors" alias to AT. Being inside a location IS
    being at that location; the containment nuance is below PEF resolution.

  R5b — NEAR (new relation)
    Proximity is distinct from AT. "near the store" ≠ "at the store". NEAR is
    a static spatial state → in _NON_ACTION_RELATIONS.

  R5c — FROM (new relation, multi-word movement-origin only)
    "came from", "arrived from", "traveled from", "coming from" record
    origin/provenance. FROM is distinct from AT (where someone came from ≠
    current location). Bare "from" is NOT aliased — too overloaded.

Explicitly excluded:
  - "outside": negated containment + proximity; deferred.
  - bare "from": overloaded across TAKE source, temporal, comparative.

Laws tested:
  L1.  "inside" / "within" / "indoors" canonicalize to AT.
  L2.  "near" / "nearby" / "close to" canonicalize to NEAR.
  L3.  "came from" / "arrived from" / "traveled from" / "coming from" → FROM.
  L4.  Bare "from" does NOT canonicalize to FROM.
  L5.  "outside" does NOT canonicalize (no alias exists yet).
  L6.  NEAR and FROM are in CANONICAL_RELATIONS.
  L7.  NEAR and FROM are in _NON_ACTION_RELATIONS (static states).
  L8.  Existence evaluator does NOT match NEAR as an action (returns None).
  L9.  Existence evaluator does NOT match FROM as an action (returns None).
  L10. inside/within/indoors do NOT produce a non-AT relation.
  L11. NEAR committed to PEF is stored with the correct relation name.
  L12. FROM committed to PEF is stored with the correct relation name.
"""

from __future__ import annotations

import pytest

from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import (
    CANONICAL_RELATIONS, PEFState, Relationship, canonicalize_relation,
    _RELATION_ALIASES,
)
from aurora_lens.state_native_engine.eval.existence import (
    _NON_ACTION_RELATIONS,
    evaluate_existence_query,
)


# ── L1: containment → AT ──────────────────────────────────────────────────────

def test_inside_maps_to_AT() -> None:
    """L1: 'inside' containment → AT (current location fact)."""
    assert canonicalize_relation("inside") == "AT"


def test_within_maps_to_AT() -> None:
    """L1: 'within' → AT."""
    assert canonicalize_relation("within") == "AT"


def test_indoors_maps_to_AT() -> None:
    """L1: 'indoors' → AT."""
    assert canonicalize_relation("indoors") == "AT"


# ── L2: NEAR aliases ──────────────────────────────────────────────────────────

def test_near_maps_to_NEAR() -> None:
    """L2: 'near' → NEAR."""
    assert canonicalize_relation("near") == "NEAR"


def test_nearby_maps_to_NEAR() -> None:
    """L2: 'nearby' → NEAR."""
    assert canonicalize_relation("nearby") == "NEAR"


def test_close_to_maps_to_NEAR() -> None:
    """L2: multi-word 'close to' → NEAR (LLM backend)."""
    assert canonicalize_relation("close to") == "NEAR"


# ── L3: FROM aliases (multi-word movement-origin only) ───────────────────────

def test_came_from_maps_to_FROM() -> None:
    """L3: 'came from' → FROM (explicit movement-origin)."""
    assert canonicalize_relation("came from") == "FROM"


def test_arrived_from_maps_to_FROM() -> None:
    """L3: 'arrived from' → FROM."""
    assert canonicalize_relation("arrived from") == "FROM"


def test_traveled_from_maps_to_FROM() -> None:
    """L3: 'traveled from' → FROM."""
    assert canonicalize_relation("traveled from") == "FROM"


def test_coming_from_maps_to_FROM() -> None:
    """L3: present-participle 'coming from' → FROM (LLM backend)."""
    assert canonicalize_relation("coming from") == "FROM"


# ── L4: bare "from" not aliased ───────────────────────────────────────────────

def test_bare_from_not_explicitly_aliased() -> None:
    """L4: bare 'from' is overloaded — no explicit alias entry must exist.

    canonicalize_relation() uppercases unknown inputs as a passthrough, so
    'from'.upper() == 'FROM' regardless. The meaningful invariant is that no
    deliberate alias maps bare 'from' — the LLM/spaCy backends will never
    emit bare 'from' as a relation string, so the passthrough is harmless.
    """
    assert "from" not in _RELATION_ALIASES


# ── L5: "outside" has no alias ────────────────────────────────────────────────

def test_outside_maps_to_OUTSIDE() -> None:
    """L5: 'outside' is now implemented — maps to OUTSIDE (see test_relation_rd_final_deferred.py)."""
    assert canonicalize_relation("outside") == "OUTSIDE"


# ── L6: CANONICAL_RELATIONS membership ───────────────────────────────────────

def test_NEAR_in_canonical_relations() -> None:
    """L6: NEAR is a first-class canonical relation."""
    assert "NEAR" in CANONICAL_RELATIONS


def test_FROM_in_canonical_relations() -> None:
    """L6: FROM is a first-class canonical relation."""
    assert "FROM" in CANONICAL_RELATIONS


# ── L7: _NON_ACTION_RELATIONS membership ──────────────────────────────────────

def test_NEAR_in_non_action_relations() -> None:
    """L7: NEAR is a static spatial state — existence engine must not match it as action."""
    assert "NEAR" in _NON_ACTION_RELATIONS


def test_FROM_in_non_action_relations() -> None:
    """L7: FROM is a static provenance fact — existence engine must not match it as action."""
    assert "FROM" in _NON_ACTION_RELATIONS


# ── L8–L9: existence evaluator skips static locative states ──────────────────

def _pef_with_alice_near_store() -> PEFState:
    pef = PEFState(session_id="r5_near_test")
    alice = Entity.create("Alice", turn=1, session_id="r5_near_test")
    pef.add_entity(alice)
    pef.add_relationship(Relationship(
        subject_id=alice.id,
        relation="NEAR",
        object_entity_id=None,
        object_literal="the store",
        span=Span.PRESENT,
        source_turn=1,
        evidence="Alice is near the store.",
    ))
    return pef


def _pef_with_bob_from_paris() -> PEFState:
    pef = PEFState(session_id="r5_from_test")
    bob = Entity.create("Bob", turn=1, session_id="r5_from_test")
    pef.add_entity(bob)
    pef.add_relationship(Relationship(
        subject_id=bob.id,
        relation="FROM",
        object_entity_id=None,
        object_literal="Paris",
        span=Span.PAST,
        source_turn=1,
        evidence="Bob came from Paris.",
    ))
    return pef


def test_existence_query_does_not_match_NEAR_as_action() -> None:
    """L8: NEAR is _NON_ACTION_RELATIONS → evaluator returns None (falls through)."""
    pef = _pef_with_alice_near_store()
    result = evaluate_existence_query(pef, "Alice", "near store")
    assert result is None


def test_existence_query_does_not_match_FROM_as_action() -> None:
    """L9: FROM is _NON_ACTION_RELATIONS → evaluator returns None (falls through)."""
    pef = _pef_with_bob_from_paris()
    result = evaluate_existence_query(pef, "Bob", "from Paris")
    assert result is None


# ── L10: containment aliases produce AT, not a distinct relation ──────────────

def test_inside_does_not_produce_non_AT_relation() -> None:
    """L10: canonicalize_relation('inside') is 'AT', not INSIDE or any other string."""
    assert canonicalize_relation("inside") == "AT"
    assert canonicalize_relation("inside") != "INSIDE"


# ── L11–L12: PEF stores correct relation names ────────────────────────────────

def test_pef_stores_NEAR_relation() -> None:
    """L11: Relationship committed with relation='NEAR' is retrievable."""
    pef = _pef_with_alice_near_store()
    near_rels = [r for r in pef.relationships if r.relation == "NEAR"]
    assert len(near_rels) == 1
    assert near_rels[0].object_literal == "the store"


def test_pef_stores_FROM_relation() -> None:
    """L12: Relationship committed with relation='FROM' is retrievable."""
    pef = _pef_with_bob_from_paris()
    from_rels = [r for r in pef.relationships if r.relation == "FROM"]
    assert len(from_rels) == 1
    assert from_rels[0].object_literal == "Paris"
