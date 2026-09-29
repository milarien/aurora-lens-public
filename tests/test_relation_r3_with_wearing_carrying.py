"""Phase R3: WITH / WEARING / CARRYING temporary-association relations — invariant tests.

Governing invariant: An alias may only map to an existing relation if the
mapped relation preserves the original semantic consequence. WEARING/WITH/
CARRYING record temporary states distinct from HAS possession. Compressing
them into HAS destroys the semantic distinction between permanent ownership
and temporary custody, accompaniment, or body-associated state.

Laws tested:
  WC1. "wear" / "wore" / "wearing" canonicalize to WEARING.
  WC2. "with" canonicalizes to WITH (LLM backend form).
  WC3. "carry" / "carries" / "carried" / "carrying" canonicalize to CARRYING.
  WC4. "carrying around" canonicalizes to CARRYING (multi-word LLM form).
  WC5. "wear", "carry", "with" do NOT map to HAS (not collapsed to possession).
  WC6. WITH, WEARING, CARRYING are in CANONICAL_RELATIONS.
  WC7. WITH, WEARING, CARRYING are in _NON_ACTION_RELATIONS (static state).
  WC8. Existence evaluator does NOT match WEARING as an action (returns UNKNOWN).
  WC9. WEARING, WITH, CARRYING committed to PEF are stored with the correct relation name.
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


# ── WC1: WEARING aliases ──────────────────────────────────────────────────────

def test_wear_lemma_maps_to_WEARING() -> None:
    """WC1: spaCy lemma 'wear' → WEARING."""
    assert canonicalize_relation("wear") == "WEARING"


def test_wore_maps_to_WEARING() -> None:
    """WC1: LLM surface past tense 'wore' → WEARING."""
    assert canonicalize_relation("wore") == "WEARING"


def test_wearing_maps_to_WEARING() -> None:
    """WC1: LLM surface present participle 'wearing' → WEARING."""
    assert canonicalize_relation("wearing") == "WEARING"


# ── WC2: WITH alias ───────────────────────────────────────────────────────────

def test_with_maps_to_WITH() -> None:
    """WC2: preposition/accompaniment 'with' → WITH (LLM backend surface form)."""
    assert canonicalize_relation("with") == "WITH"


# ── WC3–WC4: CARRYING aliases ─────────────────────────────────────────────────

def test_carry_lemma_maps_to_CARRYING() -> None:
    """WC3: spaCy lemma 'carry' → CARRYING (reclaimed from HAS)."""
    assert canonicalize_relation("carry") == "CARRYING"


def test_carries_maps_to_CARRYING() -> None:
    """WC3: 'carries' → CARRYING."""
    assert canonicalize_relation("carries") == "CARRYING"


def test_carried_maps_to_CARRYING() -> None:
    """WC3: 'carried' → CARRYING."""
    assert canonicalize_relation("carried") == "CARRYING"


def test_carrying_maps_to_CARRYING() -> None:
    """WC3: present participle 'carrying' → CARRYING."""
    assert canonicalize_relation("carrying") == "CARRYING"


def test_carrying_around_maps_to_CARRYING() -> None:
    """WC4: multi-word 'carrying around' → CARRYING (LLM backend)."""
    assert canonicalize_relation("carrying around") == "CARRYING"


# ── WC5: not collapsed to HAS ─────────────────────────────────────────────────

def test_wear_not_HAS() -> None:
    """WC5: 'wear' must not compress into possession."""
    assert canonicalize_relation("wear") != "HAS"


def test_with_not_HAS() -> None:
    """WC5: 'with' accompaniment must not compress into possession."""
    assert canonicalize_relation("with") != "HAS"


def test_carry_not_HAS() -> None:
    """WC5: temporary custody 'carry' must not compress into possession."""
    assert canonicalize_relation("carry") != "HAS"


# ── WC6: CANONICAL_RELATIONS membership ──────────────────────────────────────

def test_WITH_in_canonical_relations() -> None:
    """WC6: WITH is a first-class canonical relation."""
    assert "WITH" in CANONICAL_RELATIONS


def test_WEARING_in_canonical_relations() -> None:
    """WC6: WEARING is a first-class canonical relation."""
    assert "WEARING" in CANONICAL_RELATIONS


def test_CARRYING_in_canonical_relations() -> None:
    """WC6: CARRYING is a first-class canonical relation."""
    assert "CARRYING" in CANONICAL_RELATIONS


# ── WC7: _NON_ACTION_RELATIONS membership ────────────────────────────────────

def test_WITH_in_non_action_relations() -> None:
    """WC7: WITH is a static state — existence engine must not match it as action."""
    assert "WITH" in _NON_ACTION_RELATIONS


def test_WEARING_in_non_action_relations() -> None:
    """WC7: WEARING is a static state — existence engine must not match it as action."""
    assert "WEARING" in _NON_ACTION_RELATIONS


def test_CARRYING_in_non_action_relations() -> None:
    """WC7: CARRYING is a static state — existence engine must not match it as action."""
    assert "CARRYING" in _NON_ACTION_RELATIONS


# ── WC8: existence evaluator skips WEARING as non-action ─────────────────────

def _pef_with_alice_wearing_hat() -> PEFState:
    pef = PEFState(session_id="r3_wearing_test")
    alice = Entity.create("Alice", turn=1, session_id="r3_wearing_test")
    pef.add_entity(alice)
    pef.add_relationship(
        Relationship(
            subject_id=alice.id,
            relation="WEARING",
            object_entity_id=None,
            object_literal="hat",
            span=Span.PRESENT,
            source_turn=1,
            evidence="Alice is wearing a hat.",
        )
    )
    return pef


def test_existence_query_does_not_match_WEARING_as_action() -> None:
    """WC8: 'Did Alice wear the hat?' — WEARING is _NON_ACTION_RELATIONS → None (fall-through to LLM).

    The existence evaluator returns None when it finds zero matching action
    relations. Since WEARING is in _NON_ACTION_RELATIONS the scan skips it,
    producing no matches and the correct None fall-through.
    """
    pef = _pef_with_alice_wearing_hat()
    result = evaluate_existence_query(pef, "Alice", "wear hat")
    assert result is None  # no action match — evaluator falls through


# ── WC9: PEF stores correct relation names ────────────────────────────────────

def test_pef_stores_WEARING_relation() -> None:
    """WC9: Relationship committed with relation='WEARING' is retrievable."""
    pef = _pef_with_alice_wearing_hat()
    wearing_rels = [r for r in pef.relationships if r.relation == "WEARING"]
    assert len(wearing_rels) == 1
    assert wearing_rels[0].object_literal == "hat"


def test_pef_stores_WITH_relation() -> None:
    """WC9: Relationship committed with relation='WITH' is retrievable."""
    pef = PEFState(session_id="r3_with_test")
    alice = Entity.create("Alice", turn=1, session_id="r3_with_test")
    bob = Entity.create("Bob", turn=1, session_id="r3_with_test")
    pef.add_entity(alice)
    pef.add_entity(bob)
    pef.add_relationship(
        Relationship(
            subject_id=alice.id,
            relation="WITH",
            object_entity_id=bob.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="Alice arrived with Bob.",
        )
    )
    with_rels = [r for r in pef.relationships if r.relation == "WITH"]
    assert len(with_rels) == 1
    assert with_rels[0].object_entity_id == bob.id


def test_pef_stores_CARRYING_relation() -> None:
    """WC9: Relationship committed with relation='CARRYING' is retrievable."""
    pef = PEFState(session_id="r3_carrying_test")
    alice = Entity.create("Alice", turn=1, session_id="r3_carrying_test")
    pef.add_entity(alice)
    pef.add_relationship(
        Relationship(
            subject_id=alice.id,
            relation="CARRYING",
            object_entity_id=None,
            object_literal="briefcase",
            span=Span.PRESENT,
            source_turn=1,
            evidence="Alice is carrying a briefcase.",
        )
    )
    carrying_rels = [r for r in pef.relationships if r.relation == "CARRYING"]
    assert len(carrying_rels) == 1
    assert carrying_rels[0].object_literal == "briefcase"
