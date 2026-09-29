"""Phase R4: REQUEST / PROMISE / WARN speech-act relations — invariant tests.

Governing invariant: TELL records information transfer. REQUEST, PROMISE, and
WARN are distinct illocutionary acts — directive, commitment, and advisory
respectively. Folding them into TELL destroys the semantic distinction between
transmitting information and binding a speaker to an obligation, instructing
an action, or alerting to danger.

Laws tested:
  SA1. "ask" / "asked" / "asks" canonicalize to REQUEST.
  SA2. "request" / "requested" / "requests" canonicalize to REQUEST.
  SA3. "promise" / "promised" / "promises" canonicalize to PROMISE.
  SA4. "warn" / "warned" / "warns" canonicalize to WARN.
  SA5. REQUEST, PROMISE, WARN are in CANONICAL_RELATIONS.
  SA6. REQUEST, PROMISE, WARN are NOT in _NON_ACTION_RELATIONS (they are events).
  SA7. ask / request / promise / warn do NOT alias to TELL (distinct speech acts).
  SA8. Existence evaluator finds a committed REQUEST relation → TRUE.
  SA9. Existence evaluator finds a committed PROMISE relation → TRUE.
  SA10. Existence evaluator finds a committed WARN relation → TRUE.
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
from aurora_lens.state_native_engine.epistemic import EpistemicResult


# ── SA1–SA2: REQUEST aliases ──────────────────────────────────────────────────

def test_ask_lemma_maps_to_REQUEST() -> None:
    """SA1: spaCy lemma 'ask' → REQUEST."""
    assert canonicalize_relation("ask") == "REQUEST"


def test_asked_maps_to_REQUEST() -> None:
    """SA1: LLM surface past tense 'asked' → REQUEST."""
    assert canonicalize_relation("asked") == "REQUEST"


def test_asks_maps_to_REQUEST() -> None:
    """SA1: LLM surface 'asks' → REQUEST."""
    assert canonicalize_relation("asks") == "REQUEST"


def test_request_lemma_maps_to_REQUEST() -> None:
    """SA2: bare 'request' → REQUEST."""
    assert canonicalize_relation("request") == "REQUEST"


def test_requested_maps_to_REQUEST() -> None:
    """SA2: past tense 'requested' → REQUEST."""
    assert canonicalize_relation("requested") == "REQUEST"


def test_requests_maps_to_REQUEST() -> None:
    """SA2: 'requests' → REQUEST."""
    assert canonicalize_relation("requests") == "REQUEST"


# ── SA3: PROMISE aliases ──────────────────────────────────────────────────────

def test_promise_lemma_maps_to_PROMISE() -> None:
    """SA3: bare 'promise' → PROMISE."""
    assert canonicalize_relation("promise") == "PROMISE"


def test_promised_maps_to_PROMISE() -> None:
    """SA3: 'promised' → PROMISE."""
    assert canonicalize_relation("promised") == "PROMISE"


def test_promises_maps_to_PROMISE() -> None:
    """SA3: 'promises' → PROMISE."""
    assert canonicalize_relation("promises") == "PROMISE"


# ── SA4: WARN aliases ─────────────────────────────────────────────────────────

def test_warn_lemma_maps_to_WARN() -> None:
    """SA4: bare 'warn' → WARN."""
    assert canonicalize_relation("warn") == "WARN"


def test_warned_maps_to_WARN() -> None:
    """SA4: 'warned' → WARN."""
    assert canonicalize_relation("warned") == "WARN"


def test_warns_maps_to_WARN() -> None:
    """SA4: 'warns' → WARN."""
    assert canonicalize_relation("warns") == "WARN"


# ── SA5: CANONICAL_RELATIONS membership ──────────────────────────────────────

def test_REQUEST_in_canonical_relations() -> None:
    """SA5: REQUEST is a first-class canonical relation."""
    assert "REQUEST" in CANONICAL_RELATIONS


def test_PROMISE_in_canonical_relations() -> None:
    """SA5: PROMISE is a first-class canonical relation."""
    assert "PROMISE" in CANONICAL_RELATIONS


def test_WARN_in_canonical_relations() -> None:
    """SA5: WARN is a first-class canonical relation."""
    assert "WARN" in CANONICAL_RELATIONS


# ── SA6: not in _NON_ACTION_RELATIONS ────────────────────────────────────────

def test_REQUEST_not_in_non_action_relations() -> None:
    """SA6: REQUEST is a speech-act event — existence engine must match it."""
    assert "REQUEST" not in _NON_ACTION_RELATIONS


def test_PROMISE_not_in_non_action_relations() -> None:
    """SA6: PROMISE is a speech-act event — existence engine must match it."""
    assert "PROMISE" not in _NON_ACTION_RELATIONS


def test_WARN_not_in_non_action_relations() -> None:
    """SA6: WARN is a speech-act event — existence engine must match it."""
    assert "WARN" not in _NON_ACTION_RELATIONS


# ── SA7: not aliased to TELL ──────────────────────────────────────────────────

def test_ask_not_TELL() -> None:
    """SA7: directive 'ask' must not compress into information-transfer TELL."""
    assert canonicalize_relation("ask") != "TELL"


def test_request_not_TELL() -> None:
    """SA7: 'request' must not compress into TELL."""
    assert canonicalize_relation("request") != "TELL"


def test_promise_not_TELL() -> None:
    """SA7: commitment 'promise' must not compress into TELL."""
    assert canonicalize_relation("promise") != "TELL"


def test_warn_not_TELL() -> None:
    """SA7: advisory 'warn' must not compress into TELL."""
    assert canonicalize_relation("warn") != "TELL"


# ── SA8–SA10: existence evaluator integration ─────────────────────────────────

def _two_entity_pef(session_id: str) -> tuple[PEFState, Entity, Entity]:
    pef = PEFState(session_id=session_id)
    alice = Entity.create("Alice", turn=1, session_id=session_id)
    bob = Entity.create("Bob", turn=1, session_id=session_id)
    pef.add_entity(alice)
    pef.add_entity(bob)
    return pef, alice, bob


def test_existence_query_finds_committed_REQUEST() -> None:
    """SA8: 'Did Alice ask Bob to leave?' matches committed REQUEST relation → TRUE."""
    pef, alice, bob = _two_entity_pef("r4_request")
    pef.add_relationship(Relationship(
        subject_id=alice.id,
        relation="REQUEST",
        object_entity_id=bob.id,
        object_literal=None,
        span=Span.PAST,
        source_turn=2,
        evidence="Alice asked Bob to leave.",
    ))
    result = evaluate_existence_query(pef, "Alice", "ask leave")
    assert result is not None
    assert result.outcome == StateNativeOutcome.ANSWER
    assert result.epistemic_result == EpistemicResult.TRUE


def test_existence_query_finds_committed_PROMISE() -> None:
    """SA9: 'Did Alice promise?' matches committed PROMISE relation → TRUE."""
    pef, alice, bob = _two_entity_pef("r4_promise")
    pef.add_relationship(Relationship(
        subject_id=alice.id,
        relation="PROMISE",
        object_entity_id=bob.id,
        object_literal=None,
        span=Span.PAST,
        source_turn=2,
        evidence="Alice promised Bob she would return.",
    ))
    result = evaluate_existence_query(pef, "Alice", "promise")
    assert result is not None
    assert result.outcome == StateNativeOutcome.ANSWER
    assert result.epistemic_result == EpistemicResult.TRUE


def test_existence_query_finds_committed_WARN() -> None:
    """SA10: 'Did Alice warn Bob?' matches committed WARN relation → TRUE."""
    pef, alice, bob = _two_entity_pef("r4_warn")
    pef.add_relationship(Relationship(
        subject_id=alice.id,
        relation="WARN",
        object_entity_id=bob.id,
        object_literal=None,
        span=Span.PAST,
        source_turn=2,
        evidence="Alice warned Bob about the danger.",
    ))
    result = evaluate_existence_query(pef, "Alice", "warn danger")
    assert result is not None
    assert result.outcome == StateNativeOutcome.ANSWER
    assert result.epistemic_result == EpistemicResult.TRUE
