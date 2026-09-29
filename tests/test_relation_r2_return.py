"""Phase R2: RETURN with prior-possession precondition — invariant tests.

Governing invariant: RETURN fires only when the recipient previously had
possession or ownership. Otherwise the claim is blocked (not silently committed
as delivery/GIVE). "brought back" and "bring back" surface forms → RETURN.
Plain "brought" / "bring" remain unaliased (context-dependent, still deferred).

Laws tested:
  RT1. "brought back" and "bring back" canonicalize to RETURN.
  RT2. Plain "brought" and "bring" do NOT canonicalize to RETURN.
  RT3. "returned" / "returns" / "return" continue to canonicalize to RETURN.
  RT4. RETURN is in CANONICAL_RELATIONS.
  RT5. Prior possession satisfied → RETURN claim allowed_commit=True.
  RT6. Prior possession not satisfied → RETURN claim blocked (allowed_commit=False,
       held_reason="RETURN_NO_PRIOR_POSSESSION").
  RT7. Recipient not in PEF → RETURN claim blocked (held_reason="RETURN_RECIPIENT_UNKNOWN").
  RT8. RETURN relation committed to PEF is matchable by the existence evaluator.
"""

from __future__ import annotations

import pytest

from aurora_lens.interpret.pef_updater import _build_return_transaction, _build_semantic_transactions
from aurora_lens.interpret.schema import ExtractionResult, ExtractedClaim, SemanticTransaction
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import CANONICAL_RELATIONS, PEFState, Relationship, canonicalize_relation
from aurora_lens.state_native_engine.contracts import StateNativeOutcome
from aurora_lens.state_native_engine.eval.existence import evaluate_existence_query


# ── RT1–RT4: canonicalize_relation ───────────────────────────────────────────

def test_brought_back_maps_to_RETURN() -> None:
    """RT1: multi-word 'brought back' → RETURN."""
    assert canonicalize_relation("brought back") == "RETURN"


def test_bring_back_maps_to_RETURN() -> None:
    """RT1: lemma phrase 'bring back' → RETURN."""
    assert canonicalize_relation("bring back") == "RETURN"


def test_plain_brought_not_RETURN() -> None:
    """RT2: bare 'brought' does not map to RETURN (context-dependent, deferred)."""
    assert canonicalize_relation("brought") != "RETURN"


def test_plain_bring_not_RETURN() -> None:
    """RT2: bare 'bring' does not map to RETURN."""
    assert canonicalize_relation("bring") != "RETURN"


def test_returned_still_maps_to_RETURN() -> None:
    """RT3: existing 'returned' alias is preserved."""
    assert canonicalize_relation("returned") == "RETURN"


def test_RETURN_in_canonical_relations() -> None:
    """RT4: RETURN is a first-class canonical relation."""
    assert "RETURN" in CANONICAL_RELATIONS


# ── RT5–RT7: prior-possession precondition ───────────────────────────────────

def _pef_with_bob_owning_wallet() -> tuple[PEFState, Entity, Entity]:
    """PEF where Bob has (or had) a wallet, Alice has it now after a transfer."""
    pef = PEFState(session_id="r2_return_test")
    alice = Entity.create("Alice", turn=1, session_id="r2_return_test")
    bob = Entity.create("Bob", turn=1, session_id="r2_return_test")
    pef.add_entity(alice)
    pef.add_entity(bob)

    # Bob previously had the wallet
    pef.add_relationship(Relationship(
        subject_id=bob.id,
        relation="HAS",
        object_entity_id=None,
        object_literal="wallet",
        span=Span.PAST,
        source_turn=1,
        evidence="Bob had a wallet.",
    ))
    # Alice currently has it
    pef.add_relationship(Relationship(
        subject_id=alice.id,
        relation="HAS",
        object_entity_id=None,
        object_literal="wallet",
        span=Span.PRESENT,
        source_turn=2,
        evidence="Alice has the wallet.",
    ))
    pef.current_turn = 3
    return pef, alice, bob


def _return_claim(subject: str, obj_entity_name: str) -> ExtractedClaim:
    return ExtractedClaim(
        subject=subject,
        relation="returned",
        obj=obj_entity_name,
        span=Span.PAST,
        negated=False,
        evidence=f"{subject} returned the wallet to {obj_entity_name}.",
        provenance="user_input",
        extractor_backend="test",
    )


def test_return_allowed_when_recipient_has_prior_possession() -> None:
    """RT5: Bob previously had the wallet → RETURN is allowed to commit."""
    pef, alice, bob = _pef_with_bob_owning_wallet()
    claim = _return_claim("Alice", "Bob")
    base_tx = SemanticTransaction(claim=claim)

    result = _build_return_transaction(claim, pef, base_tx)

    assert result.allowed_commit is True
    assert result.held_reason is None


def test_return_blocked_when_recipient_has_no_prior_possession() -> None:
    """RT6: Carol has never had anything → RETURN blocked."""
    pef = PEFState(session_id="r2_no_poss")
    alice = Entity.create("Alice", turn=1, session_id="r2_no_poss")
    carol = Entity.create("Carol", turn=1, session_id="r2_no_poss")
    pef.add_entity(alice)
    pef.add_entity(carol)
    # Alice has a book; Carol has nothing
    pef.add_relationship(Relationship(
        subject_id=alice.id,
        relation="HAS",
        object_entity_id=None,
        object_literal="book",
        span=Span.PRESENT,
        source_turn=1,
        evidence="Alice has a book.",
    ))

    claim = _return_claim("Alice", "Carol")
    base_tx = SemanticTransaction(claim=claim)

    result = _build_return_transaction(claim, pef, base_tx)

    assert result.allowed_commit is False
    assert result.held_reason == "RETURN_NO_PRIOR_POSSESSION"


def test_return_blocked_when_recipient_not_in_pef() -> None:
    """RT7: Recipient not in PEF → RETURN blocked (RETURN_RECIPIENT_UNKNOWN)."""
    pef = PEFState(session_id="r2_unknown")
    alice = Entity.create("Alice", turn=1, session_id="r2_unknown")
    pef.add_entity(alice)

    claim = _return_claim("Alice", "UnknownPerson")
    base_tx = SemanticTransaction(claim=claim)

    result = _build_return_transaction(claim, pef, base_tx)

    assert result.allowed_commit is False
    assert result.held_reason == "RETURN_RECIPIENT_UNKNOWN"


def test_build_semantic_transactions_routes_return() -> None:
    """RT6 via full pipeline: RETURN claim with no prior possession → held in transactions."""
    pef = PEFState(session_id="r2_pipeline")
    alice = Entity.create("Alice", turn=1, session_id="r2_pipeline")
    carol = Entity.create("Carol", turn=1, session_id="r2_pipeline")
    pef.add_entity(alice)
    pef.add_entity(carol)

    claim = _return_claim("Alice", "Carol")
    result = ExtractionResult(
        claims=[claim],
        entity_mentions=[],
        ambiguous_referents=[],
        span=Span.PAST,
    )
    txs = _build_semantic_transactions(result, pef)

    assert len(txs) == 1
    assert txs[0].allowed_commit is False
    assert txs[0].held_reason == "RETURN_NO_PRIOR_POSSESSION"


def test_build_semantic_transactions_allows_return_with_prior_possession() -> None:
    """RT5 via full pipeline: RETURN claim with prior possession → allowed."""
    pef, alice, bob = _pef_with_bob_owning_wallet()

    claim = _return_claim("Alice", "Bob")
    result = ExtractionResult(
        claims=[claim],
        entity_mentions=[],
        ambiguous_referents=[],
        span=Span.PAST,
    )
    txs = _build_semantic_transactions(result, pef)

    assert len(txs) == 1
    assert txs[0].allowed_commit is True


# ── RT8: RETURN in existence evaluator ───────────────────────────────────────

def test_existence_query_finds_committed_RETURN() -> None:
    """RT8: 'Did Alice return the wallet?' matches committed RETURN relation → TRUE."""
    pef = PEFState(session_id="r2_exist")
    alice = Entity.create("Alice", turn=1, session_id="r2_exist")
    bob = Entity.create("Bob", turn=1, session_id="r2_exist")
    pef.add_entity(alice)
    pef.add_entity(bob)
    pef.add_relationship(Relationship(
        subject_id=alice.id,
        relation="RETURN",
        object_entity_id=bob.id,
        object_literal=None,
        span=Span.PAST,
        source_turn=2,
        evidence="Alice returned the wallet to Bob.",
    ))

    result = evaluate_existence_query(pef, "Alice", "return wallet")
    assert result is not None
    assert result.outcome == StateNativeOutcome.ANSWER
    from aurora_lens.state_native_engine.epistemic import EpistemicResult
    assert result.epistemic_result == EpistemicResult.TRUE
