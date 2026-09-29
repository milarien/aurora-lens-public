"""PossessionTransaction invariant tests — Laws P1–P6.

P1: prior_count_snapshot captured at build time; validate/commit never re-read PEF for it.
P2: transfer/consume with prior < qty → validate() returns stop_reason; commit not called.
P3: supersede commit calls supersession + add_relationship atomically (one method).
P4: _commit_resolved_claims uses PossessionTransaction for GIVE replay (no inline math).
P5: spaCy arithmetic HAS claims routed as acquire — no re-computation.
P6: Phase 1/2/2b blocking unaffected by Phase 3.
"""

from __future__ import annotations

import pytest

from aurora_lens.interpret.schema import (
    ComparativeAmbiguity,
    ExtractedClaim,
    ExtractionResult,
    PossessionTransaction,
    SemanticTransaction,
)
from aurora_lens.interpret.pef_updater import _build_semantic_transactions, update_pef
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship


# ── Helpers ───────────────────────────────────────────────────────────────────


def _claim(subject: str, relation: str, obj: str, negated: bool = False) -> ExtractedClaim:
    return ExtractedClaim(
        subject=subject,
        relation=relation,
        obj=obj,
        span=Span.PRESENT,
        negated=negated,
        evidence=f"{subject} {relation} {obj}.",
    )


def _pef_with(*names: str) -> PEFState:
    pef = PEFState()
    for name in names:
        pef.get_or_create_entity(name)
    return pef


def _give_pef(giver: str, qty: int, item: str, *others: str) -> PEFState:
    pef = _pef_with(giver, *others)
    ent = pef.find_entity_by_name(giver)
    pef.add_relationship(Relationship(
        subject_id=ent.id, relation="HAS",
        object_entity_id=None, object_literal=f"{qty} {item}",
        span=Span.PRESENT, source_turn=1, evidence=f"{giver} has {qty} {item}.",
    ))
    return pef


def _make_tx(mutation_kind: str, qty: float, prior: float, item: str = "apples") -> PossessionTransaction:
    return PossessionTransaction(
        claim=_claim("James", "HAS", f"{int(prior)} {item}"),
        mutation_kind=mutation_kind,
        quantity=qty,
        prior_count_snapshot=prior,
        giver_name="James",
        item_key_str=item,
    )


# ── Law P1+P2: validate() ─────────────────────────────────────────────────────


def test_give_validates_insufficient_quantity_returns_stop():
    tx = _make_tx("transfer", qty=5.0, prior=3.0)
    result = tx.validate(None)
    assert result is not None
    assert "James" in result
    assert "5" in result


def test_give_validates_sufficient_quantity_sets_remainder():
    tx = _make_tx("transfer", qty=5.0, prior=10.0)
    result = tx.validate(None)
    assert result is None
    assert tx.projected_remainder == 5.0


def test_consume_validates_insufficient_quantity_returns_stop():
    tx = _make_tx("consume", qty=5.0, prior=3.0)
    result = tx.validate(None)
    assert result is not None
    assert "3" in result or "5" in result


def test_consume_validates_sufficient_quantity_sets_remainder():
    tx = _make_tx("consume", qty=3.0, prior=7.0)
    result = tx.validate(None)
    assert result is None
    assert tx.projected_remainder == 4.0


def test_validate_before_commit_invariant_no_pef_write():
    """validate() returns stop_reason → commit() must not be called → no PEF write."""
    pef = _give_pef("James", 3, "apples")
    before = len(pef.relationships)
    tx = _make_tx("transfer", qty=5.0, prior=3.0)
    stop = tx.validate(pef)
    assert stop is not None
    # Caller must not call commit after a non-None validate
    assert len(pef.relationships) == before


def test_acquire_validates_always_admits():
    tx = _make_tx("acquire", qty=5.0, prior=3.0)
    assert tx.validate(None) is None


def test_supersede_validates_always_admits():
    tx = PossessionTransaction(
        claim=_claim("James", "HAS", "wallet"),
        mutation_kind="supersede",
    )
    assert tx.validate(None) is None


# ── Law P3: supersede commit is atomic ────────────────────────────────────────


def test_supersede_commit_negates_prior_and_adds_new():
    pef = _pef_with("James")
    james = pef.find_entity_by_name("James")
    pef.add_relationship(Relationship(
        subject_id=james.id, relation="HAS",
        object_entity_id=None, object_literal="wallet",
        span=Span.PRESENT, source_turn=1, evidence="James has wallet.",
    ))
    before_count = sum(1 for r in pef.relationships if r.subject_id == james.id and r.relation == "HAS")
    assert before_count == 1

    tx = PossessionTransaction(
        claim=_claim("James", "HAS", "wallet"),
        mutation_kind="supersede",
    )
    assert tx.validate(pef) is None
    tx.commit(pef)

    james_has = [r for r in pef.relationships if r.subject_id == james.id and r.relation == "HAS"]
    negated = [r for r in james_has if r.negated]
    active = [r for r in james_has if not r.negated]
    assert negated, "prior HAS must be negated by supersession"
    assert active, "new HAS must be written"


def test_acquire_commit_adds_relationship():
    pef = _pef_with("James")
    james = pef.find_entity_by_name("James")
    before = len(pef.relationships)

    tx = PossessionTransaction(
        claim=_claim("James", "HAS", "5 apples"),
        mutation_kind="acquire",
    )
    assert tx.validate(pef) is None
    tx.commit(pef)

    james_has = [r for r in pef.relationships if r.subject_id == james.id and r.relation == "HAS"]
    assert james_has, "HAS must be written"
    assert len(pef.relationships) == before + 1


def test_transfer_commit_writes_not_has_and_remainder():
    """PossessionTransaction.commit(transfer) writes NOT_HAS(old) + HAS(remainder) atomically."""
    pef = _give_pef("James", 10, "apples")
    james = pef.find_entity_by_name("James")

    tx = PossessionTransaction(
        claim=_claim("James", "HAS", "10 apples"),  # old literal as obj
        mutation_kind="transfer",
        quantity=3.0,
        prior_count_snapshot=10.0,
        giver_name="James",
        item_key_str="apples",
    )
    tx.validate(pef)
    tx.commit(pef)

    james_has = [r for r in pef.relationships if r.subject_id == james.id and r.relation == "HAS"]
    negated = [r for r in james_has if r.negated]
    active_remainder = [r for r in james_has if not r.negated and r.object_literal == "7 apples"]
    assert negated, "NOT_HAS(old literal) must be written"
    assert active_remainder, "HAS(7 apples) must be written"


# ── Law P4: GIVE replay uses PossessionTransaction ────────────────────────────


def test_give_replay_via_possession_transaction():
    """_commit_resolved_claims replays GIVE via PossessionTransaction — giver gets remainder."""
    from aurora_lens.lens import _commit_resolved_claims

    pef = _give_pef("James", 10, "apples", "Alice")
    pending = {
        "original_question": "He gave Alice 3 apples.",
        "original_span": "present",
        "ambiguous_referents": ["he"],
        "blocked_claims": [
            {
                "subject": "he",
                "relation": "GIVE",
                "obj": "3 apples",
                "span": "present",
                "negated": False,
                "evidence": "He gave Alice 3 apples.",
                "held_reason": "UNRESOLVED_REFERENT",
            }
        ],
    }
    res = _commit_resolved_claims("James", pending, pef)
    assert res.stop_reason is None
    james = pef.find_entity_by_name("James")
    james_has = [
        r for r in pef.relationships
        if r.subject_id == james.id and r.relation == "HAS" and not r.negated
    ]
    remainder_lits = [r.object_literal for r in james_has]
    assert "7 apples" in remainder_lits, f"Expected '7 apples' in {remainder_lits}"


def test_give_replay_insufficient_stop():
    """GIVE replay when giver has fewer than requested → stop_reason set, no remainder HAS."""
    from aurora_lens.lens import _commit_resolved_claims

    pef = _give_pef("James", 2, "apples", "Alice")
    pending = {
        "original_question": "He gave Alice 5 apples.",
        "original_span": "present",
        "ambiguous_referents": ["he"],
        "blocked_claims": [
            {
                "subject": "he",
                "relation": "GIVE",
                "obj": "5 apples",
                "span": "present",
                "negated": False,
                "evidence": "He gave Alice 5 apples.",
                "held_reason": "UNRESOLVED_REFERENT",
            }
        ],
    }
    before_has = [r for r in pef.relationships if r.relation == "HAS" and not r.negated]
    res = _commit_resolved_claims("James", pending, pef)
    after_has = [r for r in pef.relationships if r.relation == "HAS" and not r.negated]
    assert res.stop_reason is not None, "stop_reason must be set for insufficient GIVE"
    assert len(after_has) == len(before_has), "no new HAS must be written"


# ── Law P5: spaCy arithmetic claims routed as acquire ─────────────────────────


def test_spacy_give_arithmetic_claims_committed_as_acquire():
    """SpaCy emits NOT_HAS + HAS claims; update_pef commits them as acquire without re-computation."""
    pef = _pef_with("James")

    result = ExtractionResult(
        claims=[
            _claim("James", "HAS", "10 apples", negated=True),
            _claim("James", "HAS", "7 apples", negated=False),
        ],
        entity_mentions=[],
    )
    update_pef(result, pef)

    james = pef.find_entity_by_name("James")
    all_has = [r for r in pef.relationships if r.subject_id == james.id and r.relation == "HAS"]
    negated = [r for r in all_has if r.negated]
    active = [r for r in all_has if not r.negated]
    assert any(r.object_literal == "10 apples" for r in negated)
    assert any(r.object_literal == "7 apples" for r in active)


# ── Law P6: Phase 1/2/2b blocking unaffected ──────────────────────────────────


def test_comparative_blocking_unaffected_by_phase3():
    """IS(he, bigger) with comparative_ambiguity → UNRESOLVED_COMPARAND (not PossessionTransaction)."""
    result = ExtractionResult(
        claims=[_claim("he", "IS", "bigger")],
        entity_mentions=[],
        ambiguous_referents=["he"],
        comparative_ambiguities=[ComparativeAmbiguity("bigger", "stick", ["Alice", "Bob"])],
    )
    txs = _build_semantic_transactions(result)
    assert len(txs) == 1
    tx = txs[0]
    assert not isinstance(tx, PossessionTransaction)
    assert tx.allowed_commit is False
    assert tx.held_reason == "UNRESOLVED_COMPARAND"


def test_possessive_np_blocking_unaffected_by_phase3():
    """IS(his wallet, red) with ambiguous his → UNRESOLVED_POSSESSOR (not PossessionTransaction)."""
    result = ExtractionResult(
        claims=[_claim("his wallet", "IS", "red")],
        entity_mentions=[],
        ambiguous_referents=["his"],
    )
    txs = _build_semantic_transactions(result)
    assert len(txs) == 1
    tx = txs[0]
    assert not isinstance(tx, PossessionTransaction)
    assert tx.held_reason == "UNRESOLVED_POSSESSOR"


def test_has_blocked_by_ambiguous_referent_not_wrapped_as_possession():
    """HAS with ambiguous bare pronoun subject stays blocked (UNRESOLVED_REFERENT), not routed as acquire."""
    result = ExtractionResult(
        claims=[_claim("he", "HAS", "wallet")],
        entity_mentions=[],
        ambiguous_referents=["he"],
    )
    txs = _build_semantic_transactions(result)
    assert len(txs) == 1
    tx = txs[0]
    assert tx.allowed_commit is False
    assert tx.held_reason == "UNRESOLVED_REFERENT"


# ── PossessionTransaction wrapping in _build_semantic_transactions ─────────────


def test_bare_has_wrapped_as_supersede_when_pef_provided():
    pef = _pef_with("James")
    result = ExtractionResult(
        claims=[_claim("James", "HAS", "wallet")],
        entity_mentions=[],
    )
    txs = _build_semantic_transactions(result, pef)
    assert len(txs) == 1
    tx = txs[0]
    assert isinstance(tx, PossessionTransaction)
    assert tx.mutation_kind == "supersede"
    assert tx.allowed_commit is True


def test_counted_has_wrapped_as_acquire_when_pef_provided():
    pef = _pef_with("James")
    result = ExtractionResult(
        claims=[_claim("James", "HAS", "5 apples")],
        entity_mentions=[],
    )
    txs = _build_semantic_transactions(result, pef)
    assert isinstance(txs[0], PossessionTransaction)
    assert txs[0].mutation_kind == "acquire"


def test_negated_has_wrapped_as_acquire_when_pef_provided():
    pef = _pef_with("James")
    result = ExtractionResult(
        claims=[_claim("James", "HAS", "wallet", negated=True)],
        entity_mentions=[],
    )
    txs = _build_semantic_transactions(result, pef)
    assert isinstance(txs[0], PossessionTransaction)
    assert txs[0].mutation_kind == "acquire"


def test_give_wrapped_as_transfer_with_prior_snapshot():
    """GIVE claim wrapped as PossessionTransaction(transfer) with prior_count_snapshot."""
    pef = _give_pef("James", 10, "apples", "Alice")
    result = ExtractionResult(
        claims=[_claim("James", "GIVE", "3 apples")],
        entity_mentions=[],
    )
    txs = _build_semantic_transactions(result, pef)
    assert len(txs) == 1
    tx = txs[0]
    assert isinstance(tx, PossessionTransaction)
    assert tx.mutation_kind == "transfer"
    assert tx.prior_count_snapshot == 10.0
    assert tx.quantity == 3.0


def test_no_pef_arg_does_not_wrap_has_as_possession():
    """Without pef arg, HAS claim stays plain SemanticTransaction."""
    result = ExtractionResult(
        claims=[_claim("James", "HAS", "wallet")],
        entity_mentions=[],
    )
    txs = _build_semantic_transactions(result)
    assert len(txs) == 1
    assert not isinstance(txs[0], PossessionTransaction)


# ── update_pef supersession via PossessionTransaction ─────────────────────────


def test_update_pef_supersede_replaces_prior_bare_has():
    """update_pef for bare HAS supersedes prior HAS through PossessionTransaction."""
    pef = _pef_with("James")
    james = pef.find_entity_by_name("James")
    pef.add_relationship(Relationship(
        subject_id=james.id, relation="HAS",
        object_entity_id=None, object_literal="wallet",
        span=Span.PRESENT, source_turn=1, evidence="James has wallet.",
    ))

    result = ExtractionResult(
        claims=[_claim("James", "HAS", "wallet")],
        entity_mentions=[],
    )
    update_pef(result, pef)

    james_has = [r for r in pef.relationships if r.subject_id == james.id and r.relation == "HAS"]
    negated = [r for r in james_has if r.negated]
    assert negated, "supersession must negate prior HAS"
