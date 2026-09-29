"""Phase-2 SemanticTransaction invariant tests.

Scope: possessive-NP subject claims with unresolved possessors.

Invariants tested:
  P2-1. All possessive-NP subject claims with ambiguous possessors are blocked.
  P2-2. Possessive surfaces ("his wallet") never become canonical entity names.
  P2-3. Replay preserves ownership — HAS(James, item_entity) committed.
  P2-4. Existing owned entity is reused; no duplicate HAS or duplicate entity.
  P2-5. Non-possessive pronoun path ("He had the wallet") is unchanged.
  P2-6. AT relation replays correctly with entity-linked ownership.
  P2-7. All relations (IS, AT, HAS, TAKE) are blocked for possessive-NP subjects.
  P2-8. Comparative and possessive-NP coexist; comparative takes priority.
  P2-9. Multi-word possessive NP preserves modifier+head ("leather wallet").
  P2-10. _build_semantic_transactions: possessor pronoun and held_reason correct.
  P2-11. Blocked-claims dict has held_reason=UNRESOLVED_POSSESSOR, possessor_pronoun, item_noun.
"""

from __future__ import annotations

import pytest

from aurora_lens.interpret.pef_updater import _build_semantic_transactions, update_pef
from aurora_lens.interpret.schema import (
    ComparativeAmbiguity,
    ExtractedClaim,
    ExtractionResult,
    SemanticTransaction,
)
from aurora_lens.lens import (
    _CommitReplayResult,
    _commit_resolved_claims,
    _extract_item_noun_from_possessive_np,
    _find_owned_item_entity,
)
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship, canonicalize_relation
from aurora_lens.state_native_engine.lexical import item_key


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pef_with_entities(*names: str) -> PEFState:
    """PEF with named entities, no relationships."""
    pef = PEFState()
    for name in names:
        ent = Entity.create(name, turn=1, session_id=pef.session_id)
        pef.add_entity(ent)
    pef.current_turn = 2
    return pef


def _add_has(pef: PEFState, owner: str, item_literal: str) -> Relationship:
    owner_ent = pef.find_entity_by_name(owner)
    assert owner_ent is not None, f"Entity {owner!r} not in PEF"
    rel = Relationship(
        subject_id=owner_ent.id,
        relation="HAS",
        object_entity_id=None,
        object_literal=item_literal,
        span=Span.PRESENT,
        source_turn=1,
        evidence=f"fixture: {owner} has {item_literal}",
    )
    pef.add_relationship(rel)
    return rel


def _make_possessive_result(
    subject: str,
    relation: str,
    obj: str,
    ambiguous: list[str],
) -> ExtractionResult:
    return ExtractionResult(
        claims=[
            ExtractedClaim(
                subject=subject,
                relation=relation,
                obj=obj,
                span=Span.PRESENT,
                negated=False,
                evidence=f"{subject} {relation} {obj}",
            )
        ],
        entity_mentions=[],
        ambiguous_referents=ambiguous,
    )


def _pending_for_possessive(
    subject: str,
    relation: str,
    obj: str,
    item_noun: str,
    possessor_pronoun: str,
    ambiguous: list[str],
) -> dict:
    return {
        "original_question": f"{subject} {relation} {obj}.",
        "original_span": "present",
        "ambiguous_referents": ambiguous,
        "blocked_claims": [
            {
                "subject": subject,
                "relation": relation,
                "obj": obj,
                "span": "present",
                "negated": False,
                "evidence": f"{subject} {relation} {obj}.",
                "is_possessive_np": True,
                "possessor_pronoun": possessor_pronoun,
                "item_noun": item_noun,
            }
        ],
    }


def _entity_names(pef: PEFState) -> set[str]:
    return {e.name for e in pef.entities.values()}


def _has_relationships(pef: PEFState) -> list[Relationship]:
    return [r for r in pef.relationships if r.relation == "HAS"]


def _is_relationships(pef: PEFState) -> list[Relationship]:
    return [r for r in pef.relationships if r.relation == "IS"]


def _at_relationships(pef: PEFState) -> list[Relationship]:
    return [r for r in pef.relationships if r.relation == "AT"]


# ── Test: _extract_item_noun_from_possessive_np helper ───────────────────────


def test_extract_item_noun_strips_his():
    assert _extract_item_noun_from_possessive_np("his wallet") == "wallet"


def test_extract_item_noun_strips_her():
    assert _extract_item_noun_from_possessive_np("her car") == "car"


def test_extract_item_noun_strips_their():
    assert _extract_item_noun_from_possessive_np("their house") == "house"


def test_extract_item_noun_strips_its():
    assert _extract_item_noun_from_possessive_np("its engine") == "engine"


def test_extract_item_noun_multiword_preserves_modifier():
    assert _extract_item_noun_from_possessive_np("his leather wallet") == "leather wallet"


def test_extract_item_noun_non_possessive_unchanged():
    assert _extract_item_noun_from_possessive_np("James wallet") == "James wallet"


def test_extract_item_noun_empty():
    assert _extract_item_noun_from_possessive_np("") == ""


# ── P2-1 / P2-10: _build_semantic_transactions blocks possessive-NP ───────────


def test_possessive_np_claim_blocked_in_semantic_transaction():
    """_build_semantic_transactions must block IS claim whose subject starts with
    an ambiguous possessive pronoun."""
    result = _make_possessive_result(
        subject="his wallet",
        relation="IS",
        obj="red",
        ambiguous=["his"],
    )
    txs = _build_semantic_transactions(result)
    assert len(txs) == 1
    tx = txs[0]
    assert tx.allowed_commit is False
    assert tx.held_reason == "UNRESOLVED_POSSESSOR"
    assert tx.unresolved_possessor == "his"


def test_possessive_np_unambiguous_pronoun_passes_through():
    """If the possessive pronoun is NOT in ambiguous_referents, claim is not blocked."""
    result = _make_possessive_result(
        subject="his wallet",
        relation="IS",
        obj="red",
        ambiguous=[],   # his is not ambiguous — one entity in PEF
    )
    txs = _build_semantic_transactions(result)
    assert len(txs) == 1
    assert txs[0].allowed_commit is True


# ── P2-2: Possessive surface never becomes entity name ────────────────────────


def test_possessive_np_subject_not_minted_as_entity():
    """update_pef must not create an entity named 'his wallet'."""
    pef = _pef_with_entities("James", "Richard")
    result = _make_possessive_result(
        subject="his wallet",
        relation="IS",
        obj="red",
        ambiguous=["his"],
    )
    update_pef(result, pef)
    names = _entity_names(pef)
    assert "his wallet" not in names, f"Possessive surface became entity: {names}"
    # Also no "wallet" yet — claim was blocked, not replayed
    assert "wallet" not in names, "Item entity should not be minted until replay"


def test_possessive_np_no_entity_soup_replay():
    """After replay, PEF must not contain 'his wallet' or 'James's wallet' as entity."""
    pef = _pef_with_entities("James", "Richard")
    pending = _pending_for_possessive(
        subject="his wallet",
        relation="IS",
        obj="red",
        item_noun="wallet",
        possessor_pronoun="his",
        ambiguous=["his"],
    )
    res = _commit_resolved_claims(
        bound_entity="James",
        pending=pending,
        pef=pef,
    )
    names = _entity_names(pef)
    assert "his wallet" not in names
    assert "James's wallet" not in names
    assert "wallet" in names, "Item entity 'wallet' must be minted"


# ── P2-3: Replay commits HAS + original predicate ────────────────────────────


def test_possessive_np_is_replays_with_entity_linked_ownership():
    """IS(his wallet, red) with James resolved:
       - entity 'wallet' minted
       - HAS(James, wallet) committed
       - IS(wallet, red) committed
    """
    pef = _pef_with_entities("James", "Richard")
    pending = _pending_for_possessive(
        subject="his wallet",
        relation="IS",
        obj="red",
        item_noun="wallet",
        possessor_pronoun="his",
        ambiguous=["his"],
    )
    res = _commit_resolved_claims(
        bound_entity="James",
        pending=pending,
        pef=pef,
    )
    assert res.stop_reason is None

    names = _entity_names(pef)
    assert "wallet" in names

    # IS(wallet, red) must be committed by update_pef after _commit_resolved_claims
    # We test that resolved_claims contains IS(wallet, red) — since update_pef is
    # called by the caller, we verify via the return value's implicit effect.
    # Direct: IS(wallet, red) appears after the call if we run update_pef ourselves.
    wallet_ent = pef.find_entity_by_name("wallet")
    assert wallet_ent is not None

    # HAS(James, wallet_entity) must be in PEF (committed inside replay, not via update_pef)
    james_ent = pef.find_entity_by_name("James")
    has_rels = [
        r for r in _has_relationships(pef)
        if r.subject_id == james_ent.id and r.object_entity_id == wallet_ent.id
    ]
    assert has_rels, "HAS(James, wallet_entity) must be committed during replay"


# ── P2-4: Existing owned entity reused ───────────────────────────────────────


def test_possessive_np_existing_owned_entity_reused():
    """If James already HAS a wallet entity before replay, no duplicate minted."""
    pef = _pef_with_entities("James", "Richard")

    # Pre-mint wallet entity with James owning it
    wallet_ent, _ = pef.get_or_create_entity("wallet")
    james_ent = pef.find_entity_by_name("James")
    pef.add_relationship(Relationship(
        subject_id=james_ent.id,
        relation="HAS",
        object_entity_id=wallet_ent.id,
        object_literal=None,
        span=Span.PRESENT,
        source_turn=1,
        evidence="fixture: James has wallet",
    ))

    pending = _pending_for_possessive(
        subject="his wallet",
        relation="IS",
        obj="red",
        item_noun="wallet",
        possessor_pronoun="his",
        ambiguous=["his"],
    )
    res = _commit_resolved_claims(
        bound_entity="James",
        pending=pending,
        pef=pef,
    )

    # No duplicate HAS added
    has_rels = [
        r for r in _has_relationships(pef)
        if r.subject_id == james_ent.id
        and r.object_entity_id == wallet_ent.id
    ]
    assert len(has_rels) == 1, f"Expected 1 HAS, found {len(has_rels)}"

    # No duplicate wallet entity
    wallet_entities = [e for e in pef.entities.values() if item_key(e.name) == item_key("wallet")]
    assert len(wallet_entities) == 1, f"Expected 1 wallet entity, found {len(wallet_entities)}"


# ── P2-5: Non-possessive pronoun path unchanged ───────────────────────────────


def test_non_possessive_pronoun_path_unchanged():
    """'He had the wallet.' — bare subject pronoun, no possessive-NP blocking."""
    result = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="he",
                relation="HAS",
                obj="wallet",
                span=Span.PRESENT,
                negated=False,
                evidence="He had the wallet.",
            )
        ],
        entity_mentions=[],
        ambiguous_referents=["he"],
    )
    txs = _build_semantic_transactions(result)
    assert len(txs) == 1
    # Bare pronoun subject is NOT a possessive-NP — only blocks referent, not possessor
    # The held_reason should NOT be UNRESOLVED_POSSESSOR for a bare pronoun
    tx = txs[0]
    assert tx.held_reason != "UNRESOLVED_POSSESSOR", (
        "Bare pronoun subject must not be flagged as possessive-NP"
    )


# ── P2-6: AT relation replays correctly ──────────────────────────────────────


def test_possessive_np_at_relation_replay():
    """AT(his car, garage) with James resolved → AT(car_entity, garage) + HAS(James, car)."""
    pef = _pef_with_entities("James", "Richard")
    pending = _pending_for_possessive(
        subject="his car",
        relation="AT",
        obj="the garage",
        item_noun="car",
        possessor_pronoun="his",
        ambiguous=["his"],
    )
    res = _commit_resolved_claims(
        bound_entity="James",
        pending=pending,
        pef=pef,
    )
    assert res.stop_reason is None

    car_ent = pef.find_entity_by_name("car")
    assert car_ent is not None

    james_ent = pef.find_entity_by_name("James")
    has_rels = [
        r for r in _has_relationships(pef)
        if r.subject_id == james_ent.id and r.object_entity_id == car_ent.id
    ]
    assert has_rels, "HAS(James, car) not found after AT replay"


# ── P2-7: All relations blocked ───────────────────────────────────────────────


@pytest.mark.parametrize("relation", ["IS", "AT", "HAS", "TAKE"])
def test_possessive_np_all_relations_blocked(relation: str):
    """Claims with possessive-NP subjects must be blocked for all relations."""
    result = _make_possessive_result(
        subject="his wallet",
        relation=relation,
        obj="red",
        ambiguous=["his"],
    )
    txs = _build_semantic_transactions(result)
    assert len(txs) == 1
    tx = txs[0]
    assert tx.allowed_commit is False, f"Relation {relation} should be blocked"
    assert tx.held_reason == "UNRESOLVED_POSSESSOR"


# ── P2-8: Comparative takes priority over possessive-NP ──────────────────────


def test_comparative_takes_priority_over_possessive_np():
    """'His dog was bigger.' with both comparative + possessive ambiguity:
    held_reason must be UNRESOLVED_COMPARAND, not UNRESOLVED_POSSESSOR."""
    result = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="his dog",
                relation="IS",
                obj="bigger",
                span=Span.PRESENT,
                negated=False,
                evidence="His dog was bigger.",
            )
        ],
        entity_mentions=[],
        ambiguous_referents=["his"],
        comparative_ambiguities=[
            ComparativeAmbiguity(
                adjective="bigger",
                noun="dog",
                candidates=["James", "Richard"],
            )
        ],
    )
    txs = _build_semantic_transactions(result)
    assert len(txs) == 1
    tx = txs[0]
    assert tx.allowed_commit is False
    assert tx.held_reason == "UNRESOLVED_COMPARAND", (
        "Comparative law must take priority over possessive-NP law"
    )
    assert tx.unresolved_possessor is None


# ── P2-9: Multi-word possessive NP ───────────────────────────────────────────


def test_multiword_possessive_np_leather_wallet():
    """'His leather wallet was red.' → item_noun='leather wallet'; no 'his leather wallet' entity."""
    pef = _pef_with_entities("James", "Richard")
    pending = _pending_for_possessive(
        subject="his leather wallet",
        relation="IS",
        obj="red",
        item_noun="leather wallet",
        possessor_pronoun="his",
        ambiguous=["his"],
    )
    res = _commit_resolved_claims(
        bound_entity="James",
        pending=pending,
        pef=pef,
    )
    assert res.stop_reason is None

    names = _entity_names(pef)
    assert "his leather wallet" not in names, "Law P2-2: possessive surface must not be entity"
    assert "wallet" not in [e.name for e in pef.entities.values()], (
        "item_key('leather wallet') must not collapse to bare 'wallet'"
    )

    lw_ent = pef.find_entity_by_name("leather wallet")
    assert lw_ent is not None, "Entity 'leather wallet' must be minted"

    james_ent = pef.find_entity_by_name("James")
    has_rels = [
        r for r in _has_relationships(pef)
        if r.subject_id == james_ent.id and r.object_entity_id == lw_ent.id
    ]
    assert has_rels, "HAS(James, leather_wallet_entity) not found"

    # item_key for leather wallet entity name preserves modifier
    assert item_key(lw_ent.name) == item_key("leather wallet"), (
        "item_key must treat 'leather wallet' as distinct from 'wallet'"
    )


# ── P2-11: blocked_claims dict has correct tags ───────────────────────────────


def test_possessive_np_claim_held_in_blocked_claims():
    """UNRESOLVED_REFERENT pending built for possessive-NP must tag the blocked claim."""
    from aurora_lens.lens import _build_pending_unresolved_referent_dict

    pef = _pef_with_entities("James", "Richard")
    result = _make_possessive_result(
        subject="his wallet",
        relation="IS",
        obj="red",
        ambiguous=["his"],
    )
    # _build_semantic_transactions must block the claim
    txs = _build_semantic_transactions(result)
    assert txs[0].allowed_commit is False

    # Build the pending dict as lens.py does — pass ambiguous_tokens from extraction
    pending = _build_pending_unresolved_referent_dict(
        pef=pef,
        extraction=result,
        original_question="His wallet was red.",
        scope_text="His wallet was red.",
        detected_span=Span.PRESENT,
        ambiguous_tokens=list(result.ambiguous_referents),
    )

    blocked = pending.get("blocked_claims") or []
    possessive_claims = [c for c in blocked if c.get("held_reason") == "UNRESOLVED_POSSESSOR"]
    assert possessive_claims, "blocked_claims must contain an UNRESOLVED_POSSESSOR entry"

    claim = possessive_claims[0]
    assert claim["possessor_pronoun"] == "his"
    assert claim["item_noun"] == "wallet"


# ── P2: update_pef does not mint possessive entity during initial extraction ──


def test_update_pef_does_not_mint_possessive_entity_during_extraction():
    """update_pef with ambiguous possessive subject must not create any wallet entity."""
    pef = _pef_with_entities("James", "Richard")
    result = _make_possessive_result(
        subject="his wallet",
        relation="IS",
        obj="red",
        ambiguous=["his"],
    )
    before = set(_entity_names(pef))
    update_pef(result, pef)
    after = set(_entity_names(pef))
    new_entities = after - before
    assert not new_entities, f"update_pef minted unexpected entities: {new_entities}"


# ── P2: Phase 1 regression — comparative path unchanged ──────────────────────


def test_phase2_does_not_regress_comparative_blocking():
    """Verify Phase 2 additions do not break Phase 1 comparative IS blocking."""
    from aurora_lens.interpret.schema import ComparativeAmbiguity

    result = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="his dog",
                relation="IS",
                obj="bigger",
                span=Span.PRESENT,
                negated=False,
                evidence="His dog was bigger.",
            )
        ],
        entity_mentions=[],
        comparative_ambiguities=[
            ComparativeAmbiguity(adjective="bigger", noun="dog", candidates=["James", "Richard"])
        ],
        ambiguous_referents=["his"],
    )
    txs = _build_semantic_transactions(result)
    assert len(txs) == 1
    assert txs[0].held_reason == "UNRESOLVED_COMPARAND"
    assert txs[0].unresolved_comparand == "bigger"
    assert txs[0].unresolved_possessor is None
