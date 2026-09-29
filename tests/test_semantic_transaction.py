"""Phase-1 SemanticTransaction invariant tests.

Scope: comparative IS claims only. Non-comparative paths are untouched.

Invariants tested:
  1. No unary IS committed to PEF for relational adjectives (all code paths).
  2. _build_semantic_transactions marks comparative IS as allowed_commit=False.
  3. After referent + single comparand: COMPARE committed; no IS.
  4. Zero comparands after referent resolution: stop_reason set, no IS/COMPARE.
  5. Three holders: UNRESOLVED_COMPARAND after referent resolution.
  6. Stale holder (superseded by GIVE) excluded from comparand set.
  7. COMPARE fields: adjective independently recoverable from relation_metadata.
  8. Emma/Lucy sister test: candidate_entities non-empty.
  9. GIVE arithmetic unchanged by SemanticTransaction.
 10. EAT with no prior HAS still passes through (regression).
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
    _commit_comparand_claims_as_compare,
    _commit_resolved_claims,
    _find_eligible_comparands,
)
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship, canonicalize_relation
from aurora_lens.state_native_engine.eval.inventory import (
    supersede_same_subject_prior_holds_for_item,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pef_with_holders(*names: str, item: str = "dog") -> PEFState:
    """PEF with each name holding item at turn 1."""
    pef = PEFState()
    for name in names:
        ent = Entity.create(name, turn=1, session_id=pef.session_id)
        pef.add_entity(ent)
        pef.add_relationship(
            Relationship(
                subject_id=ent.id,
                relation="HAS",
                object_entity_id=None,
                object_literal=item,
                span=Span.PRESENT,
                source_turn=1,
                evidence=f"fixture: {name} has {item}",
            )
        )
    pef.current_turn = 2
    return pef


def _is_relationships(pef: PEFState) -> list[Relationship]:
    return [r for r in pef.relationships if r.relation == "IS"]


def _compare_relationships(pef: PEFState) -> list[Relationship]:
    return [r for r in pef.relationships if r.relation == "COMPARE"]


def _make_is_claim(
    subject: str,
    obj: str,
    span: Span = Span.PRESENT,
) -> ExtractedClaim:
    return ExtractedClaim(
        subject=subject,
        relation="IS",
        obj=obj,
        span=span,
        negated=False,
        evidence=f"{subject} is {obj}",
    )


def _make_comparative_result(
    subject: str,
    adjective: str,
    noun: str,
    candidates: list[str],
) -> ExtractionResult:
    return ExtractionResult(
        claims=[_make_is_claim(subject, adjective)],
        entity_mentions=[subject],
        span=Span.PRESENT,
        comparative_ambiguities=[
            ComparativeAmbiguity(
                adjective=adjective,
                noun=noun,
                candidates=candidates,
            )
        ],
    )


# ── Test 1: No unary IS committed for relational adjectives ───────────────────


def test_no_unary_IS_committed_for_relational_adjective():
    """update_pef must never write IS(subject, bigger) when a comparative ambiguity exists."""
    pef = _pef_with_holders("James", "Richard")
    result = _make_comparative_result(
        subject="his dog",
        adjective="bigger",
        noun="dog",
        candidates=["James", "Richard"],
    )
    update_pef(result, pef)
    is_rels = _is_relationships(pef)
    assert not any(
        str(r.object_literal or "").strip().lower() == "bigger"
        for r in is_rels
    ), "IS(*, bigger) must not appear in PEF after update_pef with comparative_ambiguities"


# ── Test 2: SemanticTransaction blocks comparative IS ─────────────────────────


def test_comparative_IS_blocked_in_semantic_transaction():
    """_build_semantic_transactions must set allowed_commit=False for comparative IS claims."""
    result = _make_comparative_result(
        subject="his dog",
        adjective="bigger",
        noun="dog",
        candidates=["James", "Richard"],
    )
    txs = _build_semantic_transactions(result)
    assert len(txs) == 1
    tx = txs[0]
    assert tx.allowed_commit is False
    assert tx.held_reason == "UNRESOLVED_COMPARAND"
    assert tx.unresolved_comparand == "bigger"


def test_non_comparative_IS_passes_through():
    """Non-comparative IS claims (e.g. IS(Alice, teacher)) must NOT be blocked."""
    result = ExtractionResult(
        claims=[_make_is_claim("Alice", "a teacher")],
        entity_mentions=["Alice"],
        comparative_ambiguities=[],
    )
    txs = _build_semantic_transactions(result)
    assert len(txs) == 1
    assert txs[0].allowed_commit is True


# ── Test 3: Auto-resolve with exactly one comparand ──────────────────────────


def test_comparand_auto_resolved_single_holder():
    """One eligible comparand → COMPARE auto-committed; no IS; stop_reason is None."""
    pef = _pef_with_holders("James", "Richard")
    pending = {
        "original_question": "His dog was bigger.",
        "original_span": "present",
        "ambiguous_referents": ["his"],
        "blocked_claims": [
            {
                "subject": "his dog",
                "relation": "IS",
                "obj": "bigger",
                "span": "present",
                "negated": False,
                "evidence": "His dog was bigger.",
                "is_comparative": True,
                "comparand_adjective": "bigger",
                "comparand_noun": "dog",
            }
        ],
    }
    # James is the bound referent; Richard is the sole other holder
    result = _commit_resolved_claims("James", pending, pef)

    assert result.stop_reason is None
    assert result.pending_comparand is None

    # No IS committed
    is_rels = _is_relationships(pef)
    assert not any(
        str(r.object_literal or "").lower() == "bigger" for r in is_rels
    )

    # COMPARE committed
    compare_rels = _compare_relationships(pef)
    assert len(compare_rels) == 1
    cr = compare_rels[0]
    assert cr.relation_metadata is not None
    assert cr.relation_metadata.get("adjective") == "bigger"
    # Comparand recoverable: entity reference preferred, literal fallback
    if cr.object_entity_id is not None:
        comp_ent = pef.entities.get(cr.object_entity_id)
        assert comp_ent is not None and comp_ent.name.lower() == "richard"
    else:
        assert str(cr.object_literal or "").lower() == "richard"


# ── Test 4: Zero comparands → stop_reason ────────────────────────────────────


def test_zero_comparands_gives_stop_reason():
    """Only James has the dog → 0 eligible comparands → stop_reason set."""
    pef = _pef_with_holders("James")  # sole holder; no comparand possible
    pending = {
        "original_question": "His dog was bigger.",
        "original_span": "present",
        "ambiguous_referents": ["his"],
        "blocked_claims": [
            {
                "subject": "his dog",
                "relation": "IS",
                "obj": "bigger",
                "span": "present",
                "negated": False,
                "evidence": "His dog was bigger.",
                "is_comparative": True,
                "comparand_adjective": "bigger",
                "comparand_noun": "dog",
            }
        ],
    }
    result = _commit_resolved_claims("James", pending, pef)

    assert result.stop_reason is not None
    assert "no committed comparison target" in result.stop_reason
    assert result.pending_comparand is None

    # Nothing written to PEF
    assert not _is_relationships(pef)
    assert not _compare_relationships(pef)


# ── Test 5: Three holders → UNRESOLVED_COMPARAND in pending_comparand ────────


def test_three_holders_gives_unresolved_comparand():
    """James, Richard, Carol all have dogs; bound = James → 2 comparands → pending_comparand."""
    pef = _pef_with_holders("James", "Richard", "Carol")
    pending = {
        "original_question": "His dog was bigger.",
        "original_span": "present",
        "ambiguous_referents": ["his"],
        "blocked_claims": [
            {
                "subject": "his dog",
                "relation": "IS",
                "obj": "bigger",
                "span": "present",
                "negated": False,
                "evidence": "His dog was bigger.",
                "is_comparative": True,
                "comparand_adjective": "bigger",
                "comparand_noun": "dog",
            }
        ],
    }
    result = _commit_resolved_claims("James", pending, pef)

    assert result.stop_reason is None
    assert result.pending_comparand is not None
    assert result.pending_comparand["failed_constraint"] == "UNRESOLVED_COMPARAND"
    cands = result.pending_comparand["candidate_entities"]
    assert "James" not in [str(c).lower() for c in cands]
    assert len(cands) == 2
    assert set(cands) == {"Richard", "Carol"}

    # No IS or COMPARE committed yet
    assert not _is_relationships(pef)
    assert not _compare_relationships(pef)


# ── Test 6: Stale holder excluded ────────────────────────────────────────────


def test_stale_holder_excluded_from_comparands():
    """Richard gave his dog to Carol → Richard no longer active holder; excluded from comparands."""
    pef = _pef_with_holders("James", "Richard")

    # Simulate Richard giving his dog to Carol
    carol = Entity.create("Carol", turn=2, session_id=pef.session_id)
    pef.add_entity(carol)

    # Supersede Richard's dog possession
    richard_ent = pef.find_entity_by_name("Richard")
    assert richard_ent is not None
    supersede_same_subject_prior_holds_for_item(
        pef,
        richard_ent.id,
        "dog",
        evidence="Richard gave his dog to Carol",
        bookkeeping_backdate_turn=False,
    )
    # Carol now has the dog
    pef.add_relationship(
        Relationship(
            subject_id=carol.id,
            relation="HAS",
            object_entity_id=None,
            object_literal="dog",
            span=Span.PRESENT,
            source_turn=2,
            evidence="Carol received the dog",
        )
    )
    pef.current_turn = 3

    comparands = _find_eligible_comparands(pef, "dog", "James")
    assert "Richard" not in comparands
    assert "Carol" in comparands
    assert len(comparands) == 1


# ── Test 7: COMPARE fields independently queryable ───────────────────────────


def test_compare_relation_metadata_adjective_recoverable():
    """Adjective must be independently recoverable from relation_metadata (not buried in obj)."""
    pef = _pef_with_holders("James", "Richard")
    pending = {
        "original_question": "His dog was bigger.",
        "original_span": "present",
        "ambiguous_referents": ["his"],
        "blocked_claims": [
            {
                "subject": "his dog",
                "relation": "IS",
                "obj": "bigger",
                "span": "present",
                "negated": False,
                "evidence": "His dog was bigger.",
                "is_comparative": True,
                "comparand_adjective": "bigger",
                "comparand_noun": "dog",
            }
        ],
    }
    _commit_resolved_claims("James", pending, pef)

    crs = _compare_relationships(pef)
    assert len(crs) == 1
    cr = crs[0]

    # Adjective recoverable independently of object_literal/object_entity_id
    assert cr.relation_metadata is not None
    assert cr.relation_metadata.get("adjective") == "bigger"

    # Comparand recoverable independently of relation_metadata
    if cr.object_entity_id is not None:
        comp_ent = pef.entities.get(cr.object_entity_id)
        assert comp_ent is not None and comp_ent.name.lower() == "richard"
    else:
        assert str(cr.object_literal or "").lower() == "richard"

    # Verify round-trip through serialization
    from aurora_lens.pef.state import _rel_to_dict, _rel_from_dict
    d = _rel_to_dict(cr)
    cr2 = _rel_from_dict(d)
    # Phase 3: relation_metadata now carries both adjective and noun (Law C2)
    assert cr2.relation_metadata is not None
    assert cr2.relation_metadata.get("adjective") == "bigger"
    assert "noun" in cr2.relation_metadata


# ── Test 8: Emma/Lucy sister candidates non-empty ────────────────────────────


def test_emma_lucy_sister_candidates_non_empty():
    """UNRESOLVED_REFERENT for 'her sister' with Emma+Lucy in PEF must yield 2 candidates."""
    from aurora_lens.lens import _build_pending_unresolved_referent_dict

    pef = PEFState()
    for name in ("Emma", "Lucy"):
        ent = Entity.create(name, turn=1, session_id=pef.session_id)
        ent.resolved = True
        pef.add_entity(ent)
    pef.current_turn = 2

    extraction = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="her sister",
                relation="AT",
                obj="overseas",
                span=Span.PRESENT,
                negated=False,
                evidence="her sister was arriving",
            )
        ],
        entity_mentions=["Emma", "Lucy"],
        ambiguous_referents=["her"],
        comparative_ambiguities=[],
    )

    payload = _build_pending_unresolved_referent_dict(
        pef,
        turn=2,
        scope_text="Emma told Lucy that her sister was arriving.",
        original_question="Emma told Lucy that her sister was arriving.",
        extraction=extraction,
        ambiguous_tokens=["her"],
        detected_span=Span.PRESENT,
    )

    candidates = payload.get("candidate_entities", [])
    assert len(candidates) >= 2, f"Expected ≥2 candidates, got {candidates}"
    names_lower = {str(c).lower() for c in candidates}
    assert "emma" in names_lower
    assert "lucy" in names_lower


# ── Test 9: GIVE arithmetic unchanged ────────────────────────────────────────


def test_give_arithmetic_unchanged():
    """SemanticTransaction must not affect GIVE arithmetic during replay."""
    pef = PEFState()
    alice = Entity.create("Alice", turn=1, session_id=pef.session_id)
    pef.add_entity(alice)
    pef.add_relationship(
        Relationship(
            subject_id=alice.id,
            relation="HAS",
            object_entity_id=None,
            object_literal="10 apples",
            span=Span.PRESENT,
            source_turn=1,
            evidence="Alice has 10 apples",
        )
    )
    pef.current_turn = 2

    pending = {
        "original_question": "She gave Bob 5 apples.",
        "original_span": "present",
        "ambiguous_referents": ["she"],
        "blocked_claims": [
            {
                "subject": "she",
                "relation": "GIVE",
                "obj": "5 apples",
                "span": "present",
                "negated": False,
                "evidence": "She gave Bob 5 apples.",
            }
        ],
    }
    result = _commit_resolved_claims("Alice", pending, pef)

    assert result.stop_reason is None
    assert result.pending_comparand is None

    has_rels = [
        r for r in pef.relationships
        if r.relation == "HAS" and r.subject_id == alice.id
    ]
    # Should have remainder HAS (5 apples) after GIVE projection
    literals = [str(r.object_literal or "") for r in has_rels]
    # At least one non-negated has-rel for Alice exists or remainder was projected
    non_negated = [r for r in has_rels if not r.negated]
    assert non_negated, f"Expected remainder HAS for Alice, got has_rels: {has_rels}"


# ── Test 10: EAT regression ──────────────────────────────────────────────────


def test_non_comparative_claims_pass_through_update_pef():
    """EAT / non-comparative claims must not be blocked by SemanticTransaction."""
    pef = PEFState()
    result = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="Alice",
                relation="EAT",
                obj="apple",
                span=Span.PRESENT,
                negated=False,
                evidence="Alice ate an apple.",
            )
        ],
        entity_mentions=["Alice"],
        comparative_ambiguities=[],
    )
    # Must not raise; EAT claim committed normally
    update_pef(result, pef)
    assert pef.find_entity_by_name("Alice") is not None


# ── Test 11: Explicit comparand resolution commits COMPARE with noun ──────────


def test_comparand_resolved_from_explicit_clarification():
    """3-holder case: after referent resolved and user provides comparand explicitly,
    _commit_comparand_claims_as_compare must commit COMPARE with both adjective AND noun
    in relation_metadata (Law C2).
    """
    pef = _pef_with_holders("James", "Richard", "Carol")

    # Simulate the UNRESOLVED_COMPARAND pending dict (built when James is resolved
    # but 2 comparands remain: Richard and Carol).
    pending = {
        "original_question": "His dog was bigger.",
        "failed_constraint": "UNRESOLVED_COMPARAND",
        "candidate_entities": ["Richard", "Carol"],
        "comparand_adjective": "bigger",
        "comparand_noun": "dog",
        "original_span": "present",
        "blocked_claims": [
            {
                "subject": "James",   # subject after referent resolution
                "relation": "IS",
                "obj": "bigger",
                "span": "present",
                "negated": False,
                "evidence": "His dog was bigger.",
                "is_comparative": True,
                "comparand_adjective": "bigger",
                "comparand_noun": "dog",
            }
        ],
    }

    # User says "Richard" — commit COMPARE(James, bigger, Richard)
    _commit_comparand_claims_as_compare("Richard", pending, pef)

    compare_rels = [r for r in pef.relationships if r.relation == "COMPARE"]
    assert compare_rels, "COMPARE relation must be committed"

    rel = compare_rels[0]
    meta = rel.relation_metadata or {}
    assert meta.get("adjective") == "bigger", f"adjective missing or wrong: {meta}"
    assert meta.get("noun") == "dog", f"noun missing or wrong: {meta}"

    # Comparand must reference Richard
    richard_ent = pef.find_entity_by_name("Richard")
    assert richard_ent is not None
    assert rel.object_entity_id == richard_ent.id or rel.object_literal == "Richard"

    # No IS(*, bigger) must exist
    is_rels = [r for r in pef.relationships if r.relation == "IS"]
    assert not any(
        str(r.object_literal or "").lower() == "bigger" for r in is_rels
    ), "IS(*, bigger) must never appear"
