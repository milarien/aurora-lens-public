"""SemanticTransaction Phase 2b — Referent-Transaction Unification invariant tests.

Laws tested:
  R1. SemanticTransaction is the sole blocking authority.
  R2. Bare pronoun subjects in ambiguous_referents → UNRESOLVED_REFERENT (allowed_commit=False).
  R3. blocked_claims carries held_reason, not is_comparative / is_possessive_np.
  R4. Back-compat: old-format dicts without held_reason route correctly via _infer_held_reason.
  R5. Defense-in-depth guards preserved; no double-blocking artifact.
"""

from __future__ import annotations

import pytest

from aurora_lens.interpret.pef_updater import _build_semantic_transactions, update_pef
from aurora_lens.interpret.schema import (
    ComparativeAmbiguity,
    ExtractedClaim,
    ExtractionResult,
)
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState


# ── Helpers ───────────────────────────────────────────────────────────────────


def _claim(subject: str, relation: str, obj: str) -> ExtractedClaim:
    return ExtractedClaim(
        subject=subject,
        relation=relation,
        obj=obj,
        span=Span.PRESENT,
        negated=False,
        evidence=f"{subject} {relation} {obj}.",
    )


def _pef_with(*names: str) -> PEFState:
    pef = PEFState()
    for name in names:
        pef.get_or_create_entity(name)
    return pef


# ── R2: bare pronoun blocking in _build_semantic_transactions ─────────────────


def test_bare_pronoun_IS_blocked_as_unresolved_referent():
    """IS(he, tall) with he in ambiguous_referents → UNRESOLVED_REFERENT, allowed_commit=False."""
    result = ExtractionResult(
        claims=[_claim("he", "IS", "tall")],
        entity_mentions=[],
        ambiguous_referents=["he"],
    )
    txs = _build_semantic_transactions(result)
    assert len(txs) == 1
    tx = txs[0]
    assert tx.allowed_commit is False
    assert tx.held_reason == "UNRESOLVED_REFERENT"


def test_bare_pronoun_HAS_blocked_as_unresolved_referent():
    """HAS(she, wallet) with she ambiguous → UNRESOLVED_REFERENT."""
    result = ExtractionResult(
        claims=[_claim("she", "HAS", "wallet")],
        entity_mentions=[],
        ambiguous_referents=["she"],
    )
    txs = _build_semantic_transactions(result)
    assert txs[0].allowed_commit is False
    assert txs[0].held_reason == "UNRESOLVED_REFERENT"


def test_bare_pronoun_not_blocked_when_not_ambiguous():
    """IS(he, tall) with empty ambiguous_referents → allowed_commit=True (pronoun resolved)."""
    result = ExtractionResult(
        claims=[_claim("he", "IS", "tall")],
        entity_mentions=[],
        ambiguous_referents=[],
    )
    txs = _build_semantic_transactions(result)
    assert txs[0].allowed_commit is True
    assert txs[0].held_reason is None


def test_comparative_takes_priority_over_bare_pronoun():
    """IS(he, bigger) with he ambiguous AND comparative → UNRESOLVED_COMPARAND (Phase 1 priority)."""
    result = ExtractionResult(
        claims=[_claim("he", "IS", "bigger")],
        entity_mentions=[],
        ambiguous_referents=["he"],
        comparative_ambiguities=[ComparativeAmbiguity("bigger", "stick", ["Alice", "Bob"])],
    )
    txs = _build_semantic_transactions(result)
    assert txs[0].allowed_commit is False
    assert txs[0].held_reason == "UNRESOLVED_COMPARAND"


def test_possessive_np_takes_priority_over_bare_pronoun():
    """IS(his dog, tall) with his ambiguous → UNRESOLVED_POSSESSOR (Phase 2 priority)."""
    result = ExtractionResult(
        claims=[_claim("his dog", "IS", "tall")],
        entity_mentions=[],
        ambiguous_referents=["his"],
    )
    txs = _build_semantic_transactions(result)
    assert txs[0].allowed_commit is False
    assert txs[0].held_reason == "UNRESOLVED_POSSESSOR"


def test_non_ambiguous_pronoun_subject_passes_through():
    """IS(he, tall) when 'he' is NOT in ambiguous_referents → not blocked."""
    result = ExtractionResult(
        claims=[_claim("he", "IS", "tall")],
        entity_mentions=[],
        ambiguous_referents=["she"],  # different pronoun
    )
    txs = _build_semantic_transactions(result)
    assert txs[0].allowed_commit is True
    assert txs[0].held_reason is None


# ── R3: blocked_claims format from _build_pending_unresolved_referent_dict ────


def test_blocked_claims_bare_pronoun_has_held_reason():
    """IS(he, tall) → blocked_claims entry carries held_reason=UNRESOLVED_REFERENT (Law R3)."""
    from aurora_lens.lens import _build_pending_unresolved_referent_dict

    pef = _pef_with("James", "Richard")
    result = ExtractionResult(
        claims=[_claim("he", "IS", "tall")],
        entity_mentions=[],
        ambiguous_referents=["he"],
    )
    pending = _build_pending_unresolved_referent_dict(
        pef=pef,
        extraction=result,
        original_question="He is tall.",
        scope_text="He is tall.",
        detected_span=Span.PRESENT,
        ambiguous_tokens=["he"],
    )
    blocked = pending.get("blocked_claims") or []
    assert len(blocked) == 1
    assert blocked[0].get("held_reason") == "UNRESOLVED_REFERENT"


def test_blocked_claims_possessive_uses_held_reason_not_is_possessive_np():
    """Possessive-NP blocked claim uses held_reason=UNRESOLVED_POSSESSOR, not is_possessive_np."""
    from aurora_lens.lens import _build_pending_unresolved_referent_dict

    pef = _pef_with("James", "Richard")
    result = ExtractionResult(
        claims=[_claim("his wallet", "IS", "red")],
        entity_mentions=[],
        ambiguous_referents=["his"],
    )
    pending = _build_pending_unresolved_referent_dict(
        pef=pef,
        extraction=result,
        original_question="His wallet was red.",
        scope_text="His wallet was red.",
        detected_span=Span.PRESENT,
        ambiguous_tokens=["his"],
    )
    blocked = pending.get("blocked_claims") or []
    poss_claims = [c for c in blocked if c.get("held_reason") == "UNRESOLVED_POSSESSOR"]
    assert poss_claims, "Expected UNRESOLVED_POSSESSOR entry in blocked_claims"
    claim = poss_claims[0]
    assert "is_possessive_np" not in claim
    assert claim["possessor_pronoun"] == "his"
    assert claim["item_noun"] == "wallet"


def test_blocked_claims_comparative_uses_held_reason_not_is_comparative():
    """Comparative blocked claim uses held_reason=UNRESOLVED_COMPARAND, not is_comparative."""
    from aurora_lens.lens import _build_pending_unresolved_referent_dict

    pef = _pef_with("James", "Richard")
    result = ExtractionResult(
        claims=[_claim("he", "IS", "bigger")],
        entity_mentions=[],
        ambiguous_referents=["he"],
        comparative_ambiguities=[ComparativeAmbiguity("bigger", "stick", ["James", "Richard"])],
    )
    pending = _build_pending_unresolved_referent_dict(
        pef=pef,
        extraction=result,
        original_question="He was bigger.",
        scope_text="He was bigger.",
        detected_span=Span.PRESENT,
        ambiguous_tokens=["he"],
    )
    blocked = pending.get("blocked_claims") or []
    comp_claims = [c for c in blocked if c.get("held_reason") == "UNRESOLVED_COMPARAND"]
    assert comp_claims, "Expected UNRESOLVED_COMPARAND entry in blocked_claims"
    claim = comp_claims[0]
    assert "is_comparative" not in claim
    assert claim.get("comparand_adjective") == "bigger"
    assert claim.get("comparand_noun") == "stick"


# ── R4: back-compat via _infer_held_reason ────────────────────────────────────


def test_infer_held_reason_from_held_reason_field():
    from aurora_lens.lens import _infer_held_reason
    assert _infer_held_reason({"held_reason": "UNRESOLVED_COMPARAND"}) == "UNRESOLVED_COMPARAND"
    assert _infer_held_reason({"held_reason": "UNRESOLVED_POSSESSOR"}) == "UNRESOLVED_POSSESSOR"
    assert _infer_held_reason({"held_reason": "UNRESOLVED_REFERENT"}) == "UNRESOLVED_REFERENT"


def test_infer_held_reason_back_compat_is_comparative():
    from aurora_lens.lens import _infer_held_reason
    assert _infer_held_reason({"is_comparative": True}) == "UNRESOLVED_COMPARAND"


def test_infer_held_reason_back_compat_is_possessive_np():
    from aurora_lens.lens import _infer_held_reason
    assert _infer_held_reason({"is_possessive_np": True}) == "UNRESOLVED_POSSESSOR"


def test_infer_held_reason_plain_dict_defaults_to_referent():
    from aurora_lens.lens import _infer_held_reason
    assert _infer_held_reason({"subject": "he", "relation": "IS", "obj": "tall"}) == "UNRESOLVED_REFERENT"


def test_back_compat_old_is_comparative_commits_correctly():
    """Old-format dict with is_comparative=True → comparand resolution, not IS commit."""
    from aurora_lens.lens import _commit_resolved_claims

    pef = _pef_with("James", "Richard", "Carol")
    pending = {
        "original_question": "He was bigger.",
        "original_span": "present",
        "ambiguous_referents": ["he"],
        "blocked_claims": [
            {
                "subject": "he",
                "relation": "IS",
                "obj": "bigger",
                "span": "present",
                "negated": False,
                "evidence": "He was bigger.",
                "is_comparative": True,       # old format — back-compat
                "comparand_adjective": "bigger",
                "comparand_noun": "stick",
            }
        ],
    }
    # James, Richard, and Carol all have sticks.
    # After excluding James (the resolved subject), 2 comparands remain → pending_comparand.
    james = pef.find_entity_by_name("James")
    richard = pef.find_entity_by_name("Richard")
    carol = pef.find_entity_by_name("Carol")
    from aurora_lens.pef.state import Relationship
    from aurora_lens.pef.span import Span as _Span
    for ent in (james, richard, carol):
        pef.add_relationship(Relationship(
            subject_id=ent.id, relation="HAS", object_entity_id=None, object_literal="stick",
            span=_Span.PRESENT, source_turn=1, evidence=f"{ent.name} has stick.",
        ))

    res = _commit_resolved_claims("James", pending, pef)
    # 2 eligible comparands (Richard, Carol) → pending_comparand issued, not IS committed
    assert res.pending_comparand is not None
    is_rels = [r for r in pef.relationships if r.relation == "IS"]
    assert not is_rels, "IS(James, bigger) must not be committed via back-compat path"


def test_back_compat_old_is_possessive_np_commits_correctly():
    """Old-format dict with is_possessive_np=True → entity-linked ownership on replay."""
    from aurora_lens.lens import _commit_resolved_claims

    pef = _pef_with("James", "Richard")
    pending = {
        "original_question": "His wallet was red.",
        "original_span": "present",
        "ambiguous_referents": ["his"],
        "blocked_claims": [
            {
                "subject": "his wallet",
                "relation": "IS",
                "obj": "red",
                "span": "present",
                "negated": False,
                "evidence": "His wallet was red.",
                "is_possessive_np": True,     # old format
                "possessor_pronoun": "his",
                "item_noun": "wallet",
            }
        ],
    }
    res = _commit_resolved_claims("James", pending, pef)
    assert res.stop_reason is None
    wallet_ent = pef.find_entity_by_name("wallet")
    assert wallet_ent is not None, "wallet entity should be minted"
    is_rels = [r for r in pef.relationships if r.relation == "IS"]
    assert is_rels, "IS(wallet, red) should be committed"


def test_back_compat_plain_referent_dict_commits_correctly():
    """Old-format dict with no tags (bare pronoun) → commits as normal IS(James, tall)."""
    from aurora_lens.lens import _commit_resolved_claims

    pef = _pef_with("James", "Richard")
    pending = {
        "original_question": "He is tall.",
        "original_span": "present",
        "ambiguous_referents": ["he"],
        "blocked_claims": [
            {
                "subject": "he",
                "relation": "IS",
                "obj": "tall",
                "span": "present",
                "negated": False,
                "evidence": "He is tall.",
                # no held_reason, no is_comparative, no is_possessive_np — old plain format
            }
        ],
    }
    res = _commit_resolved_claims("James", pending, pef)
    assert res.stop_reason is None
    james_ent = pef.find_entity_by_name("James")
    is_rels = [
        r for r in pef.relationships
        if r.relation == "IS" and r.subject_id == james_ent.id
    ]
    assert is_rels, "IS(James, tall) must be committed after pronoun resolution"


# ── R1+R2: update_pef does not commit bare pronoun IS when ambiguous ──────────


def test_bare_pronoun_IS_not_committed_when_ambiguous():
    """IS(he, tall) with he in ambiguous_referents must not be committed to PEF (Law R1/R2)."""
    pef = _pef_with("James", "Richard")
    result = ExtractionResult(
        claims=[_claim("he", "IS", "tall")],
        entity_mentions=[],
        ambiguous_referents=["he"],
    )
    before_is = [r for r in pef.relationships if r.relation == "IS"]
    update_pef(result, pef)
    after_is = [r for r in pef.relationships if r.relation == "IS"]
    assert after_is == before_is, "IS(he, tall) must not be committed when 'he' is ambiguous"


def test_bare_pronoun_IS_committed_when_not_ambiguous():
    """IS(he, tall) with no ambiguity → committed via pronoun binding."""
    pef = _pef_with("James", "Richard")
    # Put a binding so 'he' resolves to James
    james_ent = pef.find_entity_by_name("James")
    pef.resolve_pronoun("he@token_0", james_ent.id)

    result = ExtractionResult(
        claims=[_claim("he", "IS", "tall")],
        entity_mentions=[],
        ambiguous_referents=[],   # not ambiguous
    )
    update_pef(result, pef)
    # 'he' resolves → IS(James, tall) committed
    is_rels = [r for r in pef.relationships if r.relation == "IS"]
    assert is_rels, "IS should be committed when pronoun is unambiguous"
