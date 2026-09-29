"""ContinuationTransaction invariant tests — Phase 6.

Invariants protected:
- A ContinuationTransaction holds all state needed to reconstruct the
  continuation response without calling the LLM.
- A closed commitment (commitment_closed=True) is never re-opened by
  a continuation turn: applies() returning True does NOT change commitment_closed.
- ResolutionCondition.matches() gates closure: wrong turn_act or missing
  candidate keeps the hold alive.
- A continuation with interaction_open=False never applies() to any turn.
- Escalation produces a terminal ContinuationTransaction (interaction_open=False,
  resolution_condition=None).
"""

from __future__ import annotations

import pytest

from aurora_lens.interpret.schema import (
    ContinuationTransaction,
    ExtractedClaim,
    ResolutionCondition,
    SemanticTransaction,
)
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState


# ── Helpers ───────────────────────────────────────────────────────────────────


def _hold_claim(obj: str = "pending question") -> ExtractedClaim:
    return ExtractedClaim(
        subject="__continuation__",
        relation="HOLD",
        obj=obj,
        span=Span.PRESENT,
        negated=False,
        evidence="",
    )


def _continuation(
    *,
    constraint_kind: str = "UNRESOLVED_REFERENT",
    candidates: list[str] | None = None,
    required_acts: list[str] | None = None,
    requires_match: bool = False,
    commitment_closed: bool = True,
    interaction_open: bool = True,
) -> ContinuationTransaction:
    cond = ResolutionCondition(
        constraint_kind=constraint_kind,
        candidate_entities=candidates or [],
        required_turn_acts=required_acts or ["CLARIFY", "QUERY"],
        requires_candidate_match=requires_match,
    )
    return ContinuationTransaction(
        claim=_hold_claim(),
        commitment_closed=commitment_closed,
        interaction_open=interaction_open,
        resolution_condition=cond,
    )


# ── Law: ResolutionCondition gates closure ────────────────────────────────────


def test_continuation_resolves_only_on_matching_condition():
    """ResolutionCondition.matches() is the sole gate for continuation closure."""
    # UNRESOLVED_REFERENT: any CLARIFY/QUERY act resolves (no candidate match required)
    cond_referent = ResolutionCondition(
        constraint_kind="UNRESOLVED_REFERENT",
        candidate_entities=["James", "Alice"],
        required_turn_acts=["CLARIFY", "QUERY"],
        requires_candidate_match=False,
    )
    assert cond_referent.matches("anything at all", "CLARIFY") is True
    assert cond_referent.matches("something unrelated", "QUERY") is True
    assert cond_referent.matches("anything", "TELL") is False   # wrong act

    # UNRESOLVED_COMPARAND: must name a candidate
    cond_comparand = ResolutionCondition(
        constraint_kind="UNRESOLVED_COMPARAND",
        candidate_entities=["Richard", "Lucy"],
        required_turn_acts=["CLARIFY", "QUERY"],
        requires_candidate_match=True,
    )
    assert cond_comparand.matches("I meant Richard", "CLARIFY") is True
    assert cond_comparand.matches("I'm not sure", "CLARIFY") is False   # no candidate named
    assert cond_comparand.matches("Richard", "TELL") is False            # wrong act


# ── Law: closed commitment is never re-opened ─────────────────────────────────


def test_closed_commitment_cannot_reopen():
    """applies() returning True does not alter commitment_closed — it stays True."""
    tx = _continuation(
        constraint_kind="UNRESOLVED_REFERENT",
        commitment_closed=True,
        interaction_open=True,
    )
    assert tx.commitment_closed is True

    # Even when the condition is satisfied, commitment_closed must remain True
    assert tx.applies("CLARIFY", "James") is True
    assert tx.commitment_closed is True   # never re-opened

    # Repeated applies() still leaves commitment_closed intact
    tx.applies("CLARIFY", "anything")
    assert tx.commitment_closed is True


# ── Law: continuation response from stored state, not LLM ────────────────────


def test_continuation_response_from_stored_transaction_not_llm():
    """from_pending_dict() captures all state needed to reconstruct the response.

    The invariant: ContinuationTransaction carries constraint_kind, candidate_entities,
    and the original question — everything needed to re-issue the clarification prompt
    without an LLM call.
    """
    pending = {
        "original_question": "James gave them to Alice.",
        "failed_constraint": "UNRESOLVED_REFERENT",
        "ambiguous_referents": ["them"],
        "candidate_entities": ["wallet", "phone"],
        "original_span": "present",
        "blocked_proposition": "James gave them to Alice.",
        "blocked_claims": [],
        "clarification_prompt": None,
    }
    tx = ContinuationTransaction.from_pending_dict(pending)

    # All response state lives in the typed transaction — no LLM needed
    assert tx.resolution_condition is not None
    rc = tx.resolution_condition
    assert rc.constraint_kind == "UNRESOLVED_REFERENT"
    assert "wallet" in rc.candidate_entities
    assert "phone" in rc.candidate_entities
    assert "James gave them to Alice" in tx.claim.obj   # original question preserved
    assert tx.commitment_closed is True
    assert tx.interaction_open is True
    assert tx.allowed_commit is False
    assert tx.held_reason == "CONTINUATION"

    # UNRESOLVED_REFERENT: any CLARIFY/QUERY resolves (no candidate match required)
    assert rc.requires_candidate_match is False
    assert tx.applies("CLARIFY", "I meant the wallet") is True
    assert tx.applies("TELL", "anything") is False    # wrong act


# ── Law: hold survives unrelated input ───────────────────────────────────────


def test_multi_turn_hold_survives_unrelated_input():
    """Unrelated turn_acts and non-matching texts leave the hold active."""
    tx = _continuation(
        constraint_kind="UNRESOLVED_COMPARAND",
        candidates=["Richard", "Alice"],
        required_acts=["CLARIFY", "QUERY"],
        requires_match=True,
    )

    # TELL act — not in required_turn_acts → hold survives
    assert tx.applies("TELL", "James is happy") is False

    # CLARIFY act but no candidate named → hold survives
    assert tx.applies("CLARIFY", "I'm not sure what you mean") is False

    # CLARIFY act naming a candidate → hold resolves
    assert tx.applies("CLARIFY", "I meant Richard") is True

    # After checking applies(), the transaction is unchanged — still active
    assert tx.interaction_open is True
    assert tx.commitment_closed is True


# ── Law: escalation → terminal (interaction_open=False) ──────────────────────


def test_escalation_continuation_produces_hard_stop_on_repeat():
    """escalate() returns a terminal ContinuationTransaction that never applies()."""
    tx = _continuation(
        constraint_kind="UNRESOLVED_REFERENT",
        interaction_open=True,
    )
    assert tx.applies("CLARIFY", "anything") is True   # base transaction is open

    escalated = tx.escalate()

    assert escalated.continuation_kind == "escalation"
    assert escalated.interaction_open is False      # hard stop
    assert escalated.commitment_closed is True      # commitment stays closed
    assert escalated.resolution_condition is None   # no auto-close path

    # Escalated continuation never applies to any turn
    assert escalated.applies("CLARIFY", "anything") is False
    assert escalated.applies("QUERY", "James") is False
    assert escalated.applies("TELL", "anything") is False

    # Original transaction is unchanged (escalate returns a copy)
    assert tx.interaction_open is True


# ── Phase 6 — PEFState carries pending_transaction field ─────────────────────


def test_pef_state_has_pending_transaction_field():
    """PEFState.pending_transaction exists and defaults to None (Phase 6 integration point)."""
    pef = PEFState()
    assert hasattr(pef, "pending_transaction")
    assert pef.pending_transaction is None

    tx = _continuation()
    pef.pending_transaction = tx
    assert pef.pending_transaction is tx
    assert pef.pending_transaction.applies("CLARIFY", "James") is True


# ── Back-compat: from_pending_dict() for UNRESOLVED_COMPARAND ────────────────


def test_from_pending_dict_comparand_requires_candidate_match():
    """UNRESOLVED_COMPARAND hold requires the user to name a candidate to close."""
    pending = {
        "original_question": "Whose dog was bigger?",
        "failed_constraint": "UNRESOLVED_COMPARAND",
        "candidate_entities": ["Richard", "Lucy"],
        "comparand_adjective": "bigger",
        "comparand_noun": "dog",
        "original_span": "present",
    }
    tx = ContinuationTransaction.from_pending_dict(pending)

    rc = tx.resolution_condition
    assert rc is not None
    assert rc.constraint_kind == "UNRESOLVED_COMPARAND"
    assert rc.requires_candidate_match is True
    assert "Richard" in rc.candidate_entities

    # Must name a candidate
    assert tx.applies("CLARIFY", "It was Richard") is True
    assert tx.applies("CLARIFY", "I don't know") is False


# ── Back-compat: from_pending_dict() with no candidates ──────────────────────


def test_from_pending_dict_no_candidates_does_not_require_match():
    """When no candidates are known, requires_candidate_match stays False."""
    pending = {
        "original_question": "Who has them?",
        "failed_constraint": "state_native_ambiguity",
        "candidate_entities": [],
        "original_span": "present",
    }
    tx = ContinuationTransaction.from_pending_dict(pending)

    rc = tx.resolution_condition
    assert rc is not None
    assert rc.requires_candidate_match is False   # no candidates → no match required
    assert tx.applies("CLARIFY", "anything") is True
