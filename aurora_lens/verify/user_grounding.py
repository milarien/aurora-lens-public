"""Structured same-turn user grounding for verification (Lens → Checker bridge).

PEF stores user-admitted facts with ``provenance="user_input"`` and ``source_turn``.
RAG corpus turns also admit ``retrieved_context`` relationships from the bounded
``Context:`` block. This module snapshots those edges for the active turn so the
checker can ground assistant restatements against committed structure — not only
raw substring or ad hoc numeric echo heuristics on ``user_input`` text.
"""

from __future__ import annotations

from dataclasses import dataclass

from aurora_lens.pef.state import PEFState, Relationship

PROVENANCE_RETRIEVED_CONTEXT = "retrieved_context"


@dataclass(frozen=True)
class UserGroundingContext:
    """Immutable snapshot of same-turn evidence edges for verification."""

    effective_user_text: str | None
    turn: int | None
    user_committed_relationships: tuple[Relationship, ...]
    retrieved_context_relationships: tuple[Relationship, ...] = ()
    retrieved_context_text: str | None = None

    @property
    def same_turn_evidence_relationships(self) -> tuple[Relationship, ...]:
        return self.user_committed_relationships + self.retrieved_context_relationships


def build_user_grounding_context(
    pef: PEFState,
    turn: int | None,
    effective_user_text: str | None,
    *,
    retrieved_context_text: str | None = None,
) -> UserGroundingContext:
    """Collect same-turn relationships admitted from user input and retrieval.

    When ``turn`` is ``None``, the snapshot is empty (checker falls back to legacy
    behaviour only — callers should pass the Lens/PEF turn index when available).
    """
    if turn is None:
        return UserGroundingContext(
            effective_user_text=effective_user_text,
            turn=None,
            user_committed_relationships=(),
            retrieved_context_text=retrieved_context_text,
        )
    committed = tuple(
        r
        for r in pef.relationships
        if r.provenance == "user_input" and r.source_turn == turn
    )
    retrieved = tuple(
        r
        for r in pef.relationships
        if r.provenance == PROVENANCE_RETRIEVED_CONTEXT and r.source_turn == turn
    )
    ctx_text = retrieved_context_text.strip() if retrieved_context_text else None
    if ctx_text == "":
        ctx_text = None
    return UserGroundingContext(
        effective_user_text=effective_user_text,
        turn=turn,
        user_committed_relationships=committed,
        retrieved_context_relationships=retrieved,
        retrieved_context_text=ctx_text,
    )
