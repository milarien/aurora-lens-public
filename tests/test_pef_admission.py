"""Unit tests: ``update_pef`` admission (hostile contradiction vs explicit revision)."""

from __future__ import annotations

from aurora_lens.interpret.pef_updater import update_pef
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.interpret.turn_act import TurnAct
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship


def _emma_red_book_pef() -> PEFState:
    pef = PEFState()
    emma, _ = pef.get_or_create_entity("Emma")
    pef.add_relationship(
        Relationship(
            subject_id=emma.id,
            relation="HAS",
            object_entity_id=None,
            object_literal="red book",
            span=Span.PRESENT,
            source_turn=1,
            evidence="Emma has a red book.",
        )
    )
    return pef


def test_update_pef_skip_all_when_user_text_conflicts_even_if_claims_empty() -> None:
    """If extraction omits claims, user-text conflict still blocks claim application."""
    pef = _emma_red_book_pef()
    ext = ExtractionResult(claims=[], span=Span.PRESENT)
    update_pef(
        ext,
        pef,
        user_text="Actually Emma's book is blue. What colour is Emma's book?",
    )
    assert "red" in pef.to_context_summary().lower()


def test_update_pef_skips_hostile_blue_has_when_user_text_no_revision() -> None:
    """Same-slot colour flip must not commit when the turn is not a revision act."""
    pef = _emma_red_book_pef()
    ext = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="Emma",
                relation="HAS",
                obj="blue book",
                span=Span.PRESENT,
                negated=False,
                evidence="Actually Emma's book is blue.",
            )
        ],
        span=Span.PRESENT,
    )
    update_pef(
        ext,
        pef,
        user_text="Actually Emma's book is blue. What colour is Emma's book?",
    )
    ctx = pef.to_context_summary().lower()
    assert "red" in ctx
    assert "blue book" not in ctx


def test_update_pef_allows_blue_has_with_explicit_correction_user_text() -> None:
    pef = _emma_red_book_pef()
    ext = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="Emma",
                relation="HAS",
                obj="blue book",
                span=Span.PRESENT,
                negated=False,
                evidence="Correction: Emma's book is blue, not red.",
            )
        ],
        span=Span.PRESENT,
    )
    update_pef(
        ext,
        pef,
        user_text="Correction: I misspoke earlier. Emma's book is blue, not red.",
    )
    ctx = pef.to_context_summary().lower()
    assert "blue" in ctx


def test_update_pef_without_user_text_still_commits_claims() -> None:
    """Backward compat: tests and corpus replay omit user_text; no admission filter."""
    pef = _emma_red_book_pef()
    ext = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="Emma",
                relation="HAS",
                obj="blue book",
                span=Span.PRESENT,
                negated=False,
                evidence="synthetic",
            )
        ],
        span=Span.PRESENT,
    )
    update_pef(ext, pef)
    ctx = pef.to_context_summary().lower()
    assert "blue book" in ctx


def _emma_blue_claim_ext() -> ExtractionResult:
    return ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="Emma",
                relation="HAS",
                obj="blue book",
                span=Span.PRESENT,
                negated=False,
                evidence="What colour is Emma's book?",
            )
        ],
        span=Span.PRESENT,
    )


def test_update_pef_read_only_when_query_no_extracted_claims() -> None:
    """Pure QUERY (no SVO claims) is read-only for PEF; prior state stays visible."""
    pef = _emma_red_book_pef()
    ext = ExtractionResult(claims=[], span=Span.PRESENT)
    update_pef(
        ext,
        pef,
        user_text="What colour is Emma's book?",
        turn_act=TurnAct.QUERY,
    )
    assert "red" in pef.to_context_summary().lower()
    assert "blue book" not in pef.to_context_summary().lower()


def test_update_pef_commits_when_turn_act_query_but_extractor_emits_claims() -> None:
    """Classified QUERY with non-empty ``claims`` still runs normal admission (mixed turn)."""
    pef = PEFState()
    update_pef(
        _emma_blue_claim_ext(),
        pef,
        user_text="Emma has a blue book. What about you?",
        turn_act=TurnAct.QUERY,
    )
    assert "blue book" in pef.to_context_summary().lower()


def test_update_pef_commits_when_user_text_is_query_class_but_extractor_emits_claims() -> None:
    """``classify_turn_act`` may still yield QUERY; structured claims are still admitted."""
    pef = PEFState()
    update_pef(
        _emma_blue_claim_ext(),
        pef,
        user_text="Emma has a blue book. Is it hers?",
    )
    assert "blue book" in pef.to_context_summary().lower()


def test_update_pef_same_question_text_assert_commits_when_turn_act_assert() -> None:
    """Orchestrator can pass ``TurnAct.ASSERT`` so admission does not follow question-shaped text."""
    pef = PEFState()
    update_pef(
        _emma_blue_claim_ext(),
        pef,
        user_text="What colour is Emma's book?",
        turn_act=TurnAct.ASSERT,
    )
    assert "blue book" in pef.to_context_summary().lower()


def test_uncounted_has_same_category_no_cross_subject_negation() -> None:
    """Multiple subjects may each hold the same item key (e.g. apple)."""
    from aurora_lens.state_native_engine.eval.inventory import (
        _matching_holders_for_item,
    )

    pef = PEFState()

    def commit_turn(turn: int, subject: str, obj: str) -> None:
        pef.current_turn = turn
        update_pef(
            ExtractionResult(
                claims=[
                    ExtractedClaim(
                        subject=subject,
                        relation="HAS",
                        obj=obj,
                        span=Span.PRESENT,
                        negated=False,
                        evidence=f"{subject}: {obj}",
                    )
                ],
                span=Span.PRESENT,
            ),
            pef,
            user_text=None,
        )

    commit_turn(1, "James", "apple")
    commit_turn(2, "John", "apple")
    commit_turn(3, "Mary", "an apple")

    holders = _matching_holders_for_item(pef, "apples")
    assert sorted(h.name for h in holders) == ["James", "John", "Mary"]


def test_uncounted_same_subject_repeat_supersedes_prior_hold_only() -> None:
    """Supersession is scoped to (subject, item key): second James apple replaces first."""
    from aurora_lens.state_native_engine.eval.inventory import (
        _matching_holders_for_item,
        _subject_inventory_objects,
    )

    pef = PEFState()

    def commit_turn(turn: int, subject: str, obj: str) -> None:
        pef.current_turn = turn
        update_pef(
            ExtractionResult(
                claims=[
                    ExtractedClaim(
                        subject=subject,
                        relation="HAS",
                        obj=obj,
                        span=Span.PRESENT,
                        negated=False,
                        evidence=f"{subject}: {obj}",
                    )
                ],
                span=Span.PRESENT,
            ),
            pef,
            user_text=None,
        )

    commit_turn(1, "James", "apple")
    commit_turn(2, "James", "an apple")

    inv = _subject_inventory_objects(pef, pef.find_entity_by_name("James").id)
    assert len([x for x in inv if "apple" in x.lower()]) == 1
    assert _matching_holders_for_item(pef, "apple") != []
