"""Unit tests for ``revision_gate`` hostile-contradiction detection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aurora_lens.interpret.revision_gate import (
    extraction_conflicts_grounded_pef,
    user_explicit_revision_act,
    user_text_conflicts_grounded_pef,
)
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
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


def _emma_red_box_pef() -> PEFState:
    pef = PEFState()
    emma, _ = pef.get_or_create_entity("Emma")
    pef.add_relationship(
        Relationship(
            subject_id=emma.id,
            relation="HAS",
            object_entity_id=None,
            object_literal="red box",
            span=Span.PRESENT,
            source_turn=1,
            evidence="Emma has a red box.",
        )
    )
    return pef


def test_user_explicit_revision_act_detects_correction_prefix() -> None:
    assert user_explicit_revision_act("Correction: Emma's book is blue, not red.")
    assert user_explicit_revision_act("I misspoke earlier — it's blue.")
    assert user_explicit_revision_act("I was wrong about the colour.")
    assert user_explicit_revision_act("To correct that, the book is blue.")
    assert not user_explicit_revision_act("Actually Emma's book is blue.")


def test_possessive_book_is_vs_emma_has_red_book() -> None:
    """SpaCy-shaped subject ``Emma 's book`` normalizes to ``Emma's book``."""
    pef = _emma_red_book_pef()
    ext = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="Emma 's book",
                relation="IS",
                obj="blue",
                span=Span.PRESENT,
                negated=False,
                evidence="Actually Emma's book is blue.",
            )
        ],
        span=Span.PRESENT,
    )
    ok, detail = extraction_conflicts_grounded_pef(pef, ext)
    assert ok is True
    assert "HAS" in detail or "red book" in detail


def test_extraction_conflict_has_red_vs_blue_book() -> None:
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
    ok, detail = extraction_conflicts_grounded_pef(pef, ext)
    assert ok is True
    assert "red book" in detail and "blue book" in detail


def test_extraction_conflict_generalizes_owned_object_head() -> None:
    pef = _emma_red_box_pef()
    ext = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="Emma",
                relation="HAS",
                obj="blue box",
                span=Span.PRESENT,
                negated=False,
                evidence="Emma has a blue box.",
            )
        ],
        span=Span.PRESENT,
    )
    ok, detail = extraction_conflicts_grounded_pef(pef, ext)
    assert ok is True
    assert "red box" in detail and "blue box" in detail


def test_user_text_conflict_generalizes_possessive_owned_object() -> None:
    pef = _emma_red_box_pef()
    ok, detail = user_text_conflicts_grounded_pef(pef, "Actually Emma's box is blue.")
    assert ok is True
    assert "emma's box" in detail.lower()
    assert "red box" in detail.lower()


def test_extraction_conflict_ignores_user_correction_phrase() -> None:
    """``extraction_conflicts_grounded_pef`` only compares claims; Lens skips gate when
    :func:`classify_turn_act` yields ``TurnAct.REVISE`` (same signals as
    ``user_explicit_revision_act``)."""
    pef = _emma_red_book_pef()
    ext = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="Emma",
                relation="HAS",
                obj="blue book",
                span=Span.PRESENT,
                negated=False,
                evidence="Correction: Emma's book is blue.",
            )
        ],
        span=Span.PRESENT,
    )
    ok, _ = extraction_conflicts_grounded_pef(pef, ext)
    assert ok is True


def test_no_conflict_for_unknown_subject() -> None:
    pef = _emma_red_book_pef()
    ext = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="Zora",
                relation="HAS",
                obj="blue book",
                span=Span.PRESENT,
                negated=False,
                evidence="Zora has a blue book.",
            )
        ],
        span=Span.PRESENT,
    )
    assert extraction_conflicts_grounded_pef(pef, ext)[0] is False


def test_user_text_conflicts_4a_live_string() -> None:
    """Raw text scan catches hostile colour flip when extraction yields no claims."""
    pef = _emma_red_book_pef()
    hostile = "Actually Emma's book is blue. What colour is Emma's book?"
    ok, detail = user_text_conflicts_grounded_pef(pef, hostile)
    assert ok is True
    assert "blue" in detail.lower() and "red book" in detail.lower()


def test_user_text_conflicts_curly_apostrophe() -> None:
    pef = _emma_red_book_pef()
    ok, _ = user_text_conflicts_grounded_pef(pef, "Actually Emma\u2019s book is blue.")
    assert ok is True


def test_user_text_no_conflict_question_only() -> None:
    pef = _emma_red_book_pef()
    assert user_text_conflicts_grounded_pef(pef, "What colour is Emma's book?")[0] is False


@pytest.mark.asyncio
async def test_lens_blocks_hostile_overwrite_without_llm() -> None:
    """Hostile contradiction returns pre-LLM clarification; PEF keeps red book."""
    from test_lens import MockAdapter  # noqa: E402

    from aurora_lens.config import LensConfig
    from aurora_lens.lens import Lens

    adapter = MockAdapter(responses=["Acknowledged."])
    lens = Lens(LensConfig(adapter=adapter))
    await lens.process("Emma has a red book.")
    assert "red" in lens.pef.to_context_summary().lower()
    calls_after_setup = adapter._call_count

    r2 = await lens.process("Actually Emma's book is blue. What colour is Emma's book?")
    assert "conflicts" in r2.response.lower() or "correction" in r2.response.lower()
    assert "red" in lens.pef.to_context_summary().lower()
    assert adapter._call_count == calls_after_setup
    assert any(f.flag_type.name == "CONTRADICTED_FACT" for f in r2.flags)
