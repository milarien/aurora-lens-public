"""Tests for typed user-turn classification (``TurnAct`` / ``classify_turn_act``)."""

from aurora_lens.interpret.revision_gate import user_explicit_revision_act
from aurora_lens.interpret.turn_act import TurnAct, classify_turn_act


def test_classify_revise_delegates_to_revision_gate():
    assert classify_turn_act("Correction: Emma's book is blue, not red.") is TurnAct.REVISE
    assert classify_turn_act("I misspoke — the book was blue.") is TurnAct.REVISE


def test_classify_query_question_mark_or_interrogative():
    assert classify_turn_act("What colour is Emma's book?") is TurnAct.QUERY
    assert classify_turn_act("Is the book red?") is TurnAct.QUERY
    assert classify_turn_act("Have you seen the report") is TurnAct.QUERY


def test_clarify_inquiry_meta_classifies_before_query():
    """Meta-questions about what to clarify must be CLARIFY, not QUERY (lens pending path)."""
    assert classify_turn_act("What should I clarify?") is TurnAct.CLARIFY
    assert classify_turn_act("What do you still need?") is TurnAct.CLARIFY
    assert classify_turn_act("What information do you need?") is TurnAct.CLARIFY


def test_classify_clarify_meta_phrases():
    assert classify_turn_act("Just to be clear, I meant the hardcover.") is TurnAct.CLARIFY
    assert classify_turn_act("What I mean is the green box.") is TurnAct.CLARIFY


def test_classify_assert_default():
    assert classify_turn_act("Emma has a red book.") is TurnAct.ASSERT


def test_revise_takes_precedence_over_query_shape():
    # Correction: prefix is revision even if phrasing looks like a question.
    assert classify_turn_act("Correction: was it blue? I said red earlier by mistake.") is (
        TurnAct.REVISE
    )


def test_revise_matches_revision_gate_for_admission_parity():
    """Live routing uses ``TurnAct.REVISE``; it must agree with the revision phrase detector."""
    samples = [
        "Correction: Emma's book is blue, not red.",
        "I misspoke earlier — it's blue.",
        "I was wrong about the colour.",
        "To correct that, the book is blue.",
        "Actually Emma's book is blue.",
    ]
    for s in samples:
        assert (classify_turn_act(s) == TurnAct.REVISE) == user_explicit_revision_act(s), s
