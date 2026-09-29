"""Unit tests for corpus Q&A validation scoring (no proxy)."""

from __future__ import annotations

from aurora_lens.corpus.qa_validation import (
    ABSTAIN_PHRASE_EXACT,
    CaseSpec,
    answers_abstain,
    score_case,
)


def test_answers_abstain_exact_phrase():
    assert answers_abstain(ABSTAIN_PHRASE_EXACT)
    assert answers_abstain(f"Prefix. {ABSTAIN_PHRASE_EXACT}")


def test_score_in_context_passes_with_markers():
    spec = CaseSpec(
        id="T1",
        question="How does PB work?",
        expect="in_context",
        required_any=["Delta = PB", "PB - 50"],
    )
    result = score_case(
        spec,
        answer="Delta = PB - 50 shifts tiers.",
        governance="PASS",
    )
    assert result.passed


def test_score_in_context_fails_on_abstain():
    spec = CaseSpec(
        id="T2",
        question="How does PB work?",
        expect="in_context",
        required_any=["Delta"],
    )
    result = score_case(
        spec,
        answer=ABSTAIN_PHRASE_EXACT,
        governance="PASS",
    )
    assert not result.passed
    assert any("abstain" in r for r in result.reasons)


def test_score_absent_passes_on_abstain():
    spec = CaseSpec(
        id="A1",
        question="HIPAA?",
        expect="absent",
        forbidden_answer=["HIPAA compliance is required"],
    )
    result = score_case(
        spec,
        answer=ABSTAIN_PHRASE_EXACT,
        governance="PASS",
    )
    assert result.passed


def test_score_absent_fails_on_invention():
    spec = CaseSpec(
        id="A2",
        question="Stock target?",
        expect="absent",
        forbidden_answer=["price target"],
    )
    result = score_case(
        spec,
        answer="The framework recommends a price target of $120.",
        governance="PASS",
    )
    assert not result.passed
