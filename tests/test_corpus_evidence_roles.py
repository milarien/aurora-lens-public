"""Tests for corpus evidence role classification (display / review prep only)."""

from __future__ import annotations

from aurora_lens.corpus.evidence_roles import (
    classify_evidence_role,
    evidence_for_governed_review,
    group_evidence_sections,
    is_filing_provenance_question,
)


def test_admissibility_query_ranks_wipo_as_direct_support():
    role = classify_evidence_role(
        score=5,
        question="Which documents disclose admissibility-controlled governance?",
        record_id="wipo-pct-admissibility",
        document_name="WIPO PCT Application — Governance Framework",
        source_path="/data/patents/wipo-pct.pdf",
        snippet="Admissibility constraints require auditable ledger of persistent epistemic states before generation.",
    )
    assert role == "direct_support"


def test_patent_numbers_classified_as_metadata():
    role = classify_evidence_role(
        score=2,
        question="Which documents disclose admissibility constraints?",
        record_id="patent-numbers",
        document_name="Patent Numbers",
        source_path="/data/patents/Patent Numbers.txt",
        snippet="Application No. PCT/US2024/012345 · Filing date 2024-03-01 · Patent No. US-1234567",
    )
    assert role == "metadata"


def test_workspace_summary_epistemic_legitimacy_is_background():
    role = classify_evidence_role(
        score=3,
        question="Which documents disclose admissibility constraints?",
        record_id="workspace-summary",
        document_name="WORKSPACE_SUMMARY",
        source_path="/docs/WORKSPACE_SUMMARY.md",
        snippet="## Epistemic Legitimacy\nTest Philosophy and Governance Rules for operator evaluation.",
    )
    assert role == "background"


def test_zero_score_is_low_relevance_and_excluded_from_review():
    role = classify_evidence_role(
        score=0,
        question="admissibility constraints",
        record_id="repo-readme",
        document_name="README",
        source_path="/README.md",
        snippet="Running demos and repository overview with external resources.",
    )
    assert role == "low_relevance"
    rows = [
        {
            "chunk_id": "c1",
            "score": 0,
            "evidence_role": role,
        }
    ]
    assert evidence_for_governed_review(rows, question="admissibility constraints") == []


def test_filing_date_query_allows_patent_numbers_in_review():
    question = "What is the filing date and patent number for this application?"
    assert is_filing_provenance_question(question)
    rows = [
        {
            "chunk_id": "meta1",
            "score": 2,
            "evidence_role": "metadata",
        }
    ]
    assert evidence_for_governed_review(rows, question=question)


def test_group_evidence_sections_preserves_roles():
    evidence = [
        {"evidence_role": "direct_support", "score": 4},
        {"evidence_role": "background", "score": 2},
        {"evidence_role": "metadata", "score": 1},
        {"evidence_role": "low_relevance", "score": 0},
    ]
    sections = group_evidence_sections(evidence)
    assert len(sections["direct_support"]) == 1
    assert len(sections["background"]) == 1
    assert len(sections["metadata"]) == 1
    assert len(sections["low_relevance"]) == 1
