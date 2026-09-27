"""Evidence role classification for corpus display and review preparation.

Retrieval classifies evidence for operator display and chunk selection only.
Lens still disposes admissibility — this module does not decide governance outcomes.
"""

from __future__ import annotations

import re
from typing import Any, Literal

EvidenceRole = Literal["direct_support", "background", "metadata", "low_relevance"]

EVIDENCE_ROLE_LABELS: dict[str, str] = {
    "direct_support": "direct support",
    "background": "background",
    "metadata": "metadata",
    "low_relevance": "low relevance",
}

SECTION_TITLES: dict[str, str] = {
    "direct_support": "Strong evidence",
    "background": "Background / explanatory evidence",
    "metadata": "Metadata / filing evidence",
    "low_relevance": "Low relevance",
}

LOW_RELEVANCE_WARNING = (
    "Some low-relevance material was found but hidden. These blocks may be useful for "
    "navigation or provenance, but should not be treated as claim support."
)

_FILING_QUESTION_RE = re.compile(
    r"\b("
    r"filing\s+date|filing\s+record|priority\s+date|provenance|chain[- ]of[- ]title|"
    r"disclosure\s+timeline|application\s+number|patent\s+number|patent\s+numbers|"
    r"when\s+was\s+it\s+filed|filing\s+history"
    r")\b",
    re.IGNORECASE,
)

_METADATA_SOURCE_RE = re.compile(
    r"(patent\s+numbers?|filing\s+record|application\s+numbers?|provenance\s+log|"
    r"chain[- ]of[- ]title|filing\s+metadata)",
    re.IGNORECASE,
)

_BACKGROUND_SOURCE_RE = re.compile(
    r"(workspace_summary|epistemic\s+legitimacy|test\s+philosophy|governance\s+rules|"
    r"repository\s+overview|running\s+demos?|external\s+resources|general\s+notes|"
    r"file\s+structure|navigation)",
    re.IGNORECASE,
)

_PRIMARY_SOURCE_RE = re.compile(
    r"\b(wipo|pct|patent|application|specification|claims?\b)",
    re.IGNORECASE,
)

_CONCEPTUAL_QUERY_RE = re.compile(
    r"\b("
    r"admissibility|governance|constraint|ledger|epistemic|refusal|prohibition|"
    r"unjustified|persistent|auditable|controlled"
    r")\b",
    re.IGNORECASE,
)

_METADATA_SNIPPET_RE = re.compile(
    r"\b("
    r"application\s+no\.?|patent\s+no\.?|filing\s+date|priority\s+date|"
    r"provisional|publication\s+date|inventor|assignee|docket"
    r")\b",
    re.IGNORECASE,
)

_GOVERNED_REVIEW_ROLES = frozenset({"direct_support", "background"})


def is_filing_provenance_question(question: str) -> bool:
    """True when the question targets dates, numbers, or filing provenance."""
    return bool(_FILING_QUESTION_RE.search(question.strip()))


def classify_evidence_role(
    *,
    score: int | None,
    question: str,
    record_id: str,
    document_name: str,
    source_path: str = "",
    snippet: str = "",
) -> EvidenceRole:
    """Classify one evidence block for display and review preparation."""
    numeric_score = int(score or 0)
    if numeric_score == 0:
        return "low_relevance"

    blob = f"{record_id} {document_name} {source_path}".lower()
    snippet_l = snippet.lower()
    filing_q = is_filing_provenance_question(question)

    if _is_metadata_source(blob, snippet_l):
        return "metadata"

    if _is_background_source(blob, snippet_l):
        return "background"

    if filing_q and _METADATA_SNIPPET_RE.search(snippet_l):
        return "metadata"

    if _PRIMARY_SOURCE_RE.search(blob) and (
        _CONCEPTUAL_QUERY_RE.search(question) or numeric_score >= 3
    ):
        return "direct_support"

    if numeric_score > 0:
        return "direct_support"

    return "low_relevance"


def _is_metadata_source(blob: str, snippet_l: str) -> bool:
    if _METADATA_SOURCE_RE.search(blob):
        return True
    if "patent numbers" in blob:
        return True
    if _METADATA_SNIPPET_RE.search(snippet_l) and not _CONCEPTUAL_QUERY_RE.search(snippet_l):
        return True
    return False


def _is_background_source(blob: str, snippet_l: str) -> bool:
    if _BACKGROUND_SOURCE_RE.search(blob):
        return True
    if "workspace_summary" in blob:
        return True
    if any(
        token in snippet_l
        for token in (
            "epistemic legitimacy",
            "test philosophy",
            "governance rules",
            "repository overview",
            "running demo",
        )
    ):
        return True
    return False


def role_included_in_governed_review(role: EvidenceRole, *, question: str) -> bool:
    """Whether an evidence role may be passed to governed ask/validate chunk selection."""
    if role in _GOVERNED_REVIEW_ROLES:
        return True
    if role == "metadata" and is_filing_provenance_question(question):
        return True
    return False


def group_evidence_sections(
    evidence: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Split flat evidence rows into role-keyed sections (preserving rank order)."""
    sections: dict[str, list[dict[str, Any]]] = {
        "direct_support": [],
        "background": [],
        "metadata": [],
        "low_relevance": [],
    }
    for row in evidence:
        role = str(row.get("evidence_role") or "low_relevance")
        if role not in sections:
            role = "low_relevance"
        sections[role].append(row)
    return sections


def visible_evidence_for_display(
    sections: dict[str, list[dict[str, Any]]],
    *,
    question: str,
    include_low_relevance: bool = False,
) -> list[dict[str, Any]]:
    """Evidence rows shown by default in claim-support searches."""
    visible: list[dict[str, Any]] = []
    visible.extend(sections.get("direct_support") or [])
    visible.extend(sections.get("background") or [])
    if is_filing_provenance_question(question):
        visible.extend(sections.get("metadata") or [])
    if include_low_relevance:
        visible.extend(sections.get("low_relevance") or [])
    return visible


def evidence_for_governed_review(
    evidence: list[dict[str, Any]],
    *,
    question: str,
    include_low_relevance: bool = False,
) -> list[dict[str, Any]]:
    """Evidence rows eligible for governed ask/validate chunk selection."""
    if include_low_relevance:
        return list(evidence)
    return [
        row
        for row in evidence
        if role_included_in_governed_review(
            str(row.get("evidence_role") or "low_relevance"),  # type: ignore[arg-type]
            question=question,
        )
        and int(row.get("score") or 0) > 0
    ]
