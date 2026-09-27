"""Structured speech-act classification for upstream model responses.

Lexical patterns here are feature feeders only. The final label is composed from
structured features (user query shape, extraction emptiness, inability signals,
clarification solicitation) — not a single substring match on the response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from aurora_lens.interpret.schema import ExtractionResult, ExtractedClaim

_FIRST_PERSON_SUBJECTS: frozenset[str] = frozenset({"i", "we", "me", "my", "our"})
_META_EPISTEMIC_RELATIONS: frozenset[str] = frozenset({"HAS", "HAVE", "KNOW", "POSSESS"})
_META_EPISTEMIC_OBJECT_RE = re.compile(
    r"\b(?:information|context|detail|data|evidence|facts?|knowledge)\b",
    re.IGNORECASE,
)
_QUERY_STOP: frozenset[str] = frozenset({
    "have", "that", "this", "with", "from", "they", "them", "their",
    "also", "been", "were", "will", "your", "which", "what", "when",
    "here", "some", "just", "very", "more", "like", "than", "then",
    "into", "onto", "over", "under", "about", "after", "before",
})


def _content_words(text: str) -> frozenset[str]:
    words: set[str] = set()
    for w in re.split(r"[\s.,!?;:—\-\"'()\[\]]+", text.lower()):
        if len(w) > 3 and w not in _QUERY_STOP:
            words.add(w)
    return frozenset(words)


def _is_meta_epistemic_self_claim(claim: ExtractedClaim) -> bool:
    """First-person information/context disclaimers are not substantive answers."""
    subj = (claim.subject or "").strip().lower()
    if subj not in _FIRST_PERSON_SUBJECTS:
        return False
    if (claim.relation or "").upper() not in _META_EPISTEMIC_RELATIONS:
        return False
    return bool(_META_EPISTEMIC_OBJECT_RE.search(claim.obj or ""))


def _claims_substantively_answer_user_query(
    claims: list[ExtractedClaim],
    user_input: str | None,
) -> bool:
    substantive = [c for c in claims if not _is_meta_epistemic_self_claim(c)]
    if not substantive:
        return False
    if not user_input:
        return True
    query_words = _content_words(user_input)
    if not query_words:
        return True
    for claim in substantive:
        claim_words = _content_words(f"{claim.subject} {claim.obj}")
        if claim_words & query_words:
            return True
    return False

_EPISTEMIC_DISCLAIMER_RE = re.compile(
    r"\bi\s+can(?:not|'t)\b"
    r"|\bwithout\s+access\s+to\b"
    r"|\bwould\s+require\s+(?:me\s+to\s+)?speculate\b"
    r"|\bnot\s+(?:been\s+)?established\s+in\s+the\s+available\b"
    r"|\bi\s+(?:cannot|can't)\s+(?:confirm|provide|verify|explain|determine)\b"
    r"|\bi'?m\s+unable\s+to\b"
    r"|\bi\s+don'?t\s+have\s+enough\s+information\b",
    re.IGNORECASE,
)

_REFUSAL_FRAME_RE = re.compile(
    r"\b(?:i\s+(?:cannot|can['\u2019]?t|won['\u2019]?t|will\s+not|am\s+unable\s+to|"
    r"don['\u2019]?t\s+(?:provide|help|assist))|"
    r"i['\u2019]?m\s+unable\s+to|"
    r"i['\u2019]?m\s+not\s+(?:going|able)\s+to\s+(?:provide|help|share|give|assist|answer)|"
    r"(?:not|never)\s+definitive|"
    r"(?:requires?|needs?)\s+(?:further\s+)?(?:confirmation|verification|approval)|"
    r"cannot\s+(?:confirm|guarantee|verify)|"
    r"is\s+not\s+(?:yet\s+)?(?:active|cleared|enrolled|approved))\b",
    re.IGNORECASE,
)

_USER_QUERY_OPENER_RE = re.compile(
    r"^\s*(?:what|who|which|when|where|why|how)\b",
    re.IGNORECASE,
)

_CLARIFICATION_SOLICITATION_RE = re.compile(
    r"(?:"
    r"\bmore\s+(?:context|information|detail)\b|"
    r"\badditional\s+(?:context|information|detail)\b|"
    r"\bprovide\s+(?:more\s+)?(?:context|information|detail)\b|"
    r"\bneed\s+(?:more|additional)\s+(?:context|information|detail)\b|"
    r"\bcan\s+you\s+provide\b|"
    r"\bcould\s+you\s+provide\b|"
    r"\bwould\s+you\s+provide\b|"
    r"\bplease\s+provide\b|"
    r"\bclarif(?:y|ication)\b"
    r")",
    re.IGNORECASE,
)

_INABILITY_TO_ANSWER_RE = re.compile(
    r"(?:"
    r"\b(?:cannot|can['\u2019]?t|unable\s+to)\s+(?:answer|determine|provide(?:\s+a\s+response)?)\b|"
    r"\b(?:do|does)\s+not\s+have\s+(?:enough|sufficient)\s+(?:information|context)\b|"
    r"\b(?:don['\u2019]?t|do\s+not)\s+have\s+(?:enough|sufficient)\s+(?:information|context)\b|"
    r"\bnot\s+enough\s+(?:information|context)\b|"
    r"\binsufficient\s+(?:information|context|data)\b"
    r")",
    re.IGNORECASE,
)

_CLARIFYING_QUESTION_RE = re.compile(
    r"(?:\?\s*$|\?\s+[A-Z])",
)


class UpstreamResponseSpeechAct(str, Enum):
    ANSWER = "answer"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    REFUSAL = "refusal"


@dataclass(frozen=True)
class UpstreamResponseFeatures:
    user_query: bool
    extraction_empty: bool
    has_substantive_answer_claims: bool
    has_epistemic_disclaimer: bool
    has_inability_to_answer: bool
    solicits_additional_context: bool
    has_clarifying_question: bool
    is_refusal_frame: bool


def _user_input_is_query(user_input: str | None) -> bool:
    if not user_input:
        return False
    text = user_input.strip()
    if not text:
        return False
    if text.endswith("?"):
        return True
    return bool(_USER_QUERY_OPENER_RE.match(text))


def extract_upstream_response_features(
    *,
    response_text: str,
    user_input: str | None,
    extraction: ExtractionResult,
) -> UpstreamResponseFeatures:
    text = (response_text or "").strip()
    lower = text.lower().replace("\u2019", "'")
    has_disclaimer = bool(_EPISTEMIC_DISCLAIMER_RE.search(text))
    substantive_answer = _claims_substantively_answer_user_query(
        extraction.claims,
        user_input,
    )
    extraction_empty = (
        not extraction.extraction_error
        and not extraction.claims
    )
    inability = has_disclaimer or bool(_INABILITY_TO_ANSWER_RE.search(text))
    solicitation = bool(_CLARIFICATION_SOLICITATION_RE.search(text))
    clarifying_question = "?" in text and (
        solicitation or bool(_CLARIFYING_QUESTION_RE.search(text))
    )
    return UpstreamResponseFeatures(
        user_query=_user_input_is_query(user_input),
        extraction_empty=extraction_empty,
        has_substantive_answer_claims=substantive_answer,
        has_epistemic_disclaimer=has_disclaimer,
        has_inability_to_answer=inability,
        solicits_additional_context=solicitation,
        has_clarifying_question=clarifying_question,
        is_refusal_frame=bool(_REFUSAL_FRAME_RE.search(text)),
    )


def classify_upstream_response_speech_act(
    features: UpstreamResponseFeatures,
) -> UpstreamResponseSpeechAct:
    clarification_signal = (
        features.solicits_additional_context or features.has_clarifying_question
    )
    if features.is_refusal_frame and not clarification_signal:
        return UpstreamResponseSpeechAct.REFUSAL
    if (
        features.user_query
        and not features.has_substantive_answer_claims
        and features.has_inability_to_answer
        and (clarification_signal or features.has_epistemic_disclaimer)
    ):
        return UpstreamResponseSpeechAct.INSUFFICIENT_CONTEXT
    if features.has_substantive_answer_claims:
        return UpstreamResponseSpeechAct.ANSWER
    if not features.user_query or not features.extraction_empty:
        return UpstreamResponseSpeechAct.ANSWER
    if features.has_inability_to_answer and clarification_signal:
        return UpstreamResponseSpeechAct.INSUFFICIENT_CONTEXT
    if features.has_inability_to_answer and features.has_epistemic_disclaimer:
        return UpstreamResponseSpeechAct.INSUFFICIENT_CONTEXT
    return UpstreamResponseSpeechAct.ANSWER
