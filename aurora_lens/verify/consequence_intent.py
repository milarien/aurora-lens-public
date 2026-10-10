"""Strict-policy admissibility: classify user intent before empty-flag PASS.

Under strict policy, verification producing no flags is insufficient for PASS.
Ordinary harmless conversation must be explicitly classified (e.g.
``LOW_RISK_CONVERSATIONAL``). Operational or consequential intent without a
matching policy lane yields ``UNCLASSIFIED_CONSEQUENCE_INTENT`` or
``POLICY_PROFILE_REQUIRED`` (when ``request_domain`` is general).
"""

from __future__ import annotations

from dataclasses import dataclass

from aurora_lens.verify.blocked_request_normalize import normalise_request_text
from aurora_lens.verify.flags import Flag, FlagType

LOW_RISK_CONVERSATIONAL = "LOW_RISK_CONVERSATIONAL"
UNCLASSIFIED_CONSEQUENCE_INTENT = "UNCLASSIFIED_CONSEQUENCE_INTENT"
POLICY_PROFILE_REQUIRED = "POLICY_PROFILE_REQUIRED"

_GENERAL_REQUEST_DOMAINS = frozenset({"", "general", "default", "none"})

_LOW_RISK_GREETING_TOKENS = frozenset({
    "hello", "hi", "hey", "thanks", "thank", "bye", "goodbye", "morning",
    "afternoon", "evening", "howdy", "greetings",
})

_OPERATIONAL_PHRASES = (
    " how do i ",
    " how can i ",
    " how could i ",
    " how would i ",
    " how should i ",
    " walk me through ",
    " step by step ",
    " give me a plan ",
    " give me steps ",
    " outline how ",
    " outline the steps ",
    " help me draft ",
    " help me prepare ",
    " help me write ",
    " help me build ",
    " help me create ",
    " ways to ",
    " plan to ",
    " without involving ",
    " without hr ",
    " without getting caught ",
)

_DECISION_PHRASES = (
    " should i ",
    " ought i ",
    " best approach ",
    " best way ",
    " recommend ",
    " what steps ",
    " what approach ",
    " advise me ",
    " guide me ",
    " yes or no ",
)

_EDUCATIONAL_PHRASES = (
    " at a high level ",
    " high level ",
    " overview of ",
    " explain what ",
    " explain how ",
    " what is ",
    " what are ",
    " who was ",
    " who is ",
    " history of ",
    " definition of ",
)

_CATEGORY_MARKERS: tuple[tuple[str, frozenset[str]], ...] = (
    ("legal", frozenset({
        "lawsuit", "litigation", "settlement", "contract", "court", "attorney",
        "legal", "tenant", "subpoena", "binding", "nda", "severance", "dispute",
    })),
    ("medical", frozenset({
        "diagnosis", "treatment", "dosage", "symptom", "prescribe", "clinical",
        "patient", "medication", "therapy", "prognosis",
    })),
    ("financial", frozenset({
        "invest", "portfolio", "tax", "offshore", "restructure", "renegotiate", "hedge",
        "trade", "financial", "audit", "compliance", "severance", "employer", "holdings",
        "fund", "funds", "retirement", "savings", "allocation", "401k",
    })),
    ("cyber", frozenset({
        "hack", "exploit", "malware", "breach", "backdoor", "credential", "ransomware",
        "phishing", "infiltrat",
    })),
    ("authority", frozenset({
        "regulator", "regulatory", "enforcement", "jurisdiction", "fda", "sec",
        "oversight", "compliance", "licensing",
    })),
    ("identity", frozenset({
        "ssn", "passport", "impersonat", "identity", "social", "security", "number",
    })),
    ("safety", frozenset({
        "emergency", "evacuation", "unsafe", "hazard", "workplace", "injury",
    })),
    ("harmful", frozenset({
        "coerce", "intimidate", "pressure", "blackmail", "retaliate", "undetected",
    })),
)


@dataclass(frozen=True)
class ConsequenceIntentClassification:
    """Result of strict-policy request-side consequence intent scan."""

    basis: str
    reason: str
    intent_categories: tuple[str, ...] = ()


def _tokens(norm: str) -> frozenset[str]:
    return frozenset(norm.split())


def _has_operational_framing(norm: str, tokens: frozenset[str]) -> bool:
    w = f" {norm} "
    if any(p in w for p in _OPERATIONAL_PHRASES):
        return True
    if "help" in tokens and "me" in tokens:
        if tokens.isdisjoint({"understand", "learn", "know"}):
            return True
    if "prepare" in tokens or "draft" in tokens or "outline" in tokens:
        return True
    return False


def _has_decision_seeking_framing(norm: str, tokens: frozenset[str]) -> bool:
    w = f" {norm} "
    return any(p in w for p in _DECISION_PHRASES)


def _is_educational_framing(norm: str, tokens: frozenset[str]) -> bool:
    w = f" {norm} "
    return any(p in w for p in _EDUCATIONAL_PHRASES)


def _is_low_risk_greeting(norm: str, tokens: frozenset[str]) -> bool:
    if len(tokens) <= 4 and not tokens.isdisjoint(_LOW_RISK_GREETING_TOKENS):
        return True
    if norm in {"hello", "hi", "hey", "thanks", "thank you", "goodbye"}:
        return True
    return False


def _detect_categories(norm: str, tokens: frozenset[str]) -> tuple[str, ...]:
    w = f" {norm} "
    found: list[str] = []
    for name, markers in _CATEGORY_MARKERS:
        if not markers.isdisjoint(tokens):
            found.append(name)
            continue
        if any(f" {marker} " in w for marker in markers):
            found.append(name)
    return tuple(found)


def _reason_code_for_domain(request_domain: str | None) -> str:
    dom = (request_domain or "").strip().lower()
    if dom in _GENERAL_REQUEST_DOMAINS:
        return POLICY_PROFILE_REQUIRED
    return UNCLASSIFIED_CONSEQUENCE_INTENT


def classify_consequence_intent(
    user_input: str,
    *,
    request_domain: str | None = None,
) -> ConsequenceIntentClassification:
    """Classify whether a flag-free request may earn strict-policy PASS."""
    norm = normalise_request_text(user_input)
    if not norm:
        return ConsequenceIntentClassification(
            basis=LOW_RISK_CONVERSATIONAL,
            reason="Empty request",
        )

    tokens = _tokens(norm)

    if _is_low_risk_greeting(norm, tokens):
        return ConsequenceIntentClassification(
            basis=LOW_RISK_CONVERSATIONAL,
            reason="Greeting or acknowledgment",
        )

    categories = _detect_categories(norm, tokens)
    operational = _has_operational_framing(norm, tokens)
    decision_seeking = _has_decision_seeking_framing(norm, tokens)

    if categories and (operational or decision_seeking):
        reason_code = _reason_code_for_domain(request_domain)
        return ConsequenceIntentClassification(
            basis=reason_code,
            reason=(
                f"Consequential intent ({', '.join(categories)}) "
                "without matching verification policy lane"
            ),
            intent_categories=categories,
        )

    if categories and _is_educational_framing(norm, tokens) and not operational:
        return ConsequenceIntentClassification(
            basis=LOW_RISK_CONVERSATIONAL,
            reason="Educational framing without operational consequence cues",
            intent_categories=categories,
        )

    if not categories and not operational and not decision_seeking:
        return ConsequenceIntentClassification(
            basis=LOW_RISK_CONVERSATIONAL,
            reason="No operational or consequential intent cues",
        )

    if operational or decision_seeking:
        reason_code = _reason_code_for_domain(request_domain)
        return ConsequenceIntentClassification(
            basis=reason_code,
            reason="Operational or decision-seeking framing without policy lane match",
            intent_categories=categories,
        )

    return ConsequenceIntentClassification(
        basis=LOW_RISK_CONVERSATIONAL,
        reason="Non-operational request",
        intent_categories=categories,
    )


def consequence_intent_flag(classification: ConsequenceIntentClassification) -> Flag:
    """Build verification flag for unclassified consequential intent."""
    return Flag(
        flag_type=FlagType.UNCLASSIFIED_CONSEQUENCE_INTENT,
        entity_name="request",
        claim=classification.reason,
        evidence=(
            ", ".join(classification.intent_categories)
            if classification.intent_categories
            else classification.basis
        ),
        severity="warning",
        rule_id=classification.basis,
    )
