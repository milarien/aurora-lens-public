"""Explicit birth-date predicates bound to employment selection.

This does not infer age from graduation dates, labels, or other proxies.
"""
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BirthScreening:
    action: str
    target: str
    birth_predicate: str
    cutoff: str
    evidence: str
    negated: bool = False


def birth_screening(text, nlp) -> tuple[BirthScreening, ...]:
    if nlp is None:
        return ()
    doc = nlp(text)
    findings = []
    for token in doc:
        if token.lemma_.lower() != "bear" or token.text.lower() != "born":
            continue
        target = token.head
        if target.lemma_.lower() not in {"applicant", "candidate", "employee", "worker"}:
            continue
        cutoff = next((c for c in token.children if c.dep_ == "prep" and c.lemma_.lower() in {"before", "after", "in"}), None)
        if cutoff is None:
            continue
        value = next((c for c in cutoff.children if c.like_num), None)
        if value is None:
            continue
        action = next((a for a in target.ancestors if a.pos_ == "VERB"), None)
        if action is None or action.lemma_.lower() not in {"remove", "exclude", "reject", "screen", "filter", "shortlist", "hire", "recruit", "select"}:
            continue
        negated = any(c.dep_ == "neg" for c in action.children) or any(
            c.dep_ == "det" and c.lemma_.lower() == "no" for c in target.children)
        if negated and action.lemma_.lower() in {"remove", "exclude", "reject", "screen", "filter"}:
            continue
        finding = BirthScreening(action.text, target.text, token.text, cutoff.text + " " + value.text, token.sent.text, negated)
        findings.append(finding)
    logger.debug("birth_screening_bindings=%r", findings)
    return tuple(findings)
