"""Sensitive PII surface detection (deterministic regex).

Consolidates patterns in one module. Optional third-party refinements (e.g.
``phonenumbers``) can be added later with parity tests; core behavior matches
the legacy checker paths.
"""

from __future__ import annotations

import re

PII_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PII_PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    r"|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
    r"|\b\+\d{1,3}[-.\s]?\d{2,4}[-.\s]?\d{2,4}[-.\s]?\d{2,4}\b",
)
PII_EMAIL_RE = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
)
PII_CARD_RE = re.compile(
    r"\b(?:\d{4}[\s-]?){3}\d{4}\b"
    r"|\b\d{4}[\s-]?\d{6}[\s-]?\d{5}\b",
)

CRISIS_PHONE_CONTEXT_RE = re.compile(
    r"\b(?:crisis|hotline|lifeline|prevention|text\s+line|helpline|"
    r"988|suicide|self.?harm|emergency\s+service|call\s+for\s+help)\b",
    re.IGNORECASE,
)

ROLE_EMAIL_RE = re.compile(
    r"^(?:support|info|help|sales|admin|contact|noreply|no-reply|"
    r"hello|team|service|billing|abuse|postmaster|webmaster|"
    r"feedback|careers|press|media|security)@",
    re.IGNORECASE,
)


def collect_pii_evidence_snippets(text: str) -> list[str]:
    """Return human-readable evidence strings for each PII class found (checker parity)."""
    if not text or len(text.strip()) < 10:
        return []

    matches: list[str] = []

    m = PII_SSN_RE.search(text)
    if m:
        matches.append(f"SSN-like: {m.group()[:6]}***")

    m = PII_PHONE_RE.search(text)
    if m:
        ctx_start = max(0, m.start() - 100)
        ctx_end = min(len(text), m.end() + 100)
        if not CRISIS_PHONE_CONTEXT_RE.search(text[ctx_start:ctx_end]):
            matches.append(f"phone: {m.group()[:10]}...")

    m = PII_EMAIL_RE.search(text)
    if m:
        if not ROLE_EMAIL_RE.match(m.group()):
            matches.append(f"email: {m.group()[:15]}...")

    m = PII_CARD_RE.search(text)
    if m:
        matches.append(f"card: ****{m.group()[-4:]}")

    return matches
