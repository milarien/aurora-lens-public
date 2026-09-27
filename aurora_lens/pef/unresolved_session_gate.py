"""Session-level gate: open unresolved-referent registry governs dependent consequence.

While any registry entry is open, consequence-bearing turns that depend on the
unresolved attribution must not reach the adapter — even when the user omits
the original pronoun.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aurora_lens.govern.governed_copy import is_attribution_referent_query
from aurora_lens.interpret.turn_act import TurnAct, _looks_like_clarification_inquiry_meta
from aurora_lens.pef.unresolved_referents import (
    STATUS_OPEN,
    STATUS_RESOLVED,
    UnresolvedReferentEntry,
    open_entries,
    open_registry_tokens_in_text,
)

if TYPE_CHECKING:
    from aurora_lens.pef.state import PEFState

REASON_CODE = "UNRESOLVED_REFERENT"

_CONSEQUENCE_STEMS: frozenset[str] = frozenset({
    "admissible",
    "admissibility",
    "suspend",
    "suspended",
    "suspension",
    "compliant",
    "compliance",
    "responsible",
    "responsibility",
    "regulator",
    "regulatory",
    "determine",
    "decide",
    "decision",
    "action",
    "recommend",
    "recommendation",
    "liable",
    "liability",
    "enforce",
    "enforcement",
    "penalty",
    "sanction",
    "violation",
    "remove",
    "removed",
    "terminate",
    "terminated",
    "termination",
    "must",
    "shall",
    "required",
    "requirement",
    "obligation",
})

_ATTRIBUTION_PHRASES: tuple[tuple[str, ...], ...] = (
    ("which", "party"),
    ("whose", "certification"),
    ("responsible", "party"),
    ("determine", "the", "responsible"),
    ("determine", "the", "responsible", "party"),
    ("who", "must", "be", "suspended"),
    ("who", "should", "be", "suspended"),
    ("which", "party", "s", "certification"),
)

_DECISION_PHRASES: tuple[tuple[str, ...], ...] = (
    ("is", "suspension", "admissible"),
    ("should", "the", "operator", "be", "suspended"),
    ("should", "the", "contractor", "be", "suspended"),
    ("can", "we", "remove", "the", "contractor"),
    ("do", "we", "terminate", "the", "contractor"),
    ("remove", "the", "contractor"),
    ("terminate", "the", "contractor"),
    ("is", "the", "party", "compliant"),
    ("what", "action", "should"),
    ("what", "action", "should", "the", "regulator"),
)

_META_SAFE_PHRASES: tuple[str, ...] = (
    "what information do you need",
    "what info do you need",
    "what ambiguity is currently open",
    "what ambiguity is open",
    "show current unresolved referents",
    "show unresolved referents",
    "explain why the decision is blocked",
    "why is the decision blocked",
    "why is this blocked",
    "what is currently unresolved",
    "what referents are unresolved",
)

_SESSION_RESET_PHRASES: tuple[str, ...] = (
    "ignore the previous scenario and reset",
    "ignore the previous scenario",
    "ignore the previous",
    "start over",
    "reset the scenario",
    "reset the session",
    "forget the previous scenario",
    "new topic",
    "change topic",
)

_UNRELATED_SAFE_PHRASES: tuple[str, ...] = (
    "tell me a joke",
    "joke about penguins",
)


@dataclass
class OpenRegistryContext:
    entries: list[UnresolvedReferentEntry]
    candidates: list[str]
    theme_tokens: frozenset[str]
    blocked_propositions: list[str]
    introduced_utterances: list[str]


@dataclass
class UnresolvedSessionGateResult:
    blocks: bool = False
    meta_response: bool = False
    allow_through: bool = False
    reason_code: str = REASON_CODE
    reason_detail: str = ""
    act_kind: str = ""
    matched_entries: list[UnresolvedReferentEntry] = field(default_factory=list)
    ambiguous_tokens: list[str] = field(default_factory=list)
    candidate_entities: list[str] = field(default_factory=list)
    blocked_proposition: str | None = None


def _scan_word_tokens(text: str) -> tuple[str, ...]:
    out: list[str] = []
    cur: list[str] = []
    for ch in (text or "").lower():
        if ch.isalpha() or ch == "'":
            cur.append(ch)
            continue
        if cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return tuple(out)


def _contains_phrase(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    if not phrase or len(tokens) < len(phrase):
        return False
    n = len(phrase)
    for i in range(0, len(tokens) - n + 1):
        if tokens[i : i + n] == phrase:
            return True
    return False


def _theme_tokens_from_entry(entry: UnresolvedReferentEntry) -> set[str]:
    themes: set[str] = set()
    for surface in (
        entry.blocked_proposition,
        entry.introduced_utterance,
        entry.span_surface,
        entry.head,
    ):
        if not surface:
            continue
        for tok in _scan_word_tokens(str(surface)):
            if len(tok) >= 4 and tok not in {"before", "after", "both", "parties", "hold", "holds"}:
                themes.add(tok)
            if tok in {"certification", "certifications", "expired", "operator", "contractor", "party", "parties"}:
                themes.add(tok)
    return themes


def build_open_registry_context(pef: PEFState) -> OpenRegistryContext | None:
    entries = open_entries(pef)
    if not entries:
        return None
    candidates: list[str] = []
    seen_c: set[str] = set()
    themes: set[str] = set()
    blocked: list[str] = []
    intros: list[str] = []
    for entry in entries:
        for name in entry.candidate_entities:
            key = name.strip()
            if key and key.lower() not in seen_c:
                seen_c.add(key.lower())
                candidates.append(key)
        themes |= _theme_tokens_from_entry(entry)
        if entry.blocked_proposition:
            blocked.append(str(entry.blocked_proposition))
        if entry.introduced_utterance:
            intros.append(str(entry.introduced_utterance))
    return OpenRegistryContext(
        entries=entries,
        candidates=candidates,
        theme_tokens=frozenset(themes),
        blocked_propositions=blocked,
        introduced_utterances=intros,
    )


def dismiss_open_unresolved_referents(pef: PEFState, *, reason: str = "session_reset") -> int:
    """Clear open registry entries on explicit session reset (not on PASS)."""
    count = 0
    for entry in pef.unresolved_referent_registry:
        if entry.status != STATUS_OPEN:
            continue
        entry.status = STATUS_RESOLVED
        entry.resolved_entity = reason
        count += 1
    return count


def _is_disambiguation_request(user_text: str) -> bool:
    """True when the user asks to resolve an open referent rather than act on one."""
    if is_attribution_referent_query(user_text):
        return True
    tokens = _scan_word_tokens(user_text)
    if user_text.strip().endswith("?") and "whose" in tokens:
        return True
    return False


def _normalized_phrase(text: str) -> str:
    return " ".join(_scan_word_tokens(text))


def is_session_reset_request(user_text: str) -> bool:
    norm = _normalized_phrase(user_text)
    return any(p in norm for p in _SESSION_RESET_PHRASES)


def is_meta_safe_turn(user_text: str) -> bool:
    if _looks_like_clarification_inquiry_meta(user_text):
        return True
    norm = _normalized_phrase(user_text)
    return any(p in norm for p in _META_SAFE_PHRASES)


def is_unrelated_safe_turn(user_text: str) -> bool:
    norm = _normalized_phrase(user_text)
    return any(p in norm for p in _UNRELATED_SAFE_PHRASES)


def _stem_hit(tokens: tuple[str, ...], stems: frozenset[str]) -> bool:
    return any(any(tok.startswith(stem) or stem.startswith(tok) for stem in stems) for tok in tokens)


def _is_consequence_bearing_act(tokens: tuple[str, ...], text_lower: str) -> bool:
    if _stem_hit(tokens, _CONSEQUENCE_STEMS):
        return True
    for phrase in _ATTRIBUTION_PHRASES + _DECISION_PHRASES:
        if _contains_phrase(tokens, phrase):
            return True
    if text_lower.strip().endswith("?"):
        if any(w in tokens for w in ("who", "which", "whose", "should", "determine")):
            return True
    return False


def _act_kind(tokens: tuple[str, ...]) -> str:
    for phrase in _ATTRIBUTION_PHRASES:
        if _contains_phrase(tokens, phrase):
            return "attribution_seeking"
    for phrase in _DECISION_PHRASES:
        if _contains_phrase(tokens, phrase):
            return "decision_seeking"
    if _stem_hit(tokens, frozenset({"suspend", "suspended", "suspension"})):
        return "mutation_seeking"
    if _stem_hit(
        tokens,
        frozenset({"remove", "removed", "terminate", "terminated", "termination"}),
    ):
        return "mutation_seeking"
    if _stem_hit(tokens, frozenset({"recommend", "action", "regulator"})):
        return "recommendation_seeking"
    return "consequence_seeking"


def _candidate_in_text(text_lower: str, candidates: list[str]) -> bool:
    for name in candidates:
        label = name.strip().lower()
        if not label:
            continue
        if re.search(rf"\b{re.escape(label)}\b", text_lower):
            return True
    return False


def _theme_overlap(text_lower: str, tokens: tuple[str, ...], themes: frozenset[str]) -> bool:
    if not themes:
        return False
    for theme in themes:
        tl = theme.lower()
        if len(tl) >= 4 and re.search(rf"\b{re.escape(tl)}\b", text_lower):
            return True
    partyish = {"party", "parties", "certification", "certifications", "expired", "expiry", "suspension", "suspended"}
    if partyish & set(tokens) and themes & partyish:
        return True
    return False


def act_depends_on_open_registry(
    user_text: str,
    context: OpenRegistryContext,
) -> tuple[bool, str]:
    """Return (depends, act_kind) when the turn plausibly uses unresolved attribution."""
    text_lower = (user_text or "").lower()
    tokens = _scan_word_tokens(user_text)
    if not _is_consequence_bearing_act(tokens, text_lower):
        return False, ""

    kind = _act_kind(tokens)

    if _theme_overlap(text_lower, tokens, context.theme_tokens):
        return True, kind

    if _candidate_in_text(text_lower, context.candidates) and _stem_hit(tokens, _CONSEQUENCE_STEMS):
        return True, kind

    if "party" in tokens and context.theme_tokens & {"certification", "certifications", "expired", "expiry"}:
        return True, kind

    if "expired" in tokens or "expiry" in tokens:
        if context.theme_tokens & {"certification", "certifications", "expired", "expiry"}:
            return True, kind

    if "suspension" in tokens or "suspended" in tokens or "suspend" in tokens:
        if context.theme_tokens & {"certification", "certifications", "expired", "expiry", "party", "parties"}:
            return True, kind

    if _stem_hit(tokens, frozenset({"remove", "removed", "terminate", "terminated", "termination"})):
        if _candidate_in_text(text_lower, context.candidates):
            return True, kind

    return False, kind


def evaluate_unresolved_session_gate(
    pef: PEFState,
    user_text: str,
    *,
    turn_act: TurnAct | None = None,
) -> UnresolvedSessionGateResult | None:
    """Evaluate session-level unresolved registry governance for ``user_text``."""
    context = build_open_registry_context(pef)
    if context is None:
        return None

    if is_session_reset_request(user_text):
        return UnresolvedSessionGateResult(allow_through=True)

    if is_unrelated_safe_turn(user_text):
        return UnresolvedSessionGateResult(allow_through=True)

    if is_meta_safe_turn(user_text):
        tokens = [e.token for e in context.entries if e.token.strip()]
        primary_bp = context.blocked_propositions[0] if context.blocked_propositions else None
        return UnresolvedSessionGateResult(
            meta_response=True,
            reason_detail="meta_inquiry_while_registry_open",
            matched_entries=list(context.entries),
            ambiguous_tokens=tokens,
            candidate_entities=list(context.candidates),
            blocked_proposition=primary_bp,
        )

    # Direct binding attempt: single candidate name only — defer to binding resume path.
    norm = _normalized_phrase(user_text)
    if len(norm.split()) <= 3:
        for cand in context.candidates:
            if norm == cand.strip().lower():
                return None

    depends, kind = act_depends_on_open_registry(user_text, context)
    if not depends:
        return UnresolvedSessionGateResult(allow_through=True)

    if _is_disambiguation_request(user_text):
        return UnresolvedSessionGateResult(allow_through=True)

    tokens = [e.token for e in context.entries if e.token.strip()]
    primary_bp = context.blocked_propositions[0] if context.blocked_propositions else None
    detail = (
        f"Turn depends on unresolved attribution ({kind}) while registry is open; "
        "consequence is blocked until explicit resolution."
    )
    return UnresolvedSessionGateResult(
        blocks=True,
        reason_detail=detail,
        act_kind=kind,
        matched_entries=list(context.entries),
        ambiguous_tokens=tokens,
        candidate_entities=list(context.candidates),
        blocked_proposition=primary_bp,
    )
