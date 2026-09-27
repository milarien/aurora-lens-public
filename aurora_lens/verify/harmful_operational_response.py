"""Post-generation detection of harmful operational guidance in model output."""

from __future__ import annotations

from aurora_lens.verify.blocked_request_normalize import normalise_request_text
from aurora_lens.verify.blocked_request_surface_harmful_action import (
    surface_harmful_action_protective,
)

# Stable audit reason codes (response-side; mirror request policy naming).
RULE_COVERT_HARM_GUIDANCE = "blocked.response.covert_harm_guidance"
RULE_CYBER_ABUSE_GUIDANCE = "blocked.response.cyber_abuse_guidance"
RULE_HARMFUL_ACTION_GUIDANCE = "blocked.response.harmful_action_guidance"

# Soft cyber lexicon: educational/protocol discussion may use these without abuse how-to.
_SOFT_CYBER_TOKENS: frozenset[str] = frozenset(
    {
        "credential",
        "credentials",
        "vulnerability",
        "vulnerabilities",
    }
)
# Auth/session objects that complete a cyber hit only with strong operational abuse.
_CYBER_TARGET_TOKENS: frozenset[str] = _SOFT_CYBER_TOKENS | frozenset(
    {
        "password",
        "passwords",
        "token",
        "tokens",
        "session",
        "authentication",
        "auth",
        "login",
        "traffic",
    }
)
_HARD_CYBER_TOKENS: frozenset[str] = frozenset(
    {
        "malware",
        "ransomware",
        "backdoor",
        "backdoors",
        "exfiltrate",
        "exfiltration",
    }
)
# Union retained for callers/tests that inspect overall cyber lexicon.
_CYBER_TOKENS: frozenset[str] = _SOFT_CYBER_TOKENS | _HARD_CYBER_TOKENS

_COVERT_TOKENS: frozenset[str] = frozenset(
    {
        "revenge",
        "retaliation",
        "sabotage",
        "covert",
        "covertly",
        "secretly",
    }
)
_HARMFUL_TOKENS: frozenset[str] = frozenset(
    {
        "blackmail",
        "extort",
        "extortion",
        "evidence",
    }
)
# Weak connectives/advisory words — not sufficient operational-instruction signals.
_WEAK_INSTRUCTIONAL_TOKENS: frozenset[str] = frozenset(
    {
        "could",
        "then",
        "third",
        "should",
    }
)
_INSTRUCTIONAL_TOKENS: frozenset[str] = frozenset(
    {
        "step",
        "steps",
        "first",
        "second",
        "third",
        "then",
        "next",
        "approach",
        "install",
        "create",
        "obtain",
        "leverage",
        "exploit",
        "could",
        "should",
    }
)
_STRONG_INSTRUCTIONAL_TOKENS: frozenset[str] = (
    _INSTRUCTIONAL_TOKENS - _WEAK_INSTRUCTIONAL_TOKENS
)

# Soft phrase: admissible in defensive/educational prose; not an automatic cyber hit.
_SOFT_CYBER_PHRASES: tuple[str, ...] = (
    "credential theft",
)

_HARM_PHRASES: tuple[str, ...] = (
    "plant evidence",
    "steal sensitive",
    "steal company data",
    "steal data",
    "credential theft",
    "operational disruption",
    "reputational sabotage",
    "without getting caught",
    "without being traced",
    "back door",
    "freeze company computers",
    "freeze computers",
    "exploit vulnerability",
    "exploit vulnerabilities",
    "blackmail",
    "extortion",
)

_HARD_CYBER_PHRASES: tuple[str, ...] = (
    "back door",
    "steal data",
    "steal sensitive",
    "exploit vulnerability",
    "exploit vulnerabilities",
)

_STRONG_ABUSE_PHRASES: tuple[str, ...] = (
    " step by step ",
    " without getting caught ",
    " without being traced ",
    " create a backdoor ",
    " install malware ",
    " plant evidence ",
    " steal company data ",
    " freeze company computers ",
    " steal credentials ",
    " steal credential ",
    " intercept credentials ",
    " intercept login ",
    " intercept the traffic ",
    " capture passwords ",
    " capture password ",
    " capture session ",
    " steal session ",
    " bypass authentication ",
    " bypass auth ",
    " extract the credentials ",
    " extract credentials ",
    " conduct credential theft ",
    " how to steal ",
    " how to intercept ",
    " how to exploit ",
    " how to bypass ",
    " steps to intercept ",
    " steps to steal ",
    " show how to exploit ",
    " give me steps ",
    " explain how to steal ",
    " weaponise ",
    " weaponize ",
)

_ABUSE_ACTION_TOKENS: frozenset[str] = frozenset(
    {
        "steal",
        "intercept",
        "exploit",
        "bypass",
        "capture",
        "exfiltrate",
        "extract",
        "weaponise",
        "weaponize",
    }
)


def _response_has_strong_operational_abuse_instructions(
    norm: str,
    tokens: frozenset[str],
) -> bool:
    """True when the model output gives actionable wrongdoing steps, not risk overview."""
    w = f" {norm} "
    if any(phrase in w for phrase in _STRONG_ABUSE_PHRASES):
        return True
    if "create" in tokens and "backdoor" in tokens:
        return True
    if "install" in tokens and "malware" in tokens:
        return True
    # Attacker-directed execution: abuse verb tied to soft cyber / auth / session objects.
    if not tokens.isdisjoint(_ABUSE_ACTION_TOKENS):
        if not tokens.isdisjoint(_SOFT_CYBER_TOKENS):
            return True
        if not {"password", "passwords", "token", "tokens", "session", "authentication", "auth"}.isdisjoint(
            tokens
        ):
            return True
    # Concrete step sequence tied to abuse verbs.
    if "first" in tokens and ("then" in tokens or "next" in tokens):
        if not {
            "backdoor",
            "malware",
            "blackmail",
            "steal",
            "exfiltrate",
            "intercept",
            "exploit",
            "bypass",
            "capture",
            "extract",
        }.isdisjoint(tokens):
            return True
    # Imperative / procedural "steps" with abuse action.
    if ("step" in tokens or "steps" in tokens) and not tokens.isdisjoint(_ABUSE_ACTION_TOKENS):
        return True
    return False


def analyze_harmful_operational_guidance(text: str) -> dict:
    """Return score components, operational signals, and final rule_id (or None)."""
    empty = {
        "rule_id": None,
        "phrase_hits": 0,
        "hard_phrase_hits": 0,
        "soft_cyber_phrase_hits": 0,
        "cyber_hits": 0,
        "hard_cyber_hits": 0,
        "soft_cyber_hits": 0,
        "covert_hits": 0,
        "harmful_hits": 0,
        "instructional_hits": 0,
        "strong_instructional_hits": 0,
        "weak_instructional_hits": 0,
        "harm_score": 0,
        "strong_operational_abuse": False,
        "matched_operational_signals": [],
        "soft_cyber_tokens": [],
        "hard_cyber_tokens": [],
    }
    if not text or len(text.strip()) < 40:
        return empty

    norm = normalise_request_text(text)
    tokens = frozenset(norm.split())
    w = f" {norm} "

    soft_cyber_tokens = sorted(tokens & _SOFT_CYBER_TOKENS)
    hard_cyber_tokens = sorted(tokens & _HARD_CYBER_TOKENS)
    soft_cyber_phrase_hits = sum(1 for p in _SOFT_CYBER_PHRASES if p in w)
    hard_phrase_hits = sum(1 for p in _HARD_CYBER_PHRASES if p in w)
    # Full phrase table (includes soft phrases) for harm_score parity with prior metrics.
    phrase_hits = sum(1 for p in _HARM_PHRASES if p in w)

    soft_cyber_hits = len(tokens & _SOFT_CYBER_TOKENS) + soft_cyber_phrase_hits
    hard_cyber_hits = len(tokens & _HARD_CYBER_TOKENS) + (
        1 if "back door" in w or "steal data" in w or "steal sensitive" in w else 0
    )
    cyber_hits = soft_cyber_hits + hard_cyber_hits

    covert_hits = len(tokens & _COVERT_TOKENS) + (
        1 if "without getting caught" in w or "without being traced" in w else 0
    )
    harmful_hits = len(tokens & _HARMFUL_TOKENS) + (
        1 if "plant evidence" in w or "blackmail" in w or "extortion" in w else 0
    )
    weak_instructional_hits = len(tokens & _WEAK_INSTRUCTIONAL_TOKENS)
    strong_instructional_hits = len(tokens & _STRONG_INSTRUCTIONAL_TOKENS)
    # Weak connectives do not count as operational instruction.
    instructional_hits = strong_instructional_hits

    strong_ops = _response_has_strong_operational_abuse_instructions(norm, tokens)
    matched_ops: list[str] = []
    if strong_ops:
        for phrase in _STRONG_ABUSE_PHRASES:
            if phrase in w:
                matched_ops.append(phrase.strip())
        abuse_hit = sorted(tokens & _ABUSE_ACTION_TOKENS)
        if abuse_hit:
            matched_ops.extend(abuse_hit)
        if "first" in tokens and ("then" in tokens or "next" in tokens):
            matched_ops.append("first+then/next sequence")

    # Soft cyber vocabulary does not contribute to harm_score unless strong ops present.
    effective_cyber_for_score = hard_cyber_hits + (soft_cyber_hits if strong_ops else 0)
    effective_phrase_for_score = (phrase_hits - soft_cyber_phrase_hits) + (
        soft_cyber_phrase_hits if strong_ops else 0
    )
    harm_score = (
        effective_phrase_for_score + effective_cyber_for_score + covert_hits + harmful_hits
    )

    protective = surface_harmful_action_protective(norm, tokens)
    rule_id: str | None = None
    if protective and not strong_ops:
        rule_id = None
    elif strong_ops and (
        soft_cyber_hits >= 1
        or hard_cyber_hits >= 1
        or not tokens.isdisjoint(_CYBER_TARGET_TOKENS)
    ):
        # Soft cyber / auth targets complete only with strong operational-abuse instructions.
        rule_id = RULE_CYBER_ABUSE_GUIDANCE
    elif hard_cyber_hits >= 1 and (
        hard_cyber_hits >= 2
        or strong_instructional_hits >= 1
        or hard_phrase_hits >= 1
        or strong_ops
    ):
        rule_id = RULE_CYBER_ABUSE_GUIDANCE
    elif harm_score < 2 and not (harm_score >= 1 and instructional_hits >= 2):
        rule_id = None
    elif harmful_hits >= 1 and (
        harmful_hits >= 2
        or instructional_hits >= 1
        or (phrase_hits - soft_cyber_phrase_hits) >= 1
    ):
        rule_id = RULE_HARMFUL_ACTION_GUIDANCE
    else:
        # Prior behaviour: once the score gate passes without cyber/harmful, treat as covert.
        rule_id = RULE_COVERT_HARM_GUIDANCE

    return {
        "rule_id": rule_id,
        "phrase_hits": phrase_hits,
        "hard_phrase_hits": hard_phrase_hits,
        "soft_cyber_phrase_hits": soft_cyber_phrase_hits,
        "cyber_hits": cyber_hits,
        "hard_cyber_hits": hard_cyber_hits,
        "soft_cyber_hits": soft_cyber_hits,
        "covert_hits": covert_hits,
        "harmful_hits": harmful_hits,
        "instructional_hits": instructional_hits,
        "strong_instructional_hits": strong_instructional_hits,
        "weak_instructional_hits": weak_instructional_hits,
        "harm_score": harm_score,
        "strong_operational_abuse": strong_ops,
        "matched_operational_signals": matched_ops,
        "soft_cyber_tokens": soft_cyber_tokens,
        "hard_cyber_tokens": hard_cyber_tokens,
    }


def classify_harmful_operational_guidance(text: str) -> str | None:
    """Return response rule_id when output contains operational harmful guidance."""
    return analyze_harmful_operational_guidance(text)["rule_id"]
