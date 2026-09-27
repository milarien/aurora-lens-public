"""Encoded payload extraction + deprecated hazardous-synthesis surface shims.

Hazardous-synthesis closed-phrase matching has moved to
``aurora_lens.verify.hazard``. The ``surface_*`` synthesis helpers below remain
as thin shims for import stability and dual-run tests; prefer
``evaluate_hazard_request``.
"""

from __future__ import annotations

import string
import warnings

from aurora_lens.verify.blocked_request_normalize import normalise_request_text

_B64_CHARS = frozenset(string.ascii_letters + string.digits + "+/")
_HEX_CHARS = frozenset(string.hexdigits)


# ---------------------------------------------------------------------------
# Deprecated closed lists — retained for documentation / dual-run reference
# ---------------------------------------------------------------------------

_HAZARDOUS_PHRASE_TOKENS: tuple[tuple[str, ...], ...] = (
    ("chlorine", "gas"),
    ("phosgene",),
    ("mustard", "gas"),
    ("hydrogen", "cyanide"),
    ("ricin",),
    ("sarin",),
    ("tabun",),
    ("vx",),
    ("cyanogen", "bromide"),
)


def _contains_contiguous(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    pl = len(phrase)
    if pl == 0 or len(tokens) < pl:
        return False
    for i in range(0, len(tokens) - pl + 1):
        if tuple(tokens[i : i + pl]) == phrase:
            return True
    return False


def surface_hazardous_named_agent(tokens: tuple[str, ...]) -> bool:
    """Deprecated: use hazard ontology substance/class matching."""
    warnings.warn(
        "surface_hazardous_named_agent is deprecated; use evaluate_hazard_request",
        DeprecationWarning,
        stacklevel=2,
    )
    return any(_contains_contiguous(tokens, p) for p in _HAZARDOUS_PHRASE_TOKENS)


def surface_synthesis_request_verb(norm: str, tokens: tuple[str, ...]) -> bool:
    """Deprecated: procedurality is assigned by the hazard frame parser."""
    warnings.warn(
        "surface_synthesis_request_verb is deprecated; use evaluate_hazard_request",
        DeprecationWarning,
        stacklevel=2,
    )
    w = f" {norm} "
    if (
        " give me " in w
        or " walk me through " in w
        or " show me " in w
        or " tell me " in w
        or " provide " in w
        or " what are " in w
        or " what is " in w
    ):
        return True
    if "explain" in tokens or "describe" in tokens or "list" in tokens:
        return True
    for i in range(len(tokens) - 1):
        if tokens[i] in ("how", "what") and tokens[i + 1] in ("do", "can", "would", "to", "are", "is"):
            return True
    return False


def surface_synthesis_production_intent(norm: str, tokens: tuple[str, ...]) -> bool:
    """Deprecated: process family comes from hazard ontology processes.json."""
    warnings.warn(
        "surface_synthesis_production_intent is deprecated; use evaluate_hazard_request",
        DeprecationWarning,
        stacklevel=2,
    )
    w = f" {norm} "
    pairs = (
        ("synthesis", "route"),
        ("synthesis", "step"),
        ("synthesis", "steps"),
        ("synthesis", "process"),
        ("synthesis", "procedure"),
        ("synthesis", "method"),
        ("production", "process"),
        ("production", "method"),
        ("production", "procedure"),
        ("production", "route"),
    )
    if any(_contains_contiguous(tokens, p) for p in pairs):
        return True
    intent_tokens = {
        "synthesize",
        "synthesise",
        "make",
        "produce",
        "producing",
        "manufacture",
        "manufacturing",
        "create",
        "prepare",
        "preparing",
        "synthesis",
        "production",
        "recipe",
        "ingredients",
        "steps",
    }
    if not intent_tokens.isdisjoint(frozenset(tokens)):
        return True
    return bool(
        " synthesis " in w
        or " synthesize " in w
        or " synthesise " in w
        or " production " in w
    )


def surface_illegal_hazardous_synthesis(user_input: str) -> bool:
    """Deprecated shim: True when the hazard ontology decides STOP.

    Prefer :func:`aurora_lens.verify.hazard.evaluate_hazard_request`.
    """
    warnings.warn(
        "surface_illegal_hazardous_synthesis is deprecated; use evaluate_hazard_request",
        DeprecationWarning,
        stacklevel=2,
    )
    from aurora_lens.verify.hazard import HazardDecision, evaluate_hazard_request

    result = evaluate_hazard_request(user_input)
    return result.decision == HazardDecision.STOP_HAZARD_TRANSFORM


def legacy_surface_illegal_hazardous_synthesis(user_input: str) -> bool:
    """Shadow-mode reference: original closed-list detector without deprecation noise."""
    norm = normalise_request_text(user_input)
    tokens = tuple(norm.split())
    return _legacy_three_way(norm, tokens)


def _legacy_request_verb(norm: str, tokens: tuple[str, ...]) -> bool:
    w = f" {norm} "
    if (
        " give me " in w
        or " walk me through " in w
        or " show me " in w
        or " tell me " in w
        or " provide " in w
        or " what are " in w
        or " what is " in w
    ):
        return True
    if "explain" in tokens or "describe" in tokens or "list" in tokens:
        return True
    for i in range(len(tokens) - 1):
        if tokens[i] in ("how", "what") and tokens[i + 1] in ("do", "can", "would", "to", "are", "is"):
            return True
    return False


def _legacy_production_intent(norm: str, tokens: tuple[str, ...]) -> bool:
    w = f" {norm} "
    pairs = (
        ("synthesis", "route"),
        ("synthesis", "step"),
        ("synthesis", "steps"),
        ("synthesis", "process"),
        ("synthesis", "procedure"),
        ("synthesis", "method"),
        ("production", "process"),
        ("production", "method"),
        ("production", "procedure"),
        ("production", "route"),
    )
    if any(_contains_contiguous(tokens, p) for p in pairs):
        return True
    intent_tokens = {
        "synthesize",
        "synthesise",
        "make",
        "produce",
        "producing",
        "manufacture",
        "manufacturing",
        "create",
        "prepare",
        "preparing",
        "synthesis",
        "production",
        "recipe",
        "ingredients",
        "steps",
    }
    if not intent_tokens.isdisjoint(frozenset(tokens)):
        return True
    return bool(
        " synthesis " in w
        or " synthesize " in w
        or " synthesise " in w
        or " production " in w
    )


def _legacy_three_way(norm: str, tokens: tuple[str, ...]) -> bool:
    return (
        any(_contains_contiguous(tokens, p) for p in _HAZARDOUS_PHRASE_TOKENS)
        and _legacy_request_verb(norm, tokens)
        and _legacy_production_intent(norm, tokens)
    )


_DECODE_INSTRUCTION_PHRASES: tuple[str, ...] = (
    "decode",
    "decoding",
    "decode this",
    "answer the prompt",
    "interpret this",
    "execute this",
    "run this",
    "decode and answer",
    "decode and execute",
)


def surface_encoded_decode_instruction(lower_collapsed: str) -> bool:
    w = f" {lower_collapsed} "
    return any(f" {p} " in w for p in _DECODE_INSTRUCTION_PHRASES)


def extract_base64_blobs(raw: str) -> list[str]:
    """Runs of base64 alphabet + '=' padding, length >= 32 (legacy regex parity)."""
    out: list[str] = []
    n = len(raw)
    i = 0
    while i < n:
        ch = raw[i]
        if ch not in _B64_CHARS and ch != "=":
            i += 1
            continue
        start = i
        while i < n and (raw[i] in _B64_CHARS or raw[i] == "="):
            i += 1
        seg = raw[start:i].strip()
        if len(seg) >= 32:
            out.append(seg)
    return out


def extract_hex_blobs(raw: str) -> list[str]:
    """Hex digit runs length >= 32; optional 0x prefix (legacy regex parity)."""
    out: list[str] = []
    n = len(raw)
    i = 0
    while i < n:
        ch = raw[i]
        if ch in "xX" and i > 0 and raw[i - 1] == "0":
            i += 1
            continue
        if ch not in _HEX_CHARS:
            i += 1
            continue
        start = i
        while i < n and raw[i] in _HEX_CHARS:
            i += 1
        seg = raw[start:i]
        if len(seg) >= 32:
            out.append(seg)
    return out
