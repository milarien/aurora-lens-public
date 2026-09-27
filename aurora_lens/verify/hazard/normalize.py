"""Normalize wrappers for the hazard request parser."""

from __future__ import annotations

from aurora_lens.verify.blocked_request_normalize import normalise_request_text

MAX_TOKENS_SCANNED = 512


def normalize_hazard_text(user_input: str) -> str:
    return normalise_request_text(user_input or "")


def tokenize_hazard_text(user_input: str) -> tuple[str, ...]:
    norm = normalize_hazard_text(user_input)
    if not norm:
        return ()
    tokens = tuple(norm.split())
    if len(tokens) > MAX_TOKENS_SCANNED:
        return tokens[:MAX_TOKENS_SCANNED]
    return tokens


def alias_to_tokens(alias: str) -> tuple[str, ...]:
    return tuple(normalize_hazard_text(alias).split())
