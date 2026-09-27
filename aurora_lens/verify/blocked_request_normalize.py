"""Shared normalization for blocked-request surface scans (regex-free).

Folds punctuation to whitespace so downstream logic uses deterministic tokens.
"""

from __future__ import annotations

_FIN_REQUEST_PUNCT_TO_SPACE = str.maketrans(
    {c: " " for c in ",.;:!?()[]{}\"'`/\\|-_+\n\r\t"}
)


def normalise_request_text(user_input: str) -> str:
    """Lowercase + punctuation fold for deterministic phrase / token matching."""
    if not user_input:
        return ""
    lowered = user_input.lower().replace("\u2019", "'")
    return " ".join(lowered.translate(_FIN_REQUEST_PUNCT_TO_SPACE).split())
