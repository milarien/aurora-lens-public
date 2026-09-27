"""Stable source identifiers for trust-registry evaluation."""

from __future__ import annotations

import re

# Canonical stable IDs: ``src:`` or ``source:`` prefix + token body.
_SOURCE_ID_PATTERN = re.compile(
    r"^(?:src|source):[a-zA-Z0-9][a-zA-Z0-9._-]*$",
    re.IGNORECASE,
)


def normalize_source_id(raw: object) -> str | None:
    """Return a normalized stable source id, or None when invalid."""
    value = str(raw or "").strip()
    if not value or not _SOURCE_ID_PATTERN.fullmatch(value):
        return None
    prefix, _, body = value.partition(":")
    return f"{prefix.lower()}:{body.lower()}"


def source_id_key(source_id: str) -> str:
    """Case-insensitive lookup key for registry maps."""
    return str(source_id or "").strip().lower()
