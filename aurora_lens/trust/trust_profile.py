"""Declared trust profiles for evidence sources."""

from __future__ import annotations

from dataclasses import dataclass

TRUST_STATUS_TRUSTED = "trusted"
TRUST_STATUS_UNTRUSTED = "untrusted"
TRUST_STATUS_RESTRICTED = "restricted"

TRUST_STATUSES: frozenset[str] = frozenset({
    TRUST_STATUS_TRUSTED,
    TRUST_STATUS_UNTRUSTED,
    TRUST_STATUS_RESTRICTED,
})


@dataclass(frozen=True)
class TrustSourceProfile:
    """One declared source entry in a trust registry."""

    source_id: str
    source_name: str
    source_class: str
    trust_status: str
    permitted_domains: tuple[str, ...] = ()
    max_consequence_grade: str = "critical"
    certification_ref: str | None = None
