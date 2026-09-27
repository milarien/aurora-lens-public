"""Validation freshness checks for sovereign provider profiles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


def parse_validation_timestamp(raw: str) -> datetime | None:
    """Parse ISO-8601 ``last_validated_at`` values; returns None when unusable."""
    text = str(raw or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass(frozen=True)
class ValidationFreshnessPolicy:
    """When ``last_validated_at`` is older than this threshold, state is stale."""

    stale_threshold_days: int = 30


def validation_timestamp_is_stale(
    last_validated_at: str,
    *,
    policy: ValidationFreshnessPolicy,
    now: datetime | None = None,
) -> bool:
    """Return True when validation timestamp is missing or older than policy threshold."""
    if policy.stale_threshold_days <= 0:
        return False
    parsed = parse_validation_timestamp(last_validated_at)
    if parsed is None:
        return True
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return parsed < reference - timedelta(days=policy.stale_threshold_days)
