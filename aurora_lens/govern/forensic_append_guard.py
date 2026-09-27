"""Write-time forensic envelope checks shared by all bridges that persist ``forensic_event``.

Runs :func:`aurora_lens.governor.forensic_schema.validate` and
:func:`aurora_lens.governor.forensic_schema.verify_event_hash` after the envelope
is final (including ``chain_of_custody`` merged and ``event_hash`` refreshed).
"""

from __future__ import annotations

from typing import Any

from aurora_lens.governor import forensic_schema


class ForensicEnvelopeValidationError(Exception):
    """Raised when ``forensic_event`` must not be written: schema or ``event_hash`` invalid."""


def validate_forensic_event_for_append(event: dict[str, Any]) -> list[str]:
    """Return human-readable errors; empty list means the envelope may be appended."""
    errs = list(forensic_schema.validate(event))
    if not forensic_schema.verify_event_hash(event):
        errs.append("event_hash does not match canonical forensic envelope content")
    return errs


def enforce_forensic_event_for_append(event: dict[str, Any]) -> None:
    """Raise :class:`ForensicEnvelopeValidationError` if the envelope must not be persisted."""
    errs = validate_forensic_event_for_append(event)
    if errs:
        raise ForensicEnvelopeValidationError("; ".join(errs))
