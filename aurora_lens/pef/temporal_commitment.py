"""Present-bound temporal commitments in PEF (Phase 3 storage).

Committed rows here are **present commitments** about reconstruction, projection,
supersession, ambiguity, deficit of temporal anchor, or indeterminacy *within*
an anchor—not “future accomplished facts” or archival timeline primitives.

Uses :class:`~aurora_lens.state_native_engine.temporal_outcome_contract.TemporalOutcomeContract`
as the authoritative discriminant (no alias to ``EpistemicResult.UNKNOWN``).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from aurora_lens.state_native_engine.temporal_outcome_contract import (
    TemporalOutcomeContract,
)


# Wire format minor version for persistence round-trips.
PRESENT_TEMPORAL_COMMITMENT_SCHEMA_VERSION: int = 1


@dataclass
class PresentTemporalCommitment:
    """A single admitted present commitment about temporal reasoning.

    Not frozen: ``continuity_payload`` is a plain dict (not hashable) for practical
    evolution in later phases.
    """

    outcome_kind: TemporalOutcomeContract
    source_turn: int
    evidence: str
    subject_entity_id: str | None = None
    commitment_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Present-committed explanatory payload (anchors, supersession refs, ambiguity list).
    continuity_payload: dict[str, Any] = field(default_factory=dict)


def commitment_to_wire_dict(c: PresentTemporalCommitment) -> dict[str, Any]:
    return {
        "schema_version": PRESENT_TEMPORAL_COMMITMENT_SCHEMA_VERSION,
        "commitment_id": c.commitment_id,
        "outcome_kind": c.outcome_kind.value,
        "source_turn": c.source_turn,
        "evidence": c.evidence,
        "subject_entity_id": c.subject_entity_id,
        "continuity_payload": dict(c.continuity_payload),
    }


def commitment_from_wire_dict(data: dict[str, Any]) -> PresentTemporalCommitment:
    if not isinstance(data, dict):
        raise TypeError(f"present temporal commitment wire must be dict, got {type(data)}")
    raw_kind = data.get("outcome_kind")
    try:
        kind = TemporalOutcomeContract(str(raw_kind))
    except ValueError as e:
        raise ValueError(f"invalid TemporalOutcomeContract: {raw_kind!r}") from e
    payload = data.get("continuity_payload") or {}
    if not isinstance(payload, dict):
        raise TypeError("continuity_payload must be dict or absent")
    return PresentTemporalCommitment(
        outcome_kind=kind,
        source_turn=int(data["source_turn"]),
        evidence=str(data.get("evidence", "")),
        subject_entity_id=(
            None
            if data.get("subject_entity_id") is None
            else str(data["subject_entity_id"])
        ),
        commitment_id=str(data.get("commitment_id") or uuid.uuid4().hex),
        continuity_payload=dict(payload),
    )


def temporal_commitments_to_wire(rows: list[PresentTemporalCommitment]) -> list[dict[str, Any]]:
    """Deterministic serialization (sorted by source_turn then commitment_id)."""
    sorted_rows = sorted(rows, key=lambda r: (r.source_turn, r.commitment_id))
    return [commitment_to_wire_dict(r) for r in sorted_rows]


def temporal_commitments_from_wire(rows: Any) -> list[PresentTemporalCommitment]:
    if not isinstance(rows, list):
        return []
    out: list[PresentTemporalCommitment] = []
    for item in rows:
        if isinstance(item, dict):
            sv = item.get("schema_version", PRESENT_TEMPORAL_COMMITMENT_SCHEMA_VERSION)
            if sv != PRESENT_TEMPORAL_COMMITMENT_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported present temporal commitment schema_version: {sv!r}"
                )
            out.append(commitment_from_wire_dict(item))
    return out
