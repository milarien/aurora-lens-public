"""PEF admission envelope — bounded mutator semantics for Persistent Existence Framework.

Admission answers whether a bounded interpreter path may run **and reports durable
effects** with **orthogonal** mutation categories. ``ADMIT`` is **not**
``successful persistence`` — see banned vocabulary below.

Dual-count XOR rule:
  Assign each durable persistence event exactly one of:
    * world-state mutation (relations, substantive entity records used for grounding)
    * continuation-state mutation (discourse bindings that constrain subsequent turns)

  Echo traces, typography-only bookkeeping, etc. are out of scope for v1 ontology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ─── Banned vocabulary (constitutional) ───────────────────────────────────────
#
# Forbidden in names, audits, telemetry, or tests implying admission outcomes:
#
# - Using "COMMIT" as an **admission** gate result — use ``PEFAdmissionDecision``.
# - "success" as a synonym for ``ADMIT``.
# - "passed" as a synonym for "persisted" / world fact written.
# - ``committed=True`` unless a **world-state** persistence event occurred.
# - ``state_changed`` derived from ``decision == ADMIT`` alone — derive counts only.
#

__all__ = [
    "PEFAdmissionDecision",
    "PEFAdmissionEvidence",
    "PEFAdmissionResult",
    "adjudicate_update_pef_aggregate",
    "make_admission_evidence",
    "pef_admission_result_wire_dict",
]


class PEFAdmissionDecision(Enum):
    """Gate outcome for the bounded updater envelope."""

    ADMIT = "ADMIT"
    HOLD = "HOLD"
    REJECT = "REJECT"


@dataclass(frozen=True)
class PEFAdmissionEvidence:
    """Recorded reason for admission sub-slice veto or annotate (append-only semantics).

    Do not rename to embed "Commit" — reserved for durable world writes only elsewhere.
    """

    veto_kind: str
    scope: str = "claim"
    held_reason: str | None = None
    detail: str | None = None

    # Optional bookkeeping for debugging / future audit projection.
    slice_index: int | None = None


@dataclass
class PEFAdmissionResult:
    """Outcome of ``update_pef`` (bounded mutator envelope only — not whole Lens pipeline).

    * ``mutation_count`` = ``world_state_mutation_count +
      continuation_state_mutation_count`` (single derivation).

    ``None`` for count fields strictly means envelope not reached / not applicable.
    Zero means accounting executed: no durable mutations in that category occurred.
    """

    decision: PEFAdmissionDecision
    write_intent: bool
    planned_mutation_slices: int

    world_state_mutation_count: int | None
    continuation_state_mutation_count: int | None

    mutation_count: int | None
    mutation_summary: dict[str, Any] = field(default_factory=dict)

    evidence: list[PEFAdmissionEvidence] = field(default_factory=list)


def adjudicate_update_pef_aggregate(evidence: list[PEFAdmissionEvidence]) -> PEFAdmissionDecision:
    """Aggregate classifier for ``update_pef``.

    Bounded mutator presently **always completes** envelope when invoked (no envelope
    REJECT branch). Semantic holds (``U-TX*``) are evidenced but do not force aggregate
    ``HOLD`` here — clarification holds attach at Lens orchestration layer.

    If later the mutator emits batch-level rejection evidence (``REJECT_BATCH``),
    return ``PEFAdmissionDecision.REJECT``.
    """

    if any(ev.veto_kind == "REJECT_BATCH" for ev in evidence):
        return PEFAdmissionDecision.REJECT
    return PEFAdmissionDecision.ADMIT


def make_admission_evidence(
    veto_kind: str,
    *,
    scope: str = "claim",
    held_reason: str | None = None,
    detail: str | None = None,
    slice_index: int | None = None,
) -> PEFAdmissionEvidence:
    return PEFAdmissionEvidence(
        veto_kind=veto_kind,
        scope=scope,
        held_reason=held_reason,
        detail=detail,
        slice_index=slice_index,
    )


def pef_admission_result_wire_dict(result: PEFAdmissionResult) -> dict[str, Any]:
    """JSON-serializable view of ``PEFAdmissionResult`` for operator-plane HTTP payloads.

    Field names mirror the dataclass (snake_case). This does not redefine admission
    semantics; it only projects the bounded mutator envelope for observability.
    """

    return {
        "decision": result.decision.value,
        "write_intent": result.write_intent,
        "planned_mutation_slices": result.planned_mutation_slices,
        "world_state_mutation_count": result.world_state_mutation_count,
        "continuation_state_mutation_count": result.continuation_state_mutation_count,
        "mutation_count": result.mutation_count,
        "mutation_summary": dict(result.mutation_summary),
        "evidence": [
            {
                "veto_kind": e.veto_kind,
                "scope": e.scope,
                "held_reason": e.held_reason,
                "detail": e.detail,
                "slice_index": e.slice_index,
            }
            for e in result.evidence
        ],
    }


def finalized_mutation_aggregate(
    world: int,
    continuation: int,
) -> tuple[int | None, int | None, int | None, dict[str, Any]]:
    """Return (world_ct, continuation_ct, sum, summary) once accounting envelope ran."""

    summary: dict[str, Any] = {
        "world_state_writes": world,
        "continuation_state_writes": continuation,
        "ontology": "pef_admission_xor_v1",
    }
    return world, continuation, world + continuation, summary
