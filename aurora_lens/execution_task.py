"""Structured execution-boundary tasks (API/demo envelope; not user prose).

Hosts submit explicit task payloads on chat completion requests. Lens reads the
parsed task from :data:`~aurora_lens.context.execution_task_var` and adjudicates
without inferring task intent from free-text messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TASK_ADJUDICATE_CANDIDATE_RELEASE = "adjudicate_candidate_release"


def _opt_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    return None


@dataclass(frozen=True)
class NonEstablishmentRecord:
    """Structured record that present evidence does not establish a claim."""

    claim: str
    basis: str | None = None


@dataclass(frozen=True)
class EvidenceState:
    """Present evidence supplied by the host — not inferred from user chat text."""

    observations: tuple[dict[str, Any], ...] = ()
    established_claims: tuple[str, ...] = ()
    non_establishment: tuple[NonEstablishmentRecord, ...] = ()


@dataclass(frozen=True)
class GoverningPolicy:
    """Release policy supplied by the host for candidate-release adjudication."""

    require_present_evidence: bool = True
    admissibility_rule: str = "release_requires_established_claim"


@dataclass(frozen=True)
class CandidateReleaseTask:
    task_type: str
    candidate_release: str
    evidence_state: EvidenceState
    governing_policy: GoverningPolicy


ExecutionTask = CandidateReleaseTask


def _parse_non_establishment(raw: Any) -> tuple[NonEstablishmentRecord, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[NonEstablishmentRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        claim = _opt_str(item.get("claim"))
        if not claim:
            continue
        out.append(
            NonEstablishmentRecord(
                claim=claim,
                basis=_opt_str(item.get("basis")),
            )
        )
    return tuple(out)


def _parse_established_claims(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[str] = []
    for item in raw:
        s = _opt_str(item)
        if s:
            out.append(s)
    return tuple(out)


def _parse_observations(raw: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list):
        return ()
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(dict(item))
    return tuple(out)


def _parse_evidence_state(raw: Any) -> EvidenceState:
    if not isinstance(raw, dict):
        return EvidenceState()
    return EvidenceState(
        observations=_parse_observations(raw.get("observations")),
        established_claims=_parse_established_claims(raw.get("established_claims")),
        non_establishment=_parse_non_establishment(raw.get("non_establishment")),
    )


def _parse_governing_policy(raw: Any) -> GoverningPolicy:
    if not isinstance(raw, dict):
        return GoverningPolicy()
    require_raw = raw.get("require_present_evidence", True)
    if isinstance(require_raw, str):
        require = require_raw.strip().lower() not in ("0", "false", "no", "off")
    else:
        require = bool(require_raw)
    rule = _opt_str(raw.get("admissibility_rule")) or "release_requires_established_claim"
    return GoverningPolicy(
        require_present_evidence=require,
        admissibility_rule=rule,
    )


def parse_execution_task(raw: Any) -> ExecutionTask | None:
    """Parse ``execution_task`` object from a chat completion body."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    task_type = (_opt_str(raw.get("task_type")) or "").lower()
    if task_type != TASK_ADJUDICATE_CANDIDATE_RELEASE:
        return None
    candidate_release = _opt_str(raw.get("candidate_release"))
    if not candidate_release:
        return None
    return CandidateReleaseTask(
        task_type=task_type,
        candidate_release=candidate_release,
        evidence_state=_parse_evidence_state(raw.get("evidence_state")),
        governing_policy=_parse_governing_policy(raw.get("governing_policy")),
    )


def parse_execution_task_from_body(body: dict[str, Any]) -> ExecutionTask | None:
    """Read execution task from top-level body or ``aurora.execution_task``."""
    if not isinstance(body, dict):
        return None
    raw = body.get("execution_task")
    if raw is None:
        aurora = body.get("aurora")
        if isinstance(aurora, dict):
            raw = aurora.get("execution_task")
    return parse_execution_task(raw)
