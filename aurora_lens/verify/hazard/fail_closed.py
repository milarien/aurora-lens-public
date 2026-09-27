"""Fail-closed governed outcome when the hazard ontology cannot be loaded."""

from __future__ import annotations

from aurora_lens.verify.hazard.ontology_loader import OntologyLoadError
from aurora_lens.verify.hazard.schema import (
    HAZARD_FRAME_SCHEMA_VERSION,
    AmbiguityState,
    DecisionTrace,
    FramingKind,
    HazardDecision,
    HazardEvaluationResult,
    HazardRequestFrame,
    OutcomeKind,
    Procedurality,
    Severity,
)

RULE_HAZARD_ONTOLOGY_UNAVAILABLE = "blocked.request.hazard_ontology_unavailable"

_FALLBACK_OUTCOME = "HARD_STOP"


def _policy_version() -> str:
    # Late import avoids cycles with blocked_request_policy → hazard.
    from aurora_lens.verify.blocked_request_policy import BLOCKED_REQUEST_POLICY_VERSION

    return BLOCKED_REQUEST_POLICY_VERSION


def audit_evidence_for_load_failure(exc: OntologyLoadError) -> str:
    """Compact audit evidence — no filesystem paths or exception internals."""
    parts = [
        f"load_failure_category={exc.category}",
        f"component={exc.component or 'unknown'}",
        f"policy_version={_policy_version()}",
        f"fallback_outcome={_FALLBACK_OUTCOME}",
        f"decision={HazardDecision.ONTOLOGY_UNAVAILABLE.value}",
    ]
    return "; ".join(parts)


def public_claim_for_load_failure() -> str:
    """User-facing claim — no paths, no exception text."""
    return (
        "Hazard governance ontology is unavailable; the request cannot be "
        "safely evaluated and is refused"
    )


def fail_closed_ontology_unavailable(
    exc: OntologyLoadError,
) -> HazardEvaluationResult:
    """Build a deterministic fail-closed evaluation result from a load error."""
    empty = HazardRequestFrame(
        schema_version=HAZARD_FRAME_SCHEMA_VERSION,
        raw_norm="",
        tokens=(),
        action=None,
        substance=None,
        source_organism=None,
        transformation=None,
        intended_outcome=OutcomeKind.NONE,
        hazard_class_ids=(),
        procedurality=Procedurality.INFORMATIONAL,
        consequence_severity=Severity.UNKNOWN,
        user_framing=FramingKind.BARE,
        ambiguity=AmbiguityState.NONE,
        evidence_spans=(),
    )
    notes = (
        f"category={exc.category}",
        f"component={exc.component or 'unknown'}",
        f"policy_version={_policy_version()}",
        f"fallback_outcome={_FALLBACK_OUTCOME}",
    )
    trace = DecisionTrace(
        frame=empty,
        decision=HazardDecision.ONTOLOGY_UNAVAILABLE,
        rule_applied="fail_closed_ontology_load",
        matrix_key=None,
        notes=notes,
    )
    return HazardEvaluationResult(
        decision=HazardDecision.ONTOLOGY_UNAVAILABLE,
        frame=empty,
        trace=trace,
        rule_id=RULE_HAZARD_ONTOLOGY_UNAVAILABLE,
    )
