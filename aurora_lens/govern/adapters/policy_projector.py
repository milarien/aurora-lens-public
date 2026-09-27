"""PolicyProjector — maps canonical GovernorPolicy to RuntimeDecisionProjection.

ROLE
----
PolicyProjector answers: "How do I express this Governor decision in the
legacy Lens action vocabulary and in the runtime record?"

It does NOT re-evaluate flags, re-interpret the LensStatus, or make new
policy decisions. It is a vocabulary translator and nothing more.

PROJECTION INVARIANT (enforced in tests — do not remove)
---------------------------------------------------------
Projection is information-losing in only one direction:

  ALLOWED:   many Governor pathways → same InterventionAction
  FORBIDDEN: projection step alters commitment_closed, forensic_obligations,
             escalation_target, or output_mode

If you are modifying commitment_closed or forensic_obligations inside the
projector, you are writing a second policy brain. Stop and rethink.

The projector works purely from the GovernorPolicy produced by PolicyResolver.
It never reads flags, PEF state, or request context.

PATHWAY → INTERVENTION ACTION TABLE
------------------------------------
Governor pathway            InterventionAction (compatibility projection)
--------------------------  ------------------------------------------
P_ADMIT_STANDARD            PASS
P_ASK_DISAMBIGUATE          CONTAIN
P_ASK_MISSING_FACT          CONTAIN
P_REFUSE_EXPLAIN_REDIRECT   FORCE_REVISE
P_REFUSE_ESCALATE_PRO       FORCE_REVISE
P_HANDOFF_SUMMARY           FORCE_REVISE
P_STOP_TERMINAL             HARD_STOP
P_STOP_FORENSIC             HARD_STOP
P_STOP_ESCALATE             HARD_STOP
P_STOP_REFUSE_CLEAN         HARD_STOP

Note: SOFT_CORRECT is intentionally absent from this table. The canonical
Governor has no "annotate and pass" pathway — SOFT_CORRECT as an effective
outcome is handled by the bridge when the action is PASS but minor flags were
present (e.g. UNSUPPORTED_ATTRIBUTE in open mode). The projector does not
make that distinction; the bridge does.
"""

from __future__ import annotations

from typing import Literal

from aurora_lens.governor.models import (
    ContinuationPathway,
    GovernorPolicy,
)
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.govern.adapters.runtime_types import (
    ContextResolutionProvenance,
    RuntimeDecisionProjection,
)

ProjectionSource = Literal["explicit", "fallback"]

# ── Pathway → InterventionAction compatibility table ─────────────────────────

_PATHWAY_TO_ACTION: dict[ContinuationPathway, InterventionAction] = {
    ContinuationPathway.P_ADMIT_STANDARD:         InterventionAction.PASS,
    ContinuationPathway.P_ASK_DISAMBIGUATE:       InterventionAction.CONTAIN,
    ContinuationPathway.P_ASK_MISSING_FACT:       InterventionAction.CONTAIN,
    ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT: InterventionAction.FORCE_REVISE,
    ContinuationPathway.P_REFUSE_ESCALATE_PRO:    InterventionAction.FORCE_REVISE,
    ContinuationPathway.P_HANDOFF_SUMMARY:        InterventionAction.FORCE_REVISE,
    ContinuationPathway.P_STOP_TERMINAL:          InterventionAction.HARD_STOP,
    ContinuationPathway.P_STOP_FORENSIC:          InterventionAction.HARD_STOP,
    ContinuationPathway.P_STOP_ESCALATE:          InterventionAction.HARD_STOP,
    ContinuationPathway.P_STOP_REFUSE_CLEAN:      InterventionAction.HARD_STOP,
}

# Fallback for any pathway not in the table (safe default).
_PATHWAY_FALLBACK = InterventionAction.HARD_STOP


def pathway_projection_source(pathway_id: ContinuationPathway) -> ProjectionSource:
    """Return whether ``pathway_id`` resolves via explicit map or fallback."""
    if pathway_id in _PATHWAY_TO_ACTION:
        return "explicit"
    return "fallback"


class PolicyProjector:
    """Projects a canonical GovernorPolicy into a RuntimeDecisionProjection.

    Pure function with no state. Calling project() twice with the same policy
    always returns an equal projection (no side effects, no external reads).

    Usage:
        projector = PolicyProjector()
        projection = projector.project(policy)
    """

    def project(
        self,
        policy: GovernorPolicy,
        provenance: ContextResolutionProvenance | None = None,
    ) -> RuntimeDecisionProjection:
        """Map a GovernorPolicy to a RuntimeDecisionProjection.

        Args:
            policy:     The canonical policy produced by PolicyResolver.resolve().
            provenance: Optional context resolution provenance to carry through
                        into the projection for forensic envelope enrichment.

        Returns:
            A RuntimeDecisionProjection. The following fields pass through
            unchanged from GovernorPolicy (PROJECTION INVARIANT):
              - commitment_closed
              - interaction_open
              - forensic_obligations
              - escalation_target
              - output_mode
              - exposure_level
              - resolution_mode

        Only `intervention_action` (and its ``projection_source`` tag) is a
        translation product.
        """
        mapped = _PATHWAY_TO_ACTION.get(policy.pathway_id)
        if mapped is not None:
            intervention_action = mapped
            source: ProjectionSource = "explicit"
        else:
            intervention_action = _PATHWAY_FALLBACK
            source = "fallback"

        return RuntimeDecisionProjection(
            pathway_id=policy.pathway_id,
            output_mode=policy.output_mode,
            intervention_action=intervention_action,
            commitment_closed=policy.commitment_closed,
            interaction_open=policy.interaction_open,
            forensic_obligations=list(policy.forensic_obligations),
            required_disclosures=list(policy.required_disclosures),
            escalation_target=policy.escalation_target,
            exposure_level=policy.exposure_level,
            resolution_mode=policy.resolution_mode,
            provenance=provenance,
            projection_source=source,
        )
