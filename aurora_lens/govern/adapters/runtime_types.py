"""RuntimeDecisionProjection — canonical internal governance decision record.

This is the true internal meaning of a governance decision produced by the
canonical Governor pipeline. It carries the full Governor vocabulary.

InterventionAction is one projected output vocabulary (the compatibility field
`intervention_action`) for callers that still speak the legacy Lens action
surface. It is a projection, not the real internal meaning.

PROJECTION INVARIANT
--------------------
Projection is information-losing in only one direction:
  - Many Governor pathways MAY map to the same InterventionAction.
  - No projection step may alter commitment_closed, forensic_obligations,
    escalation_target, or output_mode.
If you find yourself modifying those fields in a projector, you are writing a
second policy brain, not a compatibility shim. Stop and rethink.
"""

from __future__ import annotations

import sys
from pathlib import Path

# governor/ is a sibling package of aurora_lens/ inside the aurora-lens repo.
# It is not distributed via PyPI; add the repo root to sys.path so it is
# importable regardless of working directory or entry-point location.
_REPO_ROOT = Path(__file__).resolve().parents[3]  # …/aurora-lens/
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataclasses import dataclass, field

from aurora_lens.governor.models import (
    ContinuationPathway,
    DisclosureType,
    ExposureLevel,
    ForensicObligation,
    OutputMode,
    ResolutionMode,
)
from aurora_lens.govern.decision import InterventionAction


@dataclass(frozen=True)
class ContextResolutionProvenance:
    """Records where each context value came from.

    Included in forensic envelopes so reviewers know whether domain was
    deliberately routed or lazily inferred from flag-pattern fallback.

    domain_source:
        "operator_channel" — set via domain_var ContextVar (route/header hint)
        "flag_pattern" — inferred from flag type taxonomy
        "default"      — fell through all cascade levels; Domain.GENERAL

    authority_source:
        "config"       — read from GovernanceConfig.authority_class at startup
        "default"      — GovernanceConfig not set; AuthorityClass.GP

    user_class_source:
        "header"       — read from per-request HTTP header
        "default"      — header not configured or not present; UserClass.GENERAL
    """

    domain_source: str          # "operator_channel" | "flag_pattern" | "default"
    authority_source: str       # "config" | "default"
    user_class_source: str      # "header" | "default"


@dataclass(frozen=True)
class RuntimeDecisionProjection:
    """The canonical internal governance decision record.

    Produced by PolicyProjector from a GovernorPolicy. Carries the full
    Governor vocabulary. `intervention_action` is the compatibility projection
    for external callers (bridge, audit log, API surface).

    PROJECTION INVARIANT: commitment_closed, forensic_obligations,
    escalation_target, and output_mode are never altered by the projector.
    They pass through unchanged from GovernorPolicy.
    """

    pathway_id: ContinuationPathway
    output_mode: OutputMode
    intervention_action: InterventionAction          # compatibility field
    commitment_closed: bool
    interaction_open: bool
    forensic_obligations: list[ForensicObligation] = field(default_factory=list)
    required_disclosures: list[DisclosureType] = field(default_factory=list)
    escalation_target: str | None = None
    exposure_level: ExposureLevel = ExposureLevel.MINIMAL
    resolution_mode: ResolutionMode = ResolutionMode.EXACT
    provenance: ContextResolutionProvenance | None = None
    # Observability only — not copied into GovernanceDecision / audit fields.
    # "explicit" = pathway found in PolicyProjector._PATHWAY_TO_ACTION
    # "fallback" = pathway resolved via PolicyProjector._PATHWAY_FALLBACK
    projection_source: str = "explicit"
