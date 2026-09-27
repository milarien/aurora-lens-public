"""Structural Governor -- Phase 1+2: unresolved state transition.

The structural governor evaluates LLM output for structural admissibility
against the persistent PEF state. It returns a GovernorVerdict (ADMIT/HALT/
REJECT) which is the authoritative decision. The bridge rendering layer
converts the verdict into user-visible text.

Architecture: structural Governor -> hostage-negotiator renderer
The governor decides; the bridge renders.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aurora_lens.governor.verdict import GovernorVerdict
from aurora_lens.governor.mutations import (
    ProposedMutation,
    parse_proposed_mutations,
)
from aurora_lens.governor.admissibility import (
    AdmissibilityResult,
    assess_admissibility,
    _infer_domain,
)
from aurora_lens.governor.halt_envelope import build_halt_envelope
from aurora_lens.governor.span_gate import (
    split_temporal_clauses,
    requires_span_split,
)

if TYPE_CHECKING:
    from aurora_lens.pef.state import PEFState

_log = logging.getLogger(__name__)

__all__ = [
    "StructuralGovernor",
    "GovernorVerdict",
    "ProposedMutation",
    "AdmissibilityResult",
]


class StructuralGovernor:
    """Structural governor: unresolved state transition detection (Phase 1+2).

    Evaluates LLM output by:
    1. Extracting proposed mutations (state-transition claims) via spaCy
    2. Checking each mutation against PEF state for admissibility
    3. Returning a GovernorVerdict with ADMIT/HALT/REJECT

    The verdict is authoritative. The bridge uses it to decide rendering.
    """

    def evaluate(self, response_text: str, pef: "PEFState") -> GovernorVerdict:
        """Evaluate LLM output for structural admissibility against PEF.

        Returns GovernorVerdict with status:
          ADMIT  -- no state-transition violations found
          HALT   -- at least one mutation has an unmet precondition
          REJECT -- reserved for future hard-stop cases
        """
        mutations = parse_proposed_mutations(response_text, pef)

        if not mutations:
            return GovernorVerdict(status="ADMIT")

        results: list[AdmissibilityResult] = []
        worst_status = "ADMIT"
        worst_reason = None
        worst_domain = None
        worst_details = None

        for mutation in mutations:
            result = assess_admissibility(mutation, pef)
            results.append(result)

            if result.status == "REJECT":
                worst_status = "REJECT"
                worst_reason = result.reason
                worst_domain = result.details.get("domain") if result.details else None
                worst_details = result.details
            elif result.status == "HALT" and worst_status != "REJECT":
                worst_status = "HALT"
                worst_reason = result.reason
                worst_domain = result.details.get("domain") if result.details else None
                worst_details = result.details

        if worst_status == "ADMIT":
            return GovernorVerdict(status="ADMIT", mutations=mutations)

        halt_env = build_halt_envelope(
            mutations=mutations,
            results=results,
            verdict_status=worst_status,
            reason=worst_reason,
            domain=worst_domain,
        )

        return GovernorVerdict(
            status=worst_status,
            mutations=mutations,
            reason=worst_reason,
            domain=worst_domain or _infer_domain(mutations[0], pef),
            details=worst_details,
            halt_envelope=halt_env,
        )
