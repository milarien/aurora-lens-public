"""Verify source status of PEF relationships used in state-native answers.

**Current Implementation (Conservative):**
Source verification checks if relationships are:
1. Admitted/committed (provenance is user_input or system, span is present/past, not negated)
2. Not globally blocked by PEF epistemic hold (REFUSAL or terminal STOP)

**Limitation: Global Hold Scope**
This implementation is conservative while hold scoping remains global. A response-level
refusal (blocking a generated advisory claim) will block verification of ALL relationships,
even unrelated admitted ones.

**Why this limitation exists:**
Without branch/source-scoped contamination tracking, we cannot distinguish:
- relationships that source from the refused/generated claim's chain
- relationships that are admitted and completely independent

**Future improvement:**
Branch-scoped contamination would allow:
- Response-level refusal to block only claims/branches dependent on the refused claim
- Independent admitted relationships to remain verified even with active refusal hold
- More precise hold semantics (which relationships are contaminated, which are clean)

This would require:
1. Tracking provenance chains (not just immediate provenance)
2. Marking which relationships depend on which refusals
3. Scoped hold checking (is THIS relationship blocked by THIS hold, not global yes/no)

For now, the conservative approach is correct: when doubt exists, assume contamination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aurora_lens.pef.state import (
    EPISTEMIC_MODE_AMBIGUITY,
    EPISTEMIC_MODE_REFUSAL,
    EPISTEMIC_MODE_STOP,
)

if TYPE_CHECKING:
    from aurora_lens.pef.state import PEFState, Relationship


def verify_source_relationships_admitted(
    pef: PEFState,
    relationship_indices: list[int],
) -> str:
    """Verify that relationships at given indices are uncontaminated.

    **LIMITATION: Global Hold Scoping (Not Branch-Scoped)**
    This function checks if ANY epistemic hold exists globally. Without branch-scoped
    contamination tracking, it cannot determine if a specific hold affects this specific
    set of relationships.

    A response-level refusal (blocking a generated advisory) will reject verification
    even for relationships completely independent of that advisory.

    **Why this limitation exists:**
    Without provenance dependency graphs, we cannot answer:
    - "Are these relationships linked to the refused claim's source chain?"
    - "Are these relationships independent from that refusal?"

    We can only answer:
    - "Is PEF globally under a refusal/stop hold?" (yes/no)

    **Future improvement:**
    Branch-scoped verification would require tracking:
    1. Which relationships depend on which refusals
    2. Transitive provenance chains (not just immediate provenance)
    3. Relationship-specific contamination checks

    Until then, conservative approach: if any refusal hold exists, assume all
    unverified-status.

    Returns:
    - "admitted_uncontaminated": All relationships are committed/admitted and no
      global epistemic hold blocks them (conservative while hold scoping is global)
    - "unverified": Cannot prove all relationships are uncontaminated
    """
    if not relationship_indices:
        # Empty relationship set is unverified (no explicit evidence)
        return "unverified"

    # Check if PEF is under a hold that would globally contaminate these relationships
    hold = pef.epistemic_hold
    if hold:
        mode = hold.get("mode")
        # Refusal and terminal stop contaminate available evidence globally
        if mode in (EPISTEMIC_MODE_REFUSAL, EPISTEMIC_MODE_STOP):
            # Stop may have interaction_open (reopenable); check if terminal
            if mode == EPISTEMIC_MODE_STOP:
                if hold.get("interaction_open") is False:
                    # Terminal stop contaminates globally
                    return "unverified"
                # Reopenable stop does not globally contaminate committed PEF
            else:
                # REFUSAL contaminates globally (cannot scope to individual relationships)
                return "unverified"
        # Ambiguity holds do not contaminate committed state (only ambiguous entities)

    # Check each relationship for admission status
    for idx in relationship_indices:
        if idx < 0 or idx >= len(pef.relationships):
            # Out of bounds
            return "unverified"

        rel = pef.relationships[idx]
        # Check provenance: should be from user_input or system (extracted/admitted)
        # not from generated/inferred
        if rel.provenance not in ("user_input", "system"):
            # Provenance not from admitted sources
            return "unverified"
        # Span should be present/past (committed), not future/uncertain
        if rel.span not in ("present", "past"):
            return "unverified"
        # Relationships should not be negated (they assert committed state)
        if rel.negated:
            return "unverified"

    # All checks passed: relationships are admitted and uncontaminated (under global assumptions)
    return "admitted_uncontaminated"
