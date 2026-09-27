"""GovernorVerdict — decision record from the structural governor.

The verdict is the authoritative decision. It is not user-facing text.
The bridge rendering layer turns it into the user-visible response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from aurora_lens.governor.mutations import ProposedMutation


@dataclass
class GovernorVerdict:
    """Structural admissibility verdict for an LLM response.

    status:
        ADMIT  — all proposed mutations are structurally admissible.
        HALT   — at least one mutation has an unmet precondition (e.g. unresolved
                 state transition). The response should be revised.
        REJECT — at least one mutation is structurally inadmissible and cannot
                 be revised (hard stop).

    The bridge stores this as the authoritative record on GovernanceDecision.
    A synthetic Flag is created as a rendering adapter only.
    """

    status: Literal["ADMIT", "HALT", "REJECT"]
    mutations: list[ProposedMutation] = field(default_factory=list)
    reason: str | None = None
    domain: str | None = None
    details: dict | None = None
    halt_envelope: dict | None = None
