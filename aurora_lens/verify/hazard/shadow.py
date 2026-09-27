"""Shadow dual-run comparison between legacy closed-list and hazard ontology."""

from __future__ import annotations

from dataclasses import dataclass

from aurora_lens.verify.blocked_request_surface_synthesis_encoded import (
    legacy_surface_illegal_hazardous_synthesis,
)
from aurora_lens.verify.hazard import evaluate_hazard_request
from aurora_lens.verify.hazard.schema import HazardDecision


@dataclass(frozen=True)
class HazardShadowCompare:
    prompt: str
    legacy_stop: bool
    ontology_decision: str
    agree_on_stop: bool


def shadow_compare_hazard_request(user_input: str) -> HazardShadowCompare:
    """Compare legacy synthesis surface STOP vs ontology STOP (for dual-run logs)."""
    legacy = legacy_surface_illegal_hazardous_synthesis(user_input)
    result = evaluate_hazard_request(user_input)
    onto_stop = result.decision == HazardDecision.STOP_HAZARD_TRANSFORM
    return HazardShadowCompare(
        prompt=user_input,
        legacy_stop=legacy,
        ontology_decision=result.decision.value,
        agree_on_stop=legacy == onto_stop,
    )
