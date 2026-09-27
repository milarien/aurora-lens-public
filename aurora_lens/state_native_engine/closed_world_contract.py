"""Closed-world output contract for state-native solved answers.

This contract keeps solved-state outcomes explicit so output handling never
re-opens referents once a unique committed-state solution is available.
"""

from __future__ import annotations

from enum import Enum

from aurora_lens.state_native_engine.contracts import StateNativeOutcome, StateNativeSolverFamily


class ClosedWorldResolution(str, Enum):
    SOLVED_UNIQUE = "SOLVED_UNIQUE"
    MULTIPLE_SOLUTIONS = "MULTIPLE_SOLUTIONS"
    NO_CONSISTENT_SOLUTION = "NO_CONSISTENT_SOLUTION"
    NEEDS_FORMALISATION = "NEEDS_FORMALISATION"


_STOP_REASON_NO_CONSISTENT = frozenset(
    {
        "state_native_no_at",
        "state_native_at_incomplete",
        "state_native_no_inventory",
        "state_native_no_holder",
    }
)


def classify_closed_world_resolution(
    *,
    outcome: StateNativeOutcome,
    stop_reason_code: str | None,
    solver_family: StateNativeSolverFamily | None,
) -> ClosedWorldResolution | None:
    """Map state-native outcomes to the closed-world solved-answer contract."""
    if solver_family != StateNativeSolverFamily.CLOSED_WORLD:
        return None
    if outcome == StateNativeOutcome.ANSWER:
        return ClosedWorldResolution.SOLVED_UNIQUE
    if outcome == StateNativeOutcome.CLARIFY:
        return ClosedWorldResolution.MULTIPLE_SOLUTIONS
    if outcome == StateNativeOutcome.STOP:
        if (stop_reason_code or "").strip().lower() in _STOP_REASON_NO_CONSISTENT:
            return ClosedWorldResolution.NO_CONSISTENT_SOLUTION
        return ClosedWorldResolution.NEEDS_FORMALISATION
    raise ValueError(f"unsupported state-native outcome: {outcome!r}")
