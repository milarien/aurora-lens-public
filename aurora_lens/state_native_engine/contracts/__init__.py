"""State-native engine contracts — outcome + delegation inputs (no governance imports)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from aurora_lens.interpret.turn_act import TurnAct
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState
from aurora_lens.state_native_engine.epistemic import EpistemicResult
from aurora_lens.state_native_engine.temporal_eval_result import (
    PresentBoundTemporalEvalResult,
)

from .possession_mutation_frame import (
    PossessionMutationAction,
    PossessionMutationConfidence,
    PossessionMutationDeterministicParse,
    PossessionMutationFrame,
    PossessionMutationParseSource,
    PossessionMutationVoice,
)


class StateNativeOutcome(str, Enum):
    """Epistemic outcome from committed state only."""

    ANSWER = "answer"
    CLARIFY = "clarify"
    STOP = "stop"


class StateNativeSolverFamily(str, Enum):
    CLOSED_WORLD = "closed_world"
    COMMITTED_LOCATION_READ = "committed_location_read"
    COMMITTED_INVENTORY_READ = "committed_inventory_read"
    COMMITTED_ACTION_AGENT_READ = "committed_action_agent_read"
    COMMITTED_EXISTENCE_READ = "committed_existence_read"
    COMMITTED_TEMPORAL_CONTACT_READ = "committed_temporal_contact_read"
    COMMITTED_TYPED_TRANSITION = "committed_typed_transition"
    COMMITTED_COMPARE_READ = "committed_compare_read"


@dataclass(frozen=True)
class QueryEphemeralBindings:
    """Turn-scoped binding snapshot for QUERY evaluation only.

    This payload is non-persistent by contract and must not be committed to PEF.
    """

    turn: int
    discourse_bindings: dict[str, str]


@dataclass(frozen=True)
class StateNativeRequest:
    """Inputs for one delegation attempt (data only)."""

    user_text: str
    pef: PEFState
    turn_act: TurnAct
    detected_span: Span
    #: Clarification replay applied blocked claims already; regex possession mutations
    #: must not reinterpret reconstructed ``user_text``.
    binding_resumed: bool = False
    #: Optional spaCy ``Language`` for possession-frame builders (Lens wires SpacyBackend).
    possession_nlp: Any | None = None
    #: When True, ``evaluate_possession_transfer_mutations`` applies only **HIGH** frames;
    #: LOW proposals behave like abstention on this route (see ``LensConfig``).
    require_high_confidence_possession_transfer: bool = False
    #: Non-persistent turn-scoped query bindings, when available.
    query_ephemeral_bindings: QueryEphemeralBindings | None = None


@dataclass(frozen=True)
class StateNativeDelegationResult:
    """Result of attempting a state-native delegation for one user turn."""

    handled: bool
    outcome: StateNativeOutcome | None = None
    user_visible_text: str = ""
    clarify_context: dict[str, Any] | None = None
    stop_reason_code: str | None = None
    solver_family: StateNativeSolverFamily | None = None
    source_status: str | None = None  # "admitted_uncontaminated" or "unverified"
    #: Explicit epistemic state from the adjudication layer.  Governance routing
    #: is derived from this when present; the ternary outcome field is secondary.
    epistemic_result: EpistemicResult | None = None
    #: Optional present-bound temporal evaluator payload, when this result originates
    #: from temporal contact adjudication.
    temporal_eval_result: PresentBoundTemporalEvalResult | None = None

    def __post_init__(self) -> None:
        if self.handled:
            if self.outcome is None:
                raise ValueError("handled=True requires outcome")
            if not (self.user_visible_text or "").strip():
                raise ValueError("handled=True requires non-empty user_visible_text")
            if self.solver_family is None:
                raise ValueError("handled=True requires solver_family")


__all__ = [
    "PossessionMutationAction",
    "PossessionMutationConfidence",
    "PossessionMutationDeterministicParse",
    "PossessionMutationFrame",
    "PossessionMutationParseSource",
    "PossessionMutationVoice",
    "StateNativeDelegationResult",
    "StateNativeOutcome",
    "QueryEphemeralBindings",
    "StateNativeRequest",
    "StateNativeSolverFamily",
]
