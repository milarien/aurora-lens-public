"""Types for the universal pre-model state router (:mod:`aurora_lens.pre_model_state`).

Used for **every** operator corridor (general, medical, legal, finance): the router
runs before blocked-act, extraction, and the LLM. ``StreamKind`` identifies which
adapter family produced a streaming shape for SSE emission, not the user's domain.

Operator ``request_domain`` (and related signals) are orthogonal to whether routing runs;
they are authority hints that may affect which adapter can lawfully match a
state-bearing utterance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from aurora_lens.interpret.turn_act import TurnAct

if TYPE_CHECKING:
    from aurora_lens.lens import LensResult


@dataclass(frozen=True)
class PreModelContext:
    """Inputs for pre-model state routing for the current turn (all corridors).

    Session state is :class:`~aurora_lens.pef.state.PEFState` on the active :class:`~aurora_lens.lens.Lens`.
    Corridor and policy hints live on the lens / session context, not here; they are
    not prerequisites for invoking the router.
    """

    user_input: str
    history_user_input: str
    turn: int
    binding_resumed: bool
    turn_act: TurnAct
    for_stream: bool


StreamKind = Literal["state_native", "closed_world"]


@dataclass
class PreModelDispatchResult:
    """Result of :func:`pre_model_state_dispatch`."""

    handled: bool
    result: "LensResult | None" = None
    stream_kind: StreamKind | None = None


NOT_HANDLED = PreModelDispatchResult(handled=False)