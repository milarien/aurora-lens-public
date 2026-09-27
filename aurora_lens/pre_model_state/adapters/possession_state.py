"""State-native adapter: committed PEF (possession, transfer, inventory, quantity, ``Who has``).

Same universal pre-model router for every corridor. Matches QUERY surfaces against
committed session state — including cross-turn transfer and counted-item arithmetic.
"""

from __future__ import annotations

from typing import Any

from aurora_lens.govern.forensic_append_guard import ForensicEnvelopeValidationError
from aurora_lens.pre_model_state.protocol import PreModelStateAdapter
from aurora_lens.pre_model_state.types import (
    NOT_HANDLED,
    PreModelContext,
    PreModelDispatchResult,
)
from aurora_lens.pef.span import Span


class PossessionStateAdapter(PreModelStateAdapter):
    """Jurisdiction when the state-native engine answers from committed session state."""

    def try_dispatch(self, lens: Any, ctx: PreModelContext) -> PreModelDispatchResult:
        if ctx.binding_resumed:
            # Resume path uses reconstructed question + resumed span after extraction.
            return NOT_HANDLED
        try:
            r = lens._finish_turn_with_state_native_if_handled(
                ctx.history_user_input,
                ctx.turn_act,
                Span.PRESENT,
                ctx.turn,
                for_stream=ctx.for_stream,
                binding_resumed=ctx.binding_resumed,
            )
        except ForensicEnvelopeValidationError:
            return NOT_HANDLED
        if r is None:
            return NOT_HANDLED
        return PreModelDispatchResult(
            handled=True,
            result=r,
            stream_kind="state_native",
        )
