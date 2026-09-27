"""Plug-in for the bounded puzzle surface only (not transfer/inventory arithmetic).

Transfer, inventory, and quantity-from-PEF flows use :class:`PossessionStateAdapter`.
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


class ArithmeticClosedWorldAdapter(PreModelStateAdapter):
    """Jurisdiction for deterministic puzzle surfaces before the LLM (any corridor)."""

    def try_dispatch(self, lens: Any, ctx: PreModelContext) -> PreModelDispatchResult:
        if ctx.binding_resumed:
            return NOT_HANDLED
        try:
            r = lens._finish_turn_with_closed_world_puzzle_if_handled(
                ctx.history_user_input,
                ctx.turn_act,
                Span.PRESENT,
                ctx.turn,
                for_stream=ctx.for_stream,
            )
        except ForensicEnvelopeValidationError:
            return NOT_HANDLED
        if r is None:
            return NOT_HANDLED
        return PreModelDispatchResult(
            handled=True,
            result=r,
            stream_kind="closed_world",
        )
