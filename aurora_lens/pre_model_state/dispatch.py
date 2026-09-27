"""Generic pre-model state router: PEF/state classification before LLM, blocked-act, or extraction.

Runs on **every** turn for **every** corridor. The router is always invoked; it does
not consult ``request_domain`` to decide whether to run. Adapters run in order; each
may use PEF, utterance shape, and authority hints (including corridor domain) to decide
whether it assumes jurisdiction. Individual adapters may be domain-aware; this entry
point is not.
"""

from __future__ import annotations

from typing import Any, Sequence

from aurora_lens.pre_model_state.adapters import (
    ArithmeticClosedWorldAdapter,
    PossessionStateAdapter,
)
from aurora_lens.pre_model_state.protocol import PreModelStateAdapter
from aurora_lens.pre_model_state.types import (
    NOT_HANDLED,
    PreModelContext,
    PreModelDispatchResult,
)


def default_pre_model_adapters() -> tuple[PreModelStateAdapter, ...]:
    """Ordered jurisdictional probes (first handler wins).

    This chain is installed once; it runs regardless of ``request_domain``. Domain and
    operator hints may affect whether a domain-shaped adapter *matches*, not whether
    routing occurs. Order: possession/state-native (inventory, transfer, quantity, typed
    transition updates/queries over committed PEF), then the bounded-puzzle adapter
    (``ArithmeticClosedWorldAdapter``).
    """
    return (
        PossessionStateAdapter(),
        ArithmeticClosedWorldAdapter(),
    )


def pre_model_state_dispatch(
    lens: Any,
    ctx: PreModelContext,
    *,
    adapters: Sequence[PreModelStateAdapter] | None = None,
) -> PreModelDispatchResult:
    """Ask whether PEF/state assumes jurisdiction for this turn before the LLM path.

    Applies across **general**, **medical**, **legal**, and **finance** contexts: if an
    adapter returns a governed :class:`~aurora_lens.lens.LensResult`, the upstream model
    is not invoked for that turn.

    ``request_domain`` is not what causes this function to run; it may inform adapters
    where corridor policy applies.

    If every adapter declines, returns ``NOT_HANDLED`` and the caller continues to
    blocked-act classification, extraction, and adapter generation unchanged.
    """
    chain = adapters if adapters is not None else default_pre_model_adapters()
    for adapter in chain:
        out = adapter.try_dispatch(lens, ctx)
        if out.handled and out.result is not None:
            return out
    return NOT_HANDLED
