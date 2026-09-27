"""Plug-in protocol for the universal :mod:`aurora_lens.pre_model_state` router."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from aurora_lens.pre_model_state.types import PreModelContext, PreModelDispatchResult

if TYPE_CHECKING:
    pass


class PreModelStateAdapter(Protocol):
    """One ordered probe in the shared pre-model chain (all operator corridors).

    Declines unless utterance shape, PEF, and relevant authority/policy gates match.
    """

    def try_dispatch(self, lens: Any, ctx: PreModelContext) -> PreModelDispatchResult:
        """Return ``NOT_HANDLED`` if this adapter does not assume jurisdiction."""
