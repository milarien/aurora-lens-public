"""Built-in pre-model state adapters (plug into the universal router in :mod:`aurora_lens.pre_model_state`)."""

from aurora_lens.pre_model_state.adapters.arithmetic_closed_world import (
    ArithmeticClosedWorldAdapter,
)
from aurora_lens.pre_model_state.adapters.possession_state import (
    PossessionStateAdapter,
)

__all__ = (
    "ArithmeticClosedWorldAdapter",
    "PossessionStateAdapter",
)
