"""Pre-model PEF/state dispatch — universal jurisdiction layer before the LLM.

**All** requests pass through this router first. Ordered adapters include
state-native committed-state handling (inventory/transfer/typed transitions) and
a separate bounded-puzzle plug-in. The router itself is not domain-specialized.

Operator ``request_domain`` and similar authority signals are **policy and governance
hints** on the session; they do **not** enable or disable pre-model dispatch. Only
when no adapter handles the turn does control pass to blocked-act policy, extraction,
and the upstream model.
"""

from aurora_lens.pre_model_state.dispatch import (
    default_pre_model_adapters,
    pre_model_state_dispatch,
)
from aurora_lens.pre_model_state.types import (
    NOT_HANDLED,
    PreModelContext,
    PreModelDispatchResult,
    StreamKind,
)

__all__ = (
    "NOT_HANDLED",
    "PreModelContext",
    "PreModelDispatchResult",
    "StreamKind",
    "default_pre_model_adapters",
    "pre_model_state_dispatch",
)
