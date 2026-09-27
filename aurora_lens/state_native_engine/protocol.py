"""Protocol for pluggable state-native engines."""

from __future__ import annotations

from typing import Protocol

from aurora_lens.state_native_engine.contracts import (
    StateNativeDelegationResult,
    StateNativeRequest,
)


class StateNativeEngine(Protocol):
    def evaluate(self, req: StateNativeRequest) -> StateNativeDelegationResult:
        """Return handled state-native result, or not_handled for Lane 2."""
