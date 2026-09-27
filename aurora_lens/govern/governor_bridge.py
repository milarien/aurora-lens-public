"""Backward-compatibility shim — re-exports from scanner_gate_bridge.

The AuroraGovernorBridge class was renamed to AuroraScannerGateBridge.
This module preserves the old import path for any code that has not been
updated yet.  New code should import from scanner_gate_bridge directly.
"""

from aurora_lens.govern.scanner_gate_bridge import (  # noqa: F401
    AuroraScannerGateBridge as AuroraGovernorBridge,
    _SCANNER_GATE_AVAILABLE as _GOVERNOR_AVAILABLE,
)

__all__ = ["AuroraGovernorBridge", "_GOVERNOR_AVAILABLE"]
