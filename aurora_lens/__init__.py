"""aurora-lens: Governance substrate for LLMs — PEF world-state, claim verification, forensic audit."""

from .lens import Lens, LensResult
from .config import LensConfig
from .adapters.base import LLMAdapter, AdapterResponse
from .govern.decision import InterventionAction, GovernanceDecision
from .verify.flags import Flag, FlagType
from .pef.state import PEFState, Relationship
from .pef.span import Span

__version__ = "3.0.1"
__project__ = "Aurora-Lens"
__author__ = "Margaret Stokes"
__license__ = "Proprietary"
__copyright__ = "Copyright (C) Margaret Stokes"

__all__ = [
    # Core pipeline
    "Lens",
    "LensResult",
    "LensConfig",
    # Adapter interface (transport layer — implementations are optional extras)
    "LLMAdapter",
    "AdapterResponse",
    # Governance
    "InterventionAction",
    "GovernanceDecision",
    # Verification
    "Flag",
    "FlagType",
    # World state
    "PEFState",
    "Relationship",
    "Span",
    # Version
    "__version__",
    "__project__",
    "__author__",
    "__license__",
    "__copyright__",
]
