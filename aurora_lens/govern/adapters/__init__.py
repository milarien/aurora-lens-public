"""Canonical Governor adapter layer.

Three adapters, three failure modes:
  StatusTranslator  — epistemic failures (wrong LensStatus computation)
  ContextResolver   — routing/configuration failures (wrong corridor)
  PolicyProjector   — compatibility/reporting failures (wrong output vocabulary)

Import the concrete classes from their modules; this package exposes them flat.
"""

from aurora_lens.govern.adapters.runtime_types import (
    RuntimeDecisionProjection,
    ContextResolutionProvenance,
)
from aurora_lens.govern.adapters.status_translator import StatusTranslator
from aurora_lens.govern.adapters.context_resolver import ContextResolver
from aurora_lens.govern.adapters.policy_projector import PolicyProjector

__all__ = [
    "RuntimeDecisionProjection",
    "ContextResolutionProvenance",
    "StatusTranslator",
    "ContextResolver",
    "PolicyProjector",
]
