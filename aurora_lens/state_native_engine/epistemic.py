"""Epistemic result layer for state-native query adjudication.

Sits between PEF evaluation and governance outcome selection.  Each evaluator
tags its StateNativeDelegationResult with the appropriate EpistemicResult so
governance routing is derived from explicit epistemic state, not inferred from
stop codes or text patterns.

Governance derivation (canonical):
  TRUE / FALSE / VALUE   → ANSWER  (InterventionAction.PASS)
  UNKNOWN                → STOP    (InterventionAction.HARD_STOP)
  AMBIGUOUS              → CLARIFY (InterventionAction.CONTAIN)
  CONTRADICTED           → FORCE_REVISE or HARD_STOP
"""

from __future__ import annotations

from enum import Enum


class EpistemicResult(str, Enum):
    TRUE = "true"                  # Committed relation/event matches the query exactly
    FALSE = "false"                # Committed absence or competing committed subject
    VALUE = "value"                # Committed value directly answers a who/what/where query
    UNKNOWN = "unknown"            # No committed data exists to evaluate the query
    AMBIGUOUS = "ambiguous"        # Unresolved referent or entity ambiguity
    CONTRADICTED = "contradicted"  # Committed state explicitly contradicts an asserted claim
