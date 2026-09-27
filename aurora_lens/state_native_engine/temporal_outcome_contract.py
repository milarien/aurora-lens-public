"""Canonical vocabulary for the Temporal Outcome Contract (Phase 2).

Law: ``docs/pef_temporality_present_bound_ontology.md`` § **Temporal Outcome Contract**.
These names are stable for evaluator results, governance, and audit in later phases.

**Hinge**

- ``TEMPORAL_MISSING`` — failure to **locate** the governing temporal anchor.
- ``TEMPORAL_UNKNOWN`` — failure to determine the answer **within** an already located anchor.

Neither outcome is aliased to :class:`~aurora_lens.state_native_engine.epistemic.EpistemicResult.UNKNOWN`.
``TEMPORAL_UNKNOWN`` names a *temporal* deficit; ``EpistemicResult.UNKNOWN`` names relational snapshot
ignorance for non-temporal state-native paths. Do not conflate them in routing or storage.
"""

from __future__ import annotations

from enum import Enum
from typing import Final


class TemporalOutcomeContract(str, Enum):
    """Outcomes from the binding Temporal Outcome Contract table (1:1 by member name)."""

    TEMPORAL_MISSING = "TEMPORAL_MISSING"
    TEMPORAL_UNKNOWN = "TEMPORAL_UNKNOWN"
    TEMPORAL_AMBIGUOUS = "TEMPORAL_AMBIGUOUS"
    SUPERSEDED = "SUPERSEDED"
    FUTURE_PROJECTION = "FUTURE_PROJECTION"
    PAST_RECONSTRUCTION = "PAST_RECONSTRUCTION"


# Frozen set of wire values for membership checks (stable for tests and validators).
TEMPORAL_OUTCOME_CONTRACT_VALUES: Final[frozenset[str]] = frozenset(
    m.value for m in TemporalOutcomeContract
)
