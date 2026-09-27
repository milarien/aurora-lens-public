"""Evaluator-facing result type for present-bound temporal reasoning (Phase 4).

Used by state-native temporal evaluators and mapped into Lens governance outcomes.
:class:`TemporalGovernanceCue` is consumed in ``lens.py`` for temporal-contact mapping.

Law: ``docs/pef_temporality_present_bound_ontology.md`` § **Temporal Outcome Contract**.

This module does **not** import :class:`~aurora_lens.state_native_engine.epistemic.EpistemicResult`.
``TEMPORAL_UNKNOWN`` is a :class:`TemporalOutcomeContract` member, not an epistemic alias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aurora_lens.state_native_engine.temporal_outcome_contract import TemporalOutcomeContract


class TemporalGovernanceCue(str, Enum):
    """Stable cue consumed by Lens temporal mapping (wired in lens.py)."""

    ASK_TEMPORAL_SCOPE = "ask_temporal_scope"
    GOVERNED_NON_ANSWER = "governed_non_answer"
    PRESENT_COMMITTED_ANSWER = "present_committed_answer"


@dataclass
class PresentBoundTemporalEvalResult:
    """Structured return for present-bound temporal evaluators; mapped in Lens (commit bbd79da)."""

    outcome_kind: TemporalOutcomeContract
    governance_cue: TemporalGovernanceCue
    rationale: str
    temporal_anchor_surface: str | None = None
    temporal_span_label: str | None = None
    source_commitment_id: str | None = None
    evidence_payload: dict[str, Any] = field(default_factory=dict)
