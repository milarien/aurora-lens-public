"""Identity tests for Temporal Outcome Contract vocabulary (Phase 2)."""

from __future__ import annotations

import pytest

from aurora_lens.state_native_engine.epistemic import EpistemicResult
from aurora_lens.state_native_engine.temporal_outcome_contract import (
    TEMPORAL_OUTCOME_CONTRACT_VALUES,
    TemporalOutcomeContract,
)


def test_temporal_outcome_contract_member_set() -> None:
    assert {m.name for m in TemporalOutcomeContract} == {
        "TEMPORAL_MISSING",
        "TEMPORAL_UNKNOWN",
        "TEMPORAL_AMBIGUOUS",
        "SUPERSEDED",
        "FUTURE_PROJECTION",
        "PAST_RECONSTRUCTION",
    }
    assert TEMPORAL_OUTCOME_CONTRACT_VALUES == {m.value for m in TemporalOutcomeContract}


def test_temporal_missing_never_epistemic_unknown() -> None:
    assert TemporalOutcomeContract.TEMPORAL_MISSING is not EpistemicResult.UNKNOWN
    assert TemporalOutcomeContract.TEMPORAL_MISSING.value != EpistemicResult.UNKNOWN.value
    assert TemporalOutcomeContract.TEMPORAL_MISSING.value.casefold() != "unknown"


@pytest.mark.parametrize("member", list(TemporalOutcomeContract))
def test_no_contract_value_equals_epistemic_unknown_string(member: TemporalOutcomeContract) -> None:
    """Contract wire values must not alias generic epistemic UNKNOWN ('unknown')."""
    assert member.value != EpistemicResult.UNKNOWN.value
    assert member.value.casefold() != EpistemicResult.UNKNOWN.value.casefold()
