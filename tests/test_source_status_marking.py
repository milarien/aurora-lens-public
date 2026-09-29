"""Regressions for explicit source-status marking in state-native answers."""

from __future__ import annotations

import pytest

from aurora_lens.lens import Lens
from aurora_lens.config import LensConfig
from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.pef.state import PEFState
from aurora_lens.verify.flags import FlagType


class _MockAdapter(LLMAdapter):
    """Mock adapter that records calls."""

    def __init__(self):
        self.calls = 0

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        self.calls += 1
        return AdapterResponse(text="UPSTREAM_SHOULD_NOT_RUN", model="mock")


@pytest.fixture
def mock_adapter():
    return _MockAdapter()


@pytest.mark.asyncio
async def test_valid_medical_chain_gets_admitted_uncontaminated_status(mock_adapter):
    """Valid admitted prescription/reduction chain should mark source as admitted_uncontaminated.

    This verifies that queries over an admitted (unrefused) chain of relationships
    receive explicit "admitted_uncontaminated" source status.
    """
    lens = Lens(
        LensConfig(
            adapter=mock_adapter,
            enable_state_native_delegation=True,
        ),
    )

    # Turn 1: Prescription (admitted)
    r1 = await lens.process("Dr Silva prescribed Rina 20mg of Amlodipine daily.")
    assert r1.response == "Recorded state transition."
    assert mock_adapter.calls == 0

    # Turn 2: Dose reduction (admitted, depends on turn 1)
    r2 = await lens.process("Dr Silva halved the dosage.")
    assert r2.response == "Recorded state transition."
    assert mock_adapter.calls == 0

    # Turn 3: Query about current dosage (should be answered and marked admitted_uncontaminated)
    r3 = await lens.process("What dosage is Rina currently receiving?")
    assert "10mg" in r3.response.lower()
    assert mock_adapter.calls == 0


def test_bare_handled_true_does_not_downgrade_held_refusal(mock_adapter):
    """Bare state_native_handled=True never downgrades HELD_REFUSAL classification.

    This is the core conservative requirement: without explicit source status,
    refusal holds remain globally governing.
    """
    from aurora_lens.pef.audit_linkage import classify_pef_turn_start, PefTurnClassification
    from aurora_lens.pef.state import EPISTEMIC_HOLD_SCHEMA_VERSION, EPISTEMIC_MODE_REFUSAL

    pef = PEFState()
    pef.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_REFUSAL,
        "since_turn": 2,
        "pathway_id": "P_REFUSE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "cid-test",
    }

    # Without source status, stays HELD_REFUSAL even with handled=True
    assert (
        classify_pef_turn_start(pef, state_native_source_status=None)
        == PefTurnClassification.HELD_REFUSAL
    )

    # With unverified status, stays HELD_REFUSAL
    assert (
        classify_pef_turn_start(pef, state_native_source_status="unverified")
        == PefTurnClassification.HELD_REFUSAL
    )

    # Only explicit "admitted_uncontaminated" downgrades
    assert (
        classify_pef_turn_start(pef, state_native_source_status="admitted_uncontaminated")
        == PefTurnClassification.HELD_STATE
    )


@pytest.mark.asyncio
async def test_audit_classification_only_changes_with_explicit_source_status(mock_adapter):
    """Audit classification remains HELD_REFUSAL unless source_status is explicit.

    This verifies the conservative default: responses marked handled=True do not
    automatically get better audit classification without explicit proof.
    """
    lens = Lens(
        LensConfig(
            adapter=mock_adapter,
            enable_state_native_delegation=True,
        ),
    )

    # Create a scenario with an admitted transition
    await lens.process("Dr Silva prescribed Rina 20mg of Amlodipine daily.")

    assert mock_adapter.calls == 0
