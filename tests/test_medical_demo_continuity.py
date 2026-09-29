"""Medical typed transitions stored on ordinary PEF relationships."""

from __future__ import annotations

import pytest

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter
from aurora_lens.config import LensConfig
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.lens import Lens


class _CountingAdapter(LLMAdapter):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        self.calls += 1
        return AdapterResponse(text="UPSTREAM_SHOULD_NOT_RUN", model="mock-med")


def _build_lens(adapter: _CountingAdapter, session_id: str) -> Lens:
    return Lens(
        LensConfig(
            adapter=adapter,
            audit_log_path=None,
            enable_state_native_delegation=True,
            auto_verify=True,
            auto_interpret=True,
        ),
        session_id=session_id,
    )


@pytest.mark.asyncio
async def test_typed_transitions_generic_two_entity_medication_sets_pre_llm() -> None:
    ada = _CountingAdapter()

    lens_a = _build_lens(ada, "medical-pef-generic-a")
    assert (await lens_a.process("Dr Rivera prescribed Lina 24mg of Captopril daily.")).action == InterventionAction.PASS
    assert (await lens_a.process("Nurse Omar withheld the evening dose pending blood pressure review.")).action == InterventionAction.PASS
    assert (await lens_a.process("Dr Rivera halved the dosage.")).action == InterventionAction.PASS

    q1 = await lens_a.process("What dosage is Lina currently receiving?")
    assert "12mg" in q1.response.lower()
    assert "captopril" in q1.response.lower()

    q2 = await lens_a.process("Who authorized the current dosage?")
    assert "dr rivera" in q2.response.lower()

    q3 = await lens_a.process("Was the evening dose administered?")
    assert "withheld pending blood pressure review" in q3.response.lower()

    # Withhold must not erase active dosage.
    q4 = await lens_a.process("What dosage is Lina currently receiving?")
    assert "12mg" in q4.response.lower()

    lens_b = _build_lens(ada, "medical-pef-generic-b")
    assert (await lens_b.process("Dr Aoki prescribed Mateo 30mg of Sertraline daily.")).action == InterventionAction.PASS
    assert (await lens_b.process("Dr Aoki changed the dosage to 18mg.")).action == InterventionAction.PASS

    q5 = await lens_b.process("What dosage is Mateo currently receiving?")
    assert "18mg" in q5.response.lower()
    assert "sertraline" in q5.response.lower()

    q6 = await lens_b.process("Who authorized the current dosage?")
    assert "dr aoki" in q6.response.lower()

    assert ada.calls == 0


@pytest.mark.asyncio
async def test_provenance_uses_latest_relevant_dose_transition() -> None:
    ada = _CountingAdapter()
    lens = _build_lens(ada, "medical-pef-provenance")

    await lens.process("Dr Imani prescribed Nora 20mg of Metoprolol daily.")
    q1 = await lens.process("Who authorized the current dosage?")
    assert "dr imani" in q1.response.lower()

    await lens.process("Dr Kline reduced the dosage to 10mg.")
    q2 = await lens.process("Who authorized the current dosage?")
    assert "dr kline" in q2.response.lower()

    q3 = await lens.process("What dosage is Nora currently receiving?")
    assert "10mg" in q3.response.lower()
    assert ada.calls == 0
