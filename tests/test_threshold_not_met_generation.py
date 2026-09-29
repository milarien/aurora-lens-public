"""Integration tests for deterministic threshold_not_met generation."""

from __future__ import annotations

import pytest

from aurora_lens.config import LensConfig
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import Relationship
from tests.test_lens import MockAdapter


def _seed_threshold_not_met_relationship(lens: Lens) -> None:
    e = Entity.create("risk_assessment", turn=1)
    lens.pef.entities[e.id] = e
    lens.pef.relationships.append(
        Relationship(
            subject_id=e.id,
            relation="IS",
            object_entity_id=None,
            object_literal="assessment_ready",
            span=Span.PRESENT,
            source_turn=1,
            evidence="risk review",
            relation_metadata={
                "threshold": {
                    "required_strength": 0.8,
                    "observed_strength": 0.52,
                    "consequence_grade": "high",
                    "threshold_ref": "policy.threshold.v1",
                    "comparison_mode": "gte",
                    "policy_ref": "decision.policy.v3",
                }
            },
        )
    )


def _build_lens() -> tuple[Lens, MockAdapter]:
    adapter = MockAdapter(responses=["unused"])
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=SpacyBackend(model="en_core_web_sm"),
        )
    )
    _seed_threshold_not_met_relationship(lens)
    return lens, adapter


class TestThresholdNotMetGenerationIntegration:
    @pytest.mark.asyncio
    async def test_threshold_generated_record_blocks_decision_turn_sync(self):
        lens, adapter = _build_lens()
        calls_before = adapter._call_count
        r = await lens.process("Which option should we choose?")
        assert r.action != InterventionAction.PASS
        assert adapter._call_count == calls_before
        generated = [
            u
            for u in lens.pef.open_epistemic_uncertainties
            if u.get("kind") == "threshold_not_met"
        ]
        assert generated
        assert "does not establish a justified recommendation" in (r.response or "").lower()
