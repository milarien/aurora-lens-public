"""Integration tests for deterministic inferential_gap generation."""

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


def _seed_inferential_gap_relationship(lens: Lens) -> None:
    e = Entity.create("advisory_claim", turn=1)
    lens.pef.entities[e.id] = e
    lens.pef.relationships.append(
        Relationship(
            subject_id=e.id,
            relation="IS",
            object_entity_id=None,
            object_literal="recommended",
            span=Span.PRESENT,
            source_turn=1,
            evidence="evacuation advisory",
            relation_metadata={
                "inference": {
                    "premises": ["inspection_report", "sensor_reading"],
                    "conclusion": "evacuation_route.closure_status",
                    "bridge_ref": "domain.warrant.v1",
                    "bridge_status": "missing",
                    "bridge_mode": "requires_declared_bridge",
                    "policy_ref": "support.entailment.v1",
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
    _seed_inferential_gap_relationship(lens)
    return lens, adapter


class TestInferentialGapGenerationIntegration:
    @pytest.mark.asyncio
    async def test_inferential_gap_generated_record_blocks_decision_turn_sync(self):
        lens, adapter = _build_lens()
        calls_before = adapter._call_count
        r = await lens.process("Which option should we choose?")
        assert r.action != InterventionAction.PASS
        assert adapter._call_count == calls_before
        generated = [
            u
            for u in lens.pef.open_epistemic_uncertainties
            if u.get("kind") == "inferential_gap"
        ]
        assert generated
        assert str(generated[0].get("id", "")).startswith("generated_inferential_gap:")
        assert "does not establish a justified recommendation" in (r.response or "").lower()

    @pytest.mark.asyncio
    async def test_inferential_gap_generated_record_blocks_decision_turn_stream(self):
        lens, adapter = _build_lens()
        calls_before = adapter._call_count
        events: dict[str, list] = {}
        async for kind, payload in lens.process_stream("Which option should we choose?"):
            events.setdefault(kind, []).append(payload)
        assert "epistemic_uncertainty_gate" in events
        gate_result = events["epistemic_uncertainty_gate"][0]
        assert gate_result.action != InterventionAction.PASS
        assert adapter._call_count == calls_before
        generated = [
            u
            for u in lens.pef.open_epistemic_uncertainties
            if u.get("kind") == "inferential_gap"
        ]
        assert generated

    @pytest.mark.asyncio
    async def test_inferential_gap_sync_stream_generated_record_parity(self):
        sync_lens, _ = _build_lens()
        await sync_lens.process("Which option should we choose?")
        sync_generated = [
            u
            for u in sync_lens.pef.open_epistemic_uncertainties
            if u.get("kind") == "inferential_gap"
        ]
        assert sync_generated

        stream_lens, _ = _build_lens()
        async for _kind, _payload in stream_lens.process_stream("Which option should we choose?"):
            pass
        stream_generated = [
            u
            for u in stream_lens.pef.open_epistemic_uncertainties
            if u.get("kind") == "inferential_gap"
        ]
        assert stream_generated

        s1 = sync_generated[0]
        s2 = stream_generated[0]
        assert s1.get("id") == s2.get("id")
        assert s1.get("kind") == s2.get("kind")
        assert s1.get("status") == s2.get("status")
        assert s1.get("bears_on") == s2.get("bears_on")
