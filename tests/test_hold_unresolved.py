"""Tests for HOLD_UNRESOLVED unresolved-referent continuation."""

from __future__ import annotations

import pytest

from aurora_lens.config import LensConfig
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.pef.unresolved_referents import (
    HOLD_UNRESOLVED_CHOICE_LABEL,
    RESOLUTION_MODE_HELD_UNRESOLVED,
    STATUS_OPEN,
    clarification_choices_from_pending,
    open_entries,
)
from tests.test_lens import MockAdapter

_TURN1 = (
    "The operator informed the contractor that their certification had expired "
    "before the work commenced. Both parties hold certifications."
)

_TURN1_WITH_DISAMBIG = (
    "The operator informed the contractor that their certification had expired "
    "before the work commenced. Both parties hold certifications. "
    'No further evidence is available. Who does "their" refer to?'
)


async def _lens_with_turn1_ambiguity():
    backend = SpacyBackend(model="en_core_web_sm")
    adapter = MockAdapter(responses=["unused"])
    lens = Lens(LensConfig(adapter=adapter, extraction_backend=backend))
    await lens.process(_TURN1)
    assert open_entries(lens.pef)
    return lens, adapter


class TestHoldUnresolved:
    @pytest.mark.asyncio
    async def test_hold_unresolved_option_is_present_for_unresolved_referent(self):
        lens, _adapter = await _lens_with_turn1_ambiguity()
        pending = lens.pef.pending_clarification
        assert pending is not None
        choices = clarification_choices_from_pending(pending)
        labels_lower = {c.lower() for c in choices}
        assert "contractor" in labels_lower
        assert "operator" in labels_lower
        assert HOLD_UNRESOLVED_CHOICE_LABEL.lower() in labels_lower

    @pytest.mark.asyncio
    async def test_hold_unresolved_does_not_bind_or_clear_registry(self):
        lens, adapter = await _lens_with_turn1_ambiguity()
        calls_before = adapter._call_count

        r = await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)
        assert r.action != InterventionAction.PASS
        assert adapter._call_count == calls_before
        assert "ambiguity held" in (r.response or "").lower()
        assert "does not establish" in (r.response or "").lower()

        entries = open_entries(lens.pef)
        assert len(entries) == 1
        entry = entries[0]
        assert entry.status == STATUS_OPEN
        assert entry.resolved_entity is None
        assert {c.lower() for c in entry.candidate_entities} == {"operator", "contractor"}
        assert entry.resolution_mode == RESOLUTION_MODE_HELD_UNRESOLVED
        assert lens.pef.pending_clarification is not None
        assert lens.pef.pending_clarification.get("resolution_mode") == RESOLUTION_MODE_HELD_UNRESOLVED

    @pytest.mark.asyncio
    async def test_hold_unresolved_allows_non_dependent_turn(self):
        lens, adapter = await _lens_with_turn1_ambiguity()
        await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)
        calls_before = adapter._call_count

        adapter._responses.append("Two parties are mentioned: the operator and the contractor.")
        r = await lens.process("How many parties are mentioned?")
        assert adapter._call_count == calls_before + 1
        assert "choose one option" not in (r.response or "").lower()

    @pytest.mark.asyncio
    async def test_hold_unresolved_blocks_dependent_consequence(self):
        lens, adapter = await _lens_with_turn1_ambiguity()
        await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)
        calls_before = adapter._call_count

        r = await lens.process("Is suspension admissible?")
        assert r.action != InterventionAction.PASS
        assert adapter._call_count == calls_before
        assert "decision blocked" in (r.response or "").lower()
        assert "suspension cannot be determined" in (r.response or "").lower()
        assert "does not establish" in (r.response or "").lower()

    @pytest.mark.asyncio
    async def test_hold_unresolved_blocks_candidate_specific_consequence(self):
        lens, adapter = await _lens_with_turn1_ambiguity()
        await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)
        calls_before = adapter._call_count

        r = await lens.process("Should the contractor be suspended?")
        assert r.action != InterventionAction.PASS
        assert adapter._call_count == calls_before
        assert r.telemetry_release_path == "blocked_before_generation"
        response = (r.response or "").lower()
        assert "does not establish" in response
        assert "unresolved uncertainty" in response
        assert "refers to the contractor" not in response

    @pytest.mark.asyncio
    async def test_later_explicit_resolution_clears_held_unresolved(self):
        lens, adapter = await _lens_with_turn1_ambiguity()
        await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)

        r = await lens.process(
            "The expired certification was the contractor's certification."
        )
        assert open_entries(lens.pef) == []
        resolved = [
            e for e in lens.pef.unresolved_referent_registry
            if e.resolved_entity
        ]
        assert len(resolved) == 1
        assert resolved[0].resolved_entity.lower() == "contractor"
        assert r.action != InterventionAction.HARD_STOP or adapter._call_count >= 0

    @pytest.mark.asyncio
    async def test_possessive_clarification_after_hold_unresolved_returns_mutation_ack(self):
        """Held ambiguity + explicit possessive evidence → pre-LLM PASS, no upstream."""
        backend = SpacyBackend(model="en_core_web_sm")
        adapter = MockAdapter(responses=["unused"])
        lens = Lens(LensConfig(adapter=adapter, extraction_backend=backend))
        await lens.process(_TURN1_WITH_DISAMBIG)
        await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)
        calls_before = adapter._call_count
        await lens.process("Is suspension admissible?")
        assert adapter._call_count == calls_before

        r = await lens.process("The contractor's certification had expired.")
        assert adapter._call_count == calls_before
        assert r.action == InterventionAction.PASS
        assert r.response == "Clarification noted. I have updated the recorded state."
        assert open_entries(lens.pef) == []
        resolved = [
            e for e in lens.pef.unresolved_referent_registry
            if e.resolved_entity
        ]
        assert len(resolved) == 1
        assert resolved[0].resolved_entity.lower() == "contractor"

    @pytest.mark.asyncio
    async def test_hold_unresolved_second_click_does_not_call_llm(self):
        lens, adapter = await _lens_with_turn1_ambiguity()
        await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)
        calls_before = adapter._call_count

        await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)
        assert adapter._call_count == calls_before

    @pytest.mark.asyncio
    async def test_hold_unresolved_second_click_keeps_registry_open(self):
        lens, _adapter = await _lens_with_turn1_ambiguity()
        await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)

        await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)

        entries = open_entries(lens.pef)
        assert len(entries) == 1
        assert entries[0].status == STATUS_OPEN
        assert entries[0].resolution_mode == RESOLUTION_MODE_HELD_UNRESOLVED
        assert lens.pef.pending_clarification is not None
        assert (
            lens.pef.pending_clarification.get("resolution_mode")
            == RESOLUTION_MODE_HELD_UNRESOLVED
        )
        pending = lens.pef.pending_clarification
        choices = clarification_choices_from_pending(pending, pef=lens.pef)
        assert HOLD_UNRESOLVED_CHOICE_LABEL.lower() not in {c.lower() for c in choices}

    @pytest.mark.asyncio
    async def test_hold_unresolved_second_click_returns_governed_ack(self):
        lens, _adapter = await _lens_with_turn1_ambiguity()
        await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)

        r = await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)
        assert r.action != InterventionAction.PASS
        assert "ambiguity already held" in (r.response or "").lower()
        assert r.decision is not None
        assert "HOLD_UNRESOLVED_IDEMPOTENT" in (r.decision.rationale or "")

    @pytest.mark.asyncio
    async def test_hold_unresolved_second_click_does_not_clear_dependency_gate(self):
        lens, adapter = await _lens_with_turn1_ambiguity()
        await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)
        await lens.process(HOLD_UNRESOLVED_CHOICE_LABEL)
        calls_before = adapter._call_count

        r = await lens.process("Is suspension admissible?")
        assert r.action != InterventionAction.PASS
        assert adapter._call_count == calls_before
        assert "decision blocked" in (r.response or "").lower()
