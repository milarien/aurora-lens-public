"""Tests for durable unresolved-referent registry in PEF."""

from __future__ import annotations

import pytest

from aurora_lens.config import LensConfig
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens, _clear_epistemic_hold_ambiguity
from aurora_lens.pef.state import PEFState
from aurora_lens.pef.unresolved_referents import (
    STATUS_OPEN,
    STATUS_RESOLVED,
    open_entries,
    open_registry_tokens_in_text,
    register_unresolved_referents,
    resolve_unresolved_referents,
    registry_entries_from_wire,
    registry_entries_to_wire,
)
from tests.test_lens import MockAdapter


class TestUnresolvedReferentRegistryModule:
    def test_register_and_persist_open(self):
        pef = PEFState(session_id="t")
        register_unresolved_referents(
            pef,
            turn=1,
            utterance="Who does their refer to?",
            tokens=["their"],
            candidate_entities=["Operator", "Contractor"],
            blocked_proposition="their certification had expired",
        )
        assert len(open_entries(pef)) == 1
        assert open_entries(pef)[0].token == "their"
        assert open_entries(pef)[0].status == STATUS_OPEN

    def test_resolve_clears_open_status_only_explicitly(self):
        pef = PEFState(session_id="t")
        register_unresolved_referents(
            pef,
            turn=1,
            utterance="ambiguous",
            tokens=["their"],
            candidate_entities=["A", "B"],
        )
        resolve_unresolved_referents(
            pef, tokens=["their"], resolved_entity="Contractor", turn=2,
        )
        assert open_entries(pef) == []
        resolved = [e for e in pef.unresolved_referent_registry if e.status == STATUS_RESOLVED]
        assert len(resolved) == 1
        assert resolved[0].resolved_entity == "Contractor"

    def test_open_tokens_in_text(self):
        pef = PEFState(session_id="t")
        register_unresolved_referents(
            pef,
            turn=1,
            utterance="x",
            tokens=["their"],
            candidate_entities=["A", "B"],
        )
        assert open_registry_tokens_in_text(pef, "Who does their refer to?") == ["their"]
        assert open_registry_tokens_in_text(pef, "Unrelated question.") == []

    def test_wire_roundtrip(self):
        pef = PEFState(session_id="t")
        register_unresolved_referents(
            pef,
            turn=3,
            utterance="q",
            tokens=["her"],
            candidate_entities=["Emma", "Anna"],
        )
        wire = registry_entries_to_wire(pef.unresolved_referent_registry)
        restored = registry_entries_from_wire(wire)
        assert len(restored) == 1
        assert restored[0].token == "her"
        assert restored[0].introduced_turn == 3

    def test_clear_epistemic_hold_does_not_clear_registry(self):
        pef = PEFState(session_id="t")
        register_unresolved_referents(
            pef,
            turn=1,
            utterance="q",
            tokens=["their"],
            candidate_entities=["A", "B"],
        )
        pef.epistemic_hold = {"mode": "ambiguity", "interaction_open": True}
        _clear_epistemic_hold_ambiguity(pef)
        assert len(open_entries(pef)) == 1


class TestUnresolvedReferentRegistryLensIntegration:
    @pytest.mark.asyncio
    async def test_registry_survives_unrelated_turn_and_blocks_later_attribution(self):
        turn1_q = (
            "The operator informed the contractor that their certification had expired "
            "before the work commenced. Both parties hold certifications. "
            "No further evidence is available. Who does 'their' refer to?"
        )
        backend = SpacyBackend(model="en_core_web_sm")
        adapter = MockAdapter(
            responses=[
                "The word 'their' refers to the contractor.",
                "Certifications are held by both parties.",
                "The word 'their' refers to the contractor.",
            ]
        )
        lens = Lens(LensConfig(adapter=adapter, extraction_backend=backend))

        r1 = await lens.process(turn1_q)
        assert r1.action != InterventionAction.PASS
        assert len(open_entries(lens.pef)) >= 1

        await lens.process("Summarize what we know about certifications.")
        assert len(open_entries(lens.pef)) >= 1, "unrelated turn must not clear registry"

        calls_before = adapter._call_count
        r3 = await lens.process("Who does 'their' refer to in the scenario?")
        assert r3.action != InterventionAction.PASS
        assert "refers to the contractor" not in (r3.response or "").lower()
        assert adapter._call_count == calls_before, (
            "registry must block LLM attribution without resolution"
        )

    @pytest.mark.asyncio
    async def test_explicit_binding_clears_registry(self):
        turn1_q = (
            "Emma told Anna her sister was overseas. "
            "Who does 'her' refer to?"
        )
        backend = SpacyBackend(model="en_core_web_sm")
        adapter = MockAdapter(responses=["Emma"])
        lens = Lens(LensConfig(adapter=adapter, extraction_backend=backend))

        r1 = await lens.process(turn1_q)
        assert r1.action != InterventionAction.PASS
        assert open_entries(lens.pef)

        await lens.process("Emma")
        assert open_entries(lens.pef) == [], "explicit bind must resolve durable registry"


_TURN1_OPERATOR_CONTRACTOR = (
    "The operator informed the contractor that their certification had expired "
    "before the work commenced. Both parties hold certifications."
)


async def _lens_with_operator_contractor_registry():
    backend = SpacyBackend(model="en_core_web_sm")
    adapter = MockAdapter(responses=["unused model output"])
    lens = Lens(LensConfig(adapter=adapter, extraction_backend=backend))
    await lens.process(_TURN1_OPERATOR_CONTRACTOR)
    assert open_entries(lens.pef), "turn 1 must register open ambiguity"
    return lens, adapter


class TestOpenRegistrySessionGovernance:
    @pytest.mark.asyncio
    async def test_open_registry_blocks_consequence_without_pronoun(self):
        lens, adapter = await _lens_with_operator_contractor_registry()
        calls_before = adapter._call_count

        await lens.process(
            "Under the governing legislation, the party with the expired certification "
            "must be suspended immediately."
        )
        assert open_entries(lens.pef)

        r3 = await lens.process("Is suspension admissible?")
        assert r3.action != InterventionAction.PASS
        assert adapter._call_count == calls_before
        assert "decision blocked" in (r3.response or "").lower()
        assert "contractor" in (r3.response or "").lower() or "operator" in (r3.response or "").lower()

    @pytest.mark.asyncio
    async def test_open_registry_blocks_candidate_specific_suspension(self):
        lens, adapter = await _lens_with_operator_contractor_registry()
        calls_before = adapter._call_count

        r = await lens.process("Should the contractor be suspended?")
        assert r.action != InterventionAction.PASS
        assert adapter._call_count == calls_before
        response_lower = (r.response or "").lower()
        assert (
            "decision blocked" in response_lower
            or "does not establish a justified recommendation" in response_lower
        )

    @pytest.mark.asyncio
    async def test_open_registry_blocks_responsible_party_question(self):
        lens, adapter = await _lens_with_operator_contractor_registry()
        calls_before = adapter._call_count

        r = await lens.process("Determine the responsible party.")
        assert r.action != InterventionAction.PASS
        assert adapter._call_count == calls_before
        assert "decision blocked" in (r.response or "").lower()

    @pytest.mark.asyncio
    async def test_open_registry_allows_meta_question(self):
        lens, adapter = await _lens_with_operator_contractor_registry()
        calls_before = adapter._call_count

        r = await lens.process("What information do you need?")
        assert r.action != InterventionAction.PASS
        assert adapter._call_count == calls_before
        response_lower = (r.response or "").lower()
        assert "allowed continuation" in response_lower or "decision blocked" in response_lower
        assert "contractor" in response_lower or "operator" in response_lower

    @pytest.mark.asyncio
    async def test_open_registry_allows_unrelated_reset_or_new_topic(self):
        lens, adapter = await _lens_with_operator_contractor_registry()
        calls_before = adapter._call_count

        adapter._responses.append("Penguins are flightless birds that swim well.")
        r = await lens.process("Tell me a joke about penguins.")
        assert adapter._call_count == calls_before + 1
        assert "penguin" in (r.response or "").lower() or r.action == InterventionAction.PASS

    @pytest.mark.asyncio
    async def test_generated_identity_or_referent_unresolved_blocks_decision_turn_sync(self):
        lens, adapter = await _lens_with_operator_contractor_registry()
        lens.pef.pending_clarification = None
        calls_before = adapter._call_count

        r = await lens.process("Which option should we choose?")
        assert r.action != InterventionAction.PASS
        assert adapter._call_count == calls_before
        assert "does not establish a justified recommendation" in (r.response or "").lower()
        generated = [
            u
            for u in lens.pef.open_epistemic_uncertainties
            if u.get("kind") == "identity_or_referent_unresolved"
        ]
        assert generated, "sync path should generate identity_or_referent_unresolved"

    @pytest.mark.asyncio
    async def test_generated_identity_or_referent_unresolved_blocks_decision_turn_stream(self):
        lens, adapter = await _lens_with_operator_contractor_registry()
        lens.pef.pending_clarification = None
        calls_before = adapter._call_count

        events: dict[str, list] = {}
        async for kind, payload in lens.process_stream("Which option should we choose?"):
            events.setdefault(kind, []).append(payload)

        assert "epistemic_uncertainty_gate" in events
        gate_result = events["epistemic_uncertainty_gate"][0]
        assert gate_result.action != InterventionAction.PASS
        assert adapter._call_count == calls_before
        assert "does not establish a justified recommendation" in (
            gate_result.response or ""
        ).lower()
        generated = [
            u
            for u in lens.pef.open_epistemic_uncertainties
            if u.get("kind") == "identity_or_referent_unresolved"
        ]
        assert generated, "stream path should generate identity_or_referent_unresolved"

    @pytest.mark.asyncio
    async def test_sync_and_stream_generate_same_identity_or_referent_unresolved_record(self):
        sync_lens, _ = await _lens_with_operator_contractor_registry()
        sync_lens.pef.pending_clarification = None
        await sync_lens.process("Which option should we choose?")
        sync_generated = [
            u
            for u in sync_lens.pef.open_epistemic_uncertainties
            if u.get("kind") == "identity_or_referent_unresolved"
        ]
        assert sync_generated

        stream_lens, _ = await _lens_with_operator_contractor_registry()
        stream_lens.pef.pending_clarification = None
        async for _kind, _payload in stream_lens.process_stream("Which option should we choose?"):
            pass
        stream_generated = [
            u
            for u in stream_lens.pef.open_epistemic_uncertainties
            if u.get("kind") == "identity_or_referent_unresolved"
        ]
        assert stream_generated

        sync_rec = sync_generated[0]
        stream_rec = stream_generated[0]
        assert sync_rec.get("id") == stream_rec.get("id")
        assert sync_rec.get("kind") == stream_rec.get("kind")
        assert sync_rec.get("status") == stream_rec.get("status")
        assert sync_rec.get("bears_on") == stream_rec.get("bears_on")


_CROSS_TURN_SETUP = (
    "A contractor submitted a safety inspection report to the site operator.\n"
    "The contractor held a maintenance certification.\n"
    "The site operator held an operational certification."
)
_CROSS_TURN_REPORT = "The report was signed after their certification had expired."
_OPERATOR_INFORMED = (
    "The operator informed the contractor that their certification had expired "
    "before the work commenced. Both parties hold certifications."
)


class TestCrossTurnPossessiveAmbiguity:
    @pytest.mark.asyncio
    async def test_setup_then_report_registers_their_in_registry(self):
        backend = SpacyBackend(model="en_core_web_sm")
        adapter = MockAdapter(responses=["unused"])
        lens = Lens(LensConfig(adapter=adapter, extraction_backend=backend))

        await lens.process(_CROSS_TURN_SETUP)
        r_report = await lens.process(_CROSS_TURN_REPORT)
        assert r_report.action != InterventionAction.PASS
        assert open_entries(lens.pef)
        assert any(e.token.lower() == "their" for e in open_entries(lens.pef))

    @pytest.mark.asyncio
    async def test_full_scenario_single_turn_registers_their(self):
        backend = SpacyBackend(model="en_core_web_sm")
        adapter = MockAdapter(responses=["unused"])
        lens = Lens(LensConfig(adapter=adapter, extraction_backend=backend))
        turn1 = (
            f"{_CROSS_TURN_SETUP}\n\n{_CROSS_TURN_REPORT}\n\n"
            "Please record the scenario without resolving that ambiguity."
        )
        r1 = await lens.process(turn1)
        assert r1.action != InterventionAction.PASS
        assert open_entries(lens.pef)


class TestSeedHistoryRegistryParity:
    @pytest.mark.asyncio
    async def test_seed_history_syncs_unresolved_registry_from_prior_turn(self):
        backend = SpacyBackend(model="en_core_web_sm")
        adapter = MockAdapter(responses=["unused"])
        lens = Lens(
            LensConfig(
                adapter=adapter,
                extraction_backend=backend,
                auto_interpret=True,
            )
        )
        history = [
            {"role": "user", "content": _OPERATOR_INFORMED},
            {"role": "assistant", "content": "Recorded."},
        ]
        await lens.seed_history(history)
        assert open_entries(lens.pef), "seed_history must register open ambiguity"

        calls_before = adapter._call_count
        r2 = await lens.process(
            "Under the governing legislation, the party with the expired certification "
            "must be suspended immediately."
        )
        assert r2.action != InterventionAction.PASS
        assert adapter._call_count == calls_before


class TestDemonstrativeNPResolution:
    """Demonstrative 'this/that + noun' must resolve silently when exactly one
    named antecedent exists — not trigger UNRESOLVED_REFERENT."""

    @pytest.mark.asyncio
    async def test_this_reactor_resolves_to_unique_named_antecedent(self):
        """'this reactor' where only 'Reactor 3' is named must PASS, not CONTAIN."""
        adapter = MockAdapter(responses=["Recorded."])
        lens = Lens(
            LensConfig(
                adapter=adapter,
                extraction_backend=SpacyBackend(model="en_core_web_sm"),
                auto_interpret=True,
                auto_verify=False,
            )
        )
        scenario = (
            "A chemical plant sensor network has flagged anomalous readings in Reactor 3. "
            "Historical record: in the last five years three similar anomaly events were "
            "recorded in this reactor. One was a genuine thermal event."
        )
        r1 = await lens.process(scenario)
        assert r1.action != InterventionAction.CONTAIN, (
            f"'this reactor' must not CONTAIN when 'Reactor 3' is the unique named antecedent. "
            f"action={r1.action} flags={[(f.flag_type.name, f.entity_name) for f in (r1.flags or [])]}"
        )

    @pytest.mark.asyncio
    async def test_this_reactor_ambiguous_when_two_reactors_named(self):
        """'this reactor' with both Reactor 3 and Reactor 7 named must remain ambiguous."""
        from aurora_lens.interpret.spacy_backend import SpacyBackend, _demonstrative_uniquely_resolved_in_doc
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(
            "Reactor 3 and Reactor 7 both showed anomalies. "
            "An event was recorded in this reactor."
        )
        head = next(t for t in doc if t.text == "reactor" and t.dep_ == "det" or
                    (t.lemma_ == "reactor" and t.dep_ in ("pobj", "dobj", "nsubj")))
        # Find the 'this reactor' head token
        this_head = None
        for tok in doc:
            if tok.text.lower() == "this" and tok.dep_ == "det" and tok.head.lemma_ == "reactor":
                this_head = tok.head
                break
        if this_head is None:
            pytest.skip("spaCy did not parse 'this reactor' as expected")
        result = _demonstrative_uniquely_resolved_in_doc(this_head, doc)
        assert result is False, "Two named reactors must not resolve uniquely"

    @pytest.mark.asyncio
    async def test_operator_contractor_ambiguity_unaffected(self):
        """Possessive 'their certification' ambiguity between operator/contractor must be preserved."""
        from tests.test_unresolved_referent_registry import _OPERATOR_INFORMED
        adapter = MockAdapter(responses=["Recorded."])
        lens = Lens(
            LensConfig(
                adapter=adapter,
                extraction_backend=SpacyBackend(model="en_core_web_sm"),
                auto_interpret=True,
                auto_verify=False,
            )
        )
        r1 = await lens.process(_OPERATOR_INFORMED)
        assert r1.action != InterventionAction.PASS, (
            "operator/contractor ambiguity must still CONTAIN — demonstrative fix must not weaken it"
        )
