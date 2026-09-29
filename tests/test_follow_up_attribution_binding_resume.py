"""Binding resume: state-native must see follow-up attribution tail after clarification."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aurora_lens.adapters.base import AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractionResult
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.pef.span import Span
from aurora_lens.state_native_engine.parse.query_surface import (
    extract_follow_up_attribution_tail,
    state_native_text_for_binding_resume,
)
from aurora_lens.verify.flags import FlagType
from tests.test_healthcare_scenarios import TestAdultDosingInstruction
from tests.test_lens import MockAdapter, RecordingMockAdapter, SequentialBackend


MEDICAL_Q = (
    "Emma and Dr. Patel are listed on the case. "
    "She ordered the medication change. "
    "Which clinician should I follow up with?"
)
WHERE_SHE_Q = "Emma told Anna her sister was overseas. Where is she now?"


class _RecordingAdapter:
    def __init__(self) -> None:
        self.generate = AsyncMock(
            return_value=AdapterResponse(text="MODEL_SHOULD_NOT_RUN", model="mock"),
        )


def test_extract_follow_up_attribution_tail_from_multisentence():
    tail = extract_follow_up_attribution_tail(MEDICAL_Q)
    assert tail == "Which clinician should I follow up with?"


def test_state_native_text_for_binding_resume_appends_tail_only():
    sn = state_native_text_for_binding_resume(
        "Dr. Patel ordered the medication change.",
        MEDICAL_Q,
    )
    assert sn == (
        "Dr. Patel ordered the medication change. "
        "Which clinician should I follow up with?"
    )


def test_state_native_text_for_binding_resume_no_double_append():
    q = "Which clinician should I follow up with?"
    assert state_native_text_for_binding_resume(q, q) == q


@pytest.mark.asyncio
async def test_medical_clinician_follow_up_state_native_on_binding_resume():
    """After she→Dr. Patel binding, state-native answers without LLM."""
    adapter = _RecordingAdapter()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=SpacyBackend(),
            enable_state_native_delegation=True,
            auto_verify=True,
            auto_interpret=True,
        ),
    )
    r1 = await lens.process(MEDICAL_Q)
    assert r1.action == InterventionAction.CONTAIN
    assert any(f.flag_type == FlagType.UNRESOLVED_REFERENT for f in (r1.flags or []))

    r2 = await lens.process("Dr. Patel")
    assert r2.action == InterventionAction.PASS
    assert "Dr. Patel" in r2.response
    assert "Follow up with" in r2.response
    assert r2.telemetry_release_path == "state_native_retrieval"
    adapter.generate.assert_not_called()
    assert lens.pef.discourse_referent_bindings.get("she") == "Dr. Patel"


@pytest.mark.asyncio
async def test_upstream_resume_sends_reconstructed_bound_payload():
    """upstream_original holds must send reconstructed bound text (no unresolved prompt preface)."""
    clarification_ext = ExtractionResult(
        claims=[], entity_mentions=["Dr. Patel"], span=Span.PRESENT,
    )
    recheck_ext = ExtractionResult(
        claims=[], entity_mentions=[], span=Span.PRESENT, ambiguous_referents=[],
    )
    backend = SequentialBackend([clarification_ext, recheck_ext])

    adapter = RecordingMockAdapter(responses=["Acknowledged."])
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            enable_state_native_delegation=False,
            auto_verify=False,
            auto_interpret=True,
        ),
    )
    lens.pef.get_or_create_entity("Emma", resolved=True)
    lens.pef.get_or_create_entity("Dr. Patel", resolved=True)
    lens.pef.pending_clarification = {
        "original_question": MEDICAL_Q,
        "completion_strategy": "upstream_original",
        "unresolved_entity_ids": [],
        "failed_constraint": "UNRESOLVED_REFERENT",
        "ambiguous_referents": ["she"],
        "candidate_entities": ["Emma", "Dr. Patel"],
        "blocked_proposition": "She ordered the medication change.",
        "blocked_claims": [
            {
                "subject": "she",
                "relation": "ORDER",
                "obj": "medication change",
                "span": "past",
                "negated": False,
                "evidence": "She ordered the medication change.",
            }
        ],
    }

    await lens.process("Dr. Patel")

    assert adapter._call_count == 1
    sent = adapter.received_user_msgs[-1]
    assert "Dr. Patel ordered the medication change." in sent
    assert "She ordered the medication change." not in sent
    assert "[Session referents" not in sent


@pytest.mark.asyncio
async def test_where_is_she_ambiguity_regression_still_state_native():
    """Emma/Anna where-is-she path: blocked_proposition includes query; no LLM on bind."""
    adapter = MockAdapter(responses=["should not run"])
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=SpacyBackend(),
            enable_state_native_delegation=True,
            auto_verify=True,
            auto_interpret=True,
        ),
    )
    await lens.process(WHERE_SHE_Q)
    r2 = await lens.process("Emma")
    assert r2.action == InterventionAction.PASS
    assert adapter._call_count == 0
    assert "overseas" in r2.response.lower()


class TestMedicalAdviceStillHardStop(TestAdultDosingInstruction):
    """Regression: personalized dosing request remains HARD_STOP (unchanged policy)."""

    pass


class _NoLLMAdapter:
    def __init__(self) -> None:
        self.generate = AsyncMock(
            side_effect=AssertionError("LLM must not run for deterministic attribution resume"),
        )


@pytest.mark.asyncio
async def test_deterministic_attribution_hold_commits_locally_without_upstream():
    """Attribution-only holds (original turn equals surface ask) may commit pre-LLM."""
    adapter = _NoLLMAdapter()
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=SpacyBackend(),
            enable_state_native_delegation=False,
            auto_verify=True,
            auto_interpret=True,
        ),
    )
    surface = "Which party agreed?"
    lens.pef.get_or_create_entity("Alice")
    lens.pef.get_or_create_entity("Carol")
    lens.pef.pending_clarification = {
        "original_question": surface,
        "completion_strategy": "deterministic_attribution",
        "unresolved_entity_ids": [],
        "failed_constraint": "UNRESOLVED_REFERENT",
        "ambiguous_referents": ["she"],
        "candidate_entities": ["Alice", "Carol"],
        "original_span": "present",
        "blocked_proposition": "She agreed to mediation.",
        "blocked_claims": [
            {
                "subject": "she",
                "relation": "IS",
                "obj": "agreed to mediation",
                "span": "present",
                "negated": False,
                "evidence": "She agreed to mediation.",
            }
        ],
        "pending_task": {
            "task_type": "attribution_answer",
            "expected_answer_role": "subject",
            "attribution_kind": "actor",
            "target_claim": {
                "subject": "she",
                "relation": "IS",
                "obj": "agreed to mediation",
            },
            "surface_question": surface,
        },
    }
    lens.pef.epistemic_hold = {
        "schema_version": "1",
        "mode": "ambiguity",
        "since_turn": 1,
        "pathway_id": "P_ASK_DISAMBIGUATE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "",
    }

    result = await lens.process("Alice.")
    assert result.action == InterventionAction.PASS
    assert "Alice agreed to mediation" in result.response
    adapter.generate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup", "bind", "expected"),
    [
        (
            "Sarah and Emma both received customer queries. "
            "She sent an email about a billing discrepancy. "
            "Whose query should I address?",
            "Emma",
            "Address Emma's query.",
        ),
        (
            "Maria and Jennifer both reviewed the candidate shortlist. "
            "She approved the final three candidates. "
            "Which hiring manager gave the approval?",
            "Jennifer",
            "Jennifer gave the approval.",
        ),
        (
            "Alice and Dr. Park are both involved in the revision. "
            "She flagged a confidentiality concern. "
            "Who raised the concern?",
            "Alice",
            "Alice raised the concern.",
        ),
        (
            "Emma and Dr. Chen are both listed for Module 3 submissions. "
            "She sent an email about the results. "
            "Which contact should students use?",
            "Dr. Chen",
            "Students should contact Dr. Chen about their results.",
        ),
        (
            "Alice and Carol are both named in the dispute. "
            "She agreed to mediation. Which party agreed?",
            "Carol",
            "Carol agreed to mediation.",
        ),
        (
            "Nora and Priya both handled the audit notes. "
            "She escalated the missing invoice. "
            "Which analyst escalated it?",
            "Priya",
            "Priya escalated the missing invoice.",
        ),
    ],
)
async def test_attribution_binding_resume_spacy_end_to_end(setup, bind, expected):
    adapter = MockAdapter(responses=[expected])
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=SpacyBackend(),
            enable_state_native_delegation=False,
            auto_verify=False,
            auto_interpret=True,
        ),
    )
    r1 = await lens.process(setup)
    assert r1.action == InterventionAction.CONTAIN
    assert lens.pef.pending_clarification.get("completion_strategy") == "upstream_original"
    calls_after_ask = adapter._call_count
    if "Module" in setup:
        cands = set(lens.pef.pending_clarification.get("candidate_entities") or [])
        assert cands == {"Emma", "Dr. Chen"}, f"unexpected candidates: {cands!r}"
    r2 = await lens.process(bind)
    assert r2.action == InterventionAction.PASS
    assert expected in r2.response
    assert adapter._call_count == calls_after_ask + 1


@pytest.mark.asyncio
async def test_attribution_binding_resume_passes_with_auto_verify():
    """Regression: UNSUPPORTED_EVENT must not FORCE_REVISE when LLM rephrases a committed
    blocked claim with a synonym verb (e.g. 'gave the approval' instead of APPROVE).

    The verifier correctly recognises the LLM is restating a bound fact and suppresses
    both UNSUPPORTED_ATTRIBUTE and UNSUPPORTED_EVENT for the resolved entity.
    """
    llm_response = "Jennifer gave the approval."
    adapter = MockAdapter(responses=[llm_response])
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=SpacyBackend(),
            enable_state_native_delegation=False,
            auto_verify=True,
            auto_interpret=True,
        ),
    )
    setup = (
        "Maria and Jennifer both reviewed the candidate shortlist. "
        "She approved the final three candidates. "
        "Which hiring manager gave the approval?"
    )
    r1 = await lens.process(setup)
    assert r1.action == InterventionAction.CONTAIN

    r2 = await lens.process("Jennifer")
    assert r2.action == InterventionAction.PASS, (
        f"Expected PASS; got {r2.action}. "
        "Binding resume must not FORCE_REVISE when LLM uses a synonym verb for the committed blocked claim. "
        f"Flags: {[f.flag_type for f in (r2.flags or [])]}"
    )
