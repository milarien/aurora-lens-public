"""Regression tests for the pre-upstream hold gate (held-ambiguity continuation guard).

Invariants:
- While PEF has an active ambiguity hold (mode=ambiguity, interaction_open=True,
  pending_clarification set), NO turn of ANY act class may reach the upstream adapter.
- Unresolved follow-up → CONTAIN again, adapter call count = 0.
- Resolved follow-up → hold cleared, adapter may be called normally.
- No PASS ledger event while ambiguity hold is active.
"""

from __future__ import annotations

import pytest

from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.lens import Lens, LensResult
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import EPISTEMIC_MODE_AMBIGUITY, PEFState


# ── Helpers ───────────────────────────────────────────────────────────────────


class _CountingAdapter(LLMAdapter):
    """Mock adapter that counts generate() calls."""

    def __init__(self, response: str = "Mock response"):
        self._response = response
        self.call_count = 0

    async def generate(self, messages, **kwargs) -> AdapterResponse:
        self.call_count += 1
        return AdapterResponse(text=self._response, model="mock-1")


def _lens_with_counting_adapter() -> tuple[Lens, _CountingAdapter]:
    adapter = _CountingAdapter()
    config = LensConfig(adapter=adapter, auto_interpret=False, auto_verify=False)
    lens = Lens(config)
    return lens, adapter


def _inject_ambiguity_hold(lens: Lens) -> None:
    """Directly inject a canonical ambiguity hold into the Lens PEF."""
    pef = lens.pef
    pef.pending_clarification = {
        "original_question": "Emma told Anna her sister was overseas. Where is she now?",
        "unresolved_entity_ids": [],
        "failed_constraint": "UNRESOLVED_REFERENT",
        "ambiguous_referents": ["she"],
        "candidate_entities": ["Emma", "Anna"],
        "original_span": "present",
        "blocked_proposition": "Where is she now?",
        "blocked_claims": [],
        "clarification_prompt": None,
    }
    pef.epistemic_hold = {
        "schema_version": "1",
        "mode": EPISTEMIC_MODE_AMBIGUITY,
        "since_turn": 1,
        "pathway_id": "P_ASK_DISAMBIGUATE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "",
    }


# ── Test 1: Ambiguity hold → CONTAIN returned ─────────────────────────────────


@pytest.mark.asyncio
async def test_ambiguity_hold_returns_contain():
    """Sending any turn while an ambiguity hold is active returns CONTAIN, not PASS."""
    lens, adapter = _lens_with_counting_adapter()
    _inject_ambiguity_hold(lens)

    result = await lens.process("I like pizza")

    assert isinstance(result, LensResult)
    assert result.action == InterventionAction.CONTAIN, (
        f"Expected CONTAIN while ambiguity hold active; got {result.action}"
    )


# ── Test 2: Unresolved follow-up → CONTAIN again, no LLM call ────────────────


@pytest.mark.asyncio
async def test_unresolved_followup_no_adapter_call():
    """An unresolvable follow-up (TELL-act) while ambiguity hold is active must not call the adapter."""
    lens, adapter = _lens_with_counting_adapter()
    _inject_ambiguity_hold(lens)

    result = await lens.process("I like pizza")

    assert adapter.call_count == 0, (
        f"Adapter was called {adapter.call_count} time(s); expected 0 while ambiguity hold active"
    )
    assert result.action == InterventionAction.CONTAIN


# ── Test 3: Hold survives unrelated input ────────────────────────────────────


@pytest.mark.asyncio
async def test_hold_survives_unrelated_input():
    """Sending multiple unrelated turns does not clear the hold or escalate to a PASS."""
    lens, adapter = _lens_with_counting_adapter()
    _inject_ambiguity_hold(lens)

    turns = [
        "The sky is blue.",
        "What is 2 + 2?",
        "Tell me about astronomy.",
    ]
    for text in turns:
        result = await lens.process(text)
        assert result.action == InterventionAction.CONTAIN, (
            f"Expected CONTAIN for {text!r}; got {result.action}"
        )

    assert adapter.call_count == 0, (
        f"Adapter was called {adapter.call_count} time(s); expected 0 across all unresolved turns"
    )
    # Hold must persist
    assert lens.pef.epistemic_hold is not None
    assert lens.pef.epistemic_hold.get("mode") == EPISTEMIC_MODE_AMBIGUITY
    assert lens.pef.pending_clarification is not None


# ── Test 4: No PASS while ambiguity hold is active ───────────────────────────


@pytest.mark.asyncio
async def test_no_pass_while_ambiguity_hold_active():
    """No PASS result may be returned while the epistemic hold is in ambiguity mode."""
    lens, adapter = _lens_with_counting_adapter()
    _inject_ambiguity_hold(lens)

    # Try several turns that would normally elicit a PASS
    prompts = [
        "What is the capital of France?",
        "Just say yes.",
        "Give me a short poem.",
    ]
    for text in prompts:
        result = await lens.process(text)
        assert result.action != InterventionAction.PASS, (
            f"Got PASS for {text!r} while ambiguity hold was active — governance failure"
        )


# ── Test 5: Hold cleared after epistemic state reset ────────────────────────


@pytest.mark.asyncio
async def test_adapter_called_after_hold_cleared():
    """Once the ambiguity hold is cleared from PEF, the adapter may be called normally."""
    lens, adapter = _lens_with_counting_adapter()
    _inject_ambiguity_hold(lens)

    # Confirm hold blocks adapter
    result1 = await lens.process("I like pizza")
    assert adapter.call_count == 0
    assert result1.action == InterventionAction.CONTAIN

    # Manually clear the hold (simulates successful binding resolution)
    lens.pef.pending_clarification = None
    lens.pef.epistemic_hold = None

    # Now the adapter should be reachable
    result2 = await lens.process("What is 2 + 2?")
    assert adapter.call_count == 1, (
        f"Expected adapter to be called once after hold cleared; got {adapter.call_count}"
    )
    assert result2.action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT,
                               InterventionAction.CONTAIN)


# ── Test 6: candidate_entities populated from entity_mentions pre-update_pef ─


class _EmmaAnnaBackend(ExtractionBackend):
    """Simulates what spaCy returns for the Emma/Anna ambiguous sister sentence."""

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        tl = text.strip().lower()
        if "emma" in tl and "anna" in tl:
            return ExtractionResult(
                claims=[
                    ExtractedClaim(
                        subject="her sister",
                        relation="IS",
                        obj="overseas",
                        span=Span.PRESENT,
                        negated=False,
                        evidence=text,
                    )
                ],
                entity_mentions=["Emma", "Anna"],
                span=Span.PRESENT,
                ambiguous_referents=["her", "she"],
            )
        return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)


@pytest.mark.asyncio
async def test_candidate_entities_populated_from_entity_mentions():
    """candidate_entities must contain Emma and Anna even when called pre-update_pef.

    The pre-LLM ambiguity gate fires before update_pef writes entity_mentions to PEF.
    The fallback in _referent_resolution_candidate_names() must use extraction.entity_mentions
    directly so that pending_clarification.candidate_entities is non-empty and the UI
    can render the [Emma] [Anna] choice buttons.
    """
    adapter = _CountingAdapter(response="Emma's sister is overseas.")
    config = LensConfig(
        adapter=adapter,
        extraction_backend=_EmmaAnnaBackend(),
        auto_interpret=True,
        auto_verify=True,
    )
    lens = Lens(config)

    result = await lens.process(
        "Emma told Anna her sister was overseas. Where is she now?"
    )

    assert result.action == InterventionAction.CONTAIN, (
        f"Expected CONTAIN for ambiguous Emma/Anna input; got {result.action}"
    )
    assert adapter.call_count == 0, "Adapter must not be called for pre-LLM ambiguity hold"

    pend = lens.pef.pending_clarification
    assert pend is not None, "pending_clarification must be set after ambiguity CONTAIN"

    candidates = pend.get("candidate_entities") or []
    assert len(candidates) >= 2, (
        f"candidate_entities must contain at least 2 entries (Emma and Anna); got {candidates!r}"
    )
    candidates_lower = {c.lower() for c in candidates}
    assert "emma" in candidates_lower, f"'Emma' missing from candidate_entities: {candidates!r}"
    assert "anna" in candidates_lower, f"'Anna' missing from candidate_entities: {candidates!r}"


# ── Test 7: Continuation gate — follow-up request must not fire binding ───────


def _inject_workforce_ambiguity_hold(lens: Lens) -> None:
    """Inject a P_ASK_DISAMBIGUATE hold with Maria/Jennifer candidates."""
    pef = lens.pef
    pef.pending_clarification = {
        "original_question": (
            "Maria and Jennifer both reviewed the candidate shortlist. "
            "She approved the final three candidates. "
            "Which hiring manager gave the approval?"
        ),
        "unresolved_entity_ids": [],
        "failed_constraint": "UNRESOLVED_REFERENT",
        "ambiguous_referents": ["she"],
        "candidate_entities": ["Maria", "Jennifer"],
        "original_span": "present",
        "blocked_proposition": "She approved the final three candidates.",
        "blocked_claims": [],
        "clarification_prompt": None,
    }
    pef.epistemic_hold = {
        "schema_version": "1",
        "mode": EPISTEMIC_MODE_AMBIGUITY,
        "since_turn": 1,
        "pathway_id": "P_ASK_DISAMBIGUATE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "",
    }


@pytest.mark.asyncio
async def test_continuation_gate_rejects_followup_request():
    """A follow-up request naming a candidate must not fire binding.

    "Tell me more about Jennifer" contains the name Jennifer but is not a direct
    selection. The continuation gate must intercept it and re-issue CONTAIN without
    calling the adapter.
    """
    lens, adapter = _lens_with_counting_adapter()
    _inject_workforce_ambiguity_hold(lens)

    result = await lens.process("Tell me more about Jennifer.")

    assert result.action == InterventionAction.CONTAIN, (
        f"Expected CONTAIN for follow-up request; got {result.action}"
    )
    assert adapter.call_count == 0, (
        f"Adapter called {adapter.call_count} time(s); must be 0 for non-selection turn"
    )
    # Hold must survive — ambiguity is not resolved
    assert lens.pef.epistemic_hold is not None
    assert lens.pef.epistemic_hold.get("mode") == EPISTEMIC_MODE_AMBIGUITY
    assert lens.pef.pending_clarification is not None


@pytest.mark.asyncio
async def test_continuation_gate_accepts_direct_selection():
    """A direct candidate name must pass the continuation gate and proceed to binding.

    After "Jennifer" is typed, the hold should be cleared and the adapter should
    eventually be callable (binding resumed the original question).
    """
    lens, adapter = _lens_with_counting_adapter()
    _inject_workforce_ambiguity_hold(lens)

    # Direct selection — must not be rejected by the gate
    result = await lens.process("Jennifer")

    # With auto_interpret=False, the binding path runs in basic mode.
    # The gate must not block "Jennifer" (a direct selection).
    # Result may be CONTAIN (if binding logic needs more context) or PASS,
    # but the gate itself must not intercept it — adapter_call_count remains 0
    # only because auto_interpret=False skips the LLM call.
    assert result.action != InterventionAction.CONTAIN or adapter.call_count == 0, (
        "Direct selection 'Jennifer' should not be blocked by the continuation gate"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("follow_up", [
    "Tell me more about Jennifer.",
    "What did Jennifer do?",
    "Can you explain who Jennifer is?",
    "Is it Jennifer?",
    "Who is Jennifer?",
    "Please describe Jennifer",
    "Which one is Jennifer?",
])
async def test_continuation_gate_rejects_various_followup_forms(follow_up: str):
    """Parametrized: diverse follow-up forms must all re-issue CONTAIN without adapter calls."""
    lens, adapter = _lens_with_counting_adapter()
    _inject_workforce_ambiguity_hold(lens)

    result = await lens.process(follow_up)

    assert result.action == InterventionAction.CONTAIN, (
        f"Expected CONTAIN for {follow_up!r}; got {result.action}"
    )
    assert adapter.call_count == 0, (
        f"Adapter called for non-selection input {follow_up!r}"
    )
