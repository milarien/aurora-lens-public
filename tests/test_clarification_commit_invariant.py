"""Regression tests for the clarification-resolution commit invariant.

Invariant:
    When ``pending_clarification.blocked_claims`` is **non-empty**, replay must write the
    serialized claims to PEF (or surface ``CLARIFICATION_RESOLUTION_COMMIT_FAILED``).
    When ``blocked_claims`` is **empty**, replay is intentionally a no-op; successful
    candidate binding still clears the ambiguity hold after discourse bookkeeping ---
    absence of serialized claims must not force a false commit-failure loop.

Tests:
    1  – Setup turn produces CONTAIN with correctly structured pending_clarification.
    2  – "Alice." binding commits IS(Alice, …) to PEF, clears hold, resumes upstream once.
    3  – After binding, PEF contains the resolved structured relationship.
    4  – "Carol." binding commits IS(Carol, …) symmetrically.
    5  – "I don't know." leaves hold active, adapter not called, no PASS.
    6  – Empty ``blocked_claims``: replay is a no-op by design; candidate binding still clears
        the hold and resumes (adapter may run) --- finance/spaCy paths often omit serialized
        blocked claims while still requiring possessive resolution.
"""

from __future__ import annotations

import pytest

from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.lens import Lens, LensResult
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import EPISTEMIC_MODE_AMBIGUITY, PEFState


# ── Helpers ───────────────────────────────────────────────────────────────────

_SETUP_INPUT = (
    "Alice and Carol are both named in the dispute. "
    "She agreed to mediation. Which party agreed?"
)


class _CountingAdapter(LLMAdapter):
    def __init__(self, response: str = "Alice agreed to mediation."):
        self._response = response
        self.call_count = 0

    async def generate(self, messages, **kwargs) -> AdapterResponse:
        self.call_count += 1
        return AdapterResponse(text=self._response, model="mock-1")


class _DisputeBackend(ExtractionBackend):
    """Produces a structured blocked claim for the ambiguous dispute setup turn."""

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        tl = text.strip().lower()
        if "alice" in tl and "carol" in tl and "dispute" in tl:
            return ExtractionResult(
                claims=[
                    ExtractedClaim(
                        subject="she",
                        relation="IS",
                        obj="agreed to mediation",
                        span=Span.PRESENT,
                        negated=False,
                        evidence=text,
                    )
                ],
                entity_mentions=["Alice", "Carol"],
                span=Span.PRESENT,
                ambiguous_referents=["she"],
            )
        return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)


def _make_lens(
    adapter: _CountingAdapter | None = None,
    *,
    auto_verify: bool = False,
) -> tuple[Lens, _CountingAdapter]:
    adapter = adapter or _CountingAdapter()
    config = LensConfig(
        adapter=adapter,
        extraction_backend=_DisputeBackend(),
        auto_interpret=True,
        auto_verify=auto_verify,
    )
    return Lens(config), adapter


def _inject_pending(lens: Lens, *, empty_blocked_claims: bool = False) -> None:
    """Inject a canonical UNRESOLVED_REFERENT pending state into the lens."""
    pef = lens.pef
    blocked_claims: list[dict] = (
        []
        if empty_blocked_claims
        else [
            {
                "subject": "she",
                "relation": "IS",
                "obj": "agreed to mediation",
                "span": "present",
                "negated": False,
                "evidence": "She agreed to mediation.",
            }
        ]
    )
    pef.pending_clarification = {
        "original_question": _SETUP_INPUT,
        "unresolved_entity_ids": [],
        "failed_constraint": "UNRESOLVED_REFERENT",
        "ambiguous_referents": ["she"],
        "candidate_entities": ["Alice", "Carol"],
        "original_span": "present",
        "blocked_proposition": "She agreed to mediation.",
        "blocked_claims": blocked_claims,
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
    pef.get_or_create_entity("Alice")
    pef.get_or_create_entity("Carol")


# ── Test 1: Setup turn → CONTAIN with correct pending state ──────────────────


@pytest.mark.asyncio
async def test_setup_turn_contain_with_pending_state():
    """Turn 1 returns CONTAIN and populates pending_clarification correctly.

    No mediation fact must be committed to PEF before the referent is resolved.
    """
    lens, adapter = _make_lens(auto_verify=True)

    result = await lens.process(_SETUP_INPUT)

    assert result.action == InterventionAction.CONTAIN, (
        f"Expected CONTAIN for ambiguous dispute input; got {result.action}"
    )
    assert adapter.call_count == 0, "Adapter must not be called on pre-LLM CONTAIN"

    pend = lens.pef.pending_clarification
    assert pend is not None, "pending_clarification must be set after CONTAIN"
    assert pend.get("failed_constraint") == "UNRESOLVED_REFERENT"
    assert pend.get("blocked_proposition") is not None, "blocked_proposition must be set"
    assert "she" in [str(r).lower() for r in (pend.get("ambiguous_referents") or [])]

    cands = [c.lower() for c in (pend.get("candidate_entities") or [])]
    assert "alice" in cands and "carol" in cands, (
        f"candidate_entities must contain Alice and Carol; got {pend.get('candidate_entities')!r}"
    )

    hold = lens.pef.epistemic_hold
    assert hold is not None, "epistemic_hold must be set"
    assert hold.get("commitment_closed") is True

    # No mediation fact in PEF before resolution.
    alice = lens.pef.find_entity_by_name("Alice")
    if alice is not None:
        rels = lens.pef.get_relationships_for_subject(alice.id)
        mediation_committed = any(
            "mediation" in str(r.object_literal or "").lower() for r in rels
        )
        assert not mediation_committed, (
            "Mediation fact must not be committed to PEF before referent is resolved"
        )


# ── Test 2: "Alice." binding commits to PEF and clears hold ──────────────────


@pytest.mark.asyncio
async def test_alice_binding_commits_and_clears_hold():
    """Binding 'Alice.' writes the resolved relationship to PEF and clears the hold.

    Multi-sentence UNRESOLVED_REFERENT holds use upstream_original completion and
    resume the held original_question through the adapter exactly once after bind.
    """
    lens, adapter = _make_lens(_CountingAdapter(response="Alice agreed to mediation."))
    _inject_pending(lens)

    result = await lens.process("Alice.")

    assert adapter.call_count == 1, (
        "upstream_original hold must resume the held original_question through the adapter once"
    )
    assert "Alice agreed to mediation" in result.response
    assert lens.pef.pending_clarification is None, (
        "pending_clarification must be cleared after successful commit"
    )
    assert lens.pef.epistemic_hold is None, (
        "epistemic_hold must be cleared after successful commit"
    )

    alice = lens.pef.find_entity_by_name("Alice")
    assert alice is not None, "Alice entity must exist in PEF after binding"
    rels = lens.pef.get_relationships_for_subject(alice.id)
    assert len(rels) > 0, (
        "Alice must have at least one relationship in PEF after commit"
    )


# ── Test 3: PEF contains the resolved structured relationship after binding ───


@pytest.mark.asyncio
async def test_pef_contains_resolved_relationship_after_binding():
    """The resolved IS(Alice, 'agreed to mediation') claim must be in PEF.

    Structural verification: the committed state holds the binding so subsequent
    turns can query from PEF rather than relying on LLM reconstruction.
    """
    lens, adapter = _make_lens(_CountingAdapter())
    _inject_pending(lens)

    await lens.process("Alice.")

    alice = lens.pef.find_entity_by_name("Alice")
    assert alice is not None
    rels = lens.pef.get_relationships_for_subject(alice.id)
    mediation_rel = next(
        (r for r in rels if "mediation" in str(r.object_literal or "").lower()),
        None,
    )
    assert mediation_rel is not None, (
        "IS(Alice, 'agreed to mediation') must be present in PEF after successful binding"
    )
    assert mediation_rel.relation == "IS", (
        f"Relation must be IS; got {mediation_rel.relation!r}"
    )


# ── Test 4: "Carol." binding commits symmetrically ───────────────────────────


@pytest.mark.asyncio
async def test_carol_binding_commits_and_clears_hold():
    """Binding 'Carol.' writes the resolved relationship to PEF for Carol."""
    lens, adapter = _make_lens(_CountingAdapter(response="Carol agreed to mediation."))
    _inject_pending(lens)

    result = await lens.process("Carol.")

    assert adapter.call_count == 1, (
        "upstream_original hold must resume the held original_question through the adapter once"
    )
    assert "Carol agreed to mediation" in result.response
    assert lens.pef.pending_clarification is None, (
        "pending_clarification must be cleared after Carol binding"
    )
    assert lens.pef.epistemic_hold is None, (
        "epistemic_hold must be cleared after Carol binding"
    )

    carol = lens.pef.find_entity_by_name("Carol")
    assert carol is not None, "Carol entity must exist in PEF after binding"
    rels = lens.pef.get_relationships_for_subject(carol.id)
    assert len(rels) > 0, (
        "Carol must have at least one relationship in PEF after commit"
    )


# ── Test 5: "I don't know." leaves hold active, no PASS ─────────────────────


@pytest.mark.asyncio
async def test_i_dont_know_leaves_hold_active():
    """Non-binding clarification 'I don't know.' must not clear the hold or call the adapter."""
    lens, adapter = _make_lens()
    _inject_pending(lens)

    result = await lens.process("I don't know.")

    assert result.action != InterventionAction.PASS, (
        "Must not return PASS when hold is active and no binding was resolved"
    )
    assert adapter.call_count == 0, (
        f"Adapter must not be called while hold is active; got {adapter.call_count}"
    )
    assert lens.pef.pending_clarification is not None, (
        "pending_clarification must remain after non-binding turn"
    )
    assert lens.pef.epistemic_hold is not None, (
        "epistemic_hold must remain after non-binding turn"
    )
    assert lens.pef.epistemic_hold.get("mode") == EPISTEMIC_MODE_AMBIGUITY


# ── Test 6: Empty blocked_claims → replay no-op; binding still clears hold ───


@pytest.mark.asyncio
async def test_empty_blocked_claims_resume_clears_hold_when_candidate_matches():
    """No serialized blocked claims → replay writes nothing (by design).

    Candidate binding must still clear the ambiguity hold and resume the pipeline;
    ``CLARIFICATION_RESOLUTION_COMMIT_FAILED`` applies only when replay was expected
    to commit structured claims from a non-empty ``blocked_claims`` payload.
    """
    lens, adapter = _make_lens()
    _inject_pending(lens, empty_blocked_claims=True)

    result = await lens.process("Alice.")

    assert result.action == InterventionAction.PASS, (
        f"Expected PASS after binding with empty blocked_claims replay; got {result.action}"
    )
    assert adapter.call_count == 1, (
        "upstream_original hold must resume the held original_question through the adapter once"
    )
    assert "Alice agreed to mediation" in result.response
    assert lens.pef.pending_clarification is None
    assert lens.pef.epistemic_hold is None
