"""Governed clarification resume vs meta-continuation routing (finance-facing regressions).

Proves unresolved-referent pending is consumed via Step 1.5 binding before blanket
QUERY/CLARIFY continuation fires; possessive-antecedent candidate hygiene drops fiscal
period tokens (e.g. Q4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter
from aurora_lens.config import LensConfig
from aurora_lens.context import domain_var
from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.interpret.turn_act import TurnAct
from aurora_lens.lens import (
    Lens,
    _matches_candidate,
    _pending_ambiguity_continuation_applies,
    _referent_resolution_candidate_names,
    _resolve_binding_entity,
)
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import EPISTEMIC_MODE_AMBIGUITY, PEFState


def test_pending_referent_meta_continuation_only_for_clarification_inquiry():
    """Substantive binding-shaped turns must not hit unresolved-referent continuation."""
    pending = {"failed_constraint": "UNRESOLVED_REFERENT"}
    assert not _pending_ambiguity_continuation_applies(
        turn_act=TurnAct.ASSERT,
        pending=pending,
        history_user_input="I mean APAC Sub.",
    )
    assert not _pending_ambiguity_continuation_applies(
        turn_act=TurnAct.QUERY,
        pending=pending,
        history_user_input="Was its margin worse than the prior quarter?",
    )


def test_pending_referent_continuation_true_for_meta_clarify_and_help_queries():
    pending = {"failed_constraint": "UNRESOLVED_REFERENT"}
    assert _pending_ambiguity_continuation_applies(
        turn_act=TurnAct.CLARIFY,
        pending=pending,
        history_user_input="What information do you still need?",
    )
    assert _pending_ambiguity_continuation_applies(
        turn_act=TurnAct.QUERY,
        pending=pending,
        history_user_input="Which subsidiary did you mean?",
    )


def test_resolve_binding_entity_matches_extractor_mention_parity_with_matches_candidate():
    """If ``_matches_candidate`` is True via entity_mentions, resolver must agree."""
    pending = {"candidate_entities": ["James"]}
    clar = ExtractionResult(
        claims=[],
        entity_mentions=["James"],
        span=Span.PRESENT,
    )
    text = "Sure."
    assert _matches_candidate(text, clar, pending)
    assert _resolve_binding_entity(text, clar, pending, PEFState()) == "James"


def test_replay_commit_failed_not_raised_when_pending_has_no_blocked_claims():
    """Empty ``blocked_claims`` means replay is intentionally a no-op."""
    from aurora_lens.lens import (
        _CommitReplayResult,
        _replay_verification_indicates_relation_commit_failed,
    )

    pef = PEFState()
    assert not _replay_verification_indicates_relation_commit_failed(
        _CommitReplayResult(),
        relation_count_before=len(pef.relationships),
        pef_after=pef,
        pending_blocked_claims=[],
    )


@pytest.mark.parametrize(
    ("ambiguous", "expect_q4"),
    [
        (["its"], False),
        (["his"], False),
        ([], True),
    ],
)
def test_referent_candidates_exclude_period_labels_for_possessive_ambiguity(
    ambiguous: list[str],
    expect_q4: bool,
):
    """Standalone quarter tokens are not possessive-antecedent candidates."""
    pef = PEFState()
    for nm in ("Q4", "APAC Sub", "EMEA Sub"):
        ent = Entity.create(nm, turn=1)
        ent.resolved = True
        pef.add_entity(ent)

    extraction = ExtractionResult(
        claims=[
            ExtractedClaim(
                subject="APAC Sub",
                relation="IS",
                obj="example",
                span=Span.PRESENT,
                negated=False,
                evidence="setup",
            )
        ],
        entity_mentions=["Q4", "APAC Sub", "EMEA Sub"],
        span=Span.PRESENT,
        ambiguous_referents=list(ambiguous),
    )

    names = _referent_resolution_candidate_names(
        extraction,
        pef,
        ambiguous_tokens=list(ambiguous),
    )
    lower = {n.lower() for n in names}
    if expect_q4:
        assert "q4" in lower
    else:
        assert "q4" not in lower
        assert "apac sub" in lower
        assert "emea sub" in lower


@pytest.mark.asyncio
async def test_vague_followup_still_contained_when_pending_referent():
    """Genuinely unresolved clarification must still CONTAIN (adapter unreachable)."""

    class _Adapter(LLMAdapter):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            return AdapterResponse(text="noop", model="x")

    adapter = _Adapter()
    lens = Lens(LensConfig(adapter=adapter, auto_interpret=False, auto_verify=False))
    lens.pef.pending_clarification = {
        "original_question": "Was its margin worse than the prior quarter?",
        "unresolved_entity_ids": [],
        "failed_constraint": "UNRESOLVED_REFERENT",
        "ambiguous_referents": ["its"],
        "candidate_entities": ["APAC Sub", "EMEA Sub"],
        "original_span": "present",
        "blocked_proposition": "Was its margin worse than the prior quarter?",
        "blocked_claims": [],
    }
    lens.pef.epistemic_hold = {
        "schema_version": "1",
        "mode": EPISTEMIC_MODE_AMBIGUITY,
        "since_turn": 1,
        "pathway_id": "P_ASK_DISAMBIGUATE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "",
    }

    result = await lens.process("Interesting weather today.")
    assert result.action == InterventionAction.CONTAIN
    assert adapter.calls == 0


@pytest.fixture(scope="module")
def spacy_backend_sm() -> SpacyBackend:
    return SpacyBackend(model="en_core_web_sm")


@pytest.mark.asyncio
async def test_finance_spacy_possessive_ambiguity_resume_end_to_end(
    tmp_path: Path,
    spacy_backend_sm: SpacyBackend,
) -> None:
    """Finance possessive ambiguity (live-shaped): bind clears hold; Q4 not a candidate."""

    class _CompareStub(LLMAdapter):
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            return AdapterResponse(text="APAC Sub Q4 gross margin was lower than Q3.", model="stub")

    audit = tmp_path / "finance_resume.jsonl"
    bridge = CanonicalScannerGateBridge(mode="enterprise", audit_path=str(audit))
    stub = _CompareStub()
    lens = Lens(
        LensConfig(
            adapter=stub,
            extraction_backend=spacy_backend_sm,
            governance_bridge=bridge,
            auto_interpret=True,
            auto_verify=False,
        )
    )

    tok = domain_var.set("finance")
    try:
        r1 = await lens.process("APAC Sub reported a gross margin of 34% in Q4.")
        assert r1.action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT)
        r2 = await lens.process("EMEA Sub reported a gross margin of 31% in Q4.")
        assert r2.action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT)

        r3 = await lens.process("Was its margin worse than the prior quarter?")
        assert r3.action == InterventionAction.CONTAIN
        pend = lens.pef.pending_clarification
        assert pend is not None
        cands_lower = [str(x).strip().lower() for x in (pend.get("candidate_entities") or [])]
        assert "q4" not in cands_lower, f"candidates unexpectedly included period token: {cands_lower}"

        clarify = (
            "I mean APAC Sub. APAC Sub gross margin in Q4 was 34%. "
            "APAC Sub gross margin in Q3 was 32%."
        )
        r4 = await lens.process(clarify)
    finally:
        domain_var.reset(tok)

    assert r4.action != InterventionAction.CONTAIN, (
        f"Expected resume after binding; got {r4.action} response={r4.response[:200]!r}"
    )
    assert lens.pef.pending_clarification is None
    # After binding, system either resumes via LLM or answers deterministically from
    # committed Q3/Q4 facts (turn_semantics deterministic_clarification_continuation path).
    # Both are valid governance outcomes.
    assert stub.calls >= 1 or r4.action == InterventionAction.PASS, (
        f"Expected LLM call or deterministic PASS; got action={r4.action} calls={stub.calls}"
    )
