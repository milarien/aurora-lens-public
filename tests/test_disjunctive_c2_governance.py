"""Focused tests: RAG C2 disjunctive harness maps to CONTAIN (ASK), not FORCE_REVISE."""

from __future__ import annotations

import pytest

from aurora_lens.context import auth_policy_var
from aurora_lens.govern.bridge import enforce
from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.pef.state import PEFState
from aurora_lens.verify.flags import Flag, FlagType


def _c2_flag() -> Flag:
    return Flag(
        flag_type=FlagType.DISJUNCTIVE_BRANCH_COLLAPSE,
        entity_name="disjunctive",
        claim="RAG C2 harness: disjunctive identity question requires containment, not PASS.",
        evidence="Harness",
        severity="warning",
        candidates=("Emma", "Lucy"),
    )


@pytest.mark.asyncio
async def test_disjunctive_harness_decide_contain_all_modes():
    """RAG C2: ambiguity harness must CONTAIN (not PASS / not FORCE_REVISE)."""
    for mode in ("open", "enterprise", "public"):
        bridge = CanonicalScannerGateBridge(mode=mode)
        decision = await bridge.decide([_c2_flag()], "Emma's sister moved.", PEFState())
        assert decision.action == InterventionAction.CONTAIN, mode


def test_disjunctive_contain_enforce_asks_candidates():
    """CONTAIN path uses ambiguity renderer with Emma/Lucy candidates."""
    decision = GovernanceDecision(
        action=InterventionAction.CONTAIN,
        flags=[_c2_flag()],
        rationale="test",
        pathway_id="P_ASK_DISAMBIGUATE",
        interaction_open=True,
        commitment_closed=True,
    )
    out = enforce(decision, "Emma's sister moved from Vancouver.")
    assert "emma" in out.lower() and "lucy" in out.lower(), out


@pytest.mark.asyncio
async def test_moderate_key_does_not_soften_rag_causal_non_admit():
    """RAG U1 non-admit must stay FORCE_REVISE under moderate key."""
    bridge = CanonicalScannerGateBridge(mode="open")
    token = auth_policy_var.set("moderate")
    try:
        decision = await bridge.decide(
            [
                Flag(
                    flag_type=FlagType.RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT,
                    entity_name="rag_harness",
                    claim="x",
                    evidence="y",
                    severity="warning",
                )
            ],
            "The context does not state why.",
            PEFState(),
        )
        assert decision.action == InterventionAction.FORCE_REVISE
    finally:
        auth_policy_var.reset(token)
