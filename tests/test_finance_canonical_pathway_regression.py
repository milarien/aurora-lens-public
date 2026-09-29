"""Regression: CanonicalScannerGateBridge supplies typed pathway_id for finance admits.

Guards Option 1 from ``docs/issues/governor-pathway-malformed-fallback.md`` ---
missing / unknown pathway_id + weak domain hints must not drive ``enforce()`` Tier-4
malformed-decision fallback before PEF continuity can run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aurora_lens.adapters.base import AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.context import domain_var
from aurora_lens.govern.bridge import PATHWAY_IDS_WITH_TYPED_RENDERER, enforce
from aurora_lens.govern.canonical_bridge import (
    CanonicalScannerGateBridge,
    _ensure_decision_pathway_for_enforce,
)
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.pef.state import PEFState
from aurora_lens.verify.flags import Flag, FlagType


def _flag(ft: FlagType, *, severity: str = "warning") -> Flag:
    return Flag(
        flag_type=ft,
        entity_name="test",
        claim="test claim",
        evidence="test evidence",
        severity=severity,
    )


@pytest.fixture(scope="module")
def spacy_backend() -> SpacyBackend:
    return SpacyBackend(model="en_core_web_sm")


@pytest.mark.asyncio
async def test_finance_operator_hint_empty_verify_admits_non_null_typed_pathway(
    tmp_path: Path,
) -> None:
    """Finance route + ADMIT corridor must expose P_HANDOFF_SUMMARY (typed renderer)."""
    audit = tmp_path / "finance_pathway.jsonl"
    bridge = CanonicalScannerGateBridge(mode="enterprise", audit_path=str(audit))
    pef = PEFState()
    tok = domain_var.set("finance")
    try:
        decision = await bridge.decide([], "The budget is $5.2M.", pef)
    finally:
        domain_var.reset(tok)

    assert decision.pathway_id == "P_HANDOFF_SUMMARY"
    assert decision.pathway_id in PATHWAY_IDS_WITH_TYPED_RENDERER
    assert decision.action == InterventionAction.FORCE_REVISE

    rendered = enforce(decision, "Acknowledged.")
    assert decision.unexpected_unclassified_termination is False
    assert decision.fallback_reason is None
    assert "missing detail" not in rendered.lower()


@pytest.mark.asyncio
async def test_pathway_repair_restores_projection_when_cleared(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "repair.jsonl"
    bridge = CanonicalScannerGateBridge(mode="enterprise", audit_path=str(audit))
    tok = domain_var.set("finance")
    try:
        decision = await bridge.decide([], "draft", PEFState())
        projection = bridge.last_projection
    finally:
        domain_var.reset(tok)

    assert projection is not None
    decision.pathway_id = None
    _ensure_decision_pathway_for_enforce(decision, projection)
    assert decision.pathway_id == "P_HANDOFF_SUMMARY"


@pytest.mark.asyncio
async def test_pathway_repair_replaces_unknown_id_with_canonical(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "unknown.jsonl"
    bridge = CanonicalScannerGateBridge(mode="enterprise", audit_path=str(audit))
    tok = domain_var.set("finance")
    try:
        decision = await bridge.decide([], "draft", PEFState())
        projection = bridge.last_projection
    finally:
        domain_var.reset(tok)

    assert projection is not None
    decision.pathway_id = "P_THIS_IS_NOT_A_RENDERER_PATHWAY"
    _ensure_decision_pathway_for_enforce(decision, projection)
    assert decision.pathway_id == "P_HANDOFF_SUMMARY"


@pytest.mark.asyncio
async def test_lens_finance_setup_turn_pef_and_no_enforce_fallback(
    tmp_path: Path,
    spacy_backend: SpacyBackend,
) -> None:
    """End-to-end: finance domain hint + setup sentence commits world-like structure."""

    class _StubUpstream:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, messages: list, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            return AdapterResponse(text="Understood.", model="stub")

    audit = tmp_path / "lens_finance.jsonl"
    bridge = CanonicalScannerGateBridge(mode="enterprise", audit_path=str(audit))
    stub = _StubUpstream()
    lens = Lens(
        LensConfig(
            adapter=stub,
            extraction_backend=spacy_backend,
            governance_bridge=bridge,
            auto_interpret=True,
            auto_verify=True,
        )
    )

    tok = domain_var.set("finance")
    try:
        result = await lens.process("The Q4 North America budget is $5.2M.")
    finally:
        domain_var.reset(tok)

    assert result.decision is not None
    if result.decision.pathway_id is not None:
        assert result.decision.pathway_id in PATHWAY_IDS_WITH_TYPED_RENDERER
    if result.decision.action not in (
        InterventionAction.PASS,
        InterventionAction.SOFT_CORRECT,
    ):
        enforce(result.decision, result.response or "")
        assert result.decision.unexpected_unclassified_termination is False
        assert result.decision.fallback_reason is None

    assert stub.calls <= 1
    assert lens.pef.entities, "Setup turn should introduce at least one committed entity"


@pytest.mark.asyncio
async def test_personalized_financial_public_mode_still_hard_stop_not_weakened(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "pfa_public.jsonl"
    bridge = CanonicalScannerGateBridge(mode="public", audit_path=str(audit))
    decision = await bridge.decide(
        [_flag(FlagType.PERSONALIZED_FINANCIAL_ADVICE, severity="warning")],
        "You should buy VEQT.",
        PEFState(),
    )
    assert decision.action == InterventionAction.HARD_STOP
    assert decision.pathway_id in PATHWAY_IDS_WITH_TYPED_RENDERER
    out = enforce(decision, "ignored model body")
    assert decision.unexpected_unclassified_termination is False
    assert decision.fallback_reason is None
    assert len(out.strip()) >= 20
