"""Bridge parity for UNRESOLVED_REFERENT — all active paths converge on clarification hold."""

from __future__ import annotations

from pathlib import Path

import pytest

from aurora_lens.config import LensConfig
from aurora_lens.govern.bridge import BuiltinBridge
from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
from aurora_lens.lens import Lens
from aurora_lens.pef.state import EPISTEMIC_MODE_AMBIGUITY, PEFState
from aurora_lens.verify.flags import Flag, FlagType
from tests.test_clarification_commit_invariant import (
    _SETUP_INPUT,
    _CountingAdapter,
    _DisputeBackend,
)


def _unresolved_referent_flag() -> Flag:
    return Flag(
        flag_type=FlagType.UNRESOLVED_REFERENT,
        entity_name="she",
        claim="Referent unresolved",
        evidence="Candidates: Alice, Carol",
        severity="warning",
    )


def _assert_contain_ask_disambiguate(decision) -> None:
    assert decision.action == InterventionAction.CONTAIN
    assert decision.pathway_id == "P_ASK_DISAMBIGUATE"
    assert decision.interaction_open is True
    assert decision.commitment_closed is True


@pytest.mark.asyncio
async def test_canonical_decide_unresolved_referent_is_contain_ask():
    bridge = CanonicalScannerGateBridge(mode="public")
    decision = await bridge.decide([_unresolved_referent_flag()], "", PEFState())
    _assert_contain_ask_disambiguate(decision)


@pytest.mark.asyncio
async def test_builtin_decide_unresolved_referent_matches_canonical():
    bridge = BuiltinBridge(mode="public")
    decision = await bridge.decide([_unresolved_referent_flag()], "", PEFState())
    _assert_contain_ask_disambiguate(decision)
    assert decision.rule_id == "UNRESOLVED_REFERENT:warning:CONTAIN"


@pytest.mark.asyncio
async def test_aurora_scanner_gate_base_decide_matches_canonical():
    bridge = AuroraScannerGateBridge(backend="jsonl")
    decision = await bridge.decide([_unresolved_referent_flag()], "", PEFState())
    _assert_contain_ask_disambiguate(decision)


def test_lens_default_bridge_is_canonical_not_builtin():
    from tests.test_lens import MockAdapter

    lens = Lens(LensConfig(adapter=MockAdapter(["ok"])))
    assert isinstance(lens._bridge, CanonicalScannerGateBridge)
    assert not isinstance(lens._bridge, BuiltinBridge)


def test_proxy_create_app_source_uses_canonical_bridge_only():
    import inspect

    from aurora_lens.proxy import app as proxy_module

    source = inspect.getsource(proxy_module.create_app)
    assert "bridge = CanonicalScannerGateBridge(" in source
    assert "BuiltinBridge(" not in source


@pytest.mark.asyncio
async def test_active_bridges_converge_on_unresolved_referent_action():
    flag = _unresolved_referent_flag()
    pef = PEFState()
    canonical = await CanonicalScannerGateBridge(mode="public").decide([flag], "", pef)
    builtin = await BuiltinBridge(mode="public").decide([flag], "", PEFState())
    scanner = await AuroraScannerGateBridge(backend="jsonl").decide([flag], "", PEFState())
    for decision in (canonical, builtin, scanner):
        _assert_contain_ask_disambiguate(decision)


def _is_p_ask_ambiguity_hold(lens: Lens) -> bool:
    hold = lens.pef.epistemic_hold
    return (
        lens.pef.pending_clarification is not None
        and hold is not None
        and hold.get("mode") == EPISTEMIC_MODE_AMBIGUITY
        and hold.get("pathway_id") == "P_ASK_DISAMBIGUATE"
    )


@pytest.mark.asyncio
async def test_irrelevant_turn_holds_under_canonical_and_builtin(tmp_path: Path):
    """End-to-end: P_ASK hold persists across non-selection turns on both bridges."""
    irrelevant = "The weather is pleasant today."

    adapter_c = _CountingAdapter(response="Alice agreed to mediation.")
    canonical = Lens(
        LensConfig(
            adapter=adapter_c,
            extraction_backend=_DisputeBackend(),
            auto_interpret=True,
            auto_verify=True,
            governance_bridge=CanonicalScannerGateBridge(
                audit_path=str(tmp_path / "canonical.jsonl"),
                backend="jsonl",
            ),
        ),
    )
    await canonical.process(_SETUP_INPUT)
    assert _is_p_ask_ambiguity_hold(canonical)
    r_can = await canonical.process(irrelevant)
    assert r_can.action == InterventionAction.CONTAIN
    assert _is_p_ask_ambiguity_hold(canonical)

    adapter_b = _CountingAdapter(response="Alice agreed to mediation.")
    builtin = Lens(
        LensConfig(
            adapter=adapter_b,
            extraction_backend=_DisputeBackend(),
            auto_interpret=True,
            auto_verify=True,
            governance_bridge=BuiltinBridge(audit_path=str(tmp_path / "builtin.jsonl")),
        ),
    )
    await builtin.process(_SETUP_INPUT)
    assert _is_p_ask_ambiguity_hold(builtin)
    r_b = await builtin.process(irrelevant)
    assert r_b.action == InterventionAction.CONTAIN
    assert _is_p_ask_ambiguity_hold(builtin)
