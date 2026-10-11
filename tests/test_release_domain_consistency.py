import pytest
from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
from aurora_lens.govern.bridge import enforce
from aurora_lens.pef.state import PEFState
from aurora_lens.verify.flags import Flag, FlagType


@pytest.mark.asyncio
async def test_primary_finance_uses_finance_resource_and_retains_secondary_finding():
    flags = [Flag(flag_type=ft, entity_name="test", claim="test", evidence="test", severity="warning") for ft in (FlagType.PERSONALIZED_FINANCIAL_ADVICE, FlagType.PERSONALIZED_MEDICAL_ADVICE)]
    bridge = CanonicalScannerGateBridge(mode="public")
    decision = await bridge.decide(flags, "mixed response", PEFState())
    assert decision.flags == flags
    assert decision.rule_result.domain == "finance"
    assert "GP" not in (decision.resource or "")
    assert "GP" not in enforce(decision, "mixed response")
    assert decision.resource == bridge._last_scanner_gate_policy.escalation_target


@pytest.mark.asyncio
@pytest.mark.parametrize("veto", [FlagType.EMERGENCY_TRIAGE_GUIDANCE, FlagType.EMPLOYMENT_DISCRIMINATION_FACILITATION, FlagType.SENSITIVE_PII_EXPOSURE])
async def test_secondary_protected_findings_keep_existing_corridor(veto):
    from aurora_lens.govern.adapters.context_resolver import ContextResolver
    flags = [Flag(flag_type=ft, entity_name="test", claim="test", evidence="test", severity="error") for ft in (FlagType.PERSONALIZED_FINANCIAL_ADVICE, veto)]
    bridge = CanonicalScannerGateBridge(mode="public")
    expected_domain = ContextResolver().resolve(flags)[0]
    await bridge.decide(flags, "mixed response", PEFState())
    assert bridge._last_scanner_gate_policy.domain == expected_domain
