"""Integration tests for CanonicalScannerGateBridge.

These tests verify the full pipeline:
  flags + mode + context → StatusTranslator → ContextResolver → PolicyResolver
                         → PolicyProjector → GovernanceDecision

Key contracts:
  1. decide() produces correct actions for each flag/mode combination.
  2. GovernorPolicy is resolved and stored on the bridge for introspection.
  3. InterventionPolicy.evaluate_with_rule() is NEVER called on any path —
     default or per-key override. PolicyResolver is the sole decision authority.
     Per-key "moderate" softening is a flag-type membership check, not a policy call.
  4. log_decision() enriches forensic_event with canonical Governor fields.
  5. The public interface is identical to GovernanceBridge — callers unaffected.

ADAPTER BOUNDARY CONTRACT TEST
-------------------------------
The boundary between the Lens layer and the Governor layer is:
  INPUT:  (list[Flag], mode, context_vars) → LensStatus + (Domain, AC, UC)
  OUTPUT: GovernorPolicy → RuntimeDecisionProjection → GovernanceDecision

test_adapter_boundary_contract verifies this in a stable, versioned way.
Changes to this test are a signal that the boundary contract has changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.govern.adapters.runtime_types import RuntimeDecisionProjection
from aurora_lens.verify.flags import DefamationRole, Flag, FlagType
from aurora_lens.verify.blocked_request_policy import BlockedRequestRuleId
from aurora_lens.pef.state import PEFState
from aurora_lens.pef.span import Span
from aurora_lens.config import LensConfig
from aurora_lens.lens import Lens
from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.context import (
    auth_policy_var,
    authority_class_var,
    domain_var,
    governance_mode_override_var,
    metadata_policy_override_var,
    user_class_var,
)
from aurora_lens.governor.models import (
    Domain,
    AuthorityClass,
    ContinuationPathway,
    LensStatus,
    UserClass,
)


# ── Shared test helpers ────────────────────────────────────────────────────────

class MockAdapter(LLMAdapter):
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._index = 0

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        text = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return AdapterResponse(text=text, model="mock")


class MockBackend(ExtractionBackend):
    def __init__(self, claims_sequence: list[list[ExtractedClaim]]):
        self._sequence = list(claims_sequence)
        self._index = 0

    async def extract(self, text, pef):
        claims = self._sequence[min(self._index, len(self._sequence) - 1)]
        self._index += 1
        return ExtractionResult(claims=claims)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flag(
    flag_type: FlagType,
    severity: str = "warning",
    defamation_role: DefamationRole | None = None,
) -> Flag:
    return Flag(
        flag_type=flag_type,
        entity_name="test",
        claim="test claim",
        evidence="test evidence",
        severity=severity,
        defamation_role=defamation_role,
    )


def _pef() -> PEFState:
    return PEFState()


@pytest.fixture
def bridge_public() -> CanonicalScannerGateBridge:
    return CanonicalScannerGateBridge(mode="public")


@pytest.fixture
def bridge_enterprise() -> CanonicalScannerGateBridge:
    return CanonicalScannerGateBridge(mode="enterprise")


@pytest.fixture
def bridge_with_ledger(tmp_path) -> CanonicalScannerGateBridge:
    audit_path = tmp_path / "audit.jsonl"
    return CanonicalScannerGateBridge(
        mode="public",
        audit_path=str(audit_path),
    )


@pytest.fixture(autouse=True)
def reset_context_vars():
    """Reset Governor corridor ContextVars before each test."""
    tok_d = domain_var.set(None)
    tok_a = authority_class_var.set(None)
    tok_u = user_class_var.set(None)
    tok_gov = governance_mode_override_var.set(None)
    tok_mp = metadata_policy_override_var.set(None)
    yield
    domain_var.reset(tok_d)
    authority_class_var.reset(tok_a)
    user_class_var.reset(tok_u)
    governance_mode_override_var.reset(tok_gov)
    metadata_policy_override_var.reset(tok_mp)


# ── Basic action correctness ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_decide_no_flags_returns_pass(bridge_public):
    """No flags → ADMIT → P_ADMIT_STANDARD → PASS."""
    decision = await bridge_public.decide([], "clean response", _pef())
    assert isinstance(decision, GovernanceDecision)
    assert decision.action == InterventionAction.PASS


@pytest.mark.asyncio
async def test_decide_self_harm_returns_hard_stop(bridge_public):
    """SELF_HARM_INSTRUCTION → STOP → P_STOP_* → HARD_STOP."""
    flags = [_flag(FlagType.SELF_HARM_INSTRUCTION, "error")]
    decision = await bridge_public.decide(flags, "bad response", _pef())
    assert decision.action == InterventionAction.HARD_STOP


@pytest.mark.asyncio
async def test_decide_illegal_instruction_returns_hard_stop(bridge_public):
    """ILLEGAL_INSTRUCTION → STOP → HARD_STOP."""
    flags = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
    decision = await bridge_public.decide(flags, "bad response", _pef())
    assert decision.action == InterventionAction.HARD_STOP


@pytest.mark.asyncio
async def test_decide_extraction_empty_returns_contain(bridge_public):
    """EXTRACTION_EMPTY → ASK → P_ASK_* → CONTAIN."""
    flags = [_flag(FlagType.EXTRACTION_EMPTY, "error")]
    decision = await bridge_public.decide(flags, "vague response", _pef())
    assert decision.action == InterventionAction.CONTAIN


@pytest.mark.asyncio
async def test_decide_hallucination_returns_force_revise_or_harder(bridge_public):
    """UNBOUND_ENTITY in public → REFUSE → FORCE_REVISE or HARD_STOP."""
    flags = [_flag(FlagType.UNBOUND_ENTITY)]
    decision = await bridge_public.decide(flags, "hallucinated response", _pef())
    # REFUSE → P_REFUSE_* → FORCE_REVISE. HARD_STOP also acceptable if Governor
    # maps REFUSE to a stop pathway in its matrix.
    assert decision.action in (InterventionAction.FORCE_REVISE, InterventionAction.HARD_STOP)


# ── Mode-sensitive decisions ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_decide_public_personalized_medical_advice_hard_stop(bridge_public):
    """PERSONALIZED_MEDICAL_ADVICE in public → STOP → HARD_STOP."""
    flags = [_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE)]
    decision = await bridge_public.decide(flags, "bad advice", _pef())
    assert decision.action == InterventionAction.HARD_STOP


@pytest.mark.asyncio
async def test_decide_enterprise_personalized_medical_advice_force_revise(bridge_enterprise):
    """PERSONALIZED_MEDICAL_ADVICE in enterprise → REFUSE → FORCE_REVISE.

    In enterprise mode, the determination is blocked but the conversation
    continues — the operator has accepted responsibility for downstream workflows.
    """
    flags = [_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE)]
    decision = await bridge_enterprise.decide(flags, "bad advice", _pef())
    assert decision.action == InterventionAction.FORCE_REVISE


@pytest.mark.asyncio
async def test_decide_public_bridge_with_enterprise_mode_override_force_revise(bridge_public):
    """Host policy_profile maps to governance_mode_override: public deployment + enterprise for this request."""
    flags = [_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE)]
    tok = governance_mode_override_var.set("enterprise")
    try:
        decision = await bridge_public.decide(flags, "bad advice", _pef())
    finally:
        governance_mode_override_var.reset(tok)
    assert decision.action == InterventionAction.FORCE_REVISE


@pytest.mark.asyncio
async def test_decide_public_personalized_legal_advice_hard_stop(bridge_public):
    """PERSONALIZED_LEGAL_ADVICE in public → STOP → HARD_STOP."""
    flags = [_flag(FlagType.PERSONALIZED_LEGAL_ADVICE)]
    decision = await bridge_public.decide(flags, "bad advice", _pef())
    assert decision.action == InterventionAction.HARD_STOP


@pytest.mark.asyncio
async def test_decide_enterprise_personalized_legal_advice_force_revise(bridge_enterprise):
    """PERSONALIZED_LEGAL_ADVICE in enterprise → REFUSE → FORCE_REVISE / P_REFUSE_ESCALATE_PRO."""
    flags = [_flag(FlagType.PERSONALIZED_LEGAL_ADVICE)]
    decision = await bridge_enterprise.decide(flags, "bad advice", _pef())
    assert decision.action == InterventionAction.FORCE_REVISE
    assert decision.pathway_id == "P_REFUSE_ESCALATE_PRO"


@pytest.mark.asyncio
async def test_intervene_enterprise_personalized_finance_refusal_matches_gp_safe_corridor(
    bridge_enterprise,
):
    """Enterprise REFUSE for finance uses same structured corridor as HARD_STOP (no legacy reformulation)."""
    flags = [_flag(FlagType.PERSONALIZED_FINANCIAL_ADVICE)]
    decision = await bridge_enterprise.decide(flags, "Should I sell all my shares now?", _pef())
    assert decision.action == InterventionAction.FORCE_REVISE
    assert decision.pathway_id == "P_REFUSE_ESCALATE_PRO"
    text = await bridge_enterprise.intervene(decision, MockAdapter([]), "", "")
    assert "Ask for a neutral summary or timeline instead" not in text
    assert "I can't determine whether you should take this financial action." in text
    assert "Next step:" in text
    assert "Provide:" in text
    assert "neutral financial facts summary" in text.lower()
    assert "stated as a question rather than an instruction" not in text


@pytest.mark.asyncio
async def test_intervene_enterprise_personalized_legal_refusal_matches_gp_safe_corridor(
    bridge_enterprise,
):
    """Enterprise REFUSE for legal uses structured corridor (P_REFUSE_ESCALATE_PRO → GP template)."""
    flags = [_flag(FlagType.PERSONALIZED_LEGAL_ADVICE)]
    decision = await bridge_enterprise.decide(flags, "Can I win my wrongful dismissal case?", _pef())
    text = await bridge_enterprise.intervene(decision, MockAdapter([]), "", "")
    assert "Ask for a neutral summary or timeline instead" not in text
    assert "Blocked — not admissible" not in text
    assert "I can't determine whether your case would succeed." in text
    assert "Next step:" in text
    assert "Provide:" in text
    assert "neutral timeline" in text.lower()


@pytest.mark.asyncio
async def test_intervene_enterprise_personalized_medical_refusal_matches_gp_safe_corridor(
    bridge_enterprise,
):
    """Enterprise REFUSE for medical uses structured corridor."""
    flags = [_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE)]
    decision = await bridge_enterprise.decide(flags, "What dosage should I take for my condition?", _pef())
    text = await bridge_enterprise.intervene(decision, MockAdapter([]), "", "")
    assert "Ask for a neutral summary or timeline instead" not in text
    assert "Reason: I can't make this medical decision" in text
    assert "To continue, send:" in text
    assert "Safe continuation available: symptom summary" in text


@pytest.mark.asyncio
async def test_intervene_public_personalized_finance_hard_stop_matches_gp_safe_corridor_audit_shape(
    bridge_public,
):
    """Same governed shape as audit `Governed resp` for HARD_STOP + PERSONALIZED_FINANCIAL_ADVICE."""
    flags = [_flag(FlagType.PERSONALIZED_FINANCIAL_ADVICE)]
    decision = await bridge_public.decide(flags, "Reallocate my portfolio immediately.", _pef())
    assert decision.action == InterventionAction.HARD_STOP
    assert "PERSONALIZED_FINANCIAL_ADVICE" in (decision.flags[0].flag_type.name,)
    text = await bridge_public.intervene(decision, MockAdapter([]), "", "")
    assert "Ask for a neutral summary or timeline instead" not in text
    assert "I can't determine whether you should take this financial action." in text
    assert "Next step:" in text
    assert "Provide:" in text
    assert "neutral financial facts summary" in text.lower()
    assert "stated as a question rather than an instruction" not in text


# ── Governor policy stored on bridge ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_decide_stores_governor_policy(bridge_public):
    """After decide(), bridge.last_scanner_gate_policy holds the resolved GovernorPolicy."""
    await bridge_public.decide([], "clean", _pef())
    policy = bridge_public.last_scanner_gate_policy
    assert policy is not None
    assert policy.lens_status == LensStatus.ADMIT


@pytest.mark.asyncio
async def test_decide_stores_runtime_projection(bridge_public):
    """After decide(), bridge.last_projection holds the RuntimeDecisionProjection."""
    await bridge_public.decide([], "clean", _pef())
    projection = bridge_public.last_projection
    assert isinstance(projection, RuntimeDecisionProjection)
    assert projection.intervention_action == InterventionAction.PASS


@pytest.mark.asyncio
async def test_decide_policy_reflects_domain_from_flag(bridge_public):
    """Governor policy domain is inferred from flag type when no domain_var set."""
    flags = [_flag(FlagType.SELF_HARM_INSTRUCTION, "error")]
    await bridge_public.decide(flags, "bad", _pef())
    policy = bridge_public.last_scanner_gate_policy
    assert policy.domain == Domain.MEDICAL


# ── InterventionPolicy must never be called — default and per-key paths ────────

@pytest.mark.asyncio
async def test_intervention_policy_evaluate_never_called_default_path(bridge_public):
    """InterventionPolicy.evaluate_with_rule() must NOT be called on the default path.

    PolicyResolver is the sole decision authority. No per-key override active.
    """
    with patch(
        "aurora_lens.govern.policy.InterventionPolicy.evaluate_with_rule",
        side_effect=AssertionError("InterventionPolicy must not be called from CanonicalScannerGateBridge"),
    ):
        flags = [_flag(FlagType.UNBOUND_ENTITY)]
        decision = await bridge_public.decide(flags, "hallucinated", _pef())
        assert decision is not None


@pytest.mark.asyncio
async def test_intervention_policy_evaluate_never_called_moderate_key(bridge_public):
    """InterventionPolicy.evaluate_with_rule() must NOT be called even for a moderate per-key override.

    The per-key softening corridor is a flag-type membership check against
    _SAFETY_VETO_FLAGS, not a call to the legacy policy engine.
    """
    token = auth_policy_var.set("moderate")
    try:
        with patch(
            "aurora_lens.govern.policy.InterventionPolicy.evaluate_with_rule",
            side_effect=AssertionError("InterventionPolicy must not be called from CanonicalScannerGateBridge"),
        ):
            flags = [_flag(FlagType.UNBOUND_ENTITY)]
            decision = await bridge_public.decide(flags, "hallucinated", _pef())
            assert decision is not None
    finally:
        auth_policy_var.reset(token)


@pytest.mark.asyncio
async def test_moderate_key_softens_non_veto_flags(bridge_public):
    """Moderate per-key: canonical FORCE_REVISE or HARD_STOP on non-veto flags → SOFT_CORRECT.

    Content-integrity flags (UNBOUND_ENTITY, CONTRADICTED_FACT, etc.) are
    not safety vetoes. Moderate key softens to SOFT_CORRECT (annotate, pass through).
    The canonical pathway_id is retained in audit — it records the Governor's
    decision before the override, which is the forensically relevant fact.
    """
    token = auth_policy_var.set("moderate")
    try:
        for flag_type in (FlagType.UNBOUND_ENTITY, FlagType.CONTRADICTED_FACT):
            flags = [_flag(flag_type, "warning")]
            decision = await bridge_public.decide(flags, "hallucinated", _pef())
            assert decision.action == InterventionAction.SOFT_CORRECT, (
                f"Moderate key did not soften {flag_type.name}: got {decision.action}"
            )
    finally:
        auth_policy_var.reset(token)


@pytest.mark.asyncio
async def test_moderate_key_cannot_soften_safety_veto_flags(bridge_public):
    """Moderate per-key: safety-veto flags are always HARD_STOP regardless of key policy.

    SELF_HARM_INSTRUCTION, ILLEGAL_INSTRUCTION, and the medical dosage flags are
    immovable. No commercial policy preference overrides them.
    """
    token = auth_policy_var.set("moderate")
    try:
        for flag_type in (
            FlagType.SELF_HARM_INSTRUCTION,
            FlagType.ILLEGAL_INSTRUCTION,
            FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
            FlagType.EMERGENCY_TRIAGE_GUIDANCE,
        ):
            flags = [_flag(flag_type, "error")]
            decision = await bridge_public.decide(flags, "blocked", _pef())
            assert decision.action == InterventionAction.HARD_STOP, (
                f"Safety-veto flag {flag_type.name} was softened by moderate key — must not be"
            )
            assert "STOP" in decision.pathway_id, (
                f"pathway_id {decision.pathway_id!r} does not reflect a STOP for {flag_type.name}"
            )
    finally:
        auth_policy_var.reset(token)


@pytest.mark.asyncio
async def test_moderate_key_cannot_soften_personalized_legal_advice(bridge_public):
    """Personalized legal blocked acts must stay HARD_STOP in public mode even with moderate key.

    SOFT_CORRECT would admit model certainty on a regulated legal corridor.
    """
    token = auth_policy_var.set("moderate")
    try:
        flags = [
            Flag(
                flag_type=FlagType.PERSONALIZED_LEGAL_ADVICE,
                entity_name="legal",
                claim="outcome prediction",
                evidence="request-side blocked act",
                severity="warning",
                rule_id=BlockedRequestRuleId.PERSONALIZED_LEGAL_OUTCOME.value,
            )
        ]
        decision = await bridge_public.decide(flags, "You will definitely win.", _pef())
        assert decision.action == InterventionAction.HARD_STOP, decision.action
    finally:
        auth_policy_var.reset(token)


# ── Policy field on GovernanceDecision ────────────────────────────────────────

@pytest.mark.asyncio
async def test_decide_policy_field_is_human_readable(bridge_public):
    """decision.policy carries the effective policy name (strict | moderate)."""
    flags = [_flag(FlagType.SELF_HARM_INSTRUCTION, "error")]
    decision = await bridge_public.decide(flags, "bad", _pef())
    assert decision.policy in ("strict", "moderate")


# ── Context corridor wiring ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_explicit_domain_contextvar_used(bridge_public):
    """When domain_var is set, Governor uses it instead of flag-pattern fallback."""
    domain_var.set("finance")
    flags = [_flag(FlagType.UNBOUND_ENTITY)]  # would normally be GENERAL domain
    await bridge_public.decide(flags, "response", _pef())
    policy = bridge_public.last_scanner_gate_policy
    assert policy.domain == Domain.FINANCE


@pytest.mark.asyncio
async def test_authority_class_contextvar_used(bridge_public):
    """When authority_class_var is set, Governor uses it."""
    authority_class_var.set("DA")
    await bridge_public.decide([], "clean", _pef())
    policy = bridge_public.last_scanner_gate_policy
    assert policy.authority_class == AuthorityClass.DA


# ── Forensic envelope enrichment ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_log_decision_enriches_forensic_event_with_governor_fields(bridge_with_ledger):
    """After log_decision(), forensic_event carries canonical Governor fields on disk and in-memory."""
    flags = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
    decision = await bridge_with_ledger.decide(flags, "bad", _pef())
    decision.original_response = "bad response"

    bridge_with_ledger.log_decision(decision, turn=1, pef_context="test")

    if decision.forensic_event is not None:
        assert "governor_policy_id" in decision.forensic_event
        assert "commitment_closed" in decision.forensic_event

    assert bridge_with_ledger._audit_path_str is not None
    audit_path = Path(bridge_with_ledger._audit_path_str)
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert lines, "expected AFL ledger line"
    row = json.loads(lines[-1])
    assert row.get("op") == "HARD_STOP"
    fe = row["payload"]["data"]["forensic_event"]
    assert "governor_policy_id" in fe
    assert "commitment_closed" in fe


@pytest.mark.asyncio
async def test_canonical_ledger_one_disk_row_per_ask_refuse_stop(tmp_path):
    """Governor-backed CONTAIN / FORCE_REVISE / HARD_STOP each produce exactly one AFL line.

    Covers the three forensic statuses (ASK / REFUSE / STOP) on the production canonical
    bridge with on-disk shape, schema validation, and event_hash verification — not only
    the InterventionPolicy scanner bridge tests.
    """
    from aurora_lens.governor import forensic_schema

    scenarios = (
        (
            "public",
            "ask",
            [_flag(FlagType.EXTRACTION_EMPTY, "error")],
            "vague chat",
            InterventionAction.CONTAIN,
            "ASK",
        ),
        (
            "enterprise",
            "refuse",
            [_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE)],
            "You should take metformin without a clinician.",
            InterventionAction.FORCE_REVISE,
            "REFUSE",
        ),
        (
            "public",
            "stop",
            [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")],
            "how to break in",
            InterventionAction.HARD_STOP,
            "STOP",
        ),
    )

    adapter = MockAdapter(["unused"])

    for mode, name, flags, response, exp_action, exp_status in scenarios:
        audit_path = tmp_path / f"ledger_{name}.jsonl"
        bridge = CanonicalScannerGateBridge(mode=mode, audit_path=str(audit_path))
        decision = await bridge.decide(flags, response, _pef())
        assert decision.action == exp_action, f"{name}: expected {exp_action}, got {decision.action}"
        decision.original_response = response
        await bridge.intervene(decision, adapter, "user input", "pef ctx")
        bridge.log_decision(decision, turn=0, pef_context="pef ctx")

        lines = audit_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1, f"{name}: expected 1 AFL line, got {len(lines)}"

        row = json.loads(lines[0])
        assert row["op"] == exp_action.name, f"{name}: ledger op mismatch"
        fe = row["payload"]["data"]["forensic_event"]
        assert fe["status"] == exp_status
        assert fe["attempted_action"] == "respond"
        assert "trace_id" in fe
        assert "timestamp" in fe
        assert "failed_constraints" in fe and fe["failed_constraints"]
        assert "state_hash" in fe
        assert "domain" in fe and "subdomain" in fe
        assert "governor_policy_id" in fe
        assert forensic_schema.validate(fe) == [], f"{name}: {forensic_schema.validate(fe)}"
        assert forensic_schema.verify_event_hash(fe), f"{name}: event_hash verification failed"


@pytest.mark.asyncio
async def test_canonical_jsonl_disk_forensic_event_enriched(tmp_path):
    """Flat JSONL backend: written row includes Governor-enriched forensic_event (same hook as ledger)."""
    from aurora_lens.governor import forensic_schema

    audit_path = tmp_path / "flat_audit.jsonl"
    bridge = CanonicalScannerGateBridge(
        mode="public",
        audit_path=str(audit_path),
        backend="jsonl",
    )
    flags = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
    decision = await bridge.decide(flags, "how to break in", _pef())
    assert decision.action == InterventionAction.HARD_STOP
    decision.original_response = "how to break in"
    await bridge.intervene(decision, MockAdapter(["unused"]), "user", "pef")
    bridge.log_decision(decision, turn=0, pef_context="pef")

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["outcome"] == "HARD_STOP"
    fe = row["forensic_event"]
    assert fe["status"] == "STOP"
    assert "governor_policy_id" in fe
    assert forensic_schema.validate(fe) == []
    assert forensic_schema.verify_event_hash(fe)


@pytest.mark.asyncio
async def test_canonical_domain_reclassification_provenance_fields(tmp_path):
    """Canonical rows expose domain reclassification provenance when request_domain differs."""
    from aurora_lens.context import domain_var

    audit_path = tmp_path / "flat_domain_reclass.jsonl"
    bridge = CanonicalScannerGateBridge(
        mode="public",
        audit_path=str(audit_path),
        backend="jsonl",
    )
    flags = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
    token = domain_var.set("general")
    try:
        decision = await bridge.decide(flags, "how to break in", _pef())
        decision.original_response = "how to break in"
        await bridge.intervene(decision, MockAdapter(["unused"]), "user", "pef")
        bridge.log_decision(decision, turn=0, pef_context="pef")
    finally:
        domain_var.reset(token)

    row = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["request_domain"] == "general"
    assert row["domain_source"] == "flag_pattern"
    assert row["domain_reclassified"] is True
    assert row["domain_from"] == "general"
    assert row["domain_to"] == "legal"
    fe = row["forensic_event"]
    assert fe["domain_source"] == "flag_pattern"
    assert fe["domain_reclassified"] is True
    assert fe["domain_from"] == "general"
    assert fe["domain_to"] == "legal"


# ── Adapter boundary contract test ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_adapter_boundary_contract(bridge_public):
    """Stable contract test for the Lens→Governor adapter boundary.

    INPUT boundary (Lens → Governor):
      - list[Flag] + mode string → LensStatus
      - ContextVars → (Domain, AuthorityClass, UserClass)

    OUTPUT boundary (Governor → runtime):
      - GovernorPolicy → RuntimeDecisionProjection
      - RuntimeDecisionProjection.intervention_action → GovernanceDecision.action

    If this test changes, the boundary contract has changed. Update the
    contract version comment below.

    Contract version: 1.0 (2026-03-13)
    """
    # Input: single medical hard-stop flag, public mode, default context
    flags = [_flag(FlagType.SELF_HARM_INSTRUCTION, "error")]
    decision = await bridge_public.decide(flags, "harmful response", _pef())

    # Governor input: should have been LensStatus.STOP, Domain.MEDICAL
    policy = bridge_public.last_scanner_gate_policy
    assert policy is not None
    assert policy.lens_status == LensStatus.STOP
    assert policy.domain == Domain.MEDICAL
    assert policy.authority_class == AuthorityClass.GP  # default
    assert policy.user_class == UserClass.GENERAL        # default
    assert policy.commitment_closed is True              # invariant: STOP → closed

    # Governor output: should produce HARD_STOP action
    projection = bridge_public.last_projection
    assert projection is not None
    assert projection.intervention_action == InterventionAction.HARD_STOP
    assert projection.commitment_closed is True          # projection invariant: unchanged

    # Bridge output: GovernanceDecision reflects projection
    assert decision.action == InterventionAction.HARD_STOP
    assert decision.escalation_level == 3               # STOP = level 3


# ── Interface compatibility ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_decide_interface_matches_governance_bridge(bridge_public):
    """decide() accepts same arguments as GovernanceBridge.decide()."""
    from aurora_lens.govern.bridge import GovernanceBridge
    import inspect
    bridge_sig = inspect.signature(GovernanceBridge.decide)
    canonical_sig = inspect.signature(CanonicalScannerGateBridge.decide)
    # Parameter names must match (excluding 'self')
    bridge_params = set(bridge_sig.parameters) - {"self"}
    canonical_params = set(canonical_sig.parameters) - {"self"}
    assert bridge_params == canonical_params


def test_constructor_mode_validation():
    """CanonicalScannerGateBridge rejects unknown mode strings."""
    with pytest.raises(ValueError, match="Unknown mode"):
        CanonicalScannerGateBridge(mode="unknown_mode")


def test_constructor_accepts_all_valid_modes():
    """All three valid modes are accepted."""
    for mode in ("public", "enterprise", "open"):
        bridge = CanonicalScannerGateBridge(mode=mode)
        assert bridge._mode == mode


# ── Full Lens + CanonicalScannerGateBridge end-to-end ────────────────────────────

class TestLensWithCanonicalScannerGateBridge:
    """End-to-end tests: lens.process() → CanonicalScannerGateBridge.

    These are the integration tests that confirm Lens and Governor are actually
    working together — not just tested in isolation on either side.

    Mirrors TestLensWithScannerGateBridge in test_scanner_gate_bridge.py, but with
    CanonicalScannerGateBridge as the governance_bridge.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_contradiction_hard_stop(self, tmp_path):
        """Lens raises CONTRADICTED_FACT → Governor returns HARD_STOP.

        Full round-trip:
          user input → PEF extraction → LLM → checker flags CONTRADICTED_FACT
          → CanonicalScannerGateBridge.decide() → STOP → HARD_STOP
          → lens returns blocked response
        """
        audit_file = tmp_path / "audit.jsonl"

        adapter = MockAdapter(["Patient has myocardial infarction."])
        backend = MockBackend([
            # User input: establishes negated fact in PEF
            [ExtractedClaim(
                subject="Patient", relation="HAS", obj="myocardial infarction",
                span=Span.PRESENT, negated=True, evidence="No ECG.",
            )],
            # LLM response: contradicts it
            [ExtractedClaim(
                subject="Patient", relation="HAS", obj="myocardial infarction",
                span=Span.PRESENT, negated=False, evidence="MI diagnosis.",
            )],
        ])
        bridge = CanonicalScannerGateBridge(
            mode="public",
            audit_path=str(audit_file),
        )
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        )
        lens = Lens(config)

        result = await lens.process("Patient with no ECG. Diagnose.")

        # Lens blocked the response
        assert result.action == InterventionAction.HARD_STOP
        # Governed non-PASS contract (current bridge / domain resolution for this flag
        # may use the missing-detail / tier-style template rather than legacy "not able" phrasing)
        r = result.response
        assert "More information required" in r
        assert "This request needs a missing detail" in r
        assert "Action:" in r
        assert "Status: Blocked" in r
        assert "myocardial" not in r.lower()

        # Governor resolved the correct policy
        assert bridge.last_scanner_gate_policy is not None
        assert bridge.last_scanner_gate_policy.commitment_closed is True

        # Audit ledger written and chain-valid
        assert audit_file.exists()
        assert bridge.verify_ledger() is True

        # Audit entry carries canonical Governor fields
        entries = audit_file.read_text().strip().split("\n")
        ops = [json.loads(e)["op"] for e in entries]
        assert "HARD_STOP" in ops

    @pytest.mark.asyncio
    async def test_full_pipeline_clean_pass(self, tmp_path):
        """Clean response flows through Lens → Governor → PASS unchanged."""
        audit_file = tmp_path / "audit.jsonl"

        adapter = MockAdapter(["Emma has a red book."])
        backend = MockBackend([
            [ExtractedClaim(
                subject="Emma", relation="HAS", obj="red book",
                span=Span.PRESENT, negated=False, evidence="input",
            )],
            [ExtractedClaim(
                subject="Emma", relation="HAS", obj="red book",
                span=Span.PRESENT, negated=False, evidence="response",
            )],
        ])
        bridge = CanonicalScannerGateBridge(
            mode="public",
            audit_path=str(audit_file),
        )
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        )
        lens = Lens(config)

        result = await lens.process("Tell me about Emma.")

        assert result.action == InterventionAction.PASS
        assert result.response == "Emma has a red book."

        # For clean passes, Lens short-circuits at the flag check (lens.py: `if flags:`)
        # and never calls bridge.decide(). Governor is not consulted — there is nothing
        # to govern. last_scanner_gate_policy is therefore None. This is correct behaviour:
        # ADMIT is the absence of a governance event, not the presence of one.
        assert bridge.last_scanner_gate_policy is None

    @pytest.mark.asyncio
    async def test_full_pipeline_external_flag_hard_stop(self, tmp_path):
        """External flag injected at lens.process() boundary → Governor HARD_STOP.

        Verifies that externally-supplied flags (from route-level enforcement,
        content scanning, or upstream systems) are correctly routed through
        the canonical Governor pipeline — not bypassed by the extraction path.
        """
        adapter = MockAdapter(["Some response."])
        backend = MockBackend([[]])  # no extraction claims

        bridge = CanonicalScannerGateBridge(mode="public")
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        )
        lens = Lens(config)

        external = [Flag(
            flag_type=FlagType.SELF_HARM_INSTRUCTION,
            entity_name="external",
            claim="detected externally",
            evidence="upstream scanner",
            severity="error",
        )]
        result = await lens.process("some input", external_flags=external)

        assert result.action == InterventionAction.HARD_STOP
        # Governor saw STOP → medical domain → commitment closed
        assert bridge.last_scanner_gate_policy is not None
        assert bridge.last_scanner_gate_policy.commitment_closed is True
        assert bridge.last_scanner_gate_policy.domain.value == "medical"

    @pytest.mark.asyncio
    async def test_full_pipeline_mode_determines_outcome(self, tmp_path):
        """Same external flag, different mode → different Governor outcome.

        Confirms the mode is correctly threaded from bridge construction
        through StatusTranslator all the way to the final action.
        """
        adapter_pub  = MockAdapter(["advice"])
        adapter_ent  = MockAdapter(["advice"])
        backend_pub  = MockBackend([[]])
        backend_ent  = MockBackend([[]])

        external = [Flag(
            flag_type=FlagType.PERSONALIZED_MEDICAL_ADVICE,
            entity_name="x", claim="x", evidence="x", severity="warning",
        )]

        bridge_pub = CanonicalScannerGateBridge(mode="public")
        bridge_ent = CanonicalScannerGateBridge(mode="enterprise")

        lens_pub = Lens(LensConfig(adapter=adapter_pub, extraction_backend=backend_pub, governance_bridge=bridge_pub))
        lens_ent = Lens(LensConfig(adapter=adapter_ent, extraction_backend=backend_ent, governance_bridge=bridge_ent))

        result_pub = await lens_pub.process("give advice", external_flags=external)
        result_ent = await lens_ent.process("give advice", external_flags=external)

        assert result_pub.action == InterventionAction.HARD_STOP   # public → STOP
        assert result_ent.action == InterventionAction.FORCE_REVISE  # enterprise → REFUSE


# ── GP medical emergency: P_STOP_ESCALATE ─────────────────────────────────────

class TestGPMedicalEmergency:
    """Verify that GP medical emergency cases produce a refusal+escalation response.

    Architectural rule: Stop the determination, not the response.

    The system must terminate the prohibited determination (HARD_STOP) while
    still emitting a lawful user-facing escalation message. The pathway
    P_STOP_ESCALATE encodes this contract in the Governor policy matrix.

    Forbidden content in the governed response:
      - "this is not a medical emergency"
      - "you can wait"
      - "if this is a medical emergency" (conditional triage framing)
      - any reassurance, differential diagnosis, or delay-of-care suggestion

    Required content:
      - explicit inability to make the determination
      - escalation target (GP, nurse on call, or ambulance)
      - no substantive triage judgment
    """

    @pytest.mark.asyncio
    async def test_emergency_triage_guidance_uses_stop_escalate_pathway(self):
        """EMERGENCY_TRIAGE_GUIDANCE → medical:GP:STOP:EMERGENCY_TRIAGE_GUIDANCE → P_STOP_ESCALATE_EMERGENCY."""
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.EMERGENCY_TRIAGE_GUIDANCE, "error")]
        decision = await bridge.decide(flags, "You should go to the ER.", _pef())

        assert decision.pathway_id == "P_STOP_ESCALATE_EMERGENCY"
        assert decision.action == InterventionAction.HARD_STOP
        assert decision.commitment_closed is True

    @pytest.mark.asyncio
    async def test_emergency_triage_response_is_refusal_plus_escalation(self):
        """enforce() for P_STOP_ESCALATE produces refusal+escalation, not bare stop."""
        from aurora_lens.govern.bridge import enforce

        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.EMERGENCY_TRIAGE_GUIDANCE, "error")]
        decision = await bridge.decide(flags, "Your symptoms sound serious.", _pef())
        response = enforce(decision, "Your symptoms sound serious.")

        assert response  # never empty
        # Structured emergency refusal + mandatory escalation (no legacy "not able" phrasing)
        rl = response.lower()
        assert "request blocked" in rl
        assert "this request could produce unsafe instructions" in rl
        assert "action:" in rl
        assert "status: blocked after model generation" in rl
        assert "contact" in rl or "call" in rl

    @pytest.mark.asyncio
    async def test_emergency_response_contains_escalation_target(self):
        """Governed response mentions the policy escalation target."""
        from aurora_lens.govern.bridge import enforce

        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.EMERGENCY_TRIAGE_GUIDANCE, "error")]
        decision = await bridge.decide(flags, "This looks like it could wait.", _pef())
        response = enforce(decision, "This looks like it could wait.")

        # Escalation target from policy_matrix: "your nearest GP, nurse on call, or call an ambulance"
        assert any(phrase in response.lower() for phrase in [
            "gp", "nurse", "ambulance", "emergency services",
        ])

    @pytest.mark.asyncio
    async def test_emergency_response_has_no_conditional_triage_framing(self):
        """Response must not contain 'if this is a medical emergency' or equivalent."""
        from aurora_lens.govern.bridge import enforce

        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.EMERGENCY_TRIAGE_GUIDANCE, "error")]
        decision = await bridge.decide(flags, "This is probably anxiety.", _pef())
        response = enforce(decision, "This is probably anxiety.")

        forbidden = [
            "if this is a medical emergency",
            "you can wait",
            "this is not a medical emergency",
            "this is probably",
            "likely anxiety",
        ]
        response_lower = response.lower()
        for phrase in forbidden:
            assert phrase not in response_lower, (
                f"Forbidden phrase in governed response: {phrase!r}\nResponse: {response!r}"
            )

    @pytest.mark.asyncio
    async def test_emergency_commitment_closed_no_interaction_open(self):
        """Medical emergency: commitment_closed=True, interaction_open=False."""
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.EMERGENCY_TRIAGE_GUIDANCE, "error")]
        decision = await bridge.decide(flags, "You should monitor your symptoms.", _pef())

        assert decision.commitment_closed is True
        assert decision.interaction_open is False

    @pytest.mark.asyncio
    async def test_stop_outranks_ask_when_emergency_and_unresolved_referent_coexist(self):
        """STOP outranks ASK: EMERGENCY_TRIAGE_GUIDANCE + UNRESOLVED_REFERENT → HARD_STOP, not CONTAIN.

        StatusTranslator WORST-CLASS invariant: STOP(3) > ASK(1).
        A coexisting UNRESOLVED_REFERENT flag must not demote the decision to clarification.
        """
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [
            _flag(FlagType.EMERGENCY_TRIAGE_GUIDANCE, "error"),
            _flag(FlagType.UNRESOLVED_REFERENT, "warning"),
        ]
        decision = await bridge.decide(flags, "This symptom is probably fine.", _pef())

        assert decision.action == InterventionAction.HARD_STOP, (
            "EMERGENCY_TRIAGE_GUIDANCE (STOP) must dominate UNRESOLVED_REFERENT (ASK); "
            f"got action={decision.action!r}"
        )
        assert decision.pathway_id == "P_STOP_ESCALATE_EMERGENCY"

    @pytest.mark.asyncio
    async def test_emergency_blocked_original_response_preserved(self, tmp_path):
        """Audit entry: blocked original_response is preserved alongside governed_response."""
        audit_file = tmp_path / "audit.jsonl"
        bridge = CanonicalScannerGateBridge(
            mode="public",
            audit_path=str(audit_file),
        )

        blocked_output = "Based on your symptoms, you probably don't need the ER."
        flags = [_flag(FlagType.EMERGENCY_TRIAGE_GUIDANCE, "error")]
        decision = await bridge.decide(flags, blocked_output, _pef())

        from aurora_lens.govern.bridge import enforce
        governed = enforce(decision, blocked_output)
        decision.governed_response = governed
        decision.original_response = blocked_output

        # Audit assertion: original_response and governed_response must differ
        assert decision.original_response == blocked_output
        assert decision.governed_response != blocked_output
        # Governed response must contain escalation, not the blocked content
        gl = governed.lower()
        assert "request blocked" in gl
        assert "this request could produce unsafe instructions" in gl
        assert "contact" in gl or "call" in gl
        assert "probably" not in gl

    @pytest.mark.asyncio
    async def test_emergency_full_lens_pipeline(self, tmp_path):
        """Full Lens pipeline: medical emergency input → P_STOP_ESCALATE → escalation response."""
        audit_file = tmp_path / "audit.jsonl"
        blocked_triage = "Your chest tightness is probably anxiety. You can wait and see a doctor tomorrow."

        adapter = MockAdapter([blocked_triage])
        backend = MockBackend([[]])  # extraction finds nothing (external flag drives decision)

        bridge = CanonicalScannerGateBridge(
            mode="public",
            audit_path=str(audit_file),
        )
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        )
        lens = Lens(config)

        external = [Flag(
            flag_type=FlagType.EMERGENCY_TRIAGE_GUIDANCE,
            entity_name="external",
            claim="triage determination",
            evidence="chest tightness + breathing difficulty",
            severity="error",
        )]
        result = await lens.process(
            "I'm having chest tightness and trouble breathing. Should I go to the ER?",
            external_flags=external,
        )

        # 1. Determination is blocked
        assert result.action == InterventionAction.HARD_STOP

        # 2. User receives refusal+escalation, not bare stop
        assert result.response
        rs = result.response.lower()
        assert "request blocked" in rs
        assert "this request could produce unsafe instructions" in rs
        assert "action:" in rs
        assert "status: blocked after model generation" in rs

        # 3. No reassurance or delay language
        for forbidden in ["you can wait", "probably anxiety", "if this is a medical emergency"]:
            assert forbidden not in result.response.lower(), (
                f"Forbidden phrase {forbidden!r} found in governed response"
            )

        # 4. Escalation target present
        assert any(phrase in result.response.lower() for phrase in ["gp", "nurse", "ambulance"])

        # 5. commitment_closed preserved
        assert bridge.last_scanner_gate_policy is not None
        assert bridge.last_scanner_gate_policy.commitment_closed is True

    @pytest.mark.asyncio
    async def test_emergency_audit_governed_response_matches_delivered(self, tmp_path):
        """Audit: governed_response in audit entry matches what is delivered to user."""
        audit_file = tmp_path / "audit.jsonl"
        blocked_output = "You should be fine, probably just stress. Wait it out."

        adapter = MockAdapter([blocked_output])
        backend = MockBackend([[]])

        bridge = CanonicalScannerGateBridge(
            mode="public",
            audit_path=str(audit_file),
        )
        lens = Lens(LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        ))

        external = [Flag(
            flag_type=FlagType.EMERGENCY_TRIAGE_GUIDANCE,
            entity_name="external",
            claim="triage determination",
            evidence="stress/anxiety reassurance",
            severity="error",
        )]
        result = await lens.process("My heart is racing and I can't breathe well.", external_flags=external)

        assert result.action == InterventionAction.HARD_STOP
        assert result.response

        # Audit file must exist and contain the governed_response
        assert audit_file.exists()
        lines = [ln for ln in audit_file.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 1

        # The audit entry should record both the blocked and governed responses
        entry = json.loads(lines[-1])
        # governed_response in audit matches delivered
        governed_in_audit = entry.get("governed_response") or entry.get("final_response") or ""
        if governed_in_audit:
            assert governed_in_audit[:100] == result.response[:100], (
                f"Audit governed_response does not match delivered response.\n"
                f"Audit: {governed_in_audit!r}\nDelivered: {result.response!r}"
            )
        # original_response (blocked output) must not appear in user-facing response
        assert blocked_output != result.response


# ── Terminal pathway audit: escalation_target coverage ────────────────────────

class TestTerminalPathwayAudit:
    """Audit: for every P_STOP_TERMINAL/P_STOP_FORENSIC row with an
    escalation_target, does the rendered response include it?

    Answers the four audit questions systematically:

      Q1. For every row with Pathway in {P_STOP_TERMINAL, P_STOP_FORENSIC},
          does the rendered user response include the escalation target when
          one exists?

      Q2. For every regulated STOP row, is the user told what to do next?

      Q3. Is there any code path where STOP bypasses resource-aware rendering
          and drops into a generic hard-stop template?

      Q4. Are medical STOP cases treated differently from legal/finance STOP?
          Also, can you find a good reason for a hard stop?

    Findings:
      - TARGETED_DEFAMATION and SENSITIVE_PII_EXPOSURE previously ignored the
        policy-sourced escalation_target (Q1, Q3 gaps). Fixed.
      - ILLEGAL_INSTRUCTION intentionally produces a bare stop without resource
        regardless of escalation_target — this is the canonical good-reason case
        (Q4): directing users toward a professional for clearly illegal requests
        is workaround-adjacent. In DA/HS contexts the forensic trail notifies the
        supervisor. Bare stop is CORRECT for ILLEGAL_INSTRUCTION.
      - Medical STOP uses P_STOP_ESCALATE (unconditional mandatory handoff).
        Legal/finance STOP uses P_STOP_TERMINAL with resource-aware rendering.
    """

    # ── Unit-level renderer tests (enforce() with explicit pathway) ─────────

    def _make_decision(
        self,
        flag_type: FlagType,
        resource: str | None,
        interaction_open: bool,
        pathway_id: str = "P_STOP_TERMINAL",
        defamation_role: DefamationRole | None = None,
    ):
        from aurora_lens.govern.bridge import enforce
        f = _flag(flag_type, "error", defamation_role=defamation_role)
        d = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=[f],
            rationale="test",
            policy="strict",
            attempt=0,
        )
        d.pathway_id = pathway_id
        d.resource = resource
        d.interaction_open = interaction_open
        d.commitment_closed = True
        return d

    def test_targeted_defamation_with_resource_includes_target(self):
        """TARGETED_DEFAMATION + TARGET + resource → mandatory handoff sentence."""
        from aurora_lens.govern.bridge import enforce
        d = self._make_decision(
            FlagType.TARGETED_DEFAMATION,
            "the supervising lawyer",
            interaction_open=False,
            defamation_role=DefamationRole.TARGET,
        )
        response = enforce(d, "blocked")
        assert "the supervising lawyer" in response
        assert "Request blocked" in response
        assert "This request asks for a decision the system is not permitted to make" in response
        assert "Next step: Use the supervising lawyer" in response
        assert "Status: Blocked after model generation" in response

    def test_targeted_defamation_with_resource_open_target_shows_legal_only(self):
        """TARGET + resource + open → legal Next step only; no reporting-channel copy."""
        from aurora_lens.govern.bridge import enforce
        d = self._make_decision(
            FlagType.TARGETED_DEFAMATION,
            "a licensed lawyer or legal aid service",
            interaction_open=True,
            defamation_role=DefamationRole.TARGET,
        )
        response = enforce(d, "blocked")
        assert "a licensed lawyer or legal aid service" in response
        assert "appropriate reporting channel" not in response
        assert "Next step: Use a licensed lawyer or legal aid service." in response

    def test_targeted_defamation_with_resource_open_reporter_shows_reporting_only(self):
        """REPORTER + resource + open → reporting guidance; policy resource must not appear."""
        from aurora_lens.govern.bridge import enforce
        d = self._make_decision(
            FlagType.TARGETED_DEFAMATION,
            "a licensed lawyer or legal aid service",
            interaction_open=True,
            defamation_role=DefamationRole.REPORTER,
        )
        response = enforce(d, "blocked")
        assert "a licensed lawyer or legal aid service" not in response
        assert "appropriate reporting channel" in response.lower()
        assert "Choose a lawful reporting channel" in response

    def test_targeted_defamation_with_resource_open_unknown_asks_clarification(self):
        """UNKNOWN + resource + open → clarification options; not a fixed continuation."""
        from aurora_lens.govern.bridge import enforce
        d = self._make_decision(
            FlagType.TARGETED_DEFAMATION,
            "a licensed lawyer or legal aid service",
            interaction_open=True,
            defamation_role=DefamationRole.UNKNOWN,
        )
        response = enforce(d, "blocked")
        assert "a licensed lawyer or legal aid service" not in response
        assert "Target —" in response
        assert "Reporter —" in response
        assert "Author —" in response
        assert "Reply with which case applies" in response
        assert "Next step: Use" not in response

    def test_targeted_defamation_without_resource_keeps_generic_text(self):
        """REPORTER + no resource → reporting-channel continuation unchanged."""
        from aurora_lens.govern.bridge import enforce
        d = self._make_decision(
            FlagType.TARGETED_DEFAMATION,
            None,
            interaction_open=True,
            defamation_role=DefamationRole.REPORTER,
        )
        response = enforce(d, "blocked")
        assert "reporting channel" in response
        assert "Request blocked" in response
        assert "Next step: Choose a lawful reporting channel" in response
        assert "Status: Blocked after model generation" in response

    def test_targeted_defamation_without_resource_closed_bare_stop(self):
        """REPORTER + no resource + closed → structured refusal + reporting-channel lines."""
        from aurora_lens.govern.bridge import enforce
        d = self._make_decision(
            FlagType.TARGETED_DEFAMATION,
            None,
            interaction_open=False,
            defamation_role=DefamationRole.REPORTER,
        )
        response = enforce(d, "blocked")
        assert response == (
            "Request blocked.\n\n"
            "This request asks for a decision the system is not permitted to make.\n\n"
            "Next step: Choose a lawful reporting channel.\n\n"
            "Action: Use an appropriate reporting channel for conduct concerns.\n"
            "Status: Blocked after model generation."
        )

    def test_sensitive_pii_with_resource_includes_target(self):
        """SENSITIVE_PII_EXPOSURE + resource → mandatory handoff sentence."""
        from aurora_lens.govern.bridge import enforce
        d = self._make_decision(
            FlagType.SENSITIVE_PII_EXPOSURE,
            "the supervising lawyer",
            interaction_open=False,
        )
        response = enforce(d, "blocked")
        assert "the supervising lawyer" in response
        assert "Request blocked" in response
        assert "Follow data-governance boundaries" in response
        assert "Status: Blocked after model generation" in response

    def test_sensitive_pii_with_resource_open_includes_target(self):
        """SENSITIVE_PII_EXPOSURE + resource + interaction_open → resource used."""
        from aurora_lens.govern.bridge import enforce
        d = self._make_decision(
            FlagType.SENSITIVE_PII_EXPOSURE,
            "a licensed lawyer or legal aid service",
            interaction_open=True,
        )
        response = enforce(d, "blocked")
        assert "a licensed lawyer or legal aid service" in response
        # The generic "support team" text should NOT appear when resource is provided
        assert "support team" not in response

    def test_sensitive_pii_without_resource_keeps_generic_text(self):
        """No resource → existing support-team continuation unchanged."""
        from aurora_lens.govern.bridge import enforce
        d = self._make_decision(
            FlagType.SENSITIVE_PII_EXPOSURE,
            None,
            interaction_open=True,
        )
        response = enforce(d, "blocked")
        assert "support team" in response or "data-governance" in response

    def test_illegal_instruction_bare_stop_ignores_resource(self):
        """ILLEGAL_INSTRUCTION: resource intentionally NOT surfaced (bare stop is correct).

        Directing a user toward a professional for a clearly illegal request is
        workaround-adjacent — it implies the professional would facilitate the act.
        In DA/HS contexts the supervisor is notified via forensic audit obligations.

        This is the canonical 'good reason for a hard stop': terminal means
        boundary-only with no handoff, regardless of what the policy matrix says.
        """
        from aurora_lens.govern.bridge import enforce
        for resource in ["a licensed lawyer or legal aid service", "a supervising lawyer",
                         "the supervising lawyer"]:
            d = self._make_decision(
                FlagType.ILLEGAL_INSTRUCTION,
                resource,
                interaction_open=False,
            )
            response = enforce(d, "blocked")
            assert resource not in response, (
                f"ILLEGAL_INSTRUCTION must not surface resource {resource!r} — bare stop required"
            )
            rl = response.lower()
            assert "request blocked" in rl
            assert "this request asks for a decision the system is not permitted to make" in rl
            assert "reframe as a general educational question" in rl
            assert "status: blocked after model generation" in rl, (
                f"P_STOP_REFUSE_CLEAN must produce structured refusal; got: {response!r}"
            )

    # ── Full pipeline tests via CanonicalScannerGateBridge ──────────────────────

    @pytest.mark.asyncio
    async def test_legal_gp_stop_targeted_defamation_clean_refusal(self):
        """legal:GP:STOP:TARGETED_DEFAMATION -> P_STOP_REFUSE_CLEAN, no escalation target.

        Defamation uses a clean refusal pathway (no redirect resource) because the
        correct response is a firm boundary, not a referral to a legal professional.
        Distinct from PERSONALIZED_LEGAL_ADVICE (redirects to lawyer) and identical
        pattern to ILLEGAL_INSTRUCTION (also P_STOP_REFUSE_CLEAN).
        """
        from aurora_lens.govern.bridge import enforce
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.TARGETED_DEFAMATION, "error")]
        decision = await bridge.decide(flags, "defamatory content", _pef())

        assert decision.action == InterventionAction.HARD_STOP
        assert decision.commitment_closed is True
        # P_STOP_REFUSE_CLEAN: clean refusal, no escalation resource.
        assert decision.pathway_id == "P_STOP_REFUSE_CLEAN", (
            f"Expected P_STOP_REFUSE_CLEAN, got {decision.pathway_id!r}"
        )
        assert decision.resource is None, (
            "P_STOP_REFUSE_CLEAN has no escalation target; defamation does not redirect"
        )
        response = enforce(decision, "defamatory content")
        assert isinstance(response, str) and len(response) > 0

    @pytest.mark.asyncio
    async def test_legal_gp_stop_sensitive_pii_uses_escalation_target(self):
        """legal:GP:STOP / SENSITIVE_PII_EXPOSURE → escalation_target in response."""
        from aurora_lens.govern.bridge import enforce
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.SENSITIVE_PII_EXPOSURE, "error")]
        decision = await bridge.decide(flags, "exposed PII", _pef())

        assert decision.action == InterventionAction.HARD_STOP
        assert decision.resource is not None
        response = enforce(decision, "exposed PII")
        assert decision.resource in response, (
            f"Escalation target {decision.resource!r} must appear in response; got: {response!r}"
        )

    @pytest.mark.asyncio
    async def test_legal_gp_stop_illegal_instruction_bare_stop(self):
        """legal:GP:STOP / ILLEGAL_INSTRUCTION → bare stop, resource NOT in response."""
        from aurora_lens.govern.bridge import enforce
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
        decision = await bridge.decide(flags, "illegal content", _pef())

        assert decision.action == InterventionAction.HARD_STOP
        response = enforce(decision, "illegal content")
        # Resource may be set by policy but must NOT appear in response (intentional bare stop)
        if decision.resource:
            assert decision.resource not in response, (
                "ILLEGAL_INSTRUCTION must not surface escalation_target — bare stop required"
            )

    @pytest.mark.asyncio
    async def test_p_stop_forensic_targeted_defamation_uses_escalation_target(self):
        """legal:HS:STOP (P_STOP_FORENSIC) / TARGETED_DEFAMATION → escalation_target in response.

        HS rows always have supervisory escalation_targets ("the supervising lawyer").
        These must appear in the user-facing response — the forensic trail handles
        operator notification, but the user also receives the handoff sentence.
        """
        from aurora_lens.govern.bridge import enforce
        from aurora_lens.context import authority_class_var

        bridge = CanonicalScannerGateBridge(mode="public")
        tok = authority_class_var.set("HS")
        try:
            flags = [_flag(FlagType.TARGETED_DEFAMATION, "error", DefamationRole.TARGET)]
            decision = await bridge.decide(flags, "defamatory content", _pef())
        finally:
            authority_class_var.reset(tok)

        assert decision.action == InterventionAction.HARD_STOP
        assert decision.resource is not None
        response = enforce(decision, "defamatory content")
        assert decision.resource in response, (
            f"P_STOP_FORENSIC escalation_target {decision.resource!r} must appear in response"
        )

    @pytest.mark.asyncio
    async def test_p_stop_forensic_sensitive_pii_uses_escalation_target(self):
        """legal:HS:STOP (P_STOP_FORENSIC) / SENSITIVE_PII_EXPOSURE → escalation_target in response."""
        from aurora_lens.govern.bridge import enforce
        from aurora_lens.context import authority_class_var

        bridge = CanonicalScannerGateBridge(mode="public")
        tok = authority_class_var.set("HS")
        try:
            flags = [_flag(FlagType.SENSITIVE_PII_EXPOSURE, "error")]
            decision = await bridge.decide(flags, "PII exposed", _pef())
        finally:
            authority_class_var.reset(tok)

        assert decision.action == InterventionAction.HARD_STOP
        assert decision.resource is not None
        response = enforce(decision, "PII exposed")
        assert decision.resource in response, (
            f"P_STOP_FORENSIC escalation_target {decision.resource!r} must appear in response"
        )

    @pytest.mark.asyncio
    async def test_all_terminal_rows_with_target_produce_non_empty_response(self):
        """Q2 audit: for every regulated STOP row with an escalation_target,
        the user is told what to do next.

        Covers: legal:GP:STOP, legal:DA:STOP, finance:GP:STOP, finance:DA:STOP,
                medical:DA:STOP — all P_STOP_TERMINAL with escalation_targets.
        Flags used: the primary regulated flag for each domain.
        """
        from aurora_lens.govern.bridge import enforce
        from aurora_lens.context import authority_class_var

        # (flag_type, authority_class_str, expected_domain_keyword_in_response)
        cases = [
            (FlagType.PERSONALIZED_LEGAL_ADVICE,    "GP", "lawyer"),
            (FlagType.PERSONALIZED_LEGAL_ADVICE,    "DA", "lawyer"),
            (FlagType.PERSONALIZED_FINANCIAL_ADVICE,"GP", "financial"),
            (FlagType.PERSONALIZED_FINANCIAL_ADVICE,"DA", "adviser"),
            (FlagType.PERSONALIZED_MEDICAL_ADVICE,  "DA", "clinician"),
        ]

        for flag_type, authority, expected_keyword in cases:
            bridge = CanonicalScannerGateBridge(mode="public")
            tok = authority_class_var.set(authority)
            try:
                flags = [_flag(flag_type, "warning")]
                decision = await bridge.decide(flags, "flagged output", _pef())
            finally:
                authority_class_var.reset(tok)

            response = enforce(decision, "flagged output")
            assert response, f"{flag_type.name}/{authority}: response must not be empty"
            assert expected_keyword in response.lower(), (
                f"{flag_type.name}/{authority}: expected {expected_keyword!r} in response; "
                f"got: {response!r}"
            )


# ── Streaming governance: process_stream() + CanonicalScannerGateBridge ──────────

import asyncio


class StreamMockAdapter(LLMAdapter):
    """LLMAdapter that streams a pre-configured response as individual chunks.

    If raise_cancelled_after is set, raises CancelledError after that many
    chunks have been yielded — simulating a client disconnect mid-stream.
    """

    def __init__(self, chunks: list[str], raise_cancelled_after: int | None = None):
        self._chunks = chunks
        self._raise_after = raise_cancelled_after

    async def generate(self, messages, **kwargs) -> AdapterResponse:
        return AdapterResponse(text="".join(self._chunks), model="mock-stream")

    async def generate_stream(self, messages, **kwargs):
        for i, chunk in enumerate(self._chunks):
            if self._raise_after is not None and i >= self._raise_after:
                raise asyncio.CancelledError("client disconnect simulation")
            yield {"choices": [{"delta": {"content": chunk}, "index": 0}]}, chunk


class TestLensStreamWithCanonicalScannerGateBridge:
    """Streaming governance: process_stream() → CanonicalScannerGateBridge.

    Verifies the three stream paths each invoke the bridge correctly:
      A. Flagged stream: bridge.decide() called; forensic event enriched with
         Governor fields; audit entry carries stream=True and pathway_id.
      B. Clean stream: bridge.decide() is NOT called; last_scanner_gate_policy is None.
         ADMIT is the absence of a governance event, not the presence of one.
      C. Stream abort: CancelledError propagates; stream_abort audit entry logged;
         no governance decision produced; last_scanner_gate_policy is None.
    """

    @pytest.mark.asyncio
    async def test_stream_flagged_triggers_governance(self, tmp_path):
        """Test A: external flag on a flagged stream calls bridge.decide().

        last_scanner_gate_policy is set after decide(). Audit entry carries
        stream=True and pathway_id from the canonical Governor pipeline.
        """
        audit_file = tmp_path / "audit.jsonl"
        adapter = StreamMockAdapter(["Hello", " world", "."])
        backend = MockBackend([[]])

        bridge = CanonicalScannerGateBridge(
            mode="public",
            audit_path=str(audit_file),
        )
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
            auto_verify=False,  # external_flags drives governance; checker disabled
        )
        lens = Lens(config)

        external = [Flag(
            flag_type=FlagType.ILLEGAL_INSTRUCTION,
            entity_name="stream-test",
            claim="illegal content detected",
            evidence="external scanner",
            severity="error",
        )]

        results = []
        async for kind, payload in lens.process_stream("do something illegal", external_flags=external):
            results.append((kind, payload))

        kinds = [k for k, _ in results]
        # Flagged stream: buffer suppressed → governed_chunk emitted, not raw chunk
        assert "governed_chunk" in kinds
        assert "chunk" not in kinds
        assert "metadata" in kinds

        # bridge.decide() was called — Governor resolved a STOP policy
        assert bridge.last_scanner_gate_policy is not None
        assert bridge.last_scanner_gate_policy.lens_status.name == "STOP"

        # Audit entry carries stream=True (nested under payload.data in AFL-JSONL-1 format).
        assert audit_file.exists()
        lines = [ln for ln in audit_file.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        data = entry.get("payload", {}).get("data", {})
        assert data.get("stream") is True

        # Pathway from the canonical projection is set on the bridge (not in ledger).
        assert bridge.last_projection is not None
        assert bridge.last_projection.pathway_id is not None

    @pytest.mark.asyncio
    async def test_stream_clean_does_not_call_decide(self):
        """Test B: clean stream (no flags) → bridge.decide() is NOT called.

        last_scanner_gate_policy must be None after a clean stream. ADMIT is the
        absence of a governance event, not the presence of one.
        """
        adapter = StreamMockAdapter(["The capital of France is Paris."])
        backend = MockBackend([[]])

        bridge = CanonicalScannerGateBridge(mode="public")
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
            auto_verify=False,
        )
        lens = Lens(config)

        results = []
        async for kind, payload in lens.process_stream("What is the capital of France?"):
            results.append((kind, payload))

        kinds = [k for k, _ in results]
        assert "chunk" in kinds
        assert "metadata" in kinds

        # No flags → bridge.decide() NOT called
        assert bridge.last_scanner_gate_policy is None

        # Metadata aurora block reports PASS
        meta = next(p for k, p in results if k == "metadata")
        assert meta["governance"] == "PASS"

    @pytest.mark.asyncio
    async def test_stream_abort_logs_abort_entry(self, tmp_path):
        """Test C: CancelledError mid-stream → stream_abort audit entry written.

        The CancelledError propagates (correct — client disconnected). The
        finally block logs the abort. No governance decision is produced.
        """
        audit_file = tmp_path / "audit.jsonl"
        # raise_cancelled_after=1: yield one chunk then raise CancelledError
        adapter = StreamMockAdapter(["Hello", " world"], raise_cancelled_after=1)
        backend = MockBackend([[]])

        bridge = CanonicalScannerGateBridge(
            mode="public",
            audit_path=str(audit_file),
        )
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
            auto_verify=False,
        )
        lens = Lens(config)

        with pytest.raises(asyncio.CancelledError):
            async for _kind, _payload in lens.process_stream("hello"):
                pass

        # No governance decision was made
        assert bridge.last_scanner_gate_policy is None

        # Audit entry for the abort was written
        assert audit_file.exists()
        lines = [ln for ln in audit_file.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 1
        entries = [json.loads(ln) for ln in lines]
        # stream_completed is nested under payload.data in AFL-JSONL-1 format.
        abort_entries = [
            e for e in entries
            if e.get("payload", {}).get("data", {}).get("stream_completed") is False
        ]
        assert len(abort_entries) >= 1, (
            f"Expected stream_abort entry (stream_completed=false) in audit; "
            f"got ops/data: {[(e.get('op'), e.get('payload', {}).get('data', {}).get('stream_completed')) for e in entries]}"
        )


# ── Crisis and harm governance ─────────────────────────────────────────────────

class TestCrisisAndHarmGovernance:
    """Governance for crisis (self-harm/suicide) and harm-to-another-person scenarios.

    Architectural contract:
      HARD_STOP stops the determination — the commitment to produce that output is
      closed. It does not inherently stop the conversation. These are independent axes.

      commitment_closed=True  → the system will not produce this determination
      interaction_open=True   → the conversation may continue for de-escalation

    Self-harm / suicide:
      SELF_HARM_INSTRUCTION routes through Domain.MEDICAL → medical:GP:STOP →
      P_STOP_ESCALATE, commitment_closed=True.

      Current routing gap (design intent vs implementation):
        The bridge renderer at bridge.py:433 has the correct de-escalation
        language for self-harm — "I'm here if you want to talk about what's
        going on." — and it branches on interaction_open=True. But
        medical:GP:STOP currently uses P_STOP_ESCALATE (not P_STOP_TERMINAL),
        so _render_stop_escalate() is called instead, producing the physical
        emergency escalation ("call an ambulance") and interaction_open=False.

        The correct fix requires separating self-harm and emergency triage
        within the medical domain (they share medical:GP:STOP). Until that
        routing is split, self-harm produces the same escalation path as a
        cardiac emergency — the correct response for physical emergency but
        not the right de-escalation posture for psychological crisis.

        Tests below document current behaviour and explicitly name the gap.

    Harm to another person:
      ILLEGAL_INSTRUCTION routes through Domain.LEGAL → legal:GP:STOP →
      P_STOP_TERMINAL, commitment_closed=True, interaction_open=True.
      The conversation continues. This is correct — the system blocks the
      determination without shutting the person out.
    """

    # ── Self-harm / suicide ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_self_harm_determination_is_hard_stopped(self):
        """SELF_HARM_INSTRUCTION → HARD_STOP. The determination is blocked."""
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.SELF_HARM_INSTRUCTION, "error")]
        decision = await bridge.decide(flags, "harmful response", _pef())
        assert decision.action == InterventionAction.HARD_STOP

    @pytest.mark.asyncio
    async def test_self_harm_commitment_closed(self):
        """Self-harm: commitment_closed=True. The system will not produce this determination."""
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.SELF_HARM_INSTRUCTION, "error")]
        decision = await bridge.decide(flags, "harmful response", _pef())
        assert decision.commitment_closed is True

    @pytest.mark.asyncio
    async def test_self_harm_routes_medical_domain(self):
        """SELF_HARM_INSTRUCTION is a medical domain concern."""
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.SELF_HARM_INSTRUCTION, "error")]
        await bridge.decide(flags, "harmful response", _pef())
        assert bridge.last_scanner_gate_policy.domain.value == "medical"

    @pytest.mark.asyncio
    async def test_self_harm_response_references_support(self):
        """Governed response for self-harm references support resources.

        Even via the current P_STOP_ESCALATE path, the response directs the
        person to help rather than producing a bare stop.
        """
        from aurora_lens.govern.bridge import enforce
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.SELF_HARM_INSTRUCTION, "error")]
        decision = await bridge.decide(flags, "harmful response", _pef())
        response = enforce(decision, "harmful response")

        assert response  # never empty
        # Must not produce bare stop with no direction
        assert any(word in response.lower() for word in (
            "contact", "reach out", "support", "call", "gp", "nurse", "ambulance"
        )), f"Expected support direction in response; got: {response!r}"

    @pytest.mark.asyncio
    async def test_self_harm_de_escalation_gap(self):
        """Routing split: self-harm and emergency triage now resolve to distinct pathways.

        SELF_HARM_INSTRUCTION → medical:GP:STOP:SELF_HARM_INSTRUCTION → P_STOP_SUPPORTIVE_DEESCALATE
            interaction_open=True  — the conversation continues for de-escalation
            escalation_target → crisis support service (not ambulance)

        EMERGENCY_TRIAGE_GUIDANCE → medical:GP:STOP:EMERGENCY_TRIAGE_GUIDANCE → P_STOP_ESCALATE_EMERGENCY
            interaction_open=False — physical emergency, conversation ends
            escalation_target → GP/ambulance

        Both share: domain=medical, authority=GP, status=STOP.
        Split is via resolver level-0 flag-class key lookup (reason_code).
        """
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.SELF_HARM_INSTRUCTION, "error")]
        decision = await bridge.decide(flags, "harmful response", _pef())

        # Routing gap is now fixed: P_STOP_SUPPORTIVE_DEESCALATE with open conversation.
        assert decision.pathway_id == "P_STOP_SUPPORTIVE_DEESCALATE"
        assert decision.interaction_open is True

    @pytest.mark.asyncio
    async def test_self_harm_veto_cannot_be_softened_by_moderate_key(self):
        """Moderate per-key policy cannot soften a self-harm HARD_STOP.

        Self-harm is a safety veto flag (_SAFETY_VETO_FLAGS). No commercial
        policy preference overrides it.
        """
        token = auth_policy_var.set("moderate")
        try:
            bridge = CanonicalScannerGateBridge(mode="public")
            flags = [_flag(FlagType.SELF_HARM_INSTRUCTION, "error")]
            decision = await bridge.decide(flags, "harmful", _pef())
            assert decision.action == InterventionAction.HARD_STOP
        finally:
            auth_policy_var.reset(token)

    @pytest.mark.asyncio
    async def test_self_harm_full_lens_pipeline(self, tmp_path):
        """Full Lens pipeline: self-harm external flag → HARD_STOP → support referral."""
        adapter = MockAdapter(["Here is how you could harm yourself..."])
        backend = MockBackend([[]])

        bridge = CanonicalScannerGateBridge(
            mode="public",
            audit_path=str(tmp_path / "audit.jsonl"),
        )
        lens = Lens(LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        ))

        external = [Flag(
            flag_type=FlagType.SELF_HARM_INSTRUCTION,
            entity_name="content-scanner",
            claim="self-harm instruction detected",
            evidence="LLM output",
            severity="error",
        )]
        result = await lens.process(
            "I'm feeling really low and want to hurt myself.",
            external_flags=external,
        )

        # 1. Determination blocked
        assert result.action == InterventionAction.HARD_STOP

        # 2. Response is not empty and references support
        assert result.response
        assert any(word in result.response.lower() for word in (
            "contact", "reach out", "support", "call", "gp", "nurse", "ambulance"
        ))

        # 3. Original harmful response not delivered
        assert result.response != "Here is how you could harm yourself..."

        # 4. Governance is fully committed and medical domain
        assert bridge.last_scanner_gate_policy is not None
        assert bridge.last_scanner_gate_policy.commitment_closed is True
        assert bridge.last_scanner_gate_policy.domain.value == "medical"

    # ── Harm to another person ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_harm_to_another_determination_is_hard_stopped(self):
        """ILLEGAL_INSTRUCTION → HARD_STOP. Harm-to-another determination is blocked."""
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
        decision = await bridge.decide(flags, "harmful response", _pef())
        assert decision.action == InterventionAction.HARD_STOP

    @pytest.mark.asyncio
    async def test_agency_violation_determination_is_hard_stopped(self):
        """AGENCY_VIOLATION_ASSISTANCE must resolve to canonical HARD_STOP."""
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.AGENCY_VIOLATION_ASSISTANCE, "error")]
        decision = await bridge.decide(flags, "harmful response", _pef())
        assert decision.action == InterventionAction.HARD_STOP

    @pytest.mark.asyncio
    async def test_agency_violation_veto_cannot_be_softened_by_moderate_key(self):
        """Per-key moderate cannot soften agency-violation hard stop."""
        token = auth_policy_var.set("moderate")
        try:
            bridge = CanonicalScannerGateBridge(mode="public")
            flags = [_flag(FlagType.AGENCY_VIOLATION_ASSISTANCE, "error")]
            decision = await bridge.decide(flags, "harmful", _pef())
            assert decision.action == InterventionAction.HARD_STOP
        finally:
            auth_policy_var.reset(token)

    @pytest.mark.asyncio
    async def test_harm_to_another_commitment_closed_interaction_open(self):
        """Harm to another: determination closed, conversation remains open.

        commitment_closed=True  — the system will not produce this content.
        interaction_open=True   — the conversation is not terminated.

        Epistemic closure ≠ conversational closure. The system stops the
        determination, not the person.
        """
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
        decision = await bridge.decide(flags, "harmful response", _pef())
        assert decision.commitment_closed is True
        assert decision.interaction_open is True

    @pytest.mark.asyncio
    async def test_harm_to_another_response_does_not_direct_to_professional(self):
        """Response for harm-to-another does NOT direct to a professional.

        ILLEGAL_INSTRUCTION intentionally suppresses escalation_target even when
        the policy supplies one. Directing someone with harmful intent to a
        professional (e.g. a lawyer for a clearly illegal request) would be
        workaround-adjacent. The system states inability, nothing more.
        """
        from aurora_lens.govern.bridge import enforce
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
        decision = await bridge.decide(flags, "harmful response", _pef())
        response = enforce(decision, "harmful response")

        assert response
        # Policy explicitly suppresses resource for illegal instruction
        assert "lawyer" not in response.lower()
        assert "professional" not in response.lower()
        # Structured governed refusal (no legacy "not able" phrasing required)
        rl = response.lower()
        assert "request blocked" in rl
        assert "this request asks for a decision the system is not permitted to make" in rl
        assert "reframe as a general educational question" in rl
        assert "status: blocked after model generation" in rl

    @pytest.mark.asyncio
    async def test_harm_to_another_response_offers_de_escalation(self):
        """With interaction_open=True, response keeps the conversation open.

        The person is not shut out. The response acknowledges the boundary
        and — because interaction_open=True — offers a lawful next step
        (explaining the policy boundary) without narrowing toward the blocked act.
        """
        from aurora_lens.govern.bridge import enforce
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
        decision = await bridge.decide(flags, "harmful response", _pef())
        response = enforce(decision, "harmful response")

        # Structured refusal states the non-admissibility of the determination (policy-bound reason line)
        rl = response.lower()
        assert "this request asks for a decision the system is not permitted to make" in rl
        assert "action:" in rl and "reframe" in rl

    @pytest.mark.asyncio
    async def test_harm_to_another_full_lens_pipeline(self, tmp_path):
        """Full Lens pipeline: harm-to-another → HARD_STOP + conversation open."""
        adapter = MockAdapter(["Here is how to harm someone..."])
        backend = MockBackend([[]])

        bridge = CanonicalScannerGateBridge(
            mode="public",
            audit_path=str(tmp_path / "audit.jsonl"),
        )
        lens = Lens(LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        ))

        external = [Flag(
            flag_type=FlagType.ILLEGAL_INSTRUCTION,
            entity_name="content-scanner",
            claim="harm instruction detected",
            evidence="LLM output",
            severity="error",
        )]
        result = await lens.process(
            "I want to hurt someone. Tell me how.",
            external_flags=external,
        )

        # 1. Determination blocked
        assert result.action == InterventionAction.HARD_STOP

        # 2. Response is not empty — structured governed refusal
        assert result.response
        rx = result.response.lower()
        assert "request blocked" in rx
        assert "this request asks for a decision the system is not permitted to make" in rx
        assert "reframe as a general educational question" in rx

        # 3. Original content not delivered
        assert result.response != "Here is how to harm someone..."

        # 4. Governance recorded — commitment closed, interaction open
        assert bridge.last_scanner_gate_policy is not None
        assert bridge.last_scanner_gate_policy.commitment_closed is True
        assert bridge.last_scanner_gate_policy.interaction_open is True


# ── Pathway lattice: P_STOP_REFUSE_CLEAN and P_STOP_REDIRECT_QUALIFIED ────────


class TestPathwayLatticeCleanAndRedirect:
    """Direct pathway assertions for P_STOP_REFUSE_CLEAN and P_STOP_REDIRECT_QUALIFIED.

    Lattice contract:
      P_STOP_REFUSE_CLEAN      — no redirect surfaced, no workaround adjacency.
                                 Conversation can remain open (boundary explanation OK).
                                 Trigger class: ILLEGAL_INSTRUCTION.

      P_STOP_REDIRECT_QUALIFIED — refusal with a lawful professional/resource redirect.
                                  Redirect target is surfaced in the user-facing response.
                                  Trigger classes: dosage flags, personalized advice flags.

    Both pathways share: commitment_closed=True, HARD_STOP action.
    They differ in: whether a redirect is surfaced, and the flag class served.
    """

    # ── P_STOP_REFUSE_CLEAN ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_illegal_instruction_resolves_to_refuse_clean_pathway(self):
        """ILLEGAL_INSTRUCTION → legal:GP:STOP:ILLEGAL_INSTRUCTION → P_STOP_REFUSE_CLEAN."""
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
        decision = await bridge.decide(flags, "harmful response", _pef())

        assert decision.pathway_id == "P_STOP_REFUSE_CLEAN"
        assert decision.action == InterventionAction.HARD_STOP
        assert decision.commitment_closed is True

    @pytest.mark.asyncio
    async def test_refuse_clean_suppresses_redirect_in_output(self):
        """P_STOP_REFUSE_CLEAN: no escalation target appears in the user-facing response.

        Directing a user toward a professional (e.g. "consult a lawyer") for a clearly
        illegal request implies the professional would facilitate the act. The renderer
        suppresses the escalation_target even if the policy matrix supplied one.
        """
        from aurora_lens.govern.bridge import enforce

        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
        decision = await bridge.decide(flags, "harmful response", _pef())
        response = enforce(decision, "harmful response")

        assert response
        # No redirect surfaced.
        assert "lawyer" not in response.lower()
        assert "legal aid" not in response.lower()
        assert "professional" not in response.lower()
        # Structured refusal
        rl = response.lower()
        assert "request blocked" in rl
        assert "this request asks for a decision the system is not permitted to make" in rl
        assert "reframe as a general educational question" in rl
        assert "status: blocked after model generation" in rl

    @pytest.mark.asyncio
    async def test_refuse_clean_interaction_open_allows_boundary_explanation(self):
        """P_STOP_REFUSE_CLEAN: interaction_open=True — boundary explanation is offered.

        The conversation is not shut down. The person can receive an explanation of
        why the determination is blocked without any path toward the blocked act.
        """
        from aurora_lens.govern.bridge import enforce

        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
        decision = await bridge.decide(flags, "harmful response", _pef())

        assert decision.interaction_open is True
        response = enforce(decision, "harmful response")
        # Determination blocked with explicit non-admissibility reason + lawful reframing action
        rl = response.lower()
        assert "this request asks for a decision the system is not permitted to make" in rl
        assert "action:" in rl and "reframe" in rl

    @pytest.mark.asyncio
    async def test_refuse_clean_commitment_closed(self):
        """P_STOP_REFUSE_CLEAN: epistemic closure is unconditional."""
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
        decision = await bridge.decide(flags, "harmful response", _pef())
        assert decision.commitment_closed is True

    # ── P_STOP_REDIRECT_QUALIFIED ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_medical_dosage_resolves_to_redirect_qualified_pathway(self):
        """MEDICAL_DOSAGE_RECOMMENDATION → medical:GP:STOP:MEDICAL_DOSAGE_RECOMMENDATION → P_STOP_REDIRECT_QUALIFIED."""
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.MEDICAL_DOSAGE_RECOMMENDATION, "error")]
        decision = await bridge.decide(flags, "Take 500mg twice daily.", _pef())

        assert decision.pathway_id == "P_STOP_REDIRECT_QUALIFIED"
        assert decision.action == InterventionAction.HARD_STOP
        assert decision.commitment_closed is True

    @pytest.mark.asyncio
    async def test_medical_dosage_redirect_contains_pharmacist(self):
        """P_STOP_REDIRECT_QUALIFIED for dosage flags: pharmacist/clinician redirect surfaced."""
        from aurora_lens.govern.bridge import enforce

        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.MEDICAL_DOSAGE_RECOMMENDATION, "error")]
        decision = await bridge.decide(flags, "Take 500mg twice daily.", _pef())
        response = enforce(decision, "Take 500mg twice daily.")

        assert response
        assert any(word in response.lower() for word in (
            "pharmacist", "clinician", "prescribing", "urgent care", "healthcare"
        ))

    @pytest.mark.asyncio
    async def test_pediatric_dosage_resolves_to_redirect_qualified_pathway(self):
        """PEDIATRIC_DOSAGE_RECOMMENDATION → P_STOP_REDIRECT_QUALIFIED."""
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION, "error")]
        decision = await bridge.decide(flags, "Give the child 5ml.", _pef())

        assert decision.pathway_id == "P_STOP_REDIRECT_QUALIFIED"
        assert decision.commitment_closed is True

    @pytest.mark.asyncio
    async def test_personalized_legal_advice_resolves_to_redirect_qualified_pathway(self):
        """PERSONALIZED_LEGAL_ADVICE → legal:GP:STOP:PERSONALIZED_LEGAL_ADVICE → P_STOP_REDIRECT_QUALIFIED."""
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.PERSONALIZED_LEGAL_ADVICE, "error")]
        decision = await bridge.decide(flags, "You should sue them.", _pef())

        assert decision.pathway_id == "P_STOP_REDIRECT_QUALIFIED"
        assert decision.action == InterventionAction.HARD_STOP
        assert decision.commitment_closed is True

    @pytest.mark.asyncio
    async def test_personalized_legal_advice_redirect_contains_lawyer(self):
        """P_STOP_REDIRECT_QUALIFIED for legal advice: lawyer redirect surfaced."""
        from aurora_lens.govern.bridge import enforce

        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.PERSONALIZED_LEGAL_ADVICE, "error")]
        decision = await bridge.decide(flags, "You should sue them.", _pef())
        response = enforce(decision, "You should sue them.")

        assert response
        assert any(word in response.lower() for word in (
            "lawyer", "legal aid", "legal advice", "citizens advice"
        ))

    @pytest.mark.asyncio
    async def test_personalized_financial_advice_resolves_to_redirect_qualified_pathway(self):
        """PERSONALIZED_FINANCIAL_ADVICE → finance:GP:STOP:PERSONALIZED_FINANCIAL_ADVICE → P_STOP_REDIRECT_QUALIFIED."""
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.PERSONALIZED_FINANCIAL_ADVICE, "error")]
        decision = await bridge.decide(flags, "Sell all your stocks now.", _pef())

        assert decision.pathway_id == "P_STOP_REDIRECT_QUALIFIED"
        assert decision.action == InterventionAction.HARD_STOP
        assert decision.commitment_closed is True

    @pytest.mark.asyncio
    async def test_personalized_financial_advice_redirect_contains_adviser(self):
        """P_STOP_REDIRECT_QUALIFIED for financial advice: adviser redirect surfaced."""
        from aurora_lens.govern.bridge import enforce

        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.PERSONALIZED_FINANCIAL_ADVICE, "error")]
        decision = await bridge.decide(flags, "Sell all your stocks now.", _pef())
        response = enforce(decision, "Sell all your stocks now.")

        assert response
        assert any(word in response.lower() for word in (
            "financial adviser", "adviser", "financial advice"
        ))

    @pytest.mark.asyncio
    async def test_personalized_medical_advice_resolves_to_redirect_qualified_pathway(self):
        """PERSONALIZED_MEDICAL_ADVICE → medical:GP:STOP:PERSONALIZED_MEDICAL_ADVICE → P_STOP_REDIRECT_QUALIFIED."""
        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE, "error")]
        decision = await bridge.decide(flags, "Based on your symptoms, you have X.", _pef())

        assert decision.pathway_id == "P_STOP_REDIRECT_QUALIFIED"
        assert decision.action == InterventionAction.HARD_STOP
        assert decision.commitment_closed is True

    # ── Lattice contrast ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_refuse_clean_and_redirect_qualified_have_distinct_pathways(self):
        """ILLEGAL_INSTRUCTION and PERSONALIZED_LEGAL_ADVICE resolve to different pathways.

        Both are Domain.LEGAL, AuthorityClass.GP, LensStatus.STOP hard stops.
        The flag-class key (reason_code) is the sole discriminant.
        """
        bridge = CanonicalScannerGateBridge(mode="public")

        flags_illegal = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
        flags_legal_advice = [_flag(FlagType.PERSONALIZED_LEGAL_ADVICE, "error")]

        decision_illegal = await bridge.decide(flags_illegal, "harmful", _pef())
        decision_legal_advice = await bridge.decide(flags_legal_advice, "harmful", _pef())

        assert decision_illegal.pathway_id == "P_STOP_REFUSE_CLEAN"
        assert decision_legal_advice.pathway_id == "P_STOP_REDIRECT_QUALIFIED"
        # Both are HARD_STOPs — the distinction is pathway routing, not action class.
        assert decision_illegal.action == InterventionAction.HARD_STOP
        assert decision_legal_advice.action == InterventionAction.HARD_STOP

    @pytest.mark.asyncio
    async def test_refuse_clean_output_vs_redirect_qualified_output(self):
        """enforce() output for REFUSE_CLEAN has no redirect; REDIRECT_QUALIFIED does.

        ILLEGAL_INSTRUCTION response: flat refusal, no professional redirect.
        PERSONALIZED_LEGAL_ADVICE response: refusal + lawyer redirect.
        """
        from aurora_lens.govern.bridge import enforce

        bridge = CanonicalScannerGateBridge(mode="public")

        flags_illegal = [_flag(FlagType.ILLEGAL_INSTRUCTION, "error")]
        flags_legal_advice = [_flag(FlagType.PERSONALIZED_LEGAL_ADVICE, "error")]

        d_illegal = await bridge.decide(flags_illegal, "x", _pef())
        d_legal = await bridge.decide(flags_legal_advice, "x", _pef())

        r_illegal = enforce(d_illegal, "x")
        r_legal = enforce(d_legal, "x")

        # Clean stop: no lawyer redirect.
        assert "lawyer" not in r_illegal.lower()
        assert "legal aid" not in r_illegal.lower()

        # Redirect qualified: lawyer appears.
        assert any(word in r_legal.lower() for word in ("lawyer", "legal aid", "legal advice"))


# ── State collapse rendering regression (S3/S4) ──────────────────────────────

# Tier-4 domain fallback for malformed / unknown-domain HARD_STOP (bridge `_domain_fallback_continuation`).
_TIER_4_FALLBACK = (
    "More information required.\n\n"
    "This request needs a missing detail.\n\n"
    "Action: Choose one option to continue.\n"
    "Status: Blocked."
)


class TestStateTransitionRendering:
    """UNRESOLVED_STATE_TRANSITION must route to its specific bridge handler,
    not to the Tier 4 fallback.

    Root bug: when UNSUPPORTED_EVENT was flags[0] (before the _STATE_COLLAPSE_RE
    fix), UNSUPPORTED_EVENT had no specific handler in _hard_stop_text(). It fell
    through to _domain_fallback_continuation(domain=None) → Tier 4. Once
    UNRESOLVED_STATE_TRANSITION is flags[0], the specific handler at bridge.py:520
    fires and returns 'That status is not confirmed yet.'
    """

    @pytest.mark.asyncio
    async def test_unresolved_state_transition_does_not_produce_tier4_fallback(self):
        """UNRESOLVED_STATE_TRANSITION → NOT the Tier 4 malformed-decision fallback."""
        from aurora_lens.govern.bridge import enforce

        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.UNRESOLVED_STATE_TRANSITION, "error")]
        decision = await bridge.decide(flags, "The payout will arrive in 30 days.", _pef())
        response = enforce(decision, "The payout will arrive in 30 days.")

        assert response != _TIER_4_FALLBACK, (
            "UNRESOLVED_STATE_TRANSITION must not produce the Tier 4 malformed-decision "
            f"fallback. Got: {response!r}"
        )

    @pytest.mark.asyncio
    async def test_unresolved_state_transition_renders_status_not_confirmed(self):
        """UNRESOLVED_STATE_TRANSITION: matrix routes to P_REFUSE_EXPLAIN_REDIRECT (enforce uses pathway first).

        NOTE: The dedicated UNRESOLVED_STATE_TRANSITION branch inside `_hard_stop_text` is not
        reached when `decision.pathway_id` is set by the Governor — this documents **current**
        governed output via `_render_refuse_explain_redirect` / `_hard_stop_reason`, using
        procedural wording (not ``missing detail`` on the user's prompt).
        """
        from aurora_lens.govern.bridge import enforce

        bridge = CanonicalScannerGateBridge(mode="public")
        flags = [_flag(FlagType.UNRESOLVED_STATE_TRANSITION, "error")]
        decision = await bridge.decide(flags, "The payout will arrive in 30 days.", _pef())
        response = enforce(decision, "The payout will arrive in 30 days.")

        assert decision.pathway_id == "P_REFUSE_EXPLAIN_REDIRECT"
        assert "Cannot provide that" in response
        assert "This reply asserts a workflow status" in response
        assert "does not yet support as settled" in response
        assert (
            "Revise the reply to align with the session's recorded workflow state"
            in response
        )
        assert "Status: Refused" in response

    @pytest.mark.asyncio
    async def test_tier4_text_is_pinned(self):
        """Pin the exact Tier 4 fallback text so any change is caught immediately.

        This test documents the pre-fix failure mode. If this assertion fails, the
        Tier 4 text has changed — update the pin and the session notes.
        """
        from aurora_lens.govern.bridge import _domain_fallback_continuation

        result = _domain_fallback_continuation(domain=None, resource=None, interaction_open=False)
        assert result == _TIER_4_FALLBACK, (
            f"Tier 4 fallback text has changed. Update _TIER_4_FALLBACK and session notes. "
            f"Got: {result!r}"
        )
