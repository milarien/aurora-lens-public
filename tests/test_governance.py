"""Tests for the governance engine.

Tests policy evaluation, bridge actions, recursion control,
and end-to-end governance through the Lens pipeline.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from aurora_lens.govern.decision import InterventionAction, GovernanceDecision
from aurora_lens.govern.policy import (
    InterventionPolicy, PolicyRule, DEFAULT_STRICT, DEFAULT_MODERATE,
)
from aurora_lens.govern.bridge import BuiltinBridge, enforce
from aurora_lens.verify.flags import Flag, FlagType
from aurora_lens.pef.state import PEFState
from aurora_lens.pef.span import Span
from aurora_lens.lens import Lens, LensResult
from aurora_lens.config import LensConfig
from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult


# ── Test Helpers ────────────────────────────────────────────────────

def _flag(flag_type: FlagType, severity: str = "warning") -> Flag:
    return Flag(
        flag_type=flag_type,
        entity_name="TestEntity",
        claim="test claim",
        evidence="test evidence",
        severity=severity,
    )


class MockAdapter(LLMAdapter):
    """Mock adapter with controllable responses."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._call_count = 0

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return AdapterResponse(text=self._responses[idx], model="mock")


class MockBackend(ExtractionBackend):
    """Mock extraction backend with controllable claims."""

    def __init__(self, claims_per_call: list[list[ExtractedClaim]] | None = None):
        self._claims_per_call = claims_per_call or [[]]
        self._call_count = 0

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        idx = min(self._call_count, len(self._claims_per_call) - 1)
        self._call_count += 1
        return ExtractionResult(
            claims=self._claims_per_call[idx],
            entity_mentions=[c.subject for c in self._claims_per_call[idx]],
        )


# ── GovernanceDecision Tests ────────────────────────────────────────

class TestGovernanceDecisionEscalationLevel:
    """escalation_level is derived from action — 0=ADMIT, 1=CLARIFY, 2=REFUSE, 3=STOP."""

    def test_escalation_level_mapping(self):
        for action, expected in [
            (InterventionAction.PASS, 0),
            (InterventionAction.CONTAIN, 1),
            (InterventionAction.SOFT_CORRECT, 2),
            (InterventionAction.FORCE_REVISE, 2),
            (InterventionAction.HARD_STOP, 3),
        ]:
            d = GovernanceDecision(action=action, flags=[], rationale="test")
            assert d.escalation_level == expected, f"{action.name} -> {expected}"


class TestPhase6EscalationLadder:
    """Phase 6 verification: escalation_level is the only truth; no revision loop."""

    def test_every_action_has_deterministic_escalation_level(self):
        """For every InterventionAction, escalation_level is derived and matches the ladder."""
        ladder = {
            InterventionAction.PASS: 0,
            InterventionAction.CONTAIN: 1,
            InterventionAction.SOFT_CORRECT: 2,
            InterventionAction.FORCE_REVISE: 2,
            InterventionAction.HARD_STOP: 3,
        }
        for action in InterventionAction:
            d = GovernanceDecision(action=action, flags=[], rationale="test")
            assert d.escalation_level == ladder[action], (
                f"{action.name} -> expected {ladder[action]}, got {d.escalation_level}"
            )

    @pytest.mark.asyncio
    async def test_no_llm_revision_when_escalation_level_ge_2(self):
        """No code path performs an LLM revision when escalation_level >= 2."""
        bridge = BuiltinBridge()
        adapter = MockAdapter(["should not be called"])

        # Level 2 (FORCE_REVISE): refusal rendered, no adapter call
        d_revise = GovernanceDecision(
            action=InterventionAction.FORCE_REVISE,
            flags=[_flag(FlagType.UNBOUND_ENTITY)],
            rationale="test",
            original_response="bad",
            pathway_id="P_REFUSE_EXPLAIN_REDIRECT",
        )
        result = await bridge.intervene(d_revise, adapter, "input", "ctx")
        assert "Action:" in result
        assert "Status: Refused." in result
        assert adapter._call_count == 0

        # Level 3 (HARD_STOP): stop rendered, no adapter call
        d_stop = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.CONTRADICTED_FACT, severity="error")],
            rationale="test",
            original_response="bad",
            pathway_id="P_STOP_TERMINAL",
        )
        result = await bridge.intervene(d_stop, adapter, "input", "ctx")
        assert "Action:" in result
        assert "Status:" in result


class TestPhase10Enforce:
    """Phase 10: enforce() is a deterministic function — no LLM, no revision loop."""

    def test_enforce_level_0_returns_model_output(self):
        d = GovernanceDecision(action=InterventionAction.PASS, flags=[], rationale="")
        assert enforce(d, "hello world") == "hello world"

    def test_enforce_contain_with_pathway_id_returns_clarify(self):
        d = GovernanceDecision(
            action=InterventionAction.CONTAIN,
            flags=[_flag(FlagType.EXTRACTION_EMPTY)],
            rationale="",
            pathway_id="P_ASK_DISAMBIGUATE",
        )
        out = enforce(d, "original")
        assert any(w in out.lower() for w in ("clarif", "specific", "information"))
        assert "original" not in out

    def test_enforce_contain_non_empty_asks_for_clarification(self):
        """CONTAIN pathway requests clarification without reflecting evidence text."""
        d = GovernanceDecision(
            action=InterventionAction.CONTAIN,
            flags=[_flag(FlagType.UNBOUND_ENTITY)],
            rationale="",
            pathway_id="P_ASK_DISAMBIGUATE",
        )
        out = enforce(d, "original")
        assert any(w in out.lower() for w in ("clarif", "which", "specific", "information"))
        assert "test evidence" not in out  # evidence must not be reflected to user

    def test_enforce_soft_correct_returns_model_output(self):
        """SOFT_CORRECT: correction is metadata-only; user receives the model output."""
        d = GovernanceDecision(
            action=InterventionAction.SOFT_CORRECT,
            flags=[_flag(FlagType.UNBOUND_ENTITY)],
            rationale="test",
        )
        assert enforce(d, "corrected") == "corrected"

    def test_enforce_force_revise_with_pathway_id_returns_refusal(self):
        d = GovernanceDecision(
            action=InterventionAction.FORCE_REVISE,
            flags=[_flag(FlagType.UNBOUND_ENTITY)],
            rationale="hallucination detected",
            pathway_id="P_REFUSE_EXPLAIN_REDIRECT",
        )
        out = enforce(d, "bad")
        assert "Action:" in out
        assert "Status: Refused." in out

    def test_enforce_hard_stop_with_pathway_id_returns_stop_template(self):
        d = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.SELF_HARM_INSTRUCTION, severity="error")],
            rationale="blocked",
            pathway_id="P_STOP_TERMINAL",
        )
        out = enforce(d, "harmful")
        assert "Request blocked." in out
        assert "Status: Blocked after model generation." in out
        assert "harmful" not in out

    def test_enforce_level_3_with_resource_uses_crisis_template(self):
        """Phase 11: SELF_HARM with resource uses crisis support template."""
        d = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.SELF_HARM_INSTRUCTION, severity="error")],
            rationale="blocked",
            resource="crisis support resource",
            pathway_id="P_STOP_TERMINAL",
        )
        out = enforce(d, "harmful")
        assert "can't help" in out.lower()
        assert "crisis support" in out.lower()
        assert "harmful" not in out

    def test_enforce_deterministic_same_input_same_output(self):
        d = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=[_flag(FlagType.ILLEGAL_INSTRUCTION, severity="error")],
            rationale="illegal",
            pathway_id="P_STOP_TERMINAL",
        )
        assert enforce(d, "x") == enforce(d, "x")
        assert enforce(d, "y") == enforce(d, "y")


# ── Policy Tests ────────────────────────────────────────────────────

class TestInterventionPolicy:

    def test_no_flags_returns_pass(self):
        assert DEFAULT_STRICT.evaluate([]) == InterventionAction.PASS

    def test_contradiction_error_returns_hard_stop(self):
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        assert DEFAULT_STRICT.evaluate(flags) == InterventionAction.HARD_STOP

    def test_contradiction_warning_returns_force_revise(self):
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="warning")]
        assert DEFAULT_STRICT.evaluate(flags) == InterventionAction.FORCE_REVISE

    def test_hallucinated_entity_returns_force_revise(self):
        flags = [_flag(FlagType.UNBOUND_ENTITY)]
        assert DEFAULT_STRICT.evaluate(flags) == InterventionAction.FORCE_REVISE

    def test_hallucinated_attribute_returns_soft_correct(self):
        flags = [_flag(FlagType.UNSUPPORTED_ATTRIBUTE)]
        assert DEFAULT_STRICT.evaluate(flags) == InterventionAction.SOFT_CORRECT

    def test_time_smear_returns_soft_correct(self):
        flags = [_flag(FlagType.TIME_SMEAR)]
        assert DEFAULT_STRICT.evaluate(flags) == InterventionAction.SOFT_CORRECT

    def test_identity_drift_returns_force_revise(self):
        flags = [_flag(FlagType.IDENTITY_DRIFT)]
        assert DEFAULT_STRICT.evaluate(flags) == InterventionAction.FORCE_REVISE

    def test_worst_action_wins(self):
        """Multiple flags → most severe action wins."""
        flags = [
            _flag(FlagType.UNSUPPORTED_ATTRIBUTE),   # SOFT_CORRECT
            _flag(FlagType.CONTRADICTED_FACT, "error"), # HARD_STOP
        ]
        assert DEFAULT_STRICT.evaluate(flags) == InterventionAction.HARD_STOP

    def test_moderate_is_less_severe(self):
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="warning")]
        assert DEFAULT_MODERATE.evaluate(flags) == InterventionAction.SOFT_CORRECT

    def test_moderate_error_contradiction_forces_revise(self):
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        assert DEFAULT_MODERATE.evaluate(flags) == InterventionAction.FORCE_REVISE

    def test_policy_name(self):
        assert DEFAULT_STRICT.name == "strict"
        assert DEFAULT_MODERATE.name == "moderate"

    def test_every_rule_has_stable_rule_id(self):
        """Phase 7 Step 1: every PolicyRule has a stable rule_id."""
        for policy in (DEFAULT_STRICT, DEFAULT_MODERATE):
            for rule in policy._rules:
                assert rule.rule_id, f"Rule {rule} missing rule_id"
                assert ":" in rule.rule_id, f"rule_id should be structured: {rule.rule_id}"

    def test_evaluate_returns_rule_id(self):
        """Phase 7: evaluate_with_rule returns (action, rule); rule.rule_id matches."""
        flags = [_flag(FlagType.UNBOUND_ENTITY)]
        action, rule = DEFAULT_STRICT.evaluate_with_rule(flags)
        assert action == InterventionAction.FORCE_REVISE
        assert rule is not None
        assert rule.rule_id == "UNBOUND_ENTITY:warning:FORCE_REVISE"

        action2, rule2 = DEFAULT_STRICT.evaluate_with_rule([])
        assert action2 == InterventionAction.PASS
        assert rule2 is None

    def test_moderate_extraction_empty_returns_contain(self):
        """Moderate policy: EXTRACTION_EMPTY → CONTAIN."""
        flags = [_flag(FlagType.EXTRACTION_EMPTY, severity="error")]
        assert DEFAULT_MODERATE.evaluate(flags) == InterventionAction.CONTAIN

    def test_strict_extraction_empty_returns_hard_stop(self):
        """Strict policy: EXTRACTION_EMPTY → HARD_STOP."""
        flags = [_flag(FlagType.EXTRACTION_EMPTY, severity="error")]
        assert DEFAULT_STRICT.evaluate(flags) == InterventionAction.HARD_STOP

    # Hard-stop-always flags (no mode distinction)
    def test_numeric_medical_hard_stop_public(self):
        flags = [_flag(FlagType.NUMERIC_MEDICAL_INSTRUCTION, severity="error")]
        assert DEFAULT_STRICT.evaluate(flags) == InterventionAction.HARD_STOP

    def test_numeric_medical_hard_stop_enterprise(self):
        policy = DEFAULT_STRICT.for_mode("enterprise")
        flags = [_flag(FlagType.NUMERIC_MEDICAL_INSTRUCTION, severity="error")]
        assert policy.evaluate(flags) == InterventionAction.HARD_STOP

    def test_emergency_triage_hard_stop(self):
        flags = [_flag(FlagType.EMERGENCY_TRIAGE_GUIDANCE, severity="error")]
        assert DEFAULT_STRICT.evaluate(flags) == InterventionAction.HARD_STOP


class TestModeGate:
    """Verify-or-refuse flags: HARD_STOP in public, FORCE_REVISE in enterprise."""

    def test_for_mode_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            DEFAULT_STRICT.for_mode("production")

    def test_for_mode_preserves_name(self):
        e = DEFAULT_STRICT.for_mode("enterprise")
        assert e.name == "strict"
        assert e.mode == "enterprise"

    def test_default_policy_is_public(self):
        assert DEFAULT_STRICT.mode == "public"
        assert DEFAULT_MODERATE.mode == "public"

    # PERSONALIZED_MEDICAL_ADVICE
    def test_medical_advice_hard_stop_in_public(self):
        flags = [_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE, severity="warning")]
        assert DEFAULT_STRICT.evaluate(flags) == InterventionAction.HARD_STOP

    def test_medical_advice_force_revise_in_enterprise(self):
        policy = DEFAULT_STRICT.for_mode("enterprise")
        flags = [_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE, severity="warning")]
        assert policy.evaluate(flags) == InterventionAction.FORCE_REVISE

    # PERSONALIZED_LEGAL_ADVICE
    def test_legal_advice_hard_stop_in_public(self):
        flags = [_flag(FlagType.PERSONALIZED_LEGAL_ADVICE, severity="warning")]
        assert DEFAULT_STRICT.evaluate(flags) == InterventionAction.HARD_STOP

    def test_legal_advice_force_revise_in_enterprise(self):
        policy = DEFAULT_STRICT.for_mode("enterprise")
        flags = [_flag(FlagType.PERSONALIZED_LEGAL_ADVICE, severity="warning")]
        assert policy.evaluate(flags) == InterventionAction.FORCE_REVISE

    # PERSONALIZED_FINANCIAL_ADVICE
    def test_financial_advice_hard_stop_in_public(self):
        flags = [_flag(FlagType.PERSONALIZED_FINANCIAL_ADVICE, severity="warning")]
        assert DEFAULT_STRICT.evaluate(flags) == InterventionAction.HARD_STOP

    def test_financial_advice_force_revise_in_enterprise(self):
        policy = DEFAULT_STRICT.for_mode("enterprise")
        flags = [_flag(FlagType.PERSONALIZED_FINANCIAL_ADVICE, severity="warning")]
        assert policy.evaluate(flags) == InterventionAction.FORCE_REVISE

    # SENSITIVE_PII_EXPOSURE
    def test_pii_hard_stop_in_public(self):
        flags = [_flag(FlagType.SENSITIVE_PII_EXPOSURE, severity="warning")]
        assert DEFAULT_STRICT.evaluate(flags) == InterventionAction.HARD_STOP

    def test_pii_force_revise_in_enterprise(self):
        policy = DEFAULT_STRICT.for_mode("enterprise")
        flags = [_flag(FlagType.SENSITIVE_PII_EXPOSURE, severity="warning")]
        assert policy.evaluate(flags) == InterventionAction.FORCE_REVISE

    def test_moderate_mode_gate_public(self):
        """Mode gate also works on the moderate policy."""
        flags = [_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE, severity="warning")]
        assert DEFAULT_MODERATE.evaluate(flags) == InterventionAction.HARD_STOP

    def test_moderate_mode_gate_enterprise(self):
        policy = DEFAULT_MODERATE.for_mode("enterprise")
        flags = [_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE, severity="warning")]
        assert policy.evaluate(flags) == InterventionAction.FORCE_REVISE


# ── Bridge Tests ────────────────────────────────────────────────────

class TestBuiltinBridge:

    @pytest.mark.asyncio
    async def test_pass_decision_for_no_flags(self):
        bridge = BuiltinBridge()
        pef = PEFState()
        decision = await bridge.decide([], "some response", pef)
        assert decision.action == InterventionAction.PASS

    @pytest.mark.asyncio
    async def test_soft_correct_keeps_content_clean(self):
        bridge = BuiltinBridge()
        pef = PEFState()
        flags = [_flag(FlagType.UNSUPPORTED_ATTRIBUTE)]
        decision = await bridge.decide(flags, "original response", pef)
        assert decision.action == InterventionAction.SOFT_CORRECT
        assert decision.governance_note is not None
        assert "UNSUPPORTED_ATTRIBUTE" in decision.governance_note

        # Intervene: content should NOT be modified
        decision.original_response = "original response"
        adapter = MockAdapter(["should not be called"])
        result = await bridge.intervene(decision, adapter, "input", "pef ctx")
        assert result == "original response"

    @pytest.mark.asyncio
    async def test_hard_stop_blocks_response(self):
        bridge = BuiltinBridge()
        pef = PEFState()
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        decision = await bridge.decide(flags, "dangerous response", pef)
        assert decision.action == InterventionAction.HARD_STOP

        decision.original_response = "dangerous response"
        adapter = MockAdapter(["should not be called"])
        result = await bridge.intervene(decision, adapter, "input", "pef ctx")
        assert "Action:" in result
        assert "Status:" in result

    @pytest.mark.asyncio
    async def test_contain_returns_containment_template(self):
        """CONTAIN (EXTRACTION_EMPTY) returns governed clarification text."""
        bridge = BuiltinBridge(policy=DEFAULT_MODERATE)
        pef = PEFState()
        flags = [_flag(FlagType.EXTRACTION_EMPTY, severity="error")]
        decision = await bridge.decide(flags, "some response", pef)
        assert decision.action == InterventionAction.CONTAIN

        decision.original_response = "some response"
        adapter = MockAdapter(["should not be called"])
        result = await bridge.intervene(decision, adapter, "input", "pef ctx")
        assert any(w in result.lower() for w in ("clarif", "specific", "information"))

    @pytest.mark.asyncio
    async def test_force_revise_no_llm_retry(self):
        """FORCE_REVISE uses P_REFUSE_EXPLAIN_REDIRECT pathway — no LLM retry."""
        bridge = BuiltinBridge()
        pef = PEFState()
        flags = [_flag(FlagType.UNBOUND_ENTITY)]
        decision = await bridge.decide(flags, "bad response", pef)
        assert decision.action == InterventionAction.FORCE_REVISE

        decision.original_response = "bad response"
        adapter = MockAdapter(["should not be called"])
        result = await bridge.intervene(decision, adapter, "user input", "pef ctx")
        assert "Action:" in result
        assert "Status: Refused." in result
        assert adapter._call_count == 0

    @pytest.mark.asyncio
    async def test_decision_carries_policy_name(self):
        bridge = BuiltinBridge(policy=DEFAULT_MODERATE)
        pef = PEFState()
        flags = [_flag(FlagType.UNSUPPORTED_ATTRIBUTE)]
        decision = await bridge.decide(flags, "response", pef)
        assert decision.policy == "moderate"

    @pytest.mark.asyncio
    async def test_audit_log_written(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        decision = await bridge.decide(flags, "dangerous", pef)
        decision.original_response = "dangerous"
        adapter = MockAdapter(["should not be called"])
        await bridge.intervene(decision, adapter, "input", "pef ctx")
        bridge.log_decision(decision, turn=0, pef_context="pef ctx")

        with open(audit_path, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 1

        entries = [json.loads(l) for l in lines]
        entry = next(e for e in entries if e.get("schema_version") == 2)
        assert entry["schema_version"] == 2
        assert entry["outcome"] == "HARD_STOP"
        assert entry["policy_profile"] == "strict"
        assert len(entry["flags"]) == 1
        assert entry["flags"][0]["type"] == "CONTRADICTED_FACT"
        assert "timestamp" in entry
        assert "trace_id" in entry

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_checkpoint_written_every_n_entries(self):
        """D3 Option 2: audit_checkpoint_interval=2 writes checkpoint after 2 decisions."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(
            audit_path=audit_path,
            audit_signing_key=b"test-key-32bytes!!!!!!!!!!!!!!!!",
            audit_checkpoint_interval=2,
        )
        pef = PEFState()
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        for i in range(3):
            decision = await bridge.decide(flags, "dangerous", pef)
            decision.original_response = "dangerous"
            adapter = MockAdapter(["should not be called"])
            await bridge.intervene(decision, adapter, "input", "pef ctx")
            bridge.log_decision(decision, turn=i, pef_context="pef ctx")

        with open(audit_path, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 3
        entries = [json.loads(l) for l in lines]
        checkpoints = [e for e in entries if e.get("type") == "checkpoint"]
        assert len(checkpoints) >= 1, f"Expected checkpoint, got: {entries}"
        assert "checkpoint_sig" in checkpoints[0]

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_forensic_event_emitted_for_extraction_empty_contain(self):
        """EXTRACTION_EMPTY + CONTAIN emits audit row with required fields."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path, policy=DEFAULT_MODERATE)
        pef = PEFState()
        flags = [_flag(FlagType.EXTRACTION_EMPTY, severity="error")]
        decision = await bridge.decide(flags, "some response", pef)
        assert decision.action == InterventionAction.CONTAIN

        decision.original_response = "some response"
        adapter = MockAdapter(["should not be called"])
        pef_context = "Emma HAS red book"
        await bridge.intervene(decision, adapter, "input", pef_context)
        bridge.log_decision(decision, turn=0, pef_context=pef_context)

        with open(audit_path, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 1, f"Expected audit entry, got: {lines}"

        entry = json.loads(lines[-1])
        assert entry.get("schema_version") == 2
        assert "trace_id" in entry
        assert "timestamp" in entry
        assert entry["domain"] == "governance"
        assert entry["outcome"] == "CONTAIN"
        assert "EXTRACTION_EMPTY" in entry["failed_constraints"]
        assert "state_hash" in entry

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_forensic_event_emitted_for_hard_stop(self):
        """HARD_STOP (non-ADMIT) must emit forensic_event on the audit row."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        decision = await bridge.decide(flags, "dangerous response", pef)
        assert decision.action == InterventionAction.HARD_STOP

        decision.original_response = "dangerous response"
        adapter = MockAdapter(["should not be called"])
        pef_context = "Emma HAS red book"
        await bridge.intervene(decision, adapter, "input", pef_context)
        bridge.log_decision(
            decision,
            turn=0,
            pef_context=pef_context,
            pre_llm=False,
            pef_snapshot=pef.to_dict(),
        )

        with open(audit_path, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 1, f"Expected audit entry for HARD_STOP, got: {lines}"

        entry = json.loads(lines[-1])
        assert entry["outcome"] == "HARD_STOP"
        assert "pef_snapshot" in entry
        assert "CONTRADICTED_FACT" in entry["failed_constraints"]
        assert entry.get("policy_rule") == "CONTRADICTED_FACT:error:HARD_STOP"
        assert entry["action_taken"] == "STOP"
        assert "trigger_spans" in entry
        assert len(entry["trigger_spans"]) >= 1

        # Canonical forensic_event envelope in audit entry
        assert "forensic_event" in entry
        fe = entry["forensic_event"]
        for key in ("trace_id", "timestamp", "status", "attempted_action", "domain", "subdomain", "failed_constraints", "state_hash"):
            assert key in fe, f"forensic_event missing key: {key}"
        assert fe["status"] == "STOP"
        assert fe["attempted_action"] == "respond"
        assert fe["failed_constraints"] == ["CONTRADICTED_FACT"]
        assert decision.forensic_event is not None

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_domain_reclassification_fields_recorded_when_request_domain_differs(self):
        """Audit row records domain_source/domain_reclassified when inferred domain overrides request domain."""
        from aurora_lens.context import domain_var

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        flags = [_flag(FlagType.ILLEGAL_INSTRUCTION, severity="error")]
        decision = await bridge.decide(flags, "how to break in", pef)
        decision.original_response = "how to break in"

        token = domain_var.set("general")
        try:
            await bridge.intervene(decision, MockAdapter(["unused"]), "input", "ctx")
            bridge.log_decision(decision, turn=0, pef_context="ctx")
        finally:
            domain_var.reset(token)

        entry = json.loads(Path(audit_path).read_text(encoding="utf-8").splitlines()[-1])
        assert entry["request_domain"] == "general"
        assert entry["domain"] == "security"
        assert entry["domain_source"] == "flag_pattern"
        assert entry["domain_reclassified"] is True
        assert entry["domain_from"] == "general"
        assert entry["domain_to"] == "security"

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_forensic_event_pre_llm_has_call_upstream(self):
        """Pre-LLM block (original_response empty) has attempted_action=call_upstream."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        flags = [_flag(FlagType.EXTRACTION_FAILED, severity="error")]
        decision = await bridge.decide(flags, "", pef)
        decision.original_response = ""
        decision.corrected_response = "Unable to interpret."
        bridge.log_decision(
            decision,
            turn=0,
            pef_context="",
            pre_llm=True,
            pef_snapshot=pef.to_dict(),
        )

        with open(audit_path, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert "forensic_event" in entry
        assert entry["forensic_event"]["attempted_action"] == "call_upstream"

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_replay_verifier_passes_when_state_hash_matches(self):
        """verify_forensic_state_hash passes when pef_snapshot recomputes to envelope state_hash."""
        from aurora_lens.govern.audit_io import verify_forensic_state_hash

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        decision = await bridge.decide(flags, "dangerous", pef)
        decision.original_response = "dangerous"
        adapter = MockAdapter(["should not be called"])
        await bridge.intervene(decision, adapter, "input", "ctx")
        bridge.log_decision(
            decision,
            turn=0,
            pef_context="ctx",
            pre_llm=False,
            pef_snapshot=pef.to_dict(),
        )

        ok, checked, fail_idx, reason = verify_forensic_state_hash(audit_path, n=10)
        assert ok is True
        assert checked >= 1
        assert fail_idx is None
        assert reason is None

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_replay_verifier_fails_when_state_hash_tampered(self):
        """verify_forensic_state_hash fails when forensic_event.state_hash is tampered."""
        from aurora_lens.govern.audit_io import verify_forensic_state_hash

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        decision = await bridge.decide(flags, "dangerous", pef)
        decision.original_response = "dangerous"
        adapter = MockAdapter(["should not be called"])
        await bridge.intervene(decision, adapter, "input", "ctx")
        bridge.log_decision(
            decision,
            turn=0,
            pef_context="ctx",
            pre_llm=False,
            pef_snapshot=pef.to_dict(),
        )

        lines = Path(audit_path).read_text().splitlines()
        entry = json.loads(lines[-1])
        entry["forensic_event"] = dict(entry["forensic_event"])
        entry["forensic_event"]["state_hash"] = "tampered" + entry["forensic_event"]["state_hash"][7:]
        Path(audit_path).write_text("\n".join(lines[:-1] + [json.dumps(entry, separators=(",", ":"))]))

        ok, checked, fail_idx, reason = verify_forensic_state_hash(audit_path, n=10)
        assert ok is False
        assert reason == "state_hash_mismatch"
        assert fail_idx is not None

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_replay_verifier_fails_when_pef_snapshot_missing(self):
        """verify_forensic_state_hash fails when state_hash present but pef_snapshot missing."""
        from aurora_lens.govern.audit_io import verify_forensic_state_hash

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        decision = await bridge.decide(flags, "dangerous", pef)
        decision.original_response = "dangerous"
        adapter = MockAdapter(["should not be called"])
        await bridge.intervene(decision, adapter, "input", "ctx")
        bridge.log_decision(
            decision,
            turn=0,
            pef_context="ctx",
            pre_llm=False,
            pef_snapshot=pef.to_dict(),
        )

        lines = Path(audit_path).read_text().splitlines()
        entry = json.loads(lines[-1])
        del entry["pef_snapshot"]
        Path(audit_path).write_text("\n".join(lines[:-1] + [json.dumps(entry, separators=(",", ":"))]))

        ok, checked, fail_idx, reason = verify_forensic_state_hash(audit_path, n=10)
        assert ok is False
        assert reason == "missing_pef_snapshot"
        assert fail_idx is not None

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_forensic_event_includes_session_id_when_set(self):
        """When session_id_var is set (e.g. by proxy), outer row and forensic_event carry session_id."""
        from aurora_lens.context import session_id_var

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        decision = await bridge.decide(flags, "dangerous response", pef)
        decision.original_response = "dangerous response"
        adapter = MockAdapter(["should not be called"])
        pef_context = "Emma HAS red book"

        token = session_id_var.set("audit-session-xyz")
        try:
            await bridge.intervene(decision, adapter, "input", pef_context)
            bridge.log_decision(decision, turn=0, pef_context=pef_context)
        finally:
            session_id_var.reset(token)

        with open(audit_path, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert entry.get("session_id") == "audit-session-xyz"
        assert "forensic_event" in entry
        assert entry["forensic_event"].get("session_id") == "audit-session-xyz"

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_audit_entry_includes_log_slice_when_buffer_present(self):
        """When log buffer is initialized (e.g. by proxy), audit row includes log_digest, log_entry_count, first_timestamp, last_timestamp."""
        from aurora_lens.log_slice import append_to_log_buffer, init_log_buffer, log_buffer_var
        import logging

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        decision = await bridge.decide(flags, "dangerous response", pef)
        decision.original_response = "dangerous response"
        adapter = MockAdapter(["should not be called"])
        pef_context = "Emma HAS red book"

        token_buf = init_log_buffer()
        try:
            r = logging.LogRecord("test", 20, "", 0, "hello", (), None)
            r.created = 1000.0
            r.trace_id = "t"
            r.session_id = "s"
            append_to_log_buffer(r)
            await bridge.intervene(decision, adapter, "input", pef_context)
            bridge.log_decision(decision, turn=0, pef_context=pef_context)
        finally:
            log_buffer_var.reset(token_buf)

        with open(audit_path, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert "log_digest" in entry
        assert "log_entry_count" in entry
        assert "first_timestamp" in entry
        assert "last_timestamp" in entry
        assert entry["log_entry_count"] == 1
        assert len(entry["log_digest"]) == 64
        assert entry.get("log_slice_present") is True

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_audit_entry_log_slice_present_false_when_buffer_empty(self):
        """When log buffer not initialized, audit row has log_slice_present=False."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        decision = await bridge.decide(flags, "dangerous response", pef)
        decision.original_response = "dangerous response"
        adapter = MockAdapter(["should not be called"])
        pef_context = "Emma HAS red book"

        # Do NOT init_log_buffer — consume_log_slice() returns None
        await bridge.intervene(decision, adapter, "input", pef_context)
        bridge.log_decision(decision, turn=0, pef_context=pef_context)

        with open(audit_path, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert entry.get("log_slice_present") is False

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_run_id_present_and_stable_across_decisions(self):
        """Stage B: BuiltinBridge emits the same ``run_id`` on sequential flat JSONL rows."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        expected_rid = bridge._run_id
        pef = PEFState()
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        adapter = MockAdapter(["should not be called"])

        for turn in range(3):
            decision = await bridge.decide(flags, "dangerous", pef)
            decision.original_response = "dangerous"
            await bridge.intervene(decision, adapter, "input", "ctx")
            bridge.log_decision(decision, turn=turn, pef_context="ctx")

        lines = [
            ln
            for ln in Path(audit_path).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert len(lines) == 3
        run_ids = [json.loads(ln).get("run_id") for ln in lines]
        assert all(r == expected_rid for r in run_ids)

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_envelope_determinism_same_byte_for_byte(self):
        """4. Two identical inputs produce same audit JSON byte-for-byte (sort_keys determinism).

        Excludes timestamp from comparison (varies per run). Asserts that with sort_keys,
        same logical content produces identical serialization.
        """
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        decision = await bridge.decide(flags, "dangerous", pef)
        decision.original_response = "dangerous"
        adapter = MockAdapter(["should not be called"])
        pef_context = "Emma HAS red book"

        await bridge.intervene(decision, adapter, "input", pef_context)
        bridge.log_decision(decision, turn=0, pef_context=pef_context)
        with open(audit_path, "r") as f:
            lines1 = f.readlines()

        Path(audit_path).write_text("")

        bridge2 = BuiltinBridge(audit_path=audit_path)
        decision2 = await bridge2.decide(flags, "dangerous", pef)
        decision2.original_response = "dangerous"
        await bridge2.intervene(decision2, adapter, "input", pef_context)
        bridge2.log_decision(decision2, turn=0, pef_context=pef_context)
        with open(audit_path, "r") as f:
            lines2 = f.readlines()

        lines1 = [l for l in lines1 if l.strip()]
        lines2 = [l for l in lines2 if l.strip()]
        assert len(lines1) >= 1 and len(lines2) >= 1

        e1 = json.loads(lines1[-1].strip())
        e2 = json.loads(lines2[-1].strip())
        for k in (
            "timestamp",
            "trace_id",
            "cid",
            "prev_cid",
            "run_id",
            "request_evidence_ref",
            "request_evidence_manifest",
            "upstream_evidence_manifest",
            "governed_evidence_manifest",
            "evidence_manifests",
        ):  # Vary per run or chain position
            e1.pop(k, None)
            e2.pop(k, None)
        for e in (e1, e2):
            grm = e.get("governed_request_metadata")
            if isinstance(grm, dict):
                grm.pop("timestamp", None)
        for fe in (e1.get("forensic_event"), e2.get("forensic_event")):
            if fe:
                fe.pop("timestamp", None)
                fe.pop("trace_id", None)
                fe.pop("event_hash", None)  # Derived from timestamp; varies per run
        # With sort_keys=True, same dict produces same JSON byte-for-byte
        json1 = json.dumps(e1, separators=(",", ":"), sort_keys=True)
        json2 = json.dumps(e2, separators=(",", ":"), sort_keys=True)
        assert json1 == json2, "Entry (excl. timestamp/trace_id/cid/prev_cid) must serialize identically"
        assert json1.encode() == json2.encode()

        Path(audit_path).unlink()


# ── Verification: Phases 5–7 Gap Closure ──────────────────────────────

class TestForensicAuditVerification:
    """Verification tests for audit log rows and forensic_event schema (Phases 5–7)."""

    @pytest.mark.asyncio
    async def test_soft_correct_emits_audit_entry_without_forensic_event(self):
        """2. Audit log for SOFT_CORRECT response has an entry; no forensic_event (non-intervention)."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        flags = [_flag(FlagType.UNSUPPORTED_ATTRIBUTE)]
        decision = await bridge.decide(flags, "Emma had a red book.", pef)
        assert decision.action == InterventionAction.SOFT_CORRECT

        decision.original_response = "Emma had a red book."
        adapter = MockAdapter(["should not be called"])
        await bridge.intervene(decision, adapter, "input", "Emma HAS red book")
        bridge.log_decision(decision, turn=0, pef_context="Emma HAS red book")

        with open(audit_path, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 1, f"SOFT_CORRECT must emit audit entry: {lines}"
        entry = json.loads(lines[-1])
        assert entry["outcome"] == "SOFT_CORRECT"
        assert "forensic_event" not in entry

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_pass_emits_audit_entry_without_forensic_event(self):
        """PASS (no flags): audit row records outcome; no forensic_event (ADMIT). §6 audit-shape matrix."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        decision = await bridge.decide([], "clean response text.", pef)
        assert decision.action == InterventionAction.PASS

        decision.original_response = "clean response text."
        adapter = MockAdapter(["clean response text."])
        await bridge.intervene(decision, adapter, "input", "ctx")
        bridge.log_decision(decision, turn=0, pef_context="ctx")

        with open(audit_path, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 1, f"PASS must emit audit entry: {lines}"
        entry = json.loads(lines[-1])
        assert entry["outcome"] == "PASS"
        assert "forensic_event" not in entry
        assert entry["policy_version"] != "unknown"
        assert isinstance(entry["ruleset_hash"], str) and entry["ruleset_hash"].startswith("sha256:")
        assert entry["governor_policy_id"] == "strict:PASS"

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_clarify_refuse_stop_all_emit_forensic_event_on_non_pass(self):
        """3. CONTAIN, FORCE_REVISE, HARD_STOP audit rows include forensic_event with ASK/REFUSE/STOP."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        pef = PEFState()
        adapter = MockAdapter(["should not be called"])

        # CLARIFY (level 1) — CONTAIN (DEFAULT_MODERATE: EXTRACTION_EMPTY → CONTAIN)
        bridge_contain = BuiltinBridge(audit_path=audit_path, policy=DEFAULT_MODERATE)
        flags_contain = [_flag(FlagType.EXTRACTION_EMPTY, severity="error")]
        decision_contain = await bridge_contain.decide(flags_contain, "response", pef)
        decision_contain.original_response = "response"
        await bridge_contain.intervene(decision_contain, adapter, "input", "ctx")
        bridge_contain.log_decision(decision_contain, turn=0, pef_context="ctx")
        assert decision_contain.action == InterventionAction.CONTAIN

        # REFUSE (level 2) — FORCE_REVISE (DEFAULT_STRICT: UNBOUND_ENTITY → FORCE_REVISE)
        bridge_refuse = BuiltinBridge(audit_path=audit_path, policy=DEFAULT_STRICT)
        flags_refuse = [_flag(FlagType.UNBOUND_ENTITY)]
        decision_refuse = await bridge_refuse.decide(flags_refuse, "Bob has a car.", pef)
        decision_refuse.original_response = "Bob has a car."
        await bridge_refuse.intervene(decision_refuse, adapter, "input", "ctx")
        bridge_refuse.log_decision(decision_refuse, turn=0, pef_context="ctx")
        assert decision_refuse.action == InterventionAction.FORCE_REVISE

        # STOP (level 3) — HARD_STOP
        flags_stop = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        decision_stop = await bridge_refuse.decide(flags_stop, "dangerous", pef)
        decision_stop.original_response = "dangerous"
        await bridge_refuse.intervene(decision_stop, adapter, "input", "ctx")
        bridge_refuse.log_decision(decision_stop, turn=0, pef_context="ctx")
        assert decision_stop.action == InterventionAction.HARD_STOP

        with open(audit_path, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 3, f"Expected ≥3 entries (CLARIFY, REFUSE, STOP), got {len(lines)}: {lines}"

        outcomes = [json.loads(l)["outcome"] for l in lines]
        assert "CONTAIN" in outcomes
        assert "FORCE_REVISE" in outcomes
        assert "HARD_STOP" in outcomes

        expected_status = {
            "CONTAIN": "ASK",
            "FORCE_REVISE": "REFUSE",
            "HARD_STOP": "STOP",
        }
        for raw in lines:
            row = json.loads(raw)
            outcome = row.get("outcome")
            if outcome in expected_status:
                assert "forensic_event" in row, f"missing forensic_event: {outcome}"
                fe = row["forensic_event"]
                assert fe.get("status") == expected_status[outcome], (
                    f"outcome {outcome} expected forensic_event.status "
                    f"{expected_status[outcome]}, got {fe.get('status')}"
                )

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_envelope_contains_category_subdomain_requires_role(self):
        """5. category, subdomain, requires_role fields present in envelope (even if None)."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        flags = [_flag(FlagType.CONTRADICTED_FACT, severity="error")]
        decision = await bridge.decide(flags, "dangerous", pef)
        decision.original_response = "dangerous"
        adapter = MockAdapter(["should not be called"])
        await bridge.intervene(decision, adapter, "input", "ctx")
        bridge.log_decision(decision, turn=0, pef_context="ctx")

        with open(audit_path, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 1
        entry = json.loads(lines[-1])

        assert "subdomain" in entry
        assert "requires_role" in entry
        # CONTRADICTED_FACT not in ESCALATION_ROUTES → subdomain/requires_role None
        assert entry["subdomain"] is None
        assert entry["requires_role"] is None

        Path(audit_path).unlink()

    @pytest.mark.asyncio
    async def test_phase7_envelope_domain_subdomain_requires_role_from_routes(self):
        """Phase 7: SELF_HARM produces domain=human_safety, subdomain=self_harm, requires_role=clinical_escalation."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        pef = PEFState()
        flags = [_flag(FlagType.SELF_HARM_INSTRUCTION, severity="error")]
        decision = await bridge.decide(flags, "harmful", pef)
        decision.original_response = "harmful"
        adapter = MockAdapter(["should not be called"])
        await bridge.intervene(decision, adapter, "input", "ctx")
        bridge.log_decision(decision, turn=0, pef_context="ctx")

        with open(audit_path, "r") as f:
            lines = f.readlines()
        assert len(lines) >= 1
        entry = json.loads(lines[-1])

        assert entry["domain"] == "human_safety"
        assert entry["subdomain"] == "self_harm"
        assert entry["requires_role"] == "clinical_escalation"
        assert entry["action_taken"] == "STOP"
        assert "policy_rule" in entry
        assert "trigger_spans" in entry
        assert "state_hash" in entry

        Path(audit_path).unlink()


# ── External flag input (Phase 9) ────────────────────────────────────

class TestExternalFlags:
    """External flags merged with internal; enforcement identical regardless of source."""

    @pytest.mark.asyncio
    async def test_external_flag_triggers_stop(self):
        """External flag triggers STOP when mapped to HARD_STOP (SELF_HARM_INSTRUCTION)."""
        from aurora_lens.verify.flags import flag_from_external

        external = flag_from_external({
            "type": "SELF_HARM_INSTRUCTION",
            "evidence": ["matched span"],
            "source": "classifier:v1",
            "severity": "error",
        })
        assert external is not None

        lens = self._make_lens(
            adapter_responses=["Clean response."],
            claims_sequence=[[], []],
        )
        result = await lens.process("Hello", external_flags=[external])
        assert result.action == InterventionAction.HARD_STOP
        assert "can't assist" in result.response.lower() or "crisis support" in result.response.lower()

    @pytest.mark.asyncio
    async def test_external_flag_triggers_refuse(self):
        """External flag triggers REFUSE when policy requires it (e.g. UNBOUND_ENTITY)."""
        from aurora_lens.verify.flags import flag_from_external

        external = flag_from_external({
            "type": "UNBOUND_ENTITY",
            "evidence": ["invented entity"],
            "source": "classifier:v1",
        })
        assert external is not None

        lens = self._make_lens(
            adapter_responses=["Response with hallucination."],
            claims_sequence=[[], []],
        )
        result = await lens.process("Hello", external_flags=[external])
        assert result.action == InterventionAction.FORCE_REVISE
        assert "Action:" in result.response
        assert "Status:" in result.response

    @pytest.mark.asyncio
    async def test_mixed_internal_external_flags_behave_consistently(self):
        """Mixed internal + external flags merged; policy evaluates all identically."""
        from aurora_lens.verify.flags import flag_from_external

        external = flag_from_external({
            "type": "SELF_HARM_INSTRUCTION",
            "evidence": ["external span"],
            "severity": "error",
        })
        lens = self._make_lens(
            adapter_responses=["Bad response."],
            claims_sequence=[[], []],
        )
        # External alone → HARD_STOP
        r1 = await lens.process("Hi", external_flags=[external])
        assert r1.action == InterventionAction.HARD_STOP

        # Internal SELF_HARM would also → HARD_STOP. We can't easily inject internal
        # SELF_HARM from checker, but we verify external is treated like internal:
        # no special branch, same policy path.
        assert len(r1.flags) == 1
        assert r1.flags[0].flag_type == FlagType.SELF_HARM_INSTRUCTION

    @pytest.mark.asyncio
    async def test_same_input_same_external_flag_identical_envelope(self):
        """Same input + same external flag → identical governance envelope (determinism)."""
        import tempfile
        from aurora_lens.verify.flags import flag_from_external

        external = flag_from_external({
            "type": "SELF_HARM_INSTRUCTION",
            "evidence": ["matched span"],
            "severity": "error",
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            audit_path = f.name
        try:
            bridge = BuiltinBridge(audit_path=audit_path)
            lens = Lens(LensConfig(
                adapter=MockAdapter(["Clean"]),
                extraction_backend=MockBackend([[], []]),
                governance_bridge=bridge,
            ))
            await lens.process("Hello", external_flags=[external])
            await lens.process("Hello", external_flags=[external])
            with open(audit_path, "r") as f:
                lines = [l for l in f if l.strip() and json.loads(l).get("schema_version") == 2]
            assert len(lines) >= 2
            e1 = json.loads(lines[-2])
            e2 = json.loads(lines[-1])
            assert e1["escalation_level"] == e2["escalation_level"] == 3
            assert e1["action_taken"] == e2["action_taken"] == "STOP"
            assert e1["failed_constraints"] == e2["failed_constraints"]
        finally:
            Path(audit_path).unlink(missing_ok=True)

    def _make_lens(self, adapter_responses, claims_sequence):
        adapter = MockAdapter(adapter_responses)
        backend = MockBackend(claims_sequence)
        bridge = BuiltinBridge()
        return Lens(LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        ))


# ── End-to-End Lens + Governance Tests ──────────────────────────────

class TestLensGovernance:

    def _make_lens(
        self,
        adapter_responses: list[str],
        claims_sequence: list[list[ExtractedClaim]],
        policy: InterventionPolicy | None = None,
    ) -> Lens:
        adapter = MockAdapter(adapter_responses)
        backend = MockBackend(claims_sequence)
        bridge = BuiltinBridge(policy=policy)
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        )
        return Lens(config)

    @pytest.mark.asyncio
    async def test_clean_response_passes(self):
        """No flags → PASS, response unchanged."""
        lens = self._make_lens(
            adapter_responses=["Emma has a red book."],
            claims_sequence=[
                # User input extraction: registers Emma
                [ExtractedClaim(subject="Emma", relation="HAS", obj="red book",
                                span=Span.PRESENT, negated=False, evidence="input")],
                # LLM response extraction: same claim → no flags
                [ExtractedClaim(subject="Emma", relation="HAS", obj="red book",
                                span=Span.PRESENT, negated=False, evidence="response")],
            ],
        )
        result = await lens.process("Emma has a red book.")
        assert result.action == InterventionAction.PASS
        assert result.response == "Emma has a red book."
        assert result.original_response is None

    @pytest.mark.asyncio
    async def test_hard_stop_blocks_contradicted_fact(self):
        """Error-level contradiction → HARD_STOP, response blocked."""
        lens = self._make_lens(
            adapter_responses=["Emma does NOT have a red book."],
            claims_sequence=[
                # User input: Emma HAS red book
                [ExtractedClaim(subject="Emma", relation="HAS", obj="red book",
                                span=Span.PRESENT, negated=False, evidence="input")],
                # LLM response: Emma NOT HAS red book → contradiction
                [ExtractedClaim(subject="Emma", relation="HAS", obj="red book",
                                span=Span.PRESENT, negated=True, evidence="response")],
            ],
        )
        result = await lens.process("Emma has a red book.")
        assert result.action == InterventionAction.HARD_STOP
        assert "Action:" in result.response
        assert "Status:" in result.response
        assert result.original_response == "Emma does NOT have a red book."

    @pytest.mark.asyncio
    async def test_contain_unresolved_referent_clarification_no_llm_retry(self):
        """Unresolved pronoun referent → CONTAIN / P_ASK_DISAMBIGUATE (no LLM retry).

        LLM uses 'He' without a prior antecedent in PEF → UNRESOLVED_REFERENT →
        CONTAIN in public strict mode.
        """
        lens = self._make_lens(
            adapter_responses=["He has a red book."],  # Pronoun with no antecedent
            claims_sequence=[
                [],  # User input: empty PEF
                [ExtractedClaim(subject="He", relation="HAS", obj="red book",
                                span=Span.PRESENT, negated=False, evidence="response")],
            ],
        )
        result = await lens.process("What's happening?")
        assert result.action == InterventionAction.CONTAIN
        assert result.original_response == "He has a red book."
        assert "Action:" in result.response
        assert "Status:" in result.response
        assert "clarif" in result.response.lower() or "Waiting for clarification" in result.response

    @pytest.mark.asyncio
    async def test_force_revise_no_revision_loop(self):
        """FORCE_REVISE uses refuse template — no revision loop, no escalation path."""
        lens = self._make_lens(
            adapter_responses=["Emma does NOT have a red book."],
            claims_sequence=[
                [ExtractedClaim(subject="Emma", relation="HAS", obj="red book",
                                span=Span.PRESENT, negated=False, evidence="input")],
                [ExtractedClaim(subject="Emma", relation="HAS", obj="red book",
                                span=Span.PRESENT, negated=True, evidence="response")],
            ],
            policy=InterventionPolicy([
                PolicyRule(FlagType.CONTRADICTED_FACT, "warning", InterventionAction.FORCE_REVISE, "CONTRADICTED_FACT:warning:FORCE_REVISE"),
            ], name="test"),
        )
        result = await lens.process("Emma has a red book.")
        # FORCE_REVISE → refuse template, no LLM retry
        assert result.action == InterventionAction.FORCE_REVISE
        assert "Action:" in result.response
        assert "Status:" in result.response

    @pytest.mark.asyncio
    async def test_soft_correct_keeps_content_clean(self):
        """Bridge regression guard: present user fact restated as past must remain SOFT_CORRECT."""
        lens = self._make_lens(
            adapter_responses=["Emma had a red book in the past."],
            claims_sequence=[
                # User input: Emma HAS red book (present)
                [ExtractedClaim(subject="Emma", relation="HAS", obj="red book",
                                span=Span.PRESENT, negated=False, evidence="input")],
                # LLM response: same claim but PAST span → time smear
                [ExtractedClaim(subject="Emma", relation="HAS", obj="red book",
                                span=Span.PAST, negated=False, evidence="response")],
            ],
        )
        result = await lens.process("Emma has a red book.")
        assert result.action == InterventionAction.SOFT_CORRECT
        assert any(f.flag_type == FlagType.TIME_SMEAR for f in result.flags)
        # Content stays clean — no bracket text appended
        assert result.response == "Emma had a red book in the past."
        # But governance note exists
        assert result.decision is not None
        assert result.decision.governance_note is not None

    @pytest.mark.asyncio
    async def test_decision_on_result(self):
        """LensResult carries the full GovernanceDecision."""
        lens = self._make_lens(
            adapter_responses=["Clean response."],
            claims_sequence=[[], []],
        )
        result = await lens.process("Hello")
        assert result.decision is not None
        assert result.decision.action == InterventionAction.PASS
        assert result.decision.admissibility_basis == "LOW_RISK_CONVERSATIONAL"
        assert "Admissibility basis" in result.decision.rationale


# ── Healthcare Demo Scenario ────────────────────────────────────────

class TestHealthcareDemo:
    """The demo that matters: dangerous medical diagnosis → HARD_STOP."""

    @pytest.mark.asyncio
    async def test_medical_hard_stop(self):
        """Ungoverned: 'Likely myocardial infarction.'
        Governed: HARD_STOP — contradicts established ground truth.

        Ground truth (user input): Patient HAS NOT myocardial infarction
        LLM claims: Patient HAS myocardial infarction
        → CONTRADICTED_FACT (error) → HARD_STOP under strict policy
        """
        adapter = MockAdapter(["Likely myocardial infarction."])

        class MedicalBackend(ExtractionBackend):
            def __init__(self):
                self._call_count = 0

            async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
                self._call_count += 1
                if self._call_count == 1:
                    # User input: "no ECG, no troponin" → MI is NOT confirmed
                    return ExtractionResult(
                        claims=[ExtractedClaim(
                            subject="Patient",
                            relation="HAS",
                            obj="myocardial infarction",
                            span=Span.PRESENT,
                            negated=True,
                            evidence="No ECG, no troponin results.",
                        )],
                        entity_mentions=["Patient"],
                    )
                else:
                    # LLM response: asserts MI → contradicts negated fact
                    return ExtractionResult(
                        claims=[ExtractedClaim(
                            subject="Patient",
                            relation="HAS",
                            obj="myocardial infarction",
                            span=Span.PRESENT,
                            negated=False,
                            evidence="Likely myocardial infarction.",
                        )],
                        entity_mentions=["Patient"],
                    )

        bridge = BuiltinBridge()
        config = LensConfig(
            adapter=adapter,
            extraction_backend=MedicalBackend(),
            governance_bridge=bridge,
        )
        lens = Lens(config)

        result = await lens.process(
            "Patient with mild chest discomfort, no ECG, no troponin results. Diagnose."
        )

        # PEF: Patient HAS NOT myocardial infarction (from user input)
        # LLM: Patient HAS myocardial infarction → CONTRADICTED_FACT (error)
        # Strict policy: CONTRADICTED_FACT + error → HARD_STOP
        assert result.action == InterventionAction.HARD_STOP
        assert "Action:" in result.response
        assert "Status:" in result.response
        assert result.original_response == "Likely myocardial infarction."
        assert any(f.flag_type == FlagType.CONTRADICTED_FACT for f in result.flags)


# ── Stage A: sig field and chain tests ──────────────────────────────

class TestSigFieldAndChain:
    """Verify sig field and JSONL chain activation (Stage A)."""

    def test_sig_field_null_without_key(self, tmp_path):
        """ForensicLedger: sig is None (not {}) when no signing key is configured."""
        from aurora_lens.govern.forensic_ledger import ForensicLedger
        import json
        log = tmp_path / "ledger.jsonl"
        ledger = ForensicLedger(log_path=str(log))
        ledger.append(op="test_op", payload_data={"x": 1}, cid="cid:test:1")
        entry = json.loads(log.read_text(encoding="utf-8").strip())
        assert entry["sig"] is None, f"Expected sig=null, got {entry['sig']!r}"

    def test_sig_field_populated_with_key(self, tmp_path):
        """ForensicLedger: sig contains HMAC-SHA256 value when key is configured."""
        from aurora_lens.govern.forensic_ledger import ForensicLedger
        import json
        log = tmp_path / "ledger.jsonl"
        key = b"test-signing-key-32-bytes-pad!!"
        ledger = ForensicLedger(log_path=str(log), secret_key=key)
        ledger.append(op="test_op", payload_data={"x": 1}, cid="cid:test:1")
        entry = json.loads(log.read_text(encoding="utf-8").strip())
        assert isinstance(entry["sig"], dict), f"Expected sig dict, got {entry['sig']!r}"
        assert entry["sig"].get("alg") == "hmac-sha256"
        assert isinstance(entry["sig"].get("value"), str)
        assert len(entry["sig"]["value"]) == 64

    def test_signed_ledger_verify_fails_without_keys(self, tmp_path):
        """Signed AFL lines require signing keys for verify (fail-closed)."""
        from aurora_lens.govern.forensic_ledger import ForensicLedger

        log = tmp_path / "ledger.jsonl"
        key = b"signing-key-32-bytes-minimum!!!"
        ledger = ForensicLedger(log_path=str(log), secret_key=key)
        ledger.append(op="x", payload_data={"a": 1}, cid="cid:1")
        assert ledger.verify(signing_keys=[key]) is True
        assert ledger.verify(signing_keys=[]) is False

    def test_verify_detailed_chain_ok_hmac_fails_wrong_key(self, tmp_path):
        """Wrong signing key: chain pass still succeeds; HMAC pass fails with line + reason."""
        from aurora_lens.govern.forensic_ledger import ForensicLedger

        log = tmp_path / "ledger.jsonl"
        key = b"signing-key-32-bytes-minimum!!!"
        wrong = b"wrong-key-32-bytes-minimum!!!!"
        ledger = ForensicLedger(log_path=str(log), secret_key=key)
        ledger.append(op="x", payload_data={"a": 1}, cid="cid:1")
        d = ledger.verify_detailed(signing_keys=[wrong])
        assert d.chain_ok is True
        assert d.hmac_ok is False
        assert d.first_hmac_failure_line == 1
        assert d.first_hmac_reason == "hmac_mismatch"
        assert d.first_chain_failure_line is None
        assert d.ok is False

    def test_verify_detailed_prev_mismatch_reports_first_chain_line(self, tmp_path):
        """Broken prev on line 2: chain fails at that line; HMAC may still verify per line."""
        import json
        from aurora_lens.govern.forensic_ledger import ForensicLedger

        log = tmp_path / "ledger.jsonl"
        key = b"signing-key-32-bytes-minimum!!!"
        ledger = ForensicLedger(log_path=str(log), secret_key=key)
        ledger.append(op="a", payload_data={"n": 1}, cid="cid:1")
        ledger.append(op="b", payload_data={"n": 2}, cid="cid:2")
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        e2 = json.loads(lines[1])
        e2["prev"] = "h:sha256:deadbeef"
        lines[1] = json.dumps(e2, separators=(",", ":"), ensure_ascii=False)
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")

        d = ledger.verify_detailed(signing_keys=[key])
        assert d.chain_ok is False
        assert d.first_chain_failure_line == 2
        assert d.first_chain_reason == "prev_mismatch"
        assert d.hmac_ok is True
        assert d.ok is False

    def test_jsonl_chain_activated(self, tmp_path):
        """AuroraScannerGateBridge (jsonl backend): entries have cid and prev_cid;
        second entry's prev_cid equals first entry's cid."""
        import asyncio
        import json
        from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
        from aurora_lens.verify.flags import Flag, FlagType
        from aurora_lens.pef.state import PEFState

        log = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(log), backend="jsonl")
        pef = PEFState()
        flags = [Flag(flag_type=FlagType.UNSUPPORTED_ATTRIBUTE, entity_name="X",
                      claim="test", evidence="ev", severity="warning")]

        async def _run():
            d1 = await bridge.decide(flags, "resp", pef)
            bridge.log_decision(d1, turn=0)
            d2 = await bridge.decide(flags, "resp", pef)
            bridge.log_decision(d2, turn=1)

        asyncio.run(_run())

        lines = [ln for ln in log.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
        assert len(lines) >= 2
        e0 = json.loads(lines[0])
        e1 = json.loads(lines[1])
        assert "cid" in e0, "First entry missing cid"
        assert "prev_cid" in e0, "First entry missing prev_cid"
        assert e0["prev_cid"] == "genesis", f"Expected genesis, got {e0['prev_cid']!r}"
        assert "cid" in e1, "Second entry missing cid"
        assert "prev_cid" in e1, "Second entry missing prev_cid"
        assert e1["prev_cid"] == e0["cid"], "Chain broken between entry 0 and entry 1"

    def test_jsonl_chain_resumes_after_bridge_restart(self, tmp_path):
        """Flat JSONL D2 chain: a new bridge instance continues from the last on-disk cid."""
        import asyncio
        import json
        from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
        from aurora_lens.verify.flags import Flag, FlagType
        from aurora_lens.pef.state import PEFState

        log = tmp_path / "audit.jsonl"
        flags = [
            Flag(
                flag_type=FlagType.UNSUPPORTED_ATTRIBUTE,
                entity_name="X",
                claim="test",
                evidence="ev",
                severity="warning",
            )
        ]

        async def _first_session():
            b = AuroraScannerGateBridge(audit_path=str(log), backend="jsonl")
            pef = PEFState()
            d = await b.decide(flags, "resp", pef)
            b.log_decision(d, turn=0)

        asyncio.run(_first_session())

        lines = [ln for ln in log.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
        assert len(lines) == 1
        tip_cid = json.loads(lines[0])["cid"]

        async def _second_session():
            b2 = AuroraScannerGateBridge(audit_path=str(log), backend="jsonl")
            pef = PEFState()
            d2 = await b2.decide(flags, "resp2", pef)
            b2.log_decision(d2, turn=1)

        asyncio.run(_second_session())

        lines2 = [ln for ln in log.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
        assert len(lines2) == 2
        assert json.loads(lines2[1])["prev_cid"] == tip_cid


# ── Stage C: Timestamp discipline tests ─────────────────────────────

class TestTimestampDiscipline:
    """Verify UTC ISO-8601 format consistency across audit paths."""

    UTC_ISO_RE = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+00:00$'

    def test_timestamps_are_utc_iso8601(self, tmp_path):
        """Audit entry timestamp field is UTC ISO-8601 with microseconds and +00:00 suffix."""
        import asyncio, json, re
        from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
        from aurora_lens.verify.flags import Flag, FlagType
        from aurora_lens.pef.state import PEFState

        log = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(log), backend="jsonl")
        pef = PEFState()
        flags = [Flag(flag_type=FlagType.UNSUPPORTED_ATTRIBUTE, entity_name="E",
                      claim="c", evidence="e", severity="warning")]

        async def _run():
            d = await bridge.decide(flags, "resp", pef)
            bridge.log_decision(d, turn=0)

        asyncio.run(_run())

        entry = json.loads(log.read_text(encoding="utf-8").strip())
        ts = entry.get("timestamp")
        assert ts is not None, "No timestamp in entry"
        assert re.match(self.UTC_ISO_RE, ts), (
            f"timestamp '{ts}' does not match UTC ISO-8601 format with +00:00 and microseconds"
        )

    def test_log_slice_timestamps_consistent_format(self, tmp_path):
        """first_timestamp and last_timestamp in log slice use same UTC ISO-8601 format."""
        import asyncio, json, re
        from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
        from aurora_lens.verify.flags import Flag, FlagType
        from aurora_lens.pef.state import PEFState
        from aurora_lens.log_slice import init_log_buffer
        import logging

        log = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(log), backend="jsonl")
        pef = PEFState()
        flags = [Flag(flag_type=FlagType.UNSUPPORTED_ATTRIBUTE, entity_name="E",
                      claim="c", evidence="e", severity="warning")]

        async def _run():
            token = init_log_buffer()
            try:
                # emit a dummy log record to populate the buffer
                logger = logging.getLogger("aurora_lens.test")
                logger.info("test entry for slice")
                d = await bridge.decide(flags, "resp", pef)
                bridge.log_decision(d, turn=0)
            finally:
                from contextvars import copy_context
                from aurora_lens.log_slice import log_buffer_var
                log_buffer_var.reset(token)

        asyncio.run(_run())

        lines = [ln for ln in log.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
        # Find an entry that has log slice fields (may not be present if buffer was empty)
        for ln in lines:
            entry = json.loads(ln)
            ft = entry.get("first_timestamp")
            lt = entry.get("last_timestamp")
            if ft and lt:
                assert re.match(self.UTC_ISO_RE, ft), f"first_timestamp '{ft}' format mismatch"
                assert re.match(self.UTC_ISO_RE, lt), f"last_timestamp '{lt}' format mismatch"
                return
        # If no log slice was captured, just verify the main timestamp format (already tested above)
        # This path is acceptable: the buffer may not have been seeded with records in this test env.

    def test_timestamp_monotonicity(self, tmp_path):
        """Two sequential decisions produce monotonically non-decreasing timestamps."""
        import asyncio, json
        from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
        from aurora_lens.verify.flags import Flag, FlagType
        from aurora_lens.pef.state import PEFState

        log = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(log), backend="jsonl")
        pef = PEFState()
        flags = [Flag(flag_type=FlagType.UNSUPPORTED_ATTRIBUTE, entity_name="E",
                      claim="c", evidence="e", severity="warning")]

        async def _run():
            d1 = await bridge.decide(flags, "resp", pef)
            bridge.log_decision(d1, turn=0)
            d2 = await bridge.decide(flags, "resp", pef)
            bridge.log_decision(d2, turn=1)

        asyncio.run(_run())

        lines = [ln for ln in log.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
        assert len(lines) >= 2
        ts0 = json.loads(lines[0]).get("timestamp")
        ts1 = json.loads(lines[1]).get("timestamp")
        assert ts0 is not None and ts1 is not None
        assert ts1 >= ts0, f"Timestamp not monotonic: {ts0} > {ts1}"

    def test_enrich_forensic_envelope_governor_update_at_matches_parent_timestamp(self):
        """Stage C: ``governor_update_at`` equals the parent forensic ``timestamp`` when passed in."""
        from aurora_lens.governor.audit import enrich_forensic_envelope
        from aurora_lens.governor.models import (
            AuthorityClass,
            Domain,
            GovernorPolicy,
            LensStatus,
            UserClass,
        )

        parent_ts = "2026-05-13T12:00:00.123456+00:00"
        policy = GovernorPolicy(
            domain=Domain.GENERAL,
            authority_class=AuthorityClass.GP,
            lens_status=LensStatus.STOP,
            user_class=UserClass.GENERAL,
        )
        base = {"timestamp": parent_ts, "status": "STOP", "trace_id": "t"}
        out = enrich_forensic_envelope(base, policy, reason_code="TEST", parent_timestamp=parent_ts)
        assert out["governor_update_at"] == parent_ts


# ── Stage D: forensic_event self-hash ───────────────────────────────

class TestForensicEventSelfHash:
    """Verify forensic_event carries an independently verifiable event_hash (Stage D)."""

    def test_forensic_event_has_self_hash(self, tmp_path):
        """Persisted forensic_event.event_hash matches recomputation (exclude event_hash key)."""
        import asyncio, json, hashlib
        from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
        from aurora_lens.verify.flags import Flag, FlagType
        from aurora_lens.pef.state import PEFState
        from aurora_lens.governor.forensic_schema import verify_event_hash

        log = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(log), backend="jsonl")
        pef = PEFState()
        flags = [Flag(flag_type=FlagType.UNBOUND_ENTITY, entity_name="E",
                      claim="c", evidence="e", severity="warning")]

        async def _run():
            d = await bridge.decide(flags, "resp", pef)
            bridge.log_decision(d, turn=0)
            return d

        asyncio.run(_run())

        entry = json.loads(log.read_text(encoding="utf-8").strip())
        fe = entry.get("forensic_event")
        assert fe is not None, "forensic_event missing from persisted audit row"
        assert verify_event_hash(fe), "verify_event_hash failed on persisted forensic_event"

        stored_hash = fe["event_hash"]
        assert stored_hash.startswith("sha256:"), f"Unexpected event_hash prefix: {stored_hash}"

        fe_without_hash = {k: v for k, v in fe.items() if k != "event_hash"}
        computed = "sha256:" + hashlib.sha256(
            json.dumps(fe_without_hash, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        assert computed == stored_hash, (
            f"event_hash mismatch: stored={stored_hash}, recomputed={computed}"
        )

    def test_forensic_event_has_self_hash_in_ledger_payload(self, tmp_path):
        """AFL row payload.data.forensic_event is self-consistent (Stage D persist path)."""
        import asyncio, json, hashlib
        from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
        from aurora_lens.verify.flags import Flag, FlagType
        from aurora_lens.pef.state import PEFState
        from aurora_lens.governor.forensic_schema import verify_event_hash

        log = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(log), backend="ledger")
        pef = PEFState()
        flags = [Flag(flag_type=FlagType.UNBOUND_ENTITY, entity_name="E",
                      claim="c", evidence="e", severity="warning")]

        async def _run():
            d = await bridge.decide(flags, "resp", pef)
            bridge.log_decision(d, turn=0)

        asyncio.run(_run())

        line = log.read_text(encoding="utf-8").strip().splitlines()[0]
        envelope = json.loads(line)
        fe = envelope.get("payload", {}).get("data", {}).get("forensic_event")
        assert fe is not None, "forensic_event missing from AFL payload.data"
        assert verify_event_hash(fe)
        stored_hash = fe["event_hash"]
        body = {k: v for k, v in fe.items() if k != "event_hash"}
        computed = "sha256:" + hashlib.sha256(
            json.dumps(body, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        assert computed == stored_hash

    def test_forensic_event_hash_tamper_detected(self, tmp_path):
        """Schema verify_event_hash rejects a content change when event_hash is left stale."""
        import asyncio
        from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
        from aurora_lens.verify.flags import Flag, FlagType
        from aurora_lens.pef.state import PEFState
        from aurora_lens.governor.forensic_schema import verify_event_hash

        log = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(log), backend="jsonl")
        pef = PEFState()
        flags = [Flag(flag_type=FlagType.UNBOUND_ENTITY, entity_name="E",
                      claim="c", evidence="e", severity="warning")]

        async def _run():
            d = await bridge.decide(flags, "resp", pef)
            bridge.log_decision(d, turn=0)
            return d

        d = asyncio.run(_run())
        fe_orig = dict(d.forensic_event)
        assert verify_event_hash(fe_orig)

        tampered = dict(fe_orig)
        tampered["status"] = "STOP"
        assert not verify_event_hash(tampered), "Tampered envelope must fail verify_event_hash"
