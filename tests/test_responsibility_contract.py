"""Responsibility boundary contract tests.

Verifies two architectural invariants:
  1. Lens owns admissibility — it decides *what* is inadmissible (flags).
  2. The Governor owns continuation only — it decides *how* to respond (pathway rendering).

And the critical safety property:
  3. Blocked determinations cannot be smuggled back via continuation text.

These tests are structural: they verify the shape of data flowing through
the system, not the correctness of any one policy or flag type.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
from pathlib import Path

import pytest

from aurora_lens.govern.bridge import (
    AmbiguityRendererContext,
    BuiltinBridge,
    HardStopRendererContext,
    RefusalRendererContext,
    _build_ambiguity_context,
    _build_hard_stop_context,
    _build_refusal_context,
    _render_ask_disambiguate,
    _render_ask_missing_fact,
    _render_handoff_summary,
    _render_refuse_escalate_pro,
    _render_refuse_explain_redirect,
    _render_stop_escalate,
    _render_stop_escalate_emergency,
    _render_stop_forensic,
    _render_stop_redirect_qualified,
    _render_stop_refuse_clean,
    _render_stop_supportive_deescalate,
    _render_stop_terminal,
    build_forensic_event,
    enforce,
)
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.govern.policy import DEFAULT_STRICT
from aurora_lens.pef.state import PEFState
from aurora_lens.verify.flags import Flag, FlagType


# ── Helpers ──────────────────────────────────────────────────────────

def _flag(ft: FlagType, severity: str = "error", evidence: str = "secret blocked content XYZ-999") -> Flag:
    return Flag(
        flag_type=ft,
        entity_name="TestEntity",
        claim="test claim about forbidden thing",
        evidence=evidence,
        severity=severity,
    )


def _decision(
    action: InterventionAction,
    ft: FlagType = FlagType.PERSONALIZED_MEDICAL_ADVICE,
    pathway_id: str | None = "P_STOP_TERMINAL",
    original_response: str = "Take 500mg ibuprofen every 4 hours for your specific condition",
    resource: str | None = None,
    interaction_open: bool = False,
    commitment_closed: bool = True,
    output_mode: str | None = "terminal_stop",
    forensic_obligations: list[str] | None = None,
    resolution_mode: str | None = None,
) -> GovernanceDecision:
    return GovernanceDecision(
        action=action,
        flags=[_flag(ft)],
        rationale=f"{action.name}: test",
        policy="strict",
        pathway_id=pathway_id,
        original_response=original_response,
        resource=resource,
        interaction_open=interaction_open,
        commitment_closed=commitment_closed,
        output_mode=output_mode,
        forensic_obligations=forensic_obligations or [],
        resolution_mode=resolution_mode,
    )


BLOCKED_CONTENT = "Take 500mg ibuprofen every 4 hours for your specific condition"
BLOCKED_SUBSTRINGS = ["500mg", "ibuprofen", "every 4 hours", "your specific condition"]


# All flag types that produce non-PASS outcomes under DEFAULT_STRICT
_HARD_STOP_FLAG_TYPES = [
    FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
    FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
    FlagType.NUMERIC_MEDICAL_INSTRUCTION,
    FlagType.EMERGENCY_TRIAGE_GUIDANCE,
    FlagType.SELF_HARM_INSTRUCTION,
    FlagType.PERSONALIZED_MEDICAL_ADVICE,
    FlagType.PERSONALIZED_LEGAL_ADVICE,
    FlagType.PERSONALIZED_FINANCIAL_ADVICE,
    FlagType.ILLEGAL_INSTRUCTION,
    FlagType.TARGETED_DEFAMATION,
    FlagType.SENSITIVE_PII_EXPOSURE,
    FlagType.UNRESOLVED_STATE_TRANSITION,
]


# ═════════════════════════════════════════════════════════════════════
# PART 1 — Responsibility boundary: Lens owns admissibility
# ═════════════════════════════════════════════════════════════════════

class TestLensOwnsAdmissibility:
    """Lens determines what is inadmissible. The Governor never re-examines
    the response text for admissibility. It only receives flags."""

    def test_bridge_decide_does_not_inspect_response_text(self):
        """BuiltinBridge.decide() ignores response_text — its decision
        depends only on flags and policy. Two calls with different response_text
        but identical flags produce identical actions."""
        bridge = BuiltinBridge(policy=DEFAULT_STRICT)
        pef = PEFState()
        flags = [_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE)]

        d1 = asyncio.run(bridge.decide(flags, "response A with medical advice", pef))
        d2 = asyncio.run(bridge.decide(flags, "completely different text", pef))

        assert d1.action == d2.action
        assert d1.pathway_id == d2.pathway_id

    def test_bridge_decide_with_no_flags_always_passes(self):
        """With empty flags, bridge always returns PASS regardless of response text."""
        bridge = BuiltinBridge(policy=DEFAULT_STRICT)
        pef = PEFState()

        d = asyncio.run(bridge.decide([], "Take 500mg of ibuprofen", pef))
        assert d.action == InterventionAction.PASS

    def test_flag_production_is_lens_responsibility(self):
        """The bridge receives pre-built flags. It does not have a checker,
        structural governor, or any admissibility evaluator."""
        bridge = BuiltinBridge(policy=DEFAULT_STRICT)
        assert not hasattr(bridge, "_checker")
        assert not hasattr(bridge, "_governor")
        assert not hasattr(bridge, "check")
        assert not hasattr(bridge, "evaluate")


# ═════════════════════════════════════════════════════════════════════
# PART 2 — Responsibility boundary: Governor owns continuation only
# ═════════════════════════════════════════════════════════════════════

class TestScannerGateOwnsContinuation:
    """The Governor maps flags to continuation pathways and renders
    deterministic text. It never produces admissibility decisions."""

    def test_enforce_is_deterministic(self):
        """Same decision + same model_output always produces same text."""
        d = _decision(InterventionAction.HARD_STOP)
        r1 = enforce(d, BLOCKED_CONTENT)
        r2 = enforce(d, BLOCKED_CONTENT)
        assert r1 == r2

    def test_enforce_pass_returns_model_output_verbatim(self):
        """PASS and SOFT_CORRECT return model output unchanged — no injection."""
        d_pass = GovernanceDecision(
            action=InterventionAction.PASS, flags=[], rationale="clean"
        )
        d_soft = GovernanceDecision(
            action=InterventionAction.SOFT_CORRECT, flags=[], rationale="annotate"
        )
        assert enforce(d_pass, "hello") == "hello"
        assert enforce(d_soft, "hello") == "hello"

    def test_enforce_hard_stop_does_not_return_model_output(self):
        """HARD_STOP must never return the original model output."""
        d = _decision(InterventionAction.HARD_STOP)
        result = enforce(d, BLOCKED_CONTENT)
        assert result != BLOCKED_CONTENT

    def test_enforce_force_revise_does_not_return_model_output(self):
        """FORCE_REVISE must never return the original model output."""
        d = _decision(
            InterventionAction.FORCE_REVISE,
            pathway_id="P_REFUSE_EXPLAIN_REDIRECT",
            output_mode="explain_redirect",
        )
        result = enforce(d, BLOCKED_CONTENT)
        assert result != BLOCKED_CONTENT

    def test_enforce_contain_does_not_return_model_output(self):
        """CONTAIN (ASK pathway) must never return the original model output."""
        d = _decision(
            InterventionAction.CONTAIN,
            ft=FlagType.UNRESOLVED_REFERENT,
            pathway_id="P_ASK_DISAMBIGUATE",
            output_mode="ask_disambiguate",
        )
        result = enforce(d, BLOCKED_CONTENT)
        assert result != BLOCKED_CONTENT

    def test_all_non_pass_pathways_produce_nonempty_text(self):
        """Every non-PASS action produces non-empty continuation text."""
        pathways = [
            (InterventionAction.CONTAIN, "P_ASK_DISAMBIGUATE", FlagType.UNRESOLVED_REFERENT),
            (InterventionAction.CONTAIN, "P_ASK_MISSING_FACT", FlagType.UNRESOLVED_REFERENT),
            (InterventionAction.FORCE_REVISE, "P_REFUSE_EXPLAIN_REDIRECT", FlagType.PERSONALIZED_MEDICAL_ADVICE),
            (InterventionAction.FORCE_REVISE, "P_REFUSE_ESCALATE_PRO", FlagType.PERSONALIZED_MEDICAL_ADVICE),
            (InterventionAction.FORCE_REVISE, "P_HANDOFF_SUMMARY", FlagType.PERSONALIZED_MEDICAL_ADVICE),
            (InterventionAction.HARD_STOP, "P_STOP_TERMINAL", FlagType.PERSONALIZED_MEDICAL_ADVICE),
            (InterventionAction.HARD_STOP, "P_STOP_FORENSIC", FlagType.SENSITIVE_PII_EXPOSURE),
            (InterventionAction.HARD_STOP, "P_STOP_ESCALATE", FlagType.SELF_HARM_INSTRUCTION),
            (InterventionAction.HARD_STOP, "P_STOP_ESCALATE_EMERGENCY", FlagType.EMERGENCY_TRIAGE_GUIDANCE),
            (InterventionAction.HARD_STOP, "P_STOP_SUPPORTIVE_DEESCALATE", FlagType.SELF_HARM_INSTRUCTION),
            (InterventionAction.HARD_STOP, "P_STOP_REDIRECT_QUALIFIED", FlagType.MEDICAL_DOSAGE_RECOMMENDATION),
            (InterventionAction.HARD_STOP, "P_STOP_REFUSE_CLEAN", FlagType.ILLEGAL_INSTRUCTION),
        ]
        for action, pathway, ft in pathways:
            d = _decision(action, ft=ft, pathway_id=pathway)
            result = enforce(d, BLOCKED_CONTENT)
            assert result, f"Empty text for {pathway}"
            assert len(result) > 10, f"Suspiciously short text for {pathway}: {result!r}"


# ═════════════════════════════════════════════════════════════════════
# PART 3 — Renderer context stripping: no blocked content leaks
# ═════════════════════════════════════════════════════════════════════

class TestRendererContextStripping:
    """Renderer contexts must NOT carry evidence, claim, original_response,
    or any content from the blocked determination."""

    def test_ambiguity_context_has_no_evidence(self):
        d = _decision(InterventionAction.CONTAIN, pathway_id="P_ASK_DISAMBIGUATE")
        ctx = _build_ambiguity_context(d)
        assert not hasattr(ctx, "evidence")
        assert not hasattr(ctx, "claim")
        assert not hasattr(ctx, "original_response")
        assert not hasattr(ctx, "response_text")

    def test_refusal_context_has_no_evidence(self):
        d = _decision(InterventionAction.FORCE_REVISE, pathway_id="P_REFUSE_EXPLAIN_REDIRECT")
        ctx = _build_refusal_context(d)
        assert not hasattr(ctx, "evidence")
        assert not hasattr(ctx, "claim")
        assert not hasattr(ctx, "original_response")
        assert not hasattr(ctx, "response_text")

    def test_hard_stop_context_has_no_evidence(self):
        d = _decision(InterventionAction.HARD_STOP, pathway_id="P_STOP_TERMINAL")
        ctx = _build_hard_stop_context(d)
        assert not hasattr(ctx, "evidence")
        assert not hasattr(ctx, "claim")
        assert not hasattr(ctx, "original_response")
        assert not hasattr(ctx, "response_text")

    def test_ambiguity_context_only_carries_permitted_fields(self):
        """AmbiguityRendererContext fields are candidates/missing_fields/interaction_open
        plus optional routing fields for the UNRESOLVED_REFERENT path."""
        d = _decision(InterventionAction.CONTAIN, pathway_id="P_ASK_DISAMBIGUATE")
        ctx = _build_ambiguity_context(d)
        allowed = {
            "candidates", "missing_fields", "interaction_open",
            "flag_type", "ambiguous_token", "flag_claim", "flag_evidence", "original_question",
        }
        actual = {f for f in ctx.__dataclass_fields__}
        assert actual <= allowed, f"Unexpected fields: {actual - allowed}"

    def test_refusal_context_only_carries_permitted_fields(self):
        """RefusalRendererContext carries only permitted dispatch identity fields."""
        d = _decision(InterventionAction.FORCE_REVISE, pathway_id="P_REFUSE_EXPLAIN_REDIRECT")
        ctx = _build_refusal_context(d)
        allowed = {
            "flag_type",
            "resource",
            "interaction_open",
            "rule_id",
            "domain",
            "request_domain",
            "reason_code",
            "continuation_type",
            "user_facing_template_key",
        }
        actual = {f for f in ctx.__dataclass_fields__}
        assert actual == allowed, f"Unexpected fields: {actual - allowed}"

    def test_hard_stop_context_only_carries_permitted_fields(self):
        """HardStopRendererContext fields include defamation_role for TARGETED_DEFAMATION routing."""
        d = _decision(InterventionAction.HARD_STOP, pathway_id="P_STOP_TERMINAL")
        ctx = _build_hard_stop_context(d)
        allowed = {"flag_type", "resource", "interaction_open", "domain", "defamation_role", "pre_llm"}
        actual = {f for f in ctx.__dataclass_fields__}
        assert actual == allowed, f"Unexpected fields: {actual - allowed}"

    def test_frozen_contexts_cannot_be_mutated(self):
        """All renderer contexts are frozen dataclasses."""
        d = _decision(InterventionAction.HARD_STOP)
        ctx = _build_hard_stop_context(d)
        with pytest.raises(AttributeError):
            ctx.flag_type = FlagType.ILLEGAL_INSTRUCTION  # type: ignore[misc]


# ═════════════════════════════════════════════════════════════════════
# PART 4 — Anti-smuggling: blocked content cannot leak into continuation
# ═════════════════════════════════════════════════════════════════════

class TestAntiSmuggling:
    """No substring of the blocked original response may appear in the
    continuation text for any flag type under any pathway."""

    @pytest.mark.parametrize("ft", _HARD_STOP_FLAG_TYPES, ids=lambda ft: ft.name)
    def test_hard_stop_does_not_leak_blocked_content(self, ft: FlagType):
        """For each hard-stop flag type, enforce() must not leak any
        substring of the blocked original response."""
        d = _decision(
            InterventionAction.HARD_STOP,
            ft=ft,
            original_response=BLOCKED_CONTENT,
        )
        result = enforce(d, BLOCKED_CONTENT)
        for substr in BLOCKED_SUBSTRINGS:
            assert substr.lower() not in result.lower(), (
                f"Blocked content leaked for {ft.name}: found '{substr}' in: {result!r}"
            )

    @pytest.mark.parametrize("ft", _HARD_STOP_FLAG_TYPES, ids=lambda ft: ft.name)
    def test_hard_stop_does_not_return_original_verbatim(self, ft: FlagType):
        """No hard-stop pathway returns the original response verbatim."""
        d = _decision(
            InterventionAction.HARD_STOP,
            ft=ft,
            original_response=BLOCKED_CONTENT,
        )
        result = enforce(d, BLOCKED_CONTENT)
        assert result != BLOCKED_CONTENT

    def test_illegal_instruction_no_workaround_adjacency(self):
        """ILLEGAL_INSTRUCTION must not suggest alternatives, workarounds,
        or 'however' framing."""
        d = _decision(
            InterventionAction.HARD_STOP,
            ft=FlagType.ILLEGAL_INSTRUCTION,
            pathway_id="P_STOP_REFUSE_CLEAN",
        )
        result = enforce(d, "How to pick a lock on someone else's house")
        workaround_patterns = [
            r"\bhowever\b",
            r"\balternative\b",
            r"\binstead\b",
            r"\byou could\b",
            r"\btry\b",
            r"\bconsider\b",
            r"\blocksmith\b",
        ]
        for pat in workaround_patterns:
            assert not re.search(pat, result, re.IGNORECASE), (
                f"Workaround-adjacent language for ILLEGAL_INSTRUCTION: pattern '{pat}' in: {result!r}"
            )

    def test_targeted_defamation_no_content_recovery(self):
        """TARGETED_DEFAMATION must not echo or paraphrase the defamatory claim."""
        original = "John Smith is a convicted fraudster who steals from charities"
        d = _decision(
            InterventionAction.HARD_STOP,
            ft=FlagType.TARGETED_DEFAMATION,
            pathway_id="P_STOP_REFUSE_CLEAN",
            original_response=original,
        )
        result = enforce(d, original)
        for name in ["John Smith", "fraudster", "steals", "charities"]:
            assert name.lower() not in result.lower(), (
                f"Defamatory content recovered: found '{name}' in: {result!r}"
            )

    def test_sensitive_pii_no_data_leak(self):
        """SENSITIVE_PII_EXPOSURE must not echo any PII from the blocked response."""
        original = "Bob's SSN is 123-45-6789 and his address is 42 Elm Street"
        d = _decision(
            InterventionAction.HARD_STOP,
            ft=FlagType.SENSITIVE_PII_EXPOSURE,
            pathway_id="P_STOP_FORENSIC",
            original_response=original,
        )
        result = enforce(d, original)
        for pii in ["123-45-6789", "42 Elm", "Bob"]:
            assert pii not in result, (
                f"PII leaked: found '{pii}' in: {result!r}"
            )

    def test_medical_dosage_no_numbers_leak(self):
        """MEDICAL_DOSAGE_RECOMMENDATION must not echo dosage numbers."""
        original = "Give 10mg/kg amoxicillin every 8 hours for 10 days"
        d = _decision(
            InterventionAction.HARD_STOP,
            ft=FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
            pathway_id="P_STOP_REDIRECT_QUALIFIED",
            original_response=original,
        )
        result = enforce(d, original)
        for dose in ["10mg", "amoxicillin", "every 8 hours", "10 days"]:
            assert dose.lower() not in result.lower(), (
                f"Dosage leaked: found '{dose}' in: {result!r}"
            )

    def test_renderer_input_is_context_not_decision(self):
        """enforce() never passes GovernanceDecision directly to a renderer.
        It always builds a typed context first. Verify by checking that the
        renderer functions accept only their typed context, not GovernanceDecision."""
        import inspect
        renderers = [
            _render_ask_disambiguate,
            _render_ask_missing_fact,
            _render_refuse_explain_redirect,
            _render_refuse_escalate_pro,
            _render_handoff_summary,
            _render_stop_terminal,
            _render_stop_forensic,
            _render_stop_escalate,
            _render_stop_escalate_emergency,
            _render_stop_supportive_deescalate,
            _render_stop_redirect_qualified,
            _render_stop_refuse_clean,
        ]
        for renderer in renderers:
            sig = inspect.signature(renderer)
            param_types = [
                p.annotation for p in sig.parameters.values()
                if p.annotation != inspect.Parameter.empty
            ]
            for t in param_types:
                assert t is not GovernanceDecision, (
                    f"{renderer.__name__} accepts GovernanceDecision — must use typed context"
                )


# ═════════════════════════════════════════════════════════════════════
# PART 5 — Forensic envelope hardening for ASK / REFUSE / STOP
# ═════════════════════════════════════════════════════════════════════

class TestForensicEnvelopeHardening:
    """The forensic_event must record pathway metadata and content hashes
    for every non-ADMIT outcome."""

    def _build_event(
        self,
        action: InterventionAction = InterventionAction.HARD_STOP,
        ft: FlagType = FlagType.PERSONALIZED_MEDICAL_ADVICE,
        pathway_id: str = "P_STOP_TERMINAL",
        original_response: str = BLOCKED_CONTENT,
        governed_response: str = "I can't make that determination.",
    ) -> dict:
        d = _decision(action, ft=ft, pathway_id=pathway_id, original_response=original_response)
        d.governed_response = governed_response
        return build_forensic_event(
            d,
            pre_llm=False,
            pef_snapshot={"entities": {}, "turn": 1},
            trace_id="test-trace-001",
            timestamp="2026-03-18T12:00:00+00:00",
            audit_id="cid:test:001",
        )

    def test_forensic_event_has_status_field(self):
        """ASK / REFUSE / STOP status string is present."""
        for action, expected in [
            (InterventionAction.CONTAIN, "ASK"),
            (InterventionAction.FORCE_REVISE, "REFUSE"),
            (InterventionAction.HARD_STOP, "STOP"),
        ]:
            event = self._build_event(action=action)
            assert event["status"] == expected

    def test_forensic_event_has_pathway_metadata(self):
        """pathway_id, output_mode, commitment_closed, interaction_open are recorded."""
        event = self._build_event(pathway_id="P_STOP_TERMINAL")
        assert event["pathway_id"] == "P_STOP_TERMINAL"
        assert event["output_mode"] == "terminal_stop"
        assert event["commitment_closed"] is True
        assert event["interaction_open"] is False

    def test_forensic_event_records_forensic_obligations(self):
        d = _decision(
            InterventionAction.HARD_STOP,
            pathway_id="P_STOP_FORENSIC",
            forensic_obligations=["notify_privacy_officer"],
        )
        d.governed_response = "I'm not able to surface that information."
        event = build_forensic_event(
            d,
            pre_llm=False,
            pef_snapshot={"entities": {}, "turn": 1},
            trace_id="t",
            timestamp="2026-03-18T12:00:00+00:00",
        )
        assert event["forensic_obligations"] == ["notify_privacy_officer"]

    def test_forensic_event_has_blocked_response_hash(self):
        """The blocked (original) response is hashed but not stored in plaintext."""
        event = self._build_event(original_response="sensitive content here")
        expected = "sha256:" + hashlib.sha256(b"sensitive content here").hexdigest()
        assert event["blocked_response_hash"] == expected
        # The plaintext must NOT appear anywhere in the event
        event_json = json.dumps(event)
        assert "sensitive content here" not in event_json

    def test_forensic_event_has_governed_response_hash(self):
        """The governed (rendered) response is hashed."""
        governed = "I can't make that determination."
        event = self._build_event(governed_response=governed)
        expected = "sha256:" + hashlib.sha256(governed.encode()).hexdigest()
        assert event["governed_response_hash"] == expected

    def test_forensic_event_has_state_hash(self):
        event = self._build_event()
        assert event["state_hash"] is not None
        assert len(event["state_hash"]) == 64  # SHA-256 hex

    def test_forensic_event_self_hash_is_valid(self):
        """event_hash covers all fields except itself."""
        event = self._build_event()
        stored_hash = event["event_hash"]
        # Recompute: remove event_hash, serialize, hash
        check = {k: v for k, v in event.items() if k != "event_hash"}
        expected = "sha256:" + hashlib.sha256(
            json.dumps(check, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        assert stored_hash == expected

    def test_forensic_event_self_hash_detects_tampering(self):
        event = self._build_event()
        original_hash = event["event_hash"]
        event["status"] = "TAMPERED"
        check = {k: v for k, v in event.items() if k != "event_hash"}
        recomputed = "sha256:" + hashlib.sha256(
            json.dumps(check, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        assert recomputed != original_hash

    def test_forensic_event_no_original_response_plaintext(self):
        """The forensic event must never contain the original (blocked) response
        as a plaintext string in any field."""
        original = "Here is extremely dangerous medical advice: take 50 pills"
        event = self._build_event(original_response=original)
        for key, value in event.items():
            if isinstance(value, str) and key != "blocked_response_hash":
                assert original not in value, (
                    f"Original response found in plaintext in event['{key}']"
                )

    def test_ask_path_produces_forensic_event(self):
        """CONTAIN (ASK) outcomes also produce forensic events with pathway metadata."""
        event = self._build_event(
            action=InterventionAction.CONTAIN,
            ft=FlagType.UNRESOLVED_REFERENT,
            pathway_id="P_ASK_DISAMBIGUATE",
        )
        assert event["status"] == "ASK"
        assert event["pathway_id"] == "P_ASK_DISAMBIGUATE"

    def test_refuse_path_produces_forensic_event(self):
        """FORCE_REVISE (REFUSE) outcomes produce forensic events with pathway metadata."""
        event = self._build_event(
            action=InterventionAction.FORCE_REVISE,
            ft=FlagType.PERSONALIZED_MEDICAL_ADVICE,
            pathway_id="P_REFUSE_EXPLAIN_REDIRECT",
        )
        assert event["status"] == "REFUSE"
        assert event["pathway_id"] == "P_REFUSE_EXPLAIN_REDIRECT"

    def test_stop_path_produces_forensic_event(self):
        """HARD_STOP (STOP) outcomes produce forensic events with pathway metadata."""
        event = self._build_event(
            action=InterventionAction.HARD_STOP,
            ft=FlagType.ILLEGAL_INSTRUCTION,
            pathway_id="P_STOP_REFUSE_CLEAN",
        )
        assert event["status"] == "STOP"
        assert event["pathway_id"] == "P_STOP_REFUSE_CLEAN"

    def test_resolution_mode_recorded(self):
        d = _decision(
            InterventionAction.HARD_STOP,
            resolution_mode="policy_matrix_l0",
        )
        d.governed_response = "refused"
        event = build_forensic_event(
            d,
            pre_llm=False,
            pef_snapshot={"entities": {}, "turn": 1},
            trace_id="t",
            timestamp="2026-03-18T12:00:00+00:00",
        )
        assert event["resolution_mode"] == "policy_matrix_l0"


# ═════════════════════════════════════════════════════════════════════
# PART 6 — Audit log integration: forensic event flows to audit entry
# ═════════════════════════════════════════════════════════════════════

class TestForensicEventAuditIntegration:
    """When a non-PASS decision is logged, the forensic event is embedded
    in the audit entry and contains pathway metadata."""

    def test_builtin_bridge_writes_forensic_event_with_pathway(self):
        """BuiltinBridge._log_decision writes forensic_event with pathway_id."""
        with tempfile.TemporaryDirectory() as td:
            audit_path = Path(td) / "audit.jsonl"
            bridge = BuiltinBridge(
                policy=DEFAULT_STRICT,
                audit_path=str(audit_path),
            )
            pef = PEFState()
            flags = [_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE)]
            d = asyncio.run(bridge.decide(flags, BLOCKED_CONTENT, pef))
            d.original_response = BLOCKED_CONTENT
            d.governed_response = "I can't make that determination."
            d.corrected_response = "I can't make that determination."
            bridge.log_decision(
                d,
                turn=1,
                pef_context="test context",
                pef_snapshot={"entities": {}, "turn": 1},
            )

            entries = [
                json.loads(line)
                for line in audit_path.read_text().strip().splitlines()
            ]
            assert len(entries) >= 1
            entry = entries[-1]
            fe = entry.get("forensic_event")
            assert fe is not None, "No forensic_event in audit entry"
            assert fe["status"] == "STOP"
            assert "pathway_id" in fe
            assert "blocked_response_hash" in fe
            assert "governed_response_hash" in fe
            assert "event_hash" in fe

    def test_forensic_event_pathway_matches_decision(self):
        """The forensic event's pathway_id matches the decision's pathway_id."""
        with tempfile.TemporaryDirectory() as td:
            audit_path = Path(td) / "audit.jsonl"
            bridge = BuiltinBridge(
                policy=DEFAULT_STRICT,
                audit_path=str(audit_path),
            )
            pef = PEFState()
            flags = [_flag(FlagType.PERSONALIZED_MEDICAL_ADVICE)]
            d = asyncio.run(bridge.decide(flags, BLOCKED_CONTENT, pef))
            d.original_response = BLOCKED_CONTENT
            d.governed_response = "refused"
            d.corrected_response = "refused"
            bridge.log_decision(d, turn=1, pef_context="ctx", pef_snapshot={"turn": 1})

            entry = json.loads(audit_path.read_text().strip().splitlines()[-1])
            fe = entry["forensic_event"]
            assert fe["pathway_id"] == d.pathway_id
