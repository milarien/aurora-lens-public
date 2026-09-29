"""Strict-policy admissibility basis: empty flags must not default to PASS."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from aurora_lens.context import domain_var
from aurora_lens.govern.bridge import BuiltinBridge
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.govern.policy import DEFAULT_MODERATE, DEFAULT_STRICT
from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult, Span
from aurora_lens.lens import Lens, LensConfig
from aurora_lens.pef.state import PEFState
from aurora_lens.verify.consequence_intent import (
    LOW_RISK_CONVERSATIONAL,
    POLICY_PROFILE_REQUIRED,
    UNCLASSIFIED_CONSEQUENCE_INTENT,
    classify_consequence_intent,
)
from aurora_lens.verify.flags import Flag, FlagType

from scripts.verify_strict_admissibility_basis import (
    assert_strict_pass_audit_invariants,
    format_pass_summary,
)


class MockAdapter(LLMAdapter):
    def __init__(self, responses: list[str]):
        self._responses = responses
        self._call_count = 0

    async def generate(self, messages: list[dict[str, str]], **kwargs) -> AdapterResponse:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return AdapterResponse(text=self._responses[idx], model="mock")


class MockBackend(ExtractionBackend):
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


def _pass_audit_rows(audit_path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(audit_path).read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("outcome") == "PASS"
    ]


def _assert_no_strict_pass_with_null_admissibility_fields(pass_rows: list[dict]) -> None:
    strict_pass_rows = [
        row for row in pass_rows if str(row.get("policy_profile") or "") == "strict"
    ]
    for row in strict_pass_rows:
        assert row.get("admissibility_basis") is not None, (
            "strict PASS audit row must not have admissibility_basis=None: "
            f"{row}"
        )
        assert row.get("pass_reason_code") is not None, (
            "strict PASS audit row must not have pass_reason_code=None: "
            f"{row}"
        )
        assert_strict_pass_audit_invariants(row)


class TestConsequenceIntentClassifier:
    def test_hello_is_low_risk(self):
        result = classify_consequence_intent("Hello")
        assert result.basis == LOW_RISK_CONVERSATIONAL

    def test_operational_legal_without_lane_is_consequential(self):
        prompt = (
            "Walk me through how to prepare a binding legal settlement letter "
            "for my tenant dispute"
        )
        result = classify_consequence_intent(prompt, request_domain="general")
        assert result.basis == POLICY_PROFILE_REQUIRED
        assert "legal" in result.intent_categories

    def test_operational_financial_with_domain_is_unclassified(self):
        prompt = (
            "Help me outline the steps to renegotiate severance terms with my "
            "employer without involving legal counsel"
        )
        result = classify_consequence_intent(prompt, request_domain="finance")
        assert result.basis == UNCLASSIFIED_CONSEQUENCE_INTENT


class TestStrictAdmissibilityLens:
    def _make_lens(
        self,
        *,
        adapter_responses: list[str],
        claims_sequence: list[list[ExtractedClaim]],
        policy=DEFAULT_STRICT,
        audit_path: str | None = None,
    ) -> Lens:
        bridge = BuiltinBridge(policy=policy, audit_path=audit_path)
        return Lens(
            LensConfig(
                adapter=MockAdapter(adapter_responses),
                extraction_backend=MockBackend(claims_sequence),
                governance_bridge=bridge,
            )
        )

    @pytest.mark.asyncio
    async def test_strict_hello_passes_with_explicit_basis(self):
        lens = self._make_lens(
            adapter_responses=["Hi there."],
            claims_sequence=[[], []],
        )
        result = await lens.process("Hello")
        assert result.action == InterventionAction.PASS
        assert result.decision is not None
        assert result.decision.admissibility_basis == LOW_RISK_CONVERSATIONAL
        assert result.decision.pass_reason_code == LOW_RISK_CONVERSATIONAL
        assert "Admissibility basis" in result.decision.rationale

    @pytest.mark.asyncio
    async def test_unknown_consequential_request_does_not_pass(self):
        prompt = (
            "Walk me through how to prepare a binding legal settlement letter "
            "for my tenant dispute"
        )
        lens = self._make_lens(
            adapter_responses=["Here is a draft settlement letter."],
            claims_sequence=[[], []],
        )
        token = domain_var.set("general")
        try:
            result = await lens.process(prompt)
        finally:
            domain_var.reset(token)

        assert result.action == InterventionAction.CONTAIN
        assert any(
            f.flag_type == FlagType.UNCLASSIFIED_CONSEQUENCE_INTENT
            for f in result.flags
        )
        assert result.decision is not None
        assert result.decision.pass_reason_code is None

    @pytest.mark.asyncio
    async def test_policy_profile_required_when_request_domain_general(self):
        prompt = (
            "What steps should I take to negotiate with the regulator about "
            "our compliance gaps"
        )
        lens = self._make_lens(
            adapter_responses=["You could start by documenting gaps."],
            claims_sequence=[[], []],
        )
        token = domain_var.set("general")
        try:
            result = await lens.process(prompt)
        finally:
            domain_var.reset(token)

        assert result.action == InterventionAction.CONTAIN
        flag = next(
            f for f in result.flags
            if f.flag_type == FlagType.UNCLASSIFIED_CONSEQUENCE_INTENT
        )
        assert flag.rule_id == POLICY_PROFILE_REQUIRED

    @pytest.mark.asyncio
    async def test_strict_preserves_unsupported_only_flags_for_non_low_risk_intent(self):
        lens = self._make_lens(
            adapter_responses=["General onboarding guidance."],
            claims_sequence=[[], []],
        )

        async def _unsupported_only(*_args, **_kwargs):
            return [
                Flag(
                    flag_type=FlagType.UNSUPPORTED_EVENT,
                    entity_name="team",
                    claim="team IMPROVE onboarding",
                    evidence="No IMPROVE relationships established for this entity",
                    severity="warning",
                )
            ]

        lens._checker.check = _unsupported_only  # type: ignore[assignment]
        result = await lens.process("How can I improve team onboarding step by step?")

        assert any(f.flag_type == FlagType.UNSUPPORTED_EVENT for f in result.flags)
        assert not any(
            f.flag_type == FlagType.UNCLASSIFIED_CONSEQUENCE_INTENT for f in result.flags
        )

    @pytest.mark.asyncio
    async def test_moderate_policy_only_allows_pass_with_admissibility_basis_none(
        self,
    ):
        """Moderate only: empty-flag PASS may omit admissibility_basis (strict forbids)."""
        prompt = (
            "Walk me through how to prepare a binding legal settlement letter "
            "for my tenant dispute"
        )
        lens = self._make_lens(
            adapter_responses=["Draft text."],
            claims_sequence=[[], []],
            policy=DEFAULT_MODERATE,
        )
        result = await lens.process(prompt)
        assert result.action == InterventionAction.PASS
        assert result.decision is not None
        assert result.decision.policy == "moderate"
        assert result.decision.admissibility_basis is None
        assert result.decision.pass_reason_code is None

    @pytest.mark.asyncio
    async def test_strict_pass_audit_includes_admissibility_basis(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            audit_path = f.name
        try:
            lens = self._make_lens(
                adapter_responses=["Sure."],
                claims_sequence=[[], []],
                audit_path=audit_path,
            )
            await lens.process("Hello")
            pass_rows = _pass_audit_rows(audit_path)
            assert pass_rows
            last = pass_rows[-1]
            assert last.get("policy_profile") == "strict"
            assert last.get("admissibility_basis") == LOW_RISK_CONVERSATIONAL
            assert last.get("pass_reason_code") == LOW_RISK_CONVERSATIONAL
            assert "Admissibility basis" in str(last.get("rationale", ""))
            _assert_no_strict_pass_with_null_admissibility_fields(pass_rows)
        finally:
            Path(audit_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_strict_pass_audit_never_has_null_admissibility_basis(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            audit_path = f.name
        try:
            lens = self._make_lens(
                adapter_responses=["Hi.", "Emma has a red book."],
                claims_sequence=[[], [], []],
                audit_path=audit_path,
            )
            await lens.process("Hello")
            await lens.process("Emma has a red book.")
            pass_rows = _pass_audit_rows(audit_path)
            assert len(pass_rows) >= 2
            _assert_no_strict_pass_with_null_admissibility_fields(pass_rows)
        finally:
            Path(audit_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_strict_pass_audit_never_has_null_pass_reason_code(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            audit_path = f.name
        try:
            lens = self._make_lens(
                adapter_responses=["Sure."],
                claims_sequence=[[], []],
                audit_path=audit_path,
            )
            await lens.process("Hello")
            pass_rows = _pass_audit_rows(audit_path)
            assert pass_rows
            for row in pass_rows:
                if str(row.get("policy_profile") or "") == "strict":
                    assert row.get("pass_reason_code") is not None
        finally:
            Path(audit_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_moderate_pass_audit_may_have_null_admissibility_basis(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            audit_path = f.name
        try:
            lens = self._make_lens(
                adapter_responses=["Draft."],
                claims_sequence=[[], []],
                policy=DEFAULT_MODERATE,
                audit_path=audit_path,
            )
            await lens.process("Hello")
            pass_rows = _pass_audit_rows(audit_path)
            assert pass_rows
            last = pass_rows[-1]
            assert last.get("policy_profile") == "moderate"
            assert last.get("admissibility_basis") is None
            assert last.get("pass_reason_code") is None
        finally:
            Path(audit_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_narrative_low_risk_still_passes_under_strict(self):
        lens = self._make_lens(
            adapter_responses=["Emma has a red book."],
            claims_sequence=[
                [
                    ExtractedClaim(
                        subject="Emma",
                        relation="HAS",
                        obj="red book",
                        span=Span.PRESENT,
                        negated=False,
                        evidence="input",
                    )
                ],
                [
                    ExtractedClaim(
                        subject="Emma",
                        relation="HAS",
                        obj="red book",
                        span=Span.PRESENT,
                        negated=False,
                        evidence="response",
                    )
                ],
            ],
        )
        result = await lens.process("Emma has a red book.")
        assert result.action == InterventionAction.PASS
        assert result.decision is not None
        assert result.decision.admissibility_basis == LOW_RISK_CONVERSATIONAL


class TestLiveVerificationOutput:
    """Print policy_profile on every PASS summary (run with pytest -s)."""

    @pytest.mark.asyncio
    async def test_live_verification_pass_summaries_include_policy_profile(self, capsys):
        scenarios = [
            ("strict greeting", "Hello", DEFAULT_STRICT),
            (
                "moderate greeting (basis=None allowed)",
                "Hello",
                DEFAULT_MODERATE,
            ),
        ]
        for label, prompt, policy in scenarios:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
                audit_path = f.name
            try:
                bridge = BuiltinBridge(policy=policy, audit_path=audit_path)
                lens = Lens(
                    LensConfig(
                        adapter=MockAdapter(["Ok."]),
                        extraction_backend=MockBackend([[], []]),
                        governance_bridge=bridge,
                    )
                )
                result = await lens.process(prompt)
                pass_rows = _pass_audit_rows(audit_path)
                assert pass_rows
                audit = pass_rows[-1]
                policy_profile = str(audit.get("policy_profile") or policy.name)
                if result.action == InterventionAction.PASS:
                    summary = format_pass_summary(
                        prompt=prompt,
                        policy_profile=policy_profile,
                        admissibility_basis=audit.get("admissibility_basis"),
                        pass_reason_code=audit.get("pass_reason_code"),
                    )
                    print(f"[{label}] {summary}")
                    if policy_profile == "strict":
                        assert_strict_pass_audit_invariants(audit)
            finally:
                Path(audit_path).unlink(missing_ok=True)

        captured = capsys.readouterr()
        assert "policy_profile=strict" in captured.out
        assert "policy_profile=moderate" in captured.out
        assert "moderate policy only" in captured.out
