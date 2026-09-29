"""Tests for the Aurora Governor integration bridge (AuroraScannerGateBridge).

governance module is always available: real unified_rns_system if installed,
in-tree mock (tests/governance_mock/) otherwise. conftest.py handles injection.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tests.skip_reasons import SKIP_STARLETTE_HTTP_TESTCLIENT

from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.govern.policy import DEFAULT_STRICT, DEFAULT_MODERATE, InterventionPolicy, PolicyRule
from aurora_lens.verify.flags import Flag, FlagType
from aurora_lens.pef.state import PEFState
from aurora_lens.pef.span import Span
from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.config import LensConfig
from aurora_lens.lens import Lens

from aurora_lens.govern.attestation import verify_governed_output


# ── Helpers ─────────────────────────────────────────────────────────

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


def _make_flags(types_and_severities: list[tuple[FlagType, str]]) -> list[Flag]:
    return [
        Flag(
            flag_type=ft,
            entity_name="Test",
            claim=f"Test {ft.name}",
            evidence="test evidence",
            severity=sev,
        )
        for ft, sev in types_and_severities
    ]


# ── Bridge Construction Tests ───────────────────────────────────────

class TestBridgeConstruction:

    def test_creates_without_audit(self):
        bridge = AuroraScannerGateBridge()
        assert bridge._ledger is None
        assert bridge._secret_key is None

    def test_creates_with_audit(self, tmp_path):
        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file))
        assert bridge._ledger is not None
        assert bridge.ledger_stats is not None

    def test_creates_with_secret_key(self):
        bridge = AuroraScannerGateBridge(secret_key=b"test-key-123")
        assert bridge._secret_key == b"test-key-123"

    def test_creates_with_constraints(self):
        bridge = AuroraScannerGateBridge(
            constraints={"forbidden_actions": ["danger"]}
        )
        assert bridge._constraints == {"forbidden_actions": ["danger"]}

    def test_kernel_stats_accessible(self):
        bridge = AuroraScannerGateBridge()
        stats = bridge.kernel_stats
        assert "turn_index" in stats
        assert stats["turn_index"] == 0

    def test_verify_ledger_no_ledger(self):
        bridge = AuroraScannerGateBridge()
        assert bridge.verify_ledger() is True


# ── Decision Tests ──────────────────────────────────────────────────

class TestScannerGateDecide:

    @pytest.mark.asyncio
    async def test_no_flags_passes(self):
        bridge = AuroraScannerGateBridge()
        pef = PEFState()
        decision = await bridge.decide([], "Clean response.", pef)
        assert decision.action == InterventionAction.PASS

    @pytest.mark.asyncio
    async def test_contradiction_error_hard_stop(self):
        bridge = AuroraScannerGateBridge()
        pef = PEFState()
        flags = _make_flags([(FlagType.CONTRADICTED_FACT, "error")])
        decision = await bridge.decide(flags, "Bad response.", pef)
        assert decision.action == InterventionAction.HARD_STOP
        assert decision.cid is not None

    @pytest.mark.asyncio
    async def test_hallucinated_attribute_soft_correct(self):
        bridge = AuroraScannerGateBridge()
        pef = PEFState()
        flags = _make_flags([(FlagType.UNSUPPORTED_ATTRIBUTE, "warning")])
        decision = await bridge.decide(flags, "Dubious response.", pef)
        assert decision.action == InterventionAction.SOFT_CORRECT
        assert decision.governance_note is not None

    @pytest.mark.asyncio
    async def test_decision_carries_cid(self):
        bridge = AuroraScannerGateBridge()
        pef = PEFState()
        flags = _make_flags([(FlagType.TIME_SMEAR, "warning")])
        decision = await bridge.decide(flags, "Response.", pef)
        assert decision.cid is not None
        assert decision.cid.startswith("cid:")

    @pytest.mark.asyncio
    async def test_decision_carries_policy_name(self):
        bridge = AuroraScannerGateBridge(policy=DEFAULT_MODERATE)
        pef = PEFState()
        flags = _make_flags([(FlagType.CONTRADICTED_FACT, "error")])
        decision = await bridge.decide(flags, "Bad.", pef)
        assert decision.policy == "moderate"

    @pytest.mark.asyncio
    async def test_kernel_no_longer_escalates_decide(self):
        """GovernanceKernel is NOT a co-author of policy decisions in decide().

        Previously, kernel_step() with forbidden_actions could escalate
        SOFT_CORRECT → HARD_STOP. That behaviour has been removed.
        GovernanceKernel is now attestation/ledger substrate only.

        The canonical replacement is CanonicalScannerGateBridge where PolicyResolver
        is the sole decision authority. See test_canonical_bridge.py.
        """
        bridge = AuroraScannerGateBridge(
            constraints={"forbidden_actions": ["evaluate_output"]}
        )
        pef = PEFState()
        flags = _make_flags([(FlagType.UNSUPPORTED_ATTRIBUTE, "warning")])
        decision = await bridge.decide(flags, "Response.", pef)
        # InterventionPolicy returns SOFT_CORRECT for UNSUPPORTED_ATTRIBUTE.
        # Kernel no longer escalates it — the policy decision stands.
        assert decision.action == InterventionAction.SOFT_CORRECT


# ── Intervention Tests ──────────────────────────────────────────────

class TestScannerGateIntervene:

    @pytest.mark.asyncio
    async def test_hard_stop_blocks(self):
        bridge = AuroraScannerGateBridge()
        flags = _make_flags([(FlagType.CONTRADICTED_FACT, "error")])
        decision = await bridge.decide(flags, "dangerous", PEFState())
        assert decision.action == InterventionAction.HARD_STOP
        decision.original_response = "dangerous"
        adapter = MockAdapter(["should not be called"])
        result = await bridge.intervene(decision, adapter, "test", "context")
        assert "More information required." in result
        assert "This request needs a missing detail." in result
        assert "Action: Choose one option to continue." in result
        assert "Status: Blocked." in result

    @pytest.mark.asyncio
    async def test_soft_correct_keeps_original(self):
        bridge = AuroraScannerGateBridge()
        decision = GovernanceDecision(
            action=InterventionAction.SOFT_CORRECT,
            flags=[],
            rationale="minor",
            original_response="Original text.",
        )
        adapter = MockAdapter(["should not be called"])
        result = await bridge.intervene(decision, adapter, "test", "context")
        assert result == "Original text."

    @pytest.mark.asyncio
    async def test_force_revise_no_adapter_call(self):
        """FORCE_REVISE uses P_REFUSE_EXPLAIN_REDIRECT pathway — no LLM retry."""
        bridge = AuroraScannerGateBridge()
        flags = _make_flags([(FlagType.UNBOUND_ENTITY, "warning")])
        decision = await bridge.decide(flags, "ungrounded claim", PEFState())
        assert decision.action == InterventionAction.FORCE_REVISE
        decision.original_response = "Bob has a cat."
        adapter = MockAdapter(["should not be called"])
        result = await bridge.intervene(decision, adapter, "input", "context")
        assert "Cannot provide that." in result
        assert "could not be verified against the session record" in result
        assert "Rephrase the question" in result
        assert "Choose one option to continue" not in result
        assert "Status: Refused." in result
        assert adapter._index == 0


# ── Forensic Ledger Tests ───────────────────────────────────────────

class TestForensicLedger:

    def test_log_decision_writes_afl(self, tmp_path):
        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file))

        decision = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=_make_flags([(FlagType.CONTRADICTED_FACT, "error")]),
            rationale="test hard stop",
            policy="strict",
            original_response="Bad.",
            corrected_response="Blocked.",
        )
        bridge.log_decision(decision, turn=1)

        # Read ledger
        entries = audit_file.read_text().strip().split("\n")
        assert len(entries) == 1

        envelope = json.loads(entries[0])
        assert envelope["v"] == 1
        assert envelope["kind"] == "aurora.event"
        assert envelope["op"] == "HARD_STOP"
        assert "hash" in envelope
        assert "prev" in envelope
        assert "cid" in envelope
        assert envelope["payload"]["data"]["action"] == "HARD_STOP"
        assert envelope["payload"]["data"]["flags"][0]["type"] == "CONTRADICTED_FACT"

    def test_log_subsystem_audit_event_appends_afl(self, tmp_path):
        """Proxy subsystem events use the same AFL chain as governance (no JSONL interleave)."""
        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file))
        bridge.log_subsystem_audit_event(
            op="stream_abort",
            payload={"type": "stream_abort", "session_id": "s1"},
            trace_id="proxy:trace1",
        )
        entries = audit_file.read_text().strip().split("\n")
        assert len(entries) == 1
        envelope = json.loads(entries[0])
        assert envelope["op"] == "stream_abort"
        inner = envelope["payload"]["data"]
        assert inner["type"] == "stream_abort"
        assert "event_hash" in inner
        assert bridge.verify_ledger() is True

    def test_log_subsystem_audit_event_jsonl_backend_noop(self, tmp_path):
        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file), backend="jsonl")
        bridge.log_subsystem_audit_event(op="stream_abort", payload={"x": 1}, trace_id="t")
        assert not audit_file.exists()

    @pytest.mark.asyncio
    async def test_forensic_event_single_ledger_line_for_extraction_empty_contain(self, tmp_path):
        """EXTRACTION_EMPTY + CONTAIN: single governance ledger line with canonical forensic_event."""
        from aurora_lens.governor import forensic_schema

        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file), policy=DEFAULT_MODERATE)

        flags = _make_flags([(FlagType.EXTRACTION_EMPTY, "error")])
        decision = await bridge.decide(flags, "some response", PEFState())
        assert decision.action == InterventionAction.CONTAIN

        decision.original_response = "some response"
        adapter = MockAdapter(["should not be called"])
        pef_context = "Emma HAS red book"
        result = await bridge.intervene(decision, adapter, "input", pef_context)

        assert any(
            w in result.lower()
            for w in ("clarif", "specific", "information", "missing detail")
        )

        bridge.log_decision(decision, turn=0, pef_context=pef_context)

        entries = audit_file.read_text().strip().split("\n")
        assert len(entries) == 1, f"Expected single ledger line, got {len(entries)}: {entries}"

        line = json.loads(entries[0])
        assert line["op"] == "CONTAIN"
        pdata = line["payload"]["data"]
        assert isinstance(pdata.get("ruleset_hash"), str)
        assert pdata["ruleset_hash"].startswith("sha256:")
        assert pdata.get("policy_version") != "unknown"
        assert isinstance(pdata.get("governor_policy_id"), str)
        fe = pdata["forensic_event"]
        assert fe["status"] == "ASK"
        assert fe["attempted_action"] == "respond"
        assert "EXTRACTION_EMPTY" in fe["failed_constraints"]
        assert forensic_schema.validate(fe) == []
        assert forensic_schema.verify_event_hash(fe)

    @pytest.mark.asyncio
    async def test_forensic_event_single_ledger_line_for_force_revise(self, tmp_path):
        """UNBOUND_ENTITY + FORCE_REVISE: single line; forensic_event status REFUSE (scanner bridge)."""
        from aurora_lens.governor import forensic_schema

        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file), policy=DEFAULT_STRICT)

        flags = _make_flags([(FlagType.UNBOUND_ENTITY, "warning")])
        decision = await bridge.decide(flags, "ungrounded claim", PEFState())
        assert decision.action == InterventionAction.FORCE_REVISE

        decision.original_response = "ungrounded claim"
        adapter = MockAdapter(["should not be called"])
        await bridge.intervene(decision, adapter, "input", "pef ctx")
        bridge.log_decision(decision, turn=0, pef_context="pef ctx")

        entries = audit_file.read_text().strip().split("\n")
        assert len(entries) == 1
        line = json.loads(entries[0])
        assert line["op"] == "FORCE_REVISE"
        fe = line["payload"]["data"]["forensic_event"]
        assert fe["status"] == "REFUSE"
        assert "UNBOUND_ENTITY" in fe["failed_constraints"]
        assert forensic_schema.validate(fe) == []
        assert forensic_schema.verify_event_hash(fe)

    @pytest.mark.asyncio
    async def test_forensic_event_single_ledger_line_for_hard_stop(self, tmp_path):
        """HARD_STOP: single governance line; forensic_event schema-complete."""
        from aurora_lens.context import session_id_var
        from aurora_lens.governor import forensic_schema

        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file))

        flags = _make_flags([(FlagType.CONTRADICTED_FACT, "error")])
        decision = await bridge.decide(flags, "dangerous", PEFState())
        assert decision.action == InterventionAction.HARD_STOP

        decision.original_response = "dangerous"
        adapter = MockAdapter(["should not be called"])
        result = await bridge.intervene(decision, adapter, "input", "pef ctx")

        assert "More information required." in result
        assert "This request needs a missing detail." in result
        assert "Action: Choose one option to continue." in result
        assert "Status: Blocked." in result

        tok = session_id_var.set("ledger-test-session-42")
        try:
            bridge.log_decision(
                decision,
                turn=0,
                pef_context="pef ctx",
                pre_llm=False,
                pef_snapshot=PEFState().to_dict(),
            )
        finally:
            session_id_var.reset(tok)

        entries = audit_file.read_text().strip().split("\n")
        assert len(entries) == 1
        line = json.loads(entries[0])
        assert line["op"] == "HARD_STOP"
        fe = line["payload"]["data"]["forensic_event"]
        assert fe["status"] == "STOP"
        assert "CONTRADICTED_FACT" in fe["failed_constraints"]
        assert fe.get("session_id") == "ledger-test-session-42"
        assert forensic_schema.validate(fe) == []
        assert forensic_schema.verify_event_hash(fe)

    def test_multiple_entries_hash_chain(self, tmp_path):
        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file))

        for i in range(3):
            decision = GovernanceDecision(
                action=InterventionAction.SOFT_CORRECT,
                flags=_make_flags([(FlagType.TIME_SMEAR, "warning")]),
                rationale=f"entry {i}",
                policy="strict",
            )
            bridge.log_decision(decision, turn=i + 1)

        entries = audit_file.read_text().strip().split("\n")
        assert len(entries) == 3

        # Verify hash chain
        envelopes = [json.loads(line) for line in entries]
        assert envelopes[0]["prev"] == "h:null"
        assert envelopes[1]["prev"] == envelopes[0]["hash"]
        assert envelopes[2]["prev"] == envelopes[1]["hash"]

        stats = bridge.ledger_stats
        assert stats is not None
        assert stats.get("entries") == 3
        assert stats.get("entry_count") == 3

    def test_ledger_verify_passes(self, tmp_path):
        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file))

        for i in range(5):
            decision = GovernanceDecision(
                action=InterventionAction.PASS,
                flags=[],
                rationale="clean",
                policy="strict",
            )
            bridge.log_decision(decision, turn=i + 1)

        assert bridge.verify_ledger() is True
        first = json.loads(audit_file.read_text().strip().split("\n")[0])
        pdata = first["payload"]["data"]
        assert isinstance(pdata.get("ruleset_hash"), str)
        assert pdata["ruleset_hash"].startswith("sha256:")
        assert pdata.get("policy_version") != "unknown"
        assert isinstance(pdata.get("governor_policy_id"), str)

    def test_ledger_tamper_detected(self, tmp_path):
        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file))

        decision = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=_make_flags([(FlagType.CONTRADICTED_FACT, "error")]),
            rationale="original",
            policy="strict",
        )
        bridge.log_decision(decision, turn=1)

        # Tamper with the ledger
        content = audit_file.read_text()
        tampered = content.replace('"original"', '"tampered"')
        audit_file.write_text(tampered)

        assert bridge.verify_ledger() is False

    def test_ledger_line_hmac_when_secret_key_configured(self, tmp_path):
        """AFL ledger entries include sig when bridge has secret_key."""
        audit_file = tmp_path / "audit.jsonl"
        key = b"ledger-line-hmac-key-32bytes!!"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file), secret_key=key)

        decision = GovernanceDecision(
            action=InterventionAction.PASS,
            flags=[],
            rationale="ok",
            policy="strict",
        )
        bridge.log_decision(decision, turn=1)

        envelope = json.loads(audit_file.read_text(encoding="utf-8").strip().split("\n")[0])
        assert envelope["sig"] is not None
        assert envelope["sig"]["alg"] == "hmac-sha256"
        assert len(envelope["sig"]["value"]) == 64
        assert bridge.verify_ledger() is True

    def test_ledger_verify_fails_hmac_tamper(self, tmp_path):
        """Tampering sig value fails verify_ledger when keys are configured."""
        audit_file = tmp_path / "audit.jsonl"
        key = b"anti-tamper-key-32-bytes-long!!"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file), secret_key=key)
        decision = GovernanceDecision(
            action=InterventionAction.PASS,
            flags=[],
            rationale="ok",
            policy="strict",
        )
        bridge.log_decision(decision, turn=1)

        entry = json.loads(audit_file.read_text(encoding="utf-8").strip().split("\n")[0])
        v = entry["sig"]["value"]
        entry["sig"]["value"] = ("0" if v[0] != "0" else "1") + v[1:]
        audit_file.write_text(
            json.dumps(entry, separators=(",", ":"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        assert bridge.verify_ledger() is False

    def test_pef_continuity_in_entries(self, tmp_path):
        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file))

        for i in range(3):
            decision = GovernanceDecision(
                action=InterventionAction.PASS,
                flags=[],
                rationale="clean",
            )
            bridge.log_decision(decision, turn=i + 1)

        entries = audit_file.read_text().strip().split("\n")
        for entry in entries:
            envelope = json.loads(entry)
            # Every entry has a PEF continuity value
            assert "pef" in envelope.get("ext", {})
            assert len(envelope["ext"]["pef"]) == 64  # SHA-256 hex


# ── Attestation Tests ───────────────────────────────────────────────

class TestAttestation:

    def test_attestation_produced(self, tmp_path):
        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(
            audit_path=str(audit_file),
            secret_key=b"test-secret-key",
        )

        decision = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=_make_flags([(FlagType.CONTRADICTED_FACT, "error")]),
            rationale="blocked",
            policy="strict",
            original_response="Bad.",
            corrected_response="Blocked.",
        )
        bridge.log_decision(decision, turn=1)

        attested = bridge.last_attestation
        assert attested is not None
        assert attested.decision == "HARD_STOP"
        assert attested.content == "Blocked."
        assert attested.policy_cid == "policy:strict"
        assert attested.signature != ""

    def test_attestation_verifies(self, tmp_path):
        audit_file = tmp_path / "audit.jsonl"
        secret = b"verify-me-123"
        bridge = AuroraScannerGateBridge(
            audit_path=str(audit_file),
            secret_key=secret,
        )

        decision = GovernanceDecision(
            action=InterventionAction.SOFT_CORRECT,
            flags=[],
            rationale="minor",
            policy="strict",
            corrected_response="Clean output.",
        )
        bridge.log_decision(decision, turn=1)

        # Verify via bridge method
        assert bridge.verify_attestation() is True

        # Verify via governance module directly
        attested = bridge.last_attestation
        assert verify_governed_output(attested, secret) is True

    def test_attestation_fails_wrong_key(self, tmp_path):
        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(
            audit_path=str(audit_file),
            secret_key=b"correct-key",
        )

        decision = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=_make_flags([(FlagType.CONTRADICTED_FACT, "error")]),
            rationale="blocked",
            corrected_response="Blocked.",
        )
        bridge.log_decision(decision, turn=1)

        attested = bridge.last_attestation
        assert attested is not None
        # Wrong key → verification fails
        assert verify_governed_output(attested, b"wrong-key") is False

    def test_no_attestation_without_key(self, tmp_path):
        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file))

        decision = GovernanceDecision(
            action=InterventionAction.PASS,
            flags=[],
            rationale="clean",
        )
        bridge.log_decision(decision, turn=1)
        assert bridge.last_attestation is None


# ── End-to-End with Lens ────────────────────────────────────────────

class TestLensWithScannerGateBridge:

    @pytest.mark.asyncio
    async def test_full_pipeline_hard_stop(self, tmp_path):
        """Full pipeline: flag→kernel→HARD_STOP, forensic log, attestation."""
        audit_file = tmp_path / "audit.jsonl"
        secret = b"e2e-secret"

        adapter = MockAdapter(["Patient has myocardial infarction."])
        backend = MockBackend([
            # User input: establish negated fact
            [ExtractedClaim(
                subject="Patient", relation="HAS", obj="myocardial infarction",
                span=Span.PRESENT, negated=True, evidence="No ECG.",
            )],
            # LLM response: asserts MI → contradicts PEF
            [ExtractedClaim(
                subject="Patient", relation="HAS", obj="myocardial infarction",
                span=Span.PRESENT, negated=False, evidence="MI diagnosis.",
            )],
        ])
        bridge = AuroraScannerGateBridge(
            audit_path=str(audit_file),
            secret_key=secret,
        )
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        )
        lens = Lens(config)

        result = await lens.process("Patient with no ECG. Diagnose.")

        # Governance: HARD_STOP
        assert result.action == InterventionAction.HARD_STOP
        _out = result.response
        assert "More information required." in _out
        assert "This request needs a missing detail." in _out
        assert "Action: Choose one option to continue." in _out
        assert "Status: Blocked." in _out

        # Forensic ledger: hash-chained entry
        assert audit_file.exists()
        entries = audit_file.read_text().strip().split("\n")
        assert len(entries) >= 1
        envelope = json.loads(entries[-1])
        assert envelope["op"] == "HARD_STOP"
        assert "hash" in envelope

        # Attestation: signed
        assert bridge.last_attestation is not None
        assert bridge.verify_attestation() is True

        # Ledger: tamper-evident
        assert bridge.verify_ledger() is True

    @pytest.mark.asyncio
    async def test_full_pipeline_clean_pass(self, tmp_path):
        """Clean response → PASS, no attestation needed."""
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
        bridge = AuroraScannerGateBridge(
            audit_path=str(audit_file),
            secret_key=b"key",
        )
        config = LensConfig(
            adapter=adapter,
            extraction_backend=backend,
            governance_bridge=bridge,
        )
        lens = Lens(config)

        result = await lens.process("Emma has a red book.")

        assert result.action == InterventionAction.PASS
        assert result.response == "Emma has a red book."


# ── J.1: Verify endpoint wired to ledger backend ─────────────────────

class TestVerifyEndpointLedgerBackend:
    """J.1: GET /v1/audit/verify works for the ledger (AuroraScannerGateBridge) backend."""

    @staticmethod
    def _clear_audit_signing_env(monkeypatch) -> None:
        """Env overrides win over YAML; clear so tests control signing explicitly."""
        monkeypatch.delenv("AURORA_LENS_AUDIT_SIGNING_KEY", raising=False)
        monkeypatch.delenv("AURORA_LENS_AUDIT_SIGNING_KEYS", raising=False)

    def test_verify_endpoint_ledger_backend(self, monkeypatch, tmp_path):
        """Verify endpoint returns verified:true when audit_backend is ledger."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        self._clear_audit_signing_env(monkeypatch)
        audit_file = tmp_path / "audit.jsonl"

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(["Hello!"]), MockAdapter(["Hello!"])),
        )

        from aurora_lens.proxy.config import ProxyConfig
        from aurora_lens.proxy.app import create_app

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "audit_log": str(audit_file),
                "audit_backend": "ledger",
            },
            "extraction": {"backend": "llm"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        # Make a request to write ledger entries
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert r.status_code == 200

        # Verify endpoint should delegate to ForensicLedger.verify()
        r = client.get("/v1/audit/verify")
        assert r.status_code == 200
        data = r.json()
        assert data["verified"] is True
        assert data["chain_verified"] is True
        assert data["backend"] == "ledger"
        assert data["hmac_verified"] is None
        assert "hmac_checked" not in data
        assert isinstance(data["entries"], int)

    def test_verify_endpoint_ledger_no_signing_key_required(self, monkeypatch, tmp_path):
        """Ledger verify does not require audit_signing_key (unlike JSONL path)."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        self._clear_audit_signing_env(monkeypatch)
        audit_file = tmp_path / "audit.jsonl"

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(["ok"]), MockAdapter(["ok"])),
        )

        from aurora_lens.proxy.config import ProxyConfig
        from aurora_lens.proxy.app import create_app

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "audit_log": str(audit_file),
                "audit_backend": "ledger",
                # No audit_signing_key — ledger verify should still work
            },
            "extraction": {"backend": "llm"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.get("/v1/audit/verify")
        assert r.status_code == 200
        data = r.json()
        assert data["verified"] is True
        assert data["backend"] == "ledger"

    def test_verify_endpoint_ledger_with_signing_key_checks_hmac(
        self, monkeypatch, tmp_path
    ):
        """Ledger path with audit_signing_key: lines are signed; verify reports hmac_checked."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        self._clear_audit_signing_env(monkeypatch)
        audit_file = tmp_path / "audit.jsonl"
        signing_key = "ledger-hmac-test-key-32bytes!!"

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(["Hello!"]), MockAdapter(["Hello!"])),
        )

        from aurora_lens.proxy.config import ProxyConfig
        from aurora_lens.proxy.app import create_app

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "audit_log": str(audit_file),
                "audit_backend": "ledger",
                "audit_signing_key": signing_key,
            },
            "extraction": {"backend": "llm"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 200

        lines = [ln for ln in audit_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert lines, "expected at least one ledger line"
        first = json.loads(lines[0])
        assert first.get("sig") is not None
        assert first["sig"].get("alg") == "hmac-sha256"

        r = client.get("/v1/audit/verify")
        assert r.status_code == 200
        data = r.json()
        assert data["verified"] is True
        assert data["hmac_verified"] is True
        assert data["hmac_checked"] is True
        assert data["backend"] == "ledger"


# ── Stage B: run_id and AttestedOutput.meta completeness ────────────

class TestRunIdAndAttestedMeta:
    """Verify run_id is stable per bridge instance and AttestedOutput carries temporal fields."""

    def test_run_id_present_and_stable(self, tmp_path):
        """Same run_id appears in three sequential decisions from the same bridge (jsonl backend)."""
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
            for i in range(3):
                d = await bridge.decide(flags, "resp", pef)
                bridge.log_decision(d, turn=i)

        asyncio.run(_run())

        lines = [ln for ln in log.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
        assert len(lines) == 3
        run_ids = [json.loads(ln).get("run_id") for ln in lines]
        assert all(rid is not None for rid in run_ids), f"Missing run_id in some entries: {run_ids}"
        assert len(set(run_ids)) == 1, f"run_id changed across entries: {run_ids}"

    def test_run_id_varies_per_request_when_context_var_set(self, tmp_path):
        """Proxy-minted per-request run_id/proxy_run_id (context vars) override the
        bridge's own stable instance id — distinct HTTP requests must not share a
        run_id even when they hit the same long-lived bridge instance."""
        import asyncio, json, uuid
        from aurora_lens.context import proxy_run_id_var, run_id_var
        from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
        from aurora_lens.verify.flags import Flag, FlagType
        from aurora_lens.pef.state import PEFState

        log = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(log), backend="jsonl")
        pef = PEFState()
        flags = [Flag(flag_type=FlagType.UNSUPPORTED_ATTRIBUTE, entity_name="E",
                      claim="c", evidence="e", severity="warning")]

        async def _one_request(turn: int) -> None:
            token_run = run_id_var.set(str(uuid.uuid4()))
            token_proxy = proxy_run_id_var.set(str(uuid.uuid4()))
            try:
                d = await bridge.decide(flags, "resp", pef)
                bridge.log_decision(d, turn=turn)
            finally:
                run_id_var.reset(token_run)
                proxy_run_id_var.reset(token_proxy)

        async def _run():
            for i in range(3):
                await _one_request(i)

        asyncio.run(_run())

        lines = [ln for ln in log.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
        assert len(lines) == 3
        entries = [json.loads(ln) for ln in lines]
        run_ids = [e.get("run_id") for e in entries]
        proxy_run_ids = [e.get("proxy_run_id") for e in entries]
        assert len(set(run_ids)) == 3, f"each request must mint a distinct run_id: {run_ids}"
        assert len(set(proxy_run_ids)) == 3, f"each request must mint a distinct proxy_run_id: {proxy_run_ids}"
        # None of the per-request ids collide with the bridge's own fallback instance id.
        assert bridge._run_id not in run_ids

        # Outside any request context, the bridge falls back to its own stable id —
        # existing non-proxy callers (tests, scripts) keep prior behavior unchanged.
        assert run_id_var.get(None) is None
        d2 = asyncio.run(bridge.decide(flags, "resp", pef))
        bridge.log_decision(d2, turn=99)
        last_entry = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert last_entry.get("run_id") == bridge._run_id

    def test_run_id_in_ledger_entries(self, tmp_path):
        """run_id inside AFL ledger payload.data matches bridge._run_id (ledger backend)."""
        import asyncio, json
        from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
        from aurora_lens.verify.flags import Flag, FlagType
        from aurora_lens.pef.state import PEFState

        log = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(log), backend="ledger")
        expected_run_id = bridge._run_id
        pef = PEFState()
        flags = [Flag(flag_type=FlagType.UNSUPPORTED_ATTRIBUTE, entity_name="E",
                      claim="c", evidence="e", severity="warning")]

        async def _run():
            d = await bridge.decide(flags, "resp", pef)
            bridge.log_decision(d, turn=0)

        asyncio.run(_run())

        lines = [ln for ln in log.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
        envelope = json.loads(lines[0])
        data = envelope.get("payload", {}).get("data", {})
        assert data.get("run_id") == expected_run_id

    def test_attestation_meta_has_temporal_fields(self, tmp_path):
        """AttestedOutput.meta includes timestamp, trace_id, session_id, run_id; survives verify."""
        import asyncio
        from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
        from aurora_lens.verify.flags import Flag, FlagType
        from aurora_lens.pef.state import PEFState
        from aurora_lens.govern.attestation import verify_governed_output

        log = tmp_path / "audit.jsonl"
        key = b"test-signing-key-32-bytes-pad!!"
        bridge = AuroraScannerGateBridge(
            audit_path=str(log),
            backend="jsonl",
            secret_key=key,
        )
        pef = PEFState()
        flags = [Flag(flag_type=FlagType.UNSUPPORTED_ATTRIBUTE, entity_name="E",
                      claim="c", evidence="e", severity="warning")]

        async def _run():
            d = await bridge.decide(flags, "resp", pef)
            bridge.log_decision(d, turn=0)

        asyncio.run(_run())

        attest = bridge.last_attestation
        assert attest is not None, "No attestation produced"
        assert "timestamp" in attest.meta, f"timestamp missing from meta: {attest.meta}"
        assert "trace_id" in attest.meta, f"trace_id missing from meta: {attest.meta}"
        assert "session_id" in attest.meta, f"session_id missing from meta: {attest.meta}"
        assert "run_id" in attest.meta, f"run_id missing from meta: {attest.meta}"
        assert attest.meta["run_id"] == bridge._run_id
        # Verify that the enriched meta doesn't break HMAC verification
        assert verify_governed_output(attest, key), "HMAC verification failed after meta enrichment"


# ── Stage F: JSONL backend HTTP verify endpoint tests ───────────────

class TestVerifyEndpointJsonlBackend:
    """Stage F: GET /v1/audit/verify for the flat JSONL backend (BuiltinBridge path)."""

    def test_verify_endpoint_jsonl_clean(self, monkeypatch, tmp_path):
        """JSONL backend verify: both hmac_verified=True, chain_verified=True for a clean log."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        audit_file = tmp_path / "audit.jsonl"
        signing_key = "test-jsonl-signing-key"

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(["Hello!"]), MockAdapter(["Hello!"])),
        )

        from aurora_lens.proxy.config import ProxyConfig
        from aurora_lens.proxy.app import create_app

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "audit_log": str(audit_file),
                "audit_backend": "jsonl",
                "audit_signing_key": signing_key,
            },
            "extraction": {"backend": "llm"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        # Make a request so audit entries are written
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert r.status_code == 200

        r = client.get("/v1/audit/verify")
        assert r.status_code == 200
        data = r.json()
        assert data.get("verified") is True, f"Expected verified=True, got: {data}"
        assert data.get("hmac_verified") is True, f"Expected hmac_verified=True, got: {data}"
        assert data.get("chain_verified") is True, f"Expected chain_verified=True, got: {data}"

    def test_verify_endpoint_jsonl_tampered_entry(self, monkeypatch, tmp_path):
        """JSONL backend verify: tampered entry causes verified=False with first_failed_entry."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

        import json as _json

        audit_file = tmp_path / "audit.jsonl"
        signing_key = "test-jsonl-signing-key"

        monkeypatch.setattr(
            "aurora_lens.proxy.app._build_provider_adapters",
            lambda cfg: (MockAdapter(["Hello!"]), MockAdapter(["Hello!"])),
        )

        from aurora_lens.proxy.config import ProxyConfig
        from aurora_lens.proxy.app import create_app

        cfg = ProxyConfig.from_mapping({
            "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
            "governance": {
                "audit_log": str(audit_file),
                "audit_backend": "jsonl",
                "audit_signing_key": signing_key,
            },
            "extraction": {"backend": "llm"},
        })
        app = create_app(cfg)
        client = TestClient(app)

        # Write a clean entry
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert r.status_code == 200

        # Tamper the last entry by corrupting its HMAC
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        last = _json.loads(lines[-1])
        assert "hmac" in last, "JSONL audit row should carry hmac when a signing key is configured"
        tampered_cid = last.get("cid")
        last["hmac"] = "tampered" + "0" * 56
        lines[-1] = _json.dumps(last, separators=(",", ":"), ensure_ascii=False)
        audit_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        r = client.get("/v1/audit/verify")
        assert r.status_code == 200
        data = r.json()
        assert data.get("verified") is False, (
            f"Expected verified=False after tampering, got: {data}"
        )
        failed = data.get("first_failed_entry")
        assert failed, f"Expected first_failed_entry populated, got: {data}"
        assert failed == tampered_cid, (
            f"Expected first_failed_entry={tampered_cid!r}, got {failed!r}: {data}"
        )
