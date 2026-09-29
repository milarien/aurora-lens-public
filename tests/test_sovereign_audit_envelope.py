"""Tests for Sovereign Provider Registry Phase 2 audit envelope fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aurora_lens.config import LensConfig
from aurora_lens.context import request_metadata_var
from aurora_lens.govern.bridge import BuiltinBridge
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.request_metadata import ProviderRouteRequest, RequestMetadata
from aurora_lens.sovereign.audit_envelope import (
    apply_provider_route_to_audit_entry,
    build_provider_route_audit_envelope,
    compute_registry_state_hash,
)
from aurora_lens.sovereign.failover_bridge import FAILOVER_BRIDGE_REF
from aurora_lens.sovereign.provider_registry import (
    ProviderRouteOutcome,
    SovereignProviderRegistry,
    reject_failover_without_registry,
)
from aurora_lens.sovereign.provider_state import PROVIDER_UNAVAILABLE, VALIDATED_CURRENT
from aurora_lens.verify.flags import Flag, FlagType
from tests.test_lens import MockAdapter
from tests.test_sovereign_failover_bridge import (
    ALTERNATE_ID,
    PRIMARY_ID,
    TASK_DOMAIN,
    TASK_GRADE,
    _certified_alternate_profile,
    _route_request,
)

_REQUIRED_ENVELOPE_KEYS = frozenset({
    "requested_provider_id",
    "selected_provider_id",
    "selected_route",
    "primary_provider_state",
    "alternate_provider_id",
    "bridge_status",
    "bridge_ref",
    "registry_policy_result",
    "failover_attempted",
    "failover_result",
    "refusal_reason",
    "registry_state_hash",
    "adapter_called",
})


class TestProviderRouteAuditEnvelope:
    def test_valid_failover_envelope_fields(self):
        registry = SovereignProviderRegistry([_certified_alternate_profile()])
        request = _route_request()
        evaluation = registry.evaluate_failover(request)
        envelope = build_provider_route_audit_envelope(
            request,
            evaluation,
            adapter_called=True,
            alternate_profile=_certified_alternate_profile(),
        )
        assert _REQUIRED_ENVELOPE_KEYS <= frozenset(envelope.keys())
        assert envelope["requested_provider_id"] == PRIMARY_ID
        assert envelope["selected_provider_id"] == ALTERNATE_ID
        assert envelope["selected_route"] == ProviderRouteOutcome.USE_CERTIFIED_ALTERNATE.value
        assert envelope["bridge_status"] == "valid"
        assert envelope["bridge_ref"] == FAILOVER_BRIDGE_REF
        assert envelope["failover_attempted"] is True
        assert envelope["failover_result"] == "allowed"
        assert envelope["adapter_called"] is True
        assert envelope["refusal_reason"] is None
        assert str(envelope["registry_state_hash"]).startswith("sha256:")

    def test_refused_failover_envelope(self):
        registry = SovereignProviderRegistry([])
        request = _route_request()
        evaluation = registry.evaluate_failover(request)
        envelope = build_provider_route_audit_envelope(
            request,
            evaluation,
            adapter_called=False,
        )
        assert envelope["bridge_status"] == "missing"
        assert envelope["failover_result"] == "refused"
        assert envelope["refusal_reason"] == "alternate_capability_profile_missing"
        assert envelope["adapter_called"] is False
        assert envelope["selected_provider_id"] is None

    def test_registry_state_hash_is_stable(self):
        registry = SovereignProviderRegistry([_certified_alternate_profile()])
        request = _route_request()
        evaluation = registry.evaluate_failover(request)
        profile = _certified_alternate_profile()
        h1 = compute_registry_state_hash(request, evaluation, alternate_profile=profile)
        h2 = compute_registry_state_hash(request, evaluation, alternate_profile=profile)
        assert h1 == h2

    def test_apply_to_audit_entry_and_forensic_event(self):
        request = _route_request()
        evaluation = reject_failover_without_registry(request)
        envelope = build_provider_route_audit_envelope(
            request,
            evaluation,
            adapter_called=False,
        )
        decision = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=[],
            rationale="test",
            forensic_event={"schema_version": 1, "status": "STOP"},
        )
        decision.provider_route = envelope
        entry: dict = {"forensic_event": decision.forensic_event}
        apply_provider_route_to_audit_entry(entry, decision)
        assert entry["provider_route"] == envelope
        assert entry["forensic_event"]["provider_route"] == envelope
        assert str(entry["forensic_event"]["event_hash"]).startswith("sha256:")


class TestProviderRouteAuditLensIntegration:
    @pytest.mark.asyncio
    async def test_refused_failover_audit_row_has_provider_route(self, tmp_path: Path):
        audit_path = tmp_path / "audit.jsonl"
        registry = SovereignProviderRegistry([])
        adapter = MockAdapter(responses=["unused"])
        lens = Lens(
            LensConfig(
                adapter=adapter,
                extraction_backend=SpacyBackend(model="en_core_web_sm"),
                sovereign_provider_registry=registry,
                governance_bridge=BuiltinBridge(audit_path=str(audit_path)),
            )
        )
        meta = RequestMetadata(provider_route=_route_request())
        tok = request_metadata_var.set(meta)
        try:
            await lens.process("Summarise the attached notes.")
        finally:
            request_metadata_var.reset(tok)

        lines = [json.loads(ln) for ln in audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert lines
        row = lines[-1]
        assert "provider_route" in row
        pr = row["provider_route"]
        assert pr["failover_result"] == "refused"
        assert pr["adapter_called"] is False
        assert pr["bridge_status"] == "missing"
        fe = row.get("forensic_event") or {}
        assert fe.get("provider_route") == pr

    @pytest.mark.asyncio
    async def test_allowed_failover_audit_row_records_adapter_called(self, tmp_path: Path):
        audit_path = tmp_path / "audit.jsonl"
        registry = SovereignProviderRegistry([_certified_alternate_profile()])
        adapter = MockAdapter(responses=["Summary complete."])
        lens = Lens(
            LensConfig(
                adapter=adapter,
                extraction_backend=SpacyBackend(model="en_core_web_sm"),
                sovereign_provider_registry=registry,
                governance_bridge=BuiltinBridge(audit_path=str(audit_path)),
            )
        )
        meta = RequestMetadata(
            provider_route=_route_request(primary_state=PROVIDER_UNAVAILABLE),
        )
        tok = request_metadata_var.set(meta)
        try:
            await lens.process("Summarise the attached notes.")
        finally:
            request_metadata_var.reset(tok)

        lines = [json.loads(ln) for ln in audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        pass_rows = [r for r in lines if r.get("outcome") == "PASS"]
        assert pass_rows
        pr = pass_rows[-1]["provider_route"]
        assert pr["failover_result"] == "allowed"
        assert pr["adapter_called"] is True
        assert pr["bridge_status"] == "valid"
        assert pr["selected_provider_id"] == ALTERNATE_ID

    @pytest.mark.asyncio
    async def test_active_primary_audit_envelope_without_failover(self, tmp_path: Path):
        audit_path = tmp_path / "audit.jsonl"
        registry = SovereignProviderRegistry([_certified_alternate_profile()])
        adapter = MockAdapter(responses=["Summary complete."])
        lens = Lens(
            LensConfig(
                adapter=adapter,
                extraction_backend=SpacyBackend(model="en_core_web_sm"),
                sovereign_provider_registry=registry,
                governance_bridge=BuiltinBridge(audit_path=str(audit_path)),
            )
        )
        meta = RequestMetadata(
            provider_route=_route_request(primary_state=VALIDATED_CURRENT),
        )
        tok = request_metadata_var.set(meta)
        try:
            await lens.process("Summarise the attached notes.")
        finally:
            request_metadata_var.reset(tok)

        lines = [json.loads(ln) for ln in audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        pr = lines[-1]["provider_route"]
        assert pr["failover_attempted"] is False
        assert pr["failover_result"] == "not_attempted"
        assert pr["selected_provider_id"] == PRIMARY_ID
        assert pr["bridge_status"] is None
