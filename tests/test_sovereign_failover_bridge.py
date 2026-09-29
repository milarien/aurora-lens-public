"""Tests for Sovereign Provider Registry Phase 1 — failover bridge enforcement."""

from __future__ import annotations

import pytest

from aurora_lens.config import LensConfig
from aurora_lens.context import request_metadata_var
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.pef.uncertainty_analysis import merge_epistemic_uncertainties, pef_uncertainty_analysis
from aurora_lens.request_metadata import ProviderRouteRequest, RequestMetadata
from aurora_lens.sovereign.failover_bridge import FAILOVER_BRIDGE_REF, FAILOVER_CONCLUSION
from aurora_lens.sovereign.provider_profile import ProviderProfile
from aurora_lens.sovereign.provider_registry import (
    ProviderRouteOutcome,
    SovereignProviderRegistry,
    apply_route_evaluation_to_pef,
    evaluation_blocks_adapter,
    reject_failover_without_registry,
)
from aurora_lens.sovereign.provider_state import PROVIDER_UNAVAILABLE, VALIDATED_CURRENT, VALIDATION_STALE
from tests.test_lens import MockAdapter

PRIMARY_ID = "anthropic_us_claude_sonnet"
ALTERNATE_ID = "local_mistral_7b"
TASK_DOMAIN = "internal_ops"
TASK_GRADE = "high"


def _certified_alternate_profile(**overrides: object) -> ProviderProfile:
    base = ProviderProfile(
        provider_id=ALTERNATE_ID,
        provider_name="Local Mistral",
        model_id="mistral-7b",
        endpoint_url="http://localhost:8080/v1",
        hosting_jurisdiction="AU",
        data_boundary="local_au",
        allowed_data_classes=("public", "internal", "restricted"),
        permitted_domains=("general", "software", "internal_ops"),
        max_consequence_grade="high",
        supports_tools=True,
        supports_structured_output=True,
        context_window_tokens=32768,
        last_validated_at="2026-09-28T10:00:00+10:00",
        status=VALIDATED_CURRENT,
    )
    if not overrides:
        return base
    data = {
        "provider_id": base.provider_id,
        "provider_name": base.provider_name,
        "model_id": base.model_id,
        "endpoint_url": base.endpoint_url,
        "hosting_jurisdiction": base.hosting_jurisdiction,
        "data_boundary": base.data_boundary,
        "allowed_data_classes": base.allowed_data_classes,
        "permitted_domains": base.permitted_domains,
        "max_consequence_grade": base.max_consequence_grade,
        "supports_tools": base.supports_tools,
        "supports_structured_output": base.supports_structured_output,
        "context_window_tokens": base.context_window_tokens,
        "retention_policy": base.retention_policy,
        "validation_suite_id": base.validation_suite_id,
        "last_validated_at": base.last_validated_at,
        "status": base.status,
        "regression_result": base.regression_result,
        "report_hash": base.report_hash,
        "failed_checks": base.failed_checks,
        "observed_model_id": base.observed_model_id,
    }
    data.update(overrides)
    return ProviderProfile(**data)


def _route_request(**overrides: object) -> ProviderRouteRequest:
    base = {
        "primary_provider_id": PRIMARY_ID,
        "primary_state": PROVIDER_UNAVAILABLE,
        "task_domain": TASK_DOMAIN,
        "consequence_grade": TASK_GRADE,
        "alternate_provider_id": ALTERNATE_ID,
    }
    base.update(overrides)
    return ProviderRouteRequest(**base)


def _inferential_gaps(pef: PEFState) -> list[dict]:
    merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef))
    return [u for u in pef.open_epistemic_uncertainties if u.get("kind") == "inferential_gap"]


def _build_lens(registry: SovereignProviderRegistry | None) -> tuple[Lens, MockAdapter]:
    adapter = MockAdapter(responses=["ok"])
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=SpacyBackend(model="en_core_web_sm"),
            sovereign_provider_registry=registry,
        )
    )
    return lens, adapter


def _with_provider_route(meta: RequestMetadata):
    return request_metadata_var.set(meta)


class TestSovereignRegistryEvaluation:
    def test_primary_active_allows_route_without_failover_bridge(self):
        registry = SovereignProviderRegistry([_certified_alternate_profile()])
        evaluation = registry.evaluate_failover(
            _route_request(primary_state=VALIDATED_CURRENT)
        )
        assert evaluation.outcome == ProviderRouteOutcome.USE_ACTIVE_PROVIDER
        assert evaluation.inference_contract is None
        assert not evaluation.failover_attempted
        assert not evaluation_blocks_adapter(evaluation)

        pef = PEFState()
        apply_route_evaluation_to_pef(pef, evaluation, turn=1)
        assert not _inferential_gaps(pef)

    def test_primary_unavailable_certified_alternate_valid_bridge(self):
        registry = SovereignProviderRegistry([_certified_alternate_profile()])
        evaluation = registry.evaluate_failover(_route_request())
        assert evaluation.outcome == ProviderRouteOutcome.USE_CERTIFIED_ALTERNATE
        assert evaluation.bridge_status == "valid"
        contract = evaluation.inference_contract or {}
        assert contract.get("bridge_ref") == FAILOVER_BRIDGE_REF
        assert contract.get("conclusion") == FAILOVER_CONCLUSION
        assert not evaluation_blocks_adapter(evaluation)

        pef = PEFState()
        apply_route_evaluation_to_pef(pef, evaluation, turn=1)
        assert not _inferential_gaps(pef)

    def test_alternate_missing_capability_profile_bridge_missing(self):
        registry = SovereignProviderRegistry([])
        evaluation = registry.evaluate_failover(_route_request())
        assert evaluation.bridge_status == "missing"
        assert evaluation_blocks_adapter(evaluation)

        pef = PEFState()
        apply_route_evaluation_to_pef(pef, evaluation, turn=1)
        gaps = _inferential_gaps(pef)
        assert len(gaps) == 1
        assert (gaps[0].get("meta") or {}).get("bridge_status") == "missing"

    def test_alternate_max_consequence_grade_too_low_out_of_scope(self):
        registry = SovereignProviderRegistry([
            _certified_alternate_profile(max_consequence_grade="low"),
        ])
        evaluation = registry.evaluate_failover(_route_request())
        assert evaluation.bridge_status == "out_of_scope"
        assert evaluation.reason == "max_consequence_grade_too_low"
        assert evaluation_blocks_adapter(evaluation)

        pef = PEFState()
        apply_route_evaluation_to_pef(pef, evaluation, turn=1)
        gaps = _inferential_gaps(pef)
        assert len(gaps) == 1
        assert (gaps[0].get("meta") or {}).get("bridge_status") == "out_of_scope"

    def test_validation_stale_high_consequence_bridge_expired(self):
        registry = SovereignProviderRegistry([
            _certified_alternate_profile(status=VALIDATION_STALE, max_consequence_grade="high"),
        ])
        evaluation = registry.evaluate_failover(_route_request())
        assert evaluation.bridge_status == "expired"
        assert evaluation_blocks_adapter(evaluation)

        pef = PEFState()
        apply_route_evaluation_to_pef(pef, evaluation, turn=1)
        gaps = _inferential_gaps(pef)
        assert len(gaps) == 1
        assert (gaps[0].get("meta") or {}).get("bridge_status") == "expired"

    def test_no_alternate_local_refusal(self):
        registry = SovereignProviderRegistry([_certified_alternate_profile()])
        evaluation = registry.evaluate_failover(
            _route_request(alternate_provider_id=None)
        )
        assert evaluation.outcome == ProviderRouteOutcome.LOCAL_REFUSAL
        assert evaluation.reason == "no_certified_alternate"
        assert evaluation.inference_contract is None
        assert evaluation_blocks_adapter(evaluation)

    def test_generic_relationship_without_inference_does_not_generate_gap(self):
        pef = PEFState()
        sid = Entity.create("advisory_claim", turn=1)
        pef.entities[sid.id] = sid
        pef.relationships.append(
            Relationship(
                subject_id=sid.id,
                relation="IS",
                object_entity_id=None,
                object_literal="recommended",
                span=Span.PRESENT,
                source_turn=1,
                evidence="note",
                relation_metadata={"notes": "no bridge here"},
            )
        )
        assert not _inferential_gaps(pef)

    def test_failover_without_registry_evaluation_is_rejected(self):
        evaluation = reject_failover_without_registry(_route_request())
        assert not evaluation.registry_evaluated
        assert evaluation_blocks_adapter(evaluation)
        assert evaluation.reason == "failover_registry_evaluation_required"


class TestSovereignFailoverLensIntegration:
    @pytest.mark.asyncio
    async def test_certified_alternate_allows_adapter_call(self):
        registry = SovereignProviderRegistry([_certified_alternate_profile()])
        lens, adapter = _build_lens(registry)
        meta = RequestMetadata(
            provider_route=_route_request(primary_state=PROVIDER_UNAVAILABLE),
        )
        tok = _with_provider_route(meta)
        try:
            calls_before = adapter._call_count
            r = await lens.process("Summarise the attached notes.")
            assert r.action == InterventionAction.PASS
            assert adapter._call_count == calls_before + 1
            assert not _inferential_gaps(lens.pef)
        finally:
            request_metadata_var.reset(tok)

    @pytest.mark.asyncio
    async def test_no_alternate_local_refusal_no_adapter_call(self):
        registry = SovereignProviderRegistry([_certified_alternate_profile()])
        lens, adapter = _build_lens(registry)
        meta = RequestMetadata(
            provider_route=_route_request(
                primary_state=PROVIDER_UNAVAILABLE,
                alternate_provider_id=None,
            ),
        )
        tok = _with_provider_route(meta)
        try:
            calls_before = adapter._call_count
            r = await lens.process("Summarise the attached notes.")
            assert r.action == InterventionAction.HARD_STOP
            assert adapter._call_count == calls_before
            assert "no certified alternate" in (r.response or "").lower()
        finally:
            request_metadata_var.reset(tok)

    @pytest.mark.asyncio
    async def test_missing_profile_blocks_before_adapter(self):
        registry = SovereignProviderRegistry([])
        lens, adapter = _build_lens(registry)
        meta = RequestMetadata(provider_route=_route_request())
        tok = _with_provider_route(meta)
        try:
            calls_before = adapter._call_count
            r = await lens.process("Summarise the attached notes.")
            assert r.action == InterventionAction.HARD_STOP
            assert adapter._call_count == calls_before
            assert _inferential_gaps(lens.pef)
        finally:
            request_metadata_var.reset(tok)

    @pytest.mark.asyncio
    async def test_failover_without_registry_on_lens_is_rejected(self):
        lens, adapter = _build_lens(None)
        meta = RequestMetadata(provider_route=_route_request())
        tok = _with_provider_route(meta)
        try:
            calls_before = adapter._call_count
            r = await lens.process("Summarise the attached notes.")
            assert r.action == InterventionAction.HARD_STOP
            assert adapter._call_count == calls_before
            assert "without sovereign provider registry evaluation" in (r.response or "").lower()
        finally:
            request_metadata_var.reset(tok)

    @pytest.mark.asyncio
    async def test_primary_active_no_failover_metadata_required(self):
        registry = SovereignProviderRegistry([_certified_alternate_profile()])
        lens, adapter = _build_lens(registry)
        meta = RequestMetadata(
            provider_route=_route_request(primary_state=VALIDATED_CURRENT),
        )
        tok = _with_provider_route(meta)
        try:
            calls_before = adapter._call_count
            r = await lens.process("Summarise the attached notes.")
            assert r.action == InterventionAction.PASS
            assert adapter._call_count == calls_before + 1
            sovereign_rels = [
                rel
                for rel in lens.pef.relationships
                if isinstance(rel.relation_metadata, dict)
                and (rel.relation_metadata.get("sovereign") or {}).get("source") == "failover_bridge"
            ]
            assert not sovereign_rels
        finally:
            request_metadata_var.reset(tok)
