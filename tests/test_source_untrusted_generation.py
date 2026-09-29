"""Track C — deterministic source_untrusted generation and gate parity."""

from __future__ import annotations

import pytest

from aurora_lens.config import LensConfig
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.govern.epistemic_uncertainty_gate import evaluate_epistemic_uncertainty_gate
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.pef.uncertainty_analysis import (
    merge_epistemic_uncertainties,
    pef_uncertainty_analysis,
)
from aurora_lens.trust import TRACK_C_ESTABLISHMENT_FAILURE_IMPLEMENTATION
from aurora_lens.trust.trust_profile import TRUST_STATUS_UNTRUSTED, TrustSourceProfile
from aurora_lens.trust.trust_registry import TrustRegistry
from tests.test_lens import MockAdapter

_ID_PREFIX = "generated_source_untrusted:"
_REGISTRY_REF = "registry:ops:v1"


def _registry() -> TrustRegistry:
    return TrustRegistry(
        [
            TrustSourceProfile(
                source_id="source:vendor-17",
                source_name="Vendor 17 Bulletin",
                source_class="vendor_bulletin",
                trust_status=TRUST_STATUS_UNTRUSTED,
                permitted_domains=("medical",),
                max_consequence_grade="high",
            ),
            TrustSourceProfile(
                source_id="src:trusted_clinical",
                source_name="Trusted Clinical Source",
                source_class="clinical_guideline",
                trust_status="trusted",
                permitted_domains=("medical",),
                max_consequence_grade="critical",
            ),
        ],
        registry_ref=_REGISTRY_REF,
    )


def _trust_rel(**trust_overrides) -> Relationship:
    e = Entity.create("evidence_claim", turn=1)
    trust = {
        "trust_evaluation_required": True,
        "source_id": "source:vendor-17",
        "trust_registry_ref": _REGISTRY_REF,
        "task_domain": "medical",
        "consequence_grade": "high",
        "policy_ref": "ops.source.v1",
    }
    trust.update(trust_overrides)
    return Relationship(
        subject_id=e.id,
        relation="IS",
        object_entity_id=None,
        object_literal="approved",
        span=Span.PRESENT,
        source_turn=1,
        evidence="vendor note",
        relation_metadata={"trust": trust},
    )


class TestTrackCPosture:
    def test_source_untrusted_is_track_c_deterministic_generated(self):
        assert (
            TRACK_C_ESTABLISHMENT_FAILURE_IMPLEMENTATION["source_untrusted"]
            == "deterministic_generated"
        )


class TestSourceUntrustedGeneration:
    def test_generates_when_contract_complete_and_source_untrusted(self):
        pef = PEFState()
        rel = _trust_rel()
        pef.relationships.append(rel)
        records = pef_uncertainty_analysis(pef, trust_registry=_registry())
        generated = [r for r in records if r["kind"] == "source_untrusted"]
        assert len(generated) == 1
        rec = generated[0]
        assert str(rec["id"]).startswith(_ID_PREFIX)
        meta = rec.get("meta") or {}
        assert meta.get("source_id") == "source:vendor-17"
        assert meta.get("trust_decision") == "deny"
        assert meta.get("trust_basis") == "source_marked_untrusted"

    def test_no_generation_without_trust_evaluation_required(self):
        pef = PEFState()
        rel = _trust_rel()
        rel.relation_metadata["trust"].pop("trust_evaluation_required")
        pef.relationships.append(rel)
        records = pef_uncertainty_analysis(pef, trust_registry=_registry())
        assert not any(r["kind"] == "source_untrusted" for r in records)

    def test_no_generation_without_registry(self):
        pef = PEFState()
        pef.relationships.append(_trust_rel())
        records = pef_uncertainty_analysis(pef, trust_registry=None)
        assert not any(r["kind"] == "source_untrusted" for r in records)

    def test_no_generation_for_unknown_source(self):
        pef = PEFState()
        pef.relationships.append(_trust_rel(source_id="source:unknown-99"))
        records = pef_uncertainty_analysis(pef, trust_registry=_registry())
        assert not any(r["kind"] == "source_untrusted" for r in records)

    def test_no_generation_when_source_admissible(self):
        pef = PEFState()
        pef.relationships.append(
            _trust_rel(source_id="src:trusted_clinical", consequence_grade="moderate")
        )
        records = pef_uncertainty_analysis(pef, trust_registry=_registry())
        assert not any(r["kind"] == "source_untrusted" for r in records)

    def test_registry_ref_mismatch_does_not_generate(self):
        pef = PEFState()
        pef.relationships.append(
            _trust_rel(trust_registry_ref="registry:other:v1")
        )
        records = pef_uncertainty_analysis(pef, trust_registry=_registry())
        assert not any(r["kind"] == "source_untrusted" for r in records)

    def test_stable_ids_across_reanalysis(self):
        pef = PEFState()
        pef.relationships.append(_trust_rel())
        reg = _registry()
        r1 = [r for r in pef_uncertainty_analysis(pef, trust_registry=reg) if r["kind"] == "source_untrusted"]
        r2 = [r for r in pef_uncertainty_analysis(pef, trust_registry=reg) if r["kind"] == "source_untrusted"]
        assert len(r1) == 1 and len(r2) == 1
        assert r1[0]["id"] == r2[0]["id"]

    def test_merge_reconciles_stale_generated_records(self):
        pef = PEFState()
        pef.relationships.append(_trust_rel())
        reg = _registry()
        merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef, trust_registry=reg))
        pef.relationships[0].relation_metadata["trust"]["source_id"] = "src:trusted_clinical"
        merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef, trust_registry=reg))
        kinds = [u.get("kind") for u in pef.open_epistemic_uncertainties]
        assert kinds.count("source_untrusted") == 0


class TestSourceUntrustedGateParity:
    def test_generated_record_blocks_decision_turn(self):
        pef = PEFState()
        pef.relationships.append(_trust_rel())
        reg = _registry()
        records = pef_uncertainty_analysis(pef, trust_registry=reg)
        result = evaluate_epistemic_uncertainty_gate(
            records,
            "Which option should we choose?",
        )
        assert result is not None
        assert result.blocks is True

    def test_non_decision_turn_does_not_block(self):
        pef = PEFState()
        pef.relationships.append(_trust_rel())
        records = pef_uncertainty_analysis(pef, trust_registry=_registry())
        result = evaluate_epistemic_uncertainty_gate(
            records,
            "Summarise the vendor bulletin.",
        )
        assert result is None


class TestSourceUntrustedLensIntegration:
    @staticmethod
    def _build_lens() -> tuple[Lens, MockAdapter]:
        adapter = MockAdapter(responses=["unused"])
        lens = Lens(
            LensConfig(
                adapter=adapter,
                extraction_backend=SpacyBackend(model="en_core_web_sm"),
                trust_registry=_registry(),
            )
        )
        rel = _trust_rel()
        lens.pef.relationships.append(rel)
        return lens, adapter

    @pytest.mark.asyncio
    async def test_lens_blocks_decision_turn_when_source_untrusted_generated(self):
        lens, adapter = self._build_lens()
        calls_before = adapter._call_count
        r = await lens.process("Which option should we choose?")
        assert r.action != InterventionAction.PASS
        assert adapter._call_count == calls_before
        generated = [
            u
            for u in lens.pef.open_epistemic_uncertainties
            if u.get("kind") == "source_untrusted"
        ]
        assert generated
        assert "does not establish a justified recommendation" in (r.response or "").lower()
