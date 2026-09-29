"""Tests for evidence authority demotion mapping into Lens actions."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aurora_lens.corpus.evidence_admissibility import evaluate_retrieved_evidence
from aurora_lens.corpus.evidence_authority import (
    EVIDENCE_AUTHORITY_TO_LENS_ACTION,
    EvidenceAuthorityState,
    derive_evidence_authority,
    lens_action_for_evidence_authority,
)
from aurora_lens.corpus.models import CorpusChunk, CorpusRecord
from aurora_lens.govern.decision import InterventionAction


def test_authority_mapping_table():
    assert lens_action_for_evidence_authority(EvidenceAuthorityState.AUTHORITATIVE) == InterventionAction.PASS
    assert lens_action_for_evidence_authority(EvidenceAuthorityState.SIGNAL_ONLY) == InterventionAction.FORCE_REVISE
    assert lens_action_for_evidence_authority(
        EvidenceAuthorityState.REQUIRES_REVALIDATION
    ) == InterventionAction.CONTAIN
    assert lens_action_for_evidence_authority(EvidenceAuthorityState.INADMISSIBLE) == InterventionAction.HARD_STOP
    assert len(EVIDENCE_AUTHORITY_TO_LENS_ACTION) == 4


def _record(**overrides) -> CorpusRecord:
    base = dict(
        record_id="policy-v1",
        title="Policy V1",
        source_path="docs/policy-v1.md",
        sha256="deadbeef",
        ingested_at="2026-06-29T00:00:00+00:00",
        chunk_count=1,
        authority_class="corporate_policy",
        status="approved",
        effective_from="2026-01-01",
        effective_until="2026-12-31",
        scope="travel",
        parser_status="ok",
    )
    base.update(overrides)
    return CorpusRecord(**base)


def _chunks() -> list[CorpusChunk]:
    return [
        CorpusChunk.build(
            record_id="policy-v1",
            ordinal=0,
            locator="Section 1",
            text="Travel over threshold needs manager approval.",
        )
    ]


def test_admissible_evidence_is_authoritative():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(),
        now=datetime(2026, 6, 15, tzinfo=timezone.utc),
        request_scope="travel",
    )
    assert result.authority_state == EvidenceAuthorityState.AUTHORITATIVE.value
    assert lens_action_for_evidence_authority(result.authority_state) == InterventionAction.PASS


def test_stale_evidence_is_signal_only_not_absent():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status="expired", effective_until="2020-01-01"),
        now=datetime(2026, 6, 15, tzinfo=timezone.utc),
    )
    assert result.failure_kind == "stale_evidence"
    assert result.authority_state == EvidenceAuthorityState.SIGNAL_ONLY.value
    assert result.freshness_status is not None
    assert lens_action_for_evidence_authority(result.authority_state) == InterventionAction.FORCE_REVISE


def test_unresolved_authority_requires_revalidation():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(authority_class=""),
        require_authority=True,
    )
    assert result.status == "unresolved"
    assert result.authority_state == EvidenceAuthorityState.REQUIRES_REVALIDATION.value
    assert lens_action_for_evidence_authority(result.authority_state) == InterventionAction.CONTAIN


def test_authority_missing_is_inadmissible():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status="draft"),
    )
    assert result.authority_state == EvidenceAuthorityState.INADMISSIBLE.value
    assert lens_action_for_evidence_authority(result.authority_state) == InterventionAction.HARD_STOP


def test_unknown_record_status_is_not_authoritative_and_not_pass():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status="revokedd"),
    )
    assert result.authority_state != EvidenceAuthorityState.AUTHORITATIVE.value
    assert result.authority_state == EvidenceAuthorityState.REQUIRES_REVALIDATION.value
    assert lens_action_for_evidence_authority(result.authority_state) != InterventionAction.PASS
    assert lens_action_for_evidence_authority(result.authority_state) == InterventionAction.CONTAIN


def test_missing_record_status_is_not_authoritative_and_not_pass():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status=None),
    )
    assert result.authority_state != EvidenceAuthorityState.AUTHORITATIVE.value
    assert lens_action_for_evidence_authority(result.authority_state) != InterventionAction.PASS
    assert lens_action_for_evidence_authority(result.authority_state) == InterventionAction.CONTAIN


def test_revoked_record_status_is_inadmissible_not_pass():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status="revoked"),
    )
    assert result.authority_state == EvidenceAuthorityState.INADMISSIBLE.value
    assert lens_action_for_evidence_authority(result.authority_state) != InterventionAction.PASS
    assert lens_action_for_evidence_authority(result.authority_state) == InterventionAction.HARD_STOP


def test_unknown_authority_class_is_not_authoritative_and_not_pass():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(authority_class="untrusted"),
        require_authority=True,
    )
    assert result.failure_kind == "unknown_authority_class"
    assert result.authority_state != EvidenceAuthorityState.AUTHORITATIVE.value
    assert result.authority_state == EvidenceAuthorityState.REQUIRES_REVALIDATION.value
    assert lens_action_for_evidence_authority(result.authority_state) != InterventionAction.PASS
    assert lens_action_for_evidence_authority(result.authority_state) == InterventionAction.CONTAIN


def test_typo_authority_class_is_not_authoritative_and_not_pass():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(authority_class="verifed"),
        require_authority=True,
    )
    assert result.authority_state != EvidenceAuthorityState.AUTHORITATIVE.value
    assert lens_action_for_evidence_authority(result.authority_state) != InterventionAction.PASS


def test_unknown_present_authority_class_not_authoritative_even_when_not_required():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(authority_class="untrusted"),
        require_authority=False,
    )
    assert result.failure_kind == "unknown_authority_class"
    assert result.authority_state != EvidenceAuthorityState.AUTHORITATIVE.value


def test_unknown_present_authority_class_not_pass_even_when_not_required():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(authority_class="verifed"),
        require_authority=False,
    )
    assert lens_action_for_evidence_authority(result.authority_state) != InterventionAction.PASS
    assert lens_action_for_evidence_authority(result.authority_state) == InterventionAction.CONTAIN


def test_derive_evidence_authority_typo_admissible_is_inadmissible():
    state, _, _, _ = derive_evidence_authority(status="admisssible", failure_kind=None)
    assert state == EvidenceAuthorityState.INADMISSIBLE


def test_derive_evidence_authority_unknown_failure_kind_is_inadmissible():
    state, _, _, _ = derive_evidence_authority(
        status="inadmissible", failure_kind="totally_unknown_kind"
    )
    assert state == EvidenceAuthorityState.INADMISSIBLE


@pytest.mark.asyncio
async def test_lens_rag_gate_maps_signal_only_to_force_revise(monkeypatch: pytest.MonkeyPatch):
    from aurora_lens.interpret.pef_admission import PEFAdmissionDecision, PEFAdmissionResult
    from aurora_lens.interpret.schema import ExtractedClaim, Span
    from aurora_lens.lens import Lens, LensConfig
    from aurora_lens.govern.decision import InterventionAction
    from tests.test_lens import MockAdapter, _RagGateBackend

    adapter = MockAdapter(responses=["NEVER_RETURNED"])
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_RagGateBackend(
                [ExtractedClaim(subject="Mina", relation="is", obj="Osaka", span=Span.PRESENT, negated=False, evidence="ctx")],
                [],
            ),
        )
    )

    def _fake_admit(ext_ctx, pef, *, context_block, request_metadata, **kwargs):
        pef.retrieval_unresolved = [
            {
                "kind": "evidence_inadmissible",
                "failure_kind": "stale_evidence",
                "authority_state": "signal_only",
                "freshness_status": "effective_until",
                "demotion_reason": "stale_evidence",
                "chunk_ids": ["r1--chunk-0000"],
                "reasons": ["effective_until is stale."],
                "failure_meta": {"stale_source": "effective_until"},
            }
        ]
        return PEFAdmissionResult(
            decision=PEFAdmissionDecision.REJECT,
            write_intent=False,
            planned_mutation_slices=len(ext_ctx.claims),
            world_state_mutation_count=0,
            continuation_state_mutation_count=0,
            mutation_count=0,
            mutation_summary={},
            evidence=[],
        )

    monkeypatch.setattr("aurora_lens.lens.admit_retrieved_context_to_pef", _fake_admit)
    result = await lens.process("Context:\n### Section A\nMina is in Osaka.\n\nQuestion: Where is Mina?")
    assert adapter._call_count == 0
    assert result.action == InterventionAction.FORCE_REVISE
    assert "stale_evidence" in (result.response or "")
