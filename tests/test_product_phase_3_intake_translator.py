"""Product Phase 3 intake translator tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aurora_lens.corpus.document_ingestion import propose_document_ingestion
from aurora_lens.corpus.establishment_failure_proposal import (
    PROPOSAL_STATUS_ADMITTED_FOR_EVALUATION,
    PROPOSAL_STATUS_PROPOSED,
    EstablishmentFailureProposal,
)
from aurora_lens.corpus.ingestion_report import ADMISSIBILITY_DECISION_NONE
from aurora_lens.corpus.intake_translator import (
    parse_labeled_plain_english_assertion,
    preview_gate_evaluation,
    proposals_to_gate_uncertainties,
    translate_candidate_record,
    translate_intake_batch,
    translate_plain_english_intake,
    translate_structured_intake,
)

_SAMPLE_MD = """\
### Section 1
Alpha evidence for supplier review.
"""


class TestStructuredIntakeTranslation:
    def test_structured_intake_to_proposal(self):
        proposal = translate_structured_intake(
            {
                "kind": "scope_mismatch",
                "description": "Evidence scope is NSW; claim requires AU-wide jurisdiction.",
                "bears_on": ["supplier.approval"],
                "meta": {"policy_ref": "scope.policy.v1"},
            }
        )
        assert proposal.kind == "scope_mismatch"
        assert proposal.status == PROPOSAL_STATUS_PROPOSED
        assert proposal.establishes_truth is False
        assert proposal.admissibility_decision == ADMISSIBILITY_DECISION_NONE
        assert proposal.meta["policy_ref"] == "scope.policy.v1"

    def test_rejects_invalid_kind(self):
        with pytest.raises(ValueError, match="invalid establishment-failure kind"):
            translate_structured_intake(
                {
                    "kind": "not_a_real_kind",
                    "description": "x",
                    "bears_on": ["a.b"],
                }
            )


class TestPlainEnglishIntakeTranslation:
    def test_labeled_assertion_parses_exact_kind_token(self):
        kind, desc = parse_labeled_plain_english_assertion(
            "threshold_not_met: Observed strength 0.4 below required 0.8."
        )
        assert kind == "threshold_not_met"
        assert "0.4" in desc

    def test_unlabeled_plain_english_rejected(self):
        with pytest.raises(ValueError, match="labeled form"):
            translate_plain_english_intake(
                "The evidence seems weak for this decision.",
                bears_on=["claim.x"],
            )


class TestCandidateRecordTranslation:
    def test_candidate_plus_explicit_intake_fields(self, tmp_path: Path):
        src = tmp_path / "sample.md"
        src.write_text(_SAMPLE_MD, encoding="utf-8")
        report = propose_document_ingestion(src, record_id="doc-1")
        candidate = report.candidates[0]
        proposal = translate_candidate_record(
            candidate,
            kind="inferential_gap",
            description="Committed evidence does not bridge to the approval conclusion.",
            bears_on=["supplier.approval"],
        )
        assert proposal.origin == "candidate_record"
        assert proposal.source_candidate_id == candidate.candidate_id
        assert proposal.provenance is not None
        assert proposal.provenance.chunk_id == candidate.provenance.chunk_id

    def test_candidate_meta_intake_fields(self, tmp_path: Path):
        src = tmp_path / "sample.md"
        src.write_text(_SAMPLE_MD, encoding="utf-8")
        candidate = propose_document_ingestion(src, record_id="doc-1").candidates[0]
        from aurora_lens.corpus.ingestion_report import CandidateRecord

        candidate_with_meta = CandidateRecord(
            candidate_id=candidate.candidate_id,
            kind=candidate.kind,
            status=candidate.status,
            text=candidate.text,
            provenance=candidate.provenance,
            meta={
                "intake_kind": "stale_evidence",
                "intake_description": "Evidence timestamp exceeds declared freshness window.",
                "intake_bears_on": ["weather.reading"],
            },
        )
        proposal = translate_candidate_record(candidate_with_meta)
        assert proposal.kind == "stale_evidence"


class TestGateBridge:
    def test_admitted_proposal_reaches_gate_payload(self):
        proposal = translate_structured_intake(
            {
                "kind": "threshold_not_met",
                "description": "Observed strength below required threshold.",
                "bears_on": ["risk.score"],
            }
        )
        admitted = proposal.with_admitted_for_evaluation()
        assert admitted.status == PROPOSAL_STATUS_ADMITTED_FOR_EVALUATION
        records = proposals_to_gate_uncertainties([admitted])
        assert len(records) == 1
        assert records[0]["kind"] == "threshold_not_met"
        assert records[0]["status"] == "open"

    def test_preview_gate_blocks_decision_turn(self):
        batch = translate_intake_batch(
            structured=[
                {
                    "kind": "inferential_gap",
                    "description": "Bridge from evidence to conclusion is missing.",
                    "bears_on": ["decision.option"],
                }
            ]
        )
        result = preview_gate_evaluation(
            batch.proposals,
            "Which option should we choose?",
        )
        assert result is not None
        assert result.blocks is True

    def test_proposed_only_cannot_emit_gate_payload(self):
        proposal = translate_structured_intake(
            {
                "kind": "scope_mismatch",
                "description": "Scope mismatch declared.",
                "bears_on": ["claim.a"],
            }
        )
        with pytest.raises(ValueError, match="admitted_for_evaluation"):
            proposal.to_open_epistemic_uncertainty()

    def test_intake_translator_does_not_import_lens(self):
        import aurora_lens.corpus.intake_translator as mod

        source = open(mod.__file__, encoding="utf-8").read()
        assert "aurora_lens.lens" not in source
        assert "PEFState" not in source

    def test_proposal_json_round_trip(self):
        proposal = translate_plain_english_intake(
            "source_untrusted: Vendor bulletin lacks required certification.",
            bears_on=["evidence.source"],
            meta={"source_id": "source:vendor-17"},
        )
        restored = EstablishmentFailureProposal.from_dict(
            json.loads(json.dumps(proposal.to_dict()))
        )
        assert restored.kind == proposal.kind
        assert restored.establishes_truth is False
