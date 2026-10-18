"""Synthetic regression matrix for request retention, independent of rulings."""
import hashlib
import pytest

from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.govern.evidence_capture import EvidenceCaptureConfig, capture_governance_evidence
from aurora_lens.govern.evidence_vault import EvidenceAccessDenied, create_local_evidence_vault
from aurora_lens.verify.consequence_intent import LOW_RISK_CONVERSATIONAL


@pytest.mark.parametrize("action", list(InterventionAction))
@pytest.mark.parametrize("policy", ["strict", "moderate"])
@pytest.mark.parametrize("basis", [None, LOW_RISK_CONVERSATIONAL, "synthetic_established_basis"])
def test_sealed_captures_every_request_and_recovers_after_reopen(tmp_path, action, policy, basis):
    path = tmp_path / "audit.jsonl"
    key = "synthetic-all-request-encryption-material"
    prompt = "Synthetic request\u202fwith exact Unicode\u2011bytes."
    vault = create_local_evidence_vault(path, encryption_key=key)
    decision = GovernanceDecision(action=action, policy=policy, flags=[], rationale="Synthetic matrix",
        admissibility_basis=basis, original_response="Synthetic upstream", governed_response="Synthetic released")
    result = capture_governance_evidence(decision, pre_llm=False, vault=vault,
        config=EvidenceCaptureConfig(capture_mode="sealed", strict_pass_low_risk_mode="hash_only",
            moderate_pass_mode="hash_only", moderate_pass_sealed_when_configured=False), request_prompt=prompt)
    manifest = result.request_evidence_manifest
    assert result.request_capture_status == "sealed_by_policy" and result.evidence_capture_mode == "sealed"
    assert manifest["protection_mode"] == "sealed" and manifest["capture_status"] == "captured"
    assert manifest["raw_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    reopened = create_local_evidence_vault(path, encryption_key=key)
    assert reopened.get_evidence(manifest["evidence_ref"], "forensic_review", "forensic_reviewer") == prompt.encode()
    assert prompt.encode() not in reopened._blob_path(manifest["evidence_ref"]).read_bytes()
    with pytest.raises(EvidenceAccessDenied):
        reopened.get_evidence(manifest["evidence_ref"], "forensic_review", "anonymous")
    assert decision.action is action and decision.policy == policy and not decision.flags
