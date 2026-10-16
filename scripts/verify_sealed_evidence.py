"""Synthetic sealed-capture/restart check; print only PASS/FAIL, never evidence.

Run capture before restart and recover afterwards against the same configured
audit path and key. Uses the production capture function and vault, without
calling an upstream model or adding synthetic governance decisions to the ledger.
The marker contains synthetic refs/hashes only and remains in the vault.
"""

from __future__ import annotations

import argparse
import json
import os
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.govern.evidence_capture import (
    EvidenceCaptureConfig,
    attach_evidence_fields_to_audit_entry,
)
from aurora_lens.govern.evidence_vault import (
    EVIDENCE_VAULT_BLOB_MAGIC,
    EvidenceAccessDenied,
    create_local_evidence_vault,
)
from aurora_lens.proxy.config import ProxyConfig


def check(phase: str, config_path: str) -> None:
    cfg = ProxyConfig.from_yaml(config_path).governance
    if cfg.evidence_capture_mode != "sealed" or not cfg.evidence_encryption_key:
        raise RuntimeError("sealed configuration required")
    if not cfg.audit_log or "{" in cfg.audit_log:
        raise RuntimeError("stable audit path required")
    vault = create_local_evidence_vault(
        cfg.audit_log, capture_mode="sealed", encryption_key=cfg.evidence_encryption_key
    )
    if vault is None or vault.capture_mode.value != "sealed":
        raise RuntimeError("sealed vault required")
    marker = vault.root / ".synthetic-sealed-check.json"
    expected = (b"Synthetic evidence request fixture.", b"Synthetic withheld upstream fixture.")
    if phase == "capture":
        decision = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=[],
            rationale="synthetic storage verification",
            original_response=expected[1].decode(),
        )
        row = {"original_response": decision.original_response}
        result = attach_evidence_fields_to_audit_entry(
            row, decision, pre_llm=False, vault=vault,
            config=EvidenceCaptureConfig(capture_mode="sealed"),
            request_prompt=expected[0].decode(),
        )
        if result.request_capture_status != "sealed_by_policy":
            raise RuntimeError("sealed receipt required")
        manifests = [result.request_evidence_manifest, result.upstream_evidence_manifest]
        if "original_response" in row or any(x.decode() in json.dumps(row) for x in expected):
            raise RuntimeError("inline content detected")
        with marker.open("w", encoding="utf-8") as handle:
            os.chmod(marker, 0o600)
            json.dump(manifests, handle)
    else:
        manifests = json.loads(marker.read_text(encoding="utf-8"))
    if len(manifests) != 2:
        raise RuntimeError("two manifests required")
    for manifest, content in zip(manifests, expected):
        if not manifest or manifest["protection_mode"] != "sealed":
            raise RuntimeError("sealed manifest required")
        ref = manifest["evidence_ref"]
        blob = vault._blob_path(ref).read_bytes()
        if not blob.startswith(EVIDENCE_VAULT_BLOB_MAGIC) or content in blob:
            raise RuntimeError("encrypted blob required")
        if vault.get_evidence(ref, "forensic_review", "forensic_reviewer") != content:
            raise RuntimeError("recovery mismatch")
        if not vault.verify_evidence(ref, manifest["raw_sha256"]):
            raise RuntimeError("hash verification failed")
        try:
            vault.get_evidence(ref, "forensic_review", "anonymous")
        except EvidenceAccessDenied:
            pass
        else:
            raise RuntimeError("unauthorised retrieval allowed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("capture", "recover"))
    parser.add_argument("--config", default=os.environ.get("AURORA_LENS_CONFIG", "/app/aurora-lens.yaml"))
    args = parser.parse_args()
    try:
        check(args.phase, args.config)
    except Exception:
        print(f"FAIL: synthetic sealed evidence {args.phase}")
        raise SystemExit(1) from None
    print(f"PASS: synthetic sealed evidence {args.phase}; encryption, authorised recovery, hashes, access denial")
