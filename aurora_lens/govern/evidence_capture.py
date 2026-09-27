"""Governance evidence capture policy and audit-field attachment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aurora_lens.context import request_prompt_var
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.govern.evidence_vault import (
    CaptureMode,
    CaptureStatus,
    EvidenceKind,
    EvidenceManifest,
    EvidenceVault,
    EVIDENCE_ACCESS_POLICY_AUDIT_ONLY,
    LocalSealedEvidenceVault,
    create_local_evidence_vault,
    format_sha256_digest,
    public_evidence_ref,
    resolve_evidence_encryption_key,
)
from aurora_lens.verify.consequence_intent import LOW_RISK_CONVERSATIONAL

SEALED_RECEIPT_SUMMARY = (
    "Prompt content was sealed by policy. Aurora-Lens retained hash-verifiable "
    "evidence for audit without exposing forbidden material."
)
HASH_ONLY_RECEIPT_SUMMARY = (
    "Prompt content was not stored inline. Aurora-Lens retained hash-verifiable "
    "evidence for audit without exposing forbidden material."
)


@dataclass
class EvidenceCaptureConfig:
    """Deployment capture policy for the evidence vault."""

    capture_mode: str = CaptureMode.SEALED.value
    strict_pass_low_risk_mode: str = CaptureMode.HASH_ONLY.value
    moderate_pass_mode: str = CaptureMode.HASH_ONLY.value
    moderate_pass_sealed_when_configured: bool = False
    encryption_key: str | bytes | None = None


@dataclass
class EvidenceCaptureResult:
    """Audit-plane evidence fields produced for one governance decision."""

    request_capture_status: str
    request_evidence_ref: str | None
    request_raw_sha256: str | None
    request_canonical_sha256: str | None
    evidence_capture_mode: str
    evidence_retention_class: str
    evidence_access_policy: str | None = None
    request_evidence_manifest: dict[str, Any] | None = None
    upstream_evidence_manifest: dict[str, Any] | None = None
    governed_evidence_manifest: dict[str, Any] | None = None
    evidence_manifests: list[dict[str, Any]] = field(default_factory=list)
    strip_original_response: bool = False
    strip_governed_response: bool = False


def _ledger_request_capture_status(
    manifest: EvidenceManifest,
    *,
    configured_mode: str,
) -> str:
    if (
        manifest.capture_status == CaptureStatus.CAPTURED.value
        and manifest.protection_mode == CaptureMode.SEALED.value
        and configured_mode == CaptureMode.SEALED.value
    ):
        return CaptureStatus.SEALED_BY_POLICY.value
    return manifest.capture_status


def _public_request_evidence_fields(manifest: EvidenceManifest) -> dict[str, str]:
    return {
        "request_evidence_ref": public_evidence_ref(
            manifest.evidence_ref,
            protection_mode=manifest.protection_mode,
        ),
        "request_raw_sha256": format_sha256_digest(manifest.raw_sha256),
        "request_canonical_sha256": format_sha256_digest(manifest.canonical_sha256),
    }


def build_evidence_receipt_snapshot(result: EvidenceCaptureResult) -> dict[str, Any] | None:
    """Operator-safe receipt fields for sealed/hash-only request capture."""
    status = result.request_capture_status
    if status in {CaptureStatus.OMITTED.value, CaptureStatus.FORBIDDEN_BY_POLICY.value}:
        return None
    out: dict[str, Any] = {
        "request_capture_status": status,
        "evidence_capture_mode": result.evidence_capture_mode,
    }
    if result.evidence_access_policy:
        out["evidence_access_policy"] = result.evidence_access_policy
    if result.request_evidence_ref:
        out["request_evidence_ref"] = result.request_evidence_ref
    if result.request_raw_sha256:
        out["request_raw_sha256"] = result.request_raw_sha256
    if result.request_canonical_sha256:
        out["request_canonical_sha256"] = result.request_canonical_sha256
    if status == CaptureStatus.SEALED_BY_POLICY.value:
        out["summary"] = SEALED_RECEIPT_SUMMARY
    elif status == CaptureStatus.HASH_ONLY.value:
        out["summary"] = HASH_ONLY_RECEIPT_SUMMARY
    return out


def finalize_audit_receipt_snapshot(
    decision: "GovernanceDecision",
    *,
    chain_fields: dict[str, Any],
) -> None:
    """Merge chain linkage with evidence receipt without dropping either plane."""
    from aurora_lens.govern.decision import GovernanceDecision

    if not isinstance(decision, GovernanceDecision):
        return
    merged: dict[str, Any] = {}
    evidence_snap = decision.evidence_receipt_snapshot
    if isinstance(evidence_snap, dict):
        merged.update(evidence_snap)
    merged.update(chain_fields)
    decision.audit_receipt_snapshot = merged


def _vault_protection_mode(
    vault: EvidenceVault | LocalSealedEvidenceVault,
    requested_mode: str,
) -> str:
    """Never request sealed storage from a vault that cannot provide it."""
    if not isinstance(vault, LocalSealedEvidenceVault):
        return requested_mode
    if requested_mode == CaptureMode.SEALED.value:
        if vault.capture_mode == CaptureMode.HASH_ONLY or not vault.master_key:
            return CaptureMode.HASH_ONLY.value
    return requested_mode


def _effective_capture_mode(config: EvidenceCaptureConfig, decision: GovernanceDecision) -> str:
    policy = str(decision.policy or "strict").strip().lower()
    if decision.action == InterventionAction.PASS:
        basis = str(decision.admissibility_basis or decision.pass_reason_code or "").strip()
        if policy == "moderate":
            if config.moderate_pass_sealed_when_configured:
                return CaptureMode.SEALED.value
            return config.moderate_pass_mode
        if basis == LOW_RISK_CONVERSATIONAL:
            return config.strict_pass_low_risk_mode
        if basis:
            return CaptureMode.SEALED.value
        return config.capture_mode
    return config.capture_mode


def _llm_was_called(decision: GovernanceDecision, *, pre_llm: bool) -> bool:
    if pre_llm:
        return False
    upstream = decision.original_response
    return upstream is not None and str(upstream).strip() != ""


def _should_strip_inline_from_ledger(mode: str) -> bool:
    """Production modes vault content; dev plaintext keeps inline responses for debugging."""
    return mode != CaptureMode.PLAINTEXT_DEV.value


def _vault_can_capture(vault: EvidenceVault | LocalSealedEvidenceVault, mode: str) -> bool:
    vault_mode = (
        vault.capture_mode.value
        if isinstance(vault, LocalSealedEvidenceVault)
        else mode
    )
    if vault_mode == CaptureMode.SEALED.value and not getattr(vault, "master_key", None):
        return False
    return True


def capture_governance_evidence(
    decision: GovernanceDecision,
    *,
    pre_llm: bool,
    vault: EvidenceVault | LocalSealedEvidenceVault | None,
    config: EvidenceCaptureConfig,
    request_prompt: str | None = None,
) -> EvidenceCaptureResult:
    """Apply capture policy; return audit fields and manifest summaries."""
    mode = _effective_capture_mode(config, decision)
    retention_class = "standard"
    manifests: list[dict[str, Any]] = []
    request_manifest: EvidenceManifest | None = None
    upstream_manifest: EvidenceManifest | None = None
    governed_manifest: EvidenceManifest | None = None
    strip_original = False
    strip_governed = False

    prompt = (request_prompt if request_prompt is not None else request_prompt_var.get(None)) or ""
    prompt = str(prompt)

    action = decision.action
    llm_called = _llm_was_called(decision, pre_llm=pre_llm)

    must_capture_request = action in (
        InterventionAction.HARD_STOP,
        InterventionAction.CONTAIN,
        InterventionAction.FORCE_REVISE,
    ) or action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT)

    request_status = CaptureStatus.OMITTED.value
    if must_capture_request and not prompt.strip():
        request_status = CaptureStatus.OMITTED.value
    elif must_capture_request and vault is None:
        request_status = CaptureStatus.FORBIDDEN_BY_POLICY.value
    elif must_capture_request:
        protection_mode = _vault_protection_mode(vault, mode)
        meta = {
            "capture_basis": f"governance:{action.name.lower()}",
            "retention_class": retention_class,
            "protection_mode": protection_mode,
        }
        request_manifest = vault.put_evidence(
            EvidenceKind.REQUEST_PROMPT,
            prompt,
            metadata=meta,
        )
        request_status = _ledger_request_capture_status(
            request_manifest,
            configured_mode=protection_mode,
        )
        manifests.append(request_manifest.to_dict())
        if _should_strip_inline_from_ledger(mode):
            strip_original = True

    if action == InterventionAction.HARD_STOP:
        if pre_llm:
            pass
        elif llm_called and vault is not None and _vault_can_capture(vault, mode) and decision.original_response:
            upstream_manifest = vault.put_evidence(
                EvidenceKind.UPSTREAM_MODEL_OUTPUT,
                str(decision.original_response),
                metadata={
                    "capture_basis": "hard_stop_post_llm",
                    "protection_mode": _vault_protection_mode(vault, mode),
                },
            )
            manifests.append(upstream_manifest.to_dict())
            if _should_strip_inline_from_ledger(mode):
                strip_original = True

    elif action == InterventionAction.CONTAIN:
        if llm_called and vault is not None and _vault_can_capture(vault, mode) and decision.original_response:
            upstream_manifest = vault.put_evidence(
                EvidenceKind.UPSTREAM_MODEL_OUTPUT,
                str(decision.original_response),
                metadata={"capture_basis": "contain_upstream"},
            )
            manifests.append(upstream_manifest.to_dict())
            if _should_strip_inline_from_ledger(mode):
                strip_original = True
        governed_text = decision.governed_response or decision.corrected_response
        if vault is not None and _vault_can_capture(vault, mode) and governed_text and str(governed_text).strip():
            governed_manifest = vault.put_evidence(
                EvidenceKind.GOVERNED_OUTPUT,
                str(governed_text),
                metadata={"capture_basis": "contain_governed"},
            )
            manifests.append(governed_manifest.to_dict())
            if _should_strip_inline_from_ledger(mode):
                strip_governed = True

    elif action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT):
        governed_text = decision.governed_response or decision.corrected_response
        if (
            mode == CaptureMode.SEALED.value
            and vault is not None
            and _vault_can_capture(vault, mode)
            and governed_text
            and str(governed_text).strip()
        ):
            governed_manifest = vault.put_evidence(
                EvidenceKind.GOVERNED_OUTPUT,
                str(governed_text),
                metadata={"capture_basis": f"pass:{decision.admissibility_basis or 'admitted'}"},
            )
            manifests.append(governed_manifest.to_dict())
            if _should_strip_inline_from_ledger(mode):
                strip_governed = True

    elif action == InterventionAction.FORCE_REVISE:
        if llm_called and vault is not None and _vault_can_capture(vault, mode) and decision.original_response:
            upstream_manifest = vault.put_evidence(
                EvidenceKind.UPSTREAM_MODEL_OUTPUT,
                str(decision.original_response),
                metadata={"capture_basis": "force_revise_upstream"},
            )
            manifests.append(upstream_manifest.to_dict())
            if _should_strip_inline_from_ledger(mode):
                strip_original = True
        governed_text = decision.governed_response or decision.corrected_response
        if vault is not None and _vault_can_capture(vault, mode) and governed_text and str(governed_text).strip():
            governed_manifest = vault.put_evidence(
                EvidenceKind.GOVERNED_OUTPUT,
                str(governed_text),
                metadata={"capture_basis": "force_revise_governed"},
            )
            manifests.append(governed_manifest.to_dict())
            if _should_strip_inline_from_ledger(mode):
                strip_governed = True

    if action in (InterventionAction.HARD_STOP, InterventionAction.CONTAIN, InterventionAction.FORCE_REVISE):
        if request_status == CaptureStatus.OMITTED.value and not prompt.strip():
            request_status = CaptureStatus.FORBIDDEN_BY_POLICY.value

    evidence_access_policy = (
        EVIDENCE_ACCESS_POLICY_AUDIT_ONLY
        if request_manifest is not None
        else None
    )
    public_request_fields = (
        _public_request_evidence_fields(request_manifest)
        if request_manifest is not None
        else {}
    )
    reported_capture_mode = (
        request_manifest.protection_mode
        if request_manifest is not None
        else mode
    )

    return EvidenceCaptureResult(
        request_capture_status=request_status,
        request_evidence_ref=public_request_fields.get("request_evidence_ref"),
        request_raw_sha256=public_request_fields.get("request_raw_sha256"),
        request_canonical_sha256=public_request_fields.get("request_canonical_sha256"),
        evidence_capture_mode=reported_capture_mode,
        evidence_retention_class=retention_class,
        evidence_access_policy=evidence_access_policy,
        request_evidence_manifest=request_manifest.to_dict() if request_manifest else None,
        upstream_evidence_manifest=upstream_manifest.to_dict() if upstream_manifest else None,
        governed_evidence_manifest=governed_manifest.to_dict() if governed_manifest else None,
        evidence_manifests=manifests,
        strip_original_response=strip_original,
        strip_governed_response=strip_governed,
    )


def attach_evidence_fields_to_audit_entry(
    entry: dict[str, Any],
    decision: GovernanceDecision,
    *,
    pre_llm: bool,
    vault: EvidenceVault | LocalSealedEvidenceVault | None,
    config: EvidenceCaptureConfig,
    request_prompt: str | None = None,
) -> EvidenceCaptureResult:
    """Capture evidence and merge audit-plane fields into a ledger row."""
    result = capture_governance_evidence(
        decision,
        pre_llm=pre_llm,
        vault=vault,
        config=config,
        request_prompt=request_prompt,
    )

    entry["request_capture_status"] = result.request_capture_status
    entry["request_evidence_ref"] = result.request_evidence_ref
    entry["request_raw_sha256"] = result.request_raw_sha256
    entry["request_canonical_sha256"] = result.request_canonical_sha256
    entry["evidence_capture_mode"] = result.evidence_capture_mode
    entry["evidence_retention_class"] = result.evidence_retention_class
    if result.evidence_access_policy:
        entry["evidence_access_policy"] = result.evidence_access_policy

    decision.evidence_receipt_snapshot = build_evidence_receipt_snapshot(result)

    if result.request_evidence_manifest is not None:
        entry["request_evidence_manifest"] = result.request_evidence_manifest
    if result.upstream_evidence_manifest is not None:
        entry["upstream_evidence_manifest"] = result.upstream_evidence_manifest
    if result.governed_evidence_manifest is not None:
        entry["governed_evidence_manifest"] = result.governed_evidence_manifest
    if result.evidence_manifests:
        entry["evidence_manifests"] = result.evidence_manifests

    if decision.admissibility_basis is not None:
        entry["admissibility_basis"] = decision.admissibility_basis
    if decision.pass_reason_code is not None:
        entry["pass_reason_code"] = decision.pass_reason_code

    if result.strip_original_response:
        entry.pop("original_response", None)
    if result.strip_governed_response:
        entry.pop("governed_response", None)
        entry.pop("final_response", None)

    return result


def build_evidence_vault_for_bridge(
    audit_path: str | None,
    *,
    capture_mode: str = CaptureMode.SEALED.value,
    encryption_key: str | bytes | None = None,
) -> LocalSealedEvidenceVault | None:
    if audit_path is None:
        return None
    mode = (
        capture_mode
        if isinstance(capture_mode, CaptureMode)
        else CaptureMode(str(capture_mode))
    )
    key = resolve_evidence_encryption_key(encryption_key)
    if mode == CaptureMode.SEALED and not key:
        mode = CaptureMode.HASH_ONLY
    return create_local_evidence_vault(
        audit_path,
        capture_mode=mode,
        encryption_key=encryption_key,
    )
