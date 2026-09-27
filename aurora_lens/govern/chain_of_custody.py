"""Evidentiary chain-of-custody snapshot for audit rows (court-defensible packaging).

Builds a deterministic ``chain_of_custody`` object attached to forensic envelopes and
flat JSONL / AFL ledger payloads. This is layered on top of the tamper-evident hash
chain; it does not replace it.

Sources (no invented values):
  - Application version: :data:`aurora_lens.__version__`
  - Git: subprocess in ``AURORA_GIT_REPO_ROOT`` (or cwd), or ``AURORA_GIT_COMMIT`` /
    ``AURORA_GIT_DIRTY`` overrides
  - Policy / config fingerprint: canonical JSON hash of the caller-supplied governance dict
  - Runtime: ``socket.gethostname()``, ``AURORA_SERVICE_INSTANCE_ID`` (optional),
    upstream **model_id** / **provider** from :data:`aurora_lens.context.chain_of_custody_runtime_provenance_var`
    when set by the proxy for the active chat request (both non-empty), else
    ``AURORA_MODEL_ID`` / ``AURORA_PROVIDER`` (optional; absence degrades evidence status)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Any

from aurora_lens.context import chain_of_custody_runtime_provenance_var

CHAIN_OF_CUSTODY_VERSION = "1.0"


def get_application_version() -> str:
    """Package version without importing ``aurora_lens`` package root (avoids import cycles)."""
    try:
        from importlib.metadata import version

        return version("aurora-lens")
    except Exception:
        pass
    init_py = Path(__file__).resolve().parents[1] / "__init__.py"
    try:
        text = init_py.read_text(encoding="utf-8")
    except OSError:
        return "0.0.0"
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    return m.group(1) if m else "0.0.0"

# Git: optional overrides (CI / packaged deployments without .git)
ENV_GIT_COMMIT = "AURORA_GIT_COMMIT"
ENV_RAILWAY_GIT_COMMIT = "RAILWAY_GIT_COMMIT_SHA"
ENV_GIT_DIRTY = "AURORA_GIT_DIRTY"  # "true"/"false"/"1"/"0"
ENV_GIT_REPO_ROOT = "AURORA_GIT_REPO_ROOT"

# Runtime LLM attribution (set by operator / proxy)
ENV_MODEL_ID = "AURORA_MODEL_ID"
ENV_PROVIDER = "AURORA_PROVIDER"
ENV_INSTANCE_ID = "AURORA_SERVICE_INSTANCE_ID"


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def governance_config_fingerprint(governance_config: dict[str, Any]) -> str:
    """Stable SHA-256 over canonical JSON of governance-relevant config."""
    body = _canonical_json(governance_config)
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def _resolve_git(repo_root: str | None) -> tuple[str | None, bool | None, str]:
    """Return (commit_40_hex, is_clean, commit_source).

    commit_source is one of: ``env``, ``railway_env``, ``git``, ``unavailable``.
    ``is_clean`` is True when the working tree is clean, False when dirty, None unknown.
    """
    env_c = os.environ.get(ENV_GIT_COMMIT, "").strip()
    if env_c:
        dirty_raw = os.environ.get(ENV_GIT_DIRTY, "").strip().lower()
        if dirty_raw in ("1", "true", "yes", "dirty"):
            is_clean = False
        elif dirty_raw in ("0", "false", "no", "clean"):
            is_clean = True
        else:
            is_clean = None
        commit = env_c[:40] if len(env_c) >= 40 else env_c
        return commit if len(commit) >= 7 else env_c, is_clean, "env"

    railway_c = os.environ.get(ENV_RAILWAY_GIT_COMMIT, "").strip()
    if railway_c:
        # Railway commit SHAs are immutable build metadata; cleanliness is not meaningful.
        commit = railway_c[:40] if len(railway_c) >= 40 else railway_c
        if len(commit) >= 7:
            return commit, None, "railway_env"
        return railway_c, None, "railway_env"

    root = repo_root or os.getcwd()
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if r.returncode != 0 or not r.stdout:
            return None, None, "unavailable"
        commit = r.stdout.strip()
        if len(commit) < 7:
            return None, None, "unavailable"
        r2 = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if r2.returncode != 0:
            return commit, None, "git"
        clean = len(r2.stdout.strip()) == 0
        return commit, clean, "git"
    except (OSError, subprocess.TimeoutExpired):
        return None, None, "unavailable"


def _discover_repo_root() -> str | None:
    """Best-effort repository root for git metadata resolution.

    Resolution order:
      1. ``AURORA_GIT_REPO_ROOT`` when set
      2. Current working directory (if it is inside a git worktree)
      3. Parent walk from this module file to locate ``.git``
    """
    env_root = os.environ.get(ENV_GIT_REPO_ROOT, "").strip()
    if env_root:
        return env_root

    def _has_git_marker(path: str) -> bool:
        try:
            marker = Path(path) / ".git"
            return marker.exists()
        except OSError:
            return False

    cwd = os.getcwd()
    if _has_git_marker(cwd):
        return cwd

    here = Path(__file__).resolve()
    for parent in here.parents:
        if _has_git_marker(str(parent)):
            return str(parent)
    return None


def get_code_revision_snapshot() -> dict[str, Any]:
    """Public runtime identity snapshot for operator surfaces (e.g., /health)."""
    repo_root = _discover_repo_root()
    commit, clean, source = _resolve_git(repo_root)
    return {
        "application_version": get_application_version(),
        "git_commit": commit or "unknown",
        "git_commit_source": source,
        "git_working_tree_clean": clean,
    }


def build_chain_of_custody_bundle(
    *,
    policy_version: str | None,
    policy_source: str | None,
    governance_config: dict[str, Any],
    application_version: str,
) -> dict[str, Any]:
    """Assemble chain-of-custody block for one audit row.

    Sets ``evidence_audit_status`` to ``degraded`` when required runtime model/provider
    strings are absent (so missing attribution never masquerades as complete).
    """
    degradation: list[str] = []

    fp = governance_config_fingerprint(governance_config)

    if not (policy_version and str(policy_version).strip()):
        degradation.append("policy_version_missing")

    repo_root = _discover_repo_root()
    commit, clean, git_src = _resolve_git(repo_root)
    if not commit:
        # Never emit ``null`` for code revision provenance fields.
        # "unknown" is explicit and keeps audit consumers schema-stable.
        commit = "unknown"
        degradation.append("git_commit_unavailable")

    hostname = socket.gethostname()
    instance_id = os.environ.get(ENV_INSTANCE_ID, "").strip() or hostname

    prov_ctx = chain_of_custody_runtime_provenance_var.get()
    used_request_runtime = False
    if isinstance(prov_ctx, tuple) and len(prov_ctx) == 2:
        p_req, m_req = str(prov_ctx[0] or "").strip(), str(prov_ctx[1] or "").strip()
        if p_req and m_req:
            provider, model_id = p_req, m_req
            used_request_runtime = True
    if not used_request_runtime:
        model_id = os.environ.get(ENV_MODEL_ID, "").strip() or None
        provider = os.environ.get(ENV_PROVIDER, "").strip() or None
        if model_id is None or provider is None:
            degradation.append("runtime_model_provenance_incomplete")

    evidence_status = "degraded" if degradation else "complete"

    bundle: dict[str, Any] = {
        "chain_of_custody_version": CHAIN_OF_CUSTODY_VERSION,
        "evidence_audit_status": evidence_status,
        "evidence_degradation_reasons": degradation,
        "code": {
            "application_version": application_version,
            "git_commit": commit,
            "git_working_tree_clean": clean,
            "git_commit_source": git_src,
        },
        "policy": {
            "policy_version": policy_version,
            "policy_source": policy_source,
            "governance_config_fingerprint": fp,
        },
        "runtime": {
            "hostname": hostname,
            "service_instance_id": instance_id,
            "model_id": model_id,
            "provider": provider,
        },
        # Embedded for offline fingerprint replay (same bytes as hashed).
        "_governance_config_snapshot": governance_config,
    }
    return bundle


def apply_ruleset_provenance_fields(
    row: dict[str, Any],
    *,
    chain_of_custody: dict[str, Any] | None,
    policy_profile: str | None,
    outcome: str | None,
    governor_policy_id: str | None = None,
) -> None:
    """Populate stable policy provenance fields on a governance audit row.

    Fields written:
      - ``policy_version``
      - ``ruleset_hash`` (governance config fingerprint)
      - ``governor_policy_id`` (explicit id when provided, otherwise stable fallback)
    """
    policy_block = (
        chain_of_custody.get("policy")
        if isinstance(chain_of_custody, dict)
        else None
    )
    policy_version = (
        policy_block.get("policy_version")
        if isinstance(policy_block, dict)
        else None
    )
    ruleset_hash = (
        policy_block.get("governance_config_fingerprint")
        if isinstance(policy_block, dict)
        else None
    )

    row["policy_version"] = (
        str(policy_version).strip()
        if policy_version is not None and str(policy_version).strip()
        else "unknown"
    )
    row["ruleset_hash"] = (
        str(ruleset_hash).strip()
        if ruleset_hash is not None and str(ruleset_hash).strip()
        else "unknown"
    )

    gid = str(governor_policy_id).strip() if governor_policy_id is not None else ""
    if not gid:
        profile = (
            str(policy_profile).strip()
            if policy_profile is not None and str(policy_profile).strip()
            else "unknown"
        )
        out = (
            str(outcome).strip()
            if outcome is not None and str(outcome).strip()
            else "UNKNOWN"
        )
        gid = f"{profile}:{out}"
    row["governor_policy_id"] = gid


def verify_chain_of_custody_entry(
    entry: dict[str, Any],
    *,
    require_complete_evidence: bool = False,
) -> tuple[bool, str | None]:
    """Verify presence and shape of ``chain_of_custody`` on one audit row.

    Recomputes ``governance_config_fingerprint`` when possible (must match).

    Returns (ok, reason).
    """
    coc = entry.get("chain_of_custody")
    if not isinstance(coc, dict):
        return False, "missing_chain_of_custody"
    if coc.get("chain_of_custody_version") != CHAIN_OF_CUSTODY_VERSION:
        return False, "chain_of_custody_version_mismatch"

    code = coc.get("code")
    pol = coc.get("policy")
    rt = coc.get("runtime")
    if not isinstance(code, dict) or not isinstance(pol, dict) or not isinstance(rt, dict):
        return False, "malformed_chain_of_custody_sections"

    for k in ("application_version", "git_commit_source"):
        if k not in code:
            return False, f"code_missing_{k}"
    for k in ("governance_config_fingerprint",):
        if k not in pol:
            return False, f"policy_missing_{k}"
    fp = pol.get("governance_config_fingerprint")
    if not isinstance(fp, str) or not fp.startswith("sha256:") or len(fp) != 71:
        return False, "invalid_fingerprint_format"

    for k in ("hostname", "service_instance_id"):
        if k not in rt or not rt.get(k):
            return False, f"runtime_missing_{k}"

    status = coc.get("evidence_audit_status")
    if status not in ("complete", "degraded"):
        return False, "invalid_evidence_audit_status"
    if require_complete_evidence and status != "complete":
        return False, "evidence_degraded"

    reasons = coc.get("evidence_degradation_reasons")
    if not isinstance(reasons, list):
        return False, "invalid_degradation_reasons"
    if status == "complete" and reasons:
        return False, "complete_with_degradation_reasons"
    if status == "degraded" and not reasons:
        return False, "degraded_without_reasons"

    # Fingerprint replay: snapshot must be present and match policy.governance_config_fingerprint.
    gcfg = coc.get("_governance_config_snapshot")
    if not isinstance(gcfg, dict):
        return False, "missing_governance_config_snapshot"
    expected = governance_config_fingerprint(gcfg)
    if expected != fp:
        return False, "governance_config_fingerprint_mismatch"

    return True, None


def explain_chain_of_custody_failure(reason: str | None) -> str:
    """Plain-language explanation for operators and third-party reviewers (no jargon-only codes)."""
    if not reason:
        return (
            "Chain-of-custody verification failed for an unspecified reason. "
            "See docs/evidence-model.md (evidentiary packaging / chain_of_custody)."
        )
    guides: dict[str, str] = {
        "missing_chain_of_custody": (
            "This audit row has no `chain_of_custody` object. "
            "Current releases attach that block to every new row so reviewers can see "
            "code build, policy fingerprint, and runtime attribution."
        ),
        "chain_of_custody_version_mismatch": (
            "The `chain_of_custody.chain_of_custody_version` field does not match the verifier. "
            "The log may be from a different aurora-lens release than the tool you are running."
        ),
        "malformed_chain_of_custody_sections": (
            "The `chain_of_custody` object is missing required sections (`code`, `policy`, or `runtime`). "
            "The row may be corrupted or partially written."
        ),
        "invalid_fingerprint_format": (
            "The policy fingerprint is not a valid `sha256:` + 64 hex digest. "
            "The row may have been edited by hand or truncated."
        ),
        "invalid_evidence_audit_status": (
            "`evidence_audit_status` must be exactly `complete` or `degraded`. "
            "Any other value is invalid."
        ),
        "invalid_degradation_reasons": (
            "`evidence_degradation_reasons` must be a JSON list. "
            "The row structure does not match the expected schema."
        ),
        "complete_with_degradation_reasons": (
            "The row claims `evidence_audit_status: complete` but also lists degradation reasons. "
            "That combination is inconsistent and is rejected."
        ),
        "degraded_without_reasons": (
            "The row claims `evidence_audit_status: degraded` but lists no reasons. "
            "Degraded rows must say why (for example missing model/provider attribution)."
        ),
        "missing_governance_config_snapshot": (
            "The row is missing `_governance_config_snapshot`, which is required to replay-check "
            "the policy fingerprint. Without it, an outsider cannot verify what config was hashed."
        ),
        "governance_config_fingerprint_mismatch": (
            "The stored fingerprint does not match the embedded governance snapshot. "
            "Either the row was tampered with after writing, or the snapshot was altered."
        ),
        "evidence_degraded": (
            "This row records `evidence_audit_status: degraded` (not full evidentiary packaging). "
            "You passed `--require-complete-evidence`, which requires every checked row to be `complete`. "
            "Typical cause when the row was emitted: neither per-request proxy provenance "
            "(`chain_of_custody_runtime_provenance_var`) nor `AURORA_MODEL_ID` / `AURORA_PROVIDER` "
            "in the process environment supplied both model and provider, so runtime attribution "
            "is incomplete. Use the proxy chat path (which sets request-scoped provenance) or set "
            "both environment variables for deployments that must produce complete evidence rows."
        ),
        "json_error": "A line in the audit file is not valid JSON.",
        "read_error": "The audit file could not be read (permissions or disk error).",
    }
    if reason in guides:
        return guides[reason]
    if reason.startswith("code_missing_"):
        return (
            f"Required field under `chain_of_custody.code` is missing ({reason}). "
            "The provenance block is incomplete."
        )
    if reason.startswith("policy_missing_"):
        return (
            f"Required field under `chain_of_custody.policy` is missing ({reason}). "
            "The provenance block is incomplete."
        )
    if reason.startswith("runtime_missing_"):
        return (
            f"Required field under `chain_of_custody.runtime` is missing or empty ({reason}). "
            "Hostname and service instance id must be recorded."
        )
    return (
        f"Verification failed (internal code: {reason}). "
        "Compare this row to docs/evidence-model.md under `chain_of_custody`."
    )
