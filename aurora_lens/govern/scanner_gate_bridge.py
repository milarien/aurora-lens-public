"""Governor bridge implementation (`AuroraScannerGateBridge`) — AFL-JSONL-1 forensic ledger, CID, and HMAC attestation.

All infrastructure is local — no external dependencies required.

- AFL-JSONL-1 hash-chained forensic ledger (ForensicLedger) — backend="ledger"
- Flat JSONL audit via audit_io (BuiltinBridge-compatible format) — backend="jsonl"
- Content-addressed identifiers via FallbackCIDProvider (FNV-1a 64-bit)
- HMAC-signed AttestedOutput for compliance/audit
- PEF continuity chain across all ledger entries

kernel_step() was removed as a co-author of policy decisions when
CanonicalScannerGateBridge was introduced. No external dependencies are
required for any governance decision or audit operation.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from aurora_lens.context import (
    get_request_metadata,
    proxy_run_id_var,
    request_hash_var,
    run_id_var,
)
from aurora_lens.govern.bridge import (
    GovernanceBridge,
    _build_governed_request_metadata,
    apply_domain_reclassification_fields,
    apply_intervention_policy_pathway_fields,
    attach_delivery_and_epistemic_audit_fields,
    build_forensic_event,
    enforce,
    ESCALATION_ROUTES,
    _ACTION_TO_PATHWAY,
    refresh_forensic_event_hash,
)
from aurora_lens.govern.evidence_capture import (
    EvidenceCaptureConfig,
    attach_evidence_fields_to_audit_entry,
    build_evidence_vault_for_bridge,
    finalize_audit_receipt_snapshot,
)
from aurora_lens.govern.audit_io import CHAIN_GENESIS, append_audit_entry
from aurora_lens.govern.clarification_audit import (
    CLARIFICATION_RESOLUTION_OUTCOME,
    build_clarification_resolution_audit_entry,
)
from aurora_lens.govern.forensic_append_guard import enforce_forensic_event_for_append
from aurora_lens.govern.chain_of_custody import (
    apply_ruleset_provenance_fields,
    build_chain_of_custody_bundle as _build_coc_bundle,
    get_application_version,
)
from aurora_lens.govern.instrument_provenance import (
    apply_attestation_fields_to_row,
    apply_instrument_provenance_to_row,
    finalize_decision_record_hash,
    sync_instrument_provenance_to_decision,
)
from aurora_lens.log_slice import consume_log_slice
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction, apply_epistemic_state_from_flags
from aurora_lens.govern.policy import InterventionPolicy, DEFAULT_STRICT
from aurora_lens.govern.cid import PEFFingerprint, FallbackCIDProvider
from aurora_lens.govern.forensic_ledger import ForensicLedger, LedgerVerifyDetail
from aurora_lens.govern.attestation import AttestedOutput, sign_attested_output, verify_governed_output
from aurora_lens.request_metadata import request_metadata_snapshot
from aurora_lens.verify.flags import Flag, FlagType

if TYPE_CHECKING:
    from aurora_lens.adapters.base import LLMAdapter
    from aurora_lens.pef.state import PEFState


# Retained for backward compatibility — always True now that all infrastructure is local.
_SCANNER_GATE_AVAILABLE = True


def _resolve_audit_linkage_session_id(
    ctx_sid: str | None,
    pef_snapshot: dict[str, Any] | None,
) -> str | None:
    """Chat id for AFL ``payload.data.session_id`` (ContextVar wins, then ``pef_snapshot``)."""
    if isinstance(ctx_sid, str) and ctx_sid.strip():
        return ctx_sid.strip()
    if isinstance(pef_snapshot, dict):
        snap_sid = pef_snapshot.get("session_id")
        if isinstance(snap_sid, str) and snap_sid.strip():
            return snap_sid.strip()
    return None


# ── Bridge ───────────────────────────────────────────────────────────

class AuroraScannerGateBridge(GovernanceBridge):
    """Governor bridge with AFL-JSONL-1 forensic ledger and HMAC attestation.

    Provides:
    - ForensicLedger for AFL-JSONL-1 hash-chained audit (tamper-evident)
    - AttestedOutput for HMAC-signed governance decisions
    - FallbackCIDProvider for content-addressed identifiers (FNV-1a 64-bit)

    Uses InterventionPolicy for flag→action mapping.
    See ``CanonicalScannerGateBridge`` for the canonical Governor pipeline (PolicyResolver
    as sole decision authority).
    """

    def __init__(
        self,
        policy: InterventionPolicy | None = None,
        audit_path: str | Path | None = None,
        secret_key: bytes | None = None,
        constraints: dict | None = None,
        max_revision_attempts: int = 1,
        backend: str = "ledger",
        default_policy: str = "strict",
        audit_signing_keys: tuple[bytes, ...] = (),
        proxy_run_id: str | None = None,
        evidence_capture_mode: str = "plaintext_dev",
        evidence_encryption_key: str | bytes | None = None,
    ):
        """
        Args:
            policy: Intervention policy (defaults to DEFAULT_STRICT).
            audit_path: Path for audit log.
            secret_key: Primary HMAC key for AttestedOutput and AFL ledger line ``sig``.
                        If None, attestation and per-line ledger sig are skipped.
            audit_signing_keys: Additional historical keys (bytes) for verifying logs
                        after key rotation. Only used for ``verify_ledger``; new lines
                        are signed with ``secret_key`` only.
            proxy_run_id: When set (e.g. proxy process id from ``create_app``), stored on
                        each audit row as ``proxy_run_id`` so ``verify_pef_linkage`` can skip
                        pairs across deploy/restart boundaries when logs are concatenated.
            constraints: Retained for interface compatibility.
            max_revision_attempts: Hard limit on revision loop (1–10).
            backend: Audit format — "ledger" (AFL-JSONL-1) or "jsonl" (flat JSONL).
            default_policy: Deployment default policy name ("strict" | "moderate").
                            Used as policy_profile fallback in flat JSONL entries.
        """
        self._policy = policy or DEFAULT_STRICT
        self._secret_key = secret_key
        self._audit_signing_keys = audit_signing_keys
        self._max_revision_attempts = max(1, min(10, max_revision_attempts))
        self._last_attestation: AttestedOutput | None = None
        self._constraints = constraints or {}
        self._backend = backend if backend in ("ledger", "jsonl") else "ledger"
        self._default_policy = default_policy

        # Store audit path string for JSONL writing
        self._audit_path_str: str | None = str(audit_path) if audit_path else None

        self._evidence_config = EvidenceCaptureConfig(
            capture_mode=evidence_capture_mode,
            encryption_key=evidence_encryption_key,
        )
        self._evidence_vault = build_evidence_vault_for_bridge(
            self._audit_path_str,
            capture_mode=evidence_capture_mode,
            encryption_key=evidence_encryption_key,
        )

        # CID provider (FNV-1a 64-bit)
        self._cid_provider = FallbackCIDProvider()

        # Forensic ledger (hash-chained, tamper-evident) — only for "ledger" backend
        if audit_path and self._backend == "ledger":
            self._trace_id = (
                f"aurora-lens:"
                f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')}"
            )
            self._ledger: ForensicLedger | None = ForensicLedger(
                log_path=str(audit_path),
                trace_id=self._trace_id,
                secret_key=self._secret_key,
            )
        else:
            self._trace_id = None
            self._ledger = None

        # D2 hash chain for flat JSONL (Phase / Stage A): anchor first row at genesis;
        # resume from the prior tip when reopening an existing file (parity with BuiltinBridge).
        self._prev_cid: str = (
            self._read_last_jsonl_chain_tip()
            if self._audit_path_str and self._backend == "jsonl"
            else CHAIN_GENESIS
        )

        # Process-startup identifier — stable for the lifetime of this bridge instance.
        # Used only as a fallback when no per-request run_id is available (e.g. direct
        # bridge use outside the proxy). Proxy-driven requests get a fresh id per call
        # via ``run_id_var`` — see ``_current_run_id``.
        self._run_id: str = str(uuid.uuid4())
        self._proxy_run_id: str | None = (
            str(proxy_run_id).strip()
            if proxy_run_id is not None and str(proxy_run_id).strip()
            else None
        )

    def _current_run_id(self) -> str:
        """Per-request run_id when the proxy set one this request; else the bridge's own id.

        Every HTTP request must get a fresh run_id — a shared value across every
        request in a process makes audit rows from different callers/turns
        indistinguishable. The instance fallback exists for direct (non-proxy)
        bridge use, e.g. tests and scripts calling ``log_decision`` without a
        request context.
        """
        v = run_id_var.get(None)
        return v if v is not None and str(v).strip() else self._run_id

    def _current_proxy_run_id(self) -> str | None:
        """Per-request proxy_run_id when set; else the constructor-supplied process id."""
        v = proxy_run_id_var.get(None)
        if v is not None and str(v).strip():
            return str(v).strip()
        return self._proxy_run_id

    def _read_last_jsonl_chain_tip(self) -> str:
        """Last ``cid`` in the flat audit JSONL file, or genesis when absent or unreadable."""
        if not self._audit_path_str:
            return CHAIN_GENESIS
        p = Path(self._audit_path_str)
        if not p.exists():
            return CHAIN_GENESIS
        try:
            lines = [ln for ln in p.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
            if not lines:
                return CHAIN_GENESIS
            last = json.loads(lines[-1])
            return last.get("cid") or CHAIN_GENESIS
        except (OSError, json.JSONDecodeError, IndexError):
            return CHAIN_GENESIS

    def _governance_config_snapshot(self) -> dict[str, Any]:
        """Stable dict for governance_config_fingerprint (deployment + policy surface)."""
        cfg: dict[str, Any] = {
            "bridge": "aurora_scanner_gate",
            "audit_backend": self._backend,
            "default_policy": self._default_policy,
        }
        pr = getattr(self, "_policy_resolver", None)
        if pr is not None:
            cfg["policy_matrix_version"] = pr.active_version
            cfg["policy_source"] = pr.policy_source
        return cfg

    def _policy_provenance(self) -> tuple[str | None, str | None]:
        """(policy_version, policy_source) for chain-of-custody."""
        pr = getattr(self, "_policy_resolver", None)
        if pr is not None:
            return pr.active_version, pr.policy_source
        pv = getattr(self, "_policy_version", None)
        if pv is not None:
            return str(pv), "intervention_policy"
        # Legacy InterventionPolicy bridge: stable label (matrix version lives in Canonical resolver).
        return "intervention-policy", "intervention_policy"

    def _build_chain_of_custody_bundle(self) -> dict[str, Any]:
        pv, psrc = self._policy_provenance()
        return _build_coc_bundle(
            policy_version=pv,
            policy_source=psrc,
            governance_config=self._governance_config_snapshot(),
            application_version=get_application_version(),
        )

    def _attach_chain_of_custody(self, decision: GovernanceDecision, payload: dict[str, Any]) -> None:
        """Merge chain-of-custody into payload and forensic_event; refresh event_hash."""
        coc = self._build_chain_of_custody_bundle()
        payload["chain_of_custody"] = coc
        if decision.forensic_event is not None:
            decision.forensic_event["chain_of_custody"] = coc
            refresh_forensic_event_hash(decision.forensic_event)

    def _ledger_verify_keys(self) -> list[bytes]:
        """Primary signing key first, then historical keys (deduplicated)."""
        out: list[bytes] = []
        seen: set[bytes] = set()
        if self._secret_key:
            out.append(self._secret_key)
            seen.add(self._secret_key)
        for k in self._audit_signing_keys:
            if k and k not in seen:
                out.append(k)
                seen.add(k)
        return out

    # ── GovernanceBridge interface ────────────────────────────────

    async def decide(
        self,
        flags: list[Flag],
        response_text: str,
        pef: PEFState,
    ) -> GovernanceDecision:
        """Evaluate flags via policy + kernel validation."""
        # InterventionPolicy maps flags → proposed action.
        # GovernanceKernel is attestation/ledger substrate only — it does not
        # co-author the policy decision. See CanonicalScannerGateBridge for the
        # canonical Governor pipeline (PolicyResolver replaces InterventionPolicy).
        proposed_action, rule = self._policy.evaluate_with_rule(flags)
        rule_id = rule.rule_id if rule else None
        action = proposed_action

        rationale = self._build_rationale(flags, action)

        # Generate CID directly from cid_provider (no kernel_step() needed).
        from aurora_lens.context import trace_id_var as _trace_id_var
        _trace_id = _trace_id_var.get(None) or (self._trace_id or "")
        fp = PEFFingerprint(
            kind=action.name,
            span=_trace_id,
            head=rationale[:32],
            rationale=f"flags:{len(flags)}",
        )
        cid = self._cid_provider.cid_for_pef(fp)
        note = (
            self._build_governance_note(flags)
            if action == InterventionAction.SOFT_CORRECT
            else None
        )

        safe_alt = rule.safe_alt if rule else None
        resource = None
        if rule and action == InterventionAction.HARD_STOP:
            route = ESCALATION_ROUTES.get(rule.flag_type.name, (None, None, None, None))
            resource = route[3] if len(route) > 3 else None

        decision = GovernanceDecision(
            action=action,
            flags=flags,
            rationale=rationale,
            policy=self._policy.name,
            attempt=0,
            governance_note=note,
            cid=cid,
            rule_id=rule_id,
            safe_alt=safe_alt,
            resource=resource,
            pathway_id=_ACTION_TO_PATHWAY.get(action),
        )
        apply_intervention_policy_pathway_fields(
            decision, flags=flags, action=action, rule=rule,
        )
        apply_epistemic_state_from_flags(decision)
        return decision

    async def intervene(
        self,
        decision: GovernanceDecision,
        adapter: LLMAdapter,
        user_input: str,
        pef_context: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Execute the intervention. Uses enforce() for deterministic output."""
        model_output = decision.original_response or ""
        msg = enforce(decision, model_output)
        decision.governed_response = msg
        return msg

    def _postprocess_forensic_event(self, decision: GovernanceDecision) -> None:
        """Hook: mutate ``decision.forensic_event`` after :func:`build_forensic_event`, before ledger append.

        :class:`CanonicalScannerGateBridge` overrides to apply Governor enrichment and refresh
        ``event_hash``. Non-canonical scanner bridge leaves the base forensic event unchanged.
        """
        return

    def _prepare_governance_forensic_event(
        self,
        decision: GovernanceDecision,
        *,
        trace_id: str,
        timestamp: str,
        turn: int,
        pre_llm: bool,
        pef_snapshot: dict | None,
    ) -> None:
        """Single forensic pipeline for both AFL and flat JSONL: CID, ``build_forensic_event``, hook.

        Always assigns ``decision.cid`` (FNV content id). For non-PASS / non-SOFT_CORRECT,
        builds ``decision.forensic_event`` and runs :meth:`_postprocess_forensic_event`
        (Governor enrichment on :class:`CanonicalScannerGateBridge`).
        """
        fp = PEFFingerprint(
            kind=decision.action.name,
            span=trace_id,
            head=(decision.rationale or "")[:32],
            rationale=f"turn:{turn}:flags:{len(decision.flags)}",
        )
        cid = self._cid_provider.cid_for_pef(fp)
        decision.cid = cid
        if decision.action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT):
            return
        decision.forensic_event = build_forensic_event(
            decision,
            pre_llm=pre_llm,
            pef_snapshot=pef_snapshot,
            trace_id=trace_id,
            timestamp=timestamp,
            audit_id=cid,
        )
        self._postprocess_forensic_event(decision)

    def _apply_evidence_capture(
        self,
        entry: dict[str, Any],
        decision: GovernanceDecision,
        *,
        pre_llm: bool,
    ) -> None:
        if self._audit_path_str is None:
            return
        attach_evidence_fields_to_audit_entry(
            entry,
            decision,
            pre_llm=pre_llm,
            vault=self._evidence_vault,
            config=self._evidence_config,
        )

    def log_decision(
        self,
        decision: GovernanceDecision,
        turn: int = 0,
        *,
        stream: bool = False,
        stream_completed: bool = True,
        stream_abort_reason: str | None = None,
        stream_truncated: bool = False,
        stream_dropped_chars: int = 0,
        pef_context: str | None = None,
        pre_llm: bool = False,
        pef_snapshot: dict | None = None,
        pef_turn_classification: str | None = None,
        pef_hold_transition: str | None = None,
        at_verification_basis: dict | None = None,
        epistemic_normalisation_applied: bool | None = None,
        state_native_handled: bool | None = None,
    ) -> None:
        """Log to audit backend (flat JSONL or AFL-JSONL-1) + produce AttestedOutput."""
        # Consume log slice once — it is cleared after the first read.
        log_slice = consume_log_slice()

        if pef_snapshot is not None:
            # Detach from live Lens ``PEFState`` / shared dict trees so nested mutations
            # cannot alter a row's serialized ``pef_snapshot`` before the ledger line is
            # written (``verify_pef_linkage`` replays from the stored JSON).
            pef_snapshot = copy.deepcopy(pef_snapshot)

        if at_verification_basis is not None:
            at_verification_basis = copy.deepcopy(at_verification_basis)

        if self._backend == "jsonl":
            self._log_jsonl(
                decision,
                turn=turn,
                stream=stream,
                stream_completed=stream_completed,
                stream_abort_reason=stream_abort_reason,
                stream_truncated=stream_truncated,
                stream_dropped_chars=stream_dropped_chars,
                pef_context=pef_context,
                pre_llm=pre_llm,
                pef_snapshot=pef_snapshot,
                log_slice=log_slice,
                pef_turn_classification=pef_turn_classification,
                pef_hold_transition=pef_hold_transition,
                at_verification_basis=at_verification_basis,
                epistemic_normalisation_applied=epistemic_normalisation_applied,
                state_native_handled=state_native_handled,
            )
            return

        # ── AFL-JSONL-1 ledger path ────────────────────────────────────────────
        if self._ledger is None and self._secret_key is None:
            return

        # Build payload
        payload = {
            "event_type": "governance_decision",
            "run_id": self._current_run_id(),
            "turn": turn,
            "escalation_level": decision.escalation_level,
            "policy": decision.policy,
            "action": decision.action.name,
            "flags": [
                {
                    "type": f.flag_type.name,
                    "severity": f.severity,
                    "claim": f.claim,
                    **({"rule_id": f.rule_id} if f.rule_id else {}),
                }
                for f in decision.flags
            ],
            "rationale": decision.rationale,
            "attempt": decision.attempt,
            **({"original_response": decision.original_response} if decision.original_response else {}),
            **({"governed_response": decision.governed_response} if decision.governed_response else {}),
            **({"governance_note": decision.governance_note} if decision.governance_note else {}),
        }
        from aurora_lens.context import auth_label_var, domain_var
        auth_label = auth_label_var.get(None)
        payload["consumer_label"] = auth_label  # Always present; null when auth disabled
        if auth_label is not None:
            payload["auth_label"] = auth_label
        _current_proxy_run_id = self._current_proxy_run_id()
        if _current_proxy_run_id is not None:
            payload["proxy_run_id"] = _current_proxy_run_id
        _req_dom = domain_var.get(None)
        if _req_dom:
            payload["request_domain"] = _req_dom
        _request_metadata = request_metadata_snapshot(get_request_metadata())
        if _request_metadata is not None:
            payload["request_metadata"] = _request_metadata
        _request_hash = request_hash_var.get(None)
        if _request_hash:
            payload["request_hash"] = _request_hash

        attach_delivery_and_epistemic_audit_fields(
            payload,
            stream=stream,
            stream_completed=stream_completed,
            stream_abort_reason=stream_abort_reason,
            stream_truncated=stream_truncated,
            stream_dropped_chars=stream_dropped_chars,
            epistemic_normalisation_applied=epistemic_normalisation_applied,
        )
        if state_native_handled is not None:
            payload["state_native_handled"] = state_native_handled

        if log_slice is not None:
            payload["log_slice_present"] = True
            payload.update(log_slice)
        else:
            payload["log_slice_present"] = False

        if pef_turn_classification is not None:
            payload["pef_turn_classification"] = pef_turn_classification
        if pef_hold_transition is not None:
            payload["pef_hold_transition"] = pef_hold_transition

        if decision.epistemic_state is not None:
            payload["epistemic_state"] = decision.epistemic_state

        if at_verification_basis is not None:
            decision.at_verification_basis = at_verification_basis
            payload["at_verification_basis"] = at_verification_basis

        from aurora_lens.context import session_id_var as _ledger_session_var, trace_id_var
        # Chat session id on every governance payload (including PASS-only rows) so
        # ``verify_pef_linkage`` can skip interleaved sessions (``_audit_row_session_id_for_linkage``).
        _link_sid = _resolve_audit_linkage_session_id(_ledger_session_var.get(None), pef_snapshot)
        if _link_sid:
            payload["session_id"] = _link_sid
        trace_id = trace_id_var.get(None) or (self._trace_id or "")
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        payload["trace_id"] = trace_id
        payload["timestamp"] = ts
        payload["policy_profile"] = str(payload.get("policy") or "")
        payload["policy_version"] = str(payload.get("policy") or "")
        self._prepare_governance_forensic_event(
            decision,
            trace_id=trace_id,
            timestamp=ts,
            turn=turn,
            pre_llm=pre_llm,
            pef_snapshot=pef_snapshot,
        )
        self._attach_chain_of_custody(decision, payload)
        _event_gid = (
            decision.forensic_event.get("governor_policy_id")
            if isinstance(decision.forensic_event, dict)
            else None
        )
        _fe_domain = (
            decision.forensic_event.get("domain")
            if isinstance(decision.forensic_event, dict)
            else None
        )
        _fe_ctx = (
            decision.forensic_event.get("context_provenance")
            if isinstance(decision.forensic_event, dict)
            else None
        )
        _fe_domain_source = (
            _fe_ctx.get("domain_source")
            if isinstance(_fe_ctx, dict)
            else None
        )
        apply_ruleset_provenance_fields(
            payload,
            chain_of_custody=payload.get("chain_of_custody"),
            policy_profile=str(payload.get("policy")),
            outcome=decision.action.name,
            governor_policy_id=_event_gid,
        )
        apply_domain_reclassification_fields(
            payload,
            request_domain=payload.get("request_domain"),
            effective_domain=payload.get("domain") or _fe_domain,
            domain_source=_fe_domain_source,
        )
        provenance_fields = apply_instrument_provenance_to_row(
            payload,
            chain_of_custody=payload.get("chain_of_custody"),
            policy_profile=str(payload.get("policy")),
            outcome=decision.action.name,
            governor_policy_id=_event_gid,
            policy_version=str(payload.get("policy_version") or ""),
            signing_key_configured=self._secret_key is not None,
            attestation_signed=False,
        )
        cid = decision.cid

        if decision.forensic_event is not None:
            apply_ruleset_provenance_fields(
                decision.forensic_event,
                chain_of_custody=payload.get("chain_of_custody"),
                policy_profile=str(payload.get("policy")),
                outcome=decision.action.name,
                governor_policy_id=_event_gid,
            )
            apply_domain_reclassification_fields(
                decision.forensic_event,
                request_domain=payload.get("request_domain"),
                effective_domain=decision.forensic_event.get("domain"),
                domain_source=_fe_domain_source,
            )
            refresh_forensic_event_hash(decision.forensic_event)
            enforce_forensic_event_for_append(decision.forensic_event)
            payload["forensic_event"] = decision.forensic_event
            if pef_snapshot is not None:
                payload["pef_snapshot"] = pef_snapshot

        from aurora_lens.sovereign.audit_envelope import apply_provider_route_to_audit_entry

        apply_provider_route_to_audit_entry(payload, decision)
        if decision.action != InterventionAction.PASS:
            payload["governed_request_metadata"] = _build_governed_request_metadata(
                entry=payload,
                decision=decision,
            )
        self._apply_evidence_capture(payload, decision, pre_llm=pre_llm)
        attestation_fields = apply_attestation_fields_to_row(
            payload,
            chain_of_custody=payload.get("chain_of_custody"),
            signing_key=self._secret_key,
        )
        sync_instrument_provenance_to_decision(
            decision,
            {**provenance_fields, **attestation_fields},
            signature_status=payload.get("signature_status"),
        )

        # Append to forensic ledger (hash-chained, tamper-evident)
        # ext.pef: SHA-256 of payload for PEF continuity chain.
        if self._ledger is not None:
            # Re-apply after forensic / snapshot attach so linkage id cannot be dropped
            # by any intervening payload mutation (defensive; primary fix is Lens PEF id).
            _link_sid_final = _resolve_audit_linkage_session_id(
                _ledger_session_var.get(None),
                pef_snapshot,
            )
            if _link_sid_final:
                payload["session_id"] = _link_sid_final
            pef_hash = hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            entry_index = self._ledger.seq
            prev_link = self._ledger.last_hash
            row_hash = self._ledger.append(
                op=decision.action.name,
                payload_data=payload,
                cid=cid,
                scope={
                    "domain": "governance",
                    "subdomain": "aurora-lens",
                    "authority": "policy",
                },
                ext={"pef": pef_hash},
            )
            snap: dict[str, Any] = {
                "trace_id": trace_id,
                "entry_index": entry_index,
                "prev_hash": prev_link,
                "hash": row_hash,
            }
            fe_st = decision.forensic_event.get("state_hash") if decision.forensic_event else None
            if fe_st is not None:
                snap["state_hash"] = fe_st
            finalize_audit_receipt_snapshot(decision, chain_fields=snap)
            finalize_decision_record_hash(decision, payload, cid=None, ledger_hash=row_hash)

        # Produce AttestedOutput (HMAC-signed)
        if self._secret_key is not None:
            content = decision.corrected_response or decision.original_response or ""
            from aurora_lens.context import session_id_var as _sid2_var, trace_id_var as _tid2_var
            _ts_attest = datetime.datetime.now(datetime.timezone.utc).isoformat()
            attested = AttestedOutput(
                content=content,
                policy_cid=f"policy:{decision.policy}",
                rationale_code=decision.action.name,
                decision=decision.action.name,
                event_cid=cid,
                signature="",  # will be replaced by signing
                meta={
                    "turn": turn,
                    "flags": len(decision.flags),
                    "attempt": decision.attempt,
                    "timestamp": _ts_attest,
                    "trace_id": _tid2_var.get(None) or (self._trace_id or ""),
                    "session_id": _sid2_var.get(None),
                    "run_id": self._current_run_id(),
                },
            )
            self._last_attestation = sign_attested_output(attested, self._secret_key)
            decision.signature_status = str(payload.get("signature_status") or decision.signature_status or "")

    def _log_jsonl(
        self,
        decision: GovernanceDecision,
        turn: int = 0,
        *,
        stream: bool = False,
        stream_completed: bool = True,
        stream_abort_reason: str | None = None,
        stream_truncated: bool = False,
        stream_dropped_chars: int = 0,
        pef_context: str | None = None,
        pre_llm: bool = False,
        pef_snapshot: dict | None = None,
        log_slice: dict | None = None,
        pef_turn_classification: str | None = None,
        pef_hold_transition: str | None = None,
        at_verification_basis: dict | None = None,
        epistemic_normalisation_applied: bool | None = None,
        state_native_handled: bool | None = None,
    ) -> None:
        """Write flat JSONL audit entry (BuiltinBridge-compatible format)."""
        if not self._audit_path_str and self._secret_key is None:
            return

        from aurora_lens.context import (
            auth_label_var,
            auth_policy_var,
            domain_var,
            get_request_metadata,
            session_id_var,
            trace_id_var,
            request_hash_var,
        )
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        trace_id = trace_id_var.get(None) or ""
        self._prepare_governance_forensic_event(
            decision,
            trace_id=trace_id,
            timestamp=ts,
            turn=turn,
            pre_llm=pre_llm,
            pef_snapshot=pef_snapshot,
        )
        cid = decision.cid

        auth_label = auth_label_var.get(None)
        policy_profile = auth_policy_var.get(None) or self._default_policy
        failed_constraints = [f.flag_type.name for f in decision.flags] if decision.flags else []
        pef_ctx = pef_context or ""
        state_hash = hashlib.sha256(pef_ctx.encode()).hexdigest()[:32] if pef_ctx else ""
        request_hash = request_hash_var.get(None)

        entry: dict[str, Any] = {
            "schema_version": 2,
            "run_id": self._current_run_id(),
            "trace_id": trace_id,
            "timestamp": ts,
            "session_id": session_id_var.get(None),
            "turn": turn,
            "tenant_label": auth_label,
            "consumer_label": auth_label,
            "policy_profile": policy_profile,
            "policy_version": self._default_policy,
            "outcome": decision.action.name,
            "failed_constraints": failed_constraints,
            "original_response": decision.original_response,
            "governed_response": decision.governed_response,
            "state_hash": state_hash,
        }
        _request_metadata = request_metadata_snapshot(get_request_metadata())
        if _request_metadata is not None:
            entry["request_metadata"] = _request_metadata
        _current_proxy_run_id = self._current_proxy_run_id()
        if _current_proxy_run_id is not None:
            entry["proxy_run_id"] = _current_proxy_run_id
        if request_hash:
            entry["request_hash"] = request_hash
        _req_dom = domain_var.get(None)
        if _req_dom:
            entry["request_domain"] = _req_dom
        if decision.rationale:
            entry["rationale"] = decision.rationale
        if decision.governance_note:
            entry["governance_note"] = decision.governance_note
        if decision.action != InterventionAction.PASS:
            entry["governed_request_metadata"] = _build_governed_request_metadata(
                entry=entry,
                decision=decision,
            )

        self._attach_chain_of_custody(decision, entry)
        _event_gid = (
            decision.forensic_event.get("governor_policy_id")
            if isinstance(decision.forensic_event, dict)
            else None
        )
        _fe_domain = (
            decision.forensic_event.get("domain")
            if isinstance(decision.forensic_event, dict)
            else None
        )
        _fe_ctx = (
            decision.forensic_event.get("context_provenance")
            if isinstance(decision.forensic_event, dict)
            else None
        )
        _fe_domain_source = (
            _fe_ctx.get("domain_source")
            if isinstance(_fe_ctx, dict)
            else None
        )
        apply_ruleset_provenance_fields(
            entry,
            chain_of_custody=entry.get("chain_of_custody"),
            policy_profile=policy_profile,
            outcome=decision.action.name,
            governor_policy_id=_event_gid,
        )
        apply_domain_reclassification_fields(
            entry,
            request_domain=entry.get("request_domain"),
            effective_domain=entry.get("domain") or _fe_domain,
            domain_source=_fe_domain_source,
        )
        provenance_fields = apply_instrument_provenance_to_row(
            entry,
            chain_of_custody=entry.get("chain_of_custody"),
            policy_profile=policy_profile,
            outcome=decision.action.name,
            governor_policy_id=_event_gid,
            policy_version=self._default_policy,
            signing_key_configured=self._secret_key is not None,
            attestation_signed=False,
        )
        if decision.forensic_event is not None:
            apply_ruleset_provenance_fields(
                decision.forensic_event,
                chain_of_custody=entry.get("chain_of_custody"),
                policy_profile=policy_profile,
                outcome=decision.action.name,
                governor_policy_id=_event_gid,
            )
            apply_domain_reclassification_fields(
                decision.forensic_event,
                request_domain=entry.get("request_domain"),
                effective_domain=decision.forensic_event.get("domain"),
                domain_source=_fe_domain_source,
            )
            refresh_forensic_event_hash(decision.forensic_event)
            enforce_forensic_event_for_append(decision.forensic_event)
            entry["forensic_event"] = decision.forensic_event
            if pef_snapshot is not None:
                entry["pef_snapshot"] = pef_snapshot

        attach_delivery_and_epistemic_audit_fields(
            entry,
            stream=stream,
            stream_completed=stream_completed,
            stream_abort_reason=stream_abort_reason,
            stream_truncated=stream_truncated,
            stream_dropped_chars=stream_dropped_chars,
            epistemic_normalisation_applied=epistemic_normalisation_applied,
        )
        if state_native_handled is not None:
            entry["state_native_handled"] = state_native_handled

        if pef_turn_classification is not None:
            entry["pef_turn_classification"] = pef_turn_classification
        if pef_hold_transition is not None:
            entry["pef_hold_transition"] = pef_hold_transition

        if decision.epistemic_state is not None:
            entry["epistemic_state"] = decision.epistemic_state

        if at_verification_basis is not None:
            decision.at_verification_basis = at_verification_basis
            entry["at_verification_basis"] = at_verification_basis

        if log_slice is not None:
            entry.update(log_slice)

        if decision.unexpected_unclassified_termination:
            entry["unexpected_unclassified_termination"] = True
            entry["fallback_reason"] = decision.fallback_reason

        from aurora_lens.sovereign.audit_envelope import apply_provider_route_to_audit_entry

        apply_provider_route_to_audit_entry(entry, decision)
        self._apply_evidence_capture(entry, decision, pre_llm=pre_llm)
        attestation_fields = apply_attestation_fields_to_row(
            entry,
            chain_of_custody=entry.get("chain_of_custody"),
            signing_key=self._secret_key,
        )
        sync_instrument_provenance_to_decision(
            decision,
            {**provenance_fields, **attestation_fields},
            signature_status=entry.get("signature_status"),
        )

        if self._audit_path_str:
            prev_snapshot = self._prev_cid
            new_cid = append_audit_entry(
                self._audit_path_str,
                entry,
                signing_key=self._secret_key,
                prev_cid=self._prev_cid,
            )
            if new_cid:
                decision.cid = new_cid
                self._prev_cid = new_cid
                finalize_decision_record_hash(decision, entry, cid=new_cid)
                decision.signature_status = str(entry.get("signature_status") or decision.signature_status or "")
                js_snap: dict[str, Any] = {
                    "trace_id": trace_id,
                    "prev_hash": prev_snapshot,
                    "hash": new_cid,
                }
                fe_st = decision.forensic_event.get("state_hash") if decision.forensic_event else None
                sh = fe_st if fe_st is not None else entry.get("state_hash")
                if sh is not None:
                    js_snap["state_hash"] = sh
                finalize_audit_receipt_snapshot(decision, chain_fields=js_snap)

        # Produce AttestedOutput (HMAC-signed)
        if self._secret_key is not None:
            content = decision.corrected_response or decision.original_response or ""
            from aurora_lens.context import session_id_var as _sid_var, trace_id_var as _tid_var
            attested = AttestedOutput(
                content=content,
                policy_cid=f"policy:{policy_profile}",
                rationale_code=decision.action.name,
                decision=decision.action.name,
                event_cid=cid,
                signature="",
                meta={
                    "turn": turn,
                    "flags": len(decision.flags),
                    "attempt": decision.attempt,
                    "timestamp": ts,
                    "trace_id": trace_id_var.get(None) or "",
                    "session_id": _sid_var.get(None),
                    "run_id": self._current_run_id(),
                },
            )
            self._last_attestation = sign_attested_output(attested, self._secret_key)
            decision.signature_status = str(entry.get("signature_status") or decision.signature_status or "")

    # ── Public utilities ──────────────────────────────────────────

    def log_subsystem_audit_event(
        self,
        *,
        op: str,
        payload: dict[str, Any],
        trace_id: str | None = None,
    ) -> None:
        """Append a proxy/subsystem event to the AFL ledger when backend is ``ledger``.

        No-op when the flat JSONL backend is in use (no ``ForensicLedger``). This
        avoids interleaving :func:`aurora_lens.govern.audit_io.append_audit_entry`
        lines into the same path as hash-chained AFL envelopes.
        """
        if self._ledger is None:
            return
        from aurora_lens.context import trace_id_var as _trace_id_var

        tid = trace_id if trace_id is not None else _trace_id_var.get(None)
        if tid is None:
            tid = self._trace_id or ""
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        envelope = dict(sorted({**payload, "trace_id": tid, "timestamp": ts}.items()))
        digest_hex = hashlib.sha256(
            json.dumps(envelope, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        envelope["event_hash"] = "sha256:" + digest_hex
        fp = PEFFingerprint(
            kind=op,
            span=tid,
            head=digest_hex[:16],
            rationale=f"{op}:{ts}",
        )
        cid = self._cid_provider.cid_for_pef(fp)
        self._ledger.append(
            op=op,
            payload_data=envelope,
            cid=cid,
            scope={"domain": "subsystem", "subdomain": "aurora-lens-proxy", "authority": "proxy"},
        )

    def log_clarification_resolution(
        self,
        *,
        turn: int,
        clarification_resolution: dict[str, Any],
    ) -> str | None:
        """Append USER_DISAMBIGUATION to AFL ledger or flat JSONL (never no-op on JSONL)."""
        from aurora_lens.context import (
            auth_label_var,
            domain_var,
            request_hash_var,
            session_id_var,
            trace_id_var,
        )

        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        trace_id = trace_id_var.get(None) or self._trace_id or ""
        envelope: dict[str, Any] = {
            "clarification_resolution": clarification_resolution,
            "trace_id": trace_id,
            "timestamp": ts,
            "session_id": session_id_var.get(None),
            "turn": turn,
        }
        envelope = dict(sorted(envelope.items()))
        digest_hex = hashlib.sha256(
            json.dumps(envelope, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()
        envelope["event_hash"] = "sha256:" + digest_hex

        if self._ledger is not None:
            fp = PEFFingerprint(
                kind=CLARIFICATION_RESOLUTION_OUTCOME,
                span=trace_id,
                head=digest_hex[:16],
                rationale=f"{CLARIFICATION_RESOLUTION_OUTCOME}:{ts}",
            )
            cid = self._cid_provider.cid_for_pef(fp)
            self._ledger.append(
                op=CLARIFICATION_RESOLUTION_OUTCOME,
                payload_data=envelope,
                cid=cid,
                scope={"domain": "governance", "subdomain": "clarification", "authority": "lens"},
            )
            return cid

        if not self._audit_path_str:
            return None

        auth_label = auth_label_var.get(None)
        request_hash = request_hash_var.get(None)
        _req_dom = domain_var.get(None)
        entry = build_clarification_resolution_audit_entry(
            schema_version=2,
            run_id=self._current_run_id(),
            trace_id=trace_id,
            timestamp=ts,
            session_id=session_id_var.get(None),
            turn=turn,
            tenant_label=auth_label,
            mode="public",
            policy_version=self._default_policy,
            clarification_resolution=clarification_resolution,
            request_hash=request_hash,
            request_domain=_req_dom,
        )
        _current_proxy_run_id = self._current_proxy_run_id()
        if _current_proxy_run_id is not None:
            entry["proxy_run_id"] = _current_proxy_run_id

        new_cid = append_audit_entry(
            self._audit_path_str,
            entry,
            signing_key=self._secret_key,
            prev_cid=self._prev_cid,
        )
        if new_cid:
            self._prev_cid = new_cid
        return new_cid

    @property
    def last_attestation(self) -> AttestedOutput | None:
        """The most recent AttestedOutput (None if signing not configured)."""
        return self._last_attestation

    @property
    def kernel_stats(self) -> dict:
        """Stub — GovernanceKernel is not used. Returns minimal stats dict."""
        return {"turn_index": 0, "status": "not_used"}

    def verify_ledger(self, signing_keys: Sequence[bytes] | None = None) -> bool:
        """Verify the forensic ledger hash chain and, when configured, per-line HMAC.

        Uses primary + ``audit_signing_keys`` from the bridge when ``signing_keys``
        is omitted. Pass an explicit sequence to override (e.g. proxy verify endpoint).

        Returns True if chain is valid (or no ledger configured).
        """
        return self.verify_ledger_detailed(signing_keys=signing_keys).ok

    def verify_ledger_detailed(
        self, signing_keys: Sequence[bytes] | None = None
    ) -> LedgerVerifyDetail:
        """Like ``verify_ledger`` but returns chain vs HMAC status and first failure per axis."""
        if self._ledger is None:
            return LedgerVerifyDetail(
                ok=True,
                entries_checked=0,
                chain_ok=True,
                hmac_ok=None,
                hmac_checked=False,
            )
        keys: list[bytes] = (
            list(signing_keys)
            if signing_keys is not None
            else self._ledger_verify_keys()
        )
        return self._ledger.verify_detailed(signing_keys=keys)

    def verify_attestation(
        self,
        attested: AttestedOutput | None = None,
    ) -> bool:
        """Verify an AttestedOutput signature.

        If no attestation passed, verifies the last one produced.
        Returns False if no secret key or no attestation available.
        """
        if self._secret_key is None:
            return False
        target = attested or self._last_attestation
        if target is None:
            return False
        return verify_governed_output(target, self._secret_key)

    @property
    def ledger_stats(self) -> dict | None:
        """Return ledger statistics (None if no ledger configured)."""
        if self._ledger is None:
            return None
        return self._ledger.get_stats()

    # ── Internal ──────────────────────────────────────────────────

    def _build_rationale(self, flags: list[Flag], action: InterventionAction) -> str:
        if not flags:
            return "No verification flags"

        flag_types = set(f.flag_type.name for f in flags)
        severities = set(f.severity for f in flags)
        worst = "error" if "error" in severities else "warning"

        return (
            f"{action.name}: {len(flags)} flag(s) "
            f"[{', '.join(sorted(flag_types))}] "
            f"(worst severity: {worst})"
        )

    def _build_governance_note(self, flags: list[Flag]) -> str:
        parts = []
        for flag in flags:
            parts.append(f"{flag.flag_type.name}: {flag.claim} — {flag.evidence}")
        return "; ".join(parts)
