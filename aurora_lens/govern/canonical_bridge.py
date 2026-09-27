"""CanonicalScannerGateBridge — default **Governor** bridge (canonical decision pipeline).

Architecture
------------
The canonical pipeline is:

  flags + mode + context
    -> StatusTranslator   -> LensStatus
    -> ContextResolver    -> (Domain, AuthorityClass, UserClass, Provenance)
    -> PolicyResolver     -> GovernorPolicy
    -> PolicyProjector    -> RuntimeDecisionProjection
    -> bridge plumbing    -> GovernanceDecision + forensic_event + attestation

InterventionPolicy is NOT called from decide(). PolicyResolver is the sole
decision authority. This eliminates split authority — the condition under which
governance systems start "lying politely" by allowing two policy brains to
produce incompatible verdicts and resolving the conflict silently.

This bridge extends AuroraScannerGateBridge solely to reuse its ledger, CID
provider, and HMAC signing infrastructure. The decide() method is completely
replaced. The log_decision() method enriches the audit payload with canonical
Governor metadata via aurora_lens.governor.audit.enrich_forensic_envelope().

All infrastructure is local — no external dependencies required.
"""

from __future__ import annotations

import contextvars
import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
from aurora_lens.govern.adapters.status_translator import StatusTranslator
from aurora_lens.govern.adapters.context_resolver import ContextResolver
from aurora_lens.govern.adapters.policy_projector import PolicyProjector
from aurora_lens.govern.adapters.runtime_types import RuntimeDecisionProjection
from aurora_lens.govern.decision import (
    GovernanceDecision,
    InterventionAction,
    apply_epistemic_state_from_flags,
    attach_rule_result,
)
from aurora_lens.govern.bridge import (
    ESCALATION_ROUTES,
    PATHWAY_IDS_WITH_TYPED_RENDERER,
    _ACTION_TO_PATHWAY,
    build_forensic_event,
    refresh_forensic_event_hash,
)
from aurora_lens.verify.flags import Flag, FlagType

from aurora_lens.governor.resolver import PolicyResolver
from aurora_lens.governor.audit import enrich_forensic_envelope

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aurora_lens.adapters.base import LLMAdapter
    from aurora_lens.pef.state import PEFState


def _ensure_decision_pathway_for_enforce(
    decision: GovernanceDecision,
    projection: RuntimeDecisionProjection,
) -> None:
    """Ensure ``decision.pathway_id`` is non-null and recognized by ``enforce()``.

    PASS / SOFT_CORRECT bypass pathway dispatch; all other actions require a pathway_id
    that matches a typed renderer branch. Missing or unknown IDs previously drove
    ``enforce()`` through the domain-fallback ladder with ``domain is None`` (Tier-4
    malformed-decision copy) even when the Governor projection carried a lawful pathway.

    Does not alter InterventionPolicy classification or Lens routing — pathway repair only.
    """
    if decision.action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT):
        return
    canonical = projection.pathway_id.value
    pid = decision.pathway_id
    if pid is None or (isinstance(pid, str) and not str(pid).strip()):
        decision.pathway_id = canonical
        pid = decision.pathway_id
    if pid in PATHWAY_IDS_WITH_TYPED_RENDERER:
        return
    if canonical in PATHWAY_IDS_WITH_TYPED_RENDERER:
        decision.pathway_id = canonical
        return
    fb = _ACTION_TO_PATHWAY.get(decision.action)
    if fb is not None:
        decision.pathway_id = fb


# Safety-veto flags: canonical HARD_STOP cannot be softened for any per-key
# policy, including "moderate". These flags represent immediate physical harm,
# illegal acts, or forensic obligations that exist independently of commercial
# policy preference. The Governor's HARD_STOP on these is always final.
_SAFETY_VETO_FLAGS: frozenset[FlagType] = frozenset({
    FlagType.SELF_HARM_INSTRUCTION,
    FlagType.ILLEGAL_INSTRUCTION,
    FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
    FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
    FlagType.NUMERIC_MEDICAL_INSTRUCTION,
    FlagType.EMERGENCY_TRIAGE_GUIDANCE,
    FlagType.AGENCY_VIOLATION_ASSISTANCE,
})

# RAG harness: moderate-key must not soften governed non-admit to SOFT_CORRECT (would admit a branch / absence as PASS).
_NO_MODERATE_SOFTEN_FLAGS: frozenset[FlagType] = frozenset({
    FlagType.RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT,
    FlagType.RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT,
})


# Personalized legal advisory (request-side blocked act): never soften to SOFT_CORRECT — that
# would admit model text despite the legal consequence corridor (outcome prediction, etc.).
_MODERATE_NO_SOFTEN_PERSONALIZED_LEGAL: frozenset[FlagType] = frozenset({
    FlagType.PERSONALIZED_LEGAL_ADVICE,
})


# Per-async-context storage for the GovernorPolicy resolved in decide().
# This is read by log_decision() to enrich the forensic payload.
# Using ContextVar ensures async-safe isolation between concurrent requests.
_scanner_gate_policy_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_aurora_scanner_gate_policy",
    default=None,
)
_scanner_gate_projection_var: contextvars.ContextVar[RuntimeDecisionProjection | None] = (
    contextvars.ContextVar("_aurora_scanner_gate_projection", default=None)
)


class CanonicalScannerGateBridge(AuroraScannerGateBridge):
    """Governance bridge using the canonical Governor pipeline as sole decision authority.

    PolicyResolver replaces InterventionPolicy for all admissibility decisions.
    GovernanceKernel is retained for signing, CID generation, and ledger append
    only — kernel_step() is never called.

    The public interface (decide, intervene, log_decision, verify_ledger,
    verify_attestation, last_attestation, kernel_stats, ledger_stats) is
    identical to AuroraScannerGateBridge. Existing callers work without changes.

    Additional introspection:
        bridge.last_scanner_gate_policy — the GovernorPolicy resolved on the
                                          last call to decide() (for testing)
        bridge.last_projection          — the RuntimeDecisionProjection from the
                                          last call to decide()
    """

    def __init__(
        self,
        mode: str = "public",
        audit_path: str | Path | None = None,
        secret_key: bytes | None = None,
        constraints: dict | None = None,
        max_revision_attempts: int = 1,
        backend: str = "ledger",
        default_policy: str = "strict",
        policy_version: str = "1.0",
        policy_matrix_path: str | Path | None = None,
        audit_signing_keys: tuple[bytes, ...] = (),
        proxy_run_id: str | None = None,
        evidence_capture_mode: str = "plaintext_dev",
        evidence_encryption_key: str | bytes | None = None,
    ):
        """
        Args:
            mode:                  Deployment policy mode ("public" | "enterprise" | "open").
                                   Used by StatusTranslator for mode-sensitive flags.
            audit_path:            Path for audit log.
            secret_key:            Primary HMAC key for AttestedOutput and AFL ledger ``sig``.
            audit_signing_keys:    Historical keys for ``verify_ledger`` after rotation.
            proxy_run_id:           Optional proxy process id; forwarded to the ledger for
                                    ``verify_pef_linkage`` cross-epoch pairing (see proxy app).
            constraints:           Kernel constraints (passed to substrate; not used for policy).
            max_revision_attempts: Hard limit on revision loop (1-10).
            backend:               Audit format — "ledger" (AFL-JSONL-1) or "jsonl" (flat JSONL).
            default_policy:        Deployment default policy name for flat JSONL policy_profile.
        """
        # AuroraScannerGateBridge.__init__ sets up ledger, CID provider, attestation.
        # We pass policy=None; the parent stores DEFAULT_STRICT but it is never
        # called from our override of decide().
        super().__init__(
            policy=None,
            audit_path=audit_path,
            secret_key=secret_key,
            constraints=constraints or {},
            max_revision_attempts=max_revision_attempts,
            backend=backend,
            default_policy=default_policy,
            audit_signing_keys=audit_signing_keys,
            proxy_run_id=proxy_run_id,
            evidence_capture_mode=evidence_capture_mode,
            evidence_encryption_key=evidence_encryption_key,
        )

        if mode not in ("public", "enterprise", "open"):
            raise ValueError(
                f"Unknown mode: {mode!r} — expected 'public', 'enterprise', or 'open'"
            )
        self._mode = mode

        # Canonical Governor pipeline components.
        self._status_translator = StatusTranslator()
        self._context_resolver = ContextResolver()
        self._policy_projector = PolicyProjector()
        self._policy_resolver = PolicyResolver(
            matrix_path=Path(policy_matrix_path) if policy_matrix_path else None,
            policy_version=policy_version,
        )
        self._policy_version = policy_version

        # Introspection (last call; useful for testing).
        self._last_scanner_gate_policy: Any = None
        self._last_projection: RuntimeDecisionProjection | None = None

    # -- GovernanceBridge.decide() — canonical Governor pipeline -------------

    async def decide(
        self,
        flags: list[Flag],
        response_text: str,
        pef: "PEFState",
    ) -> GovernanceDecision:
        """Evaluate flags through the canonical Governor pipeline.

        Pipeline:
          1. StatusTranslator  — flags + mode -> LensStatus
          2. ContextResolver   — flags + context vars -> (Domain, AC, UC, Provenance)
          3. PolicyResolver    — (Domain, AC, LensStatus, UC) -> GovernorPolicy
          4. PolicyProjector   — GovernorPolicy -> RuntimeDecisionProjection
          5. Build GovernanceDecision from projection.
             Canonical pathway fields are copied directly; no reinterpretation.

        GovernanceKernel.kernel_step() is NOT called here.
        InterventionPolicy.evaluate_with_rule() is NOT called here.
        Per-key "moderate" softening is a post-projection corridor check only;
        it never calls the legacy policy engine.
        """
        # 1. Terminal LensStatus from flags + mode.
        # Optional per-request mode from host policy_profile (proxy); else deployment mode.
        from aurora_lens.context import (
            auth_policy_var as _auth_policy_var,
            governance_mode_override_var as _gov_mode_ov,
            metadata_policy_override_var as _meta_pol_ov,
        )

        _gov = _gov_mode_ov.get()
        if _gov not in ("public", "enterprise", "open"):
            _gov = None
        effective_mode = _gov or self._mode
        lens_status = self._status_translator.translate(flags, effective_mode)

        # 2. Governance corridor context.
        domain, authority, user_class, provenance = self._context_resolver.resolve(flags)

        # 3. Canonical Governor decision.
        # reason_code is the primary flag type name, used by the resolver's level-0
        # flag-class key lookup (domain:authority:status:reason_code). This lets flags
        # that share a corridor — e.g. SELF_HARM_INSTRUCTION and EMERGENCY_TRIAGE_GUIDANCE
        # both in medical:GP:STOP — resolve to distinct pathways without splitting domains.
        reason_code = flags[0].flag_type.name if flags else None
        policy = self._policy_resolver.resolve(
            domain=domain,
            authority=authority,
            status=lens_status,
            user_class=user_class,
            reason_code=reason_code,
        )

        # 4. Project to runtime decision record.
        projection = self._policy_projector.project(policy, provenance)

        # Store for log_decision() and introspection — async-safe via ContextVar.
        _scanner_gate_policy_var.set(policy)
        _scanner_gate_projection_var.set(projection)
        self._last_scanner_gate_policy = policy
        self._last_projection = projection

        # 5. Build GovernanceDecision carrying canonical pathway metadata.
        action = projection.intervention_action
        cid = None  # CID assigned in log_decision() when ledger is present.

        # Per-key policy override: "moderate" key may soften a canonical HARD_STOP
        # to FORCE_REVISE, but only for non-safety-critical flags.
        # Safety-veto flags (_SAFETY_VETO_FLAGS) are always final regardless of key policy.
        # No InterventionPolicy call — the canonical Governor is the sole authority.
        _per_key_policy = _auth_policy_var.get(None)
        _meta_pol = _meta_pol_ov.get()
        if _meta_pol not in ("strict", "moderate"):
            _meta_pol = None
        _effective_policy_name = _per_key_policy or _meta_pol or self._default_policy
        _softened_pathway_id: str | None = None
        if (
            _effective_policy_name == "moderate"
            and action in (InterventionAction.FORCE_REVISE, InterventionAction.HARD_STOP)
            and flags
            and not {f.flag_type for f in flags} & _SAFETY_VETO_FLAGS
            and not {f.flag_type for f in flags} & _NO_MODERATE_SOFTEN_FLAGS
            and not {f.flag_type for f in flags} & _MODERATE_NO_SOFTEN_PERSONALIZED_LEGAL
        ):
            # SOFT_CORRECT: deliver model output with governance annotation.
            # The canonical pathway_id is kept for audit (it shows what the Governor
            # decided before the override, which is the forensically relevant fact).
            action = InterventionAction.SOFT_CORRECT
            _softened_pathway_id = None

        rationale = self._build_rationale(flags, action)
        note = (
            self._build_governance_note(flags)
            if action == InterventionAction.SOFT_CORRECT
            else None
        )

        # Resource routing: use escalation_target from Governor; fall back to
        # ESCALATION_ROUTES for backward compatibility.
        resource = projection.escalation_target
        if resource is None and flags:
            route = ESCALATION_ROUTES.get(
                flags[0].flag_type.name, (None, None, None, None)
            )
            resource = route[3] if len(route) > 3 else None

        # rule_id is not available from the canonical Governor (it resolves by
        # domain:authority:status:user_class, not per-flag rules). Leave None.
        #
        # policy: API key override > request_metadata policy_profile > deployment default.
        # This keeps the field human-readable ("strict" | "moderate") consistent with
        # BuiltinBridge and the flat JSONL audit format.
        decision = GovernanceDecision(
            action=action,
            flags=flags,
            rationale=rationale,
            policy=_effective_policy_name,
            attempt=0,
            governance_note=note,
            cid=cid,
            rule_id=None,
            safe_alt=None,
            resource=resource,
        )

        # Copy canonical continuation fields directly from the projection.
        # When a moderate-key softening occurred, _softened_pathway_id overrides
        # the canonical pathway so that action and pathway_id stay coherent in audit.
        decision.pathway_id = _softened_pathway_id or projection.pathway_id.value
        _ensure_decision_pathway_for_enforce(decision, projection)
        decision.output_mode = projection.output_mode.value
        decision.allowed_continuations = self._resolve_allowed_continuations(
            policy,
            reason_code=reason_code,
            attempted_user_request=response_text,
            pef=pef,
        )
        decision.commitment_closed = projection.commitment_closed
        decision.interaction_open = projection.interaction_open
        decision.forensic_obligations = [fo.value for fo in projection.forensic_obligations]
        decision.resolution_mode = projection.resolution_mode.value
        attach_rule_result(
            decision,
            domain=domain.value if hasattr(domain, "value") else str(domain).lower(),
            reason_code=reason_code,
            rule_id=flags[0].rule_id if flags and flags[0].rule_id else None,
        )
        apply_epistemic_state_from_flags(decision)

        return decision

    def _postprocess_forensic_event(self, decision: GovernanceDecision) -> None:
        """Enrich base forensic event with Governor fields, refresh ``event_hash``, before ledger append."""
        if decision.forensic_event is None:
            return
        policy = _scanner_gate_policy_var.get(None)
        projection = _scanner_gate_projection_var.get(None)
        if policy is None:
            return

        reason_code = decision.flags[0].flag_type.name if decision.flags else "none"
        parent_ts = decision.forensic_event.get("timestamp")
        if not isinstance(parent_ts, str) or not parent_ts.strip():
            parent_ts = None
        enriched = enrich_forensic_envelope(
            decision.forensic_event,
            policy,
            reason_code,
            parent_timestamp=parent_ts,
        )
        if projection is not None and projection.provenance is not None:
            p = projection.provenance
            enriched["context_provenance"] = {
                "domain_source": p.domain_source,
                "authority_source": p.authority_source,
                "user_class_source": p.user_class_source,
            }

        enriched["pathway_id"] = decision.pathway_id
        enriched["output_mode"] = decision.output_mode
        enriched["commitment_closed"] = decision.commitment_closed
        enriched["interaction_open"] = decision.interaction_open
        enriched["forensic_obligations"] = decision.forensic_obligations
        enriched["resolution_mode"] = decision.resolution_mode
        enriched["rendered_from_policy"] = True
        enriched["policy_version"] = self._policy_resolver.active_version
        enriched["policy_source"] = self._policy_resolver.policy_source

        decision.forensic_event = enriched
        refresh_forensic_event_hash(decision.forensic_event)

    def _resolve_allowed_continuations(
        self,
        policy: Any,
        *,
        reason_code: str | None,
        attempted_user_request: str,
        pef: "PEFState",
    ) -> list[str]:
        """Return Governor-authorized blocked continuations for this turn.

        Inputs intentionally include blocked reason, attempted request text, and
        currently admitted context (PEF) so continuation authorization remains a
        governed decision surface instead of a renderer-side string choice.
        """
        _ = reason_code
        _admitted_context_available = bool(pef.entities or pef.relationships)
        if policy.lens_status.value not in {"REFUSE", "STOP"}:
            return []
        if not policy.interaction_open:
            return []
        if not attempted_user_request.strip():
            return []
        return [cap.value for cap in policy.allowed_continuations]

    # -- log_decision() — enriched with canonical Governor fields ------------

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
        """Log to ForensicLedger + AttestedOutput; forensic_event enriched before append via hook."""
        super().log_decision(
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
            pef_turn_classification=pef_turn_classification,
            pef_hold_transition=pef_hold_transition,
            at_verification_basis=at_verification_basis,
            epistemic_normalisation_applied=epistemic_normalisation_applied,
            state_native_handled=state_native_handled,
        )

    # -- Introspection ----------------------------------------------------------

    @property
    def last_scanner_gate_policy(self) -> Any:
        """The GovernorPolicy resolved on the last call to decide(). For testing."""
        return self._last_scanner_gate_policy

    @property
    def last_projection(self) -> RuntimeDecisionProjection | None:
        """The RuntimeDecisionProjection from the last call to decide(). For testing."""
        return self._last_projection
