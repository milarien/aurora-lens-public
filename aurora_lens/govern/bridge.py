"""Governance bridge — decides and executes interventions.

GovernanceBridge is the abstract interface.
BuiltinBridge is the lightweight built-in implementation.

Key design:
- enforce() branches on pathway_id. escalation_level is NOT the execution switch.
  It is retained for telemetry and audit only.
- Both BuiltinBridge and the Governor bridge (CanonicalScannerGateBridge) always produce a pathway_id.
  pathway_id=None is error-containment, not a dispatch path.
- No LLM re-prompting. No revision loop. No ad hoc prose. No recovery of
  blocked content by implication, tone, or paraphrase.
- Audit log is append-only JSONL with a stable, boring schema.

PATHWAY RENDERING CONTRACT
--------------------------
Pathway renderers are deterministic. They render from:
  - decision.flags (to select the right hard-stop continuation)
  - decision.resource (escalation routing from policy)
  - decision.interaction_open (whether to invite further input)
  - static policy-authorized text only

No second LLM call. No improvised prose. No "helpful" framing that narrows
toward a blocked determination.

PATHWAY MAP
-----------
Pathway                      Flag class                    Continuation shape
---------------------------- ----------------------------- -----------------------------------------------
P_STOP_ESCALATE_EMERGENCY    EMERGENCY_TRIAGE_GUIDANCE     Immediate escalation; conversation closed
P_STOP_SUPPORTIVE_DEESCALATE SELF_HARM_INSTRUCTION         Supportive refusal; conversation open
P_STOP_TERMINAL              MEDICAL_DOSAGE_*              Refusal + pharmacist / clinician
                             PERSONALIZED_MEDICAL_ADVICE   Refusal + qualified clinician redirect
                             ILLEGAL_INSTRUCTION           Clean refusal only — no workaround adjacency
                             TARGETED_DEFAMATION           Role-specific refusal (see DefamationRole)
                             PERSONALIZED_LEGAL_ADVICE     Refusal + legal aid / resource offer
                             PERSONALIZED_FINANCIAL_ADVICE Refusal + licensed financial adviser redirect
                             (all others)                  Clean refusal
P_STOP_FORENSIC              SENSITIVE_PII_EXPOSURE        Stop + forensic notification chain
"""

from __future__ import annotations

import ast
import datetime
import hashlib
import json
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aurora_lens.context import (
    auth_label_var,
    auth_policy_var,
    domain_var,
    get_lock_metadata,
    get_request_hash,
    get_request_metadata,
    session_id_var,
    trace_id_var,
)
from aurora_lens.request_metadata import request_metadata_snapshot
from aurora_lens.govern.audit_io import CHAIN_GENESIS, append_audit_entry, append_checkpoint_entry
from aurora_lens.govern.clarification_audit import (
    build_clarification_resolution_audit_entry,
)
from aurora_lens.govern.chain_of_custody import (
    apply_ruleset_provenance_fields,
    build_chain_of_custody_bundle,
    get_application_version,
)
from aurora_lens.govern.instrument_provenance import (
    apply_attestation_fields_to_row,
    apply_instrument_provenance_to_row,
    finalize_decision_record_hash,
    sync_instrument_provenance_to_decision,
)
from aurora_lens.govern.forensic_append_guard import enforce_forensic_event_for_append
from aurora_lens.log_slice import consume_log_slice
from aurora_lens.govern.decision import (
    GovernanceDecision,
    InterventionAction,
    apply_epistemic_state_from_flags,
    attach_rule_result,
)
from aurora_lens.govern.evidence_capture import (
    EvidenceCaptureConfig,
    attach_evidence_fields_to_audit_entry,
    build_evidence_vault_for_bridge,
    finalize_audit_receipt_snapshot,
)
from aurora_lens.govern.policy import (
    InterventionPolicy,
    PolicyRule,
    DEFAULT_STRICT,
    DEFAULT_MODERATE,
)
from aurora_lens.govern.user_copy import (
    USER_MESSAGE_INTERPRETATION_FAILED,
    USER_MESSAGE_RESPONSE_INSUFFICIENT_DETAIL_CLOSED,
    USER_MESSAGE_RESPONSE_INSUFFICIENT_DETAIL_OPEN,
)
from aurora_lens.govern.governed_copy import (
    compose_insufficient_structure_message,
    compose_unresolved_referent_governed_message,
)
from aurora_lens.verify.flags import DefamationRole, Flag, FlagType

# Forensic envelope schema version — must match governor.forensic_schema.FORENSIC_SCHEMA_VERSION.
# Defined here to avoid a governor/ import at bridge load time (verify_audit subprocess path).
FORENSIC_SCHEMA_VERSION = "1.0"

if TYPE_CHECKING:
    from aurora_lens.adapters.base import LLMAdapter
    from aurora_lens.pef.state import PEFState


# Audit log schema version (D1: canonical envelope)
_AUDIT_SCHEMA_VERSION = 2

# Thread-local sentinel used by _domain_fallback_continuation to signal
# error-containment (Tier 3/4) back to enforce() without signature changes.
# enforce() resets this before each render call and reads it after.
_fallback_sentinel = threading.local()

# Regex to strip a leading article ("a", "an", "the") from a resource string.
# Used by _with_article() to avoid doubling the article when the resource
# already carries one (e.g. "a licensed lawyer" → "A licensed lawyer", not
# "A a licensed lawyer").
import re as _re
_LEADING_ARTICLE_RE = _re.compile(r"^(?:a|an|the)\s+", _re.IGNORECASE)


def _with_article(resource: str) -> str:
    """Prepend 'A' to *resource*, stripping any leading article first.

    Examples:
        "a licensed lawyer"       → "A licensed lawyer"
        "the supervising lawyer"  → "The supervising lawyer"
        "qualified legal aid"     → "A qualified legal aid"
    """
    stripped = _LEADING_ARTICLE_RE.sub("", resource).strip()
    return f"A {stripped}"


def _extract_chunk_ids_from_flag_evidence(flags: list[Flag]) -> list[str]:
    for flag in flags:
        evidence = str(flag.evidence or "")
        marker = "chunk_ids="
        start = evidence.find(marker)
        if start < 0:
            continue
        raw = evidence[start + len(marker):].split(";", 1)[0].strip()
        if not raw:
            continue
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return []


def _failure_kind_from_decision(decision: GovernanceDecision) -> str | None:
    rationale = str(decision.rationale or "").strip()
    if rationale.startswith("rag_evidence_admissibility_gate:"):
        _, _, value = rationale.partition(":")
        kind = value.strip()
        return kind or None
    return None


def _build_governed_request_metadata(
    *,
    entry: dict[str, object],
    decision: GovernanceDecision,
) -> dict[str, object]:
    request_meta = entry.get("request_metadata")
    request_meta_dict = request_meta if isinstance(request_meta, dict) else {}
    provider_route = (
        request_meta_dict.get("provider_route")
        if isinstance(request_meta_dict.get("provider_route"), dict)
        else {}
    )
    payload: dict[str, object] = {
        "trace_id": str(entry.get("trace_id") or ""),
        "request_hash": str(entry.get("request_hash") or ""),
        "timestamp": str(entry.get("timestamp") or ""),
        "turn": int(entry.get("turn") or 0),
        "lens_action": decision.action.name,
        "policy_profile": str(entry.get("policy_profile") or ""),
        "policy_version": str(entry.get("policy_version") or ""),
        "pathway_id": str(entry.get("pathway_id") or ""),
        "request_domain": str(entry.get("request_domain") or ""),
        "source_scope": list(request_meta_dict.get("source_scope") or []),
        "record_ids": list(request_meta_dict.get("record_ids") or []),
        "task_domain": str(provider_route.get("task_domain") or ""),
        "consequence_grade": str(provider_route.get("consequence_grade") or ""),
        "chunk_ids": _extract_chunk_ids_from_flag_evidence(decision.flags),
    }
    failure_kind = _failure_kind_from_decision(decision)
    if failure_kind:
        payload["failure_kind"] = failure_kind
    return payload


# Forensic event status: action -> status string
_ACTION_TO_FORENSIC_STATUS: dict[InterventionAction, str] = {
    InterventionAction.CONTAIN: "ASK",
    InterventionAction.FORCE_REVISE: "REFUSE",
    InterventionAction.HARD_STOP: "STOP",
}


def build_forensic_event(
    decision: GovernanceDecision,
    *,
    pre_llm: bool,
    pef_snapshot: dict | None,
    trace_id: str = "",
    timestamp: str = "",
    audit_id: str | None = None,
) -> dict[str, object]:
    """Build canonical forensic_event envelope for non-ADMIT outcomes.

    Every ASK / REFUSE / STOP path produces a forensic event that records:
    - What was blocked and why (failed_constraints, status, domain)
    - What pathway was selected (pathway_id, output_mode, resolution_mode)
    - Whether the determination was closed (commitment_closed)
    - Whether the conversation continues (interaction_open)
    - What forensic obligations attach (forensic_obligations)
    - Content hashes proving what was shown vs suppressed, without storing either
    - PEF state hash for replay verification
    - session_id from ``session_id_var`` when set (conversation correlation on the artifact)

    The event is self-hashed (event_hash) so it can be verified in isolation.
    """
    status = _ACTION_TO_FORENSIC_STATUS.get(decision.action, "REFUSE")
    attempted_action = "call_upstream" if pre_llm else "respond"
    failed_constraints = [f.flag_type.name for f in decision.flags] if decision.flags else []
    if pef_snapshot is not None:
        state_hash = hashlib.sha256(
            json.dumps(pef_snapshot, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
    else:
        state_hash = None
    route = (
        ESCALATION_ROUTES.get(decision.flags[0].flag_type.name, (None, None, None, None))
        if decision.flags
        else (None, None, None, None)
    )
    domain, subdomain = route[0], route[1]
    event: dict[str, object] = {
        "schema_version": FORENSIC_SCHEMA_VERSION,
        "trace_id": trace_id or (audit_id or ""),
        "session_id": session_id_var.get(None),
        "timestamp": timestamp,
        "status": status,
        "attempted_action": attempted_action,
        "domain": domain,
        "subdomain": subdomain,
        "state_hash": state_hash,
    }
    event["failed_constraints"] = failed_constraints

    # Pathway metadata: what continuation was selected and its contractual properties.
    event["pathway_id"] = decision.pathway_id
    event["output_mode"] = decision.output_mode
    event["allowed_continuations"] = list(decision.allowed_continuations)
    event["commitment_closed"] = decision.commitment_closed
    event["interaction_open"] = decision.interaction_open
    event["forensic_obligations"] = decision.forensic_obligations or []
    event["resolution_mode"] = decision.resolution_mode
    # Escalation target: the effective resource surfaced to the user (from
    # policy.escalation_target, optionally overridden by ESCALATION_ROUTES fallback).
    # Required for self-sufficient replay without re-running the resolver.
    event["escalation_target"] = decision.resource

    # Content hashes: prove what was suppressed and what was shown without
    # storing either text in the forensic event itself.
    if decision.original_response is not None:
        event["blocked_response_hash"] = "sha256:" + hashlib.sha256(
            decision.original_response.encode("utf-8")
        ).hexdigest()
    if decision.governed_response is not None:
        event["governed_response_hash"] = "sha256:" + hashlib.sha256(
            decision.governed_response.encode("utf-8")
        ).hexdigest()

    # Self-hash: SHA-256 of canonical JSON of event fields (excluding event_hash itself).
    # Allows the forensic_event to be verified in isolation without the outer audit entry.
    event["event_hash"] = "sha256:" + hashlib.sha256(
        json.dumps(event, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return event


def refresh_forensic_event_hash(event: dict[str, object]) -> None:
    """Recompute ``event_hash`` after mutating a forensic event dict (e.g. Governor enrichment).

    Uses the same canonical JSON rule as :func:`build_forensic_event`.
    """
    body = {k: v for k, v in event.items() if k != "event_hash"}
    event["event_hash"] = "sha256:" + hashlib.sha256(
        json.dumps(body, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


# Phase 7/12: Escalation routing — (domain, subdomain, requires_role, resource) per flag type
ESCALATION_ROUTES: dict[str, tuple[str | None, str | None, str | None, str | None]] = {
    "SELF_HARM_INSTRUCTION": ("human_safety", "self_harm", "clinical_escalation", "crisis support resource"),
    "ILLEGAL_INSTRUCTION": ("security", "illegal", "security_ops", None),
    "AGENCY_VIOLATION_ASSISTANCE": ("harmful", "agency", None, None),
    "AGENCY_RISK_CONTEXT_UNRESOLVED": ("ambiguity", "agency", None, None),
    "TARGETED_DEFAMATION": ("legal", "defamation", "legal_compliance", None),
    "SENSITIVE_PII_EXPOSURE": ("privacy", "pii", "privacy_officer", None),
    "ACADEMIC_INTEGRITY_VIOLATION": ("education", "academic_integrity", None, None),
}


# ── Renderer context types ────────────────────────────────────────────────────
#
# The governing principle:
#   Clarification may expose admissible alternatives.
#   Refusal may not expose blocked substance.
#
# Each pathway class gets its own context type carrying exactly what that class
# of renderer is permitted to see. No renderer receives a GovernanceDecision
# or any field derived from blocked content (evidence, claim, original_response).
#
# AmbiguityRendererContext  → P_ASK_DISAMBIGUATE, P_ASK_MISSING_FACT
#   May carry: normalized PEF candidate labels, missing field names.
#   May NOT carry: evidence text, claim text, any blocked content.
#
# RefusalRendererContext    → P_REFUSE_EXPLAIN_REDIRECT, P_REFUSE_ESCALATE_PRO,
#                             P_HANDOFF_SUMMARY
#   May carry: flag type (for routing), policy-sourced resource, interaction_open.
#   May NOT carry: anything from the blocked determination.
#
# HardStopRendererContext   → P_STOP_TERMINAL, P_STOP_FORENSIC
#   May carry: flag type (for continuation routing), policy-sourced resource.
#   May NOT carry: anything from the blocked determination.

@dataclass(frozen=True)
class AmbiguityRendererContext:
    """Renderer context for clarification pathways.

    candidates: normalized PEF entity/referent labels — e.g. ("Anna's sister", "Emma's sister").
                These come from Flag.candidates, which is populated by the checker from the
                world model. Never from raw evidence text.
    missing_fields: names of missing facts the interaction needs — e.g. ("dosage unit",).
    interaction_open: True for all clarification pathways.
    """
    candidates: tuple[str, ...]
    missing_fields: tuple[str, ...]
    interaction_open: bool = True
    flag_type: FlagType | None = None
    ambiguous_token: str | None = None
    flag_claim: str | None = None
    flag_evidence: str | None = None
    original_question: str | None = None


@dataclass(frozen=True)
class RefusalRendererContext:
    """Renderer context for refusal pathways.

    flag_type: the primary flag's type enum — used for routing to the right
               refusal text. No content from the flag is permitted.
    resource:  policy-sourced redirect target (e.g. "a licensed financial adviser").
               Never derived from user input or LLM output.
    interaction_open: whether the interaction continues after this refusal.
    """
    flag_type: FlagType | None
    resource: str | None
    interaction_open: bool
    domain: str | None = None
    request_domain: str | None = None
    user_facing_template_key: str | None = None
    continuation_type: str | None = None
    reason_code: str | None = None
    rule_id: str | None = None


@dataclass(frozen=True)
class HardStopRendererContext:
    """Renderer context for hard-stop pathways.

    flag_type:        the primary flag's type enum — used for domain-appropriate
                      continuation routing. No content from the flag is permitted.
    resource:         policy-sourced redirect target. Never derived from blocked content.
    interaction_open: when True, the renderer may append a bounded follow-up question
                      inside the lawful procedural corridor. Never for ILLEGAL_INSTRUCTION
                      regardless of this flag. TARGETED_DEFAMATION depends on defamation_role
                      (e.g. UNKNOWN may leave the interaction procedurally open for clarification).
    domain:           coarse domain string ("medical", "legal", "finance", "harmful",
                      "ambiguity", or None). Used only for the domain-fallback renderer
                      when flag_type does not match a specific handler. Never carries
                      blocked content — derived from flag taxonomy only.
    defamation_role:  for TARGETED_DEFAMATION only — who the user is relative to the
                      allegation (AUTHOR / TARGET / REPORTER / UNKNOWN). Renderer-only;
                      does not affect pathway or blocking.
    """
    flag_type: FlagType | None
    resource: str | None
    interaction_open: bool = False
    domain: str | None = None
    defamation_role: DefamationRole | None = None
    pre_llm: bool = False


# Minimal flag-type → domain string mapping for fallback routing.
# This is intentionally a local bridge-layer map, not the canonical governor
# domain resolution (which handles ContextVars, rule-ids, and explicit routing).
# It exists purely so the catch-all renderer can produce a lawful domain-appropriate
# fallback when a new FlagType is added and no specific handler exists yet.
_HARD_STOP_FLAG_DOMAIN: dict[FlagType, str] = {
    FlagType.MEDICAL_DOSAGE_RECOMMENDATION:   "medical",
    FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION: "medical",
    FlagType.NUMERIC_MEDICAL_INSTRUCTION:     "medical",
    FlagType.EMERGENCY_TRIAGE_GUIDANCE:       "medical",
    FlagType.SELF_HARM_INSTRUCTION:           "medical",
    FlagType.PERSONALIZED_MEDICAL_ADVICE:     "medical",
    FlagType.PERSONALIZED_LEGAL_ADVICE:       "legal",
    FlagType.ILLEGAL_INSTRUCTION:             "harmful",
    FlagType.HAZARD_ONTOLOGY_UNAVAILABLE:     "harmful",
    FlagType.AGENCY_VIOLATION_ASSISTANCE:     "harmful",
    FlagType.TARGETED_DEFAMATION:             "harmful",
    FlagType.PERSONALIZED_FINANCIAL_ADVICE:   "finance",
    FlagType.SENSITIVE_PII_EXPOSURE:          "legal",
    FlagType.UNRESOLVED_STATE_TRANSITION:     "procedural",    # data-protection is a legal/compliance domain
    # Education / workforce / enterprise compliance domains
    FlagType.ACADEMIC_INTEGRITY_VIOLATION:    "education",
    FlagType.STUDENT_RECORD_EXPOSURE:         "education",
    FlagType.EMPLOYMENT_DISCRIMINATION_FACILITATION: "workforce",
    FlagType.EMPLOYEE_RECORD_EXPOSURE:        "workforce",
    FlagType.TRADE_SECRET_DISCLOSURE:         "enterprise",
    FlagType.INSIDER_INFORMATION_ASSISTANCE:  "enterprise",
    FlagType.PROCUREMENT_FRAUD_FACILITATION:  "enterprise",
    FlagType.PROMPT_INJECTION_ATTEMPT:        "harmful",
}


# Explicit refusal-domain fallback mapping.
# This map is intentionally narrow and structured; unknowns resolve to "general".
_DOMAIN_BY_FLAG_TYPE: dict[FlagType, str] = {
    FlagType.PERSONALIZED_FINANCIAL_ADVICE: "finance",
    FlagType.PERSONALIZED_LEGAL_ADVICE: "legal",
    FlagType.PERSONALIZED_MEDICAL_ADVICE: "medical",
    FlagType.MEDICAL_DOSAGE_RECOMMENDATION: "medical",
    FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION: "medical",
    FlagType.NUMERIC_MEDICAL_INSTRUCTION: "medical",
}


def _domain_for_refusal(ctx: RefusalRendererContext) -> str:
    """Resolve refusal renderer domain using structured sources only.

    Resolution order:
      1) explicit renderer context domain
      2) request/forensic domain
      3) explicit FlagType-to-domain map
      4) general (never finance by default)
    """
    dom = str(getattr(ctx, "domain", "") or "").strip().lower()
    if dom:
        return dom
    req_dom = str(getattr(ctx, "request_domain", "") or "").strip().lower()
    if req_dom:
        return req_dom
    flag = getattr(ctx, "flag_type", None)
    if isinstance(flag, FlagType):
        return _DOMAIN_BY_FLAG_TYPE.get(flag, "general")
    return "general"


def _refusal_flag_for_template_key(template_key: str | None) -> FlagType | None:
    key = str(template_key or "").strip().lower()
    if key == "finance.blocked.personalized_decision":
        return FlagType.PERSONALIZED_FINANCIAL_ADVICE
    if key == "legal.blocked.personalized_case_outcome":
        return FlagType.PERSONALIZED_LEGAL_ADVICE
    if key == "medical.blocked.personalized_treatment_or_dosage":
        return FlagType.PERSONALIZED_MEDICAL_ADVICE
    return None


# ── Context builders ──────────────────────────────────────────────────────────
#
# These are the only functions that may read from GovernanceDecision to produce
# renderer input. They extract only permitted fields and discard everything else.
# enforce() calls these before dispatching to any renderer.

def _build_ambiguity_context(decision: GovernanceDecision) -> AmbiguityRendererContext:
    candidates: list[str] = []
    missing_fields: list[str] = []
    for flag in decision.flags:
        if flag.candidates:
            for c in flag.candidates:
                if c not in candidates:
                    candidates.append(c)
        elif flag.entity_name and flag.entity_name not in missing_fields:
            missing_fields.append(flag.entity_name)
    primary = decision.flags[0] if decision.flags else None
    return AmbiguityRendererContext(
        candidates=tuple(candidates),
        missing_fields=tuple(missing_fields),
        interaction_open=decision.interaction_open,
        flag_type=primary.flag_type if primary else None,
        ambiguous_token=primary.entity_name if primary else None,
        flag_claim=primary.claim if primary else None,
        flag_evidence=primary.evidence if primary else None,
    )


def _build_refusal_context(decision: GovernanceDecision) -> RefusalRendererContext:
    flag_type = decision.flags[0].flag_type if decision.flags else None
    request_domain = domain_var.get(None)
    rr = decision.rule_result
    return RefusalRendererContext(
        flag_type=flag_type,
        resource=decision.resource,
        interaction_open=decision.interaction_open,
        domain=(rr.domain if rr is not None else None) or (_HARD_STOP_FLAG_DOMAIN.get(flag_type) if flag_type is not None else None),
        request_domain=request_domain,
        user_facing_template_key=rr.user_facing_template_key if rr is not None else None,
        continuation_type=rr.continuation_type if rr is not None else None,
        reason_code=rr.reason_code if rr is not None else None,
        rule_id=rr.rule_id if rr is not None else None,
    )


def _build_hard_stop_context(decision: GovernanceDecision) -> HardStopRendererContext:
    flag = decision.flags[0] if decision.flags else None
    flag_type = flag.flag_type if flag else None
    dr: DefamationRole | None = None
    if flag_type == FlagType.TARGETED_DEFAMATION and flag is not None:
        dr = (
            flag.defamation_role
            if flag.defamation_role is not None
            else DefamationRole.UNKNOWN
        )
    return HardStopRendererContext(
        flag_type=flag_type,
        resource=decision.resource,
        interaction_open=decision.interaction_open,
        domain=(
            decision.rule_result.domain
            if decision.rule_result is not None
            else _HARD_STOP_FLAG_DOMAIN.get(flag_type) if flag_type is not None else None
        ),
        defamation_role=dr,
        pre_llm=decision.pre_llm,
    )


_CONTINUATION_OFFER_TEXT: dict[str, str] = {
    "neutral_timeline": "I can help format the facts you provide into a neutral timeline to discuss with them.",
    "neutral_summary": "I can help turn your own account into a clearer summary.",
    "list_documents": "I can help list documents mentioned in your account.",
}

# GP personalized advice — same structured safe corridor for HARD_STOP (P_STOP_*)
# and refusal pathways (P_REFUSE_*), so enterprise REFUSE never surfaces legacy reformulation.
_GP_PERSONALIZED_ADVICE_FLAGS: frozenset[FlagType] = frozenset({
    FlagType.PERSONALIZED_LEGAL_ADVICE,
    FlagType.PERSONALIZED_MEDICAL_ADVICE,
    FlagType.PERSONALIZED_FINANCIAL_ADVICE,
})


def _gp_personalized_has_resource(resource: str | None) -> bool:
    """True when policy supplied an explicit escalation target for personalized corridors."""
    return bool(resource and str(resource).strip())


def _gp_personalized_safe_corridor(flag_type: FlagType, resource: str | None) -> str:
    """GP personalized-advice HARD_STOP: refusal + boundary + concrete safe corridor (static copy).

    No role-based switching, no instruction restatement, no authorized professional lanes.

    When ``resource`` is set, Reason/Action use neutral professional wording only — domain-
    specific default authority phrases must not appear alongside the explicit
    ``Next step: Contact <resource>`` line (provenance contract).
    """
    has_res = _gp_personalized_has_resource(resource)
    if flag_type == FlagType.PERSONALIZED_LEGAL_ADVICE:
        boundary = "I can't determine whether your case would succeed."
        if has_res:
            r = resource.strip().rstrip(".")
            action = f"I can help you build a neutral timeline of what happened for {r}."
        else:
            action = "I can help you build a neutral timeline of what happened for a lawyer."
        send_items = (
            "key dates",
            "what happened",
            "any warnings or notices",
            "relevant documents or messages",
        )
        safe_label = "neutral timeline"
    elif flag_type == FlagType.PERSONALIZED_MEDICAL_ADVICE:
        if has_res:
            boundary = (
                "I can't make this medical decision or give medical advice. "
                "That would require a qualified professional."
            )
            action = (
                "I can help you create a neutral symptom and event summary to take to "
                "an appropriate professional."
            )
        else:
            boundary = (
                "I can't make this medical decision or give medical advice. "
                "That would require a qualified clinician."
            )
            action = (
                "I can help you create a neutral symptom and event summary to take to a doctor, "
                "clinic, emergency department, or other qualified clinician."
            )
        send_items = (
            "symptoms",
            "when they started",
            "changes over time",
            "relevant medications or conditions",
            "urgent warning signs already present, if any",
        )
        safe_label = "symptom summary"
    elif flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE:
        boundary = "I can't determine whether you should take this financial action."
        if has_res:
            r = resource.strip().rstrip(".")
            action = f"Build a neutral financial facts summary for {r}."
        else:
            action = "Build a neutral financial facts summary for a licensed adviser."
        send_items = (
            "goal or concern",
            "relevant dates",
            "amounts involved",
            "accounts or assets",
            "risk constraints",
            "questions for the adviser",
        )
        safe_label = "neutral financial facts summary"
    else:
        raise ValueError(f"_gp_personalized_safe_corridor: unsupported flag_type {flag_type!r}")

    if flag_type == FlagType.PERSONALIZED_LEGAL_ADVICE:
        lines = [
            boundary,
            "",
            "Next step:",
            action,
            "",
            "Provide:",
            *[f"- {item}" for item in send_items],
        ]
        return "\n".join(lines)
    if flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE:
        lines = [
            boundary,
            "",
            "Next step:",
            action,
            "",
            "Provide:",
            *[f"- {item}" for item in send_items],
        ]
        return "\n".join(lines)
    lines = [
        "Outside permitted scope.",
        "",
        f"Reason: {boundary}",
        "",
        f"Action: {action}",
        "",
        "To continue, send:",
        *[f"- {item}" for item in send_items],
        "",
    ]
    if resource:
        r = resource.strip().rstrip(".")
        lines.extend([f"Next step: Contact {r}.", ""])
    lines.append(f"Status: Blocked. Safe continuation available: {safe_label}.")
    return "\n".join(lines)


def _gp_refusal_action_when_corridor_available(
    flag_type: FlagType,
    resource: str | None = None,
) -> str | None:
    """Single-line Action for REFUSE pathways when a continuation corridor is authorized."""
    has_res = _gp_personalized_has_resource(resource)
    if flag_type == FlagType.PERSONALIZED_LEGAL_ADVICE:
        if has_res:
            return (
                "I can help you create a neutral timeline of what happened so you can take it "
                "to an appropriate professional or relevant body."
            )
        return (
            "I can help you create a neutral timeline of what happened so you can take it "
            "to a lawyer, union, legal aid service, or employment tribunal."
        )
    if flag_type == FlagType.PERSONALIZED_MEDICAL_ADVICE:
        if has_res:
            return (
                "I can help you create a neutral symptom and event summary to take to "
                "an appropriate professional."
            )
        return (
            "I can help you create a neutral symptom and event summary to take to a doctor, "
            "clinic, emergency department, or other qualified clinician."
        )
    if flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE:
        if has_res:
            return (
                "I can help you create a neutral financial facts summary to take to "
                "an appropriate professional."
            )
        return (
            "I can help you create a neutral financial facts summary to take to a licensed "
            "financial adviser."
        )
    return None


def _render_allowed_continuation_offer(
    allowed_continuations: tuple[str, ...],
) -> str | None:
    """Render only Governor-authorized continuation offers."""
    lines: list[str] = []
    for capability in allowed_continuations:
        text = _CONTINUATION_OFFER_TEXT.get(capability)
        if text and text not in lines:
            lines.append(text)
    if not lines:
        return None
    return " ".join(lines)


def _reformulation_action_line(
    flag_type: FlagType | None,
    allowed_continuations: tuple[str, ...],
) -> str | None:
    """In-system reformulation the user may ask for instead of the blocked act."""
    if flag_type in _GP_PERSONALIZED_ADVICE_FLAGS:
        return None
    if _render_allowed_continuation_offer(allowed_continuations) is None:
        return None
    if flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE:
        return (
            "Ask for a neutral summary of facts already provided, "
            "without buy/sell/reallocate instructions"
        )
    return "Ask for a neutral summary or timeline instead"


def _domain_external_escalation_line(flag_type: FlagType | None, resource: str | None) -> str:
    """Outside-the-system path: use policy resource when present, else domain default."""
    if resource:
        r = resource.strip().rstrip(".")
        return f"Contact {r}."
    if flag_type == FlagType.PERSONALIZED_LEGAL_ADVICE:
        return "Contact a qualified legal adviser or a Citizens Advice-style service."
    if flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE:
        return "Contact a licensed financial adviser or regulated guidance service."
    if flag_type in (
        FlagType.PERSONALIZED_MEDICAL_ADVICE,
        FlagType.EMERGENCY_TRIAGE_GUIDANCE,
    ):
        # Avoid the substring "arm" (e.g. in "pharmacist") for urgency-implication tests.
        return "Contact a clinician or appropriate urgent care service."
    if flag_type in (
        FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
        FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
        FlagType.NUMERIC_MEDICAL_INSTRUCTION,
    ):
        return "Contact a prescribing clinician or appropriate urgent care service."
    return "Contact a qualified professional."


def _regulated_hard_stop_action_items(
    flag_type: FlagType | None,
    resource: str | None,
    allowed_continuations: tuple[str, ...],
) -> tuple[str, ...]:
    """Reformulation (when Governor offers a corridor) plus domain-specific escalation."""
    reform = _reformulation_action_line(flag_type, allowed_continuations)
    ext = _domain_external_escalation_line(flag_type, resource)
    if reform:
        return (reform, ext)
    return (ext,)


# Internal sentinel ``Flag.entity_name`` labels (checker / bridge) → governed ask copy.
_MISSING_FIELD_INTERNAL_LABELS: dict[str, str] = {
    "extraction": (
        "clearer wording so I can tell what facts or questions you mean"
    ),
    "pef": (
        "enough settled context in this conversation to answer safely"
    ),
    "agency_context_unresolved": (
        "What is the intended context and target for this plan?"
    ),
}


def _humanize_missing_field_label(raw: str) -> str:
    """Translate internal diagnostics labels out of clarification prompts."""
    key = (raw or "").strip().lower()
    return _MISSING_FIELD_INTERNAL_LABELS.get(key, raw)


def _format_governed_state(
    *,
    heading: str,
    reason: str,
    action: str | None = None,
    status: str,
    options: tuple[str, ...] = (),
    next_step: str | None = None,
    action_items: tuple[str, ...] = (),
) -> str:
    """Render non-PASS user copy as heading + reason + action + status.

    When ``action_items`` has two or more entries, renders a multi-line Action block
    with bullets (reformulation + external escalation). A single entry uses the
    same ``Action: ...`` line as a plain ``action`` string.
    """
    lines = [
        f"{heading}.",
        "",
        reason,
        "",
    ]
    if options:
        lines.extend([*tuple(f"- {opt}" for opt in options), ""])
    if next_step:
        lines.extend([f"Next step: {next_step}", ""])

    items = action_items if action_items else ((action,) if action else ())
    if not items:
        if status == "Request blocked before model call":
            lines.extend([f"{status}."])
        else:
            lines.extend([f"Status: {status}."])
        return "\n".join(lines)

    def _norm_period(t: str) -> str:
        t = (t or "").strip()
        return t if t.endswith((".", "?", "!")) else f"{t}."

    if len(items) >= 2:
        lines.extend([
            "Action:",
            *[f"- {_norm_period(it)}" for it in items],
        ])
        if status == "Request blocked before model call":
            lines.extend([f"{status}."])
        else:
            lines.extend([f"Status: {status}."])
    else:
        action_text = _norm_period(items[0])
        lines.extend([
            f"Action: {action_text}",
        ])
        if status == "Request blocked before model call":
            lines.extend([f"{status}."])
        else:
            lines.extend([f"Status: {status}."])
    return "\n".join(lines)


# ── Pathway renderers ─────────────────────────────────────────────────────────
#
# Each renderer takes its typed context and returns a user-visible string.
# Renderers must not:
#   - call any LLM
#   - generate ad hoc prose
#   - recover, imply, suggest, or narratively complete a blocked determination
#   - produce workaround-adjacent language for ILLEGAL_INSTRUCTION
#   - access anything outside their typed context
#
# All text is policy-authorized and static except for the resource field
# and candidate labels, both of which are themselves policy/PEF-sourced.

def _render_ask_disambiguate(ctx: AmbiguityRendererContext) -> str:
    """P_ASK_DISAMBIGUATE: surface admissible PEF candidates, keep interaction open."""
    if ctx.flag_type == FlagType.UNRESOLVED_REFERENT:
        return compose_unresolved_referent_governed_message(
            ambiguous_tokens=[ctx.ambiguous_token] if ctx.ambiguous_token else [],
            candidate_entities=list(ctx.candidates),
            original_question=ctx.original_question,
            flag_claim=ctx.flag_claim,
            flag_evidence=ctx.flag_evidence,
        )
    if ctx.candidates:
        return _format_governed_state(
            heading="Clarification required",
            reason=(
                ctx.flag_claim.strip()
                if ctx.flag_claim and ctx.flag_claim.strip()
                else "The available evidence does not identify a unique referent."
            ),
            options=ctx.candidates,
            action="Choose one option to continue",
            status="Waiting for clarification",
        )
    return _format_governed_state(
        heading="More information required",
        reason=(
            ctx.flag_claim.strip()
            if ctx.flag_claim and ctx.flag_claim.strip()
            else "The available evidence does not identify a unique referent."
        ),
        action="Choose one option to continue",
        status="Waiting for clarification",
    )


def _render_ask_missing_fact(ctx: AmbiguityRendererContext) -> str:
    """P_ASK_MISSING_FACT: name the missing fields, keep interaction open."""
    if ctx.flag_type in (
        FlagType.EXTRACTION_EMPTY,
        FlagType.UPSTREAM_INSUFFICIENT_CONTEXT,
    ):
        missing = ctx.missing_fields[0] if ctx.missing_fields else None
        return compose_insufficient_structure_message(
            flag_type=ctx.flag_type,
            flag_claim=ctx.flag_claim,
            flag_evidence=ctx.flag_evidence,
            missing_field=missing,
        )
    if ctx.missing_fields:
        if (
            len(ctx.missing_fields) == 1
            and ctx.missing_fields[0] == "agency_context_unresolved"
        ):
            return _ensure_canonical_clarification_action_line(
                "More information required.\n\n"
                "What is the intended context and target for this plan?\n\n"
                "Status: Waiting for clarification.\n"
                "Action: Choose one option to continue."
            )
        labels = tuple(_humanize_missing_field_label(f) for f in ctx.missing_fields)
        reason = (
            ctx.flag_claim.strip()
            if ctx.flag_claim and ctx.flag_claim.strip()
            else "The available record does not include enough detail to continue safely."
        )
        return _format_governed_state(
            heading="More information required",
            reason=reason,
            options=labels,
            action="Choose one option to continue",
            status="Waiting for clarification",
        )
    return compose_insufficient_structure_message(
        flag_type=ctx.flag_type,
        flag_claim=ctx.flag_claim,
        flag_evidence=ctx.flag_evidence,
        missing_field=ctx.missing_fields[0] if ctx.missing_fields else None,
    )


def _render_refuse_explain_redirect(ctx: RefusalRendererContext) -> str:
    """P_REFUSE_EXPLAIN_REDIRECT: refusal + optional policy-sourced resource redirect."""
    rr_flag = _refusal_flag_for_template_key(ctx.user_facing_template_key)
    if rr_flag is not None:
        return _gp_personalized_safe_corridor(rr_flag, ctx.resource)
    dom = _domain_for_refusal(ctx)
    if dom == "finance":
        return _gp_personalized_safe_corridor(FlagType.PERSONALIZED_FINANCIAL_ADVICE, ctx.resource)
    if dom == "legal":
        return _gp_personalized_safe_corridor(FlagType.PERSONALIZED_LEGAL_ADVICE, ctx.resource)
    if dom in {"medical", "healthcare"}:
        return _gp_personalized_safe_corridor(FlagType.PERSONALIZED_MEDICAL_ADVICE, ctx.resource)
    if ctx.flag_type in _GP_PERSONALIZED_ADVICE_FLAGS:
        return _gp_personalized_safe_corridor(ctx.flag_type, ctx.resource)
    reason = _hard_stop_reason(ctx.flag_type)
    action = _default_action_for_flag(ctx.flag_type, ctx.resource)
    return _format_governed_state(
        heading="Cannot provide that",
        reason=reason,
        action=action,
        status="Refused",
    )


def _render_refuse_escalate_pro(ctx: RefusalRendererContext) -> str:
    """P_REFUSE_ESCALATE_PRO: refusal + professional referral."""
    rr_flag = _refusal_flag_for_template_key(ctx.user_facing_template_key)
    if rr_flag is not None:
        return _gp_personalized_safe_corridor(rr_flag, ctx.resource)
    dom = _domain_for_refusal(ctx)
    if dom == "finance":
        return _gp_personalized_safe_corridor(FlagType.PERSONALIZED_FINANCIAL_ADVICE, ctx.resource)
    if dom == "legal":
        return _gp_personalized_safe_corridor(FlagType.PERSONALIZED_LEGAL_ADVICE, ctx.resource)
    if dom in {"medical", "healthcare"}:
        return _gp_personalized_safe_corridor(FlagType.PERSONALIZED_MEDICAL_ADVICE, ctx.resource)
    if ctx.flag_type in _GP_PERSONALIZED_ADVICE_FLAGS:
        return _gp_personalized_safe_corridor(ctx.flag_type, ctx.resource)
    reason = _hard_stop_reason(ctx.flag_type)
    action = _default_action_for_flag(ctx.flag_type, ctx.resource)
    return _format_governed_state(
        heading="Outside permitted scope",
        reason=reason,
        action=action,
        status="Refused",
    )


def _render_handoff_summary(ctx: RefusalRendererContext) -> str:
    """P_HANDOFF_SUMMARY: boundary notice, offer to summarise or redirect."""
    return _format_governed_state(
        heading="Cannot provide that",
        reason="This request asks for a decision the system is not permitted to make.",
        action="Ask for a neutral summary or timeline instead",
        status="Refused",
    )


def _render_stop_terminal(
    ctx: HardStopRendererContext,
    allowed_continuations: tuple[str, ...] = (),
) -> str:
    """P_STOP_TERMINAL: hard stop with domain-appropriate continuation."""
    return _hard_stop_text(
        ctx.flag_type,
        ctx.resource,
        ctx.interaction_open,
        ctx.domain,
        allowed_continuations,
        ctx.defamation_role,
        ctx.pre_llm,
    )


def _render_stop_forensic(
    ctx: HardStopRendererContext,
    allowed_continuations: tuple[str, ...] = (),
) -> str:
    """P_STOP_FORENSIC: hard stop; forensic obligations attach via audit layer."""
    return _hard_stop_text(
        ctx.flag_type,
        ctx.resource,
        ctx.interaction_open,
        ctx.domain,
        allowed_continuations,
        ctx.defamation_role,
        ctx.pre_llm,
    )


def _render_stop_escalate(ctx: HardStopRendererContext) -> str:
    """P_STOP_ESCALATE: terminate the prohibited determination; emit immediate escalation.

    Legacy pathway — retained for BuiltinBridge and backward compatibility.
    New code should use P_STOP_ESCALATE_EMERGENCY or P_STOP_SUPPORTIVE_DEESCALATE.
    """
    return _format_governed_state(
        heading="Request blocked",
        reason="This request could produce unsafe instructions.",
        action=_default_action_for_flag(ctx.flag_type, ctx.resource),
        status=_stop_status(ctx.pre_llm),
        next_step=(f"Use {ctx.resource}." if ctx.resource else None),
    )


def _render_stop_escalate_emergency(ctx: HardStopRendererContext) -> str:
    """P_STOP_ESCALATE_EMERGENCY: physical medical emergency — immediate, unconditional escalation.

    Used when the system detects emergency triage content (cardiac, respiratory, etc.).
    The conversation is closed (interaction_open=False from policy). The renderer:
      - States inability without conditional framing ("if this is a medical emergency")
      - Directs immediately to emergency care — no assessment, no ranking, no delay suggestion
      - Does not offer a follow-up question (this is a physical emergency, not a dialogue)
    """
    return _format_governed_state(
        heading="Request blocked",
        reason="This request could produce unsafe instructions.",
        action=_default_action_for_flag(ctx.flag_type, ctx.resource),
        status=_stop_status(ctx.pre_llm),
        next_step=(f"Use {ctx.resource}." if ctx.resource else None),
    )


def _render_stop_supportive_deescalate(ctx: HardStopRendererContext) -> str:
    """P_STOP_SUPPORTIVE_DEESCALATE: psychological crisis — supportive refusal with open conversation.

    Used when the system detects self-harm or suicide content. The conversation
    remains open (interaction_open=True from policy). The renderer:
      - Blocks the determination unconditionally (no method, no assessment)
      - Acknowledges the person without clinical distance
      - Directs to crisis support — not to a GP or ambulance (wrong escalation target)
      - When ``interaction_open`` is True, appends the open-dialogue line (contractual)
    """
    base = _format_governed_state(
        heading="Request blocked",
        reason="This request could produce unsafe instructions.",
        action=_default_action_for_flag(ctx.flag_type, ctx.resource),
        status=_stop_status(ctx.pre_llm),
        next_step=(
            f"Use {ctx.resource}."
            if ctx.resource
            else "Use a crisis support service."
        ),
    )
    if ctx.interaction_open:
        return f"{base}\n\nI'm here if you want to talk."
    return base


def _render_stop_redirect_qualified(
    ctx: HardStopRendererContext,
    allowed_continuations: tuple[str, ...] = (),
) -> str:
    """P_STOP_REDIRECT_QUALIFIED: refusal with lawful professional/resource redirect.

    Used for flag classes that carry an appropriate redirect target (dosage, personalized
    medical/legal/financial advice). The refusal is clean and the redirect is surfaced.
    The conversation may remain open (per policy's interaction_open field).
    """
    return _hard_stop_text(
        ctx.flag_type,
        ctx.resource,
        ctx.interaction_open,
        ctx.domain,
        allowed_continuations,
        ctx.defamation_role,
        ctx.pre_llm,
    )


def _render_stop_refuse_clean(
    ctx: HardStopRendererContext,
    allowed_continuations: tuple[str, ...] = (),
) -> str:
    """P_STOP_REFUSE_CLEAN: refusal without redirect or workaround adjacency.

    Used for flag classes where any redirect would create an exploit surface
    (e.g. ILLEGAL_INSTRUCTION). No escalation target is surfaced. The response
    is a flat, auditable refusal. The policy's interaction_open field is respected
    so that a boundary explanation can be offered without a redirect.

    TARGETED_DEFAMATION + TARGET: policy escalation target may be surfaced as a
    legal handoff (same resource field — renderer-only exception for that role).
    """
    eff_resource: str | None = None
    if (
        ctx.flag_type == FlagType.TARGETED_DEFAMATION
        and ctx.defamation_role == DefamationRole.TARGET
    ):
        eff_resource = ctx.resource
    return _hard_stop_text(
        ctx.flag_type,
        eff_resource,
        ctx.interaction_open,
        ctx.domain,
        allowed_continuations,
        ctx.defamation_role,
        ctx.pre_llm,
    )


# Procedural governor gap — do not frame as “user omitted detail” (misleading on
# informational prompts where the model answer tripped structural workflow checks).
_UNRESOLVED_STATE_TRANSITION_REASON = (
    "This reply asserts a workflow status that the current session state does "
    "not yet support as settled."
)
_UNRESOLVED_STATE_TRANSITION_ACTION = (
    "Revise the reply to align with the session's recorded workflow state."
)

# Post-verify epistemic flags (FORCE_REVISE / REFUSE) — not disambiguation holds.
# Do not use "Choose one option" here; no candidate menu is presented to the user.
_EPISTEMIC_VERIFY_REFUSAL_REASON = (
    "The draft answer could not be verified against the session record."
)
_EPISTEMIC_VERIFY_REFUSAL_ACTION = (
    "Rephrase the question or ask for a narrower, quote-backed answer."
)

_EPISTEMIC_VERIFY_FLAG_TYPES = frozenset(
    {
        FlagType.UNSUPPORTED_ATTRIBUTE,
        FlagType.UNSUPPORTED_EVENT,
        FlagType.UNBOUND_ENTITY,
        FlagType.UNVERIFIED_FACT_ASSERTION,
        FlagType.TIME_SMEAR,
        FlagType.CONTRADICTED_FACT,
    }
)


def _hard_stop_reason(flag_type: FlagType | None) -> str:
    if flag_type == FlagType.UNRESOLVED_STATE_TRANSITION:
        return _UNRESOLVED_STATE_TRANSITION_REASON
    if flag_type in _EPISTEMIC_VERIFY_FLAG_TYPES:
        if flag_type == FlagType.CONTRADICTED_FACT:
            return (
                "The draft answer conflicted with a fact already on the session record."
            )
        return _EPISTEMIC_VERIFY_REFUSAL_REASON
    if flag_type in (
        FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
        FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
        FlagType.NUMERIC_MEDICAL_INSTRUCTION,
        FlagType.SELF_HARM_INSTRUCTION,
    ):
        return "This request could produce unsafe instructions."
    if flag_type in (
        FlagType.PERSONALIZED_MEDICAL_ADVICE,
        FlagType.EMERGENCY_TRIAGE_GUIDANCE,
    ):
        return "This request requires qualified professional judgment."
    if flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE:
        return (
            "This request asks for a financial decision the system is not permitted to make."
        )
    if flag_type in (
        FlagType.PERSONALIZED_LEGAL_ADVICE,
        FlagType.ILLEGAL_INSTRUCTION,
        FlagType.AGENCY_VIOLATION_ASSISTANCE,
        FlagType.TARGETED_DEFAMATION,
        FlagType.SENSITIVE_PII_EXPOSURE,
    ):
        return "This request asks for a decision the system is not permitted to make."
    return "This request needs a missing detail."


def _default_action_for_flag(
    flag_type: FlagType | None,
    resource: str | None,
    allowed_continuations: tuple[str, ...] = (),
) -> str:
    if flag_type == FlagType.UNRESOLVED_STATE_TRANSITION:
        return _UNRESOLVED_STATE_TRANSITION_ACTION
    if flag_type in (
        FlagType.PERSONALIZED_LEGAL_ADVICE,
        FlagType.PERSONALIZED_FINANCIAL_ADVICE,
    ):
        offer = _render_allowed_continuation_offer(allowed_continuations)
        if offer is not None:
            line = _gp_refusal_action_when_corridor_available(flag_type, resource)
            if line:
                return line
        return _domain_external_escalation_line(flag_type, resource)
    if flag_type in (
        FlagType.PERSONALIZED_MEDICAL_ADVICE,
        FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
        FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
        FlagType.NUMERIC_MEDICAL_INSTRUCTION,
        FlagType.EMERGENCY_TRIAGE_GUIDANCE,
        FlagType.SELF_HARM_INSTRUCTION,
    ):
        if flag_type == FlagType.PERSONALIZED_MEDICAL_ADVICE:
            offer = _render_allowed_continuation_offer(allowed_continuations)
            if offer is not None:
                line = _gp_refusal_action_when_corridor_available(flag_type, resource)
                if line:
                    return line
        return _domain_external_escalation_line(flag_type, resource)
    if flag_type in (
        FlagType.ILLEGAL_INSTRUCTION,
        FlagType.TARGETED_DEFAMATION,
        FlagType.SENSITIVE_PII_EXPOSURE,
    ):
        if resource:
            return "Contact a qualified professional."
        return "Reframe as a general educational question."
    if resource:
        return "Contact a qualified professional."
    if flag_type in _EPISTEMIC_VERIFY_FLAG_TYPES:
        return _EPISTEMIC_VERIFY_REFUSAL_ACTION
    return "Rephrase the question or try a narrower request."


def _stop_status(pre_llm: bool) -> str:
    if pre_llm:
        return "Request blocked before model call"
    return "Blocked after model generation"


def _hard_stop_text(
    flag_type: FlagType | None,
    resource: str | None,
    interaction_open: bool = False,
    domain: str | None = None,
    allowed_continuations: tuple[str, ...] = (),
    defamation_role: DefamationRole | None = None,
    pre_llm: bool = False,
) -> str:
    """Select the lawful user-visible continuation for a hard-stop flag class.

    When interaction_open=True, the renderer appends a bounded follow-up question
    that keeps the conversation inside the lawful procedural corridor without
    recovering any part of the prohibited determination.

    ILLEGAL_INSTRUCTION and SENSITIVE_PII_EXPOSURE are clean stops —
    interaction_open is ignored for those flag classes. TARGETED_DEFAMATION copy
    depends on ``defamation_role`` (see ``DefamationRole``); UNKNOWN may ask for
    clarification instead of a fixed continuation.

    Unknown/future flag types fall through to a domain-appropriate generic
    fallback (medical / legal / finance / harmful / None). The fallback always
    produces a lawful, auditable response — never silent failure.

    Branches on flag_type (enum value only — no content).
    resource and domain are policy/taxonomy-sourced. No field carries blocked content.
    """
    if flag_type == FlagType.EXTRACTION_FAILED:
        return _format_governed_state(
            heading="More information required",
            reason="This request needs a missing detail.",
            action="Choose one option to continue",
            status="Waiting for clarification",
        )

    if flag_type == FlagType.EXTRACTION_EMPTY:
        return _format_governed_state(
            heading="More information required",
            reason="This request needs a missing detail.",
            action="Choose one option to continue",
            status="Waiting for clarification" if interaction_open else _stop_status(pre_llm),
        )

    if flag_type in (
        FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
        FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
        FlagType.NUMERIC_MEDICAL_INSTRUCTION,
    ):
        items = _regulated_hard_stop_action_items(flag_type, resource, ())
        return _format_governed_state(
            heading="Request blocked",
            reason="This request could produce unsafe instructions.",
            action_items=items,
            status=_stop_status(pre_llm),
        )

    if flag_type == FlagType.EMERGENCY_TRIAGE_GUIDANCE:
        items = _regulated_hard_stop_action_items(
            flag_type, resource, allowed_continuations
        )
        return _format_governed_state(
            heading="Request blocked",
            reason="This request requires qualified professional judgment.",
            action_items=items,
            status=_stop_status(pre_llm),
            next_step=(f"Use {resource} immediately." if resource else None),
        )

    if flag_type == FlagType.SELF_HARM_INSTRUCTION:
        return _format_governed_state(
            heading="Request blocked",
            reason="I can't help with self-harm instructions because this request could produce unsafe instructions.",
            action="Contact a crisis support line immediately.",
            status=_stop_status(pre_llm),
            next_step=(
                f"Use {resource}. You can continue here if you want to talk about what's going on."
                if interaction_open and resource
                else (
                    "Use a crisis support service. You can continue here if you want to talk about what's going on."
                    if interaction_open
                    else (f"Use {resource}." if resource else "Use a crisis support service.")
                )
            ),
        )

    if flag_type == FlagType.PERSONALIZED_MEDICAL_ADVICE:
        return _gp_personalized_safe_corridor(flag_type, resource)

    if flag_type == FlagType.PERSONALIZED_LEGAL_ADVICE:
        return _gp_personalized_safe_corridor(flag_type, resource)

    if flag_type == FlagType.UNRESOLVED_STATE_TRANSITION:
        return _format_governed_state(
            heading="More information required",
            reason=_UNRESOLVED_STATE_TRANSITION_REASON,
            action=_UNRESOLVED_STATE_TRANSITION_ACTION,
            status="Waiting for clarification" if interaction_open else "Blocked",
        )

    if flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE:
        return _gp_personalized_safe_corridor(flag_type, resource)

    if flag_type == FlagType.ILLEGAL_INSTRUCTION:
        return _format_governed_state(
            heading="Request blocked",
            reason="This request asks for a decision the system is not permitted to make.",
            action="Reframe as a general educational question",
            status=_stop_status(pre_llm),
        )

    if flag_type == FlagType.AGENCY_VIOLATION_ASSISTANCE:
        return _format_governed_state(
            heading="Request blocked",
            reason="I can't provide tactics for manipulating or coercing people.",
            action=(
                "I can help with a neutral overview of manipulation, coercive control, "
                "warning signs, consent, or ethical persuasion"
            ),
            status=_stop_status(pre_llm),
        )

    if flag_type == FlagType.PROMPT_INJECTION_ATTEMPT:
        return _format_governed_state(
            heading="Request blocked",
            reason="Attempted authority or instruction override.",
            action="Reframe as a question within normal operating parameters.",
            status=_stop_status(pre_llm),
        )

    if flag_type == FlagType.TARGETED_DEFAMATION:
        role = defamation_role if defamation_role is not None else DefamationRole.UNKNOWN
        reason = "This request asks for a decision the system is not permitted to make."
        if role == DefamationRole.AUTHOR:
            return _format_governed_state(
                heading="Request blocked",
                reason=reason,
                action="Reframe as a general educational question",
                status=_stop_status(pre_llm),
            )
        if role == DefamationRole.REPORTER:
            return _format_governed_state(
                heading="Request blocked",
                reason=reason,
                action="Use an appropriate reporting channel for conduct concerns.",
                status=_stop_status(pre_llm),
                next_step="Choose a lawful reporting channel.",
            )
        if role == DefamationRole.TARGET:
            if resource:
                return _format_governed_state(
                    heading="Request blocked",
                    reason=reason,
                    status=_stop_status(pre_llm),
                    next_step=f"Use {resource}.",
                )
            return _format_governed_state(
                heading="Request blocked",
                reason=reason,
                status=_stop_status(pre_llm),
                next_step=(
                    "Contact a qualified legal adviser or a Citizens Advice-style service."
                ),
            )
        # UNKNOWN — ask which role applies instead of picking a continuation.
        return _format_governed_state(
            heading="Request blocked",
            reason=reason,
            options=(
                "Target — harmful statements are about you or your reputation",
                "Reporter — you are reporting another party's statements",
                "Author — you refer to your own words or writing",
            ),
            action="Reply with which case applies",
            status=(
                "Waiting for clarification"
                if interaction_open
                else _stop_status(pre_llm)
            ),
        )

    if flag_type == FlagType.SENSITIVE_PII_EXPOSURE:
        return _format_governed_state(
            heading="Request blocked",
            reason="This request asks for a decision the system is not permitted to make.",
            action="Follow data-governance boundaries and use approved support channels.",
            status=_stop_status(pre_llm),
            next_step=(
                f"Use {resource}."
                if resource
                else "Use the appropriate support team for data-governance requests."
            ),
        )

    if flag_type == FlagType.ACADEMIC_INTEGRITY_VIOLATION:
        return _format_governed_state(
            heading="Request blocked",
            reason="This request asks for assistance that would violate academic integrity policies.",
            action="Reframe as a general learning or tutoring question.",
            status=_stop_status(pre_llm),
        )

    if flag_type == FlagType.STUDENT_RECORD_EXPOSURE:
        return _format_governed_state(
            heading="Request blocked",
            reason="This request involves student record data governed by privacy regulations.",
            action="Use approved data-governance channels for student record access.",
            status=_stop_status(pre_llm),
        )

    if flag_type == FlagType.EMPLOYMENT_DISCRIMINATION_FACILITATION:
        return _format_governed_state(
            heading="Request blocked",
            reason="This request asks for assistance that would facilitate unlawful employment discrimination.",
            action="Reframe as a lawful hiring criteria question.",
            status=_stop_status(pre_llm),
        )

    if flag_type == FlagType.EMPLOYEE_RECORD_EXPOSURE:
        return _format_governed_state(
            heading="Request blocked",
            reason="This request involves confidential employee record data.",
            action="Use approved HR data-governance channels.",
            status=_stop_status(pre_llm),
        )

    if flag_type == FlagType.TRADE_SECRET_DISCLOSURE:
        return _format_governed_state(
            heading="Request blocked",
            reason="This request asks for assistance that would involve disclosing trade secrets.",
            action="Consult legal counsel regarding information-sharing obligations.",
            status=_stop_status(pre_llm),
        )

    if flag_type == FlagType.INSIDER_INFORMATION_ASSISTANCE:
        return _format_governed_state(
            heading="Request blocked",
            reason="This request involves material non-public information.",
            action="Consult legal counsel before acting on this information.",
            status=_stop_status(pre_llm),
        )

    if flag_type == FlagType.PROCUREMENT_FRAUD_FACILITATION:
        return _format_governed_state(
            heading="Request blocked",
            reason="This request asks for assistance that would facilitate procurement fraud.",
            action="Use approved procurement and conflict-of-interest channels.",
            status=_stop_status(pre_llm),
        )

    if flag_type == FlagType.PERSONALIZED_ACADEMIC_ADVICE:
        return _format_governed_state(
            heading="Request blocked",
            reason="I can't assess your specific academic chances or make admissions predictions.",
            action="I can help you build a neutral factual summary for an academic adviser.",
            status=_stop_status(pre_llm),
        )

    if flag_type == FlagType.PERSONALIZED_EMPLOYMENT_ADVICE:
        return _format_governed_state(
            heading="Request blocked",
            reason="I can't advise on the merits of your individual employment claim.",
            action="I can help you build a neutral timeline of events for an employment lawyer.",
            status=_stop_status(pre_llm),
        )

    if flag_type == FlagType.PERSONALIZED_BUSINESS_STRATEGY_ADVICE:
        return _format_governed_state(
            heading="Request blocked",
            reason="I can't provide strategic recommendations for your specific organisation.",
            action="I can help you build a neutral facts summary for a business adviser.",
            status=_stop_status(pre_llm),
        )

    # ── Domain-fallback ladder (Tier 2) ───────────────────────────────────────
    # Fires when the flag type has no specific handler above.
    # Tier 1: specific flag handler (above)
    # Tier 2: domain-appropriate REFUSE-family continuation (below)
    # Tier 3: harmful-domain STOP_TERMINAL
    # Tier 4: final clean stop
    #
    # Medical / legal / finance unknown → REFUSE_ESCALATE_PRO or
    # REFUSE_EXPLAIN_REDIRECT shape. Closes commitment. Keeps interaction
    # open procedurally. The Governor's role is to constrain, explain,
    # summarise, route, or escalate — not to be an unusually articulate bouncer.
    # Only intrinsically hostile domains get a true terminal stop.
    return _domain_fallback_continuation(
        domain,
        resource,
        interaction_open,
        allowed_continuations,
    )


def _domain_fallback_continuation(
    domain: str | None,
    resource: str | None,
    interaction_open: bool,
    allowed_continuations: tuple[str, ...] = (),
) -> str:
    """Tier-2 fallback: domain-appropriate REFUSE-family continuation.

    Called when flag_type has no specific handler in _hard_stop_text.
    Never called for flag types with specific handlers.

    Fallback ladder:
      medical  → REFUSE_ESCALATE_PRO shape  (redirect to professional + optional prep question)
      legal    → REFUSE_EXPLAIN_REDIRECT shape (boundary + redirect + optional procedural help)
      finance  → REFUSE_EXPLAIN_REDIRECT shape (boundary + redirect + optional prep question)
      harmful  → STOP_TERMINAL shape (clean stop — intrinsically hostile domain)
      None     → final clean stop

    Commitment is always closed. interaction_open governs follow-up question only.
    """
    if domain == "medical":
        return _gp_personalized_safe_corridor(FlagType.PERSONALIZED_MEDICAL_ADVICE, resource)

    if domain == "legal":
        return _gp_personalized_safe_corridor(FlagType.PERSONALIZED_LEGAL_ADVICE, resource)

    if domain == "finance":
        return _gp_personalized_safe_corridor(FlagType.PERSONALIZED_FINANCIAL_ADVICE, resource)

    # Tier 3: harmful domain (unknown illegal / exploitative / defamatory class)
    # → STOP_TERMINAL shape. No redirect. No follow-up. No workaround adjacency.
    # This path is an error-containment state — it signals that a hostile-class
    # flag type reached the system without a specific handler. Audit loudly.
    if domain == "harmful":
        _fallback_sentinel.reason = "unmapped_hostile_flag"
        return _format_governed_state(
            heading="Request blocked",
            reason="This request asks for a decision the system is not permitted to make.",
            action="Reframe as a general educational question.",
            status=_stop_status(pre_llm=False),
        )

    # Tier 4: final clean stop — domain unknown, no safe corridor available.
    # This path signals a malformed decision: a HARD_STOP with no flags,
    # or a flag type that produced no domain signal. Audit loudly.
    _fallback_sentinel.reason = "malformed_decision"
    return _format_governed_state(
        heading="More information required",
        reason="This request needs a missing detail.",
        action="Choose one option to continue",
        status="Blocked",
    )


# ── Pathway dispatch ──────────────────────────────────────────────────────────

_AMBIGUITY_PATHWAYS = frozenset({"P_ASK_DISAMBIGUATE", "P_ASK_MISSING_FACT"})
_REFUSAL_PATHWAYS = frozenset({
    "P_REFUSE_EXPLAIN_REDIRECT", "P_REFUSE_ESCALATE_PRO", "P_HANDOFF_SUMMARY",
})
_HARD_STOP_PATHWAYS = frozenset({"P_STOP_TERMINAL", "P_STOP_FORENSIC"})
_ESCALATE_PATHWAYS = frozenset({"P_STOP_ESCALATE", "P_STOP_ESCALATE_EMERGENCY"})
_SUPPORTIVE_PATHWAYS = frozenset({"P_STOP_SUPPORTIVE_DEESCALATE"})
_REDIRECT_PATHWAYS = frozenset({"P_STOP_REDIRECT_QUALIFIED"})
_CLEAN_STOP_PATHWAYS = frozenset({"P_STOP_REFUSE_CLEAN"})

# Union of all pathway_id strings that map to typed renderers inside enforce().
# Used by CanonicalScannerGateBridge to repair missing / drifted pathway wiring
# before intervene()/enforce() — avoids Tier-3/4 domain fallback when domain hints
# are absent (e.g. empty flags + unknown pathway → malformed_decision prose).
PATHWAY_IDS_WITH_TYPED_RENDERER: frozenset[str] = (
    _AMBIGUITY_PATHWAYS
    | _REFUSAL_PATHWAYS
    | _HARD_STOP_PATHWAYS
    | _ESCALATE_PATHWAYS
    | _SUPPORTIVE_PATHWAYS
    | _REDIRECT_PATHWAYS
    | _CLEAN_STOP_PATHWAYS
)

# BuiltinBridge default pathway_id mapping.
# PASS and SOFT_CORRECT are handled by the early gate in enforce() — no pathway needed.
# All other actions map to the most-specific canonical pathway available in the matrix.
_ACTION_TO_PATHWAY: dict[InterventionAction, str] = {
    InterventionAction.CONTAIN:      "P_ASK_DISAMBIGUATE",
    InterventionAction.FORCE_REVISE: "P_REFUSE_EXPLAIN_REDIRECT",
    InterventionAction.HARD_STOP:    "P_STOP_TERMINAL",
}

# Flag-class-specific pathway and interaction_open overrides for HARD_STOP.
# Mirrors the continuation_matrix entries for BuiltinBridge (which does not use the
# full ContinuationMatrix lookup).
# (pathway_id, interaction_open)
_FLAGTYPE_HARD_STOP_OVERRIDE: dict[FlagType, tuple[str, bool]] = {
    FlagType.SELF_HARM_INSTRUCTION:            ("P_STOP_SUPPORTIVE_DEESCALATE", True),
    FlagType.MEDICAL_DOSAGE_RECOMMENDATION:    ("P_STOP_REDIRECT_QUALIFIED",    True),
    FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION:  ("P_STOP_REDIRECT_QUALIFIED",    True),
    FlagType.NUMERIC_MEDICAL_INSTRUCTION:      ("P_STOP_REDIRECT_QUALIFIED",    True),
    FlagType.PERSONALIZED_MEDICAL_ADVICE:      ("P_STOP_REDIRECT_QUALIFIED",    True),
    FlagType.PERSONALIZED_LEGAL_ADVICE:        ("P_STOP_REDIRECT_QUALIFIED",    True),
    FlagType.PERSONALIZED_FINANCIAL_ADVICE:    ("P_STOP_REDIRECT_QUALIFIED",    True),
    FlagType.PERSONALIZED_ACADEMIC_ADVICE:     ("P_STOP_REDIRECT_QUALIFIED",    True),
    FlagType.PERSONALIZED_EMPLOYMENT_ADVICE:   ("P_STOP_REDIRECT_QUALIFIED",    True),
    FlagType.PERSONALIZED_BUSINESS_STRATEGY_ADVICE: ("P_STOP_REDIRECT_QUALIFIED", True),
    FlagType.EMERGENCY_TRIAGE_GUIDANCE:        ("P_STOP_ESCALATE_EMERGENCY",    False),
    FlagType.ILLEGAL_INSTRUCTION:              ("P_STOP_REFUSE_CLEAN",          True),
    FlagType.TARGETED_DEFAMATION:              ("P_STOP_REFUSE_CLEAN",          True),
    FlagType.VIOLENT_CRIMINAL_INTENT:          ("P_STOP_REFUSE_CLEAN",          False),
    # Education / workforce / enterprise compliance vetoes
    FlagType.ACADEMIC_INTEGRITY_VIOLATION:     ("P_STOP_REFUSE_CLEAN",          True),
    FlagType.STUDENT_RECORD_EXPOSURE:          ("P_STOP_REFUSE_CLEAN",          False),
    FlagType.EMPLOYMENT_DISCRIMINATION_FACILITATION: ("P_STOP_REFUSE_CLEAN",    False),
    FlagType.EMPLOYEE_RECORD_EXPOSURE:         ("P_STOP_REFUSE_CLEAN",          False),
    FlagType.TRADE_SECRET_DISCLOSURE:          ("P_STOP_REFUSE_CLEAN",          False),
    FlagType.INSIDER_INFORMATION_ASSISTANCE:   ("P_STOP_REFUSE_CLEAN",          False),
    FlagType.PROCUREMENT_FRAUD_FACILITATION:   ("P_STOP_REFUSE_CLEAN",          False),
    FlagType.PROMPT_INJECTION_ATTEMPT:         ("P_STOP_REFUSE_CLEAN",          False),
}

# Referent/comparand ambiguity — canonical P_ASK_DISAMBIGUATE hold (all InterventionPolicy bridges).
_ASK_DISAMBIGUATE_FLAG_TYPES = frozenset({
    FlagType.UNRESOLVED_REFERENT,
    FlagType.UNRESOLVED_COMPARAND,
})


def apply_intervention_policy_pathway_fields(
    decision: GovernanceDecision,
    *,
    flags: list[Flag],
    action: InterventionAction,
    rule: PolicyRule | None = None,
) -> None:
    """Set pathway_id, interaction_open, and commitment_closed for BuiltinBridge / scanner decide."""
    pathway_id = decision.pathway_id or _ACTION_TO_PATHWAY.get(action)
    interaction_open = decision.interaction_open
    commitment_closed = decision.commitment_closed

    if action == InterventionAction.CONTAIN:
        if any(f.flag_type == FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED for f in flags):
            pathway_id = "P_ASK_MISSING_FACT"
            interaction_open = True
            commitment_closed = True
        elif any(f.flag_type == FlagType.UPSTREAM_INSUFFICIENT_CONTEXT for f in flags):
            pathway_id = "P_ASK_MISSING_FACT"
            decision.output_mode = "clarification_request"
            interaction_open = True
            commitment_closed = True
        elif any(f.flag_type == FlagType.UNCLASSIFIED_CONSEQUENCE_INTENT for f in flags):
            pathway_id = "P_ASK_MISSING_FACT"
            decision.output_mode = "clarification_request"
            interaction_open = True
            commitment_closed = True
        elif any(f.flag_type in _ASK_DISAMBIGUATE_FLAG_TYPES for f in flags):
            pathway_id = "P_ASK_DISAMBIGUATE"
            interaction_open = True
            commitment_closed = True
    elif action == InterventionAction.HARD_STOP and rule is not None:
        override = _FLAGTYPE_HARD_STOP_OVERRIDE.get(rule.flag_type)
        if override is not None:
            pathway_id, interaction_open = override

    decision.pathway_id = pathway_id
    decision.interaction_open = interaction_open
    decision.commitment_closed = commitment_closed


def enforce(decision: GovernanceDecision, model_output: str) -> str:
    """Deterministic enforcement: (policy_decision, model_output) -> final response.

    No LLM re-prompting. No revision loop.

    Branching priority:
      1. PASS / SOFT_CORRECT: return model_output unchanged.
         SOFT_CORRECT is "annotate and pass" — the correction lives in governance_note
         and the audit log; the user sees the original model output.
      2. pathway_id present: build typed renderer context, dispatch.
         The renderer never receives GovernanceDecision or blocked content.
         escalation_level is NOT the execution switch.
      3. pathway_id is None: error-containment. Both CanonicalScannerGateBridge and
         BuiltinBridge always produce a pathway_id; a None value signals a malformed
         decision object. Fail safe via domain-fallback ladder and audit loudly.

    Error-containment audit marking:
      When Tier 3/4 of the domain-fallback ladder fires (unmapped hostile flag,
      malformed decision, or missing pathway_id), decision.unexpected_unclassified_termination
      is set True and decision.fallback_reason names the gap. These fields are written
      to the audit log by _log_decision. They are error signals, not product state.
    """
    # Gate 1: PASS and SOFT_CORRECT — deliver model output, no intervention.
    if decision.action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT):
        return model_output

    # Reset the thread-local fallback sentinel before any render call.
    _fallback_sentinel.reason = None

    # Gate 2: pathway_id-driven dispatch (canonical path).
    if decision.pathway_id is not None:
        allowed_continuations = tuple(decision.allowed_continuations)
        if decision.pathway_id in _AMBIGUITY_PATHWAYS:
            if any(
                f.flag_type == FlagType.UPSTREAM_INSUFFICIENT_CONTEXT
                for f in decision.flags
            ):
                return model_output
            ctx = _build_ambiguity_context(decision)
            if decision.pathway_id == "P_ASK_DISAMBIGUATE":
                return _render_ask_disambiguate(ctx)
            return _render_ask_missing_fact(ctx)

        if decision.pathway_id in _REFUSAL_PATHWAYS:
            ctx = _build_refusal_context(decision)
            if decision.pathway_id == "P_REFUSE_EXPLAIN_REDIRECT":
                return _render_refuse_explain_redirect(ctx)
            if decision.pathway_id == "P_REFUSE_ESCALATE_PRO":
                return _render_refuse_escalate_pro(ctx)
            return _render_handoff_summary(ctx)

        if decision.pathway_id in _HARD_STOP_PATHWAYS:
            ctx = _build_hard_stop_context(decision)
            result = _render_stop_terminal(
                ctx,
                allowed_continuations,
            )  # terminal and forensic produce identical text
            _mark_fallback_if_fired(decision)
            return result

        if decision.pathway_id in _ESCALATE_PATHWAYS:
            ctx = _build_hard_stop_context(decision)
            if decision.pathway_id == "P_STOP_ESCALATE_EMERGENCY":
                return _render_stop_escalate_emergency(ctx)
            return _render_stop_escalate(ctx)

        if decision.pathway_id in _SUPPORTIVE_PATHWAYS:
            ctx = _build_hard_stop_context(decision)
            return _render_stop_supportive_deescalate(ctx)

        if decision.pathway_id in _REDIRECT_PATHWAYS:
            ctx = _build_hard_stop_context(decision)
            return _render_stop_redirect_qualified(ctx, allowed_continuations)

        if decision.pathway_id in _CLEAN_STOP_PATHWAYS:
            ctx = _build_hard_stop_context(decision)
            return _render_stop_refuse_clean(ctx, allowed_continuations)

        # Unknown pathway_id — routing gap, not a hostile-class stop.
        # The decision has flags and a domain; use the domain-fallback ladder
        # rather than a bare wall. Always audit as an unexpected state.
        ctx = _build_hard_stop_context(decision)
        result = _domain_fallback_continuation(
            ctx.domain,
            ctx.resource,
            ctx.interaction_open,
            allowed_continuations,
        )
        decision.unexpected_unclassified_termination = True
        decision.fallback_reason = getattr(_fallback_sentinel, "reason", None) or "unmapped_pathway_id"
        return result

    # Gate 3: pathway_id is None — error-containment.
    # This should never be reached: both the Governor bridge (CanonicalScannerGateBridge)
    # and BuiltinBridge always produce a pathway_id. If reached, the decision object is malformed.
    ctx = _build_hard_stop_context(decision)
    result = _domain_fallback_continuation(
        ctx.domain,
        ctx.resource,
        ctx.interaction_open,
        tuple(decision.allowed_continuations),
    )
    decision.unexpected_unclassified_termination = True
    decision.fallback_reason = "missing_pathway_id"
    return result


def _mark_fallback_if_fired(decision: GovernanceDecision) -> None:
    """Check the thread-local sentinel after a render call and mark decision if Tier 3/4 fired."""
    reason = getattr(_fallback_sentinel, "reason", None)
    if reason is not None:
        decision.unexpected_unclassified_termination = True
        decision.fallback_reason = reason


def attach_delivery_and_epistemic_audit_fields(
    entry: dict[str, Any],
    *,
    stream: bool,
    stream_completed: bool,
    stream_abort_reason: str | None,
    stream_truncated: bool,
    stream_dropped_chars: int,
    epistemic_normalisation_applied: bool | None = None,
) -> None:
    """Normalize delivery (batch vs stream) and optional post-verify epistemic flags in audit rows.

    Batch paths always set ``stream: false`` and ``stream_completed: true`` so JSONL
    shape matches stream paths for forensics, aside from the boolean ``stream`` and
    stream-specific tail fields.
    """
    entry["stream"] = stream
    if stream:
        entry["stream_completed"] = stream_completed
        if not stream_completed and stream_abort_reason:
            entry["stream_abort_reason"] = stream_abort_reason
        if stream_completed and stream_truncated:
            entry["stream_truncated"] = True
            entry["stream_dropped_chars"] = stream_dropped_chars
    else:
        entry["stream_completed"] = True
    if epistemic_normalisation_applied is not None:
        entry["epistemic_normalisation_applied"] = epistemic_normalisation_applied


def apply_domain_reclassification_fields(
    row: dict[str, Any],
    *,
    request_domain: str | None = None,
    effective_domain: str | None = None,
    domain_source: str | None = None,
) -> None:
    """Record domain provenance and reclassification status on an audit payload."""

    def _norm(v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip().lower()
        return s or None

    req = _norm(request_domain if request_domain is not None else row.get("request_domain"))
    eff = _norm(effective_domain if effective_domain is not None else row.get("domain"))
    src = _norm(domain_source if domain_source is not None else row.get("domain_source"))

    if src is None and req and eff and req != eff:
        src = "flag_pattern"
    if src is not None:
        row["domain_source"] = src

    if req and eff:
        reclassified = req != eff
        row["domain_reclassified"] = reclassified
        if reclassified:
            row["domain_from"] = req
            row["domain_to"] = eff


class GovernanceBridge(ABC):
    """Abstract interface for governance integration."""

    @abstractmethod
    async def decide(
        self,
        flags: list[Flag],
        response_text: str,
        pef: PEFState,
    ) -> GovernanceDecision:
        """Evaluate flags and produce a governance decision."""
        ...

    @abstractmethod
    async def intervene(
        self,
        decision: GovernanceDecision,
        adapter: LLMAdapter,
        user_input: str,
        pef_context: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Execute the intervention. Returns the final response text."""
        ...

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
        """Log a governance decision. Override for audit trail."""
        pass


class BuiltinBridge(GovernanceBridge):
    """In-process bridge using :class:`~aurora_lens.govern.policy.InterventionPolicy`.

    ``UNRESOLVED_REFERENT`` / ``UNRESOLVED_COMPARAND`` map to ``CONTAIN`` with
    ``P_ASK_DISAMBIGUATE`` (same clarification hold as the canonical Governor path).
    Production proxy and default :class:`~aurora_lens.lens.Lens` use
    :class:`~aurora_lens.govern.canonical_bridge.CanonicalScannerGateBridge`.

    Intended for unit tests, benchmarks, eval replay, and audit-format checks.
    Logs every decision to append-only JSONL.
    Phase D: Optional HMAC signing, log rotation, hash chain (D2).
    """

    def __init__(
        self,
        policy: InterventionPolicy | None = None,
        audit_path: str | Path | None = None,
        audit_signing_key: str | bytes | None = None,
        audit_log_max_mb: int = 0,
        audit_checkpoint_interval: int = 0,
        mode: str = "public",
        policy_version: str = "1.0",
        evidence_capture_mode: str = "plaintext_dev",
        evidence_encryption_key: str | bytes | None = None,
    ):
        self._policy = policy or DEFAULT_STRICT
        self._audit_path = Path(audit_path) if audit_path else None
        self._audit_signing_key = (
            audit_signing_key.encode("utf-8") if isinstance(audit_signing_key, str) else audit_signing_key
        ) if audit_signing_key else None
        self._audit_log_max_mb = max(0, audit_log_max_mb)
        self._audit_checkpoint_interval = max(0, audit_checkpoint_interval)
        self._mode = mode
        self._policy_version = policy_version
        self._chain_lock = threading.Lock()
        self._prev_cid = self._read_last_cid()
        self._entry_count = self._read_entry_count()
        if self._audit_path:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        # Stable process-startup identifier for log attribution across restarts.
        self._run_id: str = str(uuid.uuid4())
        self._evidence_config = EvidenceCaptureConfig(
            capture_mode=evidence_capture_mode,
            encryption_key=evidence_encryption_key,
        )
        self._evidence_vault = build_evidence_vault_for_bridge(
            str(self._audit_path) if self._audit_path else None,
            capture_mode=evidence_capture_mode,
            encryption_key=evidence_encryption_key,
        )

    def _apply_evidence_capture(
        self,
        entry: dict[str, object],
        decision: GovernanceDecision,
        *,
        pre_llm: bool,
    ) -> None:
        if self._audit_path is None:
            return
        attach_evidence_fields_to_audit_entry(
            entry,
            decision,
            pre_llm=pre_llm,
            vault=self._evidence_vault,
            config=self._evidence_config,
        )

    def _read_last_cid(self) -> str:
        if not self._audit_path or not self._audit_path.exists():
            return CHAIN_GENESIS
        try:
            lines = [ln for ln in self._audit_path.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
            if not lines:
                return CHAIN_GENESIS
            last = json.loads(lines[-1])
            return last.get("cid") or CHAIN_GENESIS
        except (OSError, json.JSONDecodeError, IndexError):
            return CHAIN_GENESIS

    def _read_entry_count(self) -> int:
        if not self._audit_path or not self._audit_path.exists():
            return 0
        try:
            lines = [ln for ln in self._audit_path.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
            count = 0
            for ln in lines:
                try:
                    entry = json.loads(ln)
                    if entry.get("type") != "checkpoint":
                        count += 1
                except json.JSONDecodeError:
                    pass
            return count
        except OSError:
            return 0

    def _maybe_checkpoint(self) -> None:
        if self._audit_checkpoint_interval <= 0 or not self._audit_path or not self._audit_signing_key:
            return
        if self._entry_count % self._audit_checkpoint_interval != 0:
            return
        new_cid = append_checkpoint_entry(
            self._audit_path,
            self._prev_cid,
            signing_key=self._audit_signing_key,
            max_mb=self._audit_log_max_mb,
        )
        if new_cid is not None:
            self._prev_cid = new_cid

    async def decide(
        self,
        flags: list[Flag],
        response_text: str,
        pef: PEFState,
    ) -> GovernanceDecision:
        action, rule = self._policy.evaluate_with_rule(flags)

        # Smoke assertion: ensure primary STOP copy follows illegal when both flags present.
        # This prevents lower-priority flags (like legal) from shadowing higher-priority
        # ones (like illegal) in the final user-facing refusal.
        if action == InterventionAction.HARD_STOP and any(
            f.flag_type == FlagType.ILLEGAL_INSTRUCTION for f in flags
        ):
            if rule and rule.flag_type != FlagType.ILLEGAL_INSTRUCTION:
                # Re-evaluate focusing on the illegal flag to ensure it takes precedence.
                # This is a safety net for non-sorted input flag lists.
                illegal_only = [f for f in flags if f.flag_type == FlagType.ILLEGAL_INSTRUCTION]
                _, illegal_rule = self._policy.evaluate_with_rule(illegal_only)
                if illegal_rule:
                    rule = illegal_rule

        rule_id = rule.rule_id if rule else None

        rationale = self._build_rationale(flags, action)
        note = self._build_governance_note(flags) if action == InterventionAction.SOFT_CORRECT else None

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
            rule_id=rule_id,
            safe_alt=safe_alt,
            resource=resource,
            pathway_id=_ACTION_TO_PATHWAY.get(action),
        )
        apply_intervention_policy_pathway_fields(
            decision, flags=flags, action=action, rule=rule,
        )
        apply_epistemic_state_from_flags(decision)
        attach_rule_result(
            decision,
            reason_code=rule_id,
            rule_id=rule_id,
        )
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
        decision.corrected_response = msg
        decision.governed_response = msg
        return msg

    def _build_rationale(self, flags: list[Flag], action: InterventionAction) -> str:
        if not flags:
            return "No verification flags"
        flag_types = set(f.flag_type.name for f in flags)
        severities = set(f.severity for f in flags)
        worst_severity = "error" if "error" in severities else "warning"
        return (
            f"{action.name}: {len(flags)} flag(s) "
            f"[{', '.join(sorted(flag_types))}] "
            f"(worst severity: {worst_severity})"
        )

    def _build_governance_note(self, flags: list[Flag]) -> str:
        parts = []
        for flag in flags:
            parts.append(f"{flag.flag_type.name}: {flag.claim} — {flag.evidence}")
        return "; ".join(parts)

    def _log_decision(
        self,
        decision: GovernanceDecision,
        final_response: str | None = None,
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
        if self._audit_path is None:
            return

        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        trace_id = trace_id_var.get(None) or ""
        if decision.action not in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT):
            forensic_event = build_forensic_event(
                decision,
                pre_llm=pre_llm,
                pef_snapshot=pef_snapshot,
                trace_id=trace_id,
                timestamp=ts,
                audit_id=decision.cid,
            )
            decision.forensic_event = forensic_event
        tenant_label = auth_label_var.get(None)
        request_hash = get_request_hash()
        ctx = pef_context or ""
        state_hash = hashlib.sha256(ctx.encode()).hexdigest()[:32] if ctx else None

        failed_constraints = [f.flag_type.name for f in decision.flags] if decision.flags else []

        entry: dict[str, object] = {
            "schema_version": _AUDIT_SCHEMA_VERSION,
            "run_id": self._run_id,
            "trace_id": trace_id,
            "timestamp": ts,
            "session_id": session_id_var.get(None),
            "turn": turn,
            "tenant_label": tenant_label,
            "mode": self._mode,
            "policy_profile": decision.policy,
            "policy_version": self._policy_version,
            "outcome": decision.action.name,
            "failed_constraints": failed_constraints,
            "original_response": decision.original_response,
            "governed_response": decision.governed_response,
            "state_hash": state_hash,
            "request_hash": request_hash,
            "log_slice_present": False,
        }
        _request_metadata = request_metadata_snapshot(get_request_metadata())
        if _request_metadata is not None:
            entry["request_metadata"] = _request_metadata
        _req_dom = domain_var.get(None)
        if _req_dom:
            entry["request_domain"] = _req_dom

        log_slice = consume_log_slice()
        if log_slice is not None:
            entry["log_slice_present"] = True
            entry.update(log_slice)

        if decision.action != InterventionAction.PASS:
            flag_list = decision.flags
            trigger_spans = [f.evidence[:200] for f in flag_list[:3]]
            _LEVEL_TO_ACTION = {1: "CLARIFY", 2: "REFUSE", 3: "STOP"}
            action_taken = _LEVEL_TO_ACTION.get(decision.escalation_level, "REFUSE")
            route = ESCALATION_ROUTES.get(flag_list[0].flag_type.name, (None, None, None, None)) if flag_list else (None, None, None, None)
            entry["escalation_level"] = decision.escalation_level
            entry["domain"] = route[0] if route[0] else "governance"
            entry["subdomain"] = route[1]
            if decision.forensic_event is not None:
                entry["forensic_event"] = decision.forensic_event
                if pef_snapshot is not None:
                    entry["pef_snapshot"] = pef_snapshot
            entry["requires_role"] = route[2]
            entry["action_taken"] = action_taken
            entry["trigger_spans"] = trigger_spans
            if decision.rule_id is not None:
                entry["policy_rule"] = decision.rule_id
            entry["rationale"] = decision.rationale
            entry["final_response"] = final_response or decision.corrected_response
            if decision.governance_note:
                entry["governance_note"] = decision.governance_note
            entry["flags"] = [
                {
                    "type": f.flag_type.name,
                    "severity": f.severity,
                    "claim": f.claim,
                    **({"rule_id": f.rule_id} if f.rule_id else {}),
                    **({"extraction_diagnostic": f.extraction_diagnostic} if f.extraction_diagnostic else {}),
                }
                for f in decision.flags
            ]
            # Surface canonical pathway metadata in audit entries when present.
            if decision.pathway_id is not None:
                entry["pathway_id"] = decision.pathway_id
                entry["output_mode"] = decision.output_mode
                entry["allowed_continuations"] = list(decision.allowed_continuations)
                entry["commitment_closed"] = decision.commitment_closed
                entry["interaction_open"] = decision.interaction_open
                entry["forensic_obligations"] = decision.forensic_obligations
                entry["resolution_mode"] = decision.resolution_mode
                entry["rendered_from_policy"] = True
            entry["governed_request_metadata"] = _build_governed_request_metadata(
                entry=entry,
                decision=decision,
            )

            # Error-containment audit marking.
            # unexpected_unclassified_termination = True means a Tier 3/4 fallback
            # fired — either an unmapped hostile flag type or a malformed decision.
            # This is not normal operation; it requires operator review.
            if decision.unexpected_unclassified_termination:
                entry["unexpected_unclassified_termination"] = True
                entry["fallback_reason"] = decision.fallback_reason

        elif decision.action == InterventionAction.PASS:
            entry["rationale"] = decision.rationale

        _fe = decision.forensic_event if isinstance(decision.forensic_event, dict) else None
        _fe_domain = _fe.get("domain") if _fe is not None else None
        _fe_ctx = _fe.get("context_provenance") if _fe is not None else None
        _fe_domain_source = (
            _fe_ctx.get("domain_source")
            if isinstance(_fe_ctx, dict)
            else None
        )
        apply_domain_reclassification_fields(
            entry,
            request_domain=entry.get("request_domain"),
            effective_domain=entry.get("domain") or _fe_domain,
            domain_source=_fe_domain_source,
        )

        lock_meta = get_lock_metadata()
        if lock_meta is not None:
            entry["lock_wait_ms"] = lock_meta.get("lock_wait_ms")
            entry["lock_acquired"] = lock_meta.get("lock_acquired")
            if not lock_meta.get("lock_acquired"):
                entry["lock_timeout_ms"] = lock_meta.get("lock_timeout_ms")

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

        if decision.admissibility_basis is not None:
            entry["admissibility_basis"] = decision.admissibility_basis
        if decision.pass_reason_code is not None:
            entry["pass_reason_code"] = decision.pass_reason_code

        if at_verification_basis is not None:
            decision.at_verification_basis = at_verification_basis
            entry["at_verification_basis"] = at_verification_basis

        coc = build_chain_of_custody_bundle(
            policy_version=self._policy_version,
            policy_source="builtin_intervention_policy",
            governance_config={
                "bridge": "builtin",
                "mode": self._mode,
                "policy_version": self._policy_version,
            },
            application_version=get_application_version(),
        )
        entry["chain_of_custody"] = coc
        _event_gid = (
            decision.forensic_event.get("governor_policy_id")
            if isinstance(decision.forensic_event, dict)
            else None
        )
        apply_ruleset_provenance_fields(
            entry,
            chain_of_custody=coc,
            policy_profile=str(entry.get("policy_profile") or decision.policy),
            outcome=decision.action.name,
            governor_policy_id=_event_gid,
        )
        if decision.forensic_event is not None:
            decision.forensic_event["chain_of_custody"] = coc
            apply_ruleset_provenance_fields(
                decision.forensic_event,
                chain_of_custody=coc,
                policy_profile=str(entry.get("policy_profile") or decision.policy),
                outcome=decision.action.name,
                governor_policy_id=_event_gid,
            )
            apply_domain_reclassification_fields(
                decision.forensic_event,
                request_domain=entry.get("request_domain"),
                effective_domain=decision.forensic_event.get("domain"),
                domain_source=(
                    decision.forensic_event.get("context_provenance", {}).get("domain_source")
                    if isinstance(decision.forensic_event.get("context_provenance"), dict)
                    else None
                ),
            )
            refresh_forensic_event_hash(decision.forensic_event)
            enforce_forensic_event_for_append(decision.forensic_event)

        provenance_fields = apply_instrument_provenance_to_row(
            entry,
            chain_of_custody=coc,
            policy_profile=str(entry.get("policy_profile") or decision.policy),
            outcome=decision.action.name,
            governor_policy_id=_event_gid,
            policy_version=self._policy_version,
            signing_key_configured=self._audit_signing_key is not None,
            attestation_signed=False,
        )
        from aurora_lens.sovereign.audit_envelope import apply_provider_route_to_audit_entry

        apply_provider_route_to_audit_entry(entry, decision)
        self._apply_evidence_capture(entry, decision, pre_llm=pre_llm)
        attestation_fields = apply_attestation_fields_to_row(
            entry,
            chain_of_custody=coc,
            signing_key=self._audit_signing_key,
        )
        sync_instrument_provenance_to_decision(
            decision,
            {**provenance_fields, **attestation_fields},
            signature_status=entry.get("signature_status"),
        )

        with self._chain_lock:
            prev_link = self._prev_cid
            new_cid = append_audit_entry(
                self._audit_path,
                entry,
                signing_key=self._audit_signing_key,
                max_mb=self._audit_log_max_mb,
                prev_cid=self._prev_cid,
            )
            if new_cid is not None:
                self._prev_cid = new_cid
                decision.cid = new_cid
                finalize_decision_record_hash(decision, entry, cid=new_cid)
                decision.signature_status = str(entry.get("signature_status") or decision.signature_status or "")
            self._entry_count += 1
            if new_cid is not None:
                finalize_audit_receipt_snapshot(
                    decision,
                    chain_fields={
                        "trace_id": trace_id_var.get(None) or "",
                        "prev_hash": prev_link,
                        "hash": new_cid,
                        "entry_index": self._entry_count,
                    },
                )
            self._maybe_checkpoint()

    def _ensure_decision_forensic_event(
        self,
        decision: GovernanceDecision,
        *,
        pre_llm: bool,
        pef_snapshot: dict | None,
    ) -> None:
        """Attach forensic_event on the decision even when audit logging is disabled."""
        if decision.action in (InterventionAction.PASS, InterventionAction.SOFT_CORRECT):
            return
        if decision.forensic_event is not None:
            return
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        trace_id = trace_id_var.get(None) or ""
        decision.forensic_event = build_forensic_event(
            decision,
            pre_llm=pre_llm,
            pef_snapshot=pef_snapshot,
            trace_id=trace_id,
            timestamp=ts,
            audit_id=decision.cid,
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
        """Log a governance decision. Override for audit trail."""
        self._ensure_decision_forensic_event(
            decision,
            pre_llm=pre_llm,
            pef_snapshot=pef_snapshot,
        )
        final = decision.corrected_response or decision.original_response
        self._log_decision(
            decision,
            final_response=final,
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

    def log_clarification_resolution(
        self,
        *,
        turn: int,
        clarification_resolution: dict[str, Any],
    ) -> str | None:
        """Append a first-class USER_DISAMBIGUATION audit row (provenance only; not governance)."""
        if self._audit_path is None:
            return None

        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        trace_id = trace_id_var.get(None) or ""
        tenant_label = auth_label_var.get(None)
        request_hash = get_request_hash()
        _req_dom = domain_var.get(None)

        entry = build_clarification_resolution_audit_entry(
            schema_version=_AUDIT_SCHEMA_VERSION,
            run_id=self._run_id,
            trace_id=trace_id,
            timestamp=ts,
            session_id=session_id_var.get(None),
            turn=turn,
            tenant_label=tenant_label,
            mode=self._mode,
            policy_version=self._policy_version,
            clarification_resolution=clarification_resolution,
            request_hash=request_hash,
            request_domain=_req_dom,
        )

        with self._chain_lock:
            new_cid = append_audit_entry(
                self._audit_path,
                entry,
                signing_key=self._audit_signing_key,
                max_mb=self._audit_log_max_mb,
                prev_cid=self._prev_cid,
            )
            if new_cid is not None:
                self._prev_cid = new_cid
                self._entry_count += 1
                self._maybe_checkpoint()
            return new_cid
