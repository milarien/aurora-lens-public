"""Governance decision types.

InterventionAction defines what the system does when verification flags fire.
GovernanceDecision is the full decision record — action, evidence, and audit fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from aurora_lens.verify.flags import Flag, FlagType


class InterventionAction(Enum):
    """What the governance engine does with a flagged response."""
    PASS = auto()            # Clean — no flags, or flags below threshold
    SOFT_CORRECT = auto()    # Response delivered; correction in metadata only
    FORCE_REVISE = auto()    # Re-prompt LLM with flag context + PEF ground truth
    CONTAIN = auto()         # No determinations; acknowledge + ask intent (EXTRACTION_EMPTY)
    HARD_STOP = auto()       # Block response entirely


_ACTION_TO_ESCALATION = {
    InterventionAction.PASS: 0,         # ADMIT
    InterventionAction.CONTAIN: 1,       # CLARIFY
    InterventionAction.SOFT_CORRECT: 2,  # REFUSE
    InterventionAction.FORCE_REVISE: 2,  # REFUSE (deprecated, maps to 2)
    InterventionAction.HARD_STOP: 3,     # STOP
}


@dataclass
class GovernanceRuleResult:
    """First-class governance trigger identity for renderer and integration dispatch."""

    rule_id: str
    domain: str
    outcome: str
    flag_type: FlagType | None
    reason_code: str
    continuation_type: str | None
    interaction_open: bool
    escalation_target: str | None
    user_facing_template_key: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "domain": self.domain,
            "outcome": self.outcome,
            "flag_type": self.flag_type.name if self.flag_type is not None else None,
            "reason_code": self.reason_code,
            "continuation_type": self.continuation_type,
            "interaction_open": self.interaction_open,
            "escalation_target": self.escalation_target,
            "user_facing_template_key": self.user_facing_template_key,
        }


_RULE_DOMAIN_BY_FLAG_TYPE: dict[FlagType, str] = {
    FlagType.PERSONALIZED_FINANCIAL_ADVICE:   "finance",
    FlagType.PERSONALIZED_LEGAL_ADVICE:       "legal",
    FlagType.PERSONALIZED_MEDICAL_ADVICE:     "medical",
    FlagType.MEDICAL_DOSAGE_RECOMMENDATION:   "medical",
    FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION: "medical",
    FlagType.NUMERIC_MEDICAL_INSTRUCTION:     "medical",
    FlagType.EMERGENCY_TRIAGE_GUIDANCE:       "medical",
    FlagType.SELF_HARM_INSTRUCTION:           "medical",
    FlagType.ILLEGAL_INSTRUCTION:             "harmful",
    FlagType.HAZARD_ONTOLOGY_UNAVAILABLE:     "harmful",
    FlagType.AGENCY_VIOLATION_ASSISTANCE:     "harmful",
    FlagType.TARGETED_DEFAMATION:             "harmful",
    FlagType.VIOLENT_CRIMINAL_INTENT:         "harmful",
    FlagType.SENSITIVE_PII_EXPOSURE:          "legal",
    FlagType.UNRESOLVED_STATE_TRANSITION:     "procedural",
    FlagType.UNRESOLVED_REFERENT:             "ambiguity",
    FlagType.DISJUNCTIVE_BRANCH_COLLAPSE:     "ambiguity",
    FlagType.CLARIFICATION_AUTHORITY_OUTSIDE_HOLD: "ambiguity",
    FlagType.UPSTREAM_INSUFFICIENT_CONTEXT:    "ambiguity",
    FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED:  "ambiguity",
    FlagType.HAZARD_SUBSTANCE_UNRESOLVED:     "ambiguity",
    FlagType.UNCLASSIFIED_CONSEQUENCE_INTENT: "governance",
    FlagType.INTERPRETATION_LIMIT:            "ambiguity",
}

# Hazard ontology rule ids — always canonical domain "harmful" (never LEGAL).
_HAZARD_HARMFUL_RULE_IDS: frozenset[str] = frozenset({
    "blocked.request.hazard_operational_transform",
    "blocked.request.hazard_ontology_unavailable",
    "blocked.request.illegal_hazardous_synthesis",
    "blocked.request.bioweapon_generic_fabrication",
})


def _rule_domain_for_flag(flag: Flag | None) -> str | None:
    """Canonical rule-result domain for a primary flag."""
    if flag is None:
        return None
    rid = str(flag.rule_id) if flag.rule_id else ""
    if rid in _HAZARD_HARMFUL_RULE_IDS:
        return "harmful"
    return _RULE_DOMAIN_BY_FLAG_TYPE.get(flag.flag_type)


def apply_epistemic_state_from_flags(decision: GovernanceDecision) -> None:
    """Attach audit-facing epistemic_state when structured flags require it."""
    if any(f.flag_type == FlagType.UPSTREAM_INSUFFICIENT_CONTEXT for f in decision.flags):
        decision.epistemic_state = "insufficient_context"


def continuation_type_from_capability(capability: str | None, domain: str) -> str | None:
    cap = (capability or "").strip().lower()
    dom = (domain or "").strip().lower()
    if not cap:
        return None
    if cap == "neutral_timeline" and dom == "finance":
        return "financial_facts_summary"
    if cap == "neutral_timeline" and dom == "legal":
        return "neutral_legal_timeline"
    if cap == "medical_post_refusal_safe_followup":
        return "medical_questions_for_clinician"
    return cap


def attach_rule_result(
    decision: "GovernanceDecision",
    *,
    domain: str | None = None,
    reason_code: str | None = None,
    continuation_type: str | None = None,
    user_facing_template_key: str | None = None,
    rule_id: str | None = None,
) -> None:
    """Attach first-class rule identity to a decision when absent."""
    if decision.rule_result is not None:
        return
    primary_flag = decision.flags[0] if decision.flags else None
    flag_type = primary_flag.flag_type if primary_flag is not None else None

    # Deterministic domain resolution:
    # 1. If a flag-based domain exists, it is authoritative for non-PASS outcomes.
    #    This prevents request-level hints (e.g. "legal") from overriding the
    #    actual reason for a stop (e.g. "illegal_instruction").
    # 2. Otherwise, use the provided domain hint.
    # 3. Fall back to "general".
    flag_domain = _rule_domain_for_flag(primary_flag)
    if decision.action != InterventionAction.PASS and flag_domain:
        dom = flag_domain
    else:
        dom = (domain or "").strip().lower() or flag_domain or "general"

    rsn = (
        (reason_code or "").strip()
        or (primary_flag.rule_id if primary_flag is not None and primary_flag.rule_id else "")
        or (flag_type.name.lower() if flag_type is not None else "")
        or "governance_decision"
    )
    cont = continuation_type
    if cont is None:
        cap = decision.allowed_continuations[0] if decision.allowed_continuations else None
        cont = continuation_type_from_capability(cap, dom)
        if cont is None and decision.action == InterventionAction.CONTAIN:
            cont = "clarification"
    rid = rule_id or decision.rule_id or (primary_flag.rule_id if primary_flag is not None and primary_flag.rule_id else None) or rsn
    if user_facing_template_key:
        tmpl = user_facing_template_key
    elif (
        dom == "finance"
        and decision.action == InterventionAction.HARD_STOP
        and flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE
    ):
        tmpl = "finance.blocked.personalized_decision"
    elif (
        dom == "legal"
        and decision.action == InterventionAction.HARD_STOP
        and flag_type == FlagType.PERSONALIZED_LEGAL_ADVICE
    ):
        tmpl = "legal.blocked.personalized_case_outcome"
    elif (
        dom == "medical"
        and decision.action == InterventionAction.HARD_STOP
        and flag_type in {
            FlagType.PERSONALIZED_MEDICAL_ADVICE,
            FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
            FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
            FlagType.NUMERIC_MEDICAL_INSTRUCTION,
        }
    ):
        tmpl = "medical.blocked.personalized_treatment_or_dosage"
    elif decision.action == InterventionAction.CONTAIN:
        tmpl = f"{dom}.clarification.requested"
    else:
        tmpl = f"{dom}.{decision.action.name.lower()}.{rsn.replace(':', '.').replace('|', '.')}"
    decision.rule_result = GovernanceRuleResult(
        rule_id=rid,
        domain=dom,
        outcome=decision.action.name,
        flag_type=flag_type,
        reason_code=rsn,
        continuation_type=cont,
        interaction_open=decision.interaction_open,
        escalation_target=decision.resource,
        user_facing_template_key=tmpl,
    )


@dataclass
class GovernanceDecision:
    """Full governance decision record.

    Carries everything needed for audit: what happened, why, and context.
    escalation_level is derived from action — for compatibility/telemetry only.
    When canonical Governor fields are present (pathway_id is not None), the
    execution layer must branch on pathway_id, not on escalation_level.

    Canonical pathway fields (set by the Governor bridge, ``CanonicalScannerGateBridge.decide()``):
        pathway_id:           ContinuationPathway value string (e.g. "P_STOP_TERMINAL")
        output_mode:          OutputMode value string (e.g. "terminal_stop")
        commitment_closed:    True when no determination may be issued or recovered
        interaction_open:     True when the user may continue the conversation
        forensic_obligations: List of ForensicObligation value strings
        resolution_mode:      ResolutionMode value string (for audit provenance)

    forensic_event: canonical envelope for non-ADMIT outcomes; set by bridge when logging.
    """
    action: InterventionAction
    flags: list[Flag]
    rationale: str
    policy: str = "strict"                     # Policy that produced this decision
    rule_id: str | None = None                 # PolicyRule.rule_id that produced this action (audit)
    attempt: int = 0                           # Revision attempt (0 = first pass)
    original_response: str | None = None       # Pre-intervention response
    corrected_response: str | None = None      # Post-intervention response
    governed_response: str | None = None       # Exact text returned to caller after pathway rendering
    governance_note: str | None = None         # Human-readable note (for SOFT_CORRECT)
    cid: str | None = None                     # Content identifier (if available)
    safe_alt: str | None = None                # Phase 11: optional alternative for refuse template
    resource: str | None = None                # Phase 11/12: escalation resource (e.g. crisis support)
    forensic_event: dict | None = None         # Canonical forensic envelope (non-ADMIT only)

    # Canonical Governor continuation fields.
    # Populated by ``CanonicalScannerGateBridge.decide()`` from RuntimeDecisionProjection.
    # When pathway_id is present, enforce() branches on these rather than escalation_level.
    pathway_id: str | None = None
    output_mode: str | None = None
    allowed_continuations: list[str] = field(default_factory=list)
    commitment_closed: bool = True
    interaction_open: bool = False
    forensic_obligations: list[str] = field(default_factory=list)
    resolution_mode: str | None = None

    # Error-containment audit fields.
    # Set by enforce() when a Tier 3/4 fallback fires — either an unmapped hostile
    # flag type or a malformed decision reaching the bare stop.
    # These are not product features; they signal unexpected states that require
    # operator attention. When True, unexpected_unclassified_termination = True
    # and fallback_reason names the classification of the gap.
    unexpected_unclassified_termination: bool = False
    fallback_reason: str | None = None

    # Structural governor verdict (Phase 1).
    # Set by lens.py after StructuralGovernor.evaluate(). The verdict is the
    # authoritative decision; synthetic flags are rendering adapters only.
    governor_verdict: object | None = None

    # First-class rule identity for renderer and integration surfaces.
    rule_result: GovernanceRuleResult | None = None

    # Recoverable verification basis for locative AT (see aurora_lens.pef.at_read.at_basis_snapshot).
    # Populated when governance logs include ``at_verification_basis`` on the audit row.
    at_verification_basis: dict[str, object] | None = None

    # Operator-plane snapshot only: safe per-turn receipt fields after audit append (hashes,
    # chain slot). Never populated with signing secrets or raw config.
    audit_receipt_snapshot: dict[str, object] | None = None

    # Evidence capture receipt (merged into audit_receipt_snapshot after ledger append).
    evidence_receipt_snapshot: dict[str, object] | None = None

    # Sovereign Provider Registry route audit (Track B Phase 2). Replayable provider-route
    # decision envelope attached to audit rows when ``request_metadata.provider_route`` was
    # evaluated this turn.
    provider_route: dict[str, object] | None = None

    # Audit telemetry: epistemic posture of the governed outcome (distinct from action alone).
    epistemic_state: str | None = None

    # Pre-LLM block indicator (request-side gated)
    pre_llm: bool = False

    # Strict-policy PASS admissibility (audit: why PASS was earned, not merely no flags).
    admissibility_basis: str | None = None
    pass_reason_code: str | None = None

    # Instrument provenance (populated on audit write from policy / chain-of-custody sources).
    policy_ref: str | None = None
    instrument_id: str | None = None
    instrument_version: str | None = None
    ruleset_hash: str | None = None
    signature_status: str | None = None
    decision_record_hash: str | None = None
    provenance_status: str | None = None
    decision_signature: str | None = None
    decision_signature_status: str | None = None
    instrument_signature: str | None = None
    instrument_signature_status: str | None = None
    instrument_attestation_hash: str | None = None
    attestation_mode: str | None = None

    @property
    def escalation_level(self) -> int:
        """0=ADMIT, 1=CLARIFY, 2=REFUSE, 3=STOP. Derived from action.

        Retained for compatibility and telemetry. Do NOT use as the execution
        branch when pathway_id is present — use pathway_id instead.
        """
        return _ACTION_TO_ESCALATION.get(self.action, 0)
