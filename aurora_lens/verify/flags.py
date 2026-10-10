"""Verification flag types with evidence.

Three axes — no cross-contamination:

  Epistemic failure (ungrounded assertions)
    Model asserts world-claims without support in PEF or provided context.
    Default governance outcome: FORCE_REVISE (cite, hedge, or admit ignorance).

  PEF binding failures
    Referent resolution and span-consistency failures against the world model.

  Content-class veto (normative refusal)
    Prohibited categories regardless of truth value.
    "Even if true, you don't get to say it."
    Default governance outcome: REFUSE immediately — no revise loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class DefamationRole(Enum):
    """Who the user is relative to alleged defamation — renderer routing only.

    Does not change blocking or pathway selection; ``enforce()`` uses this only
    for TARGETED_DEFAMATION user-facing copy.
    """

    AUTHOR = auto()
    TARGET = auto()
    REPORTER = auto()
    UNKNOWN = auto()


class FlagType(Enum):
    # ── Epistemic failure (ungrounded assertions) ─────────────────────────────
    UNBOUND_ENTITY = auto()                 # Claim subject has no PEF binding; facts about unbound entity
    UNSUPPORTED_ATTRIBUTE = auto()          # Attribute asserted for known entity without PEF support
    UNSUPPORTED_EVENT = auto()              # Event/relationship asserted without PEF support
    PREDICTIVE_CLAIM_NOT_ESTABLISHED = auto()  # Predictive/modal release not established by evidence
    UNVERIFIED_FACT_ASSERTION = auto()      # Temporal/quantitative fact with no PEF/context support
    UNVERIFIED_REGULATORY_CLAIM = auto()    # FDA/EMA/CE/WHO approval asserted without verifiable support
    # RAG harness C2 only: *Whose … Emma's or Lucy's?* — governed as containment (ASK), not PASS.
    DISJUNCTIVE_BRANCH_COLLAPSE = auto()
    # RAG eval harness (Context: … Question:) — manifest U1/U2 governed non-admit.
    RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT = auto()
    RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT = auto()
    # Post-LLM: model solicited referent/binding commitment without structured containment.
    CLARIFICATION_AUTHORITY_OUTSIDE_HOLD = auto()
    # Post-LLM: upstream answer is an insufficiency/clarification speech act, not a resolved reply.
    UPSTREAM_INSUFFICIENT_CONTEXT = auto()
    # Strict admissibility: consequential request intent without a matching lane.
    UNCLASSIFIED_CONSEQUENCE_INTENT = auto()
    # Pre-commit: the interpreter reports the input is outside what it can read
    # (unsupported script, no main predicate). Nothing is committed or answered.
    INTERPRETATION_LIMIT = auto()

    # ── PEF binding failures ─────────────────────────────────────────────────
    UNRESOLVED_REFERENT = auto()            # Common-noun referent not bound into PEF (e.g. "sister")
    UNRESOLVED_COMPARAND = auto()           # Comparative adjective with 2+ eligible comparands in PEF
    # State-native committed-state read: multiple PEF entity heads match the query subject.
    # Governance mapping only (state_native_engine), not the LLM verification checker.
    STATE_NATIVE_ENTITY_AMBIGUITY = auto()
    # State-native committed-state read: insufficient committed state to answer (STOP).
    # Provides a non-empty failed_constraints list for the forensic schema invariant.
    STATE_NATIVE_COMMITTED_STATE_STOP = auto()
    # State-native present-bound temporal contact: no governing temporal anchor in PEF.
    STATE_NATIVE_TEMPORAL_ANCHOR_MISSING = auto()
    # State-native present-bound temporal contact: multiple admissible temporal scopes.
    STATE_NATIVE_TEMPORAL_SCOPE_AMBIGUITY = auto()
    IDENTITY_DRIFT = auto()                 # Entity properties changed without cause
    TIME_SMEAR = auto()                     # Past/present context mixed without span transition
    CONTRADICTED_FACT = auto()              # Directly contradicts an established PEF relationship
    CONTRADICTS_COMMITTED_STATE = auto()    # Claim vs committed current_at head (e.g. stale AT)
    UNRESOLVED_STATE_TRANSITION = auto()    # Treats an unresolved state as settled fact

    # ── Extraction failures ──────────────────────────────────────────────────
    EXTRACTION_FAILED = auto()              # Backend returned invalid/unparseable output
    EXTRACTION_EMPTY = "extraction_empty"   # Schema-valid but barren (no claims, no entity_mentions)

    # ── Content-class veto — hard-stop (all modes) ───────────────────────────
    # These are never allowed from a language model regardless of claimed support.
    # Governance outcome: HARD_STOP, no revise loop.
    MEDICAL_DOSAGE_RECOMMENDATION = "medical_dosage"           # Explicit patient dosing instruction
    PEDIATRIC_DOSAGE_RECOMMENDATION = "pediatric_dosage"       # Pediatric-specific dosage
    NUMERIC_MEDICAL_INSTRUCTION = "numeric_medical_instruction" # Titration/infusion/taper/insulin schedules
    EMERGENCY_TRIAGE_GUIDANCE = "emergency_triage"             # Definitive triage or "don't seek care" reassurance
    SELF_HARM_INSTRUCTION = "self_harm"                        # Instructions facilitating self-harm
    ILLEGAL_INSTRUCTION = "illegal_instruction"                # Weapons, fraud, hacking, evading law enforcement
    AGENCY_VIOLATION_ASSISTANCE = "agency_violation_assistance"  # Tactics/instructions to manipulate or coerce people
    AGENCY_RISK_CONTEXT_UNRESOLVED = "agency_risk_context_unresolved"  # Coercive objective present, intent/context unresolved
    # Pre-LLM hazard ontology: procedural transform requested but substance unresolved.
    HAZARD_SUBSTANCE_UNRESOLVED = "hazard_substance_unresolved"
    # Pre-LLM hazard ontology substrate unavailable — fail-closed stop.
    HAZARD_ONTOLOGY_UNAVAILABLE = "hazard_ontology_unavailable"
    PROMPT_INJECTION_ATTEMPT = "prompt_injection_attempt"        # Attempts to override governance, instructions, audit, or policy authority
    TARGETED_DEFAMATION = "targeted_defamation"                # Factual allegations about private individuals
    VIOLENT_CRIMINAL_INTENT = "violent_criminal_intent"        # Interpersonal homicide framed as legal/permissibility query

    # ── Content-class veto — hard-stop (education / workforce / enterprise) ─────
    ACADEMIC_INTEGRITY_VIOLATION = "academic_integrity_violation"  # Ghostwriting assignments or providing exam answers for submission
    STUDENT_RECORD_EXPOSURE = "student_record_exposure"            # FERPA-class data: grades, IDs, disciplinary records
    EMPLOYMENT_DISCRIMINATION_FACILITATION = "employment_discrimination"  # Hiring/firing instructions based on protected characteristics
    EMPLOYEE_RECORD_EXPOSURE = "employee_record_exposure"          # Confidential HR data: performance, medical, compensation
    TRADE_SECRET_DISCLOSURE = "trade_secret_disclosure"            # Proprietary formulas, processes, or strategy
    INSIDER_INFORMATION_ASSISTANCE = "insider_information"         # Non-public material information for trading or competitive advantage
    PROCUREMENT_FRAUD_FACILITATION = "procurement_fraud"           # Bid-rigging, conflict-of-interest concealment instructions

    # ── Content-class veto — verify-or-refuse (mode-dependent) ───────────────
    # HARD_STOP in public mode.
    # FORCE_REVISE in enterprise mode (allowed only when grounded in verified sources).
    # Mode is set per-deployment in config — never inferred from request text.
    PERSONALIZED_MEDICAL_ADVICE = "personalized_medical"       # Individualized diagnosis / treatment plan
    PERSONALIZED_LEGAL_ADVICE = "personalized_legal"           # Jurisdictional procedure / deadline advice
    PERSONALIZED_FINANCIAL_ADVICE = "personalized_financial"   # Individualized investment / tax advice
    SENSITIVE_PII_EXPOSURE = "sensitive_pii"                   # Outputting or inferring private personal data
    UNVERIFIED_TRUTHFULNESS_CLAIM = "unverified_truthfulness"  # Definitive assertion about private individual's honesty/deception
    PERSONALIZED_ACADEMIC_ADVICE = "personalized_academic"     # Specific admissions chances or credential plans for an individual
    PERSONALIZED_EMPLOYMENT_ADVICE = "personalized_employment" # Individual employment dispute posture (wrongful dismissal, discrimination claim)
    PERSONALIZED_BUSINESS_STRATEGY_ADVICE = "personalized_business_strategy"  # Named-company strategic recommendations


@dataclass
class Flag:
    """A verification flag raised against an LLM response claim."""
    flag_type: FlagType
    entity_name: str                   # Entity involved
    claim: str                         # What the LLM said
    evidence: str                      # Why this was flagged
    severity: str = "warning"          # "warning" or "error"
    extraction_diagnostic: dict | None = None  # For EXTRACTION_FAILED: provider, model, prompt_hash
    candidates: tuple[str, ...] = ()   # Normalized PEF labels for ambiguity flags
                                       # (UNRESOLVED_REFERENT, UNRESOLVED_COMPARAND).
                                       # Entity names from the world model, never raw evidence.
                                       # These are entity names from the world model, never
                                       # raw evidence text. Renderers may surface these to
                                       # the user. All other flag fields are audit-only.
    rule_id: str | None = None         # Stable policy rule id (e.g. blocked.request.*); optional
    defamation_role: DefamationRole | None = None  # TARGETED_DEFAMATION: renderer-only role hint

    def __str__(self) -> str:
        rid = f" [{self.rule_id}]" if self.rule_id else ""
        return f"[{self.flag_type.name}]{rid} {self.entity_name}: {self.claim} — {self.evidence}"


def flag_from_external(raw: dict) -> Flag | None:
    """Convert external flag dict to Flag. Returns None if type unknown.

    Expected format: {"type": "SELF_HARM_INSTRUCTION", "evidence": ["span"], "source": "..."}
    """
    type_str = raw.get("type")
    if not type_str or not isinstance(type_str, str):
        return None
    try:
        flag_type = FlagType[type_str]
    except KeyError:
        return None
    evidence = raw.get("evidence")
    if isinstance(evidence, list):
        evidence_str = "; ".join(str(s) for s in evidence[:3])
    else:
        evidence_str = str(evidence) if evidence else "external"
    source = raw.get("source", "external")
    dr_raw = raw.get("defamation_role")
    defamation_role: DefamationRole | None = None
    if isinstance(dr_raw, str):
        try:
            defamation_role = DefamationRole[dr_raw]
        except KeyError:
            defamation_role = None
    return Flag(
        flag_type=flag_type,
        entity_name="external",
        claim=f"External flag from {source}",
        evidence=evidence_str or source,
        severity=raw.get("severity", "warning"),
        defamation_role=defamation_role,
    )
