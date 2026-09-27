"""Intervention policy — maps verification flags to governance actions.

Two named policies: strict and moderate.
Three deployment modes: public, enterprise, open.

Mode is set per-deployment in config — NEVER inferred from request text.
  public     Hard-stop on verify-or-refuse categories (medical/legal/financial advice, PII).
  enterprise Allow verify-or-refuse with FORCE_REVISE if grounded in verified sources.
  open       Open-world epistemic mode: UNBOUND_ENTITY may SOFT_CORRECT (annotate) rather
             than FORCE_REVISE; UNRESOLVED_REFERENT / UNRESOLVED_COMPARAND still CONTAIN
             (clarification hold). Suitable when the PEF starts empty and most factual
             claims lack PEF grounding.
             All axis-3 normative vetoes (self-harm, illegal, defamation, medical, PII)
             remain HARD_STOP regardless of mode.

Hard-stop-always categories (HARD_STOP regardless of mode):
  dosage, titration, triage, self-harm, illegal instructions, defamation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aurora_lens.verify.flags import Flag, FlagType
from aurora_lens.govern.decision import InterventionAction


@dataclass(frozen=True)
class PolicyRule:
    """A single policy rule: flag type + severity threshold + optional modes → action.

    rule_id: Stable identifier for audit and traceability (e.g. "EXTRACTION_FAILED:error:HARD_STOP").
    modes: frozenset of mode strings this rule applies to, or None for all modes.
    safe_alt: Optional alternative for Level 2 refuse template (Phase 11 tone engine).
    """
    flag_type: FlagType
    min_severity: str                          # "warning" or "error"
    action: InterventionAction
    rule_id: str                               # Stable identifier for audit
    modes: frozenset[str] | None = None        # None = applies to all modes
    safe_alt: str | None = None                # Optional: "I can help with X instead"


class InterventionPolicy:
    """Ordered policy rules. Evaluates flags and returns the most severe action."""

    def __init__(
        self,
        rules: list[PolicyRule],
        name: str = "custom",
        mode: str = "public",
    ):
        self._rules = rules
        self.name = name
        self.mode = mode

    def for_mode(self, mode: str) -> "InterventionPolicy":
        """Return a copy of this policy configured for the given deployment mode.

        Mode is set once per deployment from config. Never derived from request text.
        """
        if mode not in ("public", "enterprise", "open"):
            raise ValueError(
                f"Unknown mode: {mode!r} — expected 'public', 'enterprise', or 'open'"
            )
        return InterventionPolicy(self._rules, name=self.name, mode=mode)

    def evaluate(self, flags: list[Flag]) -> InterventionAction:
        """Evaluate flags against policy rules. Returns the most severe action.

        Severity order: PASS < SOFT_CORRECT < FORCE_REVISE < CONTAIN < HARD_STOP
        Rules whose .modes do not include self.mode are skipped.
        """
        action, _ = self.evaluate_with_rule(flags)
        return action

    def evaluate_with_rule(self, flags: list[Flag]) -> tuple[InterventionAction, PolicyRule | None]:
        """Evaluate flags and return (action, matched_rule). Use for audit traceability.

        matched_rule is the rule that produced the worst action; None when no rule matched.
        """
        if not flags:
            return InterventionAction.PASS, None

        worst = InterventionAction.PASS
        worst_rule: PolicyRule | None = None

        for flag in flags:
            for rule in self._rules:
                if rule.flag_type != flag.flag_type:
                    continue
                if rule.modes is not None and self.mode not in rule.modes:
                    continue
                if not self._severity_meets_threshold(flag.severity, rule.min_severity):
                    continue
                if rule.action.value > worst.value:
                    worst = rule.action
                    worst_rule = rule

        # If flags exist but no rule matched, default to SOFT_CORRECT
        if worst == InterventionAction.PASS and flags:
            worst = InterventionAction.SOFT_CORRECT

        return worst, worst_rule

    @staticmethod
    def _severity_meets_threshold(actual: str, threshold: str) -> bool:
        """Check if actual severity meets or exceeds the threshold."""
        order = {"warning": 0, "error": 1}
        return order.get(actual, 0) >= order.get(threshold, 0)


# ── Built-in policies ────────────────────────────────────────────────────────

# Helper: mode-pair shorthand
_PUBLIC = frozenset({"public"})
_ENTERPRISE = frozenset({"enterprise"})
_PUBLIC_ENTERPRISE = frozenset({"public", "enterprise"})
_OPEN = frozenset({"open"})

DEFAULT_STRICT = InterventionPolicy([
    # Extraction failures
    PolicyRule(FlagType.EXTRACTION_FAILED, "error", InterventionAction.HARD_STOP, "EXTRACTION_FAILED:error:HARD_STOP"),
    PolicyRule(FlagType.EXTRACTION_EMPTY, "error", InterventionAction.HARD_STOP, "EXTRACTION_EMPTY:error:HARD_STOP", _PUBLIC_ENTERPRISE),
    PolicyRule(FlagType.EXTRACTION_EMPTY, "error", InterventionAction.CONTAIN, "EXTRACTION_EMPTY:error:CONTAIN:open", _OPEN),

    # Hard-stop always — no revise loop, no mode distinction
    PolicyRule(FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION, "error", InterventionAction.HARD_STOP, "PEDIATRIC_DOSAGE_RECOMMENDATION:error:HARD_STOP"),
    PolicyRule(FlagType.MEDICAL_DOSAGE_RECOMMENDATION, "error", InterventionAction.HARD_STOP, "MEDICAL_DOSAGE_RECOMMENDATION:error:HARD_STOP"),
    PolicyRule(FlagType.NUMERIC_MEDICAL_INSTRUCTION, "error", InterventionAction.HARD_STOP, "NUMERIC_MEDICAL_INSTRUCTION:error:HARD_STOP"),
    PolicyRule(FlagType.EMERGENCY_TRIAGE_GUIDANCE, "error", InterventionAction.HARD_STOP, "EMERGENCY_TRIAGE_GUIDANCE:error:HARD_STOP"),
    PolicyRule(FlagType.SELF_HARM_INSTRUCTION, "error", InterventionAction.HARD_STOP, "SELF_HARM_INSTRUCTION:error:HARD_STOP"),
    PolicyRule(FlagType.ILLEGAL_INSTRUCTION, "error", InterventionAction.HARD_STOP, "ILLEGAL_INSTRUCTION:error:HARD_STOP"),
    PolicyRule(FlagType.AGENCY_VIOLATION_ASSISTANCE, "error", InterventionAction.HARD_STOP, "AGENCY_VIOLATION_ASSISTANCE:error:HARD_STOP"),
    PolicyRule(FlagType.PROMPT_INJECTION_ATTEMPT, "error", InterventionAction.HARD_STOP, "PROMPT_INJECTION_ATTEMPT:error:HARD_STOP"),
    PolicyRule(FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED, "warning", InterventionAction.CONTAIN, "AGENCY_RISK_CONTEXT_UNRESOLVED:warning:CONTAIN"),
    PolicyRule(FlagType.TARGETED_DEFAMATION, "error", InterventionAction.HARD_STOP, "TARGETED_DEFAMATION:error:HARD_STOP"),
    PolicyRule(FlagType.VIOLENT_CRIMINAL_INTENT, "error", InterventionAction.HARD_STOP, "VIOLENT_CRIMINAL_INTENT:error:HARD_STOP"),
    # Education / workforce / enterprise hard-stop vetoes
    PolicyRule(FlagType.ACADEMIC_INTEGRITY_VIOLATION, "error", InterventionAction.HARD_STOP, "ACADEMIC_INTEGRITY_VIOLATION:error:HARD_STOP"),
    PolicyRule(FlagType.STUDENT_RECORD_EXPOSURE, "error", InterventionAction.HARD_STOP, "STUDENT_RECORD_EXPOSURE:error:HARD_STOP"),
    PolicyRule(FlagType.EMPLOYMENT_DISCRIMINATION_FACILITATION, "error", InterventionAction.HARD_STOP, "EMPLOYMENT_DISCRIMINATION_FACILITATION:error:HARD_STOP"),
    PolicyRule(FlagType.EMPLOYEE_RECORD_EXPOSURE, "error", InterventionAction.HARD_STOP, "EMPLOYEE_RECORD_EXPOSURE:error:HARD_STOP"),
    PolicyRule(FlagType.TRADE_SECRET_DISCLOSURE, "error", InterventionAction.HARD_STOP, "TRADE_SECRET_DISCLOSURE:error:HARD_STOP"),
    PolicyRule(FlagType.INSIDER_INFORMATION_ASSISTANCE, "error", InterventionAction.HARD_STOP, "INSIDER_INFORMATION_ASSISTANCE:error:HARD_STOP"),
    PolicyRule(FlagType.PROCUREMENT_FRAUD_FACILITATION, "error", InterventionAction.HARD_STOP, "PROCUREMENT_FRAUD_FACILITATION:error:HARD_STOP"),

    # Verify-or-refuse — mode-dependent
    # public:     HARD_STOP (model must not make these determinations)
    # enterprise: FORCE_REVISE (allowed only when grounded in verified, cited sources)
    PolicyRule(FlagType.PERSONALIZED_MEDICAL_ADVICE, "warning", InterventionAction.HARD_STOP, "PERSONALIZED_MEDICAL_ADVICE:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.PERSONALIZED_MEDICAL_ADVICE, "warning", InterventionAction.FORCE_REVISE, "PERSONALIZED_MEDICAL_ADVICE:warning:FORCE_REVISE:enterprise", _ENTERPRISE),
    PolicyRule(FlagType.PERSONALIZED_LEGAL_ADVICE, "warning", InterventionAction.HARD_STOP, "PERSONALIZED_LEGAL_ADVICE:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.PERSONALIZED_LEGAL_ADVICE, "warning", InterventionAction.FORCE_REVISE, "PERSONALIZED_LEGAL_ADVICE:warning:FORCE_REVISE:enterprise", _ENTERPRISE),
    PolicyRule(FlagType.PERSONALIZED_FINANCIAL_ADVICE, "warning", InterventionAction.HARD_STOP, "PERSONALIZED_FINANCIAL_ADVICE:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.PERSONALIZED_FINANCIAL_ADVICE, "warning", InterventionAction.FORCE_REVISE, "PERSONALIZED_FINANCIAL_ADVICE:warning:FORCE_REVISE:enterprise", _ENTERPRISE),
    PolicyRule(FlagType.SENSITIVE_PII_EXPOSURE, "warning", InterventionAction.HARD_STOP, "SENSITIVE_PII_EXPOSURE:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.SENSITIVE_PII_EXPOSURE, "warning", InterventionAction.FORCE_REVISE, "SENSITIVE_PII_EXPOSURE:warning:FORCE_REVISE:enterprise", _ENTERPRISE),
    PolicyRule(FlagType.UNVERIFIED_TRUTHFULNESS_CLAIM, "warning", InterventionAction.HARD_STOP, "UNVERIFIED_TRUTHFULNESS_CLAIM:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.UNVERIFIED_TRUTHFULNESS_CLAIM, "warning", InterventionAction.FORCE_REVISE, "UNVERIFIED_TRUTHFULNESS_CLAIM:warning:FORCE_REVISE:enterprise", _ENTERPRISE),
    PolicyRule(FlagType.PERSONALIZED_ACADEMIC_ADVICE, "warning", InterventionAction.HARD_STOP, "PERSONALIZED_ACADEMIC_ADVICE:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.PERSONALIZED_ACADEMIC_ADVICE, "warning", InterventionAction.FORCE_REVISE, "PERSONALIZED_ACADEMIC_ADVICE:warning:FORCE_REVISE:enterprise", _ENTERPRISE),
    PolicyRule(FlagType.PERSONALIZED_EMPLOYMENT_ADVICE, "warning", InterventionAction.HARD_STOP, "PERSONALIZED_EMPLOYMENT_ADVICE:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.PERSONALIZED_EMPLOYMENT_ADVICE, "warning", InterventionAction.FORCE_REVISE, "PERSONALIZED_EMPLOYMENT_ADVICE:warning:FORCE_REVISE:enterprise", _ENTERPRISE),
    PolicyRule(FlagType.PERSONALIZED_BUSINESS_STRATEGY_ADVICE, "warning", InterventionAction.HARD_STOP, "PERSONALIZED_BUSINESS_STRATEGY_ADVICE:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.PERSONALIZED_BUSINESS_STRATEGY_ADVICE, "warning", InterventionAction.FORCE_REVISE, "PERSONALIZED_BUSINESS_STRATEGY_ADVICE:warning:FORCE_REVISE:enterprise", _ENTERPRISE),

    # Epistemic failures — hallucination family
    PolicyRule(FlagType.CONTRADICTED_FACT, "error", InterventionAction.HARD_STOP, "CONTRADICTED_FACT:error:HARD_STOP"),
    PolicyRule(FlagType.CONTRADICTED_FACT, "warning", InterventionAction.FORCE_REVISE, "CONTRADICTED_FACT:warning:FORCE_REVISE"),
    PolicyRule(FlagType.UNBOUND_ENTITY, "warning", InterventionAction.FORCE_REVISE, "UNBOUND_ENTITY:warning:FORCE_REVISE", _PUBLIC_ENTERPRISE),
    PolicyRule(FlagType.UNBOUND_ENTITY, "warning", InterventionAction.SOFT_CORRECT, "UNBOUND_ENTITY:warning:SOFT_CORRECT:open", _OPEN),
    PolicyRule(FlagType.UNVERIFIED_FACT_ASSERTION, "warning", InterventionAction.FORCE_REVISE, "UNVERIFIED_FACT_ASSERTION:warning:FORCE_REVISE", _PUBLIC_ENTERPRISE),
    PolicyRule(FlagType.UNVERIFIED_FACT_ASSERTION, "warning", InterventionAction.SOFT_CORRECT, "UNVERIFIED_FACT_ASSERTION:warning:SOFT_CORRECT:open", _OPEN),
    PolicyRule(FlagType.DISJUNCTIVE_BRANCH_COLLAPSE, "warning", InterventionAction.CONTAIN, "DISJUNCTIVE_BRANCH_COLLAPSE:warning:CONTAIN:all_modes"),
    PolicyRule(
        FlagType.CLARIFICATION_AUTHORITY_OUTSIDE_HOLD,
        "warning",
        InterventionAction.CONTAIN,
        "CLARIFICATION_AUTHORITY_OUTSIDE_HOLD:warning:CONTAIN:all_modes",
    ),
    PolicyRule(
        FlagType.UPSTREAM_INSUFFICIENT_CONTEXT,
        "warning",
        InterventionAction.CONTAIN,
        "UPSTREAM_INSUFFICIENT_CONTEXT:warning:CONTAIN:all_modes",
    ),
    PolicyRule(
        FlagType.UNCLASSIFIED_CONSEQUENCE_INTENT,
        "warning",
        InterventionAction.CONTAIN,
        "UNCLASSIFIED_CONSEQUENCE_INTENT:warning:CONTAIN:strict",
    ),
    PolicyRule(FlagType.RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT, "warning", InterventionAction.FORCE_REVISE, "RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT:warning:FORCE_REVISE:all_modes"),
    PolicyRule(FlagType.RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT, "warning", InterventionAction.FORCE_REVISE, "RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT:warning:FORCE_REVISE:all_modes"),
    PolicyRule(FlagType.UNVERIFIED_REGULATORY_CLAIM, "warning", InterventionAction.FORCE_REVISE, "UNVERIFIED_REGULATORY_CLAIM:warning:FORCE_REVISE"),
    PolicyRule(FlagType.UNSUPPORTED_ATTRIBUTE, "warning", InterventionAction.SOFT_CORRECT, "UNSUPPORTED_ATTRIBUTE:warning:SOFT_CORRECT"),
    PolicyRule(FlagType.UNSUPPORTED_EVENT, "warning", InterventionAction.SOFT_CORRECT, "UNSUPPORTED_EVENT:warning:SOFT_CORRECT"),

    # PEF binding failures — clarification hold (aligned with StatusTranslator ASK / CONTAIN)
    PolicyRule(FlagType.UNRESOLVED_REFERENT, "warning", InterventionAction.CONTAIN, "UNRESOLVED_REFERENT:warning:CONTAIN", _PUBLIC_ENTERPRISE),
    PolicyRule(FlagType.UNRESOLVED_REFERENT, "warning", InterventionAction.CONTAIN, "UNRESOLVED_REFERENT:warning:CONTAIN:open", _OPEN),
    PolicyRule(FlagType.UNRESOLVED_COMPARAND, "warning", InterventionAction.CONTAIN, "UNRESOLVED_COMPARAND:warning:CONTAIN", _PUBLIC_ENTERPRISE),
    PolicyRule(FlagType.UNRESOLVED_COMPARAND, "warning", InterventionAction.CONTAIN, "UNRESOLVED_COMPARAND:warning:CONTAIN:open", _OPEN),
    PolicyRule(FlagType.IDENTITY_DRIFT, "warning", InterventionAction.FORCE_REVISE, "IDENTITY_DRIFT:warning:FORCE_REVISE"),
    PolicyRule(FlagType.TIME_SMEAR, "warning", InterventionAction.SOFT_CORRECT, "TIME_SMEAR:warning:SOFT_CORRECT"),
    PolicyRule(FlagType.UNRESOLVED_STATE_TRANSITION, "error", InterventionAction.HARD_STOP, "UNRESOLVED_STATE_TRANSITION:error:HARD_STOP"),
    PolicyRule(FlagType.UNRESOLVED_STATE_TRANSITION, "warning", InterventionAction.FORCE_REVISE, "UNRESOLVED_STATE_TRANSITION:warning:FORCE_REVISE"),
    PolicyRule(FlagType.UNRESOLVED_STATE_TRANSITION, "error", InterventionAction.HARD_STOP, "UNRESOLVED_STATE_TRANSITION:error:HARD_STOP"),
    PolicyRule(FlagType.UNRESOLVED_STATE_TRANSITION, "warning", InterventionAction.FORCE_REVISE, "UNRESOLVED_STATE_TRANSITION:warning:FORCE_REVISE"),
], name="strict")

DEFAULT_MODERATE = InterventionPolicy([
    # Extraction failures
    PolicyRule(FlagType.EXTRACTION_FAILED, "error", InterventionAction.HARD_STOP, "EXTRACTION_FAILED:error:HARD_STOP"),
    PolicyRule(FlagType.EXTRACTION_EMPTY, "error", InterventionAction.CONTAIN, "EXTRACTION_EMPTY:error:CONTAIN"),

    # Hard-stop always — no revise loop, no mode distinction
    PolicyRule(FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION, "error", InterventionAction.HARD_STOP, "PEDIATRIC_DOSAGE_RECOMMENDATION:error:HARD_STOP"),
    PolicyRule(FlagType.MEDICAL_DOSAGE_RECOMMENDATION, "error", InterventionAction.HARD_STOP, "MEDICAL_DOSAGE_RECOMMENDATION:error:HARD_STOP"),
    PolicyRule(FlagType.NUMERIC_MEDICAL_INSTRUCTION, "error", InterventionAction.HARD_STOP, "NUMERIC_MEDICAL_INSTRUCTION:error:HARD_STOP"),
    PolicyRule(FlagType.EMERGENCY_TRIAGE_GUIDANCE, "error", InterventionAction.HARD_STOP, "EMERGENCY_TRIAGE_GUIDANCE:error:HARD_STOP"),
    PolicyRule(FlagType.SELF_HARM_INSTRUCTION, "error", InterventionAction.HARD_STOP, "SELF_HARM_INSTRUCTION:error:HARD_STOP"),
    PolicyRule(FlagType.ILLEGAL_INSTRUCTION, "error", InterventionAction.HARD_STOP, "ILLEGAL_INSTRUCTION:error:HARD_STOP"),
    PolicyRule(FlagType.AGENCY_VIOLATION_ASSISTANCE, "error", InterventionAction.HARD_STOP, "AGENCY_VIOLATION_ASSISTANCE:error:HARD_STOP"),
    PolicyRule(FlagType.PROMPT_INJECTION_ATTEMPT, "error", InterventionAction.HARD_STOP, "PROMPT_INJECTION_ATTEMPT:error:HARD_STOP"),
    PolicyRule(FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED, "warning", InterventionAction.CONTAIN, "AGENCY_RISK_CONTEXT_UNRESOLVED:warning:CONTAIN"),
    PolicyRule(FlagType.TARGETED_DEFAMATION, "error", InterventionAction.HARD_STOP, "TARGETED_DEFAMATION:error:HARD_STOP"),
    PolicyRule(FlagType.VIOLENT_CRIMINAL_INTENT, "error", InterventionAction.HARD_STOP, "VIOLENT_CRIMINAL_INTENT:error:HARD_STOP"),
    # Education / workforce / enterprise hard-stop vetoes
    PolicyRule(FlagType.ACADEMIC_INTEGRITY_VIOLATION, "error", InterventionAction.HARD_STOP, "ACADEMIC_INTEGRITY_VIOLATION:error:HARD_STOP"),
    PolicyRule(FlagType.STUDENT_RECORD_EXPOSURE, "error", InterventionAction.HARD_STOP, "STUDENT_RECORD_EXPOSURE:error:HARD_STOP"),
    PolicyRule(FlagType.EMPLOYMENT_DISCRIMINATION_FACILITATION, "error", InterventionAction.HARD_STOP, "EMPLOYMENT_DISCRIMINATION_FACILITATION:error:HARD_STOP"),
    PolicyRule(FlagType.EMPLOYEE_RECORD_EXPOSURE, "error", InterventionAction.HARD_STOP, "EMPLOYEE_RECORD_EXPOSURE:error:HARD_STOP"),
    PolicyRule(FlagType.TRADE_SECRET_DISCLOSURE, "error", InterventionAction.HARD_STOP, "TRADE_SECRET_DISCLOSURE:error:HARD_STOP"),
    PolicyRule(FlagType.INSIDER_INFORMATION_ASSISTANCE, "error", InterventionAction.HARD_STOP, "INSIDER_INFORMATION_ASSISTANCE:error:HARD_STOP"),
    PolicyRule(FlagType.PROCUREMENT_FRAUD_FACILITATION, "error", InterventionAction.HARD_STOP, "PROCUREMENT_FRAUD_FACILITATION:error:HARD_STOP"),

    # Verify-or-refuse — mode-dependent
    PolicyRule(FlagType.PERSONALIZED_MEDICAL_ADVICE, "warning", InterventionAction.HARD_STOP, "PERSONALIZED_MEDICAL_ADVICE:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.PERSONALIZED_MEDICAL_ADVICE, "warning", InterventionAction.FORCE_REVISE, "PERSONALIZED_MEDICAL_ADVICE:warning:FORCE_REVISE:enterprise", _ENTERPRISE),
    PolicyRule(FlagType.PERSONALIZED_LEGAL_ADVICE, "warning", InterventionAction.HARD_STOP, "PERSONALIZED_LEGAL_ADVICE:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.PERSONALIZED_LEGAL_ADVICE, "warning", InterventionAction.FORCE_REVISE, "PERSONALIZED_LEGAL_ADVICE:warning:FORCE_REVISE:enterprise", _ENTERPRISE),
    PolicyRule(FlagType.PERSONALIZED_FINANCIAL_ADVICE, "warning", InterventionAction.HARD_STOP, "PERSONALIZED_FINANCIAL_ADVICE:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.PERSONALIZED_FINANCIAL_ADVICE, "warning", InterventionAction.FORCE_REVISE, "PERSONALIZED_FINANCIAL_ADVICE:warning:FORCE_REVISE:enterprise", _ENTERPRISE),
    PolicyRule(FlagType.SENSITIVE_PII_EXPOSURE, "warning", InterventionAction.HARD_STOP, "SENSITIVE_PII_EXPOSURE:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.SENSITIVE_PII_EXPOSURE, "warning", InterventionAction.FORCE_REVISE, "SENSITIVE_PII_EXPOSURE:warning:FORCE_REVISE:enterprise", _ENTERPRISE),
    PolicyRule(FlagType.UNVERIFIED_TRUTHFULNESS_CLAIM, "warning", InterventionAction.HARD_STOP, "UNVERIFIED_TRUTHFULNESS_CLAIM:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.UNVERIFIED_TRUTHFULNESS_CLAIM, "warning", InterventionAction.FORCE_REVISE, "UNVERIFIED_TRUTHFULNESS_CLAIM:warning:FORCE_REVISE:enterprise", _ENTERPRISE),
    PolicyRule(FlagType.PERSONALIZED_ACADEMIC_ADVICE, "warning", InterventionAction.HARD_STOP, "PERSONALIZED_ACADEMIC_ADVICE:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.PERSONALIZED_ACADEMIC_ADVICE, "warning", InterventionAction.FORCE_REVISE, "PERSONALIZED_ACADEMIC_ADVICE:warning:FORCE_REVISE:enterprise", _ENTERPRISE),
    PolicyRule(FlagType.PERSONALIZED_EMPLOYMENT_ADVICE, "warning", InterventionAction.HARD_STOP, "PERSONALIZED_EMPLOYMENT_ADVICE:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.PERSONALIZED_EMPLOYMENT_ADVICE, "warning", InterventionAction.FORCE_REVISE, "PERSONALIZED_EMPLOYMENT_ADVICE:warning:FORCE_REVISE:enterprise", _ENTERPRISE),
    PolicyRule(FlagType.PERSONALIZED_BUSINESS_STRATEGY_ADVICE, "warning", InterventionAction.HARD_STOP, "PERSONALIZED_BUSINESS_STRATEGY_ADVICE:warning:HARD_STOP:public", _PUBLIC),
    PolicyRule(FlagType.PERSONALIZED_BUSINESS_STRATEGY_ADVICE, "warning", InterventionAction.FORCE_REVISE, "PERSONALIZED_BUSINESS_STRATEGY_ADVICE:warning:FORCE_REVISE:enterprise", _ENTERPRISE),

    # Epistemic failures — hallucination family
    PolicyRule(FlagType.CONTRADICTED_FACT, "error", InterventionAction.FORCE_REVISE, "CONTRADICTED_FACT:error:FORCE_REVISE"),
    PolicyRule(FlagType.CONTRADICTED_FACT, "warning", InterventionAction.SOFT_CORRECT, "CONTRADICTED_FACT:warning:SOFT_CORRECT"),
    PolicyRule(FlagType.UNBOUND_ENTITY, "warning", InterventionAction.SOFT_CORRECT, "UNBOUND_ENTITY:warning:SOFT_CORRECT"),
    PolicyRule(FlagType.UNVERIFIED_FACT_ASSERTION, "warning", InterventionAction.SOFT_CORRECT, "UNVERIFIED_FACT_ASSERTION:warning:SOFT_CORRECT"),
    PolicyRule(FlagType.DISJUNCTIVE_BRANCH_COLLAPSE, "warning", InterventionAction.CONTAIN, "DISJUNCTIVE_BRANCH_COLLAPSE:warning:CONTAIN:moderate"),
    PolicyRule(
        FlagType.CLARIFICATION_AUTHORITY_OUTSIDE_HOLD,
        "warning",
        InterventionAction.CONTAIN,
        "CLARIFICATION_AUTHORITY_OUTSIDE_HOLD:warning:CONTAIN:moderate",
    ),
    PolicyRule(
        FlagType.UPSTREAM_INSUFFICIENT_CONTEXT,
        "warning",
        InterventionAction.CONTAIN,
        "UPSTREAM_INSUFFICIENT_CONTEXT:warning:CONTAIN:moderate",
    ),
    PolicyRule(FlagType.RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT, "warning", InterventionAction.FORCE_REVISE, "RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT:warning:FORCE_REVISE:moderate"),
    PolicyRule(FlagType.RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT, "warning", InterventionAction.FORCE_REVISE, "RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT:warning:FORCE_REVISE:moderate"),
    PolicyRule(FlagType.UNVERIFIED_REGULATORY_CLAIM, "warning", InterventionAction.SOFT_CORRECT, "UNVERIFIED_REGULATORY_CLAIM:warning:SOFT_CORRECT"),
    PolicyRule(FlagType.UNSUPPORTED_ATTRIBUTE, "warning", InterventionAction.SOFT_CORRECT, "UNSUPPORTED_ATTRIBUTE:warning:SOFT_CORRECT"),
    PolicyRule(FlagType.UNSUPPORTED_EVENT, "warning", InterventionAction.SOFT_CORRECT, "UNSUPPORTED_EVENT:warning:SOFT_CORRECT"),

    # PEF binding failures — clarification hold
    PolicyRule(FlagType.UNRESOLVED_REFERENT, "warning", InterventionAction.CONTAIN, "UNRESOLVED_REFERENT:warning:CONTAIN"),
    PolicyRule(FlagType.UNRESOLVED_COMPARAND, "warning", InterventionAction.CONTAIN, "UNRESOLVED_COMPARAND:warning:CONTAIN"),
    PolicyRule(FlagType.IDENTITY_DRIFT, "warning", InterventionAction.SOFT_CORRECT, "IDENTITY_DRIFT:warning:SOFT_CORRECT"),
    PolicyRule(FlagType.TIME_SMEAR, "warning", InterventionAction.SOFT_CORRECT, "TIME_SMEAR:warning:SOFT_CORRECT"),
    PolicyRule(FlagType.UNRESOLVED_STATE_TRANSITION, "warning", InterventionAction.FORCE_REVISE, "UNRESOLVED_STATE_TRANSITION:warning:FORCE_REVISE"),
    PolicyRule(FlagType.UNRESOLVED_STATE_TRANSITION, "error", InterventionAction.HARD_STOP, "UNRESOLVED_STATE_TRANSITION:error:HARD_STOP"),
    PolicyRule(FlagType.UNRESOLVED_STATE_TRANSITION, "warning", InterventionAction.FORCE_REVISE, "UNRESOLVED_STATE_TRANSITION:warning:FORCE_REVISE"),
], name="moderate")
