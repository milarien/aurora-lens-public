"""StatusTranslator — computes terminal LensStatus from verification outcome.

ROLE
----
StatusTranslator answers exactly one question:
  "What is the terminal epistemic state that Lens has reached for this turn?"

It does NOT answer: "What should the system do about it?"
That is the Governor's job.

INPUT
-----
  flags: list[Flag]   — all verification flags raised for this turn
  mode:  str          — deployment policy mode ("public" | "enterprise" | "open")

OUTPUT
------
  LensStatus — the single terminal admissibility verdict for the turn.

WORST-CLASS INVARIANT (critical — write this down for future maintainers)
---------
Priority is computed over terminal epistemic CLASSES, not over raw flag count,
flag order, or incidental iteration sequence. The four classes are:

  STOP   (3) — determination blocked; conversation may terminate
  REFUSE (2) — determination blocked; interaction continues
  ASK    (1) — clarification needed; no determination yet
  ADMIT  (0) — determination is admissible

"Worst wins" means the highest-priority class is returned, regardless of how
many flags produce lower-priority classes. A single STOP flag beats any number
of REFUSE or ASK flags. This ordering is enforced by integer comparison, not by
iteration order.

A later maintainer who writes "first hard flag encountered wins" will introduce
nondeterministic behaviour that depends on flag extraction ordering — which may
differ between spaCy and LLM backends. Do not do this.

MODE SENSITIVITY
----------------
Mode affects LensStatus for the verify-or-refuse class only
(PERSONALIZED_MEDICAL_ADVICE, PERSONALIZED_LEGAL_ADVICE,
PERSONALIZED_FINANCIAL_ADVICE, SENSITIVE_PII_EXPOSURE).

This is a deliberate architectural choice: in public deployments, the operator
has not established a safety pathway for individualized determinations, making
them inadmissible (STOP). In enterprise/open deployments, the operator has
accepted responsibility for downstream workflows, so the determination is
blocked (REFUSE) but the conversation is not terminated.

If you believe mode is purely a continuation concern and should live entirely in
PolicyResolver, move the verify-or-refuse class there and remove mode from this
interface. The trade-off is that PolicyResolver would need mode as an additional
key in its matrix, which it currently does not have.

EXTRACTION_FAILED INVARIANT (stated explicitly, not by convenience)
--------------------------------------------------------------------
EXTRACTION_FAILED always produces STOP regardless of mode, domain, or authority
class. Rationale: a failed extraction means verification cannot be completed.
Proceeding without verification would silently bypass governance. This is not
the same as "temporarily empty" (EXTRACTION_EMPTY → ASK) — it means the
verification substrate itself failed. If a deployment wants different handling,
it should address the extraction failure at source, not by weakening this rule.
"""

from __future__ import annotations

from aurora_lens.governor.models import LensStatus
from aurora_lens.verify.flags import Flag, FlagType


# ── Epistemic class sets ──────────────────────────────────────────────────────

# STOP always — any severity, any mode, any domain, any authority class.
# These represent content vetoes or substrate failures where proceeding without
# a valid verification outcome would silently bypass governance.
_HARD_STOP_ALWAYS: frozenset[FlagType] = frozenset({
    FlagType.EXTRACTION_FAILED,             # See EXTRACTION_FAILED INVARIANT above.
    FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
    FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
    FlagType.NUMERIC_MEDICAL_INSTRUCTION,
    FlagType.EMERGENCY_TRIAGE_GUIDANCE,
    FlagType.SELF_HARM_INSTRUCTION,
    FlagType.ILLEGAL_INSTRUCTION,
    FlagType.HAZARD_ONTOLOGY_UNAVAILABLE,
    FlagType.AGENCY_VIOLATION_ASSISTANCE,
    FlagType.PROMPT_INJECTION_ATTEMPT,
    FlagType.TARGETED_DEFAMATION,
    # Education / workforce / enterprise compliance vetoes
    FlagType.ACADEMIC_INTEGRITY_VIOLATION,
    FlagType.STUDENT_RECORD_EXPOSURE,
    FlagType.EMPLOYMENT_DISCRIMINATION_FACILITATION,
    FlagType.EMPLOYEE_RECORD_EXPOSURE,
    FlagType.TRADE_SECRET_DISCLOSURE,
    FlagType.INSIDER_INFORMATION_ASSISTANCE,
    FlagType.PROCUREMENT_FRAUD_FACILITATION,
})

# STOP in public mode; REFUSE in enterprise/open mode.
# See MODE SENSITIVITY note above.
_VERIFY_OR_REFUSE: frozenset[FlagType] = frozenset({
    FlagType.PERSONALIZED_MEDICAL_ADVICE,
    FlagType.PERSONALIZED_LEGAL_ADVICE,
    FlagType.PERSONALIZED_FINANCIAL_ADVICE,
    FlagType.SENSITIVE_PII_EXPOSURE,
    FlagType.UNVERIFIED_TRUTHFULNESS_CLAIM,
    FlagType.PERSONALIZED_ACADEMIC_ADVICE,
    FlagType.PERSONALIZED_EMPLOYMENT_ADVICE,
    FlagType.PERSONALIZED_BUSINESS_STRATEGY_ADVICE,
})

# REFUSE in all modes regardless of severity.
_ALWAYS_REFUSE: frozenset[FlagType] = frozenset({
    FlagType.IDENTITY_DRIFT,
    FlagType.UNVERIFIED_REGULATORY_CLAIM,
    # RAG harness manifest U1/U2 — bounded absence must not ADMIT as PASS in open mode.
    FlagType.RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT,
    FlagType.RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT,
})

# REFUSE in public/enterprise; ADMIT in open mode.
# In open mode the PEF starts empty and epistemic failures produce annotations
# (SOFT_CORRECT) rather than blocking. StatusTranslator maps this to ADMIT
# because the LLM's output is admitted with a governance note, not blocked.
_EPISTEMIC_REFUSE_OR_ADMIT: frozenset[FlagType] = frozenset({
    FlagType.UNBOUND_ENTITY,
    FlagType.UNSUPPORTED_ATTRIBUTE,
    FlagType.UNSUPPORTED_EVENT,
    FlagType.UNVERIFIED_FACT_ASSERTION,
    FlagType.TIME_SMEAR,
})

# ASK in all modes — interaction open, clarification needed.
# EXTRACTION_EMPTY: the extractor ran but found nothing to verify.
# This is not a hard failure (STOP) — the input may be ambiguous or short.
# The interaction remains open for clarification.
# UNRESOLVED_REFERENT / UNRESOLVED_COMPARAND: PEF cannot resolve the referent;
# the world model needs more information before a determination is possible.
# RAG C2 disjunctive harness — containment / ask, not PASS (manifest C2).
_ASK: frozenset[FlagType] = frozenset({
    FlagType.EXTRACTION_EMPTY,
    FlagType.UNRESOLVED_REFERENT,
    FlagType.UNRESOLVED_COMPARAND,
    FlagType.DISJUNCTIVE_BRANCH_COLLAPSE,
    FlagType.CLARIFICATION_AUTHORITY_OUTSIDE_HOLD,
    FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED,
    FlagType.HAZARD_SUBSTANCE_UNRESOLVED,
    FlagType.UPSTREAM_INSUFFICIENT_CONTEXT,
    FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED,
    FlagType.UNCLASSIFIED_CONSEQUENCE_INTENT,
    FlagType.INTERPRETATION_LIMIT,
})

# Priority values for the worst-class invariant.
_STATUS_PRIORITY: dict[LensStatus, int] = {
    LensStatus.STOP:   3,
    LensStatus.REFUSE: 2,
    LensStatus.ASK:    1,
    LensStatus.ADMIT:  0,
}


def _flag_to_status(flag: Flag, mode: str) -> LensStatus:
    """Map a single flag to its LensStatus given the deployment mode.

    Internal helper. Do not call directly — use StatusTranslator.translate().
    """
    ft = flag.flag_type

    if ft in _HARD_STOP_ALWAYS:
        return LensStatus.STOP

    if ft in (FlagType.CONTRADICTED_FACT, FlagType.CONTRADICTS_COMMITTED_STATE):
        # Error severity: substrate can prove contradiction → STOP.
        # Warning severity: probable contradiction, not proven → REFUSE.
        return LensStatus.STOP if flag.severity == "error" else LensStatus.REFUSE

    if ft in _VERIFY_OR_REFUSE:
        return LensStatus.STOP if mode == "public" else LensStatus.REFUSE

    if ft in _ALWAYS_REFUSE:
        return LensStatus.REFUSE

    if ft in _EPISTEMIC_REFUSE_OR_ADMIT:
        return LensStatus.REFUSE if mode in ("public", "enterprise") else LensStatus.ADMIT

    if ft in _ASK:
        return LensStatus.ASK

    # Unknown flag type — safe default is REFUSE (not ADMIT, not STOP).
    # Unknown flags should not silently pass, but should not terminate either.
    return LensStatus.REFUSE


class StatusTranslator:
    """Translates a list of verification flags into a single terminal LensStatus.

    The translator operates on the AGGREGATE turn outcome, not per-flag.
    Multiple flags can coexist; the worst epistemic class wins (see module
    docstring for the invariant).

    Usage:
        translator = StatusTranslator()
        status = translator.translate(flags, mode="public")
    """

    _VALID_MODES = frozenset({"public", "enterprise", "open"})

    def translate(self, flags: list[Flag], mode: str = "public") -> LensStatus:
        """Compute the terminal LensStatus for a turn.

        Args:
            flags: All verification flags raised against this turn's LLM output.
            mode:  Deployment policy mode ("public" | "enterprise" | "open").
                   This is a Lens-context input — it determines admissibility
                   for the verify-or-refuse class. Never inferred from text.

        Returns:
            The terminal LensStatus. Priority: STOP > REFUSE > ASK > ADMIT.
            Returns ADMIT when flags is empty.

        Raises:
            ValueError: If mode is not one of the three valid strings.
                        Unknown modes are rejected explicitly — they must never
                        silently degrade to a less restrictive status class.
                        A typo in config (e.g. "prod", "PUBLIC") that reaches
                        this layer would otherwise route verify-or-refuse flags
                        to REFUSE instead of STOP, silently weakening governance.

        INVARIANT: Result depends only on the SET of (flag_type, severity) pairs
        and the mode. It does not depend on the order flags appear in the list.
        """
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Unknown deployment mode: {mode!r}. "
                f"Expected one of: {sorted(self._VALID_MODES)}. "
                "An unrecognised mode must be rejected explicitly — it cannot "
                "silently fall through to a less restrictive status class."
            )

        if not flags:
            return LensStatus.ADMIT

        # Compute the status for each flag independently, then take the worst.
        # Sorting by priority (descending) and returning the first is equivalent
        # to max(), but makes the invariant explicit: order does not matter.
        worst = LensStatus.ADMIT
        worst_priority = _STATUS_PRIORITY[LensStatus.ADMIT]

        for flag in flags:
            status = _flag_to_status(flag, mode)
            priority = _STATUS_PRIORITY[status]
            if priority > worst_priority:
                worst = status
                worst_priority = priority
                if worst_priority == _STATUS_PRIORITY[LensStatus.STOP]:
                    # Short-circuit: nothing beats STOP.
                    break

        return worst
