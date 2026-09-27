"""ContextResolver — supplies (Domain, AuthorityClass, UserClass) from request context.

ROLE
----
ContextResolver answers: "What governance corridor are we in?"

It does NOT answer: "What should the system do?"
That is the Governor's job.

INPUTS (in cascade priority order)
-----------------------------------
1. Protected flag evidence — legal/medical/finance/harmful/ambiguity inferred from flags
   (domain is evidence; route hints must not downgrade these corridors)

2. Operator channel hint — domain_var ContextVar set by route/header
   (used only when no high-stakes flag domain is present)

3. Default — Domain.GENERAL, AuthorityClass.GP, UserClass.GENERAL

PROVENANCE
----------
Every resolved context carries a ContextResolutionProvenance record that
names where each value came from. This makes forensic reading possible —
reviewers can see whether a domain was deliberately routed or fell through
to a flag-pattern guess. See runtime_types.ContextResolutionProvenance.

NEVER-RAISES CONTRACT
---------------------
ContextResolver must never raise an exception. All four cascade levels end in
safe defaults. Governance context resolution should degrade gracefully, not
explode theatrically. If a ContextVar contains an invalid value, fall through
to the next level rather than raising.

BLEED PREVENTION
----------------
The three ContextVars (domain_var, authority_class_var, user_class_var) must be
reset in a finally block after each request. The proxy is responsible for this
using the standard token.set() / token.reset() pattern already used for
trace_id_var and session_id_var. Failure to reset will bleed one request's
corridor into the next — including authority upgrades, which would be a
governance bug with real consequences.
"""

from __future__ import annotations

from aurora_lens.governor.models import AuthorityClass, Domain, UserClass
from aurora_lens.verify.flags import Flag, FlagType
from aurora_lens.govern.adapters.runtime_types import ContextResolutionProvenance

# Import the three ContextVars added to context.py.
# These are populated by the proxy request lifecycle.
from aurora_lens.context import (
    domain_var,
    authority_class_var,
    user_class_var,
)


# ── Flag-type → Domain mapping (declarative table, not scattered through code) ─

# Priority within the table: flags are matched in definition order when
# multiple flags are present. STOP-class flags take domain precedence
# (they are listed first) over REFUSE/ASK-class flags.
# This is intentional: if a medical dosage flag and an epistemic flag both
# fire, the domain should be medical (more restrictive corridor), not general.

_FLAG_DOMAIN_PRIORITY: list[tuple[FlagType, Domain]] = [
    # Medical — hard-stop-always content vetoes
    (FlagType.SELF_HARM_INSTRUCTION,           Domain.MEDICAL),
    (FlagType.MEDICAL_DOSAGE_RECOMMENDATION,   Domain.MEDICAL),
    (FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION, Domain.MEDICAL),
    (FlagType.NUMERIC_MEDICAL_INSTRUCTION,     Domain.MEDICAL),
    (FlagType.EMERGENCY_TRIAGE_GUIDANCE,       Domain.MEDICAL),
    # Legal — hard-stop-always content vetoes
    # Generic ILLEGAL_INSTRUCTION stays LEGAL (illegal-instruction corridor).
    # Hazard ontology rule IDs override to HARMFUL via _HAZARD_HARMFUL_RULE_IDS
    # before this FlagType map is consulted.
    (FlagType.ILLEGAL_INSTRUCTION,             Domain.LEGAL),
    (FlagType.HAZARD_ONTOLOGY_UNAVAILABLE,     Domain.HARMFUL),
    (FlagType.TARGETED_DEFAMATION,             Domain.LEGAL),
    (FlagType.SENSITIVE_PII_EXPOSURE,          Domain.LEGAL),   # data-protection is a legal/compliance domain
    # Medical — verify-or-refuse
    (FlagType.PERSONALIZED_MEDICAL_ADVICE,     Domain.MEDICAL),
    # Legal — verify-or-refuse
    (FlagType.PERSONALIZED_LEGAL_ADVICE,       Domain.LEGAL),
    # Finance — verify-or-refuse
    (FlagType.PERSONALIZED_FINANCIAL_ADVICE,   Domain.FINANCE),
    # Education — hard-stop
    (FlagType.ACADEMIC_INTEGRITY_VIOLATION,    Domain.EDUCATION),
    (FlagType.STUDENT_RECORD_EXPOSURE,         Domain.EDUCATION),
    # Workforce — hard-stop
    (FlagType.EMPLOYMENT_DISCRIMINATION_FACILITATION, Domain.WORKFORCE),
    (FlagType.EMPLOYEE_RECORD_EXPOSURE,        Domain.WORKFORCE),
    # Enterprise — hard-stop
    (FlagType.TRADE_SECRET_DISCLOSURE,         Domain.ENTERPRISE),
    (FlagType.INSIDER_INFORMATION_ASSISTANCE,  Domain.ENTERPRISE),
    (FlagType.PROCUREMENT_FRAUD_FACILITATION,  Domain.ENTERPRISE),
    # Education — verify-or-refuse
    (FlagType.PERSONALIZED_ACADEMIC_ADVICE,    Domain.EDUCATION),
    # Workforce — verify-or-refuse
    (FlagType.PERSONALIZED_EMPLOYMENT_ADVICE,  Domain.WORKFORCE),
    # Enterprise — verify-or-refuse
    (FlagType.PERSONALIZED_BUSINESS_STRATEGY_ADVICE, Domain.ENTERPRISE),
    # Research — regulatory/epistemic claims
    (FlagType.UNVERIFIED_REGULATORY_CLAIM,     Domain.RESEARCH),
    (FlagType.RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT, Domain.RESEARCH),
    (FlagType.RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT, Domain.RESEARCH),
    # Ambiguity — PEF binding failures that need clarification
    (FlagType.UNRESOLVED_REFERENT,             Domain.AMBIGUITY),
    (FlagType.UNRESOLVED_COMPARAND,            Domain.AMBIGUITY),
    (FlagType.CLARIFICATION_AUTHORITY_OUTSIDE_HOLD, Domain.AMBIGUITY),
    (FlagType.DISJUNCTIVE_BRANCH_COLLAPSE,     Domain.AMBIGUITY),
    (FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED,  Domain.AMBIGUITY),
    (FlagType.HAZARD_SUBSTANCE_UNRESOLVED,     Domain.AMBIGUITY),
    (FlagType.UPSTREAM_INSUFFICIENT_CONTEXT,   Domain.AMBIGUITY),
    # General — extraction / epistemic failures with no domain signal
    # (not listed — these fall through to default Domain.GENERAL)
]

# Convert to a dict for O(1) lookup, preserving first-match priority.
_FLAG_DOMAIN_MAP: dict[FlagType, Domain] = {}
for _ft, _dom in _FLAG_DOMAIN_PRIORITY:
    if _ft not in _FLAG_DOMAIN_MAP:
        _FLAG_DOMAIN_MAP[_ft] = _dom

_PROTECTED_INFERRED_DOMAINS: frozenset[Domain] = frozenset({
    Domain.MEDICAL,
    Domain.LEGAL,
    Domain.FINANCE,
    Domain.HARMFUL,
    Domain.AMBIGUITY,
    Domain.EDUCATION,
    Domain.WORKFORCE,
    Domain.ENTERPRISE,
})


# Hazard ontology / illicit-synthesis rule ids — canonical Domain.HARMFUL.
# Checked before FlagType.ILLEGAL_INSTRUCTION so hazard stops never inherit LEGAL
# merely by sharing the illegal-instruction flag corridor.
_HAZARD_HARMFUL_RULE_IDS: frozenset[str] = frozenset({
    "blocked.request.hazard_operational_transform",
    "blocked.request.hazard_ontology_unavailable",
    "blocked.request.illegal_hazardous_synthesis",
    "blocked.request.bioweapon_generic_fabrication",
})


def _resolve_protected_flag_domain(flags: list[Flag]) -> Domain | None:
    """Return inferred protected domain from flags, if present."""
    for flag in flags:
        rid = str(flag.rule_id) if flag.rule_id else ""
        if rid in _HAZARD_HARMFUL_RULE_IDS:
            return Domain.HARMFUL

    # Flags are checked against _FLAG_DOMAIN_MAP in table order, not input order,
    # to ensure deterministic domain classification regardless of extraction order.
    for flag_type, domain in _FLAG_DOMAIN_MAP.items():
        if domain not in _PROTECTED_INFERRED_DOMAINS:
            continue
        for flag in flags:
            if flag.flag_type == flag_type:
                return domain
    return None


def _resolve_domain(flags: list[Flag]) -> tuple[Domain, str]:
    """Resolve Domain using high-stakes evidence + operator channel + default."""
    # Cross-domain agency-veto requests must not be downgraded into route hints.
    if any(
        f.flag_type in (
            FlagType.AGENCY_VIOLATION_ASSISTANCE,
            FlagType.AGENCY_RISK_CONTEXT_UNRESOLVED,
        )
        for f in flags
    ):
        return Domain.GENERAL, "flag_pattern"

    # Level 1: Protected domain from flag evidence.
    inferred = _resolve_protected_flag_domain(flags)
    if inferred is not None:
        return inferred, "flag_pattern"

    # Level 2: Operator channel hint via ContextVar (set by route handler/header).
    raw_domain = domain_var.get(None)
    if raw_domain is not None:
        try:
            if isinstance(raw_domain, Domain):
                return raw_domain, "operator_channel"
            return Domain(str(raw_domain)), "operator_channel"
        except (ValueError, KeyError):
            pass  # Invalid value — fall through

    # Level 3: Default.
    return Domain.GENERAL, "default"


def _resolve_authority_class() -> tuple[AuthorityClass, str]:
    """Resolve AuthorityClass from ContextVar set at startup. Returns (cls, source)."""
    raw = authority_class_var.get(None)
    if raw is not None:
        try:
            if isinstance(raw, AuthorityClass):
                return raw, "config"
            return AuthorityClass(str(raw)), "config"
        except (ValueError, KeyError):
            pass  # Invalid config value — fall through to default
    return AuthorityClass.GP, "default"


def _resolve_user_class() -> tuple[UserClass, str]:
    """Resolve UserClass from per-request ContextVar. Returns (cls, source)."""
    raw = user_class_var.get(None)
    if raw is not None:
        try:
            if isinstance(raw, UserClass):
                return raw, "header"
            return UserClass(str(raw)), "header"
        except (ValueError, KeyError):
            pass  # Invalid header value — fall through to default
    return UserClass.GENERAL, "default"


class ContextResolver:
    """Resolves governance corridor context from request/config/rule/flag sources.

    Returns (Domain, AuthorityClass, UserClass, ContextResolutionProvenance).
    Never raises. Always returns safe defaults if cascade fails at all levels.

    Usage:
        resolver = ContextResolver()
        domain, authority, user_class, provenance = resolver.resolve(flags)
    """

    def resolve(
        self,
        flags: list[Flag],
    ) -> tuple[Domain, AuthorityClass, UserClass, ContextResolutionProvenance]:
        """Resolve the full governance corridor context.

        Args:
            flags: Verification flags for the current turn (used for flag-pattern
                   fallback only; never used to infer AuthorityClass or UserClass).

        Returns:
            (Domain, AuthorityClass, UserClass, ContextResolutionProvenance)

        NEVER-RAISES CONTRACT: Any exception at any level falls through to the
        next level. The final level always returns safe defaults.
        """
        try:
            domain, domain_source = _resolve_domain(flags)
        except Exception:
            domain, domain_source = Domain.GENERAL, "default"

        try:
            authority, authority_source = _resolve_authority_class()
        except Exception:
            authority, authority_source = AuthorityClass.GP, "default"

        try:
            user_class, user_class_source = _resolve_user_class()
        except Exception:
            user_class, user_class_source = UserClass.GENERAL, "default"

        provenance = ContextResolutionProvenance(
            domain_source=domain_source,
            authority_source=authority_source,
            user_class_source=user_class_source,
        )

        return domain, authority, user_class, provenance
