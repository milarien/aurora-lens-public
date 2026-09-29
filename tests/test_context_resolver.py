"""Unit tests for ContextResolver.

KEY CONTRACTS UNDER TEST
------------------------
1. Never raises — always returns safe defaults at the bottom of the cascade.
2. Cascade priority: explicit context var > flag-pattern fallback > default.
3. Provenance records where each value came from.
4. ContextVar isolation — setting vars in one test does not bleed into another.
"""

import pytest
import contextvars

from aurora_lens.govern.adapters.context_resolver import ContextResolver
from aurora_lens.verify.flags import Flag, FlagType
from aurora_lens.context import domain_var, authority_class_var, user_class_var
from aurora_lens.governor.models import AuthorityClass, Domain, UserClass


def _flag(flag_type: FlagType, severity: str = "warning") -> Flag:
    return Flag(
        flag_type=flag_type,
        entity_name="test",
        claim="test claim",
        evidence="test evidence",
        severity=severity,
    )


@pytest.fixture
def resolver() -> ContextResolver:
    return ContextResolver()


@pytest.fixture(autouse=True)
def reset_context_vars():
    """Reset all three ContextVars before each test to prevent bleed."""
    tok_domain = domain_var.set(None)
    tok_auth = authority_class_var.set(None)
    tok_user = user_class_var.set(None)
    yield
    domain_var.reset(tok_domain)
    authority_class_var.reset(tok_auth)
    user_class_var.reset(tok_user)


# ── Safe defaults ─────────────────────────────────────────────────────────────

def test_no_context_no_flags_returns_defaults(resolver):
    """No ContextVars set, no flags → safe defaults across the board."""
    domain, authority, user_class, prov = resolver.resolve([])
    assert domain == Domain.GENERAL
    assert authority == AuthorityClass.GP
    assert user_class == UserClass.GENERAL


def test_provenance_reports_default_sources(resolver):
    """All sources report 'default' when nothing is set."""
    _, _, _, prov = resolver.resolve([])
    assert prov.domain_source == "default"
    assert prov.authority_source == "default"
    assert prov.user_class_source == "default"


# ── Flag-pattern fallback ─────────────────────────────────────────────────────

@pytest.mark.parametrize("flag_type,expected_domain", [
    (FlagType.SELF_HARM_INSTRUCTION,           Domain.MEDICAL),
    (FlagType.MEDICAL_DOSAGE_RECOMMENDATION,   Domain.MEDICAL),
    (FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION, Domain.MEDICAL),
    (FlagType.EMERGENCY_TRIAGE_GUIDANCE,       Domain.MEDICAL),
    (FlagType.PERSONALIZED_MEDICAL_ADVICE,     Domain.MEDICAL),
    (FlagType.ILLEGAL_INSTRUCTION,             Domain.LEGAL),
    (FlagType.TARGETED_DEFAMATION,             Domain.LEGAL),
    (FlagType.PERSONALIZED_LEGAL_ADVICE,       Domain.LEGAL),
    (FlagType.PERSONALIZED_FINANCIAL_ADVICE,   Domain.FINANCE),
    (FlagType.UNRESOLVED_REFERENT,             Domain.AMBIGUITY),
    (FlagType.UNRESOLVED_COMPARAND,            Domain.AMBIGUITY),
])
def test_protected_flag_pattern_correct_domain(resolver, flag_type, expected_domain):
    """Protected flag-pattern inference maps to regulated and ambiguity domains."""
    domain, _, _, prov = resolver.resolve([_flag(flag_type)])
    assert domain == expected_domain, (
        f"Expected {expected_domain} for {flag_type.name}, got {domain}"
    )
    assert prov.domain_source == "flag_pattern"


def test_non_high_stakes_flags_fall_through_to_general_domain(resolver):
    """Non-regulated flags are not used for domain inference and fall through to GENERAL."""
    domain, _, _, prov = resolver.resolve([_flag(FlagType.UNBOUND_ENTITY)])
    assert domain == Domain.GENERAL
    assert prov.domain_source == "default"

    domain, _, _, prov = resolver.resolve([_flag(FlagType.UNVERIFIED_REGULATORY_CLAIM)])
    assert domain == Domain.GENERAL
    assert prov.domain_source == "default"


def test_flag_pattern_priority_stop_class_wins_domain(resolver):
    """When multiple flags present, STOP-class flag domain wins over REFUSE-class.

    SELF_HARM_INSTRUCTION (medical, STOP-class) should win over
    UNBOUND_ENTITY (no domain signal) regardless of list order.
    """
    flags = [
        _flag(FlagType.UNBOUND_ENTITY),       # no domain
        _flag(FlagType.SELF_HARM_INSTRUCTION, "error"),  # medical
    ]
    domain, _, _, prov = resolver.resolve(flags)
    assert domain == Domain.MEDICAL
    assert prov.domain_source == "flag_pattern"


def test_legal_flag_present_with_medical_flag(resolver):
    """When both legal and medical flags present, table priority (STOP-class first) wins."""
    flags = [
        _flag(FlagType.PERSONALIZED_LEGAL_ADVICE),   # legal
        _flag(FlagType.SELF_HARM_INSTRUCTION, "error"),  # medical, higher priority
    ]
    domain, _, _, prov = resolver.resolve(flags)
    # SELF_HARM_INSTRUCTION is listed first in _FLAG_DOMAIN_PRIORITY
    assert domain == Domain.MEDICAL
    assert prov.domain_source == "flag_pattern"


# ── Operator channel + high-stakes precedence ────────────────────────────────

def test_high_stakes_flag_pattern_wins_over_operator_channel(resolver):
    """High-stakes legal/medical/finance flags must outrank operator channel hints."""
    domain_var.set("legal")
    domain, _, _, prov = resolver.resolve([_flag(FlagType.SELF_HARM_INSTRUCTION)])
    assert domain == Domain.MEDICAL
    assert prov.domain_source == "flag_pattern"


def test_operator_channel_domain_enum_value_accepted(resolver):
    """domain_var can be set to a Domain enum instance directly."""
    from aurora_lens.governor.models import Domain as D
    domain_var.set(D.FINANCE)
    domain, _, _, prov = resolver.resolve([])
    assert domain == Domain.FINANCE
    assert prov.domain_source == "operator_channel"


def test_non_high_stakes_flag_does_not_override_operator_channel(resolver):
    """Only high-stakes regulated domains override operator channel selection."""
    domain_var.set("finance")
    domain, _, _, prov = resolver.resolve([_flag(FlagType.UNBOUND_ENTITY)])
    assert domain == Domain.FINANCE
    assert prov.domain_source == "operator_channel"


def test_authority_class_contextvar_read(resolver):
    """authority_class_var set to 'DA' → AuthorityClass.DA returned."""
    authority_class_var.set("DA")
    _, authority, _, prov = resolver.resolve([])
    assert authority == AuthorityClass.DA
    assert prov.authority_source == "config"


def test_user_class_contextvar_clinician(resolver):
    """user_class_var set to 'clinician' → UserClass.CLINICIAN returned."""
    user_class_var.set("clinician")
    _, _, user_class, prov = resolver.resolve([])
    assert user_class == UserClass.CLINICIAN
    assert prov.user_class_source == "header"


def test_user_class_contextvar_auditor(resolver):
    """user_class_var set to 'auditor' → UserClass.AUDITOR returned."""
    user_class_var.set("auditor")
    _, _, user_class, _ = resolver.resolve([])
    assert user_class == UserClass.AUDITOR


# ── Never-raises contract ─────────────────────────────────────────────────────

def test_invalid_domain_contextvar_falls_through_to_flag_pattern(resolver):
    """Invalid domain_var value falls through to flag-pattern fallback, not exception."""
    domain_var.set("not_a_real_domain_xyz")
    domain, _, _, prov = resolver.resolve([_flag(FlagType.ILLEGAL_INSTRUCTION)])
    # Falls to flag-pattern
    assert domain == Domain.LEGAL
    assert prov.domain_source == "flag_pattern"


def test_no_high_stakes_and_invalid_operator_channel_falls_to_default(resolver):
    """Invalid operator channel with no high-stakes flags must fall back to GENERAL."""
    domain_var.set("not_a_real_domain_xyz")
    domain, _, _, prov = resolver.resolve([_flag(FlagType.UNBOUND_ENTITY)])
    assert domain == Domain.GENERAL
    assert prov.domain_source == "default"


def test_invalid_authority_class_contextvar_falls_through(resolver):
    """Invalid authority_class_var value falls through to default GP."""
    authority_class_var.set("NOT_A_CLASS")
    _, authority, _, prov = resolver.resolve([])
    assert authority == AuthorityClass.GP
    assert prov.authority_source == "default"


def test_invalid_user_class_contextvar_falls_through(resolver):
    """Invalid user_class_var value falls through to default GENERAL."""
    user_class_var.set("NOT_A_USER_CLASS")
    _, _, user_class, prov = resolver.resolve([])
    assert user_class == UserClass.GENERAL
    assert prov.user_class_source == "default"


def test_resolver_never_raises_with_empty_flags(resolver):
    """Resolver never raises even with completely empty state."""
    result = resolver.resolve([])
    assert len(result) == 4  # domain, authority, user_class, provenance


# ── ContextVar isolation (bleed prevention) ───────────────────────────────────

def test_contextvar_does_not_bleed_between_calls(resolver):
    """ContextVar set in one call does not persist to the next call.

    This test verifies the autouse reset_context_vars fixture is working.
    The fixture resets vars before each test — if vars were bleeding, this
    test (which doesn't set any vars) would inherit stale values.
    """
    domain, authority, user_class, prov = resolver.resolve([])
    assert domain == Domain.GENERAL
    assert authority == AuthorityClass.GP
    assert user_class == UserClass.GENERAL
    assert prov.domain_source == "default"
    assert prov.authority_source == "default"
    assert prov.user_class_source == "default"


def test_contextvar_reset_with_token(resolver):
    """Demonstrates safe set/reset pattern for proxy request lifecycle."""
    tok = authority_class_var.set("DA")
    _, authority, _, _ = resolver.resolve([])
    assert authority == AuthorityClass.DA
    authority_class_var.reset(tok)
    _, authority_after, _, _ = resolver.resolve([])
    assert authority_after == AuthorityClass.GP  # reset to default
