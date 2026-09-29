"""Unit tests for StatusTranslator.

KEY INVARIANT UNDER TEST
------------------------
Severity precedence is computed over terminal epistemic CLASSES
(STOP > REFUSE > ASK > ADMIT), not over raw flag count or iteration order.
A single STOP-class flag beats any number of REFUSE or ASK flags.
The result must be identical regardless of the order flags appear in the list.
"""

import pytest

from aurora_lens.govern.adapters.status_translator import StatusTranslator
from aurora_lens.verify.flags import Flag, FlagType
from aurora_lens.governor.models import LensStatus


def _flag(flag_type: FlagType, severity: str = "warning") -> Flag:
    return Flag(
        flag_type=flag_type,
        entity_name="test",
        claim="test claim",
        evidence="test evidence",
        severity=severity,
    )


@pytest.fixture
def translator() -> StatusTranslator:
    return StatusTranslator()


# ── ADMIT ─────────────────────────────────────────────────────────────────────

def test_no_flags_returns_admit(translator):
    """Empty flag list → ADMIT. The base case."""
    assert translator.translate([], "public") == LensStatus.ADMIT


def test_no_flags_any_mode(translator):
    """Empty flag list → ADMIT regardless of mode."""
    for mode in ("public", "enterprise", "open"):
        assert translator.translate([], mode) == LensStatus.ADMIT


# ── STOP — hard-stop-always set ───────────────────────────────────────────────

@pytest.mark.parametrize("flag_type", [
    FlagType.EXTRACTION_FAILED,
    FlagType.MEDICAL_DOSAGE_RECOMMENDATION,
    FlagType.PEDIATRIC_DOSAGE_RECOMMENDATION,
    FlagType.NUMERIC_MEDICAL_INSTRUCTION,
    FlagType.EMERGENCY_TRIAGE_GUIDANCE,
    FlagType.SELF_HARM_INSTRUCTION,
    FlagType.ILLEGAL_INSTRUCTION,
    FlagType.TARGETED_DEFAMATION,
])
def test_hard_stop_always_returns_stop_any_mode(translator, flag_type):
    """Each member of the hard-stop-always set → STOP regardless of mode."""
    for mode in ("public", "enterprise", "open"):
        result = translator.translate([_flag(flag_type, "error")], mode)
        assert result == LensStatus.STOP, (
            f"Expected STOP for {flag_type.name} in {mode} mode, got {result}"
        )


def test_extraction_failed_invariant_stated_consciously(translator):
    """EXTRACTION_FAILED is STOP in all modes — consciously stated invariant.

    This is not a convenience default. See StatusTranslator module docstring
    for the explicit rationale. If you weaken this, the governance system can
    be bypassed by inducing extraction failures.
    """
    flag = _flag(FlagType.EXTRACTION_FAILED, "error")
    for mode in ("public", "enterprise", "open"):
        assert translator.translate([flag], mode) == LensStatus.STOP


def test_contradicted_fact_error_severity_returns_stop(translator):
    """CONTRADICTED_FACT at error severity → STOP."""
    flag = _flag(FlagType.CONTRADICTED_FACT, "error")
    for mode in ("public", "enterprise", "open"):
        assert translator.translate([flag], mode) == LensStatus.STOP


def test_contradicts_committed_state_error_severity_returns_stop(translator):
    """CONTRADICTS_COMMITTED_STATE at error severity → STOP (same as CONTRADICTED_FACT)."""
    flag = _flag(FlagType.CONTRADICTS_COMMITTED_STATE, "error")
    for mode in ("public", "enterprise", "open"):
        assert translator.translate([flag], mode) == LensStatus.STOP


# ── STOP vs REFUSE — verify-or-refuse class ───────────────────────────────────

@pytest.mark.parametrize("flag_type", [
    FlagType.PERSONALIZED_MEDICAL_ADVICE,
    FlagType.PERSONALIZED_LEGAL_ADVICE,
    FlagType.PERSONALIZED_FINANCIAL_ADVICE,
    FlagType.SENSITIVE_PII_EXPOSURE,
])
def test_verify_or_refuse_public_mode_returns_stop(translator, flag_type):
    """Verify-or-refuse flags in public mode → STOP.

    In public mode the operator has not established a safety pathway for
    individualized determinations, so the determination is inadmissible (STOP).
    """
    assert translator.translate([_flag(flag_type)], "public") == LensStatus.STOP


@pytest.mark.parametrize("flag_type", [
    FlagType.PERSONALIZED_MEDICAL_ADVICE,
    FlagType.PERSONALIZED_LEGAL_ADVICE,
    FlagType.PERSONALIZED_FINANCIAL_ADVICE,
    FlagType.SENSITIVE_PII_EXPOSURE,
])
def test_verify_or_refuse_enterprise_mode_returns_refuse(translator, flag_type):
    """Verify-or-refuse flags in enterprise mode → REFUSE (not STOP).

    The operator has accepted responsibility for downstream workflows,
    so the determination is blocked (REFUSE) but conversation continues.
    """
    assert translator.translate([_flag(flag_type)], "enterprise") == LensStatus.REFUSE


@pytest.mark.parametrize("flag_type", [
    FlagType.PERSONALIZED_MEDICAL_ADVICE,
    FlagType.PERSONALIZED_LEGAL_ADVICE,
    FlagType.PERSONALIZED_FINANCIAL_ADVICE,
    FlagType.SENSITIVE_PII_EXPOSURE,
])
def test_verify_or_refuse_open_mode_returns_refuse(translator, flag_type):
    """Verify-or-refuse flags in open mode → REFUSE."""
    assert translator.translate([_flag(flag_type)], "open") == LensStatus.REFUSE


# ── REFUSE ────────────────────────────────────────────────────────────────────

def test_contradicted_fact_warning_severity_returns_refuse(translator):
    """CONTRADICTED_FACT at warning severity → REFUSE (not STOP)."""
    flag = _flag(FlagType.CONTRADICTED_FACT, "warning")
    assert translator.translate([flag], "public") == LensStatus.REFUSE


@pytest.mark.parametrize("flag_type", [
    FlagType.IDENTITY_DRIFT,
    FlagType.UNVERIFIED_REGULATORY_CLAIM,
    FlagType.RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT,
    FlagType.RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT,
])
def test_always_refuse_flags(translator, flag_type):
    """IDENTITY_DRIFT, UNVERIFIED_REGULATORY_CLAIM, RAG harness non-admit → REFUSE in all modes."""
    for mode in ("public", "enterprise", "open"):
        assert translator.translate([_flag(flag_type)], mode) == LensStatus.REFUSE


@pytest.mark.parametrize("flag_type", [
    FlagType.UNBOUND_ENTITY,
    FlagType.UNSUPPORTED_ATTRIBUTE,
    FlagType.UNSUPPORTED_EVENT,
    FlagType.UNVERIFIED_FACT_ASSERTION,
    FlagType.TIME_SMEAR,
])
def test_epistemic_flags_refuse_in_public_and_enterprise(translator, flag_type):
    """Epistemic failure flags → REFUSE in public and enterprise modes."""
    for mode in ("public", "enterprise"):
        assert translator.translate([_flag(flag_type)], mode) == LensStatus.REFUSE, (
            f"Expected REFUSE for {flag_type.name} in {mode} mode"
        )


@pytest.mark.parametrize("flag_type", [
    FlagType.UNBOUND_ENTITY,
    FlagType.UNSUPPORTED_ATTRIBUTE,
    FlagType.UNSUPPORTED_EVENT,
    FlagType.UNVERIFIED_FACT_ASSERTION,
    FlagType.TIME_SMEAR,
])
def test_epistemic_flags_admit_in_open_mode(translator, flag_type):
    """Epistemic failure flags → ADMIT in open mode (annotate-and-pass behaviour)."""
    assert translator.translate([_flag(flag_type)], "open") == LensStatus.ADMIT


# ── ASK ───────────────────────────────────────────────────────────────────────

def test_extraction_empty_returns_ask_all_modes(translator):
    """EXTRACTION_EMPTY → ASK in all modes.

    EXTRACTION_EMPTY means the extractor ran but found nothing to verify.
    The interaction remains open for clarification — this is not a hard failure.
    """
    flag = _flag(FlagType.EXTRACTION_EMPTY, "error")
    for mode in ("public", "enterprise", "open"):
        assert translator.translate([flag], mode) == LensStatus.ASK


@pytest.mark.parametrize("flag_type", [
    FlagType.UNRESOLVED_REFERENT,
    FlagType.UNRESOLVED_COMPARAND,
    FlagType.DISJUNCTIVE_BRANCH_COLLAPSE,
])
def test_pef_binding_failures_return_ask(translator, flag_type):
    """PEF binding failures and RAG C2 disjunctive harness → ASK (containment)."""
    for mode in ("public", "enterprise", "open"):
        assert translator.translate([_flag(flag_type)], mode) == LensStatus.ASK


# ── WORST-CLASS INVARIANT ─────────────────────────────────────────────────────

def test_stop_beats_refuse_worst_class_invariant(translator):
    """A single STOP flag beats any number of REFUSE flags — worst class wins."""
    flags = [
        _flag(FlagType.UNBOUND_ENTITY),       # REFUSE
        _flag(FlagType.IDENTITY_DRIFT),             # REFUSE
        _flag(FlagType.SELF_HARM_INSTRUCTION, "error"),  # STOP
        _flag(FlagType.UNVERIFIED_REGULATORY_CLAIM),     # REFUSE
    ]
    assert translator.translate(flags, "public") == LensStatus.STOP


def test_stop_beats_ask_worst_class_invariant(translator):
    """A single STOP flag beats any number of ASK flags."""
    flags = [
        _flag(FlagType.UNRESOLVED_REFERENT),        # ASK
        _flag(FlagType.EXTRACTION_EMPTY, "error"),  # ASK
        _flag(FlagType.ILLEGAL_INSTRUCTION, "error"),   # STOP
    ]
    assert translator.translate(flags, "public") == LensStatus.STOP


def test_refuse_beats_ask(translator):
    """REFUSE beats ASK when no STOP flags present."""
    flags = [
        _flag(FlagType.UNRESOLVED_REFERENT),     # ASK
        _flag(FlagType.UNBOUND_ENTITY),     # REFUSE in public
    ]
    assert translator.translate(flags, "public") == LensStatus.REFUSE


def test_order_independence_invariant(translator):
    """Result is identical regardless of flag list order.

    CRITICAL: This test exists to catch future maintainers who write
    "first hard flag encountered wins" — which would make behaviour
    dependent on extraction ordering (spaCy vs LLM backends differ).
    """
    flags_a = [
        _flag(FlagType.UNRESOLVED_REFERENT),         # ASK
        _flag(FlagType.UNBOUND_ENTITY),         # REFUSE
        _flag(FlagType.SELF_HARM_INSTRUCTION, "error"),  # STOP
    ]
    flags_b = list(reversed(flags_a))
    flags_c = [flags_a[2], flags_a[0], flags_a[1]]

    result_a = translator.translate(flags_a, "public")
    result_b = translator.translate(flags_b, "public")
    result_c = translator.translate(flags_c, "public")

    assert result_a == result_b == result_c == LensStatus.STOP


# ── INVALID MODE MUST RAISE, NOT DEGRADE ─────────────────────────────────────

@pytest.mark.parametrize("bad_mode", [
    "INVALID", "PUBLIC", "Enterprise", "prod", "strict", "", "enterprise2",
])
def test_invalid_mode_raises_not_silently_degrades(translator, bad_mode):
    """Unknown mode strings must raise ValueError, not silently fall to REFUSE.

    A misconfigured deployment (e.g. mode: prod in config.yaml, or a caps typo)
    that reaches StatusTranslator must be rejected explicitly. The alternative —
    silently treating unknown modes like enterprise — would degrade STOP to
    REFUSE for the verify-or-refuse flag class without any operator signal.
    """
    flag = _flag(FlagType.PERSONALIZED_MEDICAL_ADVICE)
    with pytest.raises(ValueError, match="Unknown deployment mode"):
        translator.translate([flag], bad_mode)


def test_invalid_mode_raises_even_with_no_flags(translator):
    """Mode validation fires before flag evaluation — even empty flag lists raise."""
    with pytest.raises(ValueError):
        translator.translate([], "INVALID")


# ── MODE SENSITIVITY IS PER-INVOCATION ────────────────────────────────────────

def test_same_translator_instance_different_mode(translator):
    """Same translator instance, different mode argument, different result.

    The translator is stateless — mode is always a per-call argument, never
    stored on the instance.
    """
    flag = _flag(FlagType.PERSONALIZED_MEDICAL_ADVICE)
    assert translator.translate([flag], "public") == LensStatus.STOP
    assert translator.translate([flag], "enterprise") == LensStatus.REFUSE
    assert translator.translate([flag], "open") == LensStatus.REFUSE
