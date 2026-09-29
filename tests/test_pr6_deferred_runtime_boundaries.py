"""PR6 runtime boundaries: auditor corridor enforcement and LLM parity exclusion."""

from __future__ import annotations

from aurora_lens.governor.models import (
    AuthorityClass,
    Domain,
    LensStatus,
    OutputMode,
    SpeechAct,
    UserClass,
    ForensicObligation,
    ExposureLevel,
)
from aurora_lens.governor.resolver import resolve
from aurora_lens.interpret.llm_backend import LLMExtractionBackend


class _NoopAdapter:
    async def generate(self, messages, **kwargs):  # noqa: ANN001, ARG002
        raise AssertionError("generate() should not be called in this test")


def test_auditor_stop_exact_corridor_keeps_forensic_visibility():
    policy = resolve(Domain.GENERAL, AuthorityClass.GP, LensStatus.STOP, user_class=UserClass.AUDITOR)
    assert policy.output_mode == OutputMode.FORENSIC_STOP
    assert policy.exposure_level == ExposureLevel.FULL
    assert SpeechAct.EXPOSE_AUDIT_BASIS in policy.allowed_speech_acts
    assert ForensicObligation.ATTACH_PEF_SNAPSHOT in policy.forensic_obligations


def test_auditor_non_exact_corridor_is_hard_capped_to_general_visibility():
    policy = resolve(Domain.MEDICAL, AuthorityClass.GP, LensStatus.STOP, user_class=UserClass.AUDITOR)
    assert policy.output_mode != OutputMode.FORENSIC_STOP
    assert policy.exposure_level == ExposureLevel.MINIMAL
    assert SpeechAct.EXPOSE_AUDIT_BASIS not in policy.allowed_speech_acts
    assert ForensicObligation.ATTACH_PEF_SNAPSHOT not in policy.forensic_obligations


def test_auditor_non_stop_is_hard_capped_to_general_visibility():
    policy = resolve(Domain.GENERAL, AuthorityClass.GP, LensStatus.REFUSE, user_class=UserClass.AUDITOR)
    assert policy.exposure_level == ExposureLevel.MINIMAL
    assert SpeechAct.EXPOSE_AUDIT_BASIS not in policy.allowed_speech_acts
    assert ForensicObligation.ATTACH_PEF_SNAPSHOT not in policy.forensic_obligations


def test_llm_extraction_backend_formally_excludes_ambiguity_parity():
    backend = LLMExtractionBackend(adapter=_NoopAdapter())
    assert backend.supports_ambiguous_referents() is False
