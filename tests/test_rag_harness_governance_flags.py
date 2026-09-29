"""Focused tests: RAG corpus verification flags (structural question shapes)."""

import asyncio

import pytest

from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractionResult
from aurora_lens.pef.state import PEFState
from aurora_lens.verify.checker import (
    Checker,
    _extract_disjunctive_possessive_candidates,
    _user_rag_calendar_arrival_date_question,
    _user_rag_causal_why_question,
    _user_rag_disjunctive_identity_question,
    _user_rag_salary_compensation_question,
)
from aurora_lens.verify.flags import Flag, FlagType


class _EmptyBackend(ExtractionBackend):
    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        return ExtractionResult()


@pytest.fixture
def checker() -> Checker:
    return Checker(_EmptyBackend())


_CAUSAL_WHY = (
    "Context:\nx\n\nQuestion: Why did Nora move back from Canada?"
)
_SALARY = (
    "Context:\nx\n\nQuestion: What is Nora's annual salary at Harbor Labs?"
)
_CALENDAR_ARRIVAL = (
    "Context:\nx\n\nQuestion: On what calendar date did Nora arrive in Metroville?"
)
_DISJUNCTIVE_IDENTITY = (
    "Context:\nx\n\nQuestion: Whose sister moved from Vancouver, Emma's or Lucy's?"
)


def test_structural_causal_why_question_shape():
    assert _user_rag_causal_why_question(_CAUSAL_WHY) is True
    assert _user_rag_causal_why_question("Why did Nora move back from Canada?") is False


def test_structural_salary_question_shape():
    assert _user_rag_salary_compensation_question(_SALARY) is True
    assert _user_rag_salary_compensation_question("What is Nora's annual salary?") is False


def test_structural_calendar_arrival_question_shape():
    assert _user_rag_calendar_arrival_date_question(_CALENDAR_ARRIVAL) is True
    assert _user_rag_calendar_arrival_date_question(_DISJUNCTIVE_IDENTITY) is False


def test_structural_disjunctive_identity_question_shape():
    assert _user_rag_disjunctive_identity_question(_DISJUNCTIVE_IDENTITY) is True
    assert _extract_disjunctive_possessive_candidates(
        "Whose sister moved from Vancouver, Emma's or Lucy's?"
    ) == ("Emma", "Lucy")
    assert _user_rag_disjunctive_identity_question(
        "Whose sister moved from Vancouver, Emma's or Lucy's?"
    ) is False


def test_causal_why_bounded_absence_raises_rag_causal_non_admit(checker: Checker):
    r = (
        "The context does not state why Nora moved back from Canada. "
        "No motivation for the relocation is provided in the supplied passages."
    )
    f = checker._check_rag_causal_bounded_absence_non_admit(r, _CAUSAL_WHY)
    assert f is not None
    assert f.flag_type.name == "RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT"


def test_salary_bounded_absence_raises_rag_salary_non_admit(checker: Checker):
    r = (
        "Nora's annual salary is not stated in the provided context. "
        "Compensation details are confidential and not included here."
    )
    f = checker._check_rag_salary_bounded_absence_non_admit(r, _SALARY)
    assert f is not None
    assert f.flag_type.name == "RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT"


def test_causal_why_mixed_supported_fact_and_absent_reason_raises_rag_causal_non_admit(
    checker: Checker,
):
    r = (
        "Nora moved back from Canada, but the specific reason for her return is not provided "
        "in the available information. Emma Chen simply mentioned to the Metroville Weekly "
        "that her sister moved back last week."
    )
    f = checker._check_rag_causal_bounded_absence_non_admit(r, _CAUSAL_WHY)
    assert f is not None
    assert f.flag_type.name == "RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT"


def test_causal_why_not_detailed_in_provided_context_raises_rag_causal_non_admit(
    checker: Checker,
):
    r = (
        "Nora moved back from Canada because she is Emma Chen's sister, and Emma mentioned "
        "that her sister moved back from Canada last week. However, the specific reasons "
        "for Nora's move are not detailed in the provided context."
    )
    f = checker._check_rag_causal_bounded_absence_non_admit(r, _CAUSAL_WHY)
    assert f is not None
    assert f.flag_type.name == "RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT"


def test_causal_why_not_specified_in_provided_context_raises_rag_causal_non_admit(
    checker: Checker,
):
    r = (
        "The reason why Nora moved back from Canada is not specified in the provided context."
    )
    f = checker._check_rag_causal_bounded_absence_non_admit(r, _CAUSAL_WHY)
    assert f is not None
    assert f.flag_type.name == "RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT"


def test_causal_why_no_information_provided_raises_rag_causal_non_admit(checker: Checker):
    r = (
        "There is no information provided about Nora moving back from Canada. "
        "The documents do not indicate any details about Nora's movements or reasons for moving."
    )
    f = checker._check_rag_causal_bounded_absence_non_admit(r, _CAUSAL_WHY)
    assert f is not None
    assert f.flag_type.name == "RAG_HARNESS_CAUSAL_ABSENCE_NON_ADMIT"


def test_salary_does_not_specify_plus_hr_consult_raises_rag_salary_non_admit(
    checker: Checker,
):
    r = (
        "The provided context does not specify Nora's annual salary at Harbor Labs. "
        "To find out her salary, you may need to consult HR records or other official "
        "documentation from the company."
    )
    f = checker._check_rag_salary_bounded_absence_non_admit(r, _SALARY)
    assert f is not None
    assert f.flag_type.name == "RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT"


def test_salary_does_not_contain_info_raises_rag_salary_non_admit(checker: Checker):
    r = (
        "The provided context does not contain any information regarding Nora's annual salary "
        "at Harbor Labs. Therefore, I cannot answer that question."
    )
    f = checker._check_rag_salary_bounded_absence_non_admit(r, _SALARY)
    assert f is not None
    assert f.flag_type.name == "RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT"


def test_salary_does_not_include_info_raises_rag_salary_non_admit(checker: Checker):
    r = (
        "The provided context does not include information about Nora's annual salary at Harbor Labs. "
        "Therefore, I'm unable to provide that detail."
    )
    f = checker._check_rag_salary_bounded_absence_non_admit(r, _SALARY)
    assert f is not None
    assert f.flag_type.name == "RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT"


def test_salary_i_do_not_have_information_raises_rag_salary_non_admit(checker: Checker):
    r = (
        "I do not have information regarding Nora Park's annual salary at Harbor Labs. "
        "You may need to consult HR."
    )
    f = checker._check_rag_salary_bounded_absence_non_admit(r, _SALARY)
    assert f is not None
    assert f.flag_type.name == "RAG_HARNESS_SALARY_ABSENCE_NON_ADMIT"


def test_disjunctive_identity_always_disjunctive_flag_hedged(checker: Checker):
    r = "The context is unclear whether Emma's or Lucy's sister is meant."
    f = checker._check_disjunctive_whose_sister_collapse(_DISJUNCTIVE_IDENTITY, r)
    assert f is not None
    assert f.flag_type.name == "DISJUNCTIVE_BRANCH_COLLAPSE"
    assert f.candidates == ("Emma", "Lucy")


def test_non_rag_disjunctive_plain_question_no_disjunctive_flag(checker: Checker):
    u = "Whose sister moved from Vancouver, Emma's or Lucy's?"
    f = checker._check_disjunctive_whose_sister_collapse(
        u,
        "Emma's sister moved.",
    )
    assert f is None


def test_disjunctive_prioritize_drops_epistemic_peer_flags(checker: Checker):
    u = _DISJUNCTIVE_IDENTITY
    flags = [
        Flag(
            flag_type=FlagType.DISJUNCTIVE_BRANCH_COLLAPSE,
            entity_name="disjunctive",
            claim="harness",
            evidence="e",
        ),
        Flag(
            flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
            entity_name="x",
            claim="y",
            evidence="z",
        ),
    ]
    out = checker._rag_disjunctive_prioritize_disjunctive(flags, u)
    assert len(out) == 1
    assert out[0].flag_type == FlagType.DISJUNCTIVE_BRANCH_COLLAPSE


def test_disjunctive_prioritize_first_for_policy_reason_code(checker: Checker):
    u = _DISJUNCTIVE_IDENTITY
    flags = [
        Flag(
            flag_type=FlagType.UNVERIFIED_TRUTHFULNESS_CLAIM,
            entity_name="t",
            claim="c",
            evidence="e",
        ),
        Flag(
            flag_type=FlagType.DISJUNCTIVE_BRANCH_COLLAPSE,
            entity_name="disjunctive",
            claim="harness",
            evidence="e",
        ),
    ]
    out = checker._rag_disjunctive_prioritize_disjunctive(flags, u)
    assert len(out) == 1
    assert out[0].flag_type == FlagType.DISJUNCTIVE_BRANCH_COLLAPSE


def test_disjunctive_cannot_confirmed_still_emits_disjunctive_before_refusal_frame(checker: Checker):
    """'cannot be confirmed' matches refusal regex; disjunctive flag must attach earlier in check()."""
    r = (
        "Therefore, it cannot be confirmed that either Emma's or Lucy's sister moved from Vancouver."
    )

    async def _run():
        return await checker.check(r, PEFState(), user_input=_DISJUNCTIVE_IDENTITY)

    flags = asyncio.run(_run())
    names = {f.flag_type.name for f in flags}
    assert "DISJUNCTIVE_BRANCH_COLLAPSE" in names
