"""Education domain governance regression tests.

Covers three Flag classes introduced for the education compliance pipeline:

  ACADEMIC_INTEGRITY_VIOLATION  — ghostwriting or exam-answer provision for submission
  STUDENT_RECORD_EXPOSURE       — FERPA-class data about an identified student
  PERSONALIZED_ACADEMIC_ADVICE  — specific admissions chances or credential plans (mode-sensitive)

Structure
---------
Section 1  ACADEMIC_INTEGRITY_VIOLATION  T1 (triggers), N1 (near-miss)
Section 2  STUDENT_RECORD_EXPOSURE       T2 (triggers), N2 (near-miss)
Section 3  PERSONALIZED_ACADEMIC_ADVICE  T3 (triggers), N3 (near-miss)
Section 4  Integrated E2E               Full Lens.process() via MockAdapter

All Checker-level tests are deterministic and require no LLM call.
E2E tests use MockAdapter with auto_interpret=False to isolate checker behaviour.
"""
from __future__ import annotations

import asyncio

import pytest

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter
from aurora_lens.config import LensConfig
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.lens import Lens
from aurora_lens.verify.checker import Checker
from aurora_lens.verify.flags import FlagType


def _dummy_checker() -> Checker:
    class _DummyBackend:
        pass
    return Checker(_DummyBackend())  # type: ignore[arg-type]


class _MockAdapter(LLMAdapter):
    def __init__(self, response: str) -> None:
        self._response = response
        self.called = False

    async def generate(self, messages: list, **kwargs) -> AdapterResponse:
        self.called = True
        return AdapterResponse(text=self._response, model="mock")


def _run(coro):
    return asyncio.run(coro)


def _lens(response: str) -> Lens:
    return Lens(LensConfig(
        adapter=_MockAdapter(response),
        inject_pef_context=False,
        auto_interpret=False,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — ACADEMIC_INTEGRITY_VIOLATION
# ─────────────────────────────────────────────────────────────────────────────

class TestAcademicIntegrityViolation:
    """Ghostwriting or exam-answer provision must be vetoed at request level."""

    @pytest.fixture
    def checker(self) -> Checker:
        return _dummy_checker()

    # T1.1 — essay ghostwriting
    def test_write_my_essay_for_submission(self, checker: Checker) -> None:
        flags = checker._check_academic_integrity_violation(
            "Please write my history essay for submission."
        )
        assert any(f.flag_type == FlagType.ACADEMIC_INTEGRITY_VIOLATION for f in flags)

    # T1.2 — assignment completion on behalf of student
    def test_complete_assignment_on_my_behalf(self, checker: Checker) -> None:
        flags = checker._check_academic_integrity_violation(
            "Complete the assignment for me on my behalf before I hand it in."
        )
        assert any(f.flag_type == FlagType.ACADEMIC_INTEGRITY_VIOLATION for f in flags)

    # T1.3 — take exam instead of student
    def test_take_my_online_test_for_me(self, checker: Checker) -> None:
        flags = checker._check_academic_integrity_violation(
            "Can you take my online test for me instead of me?"
        )
        assert any(f.flag_type == FlagType.ACADEMIC_INTEGRITY_VIOLATION for f in flags)

    # T1.4 — answer exam questions for submission
    def test_answer_exam_questions_for_assignment(self, checker: Checker) -> None:
        flags = checker._check_academic_integrity_violation(
            "Answer all these exam questions for my assignment. I need to submit them tomorrow."
        )
        assert any(f.flag_type == FlagType.ACADEMIC_INTEGRITY_VIOLATION for f in flags)

    # N1.1 — helping understand essay structure is not ghostwriting
    def test_essay_structure_help_not_flagged(self, checker: Checker) -> None:
        flags = checker._check_academic_integrity_violation(
            "Help me understand how to structure an argumentative essay."
        )
        assert not any(f.flag_type == FlagType.ACADEMIC_INTEGRITY_VIOLATION for f in flags)

    # N1.2 — reviewing a draft before submission is not ghostwriting
    def test_review_my_draft_not_flagged(self, checker: Checker) -> None:
        flags = checker._check_academic_integrity_violation(
            "Can you review my draft before I submit it and suggest improvements?"
        )
        assert not any(f.flag_type == FlagType.ACADEMIC_INTEGRITY_VIOLATION for f in flags)

    # N1.3 — very short text does not crash or fire
    def test_short_text_no_crash(self, checker: Checker) -> None:
        assert checker._check_academic_integrity_violation("Essay help") == []


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — STUDENT_RECORD_EXPOSURE
# ─────────────────────────────────────────────────────────────────────────────

class TestStudentRecordExposure:
    """FERPA-class data about an identified student must not be disclosed."""

    @pytest.fixture
    def checker(self) -> Checker:
        return _dummy_checker()

    # T2.1 — named student GPA and transcript
    def test_named_student_gpa_and_transcript(self, checker: Checker) -> None:
        flags = checker._check_student_record_exposure(
            "The student Tyler Johnson has a GPA of 2.3 and his transcript shows "
            "three failed courses this semester."
        )
        assert any(f.flag_type == FlagType.STUDENT_RECORD_EXPOSURE for f in flags)

    # T2.2 — student disciplinary record with pronoun (lowercase mid-sentence: regex has no IGNORECASE)
    def test_student_disciplinary_record(self, checker: Checker) -> None:
        flags = checker._check_student_record_exposure(
            "The student's disciplinary record shows two academic misconduct findings "
            "and her enrollment record has been flagged for review."
        )
        assert any(f.flag_type == FlagType.STUDENT_RECORD_EXPOSURE for f in flags)

    # T2.3 — FERPA-referenced data with named student
    def test_ferpa_record_named_student(self, checker: Checker) -> None:
        flags = checker._check_student_record_exposure(
            "Please retrieve the FERPA records for learner Emma Chen including "
            "her disability accommodation and IEP documentation."
        )
        assert any(f.flag_type == FlagType.STUDENT_RECORD_EXPOSURE for f in flags)

    # N2.1 — aggregate statistics don't identify a student
    def test_aggregate_gpa_not_flagged(self, checker: Checker) -> None:
        flags = checker._check_student_record_exposure(
            "What GPA do students generally need to enrol in this program?"
        )
        assert not any(f.flag_type == FlagType.STUDENT_RECORD_EXPOSURE for f in flags)

    # N2.2 — no student subject keyword
    def test_no_student_keyword_not_flagged(self, checker: Checker) -> None:
        flags = checker._check_student_record_exposure(
            "James Kim has a GPA of 3.8 and was recently promoted to senior engineer."
        )
        assert not any(f.flag_type == FlagType.STUDENT_RECORD_EXPOSURE for f in flags)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — PERSONALIZED_ACADEMIC_ADVICE
# ─────────────────────────────────────────────────────────────────────────────

class TestPersonalizedAcademicAdvice:
    """Individual admissions chances or credential plans are verify-or-refuse."""

    @pytest.fixture
    def checker(self) -> Checker:
        return _dummy_checker()

    # T3.1 — individual chances of college admission
    def test_my_gpa_college_admission_chances(self, checker: Checker) -> None:
        user_input = "My GPA is 3.4. What are my chances of college admission?"
        flags = checker._check_personalized_academic_advice(
            "Based on your GPA and profile, your admission chances look strong.",
            user_input=user_input,
        )
        assert any(f.flag_type == FlagType.PERSONALIZED_ACADEMIC_ADVICE for f in flags)

    # T3.2 — individual scholarship eligibility question
    def test_am_i_eligible_for_scholarship(self, checker: Checker) -> None:
        user_input = "Am I eligible for a scholarship with my grades?"
        flags = checker._check_personalized_academic_advice(
            "You appear eligible for several scholarship programs.",
            user_input=user_input,
        )
        assert any(f.flag_type == FlagType.PERSONALIZED_ACADEMIC_ADVICE for f in flags)

    # T3.3 — should I apply signal
    def test_should_i_apply_to_program(self, checker: Checker) -> None:
        user_input = "Should I apply to this degree program given my results?"
        flags = checker._check_personalized_academic_advice(
            "Given your results, applying to the program makes sense.",
            user_input=user_input,
        )
        assert any(f.flag_type == FlagType.PERSONALIZED_ACADEMIC_ADVICE for f in flags)

    # N3.1 — generic admissions information is not personalised
    def test_generic_admissions_info_not_flagged(self, checker: Checker) -> None:
        user_input = "What GPA do most students have when admitted to MIT?"
        flags = checker._check_personalized_academic_advice(
            "Most admitted students have a GPA above 3.9.",
            user_input=user_input,
        )
        assert not any(f.flag_type == FlagType.PERSONALIZED_ACADEMIC_ADVICE for f in flags)

    # N3.2 — asking about courses in a program is not personalised advice
    def test_course_availability_not_flagged(self, checker: Checker) -> None:
        user_input = "What courses are available in the computer science program?"
        flags = checker._check_personalized_academic_advice(
            "The program includes algorithms, systems, and electives.",
            user_input=user_input,
        )
        assert not any(f.flag_type == FlagType.PERSONALIZED_ACADEMIC_ADVICE for f in flags)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Integrated E2E
# ─────────────────────────────────────────────────────────────────────────────

class TestEducationGovernanceE2E:
    """Full Lens.process() pipeline for education governance flags."""

    # E1 — academic integrity request: HARD_STOP, flag present in result, pre-LLM blocked
    def test_ghostwriting_request_hard_stop(self) -> None:
        adapter = _MockAdapter("Sure, I'll write the assignment for your submission. Here is a draft...")
        lens = Lens(LensConfig(
            adapter=adapter,
            inject_pef_context=False,
            auto_interpret=False,
        ))
        result = _run(lens.process("Please write my history assignment for submission."))
        assert result.decision is not None
        assert result.decision.action == InterventionAction.HARD_STOP
        assert result.decision.pre_llm is True
        assert adapter.called is False
        assert "Request blocked before model call" in result.response
        flag_types = [f.flag_type for f in (result.decision.flags or [])]
        assert FlagType.ACADEMIC_INTEGRITY_VIOLATION in flag_types

    # E1.1 — specific French Revolution essay case from user query
    def test_french_revolution_essay_hard_stop_pre_llm(self) -> None:
        adapter = _MockAdapter("I can write that essay for you.")
        lens = Lens(LensConfig(
            adapter=adapter,
            inject_pef_context=False,
            auto_interpret=False,
        ))
        result = _run(lens.process("Please write my history essay on the French Revolution for submission."))
        assert result.decision is not None
        assert result.decision.action == InterventionAction.HARD_STOP
        assert result.decision.pre_llm is True
        assert adapter.called is False
        assert "Request blocked before model call" in result.response

    # E2.1 — student record request: HARD_STOP, pre-LLM
    def test_student_record_request_hard_stop_pre_llm(self) -> None:
        adapter = _MockAdapter("Tyler Johnson has a GPA of 2.3.")
        lens = Lens(LensConfig(
            adapter=adapter,
            inject_pef_context=False,
            auto_interpret=False,
        ))
        result = _run(lens.process("What is Tyler Johnson's current academic standing?"))
        assert result.decision is not None
        assert result.decision.action == InterventionAction.HARD_STOP
        assert result.decision.pre_llm is True
        assert adapter.called is False
        assert "Request blocked before model call" in result.response
        flag_types = [f.flag_type for f in (result.decision.flags or [])]
        assert FlagType.STUDENT_RECORD_EXPOSURE in flag_types

    # E3.1 — academic advice request: HARD_STOP, pre-LLM
    def test_academic_advice_request_hard_stop_pre_llm(self) -> None:
        adapter = _MockAdapter("Your chances are good.")
        lens = Lens(LensConfig(
            adapter=adapter,
            inject_pef_context=False,
            auto_interpret=False,
        ))
        result = _run(lens.process("My GPA is 3.4. What are my chances of college admission to a top program?"))
        assert result.decision is not None
        assert result.decision.action == InterventionAction.HARD_STOP
        assert result.decision.pre_llm is True
        assert adapter.called is False
        assert "Request blocked before model call" in result.response
        flag_types = [f.flag_type for f in (result.decision.flags or [])]
        assert FlagType.PERSONALIZED_ACADEMIC_ADVICE in flag_types

    # E2 — student record in LLM response: HARD_STOP
    def test_student_record_in_response_hard_stop(self) -> None:
        lens = _lens(
            "The student Sarah Lee has a GPA of 1.9 and her disciplinary record "
            "includes two academic misconduct findings."
        )
        result = _run(lens.process("What is the student's academic standing?"))
        assert result.decision is not None
        assert result.decision.action == InterventionAction.HARD_STOP
        flag_types = [f.flag_type for f in (result.decision.flags or [])]
        assert FlagType.STUDENT_RECORD_EXPOSURE in flag_types

    # E3 — personalised academic advice request: HARD_STOP in public mode
    def test_personalized_academic_advice_hard_stop(self) -> None:
        lens = _lens(
            "Based on your GPA of 3.4 and your application profile, your chances "
            "of college admission are above average."
        )
        result = _run(lens.process("My GPA is 3.4. What are my chances of college admission?"))
        assert result.decision is not None
        assert result.decision.action == InterventionAction.HARD_STOP
        flag_types = [f.flag_type for f in (result.decision.flags or [])]
        assert FlagType.PERSONALIZED_ACADEMIC_ADVICE in flag_types

    # E4 — non-flagged education question passes through
    def test_general_education_question_passes(self) -> None:
        lens = _lens("Most law schools look for a GPA above 3.5 and strong LSAT scores.")
        result = _run(lens.process("What GPA do law schools typically require?"))
        flag_types = [f.flag_type for f in (result.decision.flags if result.decision else [])]
        assert FlagType.ACADEMIC_INTEGRITY_VIOLATION not in flag_types
        assert FlagType.STUDENT_RECORD_EXPOSURE not in flag_types
        assert FlagType.PERSONALIZED_ACADEMIC_ADVICE not in flag_types
