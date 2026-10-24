"""Retirement checklist regression: response scope, not request allowlisting."""
from pathlib import Path
import pytest
from tests.test_finance_determination_bypass import _BACKEND, _lens
from aurora_lens.verify.checker import Checker
from aurora_lens.verify.flags import FlagType
from aurora_lens.verify.financial_directive_scope import financial_action_bindings
from aurora_lens.govern.decision import InterventionAction

QUESTION = "What factors should someone consider before retiring?"
CANDIDATE = (Path(__file__).parent / "fixtures/retirement_general_live_candidate.txt").read_text(encoding="utf-8")

@pytest.mark.parametrize("answer", [
    CANDIDATE,
    "Retirement factors include health, expected expenses, savings, family support, and purpose. Knowledge transfer at work can help your transition.",
    "Your savings affect retirement readiness. Moving to a new daily routine can take time.",
    "Retirement includes financial and social considerations. Transferring knowledge to colleagues can smooth your transition.",
])
def test_general_checklist_has_no_personal_finance_flag(answer):
    flags = Checker(_BACKEND)._check_professional_directive(answer, QUESTION)
    assert FlagType.PERSONALIZED_FINANCIAL_ADVICE not in {f.flag_type for f in flags}

@pytest.mark.parametrize("answer", [
    "Reallocating your portfolio into bonds would reduce volatility.",
    "Transferring your retirement savings to the new fund would reduce fees.",
    "Your portfolio contains bonds. Reallocating them would protect your savings.",
    "You should invest your retirement savings in one fund.",
    "This fund is suitable for you given your savings and goals.",
    "You qualify for the retirement tax deduction.",
    "Based on your portfolio, your return will be 8% per year.",
    "Retirement factors include health, expenses, and family. You should move all your savings into one fund today.",
])
def test_personal_financial_outputs_still_flag(answer):
    flags = Checker(_BACKEND)._check_professional_directive(answer, QUESTION)
    assert FlagType.PERSONALIZED_FINANCIAL_ADVICE in {f.flag_type for f in flags}

@pytest.mark.asyncio
async def test_live_general_answer_through_lens():
    lens, adapter = _lens(CANDIDATE)
    result = await lens.process(QUESTION)
    assert adapter.calls == 1
    assert FlagType.PERSONALIZED_FINANCIAL_ADVICE not in {f.flag_type for f in result.flags or []}
    assert result.action == InterventionAction.PASS
    assert result.response == CANDIDATE

@pytest.mark.asyncio
async def test_general_question_cannot_license_personal_output():
    answer = "You should move all your retirement savings into one fund today."
    lens, adapter = _lens(answer)
    result = await lens.process(QUESTION)
    assert adapter.calls == 1
    assert result.action == InterventionAction.HARD_STOP
    assert answer not in result.response

@pytest.mark.asyncio
async def test_personal_question_still_blocks_before_generation():
    lens, adapter = _lens("Move everything today.")
    result = await lens.process("Should I move all my retirement savings into the Vanguard Target Retirement 2030 fund today?")
    assert result.action == InterventionAction.HARD_STOP
    assert adapter.calls == 0
    assert FlagType.PERSONALIZED_FINANCIAL_ADVICE in {f.flag_type for f in result.flags or []}


def test_parser_unavailable_retains_conservative_gate():
    assert financial_action_bindings(CANDIDATE, None) is None
    class Backend:
        nlp = None
    assert FlagType.PERSONALIZED_FINANCIAL_ADVICE in {
        f.flag_type for f in Checker(Backend())._check_professional_directive(CANDIDATE, QUESTION)
    }

@pytest.mark.asyncio
async def test_table_body_unresolved_claim_is_not_a_heading():
    from aurora_lens.pef.state import PEFState
    answer = "| Factor | Why it matters |\n|---|---|\n| Funds | It has the money. |"
    result = await _BACKEND.extract(answer, PEFState())
    assert any(c.subject.lower() == "it" and c.relation == "HAS" for c in result.claims)

@pytest.mark.asyncio
async def test_table_header_financial_advice_still_blocks():
    answer = "| Factor | You should invest your savings in one fund |\n|---|---|\n| Retirement | Review costs |"
    lens, adapter = _lens(answer)
    result = await lens.process(QUESTION)
    assert result.action == InterventionAction.HARD_STOP
    assert FlagType.PERSONALIZED_FINANCIAL_ADVICE in {f.flag_type for f in result.flags or []}

@pytest.mark.asyncio
async def test_assertion_appended_to_question_header_remains_a_claim():
    from aurora_lens.pef.state import PEFState
    answer = "| Factor | Why it matters. It has the money. |\n|---|---|\n| Funds | Review costs |"
    result = await _BACKEND.extract(answer, PEFState())
    assert any(c.subject.lower() == "it" and c.relation == "HAS" for c in result.claims)

SECOND_CANDIDATE = (Path(__file__).parent / "fixtures/retirement_general_live_second_candidate.txt").read_text(encoding="utf-8")

@pytest.mark.asyncio
async def test_second_live_general_checklist_releases():
    lens, adapter = _lens(SECOND_CANDIDATE)
    result = await lens.process(QUESTION)
    assert adapter.calls == 1
    assert result.action == InterventionAction.PASS
    assert result.response == SECOND_CANDIDATE

@pytest.mark.parametrize("answer", [
    "Retirement involves savings and taxes. Will you need to build new social connections?",
    "Retirement involves savings and taxes. You should build new social connections.",
])
def test_unrelated_directive_is_not_financial_advice(answer):
    assert FlagType.PERSONALIZED_FINANCIAL_ADVICE not in {
        f.flag_type for f in Checker(_BACKEND)._check_professional_directive(answer, QUESTION)
    }

@pytest.mark.parametrize("answer", [
    "Retirement has social factors. You should invest.",
    "Retirement has social factors. You should retire today.",
    "Your savings are in one fund. You should move them today.",
    "Retirement has social factors. Given your savings, you should stop working today.",
    "You should pay off your debt before retiring.",
])
def test_financial_directives_keep_their_boundaries(answer):
    assert FlagType.PERSONALIZED_FINANCIAL_ADVICE in {
        f.flag_type for f in Checker(_BACKEND)._check_professional_directive(answer, QUESTION)
    }

@pytest.mark.asyncio
async def test_general_generation_guidance_reaches_model_without_bypassing_checker():
    lens, adapter = _lens("Factors include health, savings, expenses, and social support.")
    seen = []
    original = adapter.generate
    async def capture(messages, **kwargs):
        seen.extend(messages)
        return await original(messages, **kwargs)
    adapter.generate = capture
    result = await lens.process(QUESTION)
    assert result.action == InterventionAction.PASS
    assert any(m["role"] == "system" and "general third-person terms" in m["content"] for m in seen)
    assert any(m["role"] == "user" and QUESTION in m["content"] for m in seen)

RELATIVE_CANDIDATE = (Path(__file__).parent / "fixtures/retirement_general_live_relative_candidate.txt").read_text(encoding="utf-8")

@pytest.mark.asyncio
async def test_live_relative_clause_checklist_releases():
    lens, adapter = _lens(RELATIVE_CANDIDATE)
    result = await lens.process(QUESTION)
    assert adapter.calls == 1
    assert result.action == InterventionAction.PASS
    assert result.response.split() == RELATIVE_CANDIDATE.split()

@pytest.mark.asyncio
async def test_relative_subject_uses_explicit_local_noun():
    from aurora_lens.pef.state import PEFState
    result = await _BACKEND.extract("Part-time roles that could bridge income gaps are another consideration.", PEFState())
    assert any(c.relation == "BRIDGE" and "roles" in c.subject for c in result.claims)
    assert not any(c.relation == "BRIDGE" and c.subject.lower() == "that" for c in result.claims)

@pytest.mark.asyncio
async def test_deictic_that_remains_unresolved():
    from aurora_lens.pef.state import PEFState
    result = await _BACKEND.extract("Should I do that?", PEFState())
    assert "that" in result.ambiguous_referents
