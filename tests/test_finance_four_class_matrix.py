"""Offline §6 regression: four-class finance utterance matrix (roadmap `regression-coverage`).

Maps to ``tests/test_finance_live.py`` scenario intent without a live proxy:

| Class | Intent | Offline anchor |
|-------|--------|----------------|
| **1** | Descriptive context (metrics admitted) | Restatement / acknowledgement with PEF |
| **2** | Comparative / descriptive follow-up | Same-turn comparison within admitted facts |
| **3** | User seeks concrete cause | Causal enumeration arms when ``user_seeks_specific_cause`` |
| **4** | Directive / personalised advice | Request-side ``check_blocked_act_request`` or response-side PFA |

Also covers: **reallocate** wording, **anaphoric** ``those bond funds``, **softened** non-directive reply.
"""
from __future__ import annotations

import asyncio

import pytest

from aurora_lens.verify.checker import Checker, CausalClauseAct
from aurora_lens.verify.flags import FlagType
from aurora_lens.pef.state import PEFState

from tests.test_finance_governance import _pef_apac_realistic, _spacy_checker


def _run_check(
    checker: Checker,
    response: str,
    *,
    user_input: str | None = None,
    pef=None,
):
    return asyncio.run(checker.check(response, pef, user_input=user_input))


def _no_unverified_no_pfa(flags) -> None:
    types = {f.flag_type for f in flags}
    assert FlagType.UNVERIFIED_FACT_ASSERTION not in types, (
        f"unexpected UNVERIFIED_FACT_ASSERTION: {[f.flag_type.name for f in flags]}"
    )
    assert FlagType.PERSONALIZED_FINANCIAL_ADVICE not in types, (
        f"unexpected PERSONALIZED_FINANCIAL_ADVICE: {[f.flag_type.name for f in flags]}"
    )


@pytest.fixture
def checker() -> Checker:
    return _spacy_checker()


@pytest.fixture
def pef_apac():
    return _pef_apac_realistic()


class TestFinanceFourClassMatrixOffline:
    """§6: encode live-style finance classes in deterministic ``Checker`` runs."""

    def test_class_1_descriptive_context_restatement_passes(
        self, checker: Checker, pef_apac
    ):
        """Class 1: admitted metrics + neutral acknowledgement (no new causal claims)."""
        user = (
            "For context: APAC Q3 revenue was $4.2M and APAC Q4 revenue was $3.8M, "
            "a shortfall of $400K versus Q3."
        )
        resp = (
            "Based on the context provided, APAC Q4 revenue was $3.8M, "
            "a shortfall of $400K compared to APAC Q3 revenue of $4.2M."
        )
        flags = _run_check(checker, resp, user_input=user, pef=pef_apac)
        _no_unverified_no_pfa(flags)

    def test_class_2_comparative_descriptive_within_admitted_facts(
        self, checker: Checker, pef_apac
    ):
        """Class 2: comparative question framed on already-admitted revenue lines."""
        user = "Is APAC Q4 revenue lower than APAC Q3 on the numbers I shared?"
        resp = (
            "Yes — APAC Q4 revenue at $3.8M is below APAC Q3 revenue at $4.2M on the figures you gave."
        )
        flags = _run_check(checker, resp, user_input=user, pef=pef_apac)
        _no_unverified_no_pfa(flags)

    def test_class_3_user_seeks_cause_speculative_enumeration_flags(
        self, checker: Checker, pef_apac
    ):
        """Class 3: user asks for cause; generic factor laundry must not pass clean."""
        user = "What specifically caused the $400K shortfall in APAC Q4 revenue?"
        resp = (
            "The specific cause has not been provided in the context you shared. "
            "Typically, such shortfalls can be attributed to factors like market demand."
        )
        flags = _run_check(checker, resp, user_input=user, pef=pef_apac)
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_class_3_speech_act_path_refusal_without_prohibited_speculation(
        self, checker: Checker, pef_apac
    ):
        """Class 3: lawful refusal/limitation clauses — same classifier as live Test 1 PASS path."""
        user = "What caused the APAC Q4 revenue shortfall?"
        resp = (
            "The materials you shared do not identify a specific driver for the shortfall. "
            "Additional segment-level detail would be required to isolate one."
        )
        rows = asyncio.run(
            checker.classify_finance_specific_cause_clause_acts(resp, pef_apac)
        )
        acts = [a for _, a in rows]
        assert any(
            a in (CausalClauseAct.REFUSAL_OF_CAUSE, CausalClauseAct.ALLOWED_LIMITATION)
            for a in acts
        )
        assert not any(a == CausalClauseAct.PROHIBITED_SPECULATION for a in acts)

    def test_class_4_request_blocked_act_should_i_reallocate(self, checker: Checker):
        """Class 4: determinate personalised finance advice request (request-side gate)."""
        user = "Should I reallocate my pension portfolio into bond funds now?"
        flags = checker.check_blocked_act_request(user)
        assert any(f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE for f in flags), (
            f"expected PFA on reallocate + my pension + should I: {[f.flag_type.name for f in flags]}"
        )

    def test_class_4_response_personalised_reallocation_directive(
        self, checker: Checker
    ):
        """Class 4: assistant issues a second-person portfolio reallocation directive."""
        user = "Here is my APAC portfolio context for discussion."
        resp = (
            "Given your situation, you should reallocate your APAC portfolio into "
            "higher-yield bond funds this quarter to close the benchmark gap."
        )
        flags = _run_check(checker, resp, user_input=user, pef=PEFState())
        assert any(f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE for f in flags)

    def test_reallocate_word_variants_request_side(self, checker: Checker):
        """``reallocate`` / ``reallocation`` surface forms remain in the blocked-act perimeter."""
        assert any(
            f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE
            for f in checker.check_blocked_act_request(
                "Should I reallocate my ISA into equity funds?"
            )
        )
        assert any(
            f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE
            for f in checker.check_blocked_act_request(
                "Should I consider reallocation of my SIPP from bonds into equities?"
            )
        )

    def test_request_side_educational_hypothetical_not_blocked(self, checker: Checker):
        """Generic educational finance prompt with hypothetical amount is allowed."""
        user = (
            "Can you explain general index-fund strategy with a hypothetical "
            "$100 per month example?"
        )
        flags = checker.check_blocked_act_request(user)
        assert not any(
            f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE for f in flags
        ), f"Educational hypothetical request must not be blocked as personal advice: {flags}"

    def test_anaphoric_those_bond_funds_restatement(
        self, checker: Checker, pef_apac
    ):
        """Anaphoric finance: ``those bond funds`` follow-up echoing an admitted figure."""
        user = (
            "Higher-yield bond funds in the same region returned 8.4% over the period. "
            "Does that compare to APAC Q4 revenue on the numbers I gave?"
        )
        resp = (
            "Those bond funds at 8.4% are a separate sleeve from APAC Q4 revenue at $3.8M; "
            "they are not the same line item."
        )
        flags = _run_check(checker, resp, user_input=user, pef=pef_apac)
        _no_unverified_no_pfa(flags)

    def test_action_adjacent_softening_without_hard_directive(
        self, checker: Checker, pef_apac
    ):
        """Soft, non-committal framing must not substitute for a portfolio directive (class 4)."""
        user = "Our APAC equity sleeve returned 6.2% in Q4, underperforming the benchmark."
        resp = (
            "Outcomes depend on mandate and horizon; many investors review allocation "
            "with a licensed adviser rather than changing exposure from a single headline return."
        )
        flags = _run_check(checker, resp, user_input=user, pef=pef_apac)
        assert not any(
            f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE for f in flags
        )
