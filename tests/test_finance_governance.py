"""Finance governance regression test pack — Layers 1, 2, and 3.

Structure
---------
Section 1  Clean trigger tests      T1 (Layer 1), T2 (Layer 2), T3 (Layer 3)
Section 2  Near-miss / suppression  N1 (hedge), N2 (PEF deferral), N3 (predictive),
                                    N4 (rounding), N5 (unit mismatch / ambiguity)
Section 3  Overlap tests            O1–O3 (multiple layers firing / interacting)
Section 4  Regression tests         R1 (Fix 1 — PEF deferral), R2 (Fix 2 — prepositional),
                                    R3 (behaviour contracts), R4 (edge cases)
Section 5  Integrated E2E           Full Lens.process() via MockAdapter

All Checker-level tests are deterministic and require no LLM call.
E2E tests use MockAdapter and asyncio — they exercise the full pipeline.
"""
from __future__ import annotations

import asyncio

import pytest

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter
from aurora_lens.config import LensConfig
from aurora_lens.interpret.pef_updater import _relation_for_pef_write, update_pef
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship, canonicalize_relation
from aurora_lens.verify.checker import (
    CausalClauseAct,
    Checker,
    _accountability_scope,
    _claim_numeric_echoes_user_context,
    _has_business_ownership,
    _is_accountability_phrase,
    _metric_echo_entity_and_numbers_align,
    classify_causal_clause_act_from_features,
    user_seeks_specific_cause,
)
from aurora_lens.verify.user_grounding import build_user_grounding_context
from aurora_lens.verify.flags import Flag, FlagType
from aurora_lens.verify.numeric import NumericUnit, parse_all_numerics, parse_numeric


def _spacy_checker() -> Checker:
    return Checker(SpacyBackend())


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_claim(
    subject: str,
    relation: str,
    obj: str,
    evidence: str | None = None,
    negated: bool = False,
) -> ExtractedClaim:
    return ExtractedClaim(
        subject=subject,
        relation=canonicalize_relation(relation),
        obj=obj,
        span=Span.PRESENT,
        negated=negated,
        evidence=evidence or f"{subject} {relation} {obj}",
    )


def _pef_with_literal(
    entity_name: str,
    relation: str,
    literal: str,
    negated: bool = False,
) -> tuple[PEFState, list[Relationship]]:
    pef = PEFState()
    entity, _ = pef.get_or_create_entity(entity_name)
    rel = Relationship(
        subject_id=entity.id,
        relation=canonicalize_relation(relation),
        object_entity_id=None,
        object_literal=literal,
        span=Span.PRESENT,
        source_turn=0,
        evidence=f"{entity_name} {relation} {literal}",
        negated=negated,
        provenance="pre_populated",
        extractor_backend="manual",
    )
    pef.add_relationship(rel)
    return pef, pef.get_relationships_for_subject(entity.id)


def _dummy_checker() -> Checker:
    class _DummyBackend:
        pass
    return Checker(_DummyBackend())  # type: ignore[arg-type]


class MockAdapter(LLMAdapter):
    def __init__(self, response: str):
        self._response = response

    async def generate(self, messages, **kwargs) -> AdapterResponse:
        return AdapterResponse(text=self._response, model="mock")


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Clean trigger tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLayer1CleanTriggers:
    """Layer 1: UNVERIFIED_FACT_ASSERTION must fire for ungrounded assertions."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # T1.1
    def test_revenue_percent_assertion_no_pef(self, checker):
        """Ungrounded revenue % assertion with no PEF."""
        flags = checker._check_unverified_quantitative_financial(
            "Revenue increased 23% this quarter."
        )
        types = [f.flag_type for f in flags]
        assert FlagType.UNVERIFIED_FACT_ASSERTION in types

    # T1.2
    def test_roi_assertion_no_pef(self, checker):
        """Ungrounded ROI assertion with no PEF."""
        flags = checker._check_unverified_quantitative_financial(
            "The projected ROI for this initiative is 34.7%."
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # T1.3
    def test_earnings_currency_assertion_no_pef(self, checker):
        """Ungrounded currency earnings assertion with no PEF."""
        flags = checker._check_unverified_quantitative_financial(
            "Earnings reached $2.8M this period."
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # T1.4
    def test_ebitda_assertion_no_pef(self, checker):
        """EBITDA is a recognised financial metric token."""
        flags = checker._check_unverified_quantitative_financial(
            "EBITDA grew to $5.1M for the quarter."
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # T1.5
    def test_margin_bps_assertion_no_pef(self, checker):
        """Ungrounded margin assertion in basis points."""
        flags = checker._check_unverified_quantitative_financial(
            "The margin is 150 bps this period."
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)


class TestLayer2CleanTriggers:
    """Layer 2: PERSONALIZED_FINANCIAL_ADVICE must fire for personalised projections."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # T2.1
    def test_based_on_your_portfolio_roi(self, checker):
        """Strong personalisation signal + ROI projection."""
        flags = checker._check_professional_directive(
            "Based on your portfolio, your projected ROI for this initiative is 34.7%."
        )
        assert any(f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE for f in flags)

    # T2.2
    def test_given_your_situation_growth(self, checker):
        """'Given your situation' personalisation signal + earnings projection."""
        flags = checker._check_professional_directive(
            "Given your situation, your earnings growth will be 12%."
        )
        assert any(f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE for f in flags)

    # T2.3
    def test_your_roi_will_be(self, checker):
        """Possessive 'your ROI' + predictive verb."""
        flags = checker._check_professional_directive(
            "Your ROI is projected to be 18% by year end."
        )
        assert any(f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE for f in flags)


class TestHistoricalMarketLookupVsPersonalizedAdvice:
    """Lookup-style user questions must not yield PFA on generic refusal text."""

    @pytest.fixture
    def checker(self) -> Checker:
        return _dummy_checker()

    _HIST_Q = (
        "What was the exact closing share price of Zephyr Labs on 3 March 2019?"
    )

    def test_refusal_without_disclaimer_regex_not_pfa(self, checker: Checker) -> None:
        """Refusals often omit the exact disclaimer boilerplate; still not PFA."""
        flags = checker._check_professional_directive(
            "I recommend consulting a licensed financial adviser. "
            "I cannot provide investment advice.",
            user_input=self._HIST_Q,
        )
        assert not any(
            f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE for f in flags
        )

    def test_rebalance_still_pfa_when_user_asked_lookup(self, checker: Checker) -> None:
        """Portfolio action in the reply still counts as personalized advice."""
        flags = checker._check_professional_directive(
            "Given your situation, you should rebalance your portfolio.",
            user_input=self._HIST_Q,
        )
        assert any(
            f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE for f in flags
        )


class TestLayer3CleanTriggers:
    """Layer 3: CONTRADICTED_FACT must fire for numeric contradictions."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # T3.1
    def test_roi_percent_contradiction(self, checker):
        """PEF: ROI IS 23% — LLM: ROI is 25%."""
        _, rels = _pef_with_literal("roi", "IS", "23%")
        claim = _make_claim("roi", "IS", "25%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None
        assert flag.flag_type == FlagType.CONTRADICTED_FACT

    # T3.2
    def test_revenue_currency_contradiction(self, checker):
        """PEF: revenue IS $3.2M — LLM: revenue reached $3.5M."""
        _, rels = _pef_with_literal("revenue", "IS", "$3.2M")
        claim = _make_claim("revenue", "IS", "$3.5M")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None
        assert flag.flag_type == FlagType.CONTRADICTED_FACT

    # T3.3
    def test_bps_to_percent_contradiction(self, checker):
        """PEF: margin IS 120 bps (=1.2%) — LLM: margin is 1.5%."""
        _, rels = _pef_with_literal("margin", "IS", "120 bps")
        claim = _make_claim("margin", "IS", "1.5%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None
        assert flag.flag_type == FlagType.CONTRADICTED_FACT

    # T3.4
    def test_different_surface_verb_contradiction(self, checker):
        """PEF: revenue IS $3.2M — LLM: revenue reached $3.5M (relation mismatch OK)."""
        _, rels = _pef_with_literal("revenue", "IS", "$3.2M")
        claim = _make_claim("revenue", "reached", "$3.5M")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None
        assert flag.flag_type == FlagType.CONTRADICTED_FACT

    # T3.5 — Fix 2 regression
    def test_prepositional_construction_contradiction(self, checker):
        """PEF: revenue IS $3.2M — LLM: revenue increased to $3.5M.
        spaCy leaves claim.obj empty; fallback to claim.evidence must find $3.5M.
        """
        _, rels = _pef_with_literal("revenue", "IS", "$3.2M")
        # Simulate what spaCy produces: obj is empty, evidence is full sentence
        claim = _make_claim(
            "revenue", "increased", "",
            evidence="revenue increased to $3.5M",
        )
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None, (
            "Layer 3 must fire via evidence fallback for prepositional constructions"
        )
        assert flag.flag_type == FlagType.CONTRADICTED_FACT

    # T3.6
    def test_growth_percent_contradiction(self, checker):
        """PEF: growth IS 12% — LLM: growth was 10%."""
        _, rels = _pef_with_literal("growth", "IS", "12%")
        claim = _make_claim("growth", "was", "10%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Near-miss / suppression tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLayer1HedgeSuppression:
    """N1: Hedge guard must suppress Layer 1."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # N1.1
    def test_may_increase_hedged(self, checker):
        """'may increase by approximately X%' is hedged — Layer 1 must not fire."""
        flags = checker._check_unverified_quantitative_financial(
            "Revenue may increase by approximately 23% this quarter."
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # N1.2
    def test_could_reach_hedged(self, checker):
        """'could reach' is a hedge — Layer 1 must not fire."""
        flags = checker._check_unverified_quantitative_financial(
            "The ROI could reach 18% by year end."
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # N1.3
    def test_tilde_hedged(self, checker):
        """Tilde prefix is a hedge signal — Layer 1 must not fire."""
        flags = checker._check_unverified_quantitative_financial(
            "Revenue is ~23% above last year."
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # N1.4
    def test_education_question_not_flagged(self, checker):
        """Definitional question ('what is ROI?') must not trigger Layer 1."""
        flags = checker._check_unverified_quantitative_financial(
            "What is ROI? Return on investment is a measure of profitability."
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)


class TestLayer1PEFDeferral:
    """N2: Layer 1 must be suppressed when PEF has an agreeing non-predictive literal."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # N2.1
    def test_pef_backed_correct_percent_no_flag(self, checker):
        """PEF: roi IS 23% — LLM: 'The ROI is 23%' — Layer 1 suppressed."""
        pef, _ = _pef_with_literal("roi", "IS", "23%")
        flags = checker._check_unverified_quantitative_financial(
            "The ROI is 23%, as established.", pef=pef
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # N2.2
    def test_pef_backed_bps_match_no_flag(self, checker):
        """PEF: margin IS 120 bps (=1.2%) — LLM: 'The margin is 1.2%' — suppressed."""
        pef, _ = _pef_with_literal("margin", "IS", "120 bps")
        flags = checker._check_unverified_quantitative_financial(
            "The margin is 1.2%.", pef=pef
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # N2.3
    def test_pef_backed_currency_match_no_flag(self, checker):
        """PEF: revenue IS $3.2M — LLM: 'Revenue was $3.2M' — suppressed."""
        pef, _ = _pef_with_literal("revenue", "IS", "$3.2M")
        flags = checker._check_unverified_quantitative_financial(
            "Revenue was $3.2M this period.", pef=pef
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # N2.4
    def test_no_pef_still_fires(self, checker):
        """With no PEF at all, Layer 1 fires as before."""
        flags = checker._check_unverified_quantitative_financial(
            "The ROI is 23%.", pef=None
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # N2.5
    def test_pef_backed_predictive_rel_still_fires(self, checker):
        """PEF has a projected (predictive) relation — Layer 1 must still fire."""
        pef, _ = _pef_with_literal("roi", "projected", "23%")
        flags = checker._check_unverified_quantitative_financial(
            "The ROI is 25%.", pef=pef
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)


class TestLayer3PredictiveSkip:
    """N3: Predictive PEF relations must not trigger Layer 3."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # N3.1
    def test_projected_rel_skipped(self, checker):
        """PEF stores a projection — contradiction check must be skipped."""
        _, rels = _pef_with_literal("roi", "projected", "23%")
        claim = _make_claim("roi", "IS", "25%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    # N3.2
    def test_will_rel_skipped(self, checker):
        """PEF relation 'will be' is predictive — must skip."""
        _, rels = _pef_with_literal("earnings", "will", "15%")
        claim = _make_claim("earnings", "IS", "12%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    # N3.3
    def test_estimated_rel_skipped(self, checker):
        """PEF relation 'estimated at' is predictive — must skip."""
        _, rels = _pef_with_literal("revenue", "estimated", "$4M")
        claim = _make_claim("revenue", "IS", "$3.5M")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None


class TestLayer3RoundingTolerance:
    """N4: Values within rounding tolerance must not flag."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # N4.1
    def test_23_vs_23_01_no_flag(self, checker):
        """23% vs 23.01% is within 0.1% relative tolerance."""
        _, rels = _pef_with_literal("roi", "IS", "23%")
        claim = _make_claim("roi", "IS", "23.01%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    # N4.2
    def test_bps_agrees_with_percent_no_flag(self, checker):
        """120 bps = 1.2% — no contradiction when values agree after normalisation."""
        _, rels = _pef_with_literal("margin", "IS", "120 bps")
        claim = _make_claim("margin", "IS", "1.2%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None


class TestLayer3UnitMismatchAndAmbiguity:
    """N5: Incompatible units and ambiguous metric tokens must not flag."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # N5.1
    def test_percent_vs_currency_no_flag(self, checker):
        """Percent vs currency are incompatible — must not flag."""
        _, rels = _pef_with_literal("revenue", "IS", "23%")
        claim = _make_claim("revenue", "IS", "$3.5M")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    # N5.2
    def test_ambiguous_metric_rate_skipped(self, checker):
        """'rate' is not a financial metric token — must skip Layer 3."""
        _, rels = _pef_with_literal("rate", "IS", "23%")
        claim = _make_claim("rate", "IS", "25%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    # N5.3
    def test_pronoun_subject_skipped(self, checker):
        """Pronoun subject 'it' must not trigger Layer 3."""
        _, rels = _pef_with_literal("roi", "IS", "23%")
        claim = _make_claim("it", "IS", "25%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    # N5.4
    def test_negated_pef_fact_skipped(self, checker):
        """Negated PEF fact must be skipped (v1 behaviour)."""
        _, rels = _pef_with_literal("roi", "IS", "23%", negated=True)
        claim = _make_claim("roi", "IS", "25%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    # N5.5
    def test_negated_claim_skipped(self, checker):
        """Negated claim must be skipped (v1 behaviour)."""
        _, rels = _pef_with_literal("roi", "IS", "23%")
        claim = _make_claim("roi", "IS", "25%", negated=True)
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    # N5.6
    def test_empty_pef_no_flag(self, checker):
        """No PEF relationships — Layer 3 cannot fire."""
        claim = _make_claim("profit", "IS", "$3.5M")
        flag = checker._check_numeric_contradiction(claim, [])
        assert flag is None


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Overlap tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerOverlaps:
    """O: Tests for interactions between Layers 1, 2, and 3."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # O1: Layer 1 + Layer 2 co-fire
    def test_layer1_and_layer2_both_fire(self, checker):
        """Personalised ungrounded projection fires both Layer 1 and Layer 2."""
        text = "Based on your portfolio, your projected ROI for this initiative is 34.7%."
        l1_flags = checker._check_unverified_quantitative_financial(text, pef=None)
        l2_flags = checker._check_professional_directive(text)
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in l1_flags)
        assert any(f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE for f in l2_flags)

    # O2: Layer 3 fires, Layer 1 suppressed for the same entity
    def test_layer3_fires_layer1_suppressed_for_same_entity(self, checker):
        """PEF: roi IS 23% — LLM: 'The ROI is 25%'.
        Layer 3 should fire CONTRADICTED_FACT.
        Layer 1 should be suppressed (PEF has a numeric literal for 'roi', but it
        doesn't agree — so Layer 1 fires too, but CONTRADICTED_FACT is the primary flag).
        """
        pef, rels = _pef_with_literal("roi", "IS", "23%")
        # Layer 3 via direct call
        claim = _make_claim("roi", "IS", "25%")
        l3_flag = checker._check_numeric_contradiction(claim, rels)
        assert l3_flag is not None
        assert l3_flag.flag_type == FlagType.CONTRADICTED_FACT
        # Layer 1 via direct call — contradicted value does NOT agree, so Layer 1 still fires
        l1_flags = checker._check_unverified_quantitative_financial(
            "The ROI is 25%.", pef=pef
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in l1_flags)

    # O3: Layer 1 suppressed when Layer 3 would also not fire (agreeing values)
    def test_layer1_suppressed_and_layer3_also_clean(self, checker):
        """PEF: roi IS 23% — LLM: 'The ROI is 23%'.
        Neither Layer 1 nor Layer 3 should fire.
        """
        pef, rels = _pef_with_literal("roi", "IS", "23%")
        # Layer 3
        claim = _make_claim("roi", "IS", "23%")
        l3_flag = checker._check_numeric_contradiction(claim, rels)
        assert l3_flag is None
        # Layer 1
        l1_flags = checker._check_unverified_quantitative_financial(
            "The ROI is 23%.", pef=pef
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in l1_flags)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Regression tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFix1PEFDeferralRegressions:
    """R1: Fix 1 — PEF deferral regressions.

    Verifies that suppression is precise: it fires when values agree and is
    absent when values disagree or PEF relation is predictive.
    """

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # R1.1
    def test_exact_percent_match_suppressed(self, checker):
        """Exact value match (23% == 23%) suppresses Layer 1."""
        pef, _ = _pef_with_literal("roi", "IS", "23%")
        flags = checker._check_unverified_quantitative_financial(
            "The ROI is 23%.", pef=pef
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # R1.2
    def test_bps_to_percent_match_suppressed(self, checker):
        """PEF has 120 bps; LLM asserts 1.2% — values agree after normalisation."""
        pef, _ = _pef_with_literal("margin", "IS", "120 bps")
        flags = checker._check_unverified_quantitative_financial(
            "The margin is 1.2%.", pef=pef
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # R1.3
    def test_disagree_does_not_suppress(self, checker):
        """PEF has 23%; LLM asserts 25% — values disagree, Layer 1 still fires."""
        pef, _ = _pef_with_literal("roi", "IS", "23%")
        flags = checker._check_unverified_quantitative_financial(
            "The ROI is 25%.", pef=pef
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # R1.4
    def test_predictive_pef_does_not_suppress(self, checker):
        """Predictive PEF ('projected 23%') must not suppress Layer 1."""
        pef, _ = _pef_with_literal("roi", "projected", "23%")
        flags = checker._check_unverified_quantitative_financial(
            "The ROI is 23%.", pef=pef
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # R1.5
    def test_no_pef_entity_match_still_fires(self, checker):
        """PEF has 'revenue' but LLM asserts 'ROI' — different entities, still fires."""
        pef, _ = _pef_with_literal("revenue", "IS", "23%")
        flags = checker._check_unverified_quantitative_financial(
            "The ROI is 23%.", pef=pef
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)


class TestFix2PrepositionalRegressions:
    """R2: Fix 2 — prepositional construction regressions.

    Verifies that evidence fallback for Layer 3 fires correctly for
    "increased to / grew to / rose to" forms while not introducing false positives.
    """

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # R2.1
    def test_increased_to_fires_layer3(self, checker):
        """'revenue increased to $3.5M' — claim.obj empty, evidence fallback fires."""
        _, rels = _pef_with_literal("revenue", "IS", "$3.2M")
        claim = _make_claim(
            "revenue", "increased", "",
            evidence="revenue increased to $3.5M",
        )
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None
        assert flag.flag_type == FlagType.CONTRADICTED_FACT

    # R2.2
    def test_grew_to_fires_layer3(self, checker):
        """'roi grew to 25%' — claim.obj empty, evidence fallback fires."""
        _, rels = _pef_with_literal("roi", "IS", "23%")
        claim = _make_claim(
            "roi", "grew", "",
            evidence="roi grew to 25%",
        )
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None
        assert flag.flag_type == FlagType.CONTRADICTED_FACT

    # R2.3
    def test_prepositional_agreement_no_flag(self, checker):
        """'revenue increased to $3.2M' with PEF $3.2M — values agree, no flag."""
        _, rels = _pef_with_literal("revenue", "IS", "$3.2M")
        claim = _make_claim(
            "revenue", "increased", "",
            evidence="revenue increased to $3.2M",
        )
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    # R2.4
    def test_empty_obj_and_empty_evidence_no_crash(self, checker):
        """Empty obj and empty evidence must not raise — just return None."""
        _, rels = _pef_with_literal("revenue", "IS", "$3.2M")
        claim = _make_claim("revenue", "increased", "", evidence="")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None


class TestBehaviourContracts:
    """R3: Behaviour contracts that must hold regardless of other changes."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # R3.1
    def test_layer3_relation_equality_not_required(self, checker):
        """PEF 'IS'; claim 'reached' — relation mismatch must not prevent Layer 3."""
        _, rels = _pef_with_literal("revenue", "IS", "$3.2M")
        claim = _make_claim("revenue", "reached", "$3.5M")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None

    # R3.2
    def test_layer3_evidence_contains_both_values(self, checker):
        """Layer 3 flag evidence must contain both PEF value and asserted value."""
        _, rels = _pef_with_literal("roi", "IS", "23%")
        claim = _make_claim("roi", "IS", "25%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None
        assert "23%" in flag.evidence
        assert "25%" in flag.evidence

    # R3.3
    def test_layer3_non_financial_text_not_parseable(self, checker):
        """'3 tablets' is not a financial numeric — must not parse as NumericValue."""
        _, rels = _pef_with_literal("revenue", "IS", "3 tablets")
        claim = _make_claim("revenue", "IS", "$3.5M")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    # R3.4
    def test_layer1_at_most_one_flag_per_response(self, checker):
        """Layer 1 returns at most one flag even when multiple segments qualify."""
        text = (
            "Revenue increased 23% this quarter. "
            "Earnings grew to $5M this year."
        )
        flags = checker._check_unverified_quantitative_financial(text, pef=None)
        fa_flags = [f for f in flags if f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION]
        assert len(fa_flags) == 1


class TestEdgeCases:
    """R4: Edge cases and boundary conditions."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # R4.1
    def test_bare_decimal_not_parsed(self, checker):
        """Bare decimal '0.23' without a unit marker must not trigger Layer 1."""
        flags = checker._check_unverified_quantitative_financial(
            "Revenue improved by 0.23 this quarter."
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # R4.2
    def test_margin_in_layout_context_not_flagged(self, checker):
        """'margin' in a CSS/layout context must not trigger Layer 1."""
        flags = checker._check_unverified_quantitative_financial(
            "The page margin is 23px on each side."
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    # R4.3
    def test_very_short_text_no_crash(self, checker):
        """Very short text must not raise and must return empty list."""
        flags = checker._check_unverified_quantitative_financial("OK", pef=None)
        assert flags == []

    # R4.4
    def test_non_financial_numeric_not_flagged(self, checker):
        """'3 tablets' as a non-financial context — Layer 1 must not fire."""
        flags = checker._check_unverified_quantitative_financial(
            "Take 3 tablets daily as recommended."
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)


# ────────────────────────────────────────────────���────────────────────────���───
# Section 4b — Arithmetic-derivation grounding (variance rule B)
# ─────────────────────────────────────────────────────────────────────────────

def _pef_budget_actual(
    scope: str,
    budget_literal: str,
    actual_literal: str,
) -> PEFState:
    """Build a PEF with a budget entity and an actual/revenue entity sharing a scope."""
    pef = PEFState()
    for entity_name, literal in [
        (f"{scope} budget", budget_literal),
        (f"{scope} revenue", actual_literal),
    ]:
        entity, _ = pef.get_or_create_entity(entity_name)
        rel = Relationship(
            subject_id=entity.id,
            relation="IS",
            object_entity_id=None,
            object_literal=literal,
            span=Span.PRESENT,
            source_turn=0,
            evidence=f"{entity_name} IS {literal}",
            provenance="pre_populated",
            extractor_backend="manual",
        )
        pef.add_relationship(rel)
    return pef


class TestVarianceArithmeticGrounding:
    """N6: Layer 1 must be suppressed when the variance amount is deterministically
    derivable from admitted budget and actual values in the same scope.

    Rule B: variance(amount) is grounded iff there exist admitted x, y in the
    same scope such that class(x)=budget and class(y)=actual and amount=|x-y|.
    """

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # N6.1 — core case
    def test_variance_equals_budget_minus_actual_suppressed(self, checker):
        """$5.2M budget, $4.8M actual → $400K revenue shortfall is derivable; must not flag.

        The sentence must contain a recognised metric token (revenue) so it enters
        the Layer 1 triple-match gate, AND a variance word (shortfall) so rule B fires.
        """
        pef = _pef_budget_actual("Q4 North America", "$5.2M", "$4.8M")
        flags = checker._check_unverified_quantitative_financial(
            "The Q4 North America revenue shortfall was $400K.", pef=pef
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags), (
            "Derivable variance should be grounded — UNVERIFIED_FACT_ASSERTION must not fire."
        )

    # N6.2 — wrong amount still flags
    def test_wrong_variance_amount_still_flags(self, checker):
        """$5.2M budget, $4.8M actual — asserting $600K revenue variance is not derivable."""
        pef = _pef_budget_actual("Q4 North America", "$5.2M", "$4.8M")
        flags = checker._check_unverified_quantitative_financial(
            "The Q4 North America revenue variance was $600K.", pef=pef
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags), (
            "$600K is not derivable from $5.2M - $4.8M; flag must still fire."
        )

    # N6.3 — no variance keyword: rule B does not apply
    def test_no_variance_keyword_still_flags(self, checker):
        """Without a variance/shortfall/gap word, rule B must not suppress.

        'revenue was $400K' has the metric token but no variance word, so rule B
        must not suppress — $400K is not a known revenue figure in PEF.
        """
        pef = _pef_budget_actual("Q4 North America", "$5.2M", "$4.8M")
        flags = checker._check_unverified_quantitative_financial(
            "Q4 North America revenue was $400K.", pef=pef
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags), (
            "No variance keyword — rule B must not apply."
        )

    # N6.4 — different scopes: rule B must not cross scopes
    def test_cross_scope_pair_not_grounded(self, checker):
        """EMEA budget + APAC actual must not ground a North America variance claim."""
        pef = PEFState()
        for entity_name, literal in [
            ("EMEA budget", "$5.2M"),
            ("APAC revenue", "$4.8M"),
        ]:
            entity, _ = pef.get_or_create_entity(entity_name)
            rel = Relationship(
                subject_id=entity.id,
                relation="IS",
                object_entity_id=None,
                object_literal=literal,
                span=Span.PRESENT,
                source_turn=0,
                evidence=f"{entity_name} IS {literal}",
                provenance="pre_populated",
                extractor_backend="manual",
            )
            pef.add_relationship(rel)
        flags = checker._check_unverified_quantitative_financial(
            "The North America revenue shortfall was $400K.", pef=pef
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags), (
            "Cross-scope pair (EMEA budget, APAC actual) must not ground a North America claim."
        )

    # N6.5 — unit mismatch: percent variance must not be grounded by currency pair
    def test_unit_mismatch_not_grounded(self, checker):
        """Currency budget/actual must not ground a percent margin variance claim."""
        pef = _pef_budget_actual("Q4 North America", "$5.2M", "$4.8M")
        flags = checker._check_unverified_quantitative_financial(
            "The Q4 North America margin variance was 8%.", pef=pef
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags), (
            "Unit mismatch (currency pair vs percent claim) must not ground the assertion."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 4f — Descriptive context acknowledgement (rule DCA)
# ─────────────────────────────────────────────────────────────────────────────
#
# When a model acknowledges a context-setting turn using context-attribution
# phrasing — "Based on the context provided, APAC Q4 revenue was $3.8M" —
# _check_professional_directive must NOT fire PERSONALIZED_FINANCIAL_ADVICE.
#
# Bug: _FIN_STRONG_PERSONALISATION_RE contains the pattern
#   the\s+(?:details?|information|facts?|context)\s+(?:you\s+)?(?:provided|given|shared)
# The (?:you\s+)? is optional, so "the context provided" (no "you") matches.
# Combined with _financial_numeric_assertion ("revenue was $3.8M"), this fires
# a HARD_STOP on plain operational context restatements.
#
# "Context provided" is a context-attribution phrase, not a personalisation
# signal.  True personalisation means "your portfolio", "your situation",
# "based on what you've described" — applied to the user's individual finances.
#
# DCA.1 and DCA.2 currently FAIL — they pin the false-positive.
# DCA.3 (genuine advice — must still fire) always PASSES.
# ─────────────────────────────────────────────────────────────────────────────

class TestDescriptiveContextAcknowledgement:
    """DCA: context-attribution acknowledgements of admitted business metrics
    must not fire PERSONALIZED_FINANCIAL_ADVICE.

    DCA.1 and DCA.2 currently FAIL.  DCA.3 must always PASS.
    """

    @pytest.fixture
    def checker(self) -> Checker:
        return _dummy_checker()

    # DCA.1 — "based on the context provided" + metric restatement
    def test_context_provided_metric_restatement_not_flagged(
        self, checker: Checker
    ):
        """'Based on the context provided, revenue was $3.8M' must not fire.

        'The context provided' is a context-attribution phrase, not a
        personalisation signal.  The metric is an admitted past fact, not
        a projection or recommendation.
        """
        flags = checker._check_professional_directive(
            "Based on the context provided, APAC Q4 revenue was $3.8M, "
            "a shortfall of $400K versus Q3."
        )
        assert not any(
            f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE for f in flags
        ), (
            "'Based on the context provided' + metric restatement must not fire "
            f"PERSONALIZED_FINANCIAL_ADVICE.  flags={[f.flag_type.name for f in flags]}"
        )

    # DCA.2 — "based on the information provided" variant
    def test_information_provided_metric_restatement_not_flagged(
        self, checker: Checker
    ):
        """'Based on the information provided, revenue was $3.8M' must not fire."""
        flags = checker._check_professional_directive(
            "Based on the information provided, APAC Q4 revenue was $3.8M."
        )
        assert not any(
            f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE for f in flags
        ), (
            "'Based on the information provided' + metric restatement must not fire "
            f"PERSONALIZED_FINANCIAL_ADVICE.  flags={[f.flag_type.name for f in flags]}"
        )

    # DCA.3 — genuine personalised advice must still fire (negative control)
    def test_genuine_personalised_financial_advice_fires(
        self, checker: Checker
    ):
        """'Based on your portfolio, your projected ROI is 12%' must fire."""
        flags = checker._check_professional_directive(
            "Based on your portfolio, your projected ROI is 12%."
        )
        assert any(
            f.flag_type == FlagType.PERSONALIZED_FINANCIAL_ADVICE for f in flags
        ), (
            "Genuine personalised financial projection must fire "
            f"PERSONALIZED_FINANCIAL_ADVICE.  flags={[f.flag_type.name for f in flags]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 4e — Speculative causal continuation (rule SCC)
# ─────────────────────────────────────────────────────────────────────────────
#
# Pattern: the model opens with a valid causal refusal ("the cause is not
# detailed in the provided information") and then immediately launders a list
# of generic speculative causes back in ("common factors … include decreased
# sales, changes in customer demand, increased competition").
#
# The checker must fire UNSUPPORTED_EVENT on the speculative continuation.
# The speculative causes are ungrounded causal claims — no causal facts exist
# in PEF, and the "INCLUDE" enumeration of causes is an event claim with zero
# PEF support.
#
# A pure refusal (no speculative continuation) must remain clean.
#
# SCC.1 currently FAILS — the checker does not detect the INCLUDE pattern.
# SCC.2 (negative control: pure refusal) must always PASS.
# ─────────────────────────────────────────────────────────────────────────────

_SPECULATIVE_RESPONSE = (
    "The specific causes of the $400K shortfall in APAC Q4 revenue are not "
    "detailed in the provided information. Common factors that can lead to "
    "revenue shortfalls include decreased sales, changes in customer demand, "
    "increased competition, or operational challenges."
)

_PURE_REFUSAL_RESPONSE = (
    "The specific causes of the $400K shortfall are not detailed in the "
    "provided information. I cannot determine the root cause from the "
    "available context."
)


class TestSpeculativeCausalContinuation:
    """SCC: A causal refusal followed by a generic speculative cause list must
    not pass cleanly.

    The model is asked why APAC Q4 revenue was short.  No causal facts exist
    in PEF.  A governed response must refuse and stop; it must not continue
    with ungrounded causal enumerations.

    The structural fingerprint: spaCy extracts the speculative clause as
    'Common factors … INCLUDE <list-of-causes>'.  No such relation exists in
    PEF.  UNSUPPORTED_EVENT must fire on the speculative continuation.

    SCC.1 currently FAILS — pins the enforcement seam.
    SCC.2 (pure refusal) currently PASSES and must continue to.
    """

    @pytest.fixture
    def checker(self) -> Checker:
        return _spacy_checker()

    @pytest.fixture
    def pef(self) -> PEFState:
        return _pef_apac_realistic()

    # SCC.1 — speculative continuation must fire UNSUPPORTED_EVENT
    def test_causal_refusal_plus_speculation_fires_unsupported_event(
        self, checker: Checker, pef: PEFState
    ):
        """Refusal + speculative cause list must raise UNSUPPORTED_EVENT.

        Response shape
        --------------
        Clause 1 (valid refusal): 'the cause is not detailed in the provided
            information'
        Clause 2 (speculative): 'Common factors … include decreased sales,
            changes in customer demand, increased competition …'

        Clause 2 is an ungrounded causal enumeration.  PEF has no causal facts
        for APAC Q4 revenue.  UNSUPPORTED_EVENT must fire on the INCLUDE claim.
        """
        flags = asyncio.run(checker.check(_SPECULATIVE_RESPONSE, pef))
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags), (
            "Speculative causal continuation (INCLUDE enumeration with no PEF support) "
            "must raise UNVERIFIED_FACT_ASSERTION.  "
            f"flags={[f.flag_type.name for f in flags]}"
        )

    # SCC.2 — pure refusal must remain clean
    def test_pure_causal_refusal_is_clean(
        self, checker: Checker, pef: PEFState
    ):
        """A response that refuses without speculative continuation must be clean.

        No UNSUPPORTED_EVENT, no UNVERIFIED_FACT_ASSERTION.
        """
        flags = asyncio.run(checker.check(_PURE_REFUSAL_RESPONSE, pef))
        assert not any(
            f.flag_type in (FlagType.UNSUPPORTED_EVENT, FlagType.UNVERIFIED_FACT_ASSERTION)
            for f in flags
        ), (
            f"Pure causal refusal must not fire overreach flags.  "
            f"flags={[f.flag_type.name for f in flags]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 4a — Comparative metric fact grounding (rule CMF)
# ─────────────────────────────────────────────────────────────────────────────
#
# When a user input contains a comparative clause —
#   "APAC Q4 revenue was $3.8M, a shortfall of $400K versus Q3"
# — the spaCy extractor jams the whole clause into one object literal:
#   APAC Q4 revenue IS '$ 3.8 M shortfall of $ 400 K versus Q3'
#
# The $400K is embedded in that literal but NOT stored as a separate PEF fact.
# As a result, when the model later says "the revenue shortfall in Q4 was $400K",
# Layer 1 fires UNVERIFIED_FACT_ASSERTION: "revenue" is a metric token, $400K
# doesn't match any PEF entity's top-level numeric, and Rule B fails because
# both APAC entities are classified as ACTUAL-class (not BUDGET-class).
#
# Target fix: when a claim's asserted numeric is embedded inside a compound
# PEF object literal (e.g. the literal contains "$400K" as a secondary value),
# Layer 1 should treat the claim as grounded.
#
# CMF.1 and CMF.2 define the desired grounding behaviour.
# CMF.3 (negative control) asserts that a wholly invented value still flags.
# ─────────────────────────────────────────────────────────────────────────────

def _pef_apac_realistic() -> PEFState:
    """PEF as actually produced by the spaCy extractor for the two Test 1 setup turns.

    spaCy produces a compound literal for the second turn:
      APAC Q4 revenue IS '$ 3.8 M shortfall of $ 400 K versus Q3'
    Both entities are PAST-span (user said 'was').
    """
    pef = PEFState()
    for entity_name, literal, turn in [
        ("APAC Q3 revenue", "$ 4.2M.", 1),
        ("APAC Q4 revenue", "$ 3.8 M shortfall of $ 400 K versus Q3", 2),
    ]:
        entity, _ = pef.get_or_create_entity(entity_name)
        pef.add_relationship(Relationship(
            subject_id=entity.id,
            relation="IS",
            object_entity_id=None,
            object_literal=literal,
            span=Span.PAST,
            source_turn=turn,
            evidence=f"For context: {entity_name} was {literal}.",
            provenance="pre_populated",
            extractor_backend="manual",
        ))
    return pef


class TestComparativeMetricFactGrounding:
    """CMF: A variance/shortfall value embedded in a compound PEF literal must
    be treated as grounded when the model acknowledges it.

    Bug: Rule B only pairs BUDGET-class vs ACTUAL-class entities.  When both
    entities are ACTUAL-class ("revenue"), no budget/actual pair is formed and
    the $400K shortfall claim is flagged.

    Fix: before raising UNVERIFIED_FACT_ASSERTION, scan compound PEF literals
    for embedded secondary numeric values that match the asserted amount.

    CMF.1 and CMF.2 currently FAIL.  CMF.3 (negative control) must always PASS.
    """

    @pytest.fixture
    def checker(self) -> Checker:
        return _dummy_checker()

    @pytest.fixture
    def pef(self) -> PEFState:
        return _pef_apac_realistic()

    # CMF.1 — "revenue shortfall" phrasing: metric token fires Layer 1
    def test_revenue_shortfall_grounded_in_compound_literal(
        self, checker: Checker, pef: PEFState
    ):
        """'The revenue shortfall in Q4 was $400K' must not raise UNVERIFIED_FACT_ASSERTION.

        The $400K is embedded in the PEF literal for APAC Q4 revenue:
          '$ 3.8 M shortfall of $ 400 K versus Q3'
        Layer 1 sees 'revenue' as a metric token and activates.  The asserted
        $400K must be grounded from the embedded secondary value in the literal.
        """
        flags = checker._check_unverified_quantitative_financial(
            "The revenue shortfall in Q4 was $400K.", pef=pef
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags), (
            "The $400K is embedded in the APAC Q4 revenue PEF literal; "
            f"UNVERIFIED_FACT_ASSERTION must not fire.  flags={flags}"
        )

    # CMF.2 — paraphrase: "APAC Q4 revenue shortfall" variant
    def test_apac_q4_revenue_shortfall_grounded(
        self, checker: Checker, pef: PEFState
    ):
        """'The APAC Q4 revenue shortfall was $400K' must not raise UNVERIFIED_FACT_ASSERTION.

        Same embedded grounding path as CMF.1.
        """
        flags = checker._check_unverified_quantitative_financial(
            "The APAC Q4 revenue shortfall was $400K.", pef=pef
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags), (
            f"UNVERIFIED_FACT_ASSERTION must not fire for APAC Q4 revenue shortfall $400K.  flags={flags}"
        )

    # CMF.3 — negative control: invented amount must still flag
    def test_invented_shortfall_amount_still_flags(
        self, checker: Checker, pef: PEFState
    ):
        """'The revenue shortfall in Q4 was $900K' must raise UNVERIFIED_FACT_ASSERTION.

        $900K is not embedded in any PEF literal and cannot be derived.
        """
        flags = checker._check_unverified_quantitative_financial(
            "The revenue shortfall in Q4 was $900K.", pef=pef
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags), (
            "$900K is not in any PEF literal — UNVERIFIED_FACT_ASSERTION must fire."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 4a-L3 — Compound literal Layer 3 exemption (rule CLE)
# ─────────────────────────────────────────────────────────────────────────────
#
# When the spaCy extractor produces a compound object literal such as
#   '$ 3.8 M shortfall of $ 400 K versus Q3'
# for entity "APAC Q4 revenue", both $3.8M (primary) and $400K (secondary)
# are admitted values from that sentence.
#
# Layer 1 Path 3 already grounds the secondary $400K value via parse_all_numerics.
# Layer 3 (_check_numeric_contradiction) must apply the same rule: if the claim
# value matches ANY numeric embedded in the compound literal, it is an
# acknowledged sub-fact — not a contradiction.
#
# Live false-positive: Turn 2 LLM acknowledgment of "APAC Q4 revenue was $3.8M,
# a shortfall of $400K versus Q3" produces a prepositional construction
# ("came in $400K below Q3") where spaCy leaves claim.obj empty.  The evidence
# fallback picks up $400K, which contradicts the primary literal value $3.8M.
# → HARD_STOP + CONTRADICTED_FACT on a setup turn. This is the exact defect.
#
# CLE.1 — core case (was: live false positive)
# CLE.2 — negative control: value absent from literal still fires
# CLE.3 — negative control: plain single-value literal still fires
# ─────────────────────────────────────────────────────────────────────────────


class TestCompoundLiteralLayer3Exemption:
    """CLE: Layer 3 must not fire CONTRADICTED_FACT when the claim value
    matches a secondary numeric embedded in a compound PEF literal.

    Root cause: _check_numeric_contradiction uses parse_numeric (first match)
    on rel.object_literal.  For the compound literal
    '$ 3.8 M shortfall of $ 400 K versus Q3', parse_numeric returns $3.8M.
    A claim whose evidence-fallback yields $400K then contradicts $3.8M.

    Fix: after values_contradict fires, check parse_all_numerics on the
    literal.  If claim_val matches any embedded value, suppress.
    """

    @pytest.fixture
    def checker(self) -> Checker:
        return _dummy_checker()

    @pytest.fixture
    def pef(self) -> PEFState:
        return _pef_apac_realistic()

    # CLE.1 — core case: prepositional construction, evidence fallback fires $400K
    def test_secondary_embedded_value_not_contradicted(
        self, checker: Checker, pef: PEFState
    ):
        """Claim value $400K from evidence fallback must not fire CONTRADICTED_FACT.

        PEF literal: '$ 3.8 M shortfall of $ 400 K versus Q3'  (compound)
        LLM phrasing:  'APAC Q4 revenue came in $400K below Q3'
        spaCy: obj='' (prepositional), evidence fallback → $400K
        Without fix: values_contradict($400K, $3.8M) → CONTRADICTED_FACT.
        With fix: $400K is embedded in the literal → suppress.
        """
        entity = pef.find_entity_by_name("APAC Q4 revenue")
        assert entity is not None
        rels = pef.get_relationships_for_subject(entity.id)

        claim = _make_claim(
            "APAC Q4 revenue", "WAS", "",
            evidence="APAC Q4 revenue came in $400K below Q3",
        )
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None, (
            "Secondary value $400K is embedded in the compound PEF literal "
            "'$ 3.8 M shortfall of $ 400 K versus Q3'; "
            "CONTRADICTED_FACT must not fire.  "
            f"Got: {flag}"
        )

    # CLE.2 — negative control: value wholly absent from compound literal
    def test_invented_value_absent_from_literal_still_contradicted(
        self, checker: Checker, pef: PEFState
    ):
        """A claim value not embedded in the compound literal must still fire.

        $5.0M is not in '$ 3.8 M shortfall of $ 400 K versus Q3'.
        CONTRADICTED_FACT must fire.
        """
        entity = pef.find_entity_by_name("APAC Q4 revenue")
        assert entity is not None
        rels = pef.get_relationships_for_subject(entity.id)

        claim = _make_claim("APAC Q4 revenue", "IS", "$5.0M")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None, (
            "$5.0M is absent from the compound literal — CONTRADICTED_FACT must fire."
        )
        assert flag.flag_type == FlagType.CONTRADICTED_FACT

    # CLE.3 — negative control: non-compound literal contradiction unaffected
    def test_plain_literal_contradiction_still_fires(self, checker: Checker):
        """A plain single-value PEF literal must still fire CONTRADICTED_FACT normally.

        No compound literal involved — $400K vs $3.8M plain comparison.
        """
        _, rels = _pef_with_literal("APAC Q4 revenue", "IS", "$3.8M")
        claim = _make_claim("APAC Q4 revenue", "IS", "$400K")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None, (
            "$400K vs plain $3.8M literal — CONTRADICTED_FACT must fire."
        )
        assert flag.flag_type == FlagType.CONTRADICTED_FACT


# ─────────────────────────────────────────────────────────────────────────────
# Section 4b — Context-acknowledgement TIME_SMEAR exemption (rule TSA)
# ─────────────────────────────────────────────────────────────────────────────
#
# When a model re-states an admitted past-span business metric in
# conversational-present tense ("is"), that is a faithful acknowledgement of
# user-provided context — not a state-transition assertion.
#
# Target rule: if the claim's numeric value exactly matches a PAST-span PEF
# fact for the same entity × relation, TIME_SMEAR must not fire.
#
# TSA.1 and TSA.2 define the desired exemption behaviour.
# TSA.3 is the negative control: a changed value must still fire TIME_SMEAR.
# ─────────────────────────────────────────────────────────────────────────────

def _pef_apac_context_turns() -> PEFState:
    """PEF built from the two Test 1 setup turns (both PAST span, as user said 'was').

    Turn 1: APAC Q3 revenue AT $4.2M  [Span.PAST]
    Turn 2: APAC Q4 revenue AT $3.8M  [Span.PAST]
    """
    pef = PEFState()
    for entity_name, literal, turn in [
        ("APAC Q3 revenue", "$4.2M", 1),
        ("APAC Q4 revenue", "$3.8M", 2),
    ]:
        entity, _ = pef.get_or_create_entity(entity_name)
        pef.add_relationship(Relationship(
            subject_id=entity.id,
            relation=canonicalize_relation("AT"),
            object_entity_id=None,
            object_literal=literal,
            span=Span.PAST,
            source_turn=turn,
            evidence=f"For context: {entity_name} was {literal}.",
            provenance="pre_populated",
            extractor_backend="manual",
        ))
    return pef


class TestContextAcknowledgementTimeSmear:
    """TSA: Faithful restatement of admitted past-span facts in conversational-present
    must NOT raise TIME_SMEAR.

    The model often responds to "For context: APAC Q4 revenue was $3.8M" with
    "APAC Q4 revenue is $3.8M, which reflects a $400K shortfall compared to
    APAC Q3 revenue of $4.2M."  The "is" tense is a conversational present used
    to acknowledge user context — not an assertion that the past fact is now the
    current state.

    Rule: exact numeric value match between a PRESENT-span claim and a PAST-span
    PEF fact for the same entity × relation licenses the claim as an
    acknowledgement, suppressing TIME_SMEAR.

    TSA.1 and TSA.2 currently FAIL — they define the target behaviour before the
    rule is added.  TSA.3 (negative control) must always PASS.
    """

    @pytest.fixture
    def checker(self) -> Checker:
        return _dummy_checker()

    @pytest.fixture
    def pef(self) -> PEFState:
        return _pef_apac_context_turns()

    # TSA.1 — direct gate: same value, conversational-present restatement
    def test_present_restatement_of_past_fact_no_time_smear(
        self, checker: Checker, pef: PEFState
    ):
        """PRESENT-span claim with value matching a PAST-span PEF fact must not raise TIME_SMEAR.

        PEF: APAC Q4 revenue AT $3.8M  [Span.PAST]
        Claim: APAC Q4 revenue AT $3.8M  [Span.PRESENT]  ← model says "is"

        This is a faithful acknowledgement of user-admitted context, not a state
        transition.  TIME_SMEAR must be suppressed.
        """
        entity = pef.find_entity_by_name("APAC Q4 revenue")
        assert entity is not None
        rels = pef.get_relationships_for_subject(entity.id)

        claim = ExtractedClaim(
            subject="APAC Q4 revenue",
            relation=canonicalize_relation("AT"),
            obj="$3.8M",
            span=Span.PRESENT,   # model used "is"
            negated=False,
            evidence="APAC Q4 revenue is $3.8M",
        )
        result = checker._check_time_smear(claim, rels, Span.PRESENT)
        assert result is None, (
            "Faithful restatement of admitted past-span fact in conversational-present "
            "must not raise TIME_SMEAR.  "
            f"Got: {result}"
        )

    # TSA.2 — Layer 1: full comparative acknowledgement sentence must not fire UNVERIFIED
    def test_comparative_acknowledgement_sentence_not_flagged_layer1(
        self, checker: Checker, pef: PEFState
    ):
        """The full model-style comparative sentence must not raise UNVERIFIED_FACT_ASSERTION.

        Sentence: 'APAC Q4 revenue is $3.8M, which reflects a $400K shortfall
                   compared to APAC Q3 revenue of $4.2M.'

        All three numeric values are grounded in the admitted PEF:
          $3.8M  — APAC Q4 revenue (direct match)
          $4.2M  — APAC Q3 revenue (direct match)
          $400K  — variance: $4.2M − $3.8M (Rule B)
        """
        sent = (
            "APAC Q4 revenue is $3.8M, which reflects a $400K shortfall "
            "compared to APAC Q3 revenue of $4.2M."
        )
        flags = checker._check_unverified_quantitative_financial(sent, pef=pef)
        assert not any(
            f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags
        ), (
            "All numeric values in the comparative acknowledgement are PEF-grounded; "
            f"UNVERIFIED_FACT_ASSERTION must not fire.  flags={flags}"
        )

    # TSA.3 — negative control: changed value must still raise TIME_SMEAR
    def test_changed_value_still_raises_time_smear(
        self, checker: Checker, pef: PEFState
    ):
        """A PRESENT-span claim with a DIFFERENT value from the PAST-span PEF fact
        is a genuine state-transition assertion — TIME_SMEAR must still fire.

        PEF: APAC Q4 revenue AT $3.8M  [Span.PAST]
        Claim: APAC Q4 revenue AT $5.1M  [Span.PRESENT]  ← different value

        This is NOT a faithful restatement.  TIME_SMEAR must not be suppressed.
        """
        entity = pef.find_entity_by_name("APAC Q4 revenue")
        assert entity is not None
        rels = pef.get_relationships_for_subject(entity.id)

        claim = ExtractedClaim(
            subject="APAC Q4 revenue",
            relation=canonicalize_relation("AT"),
            obj="$5.1M",          # different value — not a restatement
            span=Span.PRESENT,
            negated=False,
            evidence="APAC Q4 revenue is $5.1M",
        )
        result = checker._check_time_smear(claim, rels, Span.PRESENT)
        assert result is not None, (
            "A PRESENT-span claim with a changed value for a PAST-span fact is a "
            "genuine state transition — TIME_SMEAR must still fire."
        )
        assert result.flag_type == FlagType.TIME_SMEAR


# ─────────────────────────────────────────────────────────────────────────────
# Section 4c — Business-remit PEF write (HAS → MANAGE) + checker alignment
# ─────────────────────────────────────────────────────────────────────────────


class TestBusinessRemitPefWriteNormalization:
    """Portfolio-scoped possession/control is stored as MANAGE (word-boundary object gate)."""

    def test_owns_portfolio_stored_as_manage_via_update_pef(self):
        """Sarah owns … North America portfolio → PEF relationship MANAGE."""
        pef = PEFState()
        claim = ExtractedClaim(
            subject="Sarah",
            relation="HAS",
            obj="North America portfolio",
            span=Span.PRESENT,
            negated=False,
            evidence="Sarah owns the North America portfolio.",
            provenance="user_input",
            extractor_backend="spacy",
        )
        update_pef(
            ExtractionResult(claims=[claim], entity_mentions=[], span=Span.PRESENT),
            pef,
        )
        rels = pef.get_relationships_for_subject(pef.find_entity_by_name("Sarah").id)
        assert len(rels) == 1
        assert rels[0].relation == "MANAGE"

    def test_non_portfolio_ownership_stays_has(self):
        """Non–business-remit object: HAS is not rewritten to MANAGE."""
        pef = PEFState()
        claim = ExtractedClaim(
            subject="Sarah",
            relation="HAS",
            obj="a car",
            span=Span.PRESENT,
            negated=False,
            evidence="Sarah owns a car.",
            provenance="user_input",
            extractor_backend="spacy",
        )
        update_pef(
            ExtractionResult(claims=[claim], entity_mentions=[], span=Span.PRESENT),
            pef,
        )
        rels = pef.get_relationships_for_subject(pef.find_entity_by_name("Sarah").id)
        assert len(rels) == 1
        assert rels[0].relation == "HAS"


class TestPortfolioControlRelationGrounding:
    """PEF stores business-remit portfolio control as MANAGE; assistant MANAGE aligns."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    def test_manage_claim_grounds_against_pef_manage(self, checker):
        """Sarah MANAGE North America portfolio in PEF → same claim must not hallucinate."""
        pef, rels = _pef_with_has("Sarah", "North America portfolio")
        claim = ExtractedClaim(
            subject="Sarah",
            relation="MANAGE",
            obj="North America portfolio",
            span=Span.PRESENT,
            negated=False,
            evidence="Sarah manages the North America portfolio",
        )
        result = checker._check_hallucination(claim, rels, pef)
        assert result is None


class TestSameTurnUserContextEcho:
    """Same-turn user message + assistant claim numeric alignment (live suite setup turns)."""

    def test_claim_numeric_echoes_user_context_detects_margin_setup(self):
        user = "APAC Sub reported a gross margin of 34% in Q4."
        claim = ExtractedClaim(
            subject="APAC Sub",
            relation="TELL",
            obj="34% in Q4",
            span=Span.PRESENT,
            negated=False,
            evidence="APAC Sub TELL 34% in Q4",
        )
        assert _claim_numeric_echoes_user_context(claim, user)

    def test_claim_numeric_echoes_falls_back_to_full_response_when_claim_lacks_numerics(self):
        """Extraction may omit % from obj/evidence; full response still pairs with user."""
        user = "APAC Sub reported a gross margin of 34% in Q4."
        claim = ExtractedClaim(
            subject="APAC Sub",
            relation="TELL",
            obj="",
            span=Span.PRESENT,
            negated=False,
            evidence="noted",
        )
        response = (
            "Thanks for sharing; APAC Sub's gross margin for Q4 was 34% as you stated."
        )
        assert _claim_numeric_echoes_user_context(claim, user, response_text=response)

    def test_claim_numeric_echoes_normalizes_leading_determiner(self):
        """Assistant may extract 'The gross margin' while user wrote 'APAC Sub gross margin'."""
        user = (
            "I mean APAC Sub. APAC Sub gross margin in Q4 was 34%. "
            "APAC Sub gross margin in Q3 was 32%."
        )
        claim = ExtractedClaim(
            subject="The gross margin",
            relation="IS",
            obj="34% in Q4",
            span=Span.PRESENT,
            negated=False,
            evidence="The gross margin in Q4 was 34%",
        )
        assert _claim_numeric_echoes_user_context(claim, user)

    def test_claim_numeric_echoes_normalizes_possessive(self):
        user = "APAC Sub gross margin in Q4 was 34%."
        claim = ExtractedClaim(
            subject="APAC Sub's gross margin",
            relation="IS",
            obj="34%",
            span=Span.PRESENT,
            negated=False,
            evidence="APAC Sub's gross margin in Q4 was 34%",
        )
        assert _claim_numeric_echoes_user_context(claim, user)

    def test_claim_numeric_echoes_reported_margin_sentence(self):
        """User *X reported a gross margin of N%* vs subject *X gross margin* (non-contiguous)."""
        user = "EMEA Sub reported a gross margin of 31% in Q4."
        claim = ExtractedClaim(
            subject="EMEA Sub's gross margin",
            relation="IS",
            obj="31%",
            span=Span.PRESENT,
            negated=False,
            evidence="EMEA Sub's gross margin in Q4 was 31%",
        )
        assert _claim_numeric_echoes_user_context(claim, user)

    def test_metric_echo_aligns_shortfall_when_subject_not_contiguous_in_user(self):
        """§5 class 1--2: shortfall wording without *revenue* in the same clause still pairs."""
        user = "Shortfall in APAC Q4 was $400K compared to Q3."
        claim = ExtractedClaim(
            subject="APAC Q4 shortfall",
            relation="IS",
            obj="$400K",
            span=Span.PRESENT,
            negated=False,
            evidence="APAC Q4 shortfall was $400K",
        )
        assert _metric_echo_entity_and_numbers_align(
            user, "APAC Q4 shortfall", claim
        )
        assert _claim_numeric_echoes_user_context(claim, user)

    def test_metric_echo_aligns_basis_points_subject(self):
        """Benchmark gap phrasing with bps in user line pairs when entity prefix + numerics align."""
        user = (
            "Our APAC equity sleeve underperformed the benchmark by 180 bps in Q4."
        )
        claim = ExtractedClaim(
            subject="APAC Q4 bps gap",
            relation="IS",
            obj="180 bps",
            span=Span.PRESENT,
            negated=False,
            evidence="APAC Q4 bps gap was 180 bps",
        )
        assert _metric_echo_entity_and_numbers_align(
            user, "APAC Q4 bps gap", claim
        )

    def test_check_hallucination_suppresses_unsupported_event_on_echo(self):
        """Relation mismatch + echo: no UNSUPPORTED_EVENT when user already stated the numbers."""
        checker = _dummy_checker()
        pef = PEFState()
        entity, _ = pef.get_or_create_entity("APAC Sub")
        entity.resolved = True
        # Deliberately odd literal so cross-relation numeric may not align with claim wording
        rel = Relationship(
            subject_id=entity.id,
            relation="IS",
            object_entity_id=None,
            object_literal="not the assistant wording",
            span=Span.PRESENT,
            source_turn=1,
            evidence="x",
            provenance="pre_populated",
            extractor_backend="manual",
        )
        pef.add_relationship(rel)
        rels = pef.get_relationships_for_subject(entity.id)
        user = "APAC Sub reported a gross margin of 34% in Q4."
        claim = ExtractedClaim(
            subject="APAC Sub",
            relation="REPORT",
            obj="34%",
            span=Span.PRESENT,
            negated=False,
            evidence="APAC Sub reported 34%",
        )
        r = checker._check_hallucination(claim, rels, pef, user_input=user)
        assert r is None

    def test_check_hallucination_suppresses_unsupported_event_when_numeric_only_in_response(
        self,
    ):
        """Same as echo test above but structured claim fields carry no parseable %."""
        checker = _dummy_checker()
        pef = PEFState()
        entity, _ = pef.get_or_create_entity("APAC Sub")
        entity.resolved = True
        rel = Relationship(
            subject_id=entity.id,
            relation="IS",
            object_entity_id=None,
            object_literal="not the assistant wording",
            span=Span.PRESENT,
            source_turn=1,
            evidence="x",
            provenance="pre_populated",
            extractor_backend="manual",
        )
        pef.add_relationship(rel)
        rels = pef.get_relationships_for_subject(entity.id)
        user = "APAC Sub reported a gross margin of 34% in Q4."
        claim = ExtractedClaim(
            subject="APAC Sub",
            relation="REPORT",
            obj="",
            span=Span.PRESENT,
            negated=False,
            evidence="noted",
        )
        r = checker._check_hallucination(
            claim,
            rels,
            pef,
            user_input=user,
            response_text="Confirmed: 34% gross margin for APAC Sub in Q4.",
        )
        assert r is None

    def test_check_hallucination_suppresses_unsupported_event_on_shortfall_only_echo(self):
        """§5 echo anchors + §6: relation mismatch but shortfall-only user line still suppresses UNSUPPORTED_EVENT."""
        checker = _dummy_checker()
        pef = PEFState()
        entity, _ = pef.get_or_create_entity("APAC Q4 shortfall")
        entity.resolved = True
        rel = Relationship(
            subject_id=entity.id,
            relation="IS",
            object_entity_id=None,
            object_literal="not the assistant wording",
            span=Span.PRESENT,
            source_turn=1,
            evidence="x",
            provenance="pre_populated",
            extractor_backend="manual",
        )
        pef.add_relationship(rel)
        rels = pef.get_relationships_for_subject(entity.id)
        user = "Shortfall in APAC Q4 was $400K compared to Q3."
        claim = ExtractedClaim(
            subject="APAC Q4 shortfall",
            relation="REPORT",
            obj="$400K",
            span=Span.PRESENT,
            negated=False,
            evidence="APAC Q4 shortfall reported $400K",
        )
        r = checker._check_hallucination(claim, rels, pef, user_input=user)
        assert r is None


# ─────────────────────────────────────────────────────────────────────────────
# Section 4c — Accountability derivation (rule A)
# ─────────────────────────────────────────────────────────────────────────────

def _pef_with_has(entity_name: str, has_object: str) -> tuple[PEFState, list]:
    """Build a PEF with a literal object; relation matches update_pef write rules."""
    pef = PEFState()
    entity, _ = pef.get_or_create_entity(entity_name)
    probe = ExtractedClaim(
        subject=entity_name,
        relation="HAS",
        obj=has_object,
        span=Span.PRESENT,
        negated=False,
        evidence="",
        provenance="pre_populated",
        extractor_backend="manual",
    )
    stored = _relation_for_pef_write(probe)
    rel = Relationship(
        subject_id=entity.id,
        relation=stored,
        object_entity_id=None,
        object_literal=has_object,
        span=Span.PRESENT,
        source_turn=0,
        evidence=f"{entity_name} {stored} {has_object}",
        provenance="pre_populated",
        extractor_backend="manual",
    )
    pef.add_relationship(rel)
    return pef, pef.get_relationships_for_subject(entity.id)


class TestAccountabilityScope:
    """Unit tests for _is_accountability_phrase and _accountability_scope."""

    def test_bare_accountable(self):
        assert _is_accountability_phrase("accountable")
        assert _accountability_scope("accountable") is None

    def test_bare_responsible(self):
        assert _is_accountability_phrase("responsible")
        assert _accountability_scope("responsible") is None

    def test_accountable_for_expands(self):
        assert _is_accountability_phrase("accountable for the North America portfolio")
        scope = _accountability_scope("accountable for the North America portfolio")
        assert scope == {"north", "america", "portfolio"}

    def test_responsible_for_expands(self):
        assert _is_accountability_phrase("responsible for North America")
        scope = _accountability_scope("responsible for North America")
        assert scope == {"north", "america"}

    def test_in_charge_of_expands(self):
        assert _is_accountability_phrase("in charge of the portfolio")
        scope = _accountability_scope("in charge of the portfolio")
        assert scope == {"portfolio"}

    def test_non_accountability_phrase(self):
        assert not _is_accountability_phrase("delayed")
        assert not _is_accountability_phrase("manager")


class TestAccountabilityDerivedGrounding:
    """A4: 'X IS accountable' must be grounded when X HAS/MANAGE a business object
    and the claimed scope matches the owned object.

    Rule A: HAS or MANAGE [business-object] → IS accountable for that object only.
    """

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    # A4.1 — bare accountability: no scope restriction
    def test_bare_accountable_licensed_by_portfolio_has(self, checker):
        """Sarah MANAGE North America portfolio → IS accountable must not flag."""
        pef, rels = _pef_with_has("Sarah", "North America portfolio")
        claim = _make_claim("Sarah", "IS", "accountable")
        flags = checker._check_hallucination(claim, rels, pef)
        assert flags is None or not any(
            f.flag_type in (FlagType.UNSUPPORTED_EVENT, FlagType.UNSUPPORTED_ATTRIBUTE)
            for f in ([] if flags is None else [flags])
        ), "Bare IS accountable must be grounded by MANAGE North America portfolio."

    # A4.2 — scoped accountability: scope matches owned object
    def test_scoped_accountable_matches_has_object(self, checker):
        """Sarah MANAGE North America portfolio → IS accountable for North America portfolio ✓"""
        pef, rels = _pef_with_has("Sarah", "North America portfolio")
        claim = _make_claim("Sarah", "IS", "accountable for North America portfolio")
        flags = checker._check_hallucination(claim, rels, pef)
        assert flags is None or not any(
            f.flag_type in (FlagType.UNSUPPORTED_EVENT, FlagType.UNSUPPORTED_ATTRIBUTE)
            for f in ([] if flags is None else [flags])
        ), "IS accountable for North America portfolio must be grounded by matching MANAGE."

    # A4.3 — scope mismatch: must still flag
    def test_scoped_accountable_wrong_scope_flags(self, checker):
        """Sarah MANAGE North America portfolio → IS accountable for EMEA portfolio must flag."""
        pef, rels = _pef_with_has("Sarah", "North America portfolio")
        claim = _make_claim("Sarah", "IS", "accountable for EMEA portfolio")
        result = checker._check_hallucination(claim, rels, pef)
        assert result is not None, (
            "IS accountable for EMEA must flag — scope does not match North America MANAGE."
        )

    # A4.4 — non-business HAS: must still flag
    def test_non_business_has_does_not_license_accountability(self, checker):
        """Sarah HAS a cold → IS accountable must still flag (non-business object)."""
        pef, rels = _pef_with_has("Sarah", "a cold")
        claim = _make_claim("Sarah", "IS", "accountable")
        result = checker._check_hallucination(claim, rels, pef)
        assert result is not None, (
            "IS accountable must flag when HAS object is not a business responsibility object."
        )

    # A4.5 — no HAS/MANAGE at all: must still flag
    def test_no_has_relation_still_flags(self, checker):
        """Sarah IS director (some other relation) → IS accountable must flag."""
        pef, _ = _pef_with_literal("Sarah", "IS", "director")
        rels = pef.get_relationships_for_subject(
            pef.find_entity_by_name("Sarah").id
        )
        claim = _make_claim("Sarah", "IS", "accountable")
        result = checker._check_hallucination(claim, rels, pef)
        assert result is not None, (
            "IS accountable must flag when entity has no HAS/MANAGE relations."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 4d — Final-turn business summary (Test 4 checker coverage)
# ─────────────────────────────────────────────────────────────────────────────

def _pef_test4_admitted() -> PEFState:
    """Build the PEF representing exactly the four facts admitted in Test 4 setup turns.

    Admitted facts
    --------------
    Q4 North America budget    IS  $5.2M
    Actual Q4 North America revenue  AT  $4.8M   (comes in as AT from "came in at")
    primary variance driver    IS  delayed enterprise deals
    Sarah                      MANAGE North America portfolio
    """
    pef = PEFState()
    facts = [
        ("Q4 North America budget",             "IS",  "$5.2M"),
        ("Actual Q4 North America revenue",     "AT",  "$4.8M"),
        ("primary variance driver",             "IS",  "delayed enterprise deals"),
        ("Sarah",                               "MANAGE", "North America portfolio"),
    ]
    for entity_name, relation, literal in facts:
        entity, _ = pef.get_or_create_entity(entity_name)
        pef.add_relationship(Relationship(
            subject_id=entity.id,
            relation=canonicalize_relation(relation),
            object_entity_id=None,
            object_literal=literal,
            span=Span.PRESENT,
            source_turn=0,
            evidence=f"{entity_name} {relation} {literal}",
            provenance="pre_populated",
            extractor_backend="manual",
        ))
    return pef


def _no_unsupported(checker: Checker, claim: ExtractedClaim, pef: PEFState) -> bool:
    """Return True iff _check_hallucination does NOT raise UNSUPPORTED_EVENT/ATTRIBUTE."""
    rels = pef.get_relationships_for_subject(
        pef.find_entity_by_name(claim.subject).id
        if pef.find_entity_by_name(claim.subject) is not None else ""
    ) if pef.find_entity_by_name(claim.subject) is not None else []
    result = checker._check_hallucination(claim, rels, pef)
    if result is None:
        return True
    return result.flag_type not in (FlagType.UNSUPPORTED_EVENT, FlagType.UNSUPPORTED_ATTRIBUTE)


class TestFinalTurnBusinessSummary:
    """Section 4d: Test 4 final-turn grounding.

    The three claims the LLM makes in the final turn of the finance workflow must
    all be supportable from the four admitted PEF facts — no hallucination flags.

    A4d.1  Exact grounded summary   — three canonical claims, clean PEF
    A4d.2  Paraphrase tolerance     — slight restatements of the same facts
    A4d.3  Negative control         — EMEA accountability claim must still flag
    """

    @pytest.fixture
    def checker(self) -> Checker:
        return _dummy_checker()

    @pytest.fixture
    def pef(self) -> PEFState:
        return _pef_test4_admitted()

    # A4d.1 — exact grounded final-answer
    def test_final_turn_business_summary_grounded_from_admitted_pef(
        self, checker: Checker, pef: PEFState
    ):
        """The canonical three-claim final answer must not raise any hallucination flag.

        Claims under test
        -----------------
        variance        IS  $400,000       (derived: $5.2M − $4.8M via Rule B)
        primary variance driver  IS  delayed enterprise deals  (direct PEF match)
        Sarah           IS  accountable for the North America portfolio  (Rule A)
        """
        # Variance claim — checked via Layer 1 (raw text), so we go through
        # _check_unverified_quantitative_financial with the full PEF
        variance_sent = (
            "The Q4 variance for North America is $400,000."
            " The primary variance driver is delayed enterprise deals."
            " Sarah is accountable for the North America portfolio."
        )
        layer1_flags = checker._check_unverified_quantitative_financial(
            variance_sent, pef=pef
        )
        assert not any(
            f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in layer1_flags
        ), f"Layer 1 must not flag grounded final-turn sentence; flags={layer1_flags}"

        # Accountability claim — Rule A via _check_hallucination
        claim_sarah = _make_claim(
            "Sarah", "IS", "accountable for the North America portfolio"
        )
        assert _no_unsupported(checker, claim_sarah, pef), (
            "Sarah IS accountable for the North America portfolio must be grounded "
            "by Sarah MANAGE North America portfolio (Rule A)."
        )

    # A4d.2 — bounded paraphrase tolerance
    def test_paraphrase_tolerance_slight_restatement(
        self, checker: Checker, pef: PEFState
    ):
        """Slight restatements of admitted facts must not be flagged.

        Paraphrases under test
        ----------------------
        'The Q4 variance for North America is $400,000.'   — same numerics, Rule B
        'The primary variance driver was the delay in enterprise deals.'
            — 'delay in enterprise deals' vs PEF 'delayed enterprise deals'
            — Layer 1 raw text scan: no metric token, no numeric → no flag
        'Sarah is accountable, as she oversees the North America portfolio.'
            — bare IS accountable (Rule A, bare form)
        """
        # Paraphrased variance sentence — Rule B should still ground it
        para_sent = (
            "The Q4 variance for North America is $400,000."
            " The primary variance driver was the delay in enterprise deals."
            " Sarah is accountable, as she oversees the North America portfolio."
        )
        layer1_flags = checker._check_unverified_quantitative_financial(
            para_sent, pef=pef
        )
        assert not any(
            f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in layer1_flags
        ), f"Paraphrased final-turn sentence must not fire Layer 1; flags={layer1_flags}"

        # Bare accountability — Rule A bare form
        claim_bare = _make_claim("Sarah", "IS", "accountable")
        assert _no_unsupported(checker, claim_bare, pef), (
            "Bare 'Sarah IS accountable' must be grounded by Sarah MANAGE North America portfolio."
        )

    # A4d.3 — negative control: wrong scope must still flag
    def test_wrong_accountability_scope_still_flags(
        self, checker: Checker, pef: PEFState
    ):
        """Sarah IS accountable for EMEA must still raise UNSUPPORTED_EVENT.

        PEF only admits 'Sarah MANAGE North America portfolio'.
        The EMEA scope does not match — Rule A must not fire.
        """
        claim_emea = _make_claim("Sarah", "IS", "accountable for EMEA")
        rels = pef.get_relationships_for_subject(
            pef.find_entity_by_name("Sarah").id
        )
        result = checker._check_hallucination(claim_emea, rels, pef)
        assert result is not None, (
            "IS accountable for EMEA must flag — Sarah MANAGE North America portfolio "
            "does not license EMEA accountability."
        )
        assert result.flag_type in (
            FlagType.UNSUPPORTED_EVENT, FlagType.UNSUPPORTED_ATTRIBUTE
        )


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Integrated E2E scenario
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegratedE2EFinanceScenario:
    """Full Lens.process() pipeline tests for finance governance.

    Uses MockAdapter for deterministic LLM responses and pre-populated PEF.
    """

    def _lens(self, response: str, pef: PEFState | None = None) -> Lens:
        adapter = MockAdapter(response)
        config = LensConfig(adapter=adapter, inject_pef_context=False)
        return Lens(config, initial_pef=pef)

    def _pef(self, entity: str, relation: str, literal: str) -> PEFState:
        pef = PEFState()
        e, _ = pef.get_or_create_entity(entity)
        pef.add_relationship(Relationship(
            subject_id=e.id,
            relation=canonicalize_relation(relation),
            object_entity_id=None,
            object_literal=literal,
            span=Span.PRESENT,
            source_turn=0,
            evidence=f"{entity} {relation} {literal}",
            provenance="pre_populated",
            extractor_backend="manual",
        ))
        return pef

    # E2E-1
    def test_correct_pef_backed_response_not_flagged(self):
        """PEF: roi IS 23% — LLM: 'The ROI is 23%' — Layer 1 suppressed."""
        pef = self._pef("roi", "IS", "23%")
        lens = self._lens("The ROI is 23%, as established.", pef=pef)
        result = _run(lens.process("What is the ROI?"))
        flag_types = [f.flag_type for f in result.flags]
        assert FlagType.UNVERIFIED_FACT_ASSERTION not in flag_types
        assert FlagType.CONTRADICTED_FACT not in flag_types

    # E2E-2
    def test_contradicted_pef_response_fires_layer3(self):
        """PEF: roi IS 23% — LLM: 'The ROI is 25%' — CONTRADICTED_FACT fires."""
        pef = self._pef("roi", "IS", "23%")
        lens = self._lens("Based on our records, the ROI is 25%.", pef=pef)
        result = _run(lens.process("What is our ROI?"))
        flag_types = [f.flag_type for f in result.flags]
        assert FlagType.CONTRADICTED_FACT in flag_types

    # E2E-3
    def test_ungrounded_assertion_fires_layer1(self):
        """No PEF — LLM asserts 'Revenue increased 23%' — Layer 1 fires."""
        lens = self._lens("Revenue increased 23% this quarter.", pef=None)
        result = _run(lens.process("How did revenue perform?"))
        flag_types = [f.flag_type for f in result.flags]
        assert FlagType.UNVERIFIED_FACT_ASSERTION in flag_types

    # E2E-4
    def test_personalized_projection_fires_layer2(self):
        """Personalised ROI projection fires PERSONALIZED_FINANCIAL_ADVICE (HARD_STOP)."""
        lens = self._lens(
            "Based on your portfolio, your projected ROI is 34.7%.", pef=None
        )
        result = _run(lens.process("What ROI should I expect?"))
        flag_types = [f.flag_type for f in result.flags]
        assert FlagType.PERSONALIZED_FINANCIAL_ADVICE in flag_types

    # E2E-5
    def test_hedged_assertion_clean(self):
        """Hedged assertion ('may increase by approximately 23%') — no Layer 1."""
        lens = self._lens(
            "Revenue may increase by approximately 23% this quarter.", pef=None
        )
        result = _run(lens.process("What might revenue do?"))
        flag_types = [f.flag_type for f in result.flags]
        assert FlagType.UNVERIFIED_FACT_ASSERTION not in flag_types

    # E2E-6: Known spaCy pipeline limitation — prepositional construction
    def test_prepositional_construction_layer1_fires_layer3_absent_e2e(self):
        """Full pipeline: 'increased to $3.5M' — spaCy places numeric in pobj so
        claim.obj is empty and the evidence-fallback in Layer 3 is only reached if
        spaCy emits a claim with the evidence sentence.  In practice the full pipeline
        sees Layer 1 fire (raw text scan) but Layer 3 does not — the direct-method
        tests in R2.1/R2.2 verify that _check_numeric_contradiction itself works.
        This test documents the known boundary of the full-pipeline behaviour.
        """
        pef = self._pef("revenue", "IS", "$3.2M")
        lens = self._lens("Revenue increased to $3.5M this period.", pef=pef)
        result = _run(lens.process("What was revenue?"))
        flag_types = [f.flag_type for f in result.flags]
        # Layer 1 fires (ungrounded numeric — Layer 1 can't see PEF for this form)
        assert FlagType.UNVERIFIED_FACT_ASSERTION in flag_types
        # Layer 3 is absent in the full pipeline for prepositional constructions —
        # this is a known v1 spaCy extraction limitation, not a checker bug.
        assert FlagType.CONTRADICTED_FACT not in flag_types


# ─────────────────────────────────────────────────────────────────────────────
# Phase B/D — user-input grounding, broad PEF, international currency, dedup
# ─────────────────────────────────────────────────────────────────────────────


class TestLayer1UserInputGrounding:
    """User-admitted numerics ground benign Layer-1-shaped echoes."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    def test_user_message_numeric_suppresses_layer1(self, checker):
        flags = checker._check_unverified_quantitative_financial(
            "The revenue was $4.2M for that quarter.",
            pef=None,
            user_input="For context: APAC Q3 revenue was $4.2M.",
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)


class TestLayer1BroadPEFDeferral:
    """Broad deferral: metric token not in entity name but value in PEF literal."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    def test_broad_pef_grounds_revenue_on_scope_entity(self, checker):
        pef = PEFState()
        entity, _ = pef.get_or_create_entity("APAC")
        rel = Relationship(
            subject_id=entity.id,
            relation="IS",
            object_entity_id=None,
            object_literal="$4.2M",
            span=Span.PRESENT,
            source_turn=0,
            evidence="APAC IS $4.2M",
            negated=False,
            provenance="test",
            extractor_backend="manual",
        )
        pef.add_relationship(rel)
        flags = checker._check_unverified_quantitative_financial(
            "Q3 revenue was $4.2M.",
            pef=pef,
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)


class TestInternationalCurrencyNumeric:
    """parse_numeric / parse_all_numerics accept major non-USD presentations."""

    def test_euro_sign_million(self):
        v = parse_numeric("The budget is €5.2M.")
        assert v is not None
        assert v.unit == NumericUnit.CURRENCY
        assert abs(v.magnitude - 5.2e6) / 5.2e6 < 0.01

    def test_iso_suffix_eur(self):
        v = parse_numeric("Revenue reached 4.2m EUR this quarter.")
        assert v is not None
        assert v.unit == NumericUnit.CURRENCY
        assert abs(v.magnitude - 4.2e6) / 4.2e6 < 0.01

    def test_gbp_sign(self):
        v = parse_numeric("£500k allocation")
        assert v is not None
        assert abs(v.magnitude - 500_000) < 1

    def test_parse_all_euro_and_dollar_distinct_spans(self):
        vals = parse_all_numerics("€1M and $2M")
        assert len(vals) == 2


class TestSpeculativeCausalEnumeration:
    """Text-level guard: factors like / refusal + speculative tail."""

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    def test_factors_like_triggers(self, checker):
        flags = checker._check_speculative_causal_enumeration(
            "Revenue may be due to factors like demand and competition."
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_attributed_to_factors_like_triggers(self, checker):
        flags = checker._check_speculative_causal_enumeration(
            "Shortfalls can be attributed to factors like timing and mix."
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_refusal_then_typically_shortfalls_triggers(self, checker):
        mixed = (
            "The specific cause of the variance has not been provided in the materials. "
            "Typically, such shortfalls can be attributed to factors like market demand."
        )
        flags = checker._check_speculative_causal_enumeration(mixed)
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_pure_refusal_first_sentence_only_clean(self, checker):
        flags = checker._check_speculative_causal_enumeration(
            "The specific cause has not been provided in the context you shared."
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_factors_like_skipped_when_user_did_not_seek_cause(self, checker):
        """Context-only user turn: do not flag generic causal phrasing in the reply."""
        text = "Thank you. Factors such as sector rotation may matter."
        user = (
            "Our APAC equity investment portfolio returned 6.2% in Q4, "
            "underperforming the benchmark by 180 basis points."
        )
        flags = checker._check_speculative_causal_enumeration(text, user_input=user)
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_factors_like_skipped_when_user_mentions_root_cause_without_seeking(self, checker):
        """Setup that mentions root cause (but does not ask for one) must not arm causal scans."""
        user = (
            "The Q4 deck does not include a root cause breakdown for the APAC shortfall; "
            "only the $400K figure is stated."
        )
        text = "Thanks. Revenue may be due to factors like demand and competition."
        flags = checker._check_speculative_causal_enumeration(text, user_input=user)
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_factors_like_skipped_when_user_reports_cannot_determine_reason(self, checker):
        """``cannot determine the reason`` setup must not arm causal enumeration on the reply."""
        user = (
            "We cannot determine the reason for the variance from the figures shared; "
            "only Q4 APAC revenue at $12.4M is in scope."
        )
        text = "Understood. Factors such as mix and timing may be relevant."
        flags = checker._check_speculative_causal_enumeration(text, user_input=user)
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_factors_like_still_triggers_when_user_asked_cause(self, checker):
        user = "What caused the $400K shortfall?"
        flags = checker._check_speculative_causal_enumeration(
            "Revenue may be due to factors like demand and competition.",
            user_input=user,
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_could_be_due_to_various_triggers_when_user_asked_why(self, checker):
        """RAG-style why answers sometimes pad with ungrounded modal + quantifier lists."""
        user = "Why did Emma's sister move back from Canada?"
        text = (
            "The reason isn't provided in the given context. "
            "It could be due to various personal or professional reasons, "
            "but without specific information, I can't determine the exact reason."
        )
        flags = checker._check_speculative_causal_enumeration(text, user_input=user)
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_could_be_due_to_specific_named_fact_does_not_trigger(self, checker):
        """Grounded 'could be due to <concrete referent>' must not match the open-domain quantifier rule."""
        user = "Why was the shipment delayed?"
        text = "From the email, the delay could be due to the customs hold described in paragraph two."
        flags = checker._check_speculative_causal_enumeration(text, user_input=user)
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)


class TestAbsenceWithSpeculativeFiller:
    """Evidential absence + generic speculation in one reply -> UNVERIFIED_FACT_ASSERTION."""

    _RAG_HARNESS_USER = (
        "Context:\n### Section 1\nBody line.\n\n"
        "Question: Why did Emma's sister move back from Canada?"
    )

    @pytest.fixture
    def checker(self):
        return _dummy_checker()

    def test_absence_plus_speculation_triggers(self, checker):
        text = (
            "The reason is not provided in the context. "
            "It could be due to job opportunities or family commitments."
        )
        flags = checker._check_absence_with_speculative_filler(
            text, user_input=self._RAG_HARNESS_USER
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_information_given_absence_plus_vary_widely_speculation_triggers(self, checker):
        """RAG Q4: 'information given' hedge + 'vary widely' / factor scaffolding."""
        text = (
            "The reason why Emma's sister moved back from Canada is not provided in the information given. "
            "Factors influencing such a move can vary widely, including personal, professional, "
            "or family-related reasons."
        )
        flags = checker._check_absence_with_speculative_filler(
            text, user_input=self._RAG_HARNESS_USER
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_absence_only_no_flag(self, checker):
        text = (
            "The context does not specify which city. "
            "I cannot answer without that information."
        )
        flags = checker._check_absence_with_speculative_filler(
            text, user_input=self._RAG_HARNESS_USER
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_speculation_only_no_flag(self, checker):
        text = "She may have relocated for work."
        flags = checker._check_absence_with_speculative_filler(
            text, user_input=self._RAG_HARNESS_USER
        )
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_dont_have_specific_information_about_why_plus_speculation(self, checker):
        text = (
            "I don't have specific information about why she moved. "
            "It could be for job opportunities."
        )
        flags = checker._check_absence_with_speculative_filler(
            text, user_input=self._RAG_HARNESS_USER
        )
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_absence_plus_speculation_skipped_without_rag_harness_shape(self, checker):
        """Non-RAG user_input: absence+speculation does not use this RAG-only seam."""
        text = (
            "The reason is not provided in the context. "
            "It could be due to job opportunities or family commitments."
        )
        flags = checker._check_absence_with_speculative_filler(text, user_input="Plain chat, no context block.")
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)


class TestUserSeeksSpecificCausePerimeter:
    def test_why_did_triggers(self):
        assert user_seeks_specific_cause("Why did APAC Q4 revenue miss budget?")

    def test_what_caused_triggers(self):
        assert user_seeks_specific_cause("What caused the $400K shortfall?")

    def test_root_cause_triggers(self):
        assert user_seeks_specific_cause("Identify the root cause of the variance.")

    def test_metric_definition_does_not_trigger(self):
        assert not user_seeks_specific_cause("What is the meaning of revenue variance?")

    def test_root_cause_mention_in_setup_does_not_trigger(self):
        """Class 1--2: citing absence of root-cause analysis is not a cause-seeking act."""
        assert not user_seeks_specific_cause(
            "The Q4 materials do not contain a root cause analysis for the APAC shortfall."
        )

    def test_cannot_determine_reason_in_setup_does_not_trigger(self):
        """Class 1--2: reporting inability to determine cause from materials is not seeking."""
        assert not user_seeks_specific_cause(
            "We cannot determine the reason for the APAC shortfall from the packet as provided."
        )

    def test_why_could_we_not_determine_still_seeks(self):
        """Interrogative: inability phrasing inside a question still arms the perimeter."""
        assert user_seeks_specific_cause(
            "Why could we not determine the cause of the miss from the materials?"
        )


class TestClassifyCausalClauseActFromFeatures:
    def test_structural_include_with_scaffold_subject(self):
        claim = ExtractedClaim(
            subject="common factors that can contribute to shortfalls",
            relation="INCLUDE",
            obj="decreased sales",
            span=Span.PRESENT,
            negated=False,
            evidence="",
        )
        ex = ExtractionResult(claims=[claim])
        act = classify_causal_clause_act_from_features(
            extraction=ex,
            clause_lower="common factors that can contribute to shortfalls.",
        )
        assert act == CausalClauseAct.PROHIBITED_SPECULATION

    def test_prohibited_lex_wins_over_allowed_hint(self):
        ex = ExtractionResult(claims=[])
        low = (
            "you would need more data to say. factors such as pricing, mix, and churn "
            "often matter."
        )
        assert classify_causal_clause_act_from_features(extraction=ex, clause_lower=low) == (
            CausalClauseAct.PROHIBITED_SPECULATION
        )

    def test_allowed_limitation_without_enumeration(self):
        ex = ExtractionResult(claims=[])
        low = "additional sales mix and channel data would be required to isolate the driver."
        assert classify_causal_clause_act_from_features(extraction=ex, clause_lower=low) == (
            CausalClauseAct.ALLOWED_LIMITATION
        )

    def test_refusal_of_cause_after_allowed_ruled_out(self):
        ex = ExtractionResult(claims=[])
        low = "the specific cause has not been provided in the materials."
        assert classify_causal_clause_act_from_features(extraction=ex, clause_lower=low) == (
            CausalClauseAct.REFUSAL_OF_CAUSE
        )

    def test_refusal_of_cause_not_been_detailed_variant(self):
        ex = ExtractionResult(claims=[])
        low = "the specific reasons for the shortfall have not been detailed."
        assert classify_causal_clause_act_from_features(extraction=ex, clause_lower=low) == (
            CausalClauseAct.REFUSAL_OF_CAUSE
        )

    def test_refusal_of_cause_not_provided_available_data_variant(self):
        ex = ExtractionResult(claims=[])
        low = "the specific reasons for the shortfall are not provided in the available data."
        assert classify_causal_clause_act_from_features(extraction=ex, clause_lower=low) == (
            CausalClauseAct.REFUSAL_OF_CAUSE
        )

    def test_refusal_of_cause_dont_have_information_specific_causes_variant(self):
        ex = ExtractionResult(claims=[])
        low = "i don't have information on the specific causes of the $400k shortfall."
        assert classify_causal_clause_act_from_features(extraction=ex, clause_lower=low) == (
            CausalClauseAct.REFUSAL_OF_CAUSE
        )

    def test_prohibited_speculation_factors_commonly_contribute_include_variant(self):
        ex = ExtractionResult(claims=[])
        low = "factors that commonly contribute to shortfalls can include pricing pressure and mix shifts."
        assert classify_causal_clause_act_from_features(extraction=ex, clause_lower=low) == (
            CausalClauseAct.PROHIBITED_SPECULATION
        )


class TestSpecificCauseClausePolicyIntegration:
    """Async check(): refusal-of-cause then prohibited speculation when user asked for cause."""

    @pytest.fixture
    def checker(self) -> Checker:
        return _spacy_checker()

    @pytest.fixture
    def pef(self) -> PEFState:
        return _pef_apac_realistic()

    def test_user_asks_cause_refusal_then_typical_shortfalls_flags(
        self, checker: Checker, pef: PEFState
    ):
        user = "What specifically caused the APAC Q4 revenue shortfall?"
        resp = (
            "The specific cause has not been provided in the context you shared. "
            "Typically, such shortfalls can be attributed to factors like market demand."
        )
        flags = asyncio.run(checker.check(resp, pef, user_input=user))
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_user_asks_cause_not_detailed_then_possible_factors_flags(
        self, checker: Checker, pef: PEFState
    ):
        user = "What specifically caused the APAC Q4 revenue shortfall?"
        resp = (
            "The specific reasons for the $400K shortfall in APAC Q4 revenue have not been detailed. "
            "Possible contributing factors could include demand shifts and operational challenges."
        )
        flags = asyncio.run(checker.check(resp, pef, user_input=user))
        assert any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)

    def test_user_asks_cause_pure_refusal_stays_clean(self, checker: Checker, pef: PEFState):
        user = "What caused the APAC Q4 revenue shortfall?"
        flags = asyncio.run(checker.check(_PURE_REFUSAL_RESPONSE, pef, user_input=user))
        assert not any(
            f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags
        ), f"flags={[f.flag_type.name for f in flags]}"

    def test_user_cannot_determine_reason_no_unverified_on_factors(self, checker: Checker, pef: PEFState):
        user = (
            "We cannot determine the reason for the variance from the packet as provided."
        )
        resp = "Understood. Factors such as mix and timing may be relevant to the line."
        flags = asyncio.run(checker.check(resp, pef, user_input=user))
        assert not any(f.flag_type == FlagType.UNVERIFIED_FACT_ASSERTION for f in flags)


class TestUserGroundingContextBridge:
    """Same-turn user_input PEF provenance snapshot grounds Layer 1 without user-text numerics."""

    def test_layer1_financial_restatement_via_bridge_without_user_numeric_echo(self):
        """Broad PEF must not ground *revenue* on an EBITDA-scoped entity literal alone.

        The entity name carries an EBITDA metric token so
        ``_entity_encodes_different_financial_metric`` excludes it from broad deferral
        for a *revenue* sentence; ``UserGroundingContext`` still grounds the restatement
        via same-turn committed literals without numerics in ``effective_user_text``.
        """
        checker = _spacy_checker()
        pef = PEFState()
        ent, _ = pef.get_or_create_entity("HoldCo EBITDA Segment Notes")
        rel = Relationship(
            subject_id=ent.id,
            relation="IS",
            object_entity_id=None,
            object_literal="$4.2M consolidated revenue (Q1)",
            span=Span.PRESENT,
            source_turn=1,
            evidence="user admitted metric",
            provenance="user_input",
            extractor_backend="manual",
        )
        pef.add_relationship(rel)
        ug = build_user_grounding_context(pef, 1, effective_user_text="See attached.")

        response = (
            "Quarterly revenue was $4.2 million on a consolidated basis "
            "for the period we discussed."
        )
        flags_pass = asyncio.run(
            checker.check(response, pef, user_input=None, user_grounding=ug)
        )
        assert FlagType.UNVERIFIED_FACT_ASSERTION not in {
            f.flag_type for f in flags_pass
        }

        flags_none = asyncio.run(checker.check(response, pef, user_input=None))
        assert FlagType.UNVERIFIED_FACT_ASSERTION in {f.flag_type for f in flags_none}

    def test_layer1_bridge_does_not_excuse_novel_financial_numeric(self):
        checker = _spacy_checker()
        pef = PEFState()
        ent, _ = pef.get_or_create_entity("HoldCo EBITDA Segment Notes")
        rel = Relationship(
            subject_id=ent.id,
            relation="IS",
            object_entity_id=None,
            object_literal="$4.2M consolidated revenue (Q1)",
            span=Span.PRESENT,
            source_turn=1,
            evidence="user admitted metric",
            provenance="user_input",
            extractor_backend="manual",
        )
        pef.add_relationship(rel)
        ug = build_user_grounding_context(pef, 1, effective_user_text="See attached.")

        response = (
            "Quarterly revenue was $4.2 million on a consolidated basis. "
            "Additionally one-off EBITDA included an extra $9.9 million gain."
        )
        flags = asyncio.run(
            checker.check(response, pef, user_input=None, user_grounding=ug)
        )
        assert FlagType.UNVERIFIED_FACT_ASSERTION in {f.flag_type for f in flags}


class TestUnverifiedFactAssertionDedupe:
    """At most one UNVERIFIED_FACT_ASSERTION per check() return."""

    def test_dedupe_keeps_first(self):
        f1 = Flag(
            flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
            entity_name="financial",
            claim="a",
            evidence="e1",
        )
        f2 = Flag(
            flag_type=FlagType.UNVERIFIED_FACT_ASSERTION,
            entity_name="causal_enumeration",
            claim="b",
            evidence="e2",
        )
        f3 = Flag(
            flag_type=FlagType.UNBOUND_ENTITY,
            entity_name="x",
            claim="c",
            evidence="e3",
        )
        out = Checker._dedupe_unverified_fact_assertion_flags([f1, f2, f3])
        types = [f.flag_type for f in out]
        assert types.count(FlagType.UNVERIFIED_FACT_ASSERTION) == 1
        assert FlagType.UNBOUND_ENTITY in types
        assert out[0] is f1
