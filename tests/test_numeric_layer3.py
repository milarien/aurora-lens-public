"""Layer 3 numeric contradiction tests.

Tests are split into two sections:
  1. NumericValue unit tests — parse_numeric and values_contradict in isolation
  2. Checker integration tests — _check_numeric_contradiction via direct method call
     (no spaCy required — ExtractedClaim and Relationship are constructed directly)

These tests do NOT use the full checker.check() pipeline.  They target the
Layer 3 path only and are deterministic and fast.
"""
from __future__ import annotations

import re

import pytest

from aurora_lens.pef.span import Span
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.state import PEFState, Relationship, canonicalize_relation
from aurora_lens.interpret.schema import ExtractedClaim
from aurora_lens.verify.numeric import (
    NumericUnit,
    NumericValue,
    numeric_for_metric_span,
    parse_numeric,
    values_contradict,
)
from aurora_lens.verify.checker import Checker
from aurora_lens.verify.flags import FlagType


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_claim(subject: str, relation: str, obj: str, negated: bool = False) -> ExtractedClaim:
    return ExtractedClaim(
        subject=subject,
        relation=canonicalize_relation(relation),
        obj=obj,
        span=Span.PRESENT,
        negated=negated,
        evidence=f"{subject} {relation} {obj}",
    )


def _make_pef_with_literal(
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
    rels = pef.get_relationships_for_subject(entity.id)
    return pef, rels


def _numeric_checker() -> Checker:
    """Checker with a dummy backend — only _check_numeric_contradiction is called."""
    class _DummyBackend:
        pass
    return Checker(_DummyBackend())  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — parse_numeric
# ─────────────────────────────────────────────────────────────────────────────

class TestParseNumeric:

    def test_percent_integer(self):
        v = parse_numeric("23%")
        assert v is not None
        assert v.unit == NumericUnit.PERCENT
        assert v.magnitude == pytest.approx(23.0)

    def test_percent_decimal(self):
        v = parse_numeric("23.4%")
        assert v is not None
        assert v.magnitude == pytest.approx(23.4)

    def test_bps_to_percent(self):
        v = parse_numeric("120 bps")
        assert v is not None
        assert v.unit == NumericUnit.PERCENT
        assert v.magnitude == pytest.approx(1.2)

    def test_bps_basis_points_word(self):
        v = parse_numeric("150 basis points")
        assert v is not None
        assert v.unit == NumericUnit.PERCENT
        assert v.magnitude == pytest.approx(1.5)

    def test_currency_million_suffix(self):
        v = parse_numeric("$3.2M")
        assert v is not None
        assert v.unit == NumericUnit.CURRENCY
        assert v.magnitude == pytest.approx(3_200_000.0)

    def test_currency_million_word(self):
        v = parse_numeric("$3.2 million")
        assert v is not None
        assert v.magnitude == pytest.approx(3_200_000.0)

    def test_currency_billion(self):
        v = parse_numeric("$1.5b")
        assert v is not None
        assert v.magnitude == pytest.approx(1_500_000_000.0)

    def test_currency_thousand(self):
        v = parse_numeric("$500k")
        assert v is not None
        assert v.magnitude == pytest.approx(500_000.0)

    def test_currency_commas(self):
        v = parse_numeric("$3,200,000")
        assert v is not None
        assert v.magnitude == pytest.approx(3_200_000.0)

    def test_bare_decimal_not_parsed(self):
        """Bare decimal without unit marker is ambiguous — must not parse."""
        assert parse_numeric("0.23") is None

    def test_numeric_for_metric_span_pairs_nearest_to_metric_not_first_global(self):
        """Multi-number sentences: metric-adjacent % must not lose to earlier bps in string order."""
        s = (
            "returns were 6.2% in q4, underperforming the benchmark by 180 basis points."
        )
        metric_m = re.search(r"\breturns?\b", s)
        assert metric_m is not None
        v = numeric_for_metric_span(s, metric_m.start(), metric_m.end())
        assert v is not None
        assert v.unit == NumericUnit.PERCENT
        assert v.magnitude == pytest.approx(6.2)

    def test_empty_string(self):
        assert parse_numeric("") is None

    def test_no_numeric(self):
        assert parse_numeric("high") is None

    def test_non_financial_text(self):
        assert parse_numeric("3 tablets") is None


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — values_contradict
# ─────────────────────────────────────────────────────────────────────────────

class TestValuesContradict:

    def test_same_percent_no_contradiction(self):
        a = NumericValue(23.0, NumericUnit.PERCENT)
        b = NumericValue(23.0, NumericUnit.PERCENT)
        assert not values_contradict(a, b)

    def test_different_percent_contradiction(self):
        a = NumericValue(23.0, NumericUnit.PERCENT)
        b = NumericValue(25.0, NumericUnit.PERCENT)
        assert values_contradict(a, b)

    def test_within_rounding_tolerance(self):
        """23% vs 23.01% — within 0.1% relative tolerance, not a contradiction."""
        a = NumericValue(23.0, NumericUnit.PERCENT)
        b = NumericValue(23.01, NumericUnit.PERCENT)
        assert not values_contradict(a, b)

    def test_bps_vs_percent_compatible(self):
        """120 bps = 1.2% — same unit after normalisation."""
        a = parse_numeric("120 bps")
        b = parse_numeric("1.5%")
        assert a is not None and b is not None
        assert values_contradict(a, b)

    def test_bps_vs_percent_matching(self):
        """120 bps = 1.2% — no contradiction when values agree."""
        a = parse_numeric("120 bps")
        b = parse_numeric("1.2%")
        assert a is not None and b is not None
        assert not values_contradict(a, b)

    def test_incompatible_units_no_contradiction(self):
        """Percent vs currency — cannot compare, must not flag."""
        a = NumericValue(23.0, NumericUnit.PERCENT)
        b = NumericValue(3_200_000.0, NumericUnit.CURRENCY)
        assert not values_contradict(a, b)

    def test_currency_contradiction(self):
        a = NumericValue(3_200_000.0, NumericUnit.CURRENCY)
        b = NumericValue(3_500_000.0, NumericUnit.CURRENCY)
        assert values_contradict(a, b)

    def test_currency_no_contradiction(self):
        a = NumericValue(3_200_000.0, NumericUnit.CURRENCY)
        b = NumericValue(3_200_000.0, NumericUnit.CURRENCY)
        assert not values_contradict(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — _check_numeric_contradiction (Checker method)
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericContradiction:
    """Tests for Checker._check_numeric_contradiction directly."""

    @pytest.fixture
    def checker(self):
        return _numeric_checker()

    # ── Should fire ──────────────────────────────────────────────────────────

    def test_exact_percent_contradiction(self, checker):
        """PEF: ROI IS 23% — LLM: ROI is 25% — should flag CONTRADICTED_FACT."""
        _, rels = _make_pef_with_literal("roi", "IS", "23%")
        claim = _make_claim("roi", "IS", "25%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None
        assert flag.flag_type == FlagType.CONTRADICTED_FACT
        assert "23%" in flag.evidence
        assert "25%" in flag.evidence

    def test_different_surface_verbs_still_contradicts(self, checker):
        """PEF: revenue IS $3.2M — LLM: revenue reached $3.5M.
        Relation equality NOT required — numeric contradiction must still fire.
        """
        _, rels = _make_pef_with_literal("revenue", "IS", "$3.2M")
        claim = _make_claim("revenue", "reached", "$3.5M")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None
        assert flag.flag_type == FlagType.CONTRADICTED_FACT

    def test_bps_vs_percent_contradiction(self, checker):
        """PEF: margin IS 120 bps (=1.2%) — LLM: margin is 1.5% — should flag."""
        _, rels = _make_pef_with_literal("margin", "IS", "120 bps")
        claim = _make_claim("margin", "IS", "1.5%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None
        assert flag.flag_type == FlagType.CONTRADICTED_FACT

    def test_bps_vs_percent_agreement(self, checker):
        """PEF: margin IS 120 bps (=1.2%) — LLM: margin is 1.2% — no contradiction."""
        _, rels = _make_pef_with_literal("margin", "IS", "120 bps")
        claim = _make_claim("margin", "IS", "1.2%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    def test_growth_percent_contradiction(self, checker):
        """PEF: growth IS 12% — LLM: growth was 10% — should flag."""
        _, rels = _make_pef_with_literal("growth", "IS", "12%")
        claim = _make_claim("growth", "was", "10%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None

    def test_currency_contradiction(self, checker):
        """PEF: revenue IS $3.2M — LLM: revenue is $3.5M — should flag."""
        _, rels = _make_pef_with_literal("revenue", "IS", "$3.2M")
        claim = _make_claim("revenue", "IS", "$3.5M")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is not None

    # ── Should NOT fire ──────────────────────────────────────────────────────

    def test_ambiguous_metric_skipped(self, checker):
        """Claim subject 'rate' is not in _FIN_METRIC_TOKEN_RE — must skip."""
        _, rels = _make_pef_with_literal("rate", "IS", "23%")
        claim = _make_claim("rate", "IS", "25%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    def test_metric_mismatch_skipped(self, checker):
        """PEF: revenue IS $3M — LLM: profit is $3.5M — different entities.

        In the real call flow, existing_rels contains only relationships for
        the claim's entity.  If 'profit' has no PEF relationships, the list
        is empty and no contradiction can fire.
        """
        claim = _make_claim("profit", "IS", "$3.5M")
        flag = checker._check_numeric_contradiction(claim, [])
        assert flag is None

    def test_pronoun_subject_skipped(self, checker):
        """Claim subject 'it' is not a financial metric token — must skip."""
        _, rels = _make_pef_with_literal("roi", "IS", "23%")
        claim = _make_claim("it", "IS", "25%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    def test_predictive_pef_relation_skipped(self, checker):
        """PEF stores a projection (relation contains 'projected') — must not contradict."""
        _, rels = _make_pef_with_literal("roi", "projected", "23%")
        claim = _make_claim("roi", "IS", "25%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    def test_predictive_pef_will_skipped(self, checker):
        """PEF relation contains 'will' — predictive, must skip."""
        _, rels = _make_pef_with_literal("earnings", "will", "15%")
        claim = _make_claim("earnings", "IS", "12%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    def test_negated_pef_fact_skipped(self, checker):
        """PEF has negated=True on the relationship — must skip in v1."""
        _, rels = _make_pef_with_literal("roi", "IS", "23%", negated=True)
        claim = _make_claim("roi", "IS", "25%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    def test_negated_claim_skipped(self, checker):
        """Claim is negated — must skip in v1."""
        _, rels = _make_pef_with_literal("roi", "IS", "23%")
        claim = _make_claim("roi", "IS", "25%", negated=True)
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    def test_non_numeric_pef_literal_skipped(self, checker):
        """PEF literal is text, not numeric — no parse, no compare."""
        _, rels = _make_pef_with_literal("roi", "IS", "strong")
        claim = _make_claim("roi", "IS", "25%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    def test_non_numeric_claim_obj_skipped(self, checker):
        """Claim object is text — no parse, no compare."""
        _, rels = _make_pef_with_literal("roi", "IS", "23%")
        claim = _make_claim("roi", "IS", "improving")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    def test_within_rounding_tolerance_no_flag(self, checker):
        """23% vs 23.01% — rounding noise, must not flag."""
        _, rels = _make_pef_with_literal("roi", "IS", "23%")
        claim = _make_claim("roi", "IS", "23.01%")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None

    def test_incompatible_units_no_flag(self, checker):
        """Percent vs currency — incompatible units, must not flag."""
        _, rels = _make_pef_with_literal("revenue", "IS", "23%")
        claim = _make_claim("revenue", "IS", "$3.5M")
        flag = checker._check_numeric_contradiction(claim, rels)
        assert flag is None
