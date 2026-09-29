"""Tests for reason-specific governed user copy."""

import pytest

from aurora_lens.govern.governed_copy import (
    compose_unresolved_possessive_attribution_block,
    compose_unresolved_referent_governed_message,
    compose_insufficient_structure_message,
)
from aurora_lens.verify.flags import FlagType

_OPERATOR_CASE = (
    "The operator informed the contractor that their certification had expired "
    "before the work commenced. Both parties hold certifications. "
    "No further evidence is available. Who does 'their' refer to?"
)


class TestGovernedCopyAttributionBlock:
    def test_operator_contractor_possessive_attribution_block(self):
        msg = compose_unresolved_possessive_attribution_block(
            ambiguous_token="their",
            candidate_entities=["Contractor", "Operator"],
            original_question=_OPERATOR_CASE,
        )
        assert msg is not None
        assert msg.startswith("Decision blocked.")
        assert "does not establish" in msg.lower()
        assert "expired certification" in msg.lower()
        assert "belongs to the" in msg.lower()
        assert "operator" in msg.lower()
        assert "contractor" in msg.lower()
        assert "Both parties hold certifications" in msg
        assert "No further evidence is available" in msg
        assert "cannot be released" in msg.lower()
        assert "Allowed continuation:" in msg
        assert "refers to the contractor." not in msg.lower()

    def test_compose_unresolved_referent_uses_attribution_block(self):
        msg = compose_unresolved_referent_governed_message(
            ambiguous_tokens=["their"],
            candidate_entities=["Contractor", "Operator"],
            original_question=_OPERATOR_CASE,
        )
        assert msg.startswith("Decision blocked.")
        assert "does not establish" in msg.lower()
        assert "operator" in msg.lower()
        assert "contractor" in msg.lower()


class TestGovernedCopySessionDependentBlock:
    def test_suspension_admissibility_uses_regulator_framing(self):
        from aurora_lens.govern.governed_copy import compose_session_dependent_consequence_block

        msg = compose_session_dependent_consequence_block(
            ambiguous_tokens=["their"],
            candidate_entities=["Contractor", "Operator"],
            blocked_proposition="their certification had expired before the work commenced",
            act_kind="decision_seeking",
            user_turn="Is suspension admissible?",
        )
        assert msg.startswith("Decision blocked.")
        assert "does not establish" in msg.lower()
        assert "expired certification" in msg.lower()
        assert "belongs to" in msg.lower()
        assert "Suspension cannot be determined" in msg
        assert "unresolved attribution" in msg.lower()
        assert "unresolved referent" not in msg.lower()


class TestGovernedCopyInsufficientStructure:
    def test_extraction_empty_is_specific(self):
        msg = compose_insufficient_structure_message(
            flag_type=FlagType.EXTRACTION_EMPTY,
            flag_evidence="Extraction produced no admissible structure.",
        )
        assert msg.startswith("Decision blocked.")
        assert "missing detail" not in msg.lower()
        assert "Allowed continuation:" in msg
