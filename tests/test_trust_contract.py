"""Unit tests for Track C trust contract parsing."""

from __future__ import annotations

from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import Relationship
from aurora_lens.trust.trust_contract import parse_trust_contract, trust_evaluation_required


def _rel(metadata: dict) -> Relationship:
    e = Entity.create("claim", turn=1)
    return Relationship(
        subject_id=e.id,
        relation="IS",
        object_entity_id=None,
        object_literal="value",
        span=Span.PRESENT,
        source_turn=1,
        evidence="note",
        relation_metadata={"trust": metadata},
    )


class TestTrustEvaluationRequired:
    def test_false_when_flag_missing(self):
        assert trust_evaluation_required({}) is False

    def test_true_when_bool_true(self):
        assert trust_evaluation_required({"trust_evaluation_required": True}) is True

    def test_false_for_informal_notes_only(self):
        block = {"notes": "sounds untrusted", "source_class": "unknown"}
        assert trust_evaluation_required(block) is False


class TestParseTrustContract:
    def test_complete_contract_parses(self):
        contract = parse_trust_contract(
            _rel(
                {
                    "trust_evaluation_required": True,
                    "source_id": "source:vendor-17",
                    "trust_registry_ref": "registry:ops:v1",
                    "task_domain": "medical",
                    "consequence_grade": "high",
                    "policy_ref": "ops.source.v1",
                }
            )
        )
        assert contract is not None
        assert contract.source_id == "source:vendor-17"
        assert contract.trust_registry_ref == "registry:ops:v1"
        assert contract.task_domain == "medical"
        assert contract.consequence_grade == "high"

    def test_incomplete_contract_returns_none(self):
        assert parse_trust_contract(_rel({"trust_evaluation_required": True})) is None

    def test_without_pathway_flag_returns_none(self):
        assert (
            parse_trust_contract(
                _rel(
                    {
                        "source_id": "source:vendor-17",
                        "trust_registry_ref": "registry:ops:v1",
                        "task_domain": "medical",
                        "consequence_grade": "high",
                    }
                )
            )
            is None
        )
