"""Uncommitted hiring decision context, separate from world-state assertions."""
from dataclasses import asdict, dataclass, field
from typing import Literal

Basis = Literal["unresolved", "age_exclusion", "skills_requirement"]


@dataclass
class HiringCriterion:
    kind: str
    expression: str
    comparator: str | None = None
    year: str | None = None
    basis: Basis = "unresolved"
    evidence: list[dict] = field(default_factory=list)


@dataclass
class HiringCriterionContext:
    original_question: str
    action: str
    target: str
    criteria: list[HiringCriterion]
    status: str = "held"
    screening_authorized: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None):
        if not isinstance(data, dict):
            return None
        return cls(
            original_question=data["original_question"],
            action=data["action"], target=data["target"],
            criteria=[HiringCriterion(**item) for item in data["criteria"]],
            status=data.get("status", "held"),
            # A saved clarification never supplies authorization.
            screening_authorized=False,
        )
