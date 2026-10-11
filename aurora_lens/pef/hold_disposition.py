"""Explicit closure of unresolved context; never a truth binding or authorization."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HoldDisposition:
    turn: int
    control: str
    pending: dict | None
    hiring_criterion_context: dict | None
    source: str = "explicit_user_control"
    status: str = "closed_unresolved"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(turn=int(data["turn"]), control=data["control"],
                   pending=data.get("pending"), hiring_criterion_context=data.get("hiring_criterion_context"))
