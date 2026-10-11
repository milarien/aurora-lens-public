"""Quarantined transformation evidence, never world state or an admissible binding."""
from dataclasses import asdict, dataclass, field


@dataclass
class TranslationCandidate:
    turn: int
    source_text: str
    trigger: str
    source_sha256: str
    candidate_text: str | None = None
    model: str | None = None
    candidate_sha256: str | None = None
    unresolved_items: list[str] = field(default_factory=list)
    source_provenance: str = "user_input"
    candidate_provenance: str = "model_translation_untrusted"
    call_purpose: str = "translation_only"
    calls: int = 0
    status: str = "held"
    fidelity: str = "unestablished"
    failure: str | None = None
    candidate_policy_flags: list[str] = field(default_factory=list)
    candidate_interpretation_limit: dict | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        item = cls(**data)
        # Deserialization cannot grant fidelity or admission.
        item.fidelity = "unestablished"
        item.call_purpose = "translation_only"
        item.source_provenance = "user_input"
        item.candidate_provenance = "model_translation_untrusted"
        return item
