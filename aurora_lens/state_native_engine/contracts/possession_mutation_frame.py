"""Contract surface → **PossessionMutationFrame** → PEF transition law."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

PossessionMutationVoice = Literal["active", "passive", "unknown"]

PossessionMutationParseSource = Literal[
    "regex_give_recipient_first",
    "regex_give_quantity_first_to_recipient",
    "regex_hand_recipient_first",
    "regex_hand_quantity_first_to_recipient",
    "regex_passive_put_into_container",
    "regex_put_into_container",
    "regex_return_into_container",
    "spacy_give_transfer",
    "spacy_hand_transfer",
    "spacy_passive_give_transfer",
    "spacy_passive_hand_transfer",
    "spacy_passive_put_into_container",
    "spacy_passive_return_container",
    "spacy_put_into_container",
    "spacy_return_into_container",
]

# Back-compat alias for older module names / prose (same underlying literals).
PossessionMutationDeterministicParse = PossessionMutationParseSource


class PossessionMutationAction(str, Enum):
    GIVE = "GIVE"
    PUT = "PUT"
    RETURN = "RETURN"


class PossessionMutationConfidence(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class PossessionMutationFrame:
    """Stable input to possession/container counted mutation logic (regex, spaCy, etc.).

    Interpreter abstention is valid: absent an admissible frame, mutation law must not run.

    Semantic resolution (nested container stock, binding) stays in mutation law + PEF lookup,
    not in surface tagging fields below.
    """

    action: PossessionMutationAction
    actor_surface: str
    recipient_surface: str | None
    destination_surface: str | None
    quantity: float
    item_surface: str
    source_container_surface: str | None
    original_surface: str
    voice: PossessionMutationVoice
    surface_verb: str
    parse_source: PossessionMutationParseSource
    confidence: PossessionMutationConfidence
    requires_resolution: bool
