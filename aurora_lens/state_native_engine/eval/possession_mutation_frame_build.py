"""Arbitrate interpreters → single :class:`PossessionMutationFrame` or abstention."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from aurora_lens.pef.state import PEFState
from aurora_lens.state_native_engine.contracts.possession_mutation_frame import (
    PossessionMutationConfidence,
    PossessionMutationFrame,
)
from aurora_lens.state_native_engine.eval.inventory import normalize_possession_item_surface
from aurora_lens.state_native_engine.eval.possession_regex_frame_builders import (
    regex_give_frame_builder,
    regex_hand_frame_builder,
    regex_put_into_container_frame_builder,
)
from aurora_lens.state_native_engine.eval.possession_spacy_frame_builder import (
    try_spacy_give_transfer_high,
    try_spacy_passive_give_or_hand_transfer_high,
    try_spacy_passive_put_like_frame_high,
    try_spacy_put_like_frame_high,
)

# Abstention-first: multi-sentence user surfaces are not collapsed into a single frame here.
_COMPOUND_SENTENCE_BOUNDARY = re.compile(r"(?<=[!?])\s+|(?<=\.)\s+(?=[A-Z\"'`])")


@dataclass(frozen=True)
class PossessionFrameBuildAbstained:
    """Interpreter abstention is valid — no inferred mutation from this route."""

    kind: Literal["abstained"] = "abstained"


@dataclass(frozen=True)
class PossessionFrameBuildOk:
    """Admissible semantic frame ready for mutation law."""

    frame: PossessionMutationFrame
    kind: Literal["ok"] = "ok"


PossessionFrameBuildResult = PossessionFrameBuildOk | PossessionFrameBuildAbstained


def _compound_possession_surface_abstains_first(text: str) -> bool:
    core = normalize_possession_item_surface((text or "").strip())
    if not core:
        return False
    chunks = [c.strip() for c in _COMPOUND_SENTENCE_BOUNDARY.split(core) if c.strip()]
    return len(chunks) > 1


def build_possession_mutation_frame(
    *,
    text: str,
    pef: PEFState,
    possession_nlp: object | None,
    binding_resumed: bool,
) -> PossessionFrameBuildResult:
    """Prefer spaCy **HIGH** (active/passive PUT then active/passive GIVE); fall back to regex.

    ``pef`` is reserved for forthcoming discourse-aware ambiguity / abstention hooks.
    """
    _ = pef

    if binding_resumed:
        return PossessionFrameBuildAbstained()

    if _compound_possession_surface_abstains_first(text):
        return PossessionFrameBuildAbstained()

    if possession_nlp is not None:
        sp_put = try_spacy_put_like_frame_high(possession_nlp, text)
        if sp_put is not None and sp_put.confidence == PossessionMutationConfidence.HIGH:
            return PossessionFrameBuildOk(frame=sp_put)

        sp_put_passive = try_spacy_passive_put_like_frame_high(possession_nlp, text)
        if sp_put_passive is not None and sp_put_passive.confidence == PossessionMutationConfidence.HIGH:
            return PossessionFrameBuildOk(frame=sp_put_passive)

        sp_give = try_spacy_give_transfer_high(possession_nlp, text)
        if sp_give is not None and sp_give.confidence == PossessionMutationConfidence.HIGH:
            return PossessionFrameBuildOk(frame=sp_give)

        sp_give_passive = try_spacy_passive_give_or_hand_transfer_high(possession_nlp, text)
        if sp_give_passive is not None and sp_give_passive.confidence == PossessionMutationConfidence.HIGH:
            return PossessionFrameBuildOk(frame=sp_give_passive)

    rf_put = regex_put_into_container_frame_builder(text)
    if rf_put is not None:
        return PossessionFrameBuildOk(frame=rf_put)

    rf_hand = regex_hand_frame_builder(text)
    if rf_hand is not None:
        return PossessionFrameBuildOk(frame=rf_hand)

    rf_give = regex_give_frame_builder(text)
    if rf_give is not None:
        return PossessionFrameBuildOk(frame=rf_give)

    return PossessionFrameBuildAbstained()
