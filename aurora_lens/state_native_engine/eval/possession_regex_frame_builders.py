"""Regex-only **adapter**: user text → :class:`PossessionMutationFrame` (LOW confidence).

Mutation law lives in :mod:`aurora_lens.state_native_engine.eval.possession_mutations`;
this module owns no PEF writes and no transition semantics beyond pattern capture.
"""

from __future__ import annotations

import re

from aurora_lens.state_native_engine.contracts.possession_mutation_frame import (
    PossessionMutationAction,
    PossessionMutationConfidence,
    PossessionMutationFrame,
)
from aurora_lens.state_native_engine.eval.inventory import (
    _parse_counted_item_literal,
    normalize_possession_item_surface,
)

# Lazy ``.+?`` on the recipient group so quantity capture stays anchored nearest
# the item tail in the **recipient‑first** form (“Grace gives Henry 4 coins”).
_GIVE_RECIPIENT_FIRST = re.compile(
    r"(?is)^(?P<giver>.+?)\s+(?:give|gives|gave|given)\s+"
    r"(?P<recipient>.+?)\s+(?P<qty>\d+(?:\.\d+)?)\s+(?P<item>.+)$",
)

_GIVE_QUANT_FIRST_TO_RECIPIENT = re.compile(
    r"(?is)^(?P<giver>.+?)\s+(?:give|gives|gave|given)\s+"
    r"(?P<qty>\d+(?:\.\d+)?)\s+(?P<item>.+?)\s+to\s+(?P<recipient>.+?)\s*\.?$",
)

_HAND_RECIPIENT_FIRST = re.compile(
    r"(?is)^(?P<giver>.+?)\s+(?:hand|hands|handed)\s+"
    r"(?P<recipient>.+?)\s+(?P<qty>\d+(?:\.\d+)?)\s+(?P<item>.+)$",
)

_HAND_QUANT_FIRST_TO_RECIPIENT = re.compile(
    r"(?is)^(?P<giver>.+?)\s+(?:hand|hands|handed)\s+"
    r"(?P<qty>\d+(?:\.\d+)?)\s+(?P<item>.+?)\s+to\s+(?P<recipient>.+?)\s*\.?$",
)

_PUT_INTO_CONTAINER = re.compile(
    r"(?is)^(?P<actor>.+?)\s+(?:put|puts|placed?)\s+(?P<payload>.+?)\s+"
    r"(?:back\s+)?(?:into|in|inside)\s+(?:the\s+)?(?P<dest>.+?)\s*\.?$",
)

_PASSIVE_PUT_INTO_CONTAINER = re.compile(
    r"(?is)^(?P<payload>.+?)\s+(?:was|were)\s+(?:put|placed)\s+"
    r"(?:back\s+)?(?:into|in|inside)\s+(?:the\s+)?(?P<dest>.+?)\s+by\s+(?P<actor>.+?)\s*\.?$",
)

_RETURN_TO_CONTAINER = re.compile(
    r"(?is)^(?P<actor>.+?)\s+(?:return|returns|returned)\s+(?P<payload>.+?)\s+"
    r"(?:back\s+)?(?:into|in|inside)\s+(?:the\s+)?(?P<dest>.+?)\s*\.?$",
)

_TRANSFER_PAYLOAD_DEFERRAL_PRONOUN = re.compile(r"(?is)\b(of\s+)?(them|it)\b")


def _payload_requires_extractor_binding(payload_tail: str) -> bool:
    """True when PUT payload span must see NLP/pronoun gates (same rule as possession_mutations)."""
    s = normalize_possession_item_surface((payload_tail or "").strip())
    if not s:
        return False
    return _TRANSFER_PAYLOAD_DEFERRAL_PRONOUN.search(s) is not None


def regex_give_frame_builder(user_text: str) -> PossessionMutationFrame | None:
    """Map supported **GIVE** demo surfaces to a frame; return ``None`` if no match."""
    original_surface = normalize_possession_item_surface((user_text or "").strip())
    if not original_surface:
        return None
    m = _GIVE_RECIPIENT_FIRST.match(original_surface)
    if m is not None:
        label = "regex_give_recipient_first"
        surf_verb = "give"
    else:
        m = _GIVE_QUANT_FIRST_TO_RECIPIENT.match(original_surface)
        if m is None:
            return None
        label = "regex_give_quantity_first_to_recipient"
        surf_verb = "give"
    try:
        qty = float(m.group("qty").strip())
    except ValueError:
        return None
    return PossessionMutationFrame(
        action=PossessionMutationAction.GIVE,
        actor_surface=m.group("giver").strip(),
        recipient_surface=m.group("recipient").strip(),
        destination_surface=None,
        quantity=qty,
        item_surface=m.group("item").strip(),
        source_container_surface=None,
        original_surface=original_surface,
        voice="active",
        surface_verb=surf_verb,
        parse_source=label,
        confidence=PossessionMutationConfidence.LOW,
        requires_resolution=False,
    )


def regex_hand_frame_builder(user_text: str) -> PossessionMutationFrame | None:
    """Map **HAND**/**HANDED** transfer surfaces (parallel to **GIVE** demo shapes)."""
    original_surface = normalize_possession_item_surface((user_text or "").strip())
    if not original_surface:
        return None
    m = _HAND_RECIPIENT_FIRST.match(original_surface)
    if m is not None:
        label = "regex_hand_recipient_first"
        surf_verb = "hand"
    else:
        m = _HAND_QUANT_FIRST_TO_RECIPIENT.match(original_surface)
        if m is None:
            return None
        label = "regex_hand_quantity_first_to_recipient"
        surf_verb = "hand"
    try:
        qty = float(m.group("qty").strip())
    except ValueError:
        return None
    return PossessionMutationFrame(
        action=PossessionMutationAction.GIVE,
        actor_surface=m.group("giver").strip(),
        recipient_surface=m.group("recipient").strip(),
        destination_surface=None,
        quantity=qty,
        item_surface=m.group("item").strip(),
        source_container_surface=None,
        original_surface=original_surface,
        voice="active",
        surface_verb=surf_verb,
        parse_source=label,
        confidence=PossessionMutationConfidence.LOW,
        requires_resolution=False,
    )


def _regex_try_passive_put_into_container(user_text: str) -> PossessionMutationFrame | None:
    original_surface = normalize_possession_item_surface((user_text or "").strip())
    if not original_surface:
        return None
    m = _PASSIVE_PUT_INTO_CONTAINER.match(original_surface)
    if m is None:
        return None
    payload = normalize_possession_item_surface(m.group("payload"))
    if _payload_requires_extractor_binding(payload):
        return None
    parsed = _parse_counted_item_literal(payload)
    if parsed is None:
        return None
    qty_i, item_tail = parsed
    return PossessionMutationFrame(
        action=PossessionMutationAction.PUT,
        actor_surface=m.group("actor").strip(),
        recipient_surface=None,
        destination_surface=m.group("dest").strip(),
        quantity=float(qty_i),
        item_surface=normalize_possession_item_surface(item_tail),
        source_container_surface=None,
        original_surface=original_surface,
        voice="passive",
        surface_verb="put",
        parse_source="regex_passive_put_into_container",
        confidence=PossessionMutationConfidence.LOW,
        requires_resolution=False,
    )


def _regex_try_active_put_into_container(user_text: str) -> PossessionMutationFrame | None:
    original_surface = normalize_possession_item_surface((user_text or "").strip())
    if not original_surface:
        return None
    m = _PUT_INTO_CONTAINER.match(original_surface)
    if m is None:
        return None
    payload = normalize_possession_item_surface(m.group("payload"))
    if _payload_requires_extractor_binding(payload):
        return None
    parsed = _parse_counted_item_literal(payload)
    if parsed is None:
        return None
    qty_i, item_tail = parsed
    return PossessionMutationFrame(
        action=PossessionMutationAction.PUT,
        actor_surface=m.group("actor").strip(),
        recipient_surface=None,
        destination_surface=m.group("dest").strip(),
        quantity=float(qty_i),
        item_surface=normalize_possession_item_surface(item_tail),
        source_container_surface=None,
        original_surface=original_surface,
        voice="active",
        surface_verb="put",
        parse_source="regex_put_into_container",
        confidence=PossessionMutationConfidence.LOW,
        requires_resolution=False,
    )


def regex_return_into_container_frame_builder(user_text: str) -> PossessionMutationFrame | None:
    """Active **RETURN … to/into/in …** surfaces (RETURN action → same law as PUT into container)."""
    original_surface = normalize_possession_item_surface((user_text or "").strip())
    if not original_surface:
        return None
    m = _RETURN_TO_CONTAINER.match(original_surface)
    if m is None:
        return None
    payload = normalize_possession_item_surface(m.group("payload"))
    if _payload_requires_extractor_binding(payload):
        return None
    parsed = _parse_counted_item_literal(payload)
    if parsed is None:
        return None
    qty_i, item_tail = parsed
    return PossessionMutationFrame(
        action=PossessionMutationAction.RETURN,
        actor_surface=m.group("actor").strip(),
        recipient_surface=None,
        destination_surface=m.group("dest").strip(),
        quantity=float(qty_i),
        item_surface=normalize_possession_item_surface(item_tail),
        source_container_surface=None,
        original_surface=original_surface,
        voice="active",
        surface_verb="return",
        parse_source="regex_return_into_container",
        confidence=PossessionMutationConfidence.LOW,
        requires_resolution=False,
    )


def regex_put_into_container_frame_builder(user_text: str) -> PossessionMutationFrame | None:
    """Passive PUT → active PUT → RETURN … container (single regex arbitration tier)."""
    for fn in (
        _regex_try_passive_put_into_container,
        _regex_try_active_put_into_container,
        regex_return_into_container_frame_builder,
    ):
        hit = fn(user_text)
        if hit is not None:
            return hit
    return None
