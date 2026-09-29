"""Regex frame builders ↔ :class:`PossessionMutationFrame` contract (+ law isolation)."""

from __future__ import annotations

import inspect

from aurora_lens.state_native_engine.contracts import (
    PossessionMutationAction,
    PossessionMutationConfidence,
)
from aurora_lens.state_native_engine.eval.possession_mutations import (
    _apply_possession_give_frame,
    _apply_possession_put_frame,
)
from aurora_lens.state_native_engine.eval.possession_regex_frame_builders import (
    regex_give_frame_builder,
    regex_hand_frame_builder,
    regex_put_into_container_frame_builder,
)


def test_regex_give_recipient_first_frame_fields() -> None:
    frame = regex_give_frame_builder("Grace gives Henry 4 coins.")
    assert frame is not None
    assert frame.action == PossessionMutationAction.GIVE
    assert frame.parse_source == "regex_give_recipient_first"
    assert frame.confidence == PossessionMutationConfidence.LOW
    assert frame.actor_surface == "Grace"
    assert frame.recipient_surface == "Henry"
    assert frame.quantity == 4.0
    assert frame.item_surface == "coins"
    assert frame.destination_surface is None
    assert frame.source_container_surface is None
    assert frame.original_surface == "Grace gives Henry 4 coins"


def test_regex_give_quantity_first_to_recipient_frame_fields() -> None:
    frame = regex_give_frame_builder("Ben gives 2 blue marbles to Clara.")
    assert frame is not None
    assert frame.parse_source == "regex_give_quantity_first_to_recipient"
    assert frame.actor_surface == "Ben"
    assert frame.recipient_surface == "Clara"
    assert frame.quantity == 2.0
    assert frame.item_surface == "blue marbles"
    assert frame.original_surface == "Ben gives 2 blue marbles to Clara"


def test_regex_put_into_container_frame_fields() -> None:
    frame = regex_put_into_container_frame_builder(
        "Clara puts 1 blue marble back into the red box.",
    )
    assert frame is not None
    assert frame.action == PossessionMutationAction.PUT
    assert frame.parse_source == "regex_put_into_container"
    assert frame.actor_surface == "Clara"
    assert frame.destination_surface == "red box"
    assert frame.recipient_surface is None
    assert frame.quantity == 1.0
    assert frame.item_surface == "blue marble"
    assert frame.source_container_surface is None


def test_regex_hand_recipient_first_frame_fields() -> None:
    frame = regex_hand_frame_builder("Grace handed Henry 4 coins.")
    assert frame is not None
    assert frame.action == PossessionMutationAction.GIVE
    assert frame.parse_source == "regex_hand_recipient_first"
    assert frame.surface_verb == "hand"
    assert frame.actor_surface == "Grace"
    assert frame.recipient_surface == "Henry"


def test_regex_passive_put_into_container_frame_fields() -> None:
    frame = regex_put_into_container_frame_builder(
        "1 blue marble was put back into the red box by Clara.",
    )
    assert frame is not None
    assert frame.action == PossessionMutationAction.PUT
    assert frame.parse_source == "regex_passive_put_into_container"
    assert frame.voice == "passive"
    assert frame.actor_surface == "Clara"
    assert frame.destination_surface == "red box"


def test_regex_return_into_container_frame_fields() -> None:
    frame = regex_put_into_container_frame_builder(
        "Clara returned 1 blue marble into the red box.",
    )
    assert frame is not None
    assert frame.action == PossessionMutationAction.RETURN
    assert frame.parse_source == "regex_return_into_container"
    assert frame.surface_verb == "return"


def test_mutation_apply_give_has_no_regex_match_groups_in_source() -> None:
    give_src = inspect.getsource(_apply_possession_give_frame)
    assert ".group(" not in give_src
    assert '"giver"' not in give_src


def test_mutation_apply_put_has_no_regex_match_groups_in_source() -> None:
    put_src = inspect.getsource(_apply_possession_put_frame)
    assert ".group(" not in put_src
