"""Regression for deterministic action clause realization (display only)."""

from __future__ import annotations

from aurora_lens.state_native_engine.eval.action_realization import (
    format_action_realization,
    normalize_event_object_literal,
    verb_past_tense_for_display,
)


def test_order_maps_to_ordered() -> None:
    assert verb_past_tense_for_display("ORDER") == "ordered"


def test_ordered_relation_stays_ordered() -> None:
    assert verb_past_tense_for_display("ORDERED") == "ordered"


def test_medication_change_gets_determiner() -> None:
    assert format_action_realization("ORDER", "medication change") == (
        "ordered the medication change"
    )


def test_underscores_normalized_in_object() -> None:
    assert normalize_event_object_literal("medication_change") == "medication change"
    assert format_action_realization("ORDER", "medication_change") == (
        "ordered the medication change"
    )
