"""Unit tests for ``aurora_lens.state_native_engine.lexical``."""

from __future__ import annotations

from aurora_lens.state_native_engine.lexical import (
    ItemLexeme,
    analyze_item_phrase,
    display_quantity_item,
    item_key,
    normalize_possession_item_surface,
)


def test_item_key_apples_and_determiners() -> None:
    assert item_key("apples") == "apple"
    assert item_key("the apples") == item_key("apples")
    assert item_key("these apples") == item_key("apples")


def test_item_key_glass_ss_guard() -> None:
    assert item_key("glass") == "glass"
    lx = analyze_item_phrase("glass")
    assert lx.number == "unknown"


def test_item_key_berries_boxes_movies() -> None:
    assert item_key("berries") == "berry"
    assert item_key("boxes") == "box"
    # Vowel-before-ies (movies) avoids faulty -> movy pluralisation.
    assert item_key("movies") == "movie"


def test_item_key_irregulars() -> None:
    assert item_key("children") == "child"
    assert item_key("mice") == "mouse"
    assert analyze_item_phrase("geese").lemma == "goose"


def test_multi_word_last_token() -> None:
    assert item_key("red berries") == "red berry"
    assert item_key("red berry") == "red berry"
    lx_rb = analyze_item_phrase("Red Berries")
    assert lx_rb.lemma == "red berry"


def test_display_quantity_item_qty_one_and_plural() -> None:
    assert display_quantity_item(1, "lollipops") == "1 lollipop"
    assert display_quantity_item(4, "red lollipop") == "4 red lollipops"
    assert display_quantity_item(1, "those apples.") == "1 apple"


def test_normalize_possession_item_surface_strip() -> None:
    assert normalize_possession_item_surface("apples...") == "apples"


def test_analyze_returns_item_lexeme() -> None:
    lx = analyze_item_phrase("coins")
    assert isinstance(lx, ItemLexeme)
    assert lx.number == "plural"
