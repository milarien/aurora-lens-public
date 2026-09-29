"""Negative tests for deferred multi-word comparative parsing."""

from aurora_lens.state_native_engine.parse.query_surface import parse_comparative_question


def test_multi_word_adjective_in_comparative_query_returns_none():
    """Multi-word adjective slot is deferred; returns None without error."""
    assert parse_comparative_question("Whose dog was more aggressive than yours") is None


def test_unrelated_comparative_phrasing_returns_none():
    assert parse_comparative_question("Alice has more tokens than Bob") is None


def test_single_word_adjective_still_parses():
    assert parse_comparative_question("Whose dog was bigger?") == ("dog", "bigger")
