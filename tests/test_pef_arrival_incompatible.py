"""Narrow C1 arrival seam (calendar month+day vs last week in one blob)."""

from aurora_lens.pef.arrival_incompatible import (
    entity_name_evident_in_text,
    incompatible_arrival_time_joined_text,
    incompatible_arrival_time_literal_strings,
    rag_context_body_from_user_input,
    text_contains_calendar_month_day,
)


def test_joined_text_true_when_calendar_and_last_week():
    s = "landed march 8, finally home arrived last week"
    assert incompatible_arrival_time_joined_text(s.lower()) is True


def test_joined_text_false_without_last_week():
    s = "landed march 8, finally home"
    assert incompatible_arrival_time_joined_text(s.lower()) is False


def test_joined_text_false_without_calendar():
    s = "arrived last week from the coast"
    assert incompatible_arrival_time_joined_text(s.lower()) is False


def test_literal_strings_matches_manifest_shape():
    literals = ["Landed March 8, finally home", "arrived last week"]
    assert incompatible_arrival_time_literal_strings(literals) is True


def test_text_contains_calendar_month_day():
    assert text_contains_calendar_month_day("She arrived March 8.") is True
    assert text_contains_calendar_month_day("She arrived last week.") is False


def test_rag_context_body_from_user_input():
    u = "Context:\n### Section 6\nNora landed March 8.\n\n### Section 7\nNora arrived last week.\n\nQuestion: When?"
    body = rag_context_body_from_user_input(u)
    assert body is not None
    assert "March 8" in body
    assert "last week" in body


def test_entity_name_evident_in_text():
    assert entity_name_evident_in_text("Nora Park texted Emma.", "Nora Park") is True
    assert entity_name_evident_in_text("Someone else.", "Nora Park") is False
