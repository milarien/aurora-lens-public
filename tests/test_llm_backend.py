"""Tests for the LLM extraction backend.

All tests use a mocked adapter — no live LLM calls.
Tests cover JSON parsing, code-fence stripping, and the
full extract() flow with canned responses.
"""

import asyncio
import json

import pytest

from aurora_lens.adapters.base import LLMAdapter, AdapterResponse
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState
from aurora_lens.pef.entity import Entity
from aurora_lens.interpret.llm_backend import (
    LLMExtractionBackend,
    _parse_extraction_response,
    _build_user_message,
)


# ── Helpers ──────────────────────────────────────────────────────────

class MockExtractionAdapter(LLMAdapter):
    """Mock adapter that returns canned JSON for extraction."""

    def __init__(self, response_text: str):
        self._response_text = response_text
        self.last_user_message: str = ""
        self.last_pef_context: str = ""

    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AdapterResponse:
        for m in messages:
            if m.get("role") == "system":
                self.last_pef_context = m.get("content", "")
            elif m.get("role") == "user":
                self.last_user_message = m.get("content", "")
        return AdapterResponse(text=self._response_text, model="mock-extraction")


def _make_json_response(**overrides) -> str:
    """Build a valid extraction JSON response with defaults."""
    data = {
        "claims": [],
        "entity_mentions": [],
        "span": "PRESENT",
        "pronoun_candidates": {},
    }
    data.update(overrides)
    return json.dumps(data)


# ── Parse tests ──────────────────────────────────────────────────────

class TestParseExtractionResponse:
    def test_empty_result(self):
        raw = _make_json_response()
        result = _parse_extraction_response(raw)
        assert result.claims == []
        assert result.entity_mentions == []
        assert result.span == Span.PRESENT

    def test_single_claim(self):
        raw = _make_json_response(
            claims=[{
                "subject": "Emma",
                "relation": "HAS",
                "obj": "red book",
                "span": "PRESENT",
                "negated": False,
                "evidence": "Emma has a red book.",
            }],
            entity_mentions=["Emma"],
        )
        result = _parse_extraction_response(raw)
        assert len(result.claims) == 1
        assert result.claims[0].subject == "Emma"
        assert result.claims[0].relation == "HAS"
        assert result.claims[0].obj == "red book"
        assert result.claims[0].span == Span.PRESENT
        assert result.claims[0].negated is False
        assert result.entity_mentions == ["Emma"]

    def test_past_span(self):
        raw = _make_json_response(
            span="PAST",
            claims=[{
                "subject": "Emma",
                "relation": "AT",
                "obj": "London",
                "span": "PAST",
                "negated": False,
                "evidence": "Emma was in London.",
            }],
        )
        result = _parse_extraction_response(raw)
        assert result.span == Span.PAST
        assert result.claims[0].span == Span.PAST

    def test_negated_claim(self):
        raw = _make_json_response(
            claims=[{
                "subject": "Emma",
                "relation": "HAS",
                "obj": "key",
                "span": "PRESENT",
                "negated": True,
                "evidence": "Emma does not have a key.",
            }],
        )
        result = _parse_extraction_response(raw)
        assert result.claims[0].negated is True

    def test_pronoun_candidates(self):
        raw = _make_json_response(
            pronoun_candidates={"she@token_3": "Emma"},
        )
        result = _parse_extraction_response(raw)
        assert result.pronoun_candidates == {"she@token_3": "Emma"}

    def test_multiple_claims(self):
        raw = _make_json_response(
            claims=[
                {
                    "subject": "Emma",
                    "relation": "HAS",
                    "obj": "red book",
                    "span": "PRESENT",
                    "negated": False,
                    "evidence": "Emma has a red book.",
                },
                {
                    "subject": "Emma",
                    "relation": "IS",
                    "obj": "teacher",
                    "span": "PRESENT",
                    "negated": False,
                    "evidence": "Emma is a teacher.",
                },
            ],
            entity_mentions=["Emma"],
        )
        result = _parse_extraction_response(raw)
        assert len(result.claims) == 2

    def test_code_fence_stripping(self):
        """LLMs often wrap JSON in markdown code fences."""
        inner = _make_json_response(entity_mentions=["Emma"])
        raw = f"```json\n{inner}\n```"
        result = _parse_extraction_response(raw)
        assert result.entity_mentions == ["Emma"]

    def test_missing_optional_fields(self):
        """Tolerate missing negated, evidence, pronoun_candidates."""
        raw = json.dumps({
            "claims": [{
                "subject": "Emma",
                "relation": "HAS",
                "obj": "book",
            }],
        })
        result = _parse_extraction_response(raw)
        assert len(result.claims) == 1
        assert result.claims[0].negated is False
        assert result.claims[0].evidence == ""
        assert result.claims[0].span == Span.PRESENT

    def test_invalid_json_returns_extraction_error(self):
        """Bad JSON returns ExtractionResult with extraction_error, never raises."""
        result = _parse_extraction_response("not json at all")
        assert result.extraction_error is not None
        assert result.extraction_error["reason"] == "JSON_DECODE_ERROR"
        assert "snippet" in result.extraction_error
        assert "raw_preview" in result.extraction_error
        assert result.claims == []
        assert result.entity_mentions == []

    def test_salvage_extracts_json_from_prose(self):
        """Salvage parser extracts first complete JSON object from wrapped text."""
        inner = _make_json_response(
            claims=[{"subject": "Emma", "relation": "HAS", "obj": "book", "span": "PRESENT", "negated": False, "evidence": "x"}],
            entity_mentions=["Emma"],
        )
        raw = f"Here is the extraction:\n\n{inner}\n\nHope that helps!"
        result = _parse_extraction_response(raw)
        assert result.extraction_error is None
        assert len(result.claims) == 1
        assert result.claims[0].subject == "Emma"
        assert result.entity_mentions == ["Emma"]


# ── Build user message tests ─────────────────────────────────────────

class TestBuildUserMessage:
    def test_includes_text(self):
        pef = PEFState()
        msg = _build_user_message("Emma has a red book.", pef)
        assert "Emma has a red book." in msg

    def test_includes_pef_context(self):
        pef = PEFState()
        emma = Entity.create("Emma", turn=0)
        pef.add_entity(emma)
        msg = _build_user_message("Hello.", pef)
        assert "Emma" in msg

    def test_empty_pef(self):
        pef = PEFState()
        msg = _build_user_message("Hello.", pef)
        assert "No established facts" in msg


# ── Full backend integration tests ───────────────────────────────────

class TestLLMExtractionBackend:
    def test_extract_basic(self):
        response = _make_json_response(
            claims=[{
                "subject": "Emma",
                "relation": "HAS",
                "obj": "red book",
                "span": "PRESENT",
                "negated": False,
                "evidence": "Emma has a red book.",
            }],
            entity_mentions=["Emma"],
        )
        adapter = MockExtractionAdapter(response)
        backend = LLMExtractionBackend(adapter)
        pef = PEFState()

        result = asyncio.run(backend.extract("Emma has a red book.", pef))
        assert len(result.claims) == 1
        assert result.claims[0].subject == "Emma"
        assert result.entity_mentions == ["Emma"]

    def test_extract_passes_pef_context(self):
        """The adapter should receive the existing world state."""
        response = _make_json_response()
        adapter = MockExtractionAdapter(response)
        backend = LLMExtractionBackend(adapter)

        pef = PEFState()
        emma = Entity.create("Emma", turn=0)
        pef.add_entity(emma)

        asyncio.run(backend.extract("She has a book.", pef))
        # The user message should contain Emma from PEF context
        assert "Emma" in adapter.last_user_message

    def test_extract_with_pronouns(self):
        response = _make_json_response(
            claims=[{
                "subject": "Emma",
                "relation": "HAS",
                "obj": "book",
                "span": "PRESENT",
                "negated": False,
                "evidence": "She has a book.",
            }],
            pronoun_candidates={"she@token_0": "Emma"},
        )
        adapter = MockExtractionAdapter(response)
        backend = LLMExtractionBackend(adapter)
        pef = PEFState()

        result = asyncio.run(backend.extract("She has a book.", pef))
        assert result.pronoun_candidates == {"she@token_0": "Emma"}
