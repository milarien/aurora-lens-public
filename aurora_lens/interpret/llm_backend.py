"""LLM-based extraction backend.

Uses a separate LLM call (cheap/fast model) to extract entities,
relationships, and span signals from text. This is NOT the fluency
LLM — the fluency LLM never grades its own homework.

The extraction prompt requests structured JSON matching the
ExtractionResult schema. The backend parses the response and
produces the same contract as any other backend.

When the model returns non-JSON or malformed JSON, we return a
controlled failure (extraction_error) instead of raising. Bad JSON
becomes inadmissible extraction → governance intervenes, not 500.

NOTE: ``ambiguous_referents`` is not populated on this path; ambiguity-hold parity
claims are excluded when ``backend=llm``. spaCy is the deterministic parity baseline.
See ``docs/closed-decisions.md``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from aurora_lens.adapters.base import LLMAdapter
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, canonicalize_relation
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult

logger = logging.getLogger(__name__)


_EXTRACTION_PROMPT = """\
You are a structured extraction system. Extract entities, relationships, \
and temporal signals from the given text.

The user will provide:
1. The text to analyze.
2. A summary of entities and facts that already exist in the world.

Your job is to extract what the NEW text proposes as changes to the existing world.

Respond with ONLY valid JSON matching this schema:

{
  "claims": [
    {
      "subject": "entity name",
      "relation": "CANONICAL_RELATION",
      "obj": "object text",
      "span": "PRESENT" or "PAST",
      "negated": false,
      "evidence": "source text from input"
    }
  ],
  "entity_mentions": ["Name1", "Name2"],
  "span": "PRESENT" or "PAST",
  "pronoun_candidates": {
    "pronoun@position": "resolved entity name"
  }
}

Canonical relations: HAS, IS, AT, GIVE, SEND, TELL, SHOW, RETURN, \
BECAUSE, LIKES, LOVES, WANTS, KNOWS. Use uppercase.

For pronouns: resolve them against the existing entities listed below. \
If you cannot resolve a pronoun with certainty, omit it.

Do NOT guess. Only extract what is explicitly stated or directly implied."""

_PROMPT_HASH = hashlib.sha256(_EXTRACTION_PROMPT.encode()).hexdigest()[:12]


def _strip_code_fences(text: str) -> str:
    """Strip common markdown code fences (```json, ```, etc.)."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove trailing ``` if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def _extract_first_json(text: str) -> str | None:
    """Extract the first complete JSON object or array from text.

    Uses balanced-brace matching, respecting strings (so braces inside
    string literals are ignored). Returns None if no valid substring found.
    """
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        i = 0
        while True:
            i = text.find(start_char, i)
            if i < 0:
                break
            depth = 0
            in_string = False
            escape = False
            quote_char: str | None = None
            j = i
            while j < len(text):
                c = text[j]
                if escape:
                    escape = False
                    j += 1
                    continue
                if in_string:
                    if c == quote_char:
                        in_string = False
                    j += 1
                    continue
                if c == '"':  # JSON strings are double-quoted only
                    in_string = True
                    quote_char = c
                    j += 1
                    continue
                if c == "\\":
                    escape = True
                    j += 1
                    continue
                if c == start_char:
                    depth += 1
                elif c == end_char:
                    depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
                j += 1
            i += 1
    return None


def _salvage_json(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Try to salvage valid JSON from text. Returns (data, None) on success,
    (None, error_msg) on failure.
    """
    # 1. Strip code fences
    stripped = _strip_code_fences(text)
    try:
        return json.loads(stripped), None
    except json.JSONDecodeError:
        pass

    # 2. Extract first complete JSON object/array
    substring = _extract_first_json(stripped)
    if substring:
        try:
            data = json.loads(substring)
            if isinstance(data, dict):
                return data, None
        except json.JSONDecodeError:
            pass

    return None, "All salvage attempts failed"


def _build_user_message(text: str, pef: PEFState) -> str:
    """Build the user message containing text and existing world state."""
    context = pef.to_context_summary()
    return (
        f"## Existing world state\n{context}\n\n"
        f"## Text to analyze\n{text}"
    )


def _parse_extraction_response(
    raw: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    prompt_hash: str | None = None,
) -> ExtractionResult:
    """Parse a JSON response from the LLM into an ExtractionResult.

    Uses deterministic salvage: strip code fences, extract first complete
    JSON object, then parse. Only reports EXTRACTION_FAILED when all attempts fail.
    """
    text = raw.strip()
    data, salvage_error = _salvage_json(text)

    if data is None:
        snippet = (text[:500] + "…") if len(text) > 500 else text
        raw_preview = (text[:200] + "…") if len(text) > 200 else text
        diagnostic: dict[str, Any] = {
            "reason": "JSON_DECODE_ERROR",
            "json_error": salvage_error or "Parse failed",
            "snippet": snippet,
            "raw_preview": raw_preview,
        }
        if provider is not None:
            diagnostic["provider"] = provider
        if model is not None:
            diagnostic["model"] = model
        if prompt_hash is not None:
            diagnostic["prompt_hash"] = prompt_hash
        logger.error(
            "LLM extraction JSON decode failed",
            extra={
                "error": salvage_error,
                "raw": text[:2000],
                "provider": provider,
                "model": model,
                "prompt_hash": prompt_hash,
            },
        )
        return ExtractionResult(
            claims=[],
            entity_mentions=[],
            span=Span.PRESENT,
            pronoun_candidates={},
            extraction_error=diagnostic,
        )

    # Parse span
    span_str = data.get("span", "PRESENT")
    span = Span.PAST if span_str == "PAST" else Span.PRESENT

    # Parse claims
    claims: list[ExtractedClaim] = []
    for raw_claim in data.get("claims", []):
        claim_span_str = raw_claim.get("span", span_str)
        claims.append(ExtractedClaim(
            subject=raw_claim["subject"],
            relation=canonicalize_relation(str(raw_claim.get("relation", ""))),
            obj=raw_claim["obj"],
            span=Span.PAST if claim_span_str == "PAST" else Span.PRESENT,
            negated=raw_claim.get("negated", False),
            evidence=raw_claim.get("evidence", ""),
            provenance="user_input",
            extractor_backend="llm",
        ))

    return ExtractionResult(
        claims=claims,
        entity_mentions=data.get("entity_mentions", []),
        span=span,
        pronoun_candidates=data.get("pronoun_candidates", {}),
    )


class LLMExtractionBackend(ExtractionBackend):
    """Extraction backend that uses a separate LLM call for structured extraction.

    Takes an LLMAdapter (same interface as the fluency adapter, but
    typically configured with a cheap/fast model). This is a SEPARATE
    LLM call — different prompt, different purpose, potentially different model.
    """

    def __init__(self, adapter: LLMAdapter, provider: str | None = None):
        self._adapter = adapter
        self._provider = provider

    def supports_ambiguous_referents(self) -> bool:
        """LLM extraction path is explicitly out of ambiguity-parity scope (PR6)."""
        return False

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        """Extract entities, relationships, and span signals via LLM.

        Sends a structured extraction prompt to the LLM and parses
        the JSON response into an ExtractionResult. On parse failure
        (non-JSON response), returns ExtractionResult with extraction_error
        so governance can intervene instead of raising.
        """
        user_message = _build_user_message(text, pef)
        messages = [
            {"role": "system", "content": _EXTRACTION_PROMPT},
            {"role": "user", "content": user_message},
        ]
        response = await self._adapter.generate(messages)
        return _parse_extraction_response(
            response.text or "",
            provider=self._provider,
            model=response.model or None,
            prompt_hash=_PROMPT_HASH,
        )
