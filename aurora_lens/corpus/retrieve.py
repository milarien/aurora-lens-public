"""Retrieve corpus chunks and assemble governed-proxy RAG messages."""

from __future__ import annotations

import re
from typing import Sequence

from aurora_lens.corpus.models import CorpusChunk
from aurora_lens.corpus.registry import CorpusRegistry
from aurora_lens.corpus.review import format_corpus_for_review, select_chunks

_DEFAULT_MAX_CHARS = 12000

_QUERY_STOPWORDS = frozenset(
    {
        "about",
        "are",
        "change",
        "define",
        "does",
        "from",
        "how",
        "part",
        "the",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }
)

RAG_ANSWER_RULES = """\
Answer using ONLY the Context sections below.

Rules:
- Extract direct mechanisms (formulas, thresholds, explicit rules) when the question asks how something works.
- Cite the chunk header (### line) for each factual claim.
- Quote verbatim phrases for formulas, numeric thresholds, and named protocols.
- If the Context does not contain enough detail to answer, say exactly: "Insufficient context in retrieved chunks."
- Do not substitute conceptual paraphrase when the corpus states an explicit procedure or formula.
- Do not use outside knowledge.
"""


def _query_tokens(query: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9]+", query.lower())
        if len(t) >= 3 and t not in _QUERY_STOPWORDS
    ]


def _query_phrases(query: str) -> list[str]:
    lowered = query.lower()
    phrases: list[str] = []
    for match in re.finditer(r"[a-z0-9]+(?: [a-z0-9]+)+", lowered):
        phrase = match.group(0).strip()
        if len(phrase) >= 7:
            phrases.append(phrase)
    return phrases


def rank_chunks_by_query(chunks: Sequence[CorpusChunk], query: str) -> list[CorpusChunk]:
    """Order chunks by token and phrase overlap with ``query`` (ordinal tie-break)."""
    tokens = _query_tokens(query)
    phrases = _query_phrases(query)
    if not tokens and not phrases:
        return sorted(chunks, key=lambda c: c.ordinal)

    def score(ch: CorpusChunk) -> tuple[int, int]:
        text = ch.text.lower()
        token_hits = sum(1 for t in tokens if t in text)
        phrase_hits = sum(3 for p in phrases if p in text)
        return (-(token_hits + phrase_hits), ch.ordinal)

    return sorted(chunks, key=score)


def chunk_relevance_score(ch: CorpusChunk, query: str) -> int:
    """Token + phrase overlap score for operator evidence display."""
    tokens = _query_tokens(query)
    phrases = _query_phrases(query)
    if not tokens and not phrases:
        return 0
    text = ch.text.lower()
    token_hits = sum(1 for t in tokens if t in text)
    phrase_hits = sum(3 for p in phrases if p in text)
    return token_hits + phrase_hits


def retrieve_chunks(
    registry: CorpusRegistry,
    record_id: str,
    *,
    chunk_ids: Sequence[str] | None = None,
    chunk_ordinals: set[int] | None = None,
    max_chars: int = _DEFAULT_MAX_CHARS,
    query: str | None = None,
) -> list[CorpusChunk]:
    """Load chunks for ``record_id`` and apply optional filters and char budget."""
    rid = record_id.strip()
    if not rid:
        raise ValueError("record_id must be non-empty")
    if registry.get_record(rid) is None:
        raise KeyError(f"record not found: {rid!r}")

    all_chunks = registry.load_chunks(rid)
    if not all_chunks:
        raise ValueError(f"no chunks for record {rid!r}")

    if chunk_ids is not None:
        wanted = {cid.strip() for cid in chunk_ids if cid.strip()}
        pool = [c for c in all_chunks if c.chunk_id in wanted]
        if not pool:
            raise ValueError("no chunks matched chunk_ids")
        pool = sorted(pool, key=lambda c: c.ordinal)
    else:
        pool = list(all_chunks)

    ranked = False
    if query and query.strip():
        pool = rank_chunks_by_query(pool, query.strip())
        ranked = True

    if chunk_ordinals is not None:
        pool = [c for c in pool if c.ordinal in chunk_ordinals]
        if not pool:
            raise ValueError("no chunks matched chunk ordinals")

    return select_chunks(pool, max_chars=max_chars, preserve_order=ranked)


def assemble_rag_message(chunks: Sequence[CorpusChunk], question: str) -> str:
    """Build ``Context:`` … ``Question:`` harness message for the governed proxy."""
    q = question.strip()
    if not q:
        raise ValueError("question must be non-empty")
    if not chunks:
        raise ValueError("chunks must be non-empty")
    context_body = format_corpus_for_review(chunks)
    context_with_rules = f"{RAG_ANSWER_RULES}\n\n{context_body}"
    return f"Context:\n{context_with_rules}\n\nQuestion: {q}"


def assemble_request_metadata(
    record_id: str,
    *,
    source_scope: Sequence[str] = (),
    policy_profile: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, object]:
    """Build ``request_metadata`` dict for proxy POST bodies."""
    rid = record_id.strip()
    if not rid:
        raise ValueError("record_id must be non-empty")
    meta: dict[str, object] = {"record_ids": [rid]}
    scope = [s.strip() for s in source_scope if s and str(s).strip()]
    if scope:
        meta["source_scope"] = scope
    if policy_profile and policy_profile.strip():
        meta["policy_profile"] = policy_profile.strip()
    if workspace_id and workspace_id.strip():
        meta["workspace_id"] = workspace_id.strip()
    return meta
