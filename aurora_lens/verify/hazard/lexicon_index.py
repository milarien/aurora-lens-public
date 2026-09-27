"""Deterministic multi-pattern token-sequence lexicon index."""

from __future__ import annotations

from collections import defaultdict

from aurora_lens.verify.hazard.schema import (
    HazardOntology,
    LexiconHit,
    LexiconKind,
)

MAX_MATCHES_RETAINED = 64


class LexiconIndex:
    """Longest-match-first contiguous token-sequence matcher.

    Patterns are exact token tuples from ontology aliases. Matching is
    deterministic: scan left-to-right, at each position try longest patterns first.

    When a permitted shared alias hits multiple entity kinds at the same length,
    all matching kinds are recorded (multi_match). Kind scan order matches
    ``ALIAS_RESOLUTION_ORDER`` in ontology_loader (substance → organism → class →
    process → action → framing).
    """

    def __init__(self, ontology: HazardOntology) -> None:
        # length -> kind -> record_id -> frozenset of alias tuples of that length
        self._by_length: dict[int, dict[LexiconKind, dict[str, frozenset[tuple[str, ...]]]]] = (
            defaultdict(lambda: defaultdict(dict))
        )
        self._max_len = 0
        self._ingest(ontology)

    def _add(
        self,
        kind: LexiconKind,
        record_id: str,
        aliases: tuple[tuple[str, ...], ...],
    ) -> None:
        by_len: dict[int, set[tuple[str, ...]]] = defaultdict(set)
        for alias in aliases:
            if not alias:
                continue
            by_len[len(alias)].add(alias)
            self._max_len = max(self._max_len, len(alias))
        for length, alias_set in by_len.items():
            self._by_length[length][kind][record_id] = frozenset(alias_set)

    def _ingest(self, ontology: HazardOntology) -> None:
        for rec in ontology.hazard_classes.values():
            aliases = rec.aliases
            if not aliases and rec.label:
                from aurora_lens.verify.hazard.normalize import alias_to_tokens

                aliases = (alias_to_tokens(rec.label),)
            self._add(LexiconKind.HAZARD_CLASS, rec.id, aliases)
        for rec in ontology.substances.values():
            self._add(LexiconKind.SUBSTANCE, rec.id, rec.aliases)
        for rec in ontology.organisms.values():
            self._add(LexiconKind.ORGANISM, rec.id, rec.aliases)
        for rec in ontology.processes.values():
            self._add(LexiconKind.PROCESS, rec.id, rec.aliases)
        for rec in ontology.actions.values():
            self._add(LexiconKind.ACTION, rec.id, rec.aliases)
        for rec in ontology.framings.values():
            self._add(LexiconKind.FRAMING, rec.id, rec.aliases)

    def match(self, tokens: tuple[str, ...]) -> tuple[LexiconHit, ...]:
        if not tokens or self._max_len == 0:
            return ()
        hits: list[LexiconHit] = []
        n = len(tokens)
        i = 0
        while i < n and len(hits) < MAX_MATCHES_RETAINED:
            matched = False
            for length in range(min(self._max_len, n - i), 0, -1):
                window = tuple(tokens[i : i + length])
                kind_map = self._by_length.get(length)
                if not kind_map:
                    continue
                # Stable kind order for audit determinism
                for kind in (
                    LexiconKind.SUBSTANCE,
                    LexiconKind.ORGANISM,
                    LexiconKind.HAZARD_CLASS,
                    LexiconKind.PROCESS,
                    LexiconKind.ACTION,
                    LexiconKind.FRAMING,
                ):
                    records = kind_map.get(kind)
                    if not records:
                        continue
                    for record_id in sorted(records.keys()):
                        if window in records[record_id]:
                            hits.append(
                                LexiconHit(
                                    kind=kind,
                                    record_id=record_id,
                                    start=i,
                                    end=i + length,
                                    tokens=window,
                                )
                            )
                            matched = True
                            if len(hits) >= MAX_MATCHES_RETAINED:
                                return tuple(hits)
                if matched:
                    # Advance past longest match window once any kind matched at this length
                    i += length
                    break
            if not matched:
                i += 1
        return tuple(hits)
