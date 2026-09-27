"""Durable unresolved-referent registry (v10-style session constraint in PEF).

Ambiguity detected on a turn is registered here and persists until explicit
resolution. Unlike ``pending_clarification`` (governance continuation payload),
this registry is authoritative for blocking dependent consequence across turns
even when re-extraction omits the pronoun or governance did not CONTAIN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aurora_lens.interpret.schema import ExtractionResult
    from aurora_lens.pef.state import PEFState

UNRESOLVED_REFERENT_REGISTRY_SCHEMA_VERSION = 2
STATUS_OPEN = "open"
STATUS_RESOLVED = "resolved"
RESOLUTION_MODE_PENDING = "pending"
RESOLUTION_MODE_HELD_UNRESOLVED = "held_unresolved"
HOLD_UNRESOLVED_CHOICE_LABEL = "Hold unresolved"


@dataclass
class UnresolvedReferentEntry:
    """One durable unresolved attribution constraint."""

    entry_id: str
    token: str
    span_surface: str
    candidate_entities: list[str] = field(default_factory=list)
    blocked_proposition: str | None = None
    head: str | None = None
    introduced_turn: int = 0
    introduced_utterance: str = ""
    status: str = STATUS_OPEN
    resolved_entity: str | None = None
    resolved_turn: int | None = None
    resolution_mode: str = RESOLUTION_MODE_PENDING

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": UNRESOLVED_REFERENT_REGISTRY_SCHEMA_VERSION,
            "entry_id": self.entry_id,
            "token": self.token,
            "span_surface": self.span_surface,
            "candidate_entities": list(self.candidate_entities),
            "blocked_proposition": self.blocked_proposition,
            "head": self.head,
            "introduced_turn": self.introduced_turn,
            "introduced_utterance": self.introduced_utterance,
            "status": self.status,
            "resolved_entity": self.resolved_entity,
            "resolved_turn": self.resolved_turn,
            "resolution_mode": self.resolution_mode,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> UnresolvedReferentEntry:
        return UnresolvedReferentEntry(
            entry_id=str(data.get("entry_id") or ""),
            token=str(data.get("token") or ""),
            span_surface=str(data.get("span_surface") or data.get("token") or ""),
            candidate_entities=[
                str(x) for x in (data.get("candidate_entities") or []) if str(x).strip()
            ],
            blocked_proposition=data.get("blocked_proposition"),
            head=data.get("head"),
            introduced_turn=int(data.get("introduced_turn") or 0),
            introduced_utterance=str(data.get("introduced_utterance") or ""),
            status=str(data.get("status") or STATUS_OPEN),
            resolved_entity=data.get("resolved_entity"),
            resolved_turn=(
                int(data["resolved_turn"])
                if data.get("resolved_turn") is not None
                else None
            ),
            resolution_mode=str(
                data.get("resolution_mode") or RESOLUTION_MODE_PENDING
            ),
        )


def _token_in_text(token: str, text: str) -> bool:
    if not token or not text:
        return False
    tl = text.lower()
    sub_l = token.lower()
    if sub_l not in tl:
        return False
    slen = len(sub_l)
    i = 0
    while i <= len(tl) - slen:
        j = tl.find(sub_l, i)
        if j < 0:
            return False
        left_ok = j == 0 or not tl[j - 1].isalpha()
        right_ok = j + slen == len(tl) or not tl[j + slen].isalpha()
        if left_ok and right_ok:
            return True
        i = j + 1
    return False


def open_entries(pef: PEFState) -> list[UnresolvedReferentEntry]:
    return [e for e in pef.unresolved_referent_registry if e.status == STATUS_OPEN]


def open_registry_tokens(pef: PEFState) -> frozenset[str]:
    return frozenset(e.token.lower() for e in open_entries(pef) if e.token.strip())


def open_registry_tokens_in_text(pef: PEFState, text: str) -> list[str]:
    """Open registry tokens that appear as whole words in ``text`` (surface order)."""
    if not text.strip():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for entry in open_entries(pef):
        tok = entry.token.strip()
        if not tok:
            continue
        tl = tok.lower()
        if tl in seen:
            continue
        if _token_in_text(tok, text):
            seen.add(tl)
            out.append(tok)
    return out


def candidates_for_tokens(pef: PEFState, tokens: list[str]) -> list[str]:
    want = {t.lower() for t in tokens if t.strip()}
    found: list[str] = []
    seen: set[str] = set()
    for entry in open_entries(pef):
        if entry.token.lower() not in want:
            continue
        for name in entry.candidate_entities:
            key = name.strip()
            if not key:
                continue
            kl = key.lower()
            if kl not in seen:
                seen.add(kl)
                found.append(key)
    return found


def _entry_id_for(token: str, turn: int) -> str:
    return f"ur:{token.lower()}@{turn}"


def register_unresolved_referents(
    pef: PEFState,
    *,
    turn: int,
    utterance: str,
    tokens: list[str],
    candidate_entities: list[str] | None = None,
    blocked_proposition: str | None = None,
    span_surfaces: dict[str, str] | None = None,
) -> None:
    """Upsert open registry entries for ``tokens`` (does not resolve)."""
    if not tokens:
        return
    surfaces = span_surfaces or {}
    open_by_token = {e.token.lower(): e for e in open_entries(pef)}
    for raw in tokens:
        tok = str(raw).strip()
        if not tok:
            continue
        tl = tok.lower()
        if tl in pef.discourse_referent_bindings:
            continue
        surface = surfaces.get(tl) or surfaces.get(tok) or tok
        existing = open_by_token.get(tl)
        if existing is not None:
            if candidate_entities:
                merged = list(existing.candidate_entities)
                seen = {x.lower() for x in merged}
                for c in candidate_entities:
                    if c.strip() and c.lower() not in seen:
                        merged.append(c.strip())
                        seen.add(c.lower())
                existing.candidate_entities = merged
            if blocked_proposition and not existing.blocked_proposition:
                existing.blocked_proposition = blocked_proposition
            if surface and surface != existing.span_surface and len(surface) > len(existing.span_surface):
                existing.span_surface = surface
            continue
        entry = UnresolvedReferentEntry(
            entry_id=_entry_id_for(tok, turn),
            token=tok,
            span_surface=surface,
            candidate_entities=list(candidate_entities or []),
            blocked_proposition=blocked_proposition,
            introduced_turn=turn,
            introduced_utterance=utterance,
            status=STATUS_OPEN,
        )
        pef.unresolved_referent_registry.append(entry)
        open_by_token[tl] = entry


def resolve_unresolved_referents(
    pef: PEFState,
    *,
    tokens: list[str],
    resolved_entity: str,
    turn: int,
) -> None:
    """Mark matching open entries resolved — only explicit binding path."""
    if not tokens or not resolved_entity.strip():
        return
    want = {t.lower() for t in tokens if t.strip()}
    for entry in pef.unresolved_referent_registry:
        if entry.status != STATUS_OPEN:
            continue
        if entry.token.lower() in want:
            entry.status = STATUS_RESOLVED
            entry.resolved_entity = resolved_entity.strip()
            entry.resolved_turn = turn
            entry.resolution_mode = RESOLUTION_MODE_PENDING


def hold_unresolved_referents(
    pef: PEFState,
    *,
    turn: int,
    tokens: list[str] | None = None,
) -> int:
    """Mark open entries as held-unresolved without binding to any candidate."""
    want: set[str] | None = None
    if tokens:
        want = {t.lower() for t in tokens if t.strip()}
    count = 0
    for entry in pef.unresolved_referent_registry:
        if entry.status != STATUS_OPEN:
            continue
        if want is not None and entry.token.lower() not in want:
            continue
        entry.resolution_mode = RESOLUTION_MODE_HELD_UNRESOLVED
        count += 1
    return count


def referent_registry_held_unresolved(pef: PEFState) -> bool:
    return any(
        e.status == STATUS_OPEN
        and e.resolution_mode == RESOLUTION_MODE_HELD_UNRESOLVED
        for e in pef.unresolved_referent_registry
    )


def build_unresolved_referent_clarification_choices(
    candidate_entities: list[str],
    *,
    include_hold_unresolved: bool = True,
) -> list[str]:
    """Candidate bind options plus optional Hold unresolved for UNRESOLVED_REFERENT ASK."""
    choices = [c.strip() for c in candidate_entities if str(c).strip()]
    if include_hold_unresolved and len(choices) >= 2:
        choices.append(HOLD_UNRESOLVED_CHOICE_LABEL)
    return choices


def clarification_choices_from_pending(
    pending: dict | None,
    *,
    pef: PEFState | None = None,
) -> list[str]:
    if not pending or str(pending.get("failed_constraint") or "") != "UNRESOLVED_REFERENT":
        return []
    if (
        str(pending.get("resolution_mode") or "") == RESOLUTION_MODE_HELD_UNRESOLVED
        or (pef is not None and referent_registry_held_unresolved(pef))
    ):
        return []
    explicit = pending.get("clarification_choices")
    if isinstance(explicit, list) and explicit:
        return [str(x).strip() for x in explicit if str(x).strip()]
    return build_unresolved_referent_clarification_choices(
        _normalized_candidates_from_pending(pending),
    )


def _normalized_candidates_from_pending(pending: dict) -> list[str]:
    raw = pending.get("candidate_entities")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


def _possessive_surface(name: str) -> tuple[str, ...]:
    base = name.strip().lower()
    if not base:
        return ()
    if base.endswith("s"):
        return (f"{base}'", f"{base}'s")
    return (f"{base}'s",)


def try_explicit_possessive_attribution(
    text: str,
    candidate_entities: list[str],
) -> str | None:
    """Return candidate when text explicitly attributes via possessive (e.g. contractor's certification)."""
    low = (text or "").lower()
    if not low.strip():
        return None
    matched: list[str] = []
    for cand in candidate_entities:
        label = cand.strip()
        if not label:
            continue
        if any(poss in low for poss in _possessive_surface(label)):
            matched.append(label)
    if len(matched) == 1:
        return matched[0]
    return None


def blocked_tokens_for_admission(pef: PEFState) -> frozenset[str]:
    """All open registry pronoun tokens — block PEF mint/bind until resolved."""
    return open_registry_tokens(pef)


def sync_registry_from_extraction(
    pef: PEFState,
    *,
    turn: int,
    utterance: str,
    extraction: ExtractionResult,
    candidate_entities: list[str] | None = None,
    blocked_proposition: str | None = None,
) -> None:
    """Register extractor-marked referents not yet discourse-bound."""
    if not extraction.ambiguous_referents:
        return
    bound = pef.discourse_referent_bindings
    tokens = [p for p in extraction.ambiguous_referents if p.lower() not in bound]
    if not tokens:
        return
    register_unresolved_referents(
        pef,
        turn=turn,
        utterance=utterance,
        tokens=tokens,
        candidate_entities=candidate_entities,
        blocked_proposition=blocked_proposition,
    )


def registry_entries_to_wire(entries: list[UnresolvedReferentEntry]) -> list[dict[str, Any]]:
    return [e.to_dict() for e in entries]


def registry_entries_from_wire(raw: Any) -> list[UnresolvedReferentEntry]:
    if not isinstance(raw, list):
        return []
    out: list[UnresolvedReferentEntry] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(UnresolvedReferentEntry.from_dict(item))
    return out
