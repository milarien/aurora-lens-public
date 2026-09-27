"""Location (AT / location-bearing IS) reads over committed PEF."""

from __future__ import annotations

from aurora_lens.pef.at_read import current_at
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.state_native_engine.bind import resolve_entity
from aurora_lens.state_native_engine.contracts import (
    StateNativeDelegationResult,
    StateNativeOutcome,
    StateNativeSolverFamily,
)
from aurora_lens.state_native_engine.epistemic import EpistemicResult
from aurora_lens.state_native_engine.eval.entity_typo_near_match import (
    augment_state_native_ambiguity_with_typo_recovery,
)

# Location-bearing IS predicates: bare words that are unambiguously locative.
# Prefix patterns ("in …", "at …", "on …") are handled by the startswith check.
_LOCATION_BARE_PREDICATES: frozenset[str] = frozenset({
    "overseas",
    "abroad",
    "away",
    "interstate",
    "home",
    "here",
    "there",
})

_LOCATION_READ_SOLVER = StateNativeSolverFamily.COMMITTED_LOCATION_READ


def _location_delegation(
    outcome: StateNativeOutcome,
    text: str,
    *,
    epistemic: EpistemicResult,
    clarify_context: dict | None = None,
    stop_reason_code: str | None = None,
) -> StateNativeDelegationResult:
    return StateNativeDelegationResult(
        handled=True,
        outcome=outcome,
        user_visible_text=text,
        clarify_context=clarify_context,
        stop_reason_code=stop_reason_code,
        solver_family=_LOCATION_READ_SOLVER,
        epistemic_result=epistemic,
    )


def _is_location_bearing_predicate(obj_text: str) -> bool:
    """True when an IS object literal is semantically equivalent to a location."""
    t = (obj_text or "").strip().lower()
    if not t:
        return False
    if t in _LOCATION_BARE_PREDICATES:
        return True
    return t.startswith("in ") or t.startswith("at ") or t.startswith("on ")


def _location_bearing_is_relation(pef: PEFState, subject_id: str) -> Relationship | None:
    """Return the most recent active positive IS relation whose object is location-bearing.

    Used as a fallback when no explicit AT relation exists.
    """
    best: Relationship | None = None
    for rel in pef.relationships:
        if rel.subject_id != subject_id:
            continue
        if rel.relation != "IS":
            continue
        if rel.negated:
            continue
        obj = rel.object_literal
        if not isinstance(obj, str):
            continue
        if not _is_location_bearing_predicate(obj):
            continue
        if best is None or rel.source_turn > best.source_turn:
            best = rel
    return best


def _format_at_location(pef: PEFState, rel: Relationship) -> str:
    if rel.object_literal is not None:
        return str(rel.object_literal).strip()
    if rel.object_entity_id is not None:
        ent = pef.entities.get(rel.object_entity_id)
        if ent is not None:
            return ent.name.strip()
        return rel.object_entity_id
    return ""


def _norm_place_token(s: str) -> str:
    """Lowercase, strip edges, drop a single leading ``the `` (object literal vs bare name)."""
    t = s.lower().strip(" .")
    if t.startswith("the "):
        t = t[4:].strip(" .")
    return t


def _committed_place_matches(committed_loc: str, claimed: str) -> bool:
    """True when committed AT object text matches the claimed place (checker-aligned)."""
    a = _norm_place_token(committed_loc)
    b = _norm_place_token(claimed)
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 4 and long.startswith(short):
        remainder = long[len(short) :]
        if not remainder or remainder[0] in (" ", ".", ",", "-", "/"):
            return True
    return False


def resolve_entities_for_location_subject(pef: PEFState, phrase: str) -> list[Entity]:
    """Resolve subject phrase to zero or more entities (committed PEF only)."""
    return list(resolve_entity(phrase, pef).entities)


def _plural_head_expansion_candidates(pef: PEFState, phrase: str) -> list[Entity]:
    """Best-effort plural head expansion when direct entity binding misses.

    Example: ``boxes`` can expand to entities like ``Box A``/``Box B``/``Box C``.
    """
    raw = (phrase or "").strip()
    if not raw:
        return []
    low = raw.lower()
    if low.startswith("the "):
        low = low[4:].strip()
    parts = [p for p in low.split() if p]
    if not parts:
        return []
    head = parts[-1]

    singulars: set[str] = {head}
    if head.endswith("ies") and len(head) > 3:
        singulars.add(head[:-3] + "y")
    if head.endswith("es") and len(head) > 2:
        singulars.add(head[:-2])
    if head.endswith("s") and len(head) > 1:
        singulars.add(head[:-1])

    out: list[Entity] = []
    seen: set[str] = set()
    for ent in pef.entities.values():
        name_low = ent.name.lower().strip()
        if not name_low:
            continue
        first = name_low.split()[0]
        if first in singulars and ent.id not in seen:
            seen.add(ent.id)
            out.append(ent)
    return out


def evaluate_location_from_committed_state(
    pef: PEFState,
    subject_phrase: str,
) -> StateNativeDelegationResult:
    """Strict ``Where is X?`` reads over committed AT / location-bearing IS.

    Value-returning answers use :class:`EpistemicResult`.VALUE; stops and
    clarification carry UNKNOWN / AMBIGUOUS explicitly.
    """
    entities = resolve_entities_for_location_subject(pef, subject_phrase)
    if len(entities) > 1:
        names = sorted(e.name for e in entities)
        opts = ", ".join(names)
        generic = f"Which did you mean: {opts}?"
        base_ctx = {
            "failed_constraint": "STATE_NATIVE_ENTITY_AMBIGUITY",
            "candidate_entities": names,
            "subject_phrase": subject_phrase.strip(),
        }
        text, ctx = augment_state_native_ambiguity_with_typo_recovery(
            pef,
            entities,
            ambiguity_kind="location_subject_query",
            generic_user_visible_text=generic,
            clarify_context=base_ctx,
        )
        return _location_delegation(
            StateNativeOutcome.CLARIFY,
            text,
            epistemic=EpistemicResult.AMBIGUOUS,
            clarify_context=ctx,
        )
    if len(entities) == 0:
        entities = _plural_head_expansion_candidates(pef, subject_phrase)
        if len(entities) > 1:
            names = sorted(e.name for e in entities)
            opts = ", ".join(names)
            generic = f"Which did you mean: {opts}?"
            base_ctx = {
                "failed_constraint": "UNRESOLVED_REFERENT",
                "candidate_entities": names,
                "subject_phrase": subject_phrase.strip(),
            }
            text, ctx = augment_state_native_ambiguity_with_typo_recovery(
                pef,
                entities,
                ambiguity_kind="location_subject_query",
                generic_user_visible_text=generic,
                clarify_context=base_ctx,
            )
            return _location_delegation(
                StateNativeOutcome.CLARIFY,
                text,
                epistemic=EpistemicResult.AMBIGUOUS,
                clarify_context=ctx,
            )
        if len(entities) == 0:
            return _location_delegation(
                StateNativeOutcome.STOP,
                "I cannot answer that from committed state: no matching entity for the "
                f"name you gave ({subject_phrase.strip()}).",
                epistemic=EpistemicResult.UNKNOWN,
                stop_reason_code="state_native_unknown_entity",
            )

    ent = entities[0]
    rel = current_at(pef, ent.id)
    if rel is None:
        # No explicit AT relation — fall back to location-bearing IS predicates.
        is_rel = _location_bearing_is_relation(pef, ent.id)
        if is_rel is not None:
            loc = (is_rel.object_literal or "").strip()
            if loc:
                label = subject_phrase.strip() or ent.name
                return _location_delegation(
                    StateNativeOutcome.ANSWER,
                    f"{label} is {loc}.",
                    epistemic=EpistemicResult.VALUE,
                )
        return _location_delegation(
            StateNativeOutcome.STOP,
            "I cannot answer that from committed state: no location (AT) recorded for "
            f"{ent.name}.",
            epistemic=EpistemicResult.UNKNOWN,
            stop_reason_code="state_native_no_at",
        )

    loc = _format_at_location(pef, rel)
    if not loc:
        return _location_delegation(
            StateNativeOutcome.STOP,
            "I cannot answer that from committed state: the location record is incomplete.",
            epistemic=EpistemicResult.UNKNOWN,
            stop_reason_code="state_native_at_incomplete",
        )

    return _location_delegation(
        StateNativeOutcome.ANSWER,
        f"{ent.name} is at {loc}.",
        epistemic=EpistemicResult.VALUE,
    )


def evaluate_still_in_from_committed_state(
    pef: PEFState,
    subject_phrase: str,
    claimed_place: str,
) -> StateNativeDelegationResult:
    """Boolean ``Is X still in Y?`` against ``current_at`` / formatted location."""
    entities = resolve_entities_for_location_subject(pef, subject_phrase)
    if len(entities) > 1:
        names = sorted(e.name for e in entities)
        opts = ", ".join(names)
        generic = f"Which did you mean: {opts}?"
        base_ctx = {
            "failed_constraint": "STATE_NATIVE_ENTITY_AMBIGUITY",
            "candidate_entities": names,
            "subject_phrase": subject_phrase.strip(),
        }
        text, ctx = augment_state_native_ambiguity_with_typo_recovery(
            pef,
            entities,
            ambiguity_kind="location_subject_query",
            generic_user_visible_text=generic,
            clarify_context=base_ctx,
        )
        return _location_delegation(
            StateNativeOutcome.CLARIFY,
            text,
            epistemic=EpistemicResult.AMBIGUOUS,
            clarify_context=ctx,
        )
    if len(entities) == 0:
        return _location_delegation(
            StateNativeOutcome.STOP,
            "I cannot answer that from committed state: no matching entity for the "
            f"name you gave ({subject_phrase.strip()}).",
            epistemic=EpistemicResult.UNKNOWN,
            stop_reason_code="state_native_unknown_entity",
        )

    ent = entities[0]
    rel = current_at(pef, ent.id)
    if rel is None:
        return _location_delegation(
            StateNativeOutcome.STOP,
            "I cannot answer that from committed state: no location (AT) recorded for "
            f"{ent.name}.",
            epistemic=EpistemicResult.UNKNOWN,
            stop_reason_code="state_native_no_at",
        )

    loc = _format_at_location(pef, rel)
    if not loc:
        return _location_delegation(
            StateNativeOutcome.STOP,
            "I cannot answer that from committed state: the location record is incomplete.",
            epistemic=EpistemicResult.UNKNOWN,
            stop_reason_code="state_native_at_incomplete",
        )

    if _committed_place_matches(loc, claimed_place):
        return _location_delegation(
            StateNativeOutcome.ANSWER,
            f"Yes: {ent.name} is at {loc} in committed state.",
            epistemic=EpistemicResult.TRUE,
        )
    return _location_delegation(
        StateNativeOutcome.ANSWER,
        f"No: committed state has {ent.name} at {loc}, not at {claimed_place}.",
        epistemic=EpistemicResult.FALSE,
    )
