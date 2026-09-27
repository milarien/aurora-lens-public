"""Inventory (**HAS** / **CONTAIN**) reads over committed PEF for bounded queries."""

from __future__ import annotations

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
from aurora_lens.state_native_engine.lexical import (
    analyze_item_phrase,
    display_quantity_item,
    item_key,
    normalize_possession_item_surface,
)
from aurora_lens.state_native_engine.parse.query_surface import (
    parse_inventory_quantity_holder_phrase,
)

_INVENTORY_READ_SOLVER = StateNativeSolverFamily.COMMITTED_INVENTORY_READ

_COUNTED_INVENTORY_RELATIONS: frozenset[str] = frozenset({"HAS", "CONTAIN"})


def _inventory_delegation(
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
        solver_family=_INVENTORY_READ_SOLVER,
        epistemic_result=epistemic,
    )


def _normalize_phrase(text: str) -> str:
    phrase = (text or "").strip()
    low = phrase.lower()
    if low.startswith("the "):
        return phrase[4:].strip()
    return phrase


_SMALL_NUMBER_WORDS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def _normalize_item_key(item: str) -> str:
    """Backward-compatible alias for countable / bare noun matching keys."""
    return item_key(item)


def _parse_counted_item_literal(obj_literal: str) -> tuple[int, str] | None:
    s = normalize_possession_item_surface(_normalize_phrase(obj_literal))
    if not s:
        return None
    parts = s.split(None, 1)
    if len(parts) != 2:
        return None
    n_raw, item = parts[0].lower(), normalize_possession_item_surface(parts[1])
    if not item:
        return None
    if n_raw.isdigit():
        return (int(n_raw), item)
    if n_raw in _SMALL_NUMBER_WORDS:
        return (_SMALL_NUMBER_WORDS[n_raw], item)
    return None


def _format_inventory_object(pef: PEFState, rel: Relationship) -> str:
    if rel.object_literal is not None:
        return str(rel.object_literal).strip()
    if rel.object_entity_id is not None:
        ent = pef.entities.get(rel.object_entity_id)
        if ent is not None:
            return ent.name.strip()
        return rel.object_entity_id
    return ""


def _current_state_for_item(
    pef: PEFState, subject_id: str, item_lower: str
) -> tuple[bool, int] | None:
    """Latest source_turn wins for (subject, item); same turn: NOT_HAS wins.

    Returns ``(is_held, source_turn)`` or ``None`` if no HAS assertion exists.
    """
    best_turn = -1
    best_negated: bool | None = None
    for rel in pef.get_relationships_for_subject(subject_id):
        if rel.relation != "HAS":
            continue
        obj = _format_inventory_object(pef, rel)
        if not obj or obj.lower() != item_lower:
            continue
        if rel.source_turn > best_turn or (
            rel.source_turn == best_turn and rel.negated and not best_negated
        ):
            best_turn = rel.source_turn
            best_negated = rel.negated
    if best_negated is None:
        return None
    return (not best_negated, best_turn)


def _subject_inventory_objects(pef: PEFState, subject_id: str) -> list[str]:
    items: dict[str, str] = {}  # lowercase key → display name (first seen)
    for rel in pef.get_relationships_for_subject(subject_id):
        if rel.relation != "HAS":
            continue
        obj = _format_inventory_object(pef, rel)
        if obj:
            items.setdefault(obj.lower(), obj)
    vals = []
    for low, display in items.items():
        state = _current_state_for_item(pef, subject_id, low)
        if state is not None and state[0]:
            vals.append(display)
    return sorted(set(vals), key=str.lower)


def _subject_not_has_objects(pef: PEFState, subject_id: str) -> list[str]:
    """Items whose current state for this subject is NOT_HAS (latest turn)."""
    items: dict[str, str] = {}
    for rel in pef.get_relationships_for_subject(subject_id):
        if rel.relation != "HAS":
            continue
        obj = _format_inventory_object(pef, rel)
        if obj:
            items.setdefault(obj.lower(), obj)
    vals = []
    for low, display in items.items():
        state = _current_state_for_item(pef, subject_id, low)
        if state is not None and not state[0]:
            vals.append(display)
    return sorted(set(vals), key=str.lower)


def _find_current_holder_of_item(
    pef: PEFState, item_phrase: str, exclude_subject_id: str
) -> Entity | None:
    """Find the current HAS holder of item_phrase (turn-ordered), excluding one subject."""
    low = _normalize_phrase(item_phrase).lower()
    if not low:
        return None
    candidates: dict[str, str] = {}  # subject_id → matched item_lower
    for rel in pef.get_relationships_by_relation("HAS"):
        if rel.subject_id == exclude_subject_id:
            continue
        obj = _format_inventory_object(pef, rel)
        if not obj:
            continue
        obj_low = obj.lower()
        if obj_low == low or low in obj_low or obj_low in low:
            candidates.setdefault(rel.subject_id, obj_low)
    for sid, item_low in candidates.items():
        state = _current_state_for_item(pef, sid, item_low)
        if state is not None and state[0]:
            ent = pef.entities.get(sid)
            if ent is not None:
                return ent
    return None


def _with_article(item: str) -> str:
    """Prepend 'the' to a bare noun; leave proper nouns and already-articled phrases alone."""
    low = item.lower()
    if low.startswith(("the ", "a ", "an ")):
        return item
    if item and item[0].isupper():
        return item
    return f"the {item}"


def _join_items(items: list[str]) -> str:
    articled = [_with_article(i) for i in items]
    if len(articled) == 1:
        return articled[0]
    if len(articled) == 2:
        return f"{articled[0]} and {articled[1]}"
    return ", ".join(articled[:-1]) + f", and {articled[-1]}"


def _join_holder_names(names: list[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _inventory_group_item_display(item: str) -> str:
    item = (item or "").strip()
    if not item:
        return item

    lower = item.lower()
    if lower.startswith(("a ", "an ", "the ", "some ", "any ")):
        return item

    if analyze_item_phrase(item).number == "plural":
        return item

    return f"the {item}"


_INVENTORY_HOLDER_ITEM_QUERY_PRONOUNS: frozenset[str] = frozenset(
    {"them", "it", "these", "those", "they"}
)


def _holder_query_item_is_unresolved_pronoun(item_phrase: str) -> bool:
    t = normalize_possession_item_surface(_normalize_phrase(str(item_phrase or ""))).lower()
    return t in _INVENTORY_HOLDER_ITEM_QUERY_PRONOUNS


def _object_literal_matches_normalized_item_key(obj: str, target_key: str) -> bool:
    """True iff ``obj`` designates the same countable or bare noun key as ``target_key``."""
    if not target_key:
        return False
    s = normalize_possession_item_surface(_normalize_phrase(str(obj or ""))).strip()
    if not s:
        return False
    parsed = _parse_counted_item_literal(s)
    if parsed is not None:
        return _normalize_item_key(parsed[1]) == target_key
    return _normalize_item_key(s) == target_key


def _current_holders_matching_item_key(pef: PEFState, target_key: str) -> list[Entity]:
    """Subjects with at least one **active** HAS whose object resolves to ``target_key``."""
    subject_ids: set[str] = set()
    for rel in pef.get_relationships_by_relation("HAS"):
        subject_ids.add(rel.subject_id)
    holders: list[Entity] = []
    for sid in sorted(subject_ids, key=lambda x: x.lower()):
        ent = pef.entities.get(sid)
        if ent is None:
            continue
        has_active_match = False
        for rel in pef.get_relationships_for_subject(sid):
            if rel.relation != "HAS":
                continue
            obj = _format_inventory_object(pef, rel)
            if not obj or not _object_literal_matches_normalized_item_key(obj, target_key):
                continue
            obj_lookup = str(obj).strip().lower()
            state = _current_state_for_item(pef, sid, obj_lookup)
            if state is not None and state[0]:
                has_active_match = True
                break
        if has_active_match:
            holders.append(ent)
    holders.sort(key=lambda e: e.name.lower())
    return holders


def _distinct_active_inventory_surfaces(pef: PEFState, *, limit: int = 16) -> list[str]:
    """Human-readable possessed objects currently held by someone (best-effort ordering)."""
    seen: set[str] = set()
    out: list[str] = []
    for rel in pef.get_relationships_by_relation("HAS"):
        obj = _format_inventory_object(pef, rel)
        if not obj:
            continue
        obj_lookup = str(obj).strip().lower()
        st = _current_state_for_item(pef, rel.subject_id, obj_lookup)
        if st is None or not st[0]:
            continue
        display = normalize_possession_item_surface(obj)
        k = display.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(display)
        if len(out) >= limit:
            break
    out.sort(key=str.lower)
    return out


def evaluate_inventory_for_subject(
    pef: PEFState,
    subject_phrase: str,
) -> StateNativeDelegationResult:
    """Evaluate strict ``What does X have?`` over committed PEF state.

    Explicit epistemics (no ANSWER→VALUE inference for absence):
      * held items list → :class:`EpistemicResult`.VALUE
      * confirmed negated / transfer absence lines → :class:`EpistemicResult`.FALSE
      * no matching subject → :class:`EpistemicResult`.UNKNOWN (STOP)
      * subject known but no HAS evidence at all → :class:`EpistemicResult`.UNKNOWN (STOP)
      * multiple subject candidates → :class:`EpistemicResult`.AMBIGUOUS (CLARIFY)
    """
    entities = list(resolve_entity(subject_phrase, pef).entities)
    if len(entities) > 1:
        names = sorted(e.name for e in entities)
        generic = f"Which did you mean: {', '.join(names)}?"
        base_ctx = {
            "failed_constraint": "STATE_NATIVE_ENTITY_AMBIGUITY",
            "candidate_entities": names,
            "subject_phrase": subject_phrase.strip(),
        }
        text, ctx = augment_state_native_ambiguity_with_typo_recovery(
            pef,
            entities,
            ambiguity_kind="inventory_subject_query",
            generic_user_visible_text=generic,
            clarify_context=base_ctx,
        )
        return _inventory_delegation(
            StateNativeOutcome.CLARIFY,
            text,
            epistemic=EpistemicResult.AMBIGUOUS,
            clarify_context=ctx,
        )

    if len(entities) == 0:
        return _inventory_delegation(
            StateNativeOutcome.STOP,
            "I cannot answer that from committed state: no matching entity for "
            f"{subject_phrase.strip()}.",
            epistemic=EpistemicResult.UNKNOWN,
            stop_reason_code="state_native_unknown_entity",
        )

    subject = entities[0]
    inventory = _subject_inventory_objects(pef, subject.id)
    if inventory:
        return _inventory_delegation(
            StateNativeOutcome.ANSWER,
            f"{subject.name} has {_join_items(inventory)}.",
            epistemic=EpistemicResult.VALUE,
        )

    # Entity known but positive HAS is empty — check for NOT_HAS (transfer)
    not_has_items = _subject_not_has_objects(pef, subject.id)
    if not not_has_items:
        return _inventory_delegation(
            StateNativeOutcome.STOP,
            f"I don't have any information about {subject.name}'s holdings from committed state.",
            epistemic=EpistemicResult.UNKNOWN,
            stop_reason_code="state_native_no_inventory",
        )

    lines: list[str] = []
    for item in not_has_items:
        holder = _find_current_holder_of_item(pef, item, subject.id)
        art = _with_article(item)
        if holder is not None:
            lines.append(f"{subject.name} does not have {art}. {holder.name} has it.")
        else:
            lines.append(f"{subject.name} does not have {art}.")
    return _inventory_delegation(
        StateNativeOutcome.ANSWER,
        " ".join(lines),
        epistemic=EpistemicResult.FALSE,
    )


def _matching_holders_for_item(pef: PEFState, item_phrase: str) -> list[Entity]:
    target_key = _normalize_item_key(item_phrase)
    if not target_key:
        return []
    return _current_holders_matching_item_key(pef, target_key)


def supersede_same_subject_prior_holds_for_item(
    pef: PEFState,
    subject_id: str,
    item_phrase: str,
    *,
    evidence: str,
    bookkeeping_backdate_turn: bool = False,
) -> None:
    """Negate this subject's active HAS rows for the same normalized item key only.

    Item categories (e.g. ``apple``) are not globally unique in PEF: multiple
    entities may each have an instance. Supersession applies only within
    ``(subject_id, normalized_item_key)`` so a new positive HAS for the same
    subject replaces an earlier hold of that category without touching other
    holders. Counted fungible literals (handled separately in ``update_pef``)
    skip this path.

    **Turn timestamps:** Semantic negations (real-world events) must use
    ``pef.current_turn`` — leave ``bookkeeping_backdate_turn=False`` (default).
    Only **bookkeeping** supersession (same ``update_pef`` pass closing a prior
    hold before appending the replacement positive HAS) may set
    ``bookkeeping_backdate_turn=True`` so ``_current_state_for_item``’s same-turn
    NOT_HAS tie-break does not cancel the replacement row.

    Call immediately before appending the new positive HAS for
    ``(subject_id, item_phrase)``.
    """
    target_key = _normalize_item_key(item_phrase)
    if not target_key:
        return
    literals_superseded: set[str] = set()
    for rel in list(pef.get_relationships_for_subject(subject_id)):
        if rel.relation != "HAS":
            continue
        obj = _format_inventory_object(pef, rel)
        if not obj or not _object_literal_matches_normalized_item_key(obj, target_key):
            continue
        obj_low = obj.strip().lower()
        if obj_low in literals_superseded:
            continue
        state = _current_state_for_item(pef, subject_id, obj_low)
        if state is None or not state[0]:
            continue
        literals_superseded.add(obj_low)
        if bookkeeping_backdate_turn:
            neg_turn = max(0, int(pef.current_turn) - 1)
        else:
            neg_turn = int(pef.current_turn)
        pef.add_relationship(
            Relationship(
                subject_id=subject_id,
                relation="HAS",
                object_entity_id=None,
                object_literal=obj,
                span=pef.active_span,
                source_turn=neg_turn,
                evidence=evidence,
                negated=True,
                provenance="system",
                extractor_backend="rule",
            )
        )


def _quantity_display_for_single_plural_holder_query(
    pef: PEFState,
    holder: Entity,
    item_disp: str,
) -> str | None:
    """Prefer ``display_quantity_item`` when plural query tails match one counted HAS.

    Generic ``Who has Y?`` routing cannot use lexical quantity display whenever the query
    omits ``N``. For plural ``Y`` backed by exactly one counted active literal matching the
    same item key (e.g. ``glasses`` vs committed ``one glass``), emit ``N`` + lexical tail
    instead of ``the <raw query noun>``.
    """
    trimmed = normalize_possession_item_surface((item_disp or "").strip())
    if not trimmed:
        return None
    lex = analyze_item_phrase(trimmed)
    if lex.number != "plural":
        return None

    target_key = _normalize_item_key(trimmed)
    if not target_key:
        return None

    latest_by_literal: dict[str, tuple[int, bool, int]] = {}
    for rel in pef.get_relationships_for_subject(holder.id):
        if rel.relation != "HAS":
            continue
        obj = _format_inventory_object(pef, rel)
        if not obj:
            continue
        parsed = _parse_counted_item_literal(str(obj))
        if parsed is None:
            continue
        count, item_text = parsed
        if _normalize_item_key(item_text) != target_key:
            continue
        lit_norm = normalize_possession_item_surface(obj).lower()
        prior = latest_by_literal.get(lit_norm)
        if prior is None or rel.source_turn > prior[0] or (
            rel.source_turn == prior[0] and rel.negated and not prior[1]
        ):
            latest_by_literal[lit_norm] = (rel.source_turn, rel.negated, count)

    active_qty: set[int] = set()
    for turn, negated, count in latest_by_literal.values():
        if not negated:
            active_qty.add(count)
    if len(active_qty) != 1:
        return None
    qty = active_qty.pop()
    return display_quantity_item(qty, trimmed)


def _holders_matching_exact_quantity_item(
    pef: PEFState, qty: int, item_tail: str,
) -> list[Entity]:
    """Subjects with an active counted ``HAS`` matching ``qty`` and item key from ``item_tail``."""
    target_key = _normalize_item_key(item_tail)
    if not target_key:
        return []
    subject_ids: set[str] = set()
    for rel in pef.get_relationships_by_relation("HAS"):
        subject_ids.add(rel.subject_id)
    holders: list[Entity] = []
    for sid in sorted(subject_ids, key=str.lower):
        ent = pef.entities.get(sid)
        if ent is None:
            continue
        for rel in pef.get_relationships_for_subject(sid):
            if rel.relation != "HAS":
                continue
            obj = _format_inventory_object(pef, rel)
            if not obj:
                continue
            parsed = _parse_counted_item_literal(str(obj))
            if parsed is None:
                continue
            count, item_text = parsed
            if count != qty:
                continue
            if _normalize_item_key(item_text) != target_key:
                continue
            obj_lookup = str(obj).strip().lower()
            state = _current_state_for_item(pef, sid, obj_lookup)
            if state is not None and state[0]:
                holders.append(ent)
                break
    holders.sort(key=lambda e: e.name.lower())
    return holders



def _evaluate_quantity_holder_query(
    pef: PEFState,
    qty: int,
    item_tail: str,
) -> tuple[str, str, dict | None, str | None]:
    """Answer *Who has N items?* from counted ``HAS`` literals (quantity + item class)."""
    item_tail = normalize_possession_item_surface(_normalize_phrase(item_tail))
    if not item_tail:
        return (
            StateNativeOutcome.STOP.value,
            "I cannot answer that from committed state: no item was provided.",
            None,
            "state_native_no_holder",
        )
    display_obj = display_quantity_item(qty, item_tail)
    holders = _holders_matching_exact_quantity_item(pef, qty, item_tail)
    if len(holders) == 0:
        return (
            StateNativeOutcome.STOP.value,
            "I cannot answer that from committed state: no committed holder for "
            f"{display_obj}.",
            None,
            "state_native_no_holder",
        )
    holder_names = sorted(h.name for h in holders)
    if len(holders) == 1:
        h0 = holders[0]
        return (
            StateNativeOutcome.ANSWER.value,
            f"{h0.name} has {display_obj}.",
            None,
            None,
        )
    return (
        StateNativeOutcome.ANSWER.value,
        f"{_join_holder_names(holder_names)} have {display_obj}.",
        None,
        None,
    )


def evaluate_inventory_holder_query(
    pef: PEFState,
    item_phrase: str,
    *,
    quantity_match: tuple[int, str] | None = None,
) -> tuple[str, str, dict | None, str | None]:
    """Evaluate strict ``Who has Y?`` over committed PEF state."""
    if quantity_match is None:
        trimmed = (item_phrase or "").strip()
        if trimmed:
            qm_syn = parse_inventory_quantity_holder_phrase(f"Who has {trimmed}")
            if qm_syn is not None:
                quantity_match = qm_syn

    if quantity_match is not None:
        qty, item_tail = quantity_match
        return _evaluate_quantity_holder_query(pef, qty, item_tail)

    if _holder_query_item_is_unresolved_pronoun(item_phrase):
        item_disp = _normalize_phrase(item_phrase) or item_phrase.strip()
        surfaces = _distinct_active_inventory_surfaces(pef)
        if surfaces:
            generic = (
                f'Specify which possessed item "{item_disp}" refers to '
                f"(committed holdings include: {', '.join(surfaces)})."
            )
        else:
            generic = (
                f'Specify which item "{item_disp}" refers to; pronouns alone are not '
                "a bounded inventory query here."
            )
        base_ctx = {
            "failed_constraint": "STATE_NATIVE_ENTITY_AMBIGUITY",
            "candidate_entities": surfaces,
            "item_phrase": item_disp,
        }
        text, ctx = augment_state_native_ambiguity_with_typo_recovery(
            pef,
            [],
            ambiguity_kind="inventory_holder_query_unresolved_item",
            generic_user_visible_text=generic,
            clarify_context=base_ctx,
        )
        return (
            StateNativeOutcome.CLARIFY.value,
            text,
            ctx,
            None,
        )

    holders = _matching_holders_for_item(pef, item_phrase)
    if len(holders) == 0:
        item = _normalize_phrase(item_phrase) or item_phrase.strip()
        return (
            StateNativeOutcome.STOP.value,
            "I cannot answer that from committed state: no committed holder for "
            f"{item}.",
            None,
            "state_native_no_holder",
        )

    holder_names = sorted(h.name for h in holders)
    item_disp = _normalize_phrase(item_phrase) or item_phrase.strip()

    if len(holders) == 1:
        holder = holders[0]
        qty_disp = _quantity_display_for_single_plural_holder_query(pef, holder, item_disp)
        tail_disp = qty_disp if qty_disp is not None else _with_article(item_disp)
        return (
            StateNativeOutcome.ANSWER.value,
            f"{holder.name} has {tail_disp}.",
            None,
            None,
        )

    return (
        StateNativeOutcome.ANSWER.value,
        f"{_join_holder_names(holder_names)} have {_inventory_group_item_display(item_disp)}.",
        None,
        None,
    )


def evaluate_inventory_how_many_query(
    pef: PEFState,
    subject_phrase: str,
    item_phrase: str,
    past_tense: bool = False,
) -> tuple[str, str, dict | None, str | None]:
    """Evaluate strict quantity over committed **HAS** / **CONTAIN** counted arcs."""
    entities = list(resolve_entity(subject_phrase, pef).entities)
    if len(entities) > 1:
        names = sorted(e.name for e in entities)
        generic = f"Which did you mean: {', '.join(names)}?"
        base_ctx = {
            "failed_constraint": "STATE_NATIVE_ENTITY_AMBIGUITY",
            "candidate_entities": names,
            "subject_phrase": subject_phrase.strip(),
            "item_phrase": item_phrase.strip(),
        }
        text, ctx = augment_state_native_ambiguity_with_typo_recovery(
            pef,
            entities,
            ambiguity_kind="inventory_quantity_query",
            generic_user_visible_text=generic,
            clarify_context=base_ctx,
        )
        return (
            StateNativeOutcome.CLARIFY.value,
            text,
            ctx,
            None,
        )

    if len(entities) == 0:
        return (
            StateNativeOutcome.STOP.value,
            "I cannot answer that from committed state: no matching entity for "
            f"{subject_phrase.strip()}.",
            None,
            "state_native_unknown_entity",
        )

    subject = entities[0]
    target_key = _normalize_item_key(item_phrase)
    if not target_key:
        return (
            StateNativeOutcome.STOP.value,
            "I cannot answer that from committed state: no item was provided.",
            None,
            "state_native_no_inventory",
        )

    latest_by_literal: dict[str, tuple[int, bool, int, str]] = {}
    for rel in pef.get_relationships_for_subject(subject.id):
        if rel.relation not in _COUNTED_INVENTORY_RELATIONS:
            continue
        obj = _format_inventory_object(pef, rel)
        parsed = _parse_counted_item_literal(obj)
        if parsed is None:
            continue
        count, item_text = parsed
        if _normalize_item_key(item_text) != target_key:
            continue
        literal_key = normalize_possession_item_surface(obj).lower()
        prior = latest_by_literal.get(literal_key)
        if prior is None:
            latest_by_literal[literal_key] = (rel.source_turn, rel.negated, count, item_text)
            continue
        prior_turn, prior_negated, _, _ = prior
        if rel.source_turn > prior_turn or (
            rel.source_turn == prior_turn and rel.negated and not prior_negated
        ):
            latest_by_literal[literal_key] = (rel.source_turn, rel.negated, count, item_text)

    active = [v for v in latest_by_literal.values() if not v[1]]
    if not active:
        return (
            StateNativeOutcome.STOP.value,
            "I cannot answer that from committed state: no committed quantity for "
            f"{item_phrase.strip()} for {subject.name}.",
            None,
            "state_native_no_inventory",
        )

    best_turn = max(v[0] for v in active)
    best_candidates = [v for v in active if v[0] == best_turn]
    # If several active same-turn quantities exist, prefer the largest count
    # as the most conservative current-state projection.
    _, _, best_count, best_item_text = max(best_candidates, key=lambda v: v[2])

    verb = "had" if past_tense else "has"
    return (
        StateNativeOutcome.ANSWER.value,
        f"{subject.name} {verb} {best_count} {best_item_text}.",
        None,
        None,
    )


def evaluate_inventory_subject_has_item_query(
    pef: PEFState,
    subject_phrase: str,
    item_phrase: str,
) -> StateNativeDelegationResult | None:
    """Evaluate strict ``Does X have Y`` over committed PEF state.

    Returns explicit :class:`EpistemicResult` values (never relying on inferred
    ANSWER semantics): TRUE / FALSE for Boolean possession, UNKNOWN when there
    is no evidence, AMBIGUOUS when entity resolution yields multiple subjects.
    """
    entities = list(resolve_entity(subject_phrase, pef).entities)
    if len(entities) > 1:
        names = sorted(e.name for e in entities)
        generic = f"Which did you mean: {', '.join(names)}?"
        base_ctx = {
            "failed_constraint": "STATE_NATIVE_ENTITY_AMBIGUITY",
            "candidate_entities": names,
            "subject_phrase": subject_phrase.strip(),
            "item_phrase": item_phrase.strip(),
        }
        text, ctx = augment_state_native_ambiguity_with_typo_recovery(
            pef,
            entities,
            ambiguity_kind="inventory_subject_has_item_query",
            generic_user_visible_text=generic,
            clarify_context=base_ctx,
        )
        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.CLARIFY,
            user_visible_text=text,
            clarify_context=ctx,
            solver_family=_INVENTORY_READ_SOLVER,
            epistemic_result=EpistemicResult.AMBIGUOUS,
        )

    if len(entities) == 0:
        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.STOP,
            user_visible_text=(
                "I cannot answer that from committed state: no matching entity for "
                f"{subject_phrase.strip()}."
            ),
            stop_reason_code="state_native_unknown_entity",
            solver_family=_INVENTORY_READ_SOLVER,
            epistemic_result=EpistemicResult.UNKNOWN,
        )

    subject = entities[0]
    raw_item = item_phrase.strip()
    pq = _parse_counted_item_literal(raw_item)
    expected_count: int | None = None
    match_phrase = raw_item
    if pq is not None:
        expected_count, bare_item = pq
        bk = _normalize_item_key(bare_item)
        if bk:
            target_key = bk
        else:
            target_key = _normalize_item_key(raw_item)
    else:
        target_key = _normalize_item_key(raw_item)
    if not target_key:
        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.STOP,
            user_visible_text=(
                "I cannot answer that from committed state: no item was provided."
            ),
            stop_reason_code="state_native_no_inventory",
            solver_family=_INVENTORY_READ_SOLVER,
            epistemic_result=EpistemicResult.UNKNOWN,
        )

    latest_by_literal: dict[str, tuple[int, bool, int, str]] = {}
    for rel in pef.get_relationships_for_subject(subject.id):
        if rel.relation not in _COUNTED_INVENTORY_RELATIONS:
            continue
        obj = _format_inventory_object(pef, rel)
        parsed = _parse_counted_item_literal(obj)
        if parsed is not None:
            count, item_text = parsed
            if _normalize_item_key(item_text) != target_key:
                continue
            lit = obj.lower().strip()
            prev = latest_by_literal.get(lit)
            if prev is None or rel.source_turn > prev[0] or (
                rel.source_turn == prev[0] and rel.negated and not prev[1]
            ):
                latest_by_literal[lit] = (rel.source_turn, rel.negated, count, item_text)
            continue

        item_text = _normalize_phrase(obj)
        if _normalize_item_key(item_text) != target_key:
            continue
        lit = obj.lower().strip()
        prev = latest_by_literal.get(lit)
        if prev is None or rel.source_turn > prev[0] or (
            rel.source_turn == prev[0] and rel.negated and not prev[1]
        ):
            latest_by_literal[lit] = (rel.source_turn, rel.negated, 1, item_text)

    if not latest_by_literal:
        # UNKNOWN: no committed HAS record for this item exists at all.
        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.STOP,
            user_visible_text=(
                f"I don't have any information about {subject.name} having "
                f"{match_phrase}."
            ),
            stop_reason_code="state_native_no_inventory",
            solver_family=_INVENTORY_READ_SOLVER,
            epistemic_result=EpistemicResult.UNKNOWN,
        )

    active = [v for v in latest_by_literal.values() if not v[1]]
    if not active:
        # FALSE: a HAS record exists but the latest committed state is negated
        # (item explicitly absent).  Return a definitive negative answer.
        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.ANSWER,
            user_visible_text=(
                f"No, {subject.name} does not have {match_phrase}."
            ),
            solver_family=_INVENTORY_READ_SOLVER,
            epistemic_result=EpistemicResult.FALSE,
        )

    best_turn = max(v[0] for v in active)
    best_candidates = [v for v in active if v[0] == best_turn]
    _, _, count, item_text = max(best_candidates, key=lambda v: v[2])
    if expected_count is not None and count != expected_count:
        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.ANSWER,
            user_visible_text=f"No, {subject.name} does not have {match_phrase}.",
            solver_family=_INVENTORY_READ_SOLVER,
            epistemic_result=EpistemicResult.FALSE,
        )
    if count <= 1 and not _parse_counted_item_literal(f"{count} {item_text}"):
        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.ANSWER,
            user_visible_text=f"{subject.name} has {_with_article(item_text)}.",
            solver_family=_INVENTORY_READ_SOLVER,
            epistemic_result=EpistemicResult.TRUE,
        )
    return StateNativeDelegationResult(
        handled=True,
        outcome=StateNativeOutcome.ANSWER,
        user_visible_text=f"{subject.name} has {count} {item_text}.",
        solver_family=_INVENTORY_READ_SOLVER,
        epistemic_result=EpistemicResult.TRUE,
    )
