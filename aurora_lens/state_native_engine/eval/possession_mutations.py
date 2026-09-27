"""Generic quantitative possession mutations (GIVE, PUT, CONSUME, EAT).

These mutations update active possession state by modifying counted **HAS**
and **CONTAIN** arcs on PEF entities, including quantities held **inside** a
container the actor possesses (nested stock). No domain‑specific protagonists.

``CONTAIN`` rows (extractor‑native for *contains …*) participate in quantity
queries and outbound **GIVE** / **PUT**‑into bookkeeping alongside **HAS**.

Item phrases use the same surface normalization as inventory quantity queries and
HAS literals (``normalize_possession_item_surface`` in ``inventory``): broad
regex capture, then ``[.?!,;:]+$`` stripping rather than over-narrow patterns.

Counterparties (giver/recipient/subject of consume) bind to committed entities when
present; otherwise a single-token ``get_or_create_entity`` path runs so transfers
still apply on the **pre-extraction** pre-model pass before ``Jill``/``Henry`` exist
only as mentions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import CONSUMPTION_RELATIONS, PEFState, Relationship
from aurora_lens.state_native_engine.bind import resolve_entity
from aurora_lens.state_native_engine.contracts import (
    PossessionMutationAction,
    PossessionMutationConfidence,
    PossessionMutationFrame,
    StateNativeDelegationResult,
    StateNativeOutcome,
    StateNativeRequest,
    StateNativeSolverFamily,
)
from aurora_lens.state_native_engine.eval.possession_mutation_frame_build import (
    PossessionFrameBuildAbstained,
    build_possession_mutation_frame,
)
from aurora_lens.state_native_engine.eval.inventory import (
    _format_inventory_object,
    _parse_counted_item_literal,
    normalize_possession_item_surface,
)
from aurora_lens.state_native_engine.lexical import item_key

_POSSESSION_TRANSFER_LOG = logging.getLogger(__name__)


def _log_possession_transfer_applied(frame: PossessionMutationFrame, **audit: object) -> None:
    """Structured INFO row for operator replay (parse basis + quantity deltas)."""
    if not _POSSESSION_TRANSFER_LOG.isEnabledFor(logging.INFO):
        return
    row: dict[str, object] = {
        "parse_source": frame.parse_source,
        "confidence": frame.confidence.value,
        "action": frame.action.value,
        "voice": frame.voice,
        "surface_verb": frame.surface_verb,
        "quantity": frame.quantity,
        "item_surface": frame.item_surface,
    }
    row.update({k: v for k, v in audit.items() if v is not None})
    _POSSESSION_TRANSFER_LOG.info("possession_transfer_applied %s", row)


_COUNTED_HOLD_RELATIONS: frozenset[str] = frozenset({"HAS", "CONTAIN"})

# Item tails containing third-person pronouns must not be interpreted by the regex-only
# pre-model transfer path — extractor ambiguity / quantified pronoun expansion owns binding.
_TRANSFER_ITEM_TAIL_DEFERRAL_PRONOUN = re.compile(r"(?is)\b(of\s+)?(them|it)\b")

# First-token pronouns whose reference must be settled before consequence-bearing transfer.
_AMBIGUITY_SENSITIVE_GIVER_PRONOUNS: frozenset[str] = frozenset(
    {"he", "him", "she", "her", "they", "we", "us", "it"},
)

# Subject/object pronouns that may denote the giver or recipient against committed discourse.
_PARTY_PRONOUNS: frozenset[str] = frozenset(
    {"he", "him", "she", "her", "they", "them", "we", "us", "i", "me", "you"},
)


def _most_recent_active_resolved_entity(pef: PEFState) -> Entity | None:
    """Prefer recency heuristic for unresolved pronouns (matches spaCy/discourse stubs)."""
    resolved = [e for e in pef.entities.values() if e.resolved]
    if not resolved:
        return None
    return sorted(
        resolved,
        key=lambda e: e.turn_last_active,
        reverse=True,
    )[0]


def _first_surface_token_lower(phrase: str) -> str:
    cleaned = normalize_possession_item_surface((phrase or "").strip())
    if not cleaned:
        return ""
    tok = cleaned.split()[0].lower().strip("\"'.,!?;:()[]{}")
    return tok


def _transfer_item_tail_requires_extractor_binding(item_tail: str) -> bool:
    """True when pronouns in the captured item span must go through NLP/pronoun gates."""
    s = normalize_possession_item_surface((item_tail or "").strip())
    if not s:
        return False
    return _TRANSFER_ITEM_TAIL_DEFERRAL_PRONOUN.search(s) is not None


def _giver_first_token_requires_holder_disambiguation(giver_raw: str) -> bool:
    tok = _first_surface_token_lower(giver_raw)
    return tok in _AMBIGUITY_SENSITIVE_GIVER_PRONOUNS


def _active_holder_count_for_item_key(pef: PEFState, item_match_key: str) -> int:
    """How many entities can source ``item_match_key`` (direct HAS or nested stock)."""
    if not item_match_key:
        return 0
    n = 0
    for ent in pef.entities.values():
        if resolve_holder_counted_stock(pef, ent, item_match_key) is not None:
            n += 1
    return n


@dataclass(frozen=True)
class CountedStock:
    """Affirmative counted row (HAS/CONTAIN); ``source_relationship`` is superseded on write."""

    subject_id: str
    relation: str  # HAS | CONTAIN
    quantity: float
    source_relationship: Relationship  # superseded literal for negated replay
    item_display: str


def _find_active_counted_holding(
    pef: PEFState,
    subject_id: str,
    item_match_key: str,
    *,
    before_turn: int | None = None,
    allowed_relations: frozenset[str] = _COUNTED_HOLD_RELATIONS,
) -> tuple[float, Relationship, str] | None:
    """Latest affirmative counted arc on ``subject_id`` whose item key matches."""

    best_rel = None
    best_turn = -1
    best_qty = 0.0
    best_display = ""

    for rel in pef.get_relationships_for_subject(subject_id):
        if rel.relation not in allowed_relations:
            continue
        if rel.negated:
            continue
        if before_turn is not None and rel.source_turn >= before_turn:
            continue
        obj_raw = _format_inventory_object(pef, rel)
        if not obj_raw:
            continue
        obj = normalize_possession_item_surface(str(obj_raw).strip())
        if not obj:
            continue

        qty_found: float | None = None
        item_disp_tail = ""

        m_num = re.match(r"^(\d+(?:\.\d+)?)\s+(.+)$", obj)
        if m_num:
            item_name = normalize_possession_item_surface(m_num.group(2))
            if item_name and item_key(item_name) == item_match_key:
                try:
                    qty_found = float(m_num.group(1))
                    item_disp_tail = item_name
                except ValueError:
                    qty_found = None
        if qty_found is None:
            wd = _parse_counted_item_literal(obj)
            if wd is not None:
                wi, wt = wd
                item_name = normalize_possession_item_surface(wt)
                if item_name and item_key(item_name) == item_match_key:
                    qty_found = float(wi)
                    item_disp_tail = item_name
        if qty_found is not None:
            if rel.source_turn > best_turn:
                best_rel = rel
                best_turn = rel.source_turn
                best_qty = qty_found
                best_display = item_disp_tail or item_match_key
        else:
            if item_key(obj) != item_match_key:
                continue
            if rel.source_turn > best_turn:
                best_rel = rel
                best_turn = rel.source_turn
                best_qty = 1.0
                best_display = obj

    if best_rel is None:
        return None
    return (best_qty, best_rel, best_display)


def _has_inner_entity(pef: PEFState, rel: Relationship) -> Entity | None:
    """Entity held via **HAS**: ``object_entity_id`` or lone-resolvable ``object_literal``."""
    if rel.relation != "HAS" or rel.negated:
        return None
    if rel.object_entity_id is not None:
        return pef.entities.get(rel.object_entity_id)
    lit = str(rel.object_literal or "").strip()
    if not lit:
        return None
    br = resolve_entity(lit, pef)
    if len(br.entities) == 1:
        return br.entities[0]
    return None


def resolve_holder_counted_stock(
    pef: PEFState,
    holder: Entity,
    item_match_key: str,
) -> CountedStock | None:
    """Resolve fungible quantity the holder can surrender (hands or nested container).

    If the holder has both a matching direct HAS/CONTAIN and nested stock, prefer
    the **direct** row. Nested stock is consulted only when there is exactly one
    contained source; multiple competing containers ⇒ ``None`` (defer).
    """
    direct = _find_active_counted_holding(pef, holder.id, item_match_key)
    if direct is not None:
        qty, rel, disp = direct
        return CountedStock(
            subject_id=holder.id,
            relation=str(rel.relation),
            quantity=qty,
            source_relationship=rel,
            item_display=disp,
        )

    nested: list[CountedStock] = []
    for rel in pef.get_relationships_for_subject(holder.id):
        inner = _has_inner_entity(pef, rel)
        if inner is None:
            continue
        inner_hit = _find_active_counted_holding(pef, inner.id, item_match_key)
        if inner_hit is None:
            continue
        qty, irel, idisp = inner_hit
        nested.append(
            CountedStock(
                subject_id=inner.id,
                relation=str(irel.relation),
                quantity=qty,
                source_relationship=irel,
                item_display=idisp,
            )
        )

    if len(nested) == 1:
        return nested[0]
    return None


def subject_prefers_contain_relation(pef: PEFState, subject_id: str) -> bool:
    """Prefer **CONTAIN** when subject already has an affirmative CONTAIN arc."""
    for rel in pef.get_relationships_for_subject(subject_id):
        if rel.relation == "CONTAIN" and not rel.negated:
            return True
    return False


def apply_counted_write(
    req: StateNativeRequest,
    stock: CountedStock,
    new_qty: float,
) -> None:
    """Negate the prevailing affirmative literal, optionally write new quantity."""
    subj_ent = req.pef.entities.get(stock.subject_id)
    if subj_ent is None:
        return

    literal_to_negate = str(stock.source_relationship.object_literal or "").strip()
    req.pef.add_relationship(
        Relationship(
            subject_id=stock.subject_id,
            relation=stock.relation,
            object_entity_id=None,
            object_literal=literal_to_negate,
            span=Span.PRESENT,
            source_turn=req.pef.current_turn,
            evidence="",
            negated=True,
            provenance="system",
            extractor_backend="possession_mutation",
        )
    )
    if new_qty > 0:
        qty_s = int(new_qty) if float(new_qty) == int(new_qty) else new_qty
        tail = normalize_possession_item_surface(stock.item_display)
        new_literal = f"{qty_s} {tail}"
        req.pef.add_relationship(
            Relationship(
                subject_id=stock.subject_id,
                relation=stock.relation,
                object_entity_id=None,
                object_literal=new_literal,
                span=Span.PRESENT,
                source_turn=req.pef.current_turn,
                evidence="",
                negated=False,
                provenance="system",
                extractor_backend="possession_mutation",
            )
        )


def bump_or_create_counted(
    req: StateNativeRequest,
    dest: Entity,
    item_match_key: str,
    item_display: str,
    add_qty: float,
) -> None:
    """Add ``add_qty`` to an existing counted row on ``dest`` or mint *de novo* arc."""
    if add_qty <= 0:
        return
    tail = normalize_possession_item_surface(item_display)

    existing = _find_active_counted_holding(req.pef, dest.id, item_match_key)
    if existing is None:
        rel_kind = (
            "CONTAIN" if subject_prefers_contain_relation(req.pef, dest.id) else "HAS"
        )
        qty_s = int(add_qty) if float(add_qty) == int(add_qty) else add_qty
        lit = f"{qty_s} {tail}"
        req.pef.add_relationship(
            Relationship(
                subject_id=dest.id,
                relation=rel_kind,
                object_entity_id=None,
                object_literal=lit,
                span=Span.PRESENT,
                source_turn=req.pef.current_turn,
                evidence="",
                negated=False,
                provenance="system",
                extractor_backend="possession_mutation",
            )
        )
        return

    qty_old, prev_rel, _disp = existing
    new_tot = qty_old + add_qty
    stock = CountedStock(
        subject_id=dest.id,
        relation=str(prev_rel.relation),
        quantity=qty_old,
        source_relationship=prev_rel,
        item_display=tail,
    )
    apply_counted_write(req, stock, new_tot)


def _bind_party(pef: PEFState, phrase_raw: str) -> Entity | None:
    """RESOLVED / single newly created counterpart; ambiguous or malformed → ``None``."""
    cleaned = normalize_possession_item_surface((phrase_raw or "").strip())
    if not cleaned or "\n" in cleaned:
        return None
    if len(cleaned) > 160:
        return None
    low = cleaned.lower()

    discourse = (pef.discourse_referent_bindings or {}).get(low)
    if discourse:
        br_disc = resolve_entity(str(discourse).strip(), pef)
        if len(br_disc.entities) == 1:
            return br_disc.entities[0]

    br = resolve_entity(cleaned, pef)
    if len(br.entities) == 1:
        return br.entities[0]
    if len(br.entities) > 1:
        return None
    if low in _PARTY_PRONOUNS:
        guess = _most_recent_active_resolved_entity(pef)
        if guess is not None:
            return guess

    ent, _created = pef.get_or_create_entity(cleaned, resolved=False)
    return ent


def _item_match_key_and_display(item_phrase: str) -> tuple[str | None, str]:
    """``(normalize item key for lookup, display name without redundant leading qty)``.

    ``item_phrase`` should already be ``normalize_possession_item_surface``-cleaned.
    """
    raw = normalize_possession_item_surface((item_phrase or "").strip())
    if not raw:
        return None, ""
    m = re.match(r"^(\d+(?:\.\d+)?)\s+(.+)$", raw)
    if m:
        disp = normalize_possession_item_surface(m.group(2))
        if not disp:
            return None, ""
        return item_key(disp), disp
    return item_key(raw), raw


def _find_active_possession(
    pef: PEFState,
    entity: Entity,
    item_match_key: str,
    before_turn: int | None = None,
) -> tuple[float, Relationship] | None:
    """Latest affirmative **HAS-only** counted row on ``entity``."""

    hit = _find_active_counted_holding(
        pef,
        entity.id,
        item_match_key,
        before_turn=before_turn,
        allowed_relations=frozenset({"HAS"}),
    )
    if hit is None:
        return None
    qty, rel, _ = hit
    return qty, rel


def _delegation_stop(message: str) -> StateNativeDelegationResult:
    return StateNativeDelegationResult(
        handled=True,
        outcome=StateNativeOutcome.STOP,
        user_visible_text=message,
        solver_family=StateNativeSolverFamily.COMMITTED_INVENTORY_READ,
        source_status="admitted_uncontaminated",
    )


def _apply_possession_give_frame(
    req: StateNativeRequest,
    frame: PossessionMutationFrame,
) -> StateNativeDelegationResult | None:
    """Counted **GIVE** transition from a built frame (no regex surfaces here)."""

    if frame.action != PossessionMutationAction.GIVE:
        return None
    recipient_surface = frame.recipient_surface
    if not recipient_surface:
        return None

    item_tail_preview = normalize_possession_item_surface(frame.item_surface)
    # Lexical deferral only — narrows pronoun-bearing tails so ambiguity containment runs
    # before pre-model HAS arithmetic (regex surface cannot duplicate extractor binding).
    if _transfer_item_tail_requires_extractor_binding(item_tail_preview):
        return None

    item_match_early, _ = _item_match_key_and_display(item_tail_preview)
    if (
        item_match_early
        and _giver_first_token_requires_holder_disambiguation(frame.actor_surface)
        and _active_holder_count_for_item_key(req.pef, item_match_early) != 1
    ):
        return None

    giver = _bind_party(req.pef, frame.actor_surface)
    recipient = _bind_party(req.pef, recipient_surface)
    if giver is None or recipient is None:
        return None

    qty = frame.quantity
    qty_str = str(int(qty)) if qty == int(qty) else str(qty)
    item_phrase = item_tail_preview

    item_match_key, item_display = _item_match_key_and_display(item_phrase)
    if not item_match_key or not item_display:
        return None

    giver_stock = resolve_holder_counted_stock(req.pef, giver, item_match_key)
    if giver_stock is None:
        return _delegation_stop(
            f"I cannot answer that from committed state: {giver.name} has no "
            f"{item_display}.",
        )
    giver_qty = float(giver_stock.quantity)

    if giver_qty < qty:
        return _delegation_stop(
            f"I cannot answer that from committed state: {giver.name} does not have "
            f"{qty_str} {item_display}.",
        )

    new_giver_qty = float(giver_qty - qty)
    apply_counted_write(req, giver_stock, new_giver_qty)

    bump_or_create_counted(req, recipient, item_match_key, item_display, qty)

    _log_possession_transfer_applied(
        frame,
        giver_entity_id=giver.id,
        recipient_entity_id=recipient.id,
        source_holder_qty_before=giver_qty,
        source_holder_qty_after=new_giver_qty,
        destination_qty_added=qty,
        item_match_key=item_match_key,
    )

    return StateNativeDelegationResult(
        handled=True,
        outcome=StateNativeOutcome.ANSWER,
        user_visible_text="Recorded transfer.",
        solver_family=StateNativeSolverFamily.COMMITTED_INVENTORY_READ,
        source_status="admitted_uncontaminated",
    )


def _apply_possession_put_frame(
    req: StateNativeRequest,
    frame: PossessionMutationFrame,
) -> StateNativeDelegationResult | None:
    """Counted **PUT … into** transition from a built frame (no regex surfaces here)."""

    if frame.action not in (
        PossessionMutationAction.PUT,
        PossessionMutationAction.RETURN,
    ):
        return None
    dest_surface = frame.destination_surface
    if not dest_surface:
        return None

    item_display = normalize_possession_item_surface(frame.item_surface)

    qty = frame.quantity
    item_match_key = item_key(item_display)
    if not item_match_key or not item_display:
        return None

    actor = _bind_party(req.pef, frame.actor_surface)
    dest = _bind_party(req.pef, dest_surface)
    if actor is None or dest is None:
        return None

    if (
        _giver_first_token_requires_holder_disambiguation(frame.actor_surface)
        and _active_holder_count_for_item_key(req.pef, item_match_key) != 1
    ):
        return None

    actor_stock = resolve_holder_counted_stock(req.pef, actor, item_match_key)
    if actor_stock is None:
        return _delegation_stop(
            f"I cannot answer that from committed state: {actor.name} has no "
            f"{item_display}.",
        )
    if float(actor_stock.quantity) < qty:
        return _delegation_stop(
            f"I cannot answer that from committed state: {actor.name} does not have "
            f"{int(qty) if qty == int(qty) else qty} {item_display}.",
        )

    new_actor_qty = float(actor_stock.quantity) - qty
    apply_counted_write(req, actor_stock, new_actor_qty)

    bump_or_create_counted(req, dest, item_match_key, item_display, qty)

    _log_possession_transfer_applied(
        frame,
        actor_entity_id=actor.id,
        destination_entity_id=dest.id,
        source_holder_qty_before=float(actor_stock.quantity),
        source_holder_qty_after=new_actor_qty,
        destination_qty_added=qty,
        item_match_key=item_match_key,
    )

    return StateNativeDelegationResult(
        handled=True,
        outcome=StateNativeOutcome.ANSWER,
        user_visible_text="Recorded transfer.",
        solver_family=StateNativeSolverFamily.COMMITTED_INVENTORY_READ,
        source_status="admitted_uncontaminated",
    )


def apply_possession_mutation_frame(
    req: StateNativeRequest,
    frame: PossessionMutationFrame,
) -> StateNativeDelegationResult | None:
    """Counted possession/container transition law — consumes frame only."""
    if frame.action == PossessionMutationAction.GIVE:
        return _apply_possession_give_frame(req, frame)
    if frame.action in (
        PossessionMutationAction.PUT,
        PossessionMutationAction.RETURN,
    ):
        return _apply_possession_put_frame(req, frame)
    return None


def evaluate_possession_transfer_mutations(
    req: StateNativeRequest,
) -> StateNativeDelegationResult | None:
    """Build frame via arbitration (spaCy HIGH → regex), then apply mutation law."""

    if req.binding_resumed:
        return None

    built = build_possession_mutation_frame(
        text=req.user_text,
        pef=req.pef,
        possession_nlp=req.possession_nlp,
        binding_resumed=False,
    )
    if isinstance(built, PossessionFrameBuildAbstained):
        return None
    if req.require_high_confidence_possession_transfer:
        if built.frame.confidence != PossessionMutationConfidence.HIGH:
            return None
    return apply_possession_mutation_frame(req, built.frame)


def evaluate_possession_consume_mutation(
    req: StateNativeRequest,
) -> StateNativeDelegationResult | None:
    """Evaluate consumption-class relations (EAT/CONSUME/DRINK/INGEST) via PEF dispatch.

    Spacy's EAT projection has already applied the arithmetic before this runs.
    This function detects the committed consumption relation, verifies pre-extraction
    inventory for graceful failure messaging, and returns an acknowledgement.
    """
    if req.binding_resumed:
        return None
    current_turn = req.pef.current_turn

    for rel in req.pef.relationships:
        if rel.relation not in CONSUMPTION_RELATIONS:
            continue
        if rel.source_turn != current_turn:
            continue
        if rel.negated:
            continue

        entity = req.pef.entities.get(rel.subject_id)
        if entity is None:
            continue

        obj_literal = str(rel.object_literal or "").strip()
        if not obj_literal:
            continue

        parsed = _parse_counted_item_literal(obj_literal)
        if parsed is None:
            continue

        qty, item_text = parsed
        item_match_key = item_key(item_text)
        if not item_match_key:
            continue

        prior = _find_active_possession(req.pef, entity, item_match_key, before_turn=current_turn)

        if prior is None:
            # No prior inventory baseline — admit the event as asserted fact without
            # attempting arithmetic.  Spacy wrote the EAT relation to PEF; we ACK it.
            return StateNativeDelegationResult(
                handled=True,
                outcome=StateNativeOutcome.ANSWER,
                user_visible_text="Recorded consumption.",
                solver_family=StateNativeSolverFamily.COMMITTED_INVENTORY_READ,
                source_status="admitted_uncontaminated",
            )

        prior_qty, _ = prior
        if prior_qty < qty:
            return _delegation_stop(
                f"I cannot answer that from committed state: {entity.name} does not have "
                f"{qty} {item_text}.",
            )

        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.ANSWER,
            user_visible_text="Recorded consumption.",
            solver_family=StateNativeSolverFamily.COMMITTED_INVENTORY_READ,
            source_status="admitted_uncontaminated",
        )

    return None
