"""PEFState: persistent present-state container with indexed relationships."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aurora_lens.interpret.schema import ContinuationTransaction

from .artifact_layer import ArtifactFrame
from .span import Span
from .entity import Entity, EchoTrace, TurnBindings
from .temporal_commitment import (
    PresentTemporalCommitment,
    temporal_commitments_from_wire,
    temporal_commitments_to_wire,
)
from .unresolved_referents import (
    UnresolvedReferentEntry,
    open_entries,
    registry_entries_from_wire,
    registry_entries_to_wire,
)


# ── Canonical relation mapping ───────────────────────────────────────

# Business-remit object markers and PEF write-time MANAGE normalization — see
# object_has_business_remit_marker() and interpret.pef_updater.


_RELATION_ALIASES: dict[str, str] = {
    "has": "HAS",
    "have": "HAS",
    "had": "HAS",
    "keep": "HAS",
    "keeps": "HAS",
    "kept": "HAS",
    "hold": "HAS",
    "holds": "HAS",
    "held": "HAS",
    "owns": "HAS",
    "own": "HAS",
    "manage": "MANAGE",
    "manages": "MANAGE",
    "managed": "MANAGE",
    "managing": "MANAGE",
    "possessed": "HAS",
    "possesses": "HAS",
    "be": "IS",    # spaCy lemma for is/was/are/were — must precede surface forms
    "is": "IS",
    "was": "IS",
    "are": "IS",
    "were": "IS",
    "seem": "IS",
    "seems": "IS",
    "seemed": "IS",
    "remain": "IS",
    "remains": "IS",
    "remained": "IS",
    "at": "AT",
    "in": "AT",
    "on": "AT",
    "located": "AT",
    "lives": "AT",
    "lived": "AT",
    "stay": "AT",
    "stays": "AT",
    "stayed": "AT",
    "reside": "AT",
    "resides": "AT",
    "resided": "AT",
    "gave": "GIVE",
    "gives": "GIVE",
    "give": "GIVE",
    "given": "GIVE",
    "hand": "GIVE",
    "hands": "GIVE",
    "handed": "GIVE",
    "pass": "GIVE",
    "passes": "GIVE",
    "passed": "GIVE",
    "take": "TAKE",
    "takes": "TAKE",
    "took": "TAKE",
    "taken": "TAKE",
    "grab": "TAKE",
    "grabs": "TAKE",
    "grabbed": "TAKE",
    "retrieve": "TAKE",
    "retrieves": "TAKE",
    "retrieved": "TAKE",
    "reclaim": "TAKE",
    "reclaims": "TAKE",
    "reclaimed": "TAKE",
    "sent": "SEND",
    "sends": "SEND",
    "send": "SEND",
    "told": "TELL",
    "tells": "TELL",
    "tell": "TELL",
    "say": "TELL",
    "says": "TELL",
    "said": "TELL",
    "explain": "TELL",
    "explains": "TELL",
    "explained": "TELL",
    "showed": "SHOW",
    "shows": "SHOW",
    "show": "SHOW",
    # REQUEST, PROMISE, WARN: speech-act relations — distinct from TELL (information transfer).
    # REQUEST = directive speech act (someone is instructed/asked to perform an action).
    # PROMISE = commitment speech act (subject binds themselves to a future action).
    # WARN    = advisory speech act (subject alerts another to a risk or danger).
    # All three are events, not static states — existence evaluator can match them.
    "ask":       "REQUEST",
    "asked":     "REQUEST",
    "asks":      "REQUEST",
    "request":   "REQUEST",
    "requested": "REQUEST",
    "requests":  "REQUEST",
    "promise":   "PROMISE",
    "promised":  "PROMISE",
    "promises":  "PROMISE",
    "warn":      "WARN",
    "warned":    "WARN",
    "warns":     "WARN",
    "returned": "RETURN",
    "returns": "RETURN",
    "return": "RETURN",
    # RETURN with prior-possession precondition — completed delivery back to original owner.
    # "brought back" / "bring back" are the key multi-word forms → RETURN.
    # Plain "brought" / "bring" / "brings" → GIVE (transfer/delivery consequence; see below).
    # Multi-word keys are resolved before single-word keys by dict lookup order.
    "brought back": "RETURN",
    "bring back":   "RETURN",
    # Plain brought/bring/brings: transfer/delivery — possession consequence same as GIVE.
    # "Alice brought Bob coffee" → Bob has coffee: this is GIVE semantics.
    # "brought back" above takes priority for the multi-word form (LLM backend).
    "brought": "GIVE",
    "bring":   "GIVE",
    "brings":  "GIVE",
    "because": "BECAUSE",
    "likes": "LIKES",
    "like": "LIKES",
    "liked": "LIKES",
    "prefer": "LIKES",
    "prefers": "LIKES",
    "preferred": "LIKES",
    "hate": "LIKES",
    "hates": "LIKES",
    "hated": "LIKES",
    "loves": "LOVES",
    "love": "LOVES",
    "loved": "LOVES",
    "wants": "WANTS",
    "want": "WANTS",
    "wanted": "WANTS",
    "knows": "KNOWS",
    "know": "KNOWS",
    "knew": "KNOWS",
    # BECOME: transition event — subject undergoes a role/state change.
    # Distinct from static IS (which records present state, not the change event).
    # "stayed as" is deferred — may preserve IS or need MAINTAINED_AS.
    # spaCy: "became" → lemma "become" hits the single-word key.
    # "turned into" is multi-word: resolved via LLM backend only (spaCy sees "into" as preposition).
    "become":     "BECOME",
    "became":     "BECOME",
    "turned into": "BECOME",
    # WEARING, WITH, CARRYING: temporary association states — distinct from HAS possession.
    # WEARING: body-associated temporary state (hat, coat, badge, armour).
    # WITH: accompaniment/proximity (Bob came with Alice; Alice arrived with the documents).
    # CARRYING: portable temporary custody (Alice is carrying the box; Bob carried the case).
    # spaCy: lemma "wear" covers wore/wears/wearing; lemma "carry" covers carries/carried/carrying.
    # "with" in spaCy is a preposition dep, not a verb lemma — LLM backend only.
    # "carrying around" is a multi-word LLM-backend form.
    # carry/carries/carried were previously HAS aliases — reclaimed here for correct semantics.
    "wear":            "WEARING",
    "wore":            "WEARING",
    "wearing":         "WEARING",
    "carry":           "CARRYING",
    "carries":         "CARRYING",
    "carried":         "CARRYING",
    "carrying around": "CARRYING",
    "with":            "WITH",
    # EPISTEMIC_CHANGE: epistemic state-change events (agent's knowledge state changes).
    # Distinct from static KNOWS (which records present knowledge, not the change event).
    "remembered": "EPISTEMIC_CHANGE",
    "forgot": "EPISTEMIC_CHANGE",
    "realized": "EPISTEMIC_CHANGE",
    "understood": "EPISTEMIC_CHANGE",
    "learned": "EPISTEMIC_CHANGE",
    "discovered": "EPISTEMIC_CHANGE",
    # MOTION destination (completed arrival only → current AT fact in present-bound semantics).
    # Only forms that unambiguously express completed arrival at a destination are safe here.
    # Trajectory / intention forms (heading to, left for) remain deferred.
    "went to":    "AT",
    "came to":    "AT",
    "arrived at": "AT",
    "arrived in": "AT",
    "moved to":   "AT",
    # CONTAINMENT → AT: being inside a location IS being at that location.
    # The containment nuance (inside vs at the entrance) is below PEF resolution.
    "inside":  "AT",
    "within":  "AT",
    "indoors": "AT",
    # NEAR: proximity state — distinct from AT (near the store ≠ at the store).
    # In _NON_ACTION_RELATIONS (static spatial state, not an event).
    # "close to" is multi-word: LLM backend only.
    "near":     "NEAR",
    "nearby":   "NEAR",
    "close to": "NEAR",
    # FROM: movement-origin — distinct from AT (where someone came from ≠ where they are now).
    # Only unambiguous multi-word movement-origin forms are aliased here.
    # Bare "from" is NOT aliased — it is heavily overloaded (source of TAKE, temporal, comparative).
    # In _NON_ACTION_RELATIONS (static provenance fact, not an event).
    "came from":     "FROM",
    "arrived from":  "FROM",
    "traveled from": "FROM",
    "coming from":   "FROM",
    # REMAIN: continuity/persistence assertion — subject's identity or role persisted across a
    # possible transition boundary. Distinct from BECOME (transition event) and IS (static snapshot).
    # "stayed as" is the primary surface form; "stay as" is the lemma/present form.
    # In _NON_ACTION_RELATIONS (state persistence, not a change event).
    "stayed as": "REMAIN",
    "stay as":   "REMAIN",
    # OUTSIDE: exterior/negated-containment spatial state. Ontologically distinct from NEAR (proximity)
    # and AT (co-location). Does not imply AT; does not imply NEAR. Records topology, not distance.
    # In _NON_ACTION_RELATIONS (static spatial state).
    "outside":    "OUTSIDE",
    "outside of": "OUTSIDE",
}

CANONICAL_RELATIONS = frozenset({
    "HAS", "IS", "AT", "GIVE", "TAKE", "SEND", "TELL", "SHOW",
    "RETURN", "BECAUSE", "LIKES", "LOVES", "WANTS", "KNOWS",
    "COMPARE",           # Binary comparative: subject IS adjective relative to comparand
    "EPISTEMIC_CHANGE",  # Epistemic state-change event (remembered, forgot, realized, …)
    "BECOME",            # Transition event: subject undergoes a role/state change
    "REQUEST",           # Directive speech act: subject asks/instructs another to act
    "PROMISE",           # Commitment speech act: subject binds to a future action
    "WARN",              # Advisory speech act: subject alerts another to risk/danger
    "WITH",              # Accompaniment / proximity — temporary, not possession
    "WEARING",           # Body-associated temporary state
    "CARRYING",          # Portable temporary custody
    "NEAR",              # Proximity state — close to but not at a location
    "FROM",              # Movement-origin / provenance — where someone came from
    "REMAIN",            # Continuity assertion: identity/role persisted across a transition boundary
    "OUTSIDE",           # Exterior/negated-containment spatial state
})

CONSUMPTION_RELATIONS: frozenset[str] = frozenset({"EAT", "CONSUME", "DRINK", "INGEST"})

# Epistemic holding (SPINE-aligned session mode) — serialized in PEFState.epistemic_hold
EPISTEMIC_HOLD_SCHEMA_VERSION = 1
EPISTEMIC_MODE_NONE = "none"
EPISTEMIC_MODE_AMBIGUITY = "ambiguity"
EPISTEMIC_MODE_REFUSAL = "refusal"
EPISTEMIC_MODE_STOP = "stop"

# Whole-word markers for business / portfolio remit (Rule A domain gate; PEF HAS→MANAGE rewrite).
# Match with word boundaries — not raw substring — see object_has_business_remit_marker().
BUSINESS_REMIT_MARKERS: frozenset[str] = frozenset({
    "portfolio", "account", "accounts", "region", "territory", "territories",
    "business", "unit", "division", "segment", "market", "markets",
    # Not "book": plain English "a red book" must not rewrite HAS→MANAGE; finance uses
    # portfolio/account/desk/market or explicit "order book" style phrasing via other markers.
    "client", "clients", "desk", "coverage",
})


def object_has_business_remit_marker(obj: str) -> bool:
    """True iff ``obj`` contains at least one BUSINESS_REMIT_MARKERS token as a whole word."""
    if not (obj or "").strip():
        return False
    text = obj.lower().strip()
    for marker in BUSINESS_REMIT_MARKERS:
        if re.search(r"\b" + re.escape(marker) + r"\b", text):
            return True
    return False


def canonicalize_relation(surface: str) -> str:
    """Map a surface verb form to its canonical relation string.

    ``manage`` / ``MANAGE`` are not aliased to ``HAS``; portfolio remit facts use
    ``MANAGE`` at PEF write time when the extractor produced ``HAS`` (see ``pef_updater``).

    Returns the uppercase canonical form, or the uppercased input
    if no alias is defined (allowing extension).
    """
    return _RELATION_ALIASES.get(surface.lower().strip(), surface.upper().strip())


# ── Relationship ─────────────────────────────────────────────────────

@dataclass
class Relationship:
    """A single fact linking a subject to an object (entity or literal).

    Exactly one of object_entity_id or object_literal must be non-null.

    Provenance: where the fact came from. user_input | pre_populated | llm_output | system.
    extractor_backend: spacy | llm | rule | manual | unknown.
    """
    subject_id: str
    relation: str                       # Must be canonical (pass through canonicalize_relation)

    object_entity_id: str | None        # References another entity
    object_literal: Any | None          # Literal value ("red", 42, "a book")

    span: Span
    source_turn: int                    # When established
    evidence: str                       # Surface text supporting this

    negated: bool = False               # Explicit negation ("doesn't have")

    provenance: str = "user_input"       # user_input | pre_populated | llm_output | system | retrieved_context
    extractor_backend: str = "unknown"  # spacy | llm | rule | manual | unknown
    document_id: str | None = None
    document_locator: str | None = None  # e.g. page=3, line=120-140, section=Employment
    # COMPARE relation metadata: {"adjective": "bigger"} — adjective independently recoverable
    relation_metadata: dict | None = None
    # Phase 5: authority audit trail {"actor": "...", "role": "...", "domain": "..."}
    authority_metadata: dict | None = None
    # Named source asserting this claim (e.g. "weather_model_a", "Sensor B", "Dr. Smith").
    # Distinct from provenance (which is a category). Used for conflict detection.
    source_name: str | None = None
    # Per-claim lifecycle status: asserted | contested | retracted | pending
    claim_status: str = "asserted"

    def __post_init__(self):
        has_entity = self.object_entity_id is not None
        has_literal = self.object_literal is not None
        if has_entity == has_literal:
            raise ValueError(
                "Exactly one of object_entity_id or object_literal must be non-null. "
                f"Got entity={self.object_entity_id!r}, literal={self.object_literal!r}"
            )


# ── Serialization helpers (Phase C) ───────────────────────────────────

def _entity_to_dict(e: Entity) -> dict[str, Any]:
    return {
        "id": e.id,
        "name": e.name,
        "aliases": sorted(e.aliases),
        "attributes": dict(e.attributes),
        "echo_traces": [
            {"turn": et.turn, "context": et.context, "span": et.span.value}
            for et in e.echo_traces
        ],
        "turn_introduced": e.turn_introduced,
        "turn_last_active": e.turn_last_active,
        "resolved": e.resolved,
    }


def _entity_from_dict(d: dict[str, Any]) -> Entity:
    aliases = set(d.get("aliases", []))
    if not aliases and d.get("name"):
        aliases = {d["name"]}
    echo_traces = [
        EchoTrace(
            turn=et["turn"],
            context=et.get("context", ""),
            span=Span(et["span"]) if isinstance(et.get("span"), str) else Span.PRESENT,
        )
        for et in d.get("echo_traces", [])
    ]
    return Entity(
        id=d["id"],
        name=d["name"],
        aliases=aliases,
        attributes=dict(d.get("attributes", {})),
        echo_traces=echo_traces,
        turn_introduced=int(d.get("turn_introduced", 0)),
        turn_last_active=int(d.get("turn_last_active", 0)),
        resolved=bool(d.get("resolved", True)),
    )


def _rel_to_dict(r: Relationship) -> dict[str, Any]:
    out: dict[str, Any] = {
        "subject_id": r.subject_id,
        "relation": r.relation,
        "object_entity_id": r.object_entity_id,
        "object_literal": r.object_literal,
        "span": r.span.value,
        "source_turn": r.source_turn,
        "evidence": r.evidence,
        "negated": r.negated,
        "provenance": r.provenance,
        "extractor_backend": r.extractor_backend,
        "claim_status": r.claim_status,
    }
    if r.document_id is not None:
        out["document_id"] = r.document_id
    if r.document_locator is not None:
        out["document_locator"] = r.document_locator
    if r.relation_metadata is not None:
        out["relation_metadata"] = r.relation_metadata
    if r.source_name is not None:
        out["source_name"] = r.source_name
    return out


def _rel_from_dict(d: dict[str, Any]) -> Relationship:
    span_val = d.get("span", "present")
    span = Span(span_val) if isinstance(span_val, str) else Span.PRESENT
    return Relationship(
        subject_id=d["subject_id"],
        relation=d["relation"],
        object_entity_id=d.get("object_entity_id"),
        object_literal=d.get("object_literal"),
        span=span,
        source_turn=int(d.get("source_turn", 0)),
        evidence=d.get("evidence", ""),
        negated=bool(d.get("negated", False)),
        provenance=d.get("provenance", "user_input"),
        extractor_backend=d.get("extractor_backend", "unknown"),
        document_id=d.get("document_id"),
        document_locator=d.get("document_locator"),
        relation_metadata=d.get("relation_metadata"),
        source_name=d.get("source_name"),
        claim_status=d.get("claim_status", "asserted"),
    )


def _turn_bindings_to_dict(tb: TurnBindings) -> dict[str, Any]:
    return {"turn": tb.turn, "bindings": dict(tb.bindings)}


def _turn_bindings_from_dict(d: dict[str, Any]) -> TurnBindings:
    return TurnBindings(
        turn=int(d.get("turn", 0)),
        bindings=dict(d.get("bindings", {})),
    )


def _hydrate_epistemic_hold_from_legacy(state: Any) -> None:
    """If stored JSON has pending_clarification but no epistemic_hold, infer ambiguity mode."""
    hold = state.epistemic_hold
    if hold and hold.get("mode") not in (None, "", EPISTEMIC_MODE_NONE):
        return
    if not state.pending_clarification:
        return
    state.epistemic_hold = {
        "schema_version": EPISTEMIC_HOLD_SCHEMA_VERSION,
        "mode": EPISTEMIC_MODE_AMBIGUITY,
        "since_turn": state.current_turn,
        "pathway_id": None,
        "interaction_open": True,
        "commitment_closed": False,
        "last_audit_id": "",
    }


# ── PEF State ────────────────────────────────────────────────────────

@dataclass
class PEFState:
    """The persistent present-state container.

    Maintains entities, relationships, and indexes for O(1) lookup.
    """
    entities: dict[str, Entity] = field(default_factory=dict)
    relationships: list[Relationship] = field(default_factory=list)

    # Indexes into relationships list
    rel_by_subject: dict[str, list[int]] = field(default_factory=dict)
    rel_by_object_entity: dict[str, list[int]] = field(default_factory=dict)
    rel_by_relation: dict[str, list[int]] = field(default_factory=dict)

    turn_bindings: dict[int, TurnBindings] = field(default_factory=dict)
    current_turn: int = 0
    active_span: Span = Span.PRESENT

    # Session-scoped entity IDs: name_index[canon_name] = next ordinal (2,3,...)
    #
    # ``session_id`` is also the durable PEF world/continuity handle — the proxy
    # wire calls this ``pef_context_id`` (see ``aurora_metadata.build_aurora_block``
    # and ``frame_lifecycle.extract_explicit_pef_context_id``). One value backs
    # both names; there is deliberately no separate stored field for the other
    # name, since they must never drift apart.
    session_id: str = ""
    name_index: dict[str, int] = field(default_factory=dict)

    # Pending clarification: set when governance escalates to ASK/CLARIFY so
    # that the next turn can detect whether the user has supplied a binding and
    # resume the original determination rather than treating the clarification
    # as a new, independent query.
    #
    # Schema (all keys are strings):
    #   "original_question"      str            — user input that triggered ASK
    #   "unresolved_entity_ids"  list[str]      — PEF entity IDs that were resolved=False
    #                                             at ASK time (empty = pronoun-ambiguity case)
    #   "failed_constraint"      str | None     — flag type name, e.g. "UNRESOLVED_REFERENT",
    #                                             "UNRESOLVED_COMPARAND"
    #   "ambiguous_referents"    list[str]      — pronouns that could not be uniquely resolved
    #                                             (e.g. ["his"]); empty for entity-placeholder path
    #   "candidate_entities"     list[str]      — proper-noun candidate antecedents
    #   "original_span"          str            — span value at ASK time ("present" | "past")
    #   "blocked_proposition"    str | None     — sentence(s) containing the ambiguous pronoun,
    #                                             used to scope reconstruction on binding
    #   "blocked_claims"         list[dict]     — serialized ExtractedClaim dicts whose subject
    #                                             involves the ambiguous pronoun
    #   "comparand_adjective"    str            — (UNRESOLVED_COMPARAND only) e.g. "bigger"
    #   "comparand_noun"         str            — (UNRESOLVED_COMPARAND only) e.g. "stick"
    #   "item_phrase"            str            — (state-native holder ambiguity) inventory head
    #   "subject_phrase"         str            — (state-native subject ambiguity) phrase head
    #   "typo_recovery"          dict           — optional typo clarification payload:
    #                                            typo_surface, typo_entity_id,
    #                                            suggested_canonical_name, canonical_entity_id
    #   "state_native_ambiguity_kind" str       — inventory_holder_query |
    #                                            inventory_holder_query_unresolved_item |
    #                                            inventory_subject_query | location_subject_query
    #   "entity_ambiguity_typo_recovery_suppressed" bool — user declined typo suggestion
    #
    # Not all keys are present in every instance; the binding-resolution code
    # in lens.py uses .get() with defaults for backward compatibility.
    pending_clarification: dict | None = field(default=None)
    # Phase 6: typed hold wrapper; set alongside pending_clarification when available.
    # Not serialized — reconstructed from pending_clarification via from_pending_dict().
    pending_transaction: ContinuationTransaction | None = field(default=None, repr=False)

    # After a successful clarification resume (CONTAIN → user binds a candidate),
    # maps lowercase pronoun tokens (e.g. "her", "she", "its") to the chosen
    # entity name so follow-up questions do not re-trigger UNRESOLVED_REFERENT
    # for the same narrative referent.
    discourse_referent_bindings: dict[str, str] = field(default_factory=dict)

    # Durable unresolved-referent registry (v10-style): persists across turns until
    # explicit resolution. Independent of ``pending_clarification`` lifecycle.
    unresolved_referent_registry: list[UnresolvedReferentEntry] = field(default_factory=list)

    # RAG ``Context:`` admission only: incompatible facts in the retrieved batch that
    # were not committed (e.g. multiple IS literals for the same subject). Cleared
    # and repopulated on each retrieval admission call; governance may consult it.
    retrieval_unresolved: list[dict[str, Any]] = field(default_factory=list)

    # Declarative epistemic uncertainty state supplied by the caller via
    # ``evidence_state.open_epistemic_uncertainties``.  Each entry is a raw dict
    # with at least {id, kind, description, status}.  Cleared and repopulated on
    # each request that carries an ``evidence_state`` block.
    open_epistemic_uncertainties: list[dict[str, Any]] = field(default_factory=list)

    # Durable epistemic holding: mirrors last non-ADMIT governance mode (ASK/REFUSE/STOP).
    # Updated by lens.py alongside pending_clarification for ambiguity; REFUSE/STOP
    # supersede pending_clarification. Schema keys (schema_version, mode, since_turn, ...).
    epistemic_hold: dict | None = field(default=None)

    # Active continuation capability corridor (e.g. "neutral_timeline") for
    # follow-up turns after governed blocked responses.
    active_continuation_capability: str | None = field(default=None)
    # Optional structured context associated with ``active_continuation_capability``.
    # Used by bounded post-refusal follow-up corridors (e.g. medical GP hard-stop
    # follow-up preparation prompts) without carrying blocked substantive output.
    active_continuation_context: dict[str, Any] | None = field(default=None)

    # Append-only audit log for explicit identity merges (e.g. deterministic typo confirmation).
    identity_corrections: list[dict[str, Any]] = field(default_factory=list)

    # Present-bound temporal commitments (Phase 3). Not archival timeline facts.
    temporal_commitments: list[PresentTemporalCommitment] = field(default_factory=list)

    # Active conversational frame (artifact/external layer — Phases 1-5).
    # None = EXTERNAL (default); set to ArtifactFrame when a fiction/roleplay/hypothetical
    # opener is detected. Cleared on explicit frame-closer. Governs cross-layer admissibility.
    active_frame: ArtifactFrame | None = field(default=None)

    # ── Present-bound temporal commitments (Phase 3) ───────────────

    def add_present_temporal_commitment(self, commitment: PresentTemporalCommitment) -> None:
        """Append a present-bound temporal commitment record."""
        self.temporal_commitments.append(commitment)

    # ── Entity operations ────────────────────────────────────────

    def add_entity(self, entity: Entity) -> Entity:
        """Register an entity. Returns the entity (for chaining)."""
        self.entities[entity.id] = entity
        return entity

    def find_entity_by_name(self, name: str) -> Entity | None:
        """Look up an entity by canonical name or alias (case-insensitive)."""
        name_lower = name.lower()
        for entity in self.entities.values():
            if entity.name.lower() == name_lower:
                return entity
            if any(a.lower() == name_lower for a in entity.aliases):
                return entity
        return None

    def get_or_create_entity(
        self,
        name: str,
        *,
        resolved: bool = True,
        entity_id: str | None = None,
        force_new: bool = False,
    ) -> tuple[Entity, bool]:
        """Find entity by name or create a new one.

        Unification: reuse if entity_id supplied (caller binding) or find by name.
        Sibling: when force_new=True and canon_name exists, mint ordinal #2, #3, ...
        Returns (entity, created) where created is True if new.
        """
        canon_name = name.strip().lower()
        if entity_id is not None and entity_id in self.entities:
            e = self.entities[entity_id]
            e.turn_last_active = self.current_turn
            return e, False
        if not force_new:
            existing = self.find_entity_by_name(name)
            if existing is not None:
                existing.turn_last_active = self.current_turn
                return existing, False
        # Create new: base (first) or sibling (ordinal)
        existing = self.find_entity_by_name(name)
        if existing is None:
            entity = Entity.create(
                name,
                self.current_turn,
                resolved=resolved,
                session_id=self.session_id,
                ordinal=None,
            )
        else:
            k = self.name_index.get(canon_name, 2)
            self.name_index[canon_name] = k + 1
            entity = Entity.create(
                name,
                self.current_turn,
                resolved=resolved,
                session_id=self.session_id,
                ordinal=k,
            )
        self.add_entity(entity)
        return entity, True

    # ── Relationship operations ──────────────────────────────────

    def add_relationship(self, rel: Relationship) -> int:
        """Add a relationship and update all indexes. Returns the index."""
        idx = len(self.relationships)
        self.relationships.append(rel)

        self.rel_by_subject.setdefault(rel.subject_id, []).append(idx)
        if rel.object_entity_id is not None:
            self.rel_by_object_entity.setdefault(rel.object_entity_id, []).append(idx)
        self.rel_by_relation.setdefault(rel.relation, []).append(idx)

        return idx

    def get_relationships_for_subject(self, subject_id: str) -> list[Relationship]:
        """All relationships where this entity is the subject."""
        indices = self.rel_by_subject.get(subject_id, [])
        return [self.relationships[i] for i in indices]

    def get_relationships_for_object(self, entity_id: str) -> list[Relationship]:
        """All relationships where this entity is the object."""
        indices = self.rel_by_object_entity.get(entity_id, [])
        return [self.relationships[i] for i in indices]

    def get_relationships_by_relation(self, relation: str) -> list[Relationship]:
        """All relationships of a specific type."""
        indices = self.rel_by_relation.get(relation, [])
        return [self.relationships[i] for i in indices]

    def find_relationship(
        self,
        subject_id: str,
        relation: str,
        *,
        object_entity_id: str | None = None,
        object_literal: Any | None = None,
    ) -> Relationship | None:
        """Find a specific relationship matching the given criteria."""
        for rel in self.get_relationships_for_subject(subject_id):
            if rel.relation != relation:
                continue
            if object_entity_id is not None and rel.object_entity_id == object_entity_id:
                return rel
            if object_literal is not None and rel.object_literal == object_literal:
                return rel
        return None

    def rebuild_relationship_indexes(self) -> None:
        """Rebuild relationship indexes after in-place relationship edits."""
        self.rel_by_subject = {}
        self.rel_by_object_entity = {}
        self.rel_by_relation = {}
        for idx, rel in enumerate(self.relationships):
            self.rel_by_subject.setdefault(rel.subject_id, []).append(idx)
            if rel.object_entity_id is not None:
                self.rel_by_object_entity.setdefault(rel.object_entity_id, []).append(idx)
            self.rel_by_relation.setdefault(rel.relation, []).append(idx)

    def merge_entity_identity_absorb_into_canonical(
        self,
        *,
        absorb_entity_id: str,
        canonical_entity_id: str,
        audit_reason: str,
        original_ambiguity_flag: str,
        turn: int,
    ) -> dict[str, Any]:
        """Rewire committed facts from absorbed entity into canonical; remove absorbed entity.

        Does not infer sameness — callers must obtain explicit user confirmation first.
        """
        if absorb_entity_id == canonical_entity_id:
            raise ValueError("absorb_entity_id and canonical_entity_id must differ")
        absorb_ent = self.entities.get(absorb_entity_id)
        canon_ent = self.entities.get(canonical_entity_id)
        if absorb_ent is None or canon_ent is None:
            raise KeyError("merge_entity_identity_absorb_into_canonical: missing entity")

        absorb_surface = absorb_ent.name.strip()
        canon_surface = canon_ent.name.strip()

        for rel in self.relationships:
            if rel.subject_id == absorb_entity_id:
                rel.subject_id = canonical_entity_id
            if rel.object_entity_id == absorb_entity_id:
                rel.object_entity_id = canonical_entity_id

        for tb in self.turn_bindings.values():
            for occ_key, bound_id in list(tb.bindings.items()):
                if bound_id == absorb_entity_id:
                    tb.bindings[occ_key] = canonical_entity_id

        absorb_low = absorb_surface.lower()
        for dk, dv in list(self.discourse_referent_bindings.items()):
            if dv.strip().lower() == absorb_low:
                self.discourse_referent_bindings[dk] = canon_surface

        canon_ent.aliases |= absorb_ent.aliases
        canon_ent.aliases.add(absorb_surface)
        canon_ent.echo_traces.extend(absorb_ent.echo_traces)
        canon_ent.turn_last_active = max(canon_ent.turn_last_active, absorb_ent.turn_last_active)
        canon_ent.attributes = {**absorb_ent.attributes, **canon_ent.attributes}

        del self.entities[absorb_entity_id]
        self.rebuild_relationship_indexes()

        audit_record: dict[str, Any] = {
            "source_typo_entity_id": absorb_entity_id,
            "source_typo_surface": absorb_surface,
            "target_canonical_entity_id": canonical_entity_id,
            "target_canonical_surface": canon_surface,
            "reason_type": audit_reason,
            "original_ambiguity_flag": original_ambiguity_flag,
            "turn": turn,
        }
        self.identity_corrections.append(audit_record)
        return audit_record

    # ── Pronoun binding operations ───────────────────────────────

    def resolve_pronoun(self, pronoun_key: str, entity_id: str) -> None:
        """Bind a pronoun occurrence to an entity for the current turn.

        pronoun_key should be occurrence-based, e.g. "she@token_14".
        """
        if self.current_turn not in self.turn_bindings:
            self.turn_bindings[self.current_turn] = TurnBindings(
                turn=self.current_turn
            )
        self.turn_bindings[self.current_turn].bindings[pronoun_key] = entity_id

    def get_binding(self, pronoun_key: str, turn: int | None = None) -> str | None:
        """Look up what entity a pronoun resolved to in a specific turn."""
        t = turn if turn is not None else self.current_turn
        tb = self.turn_bindings.get(t)
        if tb is None:
            return None
        return tb.bindings.get(pronoun_key)

    # ── Echo trace operations ────────────────────────────────────

    def record_echo_trace(self, entity_id: str, context: str) -> None:
        """Record an echo trace for an entity at the current turn and span."""
        entity = self.entities.get(entity_id)
        if entity is None:
            raise KeyError(f"Entity {entity_id!r} not found")
        trace = EchoTrace(
            turn=self.current_turn,
            context=context,
            span=self.active_span,
        )
        entity.echo_traces.append(trace)
        entity.turn_last_active = self.current_turn

    # ── Turn management ──────────────────────────────────────────

    def advance_turn(self) -> int:
        """Advance to the next turn. Returns the new turn number."""
        self.current_turn += 1
        return self.current_turn

    # ── Serialization (Phase C) ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-serializable dict. Round-trip with from_dict()."""
        # Sort relationships for deterministic state_hash (claims/LLM order may vary).
        def _rel_sort_key(r: Relationship) -> tuple:
            return (
                r.subject_id,
                r.relation,
                r.object_entity_id or "",
                str(r.object_literal) if r.object_literal is not None else "",
                r.source_turn,
                r.evidence,
                r.provenance,
                r.extractor_backend,
                r.document_id or "",
                r.document_locator or "",
            )
        return {
            "entities": {
                eid: _entity_to_dict(e) for eid, e in self.entities.items()
            },
            "relationships": [
                _rel_to_dict(r)
                for r in sorted(self.relationships, key=_rel_sort_key)
            ],
            "turn_bindings": {
                f"{turn:05d}": _turn_bindings_to_dict(tb)
                for turn, tb in sorted(self.turn_bindings.items())
            },
            "current_turn": self.current_turn,
            "active_span": self.active_span.value,
            "session_id": self.session_id,
            "name_index": dict(sorted(self.name_index.items())),
            "pending_clarification": self.pending_clarification,
            "epistemic_hold": self.epistemic_hold,
            "active_continuation_capability": self.active_continuation_capability,
            "active_continuation_context": self.active_continuation_context,
            "discourse_referent_bindings": dict(sorted(self.discourse_referent_bindings.items())),
            "unresolved_referent_registry": registry_entries_to_wire(
                self.unresolved_referent_registry
            ),
            "retrieval_unresolved": list(self.retrieval_unresolved),
            "open_epistemic_uncertainties": list(self.open_epistemic_uncertainties),
            "identity_corrections": list(self.identity_corrections),
            "temporal_commitments": temporal_commitments_to_wire(self.temporal_commitments),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PEFState":
        """Deserialize from dict. Rebuilds indexes from relationships."""
        entities = {
            eid: _entity_from_dict(d) for eid, d in data.get("entities", {}).items()
        }
        relationships = [_rel_from_dict(d) for d in data.get("relationships", [])]
        turn_bindings = {
            int(k): _turn_bindings_from_dict(v)
            for k, v in data.get("turn_bindings", {}).items()
        }
        current_turn = int(data.get("current_turn", 0))
        active_span_val = data.get("active_span", "present")
        active_span = Span(active_span_val) if isinstance(active_span_val, str) else Span.PRESENT
        session_id = str(data.get("session_id", ""))
        name_index = dict(data.get("name_index", {}))
        pending_clarification = data.get("pending_clarification")  # None when absent
        epistemic_hold = data.get("epistemic_hold")
        if isinstance(epistemic_hold, dict) and not epistemic_hold:
            epistemic_hold = None
        active_continuation_capability = data.get("active_continuation_capability")
        if isinstance(active_continuation_capability, str):
            active_continuation_capability = active_continuation_capability.strip() or None
        else:
            active_continuation_capability = None
        active_continuation_context_raw = data.get("active_continuation_context")
        active_continuation_context = (
            dict(active_continuation_context_raw)
            if isinstance(active_continuation_context_raw, dict)
            else None
        )
        discourse_referent_bindings = dict(data.get("discourse_referent_bindings", {}))
        unresolved_referent_registry = registry_entries_from_wire(
            data.get("unresolved_referent_registry")
        )
        retrieval_unresolved = list(data.get("retrieval_unresolved", []))
        oeu_raw = data.get("open_epistemic_uncertainties")
        open_epistemic_uncertainties = list(oeu_raw) if isinstance(oeu_raw, list) else []
        identity_corrections = [
            dict(x) for x in data.get("identity_corrections", []) if isinstance(x, dict)
        ]
        temporal_commitments = temporal_commitments_from_wire(data.get("temporal_commitments"))

        state = PEFState(
            entities=entities,
            relationships=relationships,
            turn_bindings=turn_bindings,
            current_turn=current_turn,
            active_span=active_span,
            session_id=session_id,
            name_index=name_index,
            pending_clarification=pending_clarification,
            epistemic_hold=epistemic_hold,
            active_continuation_capability=active_continuation_capability,
            active_continuation_context=active_continuation_context,
            discourse_referent_bindings=discourse_referent_bindings,
            unresolved_referent_registry=unresolved_referent_registry,
            retrieval_unresolved=retrieval_unresolved,
            open_epistemic_uncertainties=open_epistemic_uncertainties,
            identity_corrections=identity_corrections,
            temporal_commitments=temporal_commitments,
        )
        _hydrate_epistemic_hold_from_legacy(state)
        # Rebuild indexes
        state.rebuild_relationship_indexes()
        return state

    # ── Summary for LLM injection ────────────────────────────────

    def to_context_summary(self) -> str:
        """Generate a ground-truth summary suitable for LLM system prompt injection."""
        if not self.entities and not self.relationships and not self.retrieval_unresolved:
            return "No established facts."

        lines: list[str] = []

        # Entities
        if self.entities:
            lines.append("## Known Entities")
            for entity in self.entities.values():
                status = "" if entity.resolved else " [UNRESOLVED]"
                attrs = ""
                if entity.attributes:
                    attr_parts = [f"{k}={v}" for k, v in entity.attributes.items()]
                    attrs = f" ({', '.join(attr_parts)})"
                lines.append(f"- {entity.name}{status}{attrs}")

        # Relationships (grouped by subject)
        present_rels: list[str] = []
        past_rels: list[str] = []

        for rel in self.relationships:
            subj = self.entities.get(rel.subject_id)
            subj_name = subj.name if subj else rel.subject_id

            obj_str: str
            if rel.object_entity_id:
                obj_ent = self.entities.get(rel.object_entity_id)
                obj_str = obj_ent.name if obj_ent else rel.object_entity_id
            else:
                obj_str = str(rel.object_literal)

            neg = "NOT " if rel.negated else ""
            fact = f"- {subj_name} {rel.relation} {neg}{obj_str}"

            if rel.span == Span.PRESENT:
                present_rels.append(fact)
            else:
                past_rels.append(fact)

        if present_rels:
            lines.append("\n## Current Facts (present)")
            lines.extend(present_rels)

        if past_rels:
            lines.append("\n## Past Facts")
            lines.extend(past_rels)

        open_ur = open_entries(self)
        if open_ur:
            lines.append("\n## Unresolved Referents (durable — require explicit binding)")
            for entry in open_ur:
                cands = ", ".join(entry.candidate_entities) if entry.candidate_entities else "unknown"
                lines.append(
                    f"- '{entry.span_surface}' → candidates: {cands} "
                    f"(since turn {entry.introduced_turn})"
                )

        if self.retrieval_unresolved:
            lines.append("## Retrieval (unresolved)")
            for item in self.retrieval_unresolved:
                if item.get("kind") == "literal_conflict":
                    lit = item.get("literals") or []
                    subj = item.get("subject", "")
                    rel = item.get("relation", "")
                    lines.append(
                        f"- Conflicting {rel} values for {subj}: {', '.join(str(x) for x in lit)}"
                    )
                elif item.get("kind") == "temporal_arrival_incompatible":
                    subj = item.get("subject", "")
                    lines.append(
                        f"- Incompatible arrival-time signals for {subj} "
                        f"(calendar date vs last week in retrieved context)"
                    )
                else:
                    lines.append(f"- {item!r}")

        return "\n".join(lines)
