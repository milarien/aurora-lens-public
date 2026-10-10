"""Extraction schema — the contract between extraction backends and consumers.

These types define what any extraction backend must produce.
Importable without pulling in spaCy or any specific backend.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from aurora_lens.pef.span import Span


@dataclass
class ExtractedClaim:
    """A claim extracted from text before PEF integration."""
    subject: str
    relation: str           # Canonical
    obj: str                # Object text (entity name or literal)
    span: Span
    negated: bool
    evidence: str           # Source text

    provenance: str = "user_input"       # user_input | pre_populated | llm_output | system | retrieved_context
    extractor_backend: str = "unknown"  # spacy | llm | rule | manual | unknown
    document_id: str | None = None
    document_locator: str | None = None
    utterance_act: str | None = None
        # Act of the sentence the claim came from: assert | instruct | undetermined.
        # None: the backend did not assess the act.
    clause_scope: str | None = None
        # main: the claim's predicate is the sentence's main predicate.
        # embedded: relative, complement, or other subordinate clause.


@dataclass
class ComparativeAmbiguity:
    """A comparative adjective whose comparand cannot be uniquely determined.

    Produced when a JJR/RBR token modifies "be" and 2+ eligible comparands
    exist in PEF.  Exactly 1 candidate → forced collapse (not reported).
    Zero candidates → ungrounded assertion (not reported).
    """
    adjective: str          # Surface form: "bigger"
    noun: str               # Thing being compared: "stick"
    candidates: list[str]   # Eligible comparands: ["Richard", "Lucy"]


# ── Phase 7 — Forensic lifecycle ──────────────────────────────────────────────

LIFECYCLE_KINDS = frozenset(
    {"proposed", "held", "replayed", "committed", "refused", "answered"}
)


@dataclass
class LifecycleEvent:
    """A single audit event in a SemanticTransaction's lifecycle (Phase 7).

    kind: the lifecycle stage.
        "proposed"  — transaction built from extraction.
        "held"      — commit blocked; transaction pending resolution.
        "replayed"  — held transaction replayed after binding.
        "committed" — claim written to PEF.
        "refused"   — transaction refused by governance or authority check.
        "answered"  — query transaction answered from committed state (no PEF write).
    turn: conversation turn at which this event occurred.
    reason: optional annotation (held_reason, stop_reason, etc.).
    pef_snapshot_hash: "sha256:<hex>" of pef.to_dict() at this event, or None.
    """
    kind: str
    turn: int
    reason: str | None = None
    pef_snapshot_hash: str | None = None


@dataclass
class SemanticTransaction:
    """Pre-commit wrapper for a proposed claim with commit-validity status.

    Phase 1:  comparative IS claims (UNRESOLVED_COMPARAND).
    Phase 2:  possessive-NP subject claims with unresolved possessors (UNRESOLVED_POSSESSOR).
    Phase 2b: bare pronoun subjects whose token appears in ambiguous_referents (UNRESOLVED_REFERENT).

    ``allowed_commit=False`` means the claim must not be written to PEF; it is held
    pending resolution.
    """
    claim: ExtractedClaim
    unresolved_comparand: str | None = None   # Phase 1: adjective held pending comparand resolution
    unresolved_possessor: str | None = None   # Phase 2: possessor pronoun (his/her/their/its)
    allowed_commit: bool = True
    held_reason: str | None = None            # "UNRESOLVED_COMPARAND" | "UNRESOLVED_POSSESSOR"
    # Phase 7: forensic lifecycle
    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    lifecycle: list[LifecycleEvent] = field(default_factory=list)

    def record(
        self,
        kind: str,
        turn: int,
        *,
        reason: str | None = None,
        pef: object = None,
    ) -> None:
        """Append a lifecycle event. Enforces: refused transaction must not commit."""
        if kind == "committed" and any(e.kind == "refused" for e in self.lifecycle):
            raise ValueError(
                f"Invariant violation: transaction {self.transaction_id!r} was refused "
                f"and must not be committed."
            )
        hash_: str | None = None
        if pef is not None:
            import hashlib as _hl
            import json as _json
            try:
                snapshot = _json.dumps(
                    pef.to_dict(),  # type: ignore[union-attr]
                    sort_keys=True,
                    default=str,
                )
                hash_ = "sha256:" + _hl.sha256(snapshot.encode()).hexdigest()
            except Exception:
                pass
        self.lifecycle.append(
            LifecycleEvent(kind=kind, turn=turn, reason=reason, pef_snapshot_hash=hash_)
        )


@dataclass
class PossessionTransaction(SemanticTransaction):
    """SemanticTransaction for HAS/GIVE/TAKE/EAT/CONSUME possession mutations.

    mutation_kind:
      "acquire"   — plain HAS add (counted HAS, negated HAS, or from TAKE); no supersession.
      "supersede" — bare (uncounted) non-negated HAS; negate prior same-item HAS then add.
      "transfer"  — GIVE: subtract quantity from giver, add to recipient.
      "consume"   — EAT/CONSUME: subtract quantity from consumer, add remainder.
    """
    mutation_kind: str = "acquire"
    quantity: float | None = None
    prior_count_snapshot: float | None = None  # captured at build time (Law P1)
    projected_remainder: float | None = None   # set by validate()
    giver_name: str | None = None
    recipient_name: str | None = None
    item_key_str: str | None = None            # normalised item key

    def validate(self, pef: object) -> str | None:  # noqa: ARG002
        """Return stop_reason string if mutation is inadmissible; else None.

        Sets projected_remainder as a side-effect when the mutation is admitted.
        """
        if self.mutation_kind == "transfer":
            if self.quantity is not None and self.prior_count_snapshot is not None:
                if self.prior_count_snapshot < self.quantity:
                    giver = self.giver_name or str(self.claim.subject)
                    return (
                        f"I cannot answer that from committed state: {giver} does not "
                        f"have {int(self.quantity)} {self.item_key_str or 'items'}."
                    )
                self.projected_remainder = self.prior_count_snapshot - self.quantity
        elif self.mutation_kind == "consume":
            if self.quantity is not None and self.prior_count_snapshot is not None:
                if self.prior_count_snapshot < self.quantity:
                    return (
                        f"I cannot answer that from committed state: "
                        f"{self.claim.subject} does not have {int(self.quantity)} "
                        f"{self.item_key_str or 'items'}."
                    )
                self.projected_remainder = self.prior_count_snapshot - self.quantity
        return None

    def commit(self, pef: object) -> None:
        """Atomically write all HAS mutations produced by this transaction."""
        from aurora_lens.pef.state import (  # local import avoids circular dep
            Relationship,
            canonicalize_relation,
            object_has_business_remit_marker,
        )
        from aurora_lens.state_native_engine.eval.inventory import (
            supersede_same_subject_prior_holds_for_item,
        )

        c = self.claim
        span = c.span

        # Respect business-remit marker: HAS on a managed portfolio → MANAGE
        _base_rel = canonicalize_relation(c.relation)
        if object_has_business_remit_marker(c.obj) and _base_rel in ("HAS", "MANAGE"):
            _stored_rel = "MANAGE"
        else:
            _stored_rel = _base_rel

        if self.mutation_kind == "supersede":
            subj_ent = pef.find_entity_by_name(c.subject)  # type: ignore[union-attr]
            if subj_ent is None:
                subj_ent, _ = pef.get_or_create_entity(c.subject)  # type: ignore[union-attr]
            supersede_same_subject_prior_holds_for_item(
                pef,
                subj_ent.id,
                str(c.obj),
                evidence=f"Superseded by new possession assertion for {str(c.obj).strip()}",
                bookkeeping_backdate_turn=True,
            )
            pef.add_relationship(Relationship(  # type: ignore[union-attr]
                subject_id=subj_ent.id,
                relation=_stored_rel,
                object_entity_id=None,
                object_literal=str(c.obj),
                span=span,
                source_turn=pef.current_turn,  # type: ignore[union-attr]
                evidence=str(c.evidence or ""),
                negated=False,
                provenance=c.provenance,
                extractor_backend=c.extractor_backend,
                document_id=c.document_id,
                document_locator=c.document_locator,
                relation_metadata={"transaction_id": self.transaction_id},
            ))

        elif self.mutation_kind == "acquire":
            obj_ent = pef.find_entity_by_name(str(c.obj))  # type: ignore[union-attr]
            subj_ent = pef.find_entity_by_name(c.subject)  # type: ignore[union-attr]
            if subj_ent is None:
                subj_ent, _ = pef.get_or_create_entity(c.subject)  # type: ignore[union-attr]
            pef.add_relationship(Relationship(  # type: ignore[union-attr]
                subject_id=subj_ent.id,
                relation=_stored_rel,
                object_entity_id=obj_ent.id if obj_ent else None,
                object_literal=None if obj_ent else str(c.obj),
                span=span,
                source_turn=pef.current_turn,  # type: ignore[union-attr]
                evidence=str(c.evidence or ""),
                negated=c.negated,
                provenance=c.provenance,
                extractor_backend=c.extractor_backend,
                document_id=c.document_id,
                document_locator=c.document_locator,
                relation_metadata={"transaction_id": self.transaction_id},
            ))

        elif self.mutation_kind in ("transfer", "consume"):
            subj_ent = pef.find_entity_by_name(c.subject)  # type: ignore[union-attr]
            if subj_ent is None:
                subj_ent, _ = pef.get_or_create_entity(c.subject)  # type: ignore[union-attr]
            old_literal = str(c.obj)
            remainder = self.projected_remainder
            item_str = self.item_key_str or ""

            pef.add_relationship(Relationship(  # type: ignore[union-attr]
                subject_id=subj_ent.id,
                relation="HAS",
                object_entity_id=None,
                object_literal=old_literal,
                span=span,
                source_turn=pef.current_turn,  # type: ignore[union-attr]
                evidence=str(c.evidence or ""),
                negated=True,
                provenance="system",
                extractor_backend="possession_transaction",
                relation_metadata={"transaction_id": self.transaction_id},
            ))
            if remainder is not None and remainder > 0:
                new_literal = f"{int(remainder)} {item_str}"
                pef.add_relationship(Relationship(  # type: ignore[union-attr]
                    subject_id=subj_ent.id,
                    relation="HAS",
                    object_entity_id=None,
                    object_literal=new_literal,
                    span=span,
                    source_turn=pef.current_turn,  # type: ignore[union-attr]
                    evidence=str(c.evidence or ""),
                    negated=False,
                    provenance="system",
                    extractor_backend="possession_transaction",
                    relation_metadata={"transaction_id": self.transaction_id},
                ))

        self.record("committed", pef.current_turn)  # type: ignore[union-attr]


@dataclass
class QueryTransaction:
    """Typed wrapper for a read-only PEF query (Phase 4).

    Does not mutate PEF. Carries the resolved/unresolved state of query targets
    so callers can route CONTAIN (2+ unresolved candidates) vs ANSWER vs STOP
    without falling through to the LLM.

    query_kind:
      "inventory_subject"  — "What does X have?"
      "inventory_holder"   — "Who has Y?"
      "location"           — "Where is X?"
      "comparative"        — "Whose X was Y-er?"
      "temporal"           — "When should I contact X?"
      "attribution"        — "Who should I follow up with?"

    resolved_targets: entity names successfully bound from the query surface.
    unresolved_tokens: pronoun/definite-NP tokens with 2+ candidate referents;
                       non-empty signals that the outcome should be CONTAIN,
                       not STOP or LLM fallback.
    """
    query_kind: str = "inventory_holder"
    resolved_targets: list[str] = field(default_factory=list)
    unresolved_tokens: list[str] = field(default_factory=list)
    query_surface: str = ""


# ── Phase 5 — Domain-event transactions ──────────────────────────────────────

# Consequence-bearing act classes that require authority validation before commit.
# "bind"      — creates an obligation (prescribe, mandate, authorize, rule)
# "advise"    — renders professional advice
# "diagnose"  — issues a diagnosis or assessment
# "predict"   — makes a consequential prediction about future state
# "recommend" — makes a formal recommendation
# "continue"  — continues an existing authority relationship
CONSEQUENCE_ACT_CLASSES = frozenset(
    {"bind", "advise", "diagnose", "predict", "recommend", "continue"}
)


@dataclass
class AuthorityCheck:
    """Pre-consequence authority gate for a proposed domain act (Phase 5).

    Answers: may this actor/system perform the proposed act in this domain?

    proposed_act: the authority act class being proposed.
        "bind"       — creates obligation (prescribe, authorize, mandate)
        "advise"     — provides professional advice
        "diagnose"   — renders a diagnosis or assessment
        "predict"    — makes a consequential prediction
        "recommend"  — makes a formal recommendation
        "continue"   — continues an existing authority relationship

    authority_class: class of authority making the claim.
        e.g. "medical_practitioner", "legal_counsel", "financial_advisor",
             "system", "user_self", "unspecified"

    actor: entity name whose PEF state is checked.
        None for system-originated events (LLM, automated pipeline).

    required_pef_roles: role literals in PEF that confirm authority for this act.
        Empty list → act does not require a committed PEF role (admitted without lookup).

    found_role: first matching role literal found in PEF (set by evaluate()).
    result: "admitted" | "refused" | "unknown" | "pending" (updated by evaluate()).

    evaluate() contract:
        required_pef_roles empty        → "admitted" (no PEF role required)
        actor is None                   → "unknown"  (no entity to look up)
        actor not in PEF                → "unknown"
        actor in PEF, role matches      → "admitted"
        actor in PEF, has roles, no match → "refused"
        actor in PEF, no IS/ROLE rels   → "unknown"
    """
    proposed_act: str
    authority_class: str = "unspecified"
    actor: str | None = None
    required_pef_roles: list[str] = field(default_factory=list)
    found_role: str | None = None
    result: str = "pending"

    def evaluate(self, pef: object) -> str:
        """Check committed PEF. Returns "admitted", "refused", or "unknown"."""
        if not self.required_pef_roles:
            self.result = "admitted"
            return self.result

        if self.actor is None:
            self.result = "unknown"
            return self.result

        actor_ent = pef.find_entity_by_name(self.actor)  # type: ignore[union-attr]
        if actor_ent is None:
            self.result = "unknown"
            return self.result

        has_any_role = False
        for rel in pef.relationships:  # type: ignore[union-attr]
            if rel.subject_id != actor_ent.id or rel.negated:
                continue
            if rel.relation not in ("IS", "ROLE", "HAS_ROLE"):
                continue
            has_any_role = True
            role_lit = str(rel.object_literal or "").lower()
            for req in self.required_pef_roles:
                if req.lower() in role_lit:
                    self.found_role = str(rel.object_literal)
                    self.result = "admitted"
                    return self.result

        self.result = "refused" if has_any_role else "unknown"
        return self.result


@dataclass
class DomainEventTransaction(SemanticTransaction):
    """A proposed consequence-bearing domain event requiring authority validation (Phase 5).

    The invariant: no consequence-bearing event in a governed domain is committed
    to PEF until the authority to perform the proposed act is established from
    committed state — not asserted, not inferred, not derived from LLM output.

    domain: governed domain — "medical" | "legal" | "finance".
    event_verb: canonical event verb ("prescribe", "administer", "authorize", etc.).
    proposed_act: act class being proposed — mirrors authority_check.proposed_act.
        "bind" | "advise" | "diagnose" | "predict" | "recommend" | "continue"
    authority_check: gates admissibility before consequence; None skips the gate.
    admissibility: "pending" until validate() runs; then:
        "admitted" — authority established, may proceed to commit.
        "refused"  — actor/system role is known but does not confer this act.
        "unknown"  — authority not established (actor absent or unroled in PEF).

    validate(pef) returns None when admitted, or a stop_reason string otherwise.
    """
    domain: str = "medical"
    event_verb: str = ""
    proposed_act: str = "advise"
    authority_check: AuthorityCheck | None = None
    admissibility: str = "pending"

    def validate(self, pef: object) -> str | None:  # noqa: ARG002
        """Gate the proposed act. Sets self.admissibility. Returns stop_reason or None."""
        if self.authority_check is None:
            self.admissibility = "admitted"
            return None
        result = self.authority_check.evaluate(pef)
        self.admissibility = result
        actor = self.authority_check.actor or "this actor"
        if result == "admitted":
            return None
        if result == "refused":
            return (
                f"I cannot process that event: {actor} does not have authority "
                f"to {self.proposed_act} in the {self.domain} domain."
            )
        return (
            f"I cannot process that event: the authority of {actor} "
            f"to {self.proposed_act} in the {self.domain} domain "
            f"has not been established."
        )


# ── Phase 6 — Continuation transactions ──────────────────────────────────────


@dataclass
class ResolutionCondition:
    """Predicate governing when a held ContinuationTransaction can close (Phase 6).

    Evaluated against each incoming turn. matches() → True means the continuation
    may replay; → False means the hold persists for another turn.

    constraint_kind: the failed-constraint type that created this hold.
        "UNRESOLVED_REFERENT" | "UNRESOLVED_COMPARAND" | "state_native_ambiguity" | …
    candidate_entities: entity names the user may choose among.
    required_turn_acts: TurnAct.value strings eligible to satisfy this condition.
        Empty → any turn act is eligible.
    requires_candidate_match: when True, the user's turn_text must name at least
        one candidate entity explicitly.
    """
    constraint_kind: str
    candidate_entities: list[str] = field(default_factory=list)
    required_turn_acts: list[str] = field(default_factory=list)
    requires_candidate_match: bool = False

    def matches(self, turn_text: str, turn_act: str) -> bool:
        """Return True if this turn satisfies the resolution condition."""
        if self.required_turn_acts and turn_act not in self.required_turn_acts:
            return False
        if self.requires_candidate_match:
            return self._text_names_a_candidate(turn_text)
        return True

    def _text_names_a_candidate(self, text: str) -> bool:
        norm = text.lower()
        return any(c.lower() in norm for c in self.candidate_entities)


@dataclass
class ContinuationTransaction(SemanticTransaction):
    """A held interaction state pending user continuation or resolution (Phase 6).

    The invariant: the continuation response is always reconstructed from the
    stored ContinuationTransaction — the LLM is never called during a continuation
    turn. A closed commitment is never re-opened by any continuation turn.

    claim: the original claimed proposition that triggered the hold.
    trigger: the SemanticTransaction that produced this hold (lifecycle tracing).
    continuation_kind: category of hold.
        "clarification" — user must supply missing information.
        "refusal"       — held at a refusal; may resolve or escalate.
        "stop"          — terminal; interaction_open=False.
        "escalation"    — escalating to human review.
        "follow_up"     — post-refusal follow-up opportunity.
    commitment_closed: if True, no new PEF commitment may be written during this
        continuation turn. A closed commitment is NEVER re-opened.
    interaction_open: if True, the user can continue the conversation.
    resolution_condition: the predicate evaluated each turn to decide if the hold
        closes. None → hold never auto-closes.
    """
    trigger: SemanticTransaction | None = None
    continuation_kind: str = "clarification"
    commitment_closed: bool = True
    interaction_open: bool = True
    resolution_condition: ResolutionCondition | None = None

    def applies(self, turn_act: str, turn_text: str) -> bool:
        """Return True if this continuation is active and this turn satisfies the condition."""
        if not self.interaction_open:
            return False
        if self.resolution_condition is None:
            return False
        return self.resolution_condition.matches(turn_text, turn_act)

    def escalate(self) -> ContinuationTransaction:
        """Return an escalated copy: continuation_kind='escalation', interaction_open=False."""
        import dataclasses as _dc
        return _dc.replace(
            self,
            continuation_kind="escalation",
            interaction_open=False,
            resolution_condition=None,
        )

    @classmethod
    def from_pending_dict(
        cls,
        pending: dict,
        claim: ExtractedClaim | None = None,
    ) -> ContinuationTransaction:
        """Back-compat factory: construct from an existing pending_clarification dict.

        Preserves the hold schema while adding typed predicate structure.
        """
        constraint = str(pending.get("failed_constraint") or "UNRESOLVED_REFERENT")
        candidates: list[str] = list(pending.get("candidate_entities") or [])

        # UNRESOLVED_REFERENT: any CLARIFY/QUERY closes (no candidate match required)
        # UNRESOLVED_COMPARAND: must name a candidate comparand
        # Other constraint types: require candidate match when candidates are known
        requires_match = constraint != "UNRESOLVED_REFERENT" and bool(candidates)
        condition = ResolutionCondition(
            constraint_kind=constraint,
            candidate_entities=candidates,
            required_turn_acts=["CLARIFY", "QUERY"],
            requires_candidate_match=requires_match,
        )

        _claim = claim or ExtractedClaim(
            subject="__continuation__",
            relation="HOLD",
            obj=str(pending.get("original_question") or "pending"),
            span=Span.PRESENT,
            negated=False,
            evidence="",
        )

        return cls(
            claim=_claim,
            allowed_commit=False,
            held_reason="CONTINUATION",
            trigger=None,
            continuation_kind="clarification",
            commitment_closed=True,
            interaction_open=True,
            resolution_condition=condition,
        )


@dataclass
class ExtractionResult:
    """Result of extracting from a single text input."""
    claims: list[ExtractedClaim] = field(default_factory=list)
    entity_mentions: list[str] = field(default_factory=list)
    span: Span = Span.PRESENT
    pronoun_candidates: dict[str, str] = field(default_factory=dict)
        # "she@token_5" -> best-guess entity name
    ambiguous_referents: list[str] = field(default_factory=list)
        # Possessive pronouns that cannot be uniquely resolved (2+ same-category antecedents)
    comparative_ambiguities: list[ComparativeAmbiguity] = field(default_factory=list)
        # Comparative adjectives whose comparand cannot be uniquely determined
    extraction_error: dict | None = None
        # When set: parse failed (e.g. JSON_DECODE_ERROR). Keys: reason, snippet, raw_preview.
    interpretation_limit: dict | None = None
        # When set: the input is outside what this backend can interpret.
        # Keys: kind, detail, and kind-specific fields. Claims are parser
        # output only and must not be admitted as the user's assertions.
    financial_determination_probe: dict | None = None
        # Parser-grounded probe: act, concern, personal link, determination vs
        # response_format. status is established | partial | none.
