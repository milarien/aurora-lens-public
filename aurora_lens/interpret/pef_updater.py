"""PEF updater — apply extraction results to PEF state.

Backend-agnostic: any ExtractionResult (from spaCy, LLM, or anything else)
feeds through the same updater. Zero spaCy dependency.

Extraction proposes deltas; this module applies them to the existing world.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aurora_lens.pef.span import Span
from aurora_lens.pef.state import (
    PEFState,
    Relationship,
    canonicalize_relation,
    object_has_business_remit_marker,
)
from aurora_lens.interpret.schema import (
    ExtractedClaim,
    ExtractionResult,
    PossessionTransaction,
    SemanticTransaction,
)
from aurora_lens.pef.unresolved_referents import blocked_tokens_for_admission
from aurora_lens.pef_admission_debug import log_update_pef_enter, log_update_pef_exit

from aurora_lens.interpret.pef_admission import (
    PEFAdmissionDecision,
    PEFAdmissionResult,
    adjudicate_update_pef_aggregate,
    finalized_mutation_aggregate,
    make_admission_evidence,
)

if TYPE_CHECKING:
    from aurora_lens.interpret.turn_act import TurnAct


_PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "its", "their"}
_SMALL_NUMBER_WORDS: frozenset[str] = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    }
)

# Generic concept labels that must not become PEF entity placeholders.
# These are category-level abbreviations (e.g. in "AI governance", "LLM safety").
# Registering them as unresolved placeholders causes false UNSUPPORTED_ATTRIBUTE
# flags when the LLM makes ordinary claims about the concept.
_GENERIC_CONCEPT_LABELS: frozenset[str] = frozenset({
    "ai", "ml", "llm", "nlp", "api", "os", "iot", "ar", "vr",
    "ui", "ux", "ci", "cd", "qa", "hr", "pr", "llms",
})

# Possessive determiners that head NPs ("his dog"): must not mint PEF entities
# while the antecedent is unresolved (ambiguous_referents or missing discourse binding).
_POSSESSIVE_DETERMINER_LEMMAS: frozenset[str] = frozenset({"his", "her", "their", "its"})

# Definite description prefixes — subjects starting with these require a prior
# antecedent; they must NOT be auto-resolved as new concept literals.
_DEFINITE_PREFIXES: tuple[str, ...] = ("the ", "that ", "this ", "those ", "these ")


def _is_definite_description(s: str) -> bool:
    """Return True if s is a multi-word definite description (starts with 'the/that/this/...')."""
    lower = s.strip().lower()
    return any(lower.startswith(p) for p in _DEFINITE_PREFIXES)


def _possessive_lemmas_in_surface(surface: str) -> frozenset[str]:
    """Whole-word possessive-determiner lemmas in `surface` (punctuation stripped)."""
    found: set[str] = set()
    for raw in (surface or "").strip().split():
        t = raw.strip().lower().strip(".,!?;:()[]\"'`")
        if t in _POSSESSIVE_DETERMINER_LEMMAS:
            found.add(t)
    return frozenset(found)


def _surface_is_blocked_possessive_np_without_resolution(
    surface: str,
    ambiguous_referents: list[str] | tuple[str, ...],
    pef: PEFState,
) -> bool:
    """True → do not mint a PEF entity for this extraction surface."""
    lemmas = _possessive_lemmas_in_surface(surface)
    if not lemmas:
        return False
    amb_lower = {
        str(x).strip().lower()
        for x in ambiguous_referents
        if str(x).strip()
    }
    for lem in lemmas:
        if lem in amb_lower:
            return True
        if not (pef.discourse_referent_bindings.get(lem) or "").strip():
            return True
    return False


def _looks_like_counted_fungible_item(item_phrase: str) -> bool:
    """True for counted literals like ``5 lollipops`` (fungible quantities)."""
    s = str(item_phrase or "").strip()
    parts = s.split(None, 1)
    if len(parts) != 2:
        return False
    head, tail = parts[0].lower(), parts[1].strip()
    if not tail:
        return False
    return head.isdigit() or head in _SMALL_NUMBER_WORDS


def _relation_for_pef_write(claim: ExtractedClaim) -> str:
    """Canonical relation string as stored on PEF (business-remit HAS → MANAGE)."""
    rel = canonicalize_relation(claim.relation)
    if not object_has_business_remit_marker(claim.obj):
        return rel
    if rel in ("HAS", "MANAGE"):
        return "MANAGE"
    return rel


def _build_give_transaction(
    claim: ExtractedClaim,
    pef: PEFState,
    base_tx: SemanticTransaction,
) -> PossessionTransaction:
    """Build a PossessionTransaction for a GIVE claim, snapshotting prior count at build time."""
    from aurora_lens.state_native_engine.eval.possession_mutations import (
        _find_active_possession,
        _item_match_key_and_display,
    )
    from aurora_lens.state_native_engine.eval.inventory import normalize_possession_item_surface
    from aurora_lens.state_native_engine.lexical import item_key as _item_key

    obj_norm = normalize_possession_item_surface(str(claim.obj or "").strip())
    import re as _re
    m = _re.match(r"^(\d+(?:\.\d+)?)\s+(.+)$", obj_norm)
    qty: float | None = None
    item_key_str: str | None = None
    prior_snapshot: float | None = None

    if m:
        try:
            qty = float(m.group(1))
        except ValueError:
            pass
        item_key_str = _item_key(normalize_possession_item_surface(m.group(2)))

    if qty is not None and item_key_str:
        giver_ent = pef.find_entity_by_name(claim.subject)
        if giver_ent is not None:
            poss = _find_active_possession(pef, giver_ent, item_key_str)
            if poss is not None:
                prior_snapshot = poss[0]

    return PossessionTransaction(
        claim=claim,
        allowed_commit=base_tx.allowed_commit,
        held_reason=base_tx.held_reason,
        mutation_kind="transfer",
        quantity=qty,
        prior_count_snapshot=prior_snapshot,
        giver_name=claim.subject,
        item_key_str=item_key_str,
    )


def _build_return_transaction(
    claim: ExtractedClaim,
    pef: PEFState,
    base_tx: SemanticTransaction,
) -> SemanticTransaction:
    """Gate a RETURN claim on prior-possession precondition.

    RETURN fires only when the recipient (claim.obj resolved to an entity) has
    prior HAS records in PEF — confirming they previously owned the item.
    When the precondition fails the transaction is blocked (allowed_commit=False)
    rather than silently downgraded to a GIVE; the caller may still write delivery
    semantics via a separate GIVE claim extracted from the same evidence.

    If claim.obj does not resolve to a known entity we cannot verify the
    precondition, so RETURN is also blocked (RETURN_RECIPIENT_UNKNOWN).
    """
    recipient_ent = pef.find_entity_by_name(str(claim.obj or "").strip())
    if recipient_ent is None:
        import dataclasses as _dc
        return _dc.replace(
            base_tx,
            allowed_commit=False,
            held_reason="RETURN_RECIPIENT_UNKNOWN",
        )

    has_prior_possession = any(
        r.relation == "HAS"
        for r in pef.get_relationships_for_subject(recipient_ent.id)
    )
    if not has_prior_possession:
        import dataclasses as _dc
        return _dc.replace(
            base_tx,
            allowed_commit=False,
            held_reason="RETURN_NO_PRIOR_POSSESSION",
        )

    return base_tx


def _build_semantic_transactions(
    result: ExtractionResult,
    pef: PEFState | None = None,
) -> list[SemanticTransaction]:
    """Wrap each claim in a SemanticTransaction; block claims that must not commit to PEF.

    Phase 1: comparative IS claims with 2+ eligible comparands → UNRESOLVED_COMPARAND.
    Phase 2: possessive-NP subject claims with an unresolved possessor pronoun in
             result.ambiguous_referents → UNRESOLVED_POSSESSOR.
    Phase 3: when pef is provided, possession claims (HAS/GIVE) are wrapped in
             PossessionTransaction with prior-count snapshot captured at build time.

    Phase 1 takes priority: if a claim is blocked by Phase 1, Phase 2 is not checked.
    Non-blocked claims pass through with allowed_commit=True.
    """
    comparative_adjectives: frozenset[str] = frozenset(
        ca.adjective.lower()
        for ca in (result.comparative_ambiguities or [])
    )
    ambiguous_lower: frozenset[str] = frozenset(
        str(x).strip().lower()
        for x in (result.ambiguous_referents or [])
        if str(x).strip()
    )
    transactions: list[SemanticTransaction] = []
    for claim in result.claims:
        tx = SemanticTransaction(claim=claim)

        # Phase 1: comparative IS claim with unresolved comparand.
        if (
            comparative_adjectives
            and canonicalize_relation(claim.relation) == "IS"
            and str(claim.obj).strip().lower() in comparative_adjectives
        ):
            tx.allowed_commit = False
            tx.unresolved_comparand = str(claim.obj).strip()
            tx.held_reason = "UNRESOLVED_COMPARAND"

        # Phase 2 / 2b: possessive-NP or bare pronoun subject with unresolved referent.
        # Only fires when Phase 1 did not already block this claim.
        elif ambiguous_lower:
            poss_lemmas = _possessive_lemmas_in_surface(claim.subject)
            if poss_lemmas and (poss_lemmas & ambiguous_lower):
                # Phase 2: possessive-NP subject (e.g. "his wallet") — possessor is ambiguous.
                unresolved_poss = next(iter(poss_lemmas & ambiguous_lower))
                tx.allowed_commit = False
                tx.unresolved_possessor = unresolved_poss
                tx.held_reason = "UNRESOLVED_POSSESSOR"
            elif claim.subject.strip().lower() in ambiguous_lower:
                # Phase 2b: bare pronoun subject (e.g. "he") is itself an ambiguous token.
                tx.allowed_commit = False
                tx.held_reason = "UNRESOLVED_REFERENT"

        # Phase 3: possession classification — only when pef provided and claim not blocked.
        if pef is not None and tx.allowed_commit:
            rel_norm = canonicalize_relation(claim.relation)
            if rel_norm == "HAS":
                if (
                    not claim.negated
                    and not _looks_like_counted_fungible_item(claim.obj)
                ):
                    mutation_kind = "supersede"
                else:
                    mutation_kind = "acquire"
                tx = PossessionTransaction(
                    claim=claim,
                    allowed_commit=True,
                    held_reason=None,
                    mutation_kind=mutation_kind,
                )
            elif rel_norm == "GIVE":
                tx = _build_give_transaction(claim, pef, tx)
            elif rel_norm == "RETURN":
                tx = _build_return_transaction(claim, pef, tx)

        transactions.append(tx)
    return transactions


def update_pef(
    result: ExtractionResult,
    pef: PEFState,
    *,
    user_text: str | None = None,
    turn_act: TurnAct | None = None,
) -> PEFAdmissionResult:
    """Apply extraction results to PEF state.

    Returns a :class:`PEFAdmissionResult` documenting admission and orthogonal
    mutation tallies (:mod:`aurora_lens.interpret.pef_admission`). ``ADMIT`` is **not**
    a synonym for "state changed" — inspect ``world_state_mutation_count`` /
    ``continuation_state_mutation_count`` (and optionally their sum as ``mutation_count``).

    Creates/updates entities, adds relationships, records bindings.

    When ``user_text`` is provided and the turn act is not
    :attr:`~aurora_lens.interpret.turn_act.TurnAct.REVISE`, individual claims that would
    contradict existing grounded literals are skipped instead of appended.
    :attr:`~aurora_lens.interpret.turn_act.TurnAct.QUERY` is read-only for PEF only when
    there are no extracted **claims** (pure question): no SVO edges, pronoun bindings, or
    mention-driven entity creation. Mixed declarative + question (extractor may still
    emit claims from the non-question part) uses the same claim-admission path as
    other acts for those claims.

    Pass ``turn_act`` from the orchestrator (e.g. Lens) when it was already
    computed for the turn; otherwise ``user_text`` alone is classified once here.
    Callers that replay corpus or resolved bindings without user-facing text
    should omit ``user_text`` and ``turn_act`` (default).
    """
    from aurora_lens.interpret.turn_act import TurnAct, classify_turn_act

    _effective_act: TurnAct | None = turn_act
    if _effective_act is None and user_text is not None:
        _effective_act = classify_turn_act(user_text)

    if _effective_act == TurnAct.QUERY and not result.claims:
        query_admission_evidence: list = []
        log_update_pef_enter(
            user_text=user_text,
            claims_in=0,
            skip_all_claims=True,
        )
        log_update_pef_exit(
            claims_committed=0,
            claims_skipped_by_loop=0,
        )
        query_admission_evidence.append(
            make_admission_evidence(
                "U-Q0",
                scope="turn",
                detail="QUERY with no extraction claims — read-only turn for claim path.",
            )
        )
        w, c, m, summ = finalized_mutation_aggregate(0, 0)
        return PEFAdmissionResult(
            decision=PEFAdmissionDecision.ADMIT,
            write_intent=False,
            planned_mutation_slices=0,
            world_state_mutation_count=w,
            continuation_state_mutation_count=c,
            mutation_count=m,
            mutation_summary=summ,
            evidence=query_admission_evidence,
        )

    # Pronouns that the extraction has flagged as ambiguous must not be
    # committed to PEF via either path (pronoun_candidates or claims).
    # Admissibility has not licensed a binding for them; resolving by
    # recency would collapse the world state before the pre-LLM gate
    # has had a chance to intervene.
    _blocked_pronouns: frozenset[str] = frozenset(
        p.lower() for p in result.ambiguous_referents
    ) | blocked_tokens_for_admission(pef)

    continuation_ct = 0
    admission_evidence: list = []

    # Apply pronoun bindings — skip any whose base pronoun is ambiguous.
    # Key format is "{pronoun}@token_{i}"; split on "@" recovers the pronoun.
    for pronoun_key, entity_name in result.pronoun_candidates.items():
        if pronoun_key.split("@")[0].lower() in _blocked_pronouns:
            admission_evidence.append(
                make_admission_evidence(
                    "U-PR",
                    scope="pronoun_binding",
                    detail=str(pronoun_key),
                )
            )
            continue  # ambiguous — do not write turn_binding (continuation-state withheld)
        entity = pef.find_entity_by_name(entity_name)
        if entity:
            pef.resolve_pronoun(pronoun_key, entity.id)
            continuation_ct += 1

    # Process claims
    _skip_all_claims = False
    _ecf = None
    _check_per_claim_conflict = False
    if user_text is not None:
        from aurora_lens.interpret.revision_gate import (
            extraction_conflicts_grounded_pef,
            user_text_conflicts_grounded_pef,
        )

        if _effective_act != TurnAct.REVISE:
            if user_text_conflicts_grounded_pef(pef, user_text)[0]:
                _skip_all_claims = True
        _ecf = extraction_conflicts_grounded_pef
        _check_per_claim_conflict = _effective_act != TurnAct.REVISE

    n_claims_in = len(result.claims)
    log_update_pef_enter(
        user_text=user_text,
        claims_in=n_claims_in,
        skip_all_claims=_skip_all_claims,
    )
    transactions = _build_semantic_transactions(result, pef)

    slice_u_g1_logged = False
    world_ct = 0
    n_applied_claim_slices = 0

    slice_i = -1
    for tx in transactions:
        slice_i += 1
        if not tx.allowed_commit:
            held = getattr(tx, "held_reason", None) or ""
            veto_map = {
                "UNRESOLVED_COMPARAND": "U-TX1",
                "UNRESOLVED_POSSESSOR": "U-TX2",
                "UNRESOLVED_REFERENT": "U-TX3",
            }.get(str(held), "SEMANTIC_HELD_UNKNOWN")
            admission_evidence.append(
                make_admission_evidence(veto_map, slice_index=slice_i, held_reason=str(held) or None)
            )
            continue
        claim = tx.claim
        if _skip_all_claims:
            if not slice_u_g1_logged:
                admission_evidence.append(make_admission_evidence("U-G1", scope="turn"))
                slice_u_g1_logged = True
            continue
        if _ecf is not None and _check_per_claim_conflict:
            _single = ExtractionResult(
                claims=[claim],
                entity_mentions=[],
                ambiguous_referents=list(result.ambiguous_referents),
                span=result.span,
            )
            if _ecf(pef, _single)[0]:
                admission_evidence.append(
                    make_admission_evidence("U-G2", slice_index=slice_i, scope="claim")
                )
                continue

        subject_name = claim.subject

        # Resolve pronoun subjects (ambiguous tokens are already blocked above via
        # allowed_commit=False in _build_semantic_transactions, so no extra guard needed).
        if subject_name.lower() in _PRONOUNS:
            resolved_name = _resolve_single_pronoun(subject_name, pef)
            if resolved_name:
                subject_name = resolved_name

        if _surface_is_blocked_possessive_np_without_resolution(
            subject_name,
            result.ambiguous_referents,
            pef,
        ):
            admission_evidence.append(make_admission_evidence("U-POSS", slice_index=slice_i))
            # Unresolved possessive-headed NP ("His dog"): keep in extraction for audit
            # / pending blocked_claims, but never mint entities or HAS edges from it.
            continue

        # Route PossessionTransaction (HAS) through validate+commit — atomic, no inline supersession.
        if isinstance(tx, PossessionTransaction) and canonicalize_relation(claim.relation) == "HAS":
            # Propagate resolved subject (pronoun → entity name) into the transaction claim
            # so commit() looks up the correct entity rather than the raw pronoun.
            if subject_name != claim.subject:
                import dataclasses as _dc
                tx = _dc.replace(tx, claim=_dc.replace(tx.claim, subject=subject_name))
            _rels_prev = len(pef.relationships)
            stop = tx.validate(pef)
            if stop is None:
                tx.commit(pef)
                world_ct += len(pef.relationships) - _rels_prev
                n_applied_claim_slices += 1
                # Resolve subject entity for echo-trace bookkeeping
                _subj_e = pef.find_entity_by_name(subject_name)
                if _subj_e is not None:
                    pef.record_echo_trace(_subj_e.id, claim.evidence)
                    if _subj_e.resolved is False:
                        _subj_e.resolved = True
            else:
                admission_evidence.append(
                    make_admission_evidence(
                        "U-PTX",
                        slice_index=slice_i,
                        detail=str(stop),
                        held_reason="POSSESSION_VALIDATE_STOP",
                    )
                )
            continue

        # Get or create subject entity
        subj_entity, subject_created = pef.get_or_create_entity(subject_name)
        if subject_created:
            world_ct += 1

        # Admit bare nouns and proper nouns as resolved concept literals.
        # Entities previously created as unresolved placeholders (from
        # entity_mentions) are promoted here when a claim is asserted about them.
        # Pronouns and definite descriptions are excluded — they require explicit
        # PEF grounding and must not be silently resolved.
        if subject_name.lower() not in _PRONOUNS and not _is_definite_description(subject_name):
            subj_entity.resolved = True

        # Determine if object is an entity reference or literal
        obj_entity = pef.find_entity_by_name(claim.obj)
        stored_relation = _relation_for_pef_write(claim)

        if obj_entity:
            rel = Relationship(
                subject_id=subj_entity.id,
                relation=stored_relation,
                object_entity_id=obj_entity.id,
                object_literal=None,
                span=claim.span,
                source_turn=pef.current_turn,
                evidence=claim.evidence,
                negated=claim.negated,
                provenance=claim.provenance,
                extractor_backend=claim.extractor_backend,
                document_id=claim.document_id,
                document_locator=claim.document_locator,
            )
        else:
            rel = Relationship(
                subject_id=subj_entity.id,
                relation=stored_relation,
                object_entity_id=None,
                object_literal=claim.obj,
                span=claim.span,
                source_turn=pef.current_turn,
                evidence=claim.evidence,
                negated=claim.negated,
                provenance=claim.provenance,
                extractor_backend=claim.extractor_backend,
                document_id=claim.document_id,
                document_locator=claim.document_locator,
            )

        pef.add_relationship(rel)
        world_ct += 1
        n_applied_claim_slices += 1

        # Record echo traces
        pef.record_echo_trace(subj_entity.id, claim.evidence)
        if subj_entity.resolved is False and canonicalize_relation(claim.relation) != "IS":
            # Asserting a fact about the entity resolves it
            subj_entity.resolved = True

    skipped_loop = max(0, n_claims_in - n_applied_claim_slices)

    log_update_pef_exit(
        claims_committed=n_applied_claim_slices,
        claims_skipped_by_loop=skipped_loop,
    )

    # Create entity entries for mentioned-but-not-asserted entities.
    # Skip pronouns and generic concept labels (AI, ML, LLM, etc.).
    for name in result.entity_mentions:
        name_lower = name.lower()
        if name_lower in _PRONOUNS or name_lower in _GENERIC_CONCEPT_LABELS:
            admission_evidence.append(make_admission_evidence("U-M1", scope="entity_mention", detail=name_lower))
            continue
        if _surface_is_blocked_possessive_np_without_resolution(
            name,
            result.ambiguous_referents,
            pef,
        ):
            admission_evidence.append(make_admission_evidence("U-M2", scope="entity_mention", detail=name))
            continue
        # Resolution policy at creation:
        #
        #   Conceptual rule:
        #   - Proper-name / named-entity mention with no competing live candidate
        #     → resolved=True.  It is a single named target of inquiry: the user
        #     explicitly named it, there is no ambiguity, no missing antecedent.
        #     The LLM answering about it is a direct response, not hallucination.
        #   - Underspecified referent (definite description: "the X", "that Y",
        #     "this Z") → resolved=False.  It presupposes a prior antecedent that
        #     has not yet been established; the checker's "Introduced policy" must
        #     fire if the LLM fabricates attributes for it.
        #
        #   Practical signal: a definite or demonstrative determiner prefix is a
        #   reliable indicator of underspecification in NER/PROPN output.  Proper
        #   names without such a prefix are specific named targets.
        #
        #   Note: if the entity already exists in PEF (same name), get_or_create_entity
        #   returns the existing entry without touching its resolved flag — this only
        #   affects the initial creation of a new entity.
        resolved_at_creation = not _is_definite_description(name)
        _, ent_created = pef.get_or_create_entity(name, resolved=resolved_at_creation)
        if ent_created:
            world_ct += 1

    agg = adjudicate_update_pef_aggregate(admission_evidence)
    ww, cc, agg_mut, summary = finalized_mutation_aggregate(world_ct, continuation_ct)
    summary["claim_slices_applied"] = n_applied_claim_slices

    planned = len(result.claims) + len(result.entity_mentions) + len(result.pronoun_candidates)

    result_out = PEFAdmissionResult(
        decision=agg,
        write_intent=True,
        planned_mutation_slices=planned,
        world_state_mutation_count=ww,
        continuation_state_mutation_count=cc,
        mutation_count=agg_mut,
        mutation_summary=summary,
        evidence=list(admission_evidence),
    )
    return result_out


def apply_prepopulated_claims(
    claims: list[ExtractedClaim],
    pef: PEFState,
    document_id: str,
    document_locator: str | None = None,
) -> None:
    """Apply claims from a pre-populated corpus/document to PEF.

    Forces provenance="pre_populated", keeps extractor_backend from each claim,
    sets document_id and document_locator. Uses turn 0 for source_turn.
    """
    # Override provenance for all claims
    prepop_claims = [
        ExtractedClaim(
            subject=c.subject,
            relation=c.relation,
            obj=c.obj,
            span=c.span,
            negated=c.negated,
            evidence=c.evidence,
            provenance="pre_populated",
            extractor_backend=c.extractor_backend,
            document_id=document_id,
            document_locator=document_locator,
        )
        for c in claims
    ]
    # Use turn 0 for pre-populated facts (before conversation)
    saved_turn = pef.current_turn
    pef.current_turn = 0
    try:
        result = ExtractionResult(
            claims=prepop_claims,
            entity_mentions=[],
            span=prepop_claims[0].span if prepop_claims else Span.PRESENT,
        )
        update_pef(result, pef, user_text=None)
    finally:
        pef.current_turn = saved_turn


def _resolve_single_pronoun(pronoun: str, pef: PEFState) -> str | None:
    """Resolve a pronoun to the most recently active matching entity.

    Uses recency heuristic. Returns None if no resolved entities exist.
    """
    resolved_entities = [
        e for e in pef.entities.values() if e.resolved
    ]
    if not resolved_entities:
        return None

    by_recency = sorted(
        resolved_entities,
        key=lambda e: e.turn_last_active,
        reverse=True,
    )

    if by_recency:
        return by_recency[0].name

    return None

