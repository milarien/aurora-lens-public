"""Utterance → HazardRequestFrame via lexicon hits and role assignment."""

from __future__ import annotations

from aurora_lens.verify.hazard.lexicon_index import LexiconIndex
from aurora_lens.verify.hazard.normalize import normalize_hazard_text, tokenize_hazard_text
from aurora_lens.verify.hazard.schema import (
    HAZARD_FRAME_SCHEMA_VERSION,
    ActionSlot,
    AmbiguityState,
    EvidenceSpan,
    FramingKind,
    HazardOntology,
    HazardRequestFrame,
    LexiconHit,
    LexiconKind,
    MatchConfidence,
    OrganismSlot,
    OutcomeKind,
    Procedurality,
    ProcessFamily,
    ProcessSlot,
    Severity,
    SubstanceSlot,
    TRANSFORM_PROCESS_FAMILIES,
    max_outcome,
    max_severity,
)

# Horticultural product/object cues — not hazardous substance products.
_HORTICULTURAL_OBJECT_TOKENS: frozenset[str] = frozenset(
    {
        "seed",
        "seeds",
        "seedling",
        "seedlings",
        "cutting",
        "cuttings",
        "bulb",
        "bulbs",
        "rootstock",
        "scion",
        "graft",
    }
)

# Tokens that mark the process product as a hazardous chemical target.
_HAZARD_PRODUCT_TOKENS: frozenset[str] = frozenset(
    {
        "toxin",
        "toxins",
        "poison",
        "poisons",
        "agent",
        "agents",
        "concentrate",
        "concentrated",
        "extract",
        "oil",
        "glycoside",
        "glycosides",
    }
)


def _surface(tokens: tuple[str, ...], start: int, end: int) -> str:
    return " ".join(tokens[start:end])


def _pick_first(hits: tuple[LexiconHit, ...], kind: LexiconKind) -> LexiconHit | None:
    for h in hits:
        if h.kind == kind:
            return h
    return None


def _all_of(hits: tuple[LexiconHit, ...], kind: LexiconKind) -> list[LexiconHit]:
    return [h for h in hits if h.kind == kind]


def _window_tokens(tokens: tuple[str, ...], start: int, end: int, radius: int = 4) -> frozenset[str]:
    lo = max(0, start - radius)
    hi = min(len(tokens), end + radius)
    return frozenset(tokens[lo:hi])


def _process_targets_hazardous_product(
    tokens: tuple[str, ...],
    process_hit: LexiconHit,
    substance_hits: list[LexiconHit],
    ontology: HazardOntology,
) -> bool:
    """True when the operational process's object/product is a hazard substance/derivative."""
    del ontology  # substance binding uses hits; ontology reserved for future typed objects
    nearby = _window_tokens(tokens, process_hit.start, process_hit.end, radius=5)

    # Substance (or class-named product) immediately after the process verb
    for sh in substance_hits:
        if process_hit.end <= sh.start <= process_hit.end + 2:
            return True

    # Horticultural objects (seeds/cuttings) are not hazardous products
    if nearby & _HORTICULTURAL_OBJECT_TOKENS and not (nearby & _HAZARD_PRODUCT_TOKENS):
        return False

    if nearby & _HAZARD_PRODUCT_TOKENS:
        return True

    # Named substance near process without horticultural object → hazardous product
    if substance_hits and not (nearby & _HORTICULTURAL_OBJECT_TOKENS):
        for sh in substance_hits:
            if abs(sh.start - process_hit.start) <= 5:
                return True

    # Chem transform lemma with only an organism source and no horticultural object:
    # treat as hazardous (extract from oleander / distill from leaves).
    if not (nearby & _HORTICULTURAL_OBJECT_TOKENS):
        return True
    return False


def _select_process_hit(
    hits: tuple[LexiconHit, ...],
    tokens: tuple[str, ...],
    substance_hits: list[LexiconHit],
    ontology: HazardOntology,
) -> LexiconHit | None:
    """Prefer horticultural cultivate processes; demote chem transforms without hazard product."""
    process_hits = _all_of(hits, LexiconKind.PROCESS)
    if not process_hits:
        return None

    hort: list[LexiconHit] = []
    chem: list[LexiconHit] = []
    other: list[LexiconHit] = []
    for ph in process_hits:
        prec = ontology.processes[ph.record_id]
        if prec.family == ProcessFamily.OTHER_TRANSFORM and OutcomeKind.HORTICULTURAL in prec.implies_outcomes:
            hort.append(ph)
        elif prec.family in TRANSFORM_PROCESS_FAMILIES:
            if _process_targets_hazardous_product(tokens, ph, substance_hits, ontology):
                chem.append(ph)
            # else: chem lemma present but object is horticultural — ignore as hazard transform
        else:
            other.append(ph)

    if hort:
        return hort[0]
    if chem:
        # Prefer the chem transform nearest a substance mention so generic
        # lemmas (e.g. residual production cues) do not shadow a concrete
        # extract/isolate/distil verb bound to the hazard product.
        if substance_hits:
            def _chem_key(ph: LexiconHit) -> tuple[int, int, int]:
                dist = min(abs(ph.start - sh.start) for sh in substance_hits)
                return (dist, -(ph.end - ph.start), ph.start)

            chem.sort(key=_chem_key)
        return chem[0]
    if other:
        return other[0]
    return None


def parse_hazard_request(
    user_input: str,
    ontology: HazardOntology,
    index: LexiconIndex | None = None,
) -> HazardRequestFrame:
    """Deterministically parse a user request into a hazard frame."""
    raw_norm = normalize_hazard_text(user_input)
    tokens = tokenize_hazard_text(user_input)
    idx = index if index is not None else LexiconIndex(ontology)
    hits = idx.match(tokens)

    evidence: list[EvidenceSpan] = []

    action_hits = _all_of(hits, LexiconKind.ACTION)
    action_slot: ActionSlot | None = None
    procedurality = Procedurality.INFORMATIONAL
    outcome = OutcomeKind.NONE
    _PROC_RANK = {
        Procedurality.INFORMATIONAL: 0,
        Procedurality.IDENTIFYING: 1,
        Procedurality.PROCEDURAL_HOWTO: 2,
        Procedurality.IMPERATIVE_EXECUTE: 3,
    }
    # Prefer horticultural/informational actions over bare howto when both match
    # overlapping intent (e.g. "how to grow" vs residual howto ranking).
    best_action: LexiconHit | None = None
    best_rank = -1
    for action_hit in action_hits:
        arec = ontology.actions[action_hit.record_id]
        rank = _PROC_RANK[arec.procedurality]
        # Horticultural outcome actions win ties / override bare howto when longer or equal
        if arec.outcome_prior == OutcomeKind.HORTICULTURAL:
            rank = max(rank, _PROC_RANK[Procedurality.IDENTIFYING]) + 10
        if rank > best_rank:
            best_rank = rank
            best_action = action_hit
            procedurality = arec.procedurality
            outcome = max_outcome(outcome, arec.outcome_prior)
        else:
            outcome = max_outcome(outcome, arec.outcome_prior)
    if best_action is not None:
        arec = ontology.actions[best_action.record_id]
        # Keep the action's native procedurality for horticultural cultivate how-tos
        if arec.outcome_prior == OutcomeKind.HORTICULTURAL:
            procedurality = arec.procedurality
        action_slot = ActionSlot(
            action_id=arec.id,
            surface_span=(best_action.start, best_action.end),
            procedurality=procedurality,
            outcome_prior=arec.outcome_prior,
        )
        evidence.append(
            EvidenceSpan(
                field="action",
                start=best_action.start,
                end=best_action.end,
                surface=_surface(tokens, best_action.start, best_action.end),
                lexicon_id=arec.id,
            )
        )

    # Do NOT upgrade every "how to"/"how do" substring to hazard procedural intent.
    # Bare howto is sufficient only when a hazardous transform process is already present
    # (e.g. "explain how to manufacture sarin") — handled via action hits + process binding.

    substance_hits = _all_of(hits, LexiconKind.SUBSTANCE)
    organism_hits = _all_of(hits, LexiconKind.ORGANISM)
    class_hits = _all_of(hits, LexiconKind.HAZARD_CLASS)

    process_hit = _select_process_hit(hits, tokens, substance_hits, ontology)
    process_slot: ProcessSlot | None = None
    if process_hit is not None:
        prec = ontology.processes[process_hit.record_id]
        process_slot = ProcessSlot(
            process_id=prec.id,
            family=prec.family,
            surface_span=(process_hit.start, process_hit.end),
            surface=_surface(tokens, process_hit.start, process_hit.end),
            implies_outcomes=prec.implies_outcomes,
        )
        for implied in prec.implies_outcomes:
            outcome = max_outcome(outcome, implied)
        evidence.append(
            EvidenceSpan(
                field="transformation",
                start=process_hit.start,
                end=process_hit.end,
                surface=process_slot.surface,
                lexicon_id=prec.id,
            )
        )

    # Reinforce concentration only when not already in horticultural cultivate mode
    if outcome != OutcomeKind.HORTICULTURAL:
        if "concentrated" in tokens or "concentrate" in tokens or "concentration" in tokens:
            outcome = max_outcome(outcome, OutcomeKind.CONCENTRATION)

    framing_hit = _pick_first(hits, LexiconKind.FRAMING)
    user_framing = FramingKind.BARE
    if framing_hit is not None:
        frec = ontology.framings[framing_hit.record_id]
        user_framing = frec.framing
        evidence.append(
            EvidenceSpan(
                field="user_framing",
                start=framing_hit.start,
                end=framing_hit.end,
                surface=_surface(tokens, framing_hit.start, framing_hit.end),
                lexicon_id=frec.id,
            )
        )

    # Horticultural process/outcome implies horticultural framing when bare
    if outcome == OutcomeKind.HORTICULTURAL and user_framing == FramingKind.BARE:
        user_framing = FramingKind.HORTICULTURAL
    if (
        process_slot is not None
        and process_slot.family == ProcessFamily.OTHER_TRANSFORM
        and OutcomeKind.HORTICULTURAL in process_slot.implies_outcomes
        and user_framing == FramingKind.BARE
    ):
        user_framing = FramingKind.HORTICULTURAL

    substance_slot: SubstanceSlot | None = None
    organism_slot: OrganismSlot | None = None
    hazard_class_ids: list[str] = []
    ambiguity = AmbiguityState.NONE
    severity = Severity.NONE

    if substance_hits:
        best: LexiconHit | None = None
        best_sev = Severity.NONE
        tied = False
        for sh in substance_hits:
            srec = ontology.substances[sh.record_id]
            sev = Severity.NONE
            for cid in srec.hazard_class_ids:
                crec = ontology.hazard_classes[cid]
                sev = max_severity(sev, crec.default_severity)
            if best is None or _sev_rank(sev) > _sev_rank(best_sev):
                best = sh
                best_sev = sev
                tied = False
            elif best is not None and sh.record_id != best.record_id and sev == best_sev:
                tied = True
        assert best is not None
        if tied:
            ambiguity = AmbiguityState.SUBSTANCE_UNRESOLVED
            substance_slot = SubstanceSlot(
                substance_id=None,
                surface_span=(best.start, best.end),
                match_confidence=MatchConfidence.UNRESOLVED,
                surface=_surface(tokens, best.start, best.end),
            )
        else:
            srec = ontology.substances[best.record_id]
            substance_slot = SubstanceSlot(
                substance_id=srec.id,
                surface_span=(best.start, best.end),
                match_confidence=MatchConfidence.ALIAS,
                surface=_surface(tokens, best.start, best.end),
            )
            hazard_class_ids.extend(srec.hazard_class_ids)
            severity = best_sev
            if srec.source_organism_ids:
                oid = srec.source_organism_ids[0]
                orec = ontology.organisms[oid]
                oh = next(
                    (h for h in organism_hits if h.record_id == oid),
                    None,
                )
                if oh is not None:
                    organism_slot = OrganismSlot(
                        organism_id=oid,
                        surface_span=(oh.start, oh.end),
                        surface=_surface(tokens, oh.start, oh.end),
                    )
                else:
                    organism_slot = OrganismSlot(
                        organism_id=oid,
                        surface_span=(best.start, best.end),
                        surface=orec.id,
                    )
            evidence.append(
                EvidenceSpan(
                    field="substance",
                    start=best.start,
                    end=best.end,
                    surface=substance_slot.surface,
                    lexicon_id=srec.id,
                )
            )
    elif organism_hits:
        oh = organism_hits[0]
        orec = ontology.organisms[oh.record_id]
        organism_slot = OrganismSlot(
            organism_id=orec.id,
            surface_span=(oh.start, oh.end),
            surface=_surface(tokens, oh.start, oh.end),
        )
        evidence.append(
            EvidenceSpan(
                field="source_organism",
                start=oh.start,
                end=oh.end,
                surface=organism_slot.surface,
                lexicon_id=orec.id,
            )
        )
        assoc = orec.associated_substance_ids
        if len(assoc) == 1:
            srec = ontology.substances[assoc[0]]
            substance_slot = SubstanceSlot(
                substance_id=srec.id,
                surface_span=(oh.start, oh.end),
                match_confidence=MatchConfidence.ALIAS,
                surface=srec.canonical_name,
            )
            hazard_class_ids.extend(srec.hazard_class_ids)
            for cid in srec.hazard_class_ids:
                severity = max_severity(
                    severity, ontology.hazard_classes[cid].default_severity
                )
        elif len(assoc) > 1:
            ambiguity = AmbiguityState.SUBSTANCE_UNRESOLVED
            substance_slot = SubstanceSlot(
                substance_id=None,
                surface_span=(oh.start, oh.end),
                match_confidence=MatchConfidence.UNRESOLVED,
                surface=organism_slot.surface,
            )
            for sid in assoc:
                srec = ontology.substances[sid]
                hazard_class_ids.extend(srec.hazard_class_ids)
                for cid in srec.hazard_class_ids:
                    severity = max_severity(
                        severity, ontology.hazard_classes[cid].default_severity
                    )
    elif class_hits:
        ch = class_hits[0]
        crec = ontology.hazard_classes[ch.record_id]
        hazard_class_ids.append(crec.id)
        severity = crec.default_severity
        substance_slot = SubstanceSlot(
            substance_id=None,
            surface_span=(ch.start, ch.end),
            match_confidence=MatchConfidence.CLASS_ONLY,
            surface=_surface(tokens, ch.start, ch.end),
        )
        evidence.append(
            EvidenceSpan(
                field="hazard_class",
                start=ch.start,
                end=ch.end,
                surface=substance_slot.surface,
                lexicon_id=crec.id,
            )
        )

    seen: set[str] = set()
    uniq_classes: list[str] = []
    for cid in hazard_class_ids:
        if cid not in seen:
            seen.add(cid)
            uniq_classes.append(cid)

    if (
        process_slot is not None
        and action_slot is None
        and tokens
        and tokens[0] in {"make", "create", "build", "produce", "synthesize", "synthesise"}
    ):
        procedurality = Procedurality.IMPERATIVE_EXECUTE

    # PROCESS_UNRESOLVED only when a hazardous chem transform is expected but missing —
    # not for horticultural cultivate / informational plant-care how-tos.
    horticultural_mode = (
        outcome == OutcomeKind.HORTICULTURAL
        or user_framing == FramingKind.HORTICULTURAL
        or (
            process_slot is not None
            and process_slot.family == ProcessFamily.OTHER_TRANSFORM
            and OutcomeKind.HORTICULTURAL in process_slot.implies_outcomes
        )
    )
    if (
        process_slot is None
        and procedurality in (
            Procedurality.PROCEDURAL_HOWTO,
            Procedurality.IMPERATIVE_EXECUTE,
        )
        and not horticultural_mode
    ):
        if uniq_classes or substance_slot or organism_slot:
            if ambiguity == AmbiguityState.NONE:
                ambiguity = AmbiguityState.PROCESS_UNRESOLVED

    # Invariant: hazardous substance/class + chem transform process ⇒ operational
    # intent. Interrogative surface variation ("how to", "how do I", "best way to",
    # "could someone", "tell me the steps for") must not downgrade this to a
    # benign identifying/informational read. Pure identification without a
    # transform process (e.g. "what is oleandrin") remains non-operational.
    if (
        not horticultural_mode
        and process_slot is not None
        and process_slot.family in TRANSFORM_PROCESS_FAMILIES
        and (substance_slot is not None or uniq_classes)
        and procedurality
        in (Procedurality.INFORMATIONAL, Procedurality.IDENTIFYING)
    ):
        procedurality = Procedurality.PROCEDURAL_HOWTO

    return HazardRequestFrame(
        schema_version=HAZARD_FRAME_SCHEMA_VERSION,
        raw_norm=raw_norm,
        tokens=tokens,
        action=action_slot,
        substance=substance_slot,
        source_organism=organism_slot,
        transformation=process_slot,
        intended_outcome=outcome,
        hazard_class_ids=tuple(uniq_classes),
        procedurality=procedurality,
        consequence_severity=severity,
        user_framing=user_framing,
        ambiguity=ambiguity,
        evidence_spans=tuple(evidence),
    )


def _sev_rank(sev: Severity) -> int:
    order = {
        Severity.NONE: 0,
        Severity.UNKNOWN: 1,
        Severity.MILD: 2,
        Severity.MODERATE: 3,
        Severity.SEVERE: 4,
        Severity.LETHAL: 5,
    }
    return order[sev]
