"""Hazard frame → admit / ask / stop decision with auditable trace."""

from __future__ import annotations

from aurora_lens.verify.hazard.schema import (
    AmbiguityState,
    DecisionTrace,
    FramingKind,
    HazardDecision,
    HazardEvaluationResult,
    HazardOntology,
    HazardRequestFrame,
    OutcomeKind,
    Procedurality,
    ProcessFamily,
    Severity,
    TRANSFORM_PROCESS_FAMILIES,
    severity_at_least,
)

RULE_HAZARD_OPERATIONAL_TRANSFORM = "blocked.request.hazard_operational_transform"
RULE_HAZARD_SUBSTANCE_UNRESOLVED = "blocked.request.hazard_substance_unresolved"
RULE_BIOWEAPON_GENERIC_FABRICATION = "blocked.request.bioweapon_generic_fabrication"

_BENIGN_OUTCOMES: frozenset[OutcomeKind] = frozenset(
    {
        OutcomeKind.INFORMATION,
        OutcomeKind.IDENTIFICATION,
        OutcomeKind.TOXICITY_PROFILE,
        OutcomeKind.HISTORICAL,
        OutcomeKind.HORTICULTURAL,
    }
)

_BENIGN_FRAMINGS: frozenset[FramingKind] = frozenset(
    {
        FramingKind.EDUCATIONAL,
        FramingKind.HISTORICAL_ACADEMIC,
        FramingKind.HORTICULTURAL,
    }
)


def _is_bioweapon_frame(frame: HazardRequestFrame) -> bool:
    if frame.substance is not None and frame.substance.substance_id == "generic_bioweapon":
        return True
    return any(
        cid == "dangerous_biological_weapon" or "bioweapon" in cid
        for cid in frame.hazard_class_ids
    )


def _bioweapon_weapon_token_indices(tokens: tuple[str, ...]) -> list[int]:
    """Indices where a bioweapon / biological-weapon phrase begins."""
    idxs: list[int] = []
    n = len(tokens)
    i = 0
    while i < n:
        t = tokens[i]
        if t in ("bioweapon", "bioweapons"):
            idxs.append(i)
            i += 1
            continue
        if i + 1 < n:
            nxt = tokens[i + 1]
            if t in ("biological", "bio", "pathogen", "pathogenic") and nxt in (
                "weapon",
                "weapons",
            ):
                idxs.append(i)
                i += 2
                continue
        i += 1
    return idxs


def _is_vaccine_or_defensive_bioweapon_carve_out(frame: HazardRequestFrame) -> bool:
    """Public-health / vaccine / biodefense framing must not stop as fabrication.

    Matches the legacy blocked-request vaccine-before-weapon composition check,
    plus explicit DEFENSIVE_PROTECTIVE framing when no weaponisation process.
    """
    if not _is_bioweapon_frame(frame):
        return False
    if frame.user_framing == FramingKind.WEAPONISING:
        return False
    if (
        frame.transformation is not None
        and frame.transformation.family == ProcessFamily.WEAPONISATION
    ):
        return False
    if frame.intended_outcome == OutcomeKind.WEAPONISATION:
        return False

    tokens = frame.tokens
    weapon_idxs = _bioweapon_weapon_token_indices(tokens)
    if weapon_idxs:
        ws_min = min(weapon_idxs)
        # Vaccine / vaccination / biodefense strategy before the weapon phrase.
        protective_tokens = {
            "vaccine",
            "vaccines",
            "vaccination",
            "vaccinations",
            "biodefense",
            "biodefence",
            "prophylaxis",
            "immunization",
            "immunisation",
        }
        if any(t in protective_tokens and i < ws_min for i, t in enumerate(tokens)):
            return True
        # "strategy against biological weapons" / "defense against …"
        if "against" in tokens:
            against_idxs = [i for i, t in enumerate(tokens) if t == "against"]
            for ai in against_idxs:
                if ai < ws_min and any(
                    tokens[j] in {"strategy", "strategies", "defense", "defence", "protection"}
                    for j in range(0, ai)
                ):
                    return True

    if frame.user_framing == FramingKind.DEFENSIVE_PROTECTIVE:
        return True
    return False


def _stop_rule_id(frame: HazardRequestFrame) -> str:
    if _is_bioweapon_frame(frame):
        return RULE_BIOWEAPON_GENERIC_FABRICATION
    return RULE_HAZARD_OPERATIONAL_TRANSFORM


def _matrix_key(
    framing: FramingKind,
    procedurality: Procedurality,
    process_family: ProcessFamily,
    severity: Severity,
) -> str:
    return (
        f"{framing.value}:{procedurality.value}:"
        f"{process_family.value}:{severity.value}"
    )


def _lookup_matrix(
    ontology: HazardOntology,
    framing: FramingKind,
    procedurality: Procedurality,
    process_family: ProcessFamily,
    severity: Severity,
) -> tuple[HazardDecision | None, str | None]:
    """Exact key then fallback ladder: drop framing → drop process → None."""
    rows = {
        _matrix_key(r.framing, r.procedurality, r.process_family, r.severity): r.decision
        for r in ontology.decision_matrix
    }
    keys = [
        _matrix_key(framing, procedurality, process_family, severity),
        _matrix_key(FramingKind.BARE, procedurality, process_family, severity),
        _matrix_key(framing, procedurality, ProcessFamily.NONE, severity),
        _matrix_key(FramingKind.BARE, procedurality, ProcessFamily.NONE, severity),
    ]
    for key in keys:
        if key in rows:
            return rows[key], key
    return None, None


def _has_hazard_signal(frame: HazardRequestFrame) -> bool:
    if frame.hazard_class_ids:
        return True
    if frame.substance is not None:
        return True
    if frame.source_organism is not None:
        return True
    if (
        frame.transformation is not None
        and frame.transformation.family == ProcessFamily.WEAPONISATION
    ):
        return True
    if frame.user_framing == FramingKind.WEAPONISING:
        return True
    return False


def _block_on_transform(frame: HazardRequestFrame, ontology: HazardOntology) -> bool:
    if not frame.hazard_class_ids:
        # Class-only / organism unresolved may still carry classes
        return False
    return any(
        ontology.hazard_classes[cid].block_on_procedural_transform
        for cid in frame.hazard_class_ids
        if cid in ontology.hazard_classes
    )


def _is_procedural(frame: HazardRequestFrame) -> bool:
    return frame.procedurality in (
        Procedurality.PROCEDURAL_HOWTO,
        Procedurality.IMPERATIVE_EXECUTE,
    )


def _process_family(frame: HazardRequestFrame) -> ProcessFamily:
    if frame.transformation is None:
        return ProcessFamily.NONE
    return frame.transformation.family


def decide_hazard_request(
    frame: HazardRequestFrame,
    ontology: HazardOntology,
) -> HazardEvaluationResult:
    """Apply deterministic R1–R7 rules; matrix used as explicit admit/stop table."""
    if not _has_hazard_signal(frame):
        trace = DecisionTrace(
            frame=frame,
            decision=HazardDecision.NO_OP,
            rule_applied="R7",
            matrix_key=None,
            notes=("no hazard signal",),
        )
        return HazardEvaluationResult(
            decision=HazardDecision.NO_OP,
            frame=frame,
            trace=trace,
        )

    family = _process_family(frame)
    procedural = _is_procedural(frame)

    horticultural_cultivate = (
        frame.intended_outcome == OutcomeKind.HORTICULTURAL
        or frame.user_framing == FramingKind.HORTICULTURAL
        or (
            family == ProcessFamily.OTHER_TRANSFORM
            and frame.intended_outcome == OutcomeKind.HORTICULTURAL
        )
    )
    # Horticultural cultivation / seed collection / planting — admit unless a
    # hazardous chem transform family is also present.
    if horticultural_cultivate and family not in TRANSFORM_PROCESS_FAMILIES:
        trace = DecisionTrace(
            frame=frame,
            decision=HazardDecision.ADMIT,
            rule_applied="horticultural_admit",
            matrix_key=None,
            notes=("horticultural cultivate without hazardous chem transform",),
        )
        return HazardEvaluationResult(
            decision=HazardDecision.ADMIT,
            frame=frame,
            trace=trace,
        )

    # Vaccine / biodefense / public-health strategy against bioweapons — admit.
    # Must not be classified as operational bioweapon fabrication.
    if _is_vaccine_or_defensive_bioweapon_carve_out(frame):
        trace = DecisionTrace(
            frame=frame,
            decision=HazardDecision.ADMIT,
            rule_applied="bioweapon_defensive_carve_out",
            matrix_key=None,
            notes=("vaccine/biodefense/public-health framing against bioweapon",),
        )
        return HazardEvaluationResult(
            decision=HazardDecision.ADMIT,
            frame=frame,
            trace=trace,
        )

    # R5 — weaponisation framing or process always stops
    if (
        frame.user_framing == FramingKind.WEAPONISING
        or family == ProcessFamily.WEAPONISATION
        or frame.intended_outcome == OutcomeKind.WEAPONISATION
    ):
        # Fiction does not carve out (R6)
        trace = DecisionTrace(
            frame=frame,
            decision=HazardDecision.STOP_HAZARD_TRANSFORM,
            rule_applied="R5",
            matrix_key=None,
            notes=("weaponisation framing or process",),
        )
        return HazardEvaluationResult(
            decision=HazardDecision.STOP_HAZARD_TRANSFORM,
            frame=frame,
            trace=trace,
            rule_id=_stop_rule_id(frame),
        )

    # R3 — unresolved substance under procedural transform (substance only)
    if (
        frame.ambiguity == AmbiguityState.SUBSTANCE_UNRESOLVED
        and procedural
        and (family in TRANSFORM_PROCESS_FAMILIES or family == ProcessFamily.NONE)
    ):
        if family in TRANSFORM_PROCESS_FAMILIES or family == ProcessFamily.NONE:
            trace = DecisionTrace(
                frame=frame,
                decision=HazardDecision.ASK_DISAMBIGUATE,
                rule_applied="R3",
                matrix_key=None,
                notes=("substance unresolved under procedural transform",),
            )
            return HazardEvaluationResult(
                decision=HazardDecision.ASK_DISAMBIGUATE,
                frame=frame,
                trace=trace,
                rule_id=RULE_HAZARD_SUBSTANCE_UNRESOLVED,
            )

    # PROCESS_UNRESOLVED is not a substance ambiguity — do not emit
    # HAZARD_SUBSTANCE_UNRESOLVED. Without a hazardous transform, admit.
    if frame.ambiguity == AmbiguityState.PROCESS_UNRESOLVED:
        if family not in TRANSFORM_PROCESS_FAMILIES:
            trace = DecisionTrace(
                frame=frame,
                decision=HazardDecision.ADMIT,
                rule_applied="process_unresolved_admit",
                matrix_key=None,
                notes=(
                    "process unresolved without hazardous chem transform; "
                    "not substance unresolved",
                ),
            )
            return HazardEvaluationResult(
                decision=HazardDecision.ADMIT,
                frame=frame,
                trace=trace,
            )

    # R1 / R2 — benign informational / identifying
    if (
        frame.procedurality in (Procedurality.INFORMATIONAL, Procedurality.IDENTIFYING)
        and frame.user_framing in _BENIGN_FRAMINGS
        and family not in TRANSFORM_PROCESS_FAMILIES
    ):
        trace = DecisionTrace(
            frame=frame,
            decision=HazardDecision.ADMIT,
            rule_applied="R1",
            matrix_key=None,
            notes=("benign framing without transform process",),
        )
        return HazardEvaluationResult(
            decision=HazardDecision.ADMIT,
            frame=frame,
            trace=trace,
        )

    if (
        frame.intended_outcome in _BENIGN_OUTCOMES
        and not procedural
        and family not in TRANSFORM_PROCESS_FAMILIES
    ):
        trace = DecisionTrace(
            frame=frame,
            decision=HazardDecision.ADMIT,
            rule_applied="R2",
            matrix_key=None,
            notes=("benign outcome without procedural howto",),
        )
        return HazardEvaluationResult(
            decision=HazardDecision.ADMIT,
            frame=frame,
            trace=trace,
        )

    # R4 — procedural transform of blockable severe/lethal class
    if (
        procedural
        and family in TRANSFORM_PROCESS_FAMILIES
        and _block_on_transform(frame, ontology)
        and severity_at_least(frame.consequence_severity, Severity.SEVERE)
    ):
        # Fiction does not carve out (R6)
        matrix_decision, matrix_key = _lookup_matrix(
            ontology,
            frame.user_framing,
            frame.procedurality,
            family,
            frame.consequence_severity,
        )
        decision = matrix_decision or HazardDecision.STOP_HAZARD_TRANSFORM
        if decision == HazardDecision.ADMIT:
            # Matrix cannot admit R4-eligible transforms; force stop
            decision = HazardDecision.STOP_HAZARD_TRANSFORM
        if decision == HazardDecision.ASK_DISAMBIGUATE:
            if frame.ambiguity != AmbiguityState.SUBSTANCE_UNRESOLVED:
                # Do not mislabel process gaps as substance unresolved
                trace = DecisionTrace(
                    frame=frame,
                    decision=HazardDecision.STOP_HAZARD_TRANSFORM,
                    rule_applied="R4",
                    matrix_key=matrix_key,
                    notes=("procedural hazardous transform; matrix ASK ignored",),
                )
                return HazardEvaluationResult(
                    decision=HazardDecision.STOP_HAZARD_TRANSFORM,
                    frame=frame,
                    trace=trace,
                    rule_id=_stop_rule_id(frame),
                )
            trace = DecisionTrace(
                frame=frame,
                decision=decision,
                rule_applied="R4+matrix",
                matrix_key=matrix_key,
            )
            return HazardEvaluationResult(
                decision=decision,
                frame=frame,
                trace=trace,
                rule_id=RULE_HAZARD_SUBSTANCE_UNRESOLVED,
            )
        trace = DecisionTrace(
            frame=frame,
            decision=HazardDecision.STOP_HAZARD_TRANSFORM,
            rule_applied="R4",
            matrix_key=matrix_key,
            notes=("procedural transform of severe/lethal hazard class",),
        )
        return HazardEvaluationResult(
            decision=HazardDecision.STOP_HAZARD_TRANSFORM,
            frame=frame,
            trace=trace,
            rule_id=_stop_rule_id(frame),
        )

    # Matrix fallback for other combinations with hazard signal
    matrix_decision, matrix_key = _lookup_matrix(
        ontology,
        frame.user_framing,
        frame.procedurality,
        family,
        frame.consequence_severity,
    )
    if matrix_decision is not None:
        # Never map PROCESS_UNRESOLVED onto HAZARD_SUBSTANCE_UNRESOLVED.
        if (
            matrix_decision == HazardDecision.ASK_DISAMBIGUATE
            and frame.ambiguity == AmbiguityState.PROCESS_UNRESOLVED
        ):
            trace = DecisionTrace(
                frame=frame,
                decision=HazardDecision.ADMIT,
                rule_applied="matrix_process_unresolved_admit",
                matrix_key=matrix_key,
                notes=("PROCESS_UNRESOLVED must not emit substance-unresolved rule",),
            )
            return HazardEvaluationResult(
                decision=HazardDecision.ADMIT,
                frame=frame,
                trace=trace,
            )
        rule_id = None
        if matrix_decision == HazardDecision.STOP_HAZARD_TRANSFORM:
            rule_id = _stop_rule_id(frame)
        elif matrix_decision == HazardDecision.ASK_DISAMBIGUATE:
            if frame.ambiguity == AmbiguityState.SUBSTANCE_UNRESOLVED:
                rule_id = RULE_HAZARD_SUBSTANCE_UNRESOLVED
            else:
                # ASK without substance ambiguity — treat as admit for safety of reason codes
                trace = DecisionTrace(
                    frame=frame,
                    decision=HazardDecision.ADMIT,
                    rule_applied="matrix_ask_without_substance_ambiguity",
                    matrix_key=matrix_key,
                    notes=("ASK matrix row without SUBSTANCE_UNRESOLVED → ADMIT",),
                )
                return HazardEvaluationResult(
                    decision=HazardDecision.ADMIT,
                    frame=frame,
                    trace=trace,
                )
        trace = DecisionTrace(
            frame=frame,
            decision=matrix_decision,
            rule_applied="matrix",
            matrix_key=matrix_key,
        )
        return HazardEvaluationResult(
            decision=matrix_decision,
            frame=frame,
            trace=trace,
            rule_id=rule_id,
        )

    # Default: hazard mentioned but no operational transform → admit (silent)
    if not procedural or family not in TRANSFORM_PROCESS_FAMILIES:
        trace = DecisionTrace(
            frame=frame,
            decision=HazardDecision.ADMIT,
            rule_applied="default_admit",
            matrix_key=None,
            notes=("hazard signal without operational transform",),
        )
        return HazardEvaluationResult(
            decision=HazardDecision.ADMIT,
            frame=frame,
            trace=trace,
        )

    # Safe fallback: procedural + transform + unresolved severity → stop
    if procedural and family in TRANSFORM_PROCESS_FAMILIES:
        trace = DecisionTrace(
            frame=frame,
            decision=HazardDecision.STOP_HAZARD_TRANSFORM,
            rule_applied="safe_fallback_stop",
            matrix_key=None,
            notes=("procedural transform with hazard signal; safe stop",),
        )
        return HazardEvaluationResult(
            decision=HazardDecision.STOP_HAZARD_TRANSFORM,
            frame=frame,
            trace=trace,
            rule_id=_stop_rule_id(frame),
        )

    trace = DecisionTrace(
        frame=frame,
        decision=HazardDecision.ADMIT,
        rule_applied="default_admit_tail",
        matrix_key=None,
    )
    return HazardEvaluationResult(
        decision=HazardDecision.ADMIT,
        frame=frame,
        trace=trace,
    )
