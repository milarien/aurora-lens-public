"""Generic typed-transition admission/query over ordinary PEF relationships.

This module is domain-general state machinery. Domain vocabularies may define
typed relations (for example PRESCRIBE/WITHHOLD/REDUCE), but continuity remains
PEF-owned and projections read from ``pef.relationships``.

Source tracking: Query methods return source relationship indices so that
source status ("admitted_uncontaminated" vs "unverified") can be verified.
"""

from __future__ import annotations

import re
from typing import Any

from aurora_lens.interpret.turn_act import TurnAct
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.pef.state_transitions import project_latest_by_literal_key
from aurora_lens.state_native_engine.contracts import (
    StateNativeDelegationResult,
    StateNativeOutcome,
    StateNativeRequest,
    StateNativeSolverFamily,
)
from aurora_lens.state_native_engine.eval.source_verification import (
    verify_source_relationships_admitted,
)

_ACK_TEXT = "Recorded state transition."

_PRESCRIBE = re.compile(
    r"(?is)^(?P<actor>[^\n]{1,480}?)\s+prescribed\s+(?P<patient>[^\n]{1,180}?)\s+"
    r"(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg)\s+of\s+(?P<med>[^\n]{1,220}?)(?:\s+"
    r"(?P<freq>(?:once|twice)\s+daily|daily|every\s+\d+\s+hours?))?\s*\.?$"
)
_WITHHOLD = re.compile(
    r"(?is)^(?P<actor>[^\n]{1,480}?)\s+withheld\s+(?:the\s+)?(?P<schedule>morning|evening|night|noon)"
    r"\s+dose(?:\s+of\s+(?P<med>[^\.\n]+?))?(?:\s+for\s+(?P<patient>[^\.\n]+?))?"
    r"(?:\s+pending\s+(?P<reason>[^\.\n]+))?\s*\.?$"
)
_ADMINISTER = re.compile(
    r"(?is)^(?P<actor>[^\n]{1,480}?)\s+administered\s+(?:the\s+)?(?P<schedule>morning|evening|night|noon)"
    r"\s+dose(?:\s+of\s+(?P<med>[^\.\n]+?))?(?:\s+to\s+(?P<patient>[^\.\n]+?))?\s*\.?$"
)
_DOSE_CHANGE_ABS = re.compile(
    r"(?is)^(?P<actor>[^\n]{1,480}?)\s+(?:later\s+)?(?:reduced|increase[ds]?|changed)\s+"
    r"(?:the\s+)?dos(?:e|age)(?:\s+for\s+(?P<patient>[^\n]{1,120}?))?"
    r"(?:\s+of\s+(?P<med>[^\n]{1,180}?))?\s+to\s+(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg)\s*\.?$"
)
_DOSE_CHANGE_REL = re.compile(
    r"(?is)^(?P<actor>[^\n]{1,480}?)\s+(?:later\s+)?(?P<verb>halved|halve[ds]?)\s+"
    r"(?:the\s+)?dos(?:e|age)(?:\s+for\s+(?P<patient>[^\n]{1,120}?))?"
    r"(?:\s+of\s+(?P<med>[^\n]{1,180}?))?\s*\.?$"
)
_Q_DOSAGE = re.compile(
    r"(?is)^what\s+dosage\s+is\s+(?P<patient>[^\n]{1,120}?)\s+(?:currently\s+)?receiving"
    r"(?:\s+of\s+(?P<med>[^\n]{1,180}?))?\s*\?$"
)
_Q_AUTH = re.compile(
    r"(?is)^who\s+(?:authorized|authorised)\s+the\s+current\s+dosage"
    r"(?:\s+for\s+(?P<patient>[^\n]{1,120}?))?(?:\s+of\s+(?P<med>[^\n]{1,180}?))?\s*\?$"
)
_Q_SCHEDULE = re.compile(
    r"(?is)^was\s+the\s+(?P<schedule>morning|evening|night|noon)\s+dose\s+(?:administered|given)"
    r"(?:\s+for\s+(?P<patient>[^\n]{1,120}?))?(?:\s+of\s+(?P<med>[^\n]{1,180}?))?\s*\?$"
)


def _text(v: str | None) -> str | None:
    if not isinstance(v, str):
        return None
    t = v.strip()
    return t if t else None


def _dose_display(value: float, unit: str) -> str:
    n = str(int(value)) if value == float(int(value)) else str(value).rstrip("0").rstrip(".")
    return f"{n}{unit}"


def _active_dose_records(pef: PEFState) -> dict[tuple[str, ...], tuple[int, Relationship, dict[str, Any]]]:
    return project_latest_by_literal_key(
        pef,
        key_fields=("patient", "medication"),
        order_fields=("__transition_seq",),
        relation_allowlist={"PRESCRIBE", "REDUCE", "CHANGE", "HALVE"},
        literal_filter=lambda _rel, lit: lit.get("transition_kind") == "active_dose",
    )


def _resolve_single_active_record(
    pef: PEFState,
    *,
    patient: str | None,
    medication: str | None,
) -> tuple[int, Relationship, dict[str, Any]] | None:
    recs = _active_dose_records(pef)
    cand = []
    for tup in recs.values():
        _idx, _rel, lit = tup
        lp = str(lit.get("patient") or "").strip().lower()
        lm = str(lit.get("medication") or "").strip().lower()
        if patient and lp != patient.strip().lower():
            continue
        if medication and lm != medication.strip().lower():
            continue
        cand.append(tup)
    return cand[0] if len(cand) == 1 else None


def _apply_transition(req: StateNativeRequest) -> bool:
    pef = req.pef
    raw_text = req.user_text
    source_turn = pef.current_turn
    text = raw_text.strip()
    if not text:
        return False

    def _add(actor: str, relation: str, payload: dict[str, Any]) -> None:
        subj, _ = pef.get_or_create_entity(actor)
        patient = _text(str(payload.get("patient") or ""))
        if patient:
            pef.get_or_create_entity(patient)
        payload_with_seq = dict(payload)
        payload_with_seq["__transition_seq"] = len(pef.relationships) + 1
        pef.add_relationship(
            Relationship(
                subject_id=subj.id,
                relation=relation,
                object_entity_id=None,
                object_literal=payload_with_seq,
                span=req.detected_span,
                source_turn=source_turn,
                evidence=raw_text,
                negated=False,
                provenance="system",
                extractor_backend="rule",
            )
        )

    m = _PRESCRIBE.match(text)
    if m:
        actor = _text(m.group("actor"))
        patient = _text(m.group("patient"))
        dose = _text(m.group("dose"))
        unit = _text(m.group("unit"))
        med = _text(m.group("med"))
        freq = _text(m.group("freq")) or ""
        if actor and patient and dose and unit and med:
            dv = float(dose)
            _add(
                actor,
                "PRESCRIBE",
                {
                    "transition_kind": "active_dose",
                    "patient": patient,
                    "medication": med,
                    "dose_value": dv,
                    "dose_unit": unit.lower(),
                    "dose_display": _dose_display(dv, unit.lower()),
                    "frequency": freq.lower(),
                },
            )
            return True
        return False

    m = _DOSE_CHANGE_ABS.match(text)
    if m:
        actor = _text(m.group("actor"))
        dose = _text(m.group("dose"))
        unit = _text(m.group("unit"))
        patient = _text(m.group("patient"))
        med = _text(m.group("med"))
        if actor and dose and unit:
            resolved = _resolve_single_active_record(pef, patient=patient, medication=med)
            if resolved is None:
                return False
            _, _, lit = resolved
            relation = "REDUCE" if "reduc" in text.lower() else "CHANGE"
            dv = float(dose)
            _add(
                actor,
                relation,
                {
                    "transition_kind": "active_dose",
                    "patient": str(lit.get("patient")),
                    "medication": str(lit.get("medication")),
                    "dose_value": dv,
                    "dose_unit": unit.lower(),
                    "dose_display": _dose_display(dv, unit.lower()),
                    "change_kind": "absolute",
                },
            )
            return True
        return False

    m = _DOSE_CHANGE_REL.match(text)
    if m:
        actor = _text(m.group("actor"))
        patient = _text(m.group("patient"))
        med = _text(m.group("med"))
        if actor:
            resolved = _resolve_single_active_record(pef, patient=patient, medication=med)
            if resolved is None:
                return False
            _, _, lit = resolved
            prior = lit.get("dose_value")
            unit = _text(str(lit.get("dose_unit") or ""))
            if not isinstance(prior, (int, float)) or not unit:
                return False
            dv = float(prior) / 2.0
            _add(
                actor,
                "HALVE",
                {
                    "transition_kind": "active_dose",
                    "patient": str(lit.get("patient")),
                    "medication": str(lit.get("medication")),
                    "dose_value": dv,
                    "dose_unit": unit,
                    "dose_display": _dose_display(dv, unit),
                    "change_kind": "relative_halve",
                },
            )
            return True
        return False

    m = _WITHHOLD.match(text)
    if m:
        actor = _text(m.group("actor"))
        schedule = _text(m.group("schedule"))
        patient = _text(m.group("patient"))
        med = _text(m.group("med"))
        reason = (_text(m.group("reason")) or "review").lower()
        if actor and schedule:
            resolved = _resolve_single_active_record(pef, patient=patient, medication=med)
            if resolved is None:
                return False
            _, _, lit = resolved
            _add(
                actor,
                "WITHHOLD",
                {
                    "transition_kind": "scheduled_dose_status",
                    "patient": str(lit.get("patient")),
                    "medication": str(lit.get("medication")),
                    "schedule": schedule.lower(),
                    "status": "withheld",
                    "reason": reason,
                },
            )
            return True
        return False

    m = _ADMINISTER.match(text)
    if m:
        actor = _text(m.group("actor"))
        schedule = _text(m.group("schedule"))
        patient = _text(m.group("patient"))
        med = _text(m.group("med"))
        if actor and schedule:
            resolved = _resolve_single_active_record(pef, patient=patient, medication=med)
            if resolved is None:
                return False
            _, _, lit = resolved
            _add(
                actor,
                "ADMINISTER",
                {
                    "transition_kind": "scheduled_dose_status",
                    "patient": str(lit.get("patient")),
                    "medication": str(lit.get("medication")),
                    "schedule": schedule.lower(),
                    "status": "administered",
                },
            )
            return True
        return False
    return False


def _answer_query(pef: PEFState, raw_text: str) -> tuple[str | None, list[int]]:
    """Answer a query and return the source relationship indices used.

    Returns: (answer_text, source_relationship_indices)
    """
    text = raw_text.strip()
    if not text:
        return None, []

    m = _Q_DOSAGE.match(text)
    if m:
        patient = _text(m.group("patient"))
        med = _text(m.group("med"))
        if not patient:
            return None, []
        resolved = _resolve_single_active_record(pef, patient=patient, medication=med)
        if resolved is None:
            return None, []
        idx, _, lit = resolved
        answer = f"{lit.get('patient')} is currently receiving {lit.get('dose_display')} of {lit.get('medication')}."
        return answer, [idx]

    m = _Q_AUTH.match(text)
    if m:
        patient = _text(m.group("patient"))
        med = _text(m.group("med"))
        resolved = _resolve_single_active_record(pef, patient=patient, medication=med)
        if resolved is None:
            return None, []
        idx, rel, _ = resolved
        actor = pef.entities.get(rel.subject_id)
        answer = f"{actor.name} authorized the current dosage." if actor else None
        return answer, [idx] if answer else []

    m = _Q_SCHEDULE.match(text)
    if m:
        schedule = _text(m.group("schedule"))
        patient = _text(m.group("patient"))
        med = _text(m.group("med"))
        if not schedule:
            return None, []
        resolved = _resolve_single_active_record(pef, patient=patient, medication=med)
        if resolved is None:
            return None, []
        active_idx, _, active_lit = resolved
        sched = project_latest_by_literal_key(
            pef,
            key_fields=("patient", "medication", "schedule"),
            order_fields=("__transition_seq",),
            relation_allowlist={"WITHHOLD", "ADMINISTER"},
            literal_filter=lambda _rel, lit: lit.get("transition_kind") == "scheduled_dose_status",
        )
        key = (
            str(active_lit.get("patient")).strip().lower(),
            str(active_lit.get("medication")).strip().lower(),
            schedule.lower(),
        )
        row = sched.get(key)
        if row is None:
            return None, []
        sched_idx, _, lit = row
        status = str(lit.get("status") or "")
        if status == "withheld":
            answer = f"No. The {schedule.lower()} dose was withheld pending {lit.get('reason') or 'review'}."
            return answer, [active_idx, sched_idx]
        if status == "administered":
            answer = f"Yes. The {schedule.lower()} dose was administered."
            return answer, [active_idx, sched_idx]
    return None, []


def evaluate_typed_transition_request(
    req: StateNativeRequest,
) -> StateNativeDelegationResult | None:
    """Evaluate typed transition mutation/query over committed PEF."""
    if req.turn_act == TurnAct.QUERY:
        txt, source_indices = _answer_query(req.pef, req.user_text)
        if txt is None:
            return None
        # Verify source relationships are admitted and uncontaminated
        source_status = verify_source_relationships_admitted(req.pef, source_indices)
        return StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.ANSWER,
            user_visible_text=txt,
            solver_family=StateNativeSolverFamily.COMMITTED_TYPED_TRANSITION,
            source_status=source_status,
        )

    if req.turn_act == TurnAct.ASSERT:
        if _apply_transition(req):
            return StateNativeDelegationResult(
                handled=True,
                outcome=StateNativeOutcome.ANSWER,
                user_visible_text=_ACK_TEXT,
                solver_family=StateNativeSolverFamily.COMMITTED_TYPED_TRANSITION,
                source_status="admitted_uncontaminated",  # Assertions are admitted by design
            )
    return None

