"""Unresolved governance that outlives a session TTL.

Session expiry drops the conversational record. It does not resolve an open
attribution, clarification, or epistemic hold, and it does not grant permission.
The residue below is the constraint itself. Admitted facts, turn history, and
the audit snapshot stay with the expired record and are not copied forward.
"""

from __future__ import annotations

import copy
from typing import Any

from aurora_lens.pef.state import (
    EPISTEMIC_MODE_AMBIGUITY,
    EPISTEMIC_MODE_REFUSAL,
    EPISTEMIC_MODE_STOP,
    PEFState,
    _hydrate_epistemic_hold_from_legacy,
)
from aurora_lens.pef.unresolved_referents import (
    STATUS_OPEN,
    UnresolvedReferentEntry,
    build_unresolved_referent_clarification_choices,
    open_entries,
    registry_entries_from_wire,
    registry_entries_to_wire,
)

GOVERNANCE_HOLD_SCHEMA_VERSION = 1

_DURABLE_HOLD_MODES = frozenset({
    EPISTEMIC_MODE_AMBIGUITY,
    EPISTEMIC_MODE_REFUSAL,
    EPISTEMIC_MODE_STOP,
})


def unresolved_governance_residue(pef: PEFState) -> dict[str, Any] | None:
    """Return the unresolved constraint, or None when nothing is still open.

    Resolved registry entries and admitted PEF facts are omitted. Timeout is
    not a resolution, so only an explicit clear produces None.
    """
    registry = registry_entries_to_wire(open_entries(pef))
    pending = pef.pending_clarification if isinstance(pef.pending_clarification, dict) else None
    hold = pef.epistemic_hold if isinstance(pef.epistemic_hold, dict) else None
    if hold is not None and hold.get("mode") not in _DURABLE_HOLD_MODES:
        hold = None
    if not registry and pending is None and hold is None:
        return None
    return {
        "schema_version": GOVERNANCE_HOLD_SCHEMA_VERSION,
        "unresolved_referent_registry": registry,
        "pending_clarification": pending,
        "epistemic_hold": hold,
    }


def _epistemic_hold_rank(hold: dict[str, Any] | None) -> int:
    """Terminal stop outranks a reopenable stop, which outranks refusal."""
    if not isinstance(hold, dict):
        return 0
    mode = hold.get("mode")
    if mode == EPISTEMIC_MODE_STOP and hold.get("interaction_open") is False:
        return 4
    if mode == EPISTEMIC_MODE_STOP:
        return 3
    if mode == EPISTEMIC_MODE_REFUSAL:
        return 2
    if mode == EPISTEMIC_MODE_AMBIGUITY:
        return 1
    return 0


# Pending keys that describe the constraint being clarified. Identity is
# ``failed_constraint`` plus ``original_question``, not the pronoun text.
_PENDING_GOVERNING_KEYS = (
    "candidate_entities",
    "blocked_proposition",
    "blocked_claims",
    "ambiguous_referents",
    "clarification_choices",
    "allow_hold_unresolved",
    "clarification_prompt",
    "resolution_mode",
)


def _governing_signature(entry: UnresolvedReferentEntry) -> tuple[Any, ...]:
    """Constraint body used only when an entry has no ``entry_id``.

    Pronoun text alone is not the signature. Candidates, the blocked
    proposition, and the introducing turn stay in it.
    """
    return (
        entry.token.strip().lower(),
        tuple(name.strip().lower() for name in entry.candidate_entities if name.strip()),
        entry.blocked_proposition or "",
        entry.introduced_turn,
        entry.introduced_utterance,
    )


def _install_durable_entry(existing: UnresolvedReferentEntry, incoming: UnresolvedReferentEntry) -> None:
    """Install the later governing state of one registry identity.

    ``entry_id`` is the identity. The incoming record already contains the
    candidate list, blocked proposition, and span produced by
    ``register_unresolved_referents``. Re-applying that function's incremental
    union here would keep a stale blocked proposition.
    """
    existing.token = incoming.token
    existing.span_surface = incoming.span_surface
    existing.candidate_entities = list(incoming.candidate_entities)
    existing.blocked_proposition = incoming.blocked_proposition
    existing.head = incoming.head
    existing.introduced_turn = incoming.introduced_turn
    existing.introduced_utterance = incoming.introduced_utterance
    existing.resolution_mode = incoming.resolution_mode
    existing.status = STATUS_OPEN
    existing.resolved_entity = None
    existing.resolved_turn = None


def _pending_constraint_identity(pending: dict[str, Any]) -> tuple[str, str] | None:
    """Clarification identity. Pronoun text is not an identity."""
    kind = str(pending.get("failed_constraint") or "").strip()
    question = pending.get("original_question")
    if not kind or not isinstance(question, str) or not question.strip():
        return None
    return (kind, question)


def _open_entries_for_token(pef: PEFState, token: str) -> list[UnresolvedReferentEntry]:
    key = token.strip().lower()
    if not key:
        return []
    return [entry for entry in open_entries(pef) if entry.token.strip().lower() == key]


def _align_pending_to_unique_entries(pef: PEFState) -> None:
    """Make a referent clarification agree with the one open entry it names.

    When several open entries share the pronoun, pronoun text does not say
    which constraint the clarification is about, so the pending record is left
    as the durable clarification already selected.
    """
    pending = pef.pending_clarification
    if not isinstance(pending, dict):
        return
    if str(pending.get("failed_constraint") or "") != "UNRESOLVED_REFERENT":
        return
    tokens = [
        str(token).strip()
        for token in (pending.get("ambiguous_referents") or [])
        if str(token).strip()
    ]
    if not tokens:
        return
    chosen: list[UnresolvedReferentEntry] = []
    for token in tokens:
        matches = _open_entries_for_token(pef, token)
        if len(matches) != 1:
            return
        chosen.append(matches[0])
    candidates: list[str] = []
    seen: set[str] = set()
    propositions: list[str] = []
    for entry in chosen:
        for name in entry.candidate_entities:
            label = name.strip()
            key = label.lower()
            if label and key not in seen:
                seen.add(key)
                candidates.append(label)
        proposition = entry.blocked_proposition
        if proposition and proposition not in propositions:
            propositions.append(str(proposition))
    pending["candidate_entities"] = candidates
    if len(propositions) == 1:
        pending["blocked_proposition"] = propositions[0]
    pending["clarification_choices"] = build_unresolved_referent_clarification_choices(
        candidates,
        include_hold_unresolved=len(candidates) >= 2,
    )
    pending["allow_hold_unresolved"] = len(candidates) >= 2


def _reconcile_pending_clarification(pef: PEFState, residue: dict[str, Any]) -> None:
    incoming = residue.get("pending_clarification")
    live = pef.pending_clarification if isinstance(pef.pending_clarification, dict) else None
    if isinstance(incoming, dict):
        if live is None:
            # A pending clarification is its own constraint. An empty referent
            # registry does not mean the clarification was resolved.
            pef.pending_clarification = copy.deepcopy(incoming)
            _align_pending_to_unique_entries(pef)
            return
        live_id = _pending_constraint_identity(live)
        incoming_id = _pending_constraint_identity(incoming)
        if live_id is not None and live_id == incoming_id:
            for key in _PENDING_GOVERNING_KEYS:
                if key in incoming:
                    live[key] = copy.deepcopy(incoming[key])
        # One open entry for the pronoun: its governing fields win.
        # Several entries sharing the pronoun are different constraints.
        _align_pending_to_unique_entries(pef)
        return
    if "pending_clarification" in residue and incoming is None:
        pef.pending_clarification = None


def merge_unresolved_governance_residue(pef: PEFState, residue: dict[str, Any]) -> None:
    """Reconcile a durable hold onto a live session record.

    Used when the conversational record was saved without the hold, or the hold
    delete did not finish. Admitted facts already on ``pef`` are left in place.
    A weaker epistemic hold does not replace a stronger one.

    Registry identity is ``entry_id``. An older open entry with the same
    pronoun does not suppress a newer entry, and it does not keep its stale
    candidates or blocked proposition when the durable record is that same
    identity after an update.
    """
    by_id = {
        entry.entry_id: entry
        for entry in pef.unresolved_referent_registry
        if entry.entry_id.strip()
    }
    for incoming in registry_entries_from_wire(residue.get("unresolved_referent_registry")):
        if incoming.status != STATUS_OPEN or not incoming.token.strip():
            continue
        if not incoming.entry_id.strip():
            signature = _governing_signature(incoming)
            if any(_governing_signature(entry) == signature for entry in pef.unresolved_referent_registry):
                continue
            pef.unresolved_referent_registry.append(incoming)
            continue
        existing = by_id.get(incoming.entry_id)
        if existing is None:
            pef.unresolved_referent_registry.append(incoming)
            by_id[incoming.entry_id] = incoming
            continue
        _install_durable_entry(existing, incoming)
    _reconcile_pending_clarification(pef, residue)
    incoming_hold = residue.get("epistemic_hold")
    if isinstance(incoming_hold, dict) and incoming_hold.get("mode") in _DURABLE_HOLD_MODES:
        current = pef.epistemic_hold if isinstance(pef.epistemic_hold, dict) else None
        if _epistemic_hold_rank(incoming_hold) > _epistemic_hold_rank(current):
            pef.epistemic_hold = dict(incoming_hold)
    _hydrate_epistemic_hold_from_legacy(pef)


def apply_unresolved_governance_residue(pef: PEFState, residue: dict[str, Any]) -> None:
    """Attach an unresolved constraint to a fresh session state.

    Does not copy entities, relationships, discourse bindings, continuation
    corridors, or the audit snapshot from the expired session.
    """
    pef.unresolved_referent_registry = [
        entry
        for entry in registry_entries_from_wire(residue.get("unresolved_referent_registry"))
        if entry.status == STATUS_OPEN
    ]
    pending = residue.get("pending_clarification")
    pef.pending_clarification = dict(pending) if isinstance(pending, dict) else None
    hold = residue.get("epistemic_hold")
    if isinstance(hold, dict) and hold.get("mode") in _DURABLE_HOLD_MODES:
        pef.epistemic_hold = dict(hold)
    else:
        pef.epistemic_hold = None
    _hydrate_epistemic_hold_from_legacy(pef)
