"""Constitutional validation for operator matrix overrides.

FIELD CONSTITUTION
------------------
Continuation fields are classified into three tiers:

IMMUTABLE — operator may not alter these fields.
  forensic_obligations: forensic obligations are set by canonical governance
      law and are required for regulatory/audit compliance.  Operators may
      neither remove canonical obligations nor add new ones — additions must
      be made in the canonical matrix and committed through the normal review
      process.

CONSTRAINED — operator may set within defined bounds only.
  commitment_closed:  may only be tightened (canonical False → operator True).
      Never relaxed (canonical True → operator False).  Relaxing
      commitment_closed widens the epistemic commitment corridor and permits
      the system to assert, infer, or imply blocked determinations.
  interaction_open:   may only be restricted (canonical True → operator False).
      Never widened (canonical False → operator True).  Widening
      interaction_open re-opens a conversation that governance law requires
      to be closed.
  pathway_id:         the operator pathway's semantic class must be ≥ the
      canonical pathway's semantic class.  An operator may route a row to a
      more-restrictive class (e.g. REFUSE → STOP) or stay in the same class,
      but may not weaken (e.g. STOP → REFUSE, REFUSE → ASK, ASK → ADMIT).
      Semantic class order: ADMIT(0) < ASK(1) < REFUSE(2) < STOP(3).
  output_mode:        if set, must belong to the canonical set of output_modes
      for the effective pathway_id.  Each pathway_id has a fixed set of valid
      output_modes derived from the canonical matrix.

FREE — operator may set without restriction.
  escalation_target: operators may specify domain-appropriate professional or
      crisis contact strings without governance constraint.

SPEECH-ACT AND DISCLOSURE FIELDS
---------------------------------
allowed_speech_acts, allowed_procedural_actions, forbidden_speech_acts,
required_disclosures, and exposure_level are validated downstream by
GovernorPolicy.__post_init__ (which rejects substantive speech acts when
commitment_closed=True) and are not validated here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .continuation_matrix import ContinuationRow, _TABLE
from .models import ContinuationPathway, ForensicObligation, OutputMode


# ── Pathway semantic classes ──────────────────────────────────────────────────
#
# Four ordered classes: ADMIT=0 < ASK=1 < REFUSE=2 < STOP=3.
# Operator pathway_id must belong to a class ≥ the canonical class.

_PATHWAY_SEMANTIC_CLASS: dict[ContinuationPathway, int] = {
    ContinuationPathway.P_ADMIT_STANDARD:             0,  # ADMIT
    ContinuationPathway.P_ASK_DISAMBIGUATE:           1,  # ASK
    ContinuationPathway.P_ASK_MISSING_FACT:           1,  # ASK
    ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT:    2,  # REFUSE
    ContinuationPathway.P_REFUSE_ESCALATE_PRO:        2,  # REFUSE
    ContinuationPathway.P_HANDOFF_SUMMARY:            2,  # REFUSE
    ContinuationPathway.P_STOP_TERMINAL:              3,  # STOP
    ContinuationPathway.P_STOP_FORENSIC:              3,  # STOP
    ContinuationPathway.P_STOP_ESCALATE:              3,  # STOP
    ContinuationPathway.P_STOP_ESCALATE_EMERGENCY:    3,  # STOP
    ContinuationPathway.P_STOP_SUPPORTIVE_DEESCALATE: 3,  # STOP
    ContinuationPathway.P_STOP_REDIRECT_QUALIFIED:    3,  # STOP
    ContinuationPathway.P_STOP_REFUSE_CLEAN:          3,  # STOP
}

_CLASS_NAMES: dict[int, str] = {0: "ADMIT", 1: "ASK", 2: "REFUSE", 3: "STOP"}


# ── Pathway intra-class sub-levels ────────────────────────────────────────────
#
# Within a single semantic class an operator may only substitute to a pathway
# whose sub-level is ≥ the canonical pathway's sub-level.  Sub-levels reflect
# increasing governance weight within the class (more forensic obligations,
# more restricted outputs, stricter escalation requirements).
#
# Sub-level check only applies when canonical_class == op_class (same-class
# substitution).  Cross-class tightening (e.g. REFUSE → STOP) is governed
# solely by the class-level check above.
#
# ADMIT(0):  only one pathway — sub-level 0.
# ASK(1):    both pathways are weight-equivalent — sub-level 0.
# REFUSE(2): handoff/explain-redirect are baseline (0);
#            escalate-pro requires professional referral (1).
# STOP(3):   refuse-clean is the softest termination (0);
#            terminal/redirect-qualified are standard stops (1);
#            supportive-deescalate/forensic require crisis or audit weight (2);
#            escalate requires active referral to external service (3);
#            escalate-emergency is the most restrictive — immediate danger (4).

_PATHWAY_SUB_LEVEL: dict[ContinuationPathway, int] = {
    # ADMIT
    ContinuationPathway.P_ADMIT_STANDARD:             0,
    # ASK
    ContinuationPathway.P_ASK_DISAMBIGUATE:           0,
    ContinuationPathway.P_ASK_MISSING_FACT:           0,
    # REFUSE
    ContinuationPathway.P_HANDOFF_SUMMARY:            0,
    ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT:    0,
    ContinuationPathway.P_REFUSE_ESCALATE_PRO:        1,
    # STOP
    ContinuationPathway.P_STOP_REFUSE_CLEAN:          0,
    ContinuationPathway.P_STOP_TERMINAL:              1,
    ContinuationPathway.P_STOP_REDIRECT_QUALIFIED:    1,
    ContinuationPathway.P_STOP_SUPPORTIVE_DEESCALATE: 2,
    ContinuationPathway.P_STOP_FORENSIC:              2,
    ContinuationPathway.P_STOP_ESCALATE:              3,
    ContinuationPathway.P_STOP_ESCALATE_EMERGENCY:    4,
}


# ── Canonical output_mode set per pathway_id ──────────────────────────────────
#
# Each pathway_id has a canonical set of valid output_modes derived from the
# canonical matrix.  P_REFUSE_ESCALATE_PRO, P_REFUSE_EXPLAIN_REDIRECT, and
# P_STOP_TERMINAL each appear with more than one output_mode across different
# domain/authority corridors; the full set is accepted for operator overrides.

_PATHWAY_OUTPUT_MODES: dict[ContinuationPathway, frozenset[OutputMode]] = {}
for _row in _TABLE.values():
    _pid = _row.pathway_id
    if _pid not in _PATHWAY_OUTPUT_MODES:
        _PATHWAY_OUTPUT_MODES[_pid] = frozenset()
    _PATHWAY_OUTPUT_MODES[_pid] = _PATHWAY_OUTPUT_MODES[_pid] | {_row.output_mode}
del _pid, _row


# ── Field classification ──────────────────────────────────────────────────────

#: Operator may not alter these continuation fields.
IMMUTABLE_FIELDS: frozenset[str] = frozenset({"forensic_obligations"})

#: Operator may alter these fields only within defined constitutional bounds.
CONSTRAINED_FIELDS: frozenset[str] = frozenset({
    "commitment_closed",
    "interaction_open",
    "pathway_id",
    "output_mode",
})

#: Operator may alter these fields without governance restriction.
FREE_FIELDS: frozenset[str] = frozenset({"escalation_target"})


# ── Violation record ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OverrideViolation:
    """A single constitutional violation in an operator override entry."""
    field: str
    rule:  str    # which constitutional rule was violated
    reason: str   # human-readable explanation

    def __str__(self) -> str:
        return f"[{self.field} / {self.rule}] {self.reason}"


class OperatorOverrideError(ValueError):
    """Raised when an operator matrix entry violates the override constitution."""

    def __init__(self, key: str, violations: list[OverrideViolation]) -> None:
        self.key = key
        self.violations = violations
        lines = [
            f"Operator override for {key!r} has {len(violations)} "
            f"constitutional violation(s):"
        ]
        for v in violations:
            lines.append(f"  {v}")
        super().__init__("\n".join(lines))


# ── Validation ────────────────────────────────────────────────────────────────

def validate_override(
    canonical_row: ContinuationRow,
    operator_entry: dict,
) -> list[OverrideViolation]:
    """Validate an operator JSON matrix entry against the canonical continuation row.

    Args:
        canonical_row:  The ContinuationRow from CONTINUATION_MATRIX for the key.
                        This is the authoritative baseline.
        operator_entry: The raw JSON dict from the operator matrix for the same
                        key.  Fields absent from this dict are not overridden and
                        are therefore not validated.

    Returns:
        A list of OverrideViolation records.  An empty list means the override
        is constitutionally lawful.  Callers decide whether to raise or collect.

    Does not raise; use OperatorOverrideError to convert a non-empty list to an
    exception.
    """
    violations: list[OverrideViolation] = []

    # ── IMMUTABLE: forensic_obligations ───────────────────────────────────────
    #
    # Any attempt to set forensic_obligations in the operator entry is a
    # violation — whether the attempt adds, removes, or substitutes obligations.

    if "forensic_obligations" in operator_entry:
        canonical_fo = frozenset(fo.value for fo in canonical_row.forensic_obligations)
        raw_fo = operator_entry["forensic_obligations"]

        # Container type check: must be a list before iterating.
        if not isinstance(raw_fo, list):
            violations.append(OverrideViolation(
                field="forensic_obligations",
                rule="IMMUTABLE/type",
                reason=(
                    f"forensic_obligations must be a list, got {type(raw_fo).__name__!r}. "
                    f"forensic_obligations is immutable — value must match the canonical "
                    f"obligation set exactly."
                ),
            ))
            return violations  # cannot safely iterate; skip remaining checks

        # Validate each value is a known ForensicObligation.
        op_fo: frozenset[str] = frozenset()
        unknown: list[str] = []
        for val in raw_fo:
            try:
                op_fo = op_fo | {ForensicObligation(val).value}
            except ValueError:
                unknown.append(repr(val))
        if unknown:
            violations.append(OverrideViolation(
                field="forensic_obligations",
                rule="IMMUTABLE",
                reason=(
                    f"contains unknown ForensicObligation value(s): "
                    f"{', '.join(unknown)}. forensic_obligations is immutable."
                ),
            ))

        removed = canonical_fo - op_fo
        if removed:
            violations.append(OverrideViolation(
                field="forensic_obligations",
                rule="IMMUTABLE/no-removal",
                reason=(
                    f"operator removed canonical obligation(s) {sorted(removed)!r}. "
                    f"forensic_obligations are set by governance law and may not be "
                    f"reduced by operator override."
                ),
            ))

        added = op_fo - canonical_fo
        if added:
            violations.append(OverrideViolation(
                field="forensic_obligations",
                rule="IMMUTABLE/no-addition",
                reason=(
                    f"operator added obligation(s) {sorted(added)!r}. "
                    f"forensic_obligations are immutable; additions must be made "
                    f"in the canonical matrix, not operator overrides."
                ),
            ))

    # ── CONSTRAINED: commitment_closed ────────────────────────────────────────
    #
    # Operators may tighten (False → True) but not relax (True → False).

    if "commitment_closed" in operator_entry:
        op_cc = operator_entry["commitment_closed"]
        if not isinstance(op_cc, bool):
            violations.append(OverrideViolation(
                field="commitment_closed",
                rule="CONSTRAINED/type",
                reason=f"must be a bool, got {type(op_cc).__name__!r}",
            ))
        elif canonical_row.commitment_closed and not op_cc:
            violations.append(OverrideViolation(
                field="commitment_closed",
                rule="CONSTRAINED/no-widening",
                reason=(
                    "operator set commitment_closed=False but canonical row has True. "
                    "Operators may only tighten commitment (False→True). "
                    "Relaxing it widens the epistemic commitment corridor — the system "
                    "could assert, infer, or imply determinations that governance law "
                    "has closed."
                ),
            ))

    # ── CONSTRAINED: interaction_open ─────────────────────────────────────────
    #
    # Operators may restrict (True → False) but not open (False → True).

    if "interaction_open" in operator_entry:
        op_io = operator_entry["interaction_open"]
        if not isinstance(op_io, bool):
            violations.append(OverrideViolation(
                field="interaction_open",
                rule="CONSTRAINED/type",
                reason=f"must be a bool, got {type(op_io).__name__!r}",
            ))
        elif not canonical_row.interaction_open and op_io:
            violations.append(OverrideViolation(
                field="interaction_open",
                rule="CONSTRAINED/no-widening",
                reason=(
                    "operator set interaction_open=True but canonical row has False. "
                    "Operators may only restrict interaction (True→False). "
                    "Opening interaction against canonical law re-enables a "
                    "conversation that governance requires to be closed."
                ),
            ))

    # ── CONSTRAINED: pathway_id ───────────────────────────────────────────────
    #
    # Operator pathway semantic class must be ≥ canonical.
    # ADMIT(0) < ASK(1) < REFUSE(2) < STOP(3).

    op_pathway: Optional[ContinuationPathway] = None
    if "pathway_id" in operator_entry:
        try:
            op_pathway = ContinuationPathway(operator_entry["pathway_id"])
        except ValueError:
            violations.append(OverrideViolation(
                field="pathway_id",
                rule="CONSTRAINED/unknown-value",
                reason=f"unknown ContinuationPathway value {operator_entry['pathway_id']!r}",
            ))
        else:
            canonical_cls = _PATHWAY_SEMANTIC_CLASS.get(canonical_row.pathway_id, 0)
            op_cls = _PATHWAY_SEMANTIC_CLASS.get(op_pathway, 0)
            if op_cls < canonical_cls:
                violations.append(OverrideViolation(
                    field="pathway_id",
                    rule="CONSTRAINED/no-weakening",
                    reason=(
                        f"operator mapped {canonical_row.pathway_id.value!r} "
                        f"({_CLASS_NAMES[canonical_cls]}-class, level {canonical_cls}) to "
                        f"{op_pathway.value!r} "
                        f"({_CLASS_NAMES[op_cls]}-class, level {op_cls}). "
                        f"Substitution {_CLASS_NAMES[canonical_cls]}→{_CLASS_NAMES[op_cls]} "
                        f"is a weakening. Only same-class or more-restrictive substitutions "
                        f"are permitted."
                    ),
                ))
            elif op_cls == canonical_cls:
                # Same-class substitution: operator sub-level must be ≥ canonical sub-level.
                canonical_sub = _PATHWAY_SUB_LEVEL.get(canonical_row.pathway_id, 0)
                op_sub = _PATHWAY_SUB_LEVEL.get(op_pathway, 0)
                if op_sub < canonical_sub:
                    violations.append(OverrideViolation(
                        field="pathway_id",
                        rule="CONSTRAINED/intra-class-weakening",
                        reason=(
                            f"operator substituted {canonical_row.pathway_id.value!r} "
                            f"(sub-level {canonical_sub}) with {op_pathway.value!r} "
                            f"(sub-level {op_sub}) within the same "
                            f"{_CLASS_NAMES[canonical_cls]}-class. "
                            f"Same-class substitutions may only move to equal or higher "
                            f"sub-level (higher governance weight)."
                        ),
                    ))

    # ── CONSTRAINED: output_mode ──────────────────────────────────────────────
    #
    # If set, output_mode must belong to the canonical set of valid output_modes
    # for the effective pathway_id.

    if "output_mode" in operator_entry:
        try:
            op_om = OutputMode(operator_entry["output_mode"])
        except ValueError:
            violations.append(OverrideViolation(
                field="output_mode",
                rule="CONSTRAINED/unknown-value",
                reason=f"unknown OutputMode value {operator_entry['output_mode']!r}",
            ))
        else:
            effective_pathway = (
                op_pathway if op_pathway is not None else canonical_row.pathway_id
            )
            valid_modes = _PATHWAY_OUTPUT_MODES.get(effective_pathway, frozenset())
            if not valid_modes:
                violations.append(OverrideViolation(
                    field="output_mode",
                    rule="CONSTRAINED/no-canonical-mode-set",
                    reason=(
                        f"pathway_id={effective_pathway.value!r} has no canonical "
                        f"output_mode set in the continuation matrix. output_mode "
                        f"overrides are rejected when the baseline is unknown "
                        f"(fail-closed)."
                    ),
                ))
            elif op_om not in valid_modes:
                violations.append(OverrideViolation(
                    field="output_mode",
                    rule="CONSTRAINED/pathway-consistency",
                    reason=(
                        f"output_mode={op_om.value!r} is not valid for "
                        f"pathway_id={effective_pathway.value!r}. "
                        f"Valid modes for this pathway: "
                        f"{sorted(m.value for m in valid_modes)!r}."
                    ),
                ))

    # ── FREE (with type constraint): escalation_target ────────────────────────
    #
    # Operators may specify any resource string or None.  Any other type is
    # rejected — a non-string escalation_target would be surfaced to users and
    # could cause downstream rendering or logging failures.

    if "escalation_target" in operator_entry:
        et = operator_entry["escalation_target"]
        if et is not None and not isinstance(et, str):
            violations.append(OverrideViolation(
                field="escalation_target",
                rule="FREE/type",
                reason=(
                    f"escalation_target must be a str or None, "
                    f"got {type(et).__name__!r} ({et!r}). "
                    f"Non-string values cannot be safely rendered or logged."
                ),
            ))

    return violations
