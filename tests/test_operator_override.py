"""Contract tests for operator matrix override constitutional validation.

Tests are organised into four classes:

  TestFieldClassification     — IMMUTABLE / CONSTRAINED / FREE sets are disjoint
                                and cover all governed continuation fields
  TestLawfulOverrides         — validate_override() returns [] for lawful entries
  TestUnlawfulOverrides       — validate_override() returns the correct violations
                                for every distinct constitutional violation
  TestResolverEnforcement     — PolicyResolver raises OperatorOverrideError at
                                load time for unlawful matrices; succeeds for lawful
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from governor.continuation_matrix import CONTINUATION_MATRIX, _TABLE
from governor.models import (
    AuthorityClass,
    ContinuationPathway,
    Domain,
    ForensicObligation,
    LensStatus,
    OutputMode,
)
from governor.operator_override import (
    CONSTRAINED_FIELDS,
    FREE_FIELDS,
    IMMUTABLE_FIELDS,
    OperatorOverrideError,
    OverrideViolation,
    _PATHWAY_OUTPUT_MODES,
    _PATHWAY_SEMANTIC_CLASS,
    _PATHWAY_SUB_LEVEL,
    validate_override,
)
import governor.operator_override as _oo_module  # for monkeypatching
from governor.resolver import PolicyResolver


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(key: str):
    """Return the canonical ContinuationRow for a _TABLE key."""
    return _TABLE[key]


def _violations(key: str, op: dict) -> list[OverrideViolation]:
    return validate_override(_row(key), op)


def _violation_rules(key: str, op: dict) -> list[str]:
    return [v.rule for v in _violations(key, op)]


def _make_operator_matrix(key: str, override: dict) -> dict:
    """Build a minimal operator matrix JSON structure from the bundled matrix
    with one key overridden.  Only the target key is present alongside the
    metadata fields so the validator's loop stays focused.
    """
    bundled = (
        Path(__file__).resolve().parent.parent
        / "aurora_lens"
        / "governor"
        / "policy_matrix.json"
    )
    with open(bundled, encoding="utf-8-sig") as f:
        base = json.load(f)
    base[key] = override
    return base


def _write_matrix(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "operator_matrix.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ── Fixtures: canonical row samples ──────────────────────────────────────────

# general:GP:STOP — commitment_closed=True, interaction_open=False, P_STOP_TERMINAL
_STOP_KEY = "general:GP:STOP"
_STOP_ROW = _row(_STOP_KEY)

# general:GP:ASK — commitment_closed=True, interaction_open=True, P_ASK_MISSING_FACT
_ASK_KEY = "general:GP:ASK"
_ASK_ROW = _row(_ASK_KEY)

# general:GP:REFUSE — commitment_closed=True, interaction_open=True, P_REFUSE_EXPLAIN_REDIRECT
_REFUSE_KEY = "general:GP:REFUSE"
_REFUSE_ROW = _row(_REFUSE_KEY)

# general:GP:ADMIT — commitment_closed=False, interaction_open=True, P_ADMIT_STANDARD
_ADMIT_KEY = "general:GP:ADMIT"
_ADMIT_ROW = _row(_ADMIT_KEY)


# ── TestFieldClassification ───────────────────────────────────────────────────

class TestFieldClassification:
    """The three field sets (IMMUTABLE / CONSTRAINED / FREE) must be disjoint
    and collectively cover all continuation fields that operators may attempt
    to touch via their JSON matrices."""

    GOVERNED_CONTINUATION_FIELDS = {
        "forensic_obligations",
        "commitment_closed",
        "interaction_open",
        "pathway_id",
        "output_mode",
        "escalation_target",
    }

    def test_immutable_constrained_disjoint(self):
        assert not (IMMUTABLE_FIELDS & CONSTRAINED_FIELDS)

    def test_immutable_free_disjoint(self):
        assert not (IMMUTABLE_FIELDS & FREE_FIELDS)

    def test_constrained_free_disjoint(self):
        assert not (CONSTRAINED_FIELDS & FREE_FIELDS)

    def test_union_covers_all_governed_fields(self):
        covered = IMMUTABLE_FIELDS | CONSTRAINED_FIELDS | FREE_FIELDS
        assert self.GOVERNED_CONTINUATION_FIELDS <= covered

    def test_forensic_obligations_is_immutable(self):
        assert "forensic_obligations" in IMMUTABLE_FIELDS

    def test_commitment_closed_is_constrained(self):
        assert "commitment_closed" in CONSTRAINED_FIELDS

    def test_interaction_open_is_constrained(self):
        assert "interaction_open" in CONSTRAINED_FIELDS

    def test_pathway_id_is_constrained(self):
        assert "pathway_id" in CONSTRAINED_FIELDS

    def test_output_mode_is_constrained(self):
        assert "output_mode" in CONSTRAINED_FIELDS

    def test_escalation_target_is_free(self):
        assert "escalation_target" in FREE_FIELDS

    def test_pathway_semantic_classes_cover_all_pathways(self):
        """Every ContinuationPathway must have a semantic class assigned."""
        for p in ContinuationPathway:
            assert p in _PATHWAY_SEMANTIC_CLASS, (
                f"{p.value!r} is missing from _PATHWAY_SEMANTIC_CLASS"
            )

    def test_pathway_output_modes_derived_from_table(self):
        """_PATHWAY_OUTPUT_MODES must include at least the modes seen in _TABLE."""
        for row in _TABLE.values():
            modes = _PATHWAY_OUTPUT_MODES.get(row.pathway_id, frozenset())
            assert row.output_mode in modes, (
                f"{row.pathway_id.value!r} mode {row.output_mode.value!r} "
                f"not in _PATHWAY_OUTPUT_MODES"
            )


# ── TestLawfulOverrides ───────────────────────────────────────────────────────

class TestLawfulOverrides:
    """Entries that comply with every constitutional rule produce no violations."""

    def test_empty_operator_entry_is_lawful(self):
        assert _violations(_STOP_KEY, {}) == []

    def test_free_escalation_target_is_lawful(self):
        assert _violations(_STOP_KEY, {"escalation_target": "1800-MY-RESOURCE"}) == []

    def test_escalation_target_none_is_lawful(self):
        assert _violations(_STOP_KEY, {"escalation_target": None}) == []

    def test_tightening_commitment_false_to_true_is_lawful(self):
        """ADMIT row has commitment_closed=False; operator may tighten to True."""
        assert _ADMIT_ROW.commitment_closed is False
        assert _violations(_ADMIT_KEY, {"commitment_closed": True}) == []

    def test_commitment_closed_same_value_is_lawful(self):
        """Operator repeating the canonical value is a no-op and is lawful."""
        assert _violations(_STOP_KEY, {"commitment_closed": True}) == []
        assert _violations(_ADMIT_KEY, {"commitment_closed": False}) == []

    def test_restricting_interaction_open_true_to_false_is_lawful(self):
        """ASK row has interaction_open=True; operator may restrict to False."""
        assert _ASK_ROW.interaction_open is True
        assert _violations(_ASK_KEY, {"interaction_open": False}) == []

    def test_interaction_open_same_value_is_lawful(self):
        assert _violations(_STOP_KEY, {"interaction_open": False}) == []
        assert _violations(_ASK_KEY, {"interaction_open": True}) == []

    def test_same_class_pathway_substitution_is_lawful(self):
        """STOP → STOP: same class (3 → 3) is lawful."""
        assert _violations(_STOP_KEY, {"pathway_id": ContinuationPathway.P_STOP_FORENSIC.value}) == []

    def test_more_restrictive_pathway_substitution_is_lawful(self):
        """REFUSE → STOP: class 2 → class 3 is a tightening and is lawful."""
        assert _violations(_REFUSE_KEY, {"pathway_id": ContinuationPathway.P_STOP_TERMINAL.value}) == []

    def test_valid_output_mode_for_pathway_is_lawful(self):
        """terminal_stop is one of the valid modes for P_STOP_TERMINAL."""
        op = {
            "pathway_id": ContinuationPathway.P_STOP_TERMINAL.value,
            "output_mode": OutputMode.TERMINAL_STOP.value,
        }
        assert _violations(_STOP_KEY, op) == []

    def test_output_mode_consistent_with_operator_pathway_is_lawful(self):
        """forensic_stop is valid for P_STOP_TERMINAL even though the canonical
        row's output_mode is terminal_stop — the effective pathway governs."""
        op = {
            "pathway_id": ContinuationPathway.P_STOP_FORENSIC.value,
            "output_mode": OutputMode.FORENSIC_STOP.value,
        }
        assert _violations(_STOP_KEY, op) == []

    def test_all_table_rows_empty_override_is_lawful(self):
        """Every canonical row must tolerate an empty operator entry."""
        for key in _TABLE:
            viols = validate_override(_TABLE[key], {})
            assert viols == [], f"Row {key!r}: unexpected violation from empty override"


# ── TestUnlawfulOverrides ─────────────────────────────────────────────────────

class TestUnlawfulOverrides:
    """Every distinct constitutional violation must be detected and reported."""

    # ── forensic_obligations (IMMUTABLE) ──────────────────────────────────────

    def test_removing_canonical_forensic_obligation_is_unlawful(self):
        """Operator may not remove a canonical obligation."""
        assert _ASK_ROW.forensic_obligations  # must have at least one
        rules = _violation_rules(_ASK_KEY, {"forensic_obligations": []})
        assert "IMMUTABLE/no-removal" in rules

    def test_adding_forensic_obligation_is_unlawful(self):
        """Operator may not add an obligation not in the canonical row."""
        # Use ADMIT row which has no obligations; operator adds one.
        assert not _ADMIT_ROW.forensic_obligations
        op = {"forensic_obligations": [ForensicObligation.EMIT_FORENSIC_ENVELOPE.value]}
        rules = _violation_rules(_ADMIT_KEY, op)
        assert "IMMUTABLE/no-addition" in rules

    def test_unknown_forensic_obligation_value_is_unlawful(self):
        op = {"forensic_obligations": ["totally_fake_obligation"]}
        rules = _violation_rules(_ASK_KEY, op)
        assert "IMMUTABLE" in rules

    def test_substituting_forensic_obligations_is_double_violation(self):
        """Replacing canonical obligations with a different set should flag
        both removal of canonical and addition of new."""
        # ASK row has emit_forensic_envelope; replace with attach_pef_snapshot.
        assert _ASK_ROW.forensic_obligations
        op = {"forensic_obligations": [ForensicObligation.ATTACH_PEF_SNAPSHOT.value]}
        rules = _violation_rules(_ASK_KEY, op)
        assert "IMMUTABLE/no-removal" in rules
        assert "IMMUTABLE/no-addition" in rules

    # ── commitment_closed (CONSTRAINED) ───────────────────────────────────────

    def test_relaxing_commitment_closed_true_to_false_is_unlawful(self):
        """STOP row has commitment_closed=True; operator may not set False."""
        assert _STOP_ROW.commitment_closed is True
        rules = _violation_rules(_STOP_KEY, {"commitment_closed": False})
        assert "CONSTRAINED/no-widening" in rules

    def test_commitment_closed_non_bool_is_unlawful(self):
        rules = _violation_rules(_STOP_KEY, {"commitment_closed": 1})
        assert "CONSTRAINED/type" in rules

    def test_commitment_closed_string_is_unlawful(self):
        rules = _violation_rules(_STOP_KEY, {"commitment_closed": "true"})
        assert "CONSTRAINED/type" in rules

    # ── interaction_open (CONSTRAINED) ────────────────────────────────────────

    def test_widening_interaction_open_false_to_true_is_unlawful(self):
        """STOP row has interaction_open=False; operator may not set True."""
        assert _STOP_ROW.interaction_open is False
        rules = _violation_rules(_STOP_KEY, {"interaction_open": True})
        assert "CONSTRAINED/no-widening" in rules

    def test_interaction_open_non_bool_is_unlawful(self):
        rules = _violation_rules(_STOP_KEY, {"interaction_open": 0})
        assert "CONSTRAINED/type" in rules

    # ── pathway_id (CONSTRAINED) ──────────────────────────────────────────────

    def test_stop_to_refuse_is_unlawful(self):
        """STOP(3) → REFUSE(2): weakening."""
        rules = _violation_rules(
            _STOP_KEY,
            {"pathway_id": ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT.value},
        )
        assert "CONSTRAINED/no-weakening" in rules

    def test_stop_to_ask_is_unlawful(self):
        """STOP(3) → ASK(1): weakening."""
        rules = _violation_rules(
            _STOP_KEY,
            {"pathway_id": ContinuationPathway.P_ASK_DISAMBIGUATE.value},
        )
        assert "CONSTRAINED/no-weakening" in rules

    def test_stop_to_admit_is_unlawful(self):
        """STOP(3) → ADMIT(0): weakening."""
        rules = _violation_rules(
            _STOP_KEY,
            {"pathway_id": ContinuationPathway.P_ADMIT_STANDARD.value},
        )
        assert "CONSTRAINED/no-weakening" in rules

    def test_refuse_to_ask_is_unlawful(self):
        """REFUSE(2) → ASK(1): weakening."""
        rules = _violation_rules(
            _REFUSE_KEY,
            {"pathway_id": ContinuationPathway.P_ASK_MISSING_FACT.value},
        )
        assert "CONSTRAINED/no-weakening" in rules

    def test_refuse_to_admit_is_unlawful(self):
        """REFUSE(2) → ADMIT(0): weakening."""
        rules = _violation_rules(
            _REFUSE_KEY,
            {"pathway_id": ContinuationPathway.P_ADMIT_STANDARD.value},
        )
        assert "CONSTRAINED/no-weakening" in rules

    def test_ask_to_admit_is_unlawful(self):
        """ASK(1) → ADMIT(0): weakening."""
        rules = _violation_rules(
            _ASK_KEY,
            {"pathway_id": ContinuationPathway.P_ADMIT_STANDARD.value},
        )
        assert "CONSTRAINED/no-weakening" in rules

    def test_unknown_pathway_id_is_unlawful(self):
        rules = _violation_rules(_STOP_KEY, {"pathway_id": "P_TOTALLY_FAKE"})
        assert "CONSTRAINED/unknown-value" in rules

    # ── output_mode (CONSTRAINED) ─────────────────────────────────────────────

    def test_unknown_output_mode_is_unlawful(self):
        rules = _violation_rules(_STOP_KEY, {"output_mode": "not_a_real_mode"})
        assert "CONSTRAINED/unknown-value" in rules

    def test_output_mode_inconsistent_with_effective_pathway_is_unlawful(self):
        """refusal_with_explanation belongs to REFUSE pathways, not STOP."""
        op = {
            "pathway_id": ContinuationPathway.P_STOP_TERMINAL.value,
            "output_mode": OutputMode.REFUSAL_WITH_EXPLANATION.value,
        }
        rules = _violation_rules(_STOP_KEY, op)
        assert "CONSTRAINED/pathway-consistency" in rules

    def test_output_mode_inconsistent_with_canonical_pathway_is_unlawful(self):
        """terminal_stop is a STOP mode; applying it to a REFUSE row (with no
        pathway_id override) must be rejected."""
        op = {"output_mode": OutputMode.TERMINAL_STOP.value}
        rules = _violation_rules(_REFUSE_KEY, op)
        assert "CONSTRAINED/pathway-consistency" in rules

    # ── Multiple simultaneous violations ──────────────────────────────────────

    def test_multiple_violations_are_all_reported(self):
        """A single bad entry can carry several violations; all must be present."""
        op = {
            "commitment_closed": False,   # CONSTRAINED/no-widening (STOP row is True)
            "pathway_id": ContinuationPathway.P_ADMIT_STANDARD.value,  # CONSTRAINED/no-weakening
            "forensic_obligations": [],   # IMMUTABLE/no-removal
        }
        viols = _violations(_STOP_KEY, op)
        rules = [v.rule for v in viols]
        assert "CONSTRAINED/no-widening" in rules
        assert "CONSTRAINED/no-weakening" in rules
        assert "IMMUTABLE/no-removal" in rules

    # ── OverrideViolation helpers ──────────────────────────────────────────────

    def test_violation_str_format(self):
        viols = _violations(_STOP_KEY, {"commitment_closed": False})
        assert viols
        s = str(viols[0])
        assert "[commitment_closed" in s
        assert "CONSTRAINED/no-widening" in s

    def test_operator_override_error_str_contains_key(self):
        err = OperatorOverrideError("general:GP:STOP", _violations(_STOP_KEY, {"commitment_closed": False}))
        assert "general:GP:STOP" in str(err)
        assert err.key == "general:GP:STOP"


# ── TestResolverEnforcement ────────────────────────────────────────────────────

class TestResolverEnforcement:
    """PolicyResolver rejects unlawful operator matrices at load time."""

    def _bundled_as_operator(self, tmp_path: Path, overrides: dict[str, Any] = None) -> dict:
        """Load the bundled matrix and apply per-key overrides (a dict of key → patch)."""
        bundled = (
        Path(__file__).resolve().parent.parent
        / "aurora_lens"
        / "governor"
        / "policy_matrix.json"
    )
        with open(bundled, encoding="utf-8-sig") as f:
            data = json.load(f)
        if overrides:
            for key, patch in overrides.items():
                if key in data:
                    data[key] = {**data[key], **patch}
                else:
                    data[key] = patch
        return data

    def test_bundled_matrix_loads_without_error(self, tmp_path):
        """The unmodified bundled matrix must be constitutionally valid."""
        data = self._bundled_as_operator(tmp_path)
        path = _write_matrix(tmp_path, data)
        # Must not raise.
        resolver = PolicyResolver(matrix_path=path)
        assert resolver.policy_source == "operator"

    def test_commitment_widening_raises_at_load(self, tmp_path):
        data = self._bundled_as_operator(
            tmp_path, {"general:GP:STOP": {"commitment_closed": False}}
        )
        path = _write_matrix(tmp_path, data)
        with pytest.raises(OperatorOverrideError) as exc_info:
            PolicyResolver(matrix_path=path)
        assert exc_info.value.key == "general:GP:STOP"
        rules = [v.rule for v in exc_info.value.violations]
        assert "CONSTRAINED/no-widening" in rules

    def test_pathway_weakening_raises_at_load(self, tmp_path):
        data = self._bundled_as_operator(
            tmp_path,
            {"general:GP:STOP": {"pathway_id": ContinuationPathway.P_ADMIT_STANDARD.value}},
        )
        path = _write_matrix(tmp_path, data)
        with pytest.raises(OperatorOverrideError):
            PolicyResolver(matrix_path=path)

    def test_forensic_obligations_addition_raises_at_load(self, tmp_path):
        # ADMIT row has no forensic_obligations; adding one is unlawful.
        data = self._bundled_as_operator(
            tmp_path,
            {"general:GP:ADMIT": {
                "forensic_obligations": [ForensicObligation.EMIT_FORENSIC_ENVELOPE.value]
            }},
        )
        path = _write_matrix(tmp_path, data)
        with pytest.raises(OperatorOverrideError) as exc_info:
            PolicyResolver(matrix_path=path)
        rules = [v.rule for v in exc_info.value.violations]
        assert "IMMUTABLE/no-addition" in rules

    def test_lawful_tightening_loads_without_error(self, tmp_path):
        """Tightening commitment on the ADMIT row is constitutionally lawful.

        The operator matrix passes override validation at load time.  Note:
        resolving a policy from this matrix would raise a downstream GovernorPolicy
        validation error (substantive speech acts conflict with commitment_closed=True),
        which is a separate concern from the operator override constitution.
        """
        data = self._bundled_as_operator(
            tmp_path,
            {"general:GP:ADMIT": {"commitment_closed": True}},
        )
        path = _write_matrix(tmp_path, data)
        # Must not raise OperatorOverrideError.
        resolver = PolicyResolver(matrix_path=path)
        assert resolver.policy_source == "operator"

    def test_lawful_escalation_target_override_loads_and_resolves(self, tmp_path):
        """Free field override: custom escalation_target on a STOP row."""
        data = self._bundled_as_operator(
            tmp_path,
            {"general:GP:STOP": {"escalation_target": "https://example.com/crisis"}},
        )
        path = _write_matrix(tmp_path, data)
        resolver = PolicyResolver(matrix_path=path)
        policy = resolver.resolve(Domain.GENERAL, AuthorityClass.GP, LensStatus.STOP)
        assert policy.escalation_target == "https://example.com/crisis"

    def test_lawful_pathway_tightening_loads_and_resolves(self, tmp_path):
        """REFUSE → STOP (class 2→3): lawful tightening."""
        data = self._bundled_as_operator(
            tmp_path,
            {"general:GP:REFUSE": {
                "pathway_id": ContinuationPathway.P_STOP_REFUSE_CLEAN.value,
                "output_mode": OutputMode.TERMINAL_STOP.value,
            }},
        )
        path = _write_matrix(tmp_path, data)
        resolver = PolicyResolver(matrix_path=path)
        policy = resolver.resolve(Domain.GENERAL, AuthorityClass.GP, LensStatus.REFUSE)
        assert policy.pathway_id == ContinuationPathway.P_STOP_REFUSE_CLEAN

    def test_interaction_widening_raises_at_load(self, tmp_path):
        """STOP row has interaction_open=False; operator cannot set True."""
        data = self._bundled_as_operator(
            tmp_path,
            {"general:GP:STOP": {"interaction_open": True}},
        )
        path = _write_matrix(tmp_path, data)
        with pytest.raises(OperatorOverrideError) as exc_info:
            PolicyResolver(matrix_path=path)
        rules = [v.rule for v in exc_info.value.violations]
        assert "CONSTRAINED/no-widening" in rules

    def test_unknown_key_in_operator_matrix_is_silently_ignored(self, tmp_path):
        """Keys not in the canonical matrix are passed through without validation."""
        data = self._bundled_as_operator(tmp_path)
        data["custom:SPECIALIST:STOP"] = {
            "pathway_id": ContinuationPathway.P_STOP_TERMINAL.value
        }
        path = _write_matrix(tmp_path, data)
        # Must not raise — unknown domain/authority combination has no canonical row.
        PolicyResolver(matrix_path=path)

    def test_metadata_keys_are_skipped(self, tmp_path):
        """Keys starting with '_' (e.g. _version, _comment) must not be validated."""
        data = self._bundled_as_operator(tmp_path)
        data["_operator_note"] = "this is not a policy row"
        path = _write_matrix(tmp_path, data)
        # Must not raise.
        PolicyResolver(matrix_path=path)

class TestSubLevelOrdering:
    def test_sub_level_covers_all_pathways(self):
        for p in ContinuationPathway:
            assert p in _PATHWAY_SUB_LEVEL, f"{p.value!r} missing from _PATHWAY_SUB_LEVEL"

    def test_sub_levels_are_non_negative_ints(self):
        for p, level in _PATHWAY_SUB_LEVEL.items():
            assert isinstance(level, int) and level >= 0, (
                f"{p.value!r} has invalid sub-level {level!r}"
            )

    def test_same_pathway_is_lawful(self):
        assert _violations(_STOP_KEY, {"pathway_id": _STOP_ROW.pathway_id.value}) == []

    def test_stop_terminal_to_forensic_is_lawful(self):
        assert _PATHWAY_SUB_LEVEL[ContinuationPathway.P_STOP_TERMINAL] == 1
        assert _PATHWAY_SUB_LEVEL[ContinuationPathway.P_STOP_FORENSIC] == 2
        assert _STOP_ROW.pathway_id == ContinuationPathway.P_STOP_TERMINAL
        assert _violations(_STOP_KEY, {"pathway_id": ContinuationPathway.P_STOP_FORENSIC.value}) == []

    def test_stop_terminal_to_escalate_is_lawful(self):
        assert _violations(_STOP_KEY, {"pathway_id": ContinuationPathway.P_STOP_ESCALATE.value}) == []

    def test_stop_terminal_to_escalate_emergency_is_lawful(self):
        assert _violations(_STOP_KEY, {"pathway_id": ContinuationPathway.P_STOP_ESCALATE_EMERGENCY.value}) == []

    def test_refuse_explain_to_escalate_pro_is_lawful(self):
        assert _REFUSE_ROW.pathway_id == ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT
        assert _violations(_REFUSE_KEY, {"pathway_id": ContinuationPathway.P_REFUSE_ESCALATE_PRO.value}) == []

    def test_cross_class_tightening_skips_sub_level_check(self):
        # REFUSE(2) -> STOP_REFUSE_CLEAN(3, sub 0): class rises, sub-level irrelevant.
        assert _violations(_REFUSE_KEY, {"pathway_id": ContinuationPathway.P_STOP_REFUSE_CLEAN.value}) == []

    def test_stop_forensic_to_terminal_is_unlawful(self):
        forensic_key = next(
            k for k, v in _TABLE.items() if v.pathway_id == ContinuationPathway.P_STOP_FORENSIC
        )
        rules = _violation_rules(forensic_key, {"pathway_id": ContinuationPathway.P_STOP_TERMINAL.value})
        assert "CONSTRAINED/intra-class-weakening" in rules

    def test_stop_escalate_to_terminal_is_unlawful(self):
        escalate_key = next(
            k for k, v in _TABLE.items() if v.pathway_id == ContinuationPathway.P_STOP_ESCALATE
        )
        rules = _violation_rules(escalate_key, {"pathway_id": ContinuationPathway.P_STOP_TERMINAL.value})
        assert "CONSTRAINED/intra-class-weakening" in rules

    def test_stop_escalate_emergency_to_refuse_clean_is_unlawful(self):
        emerg_key = next(
            k for k, v in _TABLE.items()
            if v.pathway_id == ContinuationPathway.P_STOP_ESCALATE_EMERGENCY
        )
        rules = _violation_rules(emerg_key, {"pathway_id": ContinuationPathway.P_STOP_REFUSE_CLEAN.value})
        assert "CONSTRAINED/intra-class-weakening" in rules

    def test_refuse_escalate_pro_to_explain_redirect_is_unlawful(self):
        pro_key = next(
            k for k, v in _TABLE.items()
            if v.pathway_id == ContinuationPathway.P_REFUSE_ESCALATE_PRO
        )
        rules = _violation_rules(pro_key, {"pathway_id": ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT.value})
        assert "CONSTRAINED/intra-class-weakening" in rules

    def test_intra_class_weakening_reason_contains_sub_levels(self):
        forensic_key = next(
            k for k, v in _TABLE.items() if v.pathway_id == ContinuationPathway.P_STOP_FORENSIC
        )
        viols = _violations(forensic_key, {"pathway_id": ContinuationPathway.P_STOP_TERMINAL.value})
        v = next(v for v in viols if v.rule == "CONSTRAINED/intra-class-weakening")
        assert "sub-level" in v.reason
        assert "2" in v.reason  # canonical sub
        assert "1" in v.reason  # operator sub


class TestForensicObligationsContainerType:
    def test_string_is_unlawful(self):
        rules = _violation_rules(_ASK_KEY, {"forensic_obligations": "emit_forensic_envelope"})
        assert "IMMUTABLE/type" in rules

    def test_dict_is_unlawful(self):
        rules = _violation_rules(_ASK_KEY, {"forensic_obligations": {"v": "emit_forensic_envelope"}})
        assert "IMMUTABLE/type" in rules

    def test_int_is_unlawful(self):
        rules = _violation_rules(_STOP_KEY, {"forensic_obligations": 1})
        assert "IMMUTABLE/type" in rules

    def test_none_is_unlawful(self):
        rules = _violation_rules(_STOP_KEY, {"forensic_obligations": None})
        assert "IMMUTABLE/type" in rules

    def test_type_error_returns_exactly_one_fo_violation(self):
        """Type violation triggers early return — no additional forensic_obligations violations."""
        viols = _violations(_ASK_KEY, {"forensic_obligations": "emit_forensic_envelope"})
        fo_viols = [v for v in viols if v.field == "forensic_obligations"]
        assert len(fo_viols) == 1
        assert fo_viols[0].rule == "IMMUTABLE/type"

    def test_empty_list_is_correct_container_type(self):
        # ADMIT row has no obligations; empty list matches exactly → no violation.
        assert not _ADMIT_ROW.forensic_obligations
        fo_type_viols = [
            v for v in _violations(_ADMIT_KEY, {"forensic_obligations": []})
            if v.rule == "IMMUTABLE/type"
        ]
        assert not fo_type_viols


class TestEscalationTargetType:
    def test_string_is_lawful(self):
        assert _violations(_STOP_KEY, {"escalation_target": "1800-CRISIS"}) == []

    def test_none_is_lawful(self):
        assert _violations(_STOP_KEY, {"escalation_target": None}) == []

    def test_empty_string_is_lawful(self):
        assert _violations(_STOP_KEY, {"escalation_target": ""}) == []

    def test_int_is_unlawful(self):
        assert "FREE/type" in _violation_rules(_STOP_KEY, {"escalation_target": 18005551234})

    def test_list_is_unlawful(self):
        assert "FREE/type" in _violation_rules(_STOP_KEY, {"escalation_target": ["1800-CRISIS"]})

    def test_dict_is_unlawful(self):
        assert "FREE/type" in _violation_rules(_STOP_KEY, {"escalation_target": {"url": "x"}})

    def test_bool_is_unlawful(self):
        # bool subclasses int; must still be rejected.
        assert "FREE/type" in _violation_rules(_STOP_KEY, {"escalation_target": True})

    def test_violation_reason_names_actual_type(self):
        viols = _violations(_STOP_KEY, {"escalation_target": 42})
        v = next(v for v in viols if v.rule == "FREE/type")
        assert "int" in v.reason

    def test_type_violation_does_not_suppress_other_violations(self):
        op = {"escalation_target": 999, "commitment_closed": False}
        rules = _violation_rules(_STOP_KEY, op)
        assert "FREE/type" in rules
        assert "CONSTRAINED/no-widening" in rules


class TestOutputModeFailClosed:
    def test_all_canonical_pathways_have_mode_set(self):
        for row in _TABLE.values():
            modes = _PATHWAY_OUTPUT_MODES.get(row.pathway_id, frozenset())
            assert modes, f"{row.pathway_id.value!r} has no canonical output_mode set"

    def test_valid_output_mode_for_pathway_override_is_lawful(self):
        op = {
            "pathway_id": ContinuationPathway.P_STOP_TERMINAL.value,
            "output_mode": OutputMode.FORENSIC_STOP.value,
        }
        assert _violations(_STOP_KEY, op) == []

    def test_output_mode_without_pathway_override_uses_canonical_pathway(self):
        assert _violations(_STOP_KEY, {"output_mode": OutputMode.TERMINAL_STOP.value}) == []

    def test_wrong_output_mode_for_canonical_pathway_is_unlawful(self):
        rules = _violation_rules(_STOP_KEY, {"output_mode": OutputMode.REFUSAL_WITH_EXPLANATION.value})
        assert "CONSTRAINED/pathway-consistency" in rules

    def test_no_canonical_mode_set_triggers_fail_closed(self):
        """Monkeypatch _PATHWAY_OUTPUT_MODES to simulate missing mode set."""
        import governor.operator_override as oo

        saved = oo._PATHWAY_OUTPUT_MODES.get(ContinuationPathway.P_STOP_TERMINAL)
        try:
            del oo._PATHWAY_OUTPUT_MODES[ContinuationPathway.P_STOP_TERMINAL]
            rules = _violation_rules(_STOP_KEY, {"output_mode": OutputMode.TERMINAL_STOP.value})
            assert "CONSTRAINED/no-canonical-mode-set" in rules
        finally:
            if saved is not None:
                oo._PATHWAY_OUTPUT_MODES[ContinuationPathway.P_STOP_TERMINAL] = saved

    def test_pathway_consistency_reason_lists_valid_modes(self):
        viols = _violations(_STOP_KEY, {"output_mode": OutputMode.REFUSAL_WITH_EXPLANATION.value})
        v = next(v for v in viols if v.rule == "CONSTRAINED/pathway-consistency")
        valid = _PATHWAY_OUTPUT_MODES[ContinuationPathway.P_STOP_TERMINAL]
        assert any(m.value in v.reason for m in valid)
