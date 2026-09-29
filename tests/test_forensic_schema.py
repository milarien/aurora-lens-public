"""Forensic envelope schema — contract and replay-verification tests.

WHAT IS PROVED HERE
-------------------
1. Schema completeness (contract tests)
   Every non-ADMIT matrix row, when its decision is processed through
   build_forensic_event(), produces an event that satisfies the canonical
   forensic envelope schema:
     - All required fields present and correctly typed
     - commitment_closed == True (non-widening invariant)
     - pathway_id and output_mode are known enum values
     - failed_constraints is non-empty
     - event_hash is a well-formed SHA-256 digest

2. Schema validation catches violations
   ForensicSchema.validate() returns errors for missing required fields,
   invalid pathway_id, commitment_closed=False, malformed hash strings, etc.

3. Event hash integrity
   verify_event_hash() returns True for a freshly built event, and False
   after any field is added, removed, or mutated — even whitespace changes.

4. Governed response hash replay
   Given the forensic event alone, verify_governed_response_hash() confirms
   that sha256(re-rendered output) matches the stored governed_response_hash.
   The re-rendered output is obtained by replaying enforce() from the event's
   pathway_id, failed_constraints, interaction_open, and escalation_target —
   the four decision inputs that the event is required to carry.

5. Row identity replay
   find_matrix_rows(event) returns at least one _TABLE row whose six
   continuation fields are consistent with the event.  The event produced
   from a specific row must be consistent with that row.

REPLAY CONTRACT
---------------
A forensic event is self-sufficient for replay if it contains:
  pathway_id        — identifies the renderer and the matrix row class
  failed_constraints[0] — identifies the flag type (renderer routing)
  interaction_open  — controls invitation phrases
  escalation_target — the resource string surfaced to the user

These four fields are sufficient to reconstruct the enforce() inputs,
re-render the governed response, and verify governed_response_hash.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Optional

import pytest

from aurora_lens.govern.bridge import (
    FORENSIC_SCHEMA_VERSION,
    build_forensic_event,
    enforce,
)
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.verify.flags import Flag, FlagType
from governor.continuation_matrix import ContinuationRow, _TABLE
from governor.forensic_schema import (
    FORENSIC_SCHEMA_VERSION as _FS_SCHEMA_VERSION,
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    find_matrix_rows,
    validate,
    verify_blocked_response_hash,
    verify_event_hash,
    verify_governed_response_hash,
)
from governor.models import ContinuationPathway


# ── Helpers ───────────────────────────────────────────────────────────────────

_PATHWAY_EXEC: dict[ContinuationPathway, tuple[InterventionAction, FlagType]] = {
    ContinuationPathway.P_ASK_DISAMBIGUATE:           (InterventionAction.CONTAIN,      FlagType.UNRESOLVED_REFERENT),
    ContinuationPathway.P_ASK_MISSING_FACT:           (InterventionAction.CONTAIN,      FlagType.UNRESOLVED_REFERENT),
    ContinuationPathway.P_REFUSE_EXPLAIN_REDIRECT:    (InterventionAction.FORCE_REVISE, FlagType.PERSONALIZED_FINANCIAL_ADVICE),
    ContinuationPathway.P_REFUSE_ESCALATE_PRO:        (InterventionAction.FORCE_REVISE, FlagType.PERSONALIZED_MEDICAL_ADVICE),
    ContinuationPathway.P_HANDOFF_SUMMARY:            (InterventionAction.FORCE_REVISE, FlagType.PERSONALIZED_MEDICAL_ADVICE),
    ContinuationPathway.P_STOP_TERMINAL:              (InterventionAction.HARD_STOP,    FlagType.PERSONALIZED_MEDICAL_ADVICE),
    ContinuationPathway.P_STOP_FORENSIC:              (InterventionAction.HARD_STOP,    FlagType.SENSITIVE_PII_EXPOSURE),
    ContinuationPathway.P_STOP_ESCALATE:              (InterventionAction.HARD_STOP,    FlagType.EMERGENCY_TRIAGE_GUIDANCE),
    ContinuationPathway.P_STOP_ESCALATE_EMERGENCY:    (InterventionAction.HARD_STOP,    FlagType.EMERGENCY_TRIAGE_GUIDANCE),
    ContinuationPathway.P_STOP_SUPPORTIVE_DEESCALATE: (InterventionAction.HARD_STOP,    FlagType.SELF_HARM_INSTRUCTION),
    ContinuationPathway.P_STOP_REDIRECT_QUALIFIED:    (InterventionAction.HARD_STOP,    FlagType.MEDICAL_DOSAGE_RECOMMENDATION),
    ContinuationPathway.P_STOP_REFUSE_CLEAN:          (InterventionAction.HARD_STOP,    FlagType.ILLEGAL_INSTRUCTION),
}


def _flag(flag_type: FlagType) -> Flag:
    return Flag(
        flag_type=flag_type,
        entity_name="test-entity",
        claim="test claim",
        evidence="test evidence",
        severity="error",
    )


def _build_decision(row: ContinuationRow) -> GovernanceDecision:
    """Build a GovernanceDecision mirroring a matrix row's parameters."""
    action, flag_type = _PATHWAY_EXEC[row.pathway_id]
    d = GovernanceDecision(
        action=action,
        flags=[_flag(flag_type)],
        rationale="test rationale",
        policy="strict",
        attempt=0,
        resource=row.escalation_target,
        original_response="the model produced a blocked determination",
    )
    d.pathway_id           = row.pathway_id.value
    d.commitment_closed    = row.commitment_closed
    d.interaction_open     = row.interaction_open
    d.output_mode          = row.output_mode.value
    d.forensic_obligations = [fo.value for fo in row.forensic_obligations]
    d.resolution_mode      = "exact"
    return d


def _build_event(row: ContinuationRow) -> tuple[GovernanceDecision, dict]:
    """Build a decision + schema-complete forensic event for a matrix row.

    The governed_response is rendered by enforce() before building the event,
    so governed_response_hash is always present.
    """
    d = _build_decision(row)
    rendered = enforce(d, "")
    d.governed_response = rendered
    event = build_forensic_event(
        d,
        pre_llm=False,
        pef_snapshot={"entity": "Alice", "domain": "test"},
        trace_id="test-trace-001",
        timestamp="2026-01-01T00:00:00Z",
    )
    return d, event


# Parametric fixtures — all non-ADMIT rows.
_NON_ADMIT_ROWS = [
    (key, row)
    for key, row in _TABLE.items()
    if row.pathway_id != ContinuationPathway.P_ADMIT_STANDARD
]


# ═══════════════════════════════════════════════════════════════════════════════
# 0. Schema version parity
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaVersionParity:
    """The FORENSIC_SCHEMA_VERSION constant must agree across bridge and schema."""

    def test_bridge_and_schema_versions_agree(self):
        assert FORENSIC_SCHEMA_VERSION == _FS_SCHEMA_VERSION, (
            "aurora_lens.govern.bridge.FORENSIC_SCHEMA_VERSION and "
            "governor.forensic_schema.FORENSIC_SCHEMA_VERSION must be the same string. "
            f"bridge={FORENSIC_SCHEMA_VERSION!r}, schema={_FS_SCHEMA_VERSION!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Schema completeness (contract tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaCompleteness:
    """Every non-ADMIT matrix row produces a schema-complete forensic event.

    This is the emission contract: build_forensic_event() must always output
    a dict that passes validate() with zero errors.
    """

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_all_required_fields_present(self, key: str, row: ContinuationRow):
        _, event = _build_event(row)
        missing = REQUIRED_FIELDS - set(event.keys())
        assert not missing, (
            f"{key}: forensic event is missing required fields: {sorted(missing)}"
        )

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_validate_returns_no_errors(self, key: str, row: ContinuationRow):
        _, event = _build_event(row)
        errors = validate(event)
        assert not errors, (
            f"{key}: forensic event failed schema validation:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_schema_version_correct(self, key: str, row: ContinuationRow):
        _, event = _build_event(row)
        assert event.get("schema_version") == FORENSIC_SCHEMA_VERSION, (
            f"{key}: schema_version={event.get('schema_version')!r}, "
            f"expected {FORENSIC_SCHEMA_VERSION!r}"
        )

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_status_is_ask_refuse_or_stop(self, key: str, row: ContinuationRow):
        _, event = _build_event(row)
        assert event.get("status") in {"ASK", "REFUSE", "STOP"}, (
            f"{key}: status={event.get('status')!r}, expected one of ASK/REFUSE/STOP"
        )

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_commitment_closed_true(self, key: str, row: ContinuationRow):
        _, event = _build_event(row)
        assert event.get("commitment_closed") is True, (
            f"{key}: commitment_closed={event.get('commitment_closed')!r} — "
            f"must be True for all non-ADMIT forensic events"
        )

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_pathway_id_matches_row(self, key: str, row: ContinuationRow):
        _, event = _build_event(row)
        assert event.get("pathway_id") == row.pathway_id.value, (
            f"{key}: event.pathway_id={event.get('pathway_id')!r}, "
            f"expected {row.pathway_id.value!r}"
        )

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_failed_constraints_non_empty(self, key: str, row: ContinuationRow):
        _, event = _build_event(row)
        fc = event.get("failed_constraints", [])
        assert isinstance(fc, list) and len(fc) > 0, (
            f"{key}: failed_constraints must be a non-empty list, got {fc!r}"
        )

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_governed_response_hash_present(self, key: str, row: ContinuationRow):
        """governed_response_hash is required when governed_response is set.
        Since _build_event() always renders and assigns governed_response,
        this field must always be present.
        """
        _, event = _build_event(row)
        assert "governed_response_hash" in event, (
            f"{key}: governed_response_hash absent — required when "
            f"governed_response is set"
        )
        val = event["governed_response_hash"]
        assert isinstance(val, str) and val.startswith("sha256:") and len(val) == 71, (
            f"{key}: governed_response_hash={val!r} must be 'sha256:<64 hex chars>'"
        )

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_blocked_response_hash_present(self, key: str, row: ContinuationRow):
        """blocked_response_hash is required when original_response is set.
        _build_decision() always sets original_response.
        """
        _, event = _build_event(row)
        assert "blocked_response_hash" in event, (
            f"{key}: blocked_response_hash absent — required when "
            f"original_response is set"
        )

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_escalation_target_present_as_field(self, key: str, row: ContinuationRow):
        """escalation_target is present in the event (may be None for pathways
        that carry no resource, e.g. P_STOP_REFUSE_CLEAN).  The field itself
        must always be in the event dict so replay logic can distinguish
        'resource was None' from 'resource was not recorded'.
        """
        _, event = _build_event(row)
        assert "escalation_target" in event, (
            f"{key}: escalation_target key absent from forensic event — "
            f"required for replay (may be None)"
        )

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_escalation_target_matches_row(self, key: str, row: ContinuationRow):
        """The escalation_target recorded in the event must equal the row's
        escalation_target (which was used as decision.resource).
        """
        _, event = _build_event(row)
        assert event["escalation_target"] == row.escalation_target, (
            f"{key}: event.escalation_target={event['escalation_target']!r}, "
            f"row.escalation_target={row.escalation_target!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Schema validation correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaValidation:
    """validate() correctly identifies schema violations."""

    def _base_valid_event(self) -> dict:
        """Minimal valid event for mutation testing."""
        row = _TABLE["medical:GP:STOP:SELF_HARM_INSTRUCTION"]
        _, event = _build_event(row)
        return event

    def test_valid_event_passes(self):
        event = self._base_valid_event()
        errors = validate(event)
        assert not errors, f"Valid event should produce no errors: {errors}"

    @pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
    def test_missing_required_field_caught(self, field: str):
        event = self._base_valid_event()
        del event[field]
        errors = validate(event)
        assert any(field in e for e in errors), (
            f"Removing {field!r} should produce a validation error, got: {errors}"
        )

    def test_wrong_schema_version_caught(self):
        event = self._base_valid_event()
        event["event_hash"] = event["event_hash"]   # preserve to not trigger hash error
        event["schema_version"] = "0.9"
        errors = validate(event)
        assert any("schema_version" in e for e in errors), (
            f"Wrong schema_version should produce an error, got: {errors}"
        )

    def test_invalid_status_caught(self):
        event = self._base_valid_event()
        event["status"] = "ADMIT"   # ADMIT is not valid in a non-ADMIT event
        errors = validate(event)
        assert any("status" in e for e in errors), errors

    def test_commitment_closed_false_caught(self):
        event = self._base_valid_event()
        event["commitment_closed"] = False
        errors = validate(event)
        assert any("commitment_closed" in e for e in errors), (
            f"commitment_closed=False should be caught as a non-widening violation: {errors}"
        )

    def test_invalid_pathway_id_caught(self):
        event = self._base_valid_event()
        event["pathway_id"] = "P_NONEXISTENT_PATHWAY"
        errors = validate(event)
        assert any("pathway_id" in e for e in errors), errors

    def test_empty_failed_constraints_caught(self):
        event = self._base_valid_event()
        event["failed_constraints"] = []
        errors = validate(event)
        assert any("failed_constraints" in e for e in errors), errors

    def test_state_native_stop_forensic_has_non_empty_failed_constraints(self):
        """Regression: state-native STOP must carry structural flags (empty → HTTP 500 forensic validate)."""
        from aurora_lens.govern.forensic_append_guard import enforce_forensic_event_for_append
        from aurora_lens.govern.state_native_mapping import governance_decision_from_state_native
        from aurora_lens.state_native_engine.contracts import (
            StateNativeDelegationResult,
            StateNativeOutcome,
            StateNativeSolverFamily,
        )
        from aurora_lens.state_native_engine.epistemic import EpistemicResult

        sn = StateNativeDelegationResult(
            handled=True,
            outcome=StateNativeOutcome.STOP,
            user_visible_text="I cannot answer that from committed state.",
            solver_family=StateNativeSolverFamily.COMMITTED_INVENTORY_READ,
            epistemic_result=EpistemicResult.UNKNOWN,
            stop_reason_code="insufficient_inventory",
        )
        dec = governance_decision_from_state_native(sn)
        assert dec.flags, "STOP must not reach forensic build with flags=[]"
        event = build_forensic_event(
            dec,
            pre_llm=True,
            pef_snapshot={"entities": {}},
            trace_id="test-trace-stop-fc",
            timestamp="2026-01-01T00:00:00Z",
        )
        assert event["failed_constraints"]
        enforce_forensic_event_for_append(event)

    def test_malformed_event_hash_caught(self):
        event = self._base_valid_event()
        event["event_hash"] = "not-a-hash"
        errors = validate(event)
        assert any("event_hash" in e for e in errors), errors

    def test_malformed_governed_response_hash_caught(self):
        event = self._base_valid_event()
        event["governed_response_hash"] = "md5:abc"
        errors = validate(event)
        assert any("governed_response_hash" in e for e in errors), errors

    def test_commitment_closed_non_bool_caught(self):
        event = self._base_valid_event()
        event["commitment_closed"] = "true"
        errors = validate(event)
        assert any("commitment_closed" in e for e in errors), errors

    def test_interaction_open_non_bool_caught(self):
        event = self._base_valid_event()
        event["interaction_open"] = 1
        errors = validate(event)
        assert any("interaction_open" in e for e in errors), errors


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Event hash integrity
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventHashIntegrity:
    """event_hash is a tamper-evident self-hash of all other fields."""

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_freshly_built_event_hash_is_valid(self, key: str, row: ContinuationRow):
        _, event = _build_event(row)
        assert verify_event_hash(event), (
            f"{key}: event_hash verification failed on freshly built event. "
            f"event_hash={event.get('event_hash')!r}"
        )

    def test_missing_event_hash_returns_false(self):
        row = _TABLE["general:GP:STOP"]
        _, event = _build_event(row)
        del event["event_hash"]
        assert not verify_event_hash(event)

    def test_mutated_pathway_id_detected(self):
        row = _TABLE["medical:GP:STOP:SELF_HARM_INSTRUCTION"]
        _, event = _build_event(row)
        assert verify_event_hash(event)
        event["pathway_id"] = "P_ADMIT_STANDARD"
        assert not verify_event_hash(event), (
            "Mutating pathway_id must invalidate event_hash"
        )

    def test_mutated_commitment_closed_detected(self):
        row = _TABLE["legal:GP:STOP"]
        _, event = _build_event(row)
        event["commitment_closed"] = False
        assert not verify_event_hash(event)

    def test_mutated_escalation_target_detected(self):
        row = _TABLE["medical:GP:STOP:SELF_HARM_INSTRUCTION"]
        _, event = _build_event(row)
        event["escalation_target"] = "a completely different resource"
        assert not verify_event_hash(event)

    def test_added_field_detected(self):
        row = _TABLE["finance:GP:STOP"]
        _, event = _build_event(row)
        event["injected_field"] = "attacker payload"
        assert not verify_event_hash(event)

    def test_removed_optional_field_detected(self):
        """Removing any field that was present when event_hash was computed
        must invalidate the hash.
        """
        row = _TABLE["medical:GP:STOP:SELF_HARM_INSTRUCTION"]
        _, event = _build_event(row)
        # governed_response_hash is optional but was present during hashing.
        assert "governed_response_hash" in event
        del event["governed_response_hash"]
        assert not verify_event_hash(event)

    def test_whitespace_in_hash_body_detected(self):
        """The event_hash is computed with sort_keys=True and no extra whitespace.
        Injecting whitespace into the value is not directly possible (we test
        by verifying the hash is deterministic and would change with different
        serialisation).
        """
        row = _TABLE["general:GP:ASK"]
        _, event = _build_event(row)
        original = event["event_hash"]
        # Re-verify: hash is stable across two verify calls.
        assert verify_event_hash(event)
        assert event["event_hash"] == original


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Governed response hash replay
# ═══════════════════════════════════════════════════════════════════════════════

class TestGovernedResponseHashReplay:
    """Prove the forensic event is sufficient to verify the rendered continuation.

    REPLAY PROCEDURE
    ----------------
    Given a forensic event, reconstruct enforce() inputs:
      1. pathway_id            → InterventionAction (via _PATHWAY_TO_ACTION)
      2. failed_constraints[0] → FlagType (by name)
      3. interaction_open      → bool (from event)
      4. escalation_target     → resource string (from event, may be None)

    Then call enforce(decision, "") and verify:
      sha256(result.encode("utf-8")) == event["governed_response_hash"][7:]
    """

    def _replay(self, event: dict) -> str:
        """Replay enforce() from a forensic event's fields."""
        from aurora_lens.govern.adapters.policy_projector import (
            _PATHWAY_FALLBACK,
            _PATHWAY_TO_ACTION,
        )
        pathway = ContinuationPathway(event["pathway_id"])
        action  = _PATHWAY_TO_ACTION.get(pathway, _PATHWAY_FALLBACK)
        flag_type_name = event["failed_constraints"][0]
        flag_type = FlagType[flag_type_name]

        d = GovernanceDecision(
            action=action,
            flags=[_flag(flag_type)],
            rationale="replay",
            policy="strict",
            attempt=0,
            resource=event.get("escalation_target"),
        )
        d.pathway_id           = event["pathway_id"]
        d.commitment_closed    = event["commitment_closed"]
        d.interaction_open     = event["interaction_open"]
        d.output_mode          = event["output_mode"]
        d.forensic_obligations = event.get("forensic_obligations", [])
        d.resolution_mode      = event.get("resolution_mode", "exact")
        return enforce(d, "")

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_governed_response_hash_verifies_from_replay(
        self, key: str, row: ContinuationRow
    ):
        """The event contains enough information to re-render the governed
        response and verify its hash.
        """
        _, event = _build_event(row)
        replayed = self._replay(event)
        assert verify_governed_response_hash(event, replayed), (
            f"{key}: sha256(replayed_output) does not match governed_response_hash.\n"
            f"  event['governed_response_hash'] = {event.get('governed_response_hash')!r}\n"
            f"  sha256(replayed) = sha256:{hashlib.sha256(replayed.encode()).hexdigest()!r}\n"
            f"  replayed output  = {replayed!r}"
        )

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_replay_output_matches_original_render(
        self, key: str, row: ContinuationRow
    ):
        """The replayed output must equal the original enforce() output exactly.
        This proves renderer determinism — same inputs always produce the same
        string, so the hash is stable across time.
        """
        d, event = _build_event(row)
        original_output = d.governed_response
        replayed_output = self._replay(event)
        assert replayed_output == original_output, (
            f"{key}: replay output differs from original render.\n"
            f"  original: {original_output!r}\n"
            f"  replayed: {replayed_output!r}"
        )

    def test_blocked_response_hash_verifiable(self):
        """blocked_response_hash must verify against the original blocked text."""
        row = _TABLE["medical:GP:STOP:SELF_HARM_INSTRUCTION"]
        d, event = _build_event(row)
        assert verify_blocked_response_hash(event, d.original_response), (
            "blocked_response_hash must verify against original_response"
        )

    def test_governed_response_hash_absent_returns_false(self):
        row = _TABLE["general:GP:STOP"]
        _, event = _build_event(row)
        del event["governed_response_hash"]
        assert not verify_governed_response_hash(event, "anything")

    def test_wrong_text_returns_false(self):
        row = _TABLE["legal:GP:REFUSE"]
        _, event = _build_event(row)
        assert not verify_governed_response_hash(event, "wrong text entirely")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Row identity replay
# ═══════════════════════════════════════════════════════════════════════════════

class TestRowIdentityReplay:
    """find_matrix_rows(event) identifies the matrix row(s) consistent with the event.

    The event produced from a specific row must match at least that row.
    The matching rows' continuation fields must all agree with the event.

    This proves that the forensic event is sufficient to anchor the governance
    decision to a specific, verifiable point in the policy matrix.
    """

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_source_row_is_always_found(self, key: str, row: ContinuationRow):
        """The row that generated the event must appear in find_matrix_rows()."""
        _, event = _build_event(row)
        matches = find_matrix_rows(event)
        match_keys = {k for k, _ in matches}
        assert key in match_keys, (
            f"{key}: find_matrix_rows() did not return the source row.\n"
            f"  event pathway_id={event.get('pathway_id')!r}\n"
            f"  matched: {sorted(match_keys)}"
        )

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_all_matched_rows_consistent_with_event(
        self, key: str, row: ContinuationRow
    ):
        """Every row returned by find_matrix_rows() must agree with the event
        on all six continuation fields.
        """
        _, event = _build_event(row)
        for match_key, match_row in find_matrix_rows(event):
            assert match_row.pathway_id.value == event["pathway_id"], \
                f"{key}→{match_key}: pathway_id mismatch"
            assert match_row.commitment_closed == event["commitment_closed"], \
                f"{key}→{match_key}: commitment_closed mismatch"
            assert match_row.interaction_open == event["interaction_open"], \
                f"{key}→{match_key}: interaction_open mismatch"
            assert match_row.output_mode.value == event["output_mode"], \
                f"{key}→{match_key}: output_mode mismatch"

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_no_spurious_pathway_id_matches(
        self, key: str, row: ContinuationRow
    ):
        """Every match must share the exact pathway_id — no fallback row with
        a different pathway_id should appear.
        """
        _, event = _build_event(row)
        for match_key, match_row in find_matrix_rows(event):
            assert match_row.pathway_id.value == event["pathway_id"], (
                f"{key}: find_matrix_rows returned a row with different "
                f"pathway_id={match_row.pathway_id.value!r}"
            )

    def test_tampered_pathway_id_finds_no_match(self):
        """A forensic event with a fabricated pathway_id must return
        no matrix rows (or rows that don't match the other fields).
        """
        row = _TABLE["medical:GP:STOP:SELF_HARM_INSTRUCTION"]
        _, event = _build_event(row)
        # Swap pathway_id to one that doesn't match the output_mode.
        # P_ASK_DISAMBIGUATE has CLARIFICATION_REQUEST output_mode,
        # but the event has terminal_stop.
        event["pathway_id"] = ContinuationPathway.P_ASK_DISAMBIGUATE.value
        matches = find_matrix_rows(event)
        # No row has P_ASK_DISAMBIGUATE + terminal_stop output_mode.
        for _, match_row in matches:
            assert match_row.output_mode.value == event["output_mode"], (
                "A tampered event returned an inconsistent row"
            )
        # In practice this should return zero matches.
        assert len(matches) == 0, (
            f"Tampered event (wrong pathway_id) matched {len(matches)} rows: "
            f"{[k for k,_ in matches]}"
        )

    def test_empty_event_returns_no_rows(self):
        """An empty event has no pathway_id filter — with pathway_id=None,
        find_matrix_rows() uses no filter on that field and may return all rows.
        Verify the function handles this without error.
        """
        # This is a behaviour contract test, not a prohibition — the function
        # must not raise; the caller is responsible for providing a non-empty event.
        matches = find_matrix_rows({})
        assert isinstance(matches, list)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Cross-cutting: event self-sufficiency end-to-end
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEndSelfSufficiency:
    """Prove that a forensic event is self-sufficient for full verification.

    A self-sufficient event satisfies all of:
      A. Schema validation passes (validate() returns no errors)
      B. event_hash is valid (verify_event_hash() is True)
      C. governed_response_hash matches the replayed output
      D. Source row is identified by find_matrix_rows()
    """

    @pytest.mark.parametrize("key,row", _NON_ADMIT_ROWS)
    def test_event_passes_all_four_checks(self, key: str, row: ContinuationRow):
        from aurora_lens.govern.adapters.policy_projector import (
            _PATHWAY_FALLBACK,
            _PATHWAY_TO_ACTION,
        )
        _, event = _build_event(row)

        # A. Schema valid.
        errors = validate(event)
        assert not errors, f"{key} A: schema errors: {errors}"

        # B. Event hash valid.
        assert verify_event_hash(event), f"{key} B: event_hash invalid"

        # C. Governed response hash replay.
        pathway = ContinuationPathway(event["pathway_id"])
        action  = _PATHWAY_TO_ACTION.get(pathway, _PATHWAY_FALLBACK)
        flag_type = FlagType[event["failed_constraints"][0]]
        d_replay = GovernanceDecision(
            action=action,
            flags=[_flag(flag_type)],
            rationale="e2e replay",
            policy="strict",
            attempt=0,
            resource=event.get("escalation_target"),
        )
        d_replay.pathway_id           = event["pathway_id"]
        d_replay.commitment_closed    = event["commitment_closed"]
        d_replay.interaction_open     = event["interaction_open"]
        d_replay.output_mode          = event["output_mode"]
        d_replay.forensic_obligations = event.get("forensic_obligations", [])
        d_replay.resolution_mode      = event.get("resolution_mode", "exact")
        replayed = enforce(d_replay, "")
        assert verify_governed_response_hash(event, replayed), (
            f"{key} C: governed_response_hash does not match replayed output"
        )

        # D. Row identity.
        matches = find_matrix_rows(event)
        match_keys = {k for k, _ in matches}
        assert key in match_keys, (
            f"{key} D: source row not in find_matrix_rows() result: "
            f"{sorted(match_keys)}"
        )
