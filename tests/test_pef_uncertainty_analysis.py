"""Tests for deterministic PEF uncertainty analysis.

All tests work from committed PEF state — no adapter, no LLM.
"""

from __future__ import annotations

import pytest

from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.pef.span import Span
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.uncertainty_analysis import (
    CORE_UNCERTAINTY_KINDS,
    EXTERNAL_ESTABLISHMENT_FAILURE_KINDS,
    PHASE2_ESTABLISHMENT_FAILURE_IMPLEMENTATION,
    PHASE2_EXTERNAL_METADATA_OPTIONAL_FIELDS,
    INFERENTIAL_BRIDGE_MODE_DEFAULT,
    THRESHOLD_COMPARISON_MODE_DEFAULT,
    THRESHOLD_COMPARISON_MODES,
    THRESHOLD_CONSEQUENCE_GRADES,
    UNCERTAINTY_KINDS,
    VALID_OPEN_EPISTEMIC_KINDS,
    bridge_status_indicates_gap,
    merge_epistemic_uncertainties,
    pef_uncertainty_analysis,
    threshold_comparison_fails,
)
from aurora_lens.pef.unresolved_referents import (
    open_entries,
    register_unresolved_referents,
    resolve_unresolved_referents,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _pef() -> PEFState:
    return PEFState()


def _entity(pef: PEFState, name: str, *, resolved: bool = True) -> str:
    e = Entity.create(name, turn=1)
    e.resolved = resolved
    pef.entities[e.id] = e
    return e.id


def _rel(
    pef: PEFState,
    subject_id: str,
    relation: str,
    value: str,
    *,
    source_name: str | None = None,
    claim_status: str = "asserted",
    negated: bool = False,
    source_turn: int = 1,
    relation_metadata: dict | None = None,
) -> None:
    pef.relationships.append(
        Relationship(
            subject_id=subject_id,
            relation=relation,
            object_entity_id=None,
            object_literal=value,
            span=Span.PRESENT,
            source_turn=source_turn,
            evidence=f"{relation}={value}",
            negated=negated,
            source_name=source_name,
            claim_status=claim_status,
            relation_metadata=relation_metadata,
        )
    )


# ── CORE_UNCERTAINTY_KINDS ─────────────────────────────────────────────

class TestUncertaintyKinds:
    def test_five_kinds(self):
        assert len(CORE_UNCERTAINTY_KINDS) == 5

    def test_expected_kinds_present(self):
        assert "conflicting_models" in CORE_UNCERTAINTY_KINDS
        assert "missing_observation" in CORE_UNCERTAINTY_KINDS
        assert "blocked_access" in CORE_UNCERTAINTY_KINDS
        assert "unconfirmed_status" in CORE_UNCERTAINTY_KINDS
        assert "predictive_uncertainty" in CORE_UNCERTAINTY_KINDS

    def test_future_dependent_outcome_removed(self):
        assert "future_dependent_outcome" not in CORE_UNCERTAINTY_KINDS

    def test_authority_bearing_consequence_absent(self):
        assert "authority_bearing_consequence" not in CORE_UNCERTAINTY_KINDS

    def test_alias_matches_core(self):
        assert UNCERTAINTY_KINDS is CORE_UNCERTAINTY_KINDS


# ── pef_uncertainty_analysis: conflicting_models ───────────────────────

class TestConflictingModels:
    def test_same_predicate_different_values_yields_conflict(self):
        pef = _pef()
        sid = _entity(pef, "warm_front")
        _rel(pef, sid, "arrival_time", "48_hours", source_name="model_a")
        _rel(pef, sid, "arrival_time", "96_hours", source_name="model_b")
        records = pef_uncertainty_analysis(pef)
        kinds = {r["kind"] for r in records}
        assert "conflicting_models" in kinds

    def test_conflict_record_has_required_fields(self):
        pef = _pef()
        sid = _entity(pef, "warm_front")
        _rel(pef, sid, "arrival_time", "48h", source_name="model_a")
        _rel(pef, sid, "arrival_time", "96h", source_name="model_b")
        records = pef_uncertainty_analysis(pef)
        conflict = next(r for r in records if r["kind"] == "conflicting_models")
        assert "id" in conflict
        assert "description" in conflict
        assert "bears_on" in conflict
        assert conflict["status"] == "open"

    def test_source_names_appear_in_description(self):
        pef = _pef()
        sid = _entity(pef, "warm_front")
        _rel(pef, sid, "arrival_time", "48h", source_name="model_a")
        _rel(pef, sid, "arrival_time", "96h", source_name="model_b")
        records = pef_uncertainty_analysis(pef)
        conflict = next(r for r in records if r["kind"] == "conflicting_models")
        assert "model_a" in conflict["description"] or "model_b" in conflict["description"]

    def test_same_value_no_conflict(self):
        pef = _pef()
        sid = _entity(pef, "temperature")
        _rel(pef, sid, "reading", "85C", source_name="sensor_a")
        _rel(pef, sid, "reading", "85C", source_name="sensor_b")
        records = pef_uncertainty_analysis(pef)
        conflict_records = [r for r in records if r["kind"] == "conflicting_models"]
        assert len(conflict_records) == 0

    def test_negated_relationship_excluded_from_conflict(self):
        pef = _pef()
        sid = _entity(pef, "sensor")
        _rel(pef, sid, "reading", "85C")
        _rel(pef, sid, "reading", "62C", negated=True)
        records = pef_uncertainty_analysis(pef)
        conflict_records = [r for r in records if r["kind"] == "conflicting_models"]
        assert len(conflict_records) == 0

    def test_retracted_claim_excluded_from_conflict(self):
        pef = _pef()
        sid = _entity(pef, "contract_date")
        _rel(pef, sid, "value", "2024-01-01", claim_status="asserted")
        _rel(pef, sid, "value", "2024-03-15", claim_status="retracted")
        records = pef_uncertainty_analysis(pef)
        conflict_records = [r for r in records if r["kind"] == "conflicting_models"]
        assert len(conflict_records) == 0

    def test_different_predicates_not_a_conflict(self):
        pef = _pef()
        sid = _entity(pef, "entity")
        _rel(pef, sid, "color", "red")
        _rel(pef, sid, "size", "large")
        records = pef_uncertainty_analysis(pef)
        conflict_records = [r for r in records if r["kind"] == "conflicting_models"]
        assert len(conflict_records) == 0

    def test_multiple_conflicts_detected(self):
        pef = _pef()
        eid1 = _entity(pef, "temperature")
        _rel(pef, eid1, "reading", "85C", source_name="sensor_a")
        _rel(pef, eid1, "reading", "62C", source_name="sensor_b")
        eid2 = _entity(pef, "revenue")
        _rel(pef, eid2, "forecast", "12M", source_name="analyst_1")
        _rel(pef, eid2, "forecast", "28M", source_name="analyst_2")
        records = pef_uncertainty_analysis(pef)
        conflict_records = [r for r in records if r["kind"] == "conflicting_models"]
        assert len(conflict_records) == 2

    def test_bears_on_contains_subject_and_predicate(self):
        pef = _pef()
        sid = _entity(pef, "warm_front")
        _rel(pef, sid, "arrival_time", "48h")
        _rel(pef, sid, "arrival_time", "96h")
        records = pef_uncertainty_analysis(pef)
        conflict = next(r for r in records if r["kind"] == "conflicting_models")
        assert any("arrival_time" in b for b in conflict["bears_on"])


# ── pef_uncertainty_analysis: unconfirmed_status ──────────────────────

class TestUnconfirmedStatus:
    def test_unresolved_entity_yields_record(self):
        pef = _pef()
        _entity(pef, "suspect_compound", resolved=False)
        records = pef_uncertainty_analysis(pef)
        kinds = {r["kind"] for r in records}
        assert "unconfirmed_status" in kinds

    def test_resolved_entity_not_flagged(self):
        pef = _pef()
        _entity(pef, "known_entity", resolved=True)
        records = pef_uncertainty_analysis(pef)
        kinds = {r["kind"] for r in records}
        assert "unconfirmed_status" not in kinds

    def test_entity_name_in_description(self):
        pef = _pef()
        _entity(pef, "dam_structural_integrity", resolved=False)
        records = pef_uncertainty_analysis(pef)
        unconf = next(r for r in records if r["kind"] == "unconfirmed_status")
        assert "dam_structural_integrity" in unconf["description"]

    def test_unconfirmed_record_has_open_status(self):
        pef = _pef()
        _entity(pef, "x", resolved=False)
        records = pef_uncertainty_analysis(pef)
        unconf = next(r for r in records if r["kind"] == "unconfirmed_status")
        assert unconf["status"] == "open"


# ── pef_uncertainty_analysis: blocked_access ──────────────────────────

class TestBlockedAccess:
    def test_retrieval_unresolved_yields_blocked_record(self):
        pef = _pef()
        pef.retrieval_unresolved = [
            {"subject": "dam", "relation": "status", "description": "Dam status conflict in retrieval."}
        ]
        records = pef_uncertainty_analysis(pef)
        kinds = {r["kind"] for r in records}
        assert "blocked_access" in kinds

    def test_blocked_record_description_from_item(self):
        pef = _pef()
        pef.retrieval_unresolved = [
            {"subject": "site", "relation": "access", "description": "Ground teams cannot reach the site."}
        ]
        records = pef_uncertainty_analysis(pef)
        blocked = next(r for r in records if r["kind"] == "blocked_access")
        assert "Ground teams" in blocked["description"]

    def test_empty_retrieval_unresolved_no_blocked_record(self):
        pef = _pef()
        pef.retrieval_unresolved = []
        records = pef_uncertainty_analysis(pef)
        assert not any(r["kind"] == "blocked_access" for r in records)

    def test_blocked_record_uses_item_id_if_present(self):
        pef = _pef()
        pef.retrieval_unresolved = [{"id": "ground_access", "subject": "site", "relation": "access"}]
        records = pef_uncertainty_analysis(pef)
        blocked = next(r for r in records if r["kind"] == "blocked_access")
        assert blocked["id"] == "ground_access"


# ── pef_uncertainty_analysis: pending claims ──────────────────────────

class TestPendingClaims:
    def test_pending_claim_yields_missing_observation(self):
        pef = _pef()
        sid = _entity(pef, "scan_result")
        _rel(pef, sid, "reading", "inconclusive", claim_status="pending")
        records = pef_uncertainty_analysis(pef)
        kinds = {r["kind"] for r in records}
        assert "missing_observation" in kinds

    def test_asserted_claim_not_flagged_as_pending(self):
        pef = _pef()
        sid = _entity(pef, "x")
        _rel(pef, sid, "value", "known", claim_status="asserted")
        records = pef_uncertainty_analysis(pef)
        assert not any(r["kind"] == "missing_observation" for r in records)


# ── Empty PEF ─────────────────────────────────────────────────────────

class TestEmptyPEF:
    def test_empty_pef_returns_empty_list(self):
        pef = _pef()
        assert pef_uncertainty_analysis(pef) == []


# ── merge_epistemic_uncertainties ─────────────────────────────────────

class TestMergeEpistemicUncertainties:
    def test_adds_new_records(self):
        pef = _pef()
        merge_epistemic_uncertainties(pef, [{"id": "x", "kind": "conflicting_models", "description": "d"}])
        assert len(pef.open_epistemic_uncertainties) == 1

    def test_no_duplicate_by_id(self):
        pef = _pef()
        rec = {"id": "x", "kind": "conflicting_models", "description": "d"}
        merge_epistemic_uncertainties(pef, [rec])
        merge_epistemic_uncertainties(pef, [rec])
        assert len(pef.open_epistemic_uncertainties) == 1

    def test_preserves_existing_status(self):
        pef = _pef()
        pef.open_epistemic_uncertainties = [{"id": "x", "kind": "conflicting_models", "status": "resolved"}]
        merge_epistemic_uncertainties(pef, [{"id": "x", "kind": "conflicting_models", "description": "d", "status": "open"}])
        assert pef.open_epistemic_uncertainties[0]["status"] == "resolved"

    def test_empty_list_leaves_pef_unchanged(self):
        pef = _pef()
        pef.open_epistemic_uncertainties = [{"id": "x"}]
        merge_epistemic_uncertainties(pef, [])
        assert len(pef.open_epistemic_uncertainties) == 1

    def test_adds_to_existing_entries(self):
        pef = _pef()
        pef.open_epistemic_uncertainties = [{"id": "existing"}]
        merge_epistemic_uncertainties(pef, [{"id": "new", "kind": "blocked_access", "description": "d"}])
        ids = {u["id"] for u in pef.open_epistemic_uncertainties}
        assert "existing" in ids and "new" in ids


# ── EXTERNAL_ESTABLISHMENT_FAILURE_KINDS and VALID_OPEN_EPISTEMIC_KINDS ──

_PHASE2_KINDS = {
    "identity_or_referent_unresolved",
    "stale_evidence",
    "inferential_gap",
    "threshold_not_met",
    "source_untrusted",
    "scope_mismatch",
}

class TestExternalEstablishmentFailureKinds:
    def test_six_phase2_kinds(self):
        assert len(EXTERNAL_ESTABLISHMENT_FAILURE_KINDS) == 6

    def test_all_phase2_kinds_exported(self):
        assert _PHASE2_KINDS == EXTERNAL_ESTABLISHMENT_FAILURE_KINDS

    def test_no_phase2_kind_in_core(self):
        assert EXTERNAL_ESTABLISHMENT_FAILURE_KINDS.isdisjoint(CORE_UNCERTAINTY_KINDS)

    def test_no_core_kind_in_external(self):
        assert CORE_UNCERTAINTY_KINDS.isdisjoint(EXTERNAL_ESTABLISHMENT_FAILURE_KINDS)


class TestValidOpenEpistemicKinds:
    def test_is_union_of_core_and_external(self):
        assert VALID_OPEN_EPISTEMIC_KINDS == CORE_UNCERTAINTY_KINDS | EXTERNAL_ESTABLISHMENT_FAILURE_KINDS

    def test_all_core_kinds_included(self):
        assert CORE_UNCERTAINTY_KINDS <= VALID_OPEN_EPISTEMIC_KINDS

    def test_all_phase2_kinds_included(self):
        assert EXTERNAL_ESTABLISHMENT_FAILURE_KINDS <= VALID_OPEN_EPISTEMIC_KINDS

    def test_eleven_kinds_total(self):
        assert len(VALID_OPEN_EPISTEMIC_KINDS) == 11


class TestPhase2ImplementationModes:
    def test_identity_or_referent_unresolved_is_deterministic_generated(self):
        assert (
            PHASE2_ESTABLISHMENT_FAILURE_IMPLEMENTATION["identity_or_referent_unresolved"]
            == "deterministic_generated"
        )

    def test_stale_evidence_is_deterministic_generated(self):
        assert (
            PHASE2_ESTABLISHMENT_FAILURE_IMPLEMENTATION["stale_evidence"]
            == "deterministic_generated"
        )

    def test_scope_mismatch_is_deterministic_generated(self):
        assert (
            PHASE2_ESTABLISHMENT_FAILURE_IMPLEMENTATION["scope_mismatch"]
            == "deterministic_generated"
        )

    def test_threshold_not_met_is_deterministic_generated(self):
        assert (
            PHASE2_ESTABLISHMENT_FAILURE_IMPLEMENTATION["threshold_not_met"]
            == "deterministic_generated"
        )

    def test_inferential_gap_is_deterministic_generated(self):
        assert (
            PHASE2_ESTABLISHMENT_FAILURE_IMPLEMENTATION["inferential_gap"]
            == "deterministic_generated"
        )

    def test_remaining_phase2_kinds_are_external_declared_only(self):
        for kind in (
            "source_untrusted",
        ):
            assert PHASE2_ESTABLISHMENT_FAILURE_IMPLEMENTATION[kind] == "external_declared_only"

    def test_metadata_contract_exists_for_remaining_external_declared_kinds(self):
        for kind in (
            "stale_evidence",
            "source_untrusted",
            "scope_mismatch",
            "threshold_not_met",
            "inferential_gap",
        ):
            assert kind in PHASE2_EXTERNAL_METADATA_OPTIONAL_FIELDS
            assert PHASE2_EXTERNAL_METADATA_OPTIONAL_FIELDS[kind]


class TestPhase2KindsNotGeneratedByAnalyser:
    """pef_uncertainty_analysis() must only emit allowed Phase 2 generators."""

    _ALLOWED_AUTO_GENERATED_PHASE2 = {
        "identity_or_referent_unresolved",
        "stale_evidence",
        "scope_mismatch",
        "threshold_not_met",
        "inferential_gap",
        "source_untrusted",
    }
    _DISALLOWED_AUTO_GENERATED_PHASE2 = _PHASE2_KINDS - _ALLOWED_AUTO_GENERATED_PHASE2

    def test_analyser_emits_no_disallowed_phase2_kinds_on_empty_pef(self):
        records = pef_uncertainty_analysis(_pef())
        kinds = {r["kind"] for r in records}
        assert kinds.isdisjoint(self._DISALLOWED_AUTO_GENERATED_PHASE2)

    def test_analyser_emits_no_disallowed_phase2_kinds_on_conflict(self):
        pef = _pef()
        eid = _entity(pef, "subject")
        _rel(pef, eid, "value", "a", source_name="src_a")
        _rel(pef, eid, "value", "b", source_name="src_b")
        kinds = {r["kind"] for r in pef_uncertainty_analysis(pef)}
        assert kinds.isdisjoint(self._DISALLOWED_AUTO_GENERATED_PHASE2)

    def test_analyser_emits_no_disallowed_phase2_kinds_on_unresolved_entity(self):
        pef = _pef()
        _entity(pef, "unknown_entity", resolved=False)
        kinds = {r["kind"] for r in pef_uncertainty_analysis(pef)}
        assert kinds.isdisjoint(self._DISALLOWED_AUTO_GENERATED_PHASE2)


class TestIdentityReferentUnresolvedGeneration:
    _ID_PREFIX = "generated_identity_or_referent_unresolved:"

    def test_open_registry_entry_generates_identity_or_referent_unresolved(self):
        pef = _pef()
        register_unresolved_referents(
            pef,
            turn=1,
            utterance="Who does their refer to?",
            tokens=["their"],
            candidate_entities=["Operator", "Contractor"],
            blocked_proposition="their certification had expired",
        )
        records = pef_uncertainty_analysis(pef)
        generated = [r for r in records if r["kind"] == "identity_or_referent_unresolved"]
        assert len(generated) == 1
        rec = generated[0]
        assert str(rec["id"]).startswith(self._ID_PREFIX)
        assert rec["status"] == "open"
        assert rec["description"]
        assert isinstance(rec["bears_on"], list) and rec["bears_on"]

    def test_identity_or_referent_unresolved_id_is_stable_across_repeated_analysis_runs(self):
        pef = _pef()
        register_unresolved_referents(
            pef,
            turn=1,
            utterance="Who does their refer to?",
            tokens=["their"],
            candidate_entities=["Operator", "Contractor"],
            blocked_proposition="their certification had expired",
        )
        recs_a = pef_uncertainty_analysis(pef)
        recs_b = pef_uncertainty_analysis(pef)
        a = next(r for r in recs_a if r["kind"] == "identity_or_referent_unresolved")
        b = next(r for r in recs_b if r["kind"] == "identity_or_referent_unresolved")
        assert a["id"] == b["id"]

    def test_merge_repeated_runs_do_not_duplicate_identity_or_referent_unresolved(self):
        pef = _pef()
        register_unresolved_referents(
            pef,
            turn=1,
            utterance="Who does their refer to?",
            tokens=["their"],
            candidate_entities=["Operator", "Contractor"],
            blocked_proposition="their certification had expired",
        )
        merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef))
        merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef))
        generated = [
            u
            for u in pef.open_epistemic_uncertainties
            if u.get("kind") == "identity_or_referent_unresolved"
        ]
        assert len(generated) == 1

    def test_resolving_registry_entry_removes_or_closes_generated_identity_or_referent_unresolved(self):
        pef = _pef()
        register_unresolved_referents(
            pef,
            turn=1,
            utterance="Who does their refer to?",
            tokens=["their"],
            candidate_entities=["Operator", "Contractor"],
            blocked_proposition="their certification had expired",
        )
        assert open_entries(pef)
        merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef))
        generated_before = [
            u
            for u in pef.open_epistemic_uncertainties
            if u.get("kind") == "identity_or_referent_unresolved"
        ]
        assert generated_before

        resolve_unresolved_referents(
            pef,
            tokens=["their"],
            resolved_entity="Contractor",
            turn=2,
        )
        assert not open_entries(pef)

        merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef))
        generated_after = [
            u
            for u in pef.open_epistemic_uncertainties
            if u.get("kind") == "identity_or_referent_unresolved"
        ]
        assert all(str(u.get("status", "open")) != "open" for u in generated_after)

    def test_no_other_phase2_kinds_auto_generated(self):
        pef = _pef()
        register_unresolved_referents(
            pef,
            turn=1,
            utterance="Who does their refer to?",
            tokens=["their"],
            candidate_entities=["Operator", "Contractor"],
            blocked_proposition="their certification had expired",
        )
        kinds = {r["kind"] for r in pef_uncertainty_analysis(pef)}
        disallowed = _PHASE2_KINDS - {"identity_or_referent_unresolved"}
        assert kinds.isdisjoint(disallowed)


class TestStaleEvidenceGeneration:
    def test_stale_generator_returns_none_when_decay_mode_missing(self):
        pef = _pef()
        sid = _entity(pef, "blood_panel")
        _rel(
            pef,
            sid,
            "IS",
            "potassium=5.8",
            relation_metadata={
                "temporal": {
                    "timestamp_source": "observed_at",
                    "observed_at": "2026-06-01T00:00:00Z",
                    "freshness_window": "PT6H",
                    "assessed_at": "2026-06-01T12:30:00Z",
                    "policy_ref": "med.labs.v1",
                }
            },
        )
        assert not any(r["kind"] == "stale_evidence" for r in pef_uncertainty_analysis(pef))

    def test_stale_generator_returns_none_when_decay_mode_invalid(self):
        pef = _pef()
        sid = _entity(pef, "blood_panel")
        _rel(
            pef,
            sid,
            "IS",
            "potassium=5.8",
            relation_metadata={
                "temporal": {
                    "decay_mode": "auto",
                    "timestamp_source": "observed_at",
                    "observed_at": "2026-06-01T00:00:00Z",
                    "freshness_window": "PT6H",
                    "assessed_at": "2026-06-01T12:30:00Z",
                    "policy_ref": "med.labs.v1",
                }
            },
        )
        assert not any(r["kind"] == "stale_evidence" for r in pef_uncertainty_analysis(pef))

    def test_stale_clock_missing_required_fields_returns_none(self):
        pef = _pef()
        sid = _entity(pef, "blood_panel")
        _rel(
            pef,
            sid,
            "IS",
            "potassium=5.8",
            relation_metadata={
                "temporal": {
                    "decay_mode": "clock",
                    "timestamp_source": "observed_at",
                    "observed_at": "2026-06-01T00:00:00Z",
                    # missing freshness_window/valid_until
                    "assessed_at": "2026-06-01T12:30:00Z",
                    "policy_ref": "med.labs.v1",
                }
            },
        )
        assert not any(r["kind"] == "stale_evidence" for r in pef_uncertainty_analysis(pef))

    def test_stale_event_missing_required_fields_returns_none(self):
        pef = _pef()
        sid = _entity(pef, "statute")
        _rel(
            pef,
            sid,
            "IS",
            "active",
            relation_metadata={
                "temporal": {
                    "decay_mode": "event",
                    "timestamp_source": "asserted_at",
                    "policy_ref": "law.currency.v1",
                    # missing evidence_ref/claim_key
                }
            },
        )
        assert not any(r["kind"] == "stale_evidence" for r in pef_uncertainty_analysis(pef))

    def test_stale_clock_generates_when_assessed_after_valid_until(self):
        pef = _pef()
        sid = _entity(pef, "weather_reading")
        _rel(
            pef,
            sid,
            "IS",
            "storm_risk_high",
            relation_metadata={
                "temporal": {
                    "decay_mode": "clock",
                    "timestamp_source": "observed_at",
                    "observed_at": "2026-06-01T00:00:00Z",
                    "valid_until": "2026-06-01T03:00:00Z",
                    "assessed_at": "2026-06-01T03:00:01Z",
                    "policy_ref": "wx.forecast.v1",
                }
            },
        )
        stale = [r for r in pef_uncertainty_analysis(pef) if r["kind"] == "stale_evidence"]
        assert len(stale) == 1

    def test_stale_clock_generates_when_age_exceeds_freshness_window(self):
        pef = _pef()
        sid = _entity(pef, "blood_panel")
        _rel(
            pef,
            sid,
            "IS",
            "potassium=5.8",
            relation_metadata={
                "temporal": {
                    "decay_mode": "clock",
                    "timestamp_source": "observed_at",
                    "observed_at": "2026-06-01T00:00:00Z",
                    "freshness_window": "PT6H",
                    "assessed_at": "2026-06-01T07:00:00Z",
                    "policy_ref": "med.labs.v1",
                }
            },
        )
        stale = [r for r in pef_uncertainty_analysis(pef) if r["kind"] == "stale_evidence"]
        assert len(stale) == 1

    def test_stale_event_generates_when_matching_superseding_event_committed_after_anchor(self):
        pef = _pef()
        sid = _entity(pef, "constitutional_provision")
        _rel(
            pef,
            sid,
            "IS",
            "in_force",
            source_turn=1,
            relation_metadata={
                "temporal": {
                    "decay_mode": "event",
                    "timestamp_source": "asserted_at",
                    "asserted_at": "2010-01-01T00:00:00Z",
                    "claim_key": "constitution:s7",
                    "supersession_rule": "amendment",
                    "policy_ref": "law.currency.v1",
                }
            },
        )
        _rel(
            pef,
            sid,
            "IS",
            "amended",
            source_turn=2,
            relation_metadata={
                "temporal": {
                    "committed_at": "2020-05-01T00:00:00Z",
                    "supersedes_ref": "constitution:s7",
                }
            },
        )
        stale = [r for r in pef_uncertainty_analysis(pef) if r["kind"] == "stale_evidence"]
        assert len(stale) == 1

    def test_stale_event_no_matching_superseding_event_returns_none(self):
        pef = _pef()
        sid = _entity(pef, "constitutional_provision")
        _rel(
            pef,
            sid,
            "IS",
            "in_force",
            source_turn=1,
            relation_metadata={
                "temporal": {
                    "decay_mode": "event",
                    "timestamp_source": "asserted_at",
                    "asserted_at": "2010-01-01T00:00:00Z",
                    "claim_key": "constitution:s7",
                    "supersession_rule": "amendment",
                    "policy_ref": "law.currency.v1",
                }
            },
        )
        _rel(
            pef,
            sid,
            "IS",
            "amended",
            source_turn=2,
            relation_metadata={
                "temporal": {
                    "committed_at": "2020-05-01T00:00:00Z",
                    "supersedes_ref": "constitution:s9",
                }
            },
        )
        assert not any(r["kind"] == "stale_evidence" for r in pef_uncertainty_analysis(pef))

    def test_stale_generator_has_no_inference_fallback_from_grade_domain_source(self):
        pef = _pef()
        sid = _entity(pef, "portfolio_price")
        _rel(
            pef,
            sid,
            "IS",
            "high",
            relation_metadata={
                "temporal": {
                    "consequence_grade": "critical",
                    "source_class": "market_feed",
                    "domain": "finance",
                    "policy_ref": "fin.advice.v2",
                    "freshness_window": "PT1H",
                    "assessed_at": "2026-06-01T10:00:00Z",
                }
            },
        )
        assert not any(r["kind"] == "stale_evidence" for r in pef_uncertainty_analysis(pef))

    def test_stale_generation_id_is_stable_and_idempotent(self):
        pef = _pef()
        sid = _entity(pef, "weather_reading")
        _rel(
            pef,
            sid,
            "IS",
            "storm_risk_high",
            relation_metadata={
                "temporal": {
                    "decay_mode": "clock",
                    "timestamp_source": "observed_at",
                    "observed_at": "2026-06-01T00:00:00Z",
                    "valid_until": "2026-06-01T03:00:00Z",
                    "assessed_at": "2026-06-01T03:00:01Z",
                    "policy_ref": "wx.forecast.v1",
                }
            },
        )
        r1 = [r for r in pef_uncertainty_analysis(pef) if r["kind"] == "stale_evidence"]
        r2 = [r for r in pef_uncertainty_analysis(pef) if r["kind"] == "stale_evidence"]
        assert len(r1) == 1 and len(r2) == 1
        assert r1[0]["id"] == r2[0]["id"]
        merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef))
        merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef))
        generated = [
            u for u in pef.open_epistemic_uncertainties if u.get("kind") == "stale_evidence"
        ]
        assert len(generated) == 1


class TestScopeMismatchBoundaryLocking:
    _SUPPLIER_SCOPE_MISMATCH = {
        "required_scope": {
            "action_type": "approve_supplier",
            "domain": "medical_procurement",
            "jurisdiction": "Singapore",
            "consequence_grade": "high",
            "subject": "supplier_123",
            "timeframe": "current",
        },
        "supported_scope": {
            "action_type": "approve_supplier",
            "domain": "office_supplies",
            "jurisdiction": "Australia",
            "consequence_grade": "low",
            "subject": "supplier_123",
            "timeframe": "2024",
        },
        "scope_dimensions": [
            "subject",
            "action_type",
            "domain",
            "jurisdiction",
            "timeframe",
            "consequence_grade",
        ],
        "comparison_mode": "exact",
        "policy_ref": "scope.policy.v1",
    }

    def test_scope_mismatch_not_generated_with_missing_scope_metadata(self):
        pef = _pef()
        sid = _entity(pef, "guideline_note")
        _rel(
            pef,
            sid,
            "IS",
            "applicable",
            relation_metadata={},
        )
        assert not any(r["kind"] == "scope_mismatch" for r in pef_uncertainty_analysis(pef))

    def test_scope_mismatch_not_generated_with_partial_scope_metadata(self):
        pef = _pef()
        sid = _entity(pef, "guideline_note")
        _rel(
            pef,
            sid,
            "IS",
            "applicable",
            relation_metadata={
                "scope": {
                    "supported_scope": {
                        "domain": "office_supplies",
                        "jurisdiction": "Australia",
                    },
                    "scope_dimensions": ["domain", "jurisdiction"],
                    # required_scope intentionally missing
                }
            },
        )
        assert not any(r["kind"] == "scope_mismatch" for r in pef_uncertainty_analysis(pef))

    def test_scope_mismatch_not_generated_with_informal_scope_metadata(self):
        pef = _pef()
        sid = _entity(pef, "guideline_note")
        _rel(
            pef,
            sid,
            "IS",
            "applicable",
            relation_metadata={
                "scope": {
                    "evidence_scope": "looks-like-us-guidance",
                    "required_scope": "probably-au-state",
                    "notes": "informal free text",
                }
            },
        )
        assert not any(r["kind"] == "scope_mismatch" for r in pef_uncertainty_analysis(pef))

    def test_scope_mismatch_not_generated_with_legacy_list_scope_shapes(self):
        pef = _pef()
        sid = _entity(pef, "guideline_note")
        _rel(
            pef,
            sid,
            "IS",
            "applicable",
            relation_metadata={
                "scope": {
                    "evidence_scope": ["us-federal", "2024"],
                    "required_scope": ["au-state", "2026"],
                    "policy_ref": "scope.policy.v1",
                }
            },
        )
        assert not any(r["kind"] == "scope_mismatch" for r in pef_uncertainty_analysis(pef))

    def test_scope_mismatch_not_generated_when_comparison_mode_missing(self):
        pef = _pef()
        sid = _entity(pef, "supplier_123")
        scope = dict(self._SUPPLIER_SCOPE_MISMATCH)
        scope.pop("comparison_mode")
        _rel(pef, sid, "MAY", "approve", relation_metadata={"scope": scope})
        assert not any(r["kind"] == "scope_mismatch" for r in pef_uncertainty_analysis(pef))

    def test_scope_mismatch_not_generated_when_scopes_match(self):
        pef = _pef()
        sid = _entity(pef, "supplier_123")
        matching = {
            "action_type": "approve_supplier",
            "domain": "medical_procurement",
            "jurisdiction": "Singapore",
            "consequence_grade": "high",
            "subject": "supplier_123",
            "timeframe": "current",
        }
        _rel(
            pef,
            sid,
            "MAY",
            "approve",
            relation_metadata={
                "scope": {
                    "required_scope": matching,
                    "supported_scope": dict(matching),
                    "scope_dimensions": list(matching.keys()),
                    "comparison_mode": "exact",
                }
            },
        )
        assert not any(r["kind"] == "scope_mismatch" for r in pef_uncertainty_analysis(pef))

    def test_scope_mismatch_no_fallback_from_domain_source_class_or_wording(self):
        pef = _pef()
        sid = _entity(pef, "oncology_guideline")
        _rel(
            pef,
            sid,
            "IS",
            "applicable_to_cardiology_decision",
            source_name="cardiology_journal",
            relation_metadata={
                "scope": {
                    "domain": "medical",
                    "source_class": "journal",
                    "notes": "wrong jurisdiction for this claim",
                    "wording_hint": "does not apply here",
                }
            },
        )
        assert not any(r["kind"] == "scope_mismatch" for r in pef_uncertainty_analysis(pef))


class TestScopeMismatchGeneration:
    _ID_PREFIX = "generated_scope_mismatch:"

    def test_full_contract_mismatch_generates_scope_mismatch_with_failed_dimensions(self):
        pef = _pef()
        sid = _entity(pef, "supplier_123")
        _rel(
            pef,
            sid,
            "MAY",
            "approve",
            relation_metadata={"scope": TestScopeMismatchBoundaryLocking._SUPPLIER_SCOPE_MISMATCH},
        )
        records = pef_uncertainty_analysis(pef)
        generated = [r for r in records if r["kind"] == "scope_mismatch"]
        assert len(generated) == 1
        rec = generated[0]
        assert str(rec["id"]).startswith(self._ID_PREFIX)
        meta = rec.get("meta") or {}
        assert meta.get("comparison_mode") == "exact"
        assert set(meta.get("failed_dimensions") or []) == {
            "domain",
            "jurisdiction",
            "consequence_grade",
            "timeframe",
        }
        assert meta.get("required_scope", {}).get("jurisdiction") == "Singapore"
        assert meta.get("supported_scope", {}).get("jurisdiction") == "Australia"

    def test_evidence_scope_alias_supported_as_supported_scope(self):
        pef = _pef()
        sid = _entity(pef, "supplier_123")
        scope = dict(TestScopeMismatchBoundaryLocking._SUPPLIER_SCOPE_MISMATCH)
        supported = scope.pop("supported_scope")
        scope["evidence_scope"] = supported
        _rel(pef, sid, "MAY", "approve", relation_metadata={"scope": scope})
        generated = [r for r in pef_uncertainty_analysis(pef) if r["kind"] == "scope_mismatch"]
        assert len(generated) == 1
        meta = generated[0].get("meta") or {}
        assert meta.get("supported_scope", {}).get("domain") == "office_supplies"

    def test_merge_reconciles_stale_generated_scope_mismatch_records(self):
        pef = _pef()
        sid = _entity(pef, "supplier_123")
        scope = dict(TestScopeMismatchBoundaryLocking._SUPPLIER_SCOPE_MISMATCH)
        matching = dict(scope["required_scope"])
        _rel(pef, sid, "MAY", "approve", relation_metadata={"scope": scope})
        merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef))
        assert any(
            str(u.get("id", "")).startswith(self._ID_PREFIX)
            for u in pef.open_epistemic_uncertainties
        )
        pef.relationships[0].relation_metadata = {
            "scope": {
                **scope,
                "supported_scope": matching,
            }
        }
        merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef))
        assert not any(
            u.get("kind") == "scope_mismatch" for u in pef.open_epistemic_uncertainties
        )


class TestThresholdNotMetBoundaryLocking:
    _FULL_THRESHOLD_CONTRACT = {
        "required_strength": 0.8,
        "observed_strength": 0.52,
        "consequence_grade": "high",
        "threshold_ref": "policy.threshold.v1",
        "comparison_mode": "gte",
        "policy_ref": "decision.policy.v3",
    }

    def test_threshold_not_met_not_generated_with_missing_threshold_contract(self):
        pef = _pef()
        sid = _entity(pef, "risk_assessment")
        _rel(
            pef,
            sid,
            "IS",
            "evidence_strength_recorded",
            relation_metadata={
                "threshold": {
                    "observed_strength": 0.52,
                    # missing required_strength/threshold_ref
                }
            },
        )
        assert not any(r["kind"] == "threshold_not_met" for r in pef_uncertainty_analysis(pef))

    def test_threshold_not_met_not_generated_with_missing_measurement(self):
        pef = _pef()
        sid = _entity(pef, "risk_assessment")
        _rel(
            pef,
            sid,
            "IS",
            "policy_threshold_declared",
            relation_metadata={
                "threshold": {
                    "required_strength": 0.8,
                    "threshold_ref": "policy.threshold.v1",
                    # missing observed_strength
                }
            },
        )
        assert not any(r["kind"] == "threshold_not_met" for r in pef_uncertainty_analysis(pef))

    def test_threshold_not_met_not_generated_from_low_grade_alone(self):
        pef = _pef()
        sid = _entity(pef, "risk_assessment")
        _rel(
            pef,
            sid,
            "IS",
            "critical_decision_context",
            relation_metadata={
                "threshold": {
                    "consequence_grade": "high",
                    # no explicit comparator inputs
                }
            },
        )
        assert not any(r["kind"] == "threshold_not_met" for r in pef_uncertainty_analysis(pef))

    def test_threshold_not_met_not_generated_from_weak_evidence_label(self):
        pef = _pef()
        sid = _entity(pef, "risk_assessment")
        _rel(
            pef,
            sid,
            "IS",
            "weak_evidence",
            relation_metadata={
                "threshold": {
                    "evidence_label": "weak",
                    "notes": "appears insufficient",
                }
            },
        )
        assert not any(r["kind"] == "threshold_not_met" for r in pef_uncertainty_analysis(pef))

    def test_threshold_not_met_not_generated_with_missing_comparison_mode(self):
        pef = _pef()
        sid = _entity(pef, "risk_assessment")
        contract = dict(self._FULL_THRESHOLD_CONTRACT)
        contract.pop("comparison_mode")
        _rel(
            pef,
            sid,
            "IS",
            "assessment_ready",
            relation_metadata={"threshold": contract},
        )
        assert not any(r["kind"] == "threshold_not_met" for r in pef_uncertainty_analysis(pef))

    def test_threshold_not_met_not_generated_with_missing_consequence_grade(self):
        pef = _pef()
        sid = _entity(pef, "risk_assessment")
        contract = dict(self._FULL_THRESHOLD_CONTRACT)
        contract.pop("consequence_grade")
        _rel(
            pef,
            sid,
            "IS",
            "assessment_ready",
            relation_metadata={"threshold": contract},
        )
        assert not any(r["kind"] == "threshold_not_met" for r in pef_uncertainty_analysis(pef))

    def test_threshold_not_met_not_generated_with_invalid_comparison_mode(self):
        pef = _pef()
        sid = _entity(pef, "risk_assessment")
        contract = dict(self._FULL_THRESHOLD_CONTRACT)
        contract["comparison_mode"] = "strict_inequality"
        _rel(
            pef,
            sid,
            "IS",
            "assessment_ready",
            relation_metadata={"threshold": contract},
        )
        assert not any(r["kind"] == "threshold_not_met" for r in pef_uncertainty_analysis(pef))

    def test_threshold_not_met_not_generated_from_confidence_or_source_class_or_domain(self):
        pef = _pef()
        sid = _entity(pef, "risk_assessment")
        _rel(
            pef,
            sid,
            "IS",
            "limited_evidence_for_high_stakes_decision",
            source_name="vendor_report",
            relation_metadata={
                "threshold": {
                    "confidence_label": "low",
                    "source_class": "vendor",
                    "domain": "finance",
                    "notes": "appears insufficient for this consequence",
                }
            },
        )
        assert not any(r["kind"] == "threshold_not_met" for r in pef_uncertainty_analysis(pef))

    def test_threshold_not_met_not_generated_when_observed_meets_required(self):
        pef = _pef()
        sid = _entity(pef, "risk_assessment")
        contract = dict(self._FULL_THRESHOLD_CONTRACT)
        contract["observed_strength"] = 0.8
        _rel(
            pef,
            sid,
            "IS",
            "assessment_ready",
            relation_metadata={"threshold": contract},
        )
        assert not any(r["kind"] == "threshold_not_met" for r in pef_uncertainty_analysis(pef))


class TestThresholdNotMetGeneration:
    _ID_PREFIX = "generated_threshold_not_met:"

    def test_full_contract_failure_generates_threshold_not_met_with_meta(self):
        pef = _pef()
        sid = _entity(pef, "risk_assessment")
        _rel(
            pef,
            sid,
            "IS",
            "assessment_ready",
            relation_metadata={"threshold": TestThresholdNotMetBoundaryLocking._FULL_THRESHOLD_CONTRACT},
        )
        records = pef_uncertainty_analysis(pef)
        generated = [r for r in records if r["kind"] == "threshold_not_met"]
        assert len(generated) == 1
        rec = generated[0]
        assert str(rec["id"]).startswith(self._ID_PREFIX)
        meta = rec.get("meta") or {}
        assert meta.get("required_strength") == 0.8
        assert meta.get("observed_strength") == 0.52
        assert meta.get("consequence_grade") == "high"
        assert meta.get("threshold_ref") == "policy.threshold.v1"
        assert meta.get("comparison_mode") == "gte"
        assert meta.get("policy_ref") == "decision.policy.v3"

    def test_merge_reconciles_stale_generated_threshold_not_met_records(self):
        pef = _pef()
        sid = _entity(pef, "risk_assessment")
        contract = dict(TestThresholdNotMetBoundaryLocking._FULL_THRESHOLD_CONTRACT)
        _rel(pef, sid, "IS", "assessment_ready", relation_metadata={"threshold": contract})
        merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef))
        assert any(
            str(u.get("id", "")).startswith(self._ID_PREFIX)
            for u in pef.open_epistemic_uncertainties
        )
        contract["observed_strength"] = 0.8
        pef.relationships[0].relation_metadata = {"threshold": contract}
        merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef))
        assert not any(
            u.get("kind") == "threshold_not_met" for u in pef.open_epistemic_uncertainties
        )


class TestThresholdComparisonSemantics:
    def test_default_comparison_mode_is_gte(self):
        assert THRESHOLD_COMPARISON_MODE_DEFAULT == "gte"
        assert "gte" in THRESHOLD_COMPARISON_MODES

    def test_consequence_grade_v1_enum(self):
        assert THRESHOLD_CONSEQUENCE_GRADES == frozenset({
            "minimal", "low", "moderate", "high", "critical",
        })

    def test_gte_fails_when_observed_below_required(self):
        assert threshold_comparison_fails(0.52, 0.8, "gte") is True
        assert threshold_comparison_fails(0.8, 0.8, "gte") is False
        assert threshold_comparison_fails(0.9, 0.8, "gte") is False

    def test_gt_requires_strict_exceed(self):
        assert threshold_comparison_fails(0.8, 0.8, "gt") is True
        assert threshold_comparison_fails(0.81, 0.8, "gt") is False

    def test_exact_requires_equality(self):
        assert threshold_comparison_fails(0.79, 0.8, "exact") is True
        assert threshold_comparison_fails(0.8, 0.8, "exact") is False


class TestInferentialGapBoundaryLocking:
    _FULL_BRIDGE_CONTRACT = {
        "premises": ["inspection_report", "sensor_reading"],
        "conclusion": "evacuation_route.closure_status",
        "bridge_ref": "domain.warrant.v1",
        "bridge_status": "missing",
        "bridge_mode": "requires_declared_bridge",
        "policy_ref": "support.entailment.v1",
    }

    def test_inferential_gap_not_generated_with_missing_support_metadata(self):
        pef = _pef()
        sid = _entity(pef, "advisory_claim")
        _rel(
            pef,
            sid,
            "IS",
            "recommended",
            relation_metadata={},
        )
        assert not any(r["kind"] == "inferential_gap" for r in pef_uncertainty_analysis(pef))

    def test_inferential_gap_not_generated_with_partial_bridge_contract(self):
        pef = _pef()
        sid = _entity(pef, "advisory_claim")
        _rel(
            pef,
            sid,
            "IS",
            "recommended",
            relation_metadata={
                "inference": {
                    "conclusion": "evacuation_route.closure_status",
                    "premises": ["inspection_report"],
                    # bridge_ref / bridge_status / bridge_mode intentionally missing
                }
            },
        )
        assert not any(r["kind"] == "inferential_gap" for r in pef_uncertainty_analysis(pef))

    def test_inferential_gap_not_generated_from_weak_prose_label(self):
        pef = _pef()
        sid = _entity(pef, "advisory_claim")
        _rel(
            pef,
            sid,
            "IS",
            "recommended",
            relation_metadata={
                "inference": {
                    "notes": "seems unsupported",
                    "commentary": "sounds like it follows",
                }
            },
        )
        assert not any(r["kind"] == "inferential_gap" for r in pef_uncertainty_analysis(pef))

    def test_inferential_gap_not_generated_with_missing_conclusion(self):
        pef = _pef()
        sid = _entity(pef, "advisory_claim")
        _rel(
            pef,
            sid,
            "IS",
            "recommended",
            relation_metadata={
                "inference": {
                    "premises": ["inspection_report", "sensor_reading"],
                    "bridge_ref": "domain.warrant.v1",
                    "bridge_status": "missing",
                    "bridge_mode": "requires_declared_bridge",
                }
            },
        )
        assert not any(r["kind"] == "inferential_gap" for r in pef_uncertainty_analysis(pef))

    def test_inferential_gap_not_generated_with_missing_bridge_ref(self):
        pef = _pef()
        sid = _entity(pef, "advisory_claim")
        contract = dict(self._FULL_BRIDGE_CONTRACT)
        contract.pop("bridge_ref")
        _rel(
            pef,
            sid,
            "IS",
            "recommended",
            relation_metadata={"inference": contract},
        )
        assert not any(r["kind"] == "inferential_gap" for r in pef_uncertainty_analysis(pef))

    def test_inferential_gap_not_generated_with_invalid_bridge_mode(self):
        pef = _pef()
        sid = _entity(pef, "advisory_claim")
        contract = dict(self._FULL_BRIDGE_CONTRACT)
        contract["bridge_mode"] = "model_judgement"
        _rel(
            pef,
            sid,
            "IS",
            "recommended",
            relation_metadata={"inference": contract},
        )
        assert not any(r["kind"] == "inferential_gap" for r in pef_uncertainty_analysis(pef))

    def test_inferential_gap_not_generated_from_domain_intuition_fields(self):
        pef = _pef()
        sid = _entity(pef, "advisory_claim")
        _rel(
            pef,
            sid,
            "IS",
            "recommended",
            relation_metadata={
                "inference": {
                    "domain": "emergency_management",
                    "domain_principle": "closure implies evacuation",
                    "wording_hint": "probably follows",
                }
            },
        )
        assert not any(r["kind"] == "inferential_gap" for r in pef_uncertainty_analysis(pef))


class TestInferentialGapGeneration:
    _ID_PREFIX = "generated_inferential_gap:"
    _FULL_BRIDGE_CONTRACT = TestInferentialGapBoundaryLocking._FULL_BRIDGE_CONTRACT

    def test_full_contract_missing_bridge_generates_inferential_gap_with_meta(self):
        pef = _pef()
        sid = _entity(pef, "advisory_claim")
        _rel(
            pef,
            sid,
            "IS",
            "recommended",
            relation_metadata={"inference": dict(self._FULL_BRIDGE_CONTRACT)},
        )
        records = pef_uncertainty_analysis(pef)
        generated = [r for r in records if r["kind"] == "inferential_gap"]
        assert len(generated) == 1
        rec = generated[0]
        assert str(rec["id"]).startswith(self._ID_PREFIX)
        meta = rec.get("meta") or {}
        assert meta.get("premises") == ["inspection_report", "sensor_reading"]
        assert meta.get("conclusion") == "evacuation_route.closure_status"
        assert meta.get("bridge_ref") == "domain.warrant.v1"
        assert meta.get("bridge_status") == "missing"
        assert meta.get("bridge_mode") == "requires_declared_bridge"
        assert meta.get("policy_ref") == "support.entailment.v1"
        assert rec.get("bears_on") == ["evacuation_route.closure_status"]

    @pytest.mark.parametrize(
        "bridge_status",
        ["missing", "contradicted", "expired", "out_of_scope"],
    )
    def test_non_valid_bridge_status_generates_gap(self, bridge_status: str):
        pef = _pef()
        sid = _entity(pef, "advisory_claim")
        contract = dict(self._FULL_BRIDGE_CONTRACT)
        contract["bridge_status"] = bridge_status
        _rel(
            pef,
            sid,
            "IS",
            "recommended",
            relation_metadata={"inference": contract},
        )
        generated = [
            r for r in pef_uncertainty_analysis(pef) if r["kind"] == "inferential_gap"
        ]
        assert len(generated) == 1
        assert (generated[0].get("meta") or {}).get("bridge_status") == bridge_status

    def test_valid_bridge_status_does_not_generate(self):
        pef = _pef()
        sid = _entity(pef, "advisory_claim")
        contract = dict(self._FULL_BRIDGE_CONTRACT)
        contract["bridge_status"] = "valid"
        _rel(
            pef,
            sid,
            "IS",
            "recommended",
            relation_metadata={"inference": contract},
        )
        assert not any(
            r["kind"] == "inferential_gap" for r in pef_uncertainty_analysis(pef)
        )

    def test_missing_support_alias_generates_when_bridge_invalid(self):
        pef = _pef()
        sid = _entity(pef, "advisory_claim")
        contract = dict(self._FULL_BRIDGE_CONTRACT)
        contract.pop("premises")
        contract["missing_support"] = ["inspection_report", "sensor_reading"]
        _rel(
            pef,
            sid,
            "IS",
            "recommended",
            relation_metadata={"inference": contract},
        )
        generated = [
            r for r in pef_uncertainty_analysis(pef) if r["kind"] == "inferential_gap"
        ]
        assert len(generated) == 1
        assert (generated[0].get("meta") or {}).get("premises") == [
            "inspection_report",
            "sensor_reading",
        ]

    def test_merge_reconciles_stale_generated_inferential_gap_records(self):
        pef = _pef()
        sid = _entity(pef, "advisory_claim")
        contract = dict(self._FULL_BRIDGE_CONTRACT)
        _rel(
            pef,
            sid,
            "IS",
            "recommended",
            relation_metadata={"inference": contract},
        )
        merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef))
        assert any(
            str(u.get("id", "")).startswith(self._ID_PREFIX)
            for u in pef.open_epistemic_uncertainties
        )
        contract["bridge_status"] = "valid"
        pef.relationships[0].relation_metadata = {"inference": contract}
        merge_epistemic_uncertainties(pef, pef_uncertainty_analysis(pef))
        assert not any(
            u.get("kind") == "inferential_gap" for u in pef.open_epistemic_uncertainties
        )


class TestBridgeContractSemantics:
    def test_v1_bridge_mode_is_requires_declared_bridge(self):
        assert INFERENTIAL_BRIDGE_MODE_DEFAULT == "requires_declared_bridge"

    def test_bridge_status_indicates_gap_when_not_valid(self):
        assert bridge_status_indicates_gap("missing") is True
        assert bridge_status_indicates_gap("contradicted") is True
        assert bridge_status_indicates_gap("valid") is False


# ── Anchorage scenario integration ────────────────────────────────────

class TestAnchorageScenario:
    """Full Anchorage ice-dam case: three uncertainty kinds from one PEF."""

    def test_anchorage_pef_yields_three_uncertainty_kinds(self):
        pef = _pef()
        # Weather model conflict
        wf_id = _entity(pef, "warm_front")
        _rel(pef, wf_id, "arrival_time", "48_hours", source_name="weather_model_a")
        _rel(pef, wf_id, "arrival_time", "96_hours", source_name="weather_model_b")
        # Unresolved dam entity
        _entity(pef, "dam_structural_integrity", resolved=False)
        # Blocked retrieval
        pef.retrieval_unresolved = [
            {"id": "ground_access", "subject": "dam_site", "relation": "inspection",
             "description": "Ground teams cannot reach the site due to crevasse fields."}
        ]
        records = pef_uncertainty_analysis(pef)
        kinds = {r["kind"] for r in records}
        assert "conflicting_models" in kinds
        assert "unconfirmed_status" in kinds
        assert "blocked_access" in kinds
