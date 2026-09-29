"""Unit tests for the epistemic uncertainty gate."""
from aurora_lens.govern.epistemic_uncertainty_gate import (
    evaluate_epistemic_uncertainty_gate,
    is_decision_seeking,
    parse_open_epistemic_uncertainties,
)
from aurora_lens.pef.uncertainty_analysis import (
    EXTERNAL_ESTABLISHMENT_FAILURE_KINDS,
    VALID_OPEN_EPISTEMIC_KINDS,
)

_ANCHORAGE_UNCERTAINTIES = [
    {
        "id": "warm_front_timing",
        "kind": "predictive_uncertainty",
        "description": "Weather models disagree on warm front timing.",
        "bears_on": ["evacuation_strategy"],
        "status": "open",
    },
    {
        "id": "dam_visual_confirmation",
        "kind": "missing_observation",
        "description": "Cloud cover has prevented visual confirmation.",
        "bears_on": ["dam_status"],
        "status": "open",
    },
]


class TestIsDecisionSeeking:
    def test_choose_verb_in_question(self):
        assert is_decision_seeking("which of the four options should the emergency manager choose?")

    def test_recommend_verb_in_question(self):
        assert is_decision_seeking("What do you recommend?")

    def test_which_option_phrase_question(self):
        assert is_decision_seeking("Which option is best?")

    def test_given_the_above_explicit_request(self):
        assert is_decision_seeking("Given the above, which option should be taken?")

    def test_what_should_phrase(self):
        assert is_decision_seeking("What should they decide?")

    def test_imperative_recommend(self):
        assert is_decision_seeking("Please recommend an evacuation strategy.")

    def test_analysis_request_not_decision_seeking(self):
        assert not is_decision_seeking("Provide a structured analysis of the risks.")

    def test_empty_not_decision_seeking(self):
        assert not is_decision_seeking("")

    def test_scenario_prose_must_choose_not_decision_seeking(self):
        # Scenario describing someone else's decision — must NOT trigger.
        assert not is_decision_seeking(
            "The city emergency manager must choose one of four actions by 06:00 tomorrow."
        )

    def test_scenario_prose_has_to_decide_not_decision_seeking(self):
        assert not is_decision_seeking(
            "The committee has to decide before the deadline."
        )

    def test_scenario_with_confirm_understanding_not_decision_seeking(self):
        # The Anchorage Turn 1 text ends with a confirmation request, not a question.
        assert not is_decision_seeking(
            "The city emergency manager must choose one of four actions. "
            "Please confirm your understanding of the scenario before proceeding."
        )


class TestParseOpenEpistemicUncertainties:
    def test_parses_valid_list(self):
        result = parse_open_epistemic_uncertainties(_ANCHORAGE_UNCERTAINTIES)
        assert len(result) == 2
        assert result[0]["id"] == "warm_front_timing"

    def test_skips_missing_id(self):
        raw = [{"kind": "x", "description": "y"}]
        assert parse_open_epistemic_uncertainties(raw) == []

    def test_non_list_returns_empty(self):
        assert parse_open_epistemic_uncertainties(None) == []

    def test_parses_kind_metadata_for_stale_evidence(self):
        raw = [
            {
                "id": "s1",
                "kind": "stale_evidence",
                "description": "Evidence is stale for this decision.",
                "meta": {
                    "decay_mode": "clock",
                    "timestamp_source": "observed_at",
                    "observed_at": "2026-06-01T12:00:00Z",
                    "assessed_at": "2026-06-02T18:00:00Z",
                    "freshness_window": "P1D",
                    "policy_ref": "ops.weather.v1",
                    "ignored_field": "drop-me",
                },
            }
        ]
        parsed = parse_open_epistemic_uncertainties(raw)
        assert len(parsed) == 1
        meta = parsed[0].get("meta") or {}
        assert meta.get("decay_mode") == "clock"
        assert meta.get("timestamp_source") == "observed_at"
        assert meta.get("observed_at") == "2026-06-01T12:00:00Z"
        assert meta.get("assessed_at") == "2026-06-02T18:00:00Z"
        assert meta.get("freshness_window") == "P1D"
        assert meta.get("policy_ref") == "ops.weather.v1"
        assert "ignored_field" not in meta

    def test_stale_event_mode_metadata_normalizes_known_fields_only(self):
        raw = [
            {
                "id": "s2",
                "kind": "stale_evidence",
                "description": "Provision was superseded.",
                "meta": {
                    "decay_mode": "event",
                    "timestamp_source": "asserted_at",
                    "asserted_at": "2020-01-01T00:00:00Z",
                    "evidence_ref": "law:act:42:s7",
                    "supersession_rule": "same_jurisdiction_amendment",
                    "policy_ref": "law.currency.v1",
                    "decision_time": "2026-06-01T00:00:00Z",
                    "drop_me": "x",
                },
            }
        ]
        parsed = parse_open_epistemic_uncertainties(raw)
        assert len(parsed) == 1
        meta = parsed[0].get("meta") or {}
        assert meta.get("decay_mode") == "event"
        assert meta.get("timestamp_source") == "asserted_at"
        assert meta.get("asserted_at") == "2020-01-01T00:00:00Z"
        assert meta.get("evidence_ref") == "law:act:42:s7"
        assert meta.get("supersession_rule") == "same_jurisdiction_amendment"
        assert meta.get("decision_time") == "2026-06-01T00:00:00Z"
        assert "drop_me" not in meta

    def test_parses_kind_metadata_for_threshold_not_met_numeric_fields(self):
        raw = [
            {
                "id": "t1",
                "kind": "threshold_not_met",
                "description": "Threshold not met.",
                "meta": {
                    "consequence_grade": "high",
                    "required_strength": 0.8,
                    "observed_strength": 0.52,
                    "threshold_ref": "policy.threshold.v1",
                    "comparison_mode": "gte",
                    "policy_ref": "decision.threshold.v2",
                },
            }
        ]
        parsed = parse_open_epistemic_uncertainties(raw)
        assert len(parsed) == 1
        meta = parsed[0].get("meta") or {}
        assert meta.get("consequence_grade") == "high"
        assert meta.get("required_strength") == 0.8
        assert meta.get("observed_strength") == 0.52
        assert meta.get("threshold_ref") == "policy.threshold.v1"
        assert meta.get("comparison_mode") == "gte"
        assert meta.get("policy_ref") == "decision.threshold.v2"

    def test_threshold_not_met_drops_informal_numeric_shapes(self):
        raw = [
            {
                "id": "t2",
                "kind": "threshold_not_met",
                "description": "Threshold not met.",
                "meta": {
                    "consequence_grade": "high",
                    "required_strength": "0.8",  # string should be dropped
                    "observed_strength": "0.52",  # string should be dropped
                    "threshold_ref": "policy.threshold.v1",
                    "policy_ref": "decision.policy.v3",
                },
            }
        ]
        parsed = parse_open_epistemic_uncertainties(raw)
        assert len(parsed) == 1
        meta = parsed[0].get("meta") or {}
        assert meta.get("consequence_grade") == "high"
        assert meta.get("threshold_ref") == "policy.threshold.v1"
        assert meta.get("policy_ref") == "decision.policy.v3"
        assert "required_strength" not in meta
        assert "observed_strength" not in meta

    def test_threshold_not_met_drops_invalid_consequence_grade_and_comparison_mode(self):
        raw = [
            {
                "id": "t3",
                "kind": "threshold_not_met",
                "description": "Threshold not met.",
                "meta": {
                    "consequence_grade": "severe",
                    "required_strength": 0.8,
                    "observed_strength": 0.52,
                    "threshold_ref": "policy.threshold.v1",
                    "comparison_mode": "strict",
                },
            }
        ]
        parsed = parse_open_epistemic_uncertainties(raw)
        assert len(parsed) == 1
        meta = parsed[0].get("meta") or {}
        assert meta.get("required_strength") == 0.8
        assert meta.get("observed_strength") == 0.52
        assert meta.get("threshold_ref") == "policy.threshold.v1"
        assert "consequence_grade" not in meta
        assert "comparison_mode" not in meta

    def test_parses_kind_metadata_for_scope_and_support_list_fields(self):
        raw = [
            {
                "id": "x1",
                "kind": "scope_mismatch",
                "description": "Evidence scope differs from required scope.",
                "meta": {
                    "required_scope": {
                        "domain": "medical_procurement",
                        "jurisdiction": "Singapore",
                        "subject": "supplier_123",
                    },
                    "supported_scope": {
                        "domain": "office_supplies",
                        "jurisdiction": "Australia",
                        "subject": "supplier_123",
                    },
                    "scope_dimensions": ["subject", "domain", "jurisdiction"],
                    "comparison_mode": "exact",
                    "failed_dimensions": ["domain", "jurisdiction"],
                    "policy_ref": "scope.policy.v1",
                },
            },
            {
                "id": "x2",
                "kind": "inferential_gap",
                "description": "Support does not establish requested conclusion.",
                "meta": {
                    "premises": ["inspection_report", "sensor_reading"],
                    "conclusion": "evacuation_route.closure_status",
                    "bridge_ref": "domain.warrant.v1",
                    "bridge_status": "missing",
                    "bridge_mode": "requires_declared_bridge",
                    "policy_ref": "support.entailment.v1",
                },
            },
        ]
        parsed = parse_open_epistemic_uncertainties(raw)
        assert len(parsed) == 2
        scope_meta = parsed[0].get("meta") or {}
        infer_meta = parsed[1].get("meta") or {}
        assert scope_meta.get("required_scope", {}).get("jurisdiction") == "Singapore"
        assert scope_meta.get("supported_scope", {}).get("jurisdiction") == "Australia"
        assert scope_meta.get("scope_dimensions") == ["subject", "domain", "jurisdiction"]
        assert scope_meta.get("comparison_mode") == "exact"
        assert scope_meta.get("failed_dimensions") == ["domain", "jurisdiction"]
        assert scope_meta.get("policy_ref") == "scope.policy.v1"
        assert infer_meta.get("conclusion") == "evacuation_route.closure_status"
        assert infer_meta.get("premises") == ["inspection_report", "sensor_reading"]
        assert infer_meta.get("bridge_ref") == "domain.warrant.v1"
        assert infer_meta.get("bridge_status") == "missing"
        assert infer_meta.get("bridge_mode") == "requires_declared_bridge"
        assert infer_meta.get("policy_ref") == "support.entailment.v1"

    def test_inferential_gap_drops_informal_premises_and_invalid_bridge_fields(self):
        raw = [
            {
                "id": "x4",
                "kind": "inferential_gap",
                "description": "Support does not establish requested conclusion.",
                "meta": {
                    "conclusion": "evacuation_route.closure_status",
                    "premises": "inspection_report",  # informal shape should drop
                    "bridge_ref": "domain.warrant.v1",
                    "bridge_status": "sounds_missing",
                    "bridge_mode": "model_judgement",
                    "policy_ref": "support.entailment.v1",
                },
            }
        ]
        parsed = parse_open_epistemic_uncertainties(raw)
        assert len(parsed) == 1
        meta = parsed[0].get("meta") or {}
        assert meta.get("conclusion") == "evacuation_route.closure_status"
        assert meta.get("bridge_ref") == "domain.warrant.v1"
        assert meta.get("policy_ref") == "support.entailment.v1"
        assert "premises" not in meta
        assert "bridge_status" not in meta
        assert "bridge_mode" not in meta

    def test_inferential_gap_accepts_missing_support_alias_as_premises(self):
        raw = [
            {
                "id": "x5",
                "kind": "inferential_gap",
                "description": "Support does not establish requested conclusion.",
                "meta": {
                    "conclusion": "evacuation_route.closure_status",
                    "missing_support": ["inspection_report", "sensor_reading"],
                    "bridge_ref": "domain.warrant.v1",
                    "bridge_status": "missing",
                    "bridge_mode": "requires_declared_bridge",
                },
            }
        ]
        parsed = parse_open_epistemic_uncertainties(raw)
        assert len(parsed) == 1
        meta = parsed[0].get("meta") or {}
        assert meta.get("premises") == ["inspection_report", "sensor_reading"]

    def test_scope_mismatch_metadata_drops_informal_scope_shapes(self):
        raw = [
            {
                "id": "x3",
                "kind": "scope_mismatch",
                "description": "Evidence scope differs from required scope.",
                "meta": {
                    "evidence_scope": "free-text-scope",
                    "required_scope": {"domain": "au-state"},
                    "scope_dimensions": ["domain"],
                    "comparison_mode": "exact",
                    "policy_ref": "scope.policy.v1",
                    "ignored_field": "drop",
                },
            }
        ]
        parsed = parse_open_epistemic_uncertainties(raw)
        assert len(parsed) == 1
        meta = parsed[0].get("meta") or {}
        assert meta.get("policy_ref") == "scope.policy.v1"
        assert meta.get("required_scope") == {"domain": "au-state"}
        assert "supported_scope" not in meta
        assert "evidence_scope" not in meta
        assert "ignored_field" not in meta

    def test_parses_kind_metadata_for_source_untrusted_explicit_registry_contract(self):
        raw = [
            {
                "id": "u2",
                "kind": "source_untrusted",
                "description": "Source trust policy failed.",
                "meta": {
                    "source_id": "source:vendor-17",
                    "trust_registry_ref": "registry:ops:v1",
                    "authority_map_ref": "authmap:ops:v1",
                    "trust_decision": "deny",
                    "policy_ref": "ops.source.v1",
                    "ignored_field": "drop",
                },
            }
        ]
        parsed = parse_open_epistemic_uncertainties(raw)
        assert len(parsed) == 1
        meta = parsed[0].get("meta") or {}
        assert meta.get("source_id") == "source:vendor-17"
        assert meta.get("trust_registry_ref") == "registry:ops:v1"
        assert meta.get("authority_map_ref") == "authmap:ops:v1"
        assert meta.get("trust_decision") == "deny"
        assert meta.get("policy_ref") == "ops.source.v1"
        assert "ignored_field" not in meta

    def test_metadata_normalization_is_non_breaking_for_unknown_kind(self):
        raw = [
            {
                "id": "u1",
                "kind": "future_kind",
                "description": "Future external declaration.",
                "meta": {"whatever": "kept-or-dropped"},
            }
        ]
        parsed = parse_open_epistemic_uncertainties(raw)
        assert len(parsed) == 1


class TestEvaluateEpistemicUncertaintyGate:
    def test_blocks_decision_request_with_open_uncertainties(self):
        result = evaluate_epistemic_uncertainty_gate(
            _ANCHORAGE_UNCERTAINTIES,
            "which of the four options should the emergency manager choose, and why?"
        )
        assert result is not None
        assert result.blocks is True
        assert "does not establish a justified recommendation" in result.response_text
        assert "Weather models disagree" in result.response_text
        assert "Escalate" in result.response_text
        assert "Admissible continuation" in result.response_text
        assert len(result.matched_uncertainties) == 2

    def test_allows_through_when_no_open_uncertainties(self):
        closed = [dict(u, status="closed") for u in _ANCHORAGE_UNCERTAINTIES]
        result = evaluate_epistemic_uncertainty_gate(closed, "which option should be chosen?")
        assert result is None

    def test_allows_through_when_no_uncertainties(self):
        result = evaluate_epistemic_uncertainty_gate([], "which option should be chosen?")
        assert result is None

    def test_allows_through_non_decision_turn(self):
        result = evaluate_epistemic_uncertainty_gate(
            _ANCHORAGE_UNCERTAINTIES,
            "Provide a structured analysis of the risks and trade-offs."
        )
        assert result is None

    def test_admissible_continuation_in_response(self):
        result = evaluate_epistemic_uncertainty_gate(
            _ANCHORAGE_UNCERTAINTIES,
            "what should we decide?"
        )
        assert result is not None
        assert "structured analysis" in result.response_text


class TestPhase2KindsRecognisedByGate:
    """Phase 2 establishment-failure kinds flow through the gate identically to CORE kinds.

    The gate does not inspect kind strings — it blocks on open status and decision-seeking
    turn detection.  These tests confirm each Phase 2 kind is accepted in
    evidence_state.open_epistemic_uncertainties and produces a block on a decision turn.
    """

    def _record(self, kind: str) -> dict:
        return {
            "id": f"test_{kind}",
            "kind": kind,
            "description": f"Test declaration of {kind}.",
            "bears_on": ["test_claim"],
            "status": "open",
        }

    def test_all_phase2_kinds_in_external_establishment_failure_kinds(self):
        for kind in EXTERNAL_ESTABLISHMENT_FAILURE_KINDS:
            assert kind in VALID_OPEN_EPISTEMIC_KINDS

    def test_identity_or_referent_unresolved_blocks_decision_turn(self):
        result = evaluate_epistemic_uncertainty_gate(
            [self._record("identity_or_referent_unresolved")],
            "which option should we choose?"
        )
        assert result is not None and result.blocks is True

    def test_stale_evidence_blocks_decision_turn(self):
        result = evaluate_epistemic_uncertainty_gate(
            [self._record("stale_evidence")],
            "which option should we choose?"
        )
        assert result is not None and result.blocks is True

    def test_inferential_gap_blocks_decision_turn(self):
        result = evaluate_epistemic_uncertainty_gate(
            [self._record("inferential_gap")],
            "which option should we choose?"
        )
        assert result is not None and result.blocks is True

    def test_threshold_not_met_blocks_decision_turn(self):
        result = evaluate_epistemic_uncertainty_gate(
            [self._record("threshold_not_met")],
            "which option should we choose?"
        )
        assert result is not None and result.blocks is True

    def test_source_untrusted_blocks_decision_turn(self):
        result = evaluate_epistemic_uncertainty_gate(
            [self._record("source_untrusted")],
            "which option should we choose?"
        )
        assert result is not None and result.blocks is True

    def test_scope_mismatch_blocks_decision_turn(self):
        result = evaluate_epistemic_uncertainty_gate(
            [self._record("scope_mismatch")],
            "which option should we choose?"
        )
        assert result is not None and result.blocks is True

    def test_unknown_kind_is_non_breaking(self):
        # Gate is permissive — unknown kind strings do not raise and do block when open.
        result = evaluate_epistemic_uncertainty_gate(
            [self._record("some_future_kind_not_yet_in_taxonomy")],
            "which option should we choose?"
        )
        assert result is not None and result.blocks is True

    def test_unknown_kind_passes_parse(self):
        raw = [{"id": "x", "kind": "completely_unknown", "description": "unknown kind"}]
        parsed = parse_open_epistemic_uncertainties(raw)
        assert len(parsed) == 1
