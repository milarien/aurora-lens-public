from __future__ import annotations

from datetime import datetime, timezone

from aurora_lens.corpus.evidence_admissibility import evaluate_retrieved_evidence
from aurora_lens.corpus.models import CorpusChunk, CorpusRecord


def _record(**overrides) -> CorpusRecord:
    base = dict(
        record_id="policy-v1",
        title="Policy V1",
        source_path="docs/policy-v1.md",
        sha256="deadbeef",
        ingested_at="2026-06-29T00:00:00+00:00",
        chunk_count=1,
        authority_class="corporate_policy",
        status="approved",
        effective_from="2026-01-01",
        effective_until="2026-12-31",
        scope="travel",
        parser_status="ok",
    )
    base.update(overrides)
    return CorpusRecord(**base)


def _chunks(record_id: str = "policy-v1") -> list[CorpusChunk]:
    return [
        CorpusChunk.build(
            record_id=record_id,
            ordinal=0,
            locator="Section 1",
            text="Travel over threshold needs manager approval.",
        )
    ]


def _now() -> datetime:
    return datetime(2026, 6, 15, tzinfo=timezone.utc)


def test_approved_current_record_with_matching_chunks_is_admissible():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(),
        now=_now(),
        request_scope="travel",
    )
    assert result.status == "admissible"
    assert result.failure_kind is None


def test_record_missing_is_inadmissible_record_missing():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=None,
        now=_now(),
    )
    assert result.status == "inadmissible"
    assert result.failure_kind == "record_missing"


def test_no_chunks_is_inadmissible_no_evidence():
    result = evaluate_retrieved_evidence(
        chunks=[],
        record=_record(),
        now=_now(),
    )
    assert result.status == "inadmissible"
    assert result.failure_kind == "no_evidence"


def test_chunk_record_id_mismatch_is_inadmissible_traceability_failure():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(record_id="other-record"),
        record=_record(),
        now=_now(),
    )
    assert result.status == "inadmissible"
    assert result.failure_kind == "traceability_failure"


def test_draft_or_unapproved_is_inadmissible_authority_missing():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status="draft"),
        now=_now(),
    )
    assert result.status == "inadmissible"
    assert result.failure_kind == "authority_missing"


def test_expired_effective_until_is_inadmissible_stale_evidence():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(effective_until="2026-01-01"),
        now=_now(),
    )
    assert result.status == "inadmissible"
    assert result.failure_kind == "stale_evidence"
    assert result.failure_meta.get("stale_source") == "effective_until"


def test_effective_from_after_now_is_inadmissible_unrevalidated_orientation():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(effective_from="2026-12-01"),
        now=_now(),
    )
    assert result.status == "inadmissible"
    assert result.failure_kind == "unrevalidated_orientation"
    assert result.failure_meta.get("orientation_field") == "effective_from"


def test_superseded_by_non_empty_is_inadmissible_superseded_evidence():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(superseded_by=["policy-v2"]),
        now=_now(),
    )
    assert result.status == "inadmissible"
    assert result.failure_kind == "superseded_evidence"


def test_missing_authority_with_requirement_is_unresolved_authority_unknown():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(authority_class=None),
        now=_now(),
        require_authority=True,
    )
    assert result.status == "unresolved"
    assert result.failure_kind == "authority_unknown"


def test_known_good_authority_class_is_admissible():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(authority_class="corporate_policy"),
        now=_now(),
        request_scope="travel",
        require_authority=True,
    )
    assert result.status == "admissible"
    assert result.failure_kind is None


def test_empty_authority_class_fails_closed_when_required():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(authority_class=""),
        now=_now(),
        require_authority=True,
    )
    assert result.status == "unresolved"
    assert result.failure_kind == "authority_unknown"


def test_typo_authority_class_fails_closed_unknown_authority_class():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(authority_class="verifed"),
        now=_now(),
        require_authority=True,
    )
    assert result.status == "unresolved"
    assert result.failure_kind == "unknown_authority_class"


def test_untrusted_authority_class_fails_closed_unknown_authority_class():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(authority_class="untrusted"),
        now=_now(),
        require_authority=True,
    )
    assert result.status == "unresolved"
    assert result.failure_kind == "unknown_authority_class"


def test_missing_authority_class_may_proceed_when_not_required():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(authority_class=None),
        now=_now(),
        request_scope="travel",
        require_authority=False,
    )
    assert result.status == "admissible"
    assert result.failure_kind is None


def test_empty_authority_class_may_proceed_when_not_required():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(authority_class=""),
        now=_now(),
        request_scope="travel",
        require_authority=False,
    )
    assert result.status == "admissible"
    assert result.failure_kind is None


def test_typo_authority_class_fails_closed_even_when_not_required():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(authority_class="verifed"),
        now=_now(),
        require_authority=False,
    )
    assert result.status == "unresolved"
    assert result.failure_kind == "unknown_authority_class"


def test_untrusted_authority_class_fails_closed_even_when_not_required():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(authority_class="untrusted"),
        now=_now(),
        require_authority=False,
    )
    assert result.status == "unresolved"
    assert result.failure_kind == "unknown_authority_class"


def test_request_scope_mismatch_is_unresolved_scope_mismatch():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(scope="hr"),
        now=_now(),
        request_scope="travel",
    )
    assert result.status == "unresolved"
    assert result.failure_kind == "scope_mismatch"


def test_request_scope_allows_global_record_scope():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(scope="global"),
        now=_now(),
        request_scope="travel",
    )
    assert result.status == "admissible"
    assert result.failure_kind is None


def test_malformed_date_is_unresolved_missing_freshness():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(effective_until="2026/15/99"),
        now=_now(),
    )
    assert result.status == "unresolved"
    assert result.failure_kind == "missing_freshness"
    assert result.failure_meta.get("missing_field") == "effective_until"


def test_approved_status_remains_admissible():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status="approved"),
        now=_now(),
        request_scope="travel",
    )
    assert result.status == "admissible"
    assert result.failure_kind is None


def test_active_status_is_admissible():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status="active"),
        now=_now(),
        request_scope="travel",
    )
    assert result.status == "admissible"
    assert result.failure_kind is None


def test_missing_status_none_fails_closed_missing_record_status():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status=None),
        now=_now(),
    )
    assert result.status == "unresolved"
    assert result.failure_kind == "missing_record_status"


def test_empty_status_fails_closed_missing_record_status():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status=""),
        now=_now(),
    )
    assert result.status == "unresolved"
    assert result.failure_kind == "missing_record_status"


def test_typo_of_revoked_status_fails_closed_unknown_record_status():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status="revokedd"),
        now=_now(),
    )
    assert result.status == "unresolved"
    assert result.failure_kind == "unknown_record_status"


def test_typo_of_expired_status_fails_closed_unknown_record_status():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status="expiredd"),
        now=_now(),
    )
    assert result.status == "unresolved"
    assert result.failure_kind == "unknown_record_status"


def test_revoked_status_is_inadmissible_authority_revoked():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status="revoked"),
        now=_now(),
    )
    assert result.status == "inadmissible"
    assert result.failure_kind == "authority_revoked"


def test_withdrawn_status_is_inadmissible_authority_revoked():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status="withdrawn"),
        now=_now(),
    )
    assert result.status == "inadmissible"
    assert result.failure_kind == "authority_revoked"


def test_suspended_status_is_inadmissible_authority_revoked():
    result = evaluate_retrieved_evidence(
        chunks=_chunks(),
        record=_record(status="suspended"),
        now=_now(),
    )
    assert result.status == "inadmissible"
    assert result.failure_kind == "authority_revoked"
