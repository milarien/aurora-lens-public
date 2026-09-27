"""Deterministic admissibility checks for retrieved corpus evidence.

Authority demotion: ``authority_state`` on :class:`EvidenceAdmissibilityResult` participates
in Lens admissibility outcome selection (see :mod:`aurora_lens.corpus.evidence_authority`).
Stale evidence is ``signal_only`` — retained as contextual signal, not treated as absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal, Sequence

from aurora_lens.corpus.evidence_authority import (
    EvidenceAuthorityState,
    derive_evidence_authority,
)
from aurora_lens.corpus.models import CorpusChunk, CorpusRecord

EvidenceAdmissibilityStatus = Literal["admissible", "inadmissible", "unresolved"]

_PARSER_STATUS_ACCEPTED = frozenset({"ok", "parsed", "complete", "unknown"})
# Recognised authority-bearing (live) record statuses. Only these may carry
# consequence-bearing authority; everything else fails closed. Shared with
# aurora_lens.corpus.evidence_consistency to avoid a split-brain "active" contract.
_STATUS_ADMISSIBLE = frozenset({"approved", "active"})
_STATUS_AUTHORITY_MISSING = frozenset({"draft", "unapproved"})
_STATUS_STALE = frozenset({"archived", "retired", "expired"})
# Explicitly dead statuses: authority was granted then removed. Hard-inadmissible.
_STATUS_REVOKED = frozenset({"revoked", "withdrawn", "suspended", "cancelled", "invalid"})
# Recognised authority-bearing document authority classes. ``authority_class`` is an
# open operator-supplied descriptor, so this allowlist is intentionally narrow and
# grows only with confirmed corpus vocabulary. A non-empty string is not enough:
# unrecognised values fail closed to revalidation (CONTAIN), never silent PASS.
_AUTHORITY_CLASS_ADMISSIBLE = frozenset({"corporate_policy"})
_SCOPE_COMPATIBLE_GLOBAL = frozenset({"global", "default"})


@dataclass(frozen=True)
class EvidenceAdmissibilityResult:
    status: EvidenceAdmissibilityStatus
    failure_kind: str | None
    record_id: str
    chunk_ids: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    failure_meta: dict[str, str] = field(default_factory=dict)
    authority_state: str = EvidenceAuthorityState.AUTHORITATIVE.value
    freshness_status: str | None = None
    authority_status: str | None = None
    demotion_reason: str | None = None


def _with_authority_fields(
    *,
    status: EvidenceAdmissibilityStatus,
    failure_kind: str | None,
    record_id: str,
    chunk_ids: list[str],
    reasons: list[str],
    failure_meta: dict[str, str],
    record_status: str | None = None,
) -> EvidenceAdmissibilityResult:
    authority, freshness_status, authority_status, demotion_reason = derive_evidence_authority(
        status=status,
        failure_kind=failure_kind,
        failure_meta=failure_meta,
        record_status=record_status,
    )
    enriched_meta = dict(failure_meta)
    enriched_meta.setdefault("authority_state", authority.value)
    if freshness_status:
        enriched_meta.setdefault("freshness_status", freshness_status)
    if authority_status:
        enriched_meta.setdefault("authority_status", authority_status)
    if demotion_reason:
        enriched_meta.setdefault("demotion_reason", demotion_reason)
    return EvidenceAdmissibilityResult(
        status=status,
        failure_kind=failure_kind,
        record_id=record_id,
        chunk_ids=chunk_ids,
        reasons=reasons,
        failure_meta=enriched_meta,
        authority_state=authority.value,
        freshness_status=freshness_status,
        authority_status=authority_status,
        demotion_reason=demotion_reason,
    )


def _now_as_date(now: datetime | date | None) -> date:
    if now is None:
        return datetime.now(timezone.utc).date()
    if isinstance(now, datetime):
        return now.date()
    return now


def _parse_iso_date(value: str, *, field_name: str) -> tuple[date | None, str | None]:
    candidate = value.strip()
    if not candidate:
        return None, None
    try:
        return date.fromisoformat(candidate), None
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(candidate.replace("Z", "+00:00")).date(), None
    except ValueError:
        return None, f"{field_name} must be ISO date/date-time: {value!r}"


def evaluate_retrieved_evidence(
    *,
    chunks: Sequence[CorpusChunk],
    record: CorpusRecord | None,
    now: datetime | date | None = None,
    request_scope: str | None = None,
    require_authority: bool = True,
) -> EvidenceAdmissibilityResult:
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if record is None:
        return _with_authority_fields(
            status="inadmissible",
            failure_kind="record_missing",
            record_id="",
            chunk_ids=chunk_ids,
            reasons=["Retrieved chunks cannot be admitted without a parent CorpusRecord."],
            failure_meta={"missing": "record"},
        )

    record_id = record.record_id
    if not chunks:
        return _with_authority_fields(
            status="inadmissible",
            failure_kind="no_evidence",
            record_id=record_id,
            chunk_ids=chunk_ids,
            reasons=["No evidence chunks were retrieved for this record."],
            failure_meta={"missing": "chunks"},
            record_status=record.status,
        )

    for chunk in chunks:
        if chunk.record_id != record_id:
            return _with_authority_fields(
                status="inadmissible",
                failure_kind="traceability_failure",
                record_id=record_id,
                chunk_ids=chunk_ids,
                reasons=[
                    f"Chunk {chunk.chunk_id!r} has record_id={chunk.record_id!r}, "
                    f"expected {record_id!r}."
                ],
                failure_meta={"expected_record_id": record_id, "actual_record_id": chunk.record_id},
                record_status=record.status,
            )

    parser_status = (record.parser_status or "").strip().lower() or None
    if parser_status is not None and parser_status not in _PARSER_STATUS_ACCEPTED:
        return _with_authority_fields(
            status="inadmissible",
            failure_kind="parser_unusable",
            record_id=record_id,
            chunk_ids=chunk_ids,
            reasons=[f"parser_status={record.parser_status!r} is not admissible."],
            failure_meta={"parser_status": str(record.parser_status or "")},
            record_status=record.status,
        )

    # Explicit known-good status admission. "Ingestion proposes. Lens disposes."
    # Only recognised live statuses may proceed; missing/unknown fail closed to
    # revalidation, and known-dead statuses fail closed as inadmissible.
    status = (record.status or "").strip().lower()
    if not status:
        return _with_authority_fields(
            status="unresolved",
            failure_kind="missing_record_status",
            record_id=record_id,
            chunk_ids=chunk_ids,
            reasons=["record.status is missing; authority cannot be established without a recognised status."],
            failure_meta={"record_status": ""},
            record_status=record.status,
        )
    if status in _STATUS_AUTHORITY_MISSING:
        return _with_authority_fields(
            status="inadmissible",
            failure_kind="authority_missing",
            record_id=record_id,
            chunk_ids=chunk_ids,
            reasons=[f"record.status={record.status!r} does not establish authority."],
            failure_meta={"record_status": str(record.status or "")},
            record_status=record.status,
        )
    if status in _STATUS_STALE:
        return _with_authority_fields(
            status="inadmissible",
            failure_kind="stale_evidence",
            record_id=record_id,
            chunk_ids=chunk_ids,
            reasons=[f"record.status={record.status!r} marks evidence as stale."],
            failure_meta={"stale_source": "record_status", "record_status": str(record.status or "")},
            record_status=record.status,
        )
    if status in _STATUS_REVOKED:
        return _with_authority_fields(
            status="inadmissible",
            failure_kind="authority_revoked",
            record_id=record_id,
            chunk_ids=chunk_ids,
            reasons=[f"record.status={record.status!r} marks authority as revoked/withdrawn."],
            failure_meta={"record_status": str(record.status or "")},
            record_status=record.status,
        )
    if status not in _STATUS_ADMISSIBLE:
        return _with_authority_fields(
            status="unresolved",
            failure_kind="unknown_record_status",
            record_id=record_id,
            chunk_ids=chunk_ids,
            reasons=[f"record.status={record.status!r} is not a recognised authority-bearing status."],
            failure_meta={"record_status": str(record.status or "")},
            record_status=record.status,
        )

    today = _now_as_date(now)
    if record.effective_from:
        effective_from, parse_err = _parse_iso_date(
            record.effective_from,
            field_name="effective_from",
        )
        if parse_err:
            return _with_authority_fields(
                status="unresolved",
                failure_kind="missing_freshness",
                record_id=record_id,
                chunk_ids=chunk_ids,
                reasons=[parse_err],
                failure_meta={
                    "missing_field": "effective_from",
                    "stale_source": "effective_window",
                },
                record_status=record.status,
            )
        if effective_from is not None and effective_from > today:
            return _with_authority_fields(
                status="inadmissible",
                failure_kind="unrevalidated_orientation",
                record_id=record_id,
                chunk_ids=chunk_ids,
                reasons=[
                    f"effective_from={record.effective_from!r} is later than evaluation date {today.isoformat()}."
                ],
                failure_meta={
                    "orientation_field": "effective_from",
                    "effective_from": str(record.effective_from),
                    "evaluated_on": today.isoformat(),
                },
                record_status=record.status,
            )

    if record.effective_until:
        effective_until, parse_err = _parse_iso_date(
            record.effective_until,
            field_name="effective_until",
        )
        if parse_err:
            return _with_authority_fields(
                status="unresolved",
                failure_kind="missing_freshness",
                record_id=record_id,
                chunk_ids=chunk_ids,
                reasons=[parse_err],
                failure_meta={
                    "missing_field": "effective_until",
                    "stale_source": "effective_window",
                },
                record_status=record.status,
            )
        if effective_until is not None and effective_until < today:
            return _with_authority_fields(
                status="inadmissible",
                failure_kind="stale_evidence",
                record_id=record_id,
                chunk_ids=chunk_ids,
                reasons=[
                    f"effective_until={record.effective_until!r} is earlier than evaluation date {today.isoformat()}."
                ],
                failure_meta={
                    "stale_source": "effective_until",
                    "effective_until": str(record.effective_until),
                    "evaluated_on": today.isoformat(),
                },
                record_status=record.status,
            )

    if record.superseded_by:
        return _with_authority_fields(
            status="inadmissible",
            failure_kind="superseded_evidence",
            record_id=record_id,
            chunk_ids=chunk_ids,
            reasons=[f"record superseded_by={record.superseded_by!r} is non-empty."],
            failure_meta={"superseded_by_count": str(len(record.superseded_by))},
            record_status=record.status,
        )

    # Missing/empty authority only blocks when authority is required. But a
    # present, non-empty authority_class must be a recognised value regardless of
    # require_authority: a non-empty string may never stand in for authority.
    authority_class = (record.authority_class or "").strip().lower()
    if not authority_class:
        if require_authority:
            return _with_authority_fields(
                status="unresolved",
                failure_kind="authority_unknown",
                record_id=record_id,
                chunk_ids=chunk_ids,
                reasons=["authority_class is missing while require_authority=True."],
                failure_meta={"missing_field": "authority_class"},
                record_status=record.status,
            )
    elif authority_class not in _AUTHORITY_CLASS_ADMISSIBLE:
        return _with_authority_fields(
            status="unresolved",
            failure_kind="unknown_authority_class",
            record_id=record_id,
            chunk_ids=chunk_ids,
            reasons=[
                f"authority_class={record.authority_class!r} is not a recognised authority-bearing class."
            ],
            failure_meta={"authority_class": str(record.authority_class or "")},
            record_status=record.status,
        )

    if request_scope is not None:
        expected_scope = request_scope.strip().lower()
        record_scope = (record.scope or "").strip().lower()
        if not record_scope:
            return _with_authority_fields(
                status="unresolved",
                failure_kind="scope_unknown",
                record_id=record_id,
                chunk_ids=chunk_ids,
                reasons=["record.scope is missing for a scoped request."],
                failure_meta={"missing_field": "scope"},
                record_status=record.status,
            )
        if record_scope != expected_scope and record_scope not in _SCOPE_COMPATIBLE_GLOBAL:
            return _with_authority_fields(
                status="unresolved",
                failure_kind="scope_mismatch",
                record_id=record_id,
                chunk_ids=chunk_ids,
                reasons=[f"record.scope={record_scope!r} does not equal request_scope={expected_scope!r}."],
                failure_meta={"record_scope": record_scope, "request_scope": expected_scope},
                record_status=record.status,
            )

    return _with_authority_fields(
        status="admissible",
        failure_kind=None,
        record_id=record_id,
        chunk_ids=chunk_ids,
        reasons=[],
        failure_meta={},
        record_status=record.status,
    )
