"""Typed corpus-consistency checks over already admissible evidence sets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from aurora_lens.corpus.evidence_admissibility import _STATUS_ADMISSIBLE
from aurora_lens.corpus.models import CorpusChunk, CorpusRecord
from aurora_lens.verify.numeric import NumericValue, parse_all_numerics, values_contradict

# Single source of truth for live/authority-bearing statuses (see evidence_admissibility).
_ACTIVE_RECORD_STATUSES = _STATUS_ADMISSIBLE


@dataclass(frozen=True)
class ThresholdValue:
    record_id: str
    threshold_key: str
    value: NumericValue


def _is_active_record(record: CorpusRecord) -> bool:
    status = (record.status or "").strip().lower()
    return status in _ACTIVE_RECORD_STATUSES


def _normalize_threshold_key(line: str) -> str:
    low = line.lower()
    idx = low.find("threshold")
    if idx < 0:
        return ""
    prefix = low[max(0, idx - 48) : idx].strip()
    tokens = [t for t in prefix.replace(":", " ").replace("-", " ").split() if t]
    if not tokens:
        return "threshold"
    return " ".join(tokens[-4:]) + " threshold"


def _threshold_values_for_record(
    *,
    record_id: str,
    chunks: list[CorpusChunk],
) -> list[ThresholdValue]:
    out: list[ThresholdValue] = []
    for ch in chunks:
        for line in ch.text.splitlines():
            if "threshold" not in line.lower():
                continue
            vals = parse_all_numerics(line)
            if not vals:
                continue
            key = _normalize_threshold_key(line)
            if not key:
                continue
            # First slice: one threshold literal per line is the typed signal.
            out.append(ThresholdValue(record_id=record_id, threshold_key=key, value=vals[0]))
    return out


def _parse_iso_date_safe(raw: str | None) -> date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _effective_date_conflict_for_pair(
    *,
    left: CorpusRecord,
    right: CorpusRecord,
) -> tuple[str, list[dict]] | None:
    left_from = _parse_iso_date_safe(left.effective_from)
    left_to = _parse_iso_date_safe(left.effective_until)
    right_from = _parse_iso_date_safe(right.effective_from)
    right_to = _parse_iso_date_safe(right.effective_until)

    # If both windows are explicitly bounded and disjoint while both are active,
    # the commitment-scoped evidence set carries incompatible applicability claims.
    if left_from and left_to and right_from and right_to:
        disjoint = left_to < right_from or right_to < left_from
        if disjoint:
            return (
                "disjoint_effective_windows_for_active_records",
                [
                    {
                        "record_id": left.record_id,
                        "effective_from": left.effective_from,
                        "effective_to": left.effective_until,
                        "status": left.status,
                    },
                    {
                        "record_id": right.record_id,
                        "effective_from": right.effective_from,
                        "effective_to": right.effective_until,
                        "status": right.status,
                    },
                ],
            )

    left_supersedes_right = right.record_id in left.supersedes or left.record_id in right.superseded_by
    right_supersedes_left = left.record_id in right.supersedes or right.record_id in left.superseded_by
    if left_supersedes_right or right_supersedes_left:
        return (
            "supersession_claim_conflicts_with_active_applicability",
            [
                {
                    "record_id": left.record_id,
                    "status": left.status,
                    "supersedes": list(left.supersedes),
                    "superseded_by": list(left.superseded_by),
                },
                {
                    "record_id": right.record_id,
                    "status": right.status,
                    "supersedes": list(right.supersedes),
                    "superseded_by": list(right.superseded_by),
                },
            ],
        )
    return None


def detect_consistency_conflicts(
    *,
    chunks: list[CorpusChunk],
    records: list[CorpusRecord],
    commitment_scope: str | None,
) -> list[dict]:
    """Return typed consistency conflicts; does not alter admissibility state."""
    if len(records) < 2 or len(chunks) < 2:
        return []

    by_record: dict[str, CorpusRecord] = {r.record_id: r for r in records if _is_active_record(r)}
    if len(by_record) < 2:
        return []

    by_scope: dict[str, list[CorpusRecord]] = defaultdict(list)
    for rec in by_record.values():
        scope = (rec.scope or "").strip()
        if not scope:
            continue
        if commitment_scope and scope != commitment_scope:
            continue
        by_scope[scope].append(rec)

    conflicts: list[dict] = []
    chunks_by_record: dict[str, list[CorpusChunk]] = defaultdict(list)
    for ch in chunks:
        if ch.record_id in by_record:
            chunks_by_record[ch.record_id].append(ch)

    for scope, scoped_records in by_scope.items():
        if len(scoped_records) < 2:
            continue
        keyed_values: dict[str, list[ThresholdValue]] = defaultdict(list)
        for rec in scoped_records:
            tvals = _threshold_values_for_record(
                record_id=rec.record_id,
                chunks=chunks_by_record.get(rec.record_id, []),
            )
            for tv in tvals:
                keyed_values[tv.threshold_key].append(tv)

        for threshold_key, vals in keyed_values.items():
            if len(vals) < 2:
                continue
            v0 = vals[0].value
            incompatible = any(values_contradict(v0, v.value) for v in vals[1:])
            if not incompatible:
                continue
            rows = [
                {
                    "record_id": v.record_id,
                    "magnitude": v.value.magnitude,
                    "unit": v.value.unit.value,
                }
                for v in vals
            ]
            conflicts.append(
                {
                    "kind": "threshold_conflict",
                    "shared_scope": scope,
                    "threshold_key": threshold_key,
                    "record_ids": sorted({v.record_id for v in vals}),
                    "conflicting_values": rows,
                    "reason": "incompatible_threshold_values_same_scope",
                }
            )
            record_ids = sorted({v.record_id for v in vals})
            for i in range(len(record_ids)):
                left = by_record[record_ids[i]]
                for j in range(i + 1, len(record_ids)):
                    right = by_record[record_ids[j]]
                    detail = _effective_date_conflict_for_pair(left=left, right=right)
                    if detail is None:
                        continue
                    reason, rows = detail
                    conflicts.append(
                        {
                            "kind": "effective_date_conflict",
                            "shared_scope": scope,
                            "threshold_key": threshold_key,
                            "record_ids": [left.record_id, right.record_id],
                            "effective_windows": rows,
                            "reason": reason,
                        }
                    )
    return conflicts


def detect_threshold_conflicts(
    *,
    chunks: list[CorpusChunk],
    records: list[CorpusRecord],
    commitment_scope: str | None,
) -> list[dict]:
    """Backward-compatible alias for threshold-only callers."""
    return [
        c
        for c in detect_consistency_conflicts(
            chunks=chunks,
            records=records,
            commitment_scope=commitment_scope,
        )
        if c.get("kind") == "threshold_conflict"
    ]

