from __future__ import annotations

from aurora_lens.corpus.evidence_consistency import detect_consistency_conflicts
from aurora_lens.corpus.models import CorpusChunk, CorpusRecord


def _record(
    *,
    record_id: str,
    scope: str,
    status: str = "approved",
    effective_from: str | None = None,
    effective_until: str | None = None,
    supersedes: list[str] | None = None,
    superseded_by: list[str] | None = None,
) -> CorpusRecord:
    return CorpusRecord(
        record_id=record_id,
        title=record_id,
        source_path=f"docs/{record_id}.md",
        sha256="abc123",
        ingested_at="2026-07-01T00:00:00+00:00",
        chunk_count=1,
        status=status,
        scope=scope,
        effective_from=effective_from,
        effective_until=effective_until,
        supersedes=list(supersedes or []),
        superseded_by=list(superseded_by or []),
    )


def _threshold_chunk(record_id: str, value: str, scope: str = "policy") -> CorpusChunk:
    return CorpusChunk.build(
        record_id=record_id,
        ordinal=0,
        locator=f"### {scope}",
        text=f"{scope} approval threshold is {value}.",
    )


def test_effective_date_conflict_for_disjoint_windows():
    records = [
        _record(record_id="r1", scope="policy", effective_from="2026-01-01", effective_until="2026-03-31"),
        _record(record_id="r2", scope="policy", effective_from="2026-07-01", effective_until="2026-12-31"),
    ]
    chunks = [_threshold_chunk("r1", "$5,000"), _threshold_chunk("r2", "$10,000")]
    conflicts = detect_consistency_conflicts(chunks=chunks, records=records, commitment_scope="policy")
    ed = [c for c in conflicts if c.get("kind") == "effective_date_conflict"]
    assert len(ed) == 1
    assert ed[0]["reason"] == "disjoint_effective_windows_for_active_records"
    assert ed[0]["shared_scope"] == "policy"
    assert ed[0]["threshold_key"]


def test_effective_date_conflict_for_supersession_status():
    records = [
        _record(record_id="r1", scope="policy", superseded_by=["r2"]),
        _record(record_id="r2", scope="policy"),
    ]
    chunks = [_threshold_chunk("r1", "$5,000"), _threshold_chunk("r2", "$10,000")]
    conflicts = detect_consistency_conflicts(chunks=chunks, records=records, commitment_scope="policy")
    ed = [c for c in conflicts if c.get("kind") == "effective_date_conflict"]
    assert len(ed) == 1
    assert ed[0]["reason"] == "supersession_claim_conflicts_with_active_applicability"


def test_effective_date_conflict_not_emitted_without_threshold_pair():
    records = [
        _record(record_id="r1", scope="policy", effective_from="2026-01-01", effective_until="2026-03-31"),
        _record(record_id="r2", scope="policy", effective_from="2026-07-01", effective_until="2026-12-31"),
    ]
    chunks = [
        CorpusChunk.build(record_id="r1", ordinal=0, locator="### policy", text="no threshold phrase here"),
        CorpusChunk.build(record_id="r2", ordinal=0, locator="### policy", text="still no numeric threshold"),
    ]
    conflicts = detect_consistency_conflicts(chunks=chunks, records=records, commitment_scope="policy")
    assert not any(c.get("kind") == "effective_date_conflict" for c in conflicts)


def test_effective_date_conflict_not_emitted_for_different_scope():
    records = [
        _record(record_id="r1", scope="policy", effective_from="2026-01-01", effective_until="2026-03-31"),
        _record(record_id="r2", scope="finance", effective_from="2026-07-01", effective_until="2026-12-31"),
    ]
    chunks = [_threshold_chunk("r1", "$5,000", scope="policy"), _threshold_chunk("r2", "$10,000", scope="finance")]
    conflicts = detect_consistency_conflicts(chunks=chunks, records=records, commitment_scope="policy")
    assert not any(c.get("kind") == "effective_date_conflict" for c in conflicts)

