from __future__ import annotations

import pytest

from aurora_lens.corpus.ingest_metadata import IngestMetadata
from aurora_lens.corpus.models import CorpusRecord, utc_now_iso


def _base_record(record_id: str = "policy-v2") -> CorpusRecord:
    return CorpusRecord(
        record_id=record_id,
        title="Policy v2",
        source_path="docs/policy-v2.md",
        sha256="deadbeef",
        ingested_at=utc_now_iso(),
        chunk_count=2,
    )


def _valid_metadata(record_id: str = "policy-v2") -> dict[str, str]:
    return {
        "record_id": record_id,
        "status": "approved",
        "authority": "corporate_policy",
        "scope": "travel",
        "effective_from": "2026-01-01",
        "effective_to": "2026-12-31",
        "version": "2.0",
        "source_ref": "policy://travel/v2",
    }


def test_ingest_metadata_validates_required_fields():
    md = IngestMetadata.from_dict(_valid_metadata())
    assert md.record_id == "policy-v2"
    assert md.scope == "travel"


def test_ingest_metadata_rejects_bad_iso_dates():
    bad = _valid_metadata()
    bad["effective_to"] = "31-12-2026"
    with pytest.raises(ValueError, match="effective_to"):
        IngestMetadata.from_dict(bad)


def test_ingest_metadata_rejects_missing_required_fields():
    bad = _valid_metadata()
    bad["authority"] = "  "
    with pytest.raises(ValueError, match="authority"):
        IngestMetadata.from_dict(bad)


def test_apply_to_record_attaches_metadata():
    md = IngestMetadata.from_dict(_valid_metadata())
    rec = md.apply_to_record(_base_record())
    assert rec.status == "approved"
    assert rec.authority_class == "corporate_policy"
    assert rec.scope == "travel"
    assert rec.effective_from == "2026-01-01"
    assert rec.effective_until == "2026-12-31"
    assert rec.version == "2.0"
    assert rec.source_ref == "policy://travel/v2"


def test_apply_to_record_rejects_record_id_mismatch():
    md = IngestMetadata.from_dict(_valid_metadata(record_id="policy-v3"))
    with pytest.raises(ValueError, match="record_id"):
        md.apply_to_record(_base_record(record_id="policy-v2"))

