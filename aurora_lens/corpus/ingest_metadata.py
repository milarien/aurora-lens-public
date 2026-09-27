"""Bounded ingest metadata validation and record attachment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Any, Mapping

from aurora_lens.corpus.models import CorpusRecord


def _required_str(data: Mapping[str, Any], field: str) -> str:
    raw = data.get(field)
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"ingest metadata requires non-empty {field!r}")
    return text


def _validated_iso_date(value: str, field: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as e:
        raise ValueError(f"ingest metadata field {field!r} must be ISO date (YYYY-MM-DD)") from e
    return value


@dataclass(frozen=True)
class IngestMetadata:
    """Bounded metadata intake payload for real-corpus document ingestion."""

    record_id: str
    status: str
    authority: str
    scope: str
    effective_from: str
    effective_to: str
    version: str
    source_ref: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IngestMetadata:
        rid = _required_str(data, "record_id")
        status = _required_str(data, "status")
        authority = _required_str(data, "authority")
        scope = _required_str(data, "scope")
        effective_from = _validated_iso_date(_required_str(data, "effective_from"), "effective_from")
        effective_to = _validated_iso_date(_required_str(data, "effective_to"), "effective_to")
        version = _required_str(data, "version")
        source_ref = _required_str(data, "source_ref")
        return cls(
            record_id=rid,
            status=status,
            authority=authority,
            scope=scope,
            effective_from=effective_from,
            effective_to=effective_to,
            version=version,
            source_ref=source_ref,
        )

    def apply_to_record(self, record: CorpusRecord) -> CorpusRecord:
        if record.record_id != self.record_id:
            raise ValueError(
                "ingest metadata record_id does not match ingested record_id "
                f"({self.record_id!r} != {record.record_id!r})"
            )
        return replace(
            record,
            status=self.status,
            authority_class=self.authority,
            scope=self.scope,
            effective_from=self.effective_from,
            effective_until=self.effective_to,
            version=self.version,
            source_ref=self.source_ref,
        )

