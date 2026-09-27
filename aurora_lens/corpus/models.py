"""Corpus record and chunk models (wire-serializable)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def make_chunk_id(record_id: str, ordinal: int) -> str:
    """Stable chunk id from record id and zero-based ordinal."""
    rid = (record_id or "").strip()
    if not rid:
        raise ValueError("record_id must be non-empty")
    if ordinal < 0:
        raise ValueError("ordinal must be >= 0")
    return f"{rid}--chunk-{ordinal:04d}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CorpusRecord:
    """Metadata for one ingested corpus document."""

    record_id: str
    title: str
    source_path: str
    sha256: str
    ingested_at: str
    chunk_count: int
    authority_class: str | None = None
    status: str | None = None
    effective_from: str | None = None
    effective_until: str | None = None
    review_due: str | None = None
    jurisdiction: str | None = None
    department: str | None = None
    scope: str | None = None
    version: str | None = None
    supersedes: list[str] = field(default_factory=list)
    superseded_by: list[str] = field(default_factory=list)
    confidentiality: str | None = None
    allowed_roles: list[str] = field(default_factory=list)
    parser: str | None = None
    parser_status: str | None = None
    source_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "title": self.title,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "ingested_at": self.ingested_at,
            "chunk_count": self.chunk_count,
            "authority_class": self.authority_class,
            "status": self.status,
            "effective_from": self.effective_from,
            "effective_until": self.effective_until,
            "review_due": self.review_due,
            "jurisdiction": self.jurisdiction,
            "department": self.department,
            "scope": self.scope,
            "version": self.version,
            "supersedes": list(self.supersedes),
            "superseded_by": list(self.superseded_by),
            "confidentiality": self.confidentiality,
            "allowed_roles": list(self.allowed_roles),
            "parser": self.parser,
            "parser_status": self.parser_status,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusRecord:
        record_id = str(data.get("record_id", "")).strip()
        if not record_id:
            raise ValueError("record_id is required")
        title = str(data.get("title", "")).strip() or record_id
        source_path = str(data.get("source_path", "")).strip()
        sha256 = str(data.get("sha256", "")).strip().lower()
        ingested_at = str(data.get("ingested_at", "")).strip()
        if not ingested_at:
            raise ValueError("ingested_at is required")
        try:
            chunk_count = int(data.get("chunk_count", 0))
        except (TypeError, ValueError) as e:
            raise ValueError("chunk_count must be an integer") from e
        if chunk_count < 0:
            raise ValueError("chunk_count must be >= 0")

        def _opt_str(name: str) -> str | None:
            value = data.get(name)
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def _opt_str_list(name: str) -> list[str]:
            value = data.get(name)
            if value is None:
                return []
            if isinstance(value, str):
                candidate = value.strip()
                return [candidate] if candidate else []
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a list of strings")
            out: list[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    out.append(text)
            return out

        return cls(
            record_id=record_id,
            title=title,
            source_path=source_path,
            sha256=sha256,
            ingested_at=ingested_at,
            chunk_count=chunk_count,
            authority_class=_opt_str("authority_class"),
            status=_opt_str("status"),
            effective_from=_opt_str("effective_from"),
            effective_until=_opt_str("effective_until"),
            review_due=_opt_str("review_due"),
            jurisdiction=_opt_str("jurisdiction"),
            department=_opt_str("department"),
            scope=_opt_str("scope"),
            version=_opt_str("version"),
            supersedes=_opt_str_list("supersedes"),
            superseded_by=_opt_str_list("superseded_by"),
            confidentiality=_opt_str("confidentiality"),
            allowed_roles=_opt_str_list("allowed_roles"),
            parser=_opt_str("parser"),
            parser_status=_opt_str("parser_status"),
            source_ref=_opt_str("source_ref"),
        )


@dataclass(frozen=True)
class CorpusChunk:
    """One text segment of a corpus record."""

    chunk_id: str
    record_id: str
    locator: str
    text: str
    ordinal: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "record_id": self.record_id,
            "locator": self.locator,
            "text": self.text,
            "ordinal": self.ordinal,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusChunk:
        record_id = str(data.get("record_id", "")).strip()
        if not record_id:
            raise ValueError("record_id is required")
        try:
            ordinal = int(data.get("ordinal", -1))
        except (TypeError, ValueError) as e:
            raise ValueError("ordinal must be an integer") from e
        if ordinal < 0:
            raise ValueError("ordinal must be >= 0")
        chunk_id = str(data.get("chunk_id", "")).strip() or make_chunk_id(record_id, ordinal)
        expected = make_chunk_id(record_id, ordinal)
        if chunk_id != expected:
            raise ValueError(
                f"chunk_id {chunk_id!r} does not match expected {expected!r} "
                f"for record_id={record_id!r} ordinal={ordinal}"
            )
        locator = str(data.get("locator", "")).strip()
        text = data.get("text", "")
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        return cls(
            chunk_id=chunk_id,
            record_id=record_id,
            locator=locator,
            text=text,
            ordinal=ordinal,
        )

    @classmethod
    def build(
        cls,
        *,
        record_id: str,
        ordinal: int,
        text: str,
        locator: str = "",
    ) -> CorpusChunk:
        rid = record_id.strip()
        if not rid:
            raise ValueError("record_id must be non-empty")
        return cls(
            chunk_id=make_chunk_id(rid, ordinal),
            record_id=rid,
            locator=locator.strip(),
            text=text,
            ordinal=ordinal,
        )
