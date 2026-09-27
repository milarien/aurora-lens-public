"""File-based corpus registry and per-record chunk JSONL I/O."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from aurora_lens.corpus.models import CorpusChunk, CorpusRecord

_REGISTRY_VERSION = 1


def default_corpus_root() -> Path:
    """Default on-disk corpus root for CLI (override with ``AURORA_LENS_CORPUS_ROOT``)."""
    env = os.environ.get("AURORA_LENS_CORPUS_ROOT", "").strip()
    if env:
        return Path(env)
    return Path("data") / "corpus"


def operator_corpus_root() -> Path:
    """Isolated corpus root for the operator console (not the shared CLI registry).

    Override with ``AURORA_LENS_OPERATOR_CORPUS_ROOT``. Keeps console ingest/list
    separate from historical ``data/corpus`` entries.
    """
    env = os.environ.get("AURORA_LENS_OPERATOR_CORPUS_ROOT", "").strip()
    if env:
        return Path(env)
    return Path("data") / "corpus" / "operator"


class CorpusRegistry:
    """Load and save corpus records and chunk files under a single root directory.

    Layout::

        {root}/registry.json
        {root}/records/{record_id}/chunks.jsonl
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_corpus_root()

    @property
    def registry_path(self) -> Path:
        return self.root / "registry.json"

    def record_dir(self, record_id: str) -> Path:
        rid = record_id.strip()
        if not rid:
            raise ValueError("record_id must be non-empty")
        return self.root / "records" / rid

    def chunks_path(self, record_id: str) -> Path:
        return self.record_dir(record_id) / "chunks.jsonl"

    def load_records(self) -> dict[str, CorpusRecord]:
        """Return all records keyed by ``record_id`` (empty dict if missing file)."""
        path = self.registry_path
        if not path.is_file():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid registry format in {path}")
        version = raw.get("version", _REGISTRY_VERSION)
        if version != _REGISTRY_VERSION:
            raise ValueError(f"Unsupported registry version {version!r} in {path}")
        entries = raw.get("records", [])
        if not isinstance(entries, list):
            raise ValueError(f"registry.records must be a list in {path}")
        out: dict[str, CorpusRecord] = {}
        for item in entries:
            if not isinstance(item, dict):
                raise ValueError(f"registry.records entries must be objects in {path}")
            rec = CorpusRecord.from_dict(item)
            if rec.record_id in out:
                raise ValueError(f"Duplicate record_id {rec.record_id!r} in {path}")
            out[rec.record_id] = rec
        return out

    def save_records(self, records: dict[str, CorpusRecord]) -> None:
        """Write registry.json (sorted by record_id for stable diffs)."""
        self.root.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": _REGISTRY_VERSION,
            "records": [records[k].to_dict() for k in sorted(records)],
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        self.registry_path.write_text(text, encoding="utf-8")

    def get_record(self, record_id: str) -> CorpusRecord | None:
        return self.load_records().get(record_id.strip())

    def upsert_record(self, record: CorpusRecord) -> None:
        records = self.load_records()
        records[record.record_id] = record
        self.save_records(records)

    def delete_record(self, record_id: str) -> bool:
        """Remove record from registry (does not delete chunk files)."""
        rid = record_id.strip()
        records = self.load_records()
        if rid not in records:
            return False
        del records[rid]
        self.save_records(records)
        return True

    def load_chunks(self, record_id: str) -> list[CorpusChunk]:
        """Load chunks for a record (empty list if file missing)."""
        path = self.chunks_path(record_id)
        if not path.is_file():
            return []
        chunks: list[CorpusChunk] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Chunk line must be a JSON object at {path}:{line_no}")
            chunks.append(CorpusChunk.from_dict(obj))
        chunks.sort(key=lambda c: c.ordinal)
        rid = record_id.strip()
        for i, ch in enumerate(chunks):
            if ch.ordinal != i:
                raise ValueError(
                    f"Chunk ordinals must be contiguous from 0 at {path}; "
                    f"expected {i}, got {ch.ordinal}"
                )
            if ch.record_id != rid:
                raise ValueError(
                    f"chunk record_id mismatch at {path} ordinal {i}: "
                    f"{ch.record_id!r} != {rid!r}"
                )
        return chunks

    def save_chunks(self, record_id: str, chunks: list[CorpusChunk]) -> None:
        """Write chunks.jsonl (ordinals must be contiguous from 0)."""
        rid = record_id.strip()
        if not rid:
            raise ValueError("record_id must be non-empty")
        sorted_chunks = sorted(chunks, key=lambda c: c.ordinal)
        for i, ch in enumerate(sorted_chunks):
            if ch.ordinal != i:
                raise ValueError(f"chunk ordinals must be contiguous from 0; gap at {i}")
            if ch.record_id != rid:
                raise ValueError(
                    f"chunk record_id {ch.record_id!r} != save target {rid!r}"
                )
            expected_id = ch.chunk_id
            built_id = CorpusChunk.build(
                record_id=rid,
                ordinal=i,
                text=ch.text,
                locator=ch.locator,
            ).chunk_id
            if expected_id != built_id:
                raise ValueError(f"chunk_id {expected_id!r} != expected {built_id!r}")

        record_dir = self.record_dir(rid)
        record_dir.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(ch.to_dict(), ensure_ascii=False) for ch in sorted_chunks]
        self.chunks_path(rid).write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
