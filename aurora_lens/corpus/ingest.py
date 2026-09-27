"""Corpus ingestion: read sources, segment text, build chunks."""

from __future__ import annotations

import hashlib
import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from aurora_lens.corpus.models import CorpusChunk
from aurora_lens.corpus.registry import CorpusRegistry

_DEFAULT_MAX_CHUNK_CHARS = 4000
_DEFAULT_OVERLAP_CHARS = 200
_HEADING_LINE_RE = re.compile(
    r"^(?:PART\s+\d+\s*[—–-]|##\s+|###\s+(?:Section\s+\d+\s*)?)",
    re.IGNORECASE,
)

# Company ingest surface (v1). Planned later: .xlsx, .pptx, .eml, .msg
INGEST_SUPPORTED_SUFFIXES = frozenset({".pdf", ".md", ".markdown", ".txt", ".csv", ".json", ".html", ".htm", ".docx"})
INGEST_PLANNED_SUFFIXES = frozenset({".xlsx", ".pptx", ".eml", ".msg"})


def ingest_supported_suffix() -> frozenset[str]:
    return INGEST_SUPPORTED_SUFFIXES


def planned_ingest_suffix() -> frozenset[str]:
    return INGEST_PLANNED_SUFFIXES


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        return "\n".join(self._parts).strip()


def _read_docx(path: Path) -> tuple[str, dict[str, object]]:
    try:
        from docx import Document
    except ImportError as e:
        raise ValueError(
            "DOCX ingest requires python-docx (pip install aurora-lens[ingest])"
        ) from e
    document = Document(str(path))
    parts = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    body = "\n\n".join(parts).strip()
    meta: dict[str, object] = {"extractor": "python-docx", "page_count": None}
    return body, meta


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source_text(path: Path) -> tuple[str, dict[str, object]]:
    """Return (body text, source.meta fields)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ValueError(
                "PDF ingest requires pypdf (pip install aurora-lens[ingest])"
            ) from e
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                parts.append(text.strip())
        body = "\n\n".join(parts).strip()
        meta: dict[str, object] = {
            "extractor": "pypdf",
            "page_count": len(reader.pages),
        }
    elif suffix in {".md", ".markdown", ".txt"}:
        body = path.read_text(encoding="utf-8").strip()
        meta = {"extractor": "text", "page_count": None}
    elif suffix == ".csv":
        body = path.read_text(encoding="utf-8", errors="replace").strip()
        meta = {"extractor": "csv", "page_count": None}
    elif suffix == ".json":
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
        try:
            parsed = json.loads(raw)
            body = json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            body = raw
        meta = {"extractor": "json", "page_count": None}
    elif suffix in {".html", ".htm"}:
        raw = path.read_text(encoding="utf-8", errors="replace")
        parser = _HTMLTextExtractor()
        parser.feed(raw)
        body = html.unescape(parser.text())
        meta = {"extractor": "html", "page_count": None}
    elif suffix == ".docx":
        body, meta = _read_docx(path)
    elif suffix in INGEST_PLANNED_SUFFIXES:
        raise ValueError(f"Type {suffix!r} is planned but not supported in this release")
    else:
        supported = ", ".join(sorted(INGEST_SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported source type: {path.suffix!r} (supported: {supported})")

    if not body:
        raise ValueError(f"No extractable text in {path}")
    return body, meta


def _locator_from_heading(line: str) -> str:
    s = line.strip()
    return s[:240] if s else "section"


def segment_text_by_headings(text: str) -> list[tuple[str, str]]:
    """Split on PART / ## / ### heading lines. Returns (locator, body) pairs."""
    lines = text.splitlines()
    indices: list[int] = []
    headings: list[str] = []
    for i, line in enumerate(lines):
        if _HEADING_LINE_RE.match(line.strip()):
            indices.append(i)
            headings.append(line.strip())

    if not indices:
        return []

    segments: list[tuple[str, str]] = []
    for j, start in enumerate(indices):
        end = indices[j + 1] if j + 1 < len(indices) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        if not body:
            continue
        segments.append((_locator_from_heading(headings[j]), body))
    return segments


def segment_text_by_size(
    text: str,
    *,
    max_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
    overlap: int = _DEFAULT_OVERLAP_CHARS,
) -> list[tuple[str, str]]:
    """Fixed-size windows with overlap when structural headings are absent."""
    if max_chars < 256:
        raise ValueError("max_chars must be >= 256")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")
    body = text.strip()
    if not body:
        return []
    if len(body) <= max_chars:
        return [("Block 1", body)]

    segments: list[tuple[str, str]] = []
    start = 0
    block = 1
    while start < len(body):
        end = min(start + max_chars, len(body))
        chunk = body[start:end].strip()
        if chunk:
            segments.append((f"Block {block}", chunk))
            block += 1
        if end >= len(body):
            break
        start = max(start + 1, end - overlap)
    return segments


def build_chunks(
    record_id: str,
    segments: list[tuple[str, str]],
) -> list[CorpusChunk]:
    rid = record_id.strip()
    if not rid:
        raise ValueError("record_id must be non-empty")
    if not segments:
        raise ValueError("segments must be non-empty")
    return [
        CorpusChunk.build(
            record_id=rid,
            ordinal=i,
            locator=locator,
            text=body,
        )
        for i, (locator, body) in enumerate(segments)
    ]


def chunk_source_text(
    text: str,
    *,
    max_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
    overlap: int = _DEFAULT_OVERLAP_CHARS,
) -> list[tuple[str, str]]:
    """Prefer heading-based segments; fall back to size windows or split large sections."""
    heading_segments = segment_text_by_headings(text)
    if not heading_segments:
        return segment_text_by_size(text, max_chars=max_chars, overlap=overlap)

    out: list[tuple[str, str]] = []
    for locator, body in heading_segments:
        if len(body) <= max_chars:
            out.append((locator, body))
        else:
            for sub_loc, sub_body in segment_text_by_size(
                body, max_chars=max_chars, overlap=overlap
            ):
                out.append((f"{locator} / {sub_loc}", sub_body))
    return out


def write_source_meta(record_dir: Path, *, source_path: Path, sha256: str, meta: dict[str, object]) -> None:
    record_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_path": str(source_path),
        "sha256": sha256,
        **meta,
    }
    (record_dir / "source.meta.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
