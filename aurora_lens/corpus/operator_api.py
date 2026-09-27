"""Structured corpus operator API for the forensics console and ``/v1/corpus/*`` routes.

Retrieval proposes. Lens disposes. This module orchestrates ingest, list, retrieve,
ask, validate, and review — it does not decide admissibility.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aurora_lens.corpus.document_ingestion import ingest_file
from aurora_lens.corpus.evidence_admissibility import evaluate_retrieved_evidence
from aurora_lens.corpus.evidence_roles import (
    EVIDENCE_ROLE_LABELS,
    LOW_RELEVANCE_WARNING,
    SECTION_TITLES,
    classify_evidence_role,
    evidence_for_governed_review,
    group_evidence_sections,
    visible_evidence_for_display,
)
from aurora_lens.corpus.folder_scan import (
    FolderScanReport,
    SkippedFile,
    record_id_for_file,
    scan_path,
)
from aurora_lens.corpus.qa_validation import (
    CaseSpec,
    answers_abstain,
    load_manifest,
    score_case,
)
from aurora_lens.corpus.registry import CorpusRegistry, operator_corpus_root
from aurora_lens.corpus.retrieve import (
    assemble_rag_message,
    assemble_request_metadata,
    chunk_relevance_score,
    retrieve_chunks,
)
from aurora_lens.corpus.review import (
    format_corpus_for_review,
    run_corpus_review,
    select_chunks,
    upstream_config_status,
)
from aurora_lens.corpus.cli_support import assistant_text, post_chat
from aurora_lens.lens import split_rag_context_question

_SUPPORT_STATUSES = frozenset(
    {
        "supported",
        "not_supported",
        "partial",
        "scope_mismatch",
        "inferential_gap",
        "threshold_issue",
        "source_issue",
    }
)

_GLOBAL_SOURCE_SCOPES = frozenset({"global", "default"})


def _resolve_corpus_root(corpus_root: Path | None) -> Path:
    return corpus_root if corpus_root is not None else operator_corpus_root()


def _normalize_path_prefix(prefix: str) -> str:
    return str(Path(prefix).expanduser().resolve()).lower()


def _record_ids_from_ingest_results(results: list[dict[str, Any]]) -> list[str]:
    return [str(r["record_id"]) for r in results if r.get("record_id")]


def _file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {
        ".pdf": "pdf",
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
        ".csv": "csv",
        ".json": "json",
        ".html": "html",
        ".htm": "html",
        ".docx": "docx",
    }
    return mapping.get(suffix, suffix.lstrip(".") or "unknown")


def _slug_record_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "document"


def _chunk_to_evidence(
    chunk: Any,
    *,
    record_title: str,
    source_path: str,
    query: str,
) -> dict[str, Any]:
    text = chunk.text.strip()
    snippet = text if len(text) <= 480 else text[:477] + "…"
    score = chunk_relevance_score(chunk, query) if query.strip() else None
    role = classify_evidence_role(
        score=score,
        question=query,
        record_id=chunk.record_id,
        document_name=record_title,
        source_path=source_path,
        snippet=snippet,
    )
    return {
        "record_id": chunk.record_id,
        "document_name": record_title,
        "document_title": record_title,
        "chunk_id": chunk.chunk_id,
        "ordinal": chunk.ordinal,
        "block_number": chunk.ordinal,
        "locator": chunk.locator or f"chunk {chunk.ordinal}",
        "snippet": snippet,
        "score": score,
        "source_path": source_path,
        "evidence_role": role,
        "evidence_role_label": EVIDENCE_ROLE_LABELS[role],
    }


def _normalize_scope_token(value: str | None) -> str | None:
    token = (value or "").strip().lower()
    return token or None


def _record_matches_request_scope(
    *,
    record_scope: str | None,
    requested_scope: str | None,
) -> bool:
    """Whether a record is eligible for scoped retrieval before admissibility."""
    expected = _normalize_scope_token(requested_scope)
    if expected is None:
        return True
    actual = _normalize_scope_token(record_scope)
    if actual is None:
        return False
    if actual == expected:
        return True
    return actual in _GLOBAL_SOURCE_SCOPES


def _scope_filtered_targets(
    *,
    targets: list[str],
    records: dict[str, Any],
    request_scope: str | None,
) -> list[str]:
    if not request_scope or not request_scope.strip():
        return targets
    return [
        rid
        for rid in targets
        if _record_matches_request_scope(
            record_scope=getattr(records.get(rid), "scope", None),
            requested_scope=request_scope,
        )
    ]


def _extract_flags(aurora: dict[str, Any] | None) -> list[str]:
    if not aurora:
        return []
    raw = aurora.get("flags") or []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            name = item.get("type") or item.get("flag_type") or item.get("name")
            if name:
                out.append(str(name))
    return out


def derive_support_status(
    *,
    answer: str,
    governance: str,
    flags: list[str] | None = None,
    validation_reasons: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Map governed answer + flags to operator-facing support status (not admissibility)."""
    issues: list[str] = list(validation_reasons or [])
    gov = (governance or "?").strip().upper()
    flag_blob = " ".join(flags or []).upper()

    if "SCOPE_MISMATCH" in flag_blob:
        return "scope_mismatch", issues + ["scope mismatch flagged by Lens"]
    if "INFERENTIAL_GAP" in flag_blob:
        return "inferential_gap", issues + ["inferential gap flagged by Lens"]
    if "THRESHOLD" in flag_blob and ("UNCERTAINTY" in flag_blob or "ESTABLISHMENT" in flag_blob):
        return "threshold_issue", issues + ["threshold / establishment issue flagged by Lens"]
    if "SOURCE_UNTRUSTED" in flag_blob or ("SOURCE" in flag_blob and "TRUST" in flag_blob):
        return "source_issue", issues + ["source trust issue flagged by Lens"]

    if gov in {"HARD_STOP", "STOP", "REFUSE"}:
        return "not_supported", issues + [f"governance={gov}"]
    if gov in {"CONTAIN", "CLARIFY", "FORCE_REVISE"}:
        return "partial", issues + [f"governance={gov}"]
    if answers_abstain(answer):
        return "not_supported", issues + ["answer abstains from retrieved context"]
    if issues:
        return "partial", issues
    return "supported", []


def api_list_records(
    *,
    corpus_root: Path | None = None,
    record_ids: list[str] | None = None,
    source_prefix: str | None = None,
) -> dict[str, Any]:
    registry = CorpusRegistry(_resolve_corpus_root(corpus_root))
    records = registry.load_records()
    rows: list[dict[str, Any]] = []
    for rid in sorted(records):
        rec = records[rid]
        src = Path(rec.source_path)
        rows.append(
            {
                "record_id": rec.record_id,
                "title": rec.title,
                "filename": src.name if src.name else rec.source_path,
                "file_type": _file_type(src),
                "status": "ingested",
                "chunk_count": rec.chunk_count,
                "source_path": rec.source_path,
                "sha256_prefix": rec.sha256[:16] if rec.sha256 else "",
                "ingested_at": rec.ingested_at,
            }
        )
    total = len(rows)
    scope: dict[str, Any] = {"mode": "all_in_operator_corpus"}
    if record_ids:
        wanted = {rid.strip() for rid in record_ids if rid.strip()}
        rows = [row for row in rows if row["record_id"] in wanted]
        scope = {"mode": "record_ids", "record_ids": sorted(wanted)}
    elif source_prefix and source_prefix.strip():
        prefix = _normalize_path_prefix(source_prefix.strip())
        rows = [
            row
            for row in rows
            if _normalize_path_prefix(row["source_path"]).startswith(prefix)
            or prefix.startswith(_normalize_path_prefix(row["source_path"]))
        ]
        scope = {"mode": "source_prefix", "source_prefix": source_prefix.strip()}
    return {
        "count": len(rows),
        "total_in_registry": total,
        "corpus_root": str(registry.root.resolve()),
        "registry_path": str(registry.registry_path),
        "scope": scope,
        "records": rows,
    }


def api_scan_folder(
    *,
    path: Path,
    recursive: bool = True,
) -> dict[str, Any]:
    """Preview folder contents before ingest (no writes)."""
    return scan_path(path, recursive=recursive).to_dict()


def _ingest_from_scan(
    scan: FolderScanReport,
    *,
    registry: CorpusRegistry,
    root: Path,
    record_id: str | None,
    title: str | None,
    force: bool,
    max_chunk_chars: int,
    overlap: int,
) -> None:
    used_ids: set[str] = set()
    for file_path in scan.supported_files:
        rid = record_id_for_file(file_path, root, record_id)
        if rid in used_ids:
            base = rid
            n = 2
            while f"{base}-{n}" in used_ids:
                n += 1
            rid = f"{base}-{n}"
        used_ids.add(rid)
        try:
            record, chunks, did_write = ingest_file(
                registry,
                record_id=rid,
                source_path=file_path,
                title=title or file_path.stem,
                force=force,
                max_chunk_chars=max_chunk_chars,
                overlap=overlap,
            )
            status = "ingested" if did_write else "unchanged"
            if did_write:
                scan.files_ingested += 1
            else:
                scan.files_unchanged += 1
            scan.ingest_results.append(
                {
                    "status": status,
                    "record_id": record.record_id,
                    "title": record.title,
                    "filename": file_path.name,
                    "file_type": _file_type(file_path),
                    "source_path": str(file_path),
                    "relative_path": str(file_path.relative_to(root)),
                    "chunk_count": record.chunk_count,
                    "total_chars": sum(len(c.text) for c in chunks),
                }
            )
        except ValueError as exc:
            scan.files_skipped += 1
            suffix_key = file_path.suffix.lower() or "(no extension)"
            scan.skipped_by_type[suffix_key] = scan.skipped_by_type.get(suffix_key, 0) + 1
            reason: str = "empty_text" if "No extractable text" in str(exc) else "ingest_error"
            scan.skipped_files.append(
                SkippedFile(
                    path=str(file_path),
                    relative_path=str(file_path.relative_to(root)),
                    suffix=suffix_key,
                    reason=reason,
                    detail=str(exc),
                )
            )
            scan.ingest_errors.append({"path": str(file_path), "error": str(exc)})


def api_ingest_path(
    *,
    path: Path,
    record_id: str | None = None,
    title: str | None = None,
    force: bool = False,
    corpus_root: Path | None = None,
    max_chunk_chars: int = 4000,
    overlap: int = 200,
    recursive: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ingest one file or every supported file in a directory (recursive by default)."""
    target = path.expanduser().resolve()
    registry = CorpusRegistry(_resolve_corpus_root(corpus_root))
    scan = scan_path(target, recursive=recursive)
    root = target if target.is_dir() else target.parent

    if dry_run:
        body = scan.to_dict()
        body["corpus_root"] = str(registry.root.resolve())
        body["dry_run"] = True
        return body

    if not scan.supported_files:
        body = scan.to_dict()
        body["corpus_root"] = str(registry.root.resolve())
        body["registry_path"] = str(registry.registry_path)
        body["record_ids"] = []
        body["results"] = []
        body["ingested_count"] = 0
        body["unchanged_count"] = 0
        return body

    _ingest_from_scan(
        scan,
        registry=registry,
        root=root,
        record_id=record_id,
        title=title,
        force=force,
        max_chunk_chars=max_chunk_chars,
        overlap=overlap,
    )

    record_ids = _record_ids_from_ingest_results(scan.ingest_results)
    body = scan.to_dict()
    body.update(
        {
            "path": str(target),
            "ingested_count": scan.files_ingested,
            "unchanged_count": scan.files_unchanged,
            "record_ids": record_ids,
            "results": scan.ingest_results,
            "corpus_root": str(registry.root.resolve()),
            "registry_path": str(registry.registry_path),
        }
    )
    return body


def api_retrieve_evidence(
    *,
    question: str,
    record_id: str | None = None,
    corpus_root: Path | None = None,
    search_all: bool = False,
    record_ids: list[str] | None = None,
    max_chars: int = 12000,
    source_scope: str | None = None,
) -> dict[str, Any]:
    """Retrieve ranked passages only (no LLM, no Lens)."""
    q = question.strip()
    if not q:
        raise ValueError("question must be non-empty")
    registry = CorpusRegistry(_resolve_corpus_root(corpus_root))
    records = registry.load_records()
    if not records:
        raise ValueError("no corpus records — ingest documents first")

    targets: list[str]
    if record_ids:
        targets = sorted({rid.strip() for rid in record_ids if rid.strip() and rid.strip() in records})
        if not targets:
            raise KeyError("no scoped record_ids found in operator corpus")
    elif search_all or not record_id:
        targets = sorted(records)
    else:
        rid = record_id.strip()
        if rid not in records:
            raise KeyError(f"record not found: {rid!r}")
        targets = [rid]

    scoped_targets = _scope_filtered_targets(
        targets=targets,
        records=records,
        request_scope=source_scope,
    )
    evidence: list[dict[str, Any]] = []
    for rid in scoped_targets:
        rec = records[rid]
        chunks = retrieve_chunks(registry, rid, query=q, max_chars=max_chars)
        for ch in chunks:
            evidence.append(
                _chunk_to_evidence(
                    ch,
                    record_title=rec.title,
                    source_path=rec.source_path,
                    query=q,
                )
            )

    evidence.sort(key=lambda row: (-(row["score"] or 0), row["record_id"], row["ordinal"]))
    sections = group_evidence_sections(evidence)
    visible = visible_evidence_for_display(sections, question=q)
    low_hidden = sections.get("low_relevance") or []
    return {
        "question": q,
        "record_ids": scoped_targets,
        "evidence": evidence,
        "evidence_sections": sections,
        "evidence_visible": visible,
        "evidence_count": len(evidence),
        "evidence_visible_count": len(visible),
        "low_relevance_count": len(low_hidden),
        "low_relevance_hidden": len(low_hidden) > 0,
        "low_relevance_warning": LOW_RELEVANCE_WARNING if low_hidden else None,
        "section_titles": SECTION_TITLES,
    }


def _default_chat_fn(
    *,
    base_url: str,
    user_message: str,
    request_metadata: dict[str, object],
    session_id: str,
    bearer: str,
    authorization: str | None = None,
) -> dict[str, Any]:
    if authorization:
        chat_url = f"{base_url.rstrip('/')}/v1/chat/completions"
        body = {
            "model": "openclaw",
            "messages": [{"role": "user", "content": user_message}],
            "stream": False,
            "aurora_session_id": session_id,
            "request_metadata": request_metadata,
        }
        payload = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": authorization,
            "X-Aurora-Operator-Detail": "true",
        }
        req = Request(chat_url, data=payload, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=600.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"chat request failed: {exc}") from exc
    return post_chat(
        base_url=base_url,
        user_message=user_message,
        request_metadata=request_metadata,
        session_id=session_id,
        bearer=bearer,
    )


def _filter_chunks_for_governed_review(
    registry: CorpusRegistry,
    chunks: list[Any],
    *,
    question: str,
    include_low_relevance: bool = False,
) -> list[Any]:
    """Drop low-relevance and out-of-role chunks before governed ask/validate."""
    if not chunks:
        return []
    records = registry.load_records()
    q = question.strip()
    evidence_rows: list[dict[str, Any]] = []
    for ch in chunks:
        rec = records.get(ch.record_id)
        if rec is None:
            continue
        text = ch.text.strip()
        snippet = text if len(text) <= 480 else text[:477] + "…"
        score = chunk_relevance_score(ch, q) if q else None
        role = classify_evidence_role(
            score=score,
            question=q,
            record_id=ch.record_id,
            document_name=rec.title,
            source_path=rec.source_path,
            snippet=snippet,
        )
        evidence_rows.append(
            {
                "chunk_id": ch.chunk_id,
                "score": score,
                "evidence_role": role,
            }
        )
    allowed = {
        row["chunk_id"]
        for row in evidence_for_governed_review(
            evidence_rows,
            question=q,
            include_low_relevance=include_low_relevance,
        )
    }
    return [ch for ch in chunks if ch.chunk_id in allowed]


def _retrieve_chunks_for_question(
    registry: CorpusRegistry,
    *,
    question: str,
    record_id: str | None,
    search_all: bool,
    record_ids: list[str] | None,
    max_chars: int,
    include_low_relevance: bool = False,
    source_scope: str | None = None,
) -> tuple[list[Any], str, list[str]]:
    """Return (chunks, primary_record_id, record_ids_used)."""
    q = question.strip()
    records = registry.load_records()
    if not records:
        raise ValueError("no corpus records — ingest documents first")

    if record_id and not search_all and not record_ids:
        rid = record_id.strip()
        if rid not in records:
            raise KeyError(f"record not found: {rid!r}")
        if not _record_matches_request_scope(
            record_scope=getattr(records.get(rid), "scope", None),
            requested_scope=source_scope,
        ):
            return [], rid, []
        chunks = retrieve_chunks(registry, rid, query=q, max_chars=max_chars)
        chunks = _filter_chunks_for_governed_review(
            registry,
            chunks,
            question=q,
            include_low_relevance=include_low_relevance,
        )
        return chunks, rid, [rid]

    if record_ids:
        targets = sorted({rid.strip() for rid in record_ids if rid.strip() and rid.strip() in records})
        if not targets:
            raise KeyError("no scoped record_ids found in operator corpus")
    elif search_all or not record_id:
        targets = sorted(records)
    else:
        rid = record_id.strip()
        if rid not in records:
            raise KeyError(f"record not found: {rid!r}")
        targets = [rid]

    scoped_targets = _scope_filtered_targets(
        targets=targets,
        records=records,
        request_scope=source_scope,
    )
    if not scoped_targets:
        return [], targets[0] if targets else "", []

    per_budget = max(2000, max_chars // max(1, len(scoped_targets)))
    combined: list[Any] = []
    used: list[str] = []
    for rid in scoped_targets:
        try:
            part = retrieve_chunks(registry, rid, query=q, max_chars=per_budget)
        except (KeyError, ValueError):
            continue
        if part:
            used.append(rid)
            combined.extend(part)
    if not combined:
        return [], scoped_targets[0] if scoped_targets else "", []
    combined = _filter_chunks_for_governed_review(
        registry,
        combined,
        question=q,
        include_low_relevance=include_low_relevance,
    )
    if not combined:
        return [], scoped_targets[0] if scoped_targets else "", []
    combined = select_chunks(combined, max_chars=max_chars, preserve_order=True)
    primary = combined[0].record_id
    used_ids = sorted({c.record_id for c in combined})
    return combined, primary, used_ids


def _first_blocking_evidence_admissibility(
    *,
    registry: CorpusRegistry,
    chunks: list[Any],
    request_scope: str | None,
) -> tuple[str, Any] | None:
    """Return first blocking admissibility result over grouped record chunks."""
    by_record: dict[str, list[Any]] = {}
    for chunk in chunks:
        by_record.setdefault(chunk.record_id, []).append(chunk)
    if not by_record:
        return None

    records = registry.load_records()
    unresolved_hit: tuple[str, Any] | None = None
    for rid in sorted(by_record):
        admissibility = evaluate_retrieved_evidence(
            chunks=by_record[rid],
            record=records.get(rid),
            request_scope=request_scope,
            require_authority=True,
        )
        if admissibility.status == "inadmissible":
            return "evidence inadmissible", admissibility
        if admissibility.status == "unresolved" and unresolved_hit is None:
            unresolved_hit = ("evidence unresolved", admissibility)
    return unresolved_hit


def api_ask_corpus(
    *,
    question: str,
    record_id: str | None = None,
    corpus_root: Path | None = None,
    proxy: str,
    search_all: bool = False,
    record_ids: list[str] | None = None,
    max_chars: int = 12000,
    policy_profile: str = "enterprise_strict",
    source_scope: str = "corpus",
    authorization: str | None = None,
    chat_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Retrieve evidence, then ask through governed proxy (Lens disposes)."""
    registry = CorpusRegistry(_resolve_corpus_root(corpus_root))
    selected, primary_rid, used_ids = _retrieve_chunks_for_question(
        registry,
        question=question,
        record_id=record_id,
        search_all=search_all,
        record_ids=record_ids,
        max_chars=max_chars,
        source_scope=source_scope,
    )
    evidence_payload = api_retrieve_evidence(
        question=question,
        record_id=record_id,
        corpus_root=corpus_root,
        search_all=search_all or (not record_id and not record_ids),
        record_ids=record_ids or used_ids or None,
        max_chars=max_chars,
        source_scope=source_scope,
    )
    if not selected:
        return {
            "question": question,
            "answer": "",
            "governance": None,
            "evidence": evidence_payload.get("evidence") or [],
            "support_status": "not_supported",
            "support_issues": ["no matching evidence retrieved"],
            "flags": [],
            "session_id": None,
            "validation": None,
            "operator_note": "Retrieval found no passages; Lens was not called.",
            "record_id": primary_rid,
            "record_ids": used_ids,
        }

    blocked = _first_blocking_evidence_admissibility(
        registry=registry,
        chunks=selected,
        request_scope=source_scope or None,
    )
    if blocked is not None:
        failure_reason, admissibility = blocked
        support_status = "not_supported" if admissibility.status == "inadmissible" else "partial"
        return {
            "question": question,
            "record_id": primary_rid,
            "record_ids": used_ids,
            "answer": "",
            "governance": None,
            "evidence": evidence_payload.get("evidence_visible") or evidence_payload.get("evidence") or [],
            "evidence_sections": evidence_payload.get("evidence_sections"),
            "low_relevance_warning": evidence_payload.get("low_relevance_warning"),
            "support_status": support_status,
            "support_issues": [failure_reason, str(admissibility.failure_kind or "admissibility_blocked")],
            "flags": [],
            "session_id": None,
            "context_chars": 0,
            "chunks_used": len(selected),
            "validation": None,
            "operator_note": "Evidence admissibility blocked corpus support before Lens call.",
            "evidence_admissibility": {
                "status": admissibility.status,
                "failure_kind": admissibility.failure_kind,
                "record_id": admissibility.record_id,
                "chunk_ids": admissibility.chunk_ids,
                "reasons": admissibility.reasons,
            },
        }

    user_message = assemble_rag_message(selected, question)
    request_metadata = assemble_request_metadata(
        primary_rid,
        source_scope=(source_scope,) if source_scope else (),
        policy_profile=policy_profile,
    )
    if len(used_ids) > 1:
        request_metadata["record_ids"] = used_ids
    session_id = f"corpus-console-{uuid.uuid4().hex[:12]}"
    chat = chat_fn or _default_chat_fn
    resp = chat(
        base_url=proxy,
        user_message=user_message,
        request_metadata=request_metadata,
        session_id=session_id,
        bearer="corpus-console",
        authorization=authorization,
    )
    answer = assistant_text(resp)
    aurora = resp.get("aurora") if isinstance(resp.get("aurora"), dict) else {}
    governance = str(aurora.get("governance") or "?")
    flags = _extract_flags(aurora)
    support_status, support_issues = derive_support_status(
        answer=answer,
        governance=governance,
        flags=flags,
    )
    split = split_rag_context_question(user_message)
    return {
        "question": question,
        "record_id": primary_rid,
        "record_ids": used_ids or evidence_payload.get("record_ids") or [],
        "answer": answer,
        "governance": governance,
        "evidence": evidence_payload.get("evidence_visible") or evidence_payload.get("evidence") or [],
        "evidence_sections": evidence_payload.get("evidence_sections"),
        "low_relevance_warning": evidence_payload.get("low_relevance_warning"),
        "support_status": support_status,
        "support_issues": support_issues,
        "flags": flags,
        "session_id": session_id,
        "context_chars": len(split[0]) if split else 0,
        "chunks_used": len(selected),
        "validation": None,
        "operator_note": "Governed by Lens via proxy; console does not decide admissibility.",
    }


def _validate_review_failed(
    reason: str,
    *,
    question: str = "",
    record_id: str = "",
    detail: Any = "",
) -> dict[str, Any]:
    """Structured failure payload for Review / Validate (never silent no-op)."""
    return {
        "review_outcome": "failed",
        "review_failure_reason": reason,
        "review_failure_detail": detail,
        "question": question,
        "record_id": record_id,
        "answer": "",
        "governance": None,
        "evidence": [],
        "support_status": None,
        "support_issues": [],
        "flags": [],
        "session_id": None,
        "validation": None,
        "operator_note": "Review did not run — fix the reported reason and try again.",
    }


def preflight_validate_review(
    *,
    question: str,
    record_id: str,
    corpus_root: Path | None = None,
) -> dict[str, Any] | None:
    """Return a failed review payload when prerequisites are missing; else None."""
    q = question.strip()
    rid = record_id.strip()
    if not rid:
        return _validate_review_failed("no selected record")
    if not q:
        return _validate_review_failed("no question", record_id=rid)

    registry = CorpusRegistry(_resolve_corpus_root(corpus_root))
    if registry.get_record(rid) is None:
        return _validate_review_failed(
            "no selected record",
            question=q,
            record_id=rid,
            detail=f"Record {rid!r} is not in the operator corpus.",
        )

    try:
        evidence_payload = api_retrieve_evidence(
            question=q,
            record_id=rid,
            corpus_root=corpus_root,
        )
    except ValueError as exc:
        if "no chunks for record" in str(exc):
            return _validate_review_failed(
                "no evidence found",
                question=q,
                record_id=rid,
                detail=str(exc),
            )
        raise
    evidence = evidence_payload.get("evidence") or []
    review_evidence = evidence_for_governed_review(evidence, question=q)
    if not review_evidence:
        return _validate_review_failed(
            "no evidence found",
            question=q,
            record_id=rid,
            detail="Retrieval found no relevant passages for this question and record.",
        )
    selected_chunk_ids = {
        str(row.get("chunk_id", "")).strip()
        for row in review_evidence
        if str(row.get("chunk_id", "")).strip()
    }
    candidate_chunks = retrieve_chunks(registry, rid, query=q, max_chars=12000)
    candidate_chunks = [ch for ch in candidate_chunks if ch.chunk_id in selected_chunk_ids]
    admissibility = evaluate_retrieved_evidence(
        chunks=candidate_chunks,
        record=registry.get_record(rid),
    )
    if admissibility.status == "inadmissible":
        return _validate_review_failed(
            "evidence inadmissible",
            question=q,
            record_id=rid,
            detail={
                "failure_kind": admissibility.failure_kind,
                "reasons": admissibility.reasons,
                "record_id": admissibility.record_id,
                "chunk_ids": admissibility.chunk_ids,
            },
        )
    if admissibility.status == "unresolved":
        return _validate_review_failed(
            "evidence unresolved",
            question=q,
            record_id=rid,
            detail={
                "failure_kind": admissibility.failure_kind,
                "reasons": admissibility.reasons,
                "record_id": admissibility.record_id,
                "chunk_ids": admissibility.chunk_ids,
            },
        )

    upstream = upstream_config_status()
    if not upstream["ok"]:
        return _validate_review_failed(
            str(upstream["reason"] or "no upstream configured"),
            question=q,
            record_id=rid,
            detail=str(upstream["detail"] or ""),
        )
    return None


def api_validate_question(
    *,
    question: str,
    record_id: str,
    corpus_root: Path | None = None,
    proxy: str,
    max_chars: int = 12000,
    policy_profile: str = "enterprise_strict",
    source_scope: str = "corpus",
    authorization: str | None = None,
    chat_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one governed Q&A and score with a synthetic in-context case."""
    failed = preflight_validate_review(
        question=question,
        record_id=record_id,
        corpus_root=corpus_root,
    )
    if failed is not None:
        return failed

    try:
        ask_result = api_ask_corpus(
            question=question,
            record_id=record_id,
            corpus_root=corpus_root,
            proxy=proxy,
            max_chars=max_chars,
            policy_profile=policy_profile,
            source_scope=source_scope,
            authorization=authorization,
            chat_fn=chat_fn,
        )
    except RuntimeError as exc:
        return _validate_review_failed(
            "server error",
            question=question.strip(),
            record_id=record_id.strip(),
            detail=str(exc),
        )
    spec = CaseSpec(
        id="console-validate",
        question=question,
        expect="in_context",
        required_any=[],
        required_min_any=0,
    )
    scored = score_case(
        spec,
        answer=ask_result.get("answer") or "",
        governance=str(ask_result.get("governance") or "?"),
        chunks=int(ask_result.get("chunks_used") or 0),
        context_chars=int(ask_result.get("context_chars") or 0),
        session_id=str(ask_result.get("session_id") or ""),
    )
    validation_reasons = list(scored.reasons)
    support_status, support_issues = derive_support_status(
        answer=ask_result.get("answer") or "",
        governance=str(ask_result.get("governance") or "?"),
        flags=ask_result.get("flags") or [],
        validation_reasons=validation_reasons if not scored.passed else None,
    )
    ask_result["validation"] = {
        "passed": scored.passed,
        "reasons": validation_reasons,
    }
    ask_result["support_status"] = support_status
    ask_result["support_issues"] = support_issues
    ask_result["review_outcome"] = "complete"
    ask_result["review_failure_reason"] = None
    ask_result["review_failure_detail"] = None
    return ask_result


def api_validate_manifest(
    *,
    manifest_path: Path,
    corpus_root: Path | None = None,
    proxy: str,
    case_ids: str | None = None,
    max_chars: int = 12000,
    source_scope: str = "corpus",
    authorization: str | None = None,
    chat_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run manifest-driven validation (same cases as ``aurora-lens corpus validate``)."""
    record_id, policy_profile, cases = load_manifest(manifest_path)
    if case_ids:
        wanted = {x.strip() for x in case_ids.split(",") if x.strip()}
        cases = [c for c in cases if c.id in wanted]
        if not cases:
            raise ValueError(f"no cases matched ids {case_ids!r}")

    registry = CorpusRegistry(_resolve_corpus_root(corpus_root))
    if registry.get_record(record_id) is None:
        raise KeyError(f"record not found: {record_id!r} (run ingest first)")

    chat = chat_fn or _default_chat_fn
    case_results: list[dict[str, Any]] = []
    for spec in cases:
        chunks = retrieve_chunks(registry, record_id, max_chars=max_chars, query=spec.question)
        user_message = assemble_rag_message(chunks, spec.question)
        request_metadata = assemble_request_metadata(
            record_id,
            source_scope=(source_scope,) if source_scope else (),
            policy_profile=policy_profile,
        )
        session_id = f"validate-console-{uuid.uuid4().hex[:12]}"
        resp = chat(
            base_url=proxy,
            user_message=user_message,
            request_metadata=request_metadata,
            session_id=session_id,
            bearer="validate-corpus-console",
            authorization=authorization,
        )
        answer = assistant_text(resp)
        governance = str((resp.get("aurora") or {}).get("governance") or "?")
        scored = score_case(
            spec,
            answer=answer,
            governance=governance,
            chunks=len(chunks),
            context_chars=len(split_rag_context_question(user_message)[0] or ""),
            session_id=session_id,
        )
        flags = _extract_flags(resp.get("aurora") if isinstance(resp.get("aurora"), dict) else {})
        support_status, support_issues = derive_support_status(
            answer=answer,
            governance=governance,
            flags=flags,
            validation_reasons=scored.reasons if not scored.passed else None,
        )
        case_results.append(
            {
                "id": spec.id,
                "question": spec.question,
                "passed": scored.passed,
                "reasons": scored.reasons,
                "governance": governance,
                "answer": answer,
                "support_status": support_status,
                "support_issues": support_issues,
                "flags": flags,
                "session_id": session_id,
            }
        )

    passed = sum(1 for r in case_results if r["passed"])
    return {
        "record_id": record_id,
        "manifest_path": str(manifest_path),
        "total": len(case_results),
        "passed": passed,
        "failed": len(case_results) - passed,
        "cases": case_results,
    }


def _upstream_review_failed(
    reason: str,
    *,
    record_id: str = "",
    detail: str = "",
) -> dict[str, Any]:
    """Structured failure for upstream review (Door 1 — never silent no-op)."""
    return {
        "last_action": "upstream_review",
        "selected_record_id": record_id,
        "upstream_review_state": "failed",
        "upstream_review_failure_reason": reason,
        "upstream_review_failure_detail": detail,
        "record_id": record_id,
        "title": None,
        "provider": None,
        "model": None,
        "question": None,
        "chunks_used": 0,
        "chunks_total": 0,
        "review_text": "",
        "operator_note": "Upstream review did not run — fix the reported reason and try again.",
    }


def preflight_upstream_review(
    *,
    record_id: str,
    corpus_root: Path | None = None,
) -> dict[str, Any] | None:
    """Return a failed upstream review payload when prerequisites are missing; else None."""
    rid = record_id.strip()
    if not rid:
        return _upstream_review_failed("no selected record")

    upstream = upstream_config_status()
    if not upstream["ok"]:
        reason = str(upstream["reason"] or "upstream not configured")
        if reason == "no upstream configured":
            reason = "upstream not configured"
        return _upstream_review_failed(
            reason,
            record_id=rid,
            detail=str(upstream["detail"] or ""),
        )

    registry = CorpusRegistry(_resolve_corpus_root(corpus_root))
    record = registry.get_record(rid)
    if record is None:
        return _upstream_review_failed(
            "no selected record",
            record_id=rid,
            detail=f"Record {rid!r} is not in the operator corpus.",
        )

    all_chunks = registry.load_chunks(rid)
    if not all_chunks:
        return _upstream_review_failed(
            "no evidence found",
            record_id=rid,
            detail=f"No chunks for record {rid!r}.",
        )
    return None


def _upstream_credentials() -> tuple[str, str, str]:
    """Return (provider, api_key, model) or raise ValueError when misconfigured."""
    status = upstream_config_status()
    if not status["ok"]:
        raise ValueError(str(status["detail"] or status["reason"] or "upstream not configured"))
    provider = str(status["provider"])
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(f"Missing API key for provider {provider!r}.")
    return provider, api_key, str(status["model"])


async def api_review_record(
    *,
    record_id: str,
    question: str | None = None,
    corpus_root: Path | None = None,
    max_chars: int | None = 12000,
) -> dict[str, Any]:
    """Direct upstream review (Door 1 — no Lens)."""
    failed = preflight_upstream_review(record_id=record_id, corpus_root=corpus_root)
    if failed is not None:
        return failed

    registry = CorpusRegistry(_resolve_corpus_root(corpus_root))
    record = registry.get_record(record_id)
    if record is None:
        return _upstream_review_failed(
            "no selected record",
            record_id=record_id.strip(),
            detail=f"Record {record_id!r} is not in the operator corpus.",
        )

    all_chunks = registry.load_chunks(record_id)
    if question and question.strip():
        pool = retrieve_chunks(registry, record_id, query=question.strip(), max_chars=max_chars or 12000)
    else:
        pool = select_chunks(all_chunks, max_chars=max_chars)

    try:
        provider, api_key, model = _upstream_credentials()
        answer = await run_corpus_review(
            corpus_text=format_corpus_for_review(pool),
            provider=provider,
            api_key=api_key,
            model=model,
        )
    except Exception as exc:
        return _upstream_review_failed(
            "server error",
            record_id=record_id.strip(),
            detail=str(exc),
        )

    return {
        "last_action": "upstream_review",
        "selected_record_id": record_id,
        "upstream_review_state": "complete",
        "upstream_review_failure_reason": None,
        "upstream_review_failure_detail": None,
        "record_id": record_id,
        "title": record.title,
        "provider": provider,
        "model": model,
        "question": question,
        "chunks_used": len(pool),
        "chunks_total": len(all_chunks),
        "review_text": answer,
        "operator_note": "Direct upstream review — not governed by Lens.",
    }


def api_clear_operator_corpus(*, corpus_root: Path | None = None) -> dict[str, Any]:
    """Remove all records from the operator corpus registry (console scope only)."""
    root = _resolve_corpus_root(corpus_root).resolve()
    allowed = operator_corpus_root().resolve()
    if root != allowed:
        raise ValueError(
            f"clear allowed only for operator corpus root ({allowed}), not {root}"
        )
    registry = CorpusRegistry(root)
    records = registry.load_records()
    cleared = len(records)
    for rid in list(records):
        registry.delete_record(rid)
    records_dir = root / "records"
    if records_dir.is_dir():
        shutil.rmtree(records_dir)
    return {"cleared": cleared, "corpus_root": str(root)}
