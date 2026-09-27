"""Corpus product CLI support — shared operator functions for ``aurora-lens corpus``.

Retrieval proposes; Lens disposes. No PEF mutation or admissibility decisions here.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aurora_lens.corpus.document_ingestion import ingest_file
from aurora_lens.corpus.qa_validation import (
    load_manifest,
    render_evidence_markdown,
    results_to_json,
    score_case,
)
from aurora_lens.corpus.registry import CorpusRegistry, default_corpus_root
from aurora_lens.corpus.retrieve import assemble_rag_message, assemble_request_metadata, retrieve_chunks
from aurora_lens.corpus.review import (
    format_corpus_for_review,
    resolve_upstream,
    run_corpus_review,
    safe_print,
    select_chunks,
)
from aurora_lens.lens import split_rag_context_question

DEFAULT_PROXY = "http://localhost:8081"
DEFAULT_MODEL = "openclaw"
DEFAULT_TIMEOUT = 600.0


def parse_chunk_ordinals(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    return {int(p) for p in parts}


def check_proxy(base_url: str) -> None:
    health_url = f"{base_url.rstrip('/')}/health"
    try:
        with urlopen(health_url, timeout=5) as resp:
            health = json.loads(resp.read().decode("utf-8"))
    except (OSError, URLError) as exc:
        raise SystemExit(
            f"Proxy not reachable at {base_url}. Start with: aurora-lens proxy\n{exc}"
        ) from exc
    if health.get("status") not in ("ok", "degraded"):
        print(f"WARNING: /health status={health.get('status')!r}", file=__import__("sys").stderr)


def post_chat(
    *,
    base_url: str,
    user_message: str,
    request_metadata: dict[str, object],
    session_id: str,
    bearer: str = "corpus-cli",
) -> dict[str, Any]:
    chat_url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": user_message}],
        "stream": False,
        "aurora_session_id": session_id,
        "request_metadata": request_metadata,
    }
    payload = json.dumps(body).encode("utf-8")
    req = Request(
        chat_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer}",
        },
        method="POST",
    )
    with urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def assistant_text(resp: dict[str, Any]) -> str:
    choices = resp.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    return content.strip() if isinstance(content, str) else ""


def cmd_ingest(
    *,
    record_id: str,
    file_path: Path,
    corpus_root: Path | None = None,
    title: str | None = None,
    force: bool = False,
    max_chunk_chars: int = 4000,
    overlap: int = 200,
) -> int:
    registry = CorpusRegistry(corpus_root)
    record, chunks, did_write = ingest_file(
        registry,
        record_id=record_id,
        source_path=file_path,
        title=title,
        force=force,
        max_chunk_chars=max_chunk_chars,
        overlap=overlap,
    )
    total_chars = sum(len(c.text) for c in chunks)
    status = "ingested" if did_write else "unchanged (sha256 match; use --force)"
    print("Corpus ingest")
    print(f"  status:      {status}")
    print(f"  record_id:   {record.record_id}")
    print(f"  title:       {record.title}")
    print(f"  source:      {record.source_path}")
    print(f"  sha256:      {record.sha256[:16]}…")
    print(f"  chunks:      {record.chunk_count}")
    print(f"  total_chars: {total_chars:,}")
    print(f"  registry:    {registry.registry_path}")
    return 0


def cmd_list(*, corpus_root: Path | None = None) -> int:
    registry = CorpusRegistry(corpus_root)
    records = registry.load_records()
    if not records:
        print("No corpus records. Ingest with: aurora-lens corpus ingest …")
        return 0
    print(f"Corpus records ({len(records)})")
    for rid in sorted(records):
        rec = records[rid]
        print(f"  {rec.record_id}: {rec.title} ({rec.chunk_count} chunks)")
    return 0


def cmd_ask(
    *,
    record_id: str,
    question: str,
    corpus_root: Path | None = None,
    proxy: str = DEFAULT_PROXY,
    chunks: str | None = None,
    max_chars: int = 12000,
    policy_profile: str = "enterprise_strict",
    source_scope: str = "corpus",
) -> int:
    check_proxy(proxy)
    registry = CorpusRegistry(corpus_root)
    try:
        selected = retrieve_chunks(
            registry,
            record_id,
            chunk_ordinals=parse_chunk_ordinals(chunks),
            max_chars=max_chars,
            query=question,
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    user_message = assemble_rag_message(selected, question)
    request_metadata = assemble_request_metadata(
        record_id,
        source_scope=(source_scope,) if source_scope else (),
        policy_profile=policy_profile,
    )
    session_id = f"corpus-ask-{uuid.uuid4().hex[:12]}"
    split = split_rag_context_question(user_message)
    context_chars = len(split[0]) if split else 0

    print("Corpus Q&A (governed proxy)")
    print(f"  proxy:         {proxy.rstrip('/')}")
    print(f"  record_id:     {record_id}")
    print(f"  chunks:        {len(selected)}")
    print(f"  context_chars: {context_chars:,}")
    print(f"  question:      {question}")
    print()

    try:
        resp = post_chat(
            base_url=proxy,
            user_message=user_message,
            request_metadata=request_metadata,
            session_id=session_id,
            bearer="corpus-ask",
        )
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}\n{detail}") from exc
    except URLError as exc:
        raise SystemExit(f"Request failed: {exc}") from exc

    answer = assistant_text(resp)
    governance = (resp.get("aurora") or {}).get("governance", "?")
    safe_print("--- answer ---\n")
    safe_print(answer or f"[no content] governance={governance!r}")
    safe_print(f"\n--- end (governance={governance!r}) ---")
    return 0


def cmd_validate(
    *,
    manifest_path: Path,
    corpus_root: Path | None = None,
    proxy: str = DEFAULT_PROXY,
    max_chars: int = 12000,
    source_scope: str = "corpus",
    case_ids: str | None = None,
    write_evidence: Path | None = None,
    write_json: Path | None = None,
) -> int:
    record_id, policy_profile, cases = load_manifest(manifest_path)
    if case_ids:
        wanted = {x.strip() for x in case_ids.split(",") if x.strip()}
        cases = [c for c in cases if c.id in wanted]
        if not cases:
            raise SystemExit(f"no cases matched ids {case_ids!r}")

    check_proxy(proxy)
    registry = CorpusRegistry(corpus_root)
    if registry.get_record(record_id) is None:
        raise SystemExit(f"record not found: {record_id!r} (run ingest first)")

    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = []
    print(f"Corpus Q&A validation  record={record_id}  cases={len(cases)}")
    failures = 0

    for spec in cases:
        chunks = retrieve_chunks(
            registry,
            record_id,
            max_chars=max_chars,
            query=spec.question,
        )
        user_message = assemble_rag_message(chunks, spec.question)
        request_metadata = assemble_request_metadata(
            record_id,
            source_scope=(source_scope,) if source_scope else (),
            policy_profile=policy_profile,
        )
        session_id = f"validate-corpus-{uuid.uuid4().hex[:12]}"
        resp = post_chat(
            base_url=proxy,
            user_message=user_message,
            request_metadata=request_metadata,
            session_id=session_id,
            bearer="validate-corpus",
        )
        answer = assistant_text(resp)
        governance = str((resp.get("aurora") or {}).get("governance") or "?")
        split = split_rag_context_question(user_message)
        context_chars = len(split[0]) if split else 0
        scored = score_case(
            spec,
            answer=answer,
            governance=governance,
            chunks=len(chunks),
            context_chars=context_chars,
            session_id=session_id,
        )
        results.append(scored)
        mark = "PASS" if scored.passed else "FAIL"
        if not scored.passed:
            failures += 1
        print(f"  [{mark}] {spec.id}")
        for reason in scored.reasons:
            print(f"     - {reason}")

    if write_evidence:
        write_evidence.parent.mkdir(parents=True, exist_ok=True)
        write_evidence.write_text(
            render_evidence_markdown(
                results=results,
                record_id=record_id,
                proxy=proxy,
                run_at=run_at,
                manifest_path=manifest_path,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote evidence: {write_evidence}")
    if write_json:
        write_json.parent.mkdir(parents=True, exist_ok=True)
        write_json.write_text(
            json.dumps(
                {
                    "run_at": run_at,
                    "record_id": record_id,
                    "proxy": proxy,
                    "passed": sum(1 for r in results if r.passed),
                    "total": len(results),
                    "results": results_to_json(results),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Wrote JSON: {write_json}")

    return 1 if failures else 0


def cmd_review(
    *,
    record_id: str,
    corpus_root: Path | None = None,
    chunks: str | None = None,
    max_chars: int | None = None,
) -> int:
    registry = CorpusRegistry(corpus_root)
    record = registry.get_record(record_id)
    if record is None:
        raise SystemExit(f"Record not found: {record_id!r}. Ingest first.")

    all_chunks = registry.load_chunks(record_id)
    if not all_chunks:
        raise SystemExit(f"No chunks for record {record_id!r}")

    selected = select_chunks(
        all_chunks,
        chunk_ordinals=parse_chunk_ordinals(chunks),
        max_chars=max_chars,
    )
    corpus_text = format_corpus_for_review(selected)
    provider, api_key, model = resolve_upstream()

    print("Corpus review (direct upstream — no Lens)")
    print(f"  record_id:    {record.record_id}")
    print(f"  chunks:       {len(selected)} / {len(all_chunks)}")
    print(f"  provider:     {provider}")
    print()

    answer = asyncio.run(
        run_corpus_review(
            corpus_text,
            provider=provider,
            api_key=api_key,
            model=model,
        )
    )
    safe_print(answer)
    return 0


class CorpusProduct:
    """Stable operator surface for corpus/RAG (see docs/CORPUS_PRODUCT_UX.md)."""

    ingest = staticmethod(cmd_ingest)
    list_records = staticmethod(cmd_list)
    ask = staticmethod(cmd_ask)
    validate = staticmethod(cmd_validate)
    review = staticmethod(cmd_review)
