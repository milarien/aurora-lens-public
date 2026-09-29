#!/usr/bin/env python3
"""Run Phase 1.7 realistic corpus acceptance (offline or opt-in live mode)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aurora_lens.corpus.document_ingestion import ingest_file
from aurora_lens.corpus.ingest_metadata import IngestMetadata
from aurora_lens.corpus.qa_validation import (
    ABSTAIN_PHRASE_EXACT,
    CaseResult,
    load_manifest,
    results_to_json,
    score_case,
)
from aurora_lens.corpus.registry import CorpusRegistry
from aurora_lens.corpus.retrieve import assemble_rag_message, assemble_request_metadata, retrieve_chunks

PHASE17_FIXTURES = _REPO_ROOT / "eval" / "phase17" / "fixtures"
PHASE17_INVENTORY = PHASE17_FIXTURES / "source_inventory.yaml"
PHASE17_OFFLINE_MANIFEST = _REPO_ROOT / "eval" / "phase17" / "manifests" / "phase17_offline.manifest.yaml"
PHASE17_LIVE_MANIFEST = _REPO_ROOT / "eval" / "phase17" / "manifests" / "phase17_live.manifest.yaml"
DEFAULT_OUT_ROOT = _REPO_ROOT / "build" / "phase17"
DEFAULT_PROXY = "http://localhost:8081"

REQUIRED_INVENTORY_FIELDS = (
    "fixture_id",
    "filename",
    "source_type",
    "source_title",
    "originating_organisation",
    "origin_url",
    "licence",
    "redistribution_status",
    "transformation",
    "excerpt_boundaries",
    "sha256_digest",
    "retrieved_or_created_date",
    "notes",
)
REQUIRED_NONEMPTY_FIELDS = tuple(field for field in REQUIRED_INVENTORY_FIELDS if field != "origin_url")
ALLOWED_SOURCE_TYPES = {"public_domain", "permissive", "synthetic_derivative"}
SENSITIVE_KEYS = {"authorization", "api_key", "x-api-key", "token", "secret"}


@dataclass
class FixtureIngestResult:
    fixture_id: str
    record_id: str
    filename: str
    chunk_count: int
    status: str
    authority: str
    scope: str
    ingest_success: bool = True
    ingest_error: str = ""


@dataclass
class OwnedProxy:
    process: subprocess.Popen[str]
    port: int
    proxy_url: str
    pid_file: Path
    log_file: Path


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    payload = path.read_bytes()
    if path.suffix.lower() in {".md", ".txt", ".yaml", ".yml", ".json"}:
        # Normalize text line endings so fixture digests stay stable across
        # Windows (CRLF) and Linux/macOS (LF) checkouts.
        payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_health(url: str, *, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.4)
    raise TimeoutError(f"Timed out waiting for proxy health at {url}")


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            process.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            pass
        return
    process.terminate()
    try:
        process.wait(timeout=8.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _start_owned_proxy(
    *,
    run_dir: Path,
    provider: str,
    model: str,
    startup_timeout_s: float = 45.0,
    command_override: list[str] | None = None,
) -> OwnedProxy:
    port = _find_free_port()
    proxy_url = f"http://127.0.0.1:{port}"
    pid_file = run_dir / "phase17-live-proxy.pid"
    log_file = run_dir / "phase17-live-proxy.log"
    command = command_override or [
        sys.executable,
        str(_REPO_ROOT / "run_aurora_lens.py"),
        "--port",
        str(port),
        "--provider",
        provider,
        "--model",
        model,
    ]
    log_handle = log_file.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=_REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pid_file.write_text(str(process.pid), encoding="utf-8")
    try:
        _wait_for_health(f"{proxy_url}/health", timeout_s=startup_timeout_s)
    except Exception:
        _terminate_process_tree(process)
        if pid_file.exists():
            pid_file.unlink()
        log_handle.close()
        raise
    log_handle.close()
    return OwnedProxy(
        process=process,
        port=port,
        proxy_url=proxy_url,
        pid_file=pid_file,
        log_file=log_file,
    )


def _cleanup_owned_proxy(owned: OwnedProxy | None) -> dict[str, Any]:
    if owned is None:
        return {"owned_proxy": False, "cleaned": True, "port_listening_after_cleanup": False, "pid_file_removed": True}
    process = owned.process
    _terminate_process_tree(process)
    pid_file_removed = True
    if owned.pid_file.exists():
        owned.pid_file.unlink()
        pid_file_removed = True
    port_listening = _port_listening(owned.port)
    if port_listening:
        deadline = time.time() + 5.0
        while time.time() < deadline and port_listening:
            time.sleep(0.25)
            port_listening = _port_listening(owned.port)
    return {
        "owned_proxy": True,
        "cleaned": True,
        "port": owned.port,
        "pid": process.pid,
        "pid_file_removed": pid_file_removed,
        "port_listening_after_cleanup": port_listening,
        "log_file": str(owned.log_file),
    }


def redact_mapping(payload: Any) -> Any:
    if isinstance(payload, dict):
        redacted: dict[str, Any] = {}
        for key, value in payload.items():
            if key.lower() in SENSITIVE_KEYS:
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = redact_mapping(value)
        return redacted
    if isinstance(payload, list):
        return [redact_mapping(x) for x in payload]
    return payload


def load_source_inventory(path: Path, fixtures_dir: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("source inventory must be a mapping")
    entries = raw.get("fixtures") or []
    if not isinstance(entries, list) or not entries:
        raise ValueError("source inventory fixtures list is missing or empty")
    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("source inventory entries must be mappings")
        missing_keys = [field for field in REQUIRED_INVENTORY_FIELDS if field not in entry]
        if missing_keys:
            raise ValueError(f"source inventory entry missing required keys: {missing_keys}")
        missing = [field for field in REQUIRED_NONEMPTY_FIELDS if not str(entry.get(field) or "").strip()]
        if missing:
            raise ValueError(f"source inventory entry missing required fields: {missing}")
        source_type = str(entry.get("source_type")).strip()
        if source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"invalid source_type {source_type!r} (allowed: {sorted(ALLOWED_SOURCE_TYPES)})")
        filename = str(entry.get("filename")).strip()
        fixture_path = fixtures_dir / filename
        if not fixture_path.is_file():
            raise ValueError(f"fixture file missing for inventory entry: {fixture_path}")
        observed_digest = _sha256_file(fixture_path)
        expected_digest = str(entry.get("sha256_digest") or "").strip().lower()
        if expected_digest != observed_digest:
            raise ValueError(
                f"fixture sha256 mismatch for {filename}: expected {expected_digest!r}, observed {observed_digest!r}"
            )
        if source_type in {"public_domain", "permissive"}:
            if not str(entry.get("origin_url") or "").strip():
                raise ValueError(f"real fixture {filename} is missing origin_url")
            if not str(entry.get("excerpt_boundaries") or "").strip():
                raise ValueError(f"real fixture {filename} is missing excerpt boundaries")
        out.append(dict(entry))
    return out


def _metadata_for_fixture(entry: dict[str, Any]) -> IngestMetadata:
    fixture_id = str(entry["fixture_id"])
    record_id = str(entry.get("record_id") or f"phase17-{fixture_id}")
    status = "active"
    effective_from = "2025-01-01"
    effective_to = "2030-12-31"
    if fixture_id == "policy_v1":
        status = "superseded"
        effective_from = "2024-01-01"
        effective_to = "2025-06-30"
    elif fixture_id == "policy_v2_superseding":
        status = "approved"
        effective_from = "2025-07-01"
        effective_to = "2030-12-31"
    return IngestMetadata.from_dict(
        {
            "record_id": record_id,
            "status": status,
            "authority": "corporate_policy",
            "scope": "enterprise",
            "effective_from": effective_from,
            "effective_to": effective_to,
            "version": "v2" if fixture_id == "policy_v2_superseding" else "v1",
            "source_ref": str(entry.get("origin_url") or f"phase17://{fixture_id}"),
        }
    )


def ingest_phase17_fixtures(
    inventory: list[dict[str, Any]],
    fixtures_dir: Path,
    corpus_root: Path,
) -> tuple[CorpusRegistry, list[FixtureIngestResult]]:
    registry = CorpusRegistry(corpus_root)
    results: list[FixtureIngestResult] = []
    for entry in inventory:
        record_id = str(entry.get("record_id") or f"phase17-{entry['fixture_id']}")
        source = fixtures_dir / str(entry["filename"])
        metadata = _metadata_for_fixture(entry)
        ingest_expectation = str(entry.get("ingest_expectation") or "success").strip().lower()
        expected_error_contains = str(entry.get("expected_error_contains") or "").strip()
        try:
            record, chunks, _ = ingest_file(
                registry,
                record_id=record_id,
                source_path=source,
                title=str(entry.get("source_title") or record_id),
                ingest_metadata=metadata,
                force=True,
                max_chunk_chars=1400,
                overlap=150,
            )
            if ingest_expectation == "fail":
                raise ValueError(
                    f"fixture {entry['fixture_id']} expected ingest failure but ingested successfully"
                )
            results.append(
                FixtureIngestResult(
                    fixture_id=str(entry["fixture_id"]),
                    record_id=record.record_id,
                    filename=str(entry["filename"]),
                    chunk_count=len(chunks),
                    status=str(record.status),
                    authority=str(record.authority_class),
                    scope=str(record.scope),
                    ingest_success=True,
                    ingest_error="",
                )
            )
        except Exception as exc:
            if ingest_expectation != "fail":
                raise
            err = str(exc)
            if expected_error_contains and expected_error_contains not in err:
                raise ValueError(
                    f"fixture {entry['fixture_id']} ingest error mismatch: expected substring "
                    f"{expected_error_contains!r}, observed {err!r}"
                ) from exc
            results.append(
                FixtureIngestResult(
                    fixture_id=str(entry["fixture_id"]),
                    record_id=record_id,
                    filename=str(entry["filename"]),
                    chunk_count=0,
                    status="ingest_failed_expected",
                    authority=str(metadata.authority),
                    scope=str(metadata.scope),
                    ingest_success=False,
                    ingest_error=err,
                )
            )
    return registry, results


def _retrieve_case_chunks(
    registry: CorpusRegistry,
    record_ids: list[str],
    question: str,
    max_chars: int,
) -> dict[str, list[Any]]:
    chunks_by_record: dict[str, list[Any]] = {}
    for rid in record_ids:
        try:
            chunks_by_record[rid] = retrieve_chunks(
                registry,
                rid,
                max_chars=max_chars,
                query=question,
            )
        except Exception:
            chunks_by_record[rid] = []
    return chunks_by_record


def _offline_answer_from_chunks(chunks: list[Any]) -> str:
    if not chunks:
        return ABSTAIN_PHRASE_EXACT
    joined = "\n\n".join(chunk.text for chunk in chunks)
    return joined[:3000]


def _extract_llm_called(aurora_payload: dict[str, Any]) -> bool | None:
    if "llm_called" in aurora_payload:
        val = aurora_payload.get("llm_called")
        if isinstance(val, bool):
            return val
    receipt = aurora_payload.get("audit_receipt") or {}
    if isinstance(receipt, dict) and isinstance(receipt.get("llm_called"), bool):
        return bool(receipt.get("llm_called"))
    return None


def _run_live_case(
    *,
    proxy: str,
    model: str,
    user_message: str,
    request_metadata: dict[str, object],
) -> dict[str, Any]:
    chat_url = f"{proxy.rstrip('/')}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user_message}],
        "stream": False,
        "request_metadata": request_metadata,
    }
    response = httpx.post(
        chat_url,
        json=body,
        timeout=90.0,
        headers={"authorization": "Bearer phase17-acceptance"},
    )
    response.raise_for_status()
    return response.json()


def run_phase17(
    *,
    mode: str,
    manifest_path: Path,
    proxy: str,
    provider: str,
    model: str | None,
    max_chars: int,
    out_root: Path,
) -> tuple[int, Path, Path]:
    if mode not in {"offline", "live"}:
        raise ValueError("mode must be offline or live")
    if mode == "live" and not model:
        raise ValueError("live mode requires --model")
    if not manifest_path.is_absolute():
        manifest_path = (_REPO_ROOT / manifest_path).resolve()

    inventory = load_source_inventory(PHASE17_INVENTORY, PHASE17_FIXTURES)
    record_id_default, policy_profile, cases = load_manifest(manifest_path)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / f"{run_id}-{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    corpus_root = run_dir / "corpus"
    registry, ingested = ingest_phase17_fixtures(inventory, PHASE17_FIXTURES, corpus_root)

    owned_proxy: OwnedProxy | None = None
    cleanup_state: dict[str, Any] = {"owned_proxy": False}
    proxy_url = proxy.rstrip("/")
    if mode == "live":
        if provider == "openai" and not (
            os.environ.get("OPENAI_API_KEY") or os.environ.get("AURORA_LENS_UPSTREAM_API_KEY")
        ):
            raise RuntimeError("OPENAI_API_KEY (or AURORA_LENS_UPSTREAM_API_KEY) is required for live mode.")
        if provider == "anthropic" and not (
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("AURORA_LENS_UPSTREAM_API_KEY")
        ):
            raise RuntimeError("ANTHROPIC_API_KEY (or AURORA_LENS_UPSTREAM_API_KEY) is required for live mode.")
        owned_proxy = _start_owned_proxy(
            run_dir=run_dir,
            provider=provider,
            model=str(model),
        )
        proxy_url = owned_proxy.proxy_url

    results: list[CaseResult] = []
    case_rows: list[dict[str, Any]] = []
    try:
        for spec in cases:
            case_record_ids = spec.record_ids or [record_id_default]
            chunks_by_record = _retrieve_case_chunks(
                registry=registry,
                record_ids=case_record_ids,
                question=spec.question,
                max_chars=max_chars,
            )
            retrieved_record_ids = [rid for rid, chunks in chunks_by_record.items() if chunks]
            chosen_record_ids = list(retrieved_record_ids)
            if spec.conflict_governing_record_id and spec.conflict_governing_record_id in chunks_by_record:
                if chunks_by_record[spec.conflict_governing_record_id]:
                    chosen_record_ids = [spec.conflict_governing_record_id]
            chosen_chunks: list[Any] = []
            for rid in chosen_record_ids:
                chosen_chunks.extend(chunks_by_record.get(rid, []))
            chosen_chunks.sort(key=lambda c: c.ordinal)
            chosen_chunks = chosen_chunks[:8]
            context_chars = sum(len(c.text) for c in chosen_chunks)
            chunk_locators = [str(c.locator) for c in chosen_chunks]

            if mode == "offline":
                if spec.expect == "absent" and spec.allow_conservative_abstain:
                    answer = ABSTAIN_PHRASE_EXACT
                else:
                    answer = _offline_answer_from_chunks(chosen_chunks)
                governance = "N/A"
                llm_called = None
            else:
                if not chosen_chunks:
                    answer = ABSTAIN_PHRASE_EXACT
                    governance = "CONTAIN"
                    llm_called = False
                else:
                    user_message = assemble_rag_message(chosen_chunks, spec.question)
                    request_metadata = assemble_request_metadata(
                        record_id_default,
                        source_scope=("corpus",),
                        policy_profile=policy_profile,
                    )
                    request_metadata["record_ids"] = case_record_ids
                    live_resp = _run_live_case(
                        proxy=proxy_url,
                        model=str(model),
                        user_message=user_message,
                        request_metadata=request_metadata,
                    )
                    choices = live_resp.get("choices") or []
                    answer = ""
                    if choices:
                        answer = str((((choices[0] or {}).get("message") or {}).get("content") or "")).strip()
                    aurora_payload = live_resp.get("aurora") or {}
                    governance = str(aurora_payload.get("governance") or "?").strip().upper()
                    llm_called = _extract_llm_called(aurora_payload)

            scored = score_case(
                spec,
                answer=answer,
                governance=governance,
                llm_called=llm_called,
                cited_record_ids=chosen_record_ids,
                cited_chunk_locators=chunk_locators,
                chunks=len(chosen_chunks),
                context_chars=context_chars,
            )
            if spec.conflict_superseded_record_ids:
                missing_superseded_visibility = [
                    rid for rid in spec.conflict_superseded_record_ids if rid not in retrieved_record_ids
                ]
                if missing_superseded_visibility:
                    scored.passed = False
                    scored.reasons.append(
                        f"supersession visibility failed: superseded records not retrievable {missing_superseded_visibility!r}"
                    )
            results.append(scored)
            case_rows.append(
                {
                    "id": spec.id,
                    "question": spec.question,
                    "expect": spec.expect,
                    "expected_governance_any": spec.expected_governance_any,
                    "observed_governance": governance,
                    "retrieved_record_ids": retrieved_record_ids,
                    "cited_record_ids": chosen_record_ids,
                    "chunk_locators": chunk_locators,
                    "llm_called": llm_called,
                    "passed": scored.passed,
                    "reasons": scored.reasons,
                }
            )
    finally:
        cleanup_state = _cleanup_owned_proxy(owned_proxy)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    summary = {
        "commit_sha": _git_sha(),
        "run_timestamp_utc": _now_utc(),
        "mode": mode,
        "provider": provider if mode == "live" else "",
        "model": model if mode == "live" else "",
        "manifest_path": str(manifest_path.relative_to(_REPO_ROOT)),
        "manifest_sha256": _manifest_sha256(manifest_path),
        "fixture_inventory": [asdict(x) for x in ingested],
        "fixture_inventory_sources": inventory,
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "skipped": 0,
        "ambiguous": 0,
        "cases": case_rows,
        "lifecycle_cleanup": cleanup_state,
    }
    json_report = run_dir / "phase17-report.json"
    json_report.write_text(
        json.dumps(redact_mapping(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    licence_counts: dict[str, int] = {}
    for entry in inventory:
        label = str(entry.get("source_type") or "unknown")
        licence_counts[label] = licence_counts.get(label, 0) + 1

    md_lines = [
        "# Phase 1.7 Acceptance Report",
        "",
        f"- Commit SHA: `{summary['commit_sha']}`",
        f"- Run timestamp (UTC): {summary['run_timestamp_utc']}",
        f"- Mode: `{mode}`",
        f"- Provider: `{summary['provider']}`",
        f"- Model: `{summary['model']}`",
        f"- Manifest: `{summary['manifest_path']}`",
        f"- Manifest SHA-256: `{summary['manifest_sha256']}`",
        f"- Totals: passed={passed}, failed={failed}, skipped=0, ambiguous=0",
        f"- Source types: {licence_counts}",
        f"- Cleanup: {cleanup_state}",
        "",
        "## Fixture inventory and licence summary",
        "",
        "| fixture_id | record_id | status | source_type | licence |",
        "| --- | --- | --- | --- | --- |",
    ]
    ingest_map = {row.fixture_id: row for row in ingested}
    for source_entry in inventory:
        fixture_id = str(source_entry.get("fixture_id") or "")
        row = ingest_map.get(fixture_id)
        status = row.status if row else "not_ingested"
        md_lines.append(
            f"| {fixture_id} | {source_entry.get('record_id', '')} | {status} | "
            f"{source_entry.get('source_type', '')} | {source_entry.get('licence', '')} |"
        )
    md_lines.extend(["", "## Case results", ""])
    for row in case_rows:
        md_lines.extend(
            [
                f"### {row['id']} — {'PASS' if row['passed'] else 'FAIL'}",
                f"- Question: {row['question']}",
                f"- Expected governance: {row['expected_governance_any'] or '[not constrained]'}",
                f"- Observed governance: `{row['observed_governance']}`",
                f"- Retrieved record IDs: {row['retrieved_record_ids']}",
                f"- Cited record IDs: {row['cited_record_ids']}",
                f"- Chunk locators: {row['chunk_locators']}",
                f"- Provider called (`llm_called`): {row['llm_called']}",
            ]
        )
        if row["reasons"]:
            md_lines.append("- Failures:")
            md_lines.extend([f"  - {reason}" for reason in row["reasons"]])
        md_lines.append("")
    markdown_report = run_dir / "phase17-report.md"
    markdown_report.write_text("\n".join(md_lines), encoding="utf-8")
    return (0 if failed == 0 else 1), markdown_report, json_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1.7 realistic corpus acceptance.")
    parser.add_argument("--mode", choices=("offline", "live"), required=True)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--proxy", default=DEFAULT_PROXY)
    parser.add_argument("--max-chars", type=int, default=10000)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    manifest = args.manifest
    if manifest is None:
        manifest = PHASE17_OFFLINE_MANIFEST if args.mode == "offline" else PHASE17_LIVE_MANIFEST
    if not manifest.is_absolute():
        manifest = (_REPO_ROOT / manifest).resolve()
    code, md_report, json_report = run_phase17(
        mode=args.mode,
        manifest_path=manifest,
        proxy=args.proxy,
        provider=args.provider,
        model=args.model,
        max_chars=args.max_chars,
        out_root=args.out_root,
    )
    print(f"Report (md):   {md_report}")
    print(f"Report (json): {json_report}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
