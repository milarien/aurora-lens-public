"""aurora-lens -- Batch governance runner.

Reads a JSONL file of user turns. Runs each through the full pipeline:

    SpacyBackend (extraction) -> Checker (verification) -> Governor (CanonicalScannerGateBridge)

Writes per-turn governance decisions to an output JSONL file.
Optionally writes a tamper-evident forensic audit JSONL alongside.

INPUT FORMAT  (one JSON object per line):

    {"user": "What is the capital of France?"}
    {"user": "Alice works at Acme Corp.", "session_id": "s1"}
    {"user": "Does Alice work at Google?",  "session_id": "s1",
     "expected_action": "HARD_STOP"}

Fields:
    user             (required)  User turn text.
    session_id       (optional)  Records sharing an ID share PEF state and
                                 conversation history.  Records without an ID
                                 are each an independent single-turn session.
    expected_action  (optional)  PASS, HARD_STOP, FORCE_REVISE, SOFT_CORRECT,
                                 CONTAIN -- used for pass/fail reporting only.
    expected_pathway (optional)  ContinuationPathway value for pass/fail.

OUTPUT FORMAT  (one JSON object per line):

    {"session_id": "s1", "turn": 1, "user": "...",
     "action": "HARD_STOP",
     "pathway_id": "P_STOP_REDIRECT_QUALIFIED",
     "flags": ["PERSONALIZED_MEDICAL_ADVICE"],
     "governed_response": "I am not able to provide ...",
     "original_response": "You should take ...",
     "forensic_event": {...},
     "expected_action": "HARD_STOP", "expected_hit": true}

USAGE:

    aurora-lens batch --input scenarios.jsonl
    aurora-lens batch --input scenarios.jsonl --output results.jsonl
    aurora-lens batch --input scenarios.jsonl --audit batch_audit.jsonl
    aurora-lens batch --input scenarios.jsonl --model claude-sonnet-4-5-20250929
    aurora-lens batch --input -                      # read from stdin
    aurora-lens batch --input scenarios.jsonl --quiet  # no per-turn stderr lines

Requires OPENAI_API_KEY or ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from aurora_lens.adapters.base import LLMAdapter
from aurora_lens.config import LensConfig
from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
from aurora_lens.lens import Lens


# ---------------------------------------------------------------------------
# Adapter builder
# ---------------------------------------------------------------------------

def _build_adapter(model: str) -> tuple[LLMAdapter | None, str | None]:
    anth_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()

    if anth_key and not anth_key.startswith("${"):
        try:
            from aurora_lens.adapters.claude import ClaudeAdapter
            m = model if model and model.startswith("claude-") else "claude-sonnet-4-5-20250929"
            return ClaudeAdapter(api_key=anth_key, model=m), None
        except ImportError:
            return None, "pip install aurora-lens[claude]"

    if openai_key and not openai_key.startswith("${"):
        try:
            from aurora_lens.adapters.openai import OpenAIUpstreamAdapter
            return OpenAIUpstreamAdapter(
                api_key=openai_key,
                model=model,
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            ), None
        except ImportError:
            return None, "httpx required (included in base)"

    return None, "Set OPENAI_API_KEY or ANTHROPIC_API_KEY."


# ---------------------------------------------------------------------------
# SpacyBackend loader
# ---------------------------------------------------------------------------

def _load_spacy_backend():
    try:
        from aurora_lens.interpret.spacy_backend import SpacyBackend
        return SpacyBackend(model="en_core_web_sm")
    except ImportError as exc:
        raise ImportError(
            "spaCy required. "
            "pip install aurora-lens[spacy] && python -m spacy download en_core_web_sm"
        ) from exc


# ---------------------------------------------------------------------------
# Lens factory
# ---------------------------------------------------------------------------

def _build_lens(
    session_id: str,
    adapter: LLMAdapter,
    spacy_backend,
    bridge: CanonicalScannerGateBridge,
) -> Lens:
    config = LensConfig(
        adapter=adapter,
        extraction_backend=spacy_backend,
        governance_bridge=bridge,
        auto_interpret=True,
        auto_verify=True,
        include_operator_detail=True,
    )
    return Lens(config=config, session_id=session_id)


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def _build_result(session_id: str, record: dict, result) -> dict[str, Any]:
    decision = result.decision
    pathway_id: str | None = getattr(decision, "pathway_id", None) if decision else None
    forensic_event: dict | None = (
        getattr(decision, "forensic_event", None) if decision else None
    )
    governed_response: str | None = (
        getattr(decision, "governed_response", None) if decision else None
    )

    out: dict[str, Any] = {
        "session_id": session_id,
        "turn": result.turn,
        "user": record.get("user", ""),
        "action": result.action.name,
        "pathway_id": pathway_id,
        "flags": [f.flag_type.name for f in (result.flags or [])],
        "governed_response": governed_response or result.response,
    }

    out["epistemic_normalisation_applied"] = bool(
        getattr(result, "epistemic_normalisation_applied", False)
    )
    _upstream = getattr(result, "upstream_model_draft", None)
    if _upstream is not None:
        out["upstream_model_draft"] = _upstream

    if result.original_response:
        out["original_response"] = result.original_response

    if forensic_event:
        out["forensic_event"] = forensic_event

    expected_action = record.get("expected_action")
    expected_pathway = record.get("expected_pathway")
    if expected_action is not None:
        out["expected_action"] = expected_action
        hit = result.action.name == expected_action
        if expected_pathway is not None:
            out["expected_pathway"] = expected_pathway
            hit = hit and (pathway_id == expected_pathway)
        out["expected_hit"] = hit

    return out


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

async def run_batch(
    records: list[dict],
    *,
    adapter: LLMAdapter,
    audit_path: str | None,
    governance_mode: str,
    progress: bool = True,
) -> list[dict]:
    """Process all records through the full pipeline. Returns results in input order.

    Pass 1: group records by session_id, preserving first-seen order.
            Records without a session_id are each assigned a unique UUID
            (independent single-turn sessions).
    Pass 2: process sessions in first-seen order. Within a session, records
            are processed in file order, sharing PEF state and history.

    One SpacyBackend and one Governor bridge (CanonicalScannerGateBridge) are shared across all
    sessions. The audit file, when set, holds a single hash-chained record
    for the entire batch run.
    """
    spacy_backend = _load_spacy_backend()
    bridge = CanonicalScannerGateBridge(mode=governance_mode, audit_path=audit_path)

    session_records: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    session_order: list[str] = []

    for i, rec in enumerate(records):
        sid = rec.get("session_id") or str(uuid.uuid4())
        if sid not in session_records:
            session_order.append(sid)
        session_records[sid].append((i, rec))

    results_by_index: dict[int, dict] = {}
    total = len(records)
    done = 0

    for sid in session_order:
        lens = _build_lens(sid, adapter, spacy_backend, bridge)
        for i, rec in session_records[sid]:
            result = await lens.process(rec.get("user", ""))
            results_by_index[i] = _build_result(sid, rec, result)
            done += 1
            if progress:
                user_text = rec.get("user") or ""
                preview = user_text[:72] + ("…" if len(user_text) > 72 else "")
                en = bool(getattr(result, "epistemic_normalisation_applied", False))
                print(
                    f"  [{done}/{total}] {result.action.name} "
                    f"epistemic_norm={en} session={sid!r} {preview!r}",
                    file=sys.stderr,
                    flush=True,
                )

    return [results_by_index[i] for i in range(len(records))]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aurora-lens batch",
        description=(
            "Run aurora-lens governance over a JSONL scenario file.\n\n"
            "Each input line: {\"user\": \"...\", \"session_id\": \"s1\", "
            "\"expected_action\": \"HARD_STOP\"}.\n\n"
            "Requires OPENAI_API_KEY or ANTHROPIC_API_KEY."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", default="-",
                        help="Input JSONL path, or '-' for stdin (default: stdin)")
    parser.add_argument("--output", "-o", default="-",
                        help="Output JSONL path, or '-' for stdout (default: stdout)")
    parser.add_argument("--audit", "-a", default=None,
                        help="Forensic audit JSONL path (tamper-evident, hash-chained)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="LLM model name (default: gpt-4o-mini)")
    parser.add_argument("--mode", default="enterprise",
                        choices=["enterprise", "public"],
                        help="Governance mode (default: enterprise)")
    parser.add_argument("--fail-on-mismatch", action="store_true",
                        help="Exit 1 if any expected_action / expected_pathway mismatch")
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-turn progress lines on stderr (summary still prints).",
    )

    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "batch":
        argv = argv[1:]

    args = parser.parse_args(argv)

    # Build adapter.
    adapter, err = _build_adapter(args.model)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    # Read input.
    if args.input == "-":
        raw = sys.stdin.read()
    else:
        try:
            raw = Path(args.input).read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            print(f"Error: input file not found: {args.input}", file=sys.stderr)
            return 1

    records: list[dict] = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"Error: invalid JSON on line {lineno}: {exc}", file=sys.stderr)
            return 1

    if not records:
        print("Error: no records to process.", file=sys.stderr)
        return 1

    if not args.quiet:
        print(
            f"aurora-lens batch | {len(records)} turns | {args.model} | starting…",
            file=sys.stderr,
            flush=True,
        )

    # Run.
    try:
        results = asyncio.run(
            run_batch(
                records,
                adapter=adapter,
                audit_path=args.audit,
                governance_mode=args.mode,
                progress=not args.quiet,
            )
        )
    except (ValueError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Write output.
    output_text = (
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n"
    )
    if args.output == "-":
        sys.stdout.write(output_text)
    else:
        Path(args.output).write_text(output_text, encoding="utf-8-sig")

    # Summary to stderr.
    total = len(results)
    counts = {
        "PASS": sum(1 for r in results if r["action"] == "PASS"),
        "HARD_STOP": sum(1 for r in results if r["action"] == "HARD_STOP"),
        "FORCE_REVISE": sum(1 for r in results if r["action"] == "FORCE_REVISE"),
        "CONTAIN": sum(1 for r in results if r["action"] == "CONTAIN"),
    }
    expected_checked = [r for r in results if "expected_hit" in r]
    expected_passed = sum(1 for r in expected_checked if r["expected_hit"])

    print(f"aurora-lens batch | {total} turns | {args.model}", file=sys.stderr)
    parts = [f"{k}={v}" for k, v in counts.items() if v > 0]
    print(f"  {' '.join(parts) or 'no actions recorded'}", file=sys.stderr)
    if expected_checked:
        print(f"  Expected: {expected_passed}/{len(expected_checked)} matched",
              file=sys.stderr)
    if args.audit:
        print(f"  Audit: {args.audit}", file=sys.stderr)
    if args.output != "-":
        print(f"  Output: {args.output}", file=sys.stderr)

    if args.fail_on_mismatch and expected_checked:
        mismatches = [r for r in expected_checked if not r["expected_hit"]]
        if mismatches:
            print(f"  FAIL: {len(mismatches)} decision(s) did not match expected.",
                  file=sys.stderr)
            for r in mismatches:
                print(
                    f"    user={r['user']!r}  got={r['action']}  "
                    f"expected={r['expected_action']}",
                    file=sys.stderr,
                )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
