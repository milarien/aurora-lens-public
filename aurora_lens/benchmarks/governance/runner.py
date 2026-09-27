"""Execute governance benchmark cases (Lens-governed vs optional upstream-only baseline)."""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aurora_lens.benchmarks.governance import score as score_mod
from aurora_lens.benchmarks.governance.fixtures import (
    FixtureRuntime,
    build_fixture,
    CountingMockAdapter,
)
from aurora_lens.benchmarks.governance.models import (
    BaselineRecord,
    CaseExecution,
    LensRunRecord,
)
from aurora_lens.verify.flags import Flag, FlagType


def default_cases_path() -> Path:
    """Cases ship next to this package (``aurora_lens/benchmarks/governance/cases_v1.json``)."""
    return Path(__file__).resolve().parent / "cases_v1.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "cases" in raw:
        raw = raw["cases"]
    if not isinstance(raw, list):
        raise ValueError(f"cases file must contain a JSON array or {{cases: [...]}}: {path}")
    return raw


def spacy_available() -> bool:
    try:
        import spacy  # noqa: F401

        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


def flags_from_names(names: list[str]) -> list[Flag]:
    """Build synthetic host flags for harness-injected governance pressure tests."""
    out: list[Flag] = []
    for name in names:
        ft = FlagType[name]
        out.append(
            Flag(
                flag_type=ft,
                entity_name="benchmark_case",
                claim=f"benchmark:{name}",
                evidence="benchmark.harness",
                severity="warning",
            )
        )
    return out


def _preview(text: str, limit: int = 240) -> str:
    t = text.replace("\n", " ").strip()
    return t if len(t) <= limit else t[: limit - 3] + "..."


async def _run_lens_steps(
    case: dict[str, Any],
    rt: FixtureRuntime,
) -> tuple[LensRunRecord, int | None, int | None]:
    steps = case.get("steps")
    if steps:
        seq = steps
    else:
        seq = [{"user_input": case["user_input"], "external_flags": case.get("external_flags")}]

    last = None
    for step in seq:
        ui = step["user_input"]
        ext = step.get("external_flags") or case.get("external_flags")
        fl = flags_from_names(ext) if ext else None
        last = await rt.lens.process(ui, external_flags=fl)

    assert last is not None
    flag_names = [f.flag_type.name for f in last.flags]
    interaction_open = None
    if last.decision is not None:
        interaction_open = last.decision.interaction_open

    rec = LensRunRecord(
        action=last.action.name,
        flag_names=flag_names,
        response_preview=_preview(last.response),
        interaction_open=interaction_open,
        upstream_calls=rt.adapter.call_count,
        continuity_diagnostic=getattr(last, "continuity_diagnostic", None),
    )

    audit_lines = None
    if rt.audit_path is not None and rt.audit_path.exists():
        audit_lines = len([ln for ln in rt.audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()])

    recording_rows = None
    if rt.recording_bridge is not None:
        recording_rows = len(rt.recording_bridge.audit_log)

    return rec, audit_lines, recording_rows


def _effective_user_input(case: dict[str, Any]) -> str:
    steps = case.get("steps")
    if steps:
        return steps[-1]["user_input"]
    return case["user_input"]


async def _run_baseline(case: dict[str, Any], rt: FixtureRuntime, mode: str) -> BaselineRecord:
    if mode in ("none", "off", ""):
        return BaselineRecord(mode="none")
    if mode != "adapter_only":
        return BaselineRecord(mode="none", skipped_reason=f"unknown baseline mode: {mode!r}")

    ui = _effective_user_input(case)
    adapter = CountingMockAdapter(rt.adapter._responses)  # noqa: SLF001
    messages = [{"role": "user", "content": ui}]
    resp = await adapter.generate(messages)
    return BaselineRecord(
        mode="adapter_only",
        response_preview=_preview(resp.text),
        adapter_calls=adapter.call_count,
    )


def _audit_json_ok(audit_path: Path | None) -> bool | None:
    if audit_path is None or not audit_path.exists():
        return None
    ok = 0
    for ln in audit_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            json.loads(ln)
            ok += 1
        except json.JSONDecodeError:
            return False
    return ok > 0


async def run_benchmark(
    cases_path: Path,
    *,
    baseline_mode: str = "none",
    out_dir: Path | None = None,
) -> dict[str, Any]:
    cases = load_cases(cases_path)
    require_spacy = spacy_available()

    executions: list[CaseExecution] = []

    for case in cases:
        cid = case["id"]
        cat = case["category"]

        if case.get("requires_spacy") and not require_spacy:
            executions.append(
                CaseExecution(
                    case_id=cid,
                    category=cat,
                    skipped=True,
                    skip_reason="requires spaCy model en_core_web_sm (install: python -m spacy download en_core_web_sm)",
                )
            )
            continue

        audit_file: Path | None = None
        try:
            if case.get("use_audit_jsonl"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
                tmp.close()
                audit_file = Path(tmp.name)
            rt = build_fixture(case["fixture_id"], audit_file, case)

            lens_rec, audit_lines, rec_rows = await _run_lens_steps(case, rt)
            baseline_rec = await _run_baseline(case, rt, baseline_mode)

            audit_ok = _audit_json_ok(rt.audit_path)

            exec_row = CaseExecution(
                case_id=cid,
                category=cat,
                lens=lens_rec,
                baseline=baseline_rec,
                audit_lines_written=audit_lines,
                recording_rows=rec_rows,
            )
            exec_row.scores = score_mod.score_case(case, exec_row, audit_ok=audit_ok)
            executions.append(exec_row)
        finally:
            if audit_file is not None and audit_file.exists():
                try:
                    audit_file.unlink()
                except OSError:
                    pass

    summary = score_mod.summarize(executions)

    def _meta(c: dict[str, Any]) -> dict[str, Any]:
        return {
            "expected_outcome": c.get("expected_outcome"),
            "title": c.get("title"),
            "fixture_id": c.get("fixture_id"),
        }

    cases_by_id = {c["id"]: c for c in cases}

    payload = {
        "cases_file": str(cases_path),
        "baseline_mode": baseline_mode,
        "spacy_available": require_spacy,
        "summary": summary,
        "executions": [
            {
                "case_id": e.case_id,
                "category": e.category,
                "skipped": e.skipped,
                "skip_reason": e.skip_reason,
                "case_meta": _meta(cases_by_id.get(e.case_id, {})),
                "lens": asdict(e.lens) if e.lens else None,
                "baseline": asdict(e.baseline) if e.baseline else None,
                "audit_lines_written": e.audit_lines_written,
                "recording_rows": e.recording_rows,
                "scores": e.scores,
            }
            for e in executions
        ],
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        machine_path = out_dir / "governance_benchmark_results.json"
        machine_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Aurora-Lens governance benchmark (correctness, not fluency).")
    p.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run"],
        help="Subcommand (default: run).",
    )
    p.add_argument(
        "--cases",
        type=Path,
        default=None,
        help=f"Path to benchmark JSON (default: {default_cases_path()})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write machine-readable results + summary table here.",
    )
    p.add_argument(
        "--baseline",
        choices=["none", "adapter_only"],
        default="none",
        help="Baseline mode: none | adapter_only (upstream MockAdapter.generate only; no Lens governance).",
    )

    ns = p.parse_args(argv)

    cases_path = ns.cases or default_cases_path()
    if not cases_path.exists():
        raise SystemExit(f"cases file not found: {cases_path}")

    payload = asyncio.run(run_benchmark(cases_path, baseline_mode=ns.baseline, out_dir=ns.out_dir))

    text_report = score_mod.render_human_summary(payload)
    print(text_report)

    if ns.out_dir is not None:
        summary_path = ns.out_dir / "governance_benchmark_summary.txt"
        summary_path.write_text(text_report, encoding="utf-8")

    summary = payload["summary"]
    if summary.get("failed_cases", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
