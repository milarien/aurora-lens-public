#!/usr/bin/env python3
"""Data-driven PEF continuity runner (YAML / JSON scenarios).

Loads scenario definitions (turns + checkpoints), drives the real
:class:`~aurora_lens.lens.Lens` with :class:`~aurora_lens.interpret.spacy_backend.SpacyBackend`
and state-native delegation using the same counting-upstream stub pattern as
``tests/test_pef_turn_based_continuity.py``.

Default fixture::

    tests/fixtures/pef_continuity_scenarios.yaml

Usage from repo root::

    python tools/run_pef_continuity_scenarios.py
    python tools/run_pef_continuity_scenarios.py tests/fixtures/pef_continuity_scenarios.yaml
    python tools/run_pef_continuity_scenarios.py tests/fixtures/pef_continuity_scenarios.yaml --quiet

Exit codes: **0** all checkpoints pass | **1** checkpoint assertion failed | **2** missing fixture / load error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import spacy  # noqa: F401
except ImportError:
    print("Missing dependency: install spaCy + en_core_web_sm.", file=sys.stderr)
    raise SystemExit(2)

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:
    print("Missing dependency: pip install pyyaml (YAML scenarios).", file=sys.stderr)
    raise SystemExit(2)

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter  # noqa: E402
from aurora_lens.config import LensConfig  # noqa: E402
from aurora_lens.interpret.spacy_backend import SpacyBackend  # noqa: E402
from aurora_lens.lens import Lens  # noqa: E402


DEFAULT_FIXTURE_REL = Path("tests") / "fixtures" / "pef_continuity_scenarios.yaml"


class _CountingUpstreamAdapter(LLMAdapter):
    def __init__(self, sentinel: str = "<<UPSTREAM_LLM_UNEXPECTED>>") -> None:
        self.calls = 0
        self.sentinel = sentinel

    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AdapterResponse:
        self.calls += 1
        return AdapterResponse(text=self.sentinel, model="pef-continuity-scenarios")


def _lens_live(adapter: _CountingUpstreamAdapter) -> Lens:
    return Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=SpacyBackend(model="en_core_web_sm"),
            auto_interpret=True,
            auto_verify=True,
            enable_state_native_delegation=True,
            inject_pef_context=False,
        )
    )


def _subject_display(pef: Any, subject_id: str) -> str:
    ent = pef.entities.get(subject_id)
    return ent.name.strip() if ent is not None else subject_id


def _format_has_contain(pef: Any) -> tuple[list[str], list[str]]:
    rows = sorted(
        (rel for rel in pef.relationships if rel.relation in {"HAS", "CONTAIN"} and not rel.negated),
        key=lambda r: (r.source_turn, _subject_display(pef, r.subject_id)),
    )
    has_lines: list[str] = []
    contain_lines: list[str] = []
    for rel in rows:
        name = _subject_display(pef, rel.subject_id)
        lit = str(rel.object_literal or "").strip()
        line = f"  {name} {rel.relation} {lit!r} (turn {rel.source_turn})"
        if rel.relation == "HAS":
            has_lines.append(line)
        else:
            contain_lines.append(line)
    return has_lines, contain_lines


def _print_has_contain(pef: Any, stream: TextIO) -> None:
    has_lines, contain_lines = _format_has_contain(pef)
    stream.write("Active HAS (non-negated):\n")
    stream.write("\n".join(has_lines) if has_lines else "  (none)\n")
    stream.write("\n")
    stream.write("Active CONTAIN (non-negated):\n")
    stream.write("\n".join(contain_lines) if contain_lines else "  (none)\n")


def _active_has_contain_rows(pef: Any) -> list[Any]:
    return [
        rel
        for rel in pef.relationships
        if rel.relation in {"HAS", "CONTAIN"} and not rel.negated
    ]


def _rel_matches_spec(pef: Any, rel: Any, spec: Mapping[str, Any]) -> bool:
    rel_name = str(spec.get("relation", "")).strip().upper()
    if rel.relation != rel_name:
        return False
    subj_need = str(spec.get("subject", "")).strip().lower()
    if not subj_need:
        return False
    if _subject_display(pef, rel.subject_id).strip().lower() != subj_need:
        return False
    needle = spec.get("object_literal_contains")
    if needle is None:
        return True
    lit = str(rel.object_literal or "").strip().lower()
    return str(needle).strip().lower() in lit


def _verify_expected_relationships(pef: Any, specs: Sequence[Mapping[str, Any]] | None) -> list[str]:
    if not specs:
        return []
    rows = _active_has_contain_rows(pef)
    errs: list[str] = []
    for i, spec in enumerate(specs):
        if not any(_rel_matches_spec(pef, rel, spec) for rel in rows):
            errs.append(f"expected_active_relationships[{i}] unmatched: {spec!r}")
    return errs


def _format_rel_row(pef: Any, rel: Any) -> str:
    name = _subject_display(pef, rel.subject_id)
    lit = str(rel.object_literal or "").strip()
    return f"{name} {rel.relation} {lit!r} (turn {rel.source_turn})"


def _first_matching_rel(pef: Any, spec: Mapping[str, Any]) -> Any | None:
    for rel in _active_has_contain_rows(pef):
        if _rel_matches_spec(pef, rel, spec):
            return rel
    return None


def _print_checkpoint_expected_vs_actual(
    stream: TextIO,
    *,
    query: str,
    response: str,
    raw_cp: Mapping[str, Any],
    upstream_delta: int,
    expect_up: int,
    pef: Any,
) -> None:
    """Human-readable contract: what the fixture asked for vs what ran."""

    stream.write("\n--- Checkpoint: EXPECTED vs ACTUAL ---\n")
    stream.write(f"  Query (user input): {query!r}\n")

    # Response text expectations
    contains = raw_cp.get("expected_contains")
    if contains is not None:
        if isinstance(contains, str):
            frags = [contains]
        elif isinstance(contains, list):
            frags = list(contains)
        else:
            frags = []
        stream.write("  Expected response (substrings, all required, case-insensitive):\n")
        if frags:
            low = response.lower()
            for frag in frags:
                ok = str(frag).lower() in low
                stream.write(f"    - {frag!r} -> {'FOUND' if ok else 'MISSING'}\n")
        else:
            stream.write("    (invalid expected_contains type in YAML)\n")
    else:
        stream.write("  Expected response (contains): (not specified)\n")

    rx = raw_cp.get("expected_regex")
    if rx is not None:
        stream.write(f"  Expected response (regex): {rx!r}\n")
        try:
            ok = re.search(str(rx), response, flags=re.DOTALL) is not None
            stream.write(f"    -> {'MATCH' if ok else 'NO MATCH'}\n")
        except re.error as e:
            stream.write(f"    -> INVALID REGEX ({e})\n")
    else:
        stream.write("  Expected response (regex): (not specified)\n")

    stream.write(f"  Actual response (full): {response!r}\n")

    stream.write("  Upstream LLM adapter calls (delta during this query only):\n")
    stream.write(f"    Expected: {expect_up}\n")
    stream.write(f"    Actual:   {upstream_delta}\n")
    stream.write(
        f"    -> {'PASS' if upstream_delta == expect_up else 'FAIL'}\n",
    )

    specs = raw_cp.get("expected_active_relationships")
    if specs:
        stream.write("  Expected active relationships (each spec must match some non-negated row):\n")
        for i, spec in enumerate(specs):
            if not isinstance(spec, dict):
                stream.write(f"    [{i}] (invalid spec mapping)\n")
                continue
            rel = _first_matching_rel(pef, spec)
            if rel is not None:
                stream.write(f"    [{i}] {spec!r}\n")
                stream.write(f"        MATCH: {_format_rel_row(pef, rel)}\n")
            else:
                stream.write(f"    [{i}] {spec!r}\n")
                stream.write("        NO MATCH among current non-negated HAS/CONTAIN rows\n")
    else:
        stream.write("  Expected active relationships: (none specified)\n")

    stream.write("--- (end checkpoint comparison) ---\n")


def _checkpoint_text_checks(response: str, cp: Mapping[str, Any]) -> list[str]:
    errs: list[str] = []
    contains = cp.get("expected_contains")
    rx = cp.get("expected_regex")
    if contains is None and rx is None:
        errs.append("checkpoint must set expected_contains and/or expected_regex")

    if contains is not None:
        if isinstance(contains, str):
            contains_list = [contains]
        elif isinstance(contains, list):
            contains_list = list(contains)
        else:
            errs.append("expected_contains must be a string or list of strings")
            contains_list = []
        low = response.lower()
        for frag in contains_list:
            if str(frag).lower() not in low:
                errs.append(f"expected_contains missing substring {frag!r} in response {response!r}")

    if rx is not None:
        try:
            if not re.search(str(rx), response, flags=re.DOTALL):
                errs.append(f"expected_regex did not match: {rx!r} vs {response!r}")
        except re.error as e:
            errs.append(f"expected_regex invalid pattern {rx!r}: {e}")
    return errs


def load_scenarios_document(path: Path) -> dict[str, Any]:
    """Load YAML or JSON scenarios file (exported for callers/tests)."""

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported scenario file suffix: {path.suffix}")
    if not isinstance(data, dict) or "scenarios" not in data:
        raise ValueError("Document must be a mapping with top-level 'scenarios' list")
    return data


async def collect_checkpoint_failures(
    fixture_path: Path | str,
    *,
    quiet: bool = False,
    stream: TextIO | None = None,
) -> list[str]:
    """Run every scenario/checkpoint; return human-readable failure lines (empty = all PASS)."""

    stream = stream or sys.stdout
    path = Path(fixture_path)
    if not path.is_file():
        return [f"Fixture not found: {path.resolve()}"]

    try:
        doc = load_scenarios_document(path)
    except Exception as e:
        return [f"Fixture load error: {e}"]

    scenarios_raw = doc["scenarios"]
    if not isinstance(scenarios_raw, list):
        return ["'scenarios' must be a list"]

    all_errors: list[str] = []

    for si, raw_sc in enumerate(scenarios_raw):
        if not isinstance(raw_sc, dict):
            all_errors.append(f"scenario[{si}] must be a mapping")
            continue
        name = str(raw_sc.get("name", f"scenario_{si}"))
        turns = raw_sc.get("turns", [])
        checkpoints = raw_sc.get("checkpoints", [])
        if not isinstance(turns, list) or not isinstance(checkpoints, list):
            all_errors.append(f"[{name}] 'turns' and 'checkpoints' must be lists")
            continue

        if not quiet:
            stream.write(f"\n{'=' * 72}\nScenario: {name}\n{'=' * 72}\n")

        adapter = _CountingUpstreamAdapter()
        lens = _lens_live(adapter)

        turn_failed = False
        for ti, raw_turn in enumerate(turns, start=1):
            line = str(raw_turn).strip()
            if not quiet:
                stream.write(f"\n--- Turn {ti} ---\n")
                stream.write(f"User: {line!r}\n")
            try:
                res = await lens.process(line)
            except Exception as e:
                all_errors.append(f"[{name}] turn {ti}: Lens.process raised {e!r}")
                turn_failed = True
                break
            if not quiet:
                stream.write(f"Released: {res.response.strip()}\n")
                _print_has_contain(lens.pef, stream)
                stream.write("\n")

        if turn_failed:
            continue

        for ci, raw_cp in enumerate(checkpoints):
            if not isinstance(raw_cp, dict):
                all_errors.append(f"[{name}] checkpoint[{ci}] must be a mapping")
                continue
            tag = f"[{name}] checkpoint[{ci}]"
            query = str(raw_cp.get("query", "")).strip()
            if not query:
                all_errors.append(f"{tag}: empty query")
                continue

            if not quiet:
                stream.write(f"\n>>> Checkpoint [{ci}] ---\n")

            calls_before = adapter.calls
            try:
                rescp = await lens.process(query)
            except Exception as e:
                all_errors.append(f"{tag}: Lens.process raised {e!r}")
                continue

            response = rescp.response.strip()
            upstream_delta = adapter.calls - calls_before

            try:
                expect_up = int(raw_cp.get("expected_upstream_calls", 0))
            except (TypeError, ValueError):
                all_errors.append(f"{tag}: expected_upstream_calls must be int-like")
                continue

            errs: list[str] = []
            errs.extend(_checkpoint_text_checks(response, raw_cp))
            if upstream_delta != expect_up:
                errs.append(
                    f"upstream_calls delta want {expect_up} got {upstream_delta} "
                    f"(adapter total={adapter.calls}) response={response!r}"
                )
            errs.extend(_verify_expected_relationships(lens.pef, raw_cp.get("expected_active_relationships")))

            overall = "PASS" if not errs else "FAIL"

            if not quiet:
                _print_checkpoint_expected_vs_actual(
                    stream,
                    query=query,
                    response=response,
                    raw_cp=raw_cp,
                    upstream_delta=upstream_delta,
                    expect_up=expect_up,
                    pef=lens.pef,
                )
                stream.write(f"  Checkpoint verdict: [{overall}]")
                if errs:
                    stream.write(" - discrepancies:\n")
                    for e in errs:
                        stream.write(f"    ! {e}\n")
                else:
                    stream.write("\n")
                stream.write("\nActive PEF snapshot after this checkpoint:\n")
                _print_has_contain(lens.pef, stream)
                stream.write("\n")

            for e in errs:
                all_errors.append(f"{tag}: {e}")

    return all_errors


async def async_main(fixture: Path, *, quiet: bool) -> int:
    fails = await collect_checkpoint_failures(fixture, quiet=quiet)
    out = sys.stdout
    if not quiet:
        out.write("\n" + "=" * 72 + "\nSummary\n" + "=" * 72 + "\n")
    if fails:
        for line in fails:
            out.write(f"FAILED: {line}\n")
        return 1
    if not quiet:
        out.write("All checkpoints passed.\n")
    return 0


def _resolve_fixture_path(p: Path) -> tuple[Path, list[Path]]:
    tried: list[Path] = []
    if p.is_file():
        r = p.resolve()
        tried.append(r)
        return r, tried
    cand = (_REPO_ROOT / p).resolve()
    tried.append(cand)
    if cand.is_file():
        return cand, tried
    cwd = (Path.cwd() / p).resolve()
    tried.append(cwd)
    if cwd.is_file():
        return cwd, tried
    return cand, tried


def _print_missing_fixture_help(path_attempt: Path, tried: list[Path]) -> None:
    default_fp = (_REPO_ROOT / DEFAULT_FIXTURE_REL).resolve()
    uniq: list[Path] = []
    names: set[str] = set()
    for t in tried:
        rs = str(t.resolve())
        if rs not in names:
            names.add(rs)
            uniq.append(t.resolve())
    print("Scenario file not found.", file=sys.stderr)
    print(f"  Lookup path given: {path_attempt}", file=sys.stderr)
    print("  Resolved candidates (in order):", file=sys.stderr)
    for t in uniq:
        print(f"    - {t} {'OK' if t.is_file() else 'missing'}", file=sys.stderr)
    print("", file=sys.stderr)
    print("The placeholder `path/to/scenarios.yaml` is not a real file.", file=sys.stderr)
    print("Use an existing YAML/JSON scenarios file. Default bundled fixture:", file=sys.stderr)
    print(f"    {default_fp}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Examples:", file=sys.stderr)
    print(f'  python tools/run_pef_continuity_scenarios.py "{default_fp}"', file=sys.stderr)
    print("  python tools/run_pef_continuity_scenarios.py --quiet", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    epilog = """Examples:
  %(prog)s
  %(prog)s tests/fixtures/pef_continuity_scenarios.yaml
  %(prog)s tests/fixtures/pef_continuity_scenarios.yaml --quiet

Exit codes: 0 all checkpoints pass, 1 assertion failed, 2 scenario file missing."""

    parser = argparse.ArgumentParser(
        description="Run PEF continuity scenarios from YAML/JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "fixture",
        nargs="?",
        type=Path,
        default=_REPO_ROOT / DEFAULT_FIXTURE_REL,
        help=f"Scenario file (default: {DEFAULT_FIXTURE_REL.as_posix()} under repo root)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-turn detail; checkpoint failures still emitted as FAILED lines.",
    )
    args = parser.parse_args(argv)
    fixture, tried = _resolve_fixture_path(args.fixture)
    if not fixture.is_file():
        _print_missing_fixture_help(Path(args.fixture), tried)
        return 2
    rc = asyncio.run(async_main(fixture, quiet=args.quiet))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
