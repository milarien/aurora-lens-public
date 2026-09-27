"""Score corpus Q&A validation cases (manifest-driven, no proxy dependency)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ExpectKind = Literal["in_context", "absent"]

ABSTAIN_PHRASE_EXACT = "Insufficient context in retrieved chunks."

ABSTAIN_PATTERNS: tuple[str, ...] = (
    ABSTAIN_PHRASE_EXACT.lower(),
    "insufficient context",
    "not found in context",
    "not in the retrieved",
    "not in the context",
    "does not contain",
    "does not mention",
    "doesn't mention",
    "not mentioned",
    "not covered",
    "not discussed in the context",
    "not addressed in the context",
    "no information in the context",
    "not in the document",
    "not in this document",
)

_PASS_GOVERNANCE = frozenset({"PASS", "SOFT_CORRECT"})


def _normalize(text: str) -> str:
    return text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")


def _marker_variants(marker: str) -> list[str]:
    return [part.strip() for part in str(marker).split("||") if part.strip()]


def _marker_hit(norm_answer: str, marker: str) -> bool:
    for variant in _marker_variants(marker):
        if _normalize(variant).lower() in norm_answer:
            return True
    return False


def answers_abstain(answer: str) -> bool:
    low = _normalize(answer).lower()
    return any(p in low for p in ABSTAIN_PATTERNS)


def count_marker_hits(answer: str, markers: list[str]) -> int:
    norm = _normalize(answer).lower()
    hits = 0
    for marker in markers:
        if _marker_hit(norm, marker):
            hits += 1
    return hits


def contains_forbidden(answer: str, forbidden: list[str]) -> list[str]:
    norm = _normalize(answer).lower()
    found: list[str] = []
    for phrase in forbidden:
        if _marker_hit(norm, phrase):
            found.append(phrase)
    return found


@dataclass
class CaseSpec:
    id: str
    question: str
    expect: ExpectKind
    record_ids: list[str] = field(default_factory=list)
    expected_governance_any: list[str] = field(default_factory=list)
    expected_llm_called: bool | None = None
    expected_record_ids_any: list[str] = field(default_factory=list)
    required_any: list[str] = field(default_factory=list)
    required_all: list[str] = field(default_factory=list)
    required_min_any: int = 1
    forbidden_answer: list[str] = field(default_factory=list)
    semantic_must_include_any: list[str] = field(default_factory=list)
    semantic_must_include_all: list[str] = field(default_factory=list)
    semantic_must_exclude: list[str] = field(default_factory=list)
    semantic_min_any_hits: int = 1
    citation_required: bool = False
    citation_record_ids_any: list[str] = field(default_factory=list)
    citation_chunk_locator_any: list[str] = field(default_factory=list)
    conflict_governing_record_id: str = ""
    conflict_superseded_record_ids: list[str] = field(default_factory=list)
    conflict_superseded_version_must_not_drive_answer: bool = False
    allow_conservative_abstain: bool = False
    locator_hint: str = ""
    notes: str = ""


@dataclass
class CaseResult:
    spec: CaseSpec
    governance: str
    answer: str
    chunks: int
    context_chars: int
    passed: bool
    reasons: list[str]
    session_id: str = ""


def load_manifest(path: Path) -> tuple[str, str, list[CaseSpec]]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise RuntimeError(
            "PyYAML is required. Install: pip install pyyaml"
        ) from e
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manifest must be a mapping")
    record_id = str(raw.get("record_id") or "").strip()
    if not record_id:
        raise ValueError("manifest record_id is required")
    policy_profile = str(raw.get("policy_profile") or "enterprise_strict").strip()
    items = raw.get("questions") or []
    cases: list[CaseSpec] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "").strip()
        question = str(item.get("question") or "").strip()
        expect = str(item.get("expect") or "").strip()
        if not cid or not question or expect not in ("in_context", "absent"):
            raise ValueError(f"invalid manifest case: {item!r}")
        cases.append(
            CaseSpec(
                id=cid,
                question=question,
                expect=expect,  # type: ignore[arg-type]
                record_ids=[str(x).strip() for x in (item.get("record_ids") or []) if str(x).strip()],
                expected_governance_any=[str(x).strip().upper() for x in (item.get("expected_governance_any") or []) if str(x).strip()],
                expected_llm_called=(
                    bool(item.get("expected_llm_called"))
                    if item.get("expected_llm_called") is not None
                    else None
                ),
                expected_record_ids_any=[str(x).strip() for x in (item.get("expected_record_ids_any") or []) if str(x).strip()],
                required_any=[str(x) for x in (item.get("required_any") or [])],
                required_all=[str(x) for x in (item.get("required_all") or [])],
                required_min_any=int(item.get("required_min_any") or 1),
                forbidden_answer=[str(x) for x in (item.get("forbidden_answer") or [])],
                semantic_must_include_any=[
                    str(x) for x in ((item.get("semantic_assertions") or {}).get("must_include_any") or [])
                ],
                semantic_must_include_all=[
                    str(x) for x in ((item.get("semantic_assertions") or {}).get("must_include_all") or [])
                ],
                semantic_must_exclude=[
                    str(x) for x in ((item.get("semantic_assertions") or {}).get("must_exclude") or [])
                ],
                semantic_min_any_hits=int((item.get("semantic_assertions") or {}).get("min_any_hits") or 1),
                citation_required=bool((item.get("citation_expectation") or {}).get("required", False)),
                citation_record_ids_any=[
                    str(x) for x in ((item.get("citation_expectation") or {}).get("record_ids_any") or [])
                ],
                citation_chunk_locator_any=[
                    str(x) for x in ((item.get("citation_expectation") or {}).get("chunk_locator_any") or [])
                ],
                conflict_governing_record_id=str((item.get("conflict_expectation") or {}).get("governing_record_id") or "").strip(),
                conflict_superseded_record_ids=[
                    str(x) for x in ((item.get("conflict_expectation") or {}).get("superseded_record_ids") or [])
                ],
                conflict_superseded_version_must_not_drive_answer=bool(
                    (item.get("conflict_expectation") or {}).get("superseded_version_must_not_drive_answer", False)
                ),
                allow_conservative_abstain=bool(item.get("allow_conservative_abstain", False)),
                locator_hint=str(item.get("locator_hint") or ""),
                notes=str(item.get("notes") or ""),
            )
        )
    if not cases:
        raise ValueError("manifest has no questions")
    return record_id, policy_profile, cases


def score_case(
    spec: CaseSpec,
    *,
    answer: str,
    governance: str,
    llm_called: bool | None = None,
    cited_record_ids: list[str] | None = None,
    cited_chunk_locators: list[str] | None = None,
    chunks: int = 0,
    context_chars: int = 0,
    session_id: str = "",
) -> CaseResult:
    reasons: list[str] = []
    gov = (governance or "?").strip().upper()
    abstain = answers_abstain(answer)
    forbidden_hits = contains_forbidden(answer, spec.forbidden_answer)
    cited_record_ids = cited_record_ids or []
    cited_chunk_locators = cited_chunk_locators or []

    if spec.expected_governance_any and gov not in {x.upper() for x in spec.expected_governance_any}:
        reasons.append(f"governance={gov!r} (expected any of {spec.expected_governance_any})")
    if spec.expected_llm_called is not None and llm_called is not None and llm_called != spec.expected_llm_called:
        reasons.append(f"llm_called={llm_called!r} (expected {spec.expected_llm_called!r})")
    if spec.expected_record_ids_any and not set(spec.expected_record_ids_any).intersection(set(cited_record_ids)):
        reasons.append(
            f"expected_record_ids_any={spec.expected_record_ids_any!r} not found in observed record ids {cited_record_ids!r}"
        )

    marker_ok = True
    if spec.required_all:
        missing = [
            m
            for m in spec.required_all
            if _normalize(m).lower() not in _normalize(answer).lower()
        ]
        if missing:
            marker_ok = False
    if spec.required_any:
        hits = count_marker_hits(answer, spec.required_any)
        need = max(1, spec.required_min_any)
        if hits < need:
            marker_ok = False

    if spec.expect == "in_context":
        if abstain and not marker_ok and not spec.allow_conservative_abstain:
            reasons.append("unexpected abstain for in-context question")
        if not spec.expected_governance_any:
            if gov not in _PASS_GOVERNANCE:
                if not (spec.allow_conservative_abstain and gov in {"FORCE_REVISE", "CONTAIN"}):
                    reasons.append(f"governance={gov!r} (expected PASS or SOFT_CORRECT)")
        if spec.required_all:
            missing = [
                m
                for m in spec.required_all
                if _normalize(m).lower() not in _normalize(answer).lower()
            ]
            if missing:
                reasons.append(f"missing required markers: {missing}")
        if spec.required_any and not marker_ok:
            reasons.append(
                f"too few required_any markers "
                f"({count_marker_hits(answer, spec.required_any)}/{max(1, spec.required_min_any)}): "
                f"{spec.required_any}"
            )
        if forbidden_hits:
            reasons.append(f"forbidden phrases in answer: {forbidden_hits}")
        if not answer.strip():
            reasons.append("empty answer")
        if abstain and marker_ok:
            reasons.append(
                "note: abstain phrase present but required markers also found (mixed answer)"
            )
    else:
        if forbidden_hits:
            reasons.append(f"invented or forbidden content: {forbidden_hits}")
        if not abstain and answer.strip():
            if not forbidden_hits:
                reasons.append(
                    "expected abstain (e.g. "
                    f"{ABSTAIN_PHRASE_EXACT!r}) but got substantive answer"
                )
        if not spec.expected_governance_any:
            if gov not in _PASS_GOVERNANCE and gov not in frozenset({"FORCE_REVISE", "CONTAIN"}):
                reasons.append(f"governance={gov!r} (unexpected for absent question)")

    sem_any_markers = spec.semantic_must_include_any or spec.required_any
    sem_all_markers = spec.semantic_must_include_all or spec.required_all
    sem_exclude_markers = spec.semantic_must_exclude or spec.forbidden_answer
    sem_min_hits = max(1, spec.semantic_min_any_hits or spec.required_min_any or 1)
    norm_answer = _normalize(answer).lower()
    skip_semantic_include = bool(
        spec.allow_conservative_abstain and (abstain or gov in {"HARD_STOP", "CONTAIN", "FORCE_REVISE"})
    )

    if sem_any_markers and not skip_semantic_include:
        sem_hits = count_marker_hits(answer, sem_any_markers)
        if sem_hits < sem_min_hits:
            reasons.append(
                f"semantic must_include_any failed: expected at least {sem_min_hits} hit(s) from {sem_any_markers!r}, observed {sem_hits}"
            )
    if sem_all_markers and not skip_semantic_include:
        sem_missing = [m for m in sem_all_markers if not _marker_hit(norm_answer, m)]
        if sem_missing:
            reasons.append(
                f"semantic must_include_all failed: missing {sem_missing!r}"
            )
    if sem_exclude_markers:
        sem_found = [m for m in sem_exclude_markers if _marker_hit(norm_answer, m)]
        if sem_found:
            reasons.append(
                f"semantic must_exclude failed: found forbidden markers {sem_found!r}"
            )

    if spec.citation_required and not (cited_record_ids or cited_chunk_locators):
        reasons.append("citation expectation failed: required citation/provenance was missing")
    if spec.citation_record_ids_any and not set(spec.citation_record_ids_any).intersection(set(cited_record_ids)):
        reasons.append(
            f"citation expectation failed: expected any record id in {spec.citation_record_ids_any!r}, observed {cited_record_ids!r}"
        )
    if spec.citation_chunk_locator_any:
        low_locs = [x.lower() for x in cited_chunk_locators]
        if not any(x.lower() in low_locs for x in spec.citation_chunk_locator_any):
            reasons.append(
                f"citation expectation failed: expected any chunk locator in {spec.citation_chunk_locator_any!r}, observed {cited_chunk_locators!r}"
            )

    if spec.conflict_governing_record_id:
        if spec.conflict_governing_record_id not in cited_record_ids:
            reasons.append(
                f"conflict expectation failed: governing_record_id {spec.conflict_governing_record_id!r} not in observed record ids {cited_record_ids!r}"
            )
    if spec.conflict_superseded_version_must_not_drive_answer and spec.conflict_superseded_record_ids:
        superseded_present = [rid for rid in spec.conflict_superseded_record_ids if rid in cited_record_ids]
        if superseded_present:
            reasons.append(
                f"conflict expectation failed: superseded record ids present in governing evidence {superseded_present!r}"
            )

    reasons = [r for r in reasons if not r.startswith("note:")]

    return CaseResult(
        spec=spec,
        governance=gov,
        answer=answer,
        chunks=chunks,
        context_chars=context_chars,
        passed=not reasons,
        reasons=reasons,
        session_id=session_id,
    )


def render_evidence_markdown(
    *,
    results: list[CaseResult],
    record_id: str,
    proxy: str,
    run_at: str,
    manifest_path: Path,
) -> str:
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    lines: list[str] = [
        "# Corpus Q&A validation evidence",
        "",
        f"- **Run at:** {run_at}",
        f"- **Record:** `{record_id}`",
        f"- **Proxy:** `{proxy}`",
        f"- **Manifest:** `{manifest_path.as_posix()}`",
        f"- **Score:** {passed}/{total} passed",
        "",
        "## Before (known failure modes, pre-fix)",
        "",
        "| Issue | Symptom |",
        "| --- | --- |",
        "| Retrieval rank discarded | §12.5 Personal Baseline chunk omitted; generic paraphrase |",
        "| Post-LLM verify | `FORCE_REVISE` / `UNSUPPORTED_ATTRIBUTE` on faithful corpus restatement |",
        "| Refusal copy | \"Choose one option to continue\" with no options presented |",
        "",
        "### Example before (Personal Baseline question)",
        "",
        "Retrieval kept ordinals 0–13 (early scope/chunk text). Answer paraphrased conceptually",
        "or was blocked with:",
        "",
        "```",
        "Cannot provide that.",
        "This request needs a missing detail.",
        "Action: Choose one option to continue.",
        "Status: Refused.",
        "```",
        "",
        "## After (this run)",
        "",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.extend(
            [
                f"### {r.spec.id} — {status}",
                "",
                f"- **Expect:** `{r.spec.expect}`",
                f"- **Question:** {r.spec.question}",
                f"- **Governance:** `{r.governance}`",
                f"- **Retrieval:** {r.chunks} chunks, {r.context_chars:,} context chars",
            ]
        )
        if r.spec.locator_hint:
            lines.append(f"- **Locator hint:** {r.spec.locator_hint}")
        if r.reasons:
            lines.append(f"- **Failures:** {'; '.join(r.reasons)}")
        lines.extend(["", "**Answer:**", "", "```", r.answer.strip() or "[empty]", "```", ""])
    lines.extend(
        [
            "## Summary",
            "",
            f"- **In-context extractable:** "
            f"{sum(1 for r in results if r.spec.expect == 'in_context' and r.passed)} / "
            f"{sum(1 for r in results if r.spec.expect == 'in_context')} passed",
            f"- **Absent (must abstain):** "
            f"{sum(1 for r in results if r.spec.expect == 'absent' and r.passed)} / "
            f"{sum(1 for r in results if r.spec.expect == 'absent')} passed",
            "",
            "Abstain phrase enforced in harness: "
            f"`{ABSTAIN_PHRASE_EXACT}` (plus accepted hedge patterns in scorer).",
            "",
        ]
    )
    return "\n".join(lines)


def results_to_json(results: list[CaseResult]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in results:
        out.append(
            {
                "id": r.spec.id,
                "question": r.spec.question,
                "expect": r.spec.expect,
                "passed": r.passed,
                "reasons": r.reasons,
                "governance": r.governance,
                "chunks": r.chunks,
                "context_chars": r.context_chars,
                "session_id": r.session_id,
                "answer": r.answer,
            }
        )
    return out
