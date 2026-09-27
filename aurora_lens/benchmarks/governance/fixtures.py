"""Deterministic Lens fixtures for governance benchmarks (no chatbot scoring).

Fixture builders wire the same adapters/backends used in regression tests — they do not
relax governance policy.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from aurora_lens.adapters.base import AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.govern.bridge import BuiltinBridge, GovernanceBridge
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractedClaim, ExtractionResult
from aurora_lens.lens import Lens
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.pef.state import canonicalize_relation


class RecordingScenarioBridge(GovernanceBridge):
    """Same pattern as proof-scenario tests: scripted decide(); captures audit rows."""

    def __init__(self, decisions: list[GovernanceDecision]):
        self._decisions = list(decisions)
        self._idx = 0
        self.audit_log: list[dict[str, Any]] = []

    async def decide(
        self,
        flags,
        response_text: str,
        pef: PEFState,
    ) -> GovernanceDecision:
        if self._idx >= len(self._decisions):
            template = self._decisions[-1]
        else:
            template = self._decisions[self._idx]
            self._idx += 1
        return dataclasses.replace(template, flags=flags)

    async def intervene(
        self,
        decision: GovernanceDecision,
        adapter,
        user_input: str,
        pef_context: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        return decision.governed_response or decision.corrected_response or ""

    def log_decision(
        self,
        decision: GovernanceDecision,
        turn: int = 0,
        *,
        stream: bool = False,
        stream_completed: bool = True,
        stream_abort_reason: str | None = None,
        stream_truncated: bool = False,
        stream_dropped_chars: int = 0,
        pef_context: str | None = None,
        pre_llm: bool = False,
        pef_snapshot: dict | None = None,
        pef_turn_classification: str | None = None,
        pef_hold_transition: str | None = None,
        at_verification_basis: dict | None = None,
        epistemic_normalisation_applied: bool | None = None,
        state_native_handled: bool | None = None,
    ) -> None:
        row: dict[str, Any] = {"turn": turn, "action": decision.action}
        if pef_snapshot is not None:
            row["pef_snapshot"] = pef_snapshot
        if pre_llm:
            row["pre_llm"] = True
        _ = (
            stream,
            stream_completed,
            stream_abort_reason,
            stream_truncated,
            stream_dropped_chars,
            pef_context,
            pef_turn_classification,
            pef_hold_transition,
            at_verification_basis,
        )
        if epistemic_normalisation_applied is not None:
            row["epistemic_normalisation_applied"] = epistemic_normalisation_applied
        if state_native_handled is not None:
            row["state_native_handled"] = state_native_handled
        self.audit_log.append(row)


class CountingMockAdapter:
    """Canned adapter with call counting (deterministic upstream)."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset_calls(self) -> None:
        """Reset adapter invocation counter (e.g. between timed repeats)."""
        self._call_count = 0

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> AdapterResponse:
        idx = min(self._call_count, len(self._responses) - 1)
        text = self._responses[idx]
        self._call_count += 1
        return AdapterResponse(text=text, model="mock-benchmark")

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[tuple[dict[str, object], str]]:
        """Minimal stream: single chunk with full canned text (benchmark / tests only)."""
        idx = min(self._call_count, len(self._responses) - 1)
        text = self._responses[idx]
        self._call_count += 1
        chunk_dict: dict[str, object] = {
            "choices": [{"delta": {"content": text}, "index": 0}],
        }
        yield (chunk_dict, text)


class EmptyExtractBackend(ExtractionBackend):
    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)


class AmbiguousReferentBackend(ExtractionBackend):
    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        return ExtractionResult(
            claims=[],
            entity_mentions=["Alice", "Bob"],
            span=Span.PRESENT,
            ambiguous_referents=["she"],
        )


class AmbiguousThreePartyBackend(ExtractionBackend):
    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        return ExtractionResult(
            claims=[],
            entity_mentions=["Alice", "Bob", "Carol"],
            span=Span.PRESENT,
            ambiguous_referents=["she"],
        )


class MutationSequenceBackend(ExtractionBackend):
    """Three-turn deterministic chain: desk AT → safe AT → QUERY surface (empty claims)."""

    def __init__(self) -> None:
        self._turn = 0

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        self._turn += 1
        if self._turn == 1:
            return ExtractionResult(
                claims=[
                    ExtractedClaim(
                        subject="gold key",
                        relation="AT",
                        obj="the desk",
                        span=Span.PRESENT,
                        negated=False,
                        evidence=text,
                        extractor_backend="benchmark",
                    ),
                ],
                entity_mentions=["gold key"],
                span=Span.PRESENT,
            )
        if self._turn == 2:
            return ExtractionResult(
                claims=[
                    ExtractedClaim(
                        subject="gold key",
                        relation="AT",
                        obj="the safe",
                        span=Span.PRESENT,
                        negated=False,
                        evidence=text,
                        extractor_backend="benchmark",
                    ),
                ],
                entity_mentions=["gold key"],
                span=Span.PRESENT,
            )
        return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)


class IndexedEmmaAppleBackend(ExtractionBackend):
    """First turn HAS claim; later turns empty (proof hold scenario)."""

    def __init__(self) -> None:
        self._n = 0

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        self._n += 1
        if self._n == 1:
            return ExtractionResult(
                claims=[
                    ExtractedClaim(
                        subject="Emma",
                        relation="HAS",
                        obj="red apple",
                        span=Span.PRESENT,
                        negated=False,
                        evidence=text,
                    ),
                ],
                entity_mentions=["Emma"],
                span=Span.PRESENT,
            )
        return ExtractionResult(claims=[], entity_mentions=[], span=Span.PRESENT)


def _pef_silver_key_at_safe_literal(session_id: str = "bench") -> PEFState:
    p = PEFState(session_id=session_id)
    key = Entity.create("silver key", 0, session_id=session_id)
    p.add_entity(key)
    p.add_relationship(
        Relationship(
            subject_id=key.id,
            relation="AT",
            object_entity_id=None,
            object_literal="the safe",
            span=Span.PRESENT,
            source_turn=1,
            evidence="benchmark",
        )
    )
    return p


def _pef_silver_key_no_at(session_id: str = "bench") -> PEFState:
    p = PEFState(session_id=session_id)
    key = Entity.create("silver key", 0, session_id=session_id)
    p.add_entity(key)
    return p


def _pef_two_keys(session_id: str = "bench") -> PEFState:
    p = PEFState(session_id=session_id)
    for name in ("gold key", "silver key"):
        p.add_entity(Entity.create(name, 0, session_id=session_id))
    return p


def _pef_two_richards(session_id: str = "bench") -> PEFState:
    p = PEFState(session_id=session_id)
    p.add_entity(Entity.create("Richard Hale", 0, session_id=session_id))
    p.add_entity(Entity.create("Richard Roe", 0, session_id=session_id))
    return p


def _pef_richard_has_gold_key(session_id: str = "bench") -> PEFState:
    p = PEFState(session_id=session_id)
    richard = Entity.create("Richard", 0, session_id=session_id)
    gold_key = Entity.create("gold key", 0, session_id=session_id)
    p.add_entity(richard)
    p.add_entity(gold_key)
    p.add_relationship(
        Relationship(
            subject_id=richard.id,
            relation="HAS",
            object_entity_id=gold_key.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="benchmark",
        )
    )
    return p


def _pef_richard_only(session_id: str = "bench") -> PEFState:
    p = PEFState(session_id=session_id)
    p.add_entity(Entity.create("Richard", 0, session_id=session_id))
    return p


def _pef_two_holders_gold_key(session_id: str = "bench") -> PEFState:
    p = PEFState(session_id=session_id)
    richard = Entity.create("Richard", 0, session_id=session_id)
    emma = Entity.create("Emma", 0, session_id=session_id)
    key = Entity.create("gold key", 0, session_id=session_id)
    for e in (richard, emma, key):
        p.add_entity(e)
    for holder in (richard, emma):
        p.add_relationship(
            Relationship(
                subject_id=holder.id,
                relation="HAS",
                object_entity_id=key.id,
                object_literal=None,
                span=Span.PRESENT,
                source_turn=1,
                evidence="benchmark",
            )
        )
    return p


def _pef_one_holder_gold_key(session_id: str = "bench") -> PEFState:
    p = PEFState(session_id=session_id)
    richard = Entity.create("Richard", 0, session_id=session_id)
    key = Entity.create("gold key", 0, session_id=session_id)
    p.add_entity(richard)
    p.add_entity(key)
    p.add_relationship(
        Relationship(
            subject_id=richard.id,
            relation="HAS",
            object_entity_id=key.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="benchmark",
        )
    )
    return p


def _finance_pef_roi(is_value: str) -> PEFState:
    pef = PEFState()
    ent, _ = pef.get_or_create_entity("roi")
    pef.add_relationship(
        Relationship(
            subject_id=ent.id,
            relation=canonicalize_relation("IS"),
            object_entity_id=None,
            object_literal=is_value,
            span=Span.PRESENT,
            source_turn=0,
            evidence=f"roi IS {is_value}",
            provenance="pre_populated",
            extractor_backend="manual",
        )
    )
    return pef


@dataclasses.dataclass
class FixtureRuntime:
    """Per-run wiring returned to the runner."""

    lens: Lens
    adapter: CountingMockAdapter
    audit_path: Path | None
    recording_bridge: RecordingScenarioBridge | None = None


def _lens_builtin(
    adapter: CountingMockAdapter,
    audit_path: Path | None,
    *,
    extraction: ExtractionBackend,
    initial_pef: PEFState | None = None,
    session_id: str = "bench",
    enable_state_native: bool = False,
    auto_verify: bool = True,
    auto_interpret: bool = True,
    inject_pef_context: bool = True,
) -> FixtureRuntime:
    bridge: GovernanceBridge | None = None
    if audit_path is not None:
        bridge = BuiltinBridge(audit_path=str(audit_path))
    cfg = LensConfig(
        adapter=adapter,
        extraction_backend=extraction,
        governance_bridge=bridge,
        enable_state_native_delegation=enable_state_native,
        auto_verify=auto_verify,
        auto_interpret=auto_interpret,
        inject_pef_context=inject_pef_context,
    )
    lens = Lens(cfg, initial_pef=initial_pef, session_id=session_id)
    return FixtureRuntime(lens=lens, adapter=adapter, audit_path=audit_path)


def build_fixture(
    fixture_id: str,
    audit_path: Path | None,
    case: dict[str, Any],
) -> FixtureRuntime:
    """Construct a Lens + adapter for the given benchmark case fixture id."""

    builders: dict[str, Callable[[Path | None, dict[str, Any]], FixtureRuntime]] = {
        # --- ambiguity / state-native ---
        "amb_unresolved_referent_she": _fx_amb_unresolved,
        "amb_unresolved_three_party": _fx_amb_three_party,
        "amb_location_two_keys": _fx_amb_two_keys,
        "amb_richard_where_dup": _fx_amb_richard_where,
        "amb_who_has_multiple_holders": _fx_amb_who_has_multi,
        "amb_what_does_richard_dup": _fx_amb_what_does_dup,
        "amb_inventory_richard_empty_stop": _fx_amb_richard_empty_inv,
        # --- unsupported commitment ---
        "uns_host_unverified_flag": _fx_uns_host_flag,
        "uns_fin_revenue_ungrounded": _fx_fin_ungrounded,
        "uns_fin_roi_contradicted": _fx_fin_contradicted,
        "uns_fin_pef_backed_clean": _fx_fin_clean,
        "uns_fin_hedged_clean": _fx_fin_hedged,
        "uns_fin_pfa_projection": _fx_fin_pfa,
        # --- high stakes (request-side blocked act) ---
        "hs_pediatric_dosage": _fx_hs_pediatric,
        "hs_legal_notice_appeal": _fx_hs_legal_appeal,
        "hs_metformin_titration": _fx_hs_metformin_adult,
        "hs_metformin_general_info": _fx_hs_metformin_info,
        "hs_legal_definition": _fx_hs_legal_general,
        "hs_pfa_portfolio_question": _fx_hs_pfa_user,
        # --- persistence & contradiction ---
        "per_silver_key_safe": _fx_per_silver_safe,
        "per_silver_key_no_at": _fx_per_silver_no_at,
        "per_gold_key_location_chain": _fx_per_gold_chain,
        "per_richard_inventory": _fx_per_richard_inv,
        "per_richard_no_inventory_stop": _fx_per_richard_empty,
        "per_who_has_one_holder": _fx_per_who_has_one,
        # --- failure path / recording ---
        "rec_hold_refusal_soft_clear": _fx_rec_hold_refusal,
        "rec_stop_open_cleared": _fx_rec_stop_open,
        "rec_stop_terminal_persists": _fx_rec_stop_term,
        "rec_audit_snapshot_holds": _fx_rec_audit_snapshot,
        "rec_single_builtin_audit": _fx_rec_builtin_hello,
        "rec_blocked_with_audit": _fx_rec_blocked_audit,
    }
    if fixture_id not in builders:
        raise KeyError(f"unknown fixture_id: {fixture_id}")
    return builders[fixture_id](audit_path, case)


def _fx_amb_unresolved(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["The model should not determine the referent."])
    return _lens_builtin(
        adapter,
        ap,
        extraction=AmbiguousReferentBackend(),
        auto_verify=True,
        inject_pef_context=False,
        session_id=case.get("session_id", "amb-she"),
    )


def _fx_amb_three_party(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["Clarification only."])
    return _lens_builtin(
        adapter,
        ap,
        extraction=AmbiguousThreePartyBackend(),
        auto_verify=True,
        inject_pef_context=False,
        session_id=case.get("session_id", "amb-three"),
    )


def _fx_amb_two_keys(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["model-should-not-run"])
    return _lens_builtin(
        adapter,
        ap,
        extraction=EmptyExtractBackend(),
        initial_pef=_pef_two_keys(),
        enable_state_native=True,
        auto_verify=False,
        session_id=case.get("session_id", "amb-two-keys"),
    )


def _fx_amb_richard_where(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["model-should-not-run"])
    return _lens_builtin(
        adapter,
        ap,
        extraction=EmptyExtractBackend(),
        initial_pef=_pef_two_richards(),
        enable_state_native=True,
        auto_verify=False,
        session_id=case.get("session_id", "amb-richard-dup"),
    )


def _fx_amb_who_has_multi(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["model-should-not-run"])
    return _lens_builtin(
        adapter,
        ap,
        extraction=EmptyExtractBackend(),
        initial_pef=_pef_two_holders_gold_key(),
        enable_state_native=True,
        auto_verify=False,
        session_id=case.get("session_id", "amb-holders"),
    )


def _fx_amb_what_does_dup(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["model-should-not-run"])
    return _lens_builtin(
        adapter,
        ap,
        extraction=EmptyExtractBackend(),
        initial_pef=_pef_two_richards(),
        enable_state_native=True,
        auto_verify=False,
        session_id=case.get("session_id", "amb-what-richard"),
    )


def _fx_amb_richard_empty_inv(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["model-should-not-run"])
    return _lens_builtin(
        adapter,
        ap,
        extraction=EmptyExtractBackend(),
        initial_pef=_pef_richard_only(),
        enable_state_native=True,
        auto_verify=False,
        session_id=case.get("session_id", "amb-richard-empty"),
    )


def _fx_uns_host_flag(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["Ungrounded numeric certainty 99%."])
    return _lens_builtin(adapter, ap, extraction=EmptyExtractBackend(), auto_verify=False)


def _lens_finance_canonical(
    adapter: CountingMockAdapter,
    audit_path: Path | None,
    *,
    initial_pef: PEFState | None = None,
    session_id: str = "fin",
) -> FixtureRuntime:
    """Matches tests/test_finance_governance.py integrated E2E (default Lens bridge)."""
    from aurora_lens.interpret.spacy_backend import SpacyBackend

    cfg = LensConfig(
        adapter=adapter,
        extraction_backend=SpacyBackend(model="en_core_web_sm"),
        audit_log_path=str(audit_path) if audit_path is not None else None,
        inject_pef_context=False,
        auto_verify=True,
    )
    lens = Lens(cfg, initial_pef=initial_pef, session_id=session_id)
    return FixtureRuntime(lens=lens, adapter=adapter, audit_path=audit_path)


def _fx_fin_ungrounded(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["Revenue increased 23% this quarter."])
    return _lens_finance_canonical(adapter, ap, session_id=case.get("session_id", "fin-ungrounded"))


def _fx_fin_contradicted(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["Based on our records, the ROI is 25%."])
    return _lens_finance_canonical(
        adapter,
        ap,
        initial_pef=_finance_pef_roi("23%"),
        session_id=case.get("session_id", "fin-contra"),
    )


def _fx_fin_clean(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["The ROI is 23%, as established."])
    return _lens_finance_canonical(
        adapter,
        ap,
        initial_pef=_finance_pef_roi("23%"),
        session_id=case.get("session_id", "fin-clean"),
    )


def _fx_fin_hedged(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["Revenue may increase by approximately 23% this quarter."])
    return _lens_finance_canonical(adapter, ap, session_id=case.get("session_id", "fin-hedged"))


def _fx_fin_pfa(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["Based on your portfolio, your projected ROI is 34.7%."])
    return _lens_finance_canonical(adapter, ap, session_id=case.get("session_id", "fin-pfa"))


def _fx_hs_pediatric(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["FORBIDDEN_MODEL_BODY"])
    return _lens_builtin(adapter, ap, extraction=EmptyExtractBackend(), auto_verify=False)


def _fx_hs_legal_appeal(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["FORBIDDEN_MODEL_BODY"])
    return _lens_builtin(adapter, ap, extraction=EmptyExtractBackend(), auto_verify=False)


def _fx_hs_metformin_adult(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["FORBIDDEN_MODEL_BODY"])
    return _lens_builtin(adapter, ap, extraction=EmptyExtractBackend(), auto_verify=False)


def _fx_hs_metformin_info(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["Metformin is used for type 2 diabetes."])
    return _lens_builtin(adapter, ap, extraction=EmptyExtractBackend(), auto_verify=True)


def _fx_hs_legal_general(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["A notice to quit is a formal termination notice."])
    return _lens_builtin(adapter, ap, extraction=EmptyExtractBackend(), auto_verify=True)


def _fx_hs_pfa_user(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["FORBIDDEN_MODEL_BODY"])
    return _lens_builtin(adapter, ap, extraction=EmptyExtractBackend(), auto_verify=False)


def _fx_per_silver_safe(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["should-not-run"])
    sid = case.get("session_id", "per-safe")
    return _lens_builtin(
        adapter,
        ap,
        extraction=EmptyExtractBackend(),
        initial_pef=_pef_silver_key_at_safe_literal(sid),
        enable_state_native=True,
        auto_verify=False,
        inject_pef_context=False,
        session_id=sid,
    )


def _fx_per_silver_no_at(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["should-not-run"])
    sid = case.get("session_id", "per-no-at")
    return _lens_builtin(
        adapter,
        ap,
        extraction=EmptyExtractBackend(),
        initial_pef=_pef_silver_key_no_at(sid),
        enable_state_native=True,
        auto_verify=False,
        inject_pef_context=False,
        session_id=sid,
    )


def _fx_per_gold_chain(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["MODEL_SHOULD_NOT_RUN"])
    sid = case.get("session_id", "per-chain")
    return _lens_builtin(
        adapter,
        ap,
        extraction=MutationSequenceBackend(),
        enable_state_native=True,
        auto_verify=False,
        auto_interpret=True,
        inject_pef_context=False,
        session_id=sid,
    )


def _fx_per_richard_inv(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["should-not-run"])
    sid = case.get("session_id", "per-inv")
    return _lens_builtin(
        adapter,
        ap,
        extraction=EmptyExtractBackend(),
        initial_pef=_pef_richard_has_gold_key(sid),
        enable_state_native=True,
        auto_verify=False,
        inject_pef_context=False,
        session_id=sid,
    )


def _fx_per_richard_empty(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["should-not-run"])
    sid = case.get("session_id", "per-inv-empty")
    return _lens_builtin(
        adapter,
        ap,
        extraction=EmptyExtractBackend(),
        initial_pef=_pef_richard_only(sid),
        enable_state_native=True,
        auto_verify=False,
        inject_pef_context=False,
        session_id=sid,
    )


def _fx_per_who_has_one(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["should-not-run"])
    sid = case.get("session_id", "per-one-holder")
    return _lens_builtin(
        adapter,
        ap,
        extraction=EmptyExtractBackend(),
        initial_pef=_pef_one_holder_gold_key(sid),
        enable_state_native=True,
        auto_verify=False,
        inject_pef_context=False,
        session_id=sid,
    )


def _fx_rec_hold_refusal(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    bridge = RecordingScenarioBridge(
        [
            GovernanceDecision(
                action=InterventionAction.FORCE_REVISE,
                flags=[],
                rationale="benchmark-refuse",
                policy="strict",
                pathway_id="P_REFUSE",
                interaction_open=True,
                commitment_closed=True,
                cid="cid-bench-refuse",
                governed_response="Cannot assert that without support.",
            ),
            GovernanceDecision(
                action=InterventionAction.SOFT_CORRECT,
                flags=[],
                rationale="benchmark-soft",
                policy="strict",
                pathway_id=None,
                cid="cid-bench-soft",
                governed_response="Understood; continuing.",
            ),
        ]
    )
    adapter = CountingMockAdapter(["Emma has a red apple.", "Revenue is $1M.", "Thanks."])
    cfg = LensConfig(
        adapter=adapter,
        extraction_backend=IndexedEmmaAppleBackend(),
        governance_bridge=bridge,
        auto_verify=False,
    )
    lens = Lens(cfg)
    rt = FixtureRuntime(lens=lens, adapter=adapter, audit_path=None, recording_bridge=bridge)
    return rt


def _fx_rec_stop_open(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    bridge = RecordingScenarioBridge(
        [
            GovernanceDecision(
                action=InterventionAction.HARD_STOP,
                flags=[],
                rationale="bench-stop-open",
                policy="strict",
                pathway_id="P_STOP",
                interaction_open=True,
                commitment_closed=False,
                cid="cid-stop-open",
                governed_response="Stop (reopenable).",
            ),
            GovernanceDecision(
                action=InterventionAction.SOFT_CORRECT,
                flags=[],
                rationale="bench-soft",
                policy="strict",
                pathway_id=None,
                cid="cid-soft",
                governed_response="Continuing.",
            ),
        ]
    )
    adapter = CountingMockAdapter(["Hello.", "Okay."])
    cfg = LensConfig(
        adapter=adapter,
        extraction_backend=EmptyExtractBackend(),
        governance_bridge=bridge,
        auto_verify=False,
    )
    lens = Lens(cfg)
    return FixtureRuntime(lens=lens, adapter=adapter, audit_path=None, recording_bridge=bridge)


def _fx_rec_stop_term(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    bridge = RecordingScenarioBridge(
        [
            GovernanceDecision(
                action=InterventionAction.HARD_STOP,
                flags=[],
                rationale="bench-stop-term",
                policy="strict",
                pathway_id="P_STOP_TERMINAL",
                interaction_open=False,
                commitment_closed=True,
                cid="cid-stop-term",
                governed_response="Stop (terminal).",
            ),
            GovernanceDecision(
                action=InterventionAction.SOFT_CORRECT,
                flags=[],
                rationale="bench-soft-noop",
                policy="strict",
                pathway_id=None,
                cid="cid-soft-term",
                governed_response="Acknowledged.",
            ),
        ]
    )
    adapter = CountingMockAdapter(["Hello.", "Okay."])
    cfg = LensConfig(
        adapter=adapter,
        extraction_backend=EmptyExtractBackend(),
        governance_bridge=bridge,
        auto_verify=False,
    )
    lens = Lens(cfg)
    return FixtureRuntime(lens=lens, adapter=adapter, audit_path=None, recording_bridge=bridge)


def _fx_rec_audit_snapshot(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    bridge = RecordingScenarioBridge(
        [
            GovernanceDecision(
                action=InterventionAction.FORCE_REVISE,
                flags=[],
                rationale="audit-1",
                policy="strict",
                pathway_id="P_REFUSE",
                interaction_open=True,
                commitment_closed=True,
                cid="cid-audit-1",
                governed_response="Revise.",
            ),
            GovernanceDecision(
                action=InterventionAction.SOFT_CORRECT,
                flags=[],
                rationale="audit-2",
                policy="strict",
                pathway_id=None,
                cid="cid-audit-2",
                governed_response="Soft.",
            ),
        ]
    )
    adapter = CountingMockAdapter(["A", "B"])
    cfg = LensConfig(
        adapter=adapter,
        extraction_backend=EmptyExtractBackend(),
        governance_bridge=bridge,
        auto_verify=False,
    )
    lens = Lens(cfg)
    return FixtureRuntime(lens=lens, adapter=adapter, audit_path=None, recording_bridge=bridge)


def _fx_rec_builtin_hello(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["Hello world."])
    return _lens_builtin(adapter, ap, extraction=EmptyExtractBackend(), auto_verify=True)


def _fx_rec_blocked_audit(ap: Path | None, case: dict[str, Any]) -> FixtureRuntime:
    adapter = CountingMockAdapter(["FORBIDDEN_MODEL_BODY"])
    return _lens_builtin(adapter, ap, extraction=EmptyExtractBackend(), auto_verify=False)
