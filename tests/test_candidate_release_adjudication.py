"""Structured candidate-release adjudication at the execution boundary."""

from __future__ import annotations

import pytest

from aurora_lens.adapters.base import AdapterResponse, LLMAdapter
from aurora_lens.config import LensConfig
from aurora_lens.context import execution_task_var
from aurora_lens.execution_task import (
    CandidateReleaseTask,
    EvidenceState,
    GoverningPolicy,
    NonEstablishmentRecord,
    TASK_ADJUDICATE_CANDIDATE_RELEASE,
    parse_execution_task,
    parse_execution_task_from_body,
)
from aurora_lens.govern.candidate_release import (
    FAILED_CONSTRAINT_PREDICTIVE_CLAIM_NOT_ESTABLISHED,
    FAILED_CONSTRAINT_UNSUPPORTED_EVENT,
    adjudicate_candidate_release,
)
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens
from aurora_lens.proxy.openai_compat import parse_chat_request
from aurora_lens.verify.flags import FlagType


class _CountingAdapter(LLMAdapter):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, messages, **kwargs) -> AdapterResponse:
        self.calls += 1
        return AdapterResponse(text="should not run", model="mock")


def _dam_block_task() -> CandidateReleaseTask:
    return CandidateReleaseTask(
        task_type=TASK_ADJUDICATE_CANDIDATE_RELEASE,
        candidate_release="The dam will fail.",
        evidence_state=EvidenceState(
            observations=(),
            established_claims=(),
            non_establishment=(
                NonEstablishmentRecord(
                    claim="the dam will fail",
                    basis="no_observation_establishes_failure",
                ),
            ),
        ),
        governing_policy=GoverningPolicy(require_present_evidence=True),
    )


class TestParseExecutionTask:
    def test_parse_from_body_top_level(self):
        body = {
            "messages": [{"role": "user", "content": "x"}],
            "execution_task": {
                "task_type": "adjudicate_candidate_release",
                "candidate_release": "The dam will fail.",
                "evidence_state": {
                    "non_establishment": [{"claim": "the dam will fail"}],
                },
                "governing_policy": {"require_present_evidence": True},
            },
        }
        task = parse_execution_task_from_body(body)
        assert task is not None
        assert task.candidate_release == "The dam will fail."

    def test_ignores_unknown_task_type(self):
        assert parse_execution_task({"task_type": "other", "candidate_release": "x"}) is None


class TestAdjudicateCandidateRelease:
    def test_blocks_predictive_candidate_with_structured_non_establishment(self):
        adj = adjudicate_candidate_release(_dam_block_task())
        assert adj.admitted is False
        assert adj.failed_constraint == FAILED_CONSTRAINT_PREDICTIVE_CLAIM_NOT_ESTABLISHED
        assert adj.response_text == (
            "The requested release cannot be admitted because the current evidence "
            "does not establish that the dam will fail."
        )

    def test_blocks_factual_candidate_with_unsupported_event(self):
        task = CandidateReleaseTask(
            task_type=TASK_ADJUDICATE_CANDIDATE_RELEASE,
            candidate_release="The dam failed yesterday.",
            evidence_state=EvidenceState(
                non_establishment=(
                    NonEstablishmentRecord(
                        claim="the dam failed yesterday",
                        basis="no_observation_establishes_failure",
                    ),
                ),
            ),
            governing_policy=GoverningPolicy(require_present_evidence=True),
        )
        adj = adjudicate_candidate_release(task)
        assert adj.admitted is False
        assert adj.failed_constraint == FAILED_CONSTRAINT_UNSUPPORTED_EVENT

    def test_admits_when_established_claim_matches(self):
        task = CandidateReleaseTask(
            task_type=TASK_ADJUDICATE_CANDIDATE_RELEASE,
            candidate_release="The dam will fail.",
            evidence_state=EvidenceState(established_claims=("the dam will fail",)),
            governing_policy=GoverningPolicy(),
        )
        adj = adjudicate_candidate_release(task)
        assert adj.admitted is True
        assert adj.response_text == "The dam will fail."


class TestLensCandidateReleasePath:
    @pytest.mark.asyncio
    async def test_pre_llm_block_without_upstream_or_clarification(self):
        adapter = _CountingAdapter()
        lens = Lens(
            LensConfig(
                adapter=adapter,
                extraction_backend=SpacyBackend(model="en_core_web_sm"),
                auto_verify=True,
            )
        )
        token = execution_task_var.set(_dam_block_task())
        try:
            result = await lens.process("(structured candidate-release adjudication)")
        finally:
            execution_task_var.reset(token)

        assert adapter.calls == 0
        assert result.action == InterventionAction.HARD_STOP
        assert result.response == (
            "The requested release cannot be admitted because the current evidence "
            "does not establish that the dam will fail."
        )
        assert result.telemetry_release_path == "blocked_before_generation"
        assert result.decision is not None
        assert result.decision.pathway_id == "P_STOP_REFUSE_CLEAN"
        assert result.decision.output_mode is None
        assert result.decision.interaction_open is False
        assert len(result.flags) == 1
        assert result.flags[0].flag_type == FlagType.PREDICTIVE_CLAIM_NOT_ESTABLISHED
        assert result.flags[0].claim == "The dam will fail."
        assert not any(f.flag_type == FlagType.UNRESOLVED_REFERENT for f in result.flags)
        assert lens.pef.pending_clarification is None
        assert lens.pef.entities == {}
        assert lens.pef.relationships == []

    @pytest.mark.asyncio
    async def test_prose_requested_release_prefix_does_not_trigger_without_task(self):
        adapter = _CountingAdapter()
        lens = Lens(
            LensConfig(
                adapter=adapter,
                extraction_backend=SpacyBackend(model="en_core_web_sm"),
                auto_verify=False,
            )
        )
        token = execution_task_var.set(None)
        try:
            result = await lens.process(
                'Requested release: The dam will fail.\n'
                "Evidence: no observation establishes failure."
            )
        finally:
            execution_task_var.reset(token)
        assert "The requested release cannot be admitted" not in (result.response or "")


class TestProxyParseChatRequest:
    def test_execution_task_parsed_on_chat_body(self):
        parsed = parse_chat_request(
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "boundary"}],
                "execution_task": {
                    "task_type": "adjudicate_candidate_release",
                    "candidate_release": "The dam will fail.",
                    "evidence_state": {
                        "non_establishment": [{"claim": "the dam will fail"}],
                    },
                    "governing_policy": {"require_present_evidence": True},
                },
            }
        )
        assert parsed.execution_task is not None
        assert parsed.execution_task.candidate_release == "The dam will fail."
