"""Structured candidate-release adjudication (pre-LLM execution boundary)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from aurora_lens.execution_task import CandidateReleaseTask
from aurora_lens.verify.flags import FlagType

FAILED_CONSTRAINT_PREDICTIVE_CLAIM_NOT_ESTABLISHED = "PREDICTIVE_CLAIM_NOT_ESTABLISHED"
FAILED_CONSTRAINT_UNSUPPORTED_EVENT = "UNSUPPORTED_EVENT"

# Align with checker predictive/modal guard — projections are not established facts.
_PREDICTIVE_SURFACE_RE = re.compile(
    r"\b(?:will|would|may|might|could|should|"
    r"expect|expects|expected|"
    r"forecast|forecasts|"
    r"project|projects|projected|"
    r"anticipate|anticipates|anticipated|"
    r"estimate|estimates|estimated|"
    r"target|targets)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CandidateReleaseAdjudication:
    admitted: bool
    response_text: str
    failed_constraint: str | None = None
    rationale: str = ""


def normalize_claim_phrase(text: str) -> str:
    """Normalize a claim surface for structured equality checks (no regex triggers)."""
    s = " ".join(str(text or "").strip().lower().split())
    if s.startswith("the "):
        s = s[4:]
    return s.rstrip(".")


def _claim_phrase_for_message(candidate_release: str) -> str:
    return candidate_release.strip().rstrip(".").lower()


def _block_message(candidate_release: str) -> str:
    phrase = _claim_phrase_for_message(candidate_release)
    return (
        "The requested release cannot be admitted because the current evidence "
        f"does not establish that {phrase}."
    )


def is_predictive_candidate_surface(text: str) -> bool:
    """True when candidate_release surface is modal/predictive (not an established fact)."""
    return bool(_PREDICTIVE_SURFACE_RE.search(str(text or "")))


def failed_constraint_for_blocked_candidate(candidate_release: str) -> str:
    """Map blocked candidate_release to forensic failed_constraint name."""
    if is_predictive_candidate_surface(candidate_release):
        return FAILED_CONSTRAINT_PREDICTIVE_CLAIM_NOT_ESTABLISHED
    return FAILED_CONSTRAINT_UNSUPPORTED_EVENT


def failed_constraint_to_flag_type(failed_constraint: str | None) -> FlagType:
    """Resolve structured failed_constraint to verification FlagType."""
    if failed_constraint == FAILED_CONSTRAINT_PREDICTIVE_CLAIM_NOT_ESTABLISHED:
        return FlagType.PREDICTIVE_CLAIM_NOT_ESTABLISHED
    return FlagType.UNSUPPORTED_EVENT


def adjudicate_candidate_release(task: CandidateReleaseTask) -> CandidateReleaseAdjudication:
    """Decide admissibility from structured evidence_state and governing_policy only."""
    candidate_norm = normalize_claim_phrase(task.candidate_release)

    for established in task.evidence_state.established_claims:
        if normalize_claim_phrase(established) == candidate_norm:
            return CandidateReleaseAdjudication(
                admitted=True,
                response_text=task.candidate_release.strip(),
                rationale=(
                    "CANDIDATE_RELEASE_ADMITTED: structured evidence_state establishes "
                    "the requested release."
                ),
            )

    failed_fc = failed_constraint_for_blocked_candidate(task.candidate_release)

    for record in task.evidence_state.non_establishment:
        if normalize_claim_phrase(record.claim) == candidate_norm:
            return CandidateReleaseAdjudication(
                admitted=False,
                response_text=_block_message(task.candidate_release),
                failed_constraint=failed_fc,
                rationale=(
                    "CANDIDATE_RELEASE_BLOCKED: structured evidence_state marks the "
                    "requested release as not established by present observation."
                ),
            )

    if task.governing_policy.require_present_evidence:
        return CandidateReleaseAdjudication(
            admitted=False,
            response_text=_block_message(task.candidate_release),
            failed_constraint=failed_fc,
            rationale=(
                "CANDIDATE_RELEASE_BLOCKED: governing_policy requires present evidence "
                "and no establishing observation was supplied."
            ),
        )

    return CandidateReleaseAdjudication(
        admitted=True,
        response_text=task.candidate_release.strip(),
        rationale="CANDIDATE_RELEASE_ADMITTED: governing_policy permits release.",
    )
