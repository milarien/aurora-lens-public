"""Streaming governance: private accumulation buffer and gated release.

STREAMING STATE MACHINE
-----------------------
Provider output is accumulated in a private buffer. No content escapes to
the user until Lens has made its admissibility decision on the complete text.

                         UPSTREAM_STREAMING
                               │
                               │  (all chunks received)
                               ▼
                         BUFFER_COMPLETE
                               │
                               │  (checker + governor run)
                               ▼
                           VERIFYING
                               │
               ┌───────────────┴───────────────┐
               │                               │
     ADMIT (PASS / SOFT_CORRECT)      NON-ADMIT (ASK / REFUSE / STOP)
               │                               │
               ▼                               ▼
       RELEASING_BUFFER               SUPPRESSING_BUFFER
    Buffered chunks emitted         Buffer permanently suppressed.
    as "chunk" events.              Governed continuation emitted
                                    as single "governed_chunk" event.
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
                        METADATA_EMITTED
                               │
                               ▼
                           COMPLETE

CONSTITUTIONAL INVARIANTS
--------------------------
- No content escapes the buffer before the admissibility decision.
- Progress signals contain no substantive commitment and never echo content
  from the buffer or any field derived from user input / LLM output.
- Non-ADMIT paths produce identical forensic artifacts to non-streaming paths:
  blocked_response_hash is sha256 of the suppressed buffer;
  governed_response_hash is sha256 of the governed continuation.
- No semantic smuggling via progress signals: only a lifecycle phase label
  ("streaming" | "verifying" | "releasing") is permitted.
- The suppression decision is final: once SUPPRESSING_BUFFER is entered,
  no fragment of the buffer may appear in any subsequent user-visible output.
"""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class StreamPhase(Enum):
    """Phase of the governed stream state machine."""
    UPSTREAM_STREAMING  = auto()  # Accumulating provider chunks (private)
    BUFFER_COMPLETE     = auto()  # All chunks received; governance not yet run
    VERIFYING           = auto()  # Checker + governor evaluating full buffer
    RELEASING_BUFFER    = auto()  # ADMIT: emitting buffered chunks to user
    SUPPRESSING_BUFFER  = auto()  # NON-ADMIT: buffer suppressed, governed emitted
    COMPLETE            = auto()  # Final metadata emitted
    ABORTED             = auto()  # Provider stream aborted before buffer complete


@dataclass
class BufferedStreamResult:
    """The complete, privately buffered output from an upstream provider stream.

    Produced by the private accumulation loop in process_stream().  Never
    exposed to the user directly — the release controller decides what (if
    anything) to emit based on the governance decision.

    chunks: ordered list of (chunk_dict, content_delta) as delivered by the
            adapter.  chunk_dict is the raw OpenAI-compatible SSE payload;
            content_delta is the text fragment.  On ADMIT these are re-emitted
            to the user as "chunk" events, preserving the original streaming
            cadence.

    full_text: "".join(delta for _, delta in chunks) — the complete candidate
               response.  This is the text Lens verifies.
    """
    chunks: list[tuple[dict[str, Any], str]]
    full_text: str
    total_bytes: int
    truncated: bool = False
    dropped_chars: int = 0
    model: str = ""
    usage: dict[str, Any] | None = None


# ── Progress signal ────────────────────────────────────────────────────────────
#
# Valid status labels.  These are the ONLY strings permitted in a ProgressSignal.
# They identify a lifecycle phase — never a governance outcome, never content.

_VALID_PROGRESS_STATUSES: frozenset[str] = frozenset({
    "streaming",   # Provider stream in progress (buffer filling)
    "verifying",   # Buffer complete; Lens running checker + governor
    "releasing",   # Admissibility confirmed; buffer or governed output releasing
})


@dataclass(frozen=True)
class ProgressSignal:
    """Safe user-facing progress signal.

    Contains scaffolding status only.  Constitutional constraints:

    MUST NOT contain:
      - Any content from the provider buffer.
      - Any claim, inference, or implication about the buffer contents.
      - Any flag name, pathway name, or governance metadata that could leak the
        admissibility decision before it is made.
      - Any partial text that could constitute a substantive commitment.

    Allowed:
      - A lifecycle phase label (one of _VALID_PROGRESS_STATUSES).
      - Nothing else.  No dynamic interpolation.  No fields derived from user
        input, LLM output, or governance state.

    Consumers (e.g. the proxy SSE layer) may forward these as-is or suppress
    them.  They MUST NOT use a ProgressSignal to infer governance outcome before
    the "metadata" event arrives.
    """
    status: str  # one of _VALID_PROGRESS_STATUSES

    def __post_init__(self) -> None:
        if self.status not in _VALID_PROGRESS_STATUSES:
            raise ValueError(
                f"ProgressSignal.status must be one of "
                f"{sorted(_VALID_PROGRESS_STATUSES)!r}, got {self.status!r}"
            )

    def as_event_dict(self) -> dict[str, str]:
        """Serialise to the SSE payload dict for the 'progress' event kind."""
        return {"type": "aurora_progress", "status": self.status}


def make_governed_chunk_dict(governed_text: str, model: str = "") -> dict[str, Any]:
    """Construct a minimal OpenAI-compatible chunk_dict for governed output.

    Governed continuations are deterministic single-chunk responses.  We do not
    re-use the raw provider chunk_dicts here because:
      1. The provider chunks carry the suppressed candidate's delta structure.
      2. Re-using them would leak structural information about the suppressed text.
      3. The governed text is rendered from policy, not from the upstream model.

    The returned dict is schema-compatible with the OpenAI streaming chunk
    format so the proxy can forward it verbatim via SSE.
    """
    return {
        "id": f"gov-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(datetime.datetime.now(datetime.timezone.utc).timestamp()),
        "model": model or "aurora-governed",
        "choices": [
            {
                "index": 0,
                "delta": {"content": governed_text},
                "finish_reason": "stop",
            }
        ],
    }
