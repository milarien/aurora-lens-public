"""Frame detector: classify user input as EXTERNAL or ARTIFACT.

Detection is deterministic and explicit. Only unambiguous roleplay/fiction/
hypothetical openers and clear frame-closers are matched. No inference of
subtle fictionality, author intent, or narrative semantics.

Opener matching: lowercased prefix (first 120 chars).
Closer matching: full lowercased text (explicit out-of-character markers).
"""

from __future__ import annotations

from aurora_lens.pef.artifact_layer import FrameKind

_PREFIX_LIMIT = 120

_ARTIFACT_OPENERS: frozenset[str] = frozenset({
    "let's pretend",
    "let us pretend",
    "imagine you are",
    "imagine you're",
    "in this story",
    "in this scenario",
    "roleplay as",
    "role play as",
    "you are playing",
    "you're playing",
    "as a character",
    "as the character",
    "pretend you are",
    "pretend you're",
    "for this exercise",
    "hypothetically speaking",
})

_ARTIFACT_CLOSERS: frozenset[str] = frozenset({
    "end roleplay",
    "stop roleplay",
    "back to reality",
    "out of character",
    "end scene",
    "ooc:",
})


def classify_frame(user_text: str) -> FrameKind:
    """Return the frame kind implied by user_text.

    ARTIFACT when an explicit opener is found in the first 120 chars,
    or EXTERNAL when a closer appears anywhere in the text.
    When both opener and closer are present, closer wins (explicit exit).
    Returns EXTERNAL when neither is found (default frame).
    """
    lower = (user_text or "").lower()
    prefix = lower[:_PREFIX_LIMIT]

    is_closer = any(c in lower for c in _ARTIFACT_CLOSERS)
    if is_closer:
        return FrameKind.EXTERNAL

    is_opener = any(o in prefix for o in _ARTIFACT_OPENERS)
    if is_opener:
        return FrameKind.ARTIFACT

    return FrameKind.EXTERNAL


def detect_frame_transition(
    user_text: str,
    current_frame: FrameKind,
) -> FrameKind | None:
    """Return the new frame kind if user_text contains an explicit frame signal, else None.

    Returns None when there is no explicit opener or closer (frame persists unchanged).
    Only explicit closers trigger EXTERNAL; neutral input never exits an ARTIFACT frame.
    """
    lower = (user_text or "").lower()
    prefix = lower[:_PREFIX_LIMIT]

    is_closer = any(c in lower for c in _ARTIFACT_CLOSERS)
    if is_closer:
        return FrameKind.EXTERNAL if current_frame != FrameKind.EXTERNAL else None

    is_opener = any(o in prefix for o in _ARTIFACT_OPENERS)
    if is_opener:
        return FrameKind.ARTIFACT if current_frame != FrameKind.ARTIFACT else None

    return None
