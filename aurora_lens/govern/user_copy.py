"""Governed wording for **people-facing** assistant responses (release text).

Internal diagnostics, audit rows, flag ``claim`` fields, and operator APIs may still
use engineer-oriented vocabulary elsewhere.
"""

# Substrate could not interpret user or model text enough to continue safely.
USER_MESSAGE_INTERPRETATION_FAILED = (
    "I couldn't reliably identify the facts I needed from that message. "
    "Please restate them one at a time."
)

# Response had little usable substance to check (barren parse / too thin).
USER_MESSAGE_RESPONSE_INSUFFICIENT_DETAIL_OPEN = (
    "I couldn't pick out specific, checkable details in that message. "
    "Please restate what you need in shorter, simpler sentences."
)

USER_MESSAGE_RESPONSE_INSUFFICIENT_DETAIL_CLOSED = (
    "I couldn't pick out specific, checkable details in that message. "
    "Please try again with a shorter, clearer message."
)
