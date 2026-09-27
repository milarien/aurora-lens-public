"""Opt-in diagnostics for PEF admission (set ``AURORA_DEBUG_PEF_ADMISSION=1``)."""

from __future__ import annotations

import logging
import os

_LOG = logging.getLogger("aurora_lens.pef_admission")


def pef_admission_debug_enabled() -> bool:
    v = os.environ.get("AURORA_DEBUG_PEF_ADMISSION", "").strip().lower()
    return v in ("1", "true", "yes")


def log_update_pef_enter(
    *,
    user_text: str | None,
    claims_in: int,
    skip_all_claims: bool,
) -> None:
    if not pef_admission_debug_enabled():
        return
    _LOG.info(
        "pef_admission update_pef enter user_text_is_none=%s user_text_len=%s "
        "claims_in=%s skip_all_claims=%s",
        user_text is None,
        0 if user_text is None else len(user_text),
        claims_in,
        skip_all_claims,
    )


def log_update_pef_exit(*, claims_committed: int, claims_skipped_by_loop: int) -> None:
    if not pef_admission_debug_enabled():
        return
    _LOG.info(
        "pef_admission update_pef exit claims_committed=%s claims_skipped_conflict=%s",
        claims_committed,
        claims_skipped_by_loop,
    )


def log_revision_gate(
    *,
    extraction_conflict: bool,
    user_text_conflict: bool,
    detail: str,
) -> None:
    if not pef_admission_debug_enabled():
        return
    _LOG.info(
        "pef_admission revision_gate extraction_conflict=%s user_text_conflict=%s detail=%r",
        extraction_conflict,
        user_text_conflict,
        detail[:300] if detail else "",
    )
