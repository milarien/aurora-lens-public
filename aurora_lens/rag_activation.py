"""When retrieval-aware RAG referent handling runs (request structure + env overrides).

Ordinary chat keeps the default non-RAG ambiguity path. Evidence-bearing requests
(``request_metadata`` and/or ``Context:`` … ``Question:`` harness shape) activate
the retrieval-aware path without requiring a manual env flag.

Environment (``AURORA_LENS_RAG_RETRIEVAL_AWARE_REFERENTS``):

- **Force off:** ``0``, ``false``, ``no``, ``off`` — disables the feature for every
  request (rollback).
- **Force on:** ``1``, ``true``, ``yes``, ``on`` — enables for every request
  (legacy / full-deployment opt-in).
- **Unset or any other value:** **auto** — use :func:`effective_rag_retrieval_aware_referents`.
"""

from __future__ import annotations

import os

from aurora_lens.context import get_request_metadata
from aurora_lens.request_metadata import RequestMetadata


def _env_rag_retrieval_mode() -> str:
    """Return ``force_off`` | ``force_on`` | ``auto``."""
    raw = os.environ.get("AURORA_LENS_RAG_RETRIEVAL_AWARE_REFERENTS", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return "force_off"
    if raw in ("1", "true", "yes", "on"):
        return "force_on"
    return "auto"


def evidence_bearing_request_metadata(meta: RequestMetadata | None) -> bool:
    """True when host metadata indicates an evidence-grounded / retrieval-scoped request.

    ``workspace_id`` alone does not activate; non-empty ``record_ids`` or
    ``source_scope`` does (bounded scope for a workspace).
    """
    if meta is None:
        return False
    if meta.record_ids:
        return True
    if meta.source_scope:
        return True
    return False


def effective_rag_retrieval_aware_referents(
    user_input: str,
    *,
    config_rag: bool = False,
) -> bool:
    """Whether to run the Context:/Question seeding and question-line referent path.

    * ``config_rag``: :attr:`LensConfig.rag_retrieval_aware_referents` — programmatic
      force-on for tests/scripts when env is not ``force_off``.
    """
    mode = _env_rag_retrieval_mode()
    if mode == "force_off":
        return False
    if mode == "force_on":
        return True
    if config_rag:
        return True
    if evidence_bearing_request_metadata(get_request_metadata()):
        return True
    # Lazy import: rag_activation is imported from lens; avoid circular module load.
    from aurora_lens.lens import split_rag_context_question

    if split_rag_context_question(user_input) is not None:
        return True
    return False
