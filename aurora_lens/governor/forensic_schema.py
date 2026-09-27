"""Canonical forensic envelope schema for Aurora-Lens non-ADMIT outcomes.

SCHEMA VERSION
--------------
FORENSIC_SCHEMA_VERSION = "1.0"

Every ASK / REFUSE / STOP governance decision must produce a forensic event
that satisfies this schema.  The schema is:

  REQUIRED (always present in a schema-complete event):
    schema_version      str   — must equal FORENSIC_SCHEMA_VERSION
    trace_id            str   — request correlation ID (may be "")
    timestamp           str   — ISO-8601 UTC timestamp (may be "")
    status              str   — "ASK" | "REFUSE" | "STOP"
    attempted_action    str   — "call_upstream" | "respond"
    failed_constraints  list  — non-empty list of FlagType name strings
    pathway_id          str   — must be a known ContinuationPathway value
    output_mode         str   — must be a known OutputMode value
    commitment_closed   bool  — must be True (non-ADMIT invariant)
    interaction_open    bool
    forensic_obligations list — list of ForensicObligation value strings
    resolution_mode     str   — ResolutionMode value string
    event_hash          str   — "sha256:<64-hex>" self-hash of all other fields

  OPTIONAL (present when the pipeline has the information):
    chain_of_custody    dict  — evidentiary packaging (code/policy/runtime provenance); included in event_hash
    session_id          str | None  — conversation id (ContextVar / proxy); mirrors outer audit row
    domain              str | None
    subdomain           str | None
    state_hash          str | None  — PEF state hash
    escalation_target   str | None  — effective resource surfaced to user
    blocked_response_hash   str     — "sha256:<64-hex>" of suppressed response
    governed_response_hash  str     — "sha256:<64-hex>" of rendered output

SELF-INTEGRITY
--------------
event_hash is a SHA-256 of the canonical JSON of all other fields:
    json.dumps(event_without_hash, separators=(',',':'), sort_keys=True, ensure_ascii=False)

REPLAY CONTRACT
---------------
An event is replay-sufficient if it contains enough information to:
  1. Identify the matrix row (pathway_id + continuation fields)
  2. Reconstruct the governance decision inputs (pathway_id, failed_constraints,
     interaction_open, escalation_target)
  3. Re-render the governed response via enforce()
  4. Verify governed_response_hash == sha256(re-rendered output)
  5. Verify the event_hash has not been tampered with

ENRICHED EVENTS
---------------
On CanonicalScannerGateBridge, build_forensic_event runs first, then
enrich_forensic_envelope and pathway/context fields are applied, then
refresh_forensic_event_hash recomputes event_hash over all fields except
event_hash.  The stored hash therefore covers Governor enrichment,
not only the pre-enrichment base.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .continuation_matrix import ContinuationRow, _TABLE
from .models import ContinuationPathway, LensStatus, OutputMode


# ── Version ───────────────────────────────────────────────────────────────────

# Must equal aurora_lens.govern.bridge.FORENSIC_SCHEMA_VERSION.
# Defined independently to avoid a circular/subprocess-path import.
FORENSIC_SCHEMA_VERSION = "1.0"


# ── Field sets ────────────────────────────────────────────────────────────────

#: Fields that MUST be present in every schema-complete forensic event.
REQUIRED_FIELDS: frozenset[str] = frozenset({
    "schema_version",
    "trace_id",
    "timestamp",
    "status",
    "attempted_action",
    "failed_constraints",
    "pathway_id",
    "output_mode",
    "commitment_closed",
    "interaction_open",
    "forensic_obligations",
    "resolution_mode",
    "event_hash",
})

#: Fields that MAY be present (pipeline-conditional).
OPTIONAL_FIELDS: frozenset[str] = frozenset({
    "session_id",
    "domain",
    "subdomain",
    "state_hash",
    "escalation_target",
    "blocked_response_hash",
    "governed_response_hash",
    # Evidentiary packaging (chain-of-custody); included in event_hash when present.
    "chain_of_custody",
    # Phase 7: typed transaction lifecycle list; covered by event_hash when present.
    # Schema: list[{"transaction_id": str, "lifecycle": list[LifecycleEvent dict]}]
    "transactions",
})

VALID_STATUSES: frozenset[str] = frozenset({"ASK", "REFUSE", "STOP"})
VALID_ATTEMPTED_ACTIONS: frozenset[str] = frozenset({"call_upstream", "respond"})

_HASH_PREFIX = "sha256:"
_HASH_FIELD_LEN = len(_HASH_PREFIX) + 64   # "sha256:" + 64 lower-hex digits = 71
_VALID_PATHWAY_VALUES: frozenset[str] = frozenset(p.value for p in ContinuationPathway)
_VALID_OUTPUT_MODE_VALUES: frozenset[str] = frozenset(m.value for m in OutputMode)


# ── Validation ────────────────────────────────────────────────────────────────

def validate(event: dict[str, Any]) -> list[str]:
    """Validate a forensic event dict against the schema.

    Returns a list of error strings.  An empty list means the event is
    schema-complete.  Does NOT verify event_hash integrity (use
    verify_event_hash() for that).

    Args:
        event: A forensic event dict, as produced by build_forensic_event().

    Returns:
        List of human-readable error strings.  Empty == valid.
    """
    errors: list[str] = []

    # 1. Required fields present.
    for field in sorted(REQUIRED_FIELDS):
        if field not in event:
            errors.append(f"Missing required field: {field!r}")

    # 2. schema_version.
    if "schema_version" in event:
        if event["schema_version"] != FORENSIC_SCHEMA_VERSION:
            errors.append(
                f"schema_version={event['schema_version']!r}, "
                f"expected {FORENSIC_SCHEMA_VERSION!r}"
            )

    # 3. status.
    if "status" in event:
        if event["status"] not in VALID_STATUSES:
            errors.append(
                f"status={event['status']!r} is not one of {sorted(VALID_STATUSES)}"
            )

    # 4. attempted_action.
    if "attempted_action" in event:
        if event["attempted_action"] not in VALID_ATTEMPTED_ACTIONS:
            errors.append(
                f"attempted_action={event['attempted_action']!r} is not one of "
                f"{sorted(VALID_ATTEMPTED_ACTIONS)}"
            )

    # 5. failed_constraints: non-empty list.
    if "failed_constraints" in event:
        fc = event["failed_constraints"]
        if not isinstance(fc, list):
            errors.append("failed_constraints must be a list")
        elif not fc:
            errors.append("failed_constraints must be non-empty")

    # 6. pathway_id: must be a known ContinuationPathway value.
    if "pathway_id" in event and event["pathway_id"] is not None:
        if event["pathway_id"] not in _VALID_PATHWAY_VALUES:
            errors.append(f"pathway_id={event['pathway_id']!r} is not a known ContinuationPathway")

    # 7. output_mode: must be a known OutputMode value.
    if "output_mode" in event and event["output_mode"] is not None:
        if event["output_mode"] not in _VALID_OUTPUT_MODE_VALUES:
            errors.append(f"output_mode={event['output_mode']!r} is not a known OutputMode")

    # 8. commitment_closed: must be bool True (non-ADMIT invariant).
    if "commitment_closed" in event:
        if not isinstance(event["commitment_closed"], bool):
            errors.append("commitment_closed must be a bool")
        elif not event["commitment_closed"]:
            errors.append(
                "commitment_closed must be True for non-ADMIT forensic events "
                "(Governor non-widening invariant)"
            )

    # 9. interaction_open: must be bool.
    if "interaction_open" in event:
        if not isinstance(event["interaction_open"], bool):
            errors.append("interaction_open must be a bool")

    # 10. forensic_obligations: must be a list.
    if "forensic_obligations" in event:
        if not isinstance(event["forensic_obligations"], list):
            errors.append("forensic_obligations must be a list")

    # 11. session_id: optional str or null.
    if "session_id" in event and event["session_id"] is not None:
        if not isinstance(event["session_id"], str):
            errors.append("session_id must be a string or null")

    # 11b. transactions (Phase 7): optional list of transaction lifecycle dicts.
    if "transactions" in event and event["transactions"] is not None:
        if not isinstance(event["transactions"], list):
            errors.append("transactions must be a list")
        else:
            for i, txn in enumerate(event["transactions"]):
                if not isinstance(txn, dict):
                    errors.append(f"transactions[{i}] must be a dict")
                    continue
                if "transaction_id" not in txn:
                    errors.append(f"transactions[{i}] missing 'transaction_id'")
                if "lifecycle" not in txn:
                    errors.append(f"transactions[{i}] missing 'lifecycle'")
                elif not isinstance(txn["lifecycle"], list):
                    errors.append(f"transactions[{i}]['lifecycle'] must be a list")

    # 12. Hash field format: "sha256:<64 lower-hex chars>".
    for hf in ("event_hash", "blocked_response_hash", "governed_response_hash"):
        if hf in event and event[hf] is not None:
            val = event[hf]
            if not isinstance(val, str):
                errors.append(f"{hf!r} must be a string")
            elif not val.startswith(_HASH_PREFIX) or len(val) != _HASH_FIELD_LEN:
                errors.append(
                    f"{hf!r}={val!r} must be '{_HASH_PREFIX}<64 hex chars>' "
                    f"(length {_HASH_FIELD_LEN})"
                )

    return errors


# ── Integrity verification ────────────────────────────────────────────────────

def verify_event_hash(event: dict[str, Any]) -> bool:
    """Recompute event_hash from all other fields and compare.

    The event_hash is a SHA-256 of the canonical JSON of the event dict
    with event_hash excluded, using sort_keys=True and no extra whitespace.

    Returns True iff the stored event_hash matches the recomputed value.
    Returns False if event_hash is absent or the event has been tampered with.
    """
    stored = event.get("event_hash")
    if not stored:
        return False
    candidate = {k: v for k, v in event.items() if k != "event_hash"}
    computed = _HASH_PREFIX + hashlib.sha256(
        json.dumps(
            candidate, separators=(",", ":"), sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    return stored == computed


def verify_governed_response_hash(event: dict[str, Any], rendered_text: str) -> bool:
    """Verify that governed_response_hash matches the SHA-256 of rendered_text.

    Returns True iff:
      - event contains "governed_response_hash"
      - sha256(rendered_text.encode("utf-8")) matches the stored hash

    Returns False if governed_response_hash is absent.
    """
    stored = event.get("governed_response_hash")
    if not stored:
        return False
    expected = _HASH_PREFIX + hashlib.sha256(
        rendered_text.encode("utf-8")
    ).hexdigest()
    return stored == expected


def verify_blocked_response_hash(event: dict[str, Any], blocked_text: str) -> bool:
    """Verify that blocked_response_hash matches the SHA-256 of blocked_text.

    Returns True iff the stored hash matches. Returns False if absent.
    """
    stored = event.get("blocked_response_hash")
    if not stored:
        return False
    expected = _HASH_PREFIX + hashlib.sha256(
        blocked_text.encode("utf-8")
    ).hexdigest()
    return stored == expected


# ── Row identity lookup ───────────────────────────────────────────────────────

def find_matrix_rows(event: dict[str, Any]) -> list[tuple[str, ContinuationRow]]:
    """Find all _TABLE rows whose continuation fields are consistent with event.

    Matches on pathway_id, commitment_closed, interaction_open, and output_mode.
    Returns a list of (key, row) pairs — may be multiple if different domain/
    authority combinations share the same continuation parameters.

    An empty result means no matrix row is consistent with the event, which
    indicates a schema violation or a row that has since been removed.
    """
    pid   = event.get("pathway_id")
    cc    = event.get("commitment_closed")
    io    = event.get("interaction_open")
    om    = event.get("output_mode")

    results: list[tuple[str, ContinuationRow]] = []
    for key, row in _TABLE.items():
        if pid is not None and row.pathway_id.value != pid:
            continue
        if cc is not None and row.commitment_closed != cc:
            continue
        if io is not None and row.interaction_open != io:
            continue
        if om is not None and row.output_mode.value != om:
            continue
        results.append((key, row))
    return results
