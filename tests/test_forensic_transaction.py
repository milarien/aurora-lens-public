"""Forensic transaction lifecycle tests — Phase 7.

Invariants protected:
- Every committed Relationship carries a transaction_id traceable to its transaction.
- A 'replayed' lifecycle event always follows a 'held' event for the same transaction_id.
- A refused transaction must not have a subsequent 'committed' event (enforced by record()).
- The forensic verify_event_hash digest is broken if the 'transactions' list is mutated.
- A PEF fact's relation_metadata['transaction_id'] links back to the ledger entry.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from aurora_lens.interpret.schema import (
    ExtractedClaim,
    LifecycleEvent,
    LIFECYCLE_KINDS,
    PossessionTransaction,
    SemanticTransaction,
)
from aurora_lens.governor.forensic_schema import verify_event_hash
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState


# ── Helpers ───────────────────────────────────────────────────────────────────


def _claim(subject: str = "James", obj: str = "10 apples") -> ExtractedClaim:
    return ExtractedClaim(
        subject=subject, relation="HAS", obj=obj,
        span=Span.PRESENT, negated=False, evidence="",
    )


def _pef_with(name: str) -> PEFState:
    pef = PEFState()
    pef.get_or_create_entity(name)
    return pef


def _forensic_event_with_transactions(transactions: list[dict]) -> dict:
    """Minimal forensic event dict with a transactions key and a valid event_hash."""
    event: dict = {
        "schema_version": "1.0",
        "transactions": transactions,
    }
    canonical = {k: v for k, v in event.items() if k != "event_hash"}
    digest = "sha256:" + hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    event["event_hash"] = digest
    return event


# ── LIFECYCLE_KINDS constant ──────────────────────────────────────────────────


def test_lifecycle_kinds_complete():
    """All six lifecycle stages are exported."""
    assert LIFECYCLE_KINDS == {"proposed", "held", "replayed", "committed", "refused", "answered"}


# ── SemanticTransaction.transaction_id ───────────────────────────────────────


def test_semantic_transaction_has_unique_transaction_id():
    """Every SemanticTransaction gets a distinct UUID4 transaction_id at construction."""
    tx1 = SemanticTransaction(claim=_claim())
    tx2 = SemanticTransaction(claim=_claim())
    assert tx1.transaction_id != tx2.transaction_id
    assert len(tx1.transaction_id) == 36   # UUID4 canonical form


def test_semantic_transaction_lifecycle_starts_empty():
    """lifecycle is an empty list at construction."""
    tx = SemanticTransaction(claim=_claim())
    assert tx.lifecycle == []


# ── Law: committed Relationship carries transaction_id ────────────────────────


def test_committed_relationship_carries_transaction_id():
    """PossessionTransaction.commit() stamps transaction_id on every created Relationship."""
    pef = _pef_with("James")
    tx = PossessionTransaction(
        claim=_claim("James", "10 apples"),
        mutation_kind="acquire",
    )
    tx.commit(pef)

    stamped = [
        r for r in pef.relationships
        if r.relation_metadata and "transaction_id" in r.relation_metadata
    ]
    assert len(stamped) >= 1, "At least one committed relationship must carry transaction_id"
    for rel in stamped:
        assert rel.relation_metadata["transaction_id"] == tx.transaction_id


def test_commit_records_committed_lifecycle_event():
    """commit() automatically appends a 'committed' LifecycleEvent."""
    pef = _pef_with("Alice")
    tx = PossessionTransaction(
        claim=_claim("Alice", "wallet"),
        mutation_kind="acquire",
    )
    tx.commit(pef)

    assert any(e.kind == "committed" for e in tx.lifecycle)
    committed = next(e for e in tx.lifecycle if e.kind == "committed")
    assert committed.turn == pef.current_turn


# ── Law: held before replayed ─────────────────────────────────────────────────


def test_lifecycle_held_before_replayed():
    """'replayed' event must follow a 'held' event for the same transaction."""
    tx = SemanticTransaction(
        claim=_claim("he", "wallet"),
        allowed_commit=False,
        held_reason="UNRESOLVED_REFERENT",
    )
    tx.record("proposed", turn=1)
    tx.record("held", turn=1, reason="UNRESOLVED_REFERENT")
    tx.record("replayed", turn=2)

    kinds = [e.kind for e in tx.lifecycle]
    assert "held" in kinds
    assert "replayed" in kinds
    held_idx = kinds.index("held")
    replayed_idx = kinds.index("replayed")
    assert held_idx < replayed_idx, "'held' must precede 'replayed' in lifecycle"


# ── Law: refused transaction must not commit ──────────────────────────────────


def test_refused_transaction_never_committed():
    """record('committed') after record('refused') raises ValueError."""
    tx = SemanticTransaction(
        claim=_claim(),
        allowed_commit=False,
    )
    tx.record("proposed", turn=1)
    tx.record("refused", turn=1, reason="authority_check_failed")

    with pytest.raises(ValueError, match="refused"):
        tx.record("committed", turn=1)

    # 'refused' is in lifecycle; 'committed' is not
    kinds = [e.kind for e in tx.lifecycle]
    assert "refused" in kinds
    assert "committed" not in kinds


# ── Law: forensic hash covers transactions list ───────────────────────────────


def test_forensic_hash_covers_transaction_list():
    """verify_event_hash() is False when the transactions list is mutated after signing."""
    event = _forensic_event_with_transactions([
        {"transaction_id": "txn-001", "lifecycle": [{"kind": "committed", "turn": 1}]},
    ])

    # Before mutation: hash is valid
    assert verify_event_hash(event) is True

    # Mutate: inject a new transaction — hash must break
    event["transactions"].append({"transaction_id": "injected", "lifecycle": []})
    assert verify_event_hash(event) is False


def test_forensic_hash_covers_lifecycle_mutation():
    """verify_event_hash() is False when a lifecycle entry inside transactions is mutated."""
    event = _forensic_event_with_transactions([
        {"transaction_id": "txn-A", "lifecycle": [{"kind": "proposed", "turn": 1}]},
    ])
    assert verify_event_hash(event) is True

    # Mutate a lifecycle entry
    event["transactions"][0]["lifecycle"][0]["kind"] = "committed"
    assert verify_event_hash(event) is False


# ── Law: ledger entry traceable to PEF fact ──────────────────────────────────


def test_ledger_entry_traceable_to_pef_fact():
    """A committed Relationship's transaction_id links to the simulated ledger entry."""
    pef = _pef_with("James")
    tx = PossessionTransaction(
        claim=_claim("James", "5 lollipops"),
        mutation_kind="acquire",
    )
    tx.record("proposed", turn=1)
    tx.commit(pef)

    # Find the stamped relationship
    stamped = [
        r for r in pef.relationships
        if r.relation_metadata and r.relation_metadata.get("transaction_id") == tx.transaction_id
    ]
    assert len(stamped) >= 1

    rel_tx_id = stamped[0].relation_metadata["transaction_id"]

    # Simulate what a ledger entry would contain
    ledger_entry = {
        "transactions": [
            {
                "transaction_id": tx.transaction_id,
                "lifecycle": [
                    {"kind": e.kind, "turn": e.turn, "reason": e.reason}
                    for e in tx.lifecycle
                ],
            }
        ]
    }

    # The PEF fact's transaction_id appears in the ledger
    ledger_ids = {t["transaction_id"] for t in ledger_entry["transactions"]}
    assert rel_tx_id in ledger_ids

    # The ledger entry's lifecycle includes a 'committed' event
    ledger_tx = next(t for t in ledger_entry["transactions"] if t["transaction_id"] == rel_tx_id)
    assert any(e["kind"] == "committed" for e in ledger_tx["lifecycle"])


# ── Phase 7 forensic schema: validate() accepts transactions ──────────────────


def test_forensic_schema_validate_accepts_valid_transactions():
    """validate() produces no errors for a well-formed transactions list."""
    from aurora_lens.governor.forensic_schema import validate

    event = {
        "transactions": [
            {
                "transaction_id": "uuid-abc",
                "lifecycle": [{"kind": "proposed", "turn": 1}, {"kind": "committed", "turn": 1}],
            }
        ]
    }
    # Only validating the transactions sub-schema here; rest of required fields absent
    # (validate checks each field independently when present)
    errors = validate(event)
    transaction_errors = [e for e in errors if "transaction" in e.lower()]
    assert transaction_errors == [], f"Unexpected transaction validation errors: {transaction_errors}"


def test_forensic_schema_validate_rejects_malformed_transactions():
    """validate() reports errors for malformed transactions entries."""
    from aurora_lens.governor.forensic_schema import validate

    event = {
        "transactions": [
            {"lifecycle": []},          # missing transaction_id
            "not-a-dict",               # wrong type
            {"transaction_id": "x"},    # missing lifecycle
        ]
    }
    errors = validate(event)
    assert any("transaction_id" in e for e in errors)
    assert any("must be a dict" in e for e in errors)
    assert any("lifecycle" in e for e in errors)


# ── PEF snapshot hash in record() ────────────────────────────────────────────


def test_record_with_pef_computes_snapshot_hash():
    """record() with pef= captures a 'sha256:' snapshot hash."""
    pef = _pef_with("James")
    tx = SemanticTransaction(claim=_claim())
    tx.record("proposed", turn=1, pef=pef)

    event = tx.lifecycle[0]
    assert event.pef_snapshot_hash is not None
    assert event.pef_snapshot_hash.startswith("sha256:")
    assert len(event.pef_snapshot_hash) == 71   # "sha256:" + 64 hex


def test_record_without_pef_has_no_snapshot_hash():
    """record() without pef= leaves pef_snapshot_hash as None."""
    tx = SemanticTransaction(claim=_claim())
    tx.record("proposed", turn=1)
    assert tx.lifecycle[0].pef_snapshot_hash is None
