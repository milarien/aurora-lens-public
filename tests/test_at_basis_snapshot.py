"""at_basis_snapshot — recoverable AT head/prior for audit."""

import json

from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.govern.scanner_gate_bridge import AuroraScannerGateBridge
from aurora_lens.pef.at_read import AT_BASIS_SCHEMA_VERSION, at_basis_snapshot, current_at
from aurora_lens.pef.entity import Entity
from aurora_lens.pef.state import PEFState, Relationship
from aurora_lens.pef.span import Span


def test_at_basis_snapshot_matches_current_and_prior_indices():
    p = PEFState()
    key = Entity.create("gold key", 0, session_id="s")
    desk = Entity.create("desk", 0, session_id="s")
    study = Entity.create("study", 0, session_id="s")
    for e in (key, desk, study):
        p.add_entity(e)

    p.add_relationship(
        Relationship(
            subject_id=key.id,
            relation="AT",
            object_entity_id=desk.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="t1",
        )
    )
    p.add_relationship(
        Relationship(
            subject_id=key.id,
            relation="AT",
            object_entity_id=study.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=2,
            evidence="t2",
        )
    )

    snap = at_basis_snapshot(p)
    assert snap["schema_version"] == AT_BASIS_SCHEMA_VERSION
    assert snap["pef_current_turn"] == p.current_turn
    assert len(snap["subjects"]) == 1
    row = snap["subjects"][0]
    assert row["subject_id"] == key.id
    assert row["subject_name"] == key.name
    cur = row["current"]
    pr = row["prior"]
    assert cur["source_turn"] == 2
    assert cur["object_entity_id"] == study.id
    assert pr["source_turn"] == 1
    assert pr["object_entity_id"] == desk.id

    cur_rel = current_at(p, key.id)
    assert cur_rel is not None
    assert cur_rel.object_entity_id == study.id
    assert cur["relationship_index"] == p.relationships.index(cur_rel)


def test_at_basis_snapshot_empty_when_no_at():
    p = PEFState()
    snap = at_basis_snapshot(p)
    assert snap["subjects"] == []


def _pef_with_key_at_desk_then_study() -> PEFState:
    p = PEFState()
    key = Entity.create("gold key", 0, session_id="s")
    desk = Entity.create("desk", 0, session_id="s")
    study = Entity.create("study", 0, session_id="s")
    for e in (key, desk, study):
        p.add_entity(e)
    p.add_relationship(
        Relationship(
            subject_id=key.id,
            relation="AT",
            object_entity_id=desk.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=1,
            evidence="t1",
        )
    )
    p.add_relationship(
        Relationship(
            subject_id=key.id,
            relation="AT",
            object_entity_id=study.id,
            object_literal=None,
            span=Span.PRESENT,
            source_turn=2,
            evidence="t2",
        )
    )
    return p


class TestAtVerificationBasisAuditIntegration:
    """Audit log rows include ``at_verification_basis`` without a running HTTP server."""

    def test_flat_jsonl_backend_pass_includes_basis(self, tmp_path):
        """backend=jsonl: PASS row carries recoverable AT ranking (matches Lens wiring)."""
        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file), backend="jsonl")

        pef = _pef_with_key_at_desk_then_study()
        basis = at_basis_snapshot(pef)

        decision = GovernanceDecision(
            action=InterventionAction.PASS,
            flags=[],
            rationale="clean",
            policy="strict",
        )
        bridge.log_decision(
            decision,
            turn=2,
            pef_context="ctx",
            at_verification_basis=basis,
        )

        lines = [ln for ln in audit_file.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["outcome"] == "PASS"
        assert row["turn"] == 2
        assert "at_verification_basis" in row
        stored = row["at_verification_basis"]
        assert stored["schema_version"] == AT_BASIS_SCHEMA_VERSION
        assert len(stored["subjects"]) == 1
        study_ent = next(e for e in pef.entities.values() if e.name == "study")
        assert stored["subjects"][0]["current"]["object_entity_id"] == study_ent.id
        assert decision.at_verification_basis == stored

    def test_afl_ledger_pass_includes_basis(self, tmp_path):
        """Default AFL ledger: PASS governance line embeds ``at_verification_basis`` in payload.data."""
        audit_file = tmp_path / "audit.jsonl"
        bridge = AuroraScannerGateBridge(audit_path=str(audit_file))

        pef = _pef_with_key_at_desk_then_study()
        basis = at_basis_snapshot(pef)

        decision = GovernanceDecision(
            action=InterventionAction.PASS,
            flags=[],
            rationale="clean",
            policy="strict",
        )
        bridge.log_decision(
            decision,
            turn=1,
            pef_context="ctx",
            at_verification_basis=basis,
        )

        lines = [ln for ln in audit_file.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
        assert len(lines) == 1
        envelope = json.loads(lines[0])
        data = envelope["payload"]["data"]
        assert data["action"] == "PASS"
        assert "at_verification_basis" in data
        assert data["at_verification_basis"]["schema_version"] == AT_BASIS_SCHEMA_VERSION
        assert len(data["at_verification_basis"]["subjects"]) == 1
