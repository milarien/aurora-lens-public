"""Session expiry must not release an unresolved governance hold.

A latest-turn request on the same session id stays blocked after the
conversational TTL, with no upstream call. Explicit resolution and a different
session id are unaffected. The expired audit snapshot is not restored.
"""

from __future__ import annotations

import pytest

from aurora_lens.config import LensConfig
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.interpret.spacy_backend import SpacyBackend
from aurora_lens.lens import Lens, _apply_epistemic_hold_after_non_admit
from aurora_lens.pef.state import PEFState
from aurora_lens.pef.unresolved_referents import (
    UnresolvedReferentEntry,
    open_entries,
    register_unresolved_referents,
)
from aurora_lens.pef.unresolved_session_gate import evaluate_unresolved_session_gate
from aurora_lens.proxy.session import SessionManager
from aurora_lens.proxy.session_store import (
    MemorySessionStore,
    RedisSessionStore,
    SessionStoreError,
)
from tests.skip_reasons import SKIP_SPACY_MODULE
from tests.test_lens import MockAdapter

_TURN1 = (
    "The operator informed the contractor that their certification had expired "
    "before the work commenced. Both parties hold certifications."
)
_SUSPENSION = "Is suspension admissible?"
_CONTRACTOR_SUSPEND = "Should the contractor be suspended?"
_RESOLUTION = "The contractor's certification had expired."
_TTL = 30


class _Clock:
    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeRedis:
    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        self.values: dict[str, tuple[str, float | None]] = {}

    def get(self, key: str) -> str | None:
        item = self.values.get(key)
        if item is None:
            return None
        raw, deadline = item
        if deadline is not None and self._clock() >= deadline:
            del self.values[key]
            return None
        return raw

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.values[key] = (value, self._clock() + float(ttl))

    def set(self, key: str, value: str) -> None:
        self.values[key] = (value, None)

    def delete(self, key: str) -> None:
        self.values.pop(key, None)

    def lock(self, *_args, **_kwargs):
        return self._Lock()

    class _Lock:
        def acquire(self, blocking: bool = True) -> bool:
            return True

        def release(self) -> None:
            return None


def _patch_clock(monkeypatch: pytest.MonkeyPatch, clock: _Clock) -> None:
    monkeypatch.setattr("aurora_lens.proxy.session.time.time", clock)
    monkeypatch.setattr("aurora_lens.proxy.session_store.time.time", clock)


def _manager(monkeypatch: pytest.MonkeyPatch, backend: str, clock: _Clock) -> SessionManager:
    _patch_clock(monkeypatch, clock)
    if backend == "memory":
        store = MemorySessionStore(ttl_seconds=_TTL)
    elif backend == "redis":
        store = RedisSessionStore(
            "",
            ttl_seconds=_TTL,
            key_prefix="aurora:test:expiry:session:",
            client=_FakeRedis(clock),
        )
    else:
        raise AssertionError(backend)
    return SessionManager(
        config_factory=lambda: LensConfig(adapter=MockAdapter(responses=["unused"])),
        store=store,
        ttl_seconds=_TTL,
    )


def _seed_hold(pef: PEFState) -> None:
    register_unresolved_referents(
        pef,
        turn=1,
        utterance=_TURN1,
        tokens=["their"],
        candidate_entities=["Operator", "Contractor"],
        blocked_proposition="their certification had expired before the work commenced",
    )
    pef.pending_clarification = {
        "failed_constraint": "UNRESOLVED_REFERENT",
        "candidate_entities": ["Operator", "Contractor"],
        "ambiguous_referents": ["their"],
    }
    pef.epistemic_hold = {
        "schema_version": 1,
        "mode": "ambiguity",
        "since_turn": 1,
        "pathway_id": "P_ASK_DISAMBIGUATE",
        "interaction_open": True,
        "commitment_closed": False,
        "last_audit_id": "cid:expired-snapshot",
    }
    ent, _ = pef.get_or_create_entity("Operator", resolved=True)
    ent.turn_last_active = 1
    pef.current_turn = 2


@pytest.mark.parametrize("backend", ["memory", "redis"])
def test_expiry_drops_snapshot_and_keeps_open_attribution(monkeypatch, backend):
    clock = _Clock()
    mgr = _manager(monkeypatch, backend, clock)
    sid = "session-held"
    with mgr.with_lock(sid):
        lens, is_new = mgr.get_or_create(sid)
        assert is_new
        _seed_hold(lens.pef)
        mgr.persist(sid, lens)
        assert lens.pef.entities

    clock.advance(_TTL + 1)
    removed = mgr.cleanup_expired()
    if backend == "memory":
        assert removed == 1
    assert mgr._store.get(sid) is None
    hold = mgr._store.get_governance_hold(sid)
    assert hold is not None
    assert "entities" not in hold
    assert "relationships" not in hold
    assert hold["unresolved_referent_registry"]

    with mgr.with_lock(sid):
        restored, is_new = mgr.get_or_create(sid)
        assert is_new is False
        assert restored.pef.entities == {}
        assert restored.pef.relationships == []
        assert restored.pef.current_turn == 0
        assert open_entries(restored.pef)
        gate = evaluate_unresolved_session_gate(restored.pef, _SUSPENSION)
        assert gate is not None and gate.blocks is True
    with mgr.with_lock("session-other"):
        other, other_new = mgr.get_or_create("session-other")
        assert other_new is True
        assert open_entries(other.pef) == []


def _spacy_backend():
    try:
        import spacy
    except ImportError:
        pytest.skip(SKIP_SPACY_MODULE)
    try:
        spacy.load("en_core_web_sm")
    except Exception as exc:
        pytest.skip(f"{SKIP_SPACY_MODULE} Underlying error: {exc!r}")
    return SpacyBackend(model="en_core_web_sm")


async def _turn(mgr: SessionManager, sid: str, text: str):
    with mgr.with_lock(sid):
        lens, is_new = mgr.get_or_create(sid)
        result = await lens.process(text)
        mgr.persist(sid, lens)
        return result, is_new, lens


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_dependent_requests_stay_blocked_after_expiry(monkeypatch, backend):
    extraction = _spacy_backend()
    clock = _Clock()
    adapter = MockAdapter(responses=["Recorded."] * 8)
    _patch_clock(monkeypatch, clock)
    if backend == "memory":
        store = MemorySessionStore(ttl_seconds=_TTL)
    else:
        store = RedisSessionStore(
            "",
            ttl_seconds=_TTL,
            key_prefix="aurora:test:expiry-lens:session:",
            client=_FakeRedis(clock),
        )

    def factory():
        return LensConfig(adapter=adapter, extraction_backend=extraction)

    mgr = SessionManager(config_factory=factory, store=store, ttl_seconds=_TTL)
    sid = "session-operator"

    await _turn(mgr, sid, _TURN1)
    calls_after_setup = adapter._call_count

    before, _, held = await _turn(mgr, sid, _SUSPENSION)
    assert before.action != InterventionAction.PASS
    assert adapter._call_count == calls_after_setup
    assert "does not establish" in (before.response or "").lower()
    assert held.pef.current_turn > 0

    clock.advance(_TTL + 1)
    assert store.get(sid) is None
    with mgr.with_lock(sid):
        carried_lens, carried_new = mgr.get_or_create(sid)
        assert carried_new is False
        assert carried_lens.pef.entities == {}
        assert carried_lens.pef.relationships == []
        assert carried_lens.pef.current_turn == 0
        assert open_entries(carried_lens.pef)

    blocked, is_new, _carried = await _turn(mgr, sid, _SUSPENSION)
    assert is_new is False
    assert blocked.action != InterventionAction.PASS
    assert adapter._call_count == calls_after_setup
    assert "does not establish" in (blocked.response or "").lower()

    calls_before_second = adapter._call_count
    second, _, _ = await _turn(mgr, sid, _CONTRACTOR_SUSPEND)
    assert second.action != InterventionAction.PASS
    assert adapter._call_count == calls_before_second
    assert "does not establish" in (second.response or "").lower()

    fresh_adapter = MockAdapter(responses=["Recorded."])
    fresh = Lens(LensConfig(adapter=fresh_adapter, extraction_backend=extraction))
    fresh_result = await fresh.process(_CONTRACTOR_SUSPEND)

    new_calls_before = adapter._call_count
    new_result, new_session, _ = await _turn(mgr, "session-brand-new", _CONTRACTOR_SUSPEND)
    assert new_session is True
    assert adapter._call_count - new_calls_before == fresh_adapter._call_count
    assert new_result.action == fresh_result.action

    resolved, _, resolved_lens = await _turn(mgr, sid, _RESOLUTION)
    assert open_entries(resolved_lens.pef) == []
    assert resolved.action == InterventionAction.PASS
    assert store.get_governance_hold(sid) is None

    clock.advance(_TTL + 1)
    baseline_adapter = MockAdapter(responses=["Recorded."])
    baseline = Lens(LensConfig(adapter=baseline_adapter, extraction_backend=extraction))
    baseline_result = await baseline.process(_SUSPENSION)
    calls_before_release = adapter._call_count
    released, released_new, _ = await _turn(mgr, sid, _SUSPENSION)
    assert released_new is True
    assert adapter._call_count - calls_before_release == baseline_adapter._call_count
    assert released.action == baseline_result.action


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_terminal_stop_survives_expiry_without_upstream(monkeypatch, backend):
    clock = _Clock()
    adapter = MockAdapter(responses=["should not be called"])
    _patch_clock(monkeypatch, clock)
    if backend == "memory":
        store = MemorySessionStore(ttl_seconds=_TTL)
    else:
        store = RedisSessionStore(
            "",
            ttl_seconds=_TTL,
            key_prefix="aurora:test:expiry-stop:session:",
            client=_FakeRedis(clock),
        )
    mgr = SessionManager(
        config_factory=lambda: LensConfig(adapter=adapter),
        store=store,
        ttl_seconds=_TTL,
    )
    sid = "session-stopped"
    with mgr.with_lock(sid):
        lens, _ = mgr.get_or_create(sid)
        ent, _created = lens.pef.get_or_create_entity("Reactor 3", resolved=True)
        ent.turn_last_active = 1
        lens.pef.epistemic_hold = {
            "schema_version": 1,
            "mode": "stop",
            "since_turn": 1,
            "pathway_id": "P_STOP",
            "interaction_open": False,
            "commitment_closed": True,
            "last_audit_id": "cid:stop-snapshot",
        }
        mgr.persist(sid, lens)

    clock.advance(_TTL + 1)
    calls_before = adapter._call_count
    result, is_new, carried = await _turn(mgr, sid, "What action should the regulator take?")
    assert is_new is False
    assert result.action == InterventionAction.HARD_STOP
    assert adapter._call_count == calls_before
    assert carried.pef.entities == {}
    assert carried.pef.epistemic_hold is not None
    assert carried.pef.epistemic_hold.get("mode") == "stop"


class _FlakyMemoryStore(MemorySessionStore):
    """Session put can fail after the durable hold write has already succeeded."""

    def __init__(self) -> None:
        super().__init__(ttl_seconds=_TTL)
        self.fail_session_put = False
        self.hold_deletes = 0

    def put(self, session_id: str, record) -> None:
        if self.fail_session_put:
            raise SessionStoreError("injected session put failure")
        super().put(session_id, record)

    def delete_governance_hold(self, session_id: str) -> None:
        self.hold_deletes += 1
        super().delete_governance_hold(session_id)


def _stop_hold() -> dict:
    return {
        "schema_version": 1,
        "mode": "stop",
        "since_turn": 1,
        "pathway_id": "P_STOP",
        "interaction_open": False,
        "commitment_closed": True,
        "last_audit_id": "cid:stop",
    }


def _refusal_hold() -> dict:
    return {
        "schema_version": 1,
        "mode": "refusal",
        "since_turn": 1,
        "pathway_id": "P_REFUSE",
        "interaction_open": True,
        "commitment_closed": True,
        "last_audit_id": "cid:refusal",
    }


def test_failed_session_write_leaves_constraint_in_durable_hold(monkeypatch):
    clock = _Clock()
    _patch_clock(monkeypatch, clock)
    store = _FlakyMemoryStore()
    mgr = SessionManager(
        config_factory=lambda: LensConfig(adapter=MockAdapter(responses=["unused"])),
        store=store,
        ttl_seconds=_TTL,
    )
    sid = "session-fail"
    with mgr.with_lock(sid):
        lens, is_new = mgr.get_or_create(sid)
        assert is_new
        _seed_hold(lens.pef)
        lens.pef.epistemic_hold = _stop_hold()
        store.fail_session_put = True
        with pytest.raises(SessionStoreError, match="injected session put failure"):
            mgr.persist(sid, lens)

    hold = store.get_governance_hold(sid)
    assert hold is not None
    assert hold["unresolved_referent_registry"]
    assert hold["epistemic_hold"]["mode"] == "stop"
    raw = store._store[sid]
    assert raw.pef_state.get("unresolved_referent_registry") in (None, [])

    with mgr.with_lock(sid):
        enforced, is_new = mgr.get_or_create(sid)
    assert is_new is False
    assert open_entries(enforced.pef)
    assert enforced.pef.epistemic_hold is not None
    assert enforced.pef.epistemic_hold["mode"] == "stop"
    gate = evaluate_unresolved_session_gate(enforced.pef, _SUSPENSION)
    assert gate is not None and gate.blocks is True

    clock.advance(_TTL + 1)
    assert store.get(sid) is None
    with mgr.with_lock(sid):
        with pytest.raises(SessionStoreError, match="injected session put failure"):
            mgr.get_or_create(sid)
    assert store.get_governance_hold(sid)["epistemic_hold"]["mode"] == "stop"
    store.fail_session_put = False
    with mgr.with_lock(sid):
        restored, _ = mgr.get_or_create(sid)
    assert open_entries(restored.pef)
    assert restored.pef.epistemic_hold["mode"] == "stop"


def test_failed_resolution_write_does_not_delete_durable_hold(monkeypatch):
    clock = _Clock()
    _patch_clock(monkeypatch, clock)
    store = _FlakyMemoryStore()
    mgr = SessionManager(
        config_factory=lambda: LensConfig(adapter=MockAdapter(responses=["unused"])),
        store=store,
        ttl_seconds=_TTL,
    )
    sid = "session-resolve-fail"
    with mgr.with_lock(sid):
        lens, _ = mgr.get_or_create(sid)
        _seed_hold(lens.pef)
        lens.pef.epistemic_hold = _stop_hold()
        mgr.persist(sid, lens)
        deletes_before = store.hold_deletes
        lens.pef.unresolved_referent_registry.clear()
        lens.pef.pending_clarification = None
        lens.pef.epistemic_hold = None
        store.fail_session_put = True
        with pytest.raises(SessionStoreError, match="injected session put failure"):
            mgr.persist(sid, lens)
    assert store.hold_deletes == deletes_before
    assert store.get_governance_hold(sid) is not None
    assert store.get_governance_hold(sid)["epistemic_hold"]["mode"] == "stop"


def test_contain_does_not_replace_refusal_or_terminal_stop():
    for hold in (_refusal_hold(), _stop_hold()):
        pef = PEFState()
        _seed_hold(pef)
        pef.epistemic_hold = dict(hold)
        decision = GovernanceDecision(
            action=InterventionAction.CONTAIN,
            flags=[],
            rationale="unrelated contain",
            policy="strict",
            pathway_id="P_ASK_DISAMBIGUATE",
            interaction_open=True,
            commitment_closed=False,
        )
        _apply_epistemic_hold_after_non_admit(pef, decision, turn=4)
        assert pef.epistemic_hold is not None
        assert pef.epistemic_hold["mode"] == hold["mode"]
        assert pef.epistemic_hold["pathway_id"] == hold["pathway_id"]
        assert open_entries(pef)


@pytest.mark.asyncio
@pytest.mark.parametrize("hold_factory", [_refusal_hold, _stop_hold])
async def test_unrelated_turn_does_not_discharge_separate_refusal_or_stop(monkeypatch, hold_factory):
    clock = _Clock()
    _patch_clock(monkeypatch, clock)
    adapter = MockAdapter(responses=["A joke."])
    store = MemorySessionStore(ttl_seconds=_TTL)
    mgr = SessionManager(
        config_factory=lambda: LensConfig(adapter=adapter, auto_interpret=False, auto_verify=False),
        store=store,
        ttl_seconds=_TTL,
    )
    sid = "session-unrelated"
    hold = hold_factory()
    with mgr.with_lock(sid):
        lens, _ = mgr.get_or_create(sid)
        _seed_hold(lens.pef)
        lens.pef.epistemic_hold = dict(hold)
        mgr.persist(sid, lens)

    calls_before = adapter._call_count
    result, _, carried = await _turn(mgr, sid, "Tell me a joke about penguins.")
    assert carried.pef.epistemic_hold is not None
    assert carried.pef.epistemic_hold["mode"] == hold["mode"]
    assert carried.pef.epistemic_hold["pathway_id"] == hold["pathway_id"]
    assert open_entries(carried.pef)
    stored = store.get_governance_hold(sid)
    assert stored is not None
    assert stored["epistemic_hold"]["mode"] == hold["mode"]
    assert stored["unresolved_referent_registry"]
    if hold["mode"] == "stop":
        assert result.action == InterventionAction.HARD_STOP
        assert adapter._call_count == calls_before


@pytest.mark.asyncio
@pytest.mark.parametrize("hold_factory", [_refusal_hold, _stop_hold])
async def test_attribution_clarification_does_not_discharge_separate_refusal_or_stop(
    monkeypatch, hold_factory
):
    extraction = _spacy_backend()
    clock = _Clock()
    _patch_clock(monkeypatch, clock)
    adapter = MockAdapter(responses=["Recorded."])
    store = MemorySessionStore(ttl_seconds=_TTL)
    mgr = SessionManager(
        config_factory=lambda: LensConfig(adapter=adapter, extraction_backend=extraction),
        store=store,
        ttl_seconds=_TTL,
    )
    sid = "session-clarify"
    hold = hold_factory()
    with mgr.with_lock(sid):
        lens, _ = mgr.get_or_create(sid)
        _seed_hold(lens.pef)
        lens.pef.pending_clarification = {
            "original_question": _TURN1,
            "unresolved_entity_ids": [],
            "failed_constraint": "UNRESOLVED_REFERENT",
            "ambiguous_referents": ["their"],
            "candidate_entities": ["Operator", "Contractor"],
            "original_span": "present",
            "blocked_proposition": "their certification had expired before the work commenced",
            "blocked_claims": [],
        }
        lens.pef.epistemic_hold = dict(hold)
        mgr.persist(sid, lens)

    calls_before = adapter._call_count
    _result, _, carried = await _turn(mgr, sid, _RESOLUTION)
    assert carried.pef.epistemic_hold is not None
    assert carried.pef.epistemic_hold["mode"] == hold["mode"]
    assert carried.pef.epistemic_hold["pathway_id"] == hold["pathway_id"]
    stored = store.get_governance_hold(sid)
    assert stored is not None
    assert stored["epistemic_hold"]["mode"] == hold["mode"]
    if hold["mode"] == "stop":
        assert adapter._call_count == calls_before
        assert open_entries(carried.pef)
        assert stored["unresolved_referent_registry"]
    else:
        assert open_entries(carried.pef) == []
        assert stored["unresolved_referent_registry"] == []


_UPDATED_CANDIDATES = ["Operator", "Contractor", "Inspector", "Licensee"]
_UPDATED_PROPOSITION = "their site access had been revoked before the shift"
_OTHER_PROPOSITION = "their report was unsigned"
_OTHER_CANDIDATES = ["Auditor", "Witness"]


def _assert_updated_constraint_survives(pef) -> None:
    by_id = {entry.entry_id: entry for entry in open_entries(pef)}
    updated = by_id["ur:their@1"]
    assert updated.token == "their"
    assert updated.candidate_entities == _UPDATED_CANDIDATES
    assert updated.blocked_proposition == _UPDATED_PROPOSITION
    other = by_id["ur:their@9"]
    assert other.token == "their"
    assert other.candidate_entities == _OTHER_CANDIDATES
    assert other.blocked_proposition == _OTHER_PROPOSITION
    pending = pef.pending_clarification
    assert pending is not None
    assert pending["original_question"] == _TURN1
    assert pending["candidate_entities"] == _UPDATED_CANDIDATES
    assert pending["blocked_proposition"] == _UPDATED_PROPOSITION
    assert "Auditor" not in pending["candidate_entities"]
    assert "Witness" not in pending["candidate_entities"]


def test_updated_same_token_constraint_survives_failed_session_write(monkeypatch):
    """A newer governing constraint must survive when the session write fails.

    The live record keeps the older entry for the same pronoun. A separate
    open entry that uses that pronoun is a different constraint.
    """
    clock = _Clock()
    _patch_clock(monkeypatch, clock)
    store = _FlakyMemoryStore()
    mgr = SessionManager(
        config_factory=lambda: LensConfig(adapter=MockAdapter(responses=["unused"])),
        store=store,
        ttl_seconds=_TTL,
    )
    sid = "session-updated-token"
    with mgr.with_lock(sid):
        lens, _ = mgr.get_or_create(sid)
        _seed_hold(lens.pef)
        lens.pef.pending_clarification = {
            "original_question": _TURN1,
            "unresolved_entity_ids": [],
            "failed_constraint": "UNRESOLVED_REFERENT",
            "ambiguous_referents": ["their"],
            "candidate_entities": ["Operator", "Contractor"],
            "original_span": "present",
            "blocked_proposition": "their certification had expired before the work commenced",
            "blocked_claims": [],
        }
        lens.pef.unresolved_referent_registry.append(
            UnresolvedReferentEntry(
                entry_id="ur:their@9",
                token="their",
                span_surface="their",
                candidate_entities=list(_OTHER_CANDIDATES),
                blocked_proposition=_OTHER_PROPOSITION,
                introduced_turn=9,
                introduced_utterance="The auditor asked the witness whether their report was unsigned.",
                status="open",
            )
        )
        mgr.persist(sid, lens)

        updated = next(entry for entry in lens.pef.unresolved_referent_registry if entry.entry_id == "ur:their@1")
        updated.candidate_entities = list(_UPDATED_CANDIDATES)
        updated.blocked_proposition = _UPDATED_PROPOSITION
        pending = dict(lens.pef.pending_clarification)
        pending["candidate_entities"] = list(_UPDATED_CANDIDATES)
        pending["blocked_proposition"] = _UPDATED_PROPOSITION
        lens.pef.pending_clarification = pending
        store.fail_session_put = True
        with pytest.raises(SessionStoreError, match="injected session put failure"):
            mgr.persist(sid, lens)

    raw_entries = store._store[sid].pef_state["unresolved_referent_registry"]
    raw_updated = next(entry for entry in raw_entries if entry["entry_id"] == "ur:their@1")
    assert raw_updated["candidate_entities"] == ["Operator", "Contractor"]
    assert raw_updated["blocked_proposition"] == "their certification had expired before the work commenced"
    held = store.get_governance_hold(sid)
    assert held is not None
    held_updated = next(
        entry for entry in held["unresolved_referent_registry"] if entry["entry_id"] == "ur:their@1"
    )
    assert held_updated["candidate_entities"] == _UPDATED_CANDIDATES
    assert held_updated["blocked_proposition"] == _UPDATED_PROPOSITION

    with mgr.with_lock(sid):
        restored, is_new = mgr.get_or_create(sid)
    assert is_new is False
    _assert_updated_constraint_survives(restored.pef)
    gate = evaluate_unresolved_session_gate(restored.pef, _SUSPENSION)
    assert gate is not None and gate.blocks is True

    clock.advance(_TTL + 1)
    assert store.get(sid) is None
    assert store.get_governance_hold(sid)["unresolved_referent_registry"]
    store.fail_session_put = False
    with mgr.with_lock(sid):
        expired, expired_new = mgr.get_or_create(sid)
    assert expired_new is False
    _assert_updated_constraint_survives(expired.pef)
    expired_gate = evaluate_unresolved_session_gate(expired.pef, _SUSPENSION)
    assert expired_gate is not None and expired_gate.blocks is True


_STANDALONE_PENDING = {
    "original_question": "Which stick is bigger?",
    "unresolved_entity_ids": [],
    "failed_constraint": "UNRESOLVED_COMPARAND",
    "ambiguous_referents": [],
    "candidate_entities": ["the oak stick", "the pine stick"],
    "original_span": "present",
    "blocked_proposition": "the bigger stick was selected",
    "blocked_claims": [],
    "comparand_adjective": "bigger",
    "comparand_noun": "stick",
}


def _assert_standalone_clarification(pending: dict | None) -> None:
    assert pending is not None
    assert pending["failed_constraint"] == "UNRESOLVED_COMPARAND"
    assert pending["original_question"] == "Which stick is bigger?"
    assert pending["comparand_adjective"] == "bigger"
    assert pending["comparand_noun"] == "stick"
    assert pending["candidate_entities"] == ["the oak stick", "the pine stick"]
    assert pending["blocked_proposition"] == "the bigger stick was selected"


def test_standalone_pending_clarification_survives_failed_write_and_expiry(monkeypatch):
    """A clarification with no referent-registry entries is still unresolved.

    The durable hold is written, the session write fails, and reload before
    expiry must recover it. A later successful persist and session expiry do
    not count as resolution.
    """
    clock = _Clock()
    _patch_clock(monkeypatch, clock)
    store = _FlakyMemoryStore()
    mgr = SessionManager(
        config_factory=lambda: LensConfig(adapter=MockAdapter(responses=["unused"])),
        store=store,
        ttl_seconds=_TTL,
    )
    sid = "session-standalone-pending"
    with mgr.with_lock(sid):
        lens, is_new = mgr.get_or_create(sid)
        assert is_new
        mgr.persist(sid, lens)
        assert lens.pef.pending_clarification is None
        assert lens.pef.unresolved_referent_registry == []
        lens.pef.pending_clarification = dict(_STANDALONE_PENDING)
        store.fail_session_put = True
        with pytest.raises(SessionStoreError, match="injected session put failure"):
            mgr.persist(sid, lens)

    assert store._store[sid].pef_state.get("pending_clarification") is None
    assert store._store[sid].pef_state.get("unresolved_referent_registry") in (None, [])
    held = store.get_governance_hold(sid)
    assert held is not None
    assert held["unresolved_referent_registry"] == []
    _assert_standalone_clarification(held["pending_clarification"])

    with mgr.with_lock(sid):
        restored, is_new = mgr.get_or_create(sid)
    assert is_new is False
    assert restored.pef.unresolved_referent_registry == []
    _assert_standalone_clarification(restored.pef.pending_clarification)

    store.fail_session_put = False
    with mgr.with_lock(sid):
        mgr.persist(sid, restored)
    held_after_persist = store.get_governance_hold(sid)
    assert held_after_persist is not None
    _assert_standalone_clarification(held_after_persist["pending_clarification"])
    _assert_standalone_clarification(store._store[sid].pef_state.get("pending_clarification"))

    clock.advance(_TTL + 1)
    assert store.get(sid) is None
    with mgr.with_lock(sid):
        expired, expired_new = mgr.get_or_create(sid)
    assert expired_new is False
    assert expired.pef.unresolved_referent_registry == []
    _assert_standalone_clarification(expired.pef.pending_clarification)
    _assert_standalone_clarification(store.get_governance_hold(sid)["pending_clarification"])
