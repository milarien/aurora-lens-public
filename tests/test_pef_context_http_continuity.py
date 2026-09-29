"""HTTP-level proof of the PEF context boundary repair.

This is the core test for the repair: language is downstream of world, so
extraction and adjudication must run against a *loaded* present world, not a
freshly minted empty frame on every request — and when no durable world
handle is present, continuation language must resolve to a hold, never a
guess.

Two scenarios, both driven through real HTTP calls against the FastAPI app
(Starlette ``TestClient``), with the real spaCy extraction backend so the
referential-admission behavior under test is not mocked away:

Scenario A — continuity across separate HTTP calls:
    Call 1 establishes ``Escrow HAS a transaction`` and returns
    ``aurora.pef_context_id``. Call 2, a *separate* HTTP request carrying that
    same ``pef_context_id``, says "That transaction must remain off the
    books." The demonstrative resolves against the entity/relationship
    committed in call 1 — it is not held as an unresolved referent, and no
    duplicate/fabricated entity is created for it.

Scenario B — no context handle => hold, never a guess:
    The same harmful-adjacent continuation sentence, sent as the very first
    turn of a brand-new session (no ``pef_context_id``, no cookie). There is
    nothing in the loaded (empty) PEF for "that transaction" to resolve to,
    so the turn must come back as ``UNRESOLVED_REFERENT`` / CONTAIN — never a
    silently-admitted mutation with a fabricated entity named "that
    transaction".
"""

from __future__ import annotations

import pytest

from tests.skip_reasons import SKIP_STARLETTE_HTTP_TESTCLIENT


def _get_test_client_and_app(monkeypatch):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

    from aurora_lens.proxy.app import create_app
    from aurora_lens.proxy.config import ProxyConfig
    from tests.test_proxy import MockAdapter

    monkeypatch.setattr(
        "aurora_lens.proxy.app._build_provider_adapters",
        lambda cfg: (MockAdapter(), MockAdapter()),
    )
    cfg = ProxyConfig.from_mapping({
        "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        "governance": {
            "default_policy": "strict",
            "audit_log": None,
            "include_operator_detail": True,
        },
        "extraction": {"backend": "spacy"},
    })
    app = create_app(cfg)
    return TestClient(app), app


class TestPefContextContinuityAcrossHttpCalls:
    """Scenario A: same pef_context_id, two separate HTTP calls, world persists."""

    def test_second_call_resolves_demonstrative_against_first_calls_world(self, monkeypatch):
        client, _ = _get_test_client_and_app(monkeypatch)

        r1 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Escrow has a transaction."}],
            },
        )
        assert r1.status_code == 200, r1.text
        body1 = r1.json()
        pef_context_id = body1["aurora"].get("pef_context_id")
        assert pef_context_id, "first response must mint and return aurora.pef_context_id"
        # session_id remains the backward-compatible alias for the same handle.
        assert body1["aurora"].get("session_id") == pef_context_id

        # Second call is a genuinely separate HTTP request (fresh headers dict,
        # no reliance on cookie jar) carrying only the returned pef_context_id.
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "That transaction must remain off the books."}],
            },
            headers={"x-aurora-pef-context-id": pef_context_id},
        )
        assert r2.status_code == 200, r2.text
        body2 = r2.json()
        aurora2 = body2["aurora"]

        # Round-trips to the same world handle.
        assert aurora2.get("pef_context_id") == pef_context_id

        # The demonstrative resolved against a real antecedent from call 1 —
        # it must not come back as an unresolved-referent hold.
        assert aurora2.get("governance") != "CONTAIN", (
            f"demonstrative with a real antecedent from call 1 must not hold as "
            f"unresolved; got aurora={aurora2!r}"
        )
        flags2 = set(aurora2.get("flags") or [])
        assert "UNRESOLVED_REFERENT" not in flags2

        # World state from call 1 (the "Escrow" entity, committed via the HAS
        # claim) survived into call 2, and resolving "that transaction" against
        # it did not fabricate a *new* entity for the unresolved surface form —
        # the operator-plane entity count must not grow across the boundary.
        entity_count_call1 = (
            (body1["aurora"].get("operator_pef") or {}).get("counts", {}).get("entities")
        )
        entity_count_call2 = (
            (aurora2.get("operator_pef") or {}).get("counts", {}).get("entities")
        )
        assert entity_count_call1 == 1, f"call 1 must commit exactly the Escrow entity; got {entity_count_call1}"
        assert entity_count_call2 == entity_count_call1, (
            f"resolving a demonstrative against a real antecedent must not mint a new "
            f"entity; call1={entity_count_call1} call2={entity_count_call2}"
        )


class TestPefContextMissingHandleHoldsUnresolved:
    """Scenario B: no pef_context_id at all => hold, never a guess, never a mutation."""

    def test_continuation_language_without_context_holds_unresolved(self, monkeypatch):
        client, _ = _get_test_client_and_app(monkeypatch)

        # Brand-new session: no pef_context_id, no cookie, no prior turn — the
        # loaded world is empty, so "that transaction" has no antecedent.
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "That transaction must remain off the books."}],
            },
        )
        assert r.status_code == 200, r.text
        aurora = r.json()["aurora"]

        assert aurora.get("governance") == "CONTAIN", (
            f"continuation language with no loaded antecedent must hold, not pass or "
            f"guess; got aurora={aurora!r}"
        )
        flags = set(aurora.get("flags") or [])
        assert "UNRESOLVED_REFERENT" in flags

        # No world mutation happened for the unresolved reference: no entity
        # (e.g. a fabricated "that transaction") was minted to satisfy the claim.
        op_pef = aurora.get("operator_pef") or {}
        entity_count = (op_pef.get("counts") or {}).get("entities")
        assert entity_count == 0, (
            f"no entity may be fabricated for an unresolved demonstrative with no "
            f"loaded context; operator_pef={op_pef!r}"
        )
        pef_admission = aurora.get("pef_admission_result")
        if isinstance(pef_admission, dict):
            assert not pef_admission.get("mutation_count"), (
                f"no state mutation may be acknowledged for an unresolved referent; "
                f"pef_admission_result={pef_admission!r}"
            )
