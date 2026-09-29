"""OpenAI Chat Completions schema conformance tests.

Verifies that aurora-lens proxy responses conform to the OpenAI API schema
across all governance outcomes, error shapes, and streaming formats.
"""

import pytest

from tests.skip_reasons import SKIP_STARLETTE_HTTP_TESTCLIENT

from aurora_lens.lens import LensResult
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState
from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
from aurora_lens.verify.flags import Flag, FlagType
from aurora_lens.proxy.openai_compat import format_chat_response


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _pass_result(response: str = "Sure, here you go.") -> LensResult:
    return LensResult(
        response=response,
        flags=[],
        pef_snapshot="",
        turn=1,
        span=Span.PRESENT,
        action=InterventionAction.PASS,
        model="mock",
    )


def _hard_stop_result() -> LensResult:
    flags = [Flag(
        flag_type=FlagType.SELF_HARM_INSTRUCTION,
        entity_name="self_harm",
        claim="Procedural self-harm guidance detected",
        evidence="how to kill yourself",
        severity="error",
    )]
    decision = GovernanceDecision(
        action=InterventionAction.HARD_STOP,
        flags=flags,
        rationale="Self-harm instruction detected",
        policy="strict",
        resource="a crisis helpline",
    )
    return LensResult(
        response="This response has been blocked by governance policy.",
        flags=flags,
        pef_snapshot="",
        turn=1,
        span=Span.PRESENT,
        action=InterventionAction.HARD_STOP,
        decision=decision,
        original_response="Here is how to kill yourself...",
        model="mock",
    )


def _make_proxy_app(monkeypatch):
    """Build a TestClient-ready proxy app backed by a minimal echo adapter."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

    from aurora_lens.proxy.config import ProxyConfig
    from aurora_lens.proxy.app import create_app
    from aurora_lens.adapters.base import LLMAdapter, AdapterResponse

    class _EchoAdapter(LLMAdapter):
        async def generate(self, messages, **kwargs):
            return AdapterResponse(text="test response", model="mock")

        async def generate_stream(self, messages, **kwargs):
            for chunk in ["test", " response"]:
                yield ({"choices": [{"delta": {"content": chunk}, "index": 0}]}, chunk)

    monkeypatch.setattr(
        "aurora_lens.proxy.app._build_provider_adapters",
        lambda cfg: (_EchoAdapter(), _EchoAdapter()),
    )
    cfg = ProxyConfig.from_mapping({
        "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        "governance": {"audit_log": None},
        "extraction": {"backend": "spacy"},
    })
    return TestClient(create_app(cfg))


# ── Response schema ───────────────────────────────────────────────────────────

class TestResponseSchema:
    """format_chat_response output conforms to OpenAI chat.completion schema."""

    def test_required_fields_present(self):
        """All required OpenAI Chat Completion top-level fields present."""
        body = format_chat_response(_pass_result())
        for field in ("id", "object", "created", "model", "choices", "usage", "aurora"):
            assert field in body, f"Missing field: {field}"

    def test_object_is_chat_completion(self):
        """object field is always 'chat.completion'."""
        assert format_chat_response(_pass_result())["object"] == "chat.completion"

    def test_id_is_nonempty_string(self):
        """id is a non-empty string."""
        body = format_chat_response(_pass_result())
        assert isinstance(body["id"], str) and len(body["id"]) > 0

    def test_created_is_integer_timestamp(self):
        """created is an integer Unix timestamp (after 2023)."""
        body = format_chat_response(_pass_result())
        assert isinstance(body["created"], int)
        assert body["created"] > 1_700_000_000

    def test_choices_message_has_role_and_content(self):
        """choices[0].message has role == 'assistant' and correct content."""
        body = format_chat_response(_pass_result("Hello!"))
        msg = body["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello!"

    def test_finish_reason_is_stop(self):
        """finish_reason is 'stop' for both PASS and HARD_STOP outcomes."""
        for result in [_pass_result(), _hard_stop_result()]:
            body = format_chat_response(result)
            assert body["choices"][0]["finish_reason"] == "stop"

    def test_usage_fields_are_integers(self):
        """prompt_tokens, completion_tokens, total_tokens are all integers."""
        usage = format_chat_response(_pass_result())["usage"]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert key in usage, f"Missing usage field: {key}"
            assert isinstance(usage[key], int), f"{key} must be int, got {type(usage[key])}"

    def test_aurora_always_present_on_pass(self):
        """aurora key is always present, even on PASS with no flags."""
        body = format_chat_response(_pass_result())
        assert "aurora" in body
        assert body["aurora"]["governance"] == "PASS"

    def test_aurora_hard_stop_user_plane_only_by_default(self):
        """Default: aurora on HARD_STOP contains governance + safe hint; no forensic envelope."""
        body = format_chat_response(_hard_stop_result())
        aurora = body["aurora"]
        assert aurora["governance"] == "HARD_STOP"
        assert aurora["governance_hint"] == "blocked"
        assert "flags" not in aurora
        assert "rationale" not in aurora
        assert "original_response" not in aurora
        assert "forensic_event" not in aurora

    def test_aurora_hard_stop_operator_detail_when_enabled(self):
        """include_operator_detail=True exposes diagnostics; withheld upstream is redacted in wire."""
        body = format_chat_response(
            _hard_stop_result(), include_operator_detail=True, pef=PEFState(),
        )
        aurora = body["aurora"]
        assert aurora["governance"] == "HARD_STOP"
        assert "SELF_HARM_INSTRUCTION" in aurora["flags"]
        assert "rationale" in aurora
        assert "original_response" not in aurora
        assert aurora["intercepted_upstream_redacted_reason"] == (
            "suppressed_blocked_upstream_output"
        )
        assert "intercepted_upstream_output" not in aurora
        assert "operator_pef" in aurora
        assert aurora["operator_pef"]["session_mode"] == "normal"

    def test_audit_receipt_absent_without_snapshot_even_when_operator_detail(self):
        """No fabricated receipt: omit aurora.audit_receipt when snapshot was not recorded."""
        body = format_chat_response(_pass_result(), include_operator_detail=True, pef=PEFState())
        aurora = body["aurora"]
        assert "audit_receipt" not in aurora
        assert aurora["llm_called"] is False
        assert aurora["pre_llm_blocked"] is False
        assert aurora["release_path"] == "released"
        assert aurora["has_blocked_upstream_output"] is False

    def test_runtime_truth_pre_llm_hard_stop_fields(self):
        """Pre-LLM HARD_STOP reports blocked-before-generation with no candidate output."""
        flags = [Flag(
            flag_type=FlagType.ILLEGAL_INSTRUCTION,
            entity_name="illegal",
            claim="Illegal instruction requested",
            evidence="forbidden instructions",
            severity="error",
        )]
        decision = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=flags,
            rationale="Illegal instruction blocked pre-LLM",
            policy="strict",
        )
        result = LensResult(
            response="Request blocked before model call.",
            flags=flags,
            pef_snapshot="",
            turn=1,
            span=Span.PRESENT,
            action=InterventionAction.HARD_STOP,
            decision=decision,
            original_response=None,
            model="mock",
        )
        body = format_chat_response(result, include_operator_detail=True, pef=PEFState())
        aurora = body["aurora"]
        assert aurora["llm_called"] is False
        assert aurora["pre_llm_blocked"] is True
        assert aurora["release_path"] == "blocked_before_generation"
        assert aurora["blocked_phase"] == "pre_generation"
        assert aurora["released"] is False
        assert aurora["has_blocked_upstream_output"] is False
        assert "intercepted_upstream_redacted_reason" not in aurora

    def test_runtime_truth_post_llm_hard_stop_fields(self):
        """Post-LLM HARD_STOP reports blocked-after-generation with upstream suppression markers."""
        flags = [Flag(
            flag_type=FlagType.SELF_HARM_INSTRUCTION,
            entity_name="self_harm",
            claim="Self-harm instruction",
            evidence="unsafe guidance",
            severity="error",
        )]
        decision = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=flags,
            rationale="Post-generation hard stop",
            policy="strict",
        )
        result = LensResult(
            response="Request blocked.",
            flags=flags,
            pef_snapshot="",
            turn=1,
            span=Span.PRESENT,
            action=InterventionAction.HARD_STOP,
            decision=decision,
            original_response="Unsafe upstream draft that was blocked.",
            model="mock",
            usage={"completion_tokens": 12, "prompt_tokens": 4, "total_tokens": 16},
        )
        body = format_chat_response(result, include_operator_detail=True, pef=PEFState())
        aurora = body["aurora"]
        assert aurora["llm_called"] is True
        assert aurora["pre_llm_blocked"] is False
        assert aurora["release_path"] == "blocked_after_generation"
        assert aurora["blocked_phase"] == "post_generation"
        assert aurora["released"] is False
        assert aurora["has_blocked_upstream_output"] is True
        assert aurora["intercepted_upstream_redacted_reason"] == "suppressed_blocked_upstream_output"

    def test_pef_admission_result_wire_optional_on_operator_plane(self):
        wire = {"decision": "ADMIT", "write_intent": True}
        body = format_chat_response(
            _pass_result(),
            include_operator_detail=True,
            pef=PEFState(),
            pef_admission_result_wire=wire,
        )
        assert body["aurora"]["pef_admission_result"] == wire

    def test_pef_admission_result_omitted_when_not_supplied_operator_plane(self):
        body = format_chat_response(
            _pass_result(),
            include_operator_detail=True,
            pef=PEFState(),
        )
        assert "pef_admission_result" not in body["aurora"]

    def test_audit_receipt_operator_plane_when_snapshot_present(self):
        """Operator plane surfaces safe per-turn receipt fields from audit_receipt_snapshot."""
        decision = GovernanceDecision(
            action=InterventionAction.PASS,
            flags=[],
            rationale="",
            audit_receipt_snapshot={
                "trace_id": "tr-test",
                "entry_index": 3,
                "prev_hash": "prev_hex",
                "hash": "row_hex",
                "state_hash": "state_hex",
                "hmac_verified": True,
                "chain_verified_to_entry": False,
            },
        )
        result = LensResult(
            response="ok",
            flags=[],
            pef_snapshot="",
            turn=2,
            span=Span.PRESENT,
            action=InterventionAction.PASS,
            decision=decision,
            model="mock",
        )
        body = format_chat_response(
            result,
            include_operator_detail=True,
            pef=PEFState(),
            session_id="sess-op",
        )
        rec = body["aurora"]["audit_receipt"]
        assert rec["trace_id"] == "tr-test"
        assert rec["session_id"] == "sess-op"
        assert rec["entry_index"] == 3
        assert rec["prev_hash"] == "prev_hex"
        assert rec["hash"] == "row_hex"
        assert rec["state_hash"] == "state_hex"
        assert rec["hmac_verified"] is True
        assert rec["chain_verified_to_entry"] is False

    def test_audit_receipt_omitted_when_insufficient_public_fields(self):
        """Receipt builder requires at least two surfaced fields (no trivial objects)."""
        decision = GovernanceDecision(
            action=InterventionAction.PASS,
            flags=[],
            rationale="",
            audit_receipt_snapshot={"trace_id": "only-one-field"},
        )
        result = LensResult(
            response="ok",
            flags=[],
            pef_snapshot="",
            turn=1,
            span=Span.PRESENT,
            action=InterventionAction.PASS,
            decision=decision,
            model="mock",
        )
        body = format_chat_response(
            result,
            include_operator_detail=True,
            pef=PEFState(),
            session_id=None,
        )
        assert "audit_receipt" not in body["aurora"]

    def test_force_revise_has_failed_constraint_fallback(self):
        decision = GovernanceDecision(
            action=InterventionAction.FORCE_REVISE,
            flags=[],
            rationale="Needs revision",
            policy="strict",
        )
        result = LensResult(
            response="Revised response",
            flags=[],
            pef_snapshot="",
            turn=1,
            span=Span.PRESENT,
            action=InterventionAction.FORCE_REVISE,
            decision=decision,
            model="mock",
        )
        body = format_chat_response(result, include_operator_detail=True, pef=PEFState())
        aurora = body["aurora"]
        assert aurora["governance"] == "FORCE_REVISE"
        assert aurora["failed_constraint"] == "GOVERNANCE_REVISION_REQUIRED"
        assert aurora["failed_constraints"] == ["GOVERNANCE_REVISION_REQUIRED"]

    def test_pre_llm_covert_harm_hard_stop_surfaces_failed_constraint(self):
        """Eric-class pre-LLM HARD_STOP must expose flag type and blocked-request rule_id."""
        flags = [
            Flag(
                flag_type=FlagType.ILLEGAL_INSTRUCTION,
                entity_name="request",
                claim="Covert harm guidance",
                evidence="revenge, CEO",
                severity="error",
                rule_id="blocked.request.covert_harm_guidance",
            )
        ]
        decision = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=flags,
            rationale="Pre-LLM blocked-act",
            policy="strict",
            pre_llm=True,
            admissibility_basis="blocked_illicit_intent",
        )
        result = LensResult(
            response="Request blocked.",
            flags=flags,
            pef_snapshot="",
            turn=1,
            span=Span.PRESENT,
            action=InterventionAction.HARD_STOP,
            decision=decision,
            model="",
            original_response=None,
        )
        body = format_chat_response(result, include_operator_detail=True, pef=PEFState())
        aurora = body["aurora"]
        assert aurora["governance"] == "HARD_STOP"
        assert aurora["failed_constraints"] == [
            "ILLEGAL_INSTRUCTION",
            "blocked.request.covert_harm_guidance",
        ]
        assert aurora["failed_constraint"] == "ILLEGAL_INSTRUCTION"
        assert aurora["llm_called"] is False
        assert aurora["pre_llm_blocked"] is True
        assert aurora.get("failed_constraint") != "none"
        assert "none" not in [str(x).lower() for x in aurora.get("failed_constraints", [])]

    def test_pre_llm_covert_harm_user_plane_surfaces_failed_constraint(self):
        """User plane must expose stop reason without operator-detail opt-in."""
        flags = [
            Flag(
                flag_type=FlagType.ILLEGAL_INSTRUCTION,
                entity_name="request",
                claim="Covert harm guidance",
                evidence="revenge, CEO",
                severity="error",
                rule_id="blocked.request.covert_harm_guidance",
            )
        ]
        decision = GovernanceDecision(
            action=InterventionAction.HARD_STOP,
            flags=flags,
            rationale="Pre-LLM blocked-act",
            policy="strict",
            pre_llm=True,
            admissibility_basis="blocked_illicit_intent",
        )
        result = LensResult(
            response="Request blocked.",
            flags=flags,
            pef_snapshot="",
            turn=1,
            span=Span.PRESENT,
            action=InterventionAction.HARD_STOP,
            decision=decision,
            model="",
            original_response=None,
        )
        body = format_chat_response(result, include_operator_detail=False, pef=PEFState())
        aurora = body["aurora"]
        assert aurora["governance"] == "HARD_STOP"
        assert aurora["failed_constraint"] == "ILLEGAL_INSTRUCTION"
        assert "blocked.request.covert_harm_guidance" in aurora["failed_constraints"]
        assert str(aurora.get("failed_constraint")).lower() != "none"

    def test_audit_receipt_non_boolean_verification_fields_dropped(self):
        """Non-boolean hmac/chain verification hints must not appear as bogus booleans."""
        decision = GovernanceDecision(
            action=InterventionAction.PASS,
            flags=[],
            rationale="",
            audit_receipt_snapshot={
                "trace_id": "tr-x",
                "hash": "h1",
                "hmac_verified": "maybe",
                "chain_verified_to_entry": 1,
            },
        )
        result = LensResult(
            response="ok",
            flags=[],
            pef_snapshot="",
            turn=1,
            span=Span.PRESENT,
            action=InterventionAction.PASS,
            decision=decision,
            model="mock",
        )
        body = format_chat_response(
            result,
            include_operator_detail=True,
            pef=PEFState(),
            session_id="s2",
        )
        rec = body["aurora"]["audit_receipt"]
        assert "hmac_verified" not in rec
        assert "chain_verified_to_entry" not in rec

    def test_session_id_in_user_plane(self):
        """session_id routing handle is always in user-plane aurora when provided."""
        body = format_chat_response(_pass_result(), session_id="sess-abc123")
        assert body["aurora"]["session_id"] == "sess-abc123"

    def test_session_id_absent_when_not_provided(self):
        """session_id absent from aurora when not passed (stateless or test context)."""
        body = format_chat_response(_pass_result())
        assert "session_id" not in body["aurora"]

    def test_soft_correct_unverified_hint(self):
        """SOFT_CORRECT adds aurora.unverified=True; PASS does not."""
        from aurora_lens.govern.decision import InterventionAction as IA
        from aurora_lens.verify.flags import Flag, FlagType
        from aurora_lens.govern.decision import GovernanceDecision
        from aurora_lens.pef.span import Span
        from aurora_lens.lens import LensResult
        sc_result = LensResult(
            response="General knowledge response.",
            flags=[Flag(FlagType.UNRESOLVED_REFERENT, "e", "c", "ev", "warning")],
            pef_snapshot="",
            turn=1,
            span=Span.PRESENT,
            action=IA.SOFT_CORRECT,
            decision=GovernanceDecision(action=IA.SOFT_CORRECT, flags=[], rationale="r"),
        )
        body = format_chat_response(sc_result)
        assert body["aurora"]["unverified"] is True

        # PASS has no unverified hint
        pass_body = format_chat_response(_pass_result())
        assert "unverified" not in pass_body["aurora"]

    def test_audit_log_records_governance_note(self):
        """governance_note is written to the audit log even when absent from response body."""
        import tempfile, json
        from pathlib import Path
        from aurora_lens.govern.bridge import BuiltinBridge
        from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
        from aurora_lens.verify.flags import Flag, FlagType

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = f.name

        bridge = BuiltinBridge(audit_path=audit_path)
        decision = GovernanceDecision(
            action=InterventionAction.SOFT_CORRECT,
            flags=[],
            rationale="test",
            governance_note="This response draws on general knowledge, not the conversation.",
        )
        decision.original_response = "Some response."
        bridge.log_decision(decision)

        entries = [json.loads(line) for line in Path(audit_path).read_text().splitlines() if line.strip()]
        assert len(entries) == 1
        assert entries[0]["governance_note"] == "This response draws on general knowledge, not the conversation."

    def test_model_field_from_result(self):
        """model field reflects the result's model name."""
        body = format_chat_response(_pass_result(), request_model="gpt-4")
        assert body["model"] == "gpt-4"


# ── All governance outcomes return HTTP 200 ───────────────────────────────────

class TestAllOutcomesReturn200:
    """Governance outcomes must never change the HTTP status code (OpenAI compat)."""

    def test_pass_returns_200(self, monkeypatch):
        client = _make_proxy_app(monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 200
        assert r.json()["aurora"]["governance"] == "PASS"

    def test_hard_stop_via_external_flag_returns_200(self, monkeypatch):
        """HARD_STOP from external_flags still returns HTTP 200."""
        client = _make_proxy_app(monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
                "aurora": {
                    "external_flags": [
                        {"type": "SELF_HARM_INSTRUCTION", "evidence": ["test"], "severity": "error"},
                    ],
                },
            },
        )
        assert r.status_code == 200
        assert r.json()["aurora"]["governance"] == "HARD_STOP"


# ── Error response shape ──────────────────────────────────────────────────────

class TestErrorResponseShape:
    """Error responses follow OpenAI error schema: {error: {message, ...}}."""

    def test_invalid_json_returns_400_with_error_key(self, monkeypatch):
        """400 body has top-level 'error' key with 'message'."""
        client = _make_proxy_app(monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400
        body = r.json()
        assert "error" in body
        assert "message" in body["error"]

    def test_empty_messages_returns_422_with_error_key(self, monkeypatch):
        """422 body has top-level 'error' key with 'message'."""
        client = _make_proxy_app(monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4"},
        )
        assert r.status_code == 422
        body = r.json()
        assert "error" in body
        assert "message" in body["error"]


# ── Streaming format ──────────────────────────────────────────────────────────

class TestStreamingFormat:
    """Streaming responses are valid SSE terminating with data: [DONE]."""

    def test_stream_content_type_is_event_stream(self, monkeypatch):
        """Streaming response Content-Type is text/event-stream."""
        client = _make_proxy_app(monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_stream_lines_are_sse_format(self, monkeypatch):
        """Non-empty lines in stream response start with 'data: '."""
        client = _make_proxy_app(monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        )
        lines = [ln for ln in r.text.splitlines() if ln.strip()]
        for line in lines:
            assert line.startswith("data: "), f"SSE line malformed: {line!r}"

    def test_stream_terminates_with_done(self, monkeypatch):
        """SSE stream terminates with 'data: [DONE]'."""
        client = _make_proxy_app(monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        )
        assert "data: [DONE]" in r.text

    def test_stream_contains_aurora_metadata_event(self, monkeypatch):
        """SSE stream contains a final aurora metadata event."""
        client = _make_proxy_app(monkeypatch)
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        )
        assert '"aurora"' in r.text
        assert '"governance"' in r.text
