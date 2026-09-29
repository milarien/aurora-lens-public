"""Cross-pipeline regression: STOP hold corridor before any upstream/model invocation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from aurora_lens.config import LensConfig
from aurora_lens.govern.bridge import BuiltinBridge
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractionResult
from aurora_lens.lens import Lens, _GOVERNED_STOP_CONTINUATION_TEXT
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import EPISTEMIC_MODE_STOP, PEFState
from tests.skip_reasons import SKIP_STARLETTE_HTTP_TESTCLIENT
from tests.stop_corridor_helpers import (
    TURN1_TEXT,
    TURN2_LAUNDER,
    TURN3_RECALL,
    assert_continued_stop_turn,
    assert_turn1_hard_stop,
    assert_turn3_recall_guards,
    adapter_counter,
    run_stream_stop_corridor_sequence,
    run_sync_stop_corridor_sequence,
)
from tests.test_lens import MockAdapter, _read_last_jsonl_object
from tests.test_state_native_engine_delegation import (
    _EmptyExtractBackend,
    _pef_silver_key_at_safe_literal,
)


def _load_stream_helpers():
    path = Path(__file__).resolve().parent / "test_streaming_governance.py"
    spec = importlib.util.spec_from_file_location("stream_helpers_stop_xpipe", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FakeStreamingAdapter, mod._collect_stream


class _ClarificationHeavyBackend(ExtractionBackend):
    """Would run extraction on every turn if the stop corridor did not short-circuit."""

    def __init__(self) -> None:
        self.extract_calls = 0

    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:
        self.extract_calls += 1
        return ExtractionResult(
            claims=[],
            entity_mentions=["James", "Richard"],
            span=Span.PRESENT,
            ambiguous_referents=["his"],
        )


def _stop_corridor_lens_config(
    adapter,
    audit_path: Path,
    **extra,
) -> LensConfig:
    base = dict(
        adapter=adapter,
        governance_bridge=BuiltinBridge(audit_path=str(audit_path)),
        auto_interpret=False,
        auto_verify=True,
    )
    base.update(extra)
    return LensConfig(**base)


@pytest.mark.asyncio
async def test_stop_corridor_sync_process(tmp_path: Path) -> None:
    audit = tmp_path / "sync.jsonl"
    adapter = MockAdapter(responses=["NEVER_RETURNED"])
    lens = Lens(_stop_corridor_lens_config(adapter, audit))
    await run_sync_stop_corridor_sequence(lens, adapter, audit)


@pytest.mark.asyncio
async def test_stop_corridor_stream_process(tmp_path: Path) -> None:
    FakeStreamingAdapter, collect_stream = _load_stream_helpers()
    audit = tmp_path / "stream.jsonl"
    adapter = FakeStreamingAdapter(["NEVER_RETURNED"])
    lens = Lens(
        _stop_corridor_lens_config(
            adapter,
            audit,
            stream_emit_progress=False,
        )
    )
    await run_stream_stop_corridor_sequence(
        lens, adapter, audit, collect_stream=collect_stream
    )


@pytest.mark.asyncio
async def test_stop_corridor_sync_auto_interpret_clarification_route(
    tmp_path: Path,
) -> None:
    """auto_interpret + verify: clarification/extraction paths must not bypass STOP."""
    audit = tmp_path / "clarify_sync.jsonl"
    backend = _ClarificationHeavyBackend()
    adapter = MockAdapter(responses=["NEVER_RETURNED"])
    lens = Lens(
        _stop_corridor_lens_config(
            adapter,
            audit,
            extraction_backend=backend,
            auto_interpret=True,
            inject_pef_context=False,
        )
    )
    await run_sync_stop_corridor_sequence(lens, adapter, audit)
    assert backend.extract_calls == 0, "stop corridor must run before extraction"


@pytest.mark.asyncio
async def test_stop_corridor_stream_auto_interpret_clarification_route(
    tmp_path: Path,
) -> None:
    FakeStreamingAdapter, collect_stream = _load_stream_helpers()
    audit = tmp_path / "clarify_stream.jsonl"
    backend = _ClarificationHeavyBackend()
    adapter = FakeStreamingAdapter(["NEVER_RETURNED"])
    lens = Lens(
        _stop_corridor_lens_config(
            adapter,
            audit,
            extraction_backend=backend,
            auto_interpret=True,
            inject_pef_context=False,
            stream_emit_progress=False,
        )
    )
    await run_stream_stop_corridor_sequence(
        lens, adapter, audit, collect_stream=collect_stream
    )
    assert backend.extract_calls == 0


@pytest.mark.asyncio
async def test_stop_corridor_sync_state_native_route(tmp_path: Path) -> None:
    audit = tmp_path / "state_native_sync.jsonl"
    adapter = MockAdapter(responses=["NEVER_RETURNED"])
    lens = Lens(
        _stop_corridor_lens_config(
            adapter,
            audit,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
        ),
        initial_pef=_pef_silver_key_at_safe_literal(),
    )
    await run_sync_stop_corridor_sequence(lens, adapter, audit)


@pytest.mark.asyncio
async def test_stop_corridor_stream_state_native_route(tmp_path: Path) -> None:
    FakeStreamingAdapter, collect_stream = _load_stream_helpers()
    audit = tmp_path / "state_native_stream.jsonl"
    adapter = FakeStreamingAdapter(["NEVER_RETURNED"])
    lens = Lens(
        _stop_corridor_lens_config(
            adapter,
            audit,
            extraction_backend=_EmptyExtractBackend(),
            enable_state_native_delegation=True,
            stream_emit_progress=False,
        ),
        initial_pef=_pef_silver_key_at_safe_literal(),
    )
    await run_stream_stop_corridor_sequence(
        lens, adapter, audit, collect_stream=collect_stream
    )


@pytest.mark.asyncio
async def test_stop_corridor_sync_active_continuation_route(tmp_path: Path) -> None:
    """Stale continuation capability must not bypass STOP hold on follow-up turns."""
    audit = tmp_path / "continuation_sync.jsonl"
    adapter = MockAdapter(responses=["NEVER_RETURNED"])
    lens = Lens(_stop_corridor_lens_config(adapter, audit))
    counter = adapter_counter(adapter)
    r1 = await lens.process(TURN1_TEXT)
    assert_turn1_hard_stop(r1, lens, counter)
    lens.pef.active_continuation_capability = "neutral_timeline"
    lens.pef.active_continuation_context = {"domain": "legal"}
    r2 = await lens.process(TURN2_LAUNDER)
    assert_continued_stop_turn(
        r2, lens, counter, audit, turn_num=2, expected_upstream_calls=0
    )
    r3 = await lens.process(TURN3_RECALL)
    assert_continued_stop_turn(
        r3, lens, counter, audit, turn_num=3, expected_upstream_calls=0
    )
    assert_turn3_recall_guards(r3)


def _proxy_stop_corridor_client(monkeypatch, tmp_path: Path, *, stream: bool):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip(SKIP_STARLETTE_HTTP_TESTCLIENT)

    from aurora_lens.proxy.app import create_app
    from aurora_lens.proxy.config import ProxyConfig
    from tests.test_proxy import CountingMockAdapter

    upstream = CountingMockAdapter()
    extraction_adapter = CountingMockAdapter()
    monkeypatch.setattr(
        "aurora_lens.proxy.app._build_provider_adapters",
        lambda cfg: (upstream, extraction_adapter),
    )
    audit = tmp_path / f"proxy_{'stream' if stream else 'sync'}.jsonl"
    cfg = ProxyConfig.from_mapping({
        "upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        "governance": {
            "default_policy": "strict",
            "audit_log": str(audit),
            "audit_backend": "jsonl",
            "include_operator_detail": True,
            "allow_operator_detail_via_header": True,
        },
        "extraction": {"backend": "spacy"},
    })
    client = TestClient(create_app(cfg))
    sid = f"stop-corridor-{'stream' if stream else 'sync'}"
    headers = {
        "x-aurora-operator-detail": "1",
        "x-aurora-session-id": sid,
    }
    return client, upstream, extraction_adapter, audit, headers, stream


def _proxy_post_turn(
    client, headers, stream: bool, content: str, *, external_flags: list | None = None
) -> dict:
    body = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }
    if external_flags:
        body["aurora"] = {"external_flags": external_flags}
    r = client.post("/v1/chat/completions", json=body, headers=headers)
    assert r.status_code == 200
    return r.json()


def _assert_proxy_continued_stop(body: dict, audit: Path, turn_num: int) -> None:
    aurora = body["aurora"]
    content = body["choices"][0]["message"]["content"]
    assert aurora["governance"] == "HARD_STOP"
    assert aurora["governance"] != "PASS"
    assert content == _GOVERNED_STOP_CONTINUATION_TEXT
    assert aurora.get("llm_invoked") is False
    row = _read_last_jsonl_object(audit)
    assert row.get("pef_turn_classification") == "stopped"
    assert "continued-stop" in (row.get("rationale") or "")


def test_stop_corridor_proxy_session_sync(monkeypatch, tmp_path: Path) -> None:
    client, upstream, extraction, audit, headers, _stream = _proxy_stop_corridor_client(
        monkeypatch, tmp_path, stream=False
    )
    t1_ext_flags = [{"type": "EMPLOYMENT_DISCRIMINATION_FACILITATION", "evidence": ["age discrimination screening"]}]
    b1 = _proxy_post_turn(client, headers, False, TURN1_TEXT, external_flags=t1_ext_flags)
    assert b1["aurora"]["governance"] == "HARD_STOP"
    assert upstream.generate_calls == 0
    assert extraction.generate_calls == 0

    b2 = _proxy_post_turn(client, headers, False, TURN2_LAUNDER)
    _assert_proxy_continued_stop(b2, audit, turn_num=2)
    assert upstream.generate_calls == 0

    b3 = _proxy_post_turn(client, headers, False, TURN3_RECALL)
    _assert_proxy_continued_stop(b3, audit, turn_num=3)
    assert "conversation just started" not in b3["choices"][0]["message"]["content"].lower()
    assert "discrimination" not in b3["choices"][0]["message"]["content"].lower()
    assert upstream.generate_calls == 0


def test_stop_corridor_proxy_session_stream(monkeypatch, tmp_path: Path) -> None:
    client, upstream, extraction, audit, headers, _stream = _proxy_stop_corridor_client(
        monkeypatch, tmp_path, stream=True
    )
    t1_ext_flags = [{"type": "EMPLOYMENT_DISCRIMINATION_FACILITATION", "evidence": ["age discrimination screening"]}]
    b1 = _proxy_post_turn(client, headers, True, TURN1_TEXT, external_flags=t1_ext_flags)
    assert b1["aurora"]["governance"] == "HARD_STOP"
    assert upstream.generate_calls == 0
    assert upstream.generate_stream_calls == 0

    b2 = _proxy_post_turn(client, headers, True, TURN2_LAUNDER)
    _assert_proxy_continued_stop(b2, audit, turn_num=2)
    assert upstream.generate_stream_calls == 0

    b3 = _proxy_post_turn(client, headers, True, TURN3_RECALL)
    _assert_proxy_continued_stop(b3, audit, turn_num=3)
    assert upstream.generate_stream_calls == 0
