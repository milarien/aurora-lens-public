"""Store/config errors must not be reinterpreted as fresh session state."""
import pytest
from aurora_lens.proxy.config import ProxyConfig
from aurora_lens.proxy.session import _create_session_store
from aurora_lens.proxy.session_store import RedisSessionStore, SessionStoreError

@pytest.mark.parametrize("backend", ["redsi", "unknown"])
def test_bad_mapping_backend_refuses_memory(backend):
    with pytest.raises(ValueError, match="refusing fallback"):
        ProxyConfig.from_mapping({"upstream": {"provider": "openai", "model": "test"}, "session": {"backend": backend}})

@pytest.mark.parametrize("backend", ["redsi", "", "unknown"])
def test_bad_environment_backend_refuses_memory(monkeypatch, backend):
    monkeypatch.setenv("AURORA_LENS_SESSION_BACKEND", backend)
    with pytest.raises(ValueError, match="refusing fallback"):
        ProxyConfig.from_mapping({"upstream": {"provider": "openai", "model": "test"}}).apply_env_overrides()

def test_factory_refuses_unknown_backend():
    with pytest.raises(ValueError, match="refusing memory fallback"):
        _create_session_store("redsi", 3600, "")

def test_redis_without_url_refuses_startup():
    with pytest.raises(ValueError, match="requires"):
        _create_session_store("redis", 3600, "")

class ReadClient:
    def __init__(self, raw=None, unavailable=False):
        self.raw = raw
        self.unavailable = unavailable
    def get(self, key):
        if self.unavailable:
            raise ConnectionError("store unavailable")
        return self.raw

@pytest.mark.parametrize("raw", ["not-json", "{}", "[]", "null", '{"pef_state":{},"expires_at":"invalid","revision":0}'])
def test_corrupt_session_refuses_empty_state(raw):
    store = RedisSessionStore("", client=ReadClient(raw))
    with pytest.raises(SessionStoreError, match="unreadable"):
        store.get("existing")

def test_unavailable_store_refuses_both_reads():
    store = RedisSessionStore("", client=ReadClient(unavailable=True))
    for read in (store.get, store.get_governance_hold):
        with pytest.raises(SessionStoreError, match="failed"):
            read("existing")


@pytest.mark.parametrize("stream", [False, True])
def test_bootstrap_failure_releases_lock_without_generation(monkeypatch, stream):
    from contextlib import contextmanager
    from starlette.testclient import TestClient
    from tests.test_edge_token_gate import _make_app
    from tests.test_proxy import CountingMockAdapter

    class FailedSessions:
        entered = 0
        released = 0
        @contextmanager
        def with_lock(self, session_id):
            self.entered += 1
            try:
                yield
            finally:
                self.released += 1
        def get_or_create(self, session_id):
            raise SessionStoreError("unreadable session")
        def persist(self, *args):
            pytest.fail("bootstrap failure must not persist empty state")
        def cleanup_expired(self):
            return 0
        def active_count(self):
            return 0

    sessions = FailedSessions()
    monkeypatch.setattr("aurora_lens.proxy.app.SessionManager", lambda *a, **k: sessions)
    counter = CountingMockAdapter()
    app, _ = _make_app(monkeypatch, adapter=counter)
    with TestClient(app) as client:
        for _ in range(2):
            response = client.post("/v1/chat/completions", json={
                "model": "gpt-4", "stream": stream,
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
            })
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "session_store_error"
    assert sessions.entered == sessions.released == 2
    assert counter.generate_calls == 0
