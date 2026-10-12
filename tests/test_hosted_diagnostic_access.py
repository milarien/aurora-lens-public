"""Hosted diagnostics fail closed before handlers; public summaries contain no forensic state."""
import pytest
from starlette.testclient import TestClient
from tests.test_edge_token_gate import _make_app
from tests.conftest import PYTEST_AURORA_EDGE_TOKEN

@pytest.mark.parametrize('path', ['/v1/audit/recent?n=1','/v1/audit/verify','/v1/audit/search','/v1/audit/entry','/v1/audit/anomaly-check','/v1/audit/future-route','/v1/session/operator-pef'])
@pytest.mark.parametrize('token', [None,'wrong'])
def test_hosted_diagnostics_denied_before_handlers(monkeypatch,path,token):
    monkeypatch.setenv('RAILWAY_ENVIRONMENT_NAME','production')
    app,_ = _make_app(monkeypatch)
    def forbidden(*a,**k): raise AssertionError('backend reached')
    monkeypatch.setattr('aurora_lens.proxy.app._read_audit_tail',forbidden)
    headers={'x-test-omit-edge-token':'1'}
    if token: headers['x-aurora-edge-token']=token
    response=TestClient(app).get(path,headers=headers)
    assert response.status_code==403
    assert response.json()=={'error':{'message':'Forbidden','type':'authentication_error'}}

def test_hosted_public_health_and_summary_are_minimal(monkeypatch):
    monkeypatch.setenv('RAILWAY_ENVIRONMENT_NAME','production')
    app,_=_make_app(monkeypatch);client=TestClient(app)
    anon={'x-test-omit-edge-token':'1'}
    assert set(client.get('/health',headers=anon).json())=={'status','application_version','git_commit'}
    assert set(client.get('/v1/operator/summary',headers=anon).json())=={'system_status','sessions','decision_window_n','recent_decisions','warning_count','block_count'}
    assert 'proxy_run_id' in client.get('/health',headers={'x-aurora-edge-token':PYTEST_AURORA_EDGE_TOKEN}).json()
    assert client.get('/v1/audit/recent?n=1',headers={'x-aurora-edge-token':PYTEST_AURORA_EDGE_TOKEN}).status_code==200

def test_hosted_diagnostics_missing_secret_fails_closed(monkeypatch):
    monkeypatch.setenv('RAILWAY_ENVIRONMENT_NAME','production')
    monkeypatch.delenv('AURORA_EDGE_TOKEN',raising=False)
    app,_=_make_app(monkeypatch)
    assert TestClient(app).get('/v1/audit/recent?n=1').status_code==403
