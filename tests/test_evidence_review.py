"""Synthetic-only evidence review and denial regression tests; no live calls."""
import base64
import hashlib
import hmac
import json
import time
from types import SimpleNamespace
import uuid

import pytest
from starlette.requests import Request

from aurora_lens.govern.evidence_vault import (
    EvidenceAccessDenied, create_local_evidence_vault,
)
from aurora_lens.proxy.evidence_review import (
    AccessAudit, EvidenceState, ReviewerAuth, REVIEW_PATH, assertion_message,
    decisions, evidence_view, render_review, review_query,
)


def signed_headers(secret, decision=None, **overrides):
    data = {"actor": "synthetic-reviewer@example.test", "purpose": "forensic_review",
            "iat": int(time.time()), "nonce": uuid.uuid4().hex, **overrides}
    payload = base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode()).decode().rstrip("=")
    key = hmac.new(secret.encode(), b"aurora-evidence-review-v1", hashlib.sha256).digest()
    return {"x-aurora-edge-token": secret, "x-aurora-review-assertion": payload,
            "x-aurora-review-signature": hmac.new(key, assertion_message(payload, review_query(decision)), hashlib.sha256).hexdigest()}


def request(headers):
    return Request({"type": "http", "headers": [(k.encode(), v.encode()) for k, v in headers.items()]})


@pytest.fixture
def captured(tmp_path):
    path = tmp_path / "audit.jsonl"
    vault = create_local_evidence_vault(path, encryption_key="synthetic-review-key")
    manifests = [vault.put_evidence("request_prompt", "Synthetic request <script>unsafe</script>"),
                 vault.put_evidence("upstream_model_output", "Synthetic withheld output\u202fwith punctuation\u2011only")]
    row = {"seq": 7, "time": "2026-10-16T00:00:00Z", "payload": {"data": {
        "action": "HARD_STOP", "rationale": "Synthetic ruling", "flags": [{"flag_type": "MEDICAL_DOSAGE_RECOMMENDATION"}],
        "request_evidence_manifest": manifests[0].to_dict(), "upstream_evidence_manifest": manifests[1].to_dict()}}}
    path.write_text(json.dumps(row) + "\n")
    return path, vault, manifests, row


def test_authorised_recovery_matches_audit_manifest_and_is_read_only(captured):
    path, vault, manifests, _ = captured
    before = {p: p.read_bytes() for p in path.parent.rglob("*") if p.is_file()}
    for m, kind in zip(manifests, ["request_prompt", "upstream_model_output"]):
        view = evidence_view(vault, m.to_dict(), kind)
        assert view.state == EvidenceState.VERIFIED
        assert hashlib.sha256(view.text.encode()).hexdigest() == view.sha256 == m.raw_sha256
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_unavailable_hash_only_and_denied_are_distinct(captured, monkeypatch):
    _, vault, manifests, _ = captured
    m = manifests[0].to_dict()
    assert evidence_view(vault, None, "request_prompt").state == EvidenceState.UNAVAILABLE
    assert evidence_view(None, m, "request_prompt").state == EvidenceState.UNAVAILABLE
    hashed = {**m, "capture_status": "hash_only", "protection_mode": "hash_only"}
    assert evidence_view(None, hashed, "request_prompt").state == EvidenceState.HASH_ONLY
    monkeypatch.setattr(vault, "get_evidence", lambda *a, **k: (_ for _ in ()).throw(EvidenceAccessDenied("sensitive diagnostic")))
    denied = evidence_view(vault, m, "request_prompt")
    assert denied.state == EvidenceState.ACCESS_DENIED and denied.text is None


def test_corrupt_blob_and_manifest_never_release_plaintext(captured):
    _, vault, manifests, _ = captured
    m = manifests[0].to_dict()
    assert evidence_view(vault, {**m, "raw_sha256": "0" * 64}, "request_prompt").state == EvidenceState.INTEGRITY_FAILED
    blob = vault._blob_path(m["evidence_ref"])
    raw = blob.read_bytes()
    blob.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    result = evidence_view(vault, m, "request_prompt")
    assert result.state == EvidenceState.INTEGRITY_FAILED and result.text is None


def test_wrong_key_and_missing_blob_fail_safely(captured):
    path, vault, manifests, _ = captured
    wrong = create_local_evidence_vault(path, encryption_key="different-synthetic-key")
    assert evidence_view(wrong, manifests[0].to_dict(), "request_prompt").state == EvidenceState.INTEGRITY_FAILED
    vault._blob_path(manifests[0].evidence_ref).unlink()
    assert evidence_view(vault, manifests[0].to_dict(), "request_prompt").state == EvidenceState.UNAVAILABLE


def test_configuration_secret_in_evidence_is_withheld(captured):
    _, vault, manifests, _ = captured
    assert evidence_view(vault, manifests[0].to_dict(), "request_prompt", ["Synthetic request"]).text is None


def test_owner_assertion_is_bound_to_selection_purpose_time_and_transport(monkeypatch):
    monkeypatch.setenv("AURORA_EDGE_TOKEN", "synthetic-transport")
    monkeypatch.setenv("AURORA_LENS_EVIDENCE_KEY", "synthetic-review-key")
    cfg = SimpleNamespace(auth=SimpleNamespace(enabled=False))
    auth = ReviewerAuth()
    headers = signed_headers("synthetic-transport", "a" * 64)
    assert auth.actor(request(headers), cfg, "b" * 64) is None
    assert auth.actor(request({**headers, "x-aurora-edge-token": "wrong"}), cfg, "a" * 64) is None
    assert auth.actor(request(headers), cfg, "a" * 64) == "owner:synthetic-reviewer@example.test"
    assert auth.actor(request(headers), cfg, "a" * 64) is None  # replay
    for change in ({"iat": 1}, {"iat": int(time.time()) + 100}, {"purpose": "other"}, {"actor": "bad\nactor"}):
        assert auth.actor(request(signed_headers("synthetic-transport", **change)), cfg, None) is None
    assert auth.actor(request({"x-aurora-edge-token": "synthetic-transport", "x-aurora-review-actor": "owner"}), cfg, None) is None


def test_local_api_review_requires_explicit_label_and_enabled_auth(monkeypatch):
    for name in ["RAILWAY_ENVIRONMENT_NAME", "RAILWAY_ENVIRONMENT"]:
        monkeypatch.delenv(name, raising=False)
    cfg = SimpleNamespace(auth=SimpleNamespace(enabled=True, keys=[SimpleNamespace(label="reviewer", key="synthetic-login")]))
    auth = ReviewerAuth()
    headers = {"authorization": "Basic " + base64.b64encode(b"reviewer:synthetic-login").decode()}
    assert auth.actor(request(headers), cfg, None) is None
    monkeypatch.setenv("AURORA_LENS_EVIDENCE_REVIEWERS", "reviewer")
    assert auth.actor(request(headers), cfg, None) == "api:reviewer"
    cfg.auth.enabled = False
    assert auth.actor(request(headers), cfg, None) is None


def test_selection_handles_seq_collision_and_excludes_unrelated_files(captured):
    path, _, _, row = captured
    old = path.with_name("audit.jsonl.pre_restart.20261015T010000Z")
    old.write_text(json.dumps({**row, "time": "older"}) + "\n")
    path.with_name("secret.jsonl").write_text(json.dumps(row))
    items = decisions(path)
    assert len(items) == 2 and len({r["id"] for r in items}) == 2
    for item in items:
        selected, data = decisions(path, item["id"])
        assert selected == item and data["action"] == "HARD_STOP"
    assert decisions(path, "f" * 64) is None


def test_access_audit_contains_identity_result_but_no_content(captured):
    path, _, _, _ = captured
    log = AccessAudit(path)
    log.record("owner:synthetic-reviewer@example.test", "a" * 64, "viewed", {"request": "verified"})
    log.record(None, "untrusted raw content", "denied")
    rows = [json.loads(line) for line in log.path.read_text().splitlines()]
    assert rows[0]["result"] == "viewed" and rows[1]["actor"] == "unauthenticated"
    assert rows[1]["decision_id"] is None
    assert "Synthetic request" not in log.path.read_text()
    with pytest.raises(OSError):
        AccessAudit(None).record(None, None, "denied")


def test_html_escapes_evidence_and_distinguishes_verification_states():
    view = {"state": "verified", "reason": "sha256_and_length_match", "sha256": "a" * 64, "text": "<script>not executable</script>"}
    body = {"decision": {"action": "HARD_STOP", "seq": 1, "time": "synthetic"}, "ruling": {"action": "HARD_STOP"}, "request": view,
            "withheld_response": {**view, "state": "hash_only", "reason": "no_recoverable_blob", "text": None}}
    html = render_review(body)
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "Hash Only" in html and "PASS" in html and "Not verified" in html


@pytest.fixture
def review_client(captured, monkeypatch):
    from starlette.testclient import TestClient
    from aurora_lens.proxy.app import create_app
    from aurora_lens.proxy.config import ProxyConfig
    from tests.test_proxy import MockAdapter
    path, vault, manifests, row = captured
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_NAME", "synthetic")
    monkeypatch.setenv("AURORA_EDGE_TOKEN", "synthetic-transport")
    monkeypatch.setenv("AURORA_LENS_EVIDENCE_KEY", "synthetic-review-key")
    monkeypatch.setattr("aurora_lens.proxy.app._build_provider_adapters", lambda cfg: (MockAdapter(), MockAdapter()))
    cfg = ProxyConfig.from_mapping({"upstream": {"provider": "openai", "api_key": "synthetic-provider-credential", "model": "synthetic"},
        "auth": {"enabled": False}, "governance": {"audit_log": str(path), "audit_backend": "jsonl", "evidence_encryption_key": "synthetic-review-key"}})
    return TestClient(create_app(cfg)), path, vault, manifests, row


def test_complete_http_workflow_lists_opens_verifies_and_audits(review_client, caplog):
    client, path, _, manifests, _ = review_client
    before = path.read_bytes()
    catalog = client.get(REVIEW_PATH, headers=signed_headers("synthetic-transport"))
    assert catalog.status_code == 200
    selected = catalog.json()["decisions"][0]["id"]
    detail = client.get(REVIEW_PATH + "?decision=" + selected, headers=signed_headers("synthetic-transport", selected))
    assert detail.status_code == 200
    data = detail.json()
    assert data["ruling"]["action"] == "HARD_STOP"
    assert data["request"]["state"] == data["withheld_response"]["state"] == "verified"
    assert data["request"]["sha256"] == manifests[0].raw_sha256
    headers = {**signed_headers("synthetic-transport", selected), "accept": "text/html"}
    page = client.get(REVIEW_PATH + "?decision=" + selected, headers=headers)
    assert page.status_code == 200 and "Captured request" in page.text and "Withheld model response" in page.text
    assert "&lt;script&gt;" in page.text and "<script>" not in page.text
    assert page.headers["cache-control"] == "private, no-store"
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert path.read_bytes() == before
    access = path.with_name(path.name + ".evidence-access.jsonl").read_text()
    assert [json.loads(line)["result"] for line in access.splitlines()] == ["listed", "viewed", "viewed"]
    assert "Synthetic request" not in caplog.text + access
    assert "Synthetic withheld" not in caplog.text + access
    assert "synthetic-review-key" not in caplog.text + access + detail.text


def test_edge_credential_alone_and_forged_actor_denied_without_vault_read(review_client, monkeypatch):
    client, path, _, _, _ = review_client
    monkeypatch.setattr("aurora_lens.proxy.evidence_review.existing_vault", lambda *a: pytest.fail("denied access reached vault"))
    for headers in ({"x-test-omit-edge-token": "1"}, {"x-aurora-edge-token": "synthetic-transport"},
                    {"x-aurora-edge-token": "synthetic-transport", "x-aurora-review-actor": "owner"}):
        r = client.get(REVIEW_PATH, headers=headers)
        assert r.status_code == 403 and r.json()["state"] == "access_denied"
        assert "Synthetic request" not in r.text
    records = [json.loads(x) for x in path.with_name(path.name + ".evidence-access.jsonl").read_text().splitlines()]
    assert len(records) == 3 and all(r["result"] == "denied" for r in records)


def test_access_audit_failure_prevents_http_plaintext_release(review_client, monkeypatch):
    client, path, _, _, _ = review_client
    selected = decisions(path)[0]["id"]
    monkeypatch.setattr(AccessAudit, "record", lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic failure")))
    r = client.get(REVIEW_PATH + "?decision=" + selected, headers=signed_headers("synthetic-transport", selected))
    assert r.status_code == 503 and r.json()["state"] == "unavailable"
    assert "Synthetic request" not in r.text and "Synthetic withheld" not in r.text


@pytest.mark.parametrize("query", ["?decision=../secret", "?decision=" + "a" * 64 + "&decision=" + "b" * 64, "?key=secret"])
def test_invalid_selection_never_reads_vault(review_client, monkeypatch, query):
    client, _, _, _, _ = review_client
    from urllib.parse import parse_qs
    selected = parse_qs(query[1:]).get("decision", [None])[-1]
    monkeypatch.setattr("aurora_lens.proxy.evidence_review.existing_vault", lambda *a: pytest.fail("invalid query reached vault"))
    r = client.get(REVIEW_PATH + query, headers=signed_headers("synthetic-transport", selected))
    assert r.status_code == 400


def test_read_only_endpoint_rejects_post_and_replayed_assertion(review_client):
    client, path, _, _, _ = review_client
    before = path.read_bytes()
    assert client.post(REVIEW_PATH, headers=signed_headers("synthetic-transport")).status_code == 405
    headers = signed_headers("synthetic-transport")
    assert client.get(REVIEW_PATH, headers=headers).status_code == 200
    assert client.get(REVIEW_PATH, headers=headers).status_code == 403
    assert path.read_bytes() == before


def test_http_missing_blob_and_vault_denial_are_visibly_distinct(review_client, monkeypatch):
    client, path, vault, manifests, _ = review_client
    selected = decisions(path)[0]["id"]
    vault._blob_path(manifests[1].evidence_ref).unlink()
    url = REVIEW_PATH + "?decision=" + selected
    data = client.get(url, headers=signed_headers("synthetic-transport", selected)).json()
    assert data["request"]["state"] == "verified"
    assert data["withheld_response"]["state"] == "unavailable"
    def deny(*args, **kwargs):
        raise EvidenceAccessDenied("sensitive diagnostic never returned")
    monkeypatch.setattr(type(vault), "get_evidence", deny)
    headers = {**signed_headers("synthetic-transport", selected), "accept": "text/html"}
    page = client.get(url, headers=headers)
    assert "Access Denied" in page.text and "Not verified" in page.text
    assert "sensitive diagnostic" not in page.text and "Synthetic request" not in page.text


def test_http_hash_only_and_released_answer_have_no_false_verification(review_client):
    client, path, vault, _, row = review_client
    m = vault.put_evidence("request_prompt", "Synthetic hash-only content", {"protection_mode": "hash_only"})
    d = row["payload"]["data"]
    d.update(action="PASS", request_evidence_manifest=m.to_dict(), upstream_evidence_manifest=None,
             governed_response="Synthetic released answer", flags=[])
    path.write_text(json.dumps(row) + "\n")
    selected = decisions(path)[0]["id"]
    headers = {**signed_headers("synthetic-transport", selected), "accept": "text/html"}
    page = client.get(REVIEW_PATH + "?decision=" + selected, headers=headers)
    assert "Hash Only" in page.text and "Synthetic released answer" in page.text
    assert "Hash verification: PASS" not in page.text and "Synthetic hash-only content" not in page.text


def test_http_missing_key_withholds_content_without_modifying_vault(review_client, monkeypatch):
    from aurora_lens.proxy.evidence_review import EvidenceReview
    client, path, vault, _, _ = review_client
    selected = decisions(path)[0]["id"]
    before = {p: p.read_bytes() for p in vault.root.rglob("*") if p.is_file()}
    monkeypatch.delenv("AURORA_LENS_EVIDENCE_KEY")
    # A retrieval configuration with no key; no startup/capture policy changes.
    from aurora_lens.proxy.config import ProxyConfig
    cfg = ProxyConfig.from_mapping({"upstream": {"provider": "openai", "api_key": "synthetic", "model": "synthetic"},
        "governance": {"audit_log": str(path), "evidence_capture_mode": "hash_only"}})
    service = EvidenceReview(cfg)
    scope = {"type": "http", "method": "GET", "path": REVIEW_PATH,
             "query_string": ("decision=" + selected).encode(),
             "headers": [(k.encode(), v.encode()) for k, v in signed_headers("synthetic-transport", selected).items()]}
    data = json.loads(service.handle(Request(scope)).body)
    assert data["request"]["state"] == data["withheld_response"]["state"] == "unavailable"
    assert data["request"]["text"] is None
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_catalog_accidental_configuration_secret_is_never_returned(review_client):
    client, path, _, _, row = review_client
    row["payload"]["data"]["action"] = "synthetic-transport"
    path.write_text(json.dumps(row) + "\n")
    r = client.get(REVIEW_PATH, headers=signed_headers("synthetic-transport"))
    assert r.status_code == 503 and "synthetic-transport" not in r.text


def test_preview_cannot_inherit_deployment_configuration(tmp_path, monkeypatch):
    from scripts.preview_evidence_review import preview_service
    monkeypatch.setenv("AURORA_LENS_UPSTREAM_API_KEY", "synthetic-unrelated-credential")
    monkeypatch.setenv("AURORA_LENS_EVIDENCE_KEY", "synthetic-unrelated-key")
    monkeypatch.setenv("AURORA_LENS_AUTH_ENABLED", "true")
    service = preview_service(tmp_path)
    assert service.cfg.governance.audit_log == str(tmp_path / "synthetic-audit.jsonl")
    assert service.cfg.governance.evidence_encryption_key == "synthetic-preview-encryption-material"
    assert service.cfg.upstream.api_key == "synthetic-unused-provider"
    assert service.cfg.auth.enabled is False
    import os
    assert os.environ["AURORA_LENS_EVIDENCE_KEY"] == "synthetic-unrelated-key"
