import asyncio
import hashlib
import io
import json
import zipfile

from aurora_lens.proxy.audit_archive import archive_response, file_id, retained_files


def body(response):
    async def read():
        return b"".join([chunk async for chunk in response.body_iterator])
    return asyncio.run(read())


def test_full_archive_preserves_all_rows_and_raw_bytes(tmp_path):
    current = tmp_path / "audit.jsonl"
    old = tmp_path / "audit.jsonl.pre_restart.20260429T011700Z"
    dated = tmp_path / "audit.archive.20260427T094841Z.jsonl"
    raw = b"".join(json.dumps({"turn": i}).encode() + b"\n" for i in range(305))
    current.write_bytes(raw)
    old.write_bytes(b'not-json\n{"hash_only":true}\n')
    dated.write_bytes(b'{"earliest":true}')
    (tmp_path / "unrelated-secret.json").write_text("private")
    before = {p.name: p.read_bytes() for p in (current, old, dated)}
    catalog = archive_response(current, None, False, "test-secret")
    assert len(json.loads(catalog.body)["files"]) == 3
    response = archive_response(current, None, True, "test-secret")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(body(response))) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        assert set(bundle.namelist()) == set(before) | {"manifest.json"}
        for info in manifest["files"]:
            assert bundle.read(info["name"]) == before[info["name"]]
            assert info["sha256"] == hashlib.sha256(before[info["name"]]).hexdigest()
    assert {p.name: p.read_bytes() for p in (current, old, dated)} == before


def test_raw_view_and_download_are_complete_and_read_only(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_bytes(b'{"raw":"<script>unsafe</script>"}\n' * 150)
    for download in (True, False):
        r = archive_response(path, file_id(path), download, "secret")
        assert body(r) == path.read_bytes()
        assert r.headers["content-type"].startswith("text/plain")
        assert r.headers["cache-control"] == "private, no-store"
        assert r.headers["content-disposition"].startswith("attachment" if download else "inline")
    assert archive_response(path, "../unrelated", False, "secret").status_code == 404


def test_credential_across_chunk_boundary_withholds_entire_export(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_bytes(b"x" * 65533 + b"server-secret" + b"\n")
    for selected in (None, file_id(path)):
        r = archive_response(path, selected, True, "server-secret")
        assert r.status_code == 503
        assert b"server-secret" not in r.body


def test_snapshot_does_not_include_later_appends(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_bytes(b"original\n")
    r = archive_response(path, file_id(path), False, "secret")
    with path.open("ab") as f:
        f.write(b"later\n")
    assert body(r) == b"original\n"


def test_replaced_current_file_does_not_reuse_selection(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_bytes(b"old\n")
    selected = file_id(path)
    path.rename(tmp_path / "audit.jsonl.pre_restart.20261015T010000Z")
    path.write_bytes(b"new\n")
    assert archive_response(path, selected, False, "secret").status_code == 404
    assert len(retained_files(path)) == 2


def test_authenticated_route_lists_and_exports_without_session_writes(tmp_path, monkeypatch):
    from starlette.testclient import TestClient
    from aurora_lens.proxy.app import create_app
    from aurora_lens.proxy.config import ProxyConfig
    from tests.test_proxy import MockAdapter
    from tests.conftest import PYTEST_AURORA_EDGE_TOKEN

    path = tmp_path / "audit.jsonl"
    raw = b'{"retained":true}\n' * 151
    path.write_bytes(raw)
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_NAME", "production")
    monkeypatch.setattr("aurora_lens.proxy.app._build_provider_adapters", lambda cfg: (MockAdapter(), MockAdapter()))
    cfg = ProxyConfig.from_mapping({"upstream": {"provider": "openai", "api_key": "sk-test", "model": "gpt-4"},
        "auth": {"enabled": False}, "governance": {"audit_log": str(path), "audit_backend": "jsonl"}})
    client = TestClient(create_app(cfg))
    headers = {"x-aurora-edge-token": PYTEST_AURORA_EDGE_TOKEN}
    catalog = client.get("/v1/audit/archive", headers=headers)
    assert catalog.status_code == 200
    selected = catalog.json()["files"][0]["id"]
    assert client.get("/v1/audit/archive", params={"file": selected}, headers=headers).content == raw
    response = client.get("/v1/audit/archive?download=true", headers=headers)
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        assert bundle.read("audit.jsonl") == raw
    assert path.read_bytes() == raw

