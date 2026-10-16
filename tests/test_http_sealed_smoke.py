"""Offline contract tests; importing/running pytest never restarts a service."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from scripts import smoke_http_sealed_evidence as smoke


PROJECT = "12345678-1234-4234-8234-123456789012"
SECRET = "synthetic-secret-must-not-be-printed"
OPTIONS = smoke.Options(PROJECT, public_origin="https://public.invalid", railway_origin="https://origin.invalid")


class FakeRailway:
    def __init__(self):
        self.restarted = False
        self.calls = []
        self.after_change = None
        self.link_closed = False
        self.remote_result = "PASS\n"
        self.transient = False

    @contextmanager
    def linked(self, options):
        try:
            yield self
        finally:
            self.link_closed = True

    def json(self, args):
        self.calls.append(args)
        if args[0] == "redeploy":
            self.restarted = True
            return {"id": "new-deployment"}
        if args[0] == "variables":
            return {
                "AURORA_LENS_EVIDENCE_KEY": "changed-key" if self.restarted and self.after_change == "key" else SECRET,
                "AURORA_LENS_EVIDENCE_CAPTURE_MODE": "sealed",
                "AURORA_LENS_GOV_AUDIT_LOG": "/data/audit.jsonl",
                "AURORA_EDGE_TOKEN": "synthetic-edge-secret",
            }
        assert args == ["status", "--json"]
        if self.restarted and self.transient:
            self.transient = False
            raise smoke.Unavailable()
        deployment = {
            "id": "new-deployment" if self.restarted else "old-deployment",
            "status": "SUCCESS", "meta": {"commitHash": "revision"},
        }
        if self.restarted and self.after_change in ("commit", "failed", "building"):
            if self.after_change == "commit":
                deployment["meta"]["commitHash"] = "different-revision"
            else:
                deployment["status"] = "FAILED" if self.after_change == "failed" else "BUILDING"
        service = {"serviceId": "service-id", "serviceName": "aurora_lens", "latestDeployment": deployment,
                   "domains": {"serviceDomains": [{"domain": "origin.invalid"}], "customDomains": []}}
        return {"id": PROJECT, "environments": {"edges": [{"node": {
            "id": "environment-id", "name": "production",
            "serviceInstances": {"edges": [{"node": service}]},
            "volumeInstances": {"edges": [{"node": {"serviceId": "service-id", "mountPath": "/data",
                "volume": {"id": "changed-volume" if self.restarted and self.after_change == "volume" else "original-volume"}}}]},
        }}]}}

    def text(self, args):
        self.calls.append(args)
        assert args[0] == "ssh"
        assert "gosu aurora python -E -c" in args[-1]
        assert SECRET not in args[-1]
        return self.remote_result


class FakeWeb:
    def __init__(self, railway):
        self.railway = railway
        self.calls = []
        self.same_process = False
        self.change_audit = False
        self.row = {
            "session_id": "fresh-session", "action": "HARD_STOP",
            "request_raw_sha256": "sha256:" + hashlib.sha256(smoke.PROMPT.encode()).hexdigest(),
            "request_capture_status": "sealed_by_policy", "evidence_capture_mode": "sealed",
        }
        for field, kind, ref, basis in (
            ("request_evidence_manifest", "request_prompt", "ev:request", "governance:hard_stop"),
            ("upstream_evidence_manifest", "upstream_model_output", "ev:upstream", "hard_stop_post_llm"),
        ):
            self.row[field] = {"evidence_kind": kind, "evidence_ref": ref, "capture_basis": basis,
                "protection_mode": "sealed", "capture_status": "captured", "byte_length": 30,
                "raw_sha256": "0" * 64, "encryption_algorithm": "AELV1-HMAC-CTR"}

    def json(self, url, *, data=None, headers=None):
        self.calls.append((url, data, headers))
        path = urlsplit(url).path
        if path == "/health":
            return {
                "status": "ok", "audit_sink": "ledger", "audit_log_path": "/data/audit.jsonl",
                "git_commit": "revision",
                "proxy_run_id": "new-process" if self.railway.restarted and not self.same_process else "old-process",
                "evidence_capture": {"enabled": True, "key_available": True, "vault_available": True,
                    "required_sealed": True, "ready": True, "requested_mode": "sealed", "effective_mode": "sealed", "failure_reason": None},
            }
        if path == "/readyz":
            return {"ready": True}
        if path == "/v1/session/new-scenario":
            assert data == {}
            return {"session_id": "fresh-session", "demo_session_token": "synthetic-session-secret"}
        if path == "/v1/chat/completions":
            assert data == {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": smoke.PROMPT}], "aurora_session_id": "fresh-session"}
            assert "x-aurora-edge-token" not in headers
            return {"sensitive_body": SECRET}
        assert path == "/v1/audit/recent"
        row = deepcopy(self.row)
        if self.railway.restarted and self.change_audit:
            row["timestamp"] = "changed"
        return {"entries": [{"payload": {"data": row}}]}


def exercise(railway=None, web=None, **kwargs):
    railway = railway or FakeRailway()
    web = web or FakeWeb(railway)
    smoke.run(OPTIONS, railway=railway, web=web, sleep=lambda _: None, **kwargs)
    return railway, web


def test_complete_flow_uses_exact_refs_and_never_mutates_key_or_volume(capsys):
    rail, web = exercise()
    assert sum(c[0] == "redeploy" for c in rail.calls) == 1
    checks = [c[-1] for c in rail.calls if c[0] == "ssh"]
    assert len(checks) == 2 and checks[0] == checks[1]
    assert all(c[0] in ("status", "variables", "ssh", "redeploy") for c in rail.calls)
    assert sum(urlsplit(c[0]).path == "/v1/chat/completions" for c in web.calls) == 1
    assert rail.link_closed and capsys.readouterr().out == ""


@pytest.mark.parametrize("change", ["pre_llm", "hash_only", "same_ref", "wrong_action", "raw_prompt", "old_session"])
def test_no_restart_unless_exact_post_llm_sealed_capture_is_proven(change):
    rail = FakeRailway(); web = FakeWeb(rail)
    if change == "pre_llm":
        web.row.pop("upstream_evidence_manifest")
    elif change == "hash_only":
        web.row["upstream_evidence_manifest"]["protection_mode"] = "hash_only"
    elif change == "same_ref":
        web.row["upstream_evidence_manifest"]["evidence_ref"] = "ev:request"
    elif change == "wrong_action":
        web.row["action"] = "FORCE_REVISE"
    elif change == "raw_prompt":
        web.row["unexpected_text"] = smoke.PROMPT
    else:
        web.row["session_id"] = "old-session"
    with pytest.raises((smoke.Failure, KeyError)):
        exercise(rail, web)
    assert not rail.restarted and rail.link_closed


@pytest.mark.parametrize("change", ["key", "volume", "commit", "failed"])
def test_fails_if_restart_changes_existing_deployment_requirements(change):
    rail = FakeRailway(); rail.after_change = change
    with pytest.raises(smoke.Failure):
        exercise(rail)


@pytest.mark.parametrize("same_process,changed_audit", [(True, False), (False, True)])
def test_same_process_or_changed_original_audit_cannot_pass(same_process, changed_audit):
    rail = FakeRailway(); web = FakeWeb(rail)
    web.same_process = same_process; web.change_audit = changed_audit
    with pytest.raises(smoke.Failure):
        exercise(rail, web)


def test_transient_restart_reads_do_not_duplicate_model_or_restart():
    rail = FakeRailway(); rail.transient = True
    rail, web = exercise(rail)
    assert sum(c[0] == "redeploy" for c in rail.calls) == 1
    assert sum(urlsplit(c[0]).path == "/v1/chat/completions" for c in web.calls) == 1


def test_restart_timeout_is_bounded():
    rail = FakeRailway(); rail.after_change = "building"
    ticks = iter([0, 1, 400])
    with pytest.raises(smoke.Failure):
        exercise(rail, clock=lambda: next(ticks))


@pytest.mark.parametrize("argv", [[], ["--run"], ["--project", PROJECT], ["--run", "--project", "bad"], ["--unknown", SECRET]])
def test_explicit_opt_in_and_valid_project_required_before_any_io(argv, capsys):
    calls = []
    assert smoke.main(argv, runner=lambda options: calls.append(options)) == 1
    output = capsys.readouterr()
    assert output.out == "FAIL\n" and output.err == "" and not calls


@pytest.mark.parametrize("fails", [False, True])
def test_cli_prints_only_pass_fail_even_if_underlying_tools_leak(fails, capsys):
    def runner(options):
        print(SECRET)
        print(SECRET, file=smoke.sys.stderr)
        if fails:
            raise RuntimeError(SECRET)
    assert smoke.main(["--run", "--project", PROJECT], runner=runner) == int(fails)
    output = capsys.readouterr()
    assert output.out == ("FAIL\n" if fails else "PASS\n") and output.err == ""


@pytest.mark.parametrize("origin", ["http://origin.invalid", "https://user:password@origin.invalid", "https://origin.invalid/?secret=x"])
def test_origins_cannot_leak_credentials(origin):
    with pytest.raises(smoke.Failure):
        replace(OPTIONS, railway_origin=origin).validate()


@pytest.mark.parametrize("damage", [None, "tamper", "wrong_key", "auth_bypass"])
def test_remote_program_checks_real_encrypted_blobs_and_access(tmp_path, monkeypatch, capsys, damage):
    from aurora_lens.govern.evidence_vault import create_local_evidence_vault, LocalSealedEvidenceVault
    from aurora_lens.proxy.config import ProxyConfig

    path = tmp_path / "audit.jsonl"
    vault = create_local_evidence_vault(path, encryption_key=SECRET)
    manifests = [vault.put_evidence(kind, text).to_dict() for kind, text in (
        ("request_prompt", "synthetic request"), ("upstream_model_output", "synthetic withheld response"))]
    cfg = SimpleNamespace(audit_log=str(path), evidence_capture_mode="sealed", evidence_encryption_key=SECRET)
    monkeypatch.setattr(ProxyConfig, "from_yaml", lambda _: SimpleNamespace(governance=cfg))
    if damage == "tamper":
        vault._blob_path(manifests[1]["evidence_ref"]).write_bytes(b"synthetic corrupt bytes")
    elif damage == "wrong_key":
        cfg.evidence_encryption_key = "synthetic-different-key"
    elif damage == "auth_bypass":
        original = LocalSealedEvidenceVault.get_evidence
        def bypass(self, ref, purpose, actor):
            return b"synthetic" if actor == "anonymous" else original(self, ref, purpose, actor)
        monkeypatch.setattr(LocalSealedEvidenceVault, "get_evidence", bypass)
    exec(smoke.REMOTE_CHECK, {"target": {"audit_path": str(path), "manifests": manifests}})
    output = capsys.readouterr()
    assert output.out == ("PASS\n" if damage is None else "FAIL\n") and output.err == ""
