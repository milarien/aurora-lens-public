"""Opt-in public HTTP HARD_STOP -> Railway restart -> exact-blob recovery.

See docs/evidence-vault-operations.md. This command prints exactly PASS or FAIL.
No runtime policy, key, variable or volume configuration is modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import hmac
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID


PROMPT = (
    "What is a one-sentence example of a dosage label for an invented medicine "
    "called Examplezol, using an imaginary adult patient and a quantity "
    "expressed in milligrams?"
)


class Failure(Exception):
    pass


class Unavailable(Failure):
    """A transport operation failed; only read operations may be retried."""


def require(condition):
    if not condition:
        raise Failure()


@dataclass(frozen=True)
class Options:
    project: str
    service: str = "aurora_lens"
    environment: str = "production"
    public_origin: str = "https://aurora-lens.ai"
    railway_origin: str = "https://auroralens-production.up.railway.app"
    railway_cli: str = "railway"
    timeout: int = 300

    def validate(self):
        UUID(self.project)
        for value in (self.service, self.environment):
            require(bool(value) and all(c.isalnum() or c in "-_" for c in value))
        for origin in (self.public_origin, self.railway_origin):
            url = urlsplit(origin)
            require(url.scheme == "https" and url.hostname and not url.username and not url.password)
            require(not url.query and not url.fragment and url.path in ("", "/"))
        require(30 <= self.timeout <= 1800)


class Railway:
    def __init__(self, executable):
        found = shutil.which(executable)
        require(found is not None)
        # npm on Windows installs a wrapper; prefer its actual executable.
        binary = Path(found).parent / "node_modules/@railway/cli/bin/railway.exe"
        self.executable = str(binary) if os.name == "nt" and binary.is_file() else found
        self.cwd = None

    def text(self, args):
        try:
            result = subprocess.run(
                [self.executable, *args], cwd=self.cwd, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=45,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise Unavailable() from None
        if result.returncode:
            raise Unavailable()
        return result.stdout

    def json(self, args):
        try:
            return json.loads(self.text(args))
        except ValueError:
            raise Unavailable() from None

    @contextmanager
    def linked(self, options):
        # Do not change the operator's checkout link or depend on its environment.
        with tempfile.TemporaryDirectory(prefix="aurora-sealed-smoke-") as directory:
            self.cwd = directory
            linked = False
            try:
                self.text(["link", "--project", options.project, "--service", options.service,
                           "--environment", options.environment, "--json"])
                linked = True
                yield self
            finally:
                if linked:
                    self.text(["unlink", "--yes", "--json"])
                self.cwd = None


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Web:
    def json(self, url, *, data=None, headers=None):
        headers = {"Accept": "application/json", "User-Agent": "Aurora-Sealed-Smoke/1", **(headers or {})}
        body = None if data is None else json.dumps(data).encode()
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            with build_opener(NoRedirect()).open(Request(url, data=body, headers=headers), timeout=45) as response:
                # Bound audit-response memory; never log bodies or HTTP exceptions.
                raw = response.read(16_000_001)
                require(response.status == 200 and len(raw) <= 16_000_000)
                return json.loads(raw)
        except (OSError, ValueError):
            raise Unavailable() from None


def snapshot(railway, options):
    state = railway.json(["status", "--json"])
    require(state["id"] == options.project)
    env = next(e["node"] for e in state["environments"]["edges"]
               if options.environment in (e["node"]["id"], e["node"]["name"]))
    service = next(e["node"] for e in env["serviceInstances"]["edges"]
                   if options.service in (e["node"]["serviceId"], e["node"]["serviceName"]))
    volumes = [e["node"] for e in env["volumeInstances"]["edges"] if e["node"]["serviceId"] == service["serviceId"]]
    require(len(volumes) == 1)
    hosts = {d["domain"] for group in service["domains"].values() for d in group}
    require(urlsplit(options.railway_origin).netloc in hosts)
    return {
        "environment": env["id"], "service": service["serviceId"],
        "deployment": service["latestDeployment"], "volume": volumes[0]["volume"]["id"],
        "mount": volumes[0]["mountPath"],
    }


def variables(railway, state):
    values = railway.json(["variables", "--service", state["service"], "--environment", state["environment"], "--json"])
    require(bool(values.get("AURORA_LENS_EVIDENCE_KEY", "").strip()))
    require(values.get("AURORA_LENS_EVIDENCE_CAPTURE_MODE") == "sealed")
    require(bool(values.get("AURORA_EDGE_TOKEN", "").strip()))
    return values


def health(web, options, values):
    result = web.json(options.railway_origin.rstrip("/") + "/health",
                      headers={"x-aurora-edge-token": values["AURORA_EDGE_TOKEN"]})
    state = result["evidence_capture"]
    require(result["status"] == "ok" and result["audit_sink"] == "ledger")
    require(all(state[k] is True for k in ("enabled", "key_available", "vault_available", "required_sealed", "ready")))
    require(state["requested_mode"] == state["effective_mode"] == "sealed" and state["failure_reason"] is None)
    require(web.json(options.railway_origin.rstrip("/") + "/readyz") == {"ready": True})
    return result


def audit_target(web, options, values, session):
    rows = web.json(options.railway_origin.rstrip("/") + "/v1/audit/recent?n=200",
                    headers={"x-aurora-edge-token": values["AURORA_EDGE_TOKEN"]})["entries"]
    expected = hashlib.sha256(PROMPT.encode()).hexdigest()
    matches = []
    for row in rows:
        payload = row.get("payload", row)
        item = payload.get("data", payload)
        if item.get("session_id") == session and str(item.get("request_raw_sha256", "")).removeprefix("sha256:") == expected:
            matches.append((row, item))
    require(len(matches) == 1)
    row, item = matches[0]
    require(item.get("action", item.get("outcome")) == "HARD_STOP")
    require(item.get("request_capture_status") == "sealed_by_policy" and item.get("evidence_capture_mode") == "sealed")
    manifests = [item["request_evidence_manifest"], item["upstream_evidence_manifest"]]
    require([m["evidence_kind"] for m in manifests] == ["request_prompt", "upstream_model_output"])
    require(manifests[1]["capture_basis"] == "hard_stop_post_llm")
    require(manifests[0]["evidence_ref"] != manifests[1]["evidence_ref"])
    for manifest in manifests:
        require(manifest["protection_mode"] == "sealed" and manifest["capture_status"] == "captured")
        require(manifest["encryption_algorithm"] == "AELV1-HMAC-CTR" and manifest["byte_length"] > 0)
    require(not item.get("original_response") and PROMPT not in json.dumps(row))
    return {"manifests": manifests, "audit_digest": hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


REMOTE_CHECK = '''
import contextlib,io,json
ok=False
with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
 try:
  import hashlib,os
  from aurora_lens.proxy.config import ProxyConfig
  from aurora_lens.govern.evidence_vault import create_local_evidence_vault,EvidenceAccessDenied,EVIDENCE_VAULT_BLOB_MAGIC
  cfg=ProxyConfig.from_yaml(os.environ.get('AURORA_LENS_CONFIG','/app/aurora-lens.yaml')).governance
  assert cfg.audit_log==target['audit_path'] and cfg.evidence_capture_mode=='sealed'
  vault=create_local_evidence_vault(cfg.audit_log,capture_mode='sealed',encryption_key=cfg.evidence_encryption_key)
  for manifest in target['manifests']:
   ref=manifest['evidence_ref']
   assert vault._load_manifest(ref).to_dict()==manifest
   content=vault.get_evidence(ref,purpose='forensic_review',actor='forensic_reviewer')
   assert content and len(content)==manifest['byte_length']
   assert hashlib.sha256(content).hexdigest()==manifest['raw_sha256']
   blob=vault._blob_path(ref).read_bytes()
   assert blob.startswith(EVIDENCE_VAULT_BLOB_MAGIC) and content not in blob
   assert vault.verify_evidence(ref,manifest['raw_sha256'])
   try:
    vault.get_evidence(ref,purpose='forensic_review',actor='anonymous')
   except EvidenceAccessDenied:
    pass
   else:
    raise RuntimeError()
   del content
  ok=True
 except Exception:
  pass
print('PASS' if ok else 'FAIL')
'''


def recover(railway, options, target):
    # Only audit manifests are sent to SSH; never a key or plaintext evidence.
    payload = json.dumps(target).encode().hex()
    code = f"import json\ntarget=json.loads(bytes.fromhex('{payload}'))\n" + REMOTE_CHECK
    command = f'''gosu aurora python -E -c "exec(bytes.fromhex('{code.encode().hex()}'))"'''
    result = railway.text(["ssh", "--project", options.project, "--service", options.service,
                           "--environment", options.environment, "--", command])
    require(result.strip() == "PASS")


def same_storage(left, right):
    require(all(left[k] == right[k] for k in ("environment", "service", "volume", "mount")))


def run(options, *, railway=None, web=None, sleep=time.sleep, clock=time.monotonic):
    options.validate()
    railway = railway or Railway(options.railway_cli)
    web = web or Web()
    with railway.linked(options):
        before = snapshot(railway, options)
        values = variables(railway, before)
        ready = health(web, options, values)
        require(before["deployment"]["status"] == "SUCCESS")
        require(ready["git_commit"] == before["deployment"]["meta"]["commitHash"])
        audit_path = PurePosixPath(ready["audit_log_path"])
        require(audit_path.is_absolute() and audit_path.is_relative_to(PurePosixPath(before["mount"])))
        session = web.json(options.public_origin.rstrip("/") + "/v1/session/new-scenario", data={})
        sid = session["session_id"]
        response = web.json(options.public_origin.rstrip("/") + "/v1/chat/completions", data={
            "model": "gpt-4o-mini", "messages": [{"role": "user", "content": PROMPT}], "aurora_session_id": sid,
        }, headers={"x-aurora-demo-session": session["demo_session_token"], "x-aurora-session-id": sid,
                    "x-aurora-pef-context-id": sid, "x-aurora-operator-detail": "1"})
        del response  # Never print, save or treat the response body as evidence.
        target = audit_target(web, options, values, sid)
        target["audit_path"] = str(audit_path)
        recover(railway, options, target)
        current = snapshot(railway, options)
        same_storage(before, current)
        require(current["deployment"]["id"] == before["deployment"]["id"])
        restart = railway.json(["redeploy", "--service", before["service"], "--yes", "--json"])["id"]
        require(restart != before["deployment"]["id"])
        deadline = clock() + options.timeout
        while True:
            require(clock() < deadline)
            try:
                after = snapshot(railway, options)
                same_storage(before, after)
                deployment = after["deployment"]
                require(deployment["id"] in (before["deployment"]["id"], restart))
                if deployment["id"] == restart:
                    require(deployment["status"] not in ("FAILED", "CRASHED", "REMOVED"))
                    require(deployment["meta"]["commitHash"] == ready["git_commit"])
                    if deployment["status"] == "SUCCESS":
                        restarted_values = variables(railway, after)
                        require(hmac.compare_digest(values["AURORA_LENS_EVIDENCE_KEY"].encode(), restarted_values["AURORA_LENS_EVIDENCE_KEY"].encode()))
                        require(values.get("AURORA_LENS_GOV_AUDIT_LOG") == restarted_values.get("AURORA_LENS_GOV_AUDIT_LOG"))
                        restarted = health(web, options, restarted_values)
                        require(restarted["git_commit"] == ready["git_commit"])
                        require(restarted["proxy_run_id"] != ready["proxy_run_id"])
                        require(restarted["audit_log_path"] == ready["audit_log_path"])
                        break
            except Unavailable:
                pass  # Retry only reads; never duplicate the model call/restart.
            sleep(5)
        require(audit_target(web, options, restarted_values, sid) == {k: v for k, v in target.items() if k != "audit_path"})
        recover(railway, options, target)


def main(argv=None, *, runner=run):
    passed = False
    # Suppress parser errors, subprocess/library diagnostics and exception text.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        try:
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("--run", action="store_true")
            parser.add_argument("--project")
            for name, default in (("service", "aurora_lens"), ("environment", "production"),
                                  ("public-origin", "https://aurora-lens.ai"),
                                  ("railway-origin", "https://auroralens-production.up.railway.app"),
                                  ("railway-cli", "railway")):
                parser.add_argument("--" + name, default=default)
            parser.add_argument("--timeout", type=int, default=300)
            args = vars(parser.parse_args(argv))
            require(args.pop("run") and args["project"])
            options = Options(**args)
            options.validate()
            runner(options)
            passed = True
        except (Exception, SystemExit, KeyboardInterrupt):
            pass
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
