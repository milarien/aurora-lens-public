"""Authorised, read-only decision evidence review. Never logs recovered content."""
from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import hmac
from html import escape
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlencode

from fastapi.responses import HTMLResponse, JSONResponse

from aurora_lens.govern.evidence_vault import (
    CaptureMode, EvidenceAccessDenied, EvidenceIntegrityError,
    EvidenceNotFoundError, LocalSealedEvidenceVault,
    evidence_vault_dir_for_audit, resolve_evidence_encryption_key,
)
from aurora_lens.proxy.audit_archive import retained_files

REVIEW_PATH = "/v1/audit/evidence-review"
HEADERS = {
    "Cache-Control": "private, no-store", "Pragma": "no-cache",
    "Vary": "Authorization, x-api-key, x-aurora-review-assertion",
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'",
}


class EvidenceState(str, Enum):
    VERIFIED = "verified"
    HASH_ONLY = "hash_only"
    UNAVAILABLE = "unavailable"
    ACCESS_DENIED = "access_denied"
    INTEGRITY_FAILED = "integrity_failed"


@dataclass(frozen=True)
class EvidenceView:
    state: EvidenceState
    reason: str
    text: str | None = None
    sha256: str | None = None

    def to_dict(self):
        return {"state": self.state.value, "reason": self.reason,
                "text": self.text, "sha256": self.sha256}


def review_query(decision: str | None = None):
    return urlencode({"decision": decision}) if decision else ""


def assertion_message(payload: str, query: str) -> bytes:
    return f"evidence-review-v1\nGET\n{REVIEW_PATH}\n{query}\n{payload}".encode()


class ReviewerAuth:
    """A server-signed owner identity, or an explicitly allowed local API label.

    Edge transport credentials alone never grant evidence review. The site signs
    an identity only after checking the existing Cloudflare Access owner JWT.
    """
    def __init__(self):
        self._nonces = {}
        self._lock = threading.Lock()

    def actor(self, request, cfg, decision, *, now=None):
        now = time.time() if now is None else now
        expected = os.environ.get("AURORA_EDGE_TOKEN", "").strip()
        assertion = request.headers.get("x-aurora-review-assertion", "")
        signature = request.headers.get("x-aurora-review-signature", "")
        if assertion:
            try:
                if not expected or len(assertion) > 2048 or len(signature) != 64:
                    return None
                transport = request.headers.get("x-aurora-edge-token", "")
                if not hmac.compare_digest(hashlib.sha256(transport.encode()).digest(), hashlib.sha256(expected.encode()).digest()):
                    return None
                key = hmac.new(expected.encode(), b"aurora-evidence-review-v1", hashlib.sha256).digest()
                want = hmac.new(key, assertion_message(assertion, review_query(decision)), hashlib.sha256).hexdigest()
                if not hmac.compare_digest(want, signature):
                    return None
                data = json.loads(base64.urlsafe_b64decode(assertion + "=" * (-len(assertion) % 4)))
                actor, issued, nonce = data["actor"], data["iat"], data["nonce"]
                if (data.get("purpose") != "forensic_review" or not isinstance(issued, int)
                        or abs(now - issued) > 30 or not re.fullmatch(r"[A-Za-z0-9@._+-]{1,128}", actor)
                        or not re.fullmatch(r"[0-9a-f]{32}", nonce)):
                    return None
                with self._lock:
                    self._nonces = {n: t for n, t in self._nonces.items() if t >= now - 60}
                    if nonce in self._nonces or len(self._nonces) >= 10000:
                        return None
                    self._nonces[nonce] = now
                return "owner:" + actor
            except (ValueError, KeyError, TypeError):
                return None
        # Hosted reviewers must use the signed owner workflow. No browser
        # credential, forged identity header or auth-disabled mode grants access.
        if any(os.environ.get(k) for k in ("RAILWAY_ENVIRONMENT_NAME", "RAILWAY_ENVIRONMENT")):
            return None
        allowed = set(os.environ.get("AURORA_LENS_EVIDENCE_REVIEWERS", "").split(","))
        authorization = request.headers.get("authorization", "")
        supplied = request.headers.get("x-api-key", "")
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:]
        elif authorization.lower().startswith("basic "):
            try:
                _, supplied = base64.b64decode(authorization[6:], validate=True).decode().split(":", 1)
            except (ValueError, UnicodeError):
                return None
        if cfg.auth.enabled and supplied:
            for credential in cfg.auth.keys:
                if credential.label in allowed and hmac.compare_digest(
                    hashlib.sha256(supplied.encode()).digest(), hashlib.sha256(credential.key.encode()).digest()
                ):
                    return "api:" + credential.label
        return None


def decision_id(raw: bytes) -> str:
    return hashlib.sha256(raw.strip()).hexdigest()


def data_of(row):
    payload = row.get("payload", row)
    return payload.get("data", payload)


def decisions(path: Path, selected=None):
    """IDs bind the exact audit bytes, avoiding seq collisions across archives."""
    recent = deque(maxlen=200)
    for source in sorted(retained_files(path), key=lambda p: (p.stat().st_mtime_ns, p.name)):
        with source.open("rb") as stream:
            for raw in stream:
                if selected and decision_id(raw) != selected:
                    continue
                try:
                    row = json.loads(raw)
                    d = data_of(row)
                    if not isinstance(d, dict) or not d.get("action", row.get("op")):
                        continue
                except (ValueError, AttributeError):
                    continue
                item = {"id": decision_id(raw), "seq": row.get("seq"),
                        "time": row.get("time", d.get("timestamp")),
                        "action": d.get("action", row.get("op"))}
                if selected:
                    return item, d
                recent.append(item)
    return None if selected else list(reversed(recent))


def evidence_view(vault, manifest, expected_kind, secrets=()):
    if not manifest:
        return EvidenceView(EvidenceState.UNAVAILABLE, "not_captured")
    if manifest.get("capture_status") == "hash_only" or manifest.get("protection_mode") == "hash_only":
        return EvidenceView(EvidenceState.HASH_ONLY, "no_recoverable_blob", sha256=manifest.get("raw_sha256"))
    if vault is None:
        return EvidenceView(EvidenceState.UNAVAILABLE, "vault_or_key_unavailable")
    try:
        if manifest.get("evidence_kind") != expected_kind:
            return EvidenceView(EvidenceState.INTEGRITY_FAILED, "manifest_kind_mismatch")
        stored = vault._load_manifest(manifest["evidence_ref"]).to_dict()
        if stored != manifest:
            return EvidenceView(EvidenceState.INTEGRITY_FAILED, "manifest_mismatch")
        content = vault.get_evidence(manifest["evidence_ref"], purpose="forensic_review", actor="forensic_reviewer")
        digest = hashlib.sha256(content).hexdigest()
        if digest != manifest["raw_sha256"] or len(content) != manifest["byte_length"]:
            return EvidenceView(EvidenceState.INTEGRITY_FAILED, "hash_or_length_mismatch")
        text = content.decode("utf-8")
        if any(secret and secret in text for secret in secrets):
            return EvidenceView(EvidenceState.UNAVAILABLE, "sensitive_configuration_withheld")
        return EvidenceView(EvidenceState.VERIFIED, "sha256_and_length_match", text, digest)
    except EvidenceAccessDenied:
        return EvidenceView(EvidenceState.ACCESS_DENIED, "vault_access_denied")
    except EvidenceIntegrityError:
        return EvidenceView(EvidenceState.INTEGRITY_FAILED, "sealed_integrity_failed")
    except (EvidenceNotFoundError, OSError, RuntimeError, UnicodeError):
        return EvidenceView(EvidenceState.UNAVAILABLE, "blob_or_key_unavailable")
    except (KeyError, TypeError, ValueError):
        return EvidenceView(EvidenceState.INTEGRITY_FAILED, "invalid_manifest")


def existing_vault(cfg):
    root = evidence_vault_dir_for_audit(cfg.governance.audit_log)
    key = resolve_evidence_encryption_key(cfg.governance.evidence_encryption_key)
    if not root or not root.is_dir():
        return None
    # A retrieval handle: no sealed-mode write probes, capture, key creation or
    # manifest appends. get_evidence uses each stored manifest's protection mode.
    return LocalSealedEvidenceVault(root=root, master_key=key, capture_mode=CaptureMode.HASH_ONLY)


class AccessAudit:
    """Durable, content-free events, separate from the governance decision chain."""
    def __init__(self, audit_path):
        self.path = Path(str(audit_path) + ".evidence-access.jsonl") if audit_path else None
        self._lock = threading.Lock()

    def record(self, actor, selected, result, states=None):
        if self.path is None:
            raise OSError("access_audit_unavailable")
        event = {"event": "evidence_review_access", "time": datetime.now(timezone.utc).isoformat(),
                 "actor": actor or "unauthenticated", "purpose": "forensic_review",
                 "decision_id": selected if selected and re.fullmatch(r"[0-9a-f]{64}", selected) else None,
                 "result": result, "states": states or {}}
        raw = (json.dumps(event, separators=(",", ":")) + "\n").encode()
        with self._lock:
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(self.path, flags, 0o600)
            try:
                if os.write(fd, raw) != len(raw):
                    raise OSError("access_audit_write_failed")
                os.fsync(fd)
            finally:
                os.close(fd)


def response(body, status=200, html=False):
    return HTMLResponse(body, status_code=status, headers=HEADERS) if html else JSONResponse(body, status_code=status, headers=HEADERS)


class EvidenceReview:
    def __init__(self, cfg):
        self.cfg = cfg
        self.auth = ReviewerAuth()
        self.audit = AccessAudit(cfg.governance.audit_log)

    def handle(self, request):
        html = "text/html" in request.headers.get("accept", "")
        selected = request.query_params.get("decision")
        actor = self.auth.actor(request, self.cfg, selected)
        result = "denied"
        body = {"state": "access_denied", "message": "Evidence reviewer authorisation required."}
        status = 403
        try:
            secrets = [os.environ.get("AURORA_EDGE_TOKEN", ""), os.environ.get("AURORA_LENS_EVIDENCE_KEY", ""),
                       self.cfg.governance.evidence_encryption_key or "", self.cfg.governance.audit_signing_key or "",
                       self.cfg.upstream.api_key or "", *self.cfg.governance.audit_signing_keys,
                       *[k.key for k in self.cfg.auth.keys]]
            if actor:
                if request.method != "GET":
                    result, status = "method_denied", 405
                    body = {"state": "access_denied", "message": "Evidence review is read-only."}
                elif (any(k != "decision" for k in request.query_params)
                      or len(request.query_params.getlist("decision")) > 1
                      or (selected is not None and not re.fullmatch(r"[0-9a-f]{64}", selected))):
                    result, status = "invalid_selection", 400
                    body = {"state": "unavailable", "message": "Invalid decision selection."}
                else:
                    path = Path(self.cfg.governance.audit_log) if self.cfg.governance.audit_log else None
                    found = decisions(path, selected) if path and path.parent.is_dir() else None
                    if found is None:
                        result, status = "unavailable", 404
                        body = {"state": "unavailable", "message": "Decision unavailable. Refresh the decision list."}
                    elif not selected:
                        result, status = "listed", 200
                        body = {"decisions": found}
                    else:
                        item, data = found
                        vault = existing_vault(self.cfg)
                        req = evidence_view(vault, data.get("request_evidence_manifest"), "request_prompt", secrets)
                        upstream = evidence_view(vault, data.get("upstream_evidence_manifest"), "upstream_model_output", secrets)
                        # A legacy hash-only row may predate manifests.
                        if not data.get("request_evidence_manifest") and data.get("request_capture_status") == "hash_only":
                            req = EvidenceView(EvidenceState.HASH_ONLY, "no_recoverable_blob", sha256=data.get("request_raw_sha256"))
                        if not data.get("upstream_evidence_manifest") and data.get("blocked_response_hash"):
                            upstream = EvidenceView(EvidenceState.HASH_ONLY, "no_recoverable_blob", sha256=data["blocked_response_hash"])
                        body = {"decision": item, "ruling": {
                            "action": item["action"], "rationale": data.get("rationale"),
                            "flags": [{k: f.get(k) for k in ("flag_type", "type", "rule_id", "severity") if f.get(k) is not None}
                                      if isinstance(f, dict) else str(f) for f in data.get("flags", [])],
                        }, "request": req.to_dict(), "withheld_response": upstream.to_dict(),
                            "released_response": data.get("governed_response")}
                        result, status = "viewed", 200
            encoded = json.dumps(body, ensure_ascii=False)
            if any(s and s in encoded for s in secrets):
                body = {"state": "unavailable", "message": "Sensitive configuration withheld."}
                result, status = "configuration_withheld", 503
            states = {k: body[k]["state"] for k in ("request", "withheld_response") if k in body}
        except Exception:
            # Never return or log an exception containing a blob, path or key.
            result, status = "unavailable", 503
            body = {"state": "unavailable", "message": "Evidence temporarily unavailable."}
            states = {}
        try:
            self.audit.record(actor, selected, result, states)
        except Exception:
            # Durable access auditing is mandatory before any plaintext release.
            status = 503
            body = {"state": "unavailable", "message": "Access audit unavailable. Evidence was not released."}
        rendered = render_review(body) if html else body
        out = response(rendered, status, html)
        if (not actor and status == 403 and self.cfg.auth.enabled
                and not any(os.environ.get(k) for k in ("RAILWAY_ENVIRONMENT_NAME", "RAILWAY_ENVIRONMENT"))):
            out.status_code = 401
            out.headers["WWW-Authenticate"] = 'Basic realm="Evidence review", charset="UTF-8"'
        return out


def render_review(body, base=""):
    esc = lambda value: escape(str(value if value is not None else ""))
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Evidence review · Aurora-Lens</title>',
            '<style>body{font:16px system-ui;background:#f4f7fa;color:#17212b;max-width:1050px;margin:2rem auto;padding:0 1rem}section,article{background:white;border:1px solid #ccd5df;border-radius:8px;padding:1.2rem;margin:1rem 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px ui-monospace,monospace}a{color:#154faa}td,th{text-align:left;padding:.7rem;border-bottom:1px solid #dde}table{width:100%}.state{font-weight:700}</style>',
            '<h1>Decision evidence review</h1><p>Authorised forensic review · every access is audited · original evidence is preserved.</p>']
    if "decisions" in body:
        page += ['<p>Latest 200 decisions across retained audit files. Open a decision to recover its captured evidence.</p><table><thead><tr><th>Sequence</th><th>Recorded time (UTC)</th><th>Ruling</th><th>Review</th></tr></thead><tbody>']
        for d in body["decisions"]:
            page.append(f'<tr><td>{esc(d["seq"])}</td><td>{esc(d["time"])}</td><td>{esc(d["action"])}</td><td><a href="{esc(base)}?decision={d["id"]}">Open decision</a></td></tr>')
        page.append('</tbody></table>')
    elif "decision" in body:
        d = body["decision"]
        page += [f'<p><a href="{esc(base or "?")}">Back to decisions</a></p><section><h2>Ruling: {esc(d["action"])}</h2><p>Sequence {esc(d["seq"])} · {esc(d["time"])}</p>',
                 '<p>' + esc(body["ruling"].get("rationale") or "No rationale was recorded.") + '</p>']
        flags = body["ruling"].get("flags", [])
        if flags:
            page.append('<p>Recorded findings:</p><ul>')
            for flag in flags:
                label = (flag.get("flag_type") or flag.get("type") or flag.get("rule_id") or "Recorded finding") if isinstance(flag, dict) else flag
                page.append('<li>' + esc(label) + '</li>')
            page.append('</ul>')
        else:
            page.append('<p>No flags were recorded.</p>')
        page.append('</section>')
        for name, title in (("request", "Captured request"), ("withheld_response", "Withheld model response")):
            v = body[name]
            page += [f'<article><h2>{title}</h2><p class="state">{esc(v["state"].replace("_", " ").title())}</p><p>{esc(v["reason"].replace("_", " "))}</p>',
                     f'<p>Hash verification: {"PASS — SHA-256 and length match" if v["state"] == "verified" else "FAIL" if v["state"] == "integrity_failed" else "Not verified"}</p>']
            if v["sha256"]:
                page.append('<p>SHA-256: <code>' + esc(v["sha256"]) + '</code></p>')
            if v["text"] is not None:
                page.append('<pre>' + esc(v["text"]) + '</pre>')
            page.append('</article>')
        if body.get("released_response"):
            page += ['<article><h2>Released answer</h2><p>Recorded audit text; not a recovered sealed blob.</p><pre>' + esc(body["released_response"]) + '</pre></article>']
    else:
        page += ['<section><h2>' + esc(body.get("state", "unavailable").replace("_", " ").title()) + '</h2><p>' + esc(body.get("message", "Evidence review unavailable.")) + '</p></section>']
    return "".join(page) + '</html>'
