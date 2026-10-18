"""Opt-in, loopback-only synthetic evidence review preview. Never uses live data.

The root page simulates an authenticated reviewer. The API path still requires
the actual signed assertion. /denied demonstrates an unauthorised request.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import urlsplit
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starlette.requests import Request
from aurora_lens.govern.evidence_vault import create_local_evidence_vault
from aurora_lens.proxy.config import GovernanceConfig, ProxyConfig, UpstreamConfig
from aurora_lens.proxy.evidence_review import EvidenceReview, REVIEW_PATH, assertion_message, review_query

SYNTHETIC_TRANSPORT = "synthetic-preview-transport-not-a-live-credential"


def synthetic_headers(decision=None):
    data = {"actor": "preview@example.test", "purpose": "forensic_review",
            "iat": int(time.time()), "nonce": uuid.uuid4().hex}
    payload = base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode()).decode().rstrip("=")
    key = hmac.new(SYNTHETIC_TRANSPORT.encode(), b"aurora-evidence-review-v1", hashlib.sha256).digest()
    return {"x-aurora-edge-token": SYNTHETIC_TRANSPORT, "x-aurora-review-assertion": payload,
            "x-aurora-review-signature": hmac.new(key, assertion_message(payload, review_query(decision)), hashlib.sha256).hexdigest()}


def preview_service(directory):
    path = Path(directory) / "synthetic-audit.jsonl"
    vault = create_local_evidence_vault(path, encryption_key="synthetic-preview-encryption-material")
    request = vault.put_evidence("request_prompt", "Synthetic request: review a demonstration item. No real person or decision is involved.")
    withheld = vault.put_evidence("upstream_model_output", "Synthetic withheld response: an unsupported conclusion for the demonstration item.")
    hashed = vault.put_evidence("request_prompt", "Synthetic hash-only request.", {"protection_mode": "hash_only"})
    rows = []
    for seq, label, req, upstream in [
        (1, "Synthetic sealed capture", request.to_dict(), withheld.to_dict()),
        (2, "Synthetic hash-only capture", hashed.to_dict(), None),
        (3, "Synthetic unavailable capture", None, None),
    ]:
        rows.append({"seq": seq, "time": "2026-10-19T00:00:00Z", "payload": {"data": {
            "action": "HARD_STOP", "rationale": label, "flags": [{"flag_type": "SYNTHETIC_DEMONSTRATION", "severity": "error"}],
            "request_evidence_manifest": req, "upstream_evidence_manifest": upstream}}})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    # Construct directly: inherited deployment environment overrides must never
    # redirect this synthetic preview to an operator's real storage/config.
    cfg = ProxyConfig(
        upstream=UpstreamConfig(provider="openai", api_key="synthetic-unused-provider", model="synthetic-unused-model"),
        governance=GovernanceConfig(audit_log=str(path), evidence_encryption_key="synthetic-preview-encryption-material"),
    )
    return EvidenceReview(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly start synthetic local preview")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not args.run or not 0 <= args.port <= 65535:
        parser.error("Use --run and a valid local port.")
    if any(os.environ.get(k) for k in ("RAILWAY_ENVIRONMENT_NAME", "RAILWAY_ENVIRONMENT")):
        parser.error("Synthetic preview cannot run in a Railway deployment.")
    # Only this preview process gets synthetic credentials. Production config,
    # the existing stable key, and persistent storage are never read or changed.
    os.environ["AURORA_EDGE_TOKEN"] = SYNTHETIC_TRANSPORT
    os.environ["AURORA_LENS_EVIDENCE_KEY"] = "synthetic-preview-encryption-material"
    os.environ["RAILWAY_ENVIRONMENT_NAME"] = "synthetic-local-preview"
    with tempfile.TemporaryDirectory(prefix="aurora-evidence-review-preview-") as directory:
        service = preview_service(directory)

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                url = urlsplit(self.path)
                if url.path not in ("/", "/denied", REVIEW_PATH):
                    self.send_error(404)
                    return
                headers = {k.lower(): v for k, v in self.headers.items()}
                if url.path == "/":
                    from urllib.parse import parse_qs
                    selected = parse_qs(url.query).get("decision", [None])[-1]
                    headers.update(synthetic_headers(selected))
                if url.path == "/denied":
                    headers = {}
                headers.setdefault("accept", "text/html")
                scope = {"type": "http", "method": "GET", "path": REVIEW_PATH,
                         "query_string": url.query.encode(), "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}
                result = service.handle(Request(scope))
                body = result.body
                if "text/html" in result.headers.get("content-type", ""):
                    banner = '<p><strong>Synthetic local preview — reviewer identity is simulated on the preview home page.</strong> <a href="/">Decisions</a> · <a href="/denied">Denied access example</a></p>'
                    body = body.replace(b'<h1>', banner.encode() + b'<h1>', 1)
                self.send_response(result.status_code)
                for name, value in result.headers.items():
                    if name.lower() != "content-length":
                        self.send_header(name, value)
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        print(json.dumps({"preview_url": f"http://127.0.0.1:{server.server_port}/"}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
