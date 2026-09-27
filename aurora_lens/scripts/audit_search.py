#!/usr/bin/env python3
"""CLI to search audit log by outcome, tenant, time range.

Usage:
  python -m aurora_lens.scripts.audit_search --path audit.jsonl --outcome HARD_STOP --last 24h
  python -m aurora_lens.scripts.audit_search --url http://127.0.0.1:8081 --outcome FORCE_REVISE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import urllib.parse
    import urllib.request
    import urllib.error
except ImportError:
    urllib = None  # type: ignore


def search_via_http(
    base_url: str,
    outcome: str | None,
    tenant: str | None,
    since: str | None,
    limit: int,
    domain: str | None = None,
    non_admit_only: bool = False,
) -> list[dict]:
    params = []
    if outcome:
        params.append(f"outcome={outcome}")
    if tenant:
        params.append(f"tenant_label={tenant}")
    if domain:
        params.append(f"domain={urllib.parse.quote(domain, safe='')}")
    if non_admit_only:
        params.append("non_admit_only=true")
    if since:
        params.append(f"since={since}")
    params.append(f"limit={limit}")
    qs = "&".join(params)
    url = f"{base_url.rstrip('/')}/v1/audit/search?{qs}"
    api_key = os.environ.get("AURORA_LENS_API_KEY", "")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    return data.get("entries", [])


def search_via_file(
    path: Path,
    outcome: str | None,
    tenant: str | None,
    since: str | None,
    limit: int,
    domain: str | None = None,
    non_admit_only: bool = False,
) -> list[dict]:
    import datetime
    if not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    entries = []
    for ln in lines[-5000:]:
        try:
            entries.append(json.loads(ln))
        except Exception:
            continue
    # Filter
    since_dt = None
    if since:
        since = since.strip()
        if since.endswith("h"):
            try:
                hours = int(since[:-1])
                since_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)
            except ValueError:
                pass
        elif since.endswith("d"):
            try:
                days = int(since[:-1])
                since_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
            except ValueError:
                pass
    result = []
    for e in reversed(entries):
        if outcome and e.get("outcome") != outcome:
            continue
        if tenant is not None and e.get("tenant_label") != tenant:
            continue
        if domain is not None and e.get("request_domain") != domain:
            continue
        if non_admit_only:
            oc = e.get("outcome")
            if oc in ("PASS", "SOFT_CORRECT"):
                continue
        if since_dt and e.get("timestamp"):
            try:
                ts = datetime.datetime.fromisoformat(str(e["timestamp"]).replace("Z", "+00:00"))
                if ts < since_dt:
                    continue
            except Exception:
                pass
        result.append(e)
        if len(result) >= limit:
            break
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Search aurora-lens audit log")
    parser.add_argument("--path", help="Path to audit.jsonl (local file search)")
    parser.add_argument("--url", default="http://127.0.0.1:8081", help="Proxy base URL (HTTP search)")
    parser.add_argument("--outcome", help="Filter: PASS, SOFT_CORRECT, FORCE_REVISE, HARD_STOP")
    parser.add_argument("--tenant", help="Filter by tenant_label")
    parser.add_argument(
        "--domain",
        help="Filter by request_domain (from operator-channel header metadata at write time)",
    )
    parser.add_argument(
        "--non-admit-only",
        action="store_true",
        help="Exclude PASS and SOFT_CORRECT",
    )
    parser.add_argument("--last", dest="since", help="Time range: 24h, 7d")
    parser.add_argument("--limit", type=int, default=50, help="Max entries to return")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if urllib is None and args.url:
        print("Error: urllib required for --url", file=sys.stderr)
        return 1

    if args.path:
        entries = search_via_file(
            Path(args.path),
            outcome=args.outcome,
            tenant=args.tenant,
            since=args.since,
            limit=args.limit,
            domain=args.domain,
            non_admit_only=args.non_admit_only,
        )
    else:
        entries = search_via_http(
            args.url,
            outcome=args.outcome,
            tenant=args.tenant,
            since=args.since,
            limit=args.limit,
            domain=args.domain,
            non_admit_only=args.non_admit_only,
        )

    if args.json:
        print(json.dumps({"entries": entries, "count": len(entries)}, indent=2))
    else:
        for e in entries:
            ts = e.get("timestamp", "?")
            outcome = e.get("outcome", "?")
            tid = e.get("trace_id", "?")[:12]
            tenant = e.get("tenant_label", "")
            print(f"{ts}  {outcome:14}  {tid}  {tenant}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
