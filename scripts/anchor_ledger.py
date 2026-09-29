#!/usr/bin/env python3
"""Phase D5: Anchor the audit ledger chain tip to an external record.

Reads the last entry from the audit file, extracts its cid, and writes a signed
anchor record. Use via cron or systemd timer to prove "on date X at time T,
chain tip was CID Y" using an external timestamp.

Usage:
  python scripts/anchor_ledger.py --path audit.jsonl [--out anchor.jsonl] [--key $KEY]
  python scripts/anchor_ledger.py --path audit.jsonl --s3 s3://bucket/anchors/
  python -m scripts.anchor_ledger --path /var/log/aurora/audit.jsonl --out /backup/anchors/

Sinks: file (--out), stdout (-), S3 (--s3, requires pip install aurora-lens[s3]).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse


def _compute_hmac(payload: str, key: bytes) -> str:
    import hashlib
    import hmac
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _upload_s3(s3_uri: str, body: bytes, content_type: str = "application/json") -> None:
    """Upload body to S3. Key: prefix/YYYY-MM-DD/THH-MM-SS-anchor.json."""
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError(
            "S3 upload requires boto3. Install with: pip install aurora-lens[s3]"
        ) from e
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {s3_uri}. Use s3://bucket/prefix/")
    bucket = parsed.netloc
    prefix = (parsed.path or "").strip("/")
    now = datetime.datetime.now(datetime.timezone.utc)
    key = f"{prefix}/{now:%Y-%m-%d}/T{now:%H-%M-%S}-anchor.json" if prefix else f"{now:%Y-%m-%d}/T{now:%H-%M-%S}-anchor.json"
    client = boto3.client("s3")
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anchor audit ledger chain tip for external timestamp proof.",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to audit JSONL file",
    )
    parser.add_argument(
        "--out",
        default="-",
        help="Output: file path to append anchor record, or '-' for stdout (default)",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Signing key (hex or raw). Default: AURORA_LENS_AUDIT_SIGNING_KEY env",
    )
    parser.add_argument(
        "--s3",
        default=None,
        metavar="s3://bucket/prefix/",
        help="S3 sink: upload anchor to bucket. Requires pip install aurora-lens[s3]",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: audit file not found: {path}", file=sys.stderr)
        return 1

    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
    except OSError as e:
        print(f"Error: cannot read audit file: {e}", file=sys.stderr)
        return 1

    if not lines:
        print("Error: audit file is empty", file=sys.stderr)
        return 1

    try:
        last = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in last line: {e}", file=sys.stderr)
        return 1

    cid = last.get("cid")
    if not cid:
        print("Error: last entry has no cid (schema_version 2 chain required)", file=sys.stderr)
        return 1

    anchor_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    anchor = {
        "anchor_cid": cid,
        "anchor_time": anchor_time,
        "ledger_path": str(path.resolve()),
        "entry_count": len(lines),
    }

    key_buf = args.key or os.environ.get("AURORA_LENS_AUDIT_SIGNING_KEY", "").strip()
    if key_buf:
        key = key_buf.encode("utf-8") if isinstance(key_buf, str) else key_buf
        payload = json.dumps(anchor, separators=(",", ":"), sort_keys=True)
        anchor["anchor_hmac"] = _compute_hmac(payload, key)

    line = json.dumps(anchor, separators=(",", ":"), sort_keys=True) + "\n"
    line_bytes = line.encode("utf-8")

    if args.out == "-":
        sys.stdout.write(line)
        sys.stdout.flush()
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            print(f"Error: cannot write anchor: {e}", file=sys.stderr)
            return 1

    if args.s3:
        try:
            _upload_s3(args.s3, line_bytes)
        except (ValueError, RuntimeError) as e:
            print(f"Error: S3 upload failed: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
