"""Phase D6: CLI to verify audit log integrity (HMAC + hash chain).

Usage:
  python -m aurora_lens.scripts.verify_audit --path audit.jsonl [--n 1000] [--key $KEY]
  python -m aurora_lens.scripts.verify_audit --path audit.jsonl --keys "$OLD_KEY,$NEW_KEY"

D4 multi-key: use --keys for logs spanning key rotation. Tries each key per entry.

Optional ``--pef-linkage``: verify consecutive rows' ``pef_turn_classification`` against the
previous row's ``pef_snapshot`` (see :func:`aurora_lens.govern.audit_io.verify_pef_linkage`).

Failure messages on stderr are plain-language (what broke, where to look, internal code for support).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from aurora_lens.govern.audit_failure_messages import (
    format_chain_of_custody_failure_stderr,
    format_verify_audit_stderr_chain,
    format_verify_audit_stderr_hmac,
    format_verify_audit_stderr_linkage,
    format_verify_audit_stderr_replay,
)
from aurora_lens.govern.audit_io import (
    verify_audit_entries,
    verify_chain,
    verify_chain_of_custody_rows,
    verify_forensic_state_hash,
    verify_pef_linkage,
)


def _parse_keys(key: str | None, keys: str | None, env_key: str, env_keys: str) -> list[bytes]:
    """Build list of signing keys from --key, --keys, and env vars."""
    result: list[bytes] = []
    single = (key or os.environ.get(env_key, "")).strip()
    if single:
        result.append(single.encode("utf-8"))
    multi = (keys or os.environ.get(env_keys, "")).strip()
    if multi:
        for k in multi.split(","):
            k = k.strip()
            if k and k.encode("utf-8") not in result:
                result.append(k.encode("utf-8"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify audit log integrity (HMAC + hash chain).",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to audit JSONL file",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of last entries to verify (default: 1000)",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Signing key for HMAC. Default: AURORA_LENS_AUDIT_SIGNING_KEY env",
    )
    parser.add_argument(
        "--keys",
        default=None,
        help="Comma-separated keys for logs spanning rotation (D4). AURORA_LENS_AUDIT_SIGNING_KEYS env",
    )
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="Skip replay verification (state_hash from pef_snapshot)",
    )
    parser.add_argument(
        "--pef-linkage",
        action="store_true",
        help="Verify pef_turn_classification vs previous row pef_snapshot (optional)",
    )
    parser.add_argument(
        "--chain-of-custody",
        action="store_true",
        help="Verify chain_of_custody blocks (fingerprint replay + evidence status semantics)",
    )
    parser.add_argument(
        "--require-complete-evidence",
        action="store_true",
        help="With --chain-of-custody: fail if evidence_audit_status is degraded",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(
            f"Error: audit file not found: {path}. "
            "Check the path and that the process that should have written the log has access.",
            file=sys.stderr,
        )
        return 1

    signing_keys = _parse_keys(
        args.key, args.keys,
        "AURORA_LENS_AUDIT_SIGNING_KEY", "AURORA_LENS_AUDIT_SIGNING_KEYS",
    )
    if not signing_keys:
        print(
            "Error: no signing key provided. Pass --key or --keys, or set "
            "AURORA_LENS_AUDIT_SIGNING_KEY (and optional AURORA_LENS_AUDIT_SIGNING_KEYS for rotation). "
            "The verifier needs the same key material as the writer to check HMACs.",
            file=sys.stderr,
        )
        return 1

    hmac_ok, hmac_entries, first_failed = verify_audit_entries(
        path, args.n, signing_keys=signing_keys
    )
    chain_ok, chain_entries, first_break_idx, reason = verify_chain(
        path, args.n, signing_keys=signing_keys
    )
    replay_ok = True
    replay_checked = 0
    replay_fail_idx = None
    replay_reason = None
    if not args.no_replay:
        replay_ok, replay_checked, replay_fail_idx, replay_reason = verify_forensic_state_hash(
            path, args.n
        )

    linkage_ok = True
    linkage_checked = 0
    linkage_fail_idx = None
    linkage_reason = None
    if args.pef_linkage:
        linkage_ok, linkage_checked, linkage_fail_idx, linkage_reason = verify_pef_linkage(
            path, args.n
        )

    coc_ok = True
    coc_checked = 0
    coc_fail_idx = None
    coc_reason = None
    if args.chain_of_custody:
        coc_ok, coc_checked, coc_fail_idx, coc_reason = verify_chain_of_custody_rows(
            path,
            args.n,
            require_complete_evidence=args.require_complete_evidence,
        )

    if hmac_ok and chain_ok and replay_ok and linkage_ok and coc_ok:
        parts = [f"{chain_entries} entries verified (HMAC + chain)"]
        if replay_checked > 0:
            parts.append(f"{replay_checked} forensic state_hash replayed")
        if args.pef_linkage and linkage_checked > 0:
            parts.append(f"{linkage_checked} PEF linkage pairs checked")
        if args.chain_of_custody and coc_checked > 0:
            parts.append(f"{coc_checked} chain-of-custody rows checked")
        print(f"OK: {', '.join(parts)}")
        return 0

    if not hmac_ok:
        print(format_verify_audit_stderr_hmac(path, args.n, first_failed), file=sys.stderr)
        return 1
    if not chain_ok:
        print(format_verify_audit_stderr_chain(path, args.n, first_break_idx, reason), file=sys.stderr)
        return 1
    if not replay_ok:
        print(
            format_verify_audit_stderr_replay(path, args.n, replay_fail_idx, replay_reason),
            file=sys.stderr,
        )
        return 1
    if not linkage_ok:
        print(
            format_verify_audit_stderr_linkage(path, args.n, linkage_fail_idx, linkage_reason),
            file=sys.stderr,
        )
        return 1
    if not coc_ok:
        print(
            format_chain_of_custody_failure_stderr(path, args.n, coc_fail_idx, coc_reason),
            file=sys.stderr,
        )
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
