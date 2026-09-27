#!/usr/bin/env python3
"""One-command demo — governance in action in under 2 minutes.

Runs 3 turns:
  1. Establish: "Alice is a software engineer at Acme Corp." -> PASS
  2. Verify: "Does Alice work at Acme?" -> PASS
  3. Contradict: "Does Alice work at Google?" -> FORCE_REVISE or SOFT_CORRECT (depends on LLM)

Requires: ANTHROPIC_API_KEY or OPENAI_API_KEY set; spacy installed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys


async def _run_demo() -> int:
    parser = argparse.ArgumentParser(
        description="aurora-lens demo: governance in action in under 2 minutes.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    anth_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not anth_key and not openai_key:
        print(
            "Error: API key required. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.",
            file=sys.stderr,
        )
        return 1

    if anth_key and not anth_key.startswith("${"):
        try:
            from aurora_lens.adapters.claude import ClaudeAdapter
            adapter = ClaudeAdapter(api_key=anth_key, model=args.model or "claude-sonnet-4-5-20250929")
        except ImportError:
            print("Error: anthropic package required. pip install aurora-lens[claude]", file=sys.stderr)
            return 1
    elif openai_key and not openai_key.startswith("${"):
        try:
            from aurora_lens.adapters.openai import OpenAIUpstreamAdapter
            adapter = OpenAIUpstreamAdapter(
                api_key=openai_key,
                model=args.model,
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        except ImportError:
            print("Error: httpx required (included in base).", file=sys.stderr)
            return 1
    else:
        print("Error: API key required. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.", file=sys.stderr)
        return 1

    extraction_backend = None

    try:
        from aurora_lens.config import LensConfig
        from aurora_lens.lens import Lens
        from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        from aurora_lens.interpret.spacy_backend import SpacyBackend
        extraction_backend = SpacyBackend(model="en_core_web_sm")
    except ImportError:
        print("Error: spacy required for extraction. pip install aurora-lens[spacy]", file=sys.stderr)
        return 1

    bridge = CanonicalScannerGateBridge(mode="public")
    config = LensConfig(
        adapter=adapter,
        extraction_backend=extraction_backend,
        governance_bridge=bridge,
        audit_log_path=None,
    )
    lens = Lens(config=config, session_id="demo-")

    turns = [
        ("Alice is a software engineer at Acme Corp.", "Establish PEF"),
        ("Does Alice work at Acme?", "Verify against PEF"),
        ("Does Alice work at Google?", "Contradict PEF"),
    ]

    print("=" * 60)
    print("aurora-lens Demo")
    print("=" * 60)
    print()

    for i, (user_msg, label) in enumerate(turns, 1):
        print(f"--- Turn {i}: {label} ---")
        print(f"User: {user_msg}")
        result = await lens.process(user_msg)
        print(f"Governance: {result.action.name}")
        content = result.response[:200] + ("..." if len(result.response) > 200 else "")
        print(f"Assistant: {content}")
        if result.flags:
            print(f"Flags: {[f.flag_type.name for f in result.flags]}")
        print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("Turn 1 (establish): PEF stores Alice at Acme")
    print("Turn 2 (verify):    fact verified against PEF")
    print("Turn 3 (contradict): if LLM says Alice at Google, governance intervenes")
    print()
    print("Governance in action.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for aurora-lens console script.

    Usage: aurora-lens demo [--model ...]
           aurora-lens [--model ...]

    Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "demo":
        argv = argv[1:]
    old_argv = sys.argv
    sys.argv = ["aurora-lens"] + argv
    try:
        return asyncio.run(_run_demo())
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    sys.exit(main())
