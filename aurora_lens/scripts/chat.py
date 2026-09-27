#!/usr/bin/env python3
"""Interactive chat with governance visibility.

Type input, model replies, then inspect what actually happened:
  /state   — PEF ground-truth state
  /details — Full governance decision (flags, rationale, original response if blocked)
  /last    — Re-show last turn's details
  quit     — Exit

Usage:
  python -m aurora_lens.scripts.chat [--model gpt-4o-mini]
  aurora-lens chat [--model gpt-4o-mini]

Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys


def _build_adapter(model: str):
    anth_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not anth_key and not openai_key:
        return None, "Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
    if anth_key and not anth_key.startswith("${"):
        try:
            from aurora_lens.adapters.claude import ClaudeAdapter
            # Use Claude model when OpenAI model name passed (e.g. default gpt-4o-mini)
            claude_model = model if model and model.startswith("claude-") else "claude-sonnet-4-5-20250929"
            return ClaudeAdapter(api_key=anth_key, model=claude_model), None
        except ImportError:
            return None, "pip install aurora-lens[claude]"
    if openai_key and not openai_key.startswith("${"):
        try:
            from aurora_lens.adapters.openai import OpenAIUpstreamAdapter
            return OpenAIUpstreamAdapter(
                api_key=openai_key,
                model=model,
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            ), None
        except ImportError:
            return None, "httpx required (included in base)"
    return None, "Set OPENAI_API_KEY or ANTHROPIC_API_KEY."


def _print_details(result, label: str = "Last turn"):
    """Print full governance decision for a LensResult."""
    print(f"\n--- {label} ---")
    print(f"Governance: {result.action.name}")
    print(f"Turn: {result.turn}")
    if result.flags:
        print("Flags:")
        for f in result.flags:
            print(f"  {f.flag_type.name}: {f.claim or f.evidence}")
    if result.decision:
        if result.decision.rationale:
            print(f"Rationale: {result.decision.rationale}")
        if result.decision.original_response:
            orig = result.decision.original_response
            preview = orig[:200] + ("..." if len(orig) > 200 else "")
            print(f"Original (blocked): {preview}")
    print("---\n")


async def main(model: str | None = None) -> int:
    if model is None:
        parser = argparse.ArgumentParser(description="Interactive chat with governance visibility.")
        parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
        args = parser.parse_args()
        model = args.model

    adapter, err = _build_adapter(model)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

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
        print("Error: pip install aurora-lens[spacy]", file=sys.stderr)
        return 1

    config = LensConfig(
        adapter=adapter,
        extraction_backend=extraction_backend,
        governance_bridge=CanonicalScannerGateBridge(mode="public"),
    )
    lens = Lens(config=config)
    last_result = None

    print("aurora-lens chat — type, model replies, then inspect what happened")
    print("Commands: /state | /details | /last | quit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input == "/state":
            print(f"\n{lens.pef.to_context_summary()}")
            continue
        if user_input == "/details" and last_result:
            _print_details(last_result, "Details")
            continue
        if user_input == "/last" and last_result:
            _print_details(last_result, "Last turn")
            continue
        if user_input == "/details" or user_input == "/last":
            print("No turn yet. Send a message first.")
            continue

        result = await lens.process(user_input)
        last_result = result

        print(f"\nAssistant: {result.response}")
        print(f"[Governance: {result.action.name}]")
        if result.flags:
            print(f"[Flags: {', '.join(f.flag_type.name for f in result.flags)}]")
        print("\n  /details — full decision, rationale, original if blocked")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
