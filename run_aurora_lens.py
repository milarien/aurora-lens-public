#!/usr/bin/env python3
"""Start the Aurora-Lens OpenAI-compatible proxy.

Minimal zero-config start — just set your API key and run:

  export ANTHROPIC_API_KEY=sk-ant-...
  python run_aurora_lens.py

  export OPENAI_API_KEY=sk-...
  python run_aurora_lens.py

Provider and model are auto-detected from whichever API key env var is present.
Override with CLI args or env vars:

  python run_aurora_lens.py --port 9000 --provider anthropic --model claude-sonnet-4-6

  AURORA_LENS_UPSTREAM_MODEL=gpt-4o python run_aurora_lens.py

Config file (optional):
  Copy aurora-lens.yaml.example → aurora-lens.yaml and customise.
  The launcher checks for a YAML in this order:
    1. AURORA_LENS_CONFIG env var
    2. aurora-lens.local.yaml
    3. aurora-lens.local.openai.yaml  /  aurora-lens.local.claude.yaml
    4. aurora-lens.yaml
    5. aurora-lens.yaml.example
    6. examples/aurora-lens.blank.yaml
  If none is found, runs entirely from environment variables.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

_CONFIG_CANDIDATES = [
    ROOT / "aurora-lens.local.yaml",
    ROOT / "aurora-lens.local.openai.yaml",
    ROOT / "aurora-lens.local.claude.yaml",
    ROOT / "aurora-lens.yaml",
    ROOT / "aurora-lens.yaml.example",
    ROOT / "examples" / "aurora-lens.blank.yaml",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_aurora_lens",
        description="Start the Aurora-Lens governed LLM proxy",
        add_help=True,
    )
    parser.add_argument("--port", "-p", type=int, default=None,
                        help="Listen port (default: 8081)")
    parser.add_argument("--provider", default=None,
                        help="Upstream provider: openai | anthropic")
    parser.add_argument("--model", default=None,
                        help="Upstream model name")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to YAML config file (optional)")
    return parser.parse_args()


def _pick_config(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            print(f"Error: config file not found: {p}", file=sys.stderr)
            sys.exit(1)
        return p.resolve()
    env_path = os.environ.get("AURORA_LENS_CONFIG", "").strip()
    if env_path:
        p = Path(env_path)
        if not p.is_file():
            print(f"Error: AURORA_LENS_CONFIG file not found: {p}", file=sys.stderr)
            sys.exit(1)
        return p.resolve()
    for p in _CONFIG_CANDIDATES:
        if p.is_file():
            return p.resolve()
    return None  # no YAML — proxy will run from env vars


def main() -> None:
    ns = _parse_args()
    cfg = _pick_config(ns.config)

    os.chdir(ROOT)

    argv = [sys.executable, "-m", "aurora_lens.proxy"]

    if cfg:
        print(f"Aurora-Lens  config={cfg}", file=sys.stderr)
        argv += ["--config", str(cfg)]
    else:
        print("Aurora-Lens  no config file found — starting from environment variables",
              file=sys.stderr)
        # Pass a non-existent path so __main__ falls through to from_env()
        argv += ["--config", str(ROOT / "aurora-lens.yaml")]

    if ns.port is not None:
        argv += ["--port", str(ns.port)]
    if ns.provider is not None:
        argv += ["--provider", ns.provider]
    if ns.model is not None:
        argv += ["--model", ns.model]

    try:
        completed = subprocess.run(argv)
        raise SystemExit(completed.returncode)
    except KeyboardInterrupt:
        print("\nAurora-Lens stopped.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
