from __future__ import annotations

import argparse
from pathlib import Path
import sys

import uvicorn

from aurora_lens.proxy.config import ProxyConfig, ListenConfig
from aurora_lens.proxy.app import create_app
from aurora_lens.proxy.logging import setup_logging


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aurora-lens-proxy",
        description="Aurora-lens governed LLM proxy server",
    )
    parser.add_argument(
        "--config", "-c",
        default="aurora-lens.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Listen host (overrides config)",
    )
    parser.add_argument(
        "--port", "-p",
        default=None,
        type=int,
        help="Listen port (overrides config)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Upstream provider: mock | local | openai | anthropic (overrides config)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Upstream model name (overrides config)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import os
    ns = _parse_args(sys.argv[1:] if argv is None else argv)

    # CLI model/provider args set env vars so from_env() and apply_env_overrides() pick them up.
    if ns.provider is not None:
        os.environ["AURORA_LENS_UPSTREAM_PROVIDER"] = ns.provider
    if ns.model is not None:
        os.environ["AURORA_LENS_UPSTREAM_MODEL"] = ns.model

    cfg_path = Path(ns.config)

    try:
        config = ProxyConfig.from_yaml(cfg_path)
    except FileNotFoundError:
        # No config file — try building entirely from AURORA_LENS_* env vars.
        try:
            config = ProxyConfig.from_env()
            print("No config file found — running from environment variables.", file=sys.stderr)
        except ValueError as e:
            print(f"Error: config file not found: {cfg_path}", file=sys.stderr)
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # CLI overrides (host/port) — override only listen, keep upstream/governance intact.
    if ns.host is not None or ns.port is not None:
        new_listen = ListenConfig(
            host=ns.host if ns.host is not None else config.listen.host,
            port=ns.port if ns.port is not None else config.listen.port,
        )
        config = ProxyConfig(
            upstream=config.upstream,
            listen=new_listen,
            governance=config.governance,
        )

    try:
        config.validate()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    setup_logging()

    _startup_parts = [
        f"Upstream provider: {config.upstream.provider}",
        f"model: {config.upstream.model}",
    ]
    if config.upstream.provider == "mock":
        _startup_parts.append("base_url: (none — mock upstream)")
    elif config.upstream.base_url:
        _startup_parts.append(f"base_url: {config.upstream.base_url}")
    _startup_parts.append(f"governance: policy={config.governance.default_policy} mode={config.governance.mode}")
    _startup_parts.append(f"listen: {config.listen.host}:{config.listen.port}")
    _startup_parts.append(f"endpoint: {config.upstream.base_url or '(provider default)'}")
    _startup_parts.append(f"health_url: http://{config.listen.host}:{config.listen.port}/health")
    print(" | ".join(_startup_parts), file=sys.stderr)

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.listen.host,
        port=config.listen.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
