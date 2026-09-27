"""Single entry point for aurora-lens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aurora_lens.launcher.controller import main as launcher_main
from aurora_lens.launcher.paths import ensure_runtime_dirs, resolve_config_path, resolve_runtime_paths
from aurora_lens.launcher.support import export_support_bundle

def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="aurora-lens",
        description="Governance substrate for LLM hallucination prevention.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    start_parser = subparsers.add_parser("start", help="Start managed Aurora-Lens proxy")
    start_parser.add_argument("--config", "-c", default=None, help="Config file path")
    start_parser.set_defaults(func=_run_start)

    status_parser = subparsers.add_parser("status", help="Show managed proxy status")
    status_parser.add_argument("--config", "-c", default=None, help="Config file path")
    status_parser.set_defaults(func=_run_status)

    stop_parser = subparsers.add_parser("stop", help="Stop managed Aurora-Lens proxy")
    stop_parser.add_argument("--config", "-c", default=None, help="Config file path")
    stop_parser.set_defaults(func=_run_stop)

    init_config_parser = subparsers.add_parser("init-config", help="Write minimal example config")
    init_config_parser.add_argument("--config", "-c", default=None, help="Config file path")
    init_config_parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    init_config_parser.set_defaults(func=_run_init_config)

    support_parser = subparsers.add_parser("support-export", help="Export runtime diagnostics bundle")
    support_parser.add_argument("--config", "-c", default=None, help="Config file path")
    support_parser.set_defaults(func=_run_support_export)

    license_parser = subparsers.add_parser("license", help="Show license summary")
    license_parser.set_defaults(func=_run_license)

    # demo
    demo_parser = subparsers.add_parser("demo", help="Run governance demo")
    demo_parser.add_argument(
        "scenario",
        nargs="?",
        default=None,
        choices=["healthcare"],
        help="Demo scenario (omit for general demo requiring API key; 'healthcare' runs in-process, no API key needed but requires the [spacy] extra)",
    )
    demo_parser.add_argument("--model", default="gpt-4o-mini", help="Model name (general demo only)")
    demo_parser.set_defaults(func=_run_demo)

    # chat
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with governance visibility")
    chat_parser.add_argument("--model", default="gpt-4o-mini", help="Model name")
    chat_parser.set_defaults(func=_run_chat)

    # batch
    batch_parser = subparsers.add_parser(
        "batch",
        help="Run governance over a JSONL scenario file (requires API key)",
    )
    batch_parser.add_argument("--input", "-i", default="-",
                              help="Input JSONL path, or '-' for stdin")
    batch_parser.add_argument("--output", "-o", default="-",
                              help="Output JSONL path, or '-' for stdout")
    batch_parser.add_argument("--audit", "-a", default=None,
                              help="Forensic audit JSONL path")
    batch_parser.add_argument("--model", default="gpt-4o-mini",
                              help="LLM model name (default: gpt-4o-mini)")
    batch_parser.add_argument("--mode", default="enterprise",
                              choices=["enterprise", "public"],
                              help="Governance mode (default: enterprise)")
    batch_parser.add_argument("--fail-on-mismatch", action="store_true",
                              help="Exit 1 if any expected_action mismatch")
    batch_parser.set_defaults(func=_run_batch)

    # proxy
    proxy_parser = subparsers.add_parser("proxy", help="Run governed LLM proxy server")
    proxy_parser.add_argument("--config", "-c", default="aurora-lens.yaml", help="Config file path")
    proxy_parser.add_argument("--host", default=None, help="Listen host")
    proxy_parser.add_argument("--port", "-p", default=None, type=int, help="Listen port")
    proxy_parser.set_defaults(func=_run_proxy)

    serve_parser = subparsers.add_parser("serve", help="Run governed LLM proxy server")
    serve_parser.add_argument("--config", "-c", default="aurora-lens.yaml", help="Config file path")
    serve_parser.add_argument("--host", default=None, help="Listen host")
    serve_parser.add_argument("--port", "-p", default=None, type=int, help="Listen port")
    serve_parser.set_defaults(func=_run_proxy)

    _add_corpus_subcommands(subparsers)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


def _add_corpus_subcommands(subparsers: argparse._SubParsersAction) -> None:
    from aurora_lens.corpus.registry import default_corpus_root

    corpus = subparsers.add_parser("corpus", help="Document library (ingest, ask, review)")
    corpus_sub = corpus.add_subparsers(dest="corpus_command")

    ingest_p = corpus_sub.add_parser("ingest", help="Ingest PDF/markdown into registry")
    ingest_p.add_argument("--record-id", required=True)
    ingest_p.add_argument("--file", required=True, type=Path)
    ingest_p.add_argument("--title", default=None)
    ingest_p.add_argument("--corpus-root", type=Path, default=None)
    ingest_p.add_argument("--force", action="store_true")
    ingest_p.set_defaults(func=_corpus_ingest)

    list_p = corpus_sub.add_parser("list", help="List ingested records")
    list_p.add_argument("--corpus-root", type=Path, default=None)
    list_p.set_defaults(func=_corpus_list)

    ask_p = corpus_sub.add_parser("ask", help="Ask via governed proxy (Door 2)")
    ask_p.add_argument("--record-id", required=True)
    ask_p.add_argument("--question", required=True)
    ask_p.add_argument("--proxy", default="http://localhost:8081")
    ask_p.add_argument("--corpus-root", type=Path, default=None)
    ask_p.add_argument("--chunks", default=None)
    ask_p.add_argument("--max-chars", type=int, default=12000)
    ask_p.set_defaults(func=_corpus_ask)

    validate_p = corpus_sub.add_parser("validate", help="Run Q&A validation manifest")
    validate_p.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval/sample_corpus_qa.manifest.yaml"),
    )
    validate_p.add_argument("--proxy", default="http://localhost:8081")
    validate_p.add_argument("--corpus-root", type=Path, default=None)
    validate_p.add_argument("--write-evidence", type=Path, default=None)
    validate_p.add_argument("--write-json", type=Path, default=None)
    validate_p.set_defaults(func=_corpus_validate)

    review_p = corpus_sub.add_parser("review", help="Review record via direct upstream (Door 1)")
    review_p.add_argument("--record-id", required=True)
    review_p.add_argument("--corpus-root", type=Path, default=None)
    review_p.add_argument("--chunks", default=None)
    review_p.add_argument("--max-chars", type=int, default=None)
    review_p.set_defaults(func=_corpus_review)

    corpus.set_defaults(corpus_root_default=default_corpus_root())


def _run_start(args: argparse.Namespace) -> int:
    pass_argv = ["start"]
    if args.config is not None:
        pass_argv.extend(["--config", args.config])
    return launcher_main(pass_argv)


def _run_status(args: argparse.Namespace) -> int:
    pass_argv = ["status"]
    if args.config is not None:
        pass_argv.extend(["--config", args.config])
    return launcher_main(pass_argv)


def _run_stop(args: argparse.Namespace) -> int:
    pass_argv = ["stop"]
    if args.config is not None:
        pass_argv.extend(["--config", args.config])
    return launcher_main(pass_argv)


def _run_support_export(args: argparse.Namespace) -> int:
    paths = resolve_runtime_paths()
    ensure_runtime_dirs(paths)
    config_path = resolve_config_path(args.config, paths)
    result = export_support_bundle(paths=paths, config_path=config_path)
    print(f"Support bundle: {result['archive_path']}")
    return 0


def _run_init_config(args: argparse.Namespace) -> int:
    paths = resolve_runtime_paths()
    ensure_runtime_dirs(paths)
    config_path = resolve_config_path(args.config, paths)
    audit_path = (paths.logs_dir / "audit.jsonl").as_posix()
    if config_path.exists() and not args.force:
        print(f"Config already exists: {config_path}")
        return 0
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        (
            "upstream:\n"
            "  provider: mock\n"
            "  model: mock\n"
            "listen:\n"
            "  host: 127.0.0.1\n"
            "  port: 8081\n"
            "governance:\n"
            "  default_policy: strict\n"
            "  mode: public\n"
            f"  audit_log: {audit_path}\n"
            "extraction:\n"
            "  backend: spacy\n"
        ),
        encoding="utf-8",
    )
    print(f"Wrote example config: {config_path}")
    return 0


def _run_license(args: argparse.Namespace) -> int:  # noqa: ARG001
    print("Aurora-Lens is proprietary software. All rights reserved.")
    print("No licence is granted except under a separate written agreement.")
    print("See: LICENSE and LICENSING.md")
    return 0


def _corpus_ingest(args: argparse.Namespace) -> int:
    from aurora_lens.corpus.cli_support import cmd_ingest

    return cmd_ingest(
        record_id=args.record_id,
        file_path=args.file,
        corpus_root=args.corpus_root,
        title=args.title,
        force=args.force,
    )


def _corpus_list(args: argparse.Namespace) -> int:
    from aurora_lens.corpus.cli_support import cmd_list

    return cmd_list(corpus_root=args.corpus_root)


def _corpus_ask(args: argparse.Namespace) -> int:
    from aurora_lens.corpus.cli_support import cmd_ask

    return cmd_ask(
        record_id=args.record_id,
        question=args.question,
        corpus_root=args.corpus_root,
        proxy=args.proxy,
        chunks=args.chunks,
        max_chars=args.max_chars,
    )


def _corpus_validate(args: argparse.Namespace) -> int:
    from aurora_lens.corpus.cli_support import cmd_validate

    return cmd_validate(
        manifest_path=args.manifest,
        corpus_root=args.corpus_root,
        proxy=args.proxy,
        write_evidence=args.write_evidence,
        write_json=args.write_json,
    )


def _corpus_review(args: argparse.Namespace) -> int:
    from aurora_lens.corpus.cli_support import cmd_review

    return cmd_review(
        record_id=args.record_id,
        corpus_root=args.corpus_root,
        chunks=args.chunks,
        max_chars=args.max_chars,
    )


def _run_demo(args: argparse.Namespace) -> int:
    if args.scenario == "healthcare":
        from aurora_lens.scripts.demo_healthcare import main as healthcare_main
        return healthcare_main()
    from aurora_lens.scripts.demo import main as demo_main
    # Pass through: --model
    pass_argv = []
    if args.model:
        pass_argv.extend(["--model", args.model])
    return demo_main(pass_argv)


def _run_chat(args: argparse.Namespace) -> int:
    from aurora_lens.scripts.chat import main as chat_main
    import asyncio
    return asyncio.run(chat_main(model=args.model))


def _run_batch(args: argparse.Namespace) -> int:
    from aurora_lens.scripts.batch import main as batch_main
    pass_argv = []
    for flag, val in [
        ("--input", args.input),
        ("--output", args.output),
        ("--model", args.model),
        ("--mode", args.mode),
    ]:
        pass_argv.extend([flag, val])
    if args.audit:
        pass_argv.extend(["--audit", args.audit])
    if args.fail_on_mismatch:
        pass_argv.append("--fail-on-mismatch")
    return batch_main(pass_argv)


def _run_proxy(args: argparse.Namespace) -> int:
    from aurora_lens.proxy.__main__ import main as proxy_main
    pass_argv = ["--config", args.config]
    if args.host is not None:
        pass_argv.extend(["--host", args.host])
    if args.port is not None:
        pass_argv.extend(["--port", str(args.port)])
    proxy_main(pass_argv)
    return 0  # uvicorn blocks; only reached on shutdown


if __name__ == "__main__":
    sys.exit(main())
