"""
Rebuild C:/Git/aurora-lens-trial from C:/Git/aurora-lens.

The destination should contain only what an evaluator needs to install, run, or
understand the public trial: the shippable package (e.g. aurora_lens, governor),
evaluator-facing config, and allowlisted paths under docs/, eval/, scripts/,
and tests/ (see module constants). Everything else under those trees is omitted.

After copying, curated subtrees are **pruned** so stray files left from older
mirrors or manual drops are removed (especially under docs/). All ``*.jsonl``
audit logs plus ``flags_output.json`` / ``user_flags_output.json`` are stripped
from the destination so evaluators do not inherit a broken hash chain (integrity
failed on forensics).

Excludes deployment, internal notes, dev history, examples, root-only deploy
artifacts (via a strict root file/dir allowlist), RAG harness eval assets, and
most of the pytest tree.

Run from the aurora-lens root:  python scripts/rebuild_trial.py
"""
import shutil
import sys
from pathlib import Path

SRC = Path("C:/Git/aurora-lens")
DST = Path("C:/Git/aurora-lens-trial")

# Top-level names/dirs to exclude entirely (any path component)
EXCLUDE_NAMES = {
    # Railway / deployment (root copies use _TRIAL_ROOT_FILES_ALLOW; do not list
    # Dockerfile/docker-compose here or eval/docker-compose.yml is dropped)
    "railway.toml",
    # Personal / internal notes
    "Margaret only.txt", "INTERNAL ACQUISITION MEMO.txt",
    "TODAYS_WORK_SUMMARY.md", "GOVERNOR_RESTORE_PLAN.md", "GOVERNOR_SYNC.md",
    "NON_V1_EXTENSION_NOTES.md", "release_checklist.md",
    # Build artifacts
    "__pycache__", ".pytest_cache", "aurora_lens.egg-info",
    "build", "dist", "archive",
    # Runtime output
    "flags_output.json", "user_flags_output.json",
    # Old eval zip
    "aurora-lens-eval.zip",
    # Internal configs (never ship local/ledger/grok)
    "aurora-lens-grok.yaml", "demo_governance.yaml",
    "aurora-lens.local.yaml", "aurora-lens-ledger.yaml",
    # Windows artefact
    "nul",
    # Root scratch / local harness (not part of public trial)
    "rag_smoke_test.py",
    "rag_test_doc.txt",
    "rag_manual_smoke_test.py",
    # Claude settings
    ".claude",
    # Cursor IDE rules (trial bundle is not for IDE automation)
    ".cursor",
    # Benchmarks (internal perf testing)
    "benchmarks",
    # Secrets
    ".env",
    # rebuild script itself (no need to ship)
    "rebuild_trial.py",
}

# Relative paths (from SRC root, forward-slash) to exclude
EXCLUDE_RELATIVE = {
    "deploy",                                       # railway deploy config
    "k8s",                                          # kubernetes manifests
    "private",                                      # private internal folder
    "docs/outreach-tier2.md",                       # internal outreach
    "docs/plans",                                   # internal planning docs
    "docs/issues",                                  # internal issue tracker
    "docs/proof-session-2026-03-10.md",             # internal proof session
    "docs/STRUCTURAL_GOVERNOR_ROADMAP.md",          # internal roadmap
    "eval/.env",                                    # eval secrets file
    "eval/wheels",                                  # stale wheel
    "governor/IMPLEMENTATION_REPORT.md",            # internal status report
    "test_finance_extraction.py",                   # root-level scratchpad
    "aurora-lens.yaml",                             # production config (milamba.com CORS, ${PORT})
    ".github",                                      # CI config, not needed in trial
    "scripts/rebuild_trial.py",                     # this script
    "scripts/adversarial_stress.py",                # internal stress harness
    "scripts/anchor_ledger.py",                     # internal utility
    "scripts/hostile_smoke_numeric.py",             # internal
    "scripts/live_test_numeric_layers.py",          # internal live test
    "scripts/proof_session.py",                     # internal
    "scripts/run_benchmark.py",                     # internal
    "scripts/smoke_proof_bundle.py",                # internal
    # Tree: not in trial bundle (root files use _TRIAL_ROOT_FILES_ALLOW)
    "examples",                                     # integration samples; not required for trial
}

EXCLUDE_SUFFIXES = {".pyc", ".log"}
EXCLUDE_LOCAL_YAML_PATTERNS = {
    ".local.yaml", ".local.claude.yaml", ".local.openai.yaml",
}

# Repo root: only these top-level files and dirs are mirrored (everything else
# at depth 1 under SRC is omitted). Keep in sync with trial handoff docs.
# Not mirrored (not listed here): e.g. EVALUATOR_CONTRACT.md, CHANGELOG.md.
_TRIAL_ROOT_FILES_ALLOW = frozenset(
    {
        ".env.example",
        "aurora-lens.yaml.example",
        "entrypoint.sh",
        "INSTALL.txt",
        "LICENSE",
        "pyproject.toml",
        "README.md",
        "README_TRIAL_FIRST.md",
        "requirements.txt",
        "run_aurora_lens.py",
        "START_TRIAL_MAC.sh",
        "START_TRIAL_WINDOWS.bat",
        "CITATIONS.md",
    }
)
_TRIAL_ROOT_DIRS_ALLOW = frozenset(
    {
        "aurora_lens",
        "governor",
        "docs",
        "eval",
        "scripts",
        "tests",
    }
)

# Curated subsets: everything else under these trees is omitted from the trial mirror.
_TRIAL_TESTS_ALLOW = frozenset(
    {"tests", "tests/conftest.py", "tests/test_healthcare_trial.py"}
)
# Evaluator-facing docs (forensics UI, API, capabilities, policy, integration, healthcare trial).
_TRIAL_DOCS_ALLOW = frozenset(
    {
        "docs",
        "docs/api.md",
        "docs/CAPABILITIES.md",
        "docs/forensics-dashboard.md",
        "docs/healthcare-trial-demo.md",
        "docs/integration.md",
        "docs/policy.md",
        "docs/system-capabilities.md",
    }
)
_TRIAL_EVAL_ALLOW = frozenset(
    {
        "eval",
        "eval/README.md",
        "eval/TESTING_GUIDE.md",
        "eval/test_finance.py",
        "eval/.env.example",
        "eval/setup.sh",
        "eval/setup.bat",
        "eval/setup_offline.bat",
        "eval/start_proxy.sh",
        "eval/start_proxy.bat",
        "eval/prepare_offline.bat",
        "eval/chat.py",
        "eval/docker-compose.yml",
        "eval/Dockerfile",
    }
)
_TRIAL_SCRIPTS_ALLOW = frozenset(
    {
        "scripts",
        "scripts/demo_healthcare.py",
        "scripts/demo.ps1",
    }
)


def _relative_paths_under(prefix: str, trial_allow: frozenset[str]) -> frozenset[str]:
    """Paths relative to `prefix/` for entries like ``prefix/foo`` in trial_allow."""
    pl = prefix + "/"
    return frozenset(s[len(pl) :] for s in trial_allow if s.startswith(pl))


_TRIAL_DOCS_REL = _relative_paths_under("docs", _TRIAL_DOCS_ALLOW)
_TRIAL_EVAL_REL = _relative_paths_under("eval", _TRIAL_EVAL_ALLOW)
_TRIAL_SCRIPTS_REL = _relative_paths_under("scripts", _TRIAL_SCRIPTS_ALLOW)
_TRIAL_TESTS_REL = _relative_paths_under("tests", _TRIAL_TESTS_ALLOW)


def prune_audit_and_runtime_artifacts(dst: Path) -> int:
    """Remove audit logs and flag dumps so the trial tree starts with no chain (integrity ok on first run)."""
    if not dst.is_dir():
        return 0
    removed = 0
    for path in list(dst.rglob("*")):
        if ".git" in path.parts:
            continue
        if not path.is_file():
            continue
        name = path.name
        if name.endswith(".jsonl"):
            path.unlink()
            removed += 1
            continue
        if name in ("flags_output.json", "user_flags_output.json"):
            path.unlink()
            removed += 1
    return removed


def prune_trial_subtree(dst: Path, top: str, allowed_relpaths: frozenset[str]) -> int:
    """Delete any file under dst/top not in allowed_relpaths; remove empty dirs. Returns removed count."""
    base = dst / top
    if not base.is_dir():
        return 0
    removed = 0
    for path in list(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base).as_posix()
        if rel not in allowed_relpaths:
            path.unlink()
            removed += 1
    changed = True
    while changed:
        changed = False
        for path in sorted(base.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
                changed = True
    return removed


def _trial_root_allowlist_exclude(rel: Path) -> bool:
    """Omit any single-segment path under SRC root that is not in the trial allowlists."""
    parts = rel.parts
    if len(parts) != 1:
        return False
    name = parts[0]
    if name in _TRIAL_ROOT_FILES_ALLOW or name in _TRIAL_ROOT_DIRS_ALLOW:
        return False
    return True


def _trial_bundle_exclude(rel: Path) -> bool:
    """Exclude paths outside evaluator-focused allowlists for tests/docs/eval/scripts."""
    rel_str = str(rel).replace("\\", "/")
    if rel_str.startswith("tests/") or rel_str == "tests":
        return rel_str not in _TRIAL_TESTS_ALLOW
    if rel_str.startswith("docs/") or rel_str == "docs":
        return rel_str not in _TRIAL_DOCS_ALLOW
    if rel_str.startswith("eval/") or rel_str == "eval":
        return rel_str not in _TRIAL_EVAL_ALLOW
    if rel_str.startswith("scripts/") or rel_str == "scripts":
        return rel_str not in _TRIAL_SCRIPTS_ALLOW
    return False


def should_exclude(rel: Path) -> bool:
    parts = rel.parts
    # Check each path component
    for part in parts:
        if part in EXCLUDE_NAMES:
            return True
        for suf in EXCLUDE_SUFFIXES:
            if part.endswith(suf):
                return True
        for pat in EXCLUDE_LOCAL_YAML_PATTERNS:
            if part.endswith(pat):
                return True
    rel_str = str(rel).replace("\\", "/")
    for ex in EXCLUDE_RELATIVE:
        if rel_str == ex or rel_str.startswith(ex + "/"):
            return True
    # Audit / runtime logs: never mirror JSONL (local test chains confuse integrity / forensics)
    if rel_str.endswith(".jsonl"):
        return True
    if _trial_root_allowlist_exclude(rel):
        return True
    if _trial_bundle_exclude(rel):
        return True
    return False


# Gitignored in aurora-lens but mirrored to the trial tree when present on disk.
_TRIAL_HANDOFF_ROOT_FILES = (
    "README_TRIAL_FIRST.md",
    "START_TRIAL_WINDOWS.bat",
    "START_TRIAL_MAC.sh",
)


def ensure_handoff_root_files(src: Path, dst: Path) -> None:
    """Copy trial handoff files explicitly (listed in .gitignore; rglob still sees them, this makes intent obvious)."""
    for name in _TRIAL_HANDOFF_ROOT_FILES:
        s = src / name
        if s.is_file():
            shutil.copy2(s, dst / name)
    if not (src / "README_TRIAL_FIRST.md").is_file():
        print(
            "WARNING: README_TRIAL_FIRST.md not found under source root; "
            "trial bundle will lack the quick-start page.",
            file=sys.stderr,
        )


def clear_dst_except_git(dst: Path):
    """Remove all contents of dst except the .git directory."""
    for item in list(dst.iterdir()):
        if item.name == ".git":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def main():
    if not DST.exists():
        DST.mkdir(parents=True)
    print(f"Clearing {DST} (preserving .git)...")
    clear_dst_except_git(DST)

    copied = []
    for item in SRC.rglob("*"):
        rel = item.relative_to(SRC)
        if ".git" in rel.parts:
            continue
        if should_exclude(rel):
            continue
        dst_path = DST / rel
        if item.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst_path)
            copied.append(str(rel))

    print(f"Copied {len(copied)} files to {DST}")

    ensure_handoff_root_files(SRC, DST)

    n_docs = prune_trial_subtree(DST, "docs", _TRIAL_DOCS_REL)
    n_eval = prune_trial_subtree(DST, "eval", _TRIAL_EVAL_REL)
    n_scripts = prune_trial_subtree(DST, "scripts", _TRIAL_SCRIPTS_REL)
    n_tests = prune_trial_subtree(DST, "tests", _TRIAL_TESTS_REL)
    pruned = n_docs + n_eval + n_scripts + n_tests
    if pruned:
        print(
            f"Pruned {pruned} stray file(s) under curated trees "
            f"(docs={n_docs}, eval={n_eval}, scripts={n_scripts}, tests={n_tests})."
        )

    n_audit = prune_audit_and_runtime_artifacts(DST)
    print(
        f"Audit/runtime cleanup: removed {n_audit} file(s) "
        "(*.jsonl, flags_output.json, user_flags_output.json)."
    )

    print("\nTop-level contents of aurora-lens-trial/:")
    for p in sorted(DST.iterdir()):
        if p.name != ".git":
            print(f"  {p.name}{'/' if p.is_dir() else ''}")


if __name__ == "__main__":
    main()
