from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from release_guard import assert_clean_release_tree, inspect_distribution_archive


def _run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _run_capture(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _ensure_build_frontend() -> None:
    code, _ = _run_capture([sys.executable, "-c", "import build"], cwd=ROOT)
    if code == 0:
        return
    _run([sys.executable, "-m", "pip", "install", "build"], cwd=ROOT)


def main() -> int:
    root = ROOT
    dist = root / "dist"
    dist.mkdir(parents=True, exist_ok=True)

    assert_clean_release_tree(root)
    print("clean_tree_guard=PASS")
    _ensure_build_frontend()

    tripwire = root / "audit.jsonl"
    tripwire.write_text('{"event":"tripwire"}\n', encoding="utf-8")
    code, output = _run_capture([sys.executable, "-m", "build", "--wheel", "--outdir", "dist"], cwd=root)
    tripwire.unlink(missing_ok=True)
    if code == 0:
        raise SystemExit("pep517_guard_bypass_check=FAIL (build unexpectedly succeeded with tripwire)")
    if "Clean-audit packaging guard failed" not in output:
        raise SystemExit("pep517_guard_bypass_check=FAIL (tripwire failure did not come from guard)")
    print("pep517_guard_bypass_check=PASS")

    _run([sys.executable, "-m", "build", "--wheel", "--sdist", "--outdir", "dist"], cwd=root)

    artifacts = (
        sorted(dist.glob("aurora_lens-3.0.0-*.whl"))
        + sorted(dist.glob("aurora_lens-3.0.0*.tar.gz"))
        + sorted(dist.glob("aurora-lens-3.0.0*.tar.gz"))
    )
    if not artifacts:
        raise SystemExit("No artifacts produced in dist/.")
    for artifact in artifacts:
        issues = inspect_distribution_archive(artifact)
        status = "PASS" if not issues else "FAIL"
        print(f"artifact={artifact.name} guard={status}")
        for issue in issues:
            print(f" - {issue}")
        if issues:
            raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
