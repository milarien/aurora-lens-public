#!/usr/bin/env python3
"""Build the release packages and exercise them outside the source checkout.

Copies this repository into a temporary tree that omits local runtime
artefacts, builds a wheel and an sdist there, then installs the wheel into a
fresh virtual environment. The demo and launcher run with ``PYTHONPATH`` unset
so they import the installed package.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

_HAZARD_FILES = (
    "actions.json",
    "decision_matrix.json",
    "framings.json",
    "hazard_classes.json",
    "ontology.manifest.json",
    "organisms.json",
    "processes.json",
    "substances.json",
)

_SKIP_DIRS = {
    ".git",
    ".venv",
    ".venv-test",
    ".pytest_cache",
    ".mypy_cache",
    "__pycache__",
    "dist",
    "build",
    "logs",
    "state",
    "support",
    "tmp",
    "runtime",
    "sessions",
}


def _ignore(directory: str, names: list[str]) -> set[str]:
    del directory
    skipped: set[str] = set()
    for name in names:
        if name in _SKIP_DIRS or name.endswith(".egg-info"):
            skipped.add(name)
            continue
        lower = name.lower()
        if lower.endswith(".jsonl") or lower.endswith(".pid") or lower in {"launcher.log", "proxy.log"}:
            skipped.add(name)
        if ".evidence" in lower:
            skipped.add(name)
    return skipped


def _copy_source(repo: Path, dest: Path) -> None:
    shutil.copytree(repo, dest, ignore=_ignore, dirs_exist_ok=False)


def _python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=True, text=True, capture_output=True)


def _hazard_names_in_archive(path: Path) -> set[str]:
    names: list[str] = []
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(path) as archive:
            names = archive.getnames()
    found: set[str] = set()
    for name in names:
        normalized = name.replace("\\", "/")
        if "/data/hazard/" not in normalized and not normalized.startswith("aurora_lens/data/hazard/"):
            continue
        found.add(Path(normalized).name)
    return found


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _child_env(local_app: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("AURORA_EDGE_TOKEN", None)
    env.pop("RAILWAY_ENVIRONMENT", None)
    env.pop("RAILWAY_ENVIRONMENT_NAME", None)
    env["LOCALAPPDATA"] = str(local_app)
    return env


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    work = Path(tempfile.mkdtemp(prefix="aurora-lens-smoke-"))
    print(f"smoke workspace: {work}")
    source = work / "source"
    dist = work / "dist"
    dist.mkdir()
    _copy_source(repo, source)

    build_py = (
        "from pathlib import Path\n"
        "from aurora_lens_build_backend import build_sdist, build_wheel\n"
        "out = Path('dist')\n"
        "out.mkdir(exist_ok=True)\n"
        "print(build_wheel(str(out)))\n"
        "print(build_sdist(str(out)))\n"
    )
    built = _run([sys.executable, "-c", build_py], cwd=source)
    print(built.stdout)
    artifacts = list(dist.iterdir()) if dist.exists() else []
    # The backend writes into source/dist because cwd is source.
    built_dir = source / "dist"
    wheels = list(built_dir.glob("*.whl"))
    sdists = list(built_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(f"expected one wheel and one sdist, found {wheels} {sdists} {artifacts}")
    for package in (wheels[0], sdists[0]):
        found = _hazard_names_in_archive(package)
        missing = [name for name in _HAZARD_FILES if name not in found]
        if missing:
            raise SystemExit(f"{package.name} is missing hazard files: {missing}")
        print(f"{package.name}: {len(found)} hazard files")

    venv = work / "venv"
    _run([sys.executable, "-m", "venv", str(venv)], cwd=work)
    py = _python(venv)
    _run([str(py), "-m", "pip", "install", "-U", "pip"], cwd=work)
    wheel_req = f"{wheels[0]}[proxy,spacy]"
    _run([str(py), "-m", "pip", "install", wheel_req], cwd=work)
    _run([str(py), "-m", "spacy", "download", "en_core_web_sm"], cwd=work)

    env = _child_env(work / "localapp")
    located = _run(
        [str(py), "-c", "import aurora_lens, pathlib; p = pathlib.Path(aurora_lens.__file__); print(p); assert 'site-packages' in p.as_posix()"],
        cwd=work,
        env=env,
    )
    print(located.stdout.strip())

    outside = work / "outside"
    (outside / "tools").mkdir(parents=True)
    shutil.copy(repo / "tools" / "run_demo.py", outside / "tools" / "run_demo.py")
    demo = _run([str(py), str(outside / "tools" / "run_demo.py")], cwd=outside, env=env)
    print(demo.stdout)
    if demo.stdout.count("[OK]") < 3 or "UNEXPECTED" in demo.stdout:
        raise SystemExit("demo did not report three OK outcomes from the installed package")

    launcher = venv / ("Scripts/aurora-lens.exe" if sys.platform == "win32" else "bin/aurora-lens")
    init = _run([str(launcher), "init-config"], cwd=outside, env=env)
    print(init.stdout)
    config = work / "localapp" / "Aurora-Lens" / "config" / "aurora-lens.yaml"
    if not config.is_file():
        raise SystemExit(f"init-config did not write {config}")
    port = _free_port()
    text = config.read_text(encoding="utf-8")
    config.write_text(text.replace("  port: 8081\n", f"  port: {port}\n"), encoding="utf-8")
    stopped = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
    try:
        started = _run([str(launcher), "start"], cwd=outside, env=env)
        print(started.stdout)
        if "RUNNING_HEALTHY" not in started.stdout:
            raise SystemExit("launcher start did not become healthy")
        chat = _run(
            [
                str(py),
                "-c",
                (
                    "import json, urllib.request; "
                    f"url='http://127.0.0.1:{port}/v1/chat/completions'; "
                    "body=json.dumps({'model':'mock','messages':[{'role':'user','content':'What is 2 + 2?'}]}).encode(); "
                    "req=urllib.request.Request(url, data=body, headers={'Content-Type':'application/json'}); "
                    "resp=urllib.request.urlopen(req, timeout=60); "
                    "print(resp.status); "
                    "assert resp.status==200"
                ),
            ],
            cwd=outside,
            env=env,
        )
        print(chat.stdout)
    finally:
        stopped = subprocess.run(
            [str(launcher), "stop"],
            cwd=outside,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        print(stopped.stdout)
        print(stopped.stderr)
    if stopped.returncode != 0 or "TERMINATION_FAILED" in stopped.stdout:
        raise SystemExit("launcher stop did not confirm termination")
    if "STOPPED" not in stopped.stdout and "ALREADY_STOPPED" not in stopped.stdout:
        raise SystemExit("launcher stop did not report a stopped proxy")
    print("release artifact smoke: OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or "", file=sys.stderr)
        print(exc.stderr or "", file=sys.stderr)
        raise
