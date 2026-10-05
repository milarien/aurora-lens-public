from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from enum import Enum
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from aurora_lens.build_info import load_build_info
from aurora_lens.launcher.lifecycle import (
    LifecycleResult,
    LifecycleResultCode,
    RuntimeStatusCode,
    RuntimeStatusResult,
)
from aurora_lens.launcher.paths import RuntimePaths, runtime_paths_from_setup_home


@dataclass(frozen=True)
class ProxyStartResult:
    ok: bool
    code: RuntimeStatusCode
    message: str
    proxy_url: str | None = None
    health_url: str | None = None
    log_path: str | None = None
    warning: str | None = None


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class _PidStatus(str, Enum):
    ALIVE = "alive"
    DEAD = "dead"
    UNKNOWN = "unknown"


def _classify_windows_tasklist_result(
    pid: int,
    *,
    returncode: int,
    stdout: str,
    stderr: str,
) -> _PidStatus:
    output = (stdout or "").strip()
    combined = f"{output}\n{stderr or ''}".lower()
    if "no tasks are running" in combined:
        return _PidStatus.DEAD
    if "access denied" in combined or "access is denied" in combined:
        return _PidStatus.UNKNOWN
    if returncode != 0 or not output:
        return _PidStatus.UNKNOWN
    for line in output.splitlines():
        columns = [column.strip().strip('"') for column in line.split(",")]
        if len(columns) > 1 and columns[1] == str(pid):
            return _PidStatus.ALIVE
    return _PidStatus.UNKNOWN


def _pid_status(pid: int) -> _PidStatus:
    if pid <= 0:
        return _PidStatus.DEAD
    if sys.platform == "win32":
        try:
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                check=False,
                capture_output=True,
                text=True,
            )
            return _classify_windows_tasklist_result(
                pid,
                returncode=completed.returncode,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
            )
        except Exception:
            return _PidStatus.UNKNOWN
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return _PidStatus.DEAD
    except PermissionError:
        return _PidStatus.UNKNOWN
    except OSError:
        return _PidStatus.UNKNOWN
    return _PidStatus.ALIVE


def _is_pid_alive(pid: int) -> bool:
    """Compatibility helper for identity checks that require confirmed liveness."""

    return _pid_status(pid) == _PidStatus.ALIVE


def _reap_exited_child(pid: int) -> None:
    """Collect an exited child so a zombie is not reported as still running.

    ``os.kill(pid, 0)`` succeeds for a zombie. That happens when stop runs in
    the same process that started the proxy (the parent has not waited). A
    process that is still alive is left alone: ``WNOHANG`` returns immediately.
    """
    if pid <= 0 or sys.platform == "win32":
        return
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return
    except OSError:
        return


def _read_pid_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_pid_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def _utc_stamp_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


class LauncherState:
    """Owns deterministic runtime files and proxy process lifecycle."""

    def __init__(self, paths: RuntimePaths | None = None, *, home: Path | None = None) -> None:
        if paths is None:
            if home is None:
                raise TypeError("LauncherState requires paths= or home=")
            paths = runtime_paths_from_setup_home(home)
        self.paths = paths
        self.proxy_pid_path = paths.proxy_pid_path
        self.log_path = paths.launcher_log_path
        self.proxy_log_path = paths.proxy_log_path
        self._release_info = load_build_info()
        self.controller_pid_path = paths.state_dir / "aurora-lens.controller.pid"

    def log(self, message: str) -> None:
        line = f"{_utc_now()} {message}\n"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def _quarantine_pid_file(
        self,
        path: Path,
        *,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> Path | None:
        if not path.exists():
            return None
        stamp = _utc_stamp_compact()
        qpath = path.with_name(f"{path.name}.quarantine.{stamp}.json")
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            raw = ""
        envelope = {
            "reason": reason,
            "at_utc": _utc_now(),
            "path": str(path),
            "details": details or {},
            "raw": raw,
        }
        try:
            qpath.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        except Exception:
            qpath = None
        try:
            path.unlink()
        except OSError:
            pass
        return qpath

    def _inspect_process_identity(self, pid: int) -> dict[str, Any]:
        out: dict[str, Any] = {"pid": pid, "alive": _is_pid_alive(pid), "executable_path": "", "command_line": ""}
        if not out["alive"]:
            return out
        if sys.platform == "win32":
            cmd = (
                "Get-CimInstance Win32_Process -Filter \"ProcessId = "
                f"{pid}\" | Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
            )
            try:
                cp = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", cmd],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                txt = (cp.stdout or "").strip()
                if txt:
                    payload = json.loads(txt)
                    if isinstance(payload, list):
                        payload = payload[0] if payload else {}
                    if isinstance(payload, dict):
                        out["executable_path"] = str(payload.get("ExecutablePath") or "")
                        out["command_line"] = str(payload.get("CommandLine") or "")
            except Exception:
                pass
            return out
        try:
            cp = subprocess.run(
                ["ps", "-o", "command=", "-p", str(pid)],
                check=False,
                capture_output=True,
                text=True,
            )
            out["command_line"] = (cp.stdout or "").strip()
        except Exception:
            pass
        return out

    def _verify_expected_proxy_identity(self, identity: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        cmd = str(identity.get("command_line") or "")
        exe = str(identity.get("executable_path") or "")
        checks = {
            "alive": bool(identity.get("alive")),
            "role_marker": "aurora_lens.proxy" in cmd,
            "exe_python": (Path(exe).name.lower().startswith("python") if exe else True),
        }
        verified = checks["alive"] and checks["role_marker"] and checks["exe_python"]
        return verified, checks

    def _fallback_runtime_ownership_check(self, payload: dict[str, Any], port: int) -> tuple[bool, dict[str, Any]]:
        log_path = str(payload.get("log_path") or "")
        config_path = str(payload.get("config") or "")
        under_runtime = False
        for candidate in (log_path, config_path):
            if candidate:
                try:
                    if str(Path(candidate).resolve()).lower().startswith(str(self.paths.runtime_root).lower()):
                        under_runtime = True
                        break
                except Exception:
                    continue
        health_ok, _ = self._probe_health(port)
        checks = {"runtime_path_hint": under_runtime, "health_ok": health_ok}
        return under_runtime and health_ok, checks

    def _kill_pid(self, pid: int) -> None:
        if pid <= 0:
            return
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
            )
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return

    def _probe_health(self, port: int) -> tuple[bool, str]:
        base = f"http://127.0.0.1:{port}"
        try:
            with httpx.Client(timeout=1.5) as client:
                hz = client.get(f"{base}/healthz")
                if hz.status_code != 200:
                    return False, f"/healthz returned HTTP {hz.status_code}"
                h = client.get(f"{base}/health")
                if h.status_code != 200:
                    return False, f"/health returned HTTP {h.status_code}"
                payload = h.json()
                status = str(payload.get("status") or "").lower()
                if status in {"ok", "degraded"}:
                    return True, status
                return False, f"/health status was {status or 'unknown'}"
        except Exception as exc:
            return False, str(exc)

    def _wait_for_proxy_health(self, port: int, timeout_seconds: float = 60.0) -> tuple[bool, str]:
        deadline = time.time() + timeout_seconds
        reason = "Health endpoint did not respond in time."
        while time.time() < deadline:
            ok, detail = self._probe_health(port)
            if ok:
                return True, detail
            reason = detail
            time.sleep(0.5)
        return False, reason

    def _load_port_from_config(self, config_path: Path) -> int:
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return 8081
        listen = payload.get("listen")
        if not isinstance(listen, dict):
            return 8081
        raw = listen.get("port", 8081)
        try:
            return int(raw)
        except Exception:
            return 8081

    def _resolve_audit_log_path(self, config_path: Path) -> Path:
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return self.paths.logs_dir / "audit.jsonl"
        gov = payload.get("governance")
        if not isinstance(gov, dict):
            return self.paths.logs_dir / "audit.jsonl"
        raw = str(gov.get("audit_log") or "").strip()
        if not raw:
            return self.paths.logs_dir / "audit.jsonl"
        p = Path(raw)
        if not p.is_absolute():
            p = config_path.parent / p
        return p

    def _proxy_url(self, port: int) -> str:
        return f"http://127.0.0.1:{port}"

    def proxy_runtime_status(self, config_path: Path) -> RuntimeStatusResult:
        payload = _read_pid_file(self.proxy_pid_path)
        desired_port = self._load_port_from_config(config_path)
        proxy_url = self._proxy_url(desired_port)
        health_url = f"{proxy_url}/health"
        details = {
            "port": desired_port,
            "proxy_url": proxy_url,
            "health_url": health_url,
            "log_path": str(self.proxy_log_path),
            "pid_path": str(self.proxy_pid_path),
        }
        if not payload:
            if not _port_is_free(desired_port):
                ok, _ = self._probe_health(desired_port)
                if ok:
                    return RuntimeStatusResult(
                        RuntimeStatusCode.OWNERSHIP_UNVERIFIED,
                        "A proxy is responding, but no owned process state is available.",
                        details,
                    )
                return RuntimeStatusResult(
                    RuntimeStatusCode.PORT_OCCUPIED,
                    f"Port {desired_port} is occupied by another process.",
                    details,
                )
            return RuntimeStatusResult(RuntimeStatusCode.STOPPED, "Aurora-Lens is stopped.", details)

        pid = int(payload.get("pid") or 0)
        details["pid"] = pid
        pid_status = _pid_status(pid)
        details["pid_status"] = pid_status.value
        if pid_status == _PidStatus.DEAD:
            qpath = self._quarantine_pid_file(
                self.proxy_pid_path,
                reason="stale_pid_state",
                details={"pid": pid, "port": desired_port},
            )
            details["quarantine_path"] = str(qpath) if qpath else ""
            return RuntimeStatusResult(
                RuntimeStatusCode.STALE_STATE_REPAIRED,
                "Stale proxy PID state was repaired.",
                details,
            )

        if pid_status == _PidStatus.UNKNOWN:
            ok, health_detail = self._probe_health(desired_port)
            details["health_detail"] = health_detail
            if ok:
                return RuntimeStatusResult(
                    RuntimeStatusCode.RUNNING_HEALTHY,
                    "Aurora-Lens is healthy; operating-system PID inspection was unavailable.",
                    details,
                )
            return RuntimeStatusResult(
                RuntimeStatusCode.OWNERSHIP_UNVERIFIED,
                "Unable to determine whether the recorded proxy PID is alive.",
                details,
            )

        ident = self._inspect_process_identity(pid)
        verified, checks = self._verify_expected_proxy_identity(ident)
        phase = str(payload.get("phase") or "").lower()
        ok, health_detail = self._probe_health(desired_port)
        details["health_detail"] = health_detail
        if not verified:
            fallback_ok, fallback_checks = self._fallback_runtime_ownership_check(payload, desired_port)
            if fallback_ok:
                details["identity_checks"] = {**checks, **fallback_checks, "fallback_runtime": True}
                if phase == "starting" and not ok:
                    return RuntimeStatusResult(
                        RuntimeStatusCode.STARTING,
                        "Aurora-Lens is starting.",
                        details,
                    )
                if ok:
                    return RuntimeStatusResult(
                        RuntimeStatusCode.RUNNING_HEALTHY,
                        "Aurora-Lens is running and healthy.",
                        details,
                    )
                return RuntimeStatusResult(
                    RuntimeStatusCode.RUNNING_UNHEALTHY,
                    "Aurora-Lens process is running but health checks are failing.",
                    details,
                )
            qpath = self._quarantine_pid_file(
                self.proxy_pid_path,
                reason="ownership_unverified",
                details={"pid": pid, "port": desired_port, "identity_checks": checks, "identity": ident},
            )
            details["quarantine_path"] = str(qpath) if qpath else ""
            details["identity_checks"] = checks
            return RuntimeStatusResult(
                RuntimeStatusCode.OWNERSHIP_UNVERIFIED,
                "Unable to verify ownership of running process; no action taken.",
                details,
            )

        if phase == "starting" and not ok:
            return RuntimeStatusResult(
                RuntimeStatusCode.STARTING,
                "Aurora-Lens is starting.",
                details,
            )
        if ok:
            return RuntimeStatusResult(
                RuntimeStatusCode.RUNNING_HEALTHY,
                "Aurora-Lens is running and healthy.",
                details,
            )
        return RuntimeStatusResult(
            RuntimeStatusCode.RUNNING_UNHEALTHY,
            "Aurora-Lens process is running but health checks are failing.",
            details,
        )

    def start_proxy(self, config_path: Path, runtime_env: dict[str, str]) -> ProxyStartResult:
        status = self.proxy_runtime_status(config_path)
        if status.code in {
            RuntimeStatusCode.RUNNING_HEALTHY,
            RuntimeStatusCode.RUNNING_UNHEALTHY,
            RuntimeStatusCode.STARTING,
            RuntimeStatusCode.OWNERSHIP_UNVERIFIED,
        }:
            return ProxyStartResult(
                ok=status.code in {
                    RuntimeStatusCode.RUNNING_HEALTHY,
                    RuntimeStatusCode.RUNNING_UNHEALTHY,
                    RuntimeStatusCode.STARTING,
                },
                code=status.code,
                message=status.message,
                proxy_url=str(status.details.get("proxy_url") or ""),
                health_url=str(status.details.get("health_url") or ""),
                log_path=str(status.details.get("log_path") or self.proxy_log_path),
            )
        if status.code == RuntimeStatusCode.PORT_OCCUPIED:
            return ProxyStartResult(
                ok=False,
                code=RuntimeStatusCode.PORT_OCCUPIED,
                message=status.message,
                proxy_url=str(status.details.get("proxy_url") or ""),
                health_url=str(status.details.get("health_url") or ""),
                log_path=str(self.proxy_log_path),
            )

        port = self._load_port_from_config(config_path)
        proxy_url = self._proxy_url(port)
        health_url = f"{proxy_url}/health"
        audit_log_path = self._resolve_audit_log_path(config_path)
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not audit_log_path.exists():
            audit_log_path.write_text("", encoding="utf-8")
        cmd = [sys.executable, "-m", "aurora_lens.proxy", "--config", str(config_path)]
        env = os.environ.copy()
        env.update(runtime_env)
        creationflags = 0
        startupinfo = None
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        self.proxy_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.proxy_log_path.open("a", encoding="utf-8") as log_handle:
            proc = subprocess.Popen(
                cmd,
                cwd=str(config_path.parent),
                env=env,
                stdout=log_handle,
                stderr=log_handle,
                creationflags=creationflags,
                startupinfo=startupinfo,
            )
        _write_pid_file(
            self.proxy_pid_path,
            {
                "pid": proc.pid,
                "port": port,
                "phase": "starting",
                "started_at": _utc_now(),
                "config": str(config_path),
                "log_path": str(self.proxy_log_path),
            },
        )
        self.log(
            "proxy_start_requested "
            f"pid={proc.pid} port={port} version={self._release_info['version']} "
            f"release_tag={self._release_info['release_tag']} config={config_path}"
        )
        ok, detail = self._wait_for_proxy_health(port=port, timeout_seconds=60.0)
        if not ok:
            self._kill_pid(proc.pid)
            try:
                self.proxy_pid_path.unlink()
            except OSError:
                pass
            self.log(f"proxy_start_failed pid={proc.pid} port={port} reason={detail}")
            return ProxyStartResult(
                ok=False,
                code=RuntimeStatusCode.RUNNING_UNHEALTHY,
                message=f"Aurora-Lens failed to become healthy: {detail}",
                proxy_url=proxy_url,
                health_url=health_url,
                log_path=str(self.proxy_log_path),
            )
        _write_pid_file(
            self.proxy_pid_path,
            {
                "pid": proc.pid,
                "port": port,
                "phase": "running",
                "started_at": _utc_now(),
                "config": str(config_path),
                "log_path": str(self.proxy_log_path),
            },
        )
        self.log(
            "proxy_started "
            f"pid={proc.pid} port={port} version={self._release_info['version']} "
            f"release_tag={self._release_info['release_tag']} health={detail}"
        )
        return ProxyStartResult(
            ok=True,
            code=RuntimeStatusCode.RUNNING_HEALTHY,
            message="Aurora-Lens is running and healthy.",
            proxy_url=proxy_url,
            health_url=health_url,
            log_path=str(self.proxy_log_path),
        )

    def stop_proxy_lifecycle(self, config_path: Path | None = None) -> LifecycleResult:
        if config_path is None:
            config_path = self.paths.default_config_path
        payload = _read_pid_file(self.proxy_pid_path)
        port = self._load_port_from_config(config_path)
        if not payload:
            if self._probe_health(port)[0]:
                return LifecycleResult(
                    LifecycleResultCode.OWNERSHIP_UNVERIFIED,
                    "Aurora-Lens appears to be running, but no owned proxy PID state is available.",
                    {"role": "proxy", "port": port, "log_path": str(self.proxy_log_path)},
                )
            return LifecycleResult(
                LifecycleResultCode.ALREADY_STOPPED,
                "Aurora-Lens proxy is already stopped.",
                {"role": "proxy", "port": port},
            )

        pid = int(payload.get("pid") or 0)
        pid_status = _pid_status(pid)
        if pid_status == _PidStatus.DEAD:
            qpath = self._quarantine_pid_file(
                self.proxy_pid_path,
                reason="stale_pid_state",
                details={"role": "proxy", "pid": pid, "port": port},
            )
            self.log(f"proxy_stale_state_repaired pid={pid} port={port} quarantine={qpath}")
            return LifecycleResult(
                LifecycleResultCode.STALE_STATE_REPAIRED,
                "Stale proxy PID state was repaired.",
                {"role": "proxy", "pid": pid, "port": port, "quarantine_path": str(qpath) if qpath else ""},
            )

        if pid_status == _PidStatus.UNKNOWN:
            health_ok, health_detail = self._probe_health(port)
            return LifecycleResult(
                LifecycleResultCode.OWNERSHIP_UNVERIFIED,
                "Unable to determine whether the recorded proxy PID is alive; no termination was attempted.",
                {
                    "role": "proxy",
                    "pid": pid,
                    "port": port,
                    "pid_status": pid_status.value,
                    "health_ok": health_ok,
                    "health_detail": health_detail,
                },
            )

        ident = self._inspect_process_identity(pid)
        verified, checks = self._verify_expected_proxy_identity(ident)
        if not verified:
            fallback_ok, fallback_checks = self._fallback_runtime_ownership_check(payload, port)
            if fallback_ok:
                checks = {**checks, **fallback_checks, "fallback_runtime": True}
                verified = True
        if not verified:
            qpath = self._quarantine_pid_file(
                self.proxy_pid_path,
                reason="ownership_unverified",
                details={"role": "proxy", "pid": pid, "port": port, "identity_checks": checks, "identity": ident},
            )
            self.log(f"proxy_ownership_unverified pid={pid} port={port} quarantine={qpath}")
            return LifecycleResult(
                LifecycleResultCode.OWNERSHIP_UNVERIFIED,
                "Unable to verify ownership of the running proxy process; no termination was attempted.",
                {"role": "proxy", "pid": pid, "port": port, "identity_checks": checks, "quarantine_path": str(qpath) if qpath else ""},
            )

        self._kill_pid(pid)
        deadline = time.time() + 8.0
        while time.time() < deadline:
            _reap_exited_child(pid)
            if _pid_status(pid) == _PidStatus.DEAD and not self._probe_health(port)[0]:
                break
            time.sleep(0.25)
        _reap_exited_child(pid)
        pid_status_after = _pid_status(pid)
        health_after = self._probe_health(port)[0]
        if pid_status_after == _PidStatus.DEAD and health_after:
            qpath = self._quarantine_pid_file(
                self.proxy_pid_path,
                reason="ownership_unverified_after_stop",
                details={"role": "proxy", "pid": pid, "port": port},
            )
            self.log(f"proxy_stop_unverified_after_pid_exit pid={pid} port={port} quarantine={qpath}")
            return LifecycleResult(
                LifecycleResultCode.OWNERSHIP_UNVERIFIED,
                "Owned proxy PID exited, but another process still responds on the configured port.",
                {"role": "proxy", "pid": pid, "port": port, "quarantine_path": str(qpath) if qpath else ""},
            )
        if pid_status_after != _PidStatus.DEAD or health_after:
            self.log(f"proxy_stop_failed pid={pid} port={port}")
            return LifecycleResult(
                LifecycleResultCode.TERMINATION_FAILED,
                "Aurora-Lens proxy termination could not be confirmed.",
                {
                    "role": "proxy",
                    "pid": pid,
                    "port": port,
                    "pid_status": pid_status_after.value,
                    "health_ok": health_after,
                },
            )
        try:
            self.proxy_pid_path.unlink()
        except OSError:
            pass
        self.log(f"proxy_stopped pid={pid} port={port}")
        return LifecycleResult(
            LifecycleResultCode.STOPPED,
            "Aurora-Lens proxy stopped.",
            {"role": "proxy", "pid": pid, "port": port},
        )

    def controller_pid(self) -> int | None:
        payload = _read_pid_file(self.controller_pid_path)
        if not payload:
            if self.controller_pid_path.exists():
                try:
                    self.controller_pid_path.unlink()
                except OSError:
                    pass
            return None
        pid = int(payload.get("pid") or 0)
        if _is_pid_alive(pid):
            return pid
        try:
            self.controller_pid_path.unlink()
        except OSError:
            pass
        return None

    def proxy_pid(self) -> int | None:
        payload = _read_pid_file(self.proxy_pid_path)
        if not payload:
            if self.proxy_pid_path.exists():
                try:
                    self.proxy_pid_path.unlink()
                except OSError:
                    pass
            return None
        pid = int(payload.get("pid") or 0)
        if _is_pid_alive(pid):
            return pid
        try:
            self.proxy_pid_path.unlink()
        except OSError:
            pass
        return None

    def stop_proxy(self) -> bool:
        pid = self.proxy_pid()
        if not pid:
            return False
        self._kill_pid(pid)
        time.sleep(0.25)
        if not _is_pid_alive(pid):
            try:
                self.proxy_pid_path.unlink()
            except OSError:
                pass
            self.log(f"proxy_stopped pid={pid}")
            return True
        self.log(f"proxy_stop_failed pid={pid}")
        return False

    def proxy_status(self) -> dict[str, Any]:
        pid = self.proxy_pid()
        proxy_running = bool(pid and _is_pid_alive(pid))
        proxy_port = None
        payload = _read_pid_file(self.proxy_pid_path)
        if payload:
            proxy_port = payload.get("port")
        return {
            "running": proxy_running,
            "pid": pid,
            "port": proxy_port,
        }
