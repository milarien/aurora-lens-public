from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from aurora_lens.launcher.state import LauncherState
from aurora_lens.setup.core import (
    CheckResult,
    DEFAULT_CONFIG_NAME,
    build_readiness_report,
    check_connectivity_test,
    export_support_bundle,
    list_provider_models,
    parse_setup_input,
    provider_options_payload,
    reset_local_data,
    run_governance_smoke,
    run_preflight,
    runtime_env_from_saved_config,
    save_configuration,
)

def _setup_html_path() -> Path:
    return Path(__file__).resolve().parent / "static" / "setup.html"


def create_setup_app(*, home: Path, controller_port: int) -> FastAPI:
    app = FastAPI(title="Aurora-Lens Setup Controller")
    state = LauncherState(home=home)
    smoke_state = {"connectivity": False, "governance": False}
    csrf_token = secrets.token_urlsafe(24)

    def _require_csrf(request: Request) -> JSONResponse | None:
        cookie = request.cookies.get("aurora_setup_csrf", "")
        header = request.headers.get("x-csrf-token", "")
        if not cookie or not header or cookie != header or cookie != csrf_token:
            return JSONResponse(
                {"ok": False, "message": "Your setup session expired. Reload the page and try again."},
                status_code=403,
            )
        origin = request.headers.get("origin", "").strip()
        if origin:
            allowed = {
                f"http://127.0.0.1:{controller_port}",
                f"http://localhost:{controller_port}",
            }
            if origin not in allowed:
                return JSONResponse(
                    {"ok": False, "message": "Setup request origin is not allowed."},
                    status_code=403,
                )
        return None

    @app.get("/setup", response_class=HTMLResponse)
    async def setup_page(request: Request) -> HTMLResponse:
        html = _setup_html_path().read_text(encoding="utf-8")
        html = html.replace("__AURORA_CONTROLLER_PORT__", str(controller_port))
        response = HTMLResponse(html)
        response.set_cookie(
            key="aurora_setup_csrf",
            value=csrf_token,
            httponly=False,
            samesite="lax",
            secure=False,
        )
        return response

    @app.get("/api/providers")
    async def providers() -> dict[str, Any]:
        return provider_options_payload(home)

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        proxy = state.proxy_status()
        return {
            "ok": True,
            "controller_pid": state.controller_pid(),
            "proxy": proxy,
            "smoke": dict(smoke_state),
        }

    @app.post("/api/preflight")
    async def preflight(request: Request) -> dict[str, Any]:
        csrf_error = _require_csrf(request)
        if csrf_error is not None:
            return csrf_error
        checks = [c.to_dict() for c in run_preflight(home)]
        return {"ok": True, "checks": checks}

    @app.post("/api/test/connectivity")
    async def test_connectivity(request: Request) -> dict[str, Any]:
        csrf_error = _require_csrf(request)
        if csrf_error is not None:
            return csrf_error
        payload = await request.json()
        cfg = parse_setup_input(payload, home=home)
        result = check_connectivity_test(cfg)
        smoke_state["connectivity"] = result.status == "pass"
        return {"ok": result.status == "pass", "check": result.to_dict()}

    @app.post("/api/test/port")
    async def test_port(request: Request) -> dict[str, Any]:
        csrf_error = _require_csrf(request)
        if csrf_error is not None:
            return csrf_error
        payload = await request.json()
        cfg = parse_setup_input(payload, home=home)
        readiness = build_readiness_report(cfg, home)
        check = next(c for c in readiness["checks"] if c["id"] == "listen_port")
        return {"ok": check["status"] == "pass", "check": check}

    @app.post("/api/readiness")
    async def readiness(request: Request) -> dict[str, Any]:
        csrf_error = _require_csrf(request)
        if csrf_error is not None:
            return csrf_error
        payload = await request.json()
        cfg = parse_setup_input(
            {
                **payload,
                "connectivity_smoke_ran": smoke_state["connectivity"],
                "governance_smoke_ran": smoke_state["governance"],
            },
            home=home,
        )
        report = build_readiness_report(cfg, home)
        return {"ok": True, **report}

    @app.post("/api/save-config")
    async def save_config(request: Request) -> dict[str, Any]:
        csrf_error = _require_csrf(request)
        if csrf_error is not None:
            return csrf_error
        payload = await request.json()
        cfg = parse_setup_input(payload, home=home)
        data = save_configuration(cfg, home=home)
        return {"ok": True, **data}

    @app.post("/api/provider-models")
    async def provider_models(request: Request) -> dict[str, Any]:
        csrf_error = _require_csrf(request)
        if csrf_error is not None:
            return csrf_error
        payload = await request.json()
        cfg = parse_setup_input(payload, home=home)
        return list_provider_models(cfg)

    @app.post("/api/start")
    async def start_proxy(request: Request) -> dict[str, Any]:
        csrf_error = _require_csrf(request)
        if csrf_error is not None:
            return csrf_error
        payload = await request.json()
        config_name = str(payload.get("config_path") or DEFAULT_CONFIG_NAME).strip()
        config_path = Path(config_name)
        if not config_path.is_absolute():
            config_path = home / config_path
        runtime_env = {}
        if config_path.exists():
            runtime_env = runtime_env_from_saved_config(config_path=config_path, home=home)
        hydrated_payload = dict(payload)
        if not str(hydrated_payload.get("api_key") or "").strip():
            saved_key = runtime_env.get("AURORA_LENS_UPSTREAM_API_KEY", "")
            if saved_key:
                hydrated_payload["api_key"] = saved_key
        if (
            bool(hydrated_payload.get("evidence_vault_enabled", False))
            and not str(hydrated_payload.get("evidence_key") or "").strip()
        ):
            saved_evidence_key = runtime_env.get("AURORA_LENS_EVIDENCE_KEY", "")
            if saved_evidence_key:
                hydrated_payload["evidence_key"] = saved_evidence_key

        setup_cfg = parse_setup_input(
            {
                **hydrated_payload,
                "connectivity_smoke_ran": smoke_state["connectivity"],
                "governance_smoke_ran": smoke_state["governance"],
            },
            home=home,
        )
        readiness = build_readiness_report(setup_cfg, home)
        blocking = [
            c for c in readiness["checks"]
            if c["severity"] == "BLOCKING" and c["status"] == "fail"
        ]
        if blocking:
            return {
                "ok": False,
                "message": "Start blocked. Fix blocking readiness checks first.",
                "blocking_checks": blocking,
            }
        if not config_path.exists():
            return {
                "ok": False,
                "message": "No saved configuration was found. Save your setup first.",
            }
        result = state.start_proxy(config_path=config_path, runtime_env=runtime_env)
        out = {"ok": result.ok, "message": result.message, "proxy_url": result.proxy_url}
        if result.warning:
            out["warning"] = result.warning
        return out

    @app.post("/api/stop")
    async def stop_proxy(request: Request) -> dict[str, Any]:
        csrf_error = _require_csrf(request)
        if csrf_error is not None:
            return csrf_error
        stopped = state.stop_proxy()
        if stopped:
            return {"ok": True, "message": "Aurora-Lens stopped."}
        return {"ok": True, "message": "Aurora-Lens was not running."}

    @app.post("/api/restart")
    async def restart_proxy(request: Request) -> dict[str, Any]:
        csrf_error = _require_csrf(request)
        if csrf_error is not None:
            return csrf_error
        payload = await request.json()
        config_name = str(payload.get("config_path") or DEFAULT_CONFIG_NAME).strip()
        config_path = Path(config_name)
        if not config_path.is_absolute():
            config_path = home / config_path
        runtime_env = {}
        if config_path.exists():
            runtime_env = runtime_env_from_saved_config(config_path=config_path, home=home)
        state.stop_proxy()
        result = state.start_proxy(config_path=config_path, runtime_env=runtime_env)
        out = {"ok": result.ok, "message": result.message, "proxy_url": result.proxy_url}
        if result.warning:
            out["warning"] = result.warning
        return out

    @app.post("/api/export-logs")
    async def export_logs(request: Request) -> dict[str, Any]:
        csrf_error = _require_csrf(request)
        if csrf_error is not None:
            return csrf_error
        payload = await request.json()
        config_name = str(payload.get("config_path") or DEFAULT_CONFIG_NAME).strip()
        config_path = Path(config_name)
        if not config_path.is_absolute():
            config_path = home / config_path
        result = export_support_bundle(home=home, config_path=config_path)
        return {
            "ok": True,
            "message": "Support bundle exported.",
            **result,
        }

    @app.post("/api/reset-local-data")
    async def reset_data(request: Request) -> dict[str, Any]:
        csrf_error = _require_csrf(request)
        if csrf_error is not None:
            return csrf_error
        payload = await request.json()
        confirm_phrase = str(payload.get("confirm_phrase") or "").strip()
        if confirm_phrase != "RESET AURORA-LENS":
            return {
                "ok": False,
                "message": "Reset blocked. Type RESET AURORA-LENS to confirm.",
            }
        remove_config = bool(payload.get("remove_config", False))
        remove_audit = bool(payload.get("remove_audit", True))
        clear_secrets = bool(payload.get("clear_secrets", False))
        config_name = str(payload.get("config_path") or DEFAULT_CONFIG_NAME).strip()
        config_path = Path(config_name)
        if not config_path.is_absolute():
            config_path = home / config_path
        provider_type = "openai"
        if config_path.exists():
            try:
                payload_map = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                provider_type = str(((payload_map.get("setup") or {}).get("provider_type")) or "openai")
            except Exception:
                provider_type = "openai"
        state.stop_proxy()
        result = reset_local_data(
            home=home,
            provider_type=provider_type,
            config_path=config_path,
            remove_config=remove_config,
            remove_audit=remove_audit,
            clear_secrets=clear_secrets,
        )
        return {
            "ok": True,
            "message": "Local reset complete.",
            **result,
        }

    @app.post("/api/smoke/connectivity")
    async def smoke_connectivity(request: Request) -> dict[str, Any]:
        csrf_error = _require_csrf(request)
        if csrf_error is not None:
            return csrf_error
        payload = await request.json()
        cfg = parse_setup_input(payload, home=home)
        result = check_connectivity_test(cfg)
        smoke_state["connectivity"] = result.status == "pass"
        if result.status != "pass":
            return {"ok": False, "steps": [result.message]}
        return {"ok": True, "steps": ["Message received.", "Provider contacted.", "Model responded."]}

    @app.post("/api/smoke/governance")
    async def smoke_governance(request: Request) -> dict[str, Any]:
        csrf_error = _require_csrf(request)
        if csrf_error is not None:
            return csrf_error
        result = await run_governance_smoke(home=home)
        smoke_state["governance"] = True
        return result

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover - safety net
        _ = request
        return JSONResponse(
            {"ok": False, "message": "Something failed during setup. Please retry from the previous step."},
            status_code=500,
        )

    return app

