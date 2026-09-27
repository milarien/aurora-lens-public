from __future__ import annotations

import json
import os
import secrets
import socket
import sys
import zipfile
from datetime import UTC, datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import httpx
import yaml

from aurora_lens.adapters.base import AdapterResponse
from aurora_lens.config import LensConfig
from aurora_lens.govern.canonical_bridge import CanonicalScannerGateBridge
from aurora_lens.govern.decision import InterventionAction
from aurora_lens.interpret.base import ExtractionBackend
from aurora_lens.interpret.schema import ExtractionResult
from aurora_lens.lens import Lens
from aurora_lens.pef.span import Span
from aurora_lens.pef.state import PEFState
from aurora_lens.proxy.config import ProxyConfig
from aurora_lens.build_info import load_build_info, public_release_metadata

DEFAULT_SETUP_PORT = 8765
DEFAULT_CONFIG_NAME = "aurora-lens.yaml"
DEFAULT_ENV_NAME = ".env"

SEVERITY_PASS = "PASS"
SEVERITY_WARNING = "WARNING"
SEVERITY_BLOCKING = "BLOCKING"

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_WARN = "warn"


@dataclass(frozen=True)
class ProviderProfile:
    id: str
    label: str
    endpoint_default: str
    model_default: str
    provider_value: str
    requires_api_key: bool
    endpoint_help: str
    key_help: str


PROVIDER_PROFILES: dict[str, ProviderProfile] = {
    "openai": ProviderProfile(
        id="openai",
        label="OpenAI",
        endpoint_default="https://api.openai.com/v1",
        model_default="gpt-4.1-mini",
        provider_value="openai",
        requires_api_key=True,
        endpoint_help="Default OpenAI API base URL.",
        key_help="Get an API key from OpenAI Platform: https://platform.openai.com/api-keys",
    ),
    "anthropic": ProviderProfile(
        id="anthropic",
        label="Anthropic",
        endpoint_default="https://api.anthropic.com",
        model_default="claude-sonnet-4-20250514",
        provider_value="anthropic",
        requires_api_key=True,
        endpoint_help="Default Anthropic API base URL.",
        key_help="Get an API key from Anthropic Console: https://console.anthropic.com/settings/keys",
    ),
    "openai_compatible": ProviderProfile(
        id="openai_compatible",
        label="OpenAI-compatible endpoint",
        endpoint_default="http://127.0.0.1:11434/v1",
        model_default="local-model",
        provider_value="openai",
        requires_api_key=False,
        endpoint_help="Base URL for OpenAI-compatible APIs, for example vLLM or LM Studio.",
        key_help="API keys depend on your endpoint. If authentication is enabled, use the key issued by that endpoint.",
    ),
    "local": ProviderProfile(
        id="local",
        label="Local model endpoint",
        endpoint_default="http://127.0.0.1:11434/v1",
        model_default="llama3.1:8b",
        provider_value="local",
        requires_api_key=False,
        endpoint_help="Local endpoint URL. Aurora-Lens talks to this server on your machine.",
        key_help="No API key is usually required for local model endpoints.",
    ),
    "custom": ProviderProfile(
        id="custom",
        label="Custom OpenAI-compatible endpoint",
        endpoint_default="https://example.com/v1",
        model_default="custom-model",
        provider_value="openai",
        requires_api_key=True,
        endpoint_help="Custom OpenAI-compatible endpoint URL.",
        key_help="Use the API key issued by your custom provider endpoint.",
    ),
}

WORK_TYPE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("legal", "Legal"),
    ("medical", "Medical"),
    ("human_resources", "Human resources"),
    ("financial_services", "Financial services"),
    ("education", "Education"),
    ("general_ai_governance", "General AI governance"),
)

AUDIENCE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("customers_public", "Customers or the public"),
    ("staff_organisation", "Staff inside an organisation"),
    ("development_testing", "Development or testing"),
)

WORK_TYPE_TO_DOMAIN: dict[str, str] = {
    "legal": "legal",
    "medical": "medical",
    "human_resources": "workforce",
    "financial_services": "finance",
    "education": "education",
    "general_ai_governance": "general",
}

AUDIENCE_TO_MODE: dict[str, str] = {
    "customers_public": "public",
    "staff_organisation": "enterprise",
    "development_testing": "open",
}


@dataclass(frozen=True)
class CheckResult:
    id: str
    severity: str
    status: str
    message: str
    fix_hint: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SetupInput:
    work_type: str
    audience_type: str
    provider_type: str
    model_name: str
    api_endpoint: str
    api_key: str
    listen_port: int
    audit_log_path: str
    session_backend: str
    redis_url: str
    evidence_vault_enabled: bool
    evidence_key: str
    policy_profile: str
    extraction_backend: str = "spacy"
    config_path: str = DEFAULT_CONFIG_NAME
    connectivity_smoke_ran: bool = False
    governance_smoke_ran: bool = False


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _default_audit_log_path(home: Path) -> str:
    # Canonical audit log path is under the runtime logs/ directory so that
    # both `init-config` (which writes ``logs/audit.jsonl``) and the launcher
    # (which resolves the same path when no explicit config value is set) end
    # up writing to a single deterministic location. Previously this returned
    # ``audit/audit.jsonl``, which caused the audit stream to be split across
    # two sibling directories.
    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    if local_appdata:
        return str(Path(local_appdata) / "Aurora-Lens" / "logs" / "audit.jsonl")
    return str(home / "logs" / "audit.jsonl")


def _next_available_port(start: int = 8000, host: str = "127.0.0.1") -> int:
    for port in range(max(1, start), 65536):
        if _port_is_free(port, host=host):
            return port
    return start


class _NoCallAdapter:
    calls: int

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> AdapterResponse:  # noqa: ARG002
        self.calls += 1
        return AdapterResponse(text="unused", model="setup-mock")


class _AmbiguousBackend(ExtractionBackend):
    async def extract(self, text: str, pef: PEFState) -> ExtractionResult:  # noqa: ARG002
        return ExtractionResult(
            claims=[],
            entity_mentions=["Emma", "Anna"],
            span=Span.PRESENT,
            ambiguous_referents=["her"],
        )


def parse_setup_input(payload: dict[str, Any], home: Path | None = None) -> SetupInput:
    base_home = home or Path.cwd()
    work_type = str(payload.get("work_type") or "general_ai_governance").strip().lower()
    if work_type not in WORK_TYPE_TO_DOMAIN:
        work_type = "general_ai_governance"

    audience_type = str(payload.get("audience_type") or "customers_public").strip().lower()
    if audience_type not in AUDIENCE_TO_MODE:
        audience_type = "customers_public"

    provider_type = str(payload.get("provider_type") or "openai").strip().lower()
    if provider_type not in PROVIDER_PROFILES:
        provider_type = "openai"

    listen_port_raw = payload.get("listen_port", None)
    if listen_port_raw is None or str(listen_port_raw).strip() == "":
        listen_port = _next_available_port(8000)
    else:
        try:
            listen_port = int(listen_port_raw)
        except Exception:
            listen_port = _next_available_port(8000)

    session_backend = str(payload.get("session_backend") or "memory").strip().lower()
    if session_backend not in {"memory", "redis"}:
        session_backend = "memory"

    extraction_backend = str(payload.get("extraction_backend") or "spacy").strip().lower()
    if extraction_backend not in {"spacy", "llm"}:
        extraction_backend = "spacy"

    policy_profile = str(payload.get("policy_profile") or "strict").strip().lower()
    if policy_profile not in {"strict", "moderate"}:
        policy_profile = "strict"

    return SetupInput(
        work_type=work_type,
        audience_type=audience_type,
        provider_type=provider_type,
        model_name=str(payload.get("model_name") or "").strip(),
        api_endpoint=str(payload.get("api_endpoint") or PROVIDER_PROFILES[provider_type].endpoint_default).strip(),
        api_key=str(payload.get("api_key") or "").strip(),
        listen_port=listen_port,
        audit_log_path=str(payload.get("audit_log_path") or _default_audit_log_path(base_home)).strip(),
        session_backend=session_backend,
        redis_url=str(payload.get("redis_url") or "redis://127.0.0.1:6379/0").strip(),
        evidence_vault_enabled=_as_bool(payload.get("evidence_vault_enabled"), True),
        evidence_key=str(payload.get("evidence_key") or "").strip(),
        policy_profile=policy_profile,
        extraction_backend=extraction_backend,
        config_path=str(payload.get("config_path") or DEFAULT_CONFIG_NAME).strip(),
        connectivity_smoke_ran=bool(payload.get("connectivity_smoke_ran", False)),
        governance_smoke_ran=bool(payload.get("governance_smoke_ran", False)),
    )


def list_provider_models(cfg: SetupInput) -> dict[str, Any]:
    profile = PROVIDER_PROFILES[cfg.provider_type]
    base_url = cfg.api_endpoint.rstrip("/")
    if not base_url:
        return {"ok": False, "models": [], "message": "Provider endpoint is required before loading models."}
    if profile.requires_api_key and not cfg.api_key:
        return {"ok": False, "models": [], "message": "Enter your provider API key to load available models."}

    headers: dict[str, str] = {}
    if cfg.api_key:
        if cfg.provider_type == "anthropic":
            headers["x-api-key"] = cfg.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
    timeout = httpx.Timeout(20.0)
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{base_url}/models", headers=headers)
        if response.status_code == 401:
            return {"ok": False, "models": [], "message": "Your API key was rejected."}
        response.raise_for_status()
        payload = response.json() or {}
    except Exception:
        return {
            "ok": False,
            "models": [],
            "message": "Model listing is unavailable for this provider or endpoint. Use manual model entry in Advanced settings.",
        }

    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return {
            "ok": False,
            "models": [],
            "message": "Model listing is unavailable for this provider or endpoint. Use manual model entry in Advanced settings.",
        }
    models: list[str] = []
    seen: set[str] = set()
    for item in entries:
        if isinstance(item, dict):
            model_id = str(item.get("id") or "").strip()
            if model_id and model_id not in seen:
                seen.add(model_id)
                models.append(model_id)
    if not models:
        return {
            "ok": False,
            "models": [],
            "message": "No models were returned. Use manual model entry in Advanced settings.",
        }
    return {"ok": True, "models": models, "message": "Available models loaded."}


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def _python_version_ok() -> bool:
    return sys.version_info.major == 3 and sys.version_info.minor == 12


def _audit_path_writable(path: Path) -> bool:
    target_dir = path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    probe = target_dir / ".aurora-lens-write-test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def run_preflight(home: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []

    if _python_version_ok():
        checks.append(CheckResult("python_version", SEVERITY_PASS, STATUS_PASS, "Python 3.12 is available."))
    else:
        checks.append(
            CheckResult(
                "python_version",
                SEVERITY_BLOCKING,
                STATUS_FAIL,
                "Python 3.12 is required.",
                "Install Python 3.12 and restart Aurora-Lens.",
            )
        )

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except Exception:
        checks.append(
            CheckResult(
                "proxy_packages",
                SEVERITY_BLOCKING,
                STATUS_FAIL,
                "Required proxy packages are missing.",
                'Run: pip install ".[proxy,spacy]"',
            )
        )
    else:
        checks.append(CheckResult("proxy_packages", SEVERITY_PASS, STATUS_PASS, "Proxy packages are installed."))

    try:
        import spacy  # noqa: F401
        import en_core_web_sm  # type: ignore # noqa: F401
    except Exception:
        checks.append(
            CheckResult(
                "spacy_model",
                SEVERITY_WARNING,
                STATUS_WARN,
                "The spaCy language model is not available yet.",
                "Run: python -m spacy download en_core_web_sm",
            )
        )
    else:
        checks.append(CheckResult("spacy_model", SEVERITY_PASS, STATUS_PASS, "spaCy model is installed."))

    if _audit_path_writable(home / "audit.jsonl"):
        checks.append(CheckResult("install_directory_writable", SEVERITY_PASS, STATUS_PASS, "Install folder is writable."))
    else:
        checks.append(
            CheckResult(
                "install_directory_writable",
                SEVERITY_BLOCKING,
                STATUS_FAIL,
                "Aurora-Lens cannot write files in this folder.",
                "Choose a writable folder or run with permissions that allow writing.",
            )
        )
    return checks


def _provider_check(cfg: SetupInput) -> CheckResult:
    profile = PROVIDER_PROFILES[cfg.provider_type]
    if not cfg.model_name:
        return CheckResult(
            "model_name",
            SEVERITY_BLOCKING,
            STATUS_FAIL,
            "Model name is missing.",
            "Enter the model name you want Aurora-Lens to use.",
        )
    if not cfg.api_endpoint:
        return CheckResult(
            "api_endpoint",
            SEVERITY_BLOCKING,
            STATUS_FAIL,
            "API endpoint is missing.",
            "Enter the provider endpoint URL.",
        )
    if profile.requires_api_key and not cfg.api_key:
        return CheckResult(
            "api_key",
            SEVERITY_BLOCKING,
            STATUS_FAIL,
            "Provider API key is missing.",
            "Paste your provider API key to continue.",
        )
    return CheckResult("provider_fields", SEVERITY_PASS, STATUS_PASS, "Provider fields look valid.")


def _port_check(cfg: SetupInput) -> CheckResult:
    if cfg.listen_port <= 0 or cfg.listen_port > 65535:
        return CheckResult(
            "listen_port",
            SEVERITY_BLOCKING,
            STATUS_FAIL,
            "The selected port is invalid.",
            "Use a number between 1 and 65535. Use 8000 unless another program uses it.",
        )
    if not _port_is_free(cfg.listen_port):
        return CheckResult(
            "listen_port",
            SEVERITY_BLOCKING,
            STATUS_FAIL,
            f"The selected port {cfg.listen_port} is already in use.",
            "Choose a different port, for example 8001.",
        )
    return CheckResult("listen_port", SEVERITY_PASS, STATUS_PASS, f"Port {cfg.listen_port} is available.")


def _audit_check(cfg: SetupInput, home: Path) -> CheckResult:
    audit_path = Path(cfg.audit_log_path)
    if not audit_path.is_absolute():
        audit_path = home / audit_path
    if _audit_path_writable(audit_path):
        return CheckResult("audit_directory", SEVERITY_PASS, STATUS_PASS, "Audit directory is writable.")
    return CheckResult(
        "audit_directory",
        SEVERITY_BLOCKING,
        STATUS_FAIL,
        "Audit directory is not writable.",
        "Choose a folder where Aurora-Lens can create and update files.",
    )


def _redis_check(cfg: SetupInput) -> CheckResult:
    if cfg.session_backend != "redis":
        return CheckResult("redis_reachable", SEVERITY_PASS, STATUS_PASS, "Redis is not selected.")
    if not cfg.redis_url:
        return CheckResult(
            "redis_reachable",
            SEVERITY_BLOCKING,
            STATUS_FAIL,
            "Redis is enabled, but no Redis URL is configured.",
            "Enter a Redis URL or switch session storage to local memory.",
        )
    try:
        import redis
    except Exception:
        return CheckResult(
            "redis_reachable",
            SEVERITY_BLOCKING,
            STATUS_FAIL,
            "Redis is enabled, but the Redis client package is missing.",
            'Install Redis support: pip install ".[redis]"',
        )
    try:
        client = redis.Redis.from_url(cfg.redis_url, socket_connect_timeout=1, socket_timeout=1)
        client.ping()
    except Exception:
        return CheckResult(
            "redis_reachable",
            SEVERITY_BLOCKING,
            STATUS_FAIL,
            f"Redis is enabled, but no Redis server could be reached at {cfg.redis_url}.",
            "Start Redis or switch session storage to local memory.",
        )
    return CheckResult("redis_reachable", SEVERITY_PASS, STATUS_PASS, "Redis is reachable.")


def _evidence_check(cfg: SetupInput) -> CheckResult:
    if not cfg.evidence_vault_enabled:
        return CheckResult("evidence_vault", SEVERITY_PASS, STATUS_PASS, "Evidence vault is disabled.")
    if not cfg.evidence_key:
        return CheckResult(
            "evidence_key",
            SEVERITY_PASS,
            STATUS_PASS,
            "Evidence vault is enabled. Aurora-Lens will generate and store an encryption key automatically.",
        )
    return CheckResult("evidence_key", SEVERITY_PASS, STATUS_PASS, "Evidence vault key is configured.")


def _spacy_check(cfg: SetupInput) -> CheckResult:
    if cfg.extraction_backend != "spacy":
        return CheckResult(
            "spacy_mode",
            SEVERITY_WARNING,
            STATUS_WARN,
            "spaCy is optional for this extraction mode.",
            "You can still install it for stronger deterministic extraction coverage.",
        )
    try:
        import en_core_web_sm  # type: ignore # noqa: F401
    except Exception:
        return CheckResult(
            "spacy_mode",
            SEVERITY_BLOCKING,
            STATUS_FAIL,
            "spaCy extraction is selected, but the model is missing.",
            "Run: python -m spacy download en_core_web_sm",
        )
    return CheckResult("spacy_mode", SEVERITY_PASS, STATUS_PASS, "spaCy extraction requirements are met.")


def _smoke_status_check(name: str, ran: bool) -> CheckResult:
    if ran:
        return CheckResult(name, SEVERITY_PASS, STATUS_PASS, f"{name.replace('_', ' ').title()} was run.")
    return CheckResult(
        name,
        SEVERITY_WARNING,
        STATUS_WARN,
        f"{name.replace('_', ' ').title()} has not been run yet.",
        "Run smoke tests before production use.",
    )


def build_readiness_report(cfg: SetupInput, home: Path) -> dict[str, Any]:
    checks: list[CheckResult] = [
        _provider_check(cfg),
        _port_check(cfg),
        _audit_check(cfg, home),
        _redis_check(cfg),
        _evidence_check(cfg),
        _spacy_check(cfg),
        _smoke_status_check("connectivity_smoke", cfg.connectivity_smoke_ran),
        _smoke_status_check("governance_smoke", cfg.governance_smoke_ran),
    ]

    start_allowed = not any(
        c.severity == SEVERITY_BLOCKING and c.status == STATUS_FAIL for c in checks
    )
    return {
        "start_allowed": start_allowed,
        "checks": [c.to_dict() for c in checks],
    }


def provider_options_payload(home: Path) -> dict[str, Any]:
    return {
        "work_types": [{"id": option_id, "label": label} for option_id, label in WORK_TYPE_OPTIONS],
        "audience_types": [{"id": option_id, "label": label} for option_id, label in AUDIENCE_OPTIONS],
        "defaults": {
            "work_type": "general_ai_governance",
            "audience_type": "customers_public",
            "listen_port": _next_available_port(8000),
            "audit_log_path": _default_audit_log_path(home),
            "session_backend": "memory",
            "redis_url": "redis://127.0.0.1:6379/0",
            "evidence_vault_enabled": True,
            "policy_profile": "strict",
        },
        "providers": [
            {
                "id": p.id,
                "label": p.label,
                "endpoint_default": p.endpoint_default,
                "model_default": p.model_default,
                "requires_api_key": p.requires_api_key,
                "endpoint_help": p.endpoint_help,
                "key_help": p.key_help,
            }
            for p in PROVIDER_PROFILES.values()
        ]
    }


def _to_proxy_mapping(cfg: SetupInput) -> dict[str, Any]:
    profile = PROVIDER_PROFILES[cfg.provider_type]
    provider_value = profile.provider_value
    domain_value = WORK_TYPE_TO_DOMAIN.get(cfg.work_type, "general")
    mode_value = AUDIENCE_TO_MODE.get(cfg.audience_type, "public")
    upstream: dict[str, Any] = {
        "provider": provider_value,
        "model": cfg.model_name,
        "base_url": cfg.api_endpoint,
        "api_key": "${AURORA_LENS_UPSTREAM_API_KEY}" if profile.requires_api_key else "",
    }
    if provider_value == "local":
        upstream["api_key"] = ""
    if cfg.provider_type == "anthropic":
        # Anthropic adapter can use default base URL; include only when user changed it.
        if cfg.api_endpoint == profile.endpoint_default:
            upstream.pop("base_url", None)

    mapping: dict[str, Any] = {
        "upstream": upstream,
        "listen": {"host": "127.0.0.1", "port": cfg.listen_port},
        "governance": {
            "default_policy": cfg.policy_profile,
            "mode": mode_value,
            "default_domain": domain_value,
            "audit_log": cfg.audit_log_path,
            "evidence_capture_mode": "sealed" if cfg.evidence_vault_enabled else "hash_only",
            "evidence_encryption_key": "${AURORA_LENS_EVIDENCE_KEY}" if cfg.evidence_vault_enabled else None,
        },
        "session": {
            "backend": cfg.session_backend,
            "ttl_seconds": 3600,
            "redis_url": "${AURORA_LENS_REDIS_URL}" if cfg.session_backend == "redis" else "",
        },
        "extraction": {"backend": cfg.extraction_backend},
        "telemetry": {"enabled": False},
        "setup": {
            "work_type": cfg.work_type,
            "audience_type": cfg.audience_type,
            "provider_type": cfg.provider_type,
            "secret_backend": "credential_manager_or_env",
            "api_key_ref": f"credential_manager:{cfg.provider_type}",
            "evidence_key_ref": "credential_manager:evidence_key" if cfg.evidence_vault_enabled else "",
        },
    }
    return mapping


def generated_yaml_text(cfg: SetupInput) -> str:
    mapping = _to_proxy_mapping(cfg)
    return yaml.safe_dump(mapping, sort_keys=False, allow_unicode=False)


def _store_secret_with_keyring(service: str, username: str, secret: str) -> bool:
    if not secret:
        return True
    try:
        import keyring
    except Exception:
        return False
    try:
        keyring.set_password(service, username, secret)
    except Exception:
        return False
    return True


def _load_secret_with_keyring(service: str, username: str) -> str | None:
    try:
        import keyring
    except Exception:
        return None
    try:
        return keyring.get_password(service, username)
    except Exception:
        return None


def save_configuration(cfg: SetupInput, home: Path) -> dict[str, Any]:
    config_path = Path(cfg.config_path)
    if not config_path.is_absolute():
        config_path = home / config_path
    config_path.parent.mkdir(parents=True, exist_ok=True)

    mapping = _to_proxy_mapping(cfg)
    text = yaml.safe_dump(mapping, sort_keys=False, allow_unicode=False)
    config_path.write_text(text, encoding="utf-8")

    backend = "keyring"
    service = "aurora-lens"
    api_user = f"provider:{cfg.provider_type}"
    evidence_user = "evidence:key"
    redis_user = "redis:url"
    generated_evidence_key = False
    effective_evidence_key = cfg.evidence_key.strip()
    if cfg.evidence_vault_enabled and not effective_evidence_key:
        effective_evidence_key = secrets.token_urlsafe(32)
        generated_evidence_key = True

    if cfg.api_key and not _store_secret_with_keyring(service, api_user, cfg.api_key):
        backend = "env_file"
    if cfg.evidence_vault_enabled and effective_evidence_key and backend == "keyring":
        if not _store_secret_with_keyring(service, evidence_user, effective_evidence_key):
            backend = "env_file"
    if cfg.session_backend == "redis" and cfg.redis_url and backend == "keyring":
        if not _store_secret_with_keyring(service, redis_user, cfg.redis_url):
            backend = "env_file"

    env_path = home / DEFAULT_ENV_NAME
    env_pairs: dict[str, str] = {}
    if backend == "env_file":
        if cfg.api_key:
            env_pairs["AURORA_LENS_UPSTREAM_API_KEY"] = cfg.api_key
        if cfg.evidence_vault_enabled and effective_evidence_key:
            env_pairs["AURORA_LENS_EVIDENCE_KEY"] = effective_evidence_key
        if cfg.session_backend == "redis" and cfg.redis_url:
            env_pairs["AURORA_LENS_REDIS_URL"] = cfg.redis_url
        lines = [f"{k}={v}" for k, v in env_pairs.items()]
        env_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    else:
        if env_path.exists():
            env_path.unlink()

    return {
        "config_path": str(config_path),
        "secret_backend": backend,
        "env_path": str(env_path),
        "evidence_key_generated": generated_evidence_key,
    }


def runtime_env_from_saved_config(config_path: Path, home: Path) -> dict[str, str]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    setup_map = payload.get("setup") or {}
    provider_type = str(setup_map.get("provider_type") or "openai")
    service = "aurora-lens"
    runtime: dict[str, str] = {}

    api_user = f"provider:{provider_type}"
    key = _load_secret_with_keyring(service, api_user)
    if key:
        runtime["AURORA_LENS_UPSTREAM_API_KEY"] = key

    evidence_key = _load_secret_with_keyring(service, "evidence:key")
    if evidence_key:
        runtime["AURORA_LENS_EVIDENCE_KEY"] = evidence_key

    redis_url = _load_secret_with_keyring(service, "redis:url")
    if redis_url:
        runtime["AURORA_LENS_REDIS_URL"] = redis_url

    env_path = home / DEFAULT_ENV_NAME
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            runtime[k.strip()] = v.strip()
    return runtime


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def _resolve_path(path_text: str, home: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = home / path
    return path


def _remove_file(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        return False
    path.unlink()
    return True


def clear_saved_secrets(home: Path, provider_type: str) -> None:
    service = "aurora-lens"
    users = [f"provider:{provider_type}", "evidence:key", "redis:url"]
    try:
        import keyring
    except Exception:
        keyring = None  # type: ignore[assignment]
    if keyring is not None:
        for username in users:
            try:
                keyring.delete_password(service, username)
            except Exception:
                continue
    env_path = home / DEFAULT_ENV_NAME
    if env_path.exists():
        env_path.unlink()


def export_support_bundle(home: Path, config_path: Path | None = None) -> dict[str, Any]:
    resolved_config = config_path or (home / DEFAULT_CONFIG_NAME)
    runtime_files: list[Path] = [
        home / "launcher.log",
        home / "setup_governance_smoke_audit.jsonl",
        home / "aurora-lens.proxy.pid",
        home / "aurora-lens.controller.pid",
    ]
    if resolved_config.exists():
        runtime_files.append(resolved_config)
        try:
            payload = yaml.safe_load(resolved_config.read_text(encoding="utf-8")) or {}
            gov = payload.get("governance") or {}
            audit_path = str(gov.get("audit_log") or "").strip()
            if audit_path:
                runtime_files.append(_resolve_path(audit_path, home))
        except Exception:
            pass
    runtime_files.append(home / "audit.jsonl")

    export_dir = home / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    archive_path = export_dir / f"aurora-lens-support-{_utc_stamp()}.zip"
    included_files: list[str] = []
    seen_paths: set[Path] = set()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file_path in runtime_files:
            if not file_path.exists() or not file_path.is_file():
                continue
            resolved = file_path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            try:
                rel = file_path.relative_to(home)
                arcname = rel.as_posix()
            except ValueError:
                arcname = file_path.name
            bundle.write(file_path, arcname=arcname)
            included_files.append(arcname)
        build = load_build_info()
        manifest = {
            "created_utc": datetime.now(UTC).isoformat(),
            "included_files": included_files,
            "config_path": str(resolved_config),
            "release": public_release_metadata(build),
        }
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    return {
        "archive_path": str(archive_path),
        "included_files": included_files,
    }


def reset_local_data(
    *,
    home: Path,
    provider_type: str,
    config_path: Path | None = None,
    remove_config: bool = False,
    remove_audit: bool = True,
    clear_secrets: bool = False,
) -> dict[str, Any]:
    resolved_config = config_path or (home / DEFAULT_CONFIG_NAME)
    removed: list[str] = []
    if _remove_file(home / "aurora-lens.proxy.pid"):
        removed.append("aurora-lens.proxy.pid")
    if _remove_file(home / "aurora-lens.controller.pid"):
        removed.append("aurora-lens.controller.pid")
    if _remove_file(home / "setup_governance_smoke_audit.jsonl"):
        removed.append("setup_governance_smoke_audit.jsonl")
    if _remove_file(home / "launcher.log"):
        removed.append("launcher.log")
    if remove_audit:
        if _remove_file(home / "audit.jsonl"):
            removed.append("audit.jsonl")
        if resolved_config.exists():
            try:
                payload = yaml.safe_load(resolved_config.read_text(encoding="utf-8")) or {}
                gov = payload.get("governance") or {}
                audit_path_text = str(gov.get("audit_log") or "").strip()
                if audit_path_text:
                    audit_path = _resolve_path(audit_path_text, home)
                    if _remove_file(audit_path):
                        removed.append(str(audit_path))
            except Exception:
                pass
    if remove_config and _remove_file(resolved_config):
        removed.append(str(resolved_config))
    if clear_secrets:
        clear_saved_secrets(home=home, provider_type=provider_type)
    return {
        "removed_files": removed,
        "remove_config": remove_config,
        "remove_audit": remove_audit,
        "cleared_secrets": clear_secrets,
    }


def run_connectivity_smoke(cfg: SetupInput) -> dict[str, Any]:
    profile = PROVIDER_PROFILES[cfg.provider_type]
    base_url = cfg.api_endpoint.rstrip("/")
    model = cfg.model_name
    steps = ["Message received."]

    if profile.requires_api_key and not cfg.api_key:
        raise ValueError("Your API key is missing.")

    timeout = httpx.Timeout(20.0)
    headers: dict[str, str] = {}
    if cfg.api_key:
        if cfg.provider_type == "anthropic":
            headers["x-api-key"] = cfg.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

    with httpx.Client(timeout=timeout) as client:
        if cfg.provider_type == "anthropic":
            resp = client.post(
                f"{base_url}/messages",
                headers=headers,
                json={
                    "model": model,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "Reply with OK."}],
                },
            )
            if resp.status_code == 401:
                raise ValueError("Your API key was rejected.")
            resp.raise_for_status()
            data = resp.json()
            txt = ""
            content = data.get("content")
            if isinstance(content, list) and content:
                txt = str(content[0].get("text") or "")
        else:
            resp = client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Reply with OK."}],
                    "max_tokens": 16,
                },
            )
            if resp.status_code == 401:
                raise ValueError("Your API key was rejected.")
            resp.raise_for_status()
            data = resp.json()
            txt = (
                (((data.get("choices") or [{}])[0]).get("message") or {}).get("content")
                or ""
            )

    if not str(txt).strip():
        raise ValueError("Provider responded, but the model returned an empty response.")
    steps.append("Provider contacted.")
    steps.append("Model responded.")
    return {"ok": True, "steps": steps}


async def run_governance_smoke(home: Path) -> dict[str, Any]:
    audit_path = home / "setup_governance_smoke_audit.jsonl"
    if audit_path.exists():
        audit_path.unlink()

    bridge = CanonicalScannerGateBridge(mode="public", audit_path=str(audit_path))
    adapter = _NoCallAdapter()
    pef = PEFState()
    pef.get_or_create_entity("Emma")
    pef.get_or_create_entity("Anna")
    lens = Lens(
        LensConfig(
            adapter=adapter,
            extraction_backend=_AmbiguousBackend(),
            governance_bridge=bridge,
            auto_verify=True,
            enable_state_native_delegation=False,
        ),
        initial_pef=pef,
    )
    result = await lens.process("Emma told Anna her sister arrived. Where is she?")
    if result.action != InterventionAction.CONTAIN:
        raise ValueError("Governance smoke expected CONTAIN but got a different decision.")
    if not audit_path.exists():
        raise ValueError("Governance decision was made, but no audit record was written.")
    lines = [ln for ln in audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Audit file is empty after governance smoke.")

    steps = [
        "Message received.",
        "Governance checks ran.",
        "Decision: CONTAIN - clarification is required before continuation.",
        "Audit record written.",
    ]
    return {"ok": True, "steps": steps, "action": result.action.name}


def check_connectivity_test(cfg: SetupInput) -> CheckResult:
    try:
        run_connectivity_smoke(cfg)
    except ValueError as exc:
        return CheckResult("connectivity", SEVERITY_BLOCKING, STATUS_FAIL, str(exc), "Update provider settings and retry.")
    except Exception:
        return CheckResult(
            "connectivity",
            SEVERITY_BLOCKING,
            STATUS_FAIL,
            "Aurora-Lens could not reach the selected provider endpoint.",
            "Check endpoint URL, network access, and API key.",
        )
    return CheckResult("connectivity", SEVERITY_PASS, STATUS_PASS, "Provider connectivity test passed.")

