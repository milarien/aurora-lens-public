from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Literal
import os
import re

import yaml

from aurora_lens.sovereign.route_config import SovereignRouteConfig, parse_sovereign_route_config


Provider = Literal["openai", "anthropic", "mock", "local"]
ExtractionBackend = Literal["llm", "spacy"]

# Providers that never call remote HTTP from the proxy adapter layer.
_LOCAL_UPSTREAM_PROVIDERS: frozenset[str] = frozenset({"mock"})

# OpenAI-compatible HTTP upstream (openai vendor API or local/Ollama base URL).
_OPENAI_COMPAT_PROVIDERS: frozenset[str] = frozenset({"openai", "local"})

# Standard env var name for each provider's API key (fallback after AURORA_LENS_UPSTREAM_API_KEY).
_PROVIDER_KEY_ENVS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Provider contract metadata for validation errors and defaults.
_PROVIDER_CONTRACT: Dict[str, dict[str, Any]] = {
    "mock": {
        "requires_api_key": False,
        "requires_base_url": False,
        "default_model": "mock",
        "model_env": "AURORA_LENS_UPSTREAM_MODEL",
    },
    "local": {
        "requires_api_key": False,
        "requires_base_url": True,
        "default_model": None,
        "model_env": "AURORA_LENS_UPSTREAM_MODEL",
        "base_url_env": "AURORA_LENS_UPSTREAM_BASE_URL",
    },
    "openai": {
        "requires_api_key": True,
        "requires_base_url": False,
        "default_base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "model_env": "AURORA_LENS_UPSTREAM_MODEL",
        "base_url_env": "AURORA_LENS_UPSTREAM_BASE_URL",
        "api_key_envs": ("AURORA_LENS_UPSTREAM_API_KEY", "OPENAI_API_KEY"),
    },
    "anthropic": {
        "requires_api_key": True,
        "requires_base_url": False,
        "default_model": "claude-haiku-4-5-20251001",
        "model_env": "AURORA_LENS_UPSTREAM_MODEL",
        "api_key_envs": ("AURORA_LENS_UPSTREAM_API_KEY", "ANTHROPIC_API_KEY"),
    },
}


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")


def _env_expand(value: Any) -> Any:
    """
    Recursively expand ${VARNAME} and ${VARNAME:-default} in strings using os.environ.

    - ${VARNAME}: substituted with env value; if missing, token left intact.
    - ${VARNAME:-default}: substituted with env value; if missing, uses default.
    - Only expands on strings.
    """
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            default = match.group(2)  # None if no :- present
            if key in os.environ:
                return os.environ[key]
            if default is not None:
                return default
            return match.group(0)
        return _ENV_VAR_PATTERN.sub(repl, value)

    if isinstance(value, list):
        return [_env_expand(v) for v in value]

    if isinstance(value, dict):
        return {k: _env_expand(v) for k, v in value.items()}

    return value


def _has_unresolved_placeholder(value: str) -> bool:
    """True when a config string still contains an unexpanded ${...} token."""
    return bool(value and _ENV_VAR_PATTERN.search(value))


def _require_resolved_string(value: Any, ctx: str, *, required: bool = True) -> str:
    """Reject unresolved ${...} placeholders; optionally require non-empty."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ValueError(f"Missing required config: {ctx}")
        return ""
    if not isinstance(value, str):
        raise ValueError(f"Expected string for {ctx}, got {type(value)}")
    v = value.strip()
    if _has_unresolved_placeholder(v):
        raise ValueError(
            f"Unresolved config placeholder in {ctx}: {v!r}. "
            "Set the referenced environment variable(s) before starting the proxy."
        )
    if required and not v:
        raise ValueError(f"Empty string is not allowed for {ctx}")
    return v


def _optional_resolved_string(value: Any, ctx: str) -> Optional[str]:
    if value is None or (isinstance(value, str) and not str(value).strip()):
        return None
    v = _require_resolved_string(value, ctx, required=True)
    return v or None


def _validate_http_base_url(url: str, ctx: str) -> None:
    low = url.strip().lower()
    if not (low.startswith("http://") or low.startswith("https://")):
        raise ValueError(
            f"{ctx} must start with http:// or https:// (got {url!r}). "
            "Example for local OpenAI-compatible servers: http://localhost:11434/v1"
        )


def _provider_api_key_hint(provider: str) -> str:
    contract = _PROVIDER_CONTRACT.get(provider, {})
    envs = contract.get("api_key_envs") or (
        (_PROVIDER_KEY_ENVS.get(provider),) if provider in _PROVIDER_KEY_ENVS else ()
    )
    names = [e for e in envs if e]
    if names:
        return " or ".join(names)
    return "AURORA_LENS_UPSTREAM_API_KEY"


def _read_yaml(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML must be a mapping/object. Got: {type(data)}")
    return _env_expand(data)


def _require(mapping: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config field: {ctx}.{key}")
    return mapping[key]


def _as_str(value: Any, ctx: str) -> str:
    if value is None:
        raise ValueError(f"Missing required string: {ctx}")
    if not isinstance(value, str):
        raise ValueError(f"Expected string for {ctx}, got {type(value)}")
    v = value.strip()
    if not v:
        raise ValueError(f"Empty string is not allowed for {ctx}")
    return v


def _as_int(value: Any, ctx: str) -> int:
    if value is None:
        raise ValueError(f"Missing required int: {ctx}")
    if isinstance(value, bool):
        raise ValueError(f"Expected int for {ctx}, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Expected int for {ctx}, got {type(value)}: {value!r}")


def _as_provider(value: Any, ctx: str) -> Provider:
    s = _as_str(value, ctx).lower()
    allowed = tuple(_PROVIDER_CONTRACT.keys())
    if s not in allowed:
        raise ValueError(
            f"Invalid provider for {ctx}: {s!r} (allowed: {' | '.join(allowed)})"
        )
    return s  # type: ignore[return-value]


@dataclass(frozen=True)
class UpstreamConfig:
    provider: Provider
    api_key: str
    model: str
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None  # optional (adapters may have defaults)
    timeout_s: float = 120.0  # upstream request timeout (Ollama/local models often need more)


@dataclass(frozen=True)
class ListenConfig:
    host: str = "0.0.0.0"
    port: int = 8081


@dataclass(frozen=True)
class GovernanceConfig:
    default_policy: str = "strict"  # strict | moderate — deployment default; per-tenant override via AuthKeyConfig.policy
    mode: str = "public"          # public | enterprise | open — set per-deployment, never inferred from text
    audit_log: Optional[str] = "./audit.jsonl"
    policy_version: str = "1.0"
    audit_backend: str = "ledger"  # ledger (default when gov available) | jsonl (force BuiltinBridge)
    audit_signing_key: Optional[str] = None   # Phase D: HMAC key for JSONL; env AURORA_LENS_AUDIT_SIGNING_KEY
    audit_signing_keys: tuple[str, ...] = ()  # D4: additional historical keys for multi-key verification; env AURORA_LENS_AUDIT_SIGNING_KEYS (comma-separated)
    audit_log_max_mb: int = 100   # Phase D: rotate when exceeded; 0 = no rotation
    audit_checkpoint_interval: int = 0   # D3 Option 2: checkpoint every N entries; 0 = disabled
    max_revision_attempts: int = 1  # hard limit on revision loop (governor)
    include_operator_detail: bool = False  # Two-plane output: when False (default) flags/rationale/note stay in audit log only
    # When True, a request may opt into operator-plane JSON via X-Aurora-Operator-Detail (if global include_operator_detail is False).
    allow_operator_detail_via_header: bool = False
    enable_mock_hard_stop: bool = False  # Demo only: enable X-Aurora-Mock-Hard-Stop header. Never set in production.
    # Canonical Governor corridor context.
    # authority_class: deployment-level property — what kind of system is this endpoint.
    #   GP = General Purpose (default), DA = Domain Authorized, HS = Human Supervised.
    #   Set once per deployment; never inferred from request text.
    # user_class_header: optional HTTP header name from which per-request UserClass is read.
    #   E.g. "X-Aurora-User-Class". Empty string = always UserClass.GENERAL.
    authority_class: str = "GP"        # GP | DA | HS
    user_class_header: str = ""        # HTTP header name; empty = always GENERAL
    # Trusted runtime domain binding (server-side only). Never sourced from request headers.
    # Empty string keeps legacy behavior (domain from verified flags / default resolver fallback).
    default_domain: str = ""           # general | finance | legal | medical | ""
    # P.3 Anomaly alerts: 0 = disabled. E.g. 0.05 = 5% threshold.
    threshold_intervention_rate: float = 0.0
    threshold_extraction_failure_rate: float = 0.0
    anomaly_webhook_url: Optional[str] = None  # POST alert payload when threshold exceeded
    # Operator-supplied policy matrix path. When set, overrides the bundled policy_matrix.json.
    # The chosen matrix path is recorded in every forensic audit entry.
    policy_matrix_path: Optional[str] = None
    # When True, before opening the audit file at proxy startup, rename a non-empty file to
    # ``<name>.pre_restart.<UTC>`` so the new process starts a fresh chain at the same path.
    audit_log_fresh_chain_on_startup: bool = False
    # Forensic evidence vault (prompt/upstream stored sealed; ledger holds manifests only).
    evidence_capture_mode: str = "sealed"  # sealed | redacted | hash_only | plaintext_dev
    evidence_encryption_key: Optional[str] = None  # env AURORA_LENS_EVIDENCE_KEY


@dataclass(frozen=True)
class CorsConfig:
    enabled: bool = False
    allow_origins: tuple[str, ...] = ("*",)


@dataclass(frozen=True)
class ExtractionConfig:
    """Extraction backend for claim extraction from user input.

    spacy (default): Deterministic NLP extraction. Fast, no API cost, no LLM
        dependency for verification. Appropriate for production governance substrates.
    llm: Higher-fidelity claim extraction using the upstream adapter. Opt-in for
        deployments that accept the added latency and cost.
    """
    backend: str = "spacy"  # "spacy" (default) | "llm"
    spacy_model: str = "en_core_web_sm"  # spaCy model name
    history_window: int = 10  # max conversation turns to send to LLM


@dataclass(frozen=True)
class AuthKeyConfig:
    """Single API key with label and optional policy/domain override."""
    key: str
    label: str
    policy: Optional[str] = None  # strict | moderate | None = use deployment default
    domain: Optional[str] = None  # general | finance | legal | medical | None = use governance.default_domain


@dataclass(frozen=True)
class AuthConfig:
    """Phase B: Inbound authentication. enabled=False bypasses auth (dev mode)."""
    enabled: bool = False
    keys: tuple[AuthKeyConfig, ...] = ()


@dataclass(frozen=True)
class SessionConfig:
    """Phase C: Session persistence. memory=in-memory, redis=Redis-backed."""
    backend: str = "memory"       # memory | redis
    ttl_seconds: int = 3600
    redis_url: str = ""           # required if backend: redis
    lock_acquire_timeout_seconds: float = 10.0   # bounded wait; fail deterministically
    lock_lease_seconds: float = 240.0             # 2× upstream timeout; prevents deadlock


@dataclass(frozen=True)
class HardeningConfig:
    """Phase 5: Input validation and rate limits. 0 = disabled."""
    max_payload_bytes: int = 1_048_576       # 1 MiB
    max_messages: int = 100
    max_content_chars: int = 100_000        # per message
    rate_limit_global: int = 0               # requests/min, 0=disabled
    rate_limit_per_session: int = 0          # requests/min per session, 0=disabled
    rate_limit_per_ip: int = 0              # Phase B.4: requests/min per IP, 0=disabled
    trusted_proxy_ips: tuple[str, ...] = ()  # IPs allowed to set X-Forwarded-For
    stream_max_kb: int = 512                 # stream accumulator cap (KiB); 0 = no cap


@dataclass(frozen=True)
class ProxyConfig:
    upstream: UpstreamConfig
    listen: ListenConfig = ListenConfig()
    governance: GovernanceConfig = GovernanceConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    cors: CorsConfig = CorsConfig()
    hardening: HardeningConfig = HardeningConfig()
    auth: AuthConfig = AuthConfig()
    session: SessionConfig = SessionConfig()
    sovereign: SovereignRouteConfig = SovereignRouteConfig()

    @staticmethod
    def from_yaml(path: str | Path) -> "ProxyConfig":
        p = Path(path)
        data = _read_yaml(p)
        return ProxyConfig.from_mapping(data)

    @staticmethod
    def from_env() -> "ProxyConfig":
        """Build a ProxyConfig entirely from environment variables. No YAML required.

        Provider is auto-detected from available API key env vars when
        AURORA_LENS_UPSTREAM_PROVIDER is not set:
          ANTHROPIC_API_KEY present → anthropic
          OPENAI_API_KEY present    → openai
          neither                   → mock (no remote HTTP; governance-only local dev)

        Model defaults per provider when AURORA_LENS_UPSTREAM_MODEL is not set:
          mock      → mock
          openai    → gpt-4o-mini
          anthropic → claude-haiku-4-5-20251001
          local     → must be set explicitly
        """
        provider = os.environ.get("AURORA_LENS_UPSTREAM_PROVIDER", "").strip().lower()
        if not provider:
            if os.environ.get("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif os.environ.get("OPENAI_API_KEY") or os.environ.get("AURORA_LENS_UPSTREAM_API_KEY"):
                provider = "openai"
            else:
                provider = "mock"

        contract = _PROVIDER_CONTRACT.get(provider, {})
        model = os.environ.get("AURORA_LENS_UPSTREAM_MODEL", "").strip()
        if not model:
            default_model = contract.get("default_model")
            if not default_model and provider == "local":
                model = "local-model"
            else:
                model = str(default_model or "mock")

        base: dict = {
            "upstream": {
                "provider": provider,
                "api_key": "",   # apply_env_overrides fills this in from provider key env var
                "model": model,
            },
            "governance": {
                "audit_log": None,  # disable file audit by default; AURORA_LENS_GOV_AUDIT_LOG overrides
            },
        }
        return ProxyConfig.from_mapping(base)

    @staticmethod
    def from_mapping(data: Dict[str, Any]) -> "ProxyConfig":
        upstream_map = data.get("upstream") or {}
        if not isinstance(upstream_map, dict):
            raise ValueError("Config field 'upstream' must be a mapping/object.")

        listen_map = data.get("listen") or {}
        if not isinstance(listen_map, dict):
            raise ValueError("Config field 'listen' must be a mapping/object.")

        gov_map = data.get("governance") or {}
        if not isinstance(gov_map, dict):
            raise ValueError("Config field 'governance' must be a mapping/object.")

        cors_map = data.get("cors") or {}
        if not isinstance(cors_map, dict):
            cors_map = {}

        extraction_map = data.get("extraction") or {}
        if not isinstance(extraction_map, dict):
            extraction_map = {}
        ext_backend = (extraction_map.get("backend") or "spacy").strip().lower()
        if ext_backend not in ("llm", "spacy"):
            ext_backend = "spacy"
        spacy_model = str(extraction_map.get("spacy_model", "en_core_web_sm")).strip() or "en_core_web_sm"
        history_window = int(extraction_map.get("history_window", 10))
        extraction = ExtractionConfig(
            backend=ext_backend,
            spacy_model=spacy_model,
            history_window=max(1, min(100, history_window)),
        )

        timeout_val = upstream_map.get("timeout_s", 120)
        if isinstance(timeout_val, (int, float)) and timeout_val > 0:
            timeout_s = float(timeout_val)
        else:
            timeout_s = 120.0

        upstream = UpstreamConfig(
            provider=_as_provider(_require(upstream_map, "provider", "upstream"), "upstream.provider"),
            api_key=str(upstream_map.get("api_key", "")).strip(),
            api_key_env=_optional_resolved_string(upstream_map.get("api_key_env"), "upstream.api_key_env"),
            model=_as_str(_require(upstream_map, "model", "upstream"), "upstream.model"),
            base_url=_as_str(upstream_map.get("base_url"), "upstream.base_url") if upstream_map.get("base_url") else None,
            timeout_s=timeout_s,
        )

        listen = ListenConfig(
            host=_as_str(listen_map.get("host", "0.0.0.0"), "listen.host"),
            port=_as_int(listen_map.get("port", 8081), "listen.port"),
        )

        gov_mode_raw = (gov_map.get("mode") or "public").strip().lower()
        gov_mode = gov_mode_raw if gov_mode_raw in ("public", "enterprise", "open") else "public"

        audit_signing = (gov_map.get("audit_signing_key") or "").strip() or None
        _raw_extra_keys = gov_map.get("audit_signing_keys", [])
        if isinstance(_raw_extra_keys, str):
            audit_extra_keys = tuple(k.strip() for k in _raw_extra_keys.split(",") if k.strip())
        elif isinstance(_raw_extra_keys, list):
            audit_extra_keys = tuple(str(k).strip() for k in _raw_extra_keys if str(k).strip())
        else:
            audit_extra_keys = ()
        audit_max_mb = int(gov_map.get("audit_log_max_mb", 100)) if gov_map.get("audit_log_max_mb") is not None else 100
        audit_checkpoint = int(gov_map.get("audit_checkpoint_interval", 0)) if gov_map.get("audit_checkpoint_interval") is not None else 0
        audit_backend_raw = (gov_map.get("audit_backend") or "ledger").strip().lower()
        audit_backend = audit_backend_raw if audit_backend_raw in ("ledger", "jsonl") else "ledger"
        max_revision = int(gov_map.get("max_revision_attempts", 1))
        authority_class_raw = str(gov_map.get("authority_class", "GP")).strip().upper() or "GP"
        authority_class = authority_class_raw if authority_class_raw in ("GP", "DA", "HS") else "GP"
        user_class_header = str(gov_map.get("user_class_header", "")).strip()
        default_domain_raw = str(gov_map.get("default_domain", "")).strip().lower()
        default_domain = (
            default_domain_raw
            if default_domain_raw in (
                "", "general", "finance", "legal", "medical",
                "education", "workforce", "enterprise",
            )
            else ""
        )
        include_op_detail = bool(gov_map.get("include_operator_detail", False))
        _allow_hdr_raw = gov_map.get("allow_operator_detail_via_header", False)
        if isinstance(_allow_hdr_raw, str):
            allow_op_detail_hdr = _allow_hdr_raw.strip().lower() in ("1", "true", "yes", "on")
        else:
            allow_op_detail_hdr = bool(_allow_hdr_raw)
        enable_mock = bool(
            gov_map.get("enable_mock_hard_stop", False)
            or os.environ.get("AURORA_LENS_ENABLE_MOCK_HARD_STOP", "").strip().lower() in ("1", "true", "yes")
        )
        thr_interv = float(gov_map.get("threshold_intervention_rate", 0) or 0)
        thr_extr = float(gov_map.get("threshold_extraction_failure_rate", 0) or 0)
        anomaly_webhook = (gov_map.get("anomaly_webhook_url") or "").strip() or None
        policy_matrix_path_raw = (gov_map.get("policy_matrix_path") or "").strip() or None
        _fresh_raw = gov_map.get("audit_log_fresh_chain_on_startup", False)
        if isinstance(_fresh_raw, str):
            audit_fresh_chain = _fresh_raw.strip().lower() in ("1", "true", "yes", "on")
        else:
            audit_fresh_chain = bool(_fresh_raw)
        evidence_capture_mode = _as_str(
            gov_map.get("evidence_capture_mode", "sealed"),
            "governance.evidence_capture_mode",
        )
        evidence_encryption_key = (
            (gov_map.get("evidence_encryption_key") or "").strip()
            or os.environ.get("AURORA_LENS_EVIDENCE_KEY", "").strip()
            or None
        )
        governance = GovernanceConfig(
            default_policy=_as_str(gov_map.get("default_policy") or gov_map.get("policy", "strict"), "governance.default_policy"),
            mode=gov_mode,
            audit_log=_as_str(gov_map.get("audit_log", "./audit.jsonl"), "governance.audit_log")
            if gov_map.get("audit_log", "./audit.jsonl") is not None
            else None,
            policy_version=_as_str(gov_map.get("policy_version", "1.0"), "governance.policy_version"),
            policy_matrix_path=policy_matrix_path_raw,
            audit_backend=audit_backend,
            audit_signing_key=audit_signing,
            audit_signing_keys=audit_extra_keys,
            audit_log_max_mb=max(0, audit_max_mb),
            audit_checkpoint_interval=max(0, audit_checkpoint),
            max_revision_attempts=max(1, min(10, max_revision)),
            include_operator_detail=include_op_detail,
            allow_operator_detail_via_header=allow_op_detail_hdr,
            enable_mock_hard_stop=enable_mock,
            threshold_intervention_rate=max(0.0, min(1.0, thr_interv)),
            threshold_extraction_failure_rate=max(0.0, min(1.0, thr_extr)),
            anomaly_webhook_url=anomaly_webhook,
            authority_class=authority_class,
            user_class_header=user_class_header,
            default_domain=default_domain,
            audit_log_fresh_chain_on_startup=audit_fresh_chain,
            evidence_capture_mode=evidence_capture_mode,
            evidence_encryption_key=evidence_encryption_key,
        )

        cors_enabled = bool(cors_map.get("enabled", False))
        cors_origins_raw = cors_map.get("allow_origins", ["*"])
        if isinstance(cors_origins_raw, str):
            cors_origins = tuple(o.strip() for o in cors_origins_raw.split(",") if o.strip()) or ("*",)
        elif isinstance(cors_origins_raw, list):
            cors_origins = tuple(str(o).strip() for o in cors_origins_raw if str(o).strip()) or ("*",)
        else:
            cors_origins = ("*",)
        cors = CorsConfig(enabled=cors_enabled, allow_origins=cors_origins)

        hardening_map = data.get("hardening") or {}
        if not isinstance(hardening_map, dict):
            hardening_map = {}
        trusted_proxies_raw = hardening_map.get("trusted_proxy_ips", [])
        if isinstance(trusted_proxies_raw, str):
            trusted_proxies = tuple(p.strip() for p in trusted_proxies_raw.split(",") if p.strip())
        elif isinstance(trusted_proxies_raw, list):
            trusted_proxies = tuple(str(p).strip() for p in trusted_proxies_raw if str(p).strip())
        else:
            trusted_proxies = ()
        stream_max_kb = int(hardening_map.get("stream_max_kb", 512))
        hardening = HardeningConfig(
            max_payload_bytes=int(hardening_map.get("max_payload_bytes", 1_048_576)),
            max_messages=int(hardening_map.get("max_messages", 100)),
            max_content_chars=int(hardening_map.get("max_content_chars", 100_000)),
            rate_limit_global=int(hardening_map.get("rate_limit_global", 0)),
            rate_limit_per_session=int(hardening_map.get("rate_limit_per_session", 0)),
            rate_limit_per_ip=int(hardening_map.get("rate_limit_per_ip", 0)),
            trusted_proxy_ips=trusted_proxies,
            stream_max_kb=max(0, stream_max_kb),
        )

        auth_map = data.get("auth") or {}
        if not isinstance(auth_map, dict):
            auth_map = {}
        auth_enabled = bool(auth_map.get("enabled", False))
        auth_keys_raw = auth_map.get("keys") or []
        auth_keys: list[AuthKeyConfig] = []
        if isinstance(auth_keys_raw, list):
            for item in auth_keys_raw:
                if isinstance(item, dict):
                    k = str(item.get("key", "")).strip()
                    label = str(item.get("label", "unnamed")).strip() or "unnamed"
                    policy = (item.get("policy") or "").strip().lower() or None
                    if policy and policy not in ("strict", "moderate"):
                        policy = None
                    domain = (item.get("domain") or "").strip().lower() or None
                    if domain and domain not in (
                        "general", "finance", "legal", "medical",
                        "education", "workforce", "enterprise",
                    ):
                        domain = None
                    if k:
                        auth_keys.append(
                            AuthKeyConfig(key=k, label=label, policy=policy, domain=domain)
                        )
        auth = AuthConfig(enabled=auth_enabled, keys=tuple(auth_keys))

        session_map = data.get("session") or {}
        if not isinstance(session_map, dict):
            session_map = {}
        session_backend = (session_map.get("backend") or "memory").strip().lower()
        if session_backend not in ("memory", "redis"):
            session_backend = "memory"
        session_ttl = int(session_map.get("ttl_seconds", 3600))
        session_redis_url = str(session_map.get("redis_url", "")).strip()
        lock_acquire = float(session_map.get("lock_acquire_timeout_seconds", 10))
        lock_lease = float(session_map.get("lock_lease_seconds", 240))
        session = SessionConfig(
            backend=session_backend,
            ttl_seconds=max(60, session_ttl),
            redis_url=session_redis_url,
            lock_acquire_timeout_seconds=max(1.0, lock_acquire),
            lock_lease_seconds=max(30.0, lock_lease),
        )

        sovereign_map = data.get("sovereign") or {}
        if not isinstance(sovereign_map, dict):
            sovereign_map = {}
        sovereign = parse_sovereign_route_config(sovereign_map)

        cfg = ProxyConfig(
            upstream=upstream,
            listen=listen,
            governance=governance,
            extraction=extraction,
            cors=cors,
            hardening=hardening,
            auth=auth,
            session=session,
            sovereign=sovereign,
        )
        return cfg.apply_env_overrides()

    def validate(self) -> None:
        """Check that the resolved config is usable. Call after env overrides."""
        up = self.upstream
        provider = up.provider
        contract = _PROVIDER_CONTRACT[provider]

        model = _require_resolved_string(up.model, "upstream.model")
        api_key = (up.api_key or "").strip()
        api_key_env = (up.api_key_env or "").strip()
        if api_key and _has_unresolved_placeholder(api_key):
            raise ValueError(
                f"Unresolved config placeholder in upstream.api_key: {api_key!r}. "
                f"Set {_provider_api_key_hint(provider)} before starting the proxy."
            )
        if api_key and not api_key_env:
            raise ValueError(
                "upstream.api_key must not contain secrets. Use upstream.api_key_env with "
                "an environment variable name, for example OPENAI_API_KEY."
            )
        if api_key_env and _has_unresolved_placeholder(api_key_env):
            raise ValueError(
                f"Unresolved config placeholder in upstream.api_key_env: {api_key_env!r}. "
                "Set a literal environment variable name, for example OPENAI_API_KEY."
            )
        if api_key_env:
            env_val = os.environ.get(api_key_env, "").strip()
            if not env_val:
                raise ValueError(
                    f"Credential environment variable {api_key_env} is missing or empty."
                )
            api_key = env_val

        base_url_raw = (up.base_url or "").strip() or None
        if base_url_raw and _has_unresolved_placeholder(base_url_raw):
            hint = contract.get("base_url_env") or "AURORA_LENS_UPSTREAM_BASE_URL"
            raise ValueError(
                f"Unresolved config placeholder in upstream.base_url: {base_url_raw!r}. "
                f"Set {hint} before starting the proxy."
            )

        if provider in _LOCAL_UPSTREAM_PROVIDERS:
            return

        if provider == "local":
            if not base_url_raw:
                hint = contract.get("base_url_env") or "AURORA_LENS_UPSTREAM_BASE_URL"
                raise ValueError(
                    "upstream.provider=local requires a resolved upstream.base_url "
                    f"(set {hint}, e.g. http://localhost:11434/v1)."
                )
            _validate_http_base_url(base_url_raw, "upstream.base_url")
            _require_resolved_string(model, "upstream.model")
            return

        if contract.get("requires_api_key") and not api_key_env and not api_key:
            raise ValueError(
                f"upstream.provider={provider} requires upstream.api_key_env "
                "(for example OPENAI_API_KEY)."
            )

        if contract.get("requires_api_key") and not api_key:
            raise ValueError(
                f"upstream.provider={provider} requires an API key — set "
                f"upstream.api_key_env (for example OPENAI_API_KEY) or "
                f"{_provider_api_key_hint(provider)}."
            )

        if base_url_raw:
            _validate_http_base_url(base_url_raw, "upstream.base_url")

    def apply_env_overrides(self) -> "ProxyConfig":
        """
        Environment overrides (optional) — intended for deployment.

        Supported:
          AURORA_LENS_UPSTREAM_PROVIDER
          AURORA_LENS_UPSTREAM_API_KEY
          AURORA_LENS_UPSTREAM_MODEL
          AURORA_LENS_UPSTREAM_BASE_URL

          AURORA_LENS_LISTEN_HOST
          AURORA_LENS_LISTEN_PORT
          PORT (PaaS convention; used when AURORA_LENS_LISTEN_PORT is unset)

          AURORA_LENS_GOV_POLICY
          AURORA_LENS_GOV_AUDIT_LOG
          AURORA_LENS_AUDIT_LOG_FRESH_CHAIN_ON_STARTUP (1/true = rename non-empty audit file at proxy startup)

          AURORA_LENS_CORS_ALLOW_ORIGINS — comma-separated extra origins appended to YAML /
          AURORA_LENS_CORS_ORIGINS (does not replace them)
        """
        up = self.upstream
        ln = self.listen
        gv = self.governance

        provider_raw = os.environ.get("AURORA_LENS_UPSTREAM_PROVIDER", up.provider)
        provider_norm = _as_provider(provider_raw, "AURORA_LENS_UPSTREAM_PROVIDER")

        api_key = os.environ.get("AURORA_LENS_UPSTREAM_API_KEY", up.api_key)
        api_key_env = (up.api_key_env or "").strip()
        if api_key_env:
            api_key = os.environ.get(api_key_env, api_key)
        model = os.environ.get("AURORA_LENS_UPSTREAM_MODEL", up.model)
        base_url = os.environ.get("AURORA_LENS_UPSTREAM_BASE_URL", up.base_url or "")

        # Provider-specific env var fallback (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY)
        if provider_norm not in _LOCAL_UPSTREAM_PROVIDERS:
            if not str(api_key).strip() or _has_unresolved_placeholder(str(api_key)):
                prov_env = _PROVIDER_KEY_ENVS.get(provider_norm)
                if prov_env:
                    api_key = os.environ.get(prov_env, api_key)

        host = os.environ.get("AURORA_LENS_LISTEN_HOST", ln.host)
        # Prefer explicit app var; else honor standard PORT (Railway, Heroku, Render, etc.).
        _listen_port_env = (os.environ.get("AURORA_LENS_LISTEN_PORT") or "").strip()
        _paas_port = (os.environ.get("PORT") or "").strip()
        if _listen_port_env:
            port_raw = _listen_port_env
        elif _paas_port:
            port_raw = _paas_port
        else:
            port_raw = str(ln.port)

        policy = (os.environ.get("AURORA_LENS_GOV_DEFAULT_POLICY")
                  or os.environ.get("AURORA_LENS_GOV_POLICY")
                  or gv.default_policy)
        default_domain_env = os.environ.get("AURORA_LENS_GOV_DEFAULT_DOMAIN", gv.default_domain).strip().lower()
        default_domain = (
            default_domain_env
            if default_domain_env in ("", "general", "finance", "legal", "medical")
            else gv.default_domain
        )
        gov_mode_env = os.environ.get("AURORA_LENS_GOV_MODE", gv.mode).strip().lower()
        gov_mode = gov_mode_env if gov_mode_env in ("public", "enterprise", "open") else gv.mode
        audit_log = os.environ.get("AURORA_LENS_GOV_AUDIT_LOG", gv.audit_log or "")
        policy_version = os.environ.get("AURORA_LENS_GOV_POLICY_VERSION", gv.policy_version)
        audit_signing = os.environ.get("AURORA_LENS_AUDIT_SIGNING_KEY", gv.audit_signing_key or "").strip() or None
        _extra_keys_env = os.environ.get("AURORA_LENS_AUDIT_SIGNING_KEYS", "").strip()
        audit_extra_keys = tuple(k.strip() for k in _extra_keys_env.split(",") if k.strip()) if _extra_keys_env else gv.audit_signing_keys
        audit_max_mb = int(os.environ.get("AURORA_LENS_AUDIT_LOG_MAX_MB", gv.audit_log_max_mb))
        audit_checkpoint = int(os.environ.get("AURORA_LENS_AUDIT_CHECKPOINT_INTERVAL", gv.audit_checkpoint_interval))
        audit_backend_env = os.environ.get("AURORA_LENS_AUDIT_BACKEND", gv.audit_backend).strip().lower()
        audit_backend = audit_backend_env if audit_backend_env in ("ledger", "jsonl") else gv.audit_backend
        evidence_capture_mode_env = os.environ.get(
            "AURORA_LENS_EVIDENCE_CAPTURE_MODE", gv.evidence_capture_mode
        ).strip().lower()
        evidence_capture_mode = (
            evidence_capture_mode_env
            if evidence_capture_mode_env in ("sealed", "redacted", "hash_only", "plaintext_dev")
            else gv.evidence_capture_mode
        )
        # Carry through the already-resolved YAML/env value (config.py:474-476) rather than
        # defaulting to None here — a bare env-override pass must never discard a configured
        # evidence-vault key.
        evidence_encryption_key = (
            os.environ.get("AURORA_LENS_EVIDENCE_KEY", "").strip() or gv.evidence_encryption_key
        )

        timeout_s = up.timeout_s
        if "AURORA_LENS_UPSTREAM_TIMEOUT_S" in os.environ:
            try:
                timeout_s = float(os.environ["AURORA_LENS_UPSTREAM_TIMEOUT_S"])
            except (ValueError, TypeError):
                pass

        new_up = UpstreamConfig(
            provider=provider_norm,
            api_key=str(api_key or "").strip(),
            api_key_env=api_key_env or None,
            model=_require_resolved_string(model, "AURORA_LENS_UPSTREAM_MODEL"),
            base_url=_optional_resolved_string(base_url, "AURORA_LENS_UPSTREAM_BASE_URL"),
            timeout_s=timeout_s,
        )

        new_ln = ListenConfig(
            host=_as_str(host, "AURORA_LENS_LISTEN_HOST"),
            port=_as_int(port_raw, "listen port (AURORA_LENS_LISTEN_PORT or PORT)"),
        )

        max_revision = int(os.environ.get("AURORA_LENS_MAX_REVISION_ATTEMPTS", gv.max_revision_attempts))
        include_op_detail_env = os.environ.get("AURORA_LENS_INCLUDE_OPERATOR_DETAIL", "").strip().lower()
        include_op_detail = (
            include_op_detail_env in ("1", "true", "yes", "on")
            if include_op_detail_env
            else gv.include_operator_detail
        )
        allow_op_hdr_env = os.environ.get("AURORA_LENS_ALLOW_OPERATOR_DETAIL_HEADER", "").strip().lower()
        allow_op_detail_hdr = (
            allow_op_hdr_env in ("1", "true", "yes", "on")
            if allow_op_hdr_env
            else gv.allow_operator_detail_via_header
        )
        enable_mock = gv.enable_mock_hard_stop
        thr_interv = float(os.environ.get("AURORA_LENS_THRESHOLD_INTERVENTION_RATE", gv.threshold_intervention_rate))
        thr_extr = float(os.environ.get("AURORA_LENS_THRESHOLD_EXTRACTION_FAILURE_RATE", gv.threshold_extraction_failure_rate))
        anomaly_webhook = os.environ.get("AURORA_LENS_ANOMALY_WEBHOOK_URL", gv.anomaly_webhook_url or "").strip() or None
        fresh_chain_env = os.environ.get("AURORA_LENS_AUDIT_LOG_FRESH_CHAIN_ON_STARTUP", "").strip().lower()
        audit_fresh_chain = (
            fresh_chain_env in ("1", "true", "yes", "on")
            if fresh_chain_env
            else gv.audit_log_fresh_chain_on_startup
        )
        new_gv = GovernanceConfig(
            default_policy=_as_str(policy, "AURORA_LENS_GOV_POLICY"),
            mode=gov_mode,
            audit_log=_as_str(audit_log, "AURORA_LENS_GOV_AUDIT_LOG") if audit_log.strip() else None,
            policy_version=_as_str(policy_version, "AURORA_LENS_GOV_POLICY_VERSION"),
            audit_backend=audit_backend,
            audit_signing_key=audit_signing,
            audit_signing_keys=audit_extra_keys,
            audit_log_max_mb=max(0, audit_max_mb),
            audit_checkpoint_interval=max(0, audit_checkpoint),
            max_revision_attempts=max(1, min(10, max_revision)),
            include_operator_detail=include_op_detail,
            allow_operator_detail_via_header=allow_op_detail_hdr,
            enable_mock_hard_stop=enable_mock,
            threshold_intervention_rate=max(0.0, min(1.0, thr_interv)),
            threshold_extraction_failure_rate=max(0.0, min(1.0, thr_extr)),
            anomaly_webhook_url=anomaly_webhook,
            authority_class=gv.authority_class,
            user_class_header=gv.user_class_header,
            default_domain=default_domain,
            policy_matrix_path=gv.policy_matrix_path,
            audit_log_fresh_chain_on_startup=audit_fresh_chain,
            evidence_capture_mode=evidence_capture_mode,
            evidence_encryption_key=evidence_encryption_key,
        )

        cors_cfg = self.cors
        cors_enabled_raw = os.environ.get("AURORA_LENS_CORS_ENABLED", str(cors_cfg.enabled)).strip().lower()
        cors_enabled = cors_enabled_raw in ("1", "true", "yes", "on")
        cors_origins_raw = os.environ.get("AURORA_LENS_CORS_ORIGINS", "")
        if cors_origins_raw.strip():
            cors_origins = tuple(o.strip() for o in cors_origins_raw.split(",") if o.strip()) or ("*",)
        else:
            cors_origins = cors_cfg.allow_origins
        # Extra browser origins (e.g. LibreChat on localhost): append; never replaces YAML / CORS_ORIGINS.
        _cors_extra = os.environ.get("AURORA_LENS_CORS_ALLOW_ORIGINS", "").strip()
        if _cors_extra:
            _extra = tuple(o.strip() for o in _cors_extra.split(",") if o.strip())
            if _extra:
                seen: set[str] = set()
                merged: list[str] = []
                for o in cors_origins + _extra:
                    if o not in seen:
                        seen.add(o)
                        merged.append(o)
                cors_origins = tuple(merged)
        new_cors = CorsConfig(enabled=cors_enabled, allow_origins=cors_origins)

        ext = self.extraction
        ext_backend = os.environ.get("AURORA_LENS_EXTRACTION_BACKEND", ext.backend).strip().lower()
        if ext_backend not in ("llm", "spacy"):
            ext_backend = ext.backend
        spacy_model = os.environ.get("AURORA_LENS_SPACY_MODEL", ext.spacy_model).strip() or "en_core_web_sm"
        history_window = int(os.environ.get("AURORA_LENS_HISTORY_WINDOW", ext.history_window))
        new_ext = ExtractionConfig(
            backend=ext_backend,
            spacy_model=spacy_model,
            history_window=max(1, min(100, history_window)),
        )

        h = self.hardening
        h_max_bytes = int(os.environ.get("AURORA_LENS_MAX_PAYLOAD_BYTES", h.max_payload_bytes))
        h_max_msg = int(os.environ.get("AURORA_LENS_MAX_MESSAGES", h.max_messages))
        h_max_chars = int(os.environ.get("AURORA_LENS_MAX_CONTENT_CHARS", h.max_content_chars))
        h_rl_global = int(os.environ.get("AURORA_LENS_RATE_LIMIT_GLOBAL", h.rate_limit_global))
        h_rl_session = int(os.environ.get("AURORA_LENS_RATE_LIMIT_PER_SESSION", h.rate_limit_per_session))
        h_rl_ip = int(os.environ.get("AURORA_LENS_RATE_LIMIT_PER_IP", h.rate_limit_per_ip))
        trusted_proxies_env = os.environ.get("AURORA_LENS_TRUSTED_PROXY_IPS", "")
        trusted_proxies = tuple(p.strip() for p in trusted_proxies_env.split(",") if p.strip()) if trusted_proxies_env.strip() else h.trusted_proxy_ips
        h_stream_kb = int(os.environ.get("AURORA_LENS_STREAM_MAX_KB", h.stream_max_kb))
        new_h = HardeningConfig(
            max_payload_bytes=max(0, h_max_bytes),
            max_messages=max(0, h_max_msg),
            max_content_chars=max(0, h_max_chars),
            rate_limit_global=max(0, h_rl_global),
            rate_limit_per_session=max(0, h_rl_session),
            rate_limit_per_ip=max(0, h_rl_ip),
            trusted_proxy_ips=trusted_proxies,
            stream_max_kb=max(0, h_stream_kb),
        )

        auth_cfg = self.auth
        auth_enabled_raw = os.environ.get("AURORA_LENS_AUTH_ENABLED", str(auth_cfg.enabled)).strip().lower()
        auth_enabled = auth_enabled_raw in ("1", "true", "yes", "on")
        new_auth = AuthConfig(enabled=auth_enabled, keys=auth_cfg.keys)

        sess_cfg = self.session
        sess_backend = os.environ.get("AURORA_LENS_SESSION_BACKEND", sess_cfg.backend).strip().lower()
        if sess_backend not in ("memory", "redis"):
            sess_backend = sess_cfg.backend
        sess_ttl = int(os.environ.get("AURORA_LENS_SESSION_TTL_SECONDS", sess_cfg.ttl_seconds))
        sess_redis = os.environ.get("AURORA_LENS_REDIS_URL", sess_cfg.redis_url or "").strip()
        lock_acquire = float(os.environ.get("AURORA_LENS_LOCK_ACQUIRE_TIMEOUT", sess_cfg.lock_acquire_timeout_seconds))
        lock_lease = float(os.environ.get("AURORA_LENS_LOCK_LEASE_SECONDS", sess_cfg.lock_lease_seconds))
        new_session = SessionConfig(
            backend=sess_backend,
            ttl_seconds=max(60, sess_ttl),
            redis_url=sess_redis,
            lock_acquire_timeout_seconds=max(1.0, lock_acquire),
            lock_lease_seconds=max(30.0, lock_lease),
        )

        sov = self.sovereign

        def _env_bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name, "").strip().lower()
            if not raw:
                return default
            return raw in ("1", "true", "yes", "on")

        sov_enabled = _env_bool("AURORA_LENS_SOVEREIGN_ENABLED", sov.enabled)
        sov_enforce = _env_bool("AURORA_LENS_SOVEREIGN_ENFORCE", sov.enforce_provider_route)
        sov_primary = os.environ.get("AURORA_LENS_SOVEREIGN_PRIMARY_PROVIDER_ID", sov.primary_provider_id).strip()
        sov_primary_state = os.environ.get("AURORA_LENS_SOVEREIGN_PRIMARY_STATE", sov.primary_state).strip()
        sov_alternate = os.environ.get("AURORA_LENS_SOVEREIGN_ALTERNATE_PROVIDER_ID", sov.alternate_provider_id or "").strip()
        stale_days_raw = os.environ.get(
            "AURORA_LENS_SOVEREIGN_VALIDATION_STALE_DAYS",
            str(sov.validation_stale_threshold_days),
        )
        try:
            validation_stale_threshold_days = max(0, int(stale_days_raw))
        except (TypeError, ValueError):
            validation_stale_threshold_days = sov.validation_stale_threshold_days
        new_sov = SovereignRouteConfig(
            enabled=sov_enabled,
            enforce_provider_route=sov_enforce,
            primary_provider_id=sov_primary or sov.primary_provider_id,
            primary_state=sov_primary_state or sov.primary_state,
            alternate_provider_id=sov_alternate or sov.alternate_provider_id,
            default_policy=sov.default_policy,
            domain_policies=sov.domain_policies,
            upstream_provider_ids=sov.upstream_provider_ids,
            profiles=sov.profiles,
            validation_stale_threshold_days=validation_stale_threshold_days,
        )

        return ProxyConfig(
            upstream=new_up,
            listen=new_ln,
            governance=new_gv,
            extraction=new_ext,
            cors=new_cors,
            hardening=new_h,
            auth=new_auth,
            session=new_session,
            sovereign=new_sov,
        )

def load_config(path: str | Path) -> ProxyConfig:
    """
    Backwards-compatible entrypoint expected by proxy/__main__.py.
    """
    cfg = ProxyConfig.from_yaml(path)
    cfg.validate()
    return cfg
