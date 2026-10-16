"""Capture requirements/readiness, using synthetic fixtures and no upstream IO."""

import json
import logging

import pytest
from starlette.testclient import TestClient

from aurora_lens.adapters.base import LLMAdapter
from aurora_lens.govern.evidence_vault import (
    CaptureFailureReason,
    CaptureMode,
    EvidenceVaultConfigurationError,
    LocalSealedEvidenceVault,
    create_local_evidence_vault,
    evidence_capture_state,
    evidence_storage_available,
)
from aurora_lens.proxy.config import ProxyConfig


KEY = "synthetic-readiness-secret"


class UnusedAdapter(LLMAdapter):
    async def generate(self, messages, **kwargs):
        raise AssertionError("readiness must not call an upstream model")


@pytest.fixture
def proxy_config(monkeypatch, tmp_path):
    monkeypatch.delenv("AURORA_LENS_EVIDENCE_KEY", raising=False)
    monkeypatch.delenv("RAILWAY_ENVIRONMENT_NAME", raising=False)
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)
    monkeypatch.setenv("SYNTHETIC_UPSTREAM_KEY", "synthetic-only")
    monkeypatch.setattr(
        "aurora_lens.proxy.app._build_provider_adapters",
        lambda cfg: (UnusedAdapter(), UnusedAdapter()),
    )

    def config(*, mode="sealed", key=KEY, audit=True):
        return ProxyConfig.from_mapping({
            "upstream": {
                "provider": "openai", "model": "synthetic",
                "api_key_env": "SYNTHETIC_UPSTREAM_KEY",
            },
            "extraction": {"backend": "llm"},
            "governance": {
                "audit_log": str(tmp_path / "audit.jsonl") if audit else None,
                "evidence_capture_mode": mode,
                "evidence_encryption_key": key,
            },
        })

    return config


@pytest.mark.parametrize("key", [None, "", "   "])
def test_state_distinguishes_missing_key_from_explicit_hash_only(tmp_path, monkeypatch, key):
    monkeypatch.delenv("AURORA_LENS_EVIDENCE_KEY", raising=False)
    path = tmp_path / "audit.jsonl"
    vault = create_local_evidence_vault(path, capture_mode="hash_only")
    missing = evidence_capture_state(
        path, capture_mode="sealed", encryption_key=key, vault=vault,
        storage_available=evidence_storage_available(path, vault),
    )
    assert missing.requested_mode == "sealed"
    assert missing.effective_mode == "hash_only"
    assert missing.failure_reason == CaptureFailureReason.MISSING_KEY
    assert missing.required_sealed and not missing.ready
    explicit = evidence_capture_state(
        path, capture_mode="hash_only", encryption_key=key, vault=vault,
        storage_available=True,
    )
    assert explicit.ready and not explicit.required_sealed
    assert explicit.failure_reason is None


def test_disabled_state_is_ready_without_a_key(monkeypatch):
    monkeypatch.delenv("AURORA_LENS_EVIDENCE_KEY", raising=False)
    state = evidence_capture_state(None, capture_mode="sealed")
    assert not state.enabled and not state.required_sealed and state.ready
    assert state.effective_mode is None


def test_state_represents_unavailable_storage_and_sealing(tmp_path):
    path = tmp_path / "audit.jsonl"
    state = evidence_capture_state(path, capture_mode="sealed", encryption_key=KEY)
    assert state.failure_reason == CaptureFailureReason.STORAGE_UNAVAILABLE
    vault = create_local_evidence_vault(path, capture_mode="hash_only", encryption_key=KEY)
    state = evidence_capture_state(
        path, capture_mode="sealed", encryption_key=KEY, vault=vault,
        storage_available=True,
    )
    assert state.failure_reason == CaptureFailureReason.SEALED_UNAVAILABLE


def test_startup_and_health_expose_only_safe_capture_state(proxy_config, caplog):
    from aurora_lens.proxy.app import create_app

    with caplog.at_level(logging.INFO):
        app = create_app(proxy_config())
    state = app.state.evidence_capture_state
    assert state == {
        "enabled": True, "requested_mode": "sealed", "effective_mode": "sealed",
        "key_available": True, "vault_available": True, "required_sealed": True,
        "ready": True, "failure_reason": None,
    }
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["evidence_capture"] == state
    records = [r for r in caplog.records if r.message == "evidence_capture_state"]
    assert len(records) == 1 and records[0].evidence_capture == state
    assert KEY not in json.dumps(state) + caplog.text
    assert "key_ref" not in state and "key_fingerprint" not in state


def test_health_detects_lost_blob_storage_without_exception_text(proxy_config, monkeypatch, caplog):
    from aurora_lens.govern.evidence_vault import LocalSealedEvidenceVault
    from aurora_lens.proxy.app import create_app

    app = create_app(proxy_config())

    def unavailable(self):
        raise OSError("synthetic-sensitive-error-text")

    monkeypatch.setattr(LocalSealedEvidenceVault, "check_storage", unavailable)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.json()["status"] == "degraded"
    state = response.json()["evidence_capture"]
    assert state["requested_mode"] == state["effective_mode"] == "sealed"
    assert state["failure_reason"] == "storage_unavailable"
    assert not state["ready"] and not state["vault_available"]
    assert "synthetic-sensitive-error-text" not in response.text + caplog.text


def test_disabled_capture_health_stays_ready(proxy_config):
    from aurora_lens.proxy.app import create_app

    app = create_app(proxy_config(key=None, audit=False))
    with TestClient(app) as client:
        state = client.get("/health").json()["evidence_capture"]
    assert state["ready"] and not state["enabled"]


def test_explicit_hash_only_health_stays_ready(proxy_config):
    from aurora_lens.proxy.app import create_app

    app = create_app(proxy_config(mode="hash_only", key=None))
    with TestClient(app) as client:
        state = client.get("/health").json()["evidence_capture"]
    assert state["ready"] and not state["required_sealed"]
    assert state["effective_mode"] == CaptureMode.HASH_ONLY.value


@pytest.mark.parametrize("key", [None, "", "   "])
@pytest.mark.parametrize("factory_name", ["local", "bridge"])
def test_sealed_factories_reject_missing_or_empty_key(tmp_path, monkeypatch, key, factory_name):
    from aurora_lens.govern.evidence_capture import build_evidence_vault_for_bridge

    monkeypatch.delenv("AURORA_LENS_EVIDENCE_KEY", raising=False)
    factory = create_local_evidence_vault if factory_name == "local" else build_evidence_vault_for_bridge
    with pytest.raises(EvidenceVaultConfigurationError) as exc:
        factory(tmp_path / "audit.jsonl", capture_mode="sealed", encryption_key=key)
    assert exc.value.reason == CaptureFailureReason.MISSING_KEY


@pytest.mark.parametrize("component", ["vault", "blobs", "manifest", "audit"])
def test_required_sealed_factory_rejects_actual_storage_obstructions(tmp_path, component):
    path = tmp_path / "audit.jsonl"
    root = tmp_path / "audit.jsonl.evidence"
    if component == "audit":
        path.mkdir()
    elif component == "vault":
        root.write_bytes(b"synthetic obstruction")
    else:
        root.mkdir()
        if component == "blobs":
            (root / "blobs").write_bytes(b"synthetic obstruction")
        else:
            (root / "manifests.jsonl").mkdir()
    with pytest.raises(EvidenceVaultConfigurationError) as exc:
        create_local_evidence_vault(path, encryption_key=KEY)
    assert exc.value.reason == CaptureFailureReason.STORAGE_UNAVAILABLE
    assert KEY not in str(exc.value) and str(path) not in str(exc.value)


def test_missing_key_fails_before_provider_initialization(proxy_config, monkeypatch, caplog):
    from aurora_lens.proxy.app import create_app

    def forbidden(cfg):
        raise AssertionError("provider initialized before capture preflight")

    monkeypatch.setattr("aurora_lens.proxy.app._build_provider_adapters", forbidden)
    with pytest.raises(EvidenceVaultConfigurationError) as exc:
        create_app(proxy_config(key=None))
    assert exc.value.reason == CaptureFailureReason.MISSING_KEY
    record = next(r for r in caplog.records if r.message == "evidence_capture_unavailable")
    assert record.evidence_capture["failure_reason"] == "missing_key"
    assert not record.evidence_capture["ready"]


@pytest.mark.parametrize("failure", ["blob_probe", "bridge", "audit_append"])
def test_required_storage_failure_rejects_proxy_startup(proxy_config, monkeypatch, caplog, failure):
    from aurora_lens.proxy.app import create_app

    def unavailable(*args, **kwargs):
        raise OSError("synthetic-sensitive-error-text " + KEY)

    if failure == "blob_probe":
        monkeypatch.setattr(LocalSealedEvidenceVault, "check_storage", unavailable)
    elif failure == "bridge":
        monkeypatch.setattr("aurora_lens.govern.canonical_bridge.CanonicalScannerGateBridge", unavailable)
    else:
        monkeypatch.setattr("aurora_lens.proxy.app._check_audit_writable", lambda path: False)
    with pytest.raises(EvidenceVaultConfigurationError) as exc:
        create_app(proxy_config())
    assert exc.value.reason == CaptureFailureReason.STORAGE_UNAVAILABLE
    assert "synthetic-sensitive-error-text" not in caplog.text + str(exc.value)
    assert KEY not in caplog.text + str(exc.value)
    record = next(r for r in caplog.records if r.message == "evidence_capture_unavailable")
    assert record.evidence_capture["failure_reason"] == "storage_unavailable"
    assert record.evidence_capture["effective_mode"] is None or failure == "audit_append"


def test_readiness_fails_after_storage_loss_but_liveness_survives(proxy_config, monkeypatch):
    from aurora_lens.proxy.app import create_app

    app = create_app(proxy_config())
    original_check = LocalSealedEvidenceVault.check_storage
    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200
        monkeypatch.setattr(LocalSealedEvidenceVault, "check_storage", lambda self: (_ for _ in ()).throw(OSError()))
        assert client.get("/readyz").status_code == 503
        assert client.get("/health").status_code == 503
        assert client.get("/healthz").status_code == 200
        monkeypatch.setattr(LocalSealedEvidenceVault, "check_storage", original_check)
        assert client.get("/readyz").status_code == 200
        assert client.get("/health").status_code == 200


def test_required_capture_rejects_mismatched_or_unkeyed_vault(tmp_path):
    from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
    from aurora_lens.govern.evidence_capture import EvidenceCaptureConfig, capture_governance_evidence

    path = tmp_path / "audit.jsonl"
    vault = create_local_evidence_vault(path, capture_mode="hash_only")
    decision = GovernanceDecision(action=InterventionAction.HARD_STOP, flags=[], rationale="synthetic")
    with pytest.raises(EvidenceVaultConfigurationError):
        capture_governance_evidence(
            decision, pre_llm=True, vault=vault,
            config=EvidenceCaptureConfig(capture_mode="sealed"), request_prompt="synthetic request",
        )
    vault = create_local_evidence_vault(path, encryption_key=KEY)
    vault.master_key = None
    with pytest.raises(EvidenceVaultConfigurationError):
        capture_governance_evidence(
            decision, pre_llm=True, vault=vault,
            config=EvidenceCaptureConfig(capture_mode="sealed"), request_prompt="synthetic request",
        )


@pytest.mark.parametrize("basis", [None, "low_risk_conversational", "synthetic_admission"])
def test_explicit_hash_only_choice_is_preserved_for_pass_and_stop(tmp_path, basis):
    from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
    from aurora_lens.govern.evidence_capture import EvidenceCaptureConfig, capture_governance_evidence

    vault = create_local_evidence_vault(tmp_path / "audit.jsonl", capture_mode="hash_only")
    for action in (InterventionAction.PASS, InterventionAction.HARD_STOP):
        decision = GovernanceDecision(
            action=action, flags=[], rationale="synthetic", admissibility_basis=basis,
            original_response="synthetic upstream", governed_response="synthetic governed",
        )
        result = capture_governance_evidence(
            decision, pre_llm=False, vault=vault,
            config=EvidenceCaptureConfig(capture_mode="hash_only"), request_prompt="synthetic request",
        )
        assert result.request_capture_status == "hash_only"
        assert all(m["protection_mode"] == "hash_only" for m in result.evidence_manifests)


def test_keyed_restart_recovers_both_blobs_and_preserves_low_risk_pass(tmp_path):
    from aurora_lens.govern.decision import GovernanceDecision, InterventionAction
    from aurora_lens.govern.evidence_capture import EvidenceCaptureConfig, capture_governance_evidence
    from aurora_lens.verify.consequence_intent import LOW_RISK_CONVERSATIONAL

    path = tmp_path / "audit.jsonl"
    vault = create_local_evidence_vault(path, encryption_key=KEY)
    decision = GovernanceDecision(
        action=InterventionAction.HARD_STOP, flags=[], rationale="synthetic",
        original_response="synthetic withheld response",
    )
    captured = capture_governance_evidence(
        decision, pre_llm=False, vault=vault, config=EvidenceCaptureConfig(),
        request_prompt="synthetic request",
    )
    reopened = create_local_evidence_vault(path, encryption_key=KEY)
    for manifest, expected in zip(captured.evidence_manifests, (b"synthetic request", b"synthetic withheld response")):
        assert reopened.get_evidence(manifest["evidence_ref"], "forensic_review", "forensic_reviewer") == expected
        assert reopened.verify_evidence(manifest["evidence_ref"], manifest["raw_sha256"])
    decision.action = InterventionAction.PASS
    decision.admissibility_basis = LOW_RISK_CONVERSATIONAL
    result = capture_governance_evidence(
        decision, pre_llm=False, vault=reopened, config=EvidenceCaptureConfig(),
        request_prompt="synthetic low risk",
    )
    assert result.request_capture_status == "hash_only"
    assert result.evidence_capture_mode == "hash_only"


@pytest.mark.parametrize("factory_name", ["local", "bridge"])
def test_no_audit_path_disables_capture_without_key(monkeypatch, factory_name):
    from aurora_lens.govern.evidence_capture import build_evidence_vault_for_bridge

    monkeypatch.delenv("AURORA_LENS_EVIDENCE_KEY", raising=False)
    factory = create_local_evidence_vault if factory_name == "local" else build_evidence_vault_for_bridge
    assert factory(None, capture_mode="sealed") is None
    assert factory("", capture_mode="sealed") is None


def test_whitespace_environment_key_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("AURORA_LENS_EVIDENCE_KEY", "   ")
    with pytest.raises(EvidenceVaultConfigurationError) as exc:
        create_local_evidence_vault(tmp_path / "audit.jsonl")
    assert exc.value.reason == CaptureFailureReason.MISSING_KEY


def test_readiness_is_auth_exempt_and_has_no_diagnostics(proxy_config):
    from dataclasses import replace
    from aurora_lens.proxy.app import create_app
    from aurora_lens.proxy.config import AuthConfig

    app = create_app(replace(proxy_config(), auth=AuthConfig(enabled=True)))
    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"ready": True}


def test_hosted_health_keeps_capture_details_behind_diagnostic_boundary(proxy_config, monkeypatch):
    from aurora_lens.proxy.app import create_app

    cfg = proxy_config()
    monkeypatch.setenv("RAILWAY_ENVIRONMENT_NAME", "production")
    app = create_app(cfg)
    with TestClient(app) as client:
        public = client.get("/health")
        diagnostic = client.get("/health", headers={"x-aurora-edge-token": "pytest-aurora-edge-token"})
    assert set(public.json()) == {"status", "application_version", "git_commit"}
    assert diagnostic.json()["evidence_capture"]["ready"]
    assert KEY not in public.text + diagnostic.text


def test_hash_only_readiness_does_not_require_a_blob_directory(proxy_config, tmp_path):
    from aurora_lens.proxy.app import create_app

    cfg = proxy_config(mode="hash_only", key=None)
    root = tmp_path / "audit.jsonl.evidence"
    root.mkdir()
    (root / "blobs").write_bytes(b"synthetic unused blob obstruction")
    app = create_app(cfg)
    with TestClient(app) as client:
        response = client.get("/readyz")
    assert response.status_code == 200
    assert app.state.evidence_capture_state["vault_available"]


def test_state_uses_loaded_vault_key_availability(tmp_path):
    path = tmp_path / "audit.jsonl"
    vault = create_local_evidence_vault(path, encryption_key=KEY)
    vault.master_key = None
    state = evidence_capture_state(
        path, capture_mode="sealed", encryption_key=KEY, vault=vault,
        storage_available=True,
    )
    assert not state.ready and not state.key_available
    assert state.failure_reason == CaptureFailureReason.MISSING_KEY
