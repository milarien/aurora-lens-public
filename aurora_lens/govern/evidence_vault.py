"""Forensic evidence vault — sealed storage for prompts and model outputs.

The main governance ledger stores :class:`EvidenceManifest` summaries only;
raw prompt and upstream text live in the vault, not inline in audit rows.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

EVIDENCE_MANIFEST_SCHEMA_VERSION = "1.0"
EVIDENCE_VAULT_BLOB_MAGIC = b"AELV1\x00"
CANONICALIZATION_PROFILE = "aurora-lens/v1:utf-8-nfc-trim"
EVIDENCE_ACCESS_POLICY_AUDIT_ONLY = "audit_only"
INTERNAL_EVIDENCE_REF_PREFIX = "ev:"
SEALED_EVIDENCE_REF_PREFIX = "vault://sealed/"
HASH_ONLY_EVIDENCE_REF_PREFIX = "vault://hash_only/"


class EvidenceKind(str, Enum):
    REQUEST_PROMPT = "request_prompt"
    UPSTREAM_MODEL_OUTPUT = "upstream_model_output"
    GOVERNED_OUTPUT = "governed_output"
    RETRIEVED_CONTEXT = "retrieved_context"
    ATTACHMENT_METADATA = "attachment_metadata"


class CaptureMode(str, Enum):
    SEALED = "sealed"
    REDACTED = "redacted"
    HASH_ONLY = "hash_only"
    PLAINTEXT_DEV = "plaintext_dev"


class CaptureStatus(str, Enum):
    CAPTURED = "captured"
    SEALED_BY_POLICY = "sealed_by_policy"
    REDACTED = "redacted"
    HASH_ONLY = "hash_only"
    OMITTED = "omitted"
    FORBIDDEN_BY_POLICY = "forbidden_by_policy"


class EvidencePurpose(str, Enum):
    FORENSIC_REVIEW = "forensic_review"
    LEGAL_DISCOVERY = "legal_discovery"
    OPERATOR_DEBUG = "operator_debug"
    USER_EXPORT = "user_export"


class RetentionClass(str, Enum):
    STANDARD = "standard"
    EXTENDED = "extended"
    LEGAL_HOLD = "legal_hold"


_PURPOSE_ACTORS: dict[EvidencePurpose, frozenset[str]] = {
    EvidencePurpose.FORENSIC_REVIEW: frozenset(
        {"forensic_reviewer", "audit_service", "compliance_officer"}
    ),
    EvidencePurpose.LEGAL_DISCOVERY: frozenset(
        {"legal_counsel", "discovery_officer", "compliance_officer"}
    ),
    EvidencePurpose.OPERATOR_DEBUG: frozenset(
        {"operator", "audit_service", "forensic_reviewer"}
    ),
    EvidencePurpose.USER_EXPORT: frozenset({"user", "data_subject", "operator"}),
}


class EvidenceAccessDenied(PermissionError):
    """Raised when actor/purpose is not authorized for evidence retrieval."""


class EvidenceVaultConfigurationError(RuntimeError):
    """Raised when sealed capture is configured without a usable encryption key."""


class EvidenceIntegrityError(ValueError):
    """Raised when sealed blob integrity verification fails."""


class EvidenceNotFoundError(FileNotFoundError):
    """Raised when an evidence_ref does not exist in the vault."""


class EvidenceLegalHoldError(RuntimeError):
    """Raised when delete/expiry is attempted on legal-hold evidence."""


def canonicalize_evidence_content(content: str | bytes) -> bytes:
    """Canonical bytes for hashing (UTF-8 NFC + outer trim for text)."""
    if isinstance(content, bytes):
        return content
    text = str(content)
    try:
        import unicodedata

        text = unicodedata.normalize("NFC", text)
    except ImportError:
        pass
    return text.strip().encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def format_sha256_digest(hex_digest: str) -> str:
    """Public audit-plane digest form (``sha256:<hex>``)."""
    text = normalize_sha256_digest(hex_digest)
    return f"sha256:{text}"


def normalize_sha256_digest(value: str) -> str:
    return str(value).strip().lower().removeprefix("sha256:")


def public_evidence_ref(internal_ref: str, *, protection_mode: str) -> str:
    """Map internal ``ev:<id>`` refs to operator-safe vault URIs."""
    ref_id = str(internal_ref).strip()
    if ref_id.startswith(INTERNAL_EVIDENCE_REF_PREFIX):
        ref_id = ref_id[len(INTERNAL_EVIDENCE_REF_PREFIX) :]
    for prefix in (SEALED_EVIDENCE_REF_PREFIX, HASH_ONLY_EVIDENCE_REF_PREFIX):
        if str(internal_ref).startswith(prefix):
            return str(internal_ref)
    if protection_mode == CaptureMode.HASH_ONLY.value:
        return f"{HASH_ONLY_EVIDENCE_REF_PREFIX}{ref_id}"
    return f"{SEALED_EVIDENCE_REF_PREFIX}{ref_id}"


def normalize_evidence_ref(ref: str) -> str:
    """Accept public vault URIs or internal ``ev:<id>`` for vault operations."""
    text = str(ref).strip()
    for prefix in (SEALED_EVIDENCE_REF_PREFIX, HASH_ONLY_EVIDENCE_REF_PREFIX):
        if text.startswith(prefix):
            return f"{INTERNAL_EVIDENCE_REF_PREFIX}{text[len(prefix):]}"
    return text


def raw_and_canonical_hashes(content: str | bytes) -> tuple[str, str]:
    raw = content if isinstance(content, bytes) else str(content).encode("utf-8")
    canonical = canonicalize_evidence_content(content)
    return sha256_hex(raw), sha256_hex(canonical)


@dataclass
class EvidenceManifest:
    """Summary stored in the governance ledger (never full prompt text)."""

    evidence_ref: str
    evidence_kind: str
    capture_status: str
    capture_basis: str
    raw_sha256: str
    canonical_sha256: str
    canonicalization_profile: str = CANONICALIZATION_PROFILE
    byte_length: int = 0
    created_at: str = ""
    retention_class: str = RetentionClass.STANDARD.value
    expires_at: str | None = None
    legal_hold: bool = False
    protection_mode: str = CaptureMode.SEALED.value
    encryption_algorithm: str | None = "AELV1-HMAC-CTR"
    key_ref: str | None = None
    access_policy: str = EVIDENCE_ACCESS_POLICY_AUDIT_ONLY
    redaction_status: str = "none"
    schema_version: str = EVIDENCE_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_ref": self.evidence_ref,
            "evidence_kind": self.evidence_kind,
            "capture_status": self.capture_status,
            "capture_basis": self.capture_basis,
            "raw_sha256": self.raw_sha256,
            "canonical_sha256": self.canonical_sha256,
            "canonicalization_profile": self.canonicalization_profile,
            "byte_length": self.byte_length,
            "created_at": self.created_at,
            "retention_class": self.retention_class,
            "expires_at": self.expires_at,
            "legal_hold": self.legal_hold,
            "protection_mode": self.protection_mode,
            "encryption_algorithm": self.encryption_algorithm,
            "key_ref": self.key_ref,
            "access_policy": self.access_policy,
            "redaction_status": self.redaction_status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceManifest:
        return cls(
            evidence_ref=str(data["evidence_ref"]),
            evidence_kind=str(data["evidence_kind"]),
            capture_status=str(data["capture_status"]),
            capture_basis=str(data["capture_basis"]),
            raw_sha256=str(data["raw_sha256"]),
            canonical_sha256=str(data["canonical_sha256"]),
            canonicalization_profile=str(
                data.get("canonicalization_profile") or CANONICALIZATION_PROFILE
            ),
            byte_length=int(data.get("byte_length") or 0),
            created_at=str(data.get("created_at") or ""),
            retention_class=str(data.get("retention_class") or RetentionClass.STANDARD.value),
            expires_at=data.get("expires_at"),
            legal_hold=bool(data.get("legal_hold")),
            protection_mode=str(data.get("protection_mode") or CaptureMode.SEALED.value),
            encryption_algorithm=data.get("encryption_algorithm"),
            key_ref=data.get("key_ref"),
            access_policy=str(
                data.get("access_policy") or EVIDENCE_ACCESS_POLICY_AUDIT_ONLY
            ),
            redaction_status=str(data.get("redaction_status") or "none"),
            schema_version=str(data.get("schema_version") or EVIDENCE_MANIFEST_SCHEMA_VERSION),
        )


class EvidenceVault(Protocol):
    """Forensic evidence store — manifests in ledger, content in vault."""

    def put_evidence(
        self,
        kind: EvidenceKind | str,
        content: str | bytes,
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceManifest:
        """Persist evidence and return a ledger-safe manifest."""

    def get_evidence(
        self,
        evidence_ref: str,
        purpose: EvidencePurpose | str,
        actor: str,
    ) -> bytes:
        """Retrieve raw evidence bytes after purpose/actor authorization."""

    def verify_evidence(self, evidence_ref: str, expected_hash: str) -> bool:
        """Verify stored content matches ``expected_hash`` (raw or canonical hex)."""

    def delete_evidence(self, evidence_ref: str) -> None:
        """Delete evidence when retention permits (no-op if not implemented)."""


def _resolve_purpose(purpose: EvidencePurpose | str) -> EvidencePurpose:
    if isinstance(purpose, EvidencePurpose):
        return purpose
    return EvidencePurpose(str(purpose).strip().lower())


def _authorize_access(purpose: EvidencePurpose | str, actor: str) -> None:
    purpose_enum = _resolve_purpose(purpose)
    actor_norm = str(actor or "").strip().lower()
    allowed = _PURPOSE_ACTORS.get(purpose_enum, frozenset())
    if actor_norm not in allowed:
        raise EvidenceAccessDenied(
            f"actor {actor!r} not authorized for purpose {purpose_enum.value!r}"
        )


def _derive_key(master_key: bytes, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", master_key, salt, 200_000, dklen=32)


def _xor_keystream(key: bytes, iv: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hmac.new(key, iv + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def seal_encrypt(master_key: bytes, plaintext: bytes) -> bytes:
    """Seal plaintext with PBKDF2-derived key + HMAC-CTR (stdlib only)."""
    salt = os.urandom(16)
    iv = os.urandom(16)
    dek = _derive_key(master_key, salt)
    stream = _xor_keystream(dek, iv, len(plaintext))
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, stream))
    mac = hmac.new(dek, salt + iv + ciphertext, hashlib.sha256).digest()
    return EVIDENCE_VAULT_BLOB_MAGIC + salt + iv + mac + ciphertext


def seal_decrypt(master_key: bytes, blob: bytes) -> bytes:
    if not blob.startswith(EVIDENCE_VAULT_BLOB_MAGIC):
        raise EvidenceIntegrityError("unknown evidence blob format")
    body = blob[len(EVIDENCE_VAULT_BLOB_MAGIC) :]
    if len(body) < 16 + 16 + 32:
        raise EvidenceIntegrityError("truncated evidence blob")
    salt = body[0:16]
    iv = body[16:32]
    mac = body[32:64]
    ciphertext = body[64:]
    dek = _derive_key(master_key, salt)
    expected_mac = hmac.new(dek, salt + iv + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected_mac):
        raise EvidenceIntegrityError("evidence blob MAC mismatch")
    stream = _xor_keystream(dek, iv, len(ciphertext))
    return bytes(a ^ b for a, b in zip(ciphertext, stream))


def evidence_vault_dir_for_audit(audit_path: str | Path | None) -> Path | None:
    if audit_path is None:
        return None
    p = Path(audit_path)
    return p.parent / f"{p.name}.evidence"


def resolve_evidence_encryption_key(
    explicit: str | bytes | None = None,
) -> bytes | None:
    if explicit is not None:
        if isinstance(explicit, bytes):
            return explicit if explicit else None
        text = str(explicit).strip()
        return text.encode("utf-8") if text else None
    env = os.environ.get("AURORA_LENS_EVIDENCE_KEY", "").strip()
    return env.encode("utf-8") if env else None


@dataclass
class LocalSealedEvidenceVault:
    """Append-only local vault: encrypted blobs + sidecar manifest index."""

    root: Path
    master_key: bytes | None
    capture_mode: CaptureMode = CaptureMode.SEALED
    key_ref: str = "local/env"
    default_retention_days: int = 365
    _manifest_index_name: str = field(default="manifests.jsonl", init=False, repr=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.capture_mode == CaptureMode.SEALED and not self.master_key:
            raise EvidenceVaultConfigurationError(
                "sealed evidence capture requires AURORA_LENS_EVIDENCE_KEY or evidence_encryption_key"
            )

    def _manifest_path(self) -> Path:
        return self.root / self._manifest_index_name

    def _blob_path(self, evidence_ref: str) -> Path:
        internal_ref = normalize_evidence_ref(evidence_ref)
        safe = internal_ref.replace(":", "_").replace("/", "_")
        return self.root / "blobs" / f"{safe}.bin"

    def _append_manifest_record(self, manifest: EvidenceManifest) -> None:
        line = json.dumps(manifest.to_dict(), separators=(",", ":"), sort_keys=True)
        with self._manifest_path().open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _load_manifest(self, evidence_ref: str) -> EvidenceManifest:
        internal_ref = normalize_evidence_ref(evidence_ref)
        path = self._manifest_path()
        if not path.exists():
            raise EvidenceNotFoundError(evidence_ref)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("evidence_ref") == internal_ref:
                return EvidenceManifest.from_dict(row)
        raise EvidenceNotFoundError(evidence_ref)

    def put_evidence(
        self,
        kind: EvidenceKind | str,
        content: str | bytes,
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceManifest:
        meta = dict(metadata or {})
        kind_val = kind.value if isinstance(kind, EvidenceKind) else str(kind)
        raw_bytes = content if isinstance(content, bytes) else str(content).encode("utf-8")
        raw_hash, canonical_hash = raw_and_canonical_hashes(content)
        now = datetime.now(timezone.utc)
        retention_class = str(meta.get("retention_class") or RetentionClass.STANDARD.value)
        legal_hold = bool(meta.get("legal_hold"))
        retention_days = int(meta.get("retention_days") or self.default_retention_days)
        expires_at = None if legal_hold else (now + timedelta(days=retention_days)).isoformat()
        capture_basis = str(meta.get("capture_basis") or "governance_decision")
        protection_mode_raw = meta.get("protection_mode")
        if protection_mode_raw is not None:
            protection_mode = (
                protection_mode_raw
                if isinstance(protection_mode_raw, CaptureMode)
                else CaptureMode(str(protection_mode_raw))
            )
        else:
            protection_mode = self.capture_mode
        capture_status = CaptureStatus.CAPTURED
        redaction_status = "none"
        encryption_algorithm: str | None = "AELV1-HMAC-CTR"
        store_bytes = raw_bytes

        if protection_mode == CaptureMode.HASH_ONLY:
            capture_status = CaptureStatus.HASH_ONLY
            encryption_algorithm = None
            store_bytes = b""
        elif protection_mode == CaptureMode.REDACTED:
            capture_status = CaptureStatus.REDACTED
            redaction_status = str(meta.get("redaction_status") or "content_redacted")
            redacted = str(meta.get("redacted_preview") or "[REDACTED]")
            store_bytes = redacted.encode("utf-8")
        elif protection_mode == CaptureMode.PLAINTEXT_DEV:
            encryption_algorithm = None
        elif protection_mode == CaptureMode.SEALED:
            if not self.master_key:
                raise EvidenceVaultConfigurationError("sealed mode requires encryption key")
            store_bytes = seal_encrypt(self.master_key, raw_bytes)

        evidence_ref = str(meta.get("evidence_ref") or f"{INTERNAL_EVIDENCE_REF_PREFIX}{uuid.uuid4().hex}")
        access_policy = str(
            meta.get("access_policy") or EVIDENCE_ACCESS_POLICY_AUDIT_ONLY
        )
        manifest = EvidenceManifest(
            evidence_ref=evidence_ref,
            evidence_kind=kind_val,
            capture_status=capture_status.value,
            capture_basis=capture_basis,
            raw_sha256=raw_hash,
            canonical_sha256=canonical_hash,
            byte_length=len(raw_bytes),
            created_at=now.isoformat(),
            retention_class=retention_class,
            expires_at=expires_at,
            legal_hold=legal_hold,
            protection_mode=protection_mode.value,
            encryption_algorithm=encryption_algorithm,
            key_ref=self.key_ref if protection_mode == CaptureMode.SEALED else None,
            access_policy=access_policy,
        )

        if protection_mode != CaptureMode.HASH_ONLY:
            blob_path = self._blob_path(evidence_ref)
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            blob_path.write_bytes(store_bytes)

        self._append_manifest_record(manifest)
        return manifest

    def get_evidence(
        self,
        evidence_ref: str,
        purpose: EvidencePurpose | str,
        actor: str,
    ) -> bytes:
        _authorize_access(purpose, actor)
        manifest = self._load_manifest(evidence_ref)
        if manifest.capture_status == CaptureStatus.HASH_ONLY.value:
            raise EvidenceNotFoundError(
                f"{evidence_ref}: hash_only capture — no retrievable content"
            )
        blob_path = self._blob_path(evidence_ref)
        if not blob_path.exists():
            raise EvidenceNotFoundError(evidence_ref)
        blob = blob_path.read_bytes()
        if manifest.protection_mode == CaptureMode.SEALED.value:
            if not self.master_key:
                raise EvidenceVaultConfigurationError("sealed retrieval requires encryption key")
            return seal_decrypt(self.master_key, blob)
        return blob

    def verify_evidence(self, evidence_ref: str, expected_hash: str) -> bool:
        manifest = self._load_manifest(evidence_ref)
        expected = str(expected_hash).strip().lower().removeprefix("sha256:")
        if manifest.capture_status == CaptureStatus.HASH_ONLY.value:
            return expected in {manifest.raw_sha256, manifest.canonical_sha256}
        try:
            try:
                content = self.get_evidence(
                    evidence_ref,
                    EvidencePurpose.FORENSIC_REVIEW,
                    "forensic_reviewer",
                )
            except EvidenceAccessDenied:
                content = self._read_blob_without_auth(evidence_ref, manifest)
        except EvidenceIntegrityError:
            return False
        raw_hash, canonical_hash = raw_and_canonical_hashes(content)
        return raw_hash == expected or canonical_hash == expected

    def _read_blob_without_auth(self, evidence_ref: str, manifest: EvidenceManifest) -> bytes:
        blob = self._blob_path(evidence_ref).read_bytes()
        if manifest.protection_mode == CaptureMode.SEALED.value:
            if not self.master_key:
                raise EvidenceVaultConfigurationError("sealed verify requires encryption key")
            try:
                return seal_decrypt(self.master_key, blob)
            except EvidenceIntegrityError:
                raise
        return blob

    def delete_evidence(self, evidence_ref: str) -> None:
        manifest = self._load_manifest(evidence_ref)
        if manifest.legal_hold:
            raise EvidenceLegalHoldError(
                f"evidence {evidence_ref} is under legal_hold and cannot be deleted"
            )
        blob_path = self._blob_path(evidence_ref)
        if blob_path.exists():
            blob_path.unlink()


def create_local_evidence_vault(
    audit_path: str | Path | None,
    *,
    capture_mode: str | CaptureMode = CaptureMode.SEALED,
    encryption_key: str | bytes | None = None,
) -> LocalSealedEvidenceVault | None:
    """Construct vault adjacent to audit log path."""
    vault_dir = evidence_vault_dir_for_audit(audit_path)
    if vault_dir is None:
        return None
    mode = capture_mode if isinstance(capture_mode, CaptureMode) else CaptureMode(str(capture_mode))
    key = resolve_evidence_encryption_key(encryption_key)
    if mode == CaptureMode.SEALED and not key:
        # Sealed-and-referencable without encryption key: retain hash-only proof.
        mode = CaptureMode.HASH_ONLY
    return LocalSealedEvidenceVault(root=vault_dir, master_key=key, capture_mode=mode)


# Protocol conformance for type checkers
class _EvidenceVaultABC(ABC):
    @abstractmethod
    def put_evidence(
        self,
        kind: EvidenceKind | str,
        content: str | bytes,
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceManifest:
        raise NotImplementedError

    @abstractmethod
    def get_evidence(
        self,
        evidence_ref: str,
        purpose: EvidencePurpose | str,
        actor: str,
    ) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def verify_evidence(self, evidence_ref: str, expected_hash: str) -> bool:
        raise NotImplementedError
