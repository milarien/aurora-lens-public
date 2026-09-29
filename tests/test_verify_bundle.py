"""Tests for evidence bundle manifest (Stage E)."""

import hashlib
import json
import sys
from pathlib import Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class TestBundleManifest:
    """Verify evidence bundle creation and verification (Stage E)."""

    def test_bundle_creation_writes_manifest(self, tmp_path):
        """write_manifest() creates manifest.json with correct fields."""
        sys.path.insert(0, str(Path("scripts").absolute()))
        from smoke_proof_bundle import write_manifest

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        # Write dummy files
        (bundle_dir / "proof_bundle.txt").write_text("proof content here", encoding="utf-8")
        (bundle_dir / "audit_entries_redacted.json").write_text('{"entries": []}', encoding="utf-8")

        manifest_path, manifest_sha256 = write_manifest(bundle_dir, run_id="test-run-id")

        assert manifest_path.exists(), "manifest.json was not created"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["bundle_version"] == "1"
        assert manifest["run_id"] == "test-run-id"
        assert "created_at" in manifest
        assert len(manifest["files"]) == 2
        file_names = {f["name"] for f in manifest["files"]}
        assert "proof_bundle.txt" in file_names
        assert "audit_entries_redacted.json" in file_names
        # Verify the returned sha256 is correct
        assert manifest_sha256 == _sha256_file(manifest_path)

    def test_write_manifest_omits_run_id_when_none(self, tmp_path):
        """manifest.json excludes run_id when not provided."""
        sys.path.insert(0, str(Path("scripts").absolute()))
        from smoke_proof_bundle import write_manifest

        bundle_dir = tmp_path / "bundle2"
        bundle_dir.mkdir()
        (bundle_dir / "only.txt").write_text("x", encoding="utf-8")
        write_manifest(bundle_dir, run_id=None)
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        assert "run_id" not in manifest

    def test_verify_bundle_passes_on_clean_bundle(self, tmp_path):
        """verify_bundle() returns overall ok when all files match the manifest."""
        sys.path.insert(0, str(Path("scripts").absolute()))
        from smoke_proof_bundle import write_manifest
        from verify_bundle import verify_bundle

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "file_a.txt").write_text("content A", encoding="utf-8")
        (bundle_dir / "file_b.json").write_text('{"k": 1}', encoding="utf-8")
        write_manifest(bundle_dir, run_id="run-xyz")

        ok, issues, per_file = verify_bundle(bundle_dir)
        assert ok is True, f"Expected clean bundle to pass, got issues: {issues}"
        assert issues == []
        assert len(per_file) == 2
        assert all(ent_ok for _n, ent_ok in per_file)

    def test_verify_bundle_fails_on_tampered_file(self, tmp_path):
        """verify_bundle() fails when a file is modified after manifest creation."""
        sys.path.insert(0, str(Path("scripts").absolute()))
        from smoke_proof_bundle import write_manifest
        from verify_bundle import verify_bundle

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        target = bundle_dir / "important.txt"
        target.write_text("original content", encoding="utf-8")
        write_manifest(bundle_dir, run_id="run-abc")

        # Tamper
        target.write_text("tampered content!!", encoding="utf-8")

        ok, issues, _ = verify_bundle(bundle_dir)
        assert ok is False, "Expected tampered bundle to fail verification"
        assert any("HASH_MISMATCH" in issue for issue in issues), (
            f"Expected HASH_MISMATCH issue, got: {issues}"
        )