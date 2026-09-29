"""Product Phase 3 checkpoint posture — manifest integrity and layer boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHECKPOINT_MANIFEST = _REPO_ROOT / "eval" / "product_phase_3_checkpoint.manifest.yaml"


class TestProductPhase3CheckpointPosture:
    def test_manifest_lists_all_slices_and_expected_total(self):
        data = yaml.safe_load(_CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
        includes = data.get("includes") or {}
        assert "document_ingestion" in includes
        assert "intake_translator" in includes
        assert "corpus_ingest" in includes
        assert "checkpoint_regression" in includes
        expected = data.get("expected_pass_counts") or {}
        assert expected.get("total") == sum(
            expected[k]
            for k in ("document_ingestion", "intake_translator", "corpus_ingest", "checkpoint_regression")
        )

    def test_locked_rules_include_proposal_only_boundaries(self):
        data = yaml.safe_load(_CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
        rules = set(data.get("locked_rules") or [])
        assert "ingestion_proposes_only" in rules
        assert "intake_proposes_only" in rules
        assert "no_pef_mutation" in rules
        assert "explicit_admission_before_gate_payload" in rules

    def test_primary_code_paths_exist(self):
        data = yaml.safe_load(_CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
        for rel in data.get("primary_code") or []:
            assert (_REPO_ROOT / rel).is_file(), rel

    def _module_imports(self, path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.add(node.module)
        return names

    def test_document_ingestion_module_does_not_import_lens_or_pef(self):
        path = _REPO_ROOT / "aurora_lens" / "corpus" / "document_ingestion.py"
        imports = self._module_imports(path)
        assert not any(m.startswith("aurora_lens.lens") for m in imports)
        assert not any(m.startswith("aurora_lens.pef") for m in imports)

    def test_intake_translator_module_does_not_import_lens(self):
        path = _REPO_ROOT / "aurora_lens" / "corpus" / "intake_translator.py"
        imports = self._module_imports(path)
        assert not any(m.startswith("aurora_lens.lens") for m in imports)
