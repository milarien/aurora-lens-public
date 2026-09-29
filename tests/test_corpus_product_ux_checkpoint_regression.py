"""Corpus product UX checkpoint posture."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = _REPO / "eval" / "corpus_product_ux_checkpoint.manifest.yaml"


class TestCorpusProductUxCheckpointPosture:
    def test_manifest_expected_total(self):
        data = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
        expected = data.get("expected_pass_counts") or {}
        assert expected.get("total") == sum(
            expected[k]
            for k in ("corpus_cli", "corpus_e2e", "checkpoint_regression")
        )

    def test_cli_support_module_exists(self):
        assert (_REPO / "aurora_lens" / "corpus" / "cli_support.py").is_file()

    def test_corpus_subcommand_in_cli(self):
        text = (_REPO / "aurora_lens" / "cli.py").read_text(encoding="utf-8")
        assert "corpus" in text
        assert "_corpus_ask" in text

    def test_product_ux_doc_exists(self):
        assert (_REPO / "docs" / "CORPUS_PRODUCT_UX.md").is_file()
