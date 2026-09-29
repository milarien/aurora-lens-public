"""Corpus product UX end-to-end tests (mocked proxy)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aurora_lens.corpus.cli_support import cmd_ask, cmd_ingest


@pytest.fixture
def corpus_tree(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    source = tmp_path / "policy.md"
    source.write_text(
        "# Policy\n\nSection A: retention is seven years.\n\nSection B: audits are quarterly.\n",
        encoding="utf-8",
    )
    cmd_ingest(record_id="policy-v1", file_path=source, corpus_root=root)
    return root


def test_ingest_then_ask_mocked_proxy(corpus_tree: Path, capsys: pytest.CaptureFixture[str]):
    fake_resp = {
        "choices": [{"message": {"content": "Retention is seven years per Section A."}}],
        "aurora": {"governance": "PASS"},
    }

    with patch("aurora_lens.corpus.cli_support.check_proxy"), patch(
        "aurora_lens.corpus.cli_support.post_chat",
        return_value=fake_resp,
    ):
        assert (
            cmd_ask(
                record_id="policy-v1",
                question="What is the retention period?",
                corpus_root=corpus_tree,
                proxy="http://mock-proxy",
            )
            == 0
        )

    out = capsys.readouterr().out
    assert "seven years" in out.lower()
    assert "PASS" in out


def test_ask_assembles_rag_message_with_context(corpus_tree: Path):
    captured: dict = {}

    def _capture_post_chat(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}}],
            "aurora": {"governance": "PASS"},
        }

    with patch("aurora_lens.corpus.cli_support.check_proxy"), patch(
        "aurora_lens.corpus.cli_support.post_chat",
        side_effect=_capture_post_chat,
    ):
        cmd_ask(
            record_id="policy-v1",
            question="What is the retention period?",
            corpus_root=corpus_tree,
        )

    msg = captured.get("user_message") or ""
    meta = captured.get("request_metadata") or {}
    assert "Context:" in msg
    assert "Question:" in msg
    assert meta.get("record_ids") == ["policy-v1"]
