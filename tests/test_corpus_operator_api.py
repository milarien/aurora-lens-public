"""Tests for corpus operator API (forensics console backend)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from aurora_lens.corpus.cli_support import cmd_ingest
from aurora_lens.corpus.registry import CorpusRegistry
from aurora_lens.corpus.operator_api import (
    api_ask_corpus,
    api_clear_operator_corpus,
    api_ingest_path,
    api_list_records,
    api_retrieve_evidence,
    api_validate_question,
    derive_support_status,
    preflight_upstream_review,
    preflight_validate_review,
)
from aurora_lens.corpus.review import upstream_config_status


def _assert_preflight_failure_shape(
    out: dict,
    *,
    reason: str,
    question: str,
    record_id: str,
) -> None:
    assert out["review_outcome"] == "failed"
    assert out["review_failure_reason"] == reason
    assert out["question"] == question
    assert out["record_id"] == record_id
    assert "review_failure_detail" in out


@pytest.fixture
def corpus_tree(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    source = tmp_path / "policy.md"
    source.write_text(
        "# Policy\n\nPEF persistent state constrains model output before generation.\n"
        "Refusal is a governed outcome when admissibility fails.\n",
        encoding="utf-8",
    )
    cmd_ingest(record_id="policy-v1", file_path=source, corpus_root=root, title="Policy")
    return root


def test_api_list_records(corpus_tree: Path):
    out = api_list_records(corpus_root=corpus_tree)
    assert out["count"] == 1
    assert out["records"][0]["record_id"] == "policy-v1"
    assert out["records"][0]["file_type"] == "markdown"
    assert out["records"][0]["chunk_count"] >= 1


def test_api_ingest_file(corpus_tree: Path, tmp_path: Path):
    doc = tmp_path / "extra.txt"
    doc.write_text("Supplier approval requires ISO 9001.\n", encoding="utf-8")
    out = api_ingest_path(path=doc, record_id="extra-doc", corpus_root=corpus_tree)
    assert out["ingested_count"] == 1
    listed = api_list_records(corpus_root=corpus_tree)
    assert listed["count"] == 2


def test_api_retrieve_evidence(corpus_tree: Path):
    out = api_retrieve_evidence(
        question="Which documents mention PEF?",
        record_id="policy-v1",
        corpus_root=corpus_tree,
    )
    assert out["evidence_count"] >= 1
    assert out["evidence"][0]["snippet"]
    assert out["evidence"][0]["score"] is not None
    assert out["evidence"][0]["evidence_role"] in {"direct_support", "background", "metadata", "low_relevance"}
    assert "evidence_sections" in out
    assert out["evidence_visible_count"] >= 1


def test_derive_support_status_flags():
    status, issues = derive_support_status(
        answer="ok",
        governance="PASS",
        flags=["SCOPE_MISMATCH"],
    )
    assert status == "scope_mismatch"
    assert issues


def test_api_ask_corpus_mocked(corpus_tree: Path):
    registry = CorpusRegistry(corpus_tree)
    record = registry.get_record("policy-v1")
    assert record is not None
    registry.upsert_record(
        replace(
            record,
            authority_class="corporate_policy",
            status="approved",
            effective_from="2026-01-01",
            effective_until="2026-12-31",
            scope="corpus",
            parser_status="ok",
        )
    )
    fake = {
        "choices": [{"message": {"content": "PEF constrains output before generation."}}],
        "aurora": {"governance": "PASS", "flags": []},
    }

    with patch("aurora_lens.corpus.operator_api.post_chat", return_value=fake):
        out = api_ask_corpus(
            question="Which documents mention PEF?",
            record_id="policy-v1",
            corpus_root=corpus_tree,
            proxy="http://mock",
        )

    assert "PEF" in out["answer"]
    assert out["governance"] == "PASS"
    assert out["support_status"] == "supported"
    assert out["evidence"]


def test_api_ask_corpus_blocks_inadmissible_evidence_before_lens_call(corpus_tree: Path):
    registry = CorpusRegistry(corpus_tree)
    record = registry.get_record("policy-v1")
    assert record is not None
    registry.upsert_record(
        replace(
            record,
            authority_class="corporate_policy",
            status="draft",
            effective_from="2026-01-01",
            effective_until="2026-12-31",
            scope="corpus",
            parser_status="ok",
        )
    )
    with patch("aurora_lens.corpus.operator_api.post_chat") as post_chat:
        out = api_ask_corpus(
            question="Which documents mention PEF?",
            record_id="policy-v1",
            corpus_root=corpus_tree,
            proxy="http://mock",
        )
    post_chat.assert_not_called()
    assert out["support_status"] == "not_supported"
    assert out["evidence_admissibility"]["status"] == "inadmissible"
    assert out["evidence_admissibility"]["failure_kind"] == "authority_missing"


def test_api_ask_corpus_blocks_unresolved_evidence_before_lens_call(corpus_tree: Path):
    registry = CorpusRegistry(corpus_tree)
    record = registry.get_record("policy-v1")
    assert record is not None
    registry.upsert_record(
        replace(
            record,
            authority_class=None,
            status="approved",
            effective_from="2026-01-01",
            effective_until="2026-12-31",
            scope="corpus",
            parser_status="ok",
        )
    )
    with patch("aurora_lens.corpus.operator_api.post_chat") as post_chat:
        out = api_ask_corpus(
            question="Which documents mention PEF?",
            record_id="policy-v1",
            corpus_root=corpus_tree,
            proxy="http://mock",
        )
    post_chat.assert_not_called()
    assert out["support_status"] == "partial"
    assert out["evidence_admissibility"]["status"] == "unresolved"
    assert out["evidence_admissibility"]["failure_kind"] == "authority_unknown"


def test_api_ask_corpus_enforces_source_scope_at_retrieval(corpus_tree: Path):
    registry = CorpusRegistry(corpus_tree)
    record = registry.get_record("policy-v1")
    assert record is not None
    registry.upsert_record(
        replace(
            record,
            authority_class="corporate_policy",
            status="approved",
            effective_from="2026-01-01",
            effective_until="2026-12-31",
            scope="hr",
            parser_status="ok",
        )
    )
    with patch("aurora_lens.corpus.operator_api.post_chat") as post_chat:
        out = api_ask_corpus(
            question="Which documents mention PEF?",
            record_id="policy-v1",
            corpus_root=corpus_tree,
            proxy="http://mock",
            source_scope="corpus",
        )
    post_chat.assert_not_called()
    assert out["support_status"] == "not_supported"
    assert out["record_ids"] == []
    assert out["operator_note"] == "Retrieval found no passages; Lens was not called."


def test_api_ask_corpus_allows_global_scope_records(corpus_tree: Path):
    registry = CorpusRegistry(corpus_tree)
    record = registry.get_record("policy-v1")
    assert record is not None
    registry.upsert_record(
        replace(
            record,
            authority_class="corporate_policy",
            status="approved",
            effective_from="2026-01-01",
            effective_until="2026-12-31",
            scope="global",
            parser_status="ok",
        )
    )
    fake = {
        "choices": [{"message": {"content": "PEF constrains output before generation."}}],
        "aurora": {"governance": "PASS", "flags": []},
    }
    with patch("aurora_lens.corpus.operator_api.post_chat", return_value=fake):
        out = api_ask_corpus(
            question="Which documents mention PEF?",
            record_id="policy-v1",
            corpus_root=corpus_tree,
            proxy="http://mock",
            source_scope="workspace_documents",
        )
    assert out["support_status"] == "supported"
    assert out["record_ids"] == ["policy-v1"]


def test_api_retrieve_evidence_excludes_scope_mismatch_records(corpus_tree: Path, tmp_path: Path):
    other_doc = tmp_path / "finance.md"
    other_doc.write_text("# Finance\n\nCapital policy mentions risk appetite.\n", encoding="utf-8")
    api_ingest_path(path=other_doc, record_id="finance-v1", corpus_root=corpus_tree)
    registry = CorpusRegistry(corpus_tree)
    policy = registry.get_record("policy-v1")
    finance = registry.get_record("finance-v1")
    assert policy is not None
    assert finance is not None
    registry.upsert_record(
        replace(
            policy,
            authority_class="corporate_policy",
            status="approved",
            effective_from="2026-01-01",
            effective_until="2026-12-31",
            scope="corpus",
            parser_status="ok",
        )
    )
    registry.upsert_record(
        replace(
            finance,
            authority_class="corporate_policy",
            status="approved",
            effective_from="2026-01-01",
            effective_until="2026-12-31",
            scope="finance",
            parser_status="ok",
        )
    )
    out = api_retrieve_evidence(
        question="Which policy mentions risk appetite?",
        corpus_root=corpus_tree,
        search_all=True,
        source_scope="corpus",
    )
    assert out["record_ids"] == ["policy-v1"]
    assert all(row["record_id"] == "policy-v1" for row in out["evidence"])


def test_corpus_questioning_path_is_read_only_for_corpus_records(corpus_tree: Path):
    """Corpus questioning is inspection-only and does not mutate ingested record metadata."""
    registry = CorpusRegistry(corpus_tree)
    before = registry.get_record("policy-v1")
    assert before is not None
    fake = {
        "choices": [{"message": {"content": "PEF constrains output before generation."}}],
        "aurora": {"governance": "PASS", "flags": []},
    }
    with patch("aurora_lens.corpus.operator_api.post_chat", return_value=fake):
        out = api_ask_corpus(
            question="Which documents mention PEF?",
            record_id="policy-v1",
            corpus_root=corpus_tree,
            proxy="http://mock",
            source_scope="corpus",
        )
    after = registry.get_record("policy-v1")
    assert after == before
    assert out["record_id"] == "policy-v1"
    assert out["support_status"] in {"supported", "partial", "not_supported"}


def test_operator_corpus_root_isolated_from_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cli_root = tmp_path / "cli"
    op_root = tmp_path / "operator"
    monkeypatch.setenv("AURORA_LENS_CORPUS_ROOT", str(cli_root))
    monkeypatch.setenv("AURORA_LENS_OPERATOR_CORPUS_ROOT", str(op_root))

    source = tmp_path / "only-operator.md"
    source.write_text("# Doc\n\nOperator-only content.\n", encoding="utf-8")

    api_ingest_path(path=source, record_id="op-doc", corpus_root=op_root)
    assert api_list_records(corpus_root=cli_root)["count"] == 0
    assert api_list_records()["count"] == 1
    assert api_list_records()["corpus_root"] == str(op_root.resolve())


def test_api_clear_operator_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    op_root = tmp_path / "operator"
    monkeypatch.setenv("AURORA_LENS_OPERATOR_CORPUS_ROOT", str(op_root))
    source = tmp_path / "x.md"
    source.write_text("hello\n", encoding="utf-8")
    api_ingest_path(path=source, record_id="x", corpus_root=op_root)
    assert api_list_records()["count"] == 1
    cleared = api_clear_operator_corpus()
    assert cleared["cleared"] == 1
    assert api_list_records()["count"] == 0


def test_preflight_validate_review_no_record():
    out = preflight_validate_review(question="What is PEF?", record_id="")
    assert out is not None
    _assert_preflight_failure_shape(
        out,
        reason="no selected record",
        question="",
        record_id="",
    )
    assert isinstance(out["review_failure_detail"], str)


def test_preflight_validate_review_no_evidence(corpus_tree: Path):
    question = "quantum entanglement in section 99"
    out = preflight_validate_review(
        question=question,
        record_id="policy-v1",
        corpus_root=corpus_tree,
    )
    assert out is not None
    _assert_preflight_failure_shape(
        out,
        reason="no evidence found",
        question=question,
        record_id="policy-v1",
    )
    assert isinstance(out["review_failure_detail"], str)


def test_preflight_validate_review_no_upstream(corpus_tree: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AURORA_LENS_UPSTREAM_PROVIDER", raising=False)
    registry = CorpusRegistry(corpus_tree)
    record = registry.get_record("policy-v1")
    assert record is not None
    registry.upsert_record(
        replace(
            record,
            authority_class="corporate_policy",
            status="approved",
            effective_from="2026-01-01",
            effective_until="2026-12-31",
            scope="corpus",
            parser_status="ok",
        )
    )
    question = "Which documents mention PEF?"
    out = preflight_validate_review(
        question=question,
        record_id="policy-v1",
        corpus_root=corpus_tree,
    )
    assert out is not None
    _assert_preflight_failure_shape(
        out,
        reason="no upstream configured",
        question=question,
        record_id="policy-v1",
    )
    assert isinstance(out["review_failure_detail"], str)


def test_preflight_validate_review_evidence_inadmissible_draft(corpus_tree: Path):
    registry = CorpusRegistry(corpus_tree)
    record = registry.get_record("policy-v1")
    assert record is not None
    registry.upsert_record(
        replace(
            record,
            authority_class="corporate_policy",
            status="draft",
            effective_from="2026-01-01",
            effective_until="2026-12-31",
            parser_status="ok",
        )
    )
    question = "Which documents mention PEF?"
    out = preflight_validate_review(
        question=question,
        record_id="policy-v1",
        corpus_root=corpus_tree,
    )
    assert out is not None
    _assert_preflight_failure_shape(
        out,
        reason="evidence inadmissible",
        question=question,
        record_id="policy-v1",
    )
    detail = out["review_failure_detail"]
    assert isinstance(detail, dict)
    assert detail == {
        "failure_kind": "authority_missing",
        "reasons": ["record.status='draft' does not establish authority."],
        "record_id": "policy-v1",
        "chunk_ids": ["policy-v1--chunk-0000"],
    }


def test_preflight_validate_review_evidence_unresolved_missing_authority(corpus_tree: Path):
    registry = CorpusRegistry(corpus_tree)
    record = registry.get_record("policy-v1")
    assert record is not None
    registry.upsert_record(
        replace(
            record,
            authority_class=None,
            status="approved",
            effective_from="2026-01-01",
            effective_until="2026-12-31",
            parser_status="ok",
        )
    )
    question = "Which documents mention PEF?"
    out = preflight_validate_review(
        question=question,
        record_id="policy-v1",
        corpus_root=corpus_tree,
    )
    assert out is not None
    _assert_preflight_failure_shape(
        out,
        reason="evidence unresolved",
        question=question,
        record_id="policy-v1",
    )
    detail = out["review_failure_detail"]
    assert isinstance(detail, dict)
    assert detail == {
        "failure_kind": "authority_unknown",
        "reasons": ["authority_class is missing while require_authority=True."],
        "record_id": "policy-v1",
        "chunk_ids": ["policy-v1--chunk-0000"],
    }


def test_api_validate_question_reports_complete(corpus_tree: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    registry = CorpusRegistry(corpus_tree)
    record = registry.get_record("policy-v1")
    assert record is not None
    registry.upsert_record(
        replace(
            record,
            authority_class="corporate_policy",
            status="approved",
            effective_from="2026-01-01",
            effective_until="2026-12-31",
            scope="corpus",
            parser_status="ok",
        )
    )
    fake = {
        "choices": [{"message": {"content": "PEF constrains output before generation."}}],
        "aurora": {"governance": "PASS", "flags": []},
    }
    with patch("aurora_lens.corpus.operator_api.post_chat", return_value=fake):
        out = api_validate_question(
            question="Which documents mention PEF?",
            record_id="policy-v1",
            corpus_root=corpus_tree,
            proxy="http://mock",
        )
    assert out["review_outcome"] == "complete"
    assert out["review_failure_reason"] is None
    assert out["support_status"] == "supported"


def test_upstream_config_status_detects_missing_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AURORA_LENS_UPSTREAM_PROVIDER", raising=False)
    status = upstream_config_status()
    assert status["ok"] is False
    assert status["reason"] == "no upstream configured"


def test_preflight_upstream_review_no_upstream(corpus_tree: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AURORA_LENS_UPSTREAM_PROVIDER", raising=False)
    out = preflight_upstream_review(record_id="policy-v1", corpus_root=corpus_tree)
    assert out is not None
    assert out["last_action"] == "upstream_review"
    assert out["selected_record_id"] == "policy-v1"
    assert out["upstream_review_state"] == "failed"
    assert out["upstream_review_failure_reason"] == "upstream not configured"


def test_preflight_upstream_review_no_record(corpus_tree: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    out = preflight_upstream_review(record_id="missing-doc", corpus_root=corpus_tree)
    assert out is not None
    assert out["upstream_review_state"] == "failed"
    assert out["upstream_review_failure_reason"] == "no selected record"
