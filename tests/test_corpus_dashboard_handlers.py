"""Static checks for corpus console action wiring in dashboard.html."""

from __future__ import annotations

from pathlib import Path


def _dashboard_html() -> str:
    return (Path(__file__).resolve().parents[1] / "aurora_lens" / "proxy" / "dashboard.html").read_text(
        encoding="utf-8"
    )


def test_validate_button_posts_corpus_validate_endpoint():
    html = _dashboard_html()
    assert "corpus-btn-validate" in html
    assert "endpoint: '/v1/corpus/validate'" in html
    assert "corpusRunAction" in html
    assert "Review / Validate requires one selected document." in html


def test_ask_button_posts_corpus_ask_endpoint():
    html = _dashboard_html()
    assert "corpus-btn-ask" in html
    assert "endpoint: '/v1/corpus/ask'" in html
    assert "Ask / Governed clicked" in html


def test_corpus_actions_log_to_console_and_use_fetch():
    html = _dashboard_html()
    assert "console.info('[corpus]" in html
    assert "console.error('[corpus]" in html
    assert "fetch(corpusApiBase()" in html
    assert ".finally(function()" in html


def test_evidence_sections_rendered_in_dashboard():
    html = _dashboard_html()
    assert "renderEvidencePayload" in html
    assert "Strong evidence" in html
    assert "Show low relevance results" in html
    assert "corpus-endpoint" in html
