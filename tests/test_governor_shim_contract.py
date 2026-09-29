"""Ensure top-level ``governor/`` stays thin shims to ``aurora_lens.governor``.

Drift prevention: policy and matrix logic must not reappear under ``governor/*.py``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GOV_DIR = _REPO_ROOT / "governor"

# Modules that must remain re-export-only (no duplicate implementations).
_SHIM_MODULES = (
    "models",
    "resolver",
    "forensic_schema",
    "audit",
)


def _module_ast(name: str) -> ast.Module:
    path = _GOV_DIR / f"{name}.py"
    return ast.parse(path.read_text(encoding="utf-8"))


def _is_module_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(
        node.value, (ast.Constant, ast.JoinedStr)
    )


@pytest.mark.parametrize("mod", _SHIM_MODULES)
def test_governor_shim_is_star_reexport_only(mod: str) -> None:
    tree = _module_ast(mod)
    from_aurora: list[ast.ImportFrom] = []
    for node in tree.body:
        if _is_module_docstring(node):
            continue
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            assert node.module and node.module.startswith(
                "aurora_lens.governor"
            ), f"{mod}: disallowed import from {node.module!r}"
            from_aurora.append(node)
        else:
            pytest.fail(f"{mod}: disallowed top-level {type(node).__name__}")
    assert from_aurora, f"{mod}: missing aurora_lens.governor import"


def test_continuation_matrix_shim_has_no_matrix_table_literal() -> None:
    text = (_GOV_DIR / "continuation_matrix.py").read_text(encoding="utf-8")
    assert '"general:GP:STOP"' not in text
    assert "CONTINUATION_MATRIX" in text
    assert "aurora_lens.governor.continuation_matrix" in text


def test_operator_override_shim_has_no_field_classification_literals() -> None:
    text = (_GOV_DIR / "operator_override.py").read_text(encoding="utf-8")
    assert "IMMUTABLE_FIELDS" in text
    assert "frozenset({" not in text
    assert "aurora_lens.governor.operator_override" in text
