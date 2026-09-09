"""Regression tests for Issue #2905."""

import ast
from pathlib import Path

from src.core import text_chunking

SOURCE = Path(text_chunking.__file__).read_text(encoding="utf-8")


def test_nltk_import_is_module_level():
    tree = ast.parse(SOURCE)
    assert any(
        isinstance(node, ast.Try)
        and any(
            isinstance(stmt, ast.Import)
            and any(alias.name == "nltk" for alias in stmt.names)
            for stmt in node.body
        )
        for node in tree.body
    )
    splitter = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_split_into_sentences"
    )
    assert not any(
        isinstance(node, ast.Import)
        and any(alias.name == "nltk" for alias in node.names)
        for node in ast.walk(splitter)
    )


def test_regex_fallback_still_works_without_nltk():
    original_nltk = text_chunking.nltk
    try:
        text_chunking.nltk = None
        assert text_chunking._split_into_sentences(
            "First sentence. Second sentence!"
        ) == ["First sentence.", "Second sentence!"]
    finally:
        text_chunking.nltk = original_nltk
