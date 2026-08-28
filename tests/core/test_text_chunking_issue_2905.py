# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
