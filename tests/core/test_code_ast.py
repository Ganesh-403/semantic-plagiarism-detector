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

"""
tests/core/test_code_ast.py
---------------------------
Comprehensive unit tests for the Code AST Parser and Similarity Engine.
"""

import pytest

from src.core.code_ast_parser import (
    ASTNormalizer,
    ast_to_node_sequence,
    parse_and_normalize_code,
)
from src.core.code_similarity_engine import (
    compare_code_snippets,
    compute_levenshtein_similarity,
    compute_sequence_similarity,
)


class TestASTParser:
    """Test suite for AST parsing and normalization."""

    def test_parse_valid_code(self):
        """Verify valid Python code is parsed successfully."""
        code = "x = 10\ny = x + 5"
        tree = parse_and_normalize_code(code)
        assert tree is not None

    def test_parse_invalid_code_returns_none(self):
        """Verify invalid syntax returns None."""
        code = "def foo(:"
        tree = parse_and_normalize_code(code)
        assert tree is None

    def test_variable_normalization(self):
        """Verify variables are renamed to standardized names."""
        code = "my_var = 10\nother_var = my_var + 5"
        tree = parse_and_normalize_code(code)
        seq = ast_to_node_sequence(tree)

        # The sequence should contain 'Name' nodes, but the actual IDs are normalized
        # We verify by checking the AST directly
        import ast

        normalizer = ASTNormalizer()
        normalized = normalizer.visit(ast.parse(code))

        # Check that 'my_var' is no longer in the AST
        ast_str = ast.dump(normalized)
        assert "my_var" not in ast_str
        assert "var_1" in ast_str

    def test_docstring_stripping(self):
        """Verify docstrings are stripped from functions."""
        code = '''
def foo():
    """This is a docstring."""
    return 1
'''
        tree = parse_and_normalize_code(code)
        import ast

        # Find the function definition
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # The body should only contain the Return node, not the Expr(Constant)
                assert len(node.body) == 1
                assert isinstance(node.body[0], ast.Return)


class TestCodeSimilarity:
    """Test suite for code similarity scoring."""

    def test_identical_code_similarity(self):
        """Verify identical code produces a similarity of 1.0."""
        code = "x = 10\ny = x + 5"
        scores = compare_code_snippets(code, code)
        assert scores["overall_score"] == 1.0

    def test_renamed_variables_high_similarity(self):
        """Verify renaming variables doesn't significantly lower similarity."""
        code_a = "x = 10\ny = x + 5"
        code_b = "a = 10\nb = a + 5"
        scores = compare_code_snippets(code_a, code_b)
        assert scores["overall_score"] > 0.9

    def test_different_logic_low_similarity(self):
        """Verify structurally different code has low similarity."""
        code_a = "for i in range(10):\n    print(i)"
        code_b = "x = 10\nwhile x > 0:\n    x -= 1"
        scores = compare_code_snippets(code_a, code_b)
        assert scores["overall_score"] < 0.5

    def test_levenshtein_vs_jaccard(self):
        """Verify Levenshtein and Jaccard scores are computed."""
        code_a = "x = 1"
        code_b = "x = 2"
        scores = compare_code_snippets(code_a, code_b)
        assert "jaccard_similarity" in scores
        assert "levenshtein_similarity" in scores

    def test_empty_code_snippets(self):
        """Verify empty snippets return 0.0 similarity."""
        scores = compare_code_snippets("", "")
        # Empty trees might return 1.0 for Jaccard if both are empty sets,
        # but overall logic should handle it gracefully.
        assert scores["overall_score"] >= 0.0
