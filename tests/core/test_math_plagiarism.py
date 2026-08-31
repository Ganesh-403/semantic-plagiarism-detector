"""
tests/core/test_math_plagiarism.py
----------------------------------
Unit tests for Mathematical Equation and Proof Structural Plagiarism Detection.
"""

import pytest
from src.core.equation_ast_extractor import (
    extract_equation_ast,
    tokenize_latex,
    normalize_variables,
)
from src.core.math_structure_aligner import (
    compute_math_similarity,
    extract_preorder_sequence,
)


class TestEquationASTExtractor:
    def test_tokenize_latex(self):
        tokens = tokenize_latex("$x^2 + y^2 = z^2$")
        assert "x" in tokens
        assert "^" in tokens
        assert "2" in tokens
        assert "+" in tokens

    def test_normalize_variables(self):
        tokens = ["x", "+", "y", "=", "z"]
        normalized = normalize_variables(tokens)
        assert "VAR_1" in normalized
        assert "VAR_2" in normalized
        assert "VAR_3" in normalized
        assert "+" in normalized

    def test_extract_equation_ast(self):
        tree = extract_equation_ast(r"\frac{a}{b} + c")
        assert tree.node_type == "GROUP"
        assert len(tree.children) > 0


class TestMathStructureAligner:
    def test_extract_preorder_sequence(self):
        tree = extract_equation_ast("x + y")
        seq = extract_preorder_sequence(tree)
        assert len(seq) > 0
        assert any("VAR" in s for s in seq)

    def test_compute_similarity_identical(self):
        tree_a = extract_equation_ast("x^2 + y^2 = z^2")
        tree_b = extract_equation_ast(
            "a^2 + b^2 = c^2"
        )  # Same structure, different vars
        result = compute_math_similarity(tree_a, tree_b)
        assert result["structural_similarity"] == 1.0
        assert result["is_structural_plagiarism"] is True

    def test_compute_similarity_different(self):
        tree_a = extract_equation_ast("x + y")
        tree_b = extract_equation_ast(r"\int_{0}^{1} x^2 dx")
        result = compute_math_similarity(tree_a, tree_b)
        assert result["structural_similarity"] < 0.5
