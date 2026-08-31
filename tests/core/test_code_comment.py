"""
tests/core/test_code_comment.py
-------------------------------
Unit tests for Code Comment and Docstring Semantic Alignment Detection.
"""

import pytest
from src.core.docstring_extractor import extract_python_blocks, extract_generic_blocks
from src.core.code_comment_aligner import compute_coherence_score


class TestDocstringExtractor:
    def test_extract_python_blocks_docstring(self):
        code = '''def foo():
    """This is a docstring."""
    return 1'''
        blocks = extract_python_blocks(code)
        assert len(blocks) >= 1
        assert any("docstring" in b.comment_text for b in blocks)

    def test_extract_python_blocks_inline(self):
        code = "x = 1 # Initialize x"
        blocks = extract_python_blocks(code)
        assert len(blocks) >= 1
        assert any("Initialize x" in b.comment_text for b in blocks)

    def test_extract_generic_blocks(self):
        code = "int x = 1; // Initialize x"
        blocks = extract_generic_blocks(code)
        assert len(blocks) >= 1
        assert any("Initialize x" in b.comment_text for b in blocks)


class TestCodeCommentAligner:
    def test_compute_coherence_high(self):
        from src.core.docstring_extractor import CodeBlock

        blocks = [
            CodeBlock("calculate_sum", "Calculate the sum of two numbers", 1),
            CodeBlock("return a + b", "Return the result", 2),
        ]
        result = compute_coherence_score(blocks)
        assert result["overall_coherence"] > 0.0
        assert result["is_mismatch"] is False

    def test_compute_coherence_low(self):
        from src.core.docstring_extractor import CodeBlock

        blocks = [
            CodeBlock("calculate_sum", "The quick brown fox jumps over the lazy dog", 1)
        ]
        result = compute_coherence_score(blocks)
        assert result["overall_coherence"] < 0.15
        assert result["is_mismatch"] is True
