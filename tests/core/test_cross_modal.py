"""
tests/core/test_cross_modal.py
------------------------------
Unit tests for Cross-Modal Semantic Alignment.
"""

import pytest
from src.core.pseudocode_parser import extract_logical_blocks, LogicalBlock
from src.core.cross_modal_aligner import (
    extract_code_logical_blocks,
    compute_cross_modal_similarity,
)


class TestPseudocodeParser:
    def test_extract_logical_blocks_loop(self):
        text = "For each item in the list, print the item."
        blocks = extract_logical_blocks(text)
        assert len(blocks) >= 1
        assert blocks[0].block_type == "loop"

    def test_extract_logical_blocks_conditional(self):
        text = "If the value is greater than 10, return true."
        blocks = extract_logical_blocks(text)
        assert len(blocks) >= 1
        assert blocks[0].block_type == "conditional"


class TestCrossModalAligner:
    def test_extract_code_logical_blocks(self):
        code = "for i in range(10):\n    print(i)"
        blocks = extract_code_logical_blocks(code)
        assert len(blocks) == 2
        assert blocks[0].block_type == "loop"
        assert blocks[1].block_type == "io"

    def test_compute_cross_modal_similarity_identical_flow(self):
        text_blocks = [
            LogicalBlock("loop", "for each item", ["for", "each", "item"]),
            LogicalBlock("io", "print item", ["print", "item"]),
        ]
        code_blocks = [
            LogicalBlock(
                "loop", "for i in range(10):", ["for", "i", "in", "range", "10"]
            ),
            LogicalBlock("io", "print(i)", ["print", "i"]),
        ]
        result = compute_cross_modal_similarity(text_blocks, code_blocks)
        assert result["structural_similarity"] == 1.0
        assert result["is_translation"] is True

    def test_compute_cross_modal_similarity_different_flow(self):
        text_blocks = [LogicalBlock("loop", "for each", ["for", "each"])]
        code_blocks = [LogicalBlock("conditional", "if x > 0", ["if", "x", "0"])]
        result = compute_cross_modal_similarity(text_blocks, code_blocks)
        assert result["structural_similarity"] < 1.0
