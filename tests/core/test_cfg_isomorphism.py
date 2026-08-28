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
tests/core/test_cfg_isomorphism.py
----------------------------------
Unit tests for Control Flow Graph (CFG) Isomorphism and Plagiarism Detection.
"""

import pytest

from src.core.cfg_generator import cfg_to_adjacency_list, compute_cfg_hash, generate_cfg
from src.core.graph_isomorphism_engine import (
    compare_cfgs,
    compute_graph_edit_distance,
    compute_structural_similarity,
)


class TestCFGGenerator:
    """Test suite for CFG generation from Python source code."""

    def test_generate_cfg_basic_function(self):
        """Verify CFG is generated for a basic function."""
        code = """
def foo():
    x = 1
    return x
"""
        blocks = generate_cfg(code)
        assert len(blocks) > 0
        # Should have at least ENTRY, FUNC_DEF, and RETURN blocks
        assert any("FUNC_DEF" in b.statements for b in blocks.values())

    def test_generate_cfg_with_loop(self):
        """Verify CFG captures loop structures."""
        code = """
def foo():
    for i in range(10):
        print(i)
"""
        blocks = generate_cfg(code)
        assert any("FOR" in b.statements for b in blocks.values())

    def test_generate_cfg_syntax_error(self):
        """Verify graceful handling of syntax errors."""
        code = "def foo(:"
        blocks = generate_cfg(code)
        assert len(blocks) == 0

    def test_cfg_hash_determinism(self):
        """Verify identical code produces identical CFG hashes."""
        code = "def foo(): return 1"
        blocks_a = generate_cfg(code)
        blocks_b = generate_cfg(code)

        hash_a = compute_cfg_hash(blocks_a)
        hash_b = compute_cfg_hash(blocks_b)
        assert hash_a == hash_b


class TestGraphIsomorphism:
    """Test suite for graph edit distance and structural similarity."""

    def test_edit_distance_identical_graphs(self):
        """Verify edit distance is 0 for identical adjacency lists."""
        adj = {1: [2], 2: [3], 3: []}
        distance = compute_graph_edit_distance(adj, adj)
        assert distance == 0

    def test_edit_distance_different_graphs(self):
        """Verify edit distance > 0 for different graphs."""
        adj_a = {1: [2], 2: [3], 3: []}
        adj_b = {1: [2, 3], 2: [], 3: []}
        distance = compute_graph_edit_distance(adj_a, adj_b)
        assert distance > 0

    def test_structural_similarity_identical(self):
        """Verify structural similarity is 1.0 for identical graphs."""
        adj = {1: [2], 2: [3], 3: []}
        sim = compute_structural_similarity(adj, adj)
        assert sim == 1.0

    def test_structural_similarity_disjoint(self):
        """Verify structural similarity is low for completely different graphs."""
        adj_a = {1: [2], 2: [3]}
        adj_b = {10: [20], 20: [30]}
        sim = compute_structural_similarity(adj_a, adj_b)
        assert sim < 0.5

    def test_compare_cfgs_exact_clone(self):
        """Verify compare_cfgs flags exact clones."""
        code = "def foo():\n    if True:\n        return 1\n    return 0"
        blocks_a = generate_cfg(code)
        blocks_b = generate_cfg(code)

        result = compare_cfgs(blocks_a, blocks_b)
        assert result["is_exact_clone"] is True
        assert result["structural_similarity"] == 1.0
