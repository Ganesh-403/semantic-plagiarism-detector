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
tests/core/test_layout_scan.py
------------------------------
Unit tests for Document Structural Layout and Formatting Plagiarism Detection.
"""

import pytest

from src.core.formatting_similarity_engine import (
    compute_layout_similarity,
    tree_to_sequence,
)
from src.core.layout_tree_extractor import (
    LayoutNode,
    parse_html_layout,
    parse_markdown_layout,
)


class TestLayoutTreeExtractor:
    """Test suite for layout tree extraction."""

    def test_parse_html_layout(self):
        """Verify HTML layout is parsed into a tree."""
        html = "<h1>Title</h1><p>Text</p><h2>Sub</h2>"
        tree = parse_html_layout(html)
        assert tree.tag == "ROOT"
        assert len(tree.children) == 3
        assert tree.children[0].tag == "H1"

    def test_parse_markdown_layout(self):
        """Verify Markdown layout is parsed into a tree."""
        md = "# Title\n\nParagraph\n\n## Sub"
        tree = parse_markdown_layout(md)
        assert tree.tag == "ROOT"
        assert len(tree.children) == 3
        assert tree.children[0].tag == "H1"
        assert tree.children[2].tag == "H2"


class TestFormattingSimilarityEngine:
    """Test suite for formatting similarity engine."""

    def test_tree_to_sequence(self):
        """Verify tree flattening works correctly."""
        root = LayoutNode(
            tag="ROOT", children=[LayoutNode(tag="H1"), LayoutNode(tag="P")]
        )
        seq = tree_to_sequence(root)
        assert seq == ["ROOT", "H1", "P"]

    def test_compute_layout_similarity_identical(self):
        """Verify identical layouts have similarity 1.0."""
        tree_a = parse_markdown_layout("# Title\n\nText")
        tree_b = parse_markdown_layout("# Title\n\nText")
        result = compute_layout_similarity(tree_a, tree_b)
        assert result["structural_similarity"] == 1.0
        assert result["is_structural_clone"] is True

    def test_compute_layout_similarity_different(self):
        """Verify different layouts have lower similarity."""
        tree_a = parse_markdown_layout("# Title\n\nText")
        tree_b = parse_markdown_layout("## Sub\n\n## Sub2\n\n## Sub3")
        result = compute_layout_similarity(tree_a, tree_b)
        assert result["structural_similarity"] < 1.0
