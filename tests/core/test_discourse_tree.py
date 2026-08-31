"""
tests/core/test_discourse_tree.py
---------------------------------
Unit tests for Hierarchical Discourse Tree and Rhetorical Structure Detection.
"""

import pytest
from src.core.discourse_tree_parser import parse_discourse_tree, classify_paragraph
from src.core.rhetorical_structure_aligner import (
    compute_rhetorical_alignment,
    extract_rhetorical_sequence,
)


class TestDiscourseTreeParser:
    def test_classify_paragraph_claim(self):
        para = "We argue that this method is superior."
        assert classify_paragraph(para) == "CLAIM"

    def test_classify_paragraph_evidence(self):
        para = "For example, studies show that..."
        assert classify_paragraph(para) == "EVIDENCE"

    def test_classify_paragraph_rebuttal(self):
        para = "However, critics argue that..."
        assert classify_paragraph(para) == "REBUTTAL"

    def test_classify_paragraph_conclusion(self):
        para = "In conclusion, the results are clear."
        assert classify_paragraph(para) == "CONCLUSION"

    def test_parse_discourse_tree(self):
        text = "In this paper we propose X.\n\nFor example, Y.\n\nHowever, Z.\n\nIn conclusion, W."
        tree = parse_discourse_tree(text)
        assert len(tree.children) == 4
        assert tree.children[0].node_type == "INTRODUCTION"
        assert tree.children[3].node_type == "CONCLUSION"


class TestRhetoricalStructureAligner:
    def test_extract_rhetorical_sequence(self):
        tree = parse_discourse_tree("Intro.\n\nClaim.\n\nConclusion.")
        seq = extract_rhetorical_sequence(tree)
        assert "DOCUMENT" in seq
        assert "INTRODUCTION" in seq
        assert "CONCLUSION" in seq

    def test_compute_alignment_identical(self):
        text = "Intro.\n\nClaim.\n\nEvidence.\n\nConclusion."
        tree_a = parse_discourse_tree(text)
        tree_b = parse_discourse_tree(text)
        result = compute_rhetorical_alignment(tree_a, tree_b)
        assert result["structural_similarity"] == 1.0
        assert result["is_structural_plagiarism"] is True

    def test_compute_alignment_different(self):
        text_a = "Intro.\n\nClaim.\n\nConclusion."
        text_b = "Claim.\n\nRebuttal.\n\nEvidence.\n\nEvidence.\n\nConclusion."
        tree_a = parse_discourse_tree(text_a)
        tree_b = parse_discourse_tree(text_b)
        result = compute_rhetorical_alignment(tree_a, tree_b)
        assert result["structural_similarity"] < 1.0
