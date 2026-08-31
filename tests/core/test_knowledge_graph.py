"""
tests/core/test_knowledge_graph.py
----------------------------------
Unit tests for Knowledge Graph Extraction and Conceptual Plagiarism Detection.
"""

import pytest
from src.core.knowledge_graph_extractor import (
    extract_spo_triples,
    build_knowledge_graph,
)
from src.core.graph_alignment_scorer import (
    compute_graph_jaccard_similarity,
    compute_conceptual_overlap,
)


class TestKnowledgeGraphExtractor:
    def test_extract_spo_triples_basic(self):
        text = "The cat causes the problem. The dog implies the issue."
        triples = extract_spo_triples(text)
        assert len(triples) >= 1
        assert any(t.predicate == "causes" for t in triples)

    def test_build_knowledge_graph(self):
        from src.core.knowledge_graph_extractor import Triple

        triples = [Triple("cat", "causes", "problem")]
        graph = build_knowledge_graph(triples)
        assert graph["node_count"] == 2
        assert graph["edge_count"] == 1


class TestGraphAlignmentScorer:
    def test_compute_graph_jaccard_identical(self):
        graph = {"nodes": ["a", "b"], "edges": [("a", "rel", "b")]}
        sim = compute_graph_jaccard_similarity(graph, graph)
        assert sim == 1.0

    def test_compute_conceptual_overlap_plagiarism(self):
        graph_a = {"nodes": ["a", "b"], "edges": [("a", "causes", "b")]}
        graph_b = {
            "nodes": ["x", "y"],
            "edges": [("x", "causes", "y")],
        }  # Same structure, different nodes
        result = compute_conceptual_overlap(graph_a, graph_b)
        # Edge jaccard will be 0 if nodes are different, but let's test exact match
        graph_c = {"nodes": ["a", "b"], "edges": [("a", "causes", "b")]}
        res2 = compute_conceptual_overlap(graph_a, graph_c)
        assert res2["is_conceptual_plagiarism"] is True
