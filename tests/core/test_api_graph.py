"""
tests/core/test_api_graph.py
----------------------------
Unit tests for API Call Graph and External Dependency Plagiarism Detection.
"""

import pytest
from src.core.api_call_graph_extractor import extract_python_api_graph
from src.core.dependency_chain_aligner import compute_api_graph_similarity


class TestAPICallGraphExtractor:
    def test_extract_python_api_graph_imports(self):
        code = "import math\nimport os\nmath.sqrt(16)\nos.path.join('a', 'b')"
        graph = extract_python_api_graph(code)
        assert "math.sqrt" in graph.nodes
        assert "os.path.join" in graph.nodes
        assert len(graph.call_sequence) == 2

    def test_extract_python_api_graph_from_imports(self):
        code = "from math import sqrt\nsqrt(16)"
        graph = extract_python_api_graph(code)
        assert "math.sqrt" in graph.nodes

    def test_extract_python_api_graph_syntax_error(self):
        code = "import math\nmath.sqrt("
        graph = extract_python_api_graph(code)
        assert len(graph.nodes) == 0


class TestDependencyChainAligner:
    def test_compute_similarity_identical(self):
        code = "import math\nmath.sqrt(16)\nmath.pow(2, 3)"
        graph_a = extract_python_api_graph(code)
        graph_b = extract_python_api_graph(code)
        result = compute_api_graph_similarity(graph_a, graph_b)
        assert result["overall_score"] == 1.0
        assert result["is_clone"] is True

    def test_compute_similarity_different(self):
        code_a = "import math\nmath.sqrt(16)"
        code_b = "import os\nos.getcwd()"
        graph_a = extract_python_api_graph(code_a)
        graph_b = extract_python_api_graph(code_b)
        result = compute_api_graph_similarity(graph_a, graph_b)
        assert result["node_similarity"] == 0.0
        assert result["is_clone"] is False
