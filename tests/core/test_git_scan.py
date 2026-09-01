"""
tests/core/test_git_scan.py
---------------------------
Unit tests for Git Commit Graph and Covert Collaboration Detection.
"""

import pytest
from src.core.git_graph_extractor import parse_git_log, compute_timezone_entropy
from src.core.covert_collaboration_analyzer import analyze_covert_collaboration

MOCK_LOG = """commit a1b2c3d4e5f6
Author: Student A
Date:   Thu Jan 1 12:00:00 2024 +0000

    Initial commit
 src/main.py | 10 ++++++++++

commit b2c3d4e5f6a1
Author: Student B
Date:   Thu Jan 1 12:05:00 2024 +0000

    Fix bug
 src/main.py | 2 +-
"""


class TestGitGraphExtractor:
    def test_parse_git_log(self):
        graph = parse_git_log(MOCK_LOG)
        assert len(graph.commits) == 2
        assert graph.commits[0].author == "Student A"
        assert len(graph.edges) == 1

    def test_compute_timezone_entropy(self):
        graph = parse_git_log(MOCK_LOG)
        metrics = compute_timezone_entropy(graph)
        assert metrics["tz_entropy"] == 0.0  # Only one timezone
        assert metrics["author_entropy"] > 0.0  # Two authors


class TestCovertCollaborationAnalyzer:
    def test_analyze_covert_collaboration(self):
        graph_a = parse_git_log(MOCK_LOG)
        graph_b = parse_git_log(MOCK_LOG)
        result = analyze_covert_collaboration(graph_a, graph_b)
        assert result["graph_similarity"] == 1.0
        assert "overall_score" in result
