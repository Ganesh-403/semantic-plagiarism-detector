"""
tests/core/test_notebook_scan.py
--------------------------------
Unit tests for Jupyter Notebook Cell Execution Graph and Data Lineage Detection.
"""

import pytest
import json
from src.core.notebook_graph_extractor import extract_notebook_graph, _extract_variables
from src.core.data_lineage_aligner import compute_lineage_similarity


class TestNotebookGraphExtractor:
    def test_extract_variables(self):
        source = "x = 10\ny = x + 5\nprint(y)"
        defines, uses = _extract_variables(source)
        assert "x" in defines
        assert "y" in defines
        # 'x' is defined before 'y', so 'y' uses 'x'. But 'x' is defined in same scope.
        # Our simple regex adds to uses if not in defines at the time of match.
        assert "print" not in uses  # built-in

    def test_extract_notebook_graph_valid(self):
        nb_json = json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["x = 1"],
                        "execution_count": 1,
                        "id": "c1",
                    },
                    {
                        "cell_type": "code",
                        "source": ["y = x + 1"],
                        "execution_count": 2,
                        "id": "c2",
                    },
                ]
            }
        )
        graph = extract_notebook_graph(nb_json)
        assert len(graph.cells) == 2
        assert graph.execution_sequence == [1, 2]
        assert len(graph.lineage_edges) > 0

    def test_extract_notebook_graph_invalid_json(self):
        graph = extract_notebook_graph("not valid json")
        assert len(graph.cells) == 0


class TestDataLineageAligner:
    def test_compute_similarity_identical(self):
        nb_json = json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["x = 1"],
                        "execution_count": 1,
                        "id": "c1",
                    },
                    {
                        "cell_type": "code",
                        "source": ["y = x + 1"],
                        "execution_count": 2,
                        "id": "c2",
                    },
                ]
            }
        )
        graph_a = extract_notebook_graph(nb_json)
        graph_b = extract_notebook_graph(nb_json)
        result = compute_lineage_similarity(graph_a, graph_b)
        assert result["overall_score"] == 1.0
        assert result["is_cloned_workflow"] is True

    def test_compute_similarity_different(self):
        nb_a = json.dumps(
            {"cells": [{"cell_type": "code", "source": ["x=1"], "execution_count": 1}]}
        )
        nb_b = json.dumps(
            {
                "cells": [
                    {"cell_type": "code", "source": ["a=1"], "execution_count": 1},
                    {"cell_type": "code", "source": ["b=2"], "execution_count": 2},
                    {"cell_type": "code", "source": ["c=3"], "execution_count": 3},
                ]
            }
        )
        graph_a = extract_notebook_graph(nb_a)
        graph_b = extract_notebook_graph(nb_b)
        result = compute_lineage_similarity(graph_a, graph_b)
        assert result["overall_score"] < 1.0
