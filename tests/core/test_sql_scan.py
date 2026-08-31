"""
tests/core/test_sql_scan.py
---------------------------
Unit tests for SQL Query Execution Plan and Schema Dependency Plagiarism Detection.
"""

import pytest
from src.core.sql_ast_extractor import extract_sql_ast
from src.core.query_plan_aligner import compute_sql_similarity


class TestSQLASTExtractor:
    def test_extract_simple_select(self):
        query = "SELECT id, name FROM users WHERE active = 1"
        ast = extract_sql_ast(query)
        assert len(ast.nodes) >= 2
        assert "TBL_1" in ast.tables
        assert "COL_1" in ast.columns

    def test_normalize_aliases(self):
        query_a = "SELECT u.id FROM users u"
        query_b = "SELECT x.id FROM users x"
        ast_a = extract_sql_ast(query_a)
        ast_b = extract_sql_ast(query_b)
        # Both should normalize 'users' to TBL_1 and 'id' to COL_1
        assert ast_a.tables == ast_b.tables
        assert ast_a.columns == ast_b.columns


class TestQueryPlanAligner:
    def test_compute_similarity_identical_logic(self):
        query_a = "SELECT a, b FROM t1 WHERE a > 10 ORDER BY b"
        query_b = "SELECT x, y FROM t2 WHERE x > 10 ORDER BY y"
        ast_a = extract_sql_ast(query_a)
        ast_b = extract_sql_ast(query_b)
        result = compute_sql_similarity(ast_a, ast_b)
        assert result["overall_score"] > 0.9
        assert result["is_cloned_logic"] is True

    def test_compute_similarity_different_logic(self):
        query_a = "SELECT a FROM t1"
        query_b = (
            "SELECT x, y, z FROM t2 JOIN t3 ON t2.id = t3.id WHERE x LIKE '%test%'"
        )
        ast_a = extract_sql_ast(query_a)
        ast_b = extract_sql_ast(query_b)
        result = compute_sql_similarity(ast_a, ast_b)
        assert result["overall_score"] < 0.5
        assert result["is_cloned_logic"] is False
