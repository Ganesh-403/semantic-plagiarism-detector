"""
src/core/query_plan_aligner.py
------------------------------
SQL Query Plan Alignment Engine.

Computes tree-edit distance and relational algebra similarity between
normalized SQL execution plans to detect cloned database logic.
"""

import logging
from typing import List, Dict, Any
from src.core.sql_ast_extractor import SQLAST, SQLNode

logger = logging.getLogger(__name__)


def compute_ast_edit_distance(nodes_a: List[SQLNode], nodes_b: List[SQLNode]) -> int:
    """Compute Levenshtein distance between two sequences of normalized AST nodes."""
    seq_a = [f"{n.clause_type}:{','.join(n.normalized_tokens)}" for n in nodes_a]
    seq_b = [f"{n.clause_type}:{','.join(n.normalized_tokens)}" for n in nodes_b]

    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return dp[n][m]


def compute_schema_dependency_similarity(ast_a: SQLAST, ast_b: SQLAST) -> float:
    """Compute Jaccard similarity of the normalized schema dependencies."""
    deps_a = ast_a.tables.union(ast_a.columns)
    deps_b = ast_b.tables.union(ast_b.columns)

    if not deps_a and not deps_b:
        return 1.0

    intersection = len(deps_a.intersection(deps_b))
    union = len(deps_a.union(deps_b))

    return intersection / union if union > 0 else 0.0


def compute_sql_similarity(ast_a: SQLAST, ast_b: SQLAST) -> Dict[str, Any]:
    """Compute structural and schema similarity between two SQL queries."""
    if not ast_a.nodes and not ast_b.nodes:
        return {
            "ast_similarity": 1.0,
            "schema_similarity": 1.0,
            "overall_score": 1.0,
            "is_cloned_logic": False,
        }

    edit_dist = compute_ast_edit_distance(ast_a.nodes, ast_b.nodes)
    max_len = max(len(ast_a.nodes), len(ast_b.nodes), 1)
    ast_sim = 1.0 - (edit_dist / max_len)

    schema_sim = compute_schema_dependency_similarity(ast_a, ast_b)

    overall_score = (ast_sim * 0.7) + (schema_sim * 0.3)
    is_cloned = overall_score > 0.85

    return {
        "ast_edit_distance": edit_dist,
        "ast_similarity": round(ast_sim, 4),
        "schema_similarity": round(schema_sim, 4),
        "overall_score": round(overall_score, 4),
        "is_cloned_logic": is_cloned,
    }
