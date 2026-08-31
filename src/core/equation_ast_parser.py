"""
src/core/equation_ast_parser.py
-------------------------------
Equation AST Parser for Mathematical Plagiarism Detection.

Parses LaTeX/math blocks into normalized structural trees to detect
copied mathematical proofs and equations regardless of variable renaming.
"""

import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def tokenize_latex(latex_str: str) -> List[str]:
    """Tokenize a LaTeX string into structural components."""
    # Simplified tokenizer: splits on commands, braces, and operators
    tokens = re.findall(r"\\[a-zA-Z]+|[{}^_+\-*/=]|[a-zA-Z0-9]+", latex_str)
    return tokens


def normalize_equation_ast(tokens: List[str]) -> List[str]:
    """Normalize equation tokens to detect structural cloning.

    Replaces specific variable names with generic placeholders (e.g., VAR_1)
    to ensure that renaming variables doesn't bypass the detector.
    """
    normalized = []
    var_map = {}
    var_counter = 0

    for token in tokens:
        # If it's a single letter (likely a variable) and not a known command
        if re.match(r"^[a-zA-Z]$", token) and not token.startswith("\\"):
            if token not in var_map:
                var_counter += 1
                var_map[token] = f"VAR_{var_counter}"
            normalized.append(var_map[token])
        else:
            normalized.append(token)

    return normalized


def compute_tree_edit_distance(seq_a: List[str], seq_b: List[str]) -> int:
    """Compute the Levenshtein distance between two normalized token sequences.

    This serves as a proxy for tree-edit distance on the flattened AST.
    """
    n = len(seq_a)
    m = len(seq_b)

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
