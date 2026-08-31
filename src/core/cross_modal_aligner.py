"""
src/core/cross_modal_aligner.py
-------------------------------
Cross-Modal Semantic Alignment Engine.

Computes semantic equivalence between natural language algorithmic descriptions
and source code implementations by aligning normalized logical blocks with
code AST/CFG basic blocks.
"""

import re
import math
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter

from src.core.pseudocode_parser import LogicalBlock

logger = logging.getLogger(__name__)


def extract_code_logical_blocks(code: str) -> List[LogicalBlock]:
    """Extract logical blocks from source code using regex heuristics.

    This is a lightweight proxy for full AST/CFG parsing. It identifies
    loops, conditionals, and assignments based on language keywords.
    """
    if not code:
        return []

    # Remove comments
    code = re.sub(r"#.*", "", code)  # Python comments
    code = re.sub(r"//.*", "", code)  # C-style single line
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)  # C-style multi-line

    lines = code.split("\n")
    blocks = []

    loop_kw = re.compile(r"\b(for|while)\b")
    cond_kw = re.compile(r"\b(if|elif|else|switch|case)\b")
    assign_kw = re.compile(r"(\s*=\s*|def |function )")
    io_kw = re.compile(r"\b(print|return|yield|input|read|write)\b")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        block_type = "control"
        if loop_kw.search(line):
            block_type = "loop"
        elif cond_kw.search(line):
            block_type = "conditional"
        elif assign_kw.search(line):
            block_type = "assignment"
        elif io_kw.search(line):
            block_type = "io"

        tokens = re.findall(r"\b\w+\b", line.lower())
        blocks.append(
            LogicalBlock(block_type=block_type, content=line, normalized_tokens=tokens)
        )

    return blocks


def compute_cross_modal_similarity(
    text_blocks: List[LogicalBlock], code_blocks: List[LogicalBlock]
) -> Dict[str, Any]:
    """Compute cross-modal semantic similarity between text and code blocks.

    Aligns the sequence of logical block types and computes token overlap
    to determine if the code implements the described algorithm.

    Args:
        text_blocks: Logical blocks from natural language/pseudocode.
        code_blocks: Logical blocks from source code.

    Returns:
        Dictionary containing structural and semantic similarity scores.
    """
    if not text_blocks or not code_blocks:
        return {
            "structural_similarity": 0.0,
            "semantic_similarity": 0.0,
            "overall_score": 0.0,
            "is_translation": False,
        }

    # 1. Structural Similarity (Sequence of block types)
    seq_text = [b.block_type for b in text_blocks]
    seq_code = [b.block_type for b in code_blocks]

    # Compute Levenshtein distance for structural alignment
    n, m = len(seq_text), len(seq_code)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_text[i - 1] == seq_code[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    edit_distance = dp[n][m]
    max_len = max(n, m, 1)
    structural_sim = 1.0 - (edit_distance / max_len)

    # 2. Semantic Similarity (Token overlap across all blocks)
    tokens_text = set(t for b in text_blocks for t in b.normalized_tokens)
    tokens_code = set(t for b in code_blocks for t in b.normalized_tokens)

    # Filter out common stop words to improve signal
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "for",
        "in",
        "to",
        "and",
        "or",
        "if",
        "else",
    }
    tokens_text -= stop_words
    tokens_code -= stop_words

    intersection = len(tokens_text.intersection(tokens_code))
    union = len(tokens_text.union(tokens_code))
    semantic_sim = intersection / union if union > 0 else 0.0

    # Overall score weights structural flow higher than exact token matches
    overall_score = (structural_sim * 0.6) + (semantic_sim * 0.4)

    # Flag as translation if structural flow is highly preserved
    is_translation = structural_sim > 0.70 and semantic_sim > 0.20

    return {
        "structural_similarity": round(structural_sim, 4),
        "semantic_similarity": round(semantic_sim, 4),
        "overall_score": round(overall_score, 4),
        "is_translation": is_translation,
    }
