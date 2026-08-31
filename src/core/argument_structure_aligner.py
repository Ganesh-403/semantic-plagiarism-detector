"""
src/core/argument_structure_aligner.py
--------------------------------------
Argument Structure Alignment Engine.

Computes alignment scores between semantic role graphs to detect
deep semantic paraphrasing and idea theft.
"""

import logging
from typing import List, Dict, Any
from src.core.semantic_role_extractor import SemanticTriple

logger = logging.getLogger(__name__)


def compute_role_sequence_similarity(
    triples_a: List[SemanticTriple], triples_b: List[SemanticTriple]
) -> Dict[str, Any]:
    """Compute sequence alignment between two lists of semantic triples.

    Uses Levenshtein distance on the sequence of role types (Agent, Action, Patient)
    to determine structural alignment, ignoring exact lexical matches.

    Args:
        triples_a: Semantic triples from document A.
        triples_b: Semantic triples from document B.

    Returns:
        Dictionary containing structural similarity scores.
    """
    if not triples_a and not triples_b:
        return {"structural_similarity": 1.0, "is_paraphrase": False}
    if not triples_a or not triples_b:
        return {"structural_similarity": 0.0, "is_paraphrase": False}

    # Extract role sequences (e.g., ['A', 'V', 'P'])
    seq_a = []
    for t in triples_a:
        if t.agent:
            seq_a.append("A")
        if t.action:
            seq_a.append("V")
        if t.patient:
            seq_a.append("P")

    seq_b = []
    for t in triples_b:
        if t.agent:
            seq_b.append("A")
        if t.action:
            seq_b.append("V")
        if t.patient:
            seq_b.append("P")

    # Levenshtein distance
    n, m = len(seq_a), len(seq_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    edit_distance = dp[n][m]
    max_len = max(n, m, 1)
    structural_sim = 1.0 - (edit_distance / max_len)

    # Check lexical overlap of the actual role texts
    tokens_a = set(
        t
        for tr in triples_a
        for role in [tr.agent, tr.action, tr.patient]
        if role
        for t in role.normalized_tokens
    )
    tokens_b = set(
        t
        for tr in triples_b
        for role in [tr.agent, tr.action, tr.patient]
        if role
        for t in role.normalized_tokens
    )

    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    lexical_sim = intersection / union if union > 0 else 0.0

    # Deep paraphrase: High structural similarity, low lexical similarity
    is_deep_paraphrase = structural_sim > 0.75 and lexical_sim < 0.30

    return {
        "structural_similarity": round(structural_sim, 4),
        "lexical_similarity": round(lexical_sim, 4),
        "is_deep_paraphrase": is_deep_paraphrase,
    }
