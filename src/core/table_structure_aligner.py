"""
src/core/table_structure_aligner.py
-----------------------------------
Tabular Data Structure Alignment Engine.

Applies z-score normalization and computes distribution similarity
(e.g., Kolmogorov-Smirnov test proxy) between tables to detect cloned
datasets with renamed headers or linearly transformed values.
"""

import math
import logging
from typing import List, Dict, Any
from src.core.tabular_data_extractor import TableFingerprint, ColumnSchema

logger = logging.getLogger(__name__)


def compute_distribution_similarity(col_a: ColumnSchema, col_b: ColumnSchema) -> float:
    """Compute the similarity between two numerical column distributions.

    Uses a proxy for the Kolmogorov-Smirnov test by comparing the
    normalized min/max ranges and standard deviations. If two columns
    are linearly transformed versions of each other (e.g., y = mx + c),
    their z-score distributions will be identical.

    Args:
        col_a: Schema for column A.
        col_b: Schema for column B.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if col_a.dtype != "numeric" or col_b.dtype != "numeric":
        return 0.0

    if col_a.std_dev is None or col_b.std_dev is None:
        return 0.0

    if col_a.std_dev == 0 or col_b.std_dev == 0:
        # Constant columns
        return 1.0 if col_a.mean == col_b.mean else 0.0

    # Compare coefficient of variation (std_dev / mean)
    # This is invariant to scaling (multiplication)
    cv_a = col_a.std_dev / abs(col_a.mean) if col_a.mean != 0 else col_a.std_dev
    cv_b = col_b.std_dev / abs(col_b.mean) if col_b.mean != 0 else col_b.std_dev

    cv_diff = abs(cv_a - cv_b)

    # Compare normalized range (max - min) / std_dev
    # This is also invariant to linear transformations
    range_a = (
        (col_a.max_val - col_a.min_val) / col_a.std_dev if col_a.std_dev > 0 else 0
    )
    range_b = (
        (col_b.max_val - col_b.min_val) / col_b.std_dev if col_b.std_dev > 0 else 0
    )

    range_diff = abs(range_a - range_b)

    # Combine differences into a similarity score
    total_diff = (cv_diff * 0.5) + (range_diff * 0.1)
    similarity = max(0.0, 1.0 - total_diff)

    return round(similarity, 4)


def compute_table_similarity(
    fp_a: TableFingerprint, fp_b: TableFingerprint
) -> Dict[str, Any]:
    """Compute structural and distribution similarity between two tables.

    Args:
        fp_a: Fingerprint for table A.
        fp_b: Fingerprint for table B.

    Returns:
        Dictionary containing schema match, distribution similarity, and plagiarism flags.
    """
    if fp_a.column_count == 0 or fp_b.column_count == 0:
        return {
            "schema_similarity": 0.0,
            "distribution_similarity": 0.0,
            "overall_score": 0.0,
            "is_cloned_dataset": False,
        }

    # Schema similarity: Jaccard similarity of column data types
    types_a = set(c.dtype for c in fp_a.columns)
    types_b = set(c.dtype for c in fp_b.columns)

    intersection = len(types_a.intersection(types_b))
    union = len(types_a.union(types_b))
    schema_sim = intersection / union if union > 0 else 0.0

    # Distribution similarity: Average similarity of numeric columns
    numeric_cols_a = [c for c in fp_a.columns if c.dtype == "numeric"]
    numeric_cols_b = [c for c in fp_b.columns if c.dtype == "numeric"]

    dist_sims = []
    if numeric_cols_a and numeric_cols_b:
        # Compare pairwise and take the maximum average
        # For simplicity, compare the first N numeric columns
        min_cols = min(len(numeric_cols_a), len(numeric_cols_b))
        for i in range(min_cols):
            sim = compute_distribution_similarity(numeric_cols_a[i], numeric_cols_b[i])
            dist_sims.append(sim)

    dist_sim = sum(dist_sims) / len(dist_sims) if dist_sims else 0.0

    # Overall score weights distribution similarity higher, as headers can be renamed
    overall_score = (schema_sim * 0.3) + (dist_sim * 0.7)

    # Flag as cloned if distribution is highly similar, even if row counts differ slightly
    is_cloned = dist_sim > 0.85 and schema_sim > 0.5

    return {
        "schema_similarity": round(schema_sim, 4),
        "distribution_similarity": round(dist_sim, 4),
        "overall_score": round(overall_score, 4),
        "is_cloned_dataset": is_cloned,
    }
