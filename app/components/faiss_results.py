"""
Utilities for rendering and formatting FAISS search results.

This module provides helper functions for retrieving,
formatting, and displaying vector search results.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

RESULT_COLUMNS: list[str] = [
    "Rank",
    "Target Document",
    "Chunk",
    "Similarity Score",
    "Matching Text",
    "Stats",
]


def _record_value(record: Any, name: str, default: Any = None) -> Any:
    """
    Read a field from dataclass-like or mapping-like records.

    Args:
        record: The record to read the field from (mapping or object).
        name: The name of the field to read.
        default: The default value to return if the field is not found.

    Returns:
        The value of the field, or the default value.
    """
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def faiss_results_dataframe(
    results: Iterable[tuple[Any, float]],
    min_similarity: float | None = None,
    max_similarity: float | None = None,
) -> pd.DataFrame:
    """
    Convert FAISS records into a sortable display DataFrame.

    Args:
        results: An iterable of tuples containing a record and a raw similarity score.
        min_similarity: Minimum similarity score to include.
        max_similarity: Maximum similarity score to include.

    Returns:
        A pandas DataFrame containing the formatted search results.
    """
    rows: list[dict[str, Any]] = []

    for record, raw_score in results:
        score: float = float(raw_score)

        if min_similarity is not None and score < min_similarity:
            continue
        if max_similarity is not None and score > max_similarity:
            continue

        document: str = str(_record_value(record, "doc_name", "Unknown document"))
        chunk_index: int = int(_record_value(record, "chunk_index", 0))
        chunk_text: str = str(_record_value(record, "chunk_text", ""))

        from src.utils.text_stats import format_text_stats

        rows.append(
            {
                "Target Document": document,
                "Chunk": chunk_index + 1,
                "Similarity Score": score,
                "Matching Text": chunk_text,
                "Stats": format_text_stats(chunk_text),
            }
        )

    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    dataframe: pd.DataFrame = pd.DataFrame(rows)
    dataframe = dataframe.sort_values(
        by=["Similarity Score", "Target Document", "Chunk"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)

    dataframe.insert(0, "Rank", range(1, len(dataframe) + 1))

    return dataframe[RESULT_COLUMNS]
