"""
Utilities for rendering and formatting FAISS search results.

This module provides helper functions for retrieving,
formatting, and displaying vector search results.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
import streamlit as st


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
                "Similarity Score": round(score, 4),
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


@st.dialog("🔍 Chunk Diff Inspector", width="large")
def inspect_diff_dialog(
    query_text: str,
    matched_text: str,
    doc_name: str,
    score: float,
    pdf_bytes: bytes | None = None,
    chunk_id: str | None = None,
):
    """Render a side-by-side highlighted diff of query vs matched chunk inside a modal."""
    formatted_score = f"{score:.4f}"
    st.markdown(f"### Match Similarity: **{formatted_score}** ({score:.1%})")

    # Display quick copy buttons inside the inspector modal
    col_score, col_chunk = st.columns(2)
    with col_score:
        st.caption("📋 Similarity Score")
        st.code(formatted_score, language="text")
    with col_chunk:
        if chunk_id:
            st.caption("📋 Vector Chunk ID")
            st.code(chunk_id, language="text")

    from src.utils.diff_highlighter import highlight_overlap
    highlighted_query, highlighted_match = highlight_overlap(query_text, matched_text)

    col_q, col_m = st.columns(2)
    with col_q:
        st.markdown("### 📝 Query Text")
        st.markdown(
            f"<div style='border: 1px solid #cccccc; padding: 12px; border-radius: 6px; min-height: 150px; white-space: pre-wrap;'>{highlighted_query}</div>",
            unsafe_allow_html=True
        )
    with col_m:
        st.markdown(f"### 📄 Matched Chunk ({doc_name})")
        st.markdown(
            f"<div style='border: 1px solid #cccccc; padding: 12px; border-radius: 6px; min-height: 150px; white-space: pre-wrap;'>{highlighted_match}</div>",
            unsafe_allow_html=True
        )

    if pdf_bytes:
        st.divider()

        try:
            from src.utils.pdf_highlighter import highlight_pdf_matches
        except Exception:
            highlight_pdf_matches = None

        if highlight_pdf_matches is None:
            st.caption("⚠️ PDF highlighting is unavailable (PyMuPDF is not installed).")
        else:
            try:
                annotated_pdf = highlight_pdf_matches(pdf_bytes, [matched_text])
            except Exception as exc:
                annotated_pdf = None
                st.warning(f"Could not generate the highlighted PDF: {exc}")

            if annotated_pdf:
                base_name = os.path.splitext(doc_name)[0]
                st.download_button(
                    "⬇️ Download Highlighted PDF",
                    data=annotated_pdf,
                    file_name=f"highlighted_{base_name}.pdf",
                    mime="application/pdf",
                    key=f"download_highlighted_pdf_{doc_name}",
                )


def render_faiss_results_ui(
    results: Iterable[tuple[Any, float]],
    query_text: str,
    document_pdf_bytes: Mapping[str, bytes] | None = None,
) -> None:
    """
    Render FAISS search results with a clean interface, quick-copy score and chunk IDs,
    and an interactive 'Inspect Diff' modal dialog for side-by-side comparison.

    Args:
        results: An iterable of (record, score) tuples returned by the FAISS search.
        query_text: The text the user searched with.
        document_pdf_bytes: Optional mapping of doc_name -> original PDF bytes.
            When the matched document is available here, the diff inspector
            offers a "Download Highlighted PDF" button that annotates the
            matched passage directly on the source PDF.
    """
    if not results:
        st.info("No significant matches found above threshold.")
        return

    for i, (record, raw_score) in enumerate(results):
        score = float(raw_score)
        formatted_score = f"{score:.4f}"
        doc_name = str(_record_value(record, "doc_name", "Unknown document"))
        chunk_index = int(_record_value(record, "chunk_index", 0))
        chunk_id = str(_record_value(record, "chunk_id", f"chunk_{chunk_index + 1}"))
        chunk_text = str(_record_value(record, "chunk_text", ""))

        st.markdown(
            f"<div style='border: 1px solid #e2e8f0; padding: 12px; border-radius: 8px; margin-bottom: 8px;'>"
            f"<strong>📄 {doc_name}</strong> (Chunk #{chunk_index + 1}) · "
            f"<span style='color: #3b82f6; font-weight: bold;'>Similarity: {formatted_score} ({score:.1%})</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Quick-copy blocks for Similarity Score and Chunk ID using st.code()
        col_score, col_chunk = st.columns(2)
        with col_score:
            st.caption("📋 Similarity Score")
            st.code(formatted_score, language="text")
        with col_chunk:
            st.caption("📋 Vector Chunk ID")
            st.code(chunk_id, language="text")

        st.caption(chunk_text[:300] + ("..." if len(chunk_text) > 300 else ""))

        if st.button("🔍 Inspect Diff", key=f"diff_btn_{i}_{doc_name}_{chunk_index}"):
            source_pdf_bytes = (
                document_pdf_bytes.get(doc_name) if document_pdf_bytes else None
            )
            if source_pdf_bytes:
                inspect_diff_dialog(
                    query_text,
                    chunk_text,
                    doc_name,
                    score,
                    pdf_bytes=source_pdf_bytes,
                    chunk_id=chunk_id,
                )
            else:
                inspect_diff_dialog(
                    query_text,
                    chunk_text,
                    doc_name,
                    score,
                    chunk_id=chunk_id,
                )

        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)


def get_faiss_metric_label(faiss_index: Any = None) -> str:
    """Read distance metric type from active FAISS index wrapper or object.

    Args:
        faiss_index: FAISS index instance or wrapper object.

    Returns:
        Formatted metric description string or 'Default' if uninitialized.
    """
    if faiss_index is None:
        return "Default"

    try:
        metric_type = getattr(faiss_index, "metric_type", None)
        if metric_type is None and hasattr(faiss_index, "index"):
            metric_type = getattr(faiss_index.index, "metric_type", None)

        try:
            import faiss
            if metric_type == faiss.METRIC_INNER_PRODUCT or isinstance(faiss_index, faiss.IndexFlatIP):
                return "Inner Product (Cosine)"
            elif metric_type == faiss.METRIC_L2 or isinstance(faiss_index, faiss.IndexFlatL2):
                return "L2 (Euclidean)"
        except ImportError:
            pass

        if metric_type == 0:
            return "Inner Product (Cosine)"
        elif metric_type == 1:
            return "L2 (Euclidean)"
        elif type(faiss_index).__name__ in ("IndexFlatIP", "IndexIVFFlat"):
            return "Inner Product (Cosine)"
        elif type(faiss_index).__name__ in ("IndexFlatL2",):
            return "L2 (Euclidean)"
    except Exception:
        pass

    return "Default"


def render_faiss_metric_badge(faiss_index: Any = None) -> str:
    """Render active FAISS vector distance metric badge in sidebar.

    Args:
        faiss_index: Active FAISS index instance.

    Returns:
        Rendered metric badge label string.
    """
    label = get_faiss_metric_label(faiss_index)
    badge_text = f"Metric: {label}"
    st.sidebar.caption(f"🎯 {badge_text}")
    return badge_text
