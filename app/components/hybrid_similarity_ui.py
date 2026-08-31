"""
Hybrid Similarity UI Components.

Provides UI elements for displaying hybrid similarity scores,
lexical vs semantic comparison charts, and settings controls.
"""

from typing import Any, Dict, List, Optional, Tuple  # noqa: F401

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# HYBRID SIMILARITY DISPLAY
# ============================================================================


def render_hybrid_similarity_metrics(
    semantic_score: float, lexical_score: float, hybrid_score: float, alpha: float = 0.7
) -> None:
    """
    Render hybrid similarity metrics in columns.

    Args:
        semantic_score: Semantic similarity score
        lexical_score: Lexical similarity score
        hybrid_score: Combined hybrid score
        alpha: Semantic weight used
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🧠 Semantic",
            f"{semantic_score:.1%}",
            help="Transformer embedding similarity",
        )
    with col2:
        st.metric(
            "📝 Lexical",
            f"{lexical_score:.1%}",
            help="TF-IDF / token overlap similarity",
        )
    with col3:
        st.metric(
            "🔀 Hybrid",
            f"{hybrid_score:.1%}",
            delta=f"α={alpha:.2f}",
            help="Weighted combination of semantic + lexical",
        )
    with col4:
        # Determine which method contributed more
        if semantic_score > lexical_score:
            st.metric(
                "📊 Dominant",
                "🧠 Semantic",
                delta=f"+{semantic_score - lexical_score:.1%}",
            )
        elif lexical_score > semantic_score:
            st.metric(
                "📊 Dominant",
                "📝 Lexical",
                delta=f"+{lexical_score - semantic_score:.1%}",
            )
        else:
            st.metric("📊 Dominant", "⚖️ Balanced", delta="Equal")


def render_hybrid_comparison_chart(
    semantic_scores: List[float],
    lexical_scores: List[float],
    hybrid_scores: List[float],
    labels: List[str],
) -> None:
    """
    Render Plotly chart comparing semantic, lexical, and hybrid scores.

    Args:
        semantic_scores: List of semantic scores
        lexical_scores: List of lexical scores
        hybrid_scores: List of hybrid scores
        labels: List of pair labels
    """
    if not semantic_scores:
        st.info("No scores to display")
        return

    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Bar(
            name="Semantic",
            x=labels,
            y=semantic_scores,
            marker_color="#3B82F6",
            text=[f"{s:.1%}" for s in semantic_scores],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Lexical",
            x=labels,
            y=lexical_scores,
            marker_color="#F59E0B",
            text=[f"{s:.1%}" for s in lexical_scores],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Hybrid",
            x=labels,
            y=hybrid_scores,
            marker_color="#10B981",
            text=[f"{s:.1%}" for s in hybrid_scores],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Similarity Score Comparison",
        xaxis_title="Document Pair",
        yaxis_title="Similarity Score",
        yaxis_tickformat=".0%",
        barmode="group",
        height=400,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_hybrid_settings_panel() -> dict[str, Any]:
    """
    Render hybrid similarity settings panel.

    Returns:
        Dict with settings values
    """
    st.markdown("### 🎯 Hybrid Similarity Settings")

    col1, col2 = st.columns(2)

    with col1:
        alpha = st.slider(
            "🧠 Semantic Weight (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help=(
                "Weight for semantic similarity vs lexical similarity.\n"
                "α = 1.0: Only semantic (current default)\n"
                "α = 0.7: 70% semantic, 30% lexical (recommended)\n"
                "α = 0.5: Equal weighting\n"
                "α = 0.0: Only lexical"
            ),
            key="hybrid_alpha_slider",
        )

    with col2:
        lexical_method = st.selectbox(
            "📝 Lexical Method",
            options=["tfidf", "jaccard", "dice", "overlap", "ngram", "char_ngram"],
            index=0,
            help=(
                "Method for computing lexical similarity:\n"
                "- tfidf: TF-IDF cosine similarity\n"
                "- jaccard: Token overlap Jaccard\n"
                "- dice: Sørensen-Dice coefficient\n"
                "- overlap: Szymkiewicz-Simpson overlap\n"
                "- ngram: Word n-gram overlap\n"
                "- char_ngram: Character n-gram overlap"
            ),
            key="lexical_method_selector",
        )

    # Scaling options
    with st.expander("⚙️ Scaling Options", expanded=False):
        scale_scores = st.checkbox(
            "Apply sigmoid scaling to lexical scores",
            value=False,
            key="scale_lexical_scores",
            help="Apply non-linear scaling to lexical scores for better separation",
        )

        if scale_scores:
            col1, col2 = st.columns(2)
            with col1:
                steepness = st.slider(
                    "Steepness",
                    min_value=2.0,
                    max_value=10.0,
                    value=6.0,
                    step=0.5,
                    help="Curve steepness for sigmoid scaling",
                )
            with col2:
                midpoint = st.slider(
                    "Midpoint",
                    min_value=0.3,
                    max_value=0.7,
                    value=0.5,
                    step=0.05,
                    help="Inflection point for sigmoid scaling",
                )
        else:
            steepness = 6.0
            midpoint = 0.5

    enabled = st.toggle(
        "🔀 Enable Hybrid Scoring",
        value=st.session_state.get("use_hybrid_scoring_toggle", False),
        key="use_hybrid_scoring_toggle",
        help=(
            "Combine semantic and lexical similarity for more robust detection.\n"
            "Hybrid scoring catches both paraphrasing and exact copying."
        ),
    )

    return {
        "enabled": enabled,
        "alpha": alpha,
        "lexical_method": lexical_method,
        "scale_scores": scale_scores,
        "steepness": steepness if scale_scores else 6.0,
        "midpoint": midpoint if scale_scores else 0.5,
    }


def render_hybrid_stats_chart(hybrid_stats: dict[str, Any]) -> None:
    """
    Render hybrid statistics chart.

    Args:
        hybrid_stats: Statistics from get_hybrid_similarity_stats()
    """
    if not hybrid_stats or hybrid_stats.get("semantic_avg", 0) == 0:
        st.info("No hybrid statistics available")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Semantic Avg", f"{hybrid_stats['semantic_avg']:.1%}")
    with col2:
        st.metric("📊 Lexical Avg", f"{hybrid_stats['lexical_avg']:.1%}")
    with col3:
        st.metric("📊 Hybrid Avg", f"{hybrid_stats['hybrid_avg']:.1%}")
    with col4:
        st.metric(
            "📈 Correlation",
            f"{hybrid_stats['semantic_lexical_correlation']:.2f}",
            help="Correlation between semantic and lexical scores",
        )

    # Progress bar for hybrid vs semantic
    if hybrid_stats["semantic_avg"] > 0:
        improvement = (
            hybrid_stats["hybrid_avg"] - hybrid_stats["semantic_avg"]
        ) / hybrid_stats["semantic_avg"]
        st.caption(f"Hybrid improvement over semantic: {improvement:+.1%}")
        st.progress(min(1.0, hybrid_stats["hybrid_avg"]))


def render_lexical_vs_semantic_scatter(
    semantic_scores: List[float], lexical_scores: List[float], labels: List[str]
) -> None:
    """
    Render scatter plot of semantic vs lexical scores.

    Args:
        semantic_scores: List of semantic scores
        lexical_scores: List of lexical scores
        labels: List of pair labels
    """
    if not semantic_scores or not lexical_scores:
        st.info("No data for scatter plot")
        return

    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=semantic_scores,
            y=lexical_scores,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(
                size=12,
                color=[s + l for s, l in zip(semantic_scores, lexical_scores)],  # noqa: E741
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Total Score"),
            ),
            hovertemplate="<b>%{text}</b><br>Semantic: %{x:.1%}<br>Lexical: %{y:.1%}<extra></extra>",
        )
    )

    # Diagonal line (equal contribution)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Equal Contribution",
            line=dict(dash="dash", color="gray"),
        )
    )

    fig.update_layout(
        title="Semantic vs Lexical Similarity",
        xaxis_title="Semantic Score",
        yaxis_title="Lexical Score",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        height=400,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# HYBRID SCORING INTEGRATION
# ============================================================================


def compute_and_display_hybrid_scores(
    semantic_matrix: pd.DataFrame,
    texts: dict[str, str],
    alpha: float = 0.7,
    lexical_method: str = "tfidf",
    scale_scores: bool = False,
    steepness: float = 6.0,
    midpoint: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Compute and display hybrid similarity scores.

    Args:
        semantic_matrix: Semantic similarity matrix
        texts: Document texts
        alpha: Semantic weight
        lexical_method: Lexical method to use
        scale_scores: Whether to scale lexical scores
        steepness: Sigmoid steepness
        midpoint: Sigmoid midpoint

    Returns:
        Tuple of (hybrid_matrix, statistics)
    """
    from src.core.lexical_similarity import (
        lexical_similarity_matrix,
        scale_lexical_matrix,
    )

    doc_names = list(texts.keys())

    # Compute lexical matrix
    lexical_matrix = lexical_similarity_matrix(texts)

    # Scale lexical scores if requested
    if scale_scores:
        lexical_matrix = scale_lexical_matrix(
            lexical_matrix, steepness=steepness, midpoint=midpoint
        )

    # Compute hybrid matrix
    hybrid_matrix = alpha * semantic_matrix + (1 - alpha) * lexical_matrix

    # Get stats
    n = len(doc_names)
    semantic_scores = []
    lexical_scores = []
    hybrid_scores = []
    pair_labels = []

    for i in range(n):
        for j in range(i + 1, n):
            semantic_scores.append(float(semantic_matrix.iloc[i, j]))
            lexical_scores.append(float(lexical_matrix.iloc[i, j]))
            hybrid_scores.append(float(hybrid_matrix.iloc[i, j]))
            pair_labels.append(f"{doc_names[i]} ↔ {doc_names[j]}")

    stats = {
        "semantic_avg": sum(semantic_scores) / len(semantic_scores)
        if semantic_scores
        else 0,
        "lexical_avg": sum(lexical_scores) / len(lexical_scores)
        if lexical_scores
        else 0,
        "hybrid_avg": sum(hybrid_scores) / len(hybrid_scores) if hybrid_scores else 0,
        "semantic_lexical_correlation": np.corrcoef(semantic_scores, lexical_scores)[
            0, 1
        ]
        if len(semantic_scores) > 1
        else 0.0,
        "alpha_used": alpha,
        "lexical_method": lexical_method,
    }

    return hybrid_matrix, stats
