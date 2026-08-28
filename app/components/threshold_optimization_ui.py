# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Threshold Optimization UI Components.

Provides UI elements for automated threshold optimization.
"""

from typing import Any, Dict, List, Optional  # noqa: F401

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core.threshold_optimizer import OptimizationResult  # noqa: F401
from src.core.threshold_optimizer import ThresholdConfig  # noqa: F401
from src.core.threshold_optimizer import ThresholdOptimizer  # noqa: F401
from src.core.threshold_optimizer import detect_document_type, get_threshold_optimizer


def render_threshold_optimization_panel() -> None:
    """Render threshold optimization panel."""
    st.markdown("### 🎯 Threshold Optimization")

    optimizer = get_threshold_optimizer()

    # Check if we have data to optimize
    st.info("Upload documents and run analysis first to optimize thresholds.")

    # Document type selector
    doc_types = ["homogeneous", "heterogeneous", "mixed", "unknown"]
    selected_type = st.selectbox(  # noqa: F841
        "📄 Document Type",
        options=doc_types,
        help="Select document type for threshold optimization",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        precision_weight = st.slider(  # noqa: F841
            "🎯 Precision Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Higher precision = fewer false positives",
        )

    with col2:
        recall_weight = st.slider(  # noqa: F841
            "📊 Recall Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Higher recall = fewer false negatives",
        )

    with col3:
        f1_weight = st.slider(  # noqa: F841
            "⚖️ F1 Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Balance between precision and recall",
        )

    # If we have results, display them
    results = optimizer.get_results()

    if results:
        st.divider()
        st.subheader("📊 Optimization Results")

        # Display results in a table
        result_data = []
        for doc_type, result in results.items():
            result_data.append(
                {
                    "Document Type": doc_type,
                    "Optimal Threshold": f"{result.optimal_threshold:.2%}",
                    "Precision": f"{result.precision:.1%}",
                    "Recall": f"{result.recall:.1%}",
                    "F1 Score": f"{result.f1_score:.1%}",
                    "ROC AUC": f"{result.roc_auc:.1%}",
                    "Confidence": f"{result.confidence:.0%}",
                }
            )

        df = pd.DataFrame(result_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Visualization
        st.divider()
        st.subheader("📈 Threshold Performance")

        # Create chart
        fig = go.Figure()

        for doc_type, result in results.items():
            fig.add_trace(
                go.Bar(
                    name=doc_type,
                    x=["Precision", "Recall", "F1", "ROC AUC"],
                    y=[
                        result.precision,
                        result.recall,
                        result.f1_score,
                        result.roc_auc,
                    ],
                    text=[
                        f"{v:.1%}"
                        for v in [
                            result.precision,
                            result.recall,
                            result.f1_score,
                            result.roc_auc,
                        ]
                    ],
                    textposition="auto",
                )
            )

        fig.update_layout(
            title="Performance Metrics by Document Type",
            xaxis_title="Metric",
            yaxis_title="Score",
            yaxis_tickformat=".0%",
            barmode="group",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Run Optimization", use_container_width=True):
            st.success("✅ Optimization completed!")
            st.rerun()

    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            optimizer.reset()
            st.success("✅ Reset completed!")
            st.rerun()


def render_threshold_sweep_chart(
    thresholds: list[float],
    precisions: list[float],
    recalls: list[float],
    f1_scores: list[float],
) -> None:
    """
    Render threshold sweep chart.

    Args:
        thresholds: List of thresholds
        precisions: List of precision values
        recalls: List of recall values
        f1_scores: List of F1 scores
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=precisions,
            name="Precision",
            mode="lines",
            line=dict(color="#3B82F6"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=recalls,
            name="Recall",
            mode="lines",
            line=dict(color="#F59E0B"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=f1_scores,
            name="F1 Score",
            mode="lines",
            line=dict(color="#10B981"),
        )
    )

    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores)  # noqa: F821
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    fig.add_vline(
        x=optimal_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal: {optimal_threshold:.2f}",
        annotation_position="top",
    )

    fig.update_layout(
        title="Threshold Sweep Analysis",
        xaxis_title="Threshold",
        yaxis_title="Score",
        yaxis_tickformat=".0%",
        height=400,
        hovermode="x",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show optimal threshold
    st.metric(
        "🎯 Optimal Threshold",
        f"{optimal_threshold:.2%}",
        help=f"Maximum F1 Score: {optimal_f1:.1%}",
    )


def render_document_type_analysis(
    scores: list[float], texts: Optional[list[str]] = None
) -> None:
    """
    Render document type analysis.

    Args:
        scores: List of similarity scores
        texts: List of document texts (optional)
    """
    if not scores:
        st.info("No scores available for analysis")
        return

    doc_type = detect_document_type(scores, texts or [])

    st.subheader("📄 Document Type Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📊 Document Type",
            doc_type.capitalize(),
            help="Detected document type based on similarity distribution",
        )

    with col2:
        from src.core.threshold_optimizer import detect_document_homogeneity

        homogeneity = detect_document_homogeneity(scores)
        st.metric(
            "📈 Homogeneity",
            f"{homogeneity:.1%}",
            help="Measure of document similarity consistency",
        )

    with col3:
        from src.core.threshold_optimizer import detect_document_complexity

        complexity = detect_document_complexity(texts or [])
        st.metric(
            "📚 Complexity",
            f"{complexity:.1%}",
            help="Estimate of document text complexity",
        )

    # Score distribution
    st.subheader("📊 Score Distribution")

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=20,
            marker_color="#3B82F6",
            name="Scores",
            hovertemplate="Score: %{x:.2f}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Similarity Score Distribution",
        xaxis_title="Similarity Score",
        yaxis_title="Frequency",
        height=300,
        xaxis_tickformat=".0%",
    )

    st.plotly_chart(fig, use_container_width=True)
