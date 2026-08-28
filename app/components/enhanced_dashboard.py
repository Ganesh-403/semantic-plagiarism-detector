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
Enhanced Dashboard Components for Semantic Plagiarism Detector

Features:
- Document similarity trend analysis
- Plagiarism pattern detection
- Real-time processing metrics
- Historical comparison visualization
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==============================================================================
# DATA ANALYTICS CLASSES
# ==============================================================================


class PlagiarismPatternAnalyzer:
    """Analyze plagiarism patterns across documents."""

    @staticmethod
    def detect_collusion_rings(
        sim_matrix: pd.DataFrame, threshold: float = 0.75
    ) -> dict:
        """Detect groups of highly similar documents."""
        if sim_matrix.empty:
            return {"rings": [], "summary": "No data available"}

        # Create graph of similar documents
        graph = defaultdict(set)
        docs = sim_matrix.columns.tolist()

        for i, doc_a in enumerate(docs):
            for j, doc_b in enumerate(docs):
                if i != j and sim_matrix.iloc[i, j] >= threshold:
                    graph[doc_a].add(doc_b)
                    graph[doc_b].add(doc_a)

        # Find connected components (collusion rings)
        visited = set()
        rings = []

        for doc in graph:
            if doc not in visited:
                # BFS to find component
                component = []
                queue = [doc]
                visited.add(doc)

                while queue:
                    current = queue.pop(0)
                    component.append(current)
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                if len(component) >= 3:  # Only rings of 3+
                    rings.append(
                        {
                            "documents": component,
                            "size": len(component),
                            "avg_similarity": PlagiarismPatternAnalyzer._calculate_avg_similarity(
                                sim_matrix, component
                            ),
                        }
                    )

        return {
            "rings": rings,
            "total_rings": len(rings),
            "affected_docs": sum(r["size"] for r in rings),
            "summary": f"Found {len(rings)} collusion rings affecting {sum(r['size'] for r in rings)} documents",
        }

    @staticmethod
    def _calculate_avg_similarity(
        sim_matrix: pd.DataFrame, documents: list[str]
    ) -> float:
        """Calculate average similarity within a group."""
        similarities = []
        for i, doc_a in enumerate(documents):
            for doc_b in documents[i + 1 :]:
                similarities.append(sim_matrix.loc[doc_a, doc_b])
        return np.mean(similarities) if similarities else 0.0

    @staticmethod
    def identify_outlier_patterns(scores: list[float]) -> dict:
        """Identify outlier patterns in similarity scores."""
        if not scores:
            return {"outliers": [], "summary": "No data available"}

        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = [s for s in scores if s < lower_bound or s > upper_bound]

        return {
            "outliers": outliers,
            "outlier_count": len(outliers),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "summary": f"Found {len(outliers)} outlier patterns",
        }


class DocumentTrendAnalyzer:
    """Analyze document processing trends."""

    def __init__(self, history_data: list[dict]):
        self.history_data = history_data

    def get_daily_trends(self) -> pd.DataFrame:
        """Get daily processing trends."""
        if not self.history_data:
            return pd.DataFrame()

        df = pd.DataFrame(self.history_data)
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily_stats = (
            df.groupby("date")
            .agg({"doc_count": "sum", "avg_similarity": "mean", "flagged_count": "sum"})
            .reset_index()
        )

        return daily_stats

    def get_peak_times(self) -> dict:
        """Identify peak processing times."""
        if not self.history_data:
            return {"peak_hours": [], "summary": "No data available"}

        df = pd.DataFrame(self.history_data)
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        hour_counts = df.groupby("hour").size()

        peak_hours = hour_counts.nlargest(3).index.tolist()

        return {
            "peak_hours": peak_hours,
            "busiest_hour": peak_hours[0] if peak_hours else None,
            "summary": f"Peak hours: {', '.join(map(str, peak_hours))}",
        }


# ==============================================================================
# PERFORMANCE ANALYTICS
# ==============================================================================


class PerformanceDashboard:
    """Performance monitoring dashboard."""

    @staticmethod
    def render_performance_metrics(metrics: dict[str, list[float]]):
        """Render performance metrics dashboard."""
        if not metrics:
            st.info("No performance metrics available")
            return

        st.markdown("### ⚡ Real-time Performance")

        # Create metrics grid
        cols = st.columns(3)
        metric_names = list(metrics.keys())[:3]

        for col, name in zip(cols, metric_names):
            values = metrics.get(name, [])
            if values:
                avg = np.mean(values)
                p95 = np.percentile(values, 95)
                col.metric(
                    f"{name.replace('_', ' ').title()}",
                    f"{avg*1000:.1f}ms",
                    delta=f"P95: {p95*1000:.1f}ms",
                )

        # Performance trend chart
        if metrics:
            PerformanceDashboard._create_performance_chart(metrics)

    @staticmethod
    def _create_performance_chart(metrics: dict[str, list[float]]):
        """Create performance trend chart."""
        fig = go.Figure()

        for name, values in list(metrics.items())[:5]:
            if values:
                # Take last 50 values for chart
                recent = values[-50:]
                fig.add_trace(
                    go.Scatter(
                        y=recent,
                        name=name.replace("_", " ").title(),
                        mode="lines",
                        line=dict(width=2),
                    )
                )

        fig.update_layout(
            title="Performance Trends",
            xaxis_title="Recent Operations",
            yaxis_title="Time (seconds)",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# VISUALIZATION COMPONENTS
# ==============================================================================


def render_similarity_trend_chart(history_data: list[dict]) -> None:
    """Render similarity trend chart."""
    if not history_data:
        st.info("No history data available for trends")
        return

    df = pd.DataFrame(history_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["avg_similarity"],
            mode="lines+markers",
            name="Avg Similarity",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["max_similarity"],
            mode="lines+markers",
            name="Max Similarity",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            marker=dict(size=6),
        )
    )

    # Add threshold line
    fig.add_hline(
        y=0.75, line_dash="dot", line_color="red", annotation_text="Alert Threshold"
    )

    fig.update_layout(
        title="Similarity Trends Over Time",
        xaxis_title="Date",
        yaxis_title="Similarity Score",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_document_activity_heatmap(history_data: list[dict]) -> None:
    """Render document activity heatmap."""
    if not history_data:
        return

    df = pd.DataFrame(history_data)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    # Create pivot table
    pivot = pd.pivot_table(
        df,
        values="doc_count",
        index="hour",
        columns="date",
        aggfunc="sum",
        fill_value=0,
    )

    fig = px.imshow(
        pivot,
        title="Document Activity Heatmap",
        labels=dict(x="Date", y="Hour", color="Documents"),
        color_continuous_scale="YlOrRd",
        aspect="auto",
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_collusion_ring_dashboard(
    sim_matrix: pd.DataFrame, threshold: float = 0.75
) -> None:
    """Render collusion ring detection dashboard."""
    if sim_matrix.empty:
        st.info("No similarity data available")
        return

    analyzer = PlagiarismPatternAnalyzer()
    result = analyzer.detect_collusion_rings(sim_matrix, threshold)

    if result["rings"]:
        st.warning(f"🚨 {result['summary']}")

        for ring in result["rings"]:
            with st.expander(
                f"Ring #{result['rings'].index(ring) + 1}: {ring['size']} documents",
                expanded=False,
            ):
                st.markdown("**Documents:**")
                for doc in ring["documents"]:
                    st.markdown(f"- 📄 {doc}")
                st.metric("Average Similarity", f"{ring['avg_similarity']:.1%}")
    else:
        st.success("✅ No collusion rings detected")


def render_processing_time_breakdown(timings: dict[str, float]) -> None:
    """Render processing time breakdown chart."""
    if not timings:
        st.info("No timing data available")
        return

    # Create pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(timings.keys()),
                values=list(timings.values()),
                hole=0.4,
                textinfo="label+percent",
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Processing Time Breakdown", height=350, margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# COMPARISON HISTORY DASHBOARD
# ==============================================================================


class ComparisonHistoryDashboard:
    """Dashboard for comparison history."""

    def __init__(self, history_manager):
        self.history = history_manager

    def render_summary_stats(self):
        """Render summary statistics."""
        stats = self.history.get_statistics() if self.history else {}

        if not stats or stats.get("total_comparisons", 0) == 0:
            st.info("No comparison history available")
            return

        cols = st.columns(4)
        cols[0].metric("Total Comparisons", stats.get("total_comparisons", 0))
        cols[1].metric("Avg Similarity", f"{stats.get('avg_similarity', 0):.1%}")
        cols[2].metric("Flagged", stats.get("flagged_count", 0))
        cols[3].metric("Unique Documents", len(stats.get("unique_documents", [])))

    def render_recent_comparisons(self, limit: int = 10):
        """Render recent comparisons table."""
        if not self.history:
            return

        records = self.history.get_comparisons(limit=limit)
        if not records:
            st.info("No recent comparisons")
            return

        data = []
        for r in records:
            data.append(
                {
                    "Document A": (
                        r.document_a[:30] + "..."
                        if len(r.document_a) > 30
                        else r.document_a
                    ),
                    "Document B": (
                        r.document_b[:30] + "..."
                        if len(r.document_b) > 30
                        else r.document_b
                    ),
                    "Similarity": f"{r.similarity_score:.1%}",
                    "Flagged": "⚠️" if r.was_flagged else "✅",
                    "Time": r.timestamp[:19] if len(r.timestamp) >= 19 else r.timestamp,
                }
            )

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)


# ==============================================================================
# MAIN UI RENDER FUNCTIONS
# ==============================================================================


def render_enhanced_analytics_tab(
    sim_matrix: pd.DataFrame, history_data: list[dict], timings: dict[str, float]
):
    """Render enhanced analytics tab."""
    st.markdown("### 📊 Enhanced Analytics Dashboard")

    # Row 1: Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Scans", len(history_data))
    col2.metric(
        "Flagged Incidents", sum(h.get("flagged_count", 0) for h in history_data)
    )

    if sim_matrix is not None and not sim_matrix.empty:
        max_sim = sim_matrix.max().max()
        col3.metric("Max Similarity", f"{max_sim:.1%}")
        avg_sim = sim_matrix.values[np.triu_indices_from(sim_matrix.values, k=1)].mean()
        col4.metric("Avg Similarity", f"{avg_sim:.1%}")

    # Row 2: Charts
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "🔥 Activity", "🔍 Patterns"])

    with tab1:
        render_similarity_trend_chart(history_data)

    with tab2:
        render_document_activity_heatmap(history_data)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            render_collusion_ring_dashboard(sim_matrix)
        with col2:
            render_processing_time_breakdown(timings)

    # Row 3: Historical Data
    st.markdown("---")
    if history_data:
        st.markdown("### 📋 Detailed History")
        df = pd.DataFrame(history_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.dataframe(df, use_container_width=True)


def render_enhanced_document_analysis(doc_name: str, text: str):
    """Render enhanced document analysis."""
    with st.expander(f"📄 Document Analysis: {doc_name}", expanded=False):
        # Text Statistics
        words = text.split()
        sentences = len([s for s in text.split(".") if s.strip()])

        col1, col2, col3 = st.columns(3)
        col1.metric("Word Count", len(words))
        col2.metric("Sentence Count", sentences)
        col3.metric(
            "Avg Word Length", f"{sum(len(w) for w in words) / max(len(words), 1):.1f}"
        )

        # Readability Score
        from app.components.advanced_analytics import AdvancedTextPreprocessor

        preprocessor = AdvancedTextPreprocessor()
        readability = preprocessor.compute_readability_score(text)

        st.markdown("**Readability Metrics:**")
        cols = st.columns(4)
        for col, (key, value) in zip(cols, readability.items()):
            if isinstance(value, float):
                col.metric(key.replace("_", " ").title(), f"{value:.1f}")
            else:
                col.metric(key.replace("_", " ").title(), value)

        # Key Phrases
        phrases = preprocessor.extract_key_phrases(text)
        if phrases:
            st.markdown("**Key Phrases:**")
            st.markdown(", ".join(phrases[:10]))


# ==============================================================================
# INITIALIZATION
# ==============================================================================


def initialize_enhanced_dashboard():
    """Initialize enhanced dashboard components."""
    if "pattern_analyzer" not in st.session_state:
        st.session_state.pattern_analyzer = PlagiarismPatternAnalyzer()

    if "trend_analyzer" not in st.session_state:
        st.session_state.trend_analyzer = None

    if "performance_dashboard" not in st.session_state:
        st.session_state.performance_dashboard = PerformanceDashboard()

    if "comparison_dashboard" not in st.session_state:
        from pathlib import Path

        from app.components.advanced_analytics import ComparisonHistoryManager

        history_path = (
            Path(st.session_state.get("data_dir", ".")) / "comparison_history"
        )
        st.session_state.comparison_dashboard = ComparisonHistoryDashboard(
            ComparisonHistoryManager(history_path)
        )
