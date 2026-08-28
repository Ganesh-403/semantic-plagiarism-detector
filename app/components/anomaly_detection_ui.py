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
Plagiarism Anomaly Detection UI Component.

Streamlit-based interface for anomaly detection with
pattern analysis and visualization.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.anomaly_detector import (
    AnomalyConfig,
    AnomalyDetector,
    AnomalySeverity,
    AnomalyType,
)


def render_anomaly_detection_dashboard():
    """Render the anomaly detection dashboard."""
    st.title("🔍 Plagiarism Anomaly Detection")
    st.markdown(
        "Detect **unusual patterns** that may indicate plagiarism, collusion, or academic dishonesty."
    )

    tab_config, tab_detect, tab_results = st.tabs(
        ["⚙️ Configuration", "🔎 Detection", "📊 Results"]
    )

    with tab_config:
        _render_configuration()

    with tab_detect:
        _render_detection()

    with tab_results:
        _render_results()


def _render_configuration():
    """Render configuration panel."""
    st.subheader("Anomaly Detection Configuration")

    with st.form("anomaly_config"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Statistical Settings**")
            z_threshold = st.slider("Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)
            outlier_pct = st.slider("Outlier Percentile", 90, 99, 95)
            enable_stat = st.checkbox("Enable Statistical Analysis", value=True)

        with col2:
            st.markdown("**Cluster & Pattern Settings**")
            cluster_min = st.slider("Min Cluster Size", 2, 10, 3)
            cluster_threshold = st.slider(
                "Cluster Similarity Threshold", 0.70, 0.99, 0.85, 0.01
            )
            collusion_threshold = st.slider(
                "Collusion Threshold", 0.70, 0.99, 0.80, 0.01
            )
            enable_cluster = st.checkbox("Enable Cluster Analysis", value=True)
            enable_pattern = st.checkbox("Enable Pattern Analysis", value=True)

        if st.form_submit_button("💾 Save Configuration", use_container_width=True):
            st.session_state.anomaly_config = AnomalyConfig(
                z_score_threshold=z_threshold,
                cluster_min_size=cluster_min,
                cluster_similarity_threshold=cluster_threshold,
                collusion_threshold=collusion_threshold,
                enable_statistical=enable_stat,
                enable_cluster=enable_cluster,
                enable_pattern=enable_pattern,
            )
            st.success("✅ Configuration saved!")


def _render_detection():
    """Render detection interface."""
    st.subheader("Anomaly Detection")

    uploaded_files = st.file_uploader(
        "Upload documents for anomaly detection",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload at least 3 documents for effective anomaly detection",
    )

    use_sample = st.checkbox("Use sample data for demo", value=False)

    if st.button("🚀 Run Anomaly Detection", type="primary", use_container_width=True):
        if use_sample or (uploaded_files and len(uploaded_files) >= 3):
            documents = {}
            if use_sample:
                documents = _get_sample_documents()
            else:
                for f in uploaded_files:
                    text = f.read().decode("utf-8", errors="ignore")
                    documents[f.name] = text

            config = st.session_state.get("anomaly_config", AnomalyConfig())
            detector = AnomalyDetector(config)

            # Generate sample similarity data
            n = len(documents)
            sim_matrix = np.random.rand(n, n) * 0.5 + 0.3
            sim_matrix = (sim_matrix + sim_matrix.T) / 2
            np.fill_diagonal(sim_matrix, 1.0)
            sim_scores = [sim_matrix[i, j] for i in range(n) for j in range(i + 1, n)]

            with st.spinner("🔍 Running anomaly detection..."):
                result = detector.detect(documents, sim_matrix, sim_scores)

            st.session_state.anomaly_result = result
            st.success(
                f"✅ Detection complete! Found **{len(result.anomalies)}** anomalies."
            )
        else:
            st.warning("Please upload at least 3 documents or use sample data.")


def _render_results():
    """Render detection results."""
    st.subheader("Anomaly Detection Results")

    result = st.session_state.get("anomaly_result")
    if not result:
        st.info("Run detection first to see results here.")
        return

    data = result.to_dict()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Anomalies", data["summary"]["total_anomalies"])
    col2.metric("High Priority", data["summary"]["high_priority_count"])
    col3.metric("Documents Analyzed", data["summary"]["documents_analyzed"])
    col4.metric("Processing Time", f"{data['processing_time']:.2f}s")

    # Severity distribution
    severity = data["summary"].get("by_severity", {})
    if severity:
        st.subheader("🎯 Severity Distribution")
        fig_sev = px.bar(
            x=list(severity.keys()),
            y=list(severity.values()),
            color=list(severity.keys()),
            color_discrete_map={
                "critical": "#ef4444",
                "high": "#f97316",
                "medium": "#eab308",
                "low": "#84cc16",
                "info": "#60a5fa",
            },
            title="Anomalies by Severity",
        )
        st.plotly_chart(fig_sev, use_container_width=True)

    # Anomaly type distribution
    by_type = data["summary"].get("by_type", {})
    if by_type:
        st.subheader("📊 Anomaly Types")
        fig_type = px.pie(
            values=list(by_type.values()),
            names=list(by_type.keys()),
            title="Anomalies by Type",
        )
        st.plotly_chart(fig_type, use_container_width=True)

    # Detailed anomalies
    if data["anomalies"]:
        st.subheader("📋 Detected Anomalies")
        for i, anomaly in enumerate(data["anomalies"][:15]):
            sev_colors = {
                "critical": "#ef4444",
                "high": "#f97316",
                "medium": "#eab308",
                "low": "#84cc16",
                "info": "#60a5fa",
            }
            color = sev_colors.get(anomaly["severity"], "#64748b")
            with st.expander(
                f"**{anomaly['title']}** — {anomaly['severity'].upper()} ({anomaly['confidence']:.0%})",
                expanded=(anomaly["severity"] in ("critical", "high")),
            ):
                st.markdown(f"**Type:** {anomaly['anomaly_type']}")
                st.markdown(f"**Description:** {anomaly['description']}")
                if anomaly["affected_documents"]:
                    st.markdown(
                        f"**Affected Documents:** {', '.join(anomaly['affected_documents'])}"
                    )
                st.markdown(f"**Confidence:** {anomaly['confidence']:.1%}")
                if anomaly["evidence"]:
                    st.json(anomaly["evidence"])

    # Recommendations
    if data.get("recommendations"):
        st.subheader("💡 Recommendations")
        for rec in data["recommendations"]:
            st.markdown(f"- {rec}")

    # Export
    st.subheader("⬇️ Export Results")
    col1, col2 = st.columns(2)
    with col1:
        if data["anomalies"]:
            import json

            st.download_button(
                "Download JSON",
                json.dumps(data, indent=2, default=str),
                "anomaly_results.json",
                "application/json",
            )
    with col2:
        if data["anomalies"]:
            df = pd.DataFrame(data["anomalies"])
            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                "anomaly_results.csv",
                "text/csv",
            )


def _get_sample_documents() -> dict[str, str]:
    """Get sample documents for demo."""
    return {
        "student_a.txt": "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Deep learning is a subset of machine learning that uses neural networks with many layers. The field has grown rapidly in recent years due to increased computational power and data availability.",
        "student_b.txt": "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Deep learning is a subset of machine learning that uses neural networks with many layers. The field has experienced significant growth due to advances in computing and data.",
        "student_c.txt": "Artificial intelligence encompasses various techniques including machine learning and deep learning. Neural networks with multiple layers enable deep learning approaches. Recent developments in computational infrastructure have accelerated progress in this area.",
        "student_d.txt": "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. Deep learning is a subset of machine learning that uses neural networks with many layers. The field has grown rapidly in recent years due to increased computational power and data availability.",
        "student_e.txt": "Cooking pasta requires boiling water in a large pot. Add salt to the water for flavor. Once the water reaches a rolling boil, add the pasta and stir occasionally. Cook until al dente, then drain and serve with your favorite sauce.",
    }
