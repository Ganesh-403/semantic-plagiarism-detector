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
AI Detection Report Component

Generates detailed reports for AI-generated text detection.
"""

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core.ai_detector_enhanced import AIDetectionResult


def render_ai_detection_report(results: dict[str, AIDetectionResult]) -> None:
    """
    Render detailed AI detection report.
    """
    if not results:
        st.info("No AI detection results available.")
        return

    st.markdown("### 📊 AI Detection Report")

    # Overview metrics
    total = len(results)
    suspicious = sum(1 for r in results.values() if r.is_suspicious)
    avg_prob = sum(r.ai_probability for r in results.values()) / total

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", total)
    col2.metric("Suspicious", suspicious, delta=f"{suspicious/total*100:.0f}%")
    col3.metric("Avg AI Probability", f"{avg_prob*100:.1f}%")
    col4.metric(
        "Avg Perplexity",
        f"{sum(r.perplexity_score for r in results.values())/total:.2f}",
    )

    st.divider()

    # Score breakdown chart
    st.markdown("### 📈 Score Breakdown")

    # Prepare data
    doc_names = list(results.keys())
    ai_probs = [r.ai_probability for r in results.values()]
    perplexity = [r.perplexity_score for r in results.values()]
    burstiness = [r.burstiness_score for r in results.values()]
    pattern = [r.pattern_score for r in results.values()]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="AI Probability",
            x=doc_names,
            y=ai_probs,
            marker_color="#ef4444",
            text=[f"{p*100:.1f}%" for p in ai_probs],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Perplexity",
            x=doc_names,
            y=perplexity,
            marker_color="#3b82f6",
            text=[f"{p*100:.1f}%" for p in perplexity],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Burstiness",
            x=doc_names,
            y=burstiness,
            marker_color="#22c55e",
            text=[f"{p*100:.1f}%" for p in burstiness],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Pattern",
            x=doc_names,
            y=pattern,
            marker_color="#f59e0b",
            text=[f"{p*100:.1f}%" for p in pattern],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="AI Detection Score Breakdown",
        xaxis_title="Document",
        yaxis_title="Score",
        yaxis_tickformat=".0%",
        barmode="group",
        height=400,
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Detailed results table
    st.markdown("### 📋 Detailed Results")

    data = []
    for doc_name, result in results.items():
        data.append(
            {
                "Document": doc_name,
                "AI Probability": f"{result.ai_probability*100:.1f}%",
                "Status": "⚠️ AI" if result.is_suspicious else "✅ Human",
                "Perplexity": f"{result.perplexity_score*100:.1f}%",
                "Burstiness": f"{result.burstiness_score*100:.1f}%",
                "Pattern": f"{result.pattern_score*100:.1f}%",
                "Sentence Var": f"{result.sentence_variability*100:.1f}%",
                "Word Count": result.features.get("word_count", 0),
            }
        )

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # Suspicious documents detail
    suspicious_docs = [d for d, r in results.items() if r.is_suspicious]
    if suspicious_docs:
        st.markdown("### ⚠️ Suspicious Documents")
        for doc in suspicious_docs:
            result = results[doc]
            with st.expander(
                f"🤖 {doc} - {result.ai_probability*100:.1f}% AI", expanded=False
            ):
                st.markdown(f"**AI Probability:** {result.ai_probability*100:.1f}%")
                st.markdown(
                    f"**Perplexity:** {result.perplexity_score*100:.1f}% (lower = more AI)"
                )
                st.markdown(
                    f"**Burstiness:** {result.burstiness_score*100:.1f}% (lower = more AI)"
                )
                st.markdown(
                    f"**Pattern Score:** {result.pattern_score*100:.1f}% (higher = more AI)"
                )
                st.markdown(
                    f"**Sentence Variability:** {result.sentence_variability*100:.1f}% (lower = more AI)"
                )
                st.markdown("**Features:**")
                for key, value in result.features.items():
                    st.caption(f"- {key}: {value}")
    else:
        st.success("✅ No suspicious AI-generated documents detected.")
