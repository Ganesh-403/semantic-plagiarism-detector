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

"""Pattern Recognition & Prediction System — Streamlit UI component.

Renders the pattern recognition dashboard with 4 sub-tabs:
- Pattern Dashboard: summary cards + pattern list
- Pattern Details: evolution chart + related incidents
- Risk Matrix: document risk scatter plot
- Recommendations: actionable items table
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

# ── Lazy imports for backend modules ──────────────────────────────────────
try:
    from src.core.pattern_recognition import PatternDetectionEngine
    from src.core.recommendations import RecommendationEngine
    from src.db.pattern_repository import PatternRepository

    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False


def _get_repo():
    if not BACKEND_AVAILABLE:
        return None
    if "pattern_repo" not in st.session_state:
        st.session_state.pattern_repo = PatternRepository()
    return st.session_state.pattern_repo


def _get_engine():
    if not BACKEND_AVAILABLE:
        return None
    if "pattern_engine" not in st.session_state:
        st.session_state.pattern_engine = PatternDetectionEngine(repository=_get_repo())
    return st.session_state.pattern_engine


def _get_rec_engine():
    if not BACKEND_AVAILABLE:
        return None
    if "rec_engine" not in st.session_state:
        st.session_state.rec_engine = RecommendationEngine(repository=_get_repo())
    return st.session_state.rec_engine


# ═══════════════════════════════════════════════════════════════════════════
#  Main Render Function
# ═══════════════════════════════════════════════════════════════════════════


def render_pattern_recognition():
    """Render the full pattern recognition dashboard."""
    if not BACKEND_AVAILABLE:
        st.warning("Pattern recognition backend unavailable. Install scikit-learn.")
        return

    st.subheader("Intelligent Pattern Recognition & Prediction")

    tab_dash, tab_details, tab_risk, tab_recs = st.tabs(
        [
            "Pattern Dashboard",
            "Pattern Details",
            "Risk Matrix",
            "Recommendations",
        ]
    )

    with tab_dash:
        _render_pattern_dashboard()

    with tab_details:
        _render_pattern_details()

    with tab_risk:
        _render_risk_matrix()

    with tab_recs:
        _render_recommendations()


# ═══════════════════════════════════════════════════════════════════════════
#  Tab 1: Pattern Dashboard
# ═══════════════════════════════════════════════════════════════════════════


def _render_pattern_dashboard():
    repo = _get_repo()
    engine = _get_engine()
    if not repo or not engine:
        return

    # Summary cards
    summary = repo.get_pattern_summary()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patterns", summary.get("total", 0))
    col2.metric("Active", summary.get("by_status", {}).get("active", 0))
    col3.metric("Escalated", summary.get("by_status", {}).get("escalated", 0))
    col4.metric("Resolved", summary.get("by_status", {}).get("resolved", 0))

    st.divider()

    # Run detection button
    if st.button("Run Pattern Detection", type="primary", use_container_width=True):
        with st.spinner("Analyzing historical incidents..."):
            incidents = _load_incidents()
            if not incidents:
                st.info("No historical incidents available for pattern detection.")
                return
            patterns = engine.detect_recurring_patterns(incidents)
            st.success(f"Detected {len(patterns)} patterns")
            st.rerun()

    # Pattern list
    st.markdown("#### Detected Patterns")
    status_filter = st.selectbox(
        "Filter by status",
        ["All", "active", "resolved", "escalated"],
        key="pat_status_filter",
    )
    type_filter = st.selectbox(
        "Filter by type",
        ["All"] + list(_PATTERN_TYPE_LABELS.values()),
        key="pat_type_filter",
    )

    patterns = repo.get_patterns(
        status=status_filter if status_filter != "All" else None,
        limit=50,
    )

    if not patterns:
        st.info("No patterns detected yet. Click 'Run Pattern Detection' to analyze.")
        return

    # Filter by type if selected
    if type_filter != "All":
        reverse_labels = {v: k for k, v in _PATTERN_TYPE_LABELS.items()}
        target_type = reverse_labels.get(type_filter, type_filter)
        patterns = [p for p in patterns if p.get("pattern_type") == target_type]

    for pattern in patterns:
        ptype_label = _PATTERN_TYPE_LABELS.get(
            pattern.get("pattern_type", ""), pattern.get("pattern_type", "unknown")
        )
        severity = pattern.get("severity", "Low")
        severity_color = _SEVERITY_COLORS.get(severity, "gray")
        confidence = pattern.get("confidence_score", 0)

        with st.expander(
            f"**{ptype_label}** | {severity} | "
            f"Confidence: {confidence:.0%} | "
            f"Occurrences: {pattern.get('occurrence_count', 0)}",
            expanded=severity in ("Critical", "High"),
        ):
            st.markdown(f"**Description:** {pattern.get('description', '')}")
            st.markdown(f"**Pattern ID:** `{pattern.get('pattern_id', '')}`")
            st.markdown(f"**Status:** {pattern.get('status', 'active')}")
            st.markdown(f"**First seen:** {pattern.get('first_seen', '')}")
            st.markdown(f"**Last seen:** {pattern.get('last_seen', '')}")

            doc_group = pattern.get("document_group", "[]")
            if isinstance(doc_group, str):
                try:
                    doc_group = json.loads(doc_group)
                except json.JSONDecodeError:
                    doc_group = [doc_group]
            if doc_group:
                st.markdown("**Documents involved:**")
                for doc in doc_group:
                    st.markdown(f"  - {doc}")

            author_group = pattern.get("author_group")
            if author_group and author_group != "null":
                if isinstance(author_group, str):
                    try:
                        author_group = json.loads(author_group)
                    except json.JSONDecodeError:
                        author_group = [author_group]
                if author_group:
                    st.markdown(f"**Authors:** {', '.join(author_group)}")

            assignment = pattern.get("assignment_title")
            if assignment:
                st.markdown(f"**Assignment:** {assignment}")

            # Status management
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("Mark Resolved", key=f"res_{pattern['pattern_id']}"):
                    repo.update_pattern_status(pattern["pattern_id"], "resolved")
                    st.rerun()
            with col_b:
                if st.button("Escalate", key=f"esc_{pattern['pattern_id']}"):
                    repo.update_pattern_status(pattern["pattern_id"], "escalated")
                    st.rerun()
            with col_c:
                if st.button("Dismiss", key=f"dismiss_{pattern['pattern_id']}"):
                    repo.update_pattern_status(pattern["pattern_id"], "dismissed")
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  Tab 2: Pattern Details (Evolution)
# ═══════════════════════════════════════════════════════════════════════════


def _render_pattern_details():
    repo = _get_repo()
    engine = _get_engine()
    if not repo or not engine:
        return

    patterns = repo.get_patterns(status="active", limit=100)
    if not patterns:
        st.info("No active patterns to display details for.")
        return

    pattern_ids = [p["pattern_id"] for p in patterns]
    selected_id = st.selectbox(
        "Select a pattern",
        pattern_ids,
        format_func=lambda x: _short_pattern_label(x, patterns),
    )

    if not selected_id:
        return

    pattern = repo.get_pattern_by_id(selected_id)
    if not pattern:
        st.warning("Pattern not found.")
        return

    # Pattern info
    ptype_label = _PATTERN_TYPE_LABELS.get(pattern.get("pattern_type", ""), "unknown")
    st.markdown(f"### {ptype_label} — `{selected_id}`")
    st.markdown(pattern.get("description", ""))

    # Record evolution snapshot
    if st.button("Record Evolution Snapshot", use_container_width=True):
        repo.record_evolution_snapshot(
            pattern_id=selected_id,
            occurrence_count=pattern.get("occurrence_count", 0),
            avg_similarity=pattern.get("avg_similarity", 0),
            confidence_score=pattern.get("confidence_score", 0),
        )
        st.success("Snapshot recorded.")
        st.rerun()

    # Evolution chart
    evolution_data = repo.get_evolution_timeseries(selected_id, days=90)
    if evolution_data:
        st.markdown("#### Evolution Over Time")
        evo_df = pd.DataFrame(evolution_data)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=evo_df["snapshot_date"],
                y=evo_df["occurrence_count"],
                name="Occurrences",
                line=dict(color="#3b82f6", width=2),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=evo_df["snapshot_date"],
                y=evo_df["avg_similarity"],
                name="Avg Similarity",
                line=dict(color="#ef4444", width=2, dash="dash"),
            ),
            secondary_y=True,
        )
        if "drift_score" in evo_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=evo_df["snapshot_date"],
                    y=evo_df["drift_score"],
                    name="Drift Score",
                    line=dict(color="#f59e0b", width=1, dash="dot"),
                ),
                secondary_y=True,
            )
        fig.update_layout(
            title="Pattern Evolution",
            xaxis_title="Date",
            height=400,
            legend=dict(orientation="h", y=-0.15),
        )
        fig.update_yaxes(title_text="Occurrences", secondary_y=False)
        fig.update_yaxes(title_text="Score", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No evolution snapshots yet. Click 'Record Evolution Snapshot' to start tracking."
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Tab 3: Risk Matrix
# ═══════════════════════════════════════════════════════════════════════════


def _render_risk_matrix():
    repo = _get_repo()
    engine = _get_engine()
    if not repo or not engine:
        return

    st.markdown("#### Document Risk Assessment")

    # Score a single document
    st.markdown("**Score a Document**")
    with st.form("risk_score_form"):
        col1, col2 = st.columns(2)
        with col1:
            doc_name = st.text_input("Document name")
            max_sim = st.slider("Max similarity", 0.0, 1.0, 0.5)
            author_incidents = st.number_input("Author prior incidents", 0, 20, 0)
        with col2:
            avg_sim = st.slider("Avg similarity", 0.0, 1.0, 0.3)
            assignment_rate = st.slider("Assignment incident rate", 0.0, 1.0, 0.1)
            submission_hour = st.slider("Submission hour", 0, 23, 12)

        submitted = st.form_submit_button("Score Document", type="primary")
        if submitted and doc_name:
            result = engine.score_document_risk(
                doc_name,
                {
                    "max_similarity": max_sim,
                    "avg_similarity": avg_sim,
                    "author_incident_count": author_incidents,
                    "author_avg_similarity": avg_sim * 0.9,
                    "assignment_incident_rate": assignment_rate,
                    "document_length": 5000,
                    "submission_hour": submission_hour,
                    "days_until_deadline": 7,
                    "class_section_risk": 0.1,
                },
            )
            risk_color = _RISK_LEVEL_COLORS.get(result["risk_level"], "gray")
            st.markdown(
                f"**Risk Score:** {result['risk_score']:.2%} | "
                f"**Level:** :{risk_color}[{result['risk_level']}]"
            )
            st.markdown("**Contributing Factors:**")
            for factor in result["contributing_factors"]:
                st.markdown(f"  - {factor}")

    st.divider()

    # Risk distribution
    distribution = repo.get_risk_distribution()
    if distribution:
        st.markdown("#### Risk Distribution")
        dist_df = pd.DataFrame(
            list(distribution.items()), columns=["Risk Level", "Count"]
        )
        fig = px.bar(
            dist_df,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            color_discrete_map=_RISK_LEVEL_COLORS,
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # High-risk documents
    high_risk = repo.get_high_risk_documents(threshold=0.6)
    if high_risk:
        st.markdown("#### High-Risk Documents")
        hr_df = pd.DataFrame(high_risk)
        display_cols = ["document_name", "risk_score", "risk_level", "scored_at"]
        available_cols = [c for c in display_cols if c in hr_df.columns]
        st.dataframe(hr_df[available_cols], use_container_width=True)
    else:
        st.info("No high-risk documents identified yet.")


# ═══════════════════════════════════════════════════════════════════════════
#  Tab 4: Recommendations
# ═══════════════════════════════════════════════════════════════════════════


def _render_recommendations():
    repo = _get_repo()
    engine = _get_engine()
    rec_engine = _get_rec_engine()
    if not repo or not engine or not rec_engine:
        return

    st.markdown("#### Proactive Recommendations")

    if st.button("Generate Recommendations", type="primary", use_container_width=True):
        with st.spinner("Generating recommendations..."):
            incidents = _load_incidents()
            patterns = repo.get_patterns(status="active", limit=50)
            high_risk = repo.get_high_risk_documents(threshold=0.5)

            trend_data = {}
            if incidents:
                trend_data = engine.detect_technique_evolution(incidents)

            rec_engine.generate_recommendations(
                patterns=patterns,
                risk_scores=high_risk,
                trend_data=trend_data,
            )
            st.success("Recommendations generated.")
            st.rerun()

    # Filter
    status_filter = st.selectbox(
        "Filter by status",
        ["All", "pending", "acknowledged", "dismissed"],
        key="rec_status_filter",
    )

    recommendations = repo.get_recommendations(
        status=status_filter if status_filter != "All" else None,
        limit=50,
    )

    if not recommendations:
        st.info("No recommendations yet. Click 'Generate Recommendations' to analyze.")
        return

    for rec in recommendations:
        priority = rec.get("priority", 3)
        priority_label = _PRIORITY_LABELS.get(priority, f"P{priority}")
        rec_type_label = _REC_TYPE_LABELS.get(
            rec.get("recommendation_type", ""),
            rec.get("recommendation_type", "unknown"),
        )

        with st.expander(
            f"**{priority_label}** | {rec_type_label} | Target: {rec.get('target', '')}",
            expanded=priority <= 2,
        ):
            st.markdown(f"**Message:** {rec.get('message', '')}")

            action_items = rec.get("action_items")
            if action_items and action_items != "null":
                if isinstance(action_items, str):
                    try:
                        action_items = json.loads(action_items)
                    except json.JSONDecodeError:
                        action_items = [action_items]
                if action_items:
                    st.markdown("**Action Items:**")
                    for item in action_items:
                        st.markdown(f"  1. {item}")

            st.caption(
                f"Created: {rec.get('created_at', '')} | Status: {rec.get('status', 'pending')}"
            )

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("Acknowledge", key=f"ack_{rec['recommendation_id']}"):
                    rec_engine.acknowledge_recommendation(rec["recommendation_id"])
                    st.rerun()
            with col_b:
                if st.button("Dismiss", key=f"dismiss_{rec['recommendation_id']}"):
                    rec_engine.dismiss_recommendation(rec["recommendation_id"])
                    st.rerun()

    # Stats
    stats = repo.get_recommendation_stats()
    if stats:
        st.divider()
        st.markdown("#### Recommendation Statistics")
        stats_df = pd.DataFrame(list(stats.items()), columns=["Status", "Count"])
        st.dataframe(stats_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

_PATTERN_TYPE_LABELS = {
    "copy_paste": "Copy-Paste",
    "paraphrase": "Paraphrase",
    "source_sharing": "Source Sharing",
    "collaborative": "Collaborative",
    "template_reuse": "Template Reuse",
}

_SEVERITY_COLORS = {
    "Critical": "red",
    "High": "orange",
    "Medium": "blue",
    "Low": "green",
}

_RISK_LEVEL_COLORS = {
    "Critical": "#dc2626",
    "High": "#ea580c",
    "Medium": "#ca8a04",
    "Low": "#16a34a",
    "Negligible": "#9ca3af",
}

_PRIORITY_LABELS = {
    1: "CRITICAL",
    2: "HIGH",
    3: "MEDIUM",
    4: "LOW",
}

_REC_TYPE_LABELS = {
    "monitor_author": "Monitor Author",
    "escalate": "Escalate",
    "redesign_assignment": "Redesign Assignment",
    "policy_review": "Policy Review",
    "strengthen_detection": "Strengthen Detection",
    "stagger_deadlines": "Stagger Deadlines",
}


def _load_incidents() -> list[dict[str, Any]]:
    """Load all incidents from the database for pattern detection."""
    try:
        from src.db.incidents import get_all_incidents, get_total_incidents_count

        total = get_total_incidents_count()
        if total == 0:
            return []
        return get_all_incidents(limit=total, offset=0)
    except Exception as exc:
        logger.warning("Failed to load incidents: %s", exc)
        return []


def _short_pattern_label(pattern_id: str, patterns: list[dict]) -> str:
    """Create a short display label for a pattern ID."""
    for p in patterns:
        if p.get("pattern_id") == pattern_id:
            ptype = _PATTERN_TYPE_LABELS.get(p.get("pattern_type", ""), "unknown")
            occ = p.get("occurrence_count", 0)
            return f"{ptype} ({occ} occurrences) — {pattern_id}"
    return pattern_id


def make_subplots(specs=None):
    """Lazy import wrapper for plotly.subplots.make_subplots."""
    from plotly.subplots import make_subplots as _make_subplots

    return _make_subplots(specs=specs)
