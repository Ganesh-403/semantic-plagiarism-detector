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
dashboard_stats.py
------------------
Streamlit dashboard statistics and analytics component.
Displays real-time overview of plagiarism detection statistics, including 8 KPI cards,
responsive Plotly charts, a recent incidents feed, and summary tables.
"""

from __future__ import annotations

import html
import logging
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Setup logging
logger = logging.getLogger(__name__)


# ── DATABASE LOADING FUNCTIONS WITH CACHING ───────────────────────────────────


@st.cache_data(ttl=60)
def _load_documents_cached() -> list[Any]:
    """Fetch all non-deleted documents from the database, cached for performance."""
    try:
        from src.db.corpus_db import get_all_documents

        return get_all_documents(include_deleted=False)
    except Exception as e:
        logger.error("Failed to load documents from database: %s", e)
        return []


@st.cache_data(ttl=60)
def _load_incidents_cached() -> list[Any]:
    """Fetch all incidents from the database, cached for performance."""
    try:
        from src.db.incidents import get_all_incidents

        # Retrieve incidents with a sufficiently large limit for dashboarding
        return get_all_incidents(limit=100000)
    except Exception as e:
        logger.error("Failed to load incidents from database: %s", e)
        return []


@st.cache_data(ttl=60)
def _load_high_severity_trends_cached(days: int = 30) -> list[dict[str, Any]]:
    """Fetch daily high severity incident trend counts, cached for performance."""
    try:
        from src.db.incidents import get_high_severity_trends

        return get_high_severity_trends(days=days)
    except Exception as e:
        logger.error("Failed to load high severity trends: %s", e)
        return []


@st.cache_data(ttl=60)
def _load_most_plagiarized_documents_cached(limit: int = 10) -> list[dict[str, Any]]:
    """Fetch the most frequently plagiarized documents, cached for performance."""
    try:
        from src.db.incidents import get_most_plagiarized_documents

        return get_most_plagiarized_documents(limit=limit)
    except Exception as e:
        logger.error("Failed to load most plagiarized documents: %s", e)
        return []


@st.cache_data(ttl=60)
def _load_document_count_cached() -> int:
    """Fetch total document count, cached for performance."""
    try:
        from src.db.corpus_db import get_total_document_count

        return get_total_document_count(include_deleted=False)
    except Exception as e:
        logger.error("Failed to load document count: %s", e)
        return 0


@st.cache_data(ttl=60)
def _load_storage_footprint_cached() -> dict[str, Any]:
    """Fetch vector embedding storage footprint, cached for performance."""
    try:
        from src.db.corpus_db import get_embedding_storage_footprint

        return get_embedding_storage_footprint()
    except Exception as e:
        logger.error("Failed to load storage footprint: %s", e)
        return {
            "embedding_bytes": 0,
            "database_bytes": 0,
            "embedding_percentage": 0.0,
            "chunk_count": 0,
        }


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, handling exceptions."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int, handling exceptions."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _normalise_severity(value: Any) -> str:
    """Normalise severity value to 'High', 'Medium', or 'Low'."""
    if not value:
        return "Low"
    val_str = str(value).strip().lower()
    if "high" in val_str:
        return "High"
    elif "med" in val_str or "medium" in val_str:
        return "Medium"
    return "Low"


def _normalise_review_status(value: Any) -> str:
    """Normalise review status value to 'Pending' or 'Resolved'."""
    if not value:
        return "Pending"
    val_str = str(value).strip().lower()
    if "resolved" in val_str or "resolve" in val_str:
        return "Resolved"
    return "Pending"


def _incidents_to_dataframe(incidents: list[Any]) -> pd.DataFrame:
    """Convert raw incidents lists (dicts or Pydantic models) into a Pandas DataFrame."""
    if not incidents:
        return pd.DataFrame(
            columns=[
                "incident_id",
                "document_a",
                "document_b",
                "similarity_score",
                "severity_rank",
                "review_status",
                "date_flagged",
                "last_seen",
            ]
        )
    rows = []
    for inc in incidents:
        is_dict = isinstance(inc, dict)

        def _get_val(key: str, default: Any = None) -> Any:
            if is_dict:
                return inc.get(key, default)
            try:
                return getattr(inc, key, default)
            except AttributeError:
                return default

        inc_id = _get_val("incident_id")
        doc_a = _get_val("document_a") or _get_val("doc_a") or "Unknown"
        doc_b = _get_val("document_b") or _get_val("doc_b") or "Unknown"
        similarity = _safe_float(_get_val("similarity_score") or _get_val("similarity"))
        severity = _normalise_severity(
            _get_val("severity_rank") or _get_val("severity")
        )
        status = _normalise_review_status(_get_val("review_status"))
        date_flg = _get_val("date_flagged")
        last_seen = _get_val("last_seen")

        rows.append(
            {
                "incident_id": inc_id,
                "document_a": doc_a,
                "document_b": doc_b,
                "similarity_score": similarity,
                "severity_rank": severity,
                "review_status": status,
                "date_flagged": date_flg,
                "last_seen": last_seen,
            }
        )
    return pd.DataFrame(rows)


def _calculate_dashboard_stats(df: pd.DataFrame, total_docs: int) -> dict[str, Any]:
    """Calculate statistics for metrics, summary, and widgets from incidents DataFrame."""
    total_incidents = len(df)

    if total_incidents > 0:
        raw_scores = df["similarity_score"].astype(float)
        scaled_scores = raw_scores.apply(lambda x: x * 100.0 if x <= 1.0 else x)
        avg_similarity = float(scaled_scores.mean())
        max_similarity = float(scaled_scores.max())

        high_severity_count = int((df["severity_rank"] == "High").sum())
        medium_severity_count = int((df["severity_rank"] == "Medium").sum())

        pending_count = int((df["review_status"] == "Pending").sum())
        resolved_count = int((df["review_status"] == "Resolved").sum())

        pending_review_pct = (pending_count / total_incidents) * 100.0
    else:
        avg_similarity = 0.0
        max_similarity = 0.0
        high_severity_count = 0
        medium_severity_count = 0
        pending_count = 0
        resolved_count = 0
        pending_review_pct = 0.0

    incident_rate = (total_incidents / total_docs * 100.0) if total_docs > 0 else 0.0

    return {
        "total_documents": total_docs,
        "total_incidents": total_incidents,
        "high_severity_count": high_severity_count,
        "medium_severity_count": medium_severity_count,
        "pending_reviews_count": pending_count,
        "resolved_reviews_count": resolved_count,
        "avg_similarity": avg_similarity,
        "max_similarity": max_similarity,
        "incident_rate": incident_rate,
        "pending_review_pct": pending_review_pct,
    }


# ── UI RENDERING HELPERS ──────────────────────────────────────────────────────


def _render_metric_card(
    label: str,
    value: str | int | float,
    icon: str,
    description: str,
    accent_color_var: str,
) -> None:
    """Render a styled KPI metric card using custom HTML/CSS."""
    card_html = f"""
    <div class="kpi-card" style="border-bottom-color: var({accent_color_var});">
        <div class="kpi-card-header">
            <span class="kpi-card-title">{label}</span>
            <span class="kpi-card-icon">{icon}</span>
        </div>
        <div class="kpi-card-value">{value}</div>
        <div class="kpi-card-desc">{description}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def _render_empty_state() -> None:
    """Render an elegant empty state when there is no data."""
    st.markdown(
        """
        <div class="empty-state" style="text-align: center; padding: 50px 20px; border: 1px dashed var(--border-color, #E2E8F0); border-radius: 12px; background-color: var(--secondary-bg, #F8FAFC); margin: 20px 0;">
            <div class="empty-icon" style="font-size: 3.5rem; margin-bottom: 12px;">🛡️</div>
            <div class="empty-title" style="font-size: 1.35rem; font-weight: 700; color: var(--text-color, #0F172A); margin-bottom: 8px;">No Plagiarism Incidents Logged</div>
            <div class="empty-desc" style="font-size: 0.95rem; color: var(--secondary-text-color, #64748B); max-width: 500px; margin: 0 auto;">The system has not flagged any document matches. Ensure the document corpus contains files, and running scans will populate this panel.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_recent_incidents(recent_df: pd.DataFrame) -> None:
    """Render a grid of the latest 8 plagiarism incidents."""
    st.markdown("### Recent Flagged Incidents")
    if recent_df.empty:
        _render_empty_state()
        return

    df_recent = recent_df.head(8)
    cards_html = []

    for _, row in df_recent.iterrows():
        doc_a = html.escape(str(row["document_a"]))
        doc_b = html.escape(str(row["document_b"]))
        similarity = _safe_float(row["similarity_score"])
        similarity_pct = similarity * 100.0 if similarity <= 1.0 else similarity

        severity = row["severity_rank"]
        status = row["review_status"]
        date_flagged = (
            str(row["date_flagged"]) if pd.notna(row["date_flagged"]) else "N/A"
        )
        if " " in date_flagged:
            date_flagged = date_flagged.split(" ")[0]

        # Determine colors for badges
        if severity == "High":
            sev_bg = "var(--danger-soft, #FEE2E2)"
            sev_color = "var(--danger, #FF4B4B)"
        elif severity == "Medium":
            sev_bg = "var(--warning-soft, #FEF3C7)"
            sev_color = "var(--warning, #FFA500)"
        else:
            sev_bg = "var(--success-soft, #DCFCE7)"
            sev_color = "var(--success, #21C55D)"

        if status == "Resolved":
            stat_bg = "var(--success-soft, #DCFCE7)"
            stat_color = "var(--success, #21C55D)"
        else:
            stat_bg = "var(--warning-soft, #FEF3C7)"
            stat_color = "var(--warning, #FFA500)"

        card_html = f"""
        <div class="incident-card">
            <div class="incident-card-header">
                <span class="incident-sim-value">{similarity_pct:.1f}% Match</span>
                <span class="incident-card-date">{date_flagged}</span>
            </div>
            <div class="incident-docs-container">
                <div class="doc-line">
                    <span class="doc-prefix">Doc A:</span>
                    <span class="doc-name-text" title="{doc_a}">{doc_a}</span>
                </div>
                <div class="doc-line">
                    <span class="doc-prefix">Doc B:</span>
                    <span class="doc-name-text" title="{doc_b}">{doc_b}</span>
                </div>
            </div>
            <div class="incident-card-badges">
                <span class="incident-badge" style="background-color: {sev_bg}; color: {sev_color}; border: 1px solid {sev_color};">{severity}</span>
                <span class="incident-badge" style="background-color: {stat_bg}; color: {stat_color}; border: 1px solid {stat_color};">{status}</span>
            </div>
        </div>
        """
        cards_html.append(card_html)

    grid_html = f"""
    <div class="incidents-grid">
        {"".join(cards_html)}
    </div>
    """
    st.markdown(grid_html, unsafe_allow_html=True)


def _render_summary(stats: dict[str, Any]) -> None:
    """Render the summary section containing key percentage ratios."""
    st.markdown("### Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _render_metric_card(
            label="Incident Rate",
            value=f"{stats['incident_rate']:.1f}%",
            icon="📊",
            description="Incidents / corpus ratio",
            accent_color_var="--accent-color",
        )
    with col2:
        _render_metric_card(
            label="Pending Review %",
            value=f"{stats['pending_review_pct']:.1f}%",
            icon="⏳",
            description="Proportion awaiting review",
            accent_color_var="--warning",
        )
    with col3:
        _render_metric_card(
            label="Average Similarity",
            value=f"{stats['avg_similarity']:.1f}%",
            icon="⚖️",
            description="Mean similarity index",
            accent_color_var="--accent-color",
        )
    with col4:
        _render_metric_card(
            label="Highest Similarity",
            value=f"{stats['max_similarity']:.1f}%",
            icon="🔥",
            description="Maximum similarity logged",
            accent_color_var="--danger",
        )


# ── PLOTLY CHART HELPERS ──────────────────────────────────────────────────────


def _apply_plotly_theme(
    fig: go.Figure, title: str, theme_colors: dict[str, str], margin_left: int = 50
) -> None:
    """Style Plotly figures layout to match active light or dark themes."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                family="'Inter', sans-serif",
                size=16,
                color=theme_colors.get("ink", "#0F172A"),
            ),
            x=0.02,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="'Inter', sans-serif", color=theme_colors.get("ink", "#0F172A")
        ),
        margin=dict(l=margin_left, r=30, t=60, b=40),
        legend=dict(
            font=dict(color=theme_colors.get("muted", "#64748B")),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor=theme_colors.get("border", "#E2E8F0"),
            tickfont=dict(color=theme_colors.get("muted", "#64748B")),
            titlefont=dict(color=theme_colors.get("ink", "#0F172A")),
        ),
        yaxis=dict(
            gridcolor=theme_colors.get("border", "#E2E8F0"),
            tickfont=dict(color=theme_colors.get("muted", "#64748B")),
            titlefont=dict(color=theme_colors.get("ink", "#0F172A")),
        ),
    )


# ── MAIN COMPONENT ENTRYPOINT ─────────────────────────────────────────────────


def render_dashboard_stats() -> None:
    """Entrypoint function to render the Semantic Plagiarism Analytics Dashboard.
    Fetches database statistics, prepares data tables, and draws the metric panels and Plotly graphs.
    """
    # 1. Inject custom styling CSS matching theme variables
    st.markdown(
        """
        <style>
        .dashboard-header {
            margin-bottom: 25px;
            border-bottom: 2px solid var(--border-color, #E2E8F0);
            padding-bottom: 15px;
        }
        .dashboard-title {
            font-size: 2.25rem;
            font-weight: 800;
            color: var(--text-color, #0F172A);
            margin: 0 0 8px 0;
            letter-spacing: -0.025em;
        }
        .dashboard-subtitle {
            font-size: 1.05rem;
            color: var(--secondary-text-color, #64748B);
            margin: 0;
        }
        .kpi-card {
            background-color: var(--card, #FFFFFF);
            border: 1px solid var(--border-color, #E2E8F0);
            border-bottom: 4px solid var(--accent-color, #0D9488);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
            transition: transform 0.25s ease-in-out, box-shadow 0.25s ease-in-out;
            display: flex;
            flex-direction: column;
            min-height: 140px;
        }
        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        .kpi-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .kpi-card-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--secondary-text-color, #64748B);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .kpi-card-icon {
            font-size: 1.4rem;
        }
        .kpi-card-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--text-color, #0F172A);
            line-height: 1.2;
        }
        .kpi-card-desc {
            font-size: 0.75rem;
            color: var(--secondary-text-color, #64748B);
            margin-top: 5px;
        }
        .incidents-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 16px;
            margin-top: 10px;
            margin-bottom: 25px;
        }
        .incident-card {
            background-color: var(--card, #FFFFFF);
            border: 1px solid var(--border-color, #E2E8F0);
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-height: 155px;
        }
        .incident-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
        }
        .incident-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color, #E2E8F0);
            padding-bottom: 8px;
            margin-bottom: 12px;
        }
        .incident-sim-value {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--accent-color, #0D9488);
        }
        .incident-card-date {
            font-size: 0.75rem;
            color: var(--secondary-text-color, #64748B);
        }
        .incident-docs-container {
            margin-bottom: 12px;
        }
        .doc-line {
            display: flex;
            align-items: center;
            font-size: 0.8rem;
            margin-bottom: 6px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .doc-prefix {
            font-weight: 600;
            color: var(--secondary-text-color, #64748B);
            margin-right: 6px;
            min-width: 45px;
        }
        .doc-name-text {
            color: var(--text-color, #0F172A);
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .incident-card-badges {
            display: flex;
            gap: 8px;
            margin-top: auto;
        }
        .incident-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 700;
            text-align: center;
            font-family: 'IBM Plex Mono', monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render header
    st.markdown(
        """
        <div class="dashboard-header">
            <h1 class="dashboard-title">Semantic Plagiarism Analytics Dashboard</h1>
            <p class="dashboard-subtitle">Real-time overview of plagiarism detection statistics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Load active theme colors from theme utility
    try:
        from app.theme import get_chart_colors

        theme_colors = get_chart_colors()
    except Exception:
        theme_colors = {
            "background": "#FFFFFF",
            "surface": "#F8FAFC",
            "card": "#FFFFFF",
            "ink": "#0F172A",
            "muted": "#64748B",
            "accent": "#0D9488",
            "border": "#E2E8F0",
            "danger": "#FF4B4B",
            "warning": "#FFA500",
            "success": "#21C55D",
        }

    # 3. Load DB data
    total_docs = _load_document_count_cached()
    incidents_list = _load_incidents_cached()
    df = _incidents_to_dataframe(incidents_list)

    if total_docs == 0 and len(df) == 0:
        _render_empty_state()
        return

    # Calculate stats
    stats = _calculate_dashboard_stats(df, total_docs)

    # SECTION 1: 8 KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _render_metric_card(
            label="Total Documents",
            value=f"{stats['total_documents']:,}",
            icon="📄",
            description="Indexed in corpus",
            accent_color_var="--accent-color",
        )
    with col2:
        _render_metric_card(
            label="Total Incidents",
            value=f"{stats['total_incidents']:,}",
            icon="🚨",
            description="Total flagged incidents",
            accent_color_var="--muted",
        )
    with col3:
        _render_metric_card(
            label="High Severity",
            value=f"{stats['high_severity_count']:,}",
            icon="🔴",
            description="Similarity score >= 80%",
            accent_color_var="--danger",
        )
    with col4:
        _render_metric_card(
            label="Medium Severity",
            value=f"{stats['medium_severity_count']:,}",
            icon="🟡",
            description="Similarity score 50-79%",
            accent_color_var="--warning",
        )

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        _render_metric_card(
            label="Pending Reviews",
            value=f"{stats['pending_reviews_count']:,}",
            icon="⏳",
            description="Incidents awaiting review",
            accent_color_var="--warning",
        )
    with col6:
        _render_metric_card(
            label="Resolved Reviews",
            value=f"{stats['resolved_reviews_count']:,}",
            icon="🟢",
            description="Incidents marked resolved",
            accent_color_var="--success",
        )
    with col7:
        _render_metric_card(
            label="Average Similarity",
            value=f"{stats['avg_similarity']:.1f}%",
            icon="⚖️",
            description="Average index value",
            accent_color_var="--accent-color",
        )
    with col8:
        _render_metric_card(
            label="Highest Similarity",
            value=f"{stats['max_similarity']:.1f}%",
            icon="🔥",
            description="Maximum logged score",
            accent_color_var="--danger",
        )

    st.markdown("---")

    # SECTION 1.5: Database Vector Footprint
    footprint_stats = _load_storage_footprint_cached()
    st.markdown("### Vector Storage Footprint")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        _render_metric_card(
            label="Database Size",
            value=f"{footprint_stats['database_bytes'] / 1024 / 1024:.2f} MB",
            icon="🗄️",
            description="Total SQLite size",
            accent_color_var="--accent-color",
        )
    with col_f2:
        _render_metric_card(
            label="Embedding Size",
            value=f"{footprint_stats['embedding_bytes'] / 1024 / 1024:.2f} MB",
            icon="🧠",
            description="Total Vector storage",
            accent_color_var="--warning",
        )
    with col_f3:
        _render_metric_card(
            label="Embedding %",
            value=f"{footprint_stats['embedding_percentage']:.2f}%",
            icon="📈",
            description="Of total database size",
            accent_color_var="--accent-color",
        )
    with col_f4:
        _render_metric_card(
            label="Total Chunks",
            value=f"{footprint_stats['chunk_count']:,}",
            icon="🧩",
            description="Stored in database",
            accent_color_var="--success",
        )

    st.markdown("---")

    # SECTION 2: Plotly Charts
    st.markdown("### Visual Analytics")
    # Row 1 of charts: Severity Distribution (Pie) & Review Status (Donut)
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        labels_sev = ["High", "Medium", "Low"]
        values_sev = [
            stats["high_severity_count"],
            stats["medium_severity_count"],
            max(
                0,
                stats["total_incidents"]
                - stats["high_severity_count"]
                - stats["medium_severity_count"],
            ),
        ]
        if sum(values_sev) == 0:
            fig_sev = go.Figure()
            fig_sev.add_annotation(
                text="No severity data available", showarrow=False, font=dict(size=14)
            )
        else:
            fig_sev = go.Figure(
                data=[
                    go.Pie(
                        labels=labels_sev,
                        values=values_sev,
                        marker=dict(
                            colors=[
                                theme_colors.get("danger", "#FF4B4B"),
                                theme_colors.get("warning", "#FFA500"),
                                theme_colors.get("success", "#21C55D"),
                            ],
                            line=dict(
                                color=theme_colors.get("card", "#FFFFFF"), width=2
                            ),
                        ),
                        textinfo="percent+label",
                        insidetextorientation="radial",
                    )
                ]
            )
        _apply_plotly_theme(fig_sev, "Incident Severity Distribution", theme_colors)
        st.plotly_chart(fig_sev, use_container_width=True)

    with chart_col2:
        labels_status = ["Pending", "Resolved"]
        values_status = [
            stats["pending_reviews_count"],
            stats["resolved_reviews_count"],
        ]
        if sum(values_status) == 0:
            fig_status = go.Figure()
            fig_status.add_annotation(
                text="No review status data available",
                showarrow=False,
                font=dict(size=14),
            )
        else:
            fig_status = go.Figure(
                data=[
                    go.Pie(
                        labels=labels_status,
                        values=values_status,
                        marker=dict(
                            colors=[
                                theme_colors.get("warning", "#FFA500"),
                                theme_colors.get("success", "#21C55D"),
                            ],
                            line=dict(
                                color=theme_colors.get("card", "#FFFFFF"), width=2
                            ),
                        ),
                        textinfo="percent+label",
                        hole=0.4,
                    )
                ]
            )
        _apply_plotly_theme(fig_status, "Review Status Breakdown (Donut)", theme_colors)
        st.plotly_chart(fig_status, use_container_width=True)

    # Row 2 of charts: Similarity Score Distribution & Incident Trend over time
    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        fig_hist = go.Figure()
        if df.empty:
            fig_hist.add_annotation(
                text="No similarity score data available",
                showarrow=False,
                font=dict(size=14),
            )
        else:
            raw_scores = df["similarity_score"].astype(float)
            scaled_scores = raw_scores.apply(lambda x: x * 100.0 if x <= 1.0 else x)
            fig_hist.add_trace(
                go.Histogram(
                    x=scaled_scores,
                    nbinsx=20,
                    marker=dict(
                        color=theme_colors.get("accent", "#0D9488"),
                        line=dict(color=theme_colors.get("card", "#FFFFFF"), width=0.5),
                    ),
                    opacity=0.85,
                    hovertemplate="Score Range: %{x}%<br>Count: %{y}<extra></extra>",
                )
            )
            fig_hist.update_xaxes(range=[0, 100], title_text="Similarity Score (%)")
            fig_hist.update_yaxes(title_text="Incident Count")
        _apply_plotly_theme(fig_hist, "Similarity Score Distribution", theme_colors)
        st.plotly_chart(fig_hist, use_container_width=True)

    with chart_col4:
        fig_trend = go.Figure()
        if df.empty:
            fig_trend.add_annotation(
                text="No daily trend data available",
                showarrow=False,
                font=dict(size=14),
            )
        else:
            df["date"] = pd.to_datetime(df["date_flagged"], errors="coerce").dt.date
            trend_data = df.groupby("date").size().reset_index(name="count")
            trend_data = trend_data.sort_values("date")

            if trend_data.empty:
                fig_trend.add_annotation(
                    text="No daily trend data available",
                    showarrow=False,
                    font=dict(size=14),
                )
            else:
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_data["date"],
                        y=trend_data["count"],
                        mode="lines+markers",
                        line=dict(color=theme_colors.get("accent", "#0D9488"), width=3),
                        marker=dict(
                            color=theme_colors.get("danger", "#FF4B4B"),
                            size=8,
                            line=dict(
                                color=theme_colors.get("card", "#FFFFFF"), width=1.5
                            ),
                        ),
                        fill="tozeroy",
                        fillcolor="rgba(13, 148, 136, 0.1)",
                        hovertemplate="Date: %{x}<br>Incidents: %{y}<extra></extra>",
                    )
                )
                fig_trend.update_xaxes(title_text="Date Flagged")
                fig_trend.update_yaxes(title_text="Incident Count")
        _apply_plotly_theme(fig_trend, "Incident Trend (Daily)", theme_colors)
        st.plotly_chart(fig_trend, use_container_width=True)

    # Top Flagged Documents Horizontal Bar Chart
    most_plagiarized = _load_most_plagiarized_documents_cached(limit=10)
    fig_bar = go.Figure()
    if not most_plagiarized:
        fig_bar.add_annotation(
            text="No document plagiarism count data available",
            showarrow=False,
            font=dict(size=14),
        )
    else:
        bar_df = pd.DataFrame(most_plagiarized)
        bar_df = bar_df.sort_values("incident_count", ascending=True)
        fig_bar.add_trace(
            go.Bar(
                x=bar_df["incident_count"],
                y=bar_df["document_name"],
                orientation="h",
                marker=dict(
                    color=theme_colors.get("accent", "#0D9488"),
                    line=dict(color=theme_colors.get("card", "#FFFFFF"), width=1),
                ),
                opacity=0.9,
                hovertemplate="Document: %{y}<br>Incident Count: %{x}<extra></extra>",
            )
        )
        fig_bar.update_xaxes(title_text="Incident Count")
        fig_bar.update_yaxes(title_text="Document Name")
    _apply_plotly_theme(fig_bar, "Top Flagged Documents", theme_colors, margin_left=160)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # SECTION 3: Recent Incidents (Latest 8)
    _render_recent_incidents(df)

    st.markdown("---")

    # SECTION 4: Summary Statistics Row
    _render_summary(stats)
