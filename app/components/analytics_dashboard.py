"""
analytics_dashboard.py
----------------------
Streamlit dashboard component for the Plagiarism Analytics view.
Displays trend charts, severity distributions, risk profiles,
anomaly detection, and supports multi-format report export.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from src.core.report_export_analytics import (
    TrendPoint,
    compute_rolling_averages,
    detect_scan_anomalies,
    generate_analytics_summary,
    threshold_sensitivity_analysis,
)
from src.utils.analytics_export import export_analytics


def _load_scan_history() -> List[Dict[str, Any]]:
    try:
        from src.db.corpus_db import get_scan_history
        return get_scan_history(limit=500)
    except Exception as exc:
        logger.warning("Failed to load scan history: %s", exc)
        return []


def _load_incidents() -> List[Dict[str, Any]]:
    try:
        from src.db.corpus_db import _connect
        with _connect() as conn:
            rows = conn.execute(
                "SELECT incident_id, document_a, document_b, "
                "similarity_score, severity_rank, review_status, "
                "date_flagged, last_seen, threshold_at_time_of_flag "
                "FROM plagiarism_incidents ORDER BY date_flagged DESC"
            ).fetchall()
            return [
                {"incident_id": r[0], "document_a": r[1], "document_b": r[2],
                 "similarity_score": r[3], "severity_rank": r[4],
                 "review_status": r[5], "date_flagged": r[6],
                 "last_seen": r[7], "threshold_at_time_of_flag": r[8]}
                for r in rows
            ]
    except Exception as exc:
        logger.warning("Failed to load incidents: %s", exc)
        return []


def _load_doc_count() -> int:
    try:
        from src.db.corpus_db import get_document_count_fast
        return get_document_count_fast()
    except Exception:
        return 0


def _df_or_fallback(columns: Dict[str, list], fallback_text: str):
    if not PANDAS_AVAILABLE:
        st.info(fallback_text)
        return None
    df = pd.DataFrame(columns)
    return df


def render_analytics_dashboard() -> None:
    if not STREAMLIT_AVAILABLE:
        raise RuntimeError("Streamlit required. Install: pip install streamlit")

    st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")
    st.title("📊 Plagiarism Analytics Dashboard")

    # ── Controls ──
    c1, c2 = st.columns(2)
    with c1:
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            key="analytics_date_range",
        )
    with c2:
        granularity = st.selectbox(
            "Trend Granularity", ["daily", "weekly", "monthly"], key="analytics_gran"
        )

    # ── Load data ──
    scan_history = _load_scan_history()
    incidents = _load_incidents()
    total_documents = _load_doc_count()

    if date_range and len(date_range) == 2:
        s, e = date_range[0].isoformat(), date_range[1].isoformat()
        scan_history = [r for r in scan_history if s <= r.get("timestamp", "")[:10] <= e]

    summary = generate_analytics_summary(scan_history, incidents, total_documents)

    # ── KPI Cards ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scans", summary.total_scans)
    c2.metric("Flagged Incidents", summary.total_incidents)
    c3.metric("Documents", summary.total_documents)
    c4.metric("Flagged Rate", f"{round(summary.flagged_rate * 100, 1)}%")
    st.divider()

    # ── Trend Chart ──
    trend_map = {"daily": summary.daily_trends, "weekly": summary.weekly_trends, "monthly": summary.monthly_trends}
    active_trends = trend_map.get(granularity, summary.daily_trends)
    st.subheader(f"{granularity.title()} Trend — Avg Similarity")

    if active_trends:
        if PANDAS_AVAILABLE:
            df = pd.DataFrame({
                "Period": [t.timestamp for t in active_trends],
                "Avg Sim": [t.avg_similarity for t in active_trends],
                "Max Sim": [t.max_similarity for t in active_trends],
                "Flagged": [t.flagged_count for t in active_trends],
            }).set_index("Period")
            st.line_chart(df)
        else:
            for t in active_trends:
                st.write(f"**{t.timestamp}** — Avg: {t.avg_similarity:.3f} | Max: {t.max_similarity:.3f} | Flagged: {t.flagged_count}")
    else:
        st.info("No trend data available.")
    st.divider()

    # ── Severity + Risk side-by-side ──
    cs1, cs2 = st.columns(2)
    with cs1:
        st.subheader("Severity Distribution")
        if summary.severity_distribution:
            if PANDAS_AVAILABLE:
                df = pd.DataFrame({
                    "Severity": [b.label for b in summary.severity_distribution],
                    "Count": [b.count for b in summary.severity_distribution],
                }).set_index("Severity")
                st.bar_chart(df)
            else:
                for b in summary.severity_distribution:
                    st.write(f"**{b.label}**: {b.count} ({b.percentage}%) — Avg: {b.avg_score}")
        else:
            st.info("No severity data.")

    with cs2:
        st.subheader("Top Risk Documents")
        if summary.top_risk_documents:
            if PANDAS_AVAILABLE:
                df = pd.DataFrame({
                    "Document": [d.filename for d in summary.top_risk_documents],
                    "Incidents": [d.incident_count for d in summary.top_risk_documents],
                    "Avg": [d.avg_similarity for d in summary.top_risk_documents],
                    "Max": [d.max_similarity for d in summary.top_risk_documents],
                    "Severity": [d.severity for d in summary.top_risk_documents],
                })
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                for d in summary.top_risk_documents:
                    st.write(f"**{d.filename}** — {d.incident_count} incidents, max: {d.max_similarity}, {d.severity}")
        else:
            st.info("No flagged documents.")
    st.divider()

    # ── Rolling Averages ──
    st.subheader("Rolling Averages (7-period window)")
    rolling = compute_rolling_averages(active_trends, window=7)
    if rolling:
        if PANDAS_AVAILABLE:
            st.line_chart(pd.DataFrame(rolling).set_index("timestamp"))
        else:
            for r in rolling:
                st.write(f"**{r['timestamp']}** — Sim: {r['rolling_avg_similarity']:.3f} | Flagged: {r['rolling_avg_flagged']:.1f}")
    else:
        st.info("Insufficient data for rolling averages.")

    # ── Anomaly Detection ──
    st.subheader("Anomaly Detection")
    anomalies = detect_scan_anomalies(active_trends)
    if anomalies:
        st.warning(f"Found {len(anomalies)} anomalous data points.")
        if PANDAS_AVAILABLE:
            st.dataframe(pd.DataFrame(anomalies), use_container_width=True, hide_index=True)
    else:
        st.success("No anomalies detected.")
    st.divider()

    # ── Threshold Sensitivity ──
    st.subheader("Threshold Sensitivity Analysis")
    analysis = threshold_sensitivity_analysis(incidents, scan_history)
    if analysis:
        if PANDAS_AVAILABLE:
            st.line_chart(pd.DataFrame(analysis).set_index("threshold")["incident_count"])
        for entry in analysis:
            st.write(f"Threshold **{entry['threshold']:.2f}**: {entry['incident_count']} incidents ({entry['pct_of_total']}%)")
    else:
        st.info("No data for threshold analysis.")
    st.divider()

    # ── Export ──
    st.subheader("Export Report")
    ec1, ec2 = st.columns(2)
    with ec1:
        fmt = st.selectbox("Format", ["json", "csv", "html"], key="analytics_fmt")
    with ec2:
        inc_trends = st.checkbox("Include trends", value=True, key="analytics_inc")

    if st.button("Generate Report", key="analytics_gen"):
        output = export_analytics(summary, format=fmt)
        mime = {"json": "application/json", "csv": "text/csv", "html": "text/html"}
        fname = f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt}"
        st.download_button(f"Download {fmt.upper()}", output, fname, mime.get(fmt, "text/plain"))
