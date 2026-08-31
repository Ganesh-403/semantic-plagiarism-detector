"""
Plagiarism Trend Analytics — Streamlit Dashboard Component
==========================================================
Interactive dashboard tab for viewing plagiarism detection trends,
statistical analysis, severity distributions, and institutional reports.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import streamlit as st

from src.core.plagiarism_trends import (
    AnalyticsReport,
    OffenderProfile,
    PlagiarismIncident,
    PlagiarismTrendAnalytics,
    ReportFormat,
    SeverityLevel,
    StatisticalSummary,
    SeverityDistribution,
    TimeWindow,
    TrendDirection,
    TrendResult,
    TrendWindow,
)
from src.visualization.trend_charts import (
    create_incident_timeline,
    create_severity_donut,
    create_severity_timeline,
    create_offender_bar_chart,
    create_similarity_trend_chart,
    create_static_incident_bar,
    create_static_severity_pie,
    create_trend_summary_card,
    create_window_comparison_heatmap,
    render_analytics_summary_metrics,
)

logger = logging.getLogger(__name__)

# ── Session Key Constants ─────────────────────────────────────────────────────

KEY_ANALYTICS_ENGINE = "trend_analytics_engine"
KEY_ANALYTICS_REPORT = "trend_analytics_report"
KEY_ANALYTICS_WINDOW = "trend_analytics_window"
KEY_ANALYTICS_LOADED = "trend_analytics_loaded"


def initialize_trend_analytics() -> PlagiarismTrendAnalytics:
    """Initialize or retrieve the trend analytics engine from session state."""
    if KEY_ANALYTICS_ENGINE not in st.session_state:
        st.session_state[KEY_ANALYTICS_ENGINE] = PlagiarismTrendAnalytics()
    return st.session_state[KEY_ANALYTICS_ENGINE]


def _seed_demo_data(engine: PlagiarismTrendAnalytics) -> None:
    """Seed the analytics engine with demo data for development and preview."""
    import random
    import hashlib

    documents = [
        "essay_ai_ethics.pdf", "research_ml_pipeline.docx",
        "thesis_nlp_review.pdf", "report_cloud_computing.txt",
        "assignment_data_structures.pdf", "paper_quantum_computing.pdf",
        "homework_web_dev.docx", "project_database_design.pdf",
        "essay_climate_change.pdf", "lab_report_chemistry.txt",
        "dissertation_robotics.pdf", "paper_biotech_ethics.docx",
        "report_cybersecurity.pdf", "essay_philosophy_mind.pdf",
        "assignment_linear_algebra.pdf",
    ]
    matchees = [
        "source_wikipedia_ai.pdf", "course_textbook_ml.pdf",
        "online_article_cloud.docx", "reference_nlp_survey.pdf",
        "textbook_data_structures.pdf", "paper_arxiv_quantum.pdf",
        "tutorial_web_basics.html", "guide_database_systems.pdf",
        "journal_climate_science.pdf", "lab_manual_chem.pdf",
    ]
    severities = ["low", "medium", "high", "critical"]

    random.seed(42)
    base_date = datetime.now(timezone.utc) - timedelta(days=180)

    incidents = []
    for day_offset in range(180):
        date = base_date + timedelta(days=day_offset)
        # Simulate increasing trend with some noise
        base_rate = 0.3 + (day_offset / 180.0) * 0.8
        n_incidents = max(0, int(random.gauss(base_rate, 0.5)))

        for _ in range(n_incidents):
            doc = random.choice(documents)
            match = random.choice(matchees)
            # Random similarity score biased toward threshold
            sim = random.betavariate(5, 3) * 0.4 + 0.5
            sim = min(1.0, max(0.0, sim))

            if sim >= 0.90:
                sev = "critical"
            elif sim >= 0.75:
                sev = "high"
            elif sim >= 0.59:
                sev = "medium"
            else:
                sev = "low"

            inc_id = hashlib.md5(f"{doc}{match}{day_offset}{_}".encode()).hexdigest()[:12]
            incidents.append(PlagiarismIncident(
                incident_id=inc_id,
                document_name=doc,
                matched_against=match,
                similarity_score=round(sim, 4),
                severity=sev,
                detected_at=date,
                chunk_count=random.randint(1, 8),
                max_chunk_similarity=round(min(1.0, sim + random.uniform(0, 0.1)), 4),
            ))

    engine.add_incidents(incidents)


def render_trend_analytics_tab(
    incidents: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Render the complete Trend Analytics dashboard tab.

    Call this from the main Streamlit app inside a tab context.

    Args:
        incidents: Optional list of incident dicts to load. If None,
                   uses session state data or seeds demo data.
    """
    engine = initialize_trend_analytics()

    # Load incidents if provided and not yet loaded
    if incidents and not st.session_state.get(KEY_ANALYTICS_LOADED, False):
        count = engine.load_from_dicts(incidents)
        st.session_state[KEY_ANALYTICS_LOADED] = True
        if count > 0:
            st.success(f"Loaded {count} incidents for trend analysis.")

    # Seed demo data if engine is empty
    if engine.incident_count == 0:
        _seed_demo_data(engine)
        st.info("Demo data loaded for trend analytics preview.")

    # ── Sidebar Controls ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Trend Analytics Settings")

        window_options = {
            "Daily": TimeWindow.DAILY,
            "Weekly": TimeWindow.WEEKLY,
            "Monthly": TimeWindow.MONTHLY,
            "Quarterly": TimeWindow.QUARTERLY,
            "Yearly": TimeWindow.YEARLY,
        }
        selected_window_name = st.selectbox(
            "Time Window",
            options=list(window_options.keys()),
            index=2,  # Monthly default
            key="trend_window_select",
        )
        selected_window = window_options[selected_window_name]

        forecast_periods = st.slider(
            "Forecast Periods",
            min_value=1,
            max_value=12,
            value=3,
            key="trend_forecast_periods",
        )

        show_demo_button = st.button(
            "🔄 Regenerate Demo Data",
            key="trend_regenerate_demo",
            help="Replace current data with fresh demo incidents",
        )
        if show_demo_button:
            engine = PlagiarismTrendAnalytics(default_window=selected_window)
            _seed_demo_data(engine)
            st.session_state[KEY_ANALYTICS_ENGINE] = engine
            st.session_state.pop(KEY_ANALYTICS_REPORT, None)
            st.rerun()

    # ── Main Content ───────────────────────────────────────────────────
    st.title("📈 Plagiarism Trend Analytics")
    st.caption(
        f"Tracking **{engine.incident_count}** incidents across "
        f"**{selected_window_name.lower()}** windows"
    )

    # Generate report
    with st.spinner("Computing analytics..."):
        report = engine.generate_report(
            window=selected_window,
            forecast_periods=forecast_periods,
        )
    st.session_state[KEY_ANALYTICS_REPORT] = report

    # ── KPI Metric Cards ───────────────────────────────────────────────
    metrics = render_analytics_summary_metrics(report)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Incidents",
            value=metrics["total_incidents"],
            help="Total plagiarism incidents in the analysis period",
        )
    with col2:
        st.metric(
            "Avg Similarity",
            value=f"{metrics['avg_similarity']:.3f}",
            help="Mean similarity score across all incidents",
        )
    with col3:
        st.metric(
            "High+Critical Rate",
            value=f"{metrics['high_rate']}%",
            help="Percentage of incidents with High or Critical severity",
        )
    with col4:
        st.metric(
            "Repeat Offense Rate",
            value=f"{metrics['repeat_offense_rate']}%",
            help="Fraction of documents with 2+ incidents",
        )

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric(
            "Monthly Growth",
            value=f"{metrics['monthly_growth']}%",
            delta=f"{metrics['monthly_growth']:+.1f}%",
            delta_color="inverse",
            help="Average month-over-month change in incidents",
        )
    with col6:
        trend_icon = {
            "increasing": "📈",
            "decreasing": "📉",
            "stable": "➡️",
            "insufficient_data": "❓",
        }.get(metrics["trend_direction"], "❓")
        st.metric(
            "Trend",
            value=f"{trend_icon} {metrics['trend_direction'].title()}",
            help=f"Confidence: {metrics['trend_confidence']:.1f}%",
        )
    with col7:
        st.metric(
            "Forecast (Next)",
            value=f"{metrics['forecast_next']:.1f}",
            help="Predicted incidents in the next time window",
        )
    with col8:
        st.metric(
            "Analysis Period",
            value=f"{metrics['date_range_start']} → {metrics['date_range_end']}",
            help="Date range of the analysis",
        )

    st.divider()

    # ── Incident Timeline ──────────────────────────────────────────────
    st.subheader("📅 Incident Timeline")

    fig_timeline = create_incident_timeline(
        windows=report.windows,
        forecast_values=report.trend.forecast_values,
        forecast_timestamps=report.trend.forecast_timestamps,
    )
    if fig_timeline:
        st.plotly_chart(fig_timeline, use_container_width=True, key="fig_timeline")
    else:
        st.info("Timeline chart unavailable (plotly not installed).")

    # ── Similarity Trend ───────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📉 Similarity Score Trend")
        fig_sim = create_similarity_trend_chart(windows=report.windows)
        if fig_sim:
            st.plotly_chart(fig_sim, use_container_width=True, key="fig_sim")
        else:
            st.info("Similarity trend chart unavailable.")

    with col_right:
        st.subheader("🎯 Severity Distribution")
        fig_sev = create_severity_donut(dist=report.severity_distribution)
        if fig_sev:
            st.plotly_chart(fig_sev, use_container_width=True, key="fig_sev_donut")
        else:
            st.info("Severity chart unavailable.")

    st.divider()

    # ── Trend Confidence & Severity Timeline ───────────────────────────
    col_trend, col_sev_timeline = st.columns([1, 2])

    with col_trend:
        st.subheader("🎯 Trend Confidence")
        fig_card = create_trend_summary_card(trend=report.trend)
        if fig_card:
            st.plotly_chart(fig_card, use_container_width=True, key="fig_trend_card")

        # Trend details table
        with st.expander("Trend Details", expanded=False):
            st.json({
                "direction": report.trend.direction.value,
                "slope": report.trend.slope,
                "r_squared": report.trend.r_squared,
                "p_value": report.trend.p_value,
                "confidence": f"{report.trend.confidence:.2f}%",
                "forecast": report.trend.forecast_values,
            })

    with col_sev_timeline:
        st.subheader("📊 Severity Over Time")
        fig_sev_tl = create_severity_timeline(windows=report.windows)
        if fig_sev_tl:
            st.plotly_chart(fig_sev_tl, use_container_width=True, key="fig_sev_tl")
        else:
            st.info("Severity timeline unavailable.")

    st.divider()

    # ── Statistical Summary ────────────────────────────────────────────
    st.subheader("📋 Statistical Summary")

    stats = report.statistical_summary
    stat_cols = st.columns(5)
    with stat_cols[0]:
        st.metric("Count", stats.count)
    with stat_cols[1]:
        st.metric("Mean", f"{stats.mean:.4f}")
    with stat_cols[2]:
        st.metric("Median", f"{stats.median:.4f}")
    with stat_cols[3]:
        st.metric("Std Dev", f"{stats.std_dev:.4f}")
    with stat_cols[4]:
        st.metric("IQR", f"{stats.iqr:.4f}")

    with st.expander("Detailed Percentiles", expanded=False):
        pct_cols = st.columns(4)
        with pct_cols[0]:
            st.metric("Min", f"{stats.min_value:.4f}")
        with pct_cols[1]:
            st.metric("P25", f"{stats.percentile_25:.4f}")
        with pct_cols[2]:
            st.metric("P75", f"{stats.percentile_75:.4f}")
        with pct_cols[3]:
            st.metric("P90", f"{stats.percentile_90:.4f}")

    st.divider()

    # ── Top Offenders ──────────────────────────────────────────────────
    st.subheader("🚨 Top Plagiarism Offenders")

    if report.top_offenders:
        fig_offenders = create_offender_bar_chart(offenders=report.top_offenders)
        if fig_offenders:
            st.plotly_chart(fig_offenders, use_container_width=True, key="fig_offenders")

        # Detailed offender table
        with st.expander("Offender Details", expanded=True):
            for i, off in enumerate(report.top_offenders[:10], 1):
                sev_badges = " ".join(
                    f"`{s}`" for s in off.severity_history[:5]
                )
                st.markdown(
                    f"**{i}. {off.document_name}** — "
                    f"{off.incident_count} incidents, "
                    f"avg sim: {off.avg_similarity:.3f}, "
                    f"max sim: {off.max_similarity:.3f}  \n"
                    f"    First: {off.first_detected.strftime('%Y-%m-%d')}, "
                    f"Last: {off.last_detected.strftime('%Y-%m-%d')}  \n"
                    f"    Matched against: {', '.join(off.unique_matchees[:3])}  \n"
                    f"    Severity history: {sev_badges}"
                )
                if i < len(report.top_offenders[:10]):
                    st.markdown("---")
    else:
        st.info("No offender data available.")

    st.divider()

    # ── Window Comparison Heatmap ───────────────────────────────────────
    st.subheader("🗺️ Window Comparison")
    fig_heatmap = create_window_comparison_heatmap(windows=report.windows)
    if fig_heatmap:
        st.plotly_chart(fig_heatmap, use_container_width=True, key="fig_heatmap")

    # ── Export Section ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📥 Export Report")

    export_cols = st.columns(3)

    with export_cols[0]:
        # JSON export
        json_data = engine.export_json(report)
        st.download_button(
            label="📄 Download JSON Report",
            data=json_data,
            file_name="plagiarism_trend_report.json",
            mime="application/json",
            key="trend_export_json",
        )

    with export_cols[1]:
        # CSV export
        csv_data = engine.export_csv(report)
        st.download_button(
            label="📊 Download CSV Report",
            data=csv_data,
            file_name="plagiarism_trend_report.csv",
            mime="text/csv",
            key="trend_export_csv",
        )

    with export_cols[2]:
        # Summary JSON for API consumption
        summary_data = json.dumps({
            "total_incidents": report.total_incidents,
            "avg_similarity": report.statistical_summary.mean,
            "trend_direction": report.trend.direction.value,
            "high_rate": report.severity_distribution.high_rate,
            "repeat_offense_rate": report.repeat_offense_rate,
            "monthly_growth_rate": report.monthly_growth_rate,
            "forecast": report.trend.forecast_values,
        }, indent=2)
        st.download_button(
            label="⚡ Download Summary",
            data=summary_data,
            file_name="trend_summary.json",
            mime="application/json",
            key="trend_export_summary",
        )


def render_trend_analytics_sidebar_widget(
    engine: Optional[PlagiarismTrendAnalytics] = None,
) -> None:
    """Render a compact sidebar widget showing trend snapshot.

    Designed to be embedded in the main app sidebar for quick access.
    """
    if engine is None:
        engine = initialize_trend_analytics()

    if engine.incident_count == 0:
        st.sidebar.info("No trend data available yet.")
        return

    report = engine.generate_report()

    with st.sidebar.expander("📈 Trend Snapshot", expanded=False):
        trend_emoji = {
            "increasing": "📈",
            "decreasing": "📉",
            "stable": "➡️",
            "insufficient_data": "❓",
        }
        direction = report.trend.direction.value
        emoji = trend_emoji.get(direction, "❓")

        st.markdown(f"**Trend:** {emoji} {direction.title()}")
        st.markdown(f"**Incidents:** {report.total_incidents}")
        st.markdown(f"**Avg Score:** {report.statistical_summary.mean:.3f}")
        st.markdown(f"**High Rate:** {report.severity_distribution.high_rate:.1f}%")

        if report.trend.forecast_values:
            next_val = report.trend.forecast_values[0]
            st.markdown(f"**Next Period:** ~{next_val:.1f} incidents")
