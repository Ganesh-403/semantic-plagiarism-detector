"""
Plagiarism Report Generator UI Component.

Streamlit-based interface for generating and exporting
comprehensive plagiarism detection reports.
"""

from typing import Any, Dict

import plotly.express as px
import streamlit as st

from src.core.report_generator import (
    ReportConfig,
    ReportFormat,
    ReportGenerator,
    ReportType,
)


def render_report_generator_dashboard():
    """Render the report generator dashboard."""
    st.title("📄 Plagiarism Report Generator")
    st.markdown(
        "Generate comprehensive reports with **visualizations**, **statistics**, and **recommendations**."
    )

    tab_config, tab_generate, tab_preview = st.tabs(
        ["⚙️ Configuration", "📝 Generate Report", "👁️ Preview Report"]
    )

    with tab_config:
        _render_configuration()

    with tab_generate:
        _render_generation()

    with tab_preview:
        _render_preview()


def _render_configuration():
    """Render report configuration."""
    st.subheader("Report Configuration")

    with st.form("report_config"):
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input(
                "Organization Name", value="Plagiarism Detection System"
            )
            include_viz = st.checkbox("Include Visualizations", value=True)
            include_raw = st.checkbox("Include Raw Data", value=False)
            include_recs = st.checkbox("Include Recommendations", value=True)
        with col2:
            max_matches = st.number_input("Max Matches Displayed", 10, 200, 50)
            severity_threshold = st.slider("Severity Threshold", 0.3, 0.9, 0.5, 0.05)
            custom_footer = st.text_input("Custom Footer Text", value="")

        if st.form_submit_button("💾 Save Configuration", use_container_width=True):
            st.session_state.report_config = ReportConfig(
                company_name=company_name,
                include_visualizations=include_viz,
                include_raw_data=include_raw,
                include_recommendations=include_recs,
                max_matches_displayed=max_matches,
                severity_threshold=severity_threshold,
                custom_footer=custom_footer,
            )
            st.success("✅ Configuration saved!")


def _render_generation():
    """Render report generation interface."""
    st.subheader("Generate Report")

    report_type = st.selectbox(
        "Report Type", ["detailed", "summary", "executive", "comparison", "batch"]
    )
    report_format = st.selectbox("Export Format", ["markdown", "json", "html", "text"])

    custom_title = st.text_input(
        "Custom Report Title",
        value="",
        placeholder="Leave empty for auto-generated title",
    )

    uploaded_files = st.file_uploader(
        "Upload detection results (JSON) or paste results",
        type=["json"],
        help="Upload JSON output from plagiarism detection",
    )

    if uploaded_files:
        import json

        try:
            results = json.load(uploaded_files)
            st.session_state.detection_results = results
            st.success(
                f"✅ Loaded results: {len(results.get('matches', []))} matches found"
            )
        except Exception as e:
            st.error(f"Error loading file: {e}")

    if "detection_results" not in st.session_state:
        st.info("👆 Upload detection results to generate a report.")
        if st.button("📝 Use Sample Data for Demo", type="secondary"):
            st.session_state.detection_results = _get_sample_results()
            st.success("Sample data loaded!")
            st.rerun()

    if st.button("🚀 Generate Report", type="primary", use_container_width=True):
        if "detection_results" in st.session_state:
            config = st.session_state.get("report_config", ReportConfig())
            generator = ReportGenerator(config)
            report = generator.generate_report(
                st.session_state.detection_results,
                ReportType(report_type),
                title=custom_title or None,
            )
            st.session_state.current_report = report
            st.session_state.report_format = report_format
            st.success(f"✅ Report generated! ID: {report.report_id}")
        else:
            st.warning("Please upload detection results first.")


def _render_preview():
    """Render report preview."""
    st.subheader("Report Preview")

    report = st.session_state.get("current_report")
    if not report:
        st.info("Generate a report first to see the preview.")
        return

    st.markdown(f"### {report.title}")
    st.caption(f"Report ID: `{report.report_id}` | Generated: {report.generated_at}")

    # Summary metrics
    summary = report.summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Documents", summary.get("total_documents", 0))
    col2.metric("Matches", summary.get("total_matches", 0))
    col3.metric("Avg Similarity", f"{summary.get('average_similarity', 0):.1%}")
    col4.metric("Plagiarism Rate", f"{summary.get('plagiarism_rate', 0):.1f}%")

    # Severity distribution chart
    severity = summary.get("severity_distribution", {})
    if any(v > 0 for v in severity.values()):
        fig = px.pie(
            values=list(severity.values()),
            names=[f"{k.title()} ({v})" for k, v in severity.items()],
            title="Severity Distribution",
            color_discrete_map={
                "Critical (0)": "#ef4444",
                "High (0)": "#f97316",
                "Moderate (0)": "#eab308",
                "Low (0)": "#84cc16",
                "Clean (0)": "#22c55e",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    # Sections
    for section in report.sections:
        with st.expander(f"📋 {section.title}", expanded=(section.title == "Overview")):
            st.markdown(section.content)

    # Recommendations
    if report.recommendations:
        st.subheader("💡 Recommendations")
        for rec in report.recommendations:
            st.markdown(f"- {rec}")

    # Export buttons
    st.subheader("⬇️ Export Report")
    fmt = st.session_state.get("report_format", "markdown")
    generator = ReportGenerator()
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
        temp_path = f.name

    try:
        generator.export_report(report, ReportFormat(fmt), temp_path)
        with open(temp_path, "r") as f:
            content = f.read()
        mime = {
            "json": "application/json",
            "markdown": "text/markdown",
            "html": "text/html",
            "text": "text/plain",
        }.get(fmt, "text/plain")
        st.download_button(
            f"⬇️ Download {fmt.upper()}", content, f"plagiarism_report.{fmt}", mime
        )
    finally:
        os.unlink(temp_path)


def _get_sample_results() -> dict[str, Any]:
    """Get sample detection results for demo."""
    return {
        "total_documents": 6,
        "matches": [
            {
                "doc_a": "assignment_01.pdf",
                "doc_b": "assignment_03.pdf",
                "similarity": 0.92,
                "severity": "critical",
            },
            {
                "doc_a": "assignment_01.pdf",
                "doc_b": "assignment_05.pdf",
                "similarity": 0.78,
                "severity": "high",
            },
            {
                "doc_a": "assignment_02.pdf",
                "doc_b": "assignment_04.pdf",
                "similarity": 0.65,
                "severity": "moderate",
            },
            {
                "doc_a": "assignment_03.pdf",
                "doc_b": "assignment_06.pdf",
                "similarity": 0.55,
                "severity": "moderate",
            },
            {
                "doc_a": "assignment_02.pdf",
                "doc_b": "assignment_06.pdf",
                "similarity": 0.42,
                "severity": "low",
            },
        ],
        "flagged": [
            {
                "doc_a": "assignment_01.pdf",
                "doc_b": "assignment_03.pdf",
                "similarity": 0.92,
            },
            {
                "doc_a": "assignment_01.pdf",
                "doc_b": "assignment_05.pdf",
                "similarity": 0.78,
            },
        ],
    }
