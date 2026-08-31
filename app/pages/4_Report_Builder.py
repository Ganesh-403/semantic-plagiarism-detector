"""
app/pages/4_Report_Builder.py
-----------------------------
Streamlit multi-page app: Plagiarism Report Builder.

Generate, customize, preview, and export comprehensive plagiarism analysis
reports with charts, severity breakdowns, source comparisons, and multi-format
export (HTML, Markdown, CSV, JSON).
"""

import json
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Report Builder - Plagiarism Detector",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Mock data generators (used when no real DB data is available)
# ---------------------------------------------------------------------------

def _generate_mock_incidents(n: int = 25) -> list[dict[str, Any]]:
    """Generate mock plagiarism incident data for the report builder."""
    import random
    random.seed(42)
    doc_names = [
        "Thesis_Chapter3.docx", "Research_Paper_AI.docx", "Literature_Review.pdf",
        "Capstone_Project.docx", "Seminar_Report.pdf", "Lab_Report_Physics.docx",
        "Dissertation_Draft.docx", "Conference_Paper.docx", "Assignment_Week5.docx",
        "Midterm_Essay.docx", "Group_Project_Report.pdf", "Technical_Writeup.docx",
        "Case_Study_Analysis.docx", "Term_Paper_History.docx", "Methodology_Section.docx",
        "Abstract_Collection.pdf", "Appendix_Draft.docx", "Review_Article.docx",
        "Survey_Results.docx", "Final_Proposal.docx", "Preprint_v2.pdf",
        "Book_Chapter_Draft.docx", "Workshop_Paper.docx", "Thesis_Intro.docx",
        "Dataset_Analysis.docx",
    ]
    authors = [
        "Alice Johnson", "Bob Smith", "Carol White", "David Brown", "Eva Martinez",
        "Frank Lee", "Grace Kim", "Henry Wilson", "Iris Chen", "Jack Davis",
        "Karen Patel", "Leo Nguyen", "Mia Thompson", "Noah Garcia", "Olivia Davis",
    ]
    severity_levels = ["Critical", "High", "Medium", "Low", "Info"]
    severity_weights = [0.08, 0.18, 0.35, 0.28, 0.11]
    sources = [
        "Wikipedia", "arXiv.org", "IEEE Xplore", "SpringerLink", "PubMed",
        "JSTOR", "Google Scholar", "ResearchGate", "ScienceDirect", "Semantic Scholar",
        "DOAJ", "ACM Digital Library", "Elsevier", "Taylor & Francis", "MDPI",
    ]
    departments = [
        "Computer Science", "Electrical Engineering", "Mechanical Engineering",
        "Physics", "Mathematics", "Biology", "Chemistry", "Data Science",
    ]
    categories = [
        "Verbatim Copy", "Paraphrase", "Mosaic", "Self-Plagiarism",
        "Idea Theft", "Structure Copy", "Translation", "AI-Generated",
    ]

    incidents = []
    for i in range(n):
        sev = random.choices(severity_levels, weights=severity_weights, k=1)[0]
        sim = (
            random.uniform(85, 99) if sev == "Critical"
            else random.uniform(70, 94) if sev == "High"
            else random.uniform(50, 74) if sev == "Medium"
            else random.uniform(30, 54) if sev == "Low"
            else random.uniform(10, 29)
        )
        days_ago = random.randint(0, 90)
        created = datetime.now() - timedelta(days=days_ago)
        incidents.append({
            "id": f"INC-{1000 + i}",
            "document": random.choice(doc_names),
            "author": random.choice(authors),
            "department": random.choice(departments),
            "similarity_score": round(sim, 1),
            "severity": sev,
            "category": random.choice(categories),
            "source": random.choice(sources),
            "source_url": f"https://example.com/paper/{random.randint(1000, 9999)}",
            "matched_segments": random.randint(1, 15),
            "total_words": random.randint(800, 12000),
            "flagged_words": random.randint(50, 5000),
            "created_at": created.strftime("%Y-%m-%d %H:%M"),
            "status": random.choice(["Pending Review", "Under Investigation", "Confirmed", "Dismissed", "Escalated"]),
            "reviewer": random.choice(["Dr. Smith", "Prof. Johnson", "Admin", None]),
            "notes": random.choice([
                "Needs immediate attention",
                "Similar to previous case",
                "Possible self-plagiarism",
                "Source attribution missing",
                "Academic integrity violation suspected",
                "",
            ]),
        })
    return incidents


def _generate_mock_comparison_data() -> list[dict[str, Any]]:
    """Generate mock document comparison data."""
    import random
    random.seed(99)
    pairs = []
    doc_names = [
        "Thesis_Chapter3.docx", "Research_Paper_AI.docx", "Literature_Review.pdf",
        "Capstone_Project.docx", "Seminar_Report.pdf", "Lab_Report_Physics.docx",
        "Dissertation_Draft.docx", "Conference_Paper.docx",
    ]
    for i in range(8):
        d1, d2 = random.sample(doc_names, 2)
        pairs.append({
            "doc_a": d1,
            "doc_b": d2,
            "similarity": round(random.uniform(35, 95), 1),
            "common_segments": random.randint(3, 20),
            "common_words": random.randint(200, 3000),
        })
    return pairs


# ---------------------------------------------------------------------------
# Report configuration helpers
# ---------------------------------------------------------------------------

def _default_report_config() -> dict[str, Any]:
    """Return default report configuration."""
    return {
        "title": "Plagiarism Analysis Report",
        "subtitle": "Comprehensive Document Integrity Assessment",
        "author": "Plagiarism Detection System",
        "organization": "Academic Integrity Office",
        "date_range_days": 30,
        "include_executive_summary": True,
        "include_severity_breakdown": True,
        "include_document_details": True,
        "include_source_analysis": True,
        "include_department_analysis": True,
        "include_trend_charts": True,
        "include_recommendations": True,
        "include_raw_data": False,
        "severity_filter": ["Critical", "High", "Medium", "Low", "Info"],
        "department_filter": "All",
        "min_similarity": 0,
        "sort_by": "similarity_score",
        "sort_order": "descending",
    }


def _render_severity_badge(severity: str) -> str:
    """Return a colored badge for severity level."""
    colors = {
        "Critical": "#dc3545",
        "High": "#fd7e14",
        "Medium": "#ffc107",
        "Low": "#28a745",
        "Info": "#17a2b8",
    }
    color = colors.get(severity, "#6c757d")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:0.8em;font-weight:600">{severity}</span>'


def _render_status_badge(status: str) -> str:
    """Return a colored badge for status."""
    colors = {
        "Pending Review": "#ffc107",
        "Under Investigation": "#fd7e14",
        "Confirmed": "#dc3545",
        "Dismissed": "#28a745",
        "Escalated": "#6f42c1",
    }
    color = colors.get(status, "#6c757d")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:0.8em">{status}</span>'


# ---------------------------------------------------------------------------
# Chart generators (pure Streamlit / markdown — no matplotlib dependency)
# ---------------------------------------------------------------------------

def _render_bar_chart(data: dict[str, int], title: str, color: str = "#4a90d9", max_val: int | None = None):
    """Render a horizontal bar chart using Streamlit markdown."""
    if not data:
        st.info("No data available for this chart.")
        return
    mx = max_val or max(data.values()) or 1
    st.markdown(f"**{title}**")
    for label, value in data.items():
        pct = (value / mx) * 100
        st.markdown(
            f'<div style="margin-bottom:4px">'
            f'<span style="display:inline-block;width:140px;font-size:0.85em">{label}</span>'
            f'<span style="display:inline-block;width:60%;background:#e0e0e0;border-radius:4px;height:18px;position:relative">'
            f'<span style="display:inline-block;width:{pct:.0f}%;background:{color};border-radius:4px;height:100%"></span>'
            f'</span>'
            f'<span style="margin-left:8px;font-size:0.85em;font-weight:600">{value}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_donut_chart(data: dict[str, int], title: str, colors: list[str] | None = None):
    """Render a simple donut-style breakdown as a horizontal stacked bar."""
    if not data:
        st.info("No data available.")
        return
    total = sum(data.values()) or 1
    default_colors = ["#dc3545", "#fd7e14", "#ffc107", "#28a745", "#17a2b8", "#6f42c1", "#20c997"]
    c = colors or default_colors
    st.markdown(f"**{title}**")
    segments = ""
    legend = ""
    for i, (label, value) in enumerate(data.items()):
        pct = (value / total) * 100
        color = c[i % len(c)]
        segments += f'<span style="display:inline-block;width:{pct:.1f}%;background:{color};height:24px"></span>'
        legend += f'<span style="margin-right:14px"><span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:50%;margin-right:4px"></span>{label} ({value})</span>'
    st.markdown(
        f'<div style="background:#e0e0e0;border-radius:6px;overflow:hidden;margin-bottom:6px">{segments}</div>'
        f'<div style="font-size:0.8em;color:#666">{legend}</div>',
        unsafe_allow_html=True,
    )


def _render_trend_chart(values: list[int], labels: list[str], title: str, color: str = "#4a90d9"):
    """Render a simple trend sparkline as an SVG line chart."""
    if not values:
        st.info("No trend data.")
        return
    mx = max(values) or 1
    width, height = 500, 100
    points = []
    for i, v in enumerate(values):
        x = (i / max(len(values) - 1, 1)) * width
        y = height - (v / mx) * (height - 20)
        points.append(f"{x:.1f},{y:.1f}")
    path = "M " + " L ".join(points)
    labels_str = ", ".join(labels)
    values_str = ", ".join(str(v) for v in values)
    svg = f'''<svg width="{width}" height="{height + 30}" viewBox="0 0 {width} {height + 30}" xmlns="http://www.w3.org/2000/svg">
  <polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="10" y="{height + 20}" fill="#666" font-size="10">{labels_str}</text>
</svg>'''
    st.markdown(f"**{title}**")
    st.markdown(svg, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Report section renderers
# ---------------------------------------------------------------------------

def _render_executive_summary(incidents: list[dict], config: dict):
    """Render the executive summary section."""
    st.subheader("📋 Executive Summary")
    total = len(incidents)
    critical = sum(1 for i in incidents if i["severity"] == "Critical")
    high = sum(1 for i in incidents if i["severity"] == "High")
    confirmed = sum(1 for i in incidents if i["status"] == "Confirmed")
    avg_sim = sum(i["similarity_score"] for i in incidents) / total if total else 0
    total_flagged = sum(i["flagged_words"] for i in incidents)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Incidents", f"{total:,}")
    col2.metric("Critical", f"{critical}", delta=None, delta_color="inverse")
    col3.metric("High Severity", f"{high}", delta=None, delta_color="inverse")
    col4.metric("Confirmed", f"{confirmed}")
    col5.metric("Avg Similarity", f"{avg_sim:.1f}%")

    st.markdown("---")
    st.markdown(
        f"This report covers **{total} plagiarism incidents** identified across "
        f"the monitored corpus. Of these, **{critical} are classified as Critical** "
        f"and **{high} as High severity**, requiring immediate attention. "
        f"**{confirmed} incidents have been confirmed** as violations. "
        f"A total of **{total_flagged:,} words** were flagged across all incidents "
        f"with an average similarity score of **{avg_sim:.1f}%**."
    )

    if critical > 0:
        st.warning(
            f"⚠️ **{critical} critical incidents** require immediate review. "
            f"These cases involve similarity scores above 85% and may represent "
            f"direct verbatim copying."
        )


def _render_severity_breakdown(incidents: list[dict]):
    """Render severity breakdown charts."""
    st.subheader("🔴 Severity Breakdown")
    sev_counts = {}
    for i in incidents:
        sev_counts[i["severity"]] = sev_counts.get(i["severity"], 0) + 1

    col1, col2 = st.columns(2)
    with col1:
        _render_donut_chart(
            sev_counts, "Severity Distribution",
            colors=["#dc3545", "#fd7e14", "#ffc107", "#28a745", "#17a2b8"],
        )
    with col2:
        _render_bar_chart(sev_counts, "Incidents by Severity", color="#4a90d9")

    st.markdown("**Category Breakdown:**")
    cat_counts = {}
    for i in incidents:
        cat_counts[i["category"]] = cat_counts.get(i["category"], 0) + 1
    _render_bar_chart(cat_counts, "Plagiarism Categories", color="#6f42c1")


def _render_document_details(incidents: list[dict], config: dict):
    """Render detailed document-level analysis."""
    st.subheader("📄 Document Details")

    sort_by = config.get("sort_by", "similarity_score")
    sort_desc = config.get("sort_order") == "descending"
    sorted_incidents = sorted(incidents, key=lambda x: x.get(sort_by, 0), reverse=sort_desc)

    for idx, inc in enumerate(sorted_incidents):
        badge = _render_severity_badge(inc["severity"])
        status = _render_status_badge(inc["status"])
        sim = inc["similarity_score"]
        sim_color = (
            "#dc3545" if sim >= 80 else "#fd7e14" if sim >= 60
            else "#ffc107" if sim >= 40 else "#28a745"
        )

        with st.expander(
            f"**{inc['id']}** — {inc['document']} | {badge} | {sim}% similarity | {status}",
            expanded=(sim >= 80),
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Similarity", f"{sim}%", delta=None)
            c2.metric("Matched Segments", inc["matched_segments"])
            c3.metric("Flagged Words", f"{inc['flagged_words']:,}")

            # Similarity bar
            st.markdown(
                f'<div style="background:#e0e0e0;border-radius:4px;height:12px;margin:8px 0">'
                f'<div style="width:{sim}%;background:{sim_color};border-radius:4px;height:100%"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"**Author:** {inc['author']} | **Department:** {inc['department']} | "
                f"**Category:** {inc['category']} | **Source:** {inc['source']}"
            )
            st.markdown(
                f"**Detected:** {inc['created_at']} | **Status:** {inc['status']} | "
                f"**Reviewer:** {inc['reviewer'] or 'Unassigned'}"
            )
            if inc["notes"]:
                st.info(f"📝 {inc['notes']}")
            if inc["source_url"]:
                st.markdown(f"🔗 [Source Reference]({inc['source_url']})")


def _render_source_analysis(incidents: list[dict]):
    """Render source analysis and attribution patterns."""
    st.subheader("🌐 Source Analysis")

    source_counts = {}
    source_sims = {}
    for i in incidents:
        src = i["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
        source_sims.setdefault(src, []).append(i["similarity_score"])

    # Top sources bar chart
    sorted_sources = dict(sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    _render_bar_chart(sorted_sources, "Top 10 Sources by Incident Count", color="#17a2b8")

    # Average similarity by source
    st.markdown("**Average Similarity by Source:**")
    src_avg = {s: round(sum(v) / len(v), 1) for s, v in source_sims.items()}
    sorted_avg = dict(sorted(src_avg.items(), key=lambda x: x[1], reverse=True)[:10])
    _render_bar_chart(
        {k: v for k, v in sorted_avg.items()},
        "Average Similarity Score (%)",
        color="#fd7e14",
    )

    # Source table
    st.markdown("**Source Summary Table:**")
    src_data = []
    for src in source_counts:
        avg_sim = round(sum(source_sims[src]) / len(source_sims[src]), 1)
        max_sim = max(source_sims[src])
        src_data.append({
            "Source": src,
            "Incidents": source_counts[src],
            "Avg Similarity": f"{avg_sim}%",
            "Max Similarity": f"{max_sim}%",
        })
    src_df = pd.DataFrame(src_data).sort_values("Incidents", ascending=False)
    st.dataframe(src_df, use_container_width=True, hide_index=True)


def _render_department_analysis(incidents: list[dict]):
    """Render department-level plagiarism analysis."""
    st.subheader("🏫 Department Analysis")

    dept_counts = {}
    dept_sims = {}
    dept_severity = {}
    for i in incidents:
        dept = i["department"]
        dept_counts[dept] = dept_counts.get(dept, 0) + 1
        dept_sims.setdefault(dept, []).append(i["similarity_score"])
        dept_severity.setdefault(dept, {})
        dept_severity[dept][i["severity"]] = dept_severity[dept].get(i["severity"], 0) + 1

    # Department incident counts
    sorted_depts = dict(sorted(dept_counts.items(), key=lambda x: x[1], reverse=True))
    _render_bar_chart(sorted_depts, "Incidents by Department", color="#6f42c1")

    # Average similarity by department
    dept_avg = {d: round(sum(v) / len(v), 1) for d, v in dept_sims.items()}
    sorted_avg = dict(sorted(dept_avg.items(), key=lambda x: x[1], reverse=True))
    _render_bar_chart({k: v for k, v in sorted_avg.items()}, "Avg Similarity by Department (%)", color="#e83e8c")

    # Department cards
    st.markdown("**Department Breakdown:**")
    for dept, count in sorted_depts.items():
        avg = dept_avg[dept]
        severity = dept_severity[dept]
        critical_count = severity.get("Critical", 0)
        high_count = severity.get("High", 0)
        sev_str = ", ".join(f"{k}: {v}" for k, v in sorted(severity.items()))
        color = "#dc3545" if critical_count > 0 else "#fd7e14" if high_count > 0 else "#28a745"
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:8px 12px;margin:6px 0;background:#f8f9fa;border-radius:4px">'
            f'<strong>{dept}</strong> — {count} incidents, {avg}% avg similarity<br/>'
            f'<span style="font-size:0.85em;color:#666">{sev_str}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_trend_analysis(incidents: list[dict]):
    """Render trend analysis over time."""
    st.subheader("📈 Trend Analysis")

    # Group by date
    date_counts = {}
    for i in incidents:
        day = i["created_at"][:10]
        date_counts[day] = date_counts.get(day, 0) + 1

    if date_counts:
        sorted_dates = sorted(date_counts.keys())
        labels = [d[5:] for d in sorted_dates]  # MM-DD format
        values = [date_counts[d] for d in sorted_dates]
        _render_trend_chart(values, labels, "Daily Incident Trend", color="#4a90d9")

    # Severity trend by week
    st.markdown("**Weekly Severity Distribution:**")
    week_data: dict[str, dict[str, int]] = {}
    for i in incidents:
        day = datetime.strptime(i["created_at"], "%Y-%m-%d %H:%M")
        week_key = day.strftime("%Y-W%U")
        week_data.setdefault(week_key, {})
        week_data[week_key][i["severity"]] = week_data[week_key].get(i["severity"], 0) + 1

    if week_data:
        sorted_weeks = sorted(week_data.keys())
        for sev in ["Critical", "High", "Medium", "Low"]:
            vals = [week_data[w].get(sev, 0) for w in sorted_weeks]
            wk_labels = [w[5:] for w in sorted_weeks]
            if any(v > 0 for v in vals):
                _render_trend_chart(vals, wk_labels, f"{sev} Severity Trend", color={
                    "Critical": "#dc3545", "High": "#fd7e14",
                    "Medium": "#ffc107", "Low": "#28a745",
                }.get(sev, "#6c757d"))


def _render_recommendations(incidents: list[dict]):
    """Render AI-style recommendations."""
    st.subheader("💡 Recommendations & Action Items")

    critical = [i for i in incidents if i["severity"] == "Critical"]
    high = [i for i in incidents if i["severity"] == "High"]
    confirmed = [i for i in incidents if i["status"] == "Confirmed"]
    pending = [i for i in incidents if i["status"] == "Pending Review"]

    # Urgent actions
    if critical:
        st.error(
            f"**🚨 URGENT:** {len(critical)} critical incidents require immediate investigation. "
            f"Documents: {', '.join(i['document'] for i in critical[:3])}"
        )

    if high:
        st.warning(
            f"**⚠️ HIGH PRIORITY:** {len(high)} high-severity incidents should be reviewed within 48 hours."
        )

    if pending:
        st.info(f"**📋 PENDING:** {len(pending)} incidents are awaiting initial review.")

    # Action items
    st.markdown("**Recommended Actions:**")
    actions = []
    if critical:
        actions.append({
            "priority": "🔴 Critical",
            "action": f"Conduct immediate investigation of {len(critical)} critical plagiarism cases",
            "deadline": "Within 24 hours",
            "responsible": "Academic Integrity Committee",
        })
    actions.append({
        "priority": "🟠 High",
        "action": f"Schedule review meetings for {len(high)} high-severity incidents",
        "deadline": "Within 48 hours",
        "responsible": "Department Heads",
    })
    if confirmed:
        actions.append({
            "priority": "🔴 Confirmed",
            "action": f"Initiate academic integrity proceedings for {len(confirmed)} confirmed violations",
            "deadline": "Within 1 week",
            "responsible": "Dean of Students",
        })

    # Department-specific recommendations
    dept_sims: dict[str, list[float]] = {}
    for i in incidents:
        dept_sims.setdefault(i["department"], []).append(i["similarity_score"])
    for dept, sims in dept_sims.items():
        avg = sum(sims) / len(sims)
        if avg > 60:
            actions.append({
                "priority": "🟡 Department",
                "action": f"Conduct academic integrity workshop for {dept} (avg similarity: {avg:.0f}%)",
                "deadline": "Within 2 weeks",
                "responsible": f"{dept} Chair",
            })

    actions.append({
        "priority": "🟢 Preventive",
        "action": "Implement mandatory plagiarism awareness training for all students",
        "deadline": "Next semester",
        "responsible": "Academic Affairs",
    })

    for act in actions:
        st.markdown(
            f'<div style="border:1px solid #ddd;border-radius:6px;padding:10px;margin:8px 0">'
            f'<strong>{act["priority"]}</strong><br/>'
            f'{act["action"]}<br/>'
            f'<span style="font-size:0.8em;color:#666">Deadline: {act["deadline"]} | Responsible: {act["responsible"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_comparison_data(comparisons: list[dict]):
    """Render document comparison analysis."""
    st.subheader("🔀 Document Comparison")

    if not comparisons:
        st.info("No comparison data available.")
        return

    st.markdown("**Cross-Document Similarity Matrix:**")
    pairs_data = []
    for c in comparisons:
        pairs_data.append({
            "Document A": c["doc_a"],
            "Document B": c["doc_b"],
            "Similarity": f"{c['similarity']}%",
            "Common Segments": c["common_segments"],
            "Common Words": f"{c['common_words']:,}",
        })
    pairs_df = pd.DataFrame(pairs_data).sort_values("Similarity", ascending=False)
    st.dataframe(pairs_df, use_container_width=True, hide_index=True)

    # Most similar pairs
    st.markdown("**Top Similar Pairs:**")
    for c in sorted(comparisons, key=lambda x: x["similarity"], reverse=True)[:5]:
        sim = c["similarity"]
        color = "#dc3545" if sim >= 80 else "#fd7e14" if sim >= 60 else "#ffc107"
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:6px 10px;margin:4px 0;background:#f8f9fa;border-radius:4px">'
            f'<strong>{c["doc_a"]}</strong> ↔ <strong>{c["doc_b"]}</strong> '
            f'— <span style="color:{color};font-weight:600">{sim}%</span> '
            f'({c["common_segments"]} segments, {c["common_words"]:,} words)'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_raw_data(incidents: list[dict]):
    """Render raw data table."""
    st.subheader("📊 Raw Data Export")

    df = pd.DataFrame(incidents)
    cols_to_show = [
        "id", "document", "author", "department", "similarity_score",
        "severity", "category", "source", "status", "created_at",
    ]
    cols_available = [c for c in cols_to_show if c in df.columns]
    st.dataframe(df[cols_available], use_container_width=True, hide_index=True)

    # Export buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name=f"plagiarism_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    with col2:
        json_data = json.dumps(incidents, indent=2, default=str)
        st.download_button(
            "📥 Download JSON",
            json_data,
            file_name=f"plagiarism_report_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
        )
    with col3:
        md_lines = ["# Plagiarism Report", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
        for inc in incidents:
            md_lines.append(
                f"- **{inc['id']}** | {inc['document']} | {inc['similarity_score']}% | {inc['severity']} | {inc['status']}"
            )
        markdown = "\n".join(md_lines)
        st.download_button(
            "📥 Download Markdown",
            markdown,
            file_name=f"plagiarism_report_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
        )


def _render_html_preview(incidents: list[dict], config: dict):
    """Render an HTML preview of the report."""
    st.subheader("👁️ HTML Report Preview")

    total = len(incidents)
    critical = sum(1 for i in incidents if i["severity"] == "Critical")
    high = sum(1 for i in incidents if i["severity"] == "High")
    avg_sim = sum(i["similarity_score"] for i in incidents) / total if total else 0

    html_parts = [
        f'<div style="font-family:Arial,sans-serif;max-width:900px;margin:0 auto;padding:20px">',
        f'<h1 style="color:#1a1a2e">{config["title"]}</h1>',
        f'<p style="color:#666">{config["subtitle"]} | {config["author"]} | {datetime.now().strftime("%B %d, %Y")}</p>',
        f'<hr/>',
        f'<h2>Executive Summary</h2>',
        f'<table style="width:100%;border-collapse:collapse;margin:16px 0">',
        f'<tr>',
        f'<td style="padding:12px;background:#f0f4ff;border-radius:8px;text-align:center"><strong>{total}</strong><br/>Total Incidents</td>',
        f'<td style="padding:12px;background:#fff0f0;border-radius:8px;text-align:center"><strong style="color:#dc3545">{critical}</strong><br/>Critical</td>',
        f'<td style="padding:12px;background:#fff8f0;border-radius:8px;text-align:center"><strong style="color:#fd7e14">{high}</strong><br/>High</td>',
        f'<td style="padding:12px;background:#f0fff4;border-radius:8px;text-align:center"><strong style="color:#28a745">{avg_sim:.1f}%</strong><br/>Avg Similarity</td>',
        f'</tr></table>',
        f'<h2>Incident Details</h2>',
        f'<table style="width:100%;border-collapse:collapse;font-size:0.9em">',
        f'<tr style="background:#1a1a2e;color:white"><th style="padding:8px">ID</th><th>Document</th><th>Author</th><th>Similarity</th><th>Severity</th><th>Status</th></tr>',
    ]

    for inc in sorted(incidents, key=lambda x: x["similarity_score"], reverse=True)[:20]:
        sev_color = {"Critical": "#dc3545", "High": "#fd7e14", "Medium": "#ffc107", "Low": "#28a745"}.get(inc["severity"], "#6c757d")
        html_parts.append(
            f'<tr style="border-bottom:1px solid #eee">'
            f'<td style="padding:6px">{inc["id"]}</td>'
            f'<td>{inc["document"]}</td>'
            f'<td>{inc["author"]}</td>'
            f'<td><strong>{inc["similarity_score"]}%</strong></td>'
            f'<td><span style="color:{sev_color};font-weight:600">{inc["severity"]}</span></td>'
            f'<td>{inc["status"]}</td></tr>'
        )

    html_parts.append("</table></div>")
    html_content = "\n".join(html_parts)
    st.markdown(html_content, unsafe_allow_html=True)

    # Download HTML
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{config['title']}</title></head>
<body style="margin:40px;background:#fafafa">
<div style="max-width:1000px;margin:0 auto;background:white;padding:30px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1)">
{html_content}
</div>
</body></html>"""
    st.download_button(
        "📥 Download HTML Report",
        full_html,
        file_name=f"plagiarism_report_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html",
    )


# ---------------------------------------------------------------------------
# Main page render
# ---------------------------------------------------------------------------

def render_report_builder():
    """Render the Plagiarism Report Builder page."""
    st.title("📄 Plagiarism Report Builder")
    st.markdown(
        "Generate, customize, preview, and export comprehensive plagiarism analysis reports."
    )

    # --- Sidebar: Report Configuration ---
    with st.sidebar:
        st.header("⚙️ Report Configuration")

        config = _default_report_config()
        config["title"] = st.text_input("Report Title", value=config["title"])
        config["subtitle"] = st.text_input("Subtitle", value=config["subtitle"])
        config["author"] = st.text_input("Author", value=config["author"])
        config["organization"] = st.text_input("Organization", value=config["organization"])

        st.markdown("---")
        st.subheader("📅 Date Range")
        config["date_range_days"] = st.slider("Last N days", 7, 365, config["date_range_days"])

        st.markdown("---")
        st.subheader("🔍 Filters")
        config["severity_filter"] = st.multiselect(
            "Severity Levels",
            ["Critical", "High", "Medium", "Low", "Info"],
            default=config["severity_filter"],
        )
        config["min_similarity"] = st.slider("Minimum Similarity (%)", 0, 100, config["min_similarity"])
        config["sort_by"] = st.selectbox("Sort By", ["similarity_score", "created_at", "severity", "document"])
        config["sort_order"] = st.selectbox("Order", ["descending", "ascending"])

        st.markdown("---")
        st.subheader("📑 Sections to Include")
        config["include_executive_summary"] = st.checkbox("Executive Summary", config["include_executive_summary"])
        config["include_severity_breakdown"] = st.checkbox("Severity Breakdown", config["include_severity_breakdown"])
        config["include_document_details"] = st.checkbox("Document Details", config["include_document_details"])
        config["include_source_analysis"] = st.checkbox("Source Analysis", config["include_source_analysis"])
        config["include_department_analysis"] = st.checkbox("Department Analysis", config["include_department_analysis"])
        config["include_trend_charts"] = st.checkbox("Trend Charts", config["include_trend_charts"])
        config["include_recommendations"] = st.checkbox("Recommendations", config["include_recommendations"])
        config["include_raw_data"] = st.checkbox("Raw Data Table", config["include_raw_data"])

    # --- Load Data ---
    try:
        from src.db.incidents import get_all_incidents
        raw_incidents = get_all_incidents(limit=10000)
        if raw_incidents:
            incidents = raw_incidents
        else:
            incidents = _generate_mock_incidents(25)
            st.info("Using sample data for demonstration.")
    except Exception:
        incidents = _generate_mock_incidents(25)
        st.info("Using sample data for demonstration.")

    comparisons = _generate_mock_comparison_data()

    # Apply filters
    filtered = incidents
    if config["severity_filter"]:
        filtered = [i for i in filtered if i["severity"] in config["severity_filter"]]
    if config["min_similarity"] > 0:
        filtered = [i for i in filtered if i["similarity_score"] >= config["min_similarity"]]

    # --- Report Preview ---
    st.markdown("---")
    st.subheader(f"📋 Report Preview — {len(filtered)} incidents")

    if not filtered:
        st.warning("No incidents match the current filter criteria. Adjust filters in the sidebar.")
        return

    # Render sections
    if config["include_executive_summary"]:
        _render_executive_summary(filtered, config)

    if config["include_severity_breakdown"]:
        st.markdown("---")
        _render_severity_breakdown(filtered)

    if config["include_trend_charts"]:
        st.markdown("---")
        _render_trend_analysis(filtered)

    if config["include_source_analysis"]:
        st.markdown("---")
        _render_source_analysis(filtered)

    if config["include_department_analysis"]:
        st.markdown("---")
        _render_department_analysis(filtered)

    if config["include_document_details"]:
        st.markdown("---")
        _render_document_details(filtered, config)

    st.markdown("---")
    _render_comparison_data(comparisons)

    if config["include_recommendations"]:
        st.markdown("---")
        _render_recommendations(filtered)

    if config["include_raw_data"]:
        st.markdown("---")
        _render_raw_data(filtered)

    # --- HTML Export ---
    st.markdown("---")
    _render_html_preview(filtered, config)

    # --- Report Metadata ---
    st.markdown("---")
    st.caption(
        f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"{len(filtered)} incidents | {len(comparisons)} document comparisons | "
        f"Filters: severity={config['severity_filter']}, min_similarity={config['min_similarity']}%"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__" or True:
    render_report_builder()
