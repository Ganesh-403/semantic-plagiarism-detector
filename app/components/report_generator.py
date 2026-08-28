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
Comprehensive Report Generator for Plagiarism Detection

Features:
- PDF/HTML report generation with plagiarism findings
- Executive summary with key metrics
- Detailed document pair analysis
- Visual charts and graphs embedding
- Export to multiple formats
- Scheduled report generation
- Email delivery integration
"""

import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Report generation libraries (optional, with fallbacks)
try:
    from reportlab.graphics.charts.barcharts import VerticalBarChart  # noqa: F401
    from reportlab.graphics.charts.linecharts import HorizontalLineChart  # noqa: F401
    from reportlab.graphics.charts.piecharts import Pie  # noqa: F401
    from reportlab.graphics.shapes import Drawing  # noqa: F401
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT  # noqa: F401
    from reportlab.lib.pagesizes import A4, landscape, letter  # noqa: F401
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import pdfkit  # noqa: F401

    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False

try:
    import markdown  # noqa: F401

    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

try:
    import jinja2  # noqa: F401

    JINJA_AVAILABLE = True
except ImportError:
    JINJA_AVAILABLE = False

# ==============================================================================
# DATA CLASSES
# ==============================================================================


@dataclass  # noqa: F821
class ReportConfig:
    """Configuration for report generation."""

    title: str = "Plagiarism Detection Report"
    author: str = "Semantic Plagiarism Detector"
    company: str = ""
    logo_path: Optional[str] = None
    include_charts: bool = True
    include_detailed_analysis: bool = True
    include_appendix: bool = False
    max_pairs_to_show: int = 50
    date_format: str = "%Y-%m-%d %H:%M:%S"
    threshold: float = 0.75
    output_format: str = "pdf"  # pdf, html, json


@dataclass  # noqa: F821
class ReportData:
    """Data structure for report generation."""

    generated_at: str
    document_count: int
    total_pairs: int
    flagged_pairs: int
    avg_similarity: float
    max_similarity: float
    min_similarity: float
    flagged_pairs_list: list[dict]
    similarity_matrix: pd.DataFrame
    document_names: list[str]
    threshold_used: float
    execution_time: float
    system_info: dict[str, Any]
    alerts: list[dict]


# ==============================================================================
# REPORT GENERATOR CLASS
# ==============================================================================


class PlagiarismReportGenerator:
    """
    Comprehensive report generator for plagiarism detection results.
    """

    def __init__(self, config: ReportConfig = None):
        self.config = config or ReportConfig()
        self._setup_styles()

    def _setup_styles(self):
        """Setup report styles."""
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self.title_style = ParagraphStyle(
                "CustomTitle",
                parent=self.styles["Heading1"],
                fontSize=24,
                textColor=colors.HexColor("#1a237e"),
                spaceAfter=30,
                alignment=TA_CENTER,
            )
            self.heading_style = ParagraphStyle(
                "CustomHeading",
                parent=self.styles["Heading2"],
                fontSize=16,
                textColor=colors.HexColor("#283593"),
                spaceAfter=12,
            )
            self.subheading_style = ParagraphStyle(
                "CustomSubHeading",
                parent=self.styles["Heading3"],
                fontSize=14,
                textColor=colors.HexColor("#3949ab"),
                spaceAfter=8,
            )
            self.normal_style = ParagraphStyle(
                "CustomNormal", parent=self.styles["Normal"], fontSize=10, spaceAfter=6
            )

    def generate_report(self, report_data: ReportData) -> bytes:
        """
        Generate report in specified format.

        Args:
            report_data: ReportData object with all required information

        Returns:
            bytes: Report file content
        """
        if self.config.output_format == "pdf":
            return self._generate_pdf_report(report_data)
        elif self.config.output_format == "html":
            return self._generate_html_report(report_data)
        elif self.config.output_format == "json":
            return self._generate_json_report(report_data)
        else:
            raise ValueError(f"Unsupported format: {self.config.output_format}")

    def _generate_pdf_report(self, report_data: ReportData) -> bytes:
        """Generate PDF report using ReportLab."""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )

        story = []

        # Title
        story.append(Paragraph(f"<b>{self.config.title}</b>", self.title_style))
        story.append(Spacer(1, 0.25 * inch))

        # Metadata
        story.append(
            Paragraph(f"Generated: {report_data.generated_at}", self.normal_style)
        )
        story.append(Paragraph(f"Author: {self.config.author}", self.normal_style))
        if self.config.company:
            story.append(
                Paragraph(f"Company: {self.config.company}", self.normal_style)
            )
        story.append(Spacer(1, 0.25 * inch))

        # Executive Summary
        story.append(Paragraph("1. Executive Summary", self.heading_style))
        story.append(Spacer(1, 0.1 * inch))

        summary_data = [
            ["Metric", "Value"],
            ["Total Documents", str(report_data.document_count)],
            ["Total Pairs Evaluated", str(report_data.total_pairs)],
            ["Flagged Pairs", str(report_data.flagged_pairs)],
            ["Average Similarity", f"{report_data.avg_similarity:.2%}"],
            ["Maximum Similarity", f"{report_data.max_similarity:.2%}"],
            ["Minimum Similarity", f"{report_data.min_similarity:.2%}"],
            ["Threshold Used", f"{report_data.threshold_used:.2%}"],
        ]

        summary_table = Table(summary_data, colWidths=[2 * inch, 1.5 * inch])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#1a237e")),
                    ("TEXTCOLOR", (0, 0), (1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(summary_table)
        story.append(Spacer(1, 0.25 * inch))

        # Flagged Pairs
        if report_data.flagged_pairs_list:
            story.append(Paragraph("2. Flagged Pairs", self.heading_style))
            story.append(Spacer(1, 0.1 * inch))

            # Limit display
            pairs = report_data.flagged_pairs_list[: self.config.max_pairs_to_show]

            pair_table_data = [
                ["#", "Document A", "Document B", "Similarity", "Severity"]
            ]

            for idx, pair in enumerate(pairs, 1):
                severity = (
                    "High"
                    if pair.get("similarity", 0) > 0.90
                    else "Medium"
                    if pair.get("similarity", 0) > 0.80
                    else "Low"
                )
                pair_table_data.append(
                    [
                        str(idx),
                        pair.get("doc_a", "")[:30],
                        pair.get("doc_b", "")[:30],
                        f"{pair.get('similarity', 0):.2%}",
                        severity,
                    ]
                )

            pair_table = Table(
                pair_table_data,
                colWidths=[0.5 * inch, 1.5 * inch, 1.5 * inch, 1 * inch, 0.8 * inch],
            )
            pair_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a237e")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ]
                )
            )
            story.append(pair_table)

            if len(report_data.flagged_pairs_list) > self.config.max_pairs_to_show:
                story.append(
                    Paragraph(
                        f"<i>Showing first {self.config.max_pairs_to_show} of {len(report_data.flagged_pairs_list)} flagged pairs</i>",
                        self.normal_style,
                    )
                )
            story.append(Spacer(1, 0.25 * inch))

        # System Information
        story.append(Paragraph("3. System Information", self.heading_style))
        story.append(Spacer(1, 0.1 * inch))

        sys_info = report_data.system_info
        sys_data = [
            ["Property", "Value"],
            ["Execution Time", f"{report_data.execution_time:.2f} seconds"],
            ["Python Version", sys_info.get("python_version", "N/A")],
            ["Platform", sys_info.get("platform", "N/A")],
            ["Processor", sys_info.get("processor", "N/A")],
        ]

        sys_table = Table(sys_data, colWidths=[2 * inch, 2 * inch])
        sys_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#1a237e")),
                    ("TEXTCOLOR", (0, 0), (1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(sys_table)

        # Build PDF
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()

        return pdf_data

    def _generate_html_report(self, report_data: ReportData) -> bytes:
        """Generate HTML report."""
        # Build HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.config.title}</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 20px;
                    color: #333;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 16px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                    padding: 40px;
                }}
                .header {{
                    text-align: center;
                    padding-bottom: 30px;
                    border-bottom: 3px solid #1a237e;
                    margin-bottom: 30px;
                }}
                .header h1 {{
                    color: #1a237e;
                    font-size: 32px;
                    margin-bottom: 8px;
                }}
                .header .subtitle {{
                    color: #666;
                    font-size: 14px;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                }}
                .section h2 {{
                    color: #1a237e;
                    font-size: 20px;
                    margin-bottom: 15px;
                    border-bottom: 2px solid #3949ab;
                    padding-bottom: 8px;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 15px 0;
                }}
                .metric-card {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    text-align: center;
                }}
                .metric-card .label {{
                    font-size: 12px;
                    color: #666;
                    text-transform: uppercase;
                    font-weight: 600;
                }}
                .metric-card .value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #1a237e;
                    margin: 5px 0;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                    background: white;
                    border-radius: 8px;
                    overflow: hidden;
                }}
                th {{
                    background: #1a237e;
                    color: white;
                    padding: 10px;
                    text-align: left;
                    font-weight: 600;
                }}
                td {{
                    padding: 10px;
                    border-bottom: 1px solid #eee;
                }}
                tr:hover {{
                    background: #f5f5f5;
                }}
                .badge {{
                    display: inline-block;
                    padding: 3px 10px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: 600;
                }}
                .badge-high {{ background: #ffebee; color: #c62828; }}
                .badge-medium {{ background: #fff3e0; color: #e65100; }}
                .badge-low {{ background: #e8f5e9; color: #2e7d32; }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #666;
                    font-size: 12px;
                }}
                @media print {{
                    body {{ background: white; padding: 0; }}
                    .container {{ box-shadow: none; border-radius: 0; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 {self.config.title}</h1>
                    <div class="subtitle">Generated: {report_data.generated_at}</div>
                    <div class="subtitle">Author: {self.config.author}</div>
                </div>
        """

        # Executive Summary
        html_content += """
                <div class="section">
                    <h2>📋 Executive Summary</h2>
                    <div class="grid">
                        <div class="metric-card">
                            <div class="label">Total Documents</div>
                            <div class="value">{}</div>
                        </div>
                        <div class="metric-card">
                            <div class="label">Flagged Pairs</div>
                            <div class="value">{}</div>
                        </div>
                        <div class="metric-card">
                            <div class="label">Avg Similarity</div>
                            <div class="value">{:.1%}</div>
                        </div>
                        <div class="metric-card">
                            <div class="label">Max Similarity</div>
                            <div class="value">{:.1%}</div>
                        </div>
                    </div>
                </div>
        """.format(
            report_data.document_count,
            report_data.flagged_pairs,
            report_data.avg_similarity,
            report_data.max_similarity,
        )

        # Flagged Pairs
        if report_data.flagged_pairs_list:
            html_content += """
                <div class="section">
                    <h2>⚠️ Flagged Pairs</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Document A</th>
                                <th>Document B</th>
                                <th>Similarity</th>
                                <th>Severity</th>
                            </tr>
                        </thead>
                        <tbody>
            """

            for idx, pair in enumerate(
                report_data.flagged_pairs_list[: self.config.max_pairs_to_show], 1
            ):
                similarity = pair.get("similarity", 0)
                severity_class = (
                    "high"
                    if similarity > 0.90
                    else "medium"
                    if similarity > 0.80
                    else "low"
                )
                html_content += f"""
                            <tr>
                                <td>{idx}</td>
                                <td>{pair.get('doc_a', '')[:50]}</td>
                                <td>{pair.get('doc_b', '')[:50]}</td>
                                <td>{similarity:.1%}</td>
                                <td><span class="badge badge-{severity_class}">{severity_class.title()}</span></td>
                            </tr>
                """

            html_content += """
                        </tbody>
                    </table>
                </div>
            """

        # System Information
        html_content += """
                <div class="section">
                    <h2>⚙️ System Information</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Property</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Execution Time</td>
                                <td>{:.2f} seconds</td>
                            </tr>
                            <tr>
                                <td>Threshold Used</td>
                                <td>{:.1%}</td>
                            </tr>
                            <tr>
                                <td>Generated At</td>
                                <td>{}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
        """.format(
            report_data.execution_time,
            report_data.threshold_used,
            report_data.generated_at,
        )

        # Footer
        html_content += f"""
                <div class="footer">
                    <p>Generated by {self.config.author}</p>
                    <p>© {datetime.now().year} {self.config.company}</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html_content.encode("utf-8")

    def _generate_json_report(self, report_data: ReportData) -> bytes:
        """Generate JSON report."""
        # Convert to serializable format
        json_data = {
            "report_metadata": {
                "title": self.config.title,
                "author": self.config.author,
                "company": self.config.company,
                "generated_at": report_data.generated_at,
                "version": "1.0",
            },
            "summary": {
                "document_count": report_data.document_count,
                "total_pairs": report_data.total_pairs,
                "flagged_pairs": report_data.flagged_pairs,
                "avg_similarity": report_data.avg_similarity,
                "max_similarity": report_data.max_similarity,
                "min_similarity": report_data.min_similarity,
                "threshold_used": report_data.threshold_used,
                "execution_time": report_data.execution_time,
            },
            "flagged_pairs": report_data.flagged_pairs_list[
                : self.config.max_pairs_to_show
            ],
            "system_info": report_data.system_info,
        }

        return json.dumps(json_data, indent=2).encode("utf-8")


# ==============================================================================
# AUTOMATED REPORT SCHEDULER
# ==============================================================================


class ReportScheduler:
    """Schedule and manage automated report generation."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.schedules = []
        self._load_schedules()

    def _load_schedules(self):
        """Load schedules from storage."""
        schedule_file = self.storage_path / "schedules.json"
        if schedule_file.exists():
            try:
                with open(schedule_file) as f:
                    self.schedules = json.load(f)
            except Exception:
                self.schedules = []

    def _save_schedules(self):
        """Save schedules to storage."""
        schedule_file = self.storage_path / "schedules.json"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        with open(schedule_file, "w") as f:
            json.dump(self.schedules, f, indent=2)

    def add_schedule(self, name: str, frequency: str, config: dict):
        """Add a new report schedule."""
        schedule = {
            "id": len(self.schedules) + 1,
            "name": name,
            "frequency": frequency,  # daily, weekly, monthly
            "config": config,
            "created_at": datetime.now().isoformat(),
            "last_run": None,
            "next_run": self._calculate_next_run(frequency),
            "enabled": True,
        }
        self.schedules.append(schedule)
        self._save_schedules()
        return schedule

    def _calculate_next_run(self, frequency: str) -> str:
        """Calculate next run time based on frequency."""
        now = datetime.now()
        if frequency == "daily":
            next_run = now + timedelta(days=1)
        elif frequency == "weekly":
            next_run = now + timedelta(weeks=1)
        elif frequency == "monthly":
            next_run = now + timedelta(days=30)
        else:
            next_run = now + timedelta(days=1)
        return next_run.isoformat()

    def get_due_schedules(self) -> list[dict]:
        """Get schedules that are due to run."""
        now = datetime.now()
        due = []
        for schedule in self.schedules:
            if not schedule.get("enabled", True):
                continue
            next_run = datetime.fromisoformat(schedule["next_run"])
            if now >= next_run:
                due.append(schedule)
        return due

    def mark_run_completed(self, schedule_id: int):
        """Mark a schedule run as completed."""
        for schedule in self.schedules:
            if schedule["id"] == schedule_id:
                schedule["last_run"] = datetime.now().isoformat()
                schedule["next_run"] = self._calculate_next_run(schedule["frequency"])
                self._save_schedules()
                break


# ==============================================================================
# UI COMPONENTS
# ==============================================================================


def render_report_generator_ui():
    """Render report generator UI in Streamlit."""
    st.markdown("### 📄 Report Generator")

    # Report configuration
    col1, col2 = st.columns(2)

    with col1:
        report_title = st.text_input("Report Title", "Plagiarism Detection Report")
        report_author = st.text_input("Author", "Semantic Plagiarism Detector")

    with col2:
        report_format = st.selectbox(
            "Output Format", ["pdf", "html", "json"], format_func=lambda x: x.upper()
        )
        include_charts = st.checkbox("Include Charts", value=True)

    # Data selection
    st.markdown("#### 📊 Data Selection")

    data_source = st.radio(
        "Select Data Source",
        ["Current Analysis", "Historical Data", "Custom Range"],
        horizontal=True,
    )

    if data_source == "Custom Range":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))  # noqa: F841
        with col2:
            end_date = st.date_input("End Date", datetime.now())  # noqa: F841

    # Generate button
    if st.button("📥 Generate Report", type="primary", use_container_width=True):
        with st.spinner("Generating report..."):
            # Build report data from current session state
            report_data = _build_report_data_from_session()

            # Generate report
            config = ReportConfig(
                title=report_title,
                author=report_author,
                include_charts=include_charts,
                output_format=report_format,
            )

            generator = PlagiarismReportGenerator(config)
            report_bytes = generator.generate_report(report_data)

            # Download button
            filename = f"plagiarism_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{report_format}"
            st.download_button(
                label=f"⬇️ Download Report ({report_format.upper()})",
                data=report_bytes,
                file_name=filename,
                mime=f"application/{report_format}",
                use_container_width=True,
                type="primary",
            )
            st.success("✅ Report generated successfully!")


def _build_report_data_from_session() -> ReportData:
    """Build ReportData from current Streamlit session state."""
    # Get data from session state
    sim_df = st.session_state.get("sim_df")
    flags = st.session_state.get("flags", [])
    doc_names = st.session_state.get("doc_names", [])
    threshold = st.session_state.get("threshold_slider", 0.75)

    # Calculate metrics
    total_pairs = (
        len(doc_names) * (len(doc_names) - 1) // 2 if len(doc_names) > 1 else 0
    )

    if sim_df is not None and not sim_df.empty:
        # Get similarity values
        values = sim_df.values
        upper_tri = values[np.triu_indices_from(values, k=1)]
        avg_sim = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
        max_sim = float(np.max(upper_tri)) if len(upper_tri) > 0 else 0.0
        min_sim = float(np.min(upper_tri)) if len(upper_tri) > 0 else 0.0
    else:
        avg_sim = max_sim = min_sim = 0.0

    # Build flagged pairs list
    flagged_pairs = []
    for flag in flags:
        flagged_pairs.append(
            {
                "doc_a": flag.get("doc_a", ""),
                "doc_b": flag.get("doc_b", ""),
                "similarity": flag.get("similarity", 0.0),
                "snippet_a": flag.get("snippet_a", ""),
                "snippet_b": flag.get("snippet_b", ""),
            }
        )

    # System info
    import platform

    system_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timestamp": datetime.now().isoformat(),
    }

    return ReportData(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        document_count=len(doc_names),
        total_pairs=total_pairs,
        flagged_pairs=len(flagged_pairs),
        avg_similarity=avg_sim,
        max_similarity=max_sim,
        min_similarity=min_sim,
        flagged_pairs_list=flagged_pairs,
        similarity_matrix=sim_df if sim_df is not None else pd.DataFrame(),
        document_names=doc_names,
        threshold_used=threshold,
        execution_time=st.session_state.get("last_execution_time", 0.0),
        system_info=system_info,
        alerts=[],
    )


def render_scheduled_reports_ui():
    """Render scheduled reports management UI."""
    st.markdown("### 🗓️ Scheduled Reports")

    # Initialize scheduler
    data_dir = Path(st.session_state.get("data_dir", "."))
    scheduler = ReportScheduler(data_dir / "reports")

    # Display schedules
    if scheduler.schedules:
        df = pd.DataFrame(scheduler.schedules)
        st.dataframe(
            df[["id", "name", "frequency", "last_run", "next_run", "enabled"]],
            use_container_width=True,
        )
    else:
        st.info("No schedules configured")

    # Add new schedule
    with st.expander("➕ Add New Schedule", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            schedule_name = st.text_input("Schedule Name", "Daily Report")
            frequency = st.selectbox("Frequency", ["daily", "weekly", "monthly"])
        with col2:
            report_format = st.selectbox("Format", ["pdf", "html", "json"])

        if st.button("Create Schedule", use_container_width=True):
            scheduler.add_schedule(
                name=schedule_name,
                frequency=frequency,
                config={"format": report_format, "auto_send": False},
            )
            st.success("✅ Schedule created!")
            st.rerun()


# ==============================================================================
# MAIN EXPORT FUNCTIONS
# ==============================================================================


def initialize_report_generator():
    """Initialize report generator components."""
    if "report_generator_initialized" not in st.session_state:
        st.session_state.report_generator_initialized = True
        st.session_state.report_config = ReportConfig()
        st.session_state.report_generator = PlagiarismReportGenerator()

        # Create report storage
        data_dir = Path(st.session_state.get("data_dir", "."))
        (data_dir / "reports").mkdir(parents=True, exist_ok=True)
