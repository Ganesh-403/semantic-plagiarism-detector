# ───────────────────────────────────────────────────────────────────────────────
# ── SECTION: AUTOMATED REPORT GENERATION SYSTEM (Issue #2000) ────────────────
# ───────────────────────────────────────────────────────────────────────────────

import json
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ── Data Models ─────────────────────────────────────────────────────────────


class ReportStatus(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    DELIVERED = "delivered"
    ARCHIVED = "archived"


class ReportPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class ExportFormat(Enum):
    PDF = "pdf"
    EXCEL = "xlsx"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    DOCX = "docx"


@dataclass
class ReportTemplate:
    """Report template definition"""

    id: str
    name: str
    description: str
    sections: list[str]
    style_config: dict[str, Any]
    default_format: ExportFormat
    created_at: datetime
    updated_at: datetime
    is_custom: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class ReportJob:
    """Scheduled report job"""

    id: str
    name: str
    template_id: str
    schedule: str  # 'daily', 'weekly', 'monthly', 'custom'
    recipients: list[str]
    format: ExportFormat
    filters: dict[str, Any]
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: ReportStatus = ReportStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class Report:
    """Generated report"""

    id: str
    job_id: str
    template_id: str
    title: str
    content: str
    format: ExportFormat
    generated_at: datetime
    generated_by: str
    file_size: int = 0
    file_path: Optional[str] = None
    status: ReportStatus = ReportStatus.COMPLETED
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "generated_at": self.generated_at.isoformat(),
            "status": self.status.value,
            "format": self.format.value,
        }


@dataclass
class ReportInsight:
    """Intelligent insight for report"""

    id: str
    report_id: str
    type: str  # 'trend', 'anomaly', 'risk', 'recommendation', 'summary'
    title: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float
    data: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {**asdict(self), "created_at": self.created_at.isoformat()}


# ── Report Generator ──────────────────────────────────────────────────────


class ReportGenerator:
    """Main report generation engine"""

    def __init__(self):
        self.templates: dict[str, ReportTemplate] = {}
        self.reports: dict[str, Report] = {}
        self.jobs: dict[str, ReportJob] = {}
        self.insights: dict[str, list[ReportInsight]] = defaultdict(list)
        self._init_default_templates()

    def _init_default_templates(self):
        """Initialize default report templates"""
        default_templates = [
            {
                "id": "executive_summary",
                "name": "Executive Summary",
                "description": "High-level overview for executives",
                "sections": [
                    "Overview",
                    "Key Metrics",
                    "Risk Summary",
                    "Recommendations",
                ],
                "style_config": {
                    "color_scheme": "professional",
                    "include_charts": True,
                },
                "default_format": ExportFormat.PDF,
            },
            {
                "id": "detailed_analysis",
                "name": "Detailed Analysis",
                "description": "Comprehensive analysis with all metrics",
                "sections": [
                    "Overview",
                    "Document Analysis",
                    "Similarity Matrix",
                    "Flagged Cases",
                    "Trend Analysis",
                    "Recommendations",
                    "Appendix",
                ],
                "style_config": {"color_scheme": "analytical", "include_charts": True},
                "default_format": ExportFormat.PDF,
            },
            {
                "id": "summary_report",
                "name": "Summary Report",
                "description": "Quick summary for regular monitoring",
                "sections": ["Summary", "Key Findings", "Statistics"],
                "style_config": {"color_scheme": "clean", "include_charts": False},
                "default_format": ExportFormat.HTML,
            },
            {
                "id": "compliance_report",
                "name": "Compliance Report",
                "description": "Regulatory compliance focused report",
                "sections": [
                    "Compliance Overview",
                    "Policy Violations",
                    "Actions Taken",
                    "Audit Trail",
                ],
                "style_config": {"color_scheme": "formal", "include_charts": True},
                "default_format": ExportFormat.PDF,
            },
        ]

        for template_data in default_templates:
            template = ReportTemplate(
                id=template_data["id"],
                name=template_data["name"],
                description=template_data["description"],
                sections=template_data["sections"],
                style_config=template_data["style_config"],
                default_format=template_data["default_format"],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            self.templates[template.id] = template

    def create_template(
        self,
        name: str,
        description: str,
        sections: List[str],
        style_config: Dict,
        format: ExportFormat = ExportFormat.PDF,
    ) -> ReportTemplate:
        """Create a new custom template"""
        template = ReportTemplate(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            sections=sections,
            style_config=style_config,
            default_format=format,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            is_custom=True,
        )
        self.templates[template.id] = template
        return template

    def generate_report(
        self,
        template_id: str,
        data: Dict[str, Any],
        title: str = None,
        format: ExportFormat = None,
        generated_by: str = "system",
    ) -> Report:
        """Generate a report from template"""
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")

        if not format:
            format = template.default_format

        # Generate report content
        content = self._generate_content(template, data)

        # Generate insights
        insights = self._generate_insights(data, template)

        # Create report
        report_id = str(uuid.uuid4())
        report = Report(
            id=report_id,
            job_id="manual",
            template_id=template_id,
            title=title or f"{template.name} Report",
            content=content,
            format=format,
            generated_at=datetime.now(),
            generated_by=generated_by,
        )

        self.reports[report_id] = report

        # Store insights
        if insights:
            self.insights[report_id] = insights

        return report

    def _generate_content(self, template: ReportTemplate, data: Dict) -> str:
        """Generate report content as HTML"""
        # Build HTML content
        html_parts = []

        # CSS styles
        styles = self._get_style_css(template.style_config)
        html_parts.append(f"<style>{styles}</style>")

        # Header
        html_parts.append(f"""
        <div class="report-header">
            <h1>{data.get("title", template.name)}</h1>
            <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        """)

        # Sections
        for section in template.sections:
            section_content = self._generate_section(section, data)
            if section_content:
                html_parts.append(f"""
                <div class="report-section">
                    <h2>{section}</h2>
                    {section_content}
                </div>
                """)

        # Footer
        html_parts.append("""
        <div class="report-footer">
            <p>Generated by Automated Report System</p>
        </div>
        """)

        return "\n".join(html_parts)

    def _get_style_css(self, style_config: Dict) -> str:
        """Get CSS styles based on configuration"""
        colors = {
            "professional": {
                "primary": "#1a237e",
                "secondary": "#0d47a1",
                "bg": "#f5f5f5",
            },
            "analytical": {
                "primary": "#263238",
                "secondary": "#455a64",
                "bg": "#eceff1",
            },
            "clean": {"primary": "#37474f", "secondary": "#607d8b", "bg": "#ffffff"},
            "formal": {"primary": "#1a1a1a", "secondary": "#333333", "bg": "#fafafa"},
        }

        scheme = style_config.get("color_scheme", "professional")
        colorset = colors.get(scheme, colors["professional"])

        return f"""
        .report-header {{
            background: {colorset["primary"]};
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .report-section {{
            background: {colorset["bg"]};
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid {colorset["secondary"]};
        }}
        .report-footer {{
            text-align: center;
            padding: 10px;
            color: #666;
            font-size: 12px;
            border-top: 1px solid #ddd;
            margin-top: 20px;
        }}
        .metric-card {{
            display: inline-block;
            background: white;
            padding: 15px;
            margin: 5px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: {colorset["primary"]};
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
        }}
        .risk-high {{ color: #d32f2f; }}
        .risk-medium {{ color: #f57c00; }}
        .risk-low {{ color: #388e3c; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: var(--background-color, #ffffff);
            color: var(--text-color, #111827);
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: {colorset["secondary"]};
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        """

    def _generate_section(self, section: str, data: Dict) -> str:
        """Generate content for a specific section"""
        section_handlers = {
            "Overview": self._generate_overview_section,
            "Key Metrics": self._generate_metrics_section,
            "Risk Summary": self._generate_risk_section,
            "Recommendations": self._generate_recommendations_section,
            "Document Analysis": self._generate_document_analysis_section,
            "Similarity Matrix": self._generate_similarity_section,
            "Flagged Cases": self._generate_flagged_cases_section,
            "Trend Analysis": self._generate_trend_section,
            "Appendix": self._generate_appendix_section,
            "Summary": self._generate_summary_section,
            "Key Findings": self._generate_findings_section,
            "Statistics": self._generate_statistics_section,
            "Compliance Overview": self._generate_compliance_section,
            "Policy Violations": self._generate_violations_section,
            "Actions Taken": self._generate_actions_section,
            "Audit Trail": self._generate_audit_section,
        }

        handler = section_handlers.get(section)
        if handler:
            return handler(data)
        return f"<p>Content for {section}</p>"

    def _generate_overview_section(self, data: Dict) -> str:
        """Generate overview section"""
        total_docs = data.get("total_documents", 0)
        flagged = data.get("flagged_count", 0)
        avg_sim = data.get("avg_similarity", 0)

        return f"""
        <div class="overview">
            <div class="metric-card">
                <div class="metric-value">{total_docs}</div>
                <div class="metric-label">Total Documents</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{flagged}</div>
                <div class="metric-label">Flagged Cases</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_sim:.1%}</div>
                <div class="metric-label">Avg Similarity</div>
            </div>
        </div>
        """

    def _generate_metrics_section(self, data: Dict) -> str:
        """Generate metrics section"""
        metrics = data.get("metrics", {})
        html = '<div class="metrics-grid">'

        for key, value in metrics.items():
            html += f"""
            <div class="metric-card">
                <div class="metric-value">{value if isinstance(value, (int, float)) else value}</div>
                <div class="metric-label">{key.replace("_", " ").title()}</div>
            </div>
            """

        html += "</div>"
        return html

    def _generate_risk_section(self, data: Dict) -> str:
        """Generate risk summary section"""
        risk_levels = data.get("risk_levels", {})
        if not risk_levels:
            return "<p>No risk data available</p>"

        html = '<div class="risk-summary">'
        for level, count in risk_levels.items():
            css_class = f"risk-{level.lower()}"
            html += f"""
            <div class="metric-card">
                <div class="metric-value {css_class}">{count}</div>
                <div class="metric-label">{level.title()} Risk</div>
            </div>
            """
        html += "</div>"
        return html

    def _generate_recommendations_section(self, data: Dict) -> str:
        """Generate recommendations section"""
        recommendations = data.get("recommendations", [])
        if not recommendations:
            return "<p>No recommendations available</p>"

        html = "<ul>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul>"
        return html

    def _generate_document_analysis_section(self, data: Dict) -> str:
        """Generate document analysis section"""
        docs = data.get("documents", [])
        if not docs:
            return "<p>No document data available</p>"

        html = "<table><tr><th>Document</th><th>Word Count</th><th>Uniqueness</th><th>Risk</th></tr>"
        for doc in docs[:20]:
            html += f"""
            <tr>
                <td>{doc.get("name", "Unknown")}</td>
                <td>{doc.get("word_count", 0)}</td>
                <td>{doc.get("uniqueness", 0):.1%}</td>
                <td class="risk-{doc.get("risk", "low").lower()}">{doc.get("risk", "Low")}</td>
            </tr>
            """
        html += "</table>"
        return html

    def _generate_similarity_section(self, data: Dict) -> str:
        """Generate similarity matrix section"""
        matrix = data.get("similarity_matrix", [])
        if not matrix:
            return "<p>No similarity data available</p>"

        html = '<div style="overflow-x:auto;">'
        html += "<table><tr><th>Document</th>"

        # Headers
        docs = list(matrix[0].keys()) if matrix else []
        for doc in docs:
            html += f"<th>{doc[:20]}</th>"
        html += "</tr>"

        # Rows
        for row in matrix:
            doc_name = row.get("name", "Unknown")
            html += f"<tr><td>{doc_name[:20]}</td>"
            for doc in docs:
                value = row.get(doc, 0)
                color = self._get_similarity_color(value)
                html += f'<td style="background:{color};">{value:.1%}</td>'
            html += "</tr>"

        html += "</table></div>"
        return html

    def _get_similarity_color(self, value: float) -> str:
        """Get color for similarity value"""
        if value > 0.8:
            return "#d32f2f"
        elif value > 0.6:
            return "#f57c00"
        elif value > 0.4:
            return "#f9a825"
        else:
            return "#81c784"

    def _generate_trend_section(self, data: Dict) -> str:
        """Generate trend analysis section"""
        trends = data.get("trends", {})
        if not trends:
            return "<p>No trend data available</p>"

        html = '<div class="trend-data">'
        for key, value in trends.items():
            direction = value.get("direction", "stable")
            icon = "📈" if direction == "up" else "📉" if direction == "down" else "➡️"
            html += f"""
            <div class="metric-card">
                <div class="metric-value">{icon} {value.get("change", 0):.1%}</div>
                <div class="metric-label">{key.replace("_", " ").title()}</div>
            </div>
            """
        html += "</div>"
        return html

    def _generate_summary_section(self, data: Dict) -> str:
        """Generate summary section"""
        summary = data.get("summary", "")
        if not summary:
            return "<p>No summary available</p>"

        return f'<div class="summary-content"><p>{summary}</p></div>'

    def _generate_findings_section(self, data: Dict) -> str:
        """Generate key findings section"""
        findings = data.get("findings", [])
        if not findings:
            return "<p>No findings available</p>"

        html = "<ul>"
        for finding in findings:
            importance = finding.get("importance", "medium")
            icon = (
                "🔴"
                if importance == "high"
                else "🟡"
                if importance == "medium"
                else "🟢"
            )
            html += f"<li>{icon} {finding.get('description', '')}</li>"
        html += "</ul>"
        return html

    def _generate_statistics_section(self, data: Dict) -> str:
        """Generate statistics section"""
        stats = data.get("statistics", {})
        if not stats:
            return "<p>No statistics available</p>"

        html = '<div class="statistics-grid">'
        for key, value in stats.items():
            html += f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{key.replace("_", " ").title()}</div>
            </div>
            """
        html += "</div>"
        return html

    def _generate_compliance_section(self, data: Dict) -> str:
        """Generate compliance section"""
        compliance = data.get("compliance", {})
        if not compliance:
            return "<p>No compliance data available</p>"

        status = compliance.get("status", "Unknown")
        status_icon = "✅" if status == "compliant" else "⚠️"

        return f"""
        <div class="compliance-status">
            <div class="metric-card">
                <div class="metric-value">{status_icon} {status.title()}</div>
                <div class="metric-label">Compliance Status</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{compliance.get("score", 0):.1%}</div>
                <div class="metric-label">Compliance Score</div>
            </div>
        </div>
        """

    def _generate_violations_section(self, data: Dict) -> str:
        """Generate policy violations section"""
        violations = data.get("violations", [])
        if not violations:
            return "<p>No policy violations detected</p>"

        html = "<table><tr><th>Policy</th><th>Severity</th><th>Status</th></tr>"
        for violation in violations:
            html += f"""
            <tr>
                <td>{violation.get("policy", "Unknown")}</td>
                <td class="risk-{violation.get("severity", "medium").lower()}">{violation.get("severity", "Medium").title()}</td>
                <td>{violation.get("status", "Pending")}</td>
            </tr>
            """
        html += "</table>"
        return html

    def _generate_actions_section(self, data: Dict) -> str:
        """Generate actions taken section"""
        actions = data.get("actions", [])
        if not actions:
            return "<p>No actions recorded</p>"

        html = "<ul>"
        for action in actions:
            html += f"""
            <li>
                <strong>{action.get("type", "Action")}</strong> - 
                {action.get("description", "")}
                <small>({action.get("timestamp", "")})</small>
            </li>
            """
        html += "</ul>"
        return html

    def _generate_audit_section(self, data: Dict) -> str:
        """Generate audit trail section"""
        audit_trail = data.get("audit_trail", [])
        if not audit_trail:
            return "<p>No audit trail available</p>"

        html = "<table><tr><th>Timestamp</th><th>User</th><th>Action</th><th>Details</th></tr>"
        for entry in audit_trail[:50]:
            html += f"""
            <tr>
                <td>{entry.get("timestamp", "")}</td>
                <td>{entry.get("user", "Unknown")}</td>
                <td>{entry.get("action", "")}</td>
                <td>{entry.get("details", "")}</td>
            </tr>
            """
        html += "</table>"
        return html

    def _generate_insights(
        self, data: Dict, template: ReportTemplate
    ) -> List[ReportInsight]:
        """Generate intelligent insights from data"""
        insights = []

        # Analyze risk levels
        risk_levels = data.get("risk_levels", {})
        if risk_levels:
            critical = risk_levels.get("critical", 0)
            high = risk_levels.get("high", 0)

            if critical > 0:
                insights.append(
                    ReportInsight(
                        id=str(uuid.uuid4()),
                        report_id="",
                        type="risk",
                        title="Critical Risk Detected",
                        description=f"{critical} critical risk items found requiring immediate attention",
                        severity="critical",
                        confidence=0.95,
                        data={"count": critical},
                    )
                )

            if high > 0:
                insights.append(
                    ReportInsight(
                        id=str(uuid.uuid4()),
                        report_id="",
                        type="risk",
                        title="High Risk Items",
                        description=f"{high} high risk items detected. Review recommended.",
                        severity="high",
                        confidence=0.85,
                        data={"count": high},
                    )
                )

        # Analyze trends
        trends = data.get("trends", {})
        for key, value in trends.items():
            if (
                value.get("direction") in ["up", "down"]
                and abs(value.get("change", 0)) > 0.1
            ):
                insights.append(
                    ReportInsight(
                        id=str(uuid.uuid4()),
                        report_id="",
                        type="trend",
                        title=f"{key.replace('_', ' ').title()} Trend",
                        description=f"{key.replace('_', ' ').title()} is {value.get('direction')} by {value.get('change', 0):.1%}",
                        severity="medium",
                        confidence=0.75,
                        data={
                            "direction": value.get("direction"),
                            "change": value.get("change", 0),
                        },
                    )
                )

        # Overall recommendation
        overall_score = data.get("overall_score", 0)
        if overall_score > 0.7:
            insights.append(
                ReportInsight(
                    id=str(uuid.uuid4()),
                    report_id="",
                    type="recommendation",
                    title="High Overall Similarity Detected",
                    description="Overall similarity score exceeds 70%. Comprehensive review recommended.",
                    severity="high",
                    confidence=0.90,
                    data={"score": overall_score},
                )
            )

        return insights

    def export_report(self, report_id: str, format: ExportFormat = None) -> BytesIO:
        """Export report to specified format"""
        report = self.reports.get(report_id)
        if not report:
            raise ValueError(f"Report {report_id} not found")

        format = format or report.format

        if format == ExportFormat.PDF:
            return self._export_pdf(report)
        elif format == ExportFormat.HTML:
            return self._export_html(report)
        elif format == ExportFormat.JSON:
            return self._export_json(report)
        elif format == ExportFormat.EXCEL:
            return self._export_excel(report)
        elif format == ExportFormat.CSV:
            return self._export_csv(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_pdf(self, report: Report) -> BytesIO:
        """Export to PDF"""
        try:
            import weasyprint
        except ImportError:
            raise ImportError("WeasyPrint required for PDF export")

        html_content = self._wrap_html(report.content)
        pdf_bytes = weasyprint.HTML(string=html_content).write_pdf()
        return BytesIO(pdf_bytes)

    def _export_html(self, report: Report) -> BytesIO:
        """Export to HTML"""
        html_content = self._wrap_html(report.content)
        return BytesIO(html_content.encode("utf-8"))

    def _export_json(self, report: Report) -> BytesIO:
        """Export to JSON"""
        data = {
            "id": report.id,
            "title": report.title,
            "content": report.content,
            "generated_at": report.generated_at.isoformat(),
            "metadata": report.metadata,
        }
        json_str = json.dumps(data, indent=2)
        return BytesIO(json_str.encode("utf-8"))

    def _export_excel(self, report: Report) -> BytesIO:
        """Export to Excel"""
        output = BytesIO()

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            # Extract data from report content
            # This is a simplified version
            summary_data = {
                "Report Title": [report.title],
                "Generated At": [report.generated_at.strftime("%Y-%m-%d %H:%M:%S")],
                "Report ID": [report.id],
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        output.seek(0)
        return output

    def _export_csv(self, report: Report) -> BytesIO:
        """Export to CSV"""
        # Simplified export
        data = {
            "report_id": [report.id],
            "title": [report.title],
            "generated_at": [report.generated_at.strftime("%Y-%m-%d %H:%M:%S")],
        }
        df = pd.DataFrame(data)
        csv_content = df.to_csv(index=False)
        return BytesIO(csv_content.encode("utf-8"))

    def _wrap_html(self, content: str) -> str:
        """Wrap HTML content with full page structure"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Report</title>
            {content}
        </head>
        <body>
            {content}
        </body>
        </html>
        """

    def create_scheduled_job(
        self,
        name: str,
        template_id: str,
        schedule: str,
        recipients: List[str],
        format: ExportFormat = None,
        filters: Dict = None,
    ) -> ReportJob:
        """Create a scheduled report job"""
        if not format:
            template = self.templates.get(template_id)
            format = template.default_format if template else ExportFormat.PDF

        job = ReportJob(
            id=str(uuid.uuid4()),
            name=name,
            template_id=template_id,
            schedule=schedule,
            recipients=recipients,
            format=format,
            filters=filters or {},
            created_at=datetime.now(),
        )

        self.jobs[job.id] = job
        return job

    def run_scheduled_job(self, job_id: str, data: Dict = None) -> Optional[Report]:
        """Execute a scheduled report job"""
        job = self.jobs.get(job_id)
        if not job:
            return None

        try:
            report = self.generate_report(
                job.template_id,
                data or {},
                title=f"{job.name} - {datetime.now().strftime('%Y-%m-%d')}",
                format=job.format,
                generated_by="scheduler",
            )

            job.last_run = datetime.now()
            job.status = ReportStatus.COMPLETED

            return report
        except Exception:
            job.status = ReportStatus.FAILED
            return None

    def get_report_stats(self) -> Dict:
        """Get report generation statistics"""
        total = len(self.reports)
        if total == 0:
            return {"total": 0}

        formats = Counter(r.format.value for r in self.reports.values())

        return {
            "total": total,
            "templates": len(self.templates),
            "jobs": len(self.jobs),
            "formats": dict(formats),
            "avg_size": sum(r.file_size for r in self.reports.values()) / total
            if total
            else 0,
        }


# ── UI Components ──────────────────────────────────────────────────────────


def render_report_generator_ui(report_generator: ReportGenerator):
    """Render report generator UI"""
    st.subheader("📄 Automated Report Generator")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Generate Report", "📝 Templates", "⏰ Scheduled Jobs", "📊 History"]
    )

    with tab1:
        render_generation_tab(report_generator)

    with tab2:
        render_templates_tab(report_generator)

    with tab3:
        render_jobs_tab(report_generator)

    with tab4:
        render_history_tab(report_generator)


def render_generation_tab(report_generator: ReportGenerator):
    """Render report generation tab"""
    st.subheader("📋 Generate New Report")

    # Select template
    templates = list(report_generator.templates.values())
    template_options = {t.id: t.name for t in templates}

    col1, col2 = st.columns(2)
    with col1:
        template_id = st.selectbox(
            "Select Template",
            options=list(template_options.keys()),
            format_func=lambda x: template_options.get(x, x),
        )

    with col2:
        format_options = [f.value for f in ExportFormat]
        selected_format = st.selectbox("Export Format", options=format_options)

    # Report title
    title = st.text_input("Report Title", placeholder="Enter report title...")

    # Data selection
    st.subheader("📊 Select Data for Report")

    col1, col2, col3 = st.columns(3)
    with col1:
        include_metrics = st.checkbox("Include Metrics", value=True)
    with col2:
        include_risks = st.checkbox("Include Risk Analysis", value=True)
    with col3:
        include_trends = st.checkbox("Include Trends", value=True)

    # Generate button
    if st.button("🚀 Generate Report", type="primary"):
        if template_id:
            # Prepare data
            data = {
                "title": title or "Automated Report",
                "total_documents": st.session_state.get("total_documents", 0),
                "flagged_count": st.session_state.get("flagged_count", 0),
                "avg_similarity": st.session_state.get("avg_similarity", 0),
                "risk_levels": st.session_state.get("risk_levels", {}),
            }

            with st.spinner("Generating report..."):
                format_enum = ExportFormat(selected_format)
                report = report_generator.generate_report(
                    template_id, data, title, format_enum
                )

                if report:
                    st.success(f"✅ Report generated: {report.id}")

                    # Export
                    report_bytes = report_generator.export_report(
                        report.id, format_enum
                    )

                    st.download_button(
                        label=f"📥 Download {selected_format.upper()}",
                        data=report_bytes,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{selected_format}",
                        mime=f"application/{selected_format}",
                    )

                    # Show preview
                    st.session_state["preview_report"] = report
                    st.rerun()

    # Preview
    if "preview_report" in st.session_state:
        report = st.session_state["preview_report"]
        st.subheader("📄 Preview")
        st.components.v1.html(report.content, height=400, scrolling=True)


def render_templates_tab(report_generator: ReportGenerator):
    """Render templates management tab"""
    st.subheader("📝 Report Templates")

    templates = list(report_generator.templates.values())

    # List templates
    for template in templates:
        with st.expander(f"📄 {template.name}", expanded=False):
            st.markdown(f"**Description:** {template.description}")
            st.markdown(f"**Sections:** {', '.join(template.sections)}")
            st.markdown(f"**Default Format:** {template.default_format.value}")
            st.markdown(
                f"**Created:** {template.created_at.strftime('%Y-%m-%d %H:%M')}"
            )
            st.markdown(f"**Custom:** {'Yes' if template.is_custom else 'No'}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 Use Template", key=f"use_{template.id}"):
                    st.info("Template selected for generation")
            with col2:
                if template.is_custom:
                    if st.button("🗑️ Delete", key=f"delete_{template.id}"):
                        if template.id in report_generator.templates:
                            del report_generator.templates[template.id]
                            st.rerun()

    # Create custom template
    with st.expander("➕ Create Custom Template", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            template_name = st.text_input("Template Name")
            template_desc = st.text_area("Description")
        with col2:
            sections = st.text_area(
                "Sections (one per line)",
                placeholder="Overview\nKey Metrics\nRisk Summary\nRecommendations",
            )
            format_enum = st.selectbox(
                "Default Format", [f.value for f in ExportFormat]
            )

        if st.button("Create Template"):
            if template_name and sections:
                section_list = [s.strip() for s in sections.split("\n") if s.strip()]
                template = report_generator.create_template(
                    template_name,
                    template_desc,
                    section_list,
                    {"color_scheme": "professional"},
                    ExportFormat(format_enum),
                )
                st.success(f"✅ Template '{template.name}' created!")
                st.rerun()


def render_jobs_tab(report_generator: ReportGenerator):
    """Render scheduled jobs tab"""
    st.subheader("⏰ Scheduled Report Jobs")

    jobs = list(report_generator.jobs.values())

    # List jobs
    if jobs:
        for job in jobs:
            with st.expander(f"📅 {job.name}", expanded=False):
                st.markdown(
                    f"**Template:** {report_generator.templates.get(job.template_id, {}).name}"
                )
                st.markdown(f"**Schedule:** {job.schedule}")
                st.markdown(f"**Format:** {job.format.value}")
                st.markdown(f"**Recipients:** {', '.join(job.recipients)}")
                st.markdown(f"**Status:** {job.status.value}")
                if job.last_run:
                    st.markdown(
                        f"**Last Run:** {job.last_run.strftime('%Y-%m-%d %H:%M')}"
                    )
                if job.next_run:
                    st.markdown(
                        f"**Next Run:** {job.next_run.strftime('%Y-%m-%d %H:%M')}"
                    )

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("▶️ Run Now", key=f"run_{job.id}"):
                        report_generator.run_scheduled_job(job.id)
                        st.success("✅ Job executed!")
                        st.rerun()
                with col2:
                    if st.button("✏️ Edit", key=f"edit_{job.id}"):
                        st.info("Edit functionality")
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_job_{job.id}"):
                        if job.id in report_generator.jobs:
                            del report_generator.jobs[job.id]
                            st.rerun()

    # Create new job
    with st.expander("➕ Create Scheduled Job", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            job_name = st.text_input("Job Name")
            template_id = st.selectbox(
                "Template",
                options=list(report_generator.templates.keys()),
                format_func=lambda x: report_generator.templates.get(x, {}).name,
            )
            schedule = st.selectbox(
                "Schedule", ["daily", "weekly", "monthly", "custom"]
            )
        with col2:
            format_enum = st.selectbox("Format", [f.value for f in ExportFormat])
            recipients = st.text_input("Recipients (comma separated)")
            filters_json = st.text_area("Filters (JSON)", value="{}")

        if st.button("Create Job"):
            if job_name and recipients:
                recipient_list = [r.strip() for r in recipients.split(",") if r.strip()]
                try:
                    filters = json.loads(filters_json) if filters_json else {}
                except:
                    filters = {}

                job = report_generator.create_scheduled_job(
                    job_name,
                    template_id,
                    schedule,
                    recipient_list,
                    ExportFormat(format_enum),
                    filters,
                )
                st.success(f"✅ Job '{job.name}' created!")
                st.rerun()


def render_history_tab(report_generator: ReportGenerator):
    """Render report history tab"""
    st.subheader("📊 Report History")

    # Stats
    stats = report_generator.get_report_stats()
    if stats.get("total", 0) > 0:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reports", stats["total"])
        col2.metric("Templates", stats["templates"])
        col3.metric("Scheduled Jobs", stats["jobs"])
        col4.metric("Avg Size", f"{stats['avg_size'] / 1024:.1f} KB")

        # Format distribution
        if stats.get("formats"):
            st.subheader("📈 Format Distribution")
            format_df = pd.DataFrame(
                {
                    "Format": list(stats["formats"].keys()),
                    "Count": list(stats["formats"].values()),
                }
            )
            st.bar_chart(format_df.set_index("Format"))

    # List reports
    reports = list(report_generator.reports.values())
    if reports:
        st.subheader("📋 Generated Reports")
        for report in reports[-20:]:
            with st.expander(
                f"📄 {report.title} - {report.generated_at.strftime('%Y-%m-%d %H:%M')}"
            ):
                st.markdown(f"**ID:** {report.id}")
                st.markdown(
                    f"**Template:** {report_generator.templates.get(report.template_id, {}).name}"
                )
                st.markdown(f"**Format:** {report.format.value}")
                st.markdown(f"**Status:** {report.status.value}")
                st.markdown(f"**Generated By:** {report.generated_by}")

                # Insights
                if report.id in report_generator.insights:
                    st.markdown("**Insights:**")
                    for insight in report_generator.insights[report.id]:
                        st.markdown(
                            f"- [{insight.severity.upper()}] {insight.title}: {insight.description}"
                        )

                # Export button
                if st.button("📥 Export", key=f"export_{report.id}"):
                    report_bytes = report_generator.export_report(
                        report.id, report.format
                    )
                    st.download_button(
                        label=f"Download {report.format.value.upper()}",
                        data=report_bytes,
                        file_name=f"{report.title}.{report.format.value}",
                        mime=f"application/{report.format.value}",
                    )


# ── Integration with Main App ─────────────────────────────────────────────


def integrate_report_generator():
    """Initialize and integrate report generator"""
    if "report_generator" not in st.session_state:
        st.session_state["report_generator"] = ReportGenerator()

    # Add report generator tab to main app
    render_report_generator_ui(st.session_state["report_generator"])


# ── End of Report Generator ───────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────────
