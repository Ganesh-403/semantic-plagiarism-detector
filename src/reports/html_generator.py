"""
HTML report generator for the Report Generation Module
Creates beautiful, interactive HTML reports with embedded visualizations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import base64

from src.models.report import Report, ReportConfig, ReportSection
from .visualizations import ReportVisualizer


class HTMLGenerator:
    """
    Generates HTML reports with embedded visualizations and interactive elements.
    """
    
    def __init__(self):
        self.visualizer = ReportVisualizer()
        self._templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load HTML templates."""
        return {
            'header': """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{title}</title>
                <style>
                    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: #f0f2f5;
                        padding: 40px;
                        color: #333;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                        background: white;
                        padding: 50px;
                        border-radius: 12px;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                    }}
                    .header {{
                        border-bottom: 4px solid #3498db;
                        padding-bottom: 20px;
                        margin-bottom: 30px;
                    }}
                    .header h1 {{
                        font-size: 32px;
                        color: #2c3e50;
                        font-weight: 700;
                    }}
                    .header-meta {{
                        display: flex;
                        justify-content: space-between;
                        color: #7f8c8d;
                        font-size: 14px;
                        margin-top: 8px;
                        flex-wrap: wrap;
                        gap: 8px;
                    }}
                    .badge {{
                        display: inline-block;
                        padding: 4px 12px;
                        border-radius: 20px;
                        font-size: 12px;
                        font-weight: 600;
                    }}
                    .badge-high {{ background: #fee2e2; color: #dc2626; }}
                    .badge-medium {{ background: #fef3c7; color: #d97706; }}
                    .badge-low {{ background: #dbeafe; color: #2563eb; }}
                    .badge-none {{ background: #e5e7eb; color: #6b7280; }}
                    
                    .section {{
                        margin: 40px 0;
                        padding: 30px;
                        background: #f8fafc;
                        border-radius: 10px;
                        border: 1px solid #e2e8f0;
                    }}
                    .section h2 {{
                        font-size: 22px;
                        color: #1e293b;
                        margin-bottom: 16px;
                        padding-bottom: 10px;
                        border-bottom: 2px solid #e2e8f0;
                    }}
                    .section h3 {{
                        font-size: 18px;
                        color: #334155;
                        margin: 16px 0 10px 0;
                    }}
                    
                    .stats-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                        gap: 16px;
                        margin: 16px 0;
                    }}
                    .stat-card {{
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        text-align: center;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
                        border: 1px solid #e2e8f0;
                    }}
                    .stat-value {{
                        font-size: 28px;
                        font-weight: 700;
                        color: #0f172a;
                    }}
                    .stat-label {{
                        font-size: 13px;
                        color: #64748b;
                        margin-top: 4px;
                    }}
                    
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 16px 0;
                        font-size: 14px;
                    }}
                    th, td {{
                        padding: 12px 16px;
                        text-align: left;
                        border-bottom: 1px solid #e2e8f0;
                    }}
                    th {{
                        background: #f1f5f9;
                        font-weight: 600;
                        color: #1e293b;
                    }}
                    tr:hover {{
                        background: #f8fafc;
                    }}
                    
                    .heatmap-container {{
                        margin: 20px 0;
                        text-align: center;
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        border: 1px solid #e2e8f0;
                    }}
                    .heatmap-container img {{
                        max-width: 100%;
                        border-radius: 6px;
                    }}
                    
                    .match-card {{
                        background: white;
                        padding: 16px 20px;
                        margin: 8px 0;
                        border-radius: 6px;
                        border-left: 4px solid #3498db;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
                    }}
                    .match-card.high {{ border-left-color: #dc2626; }}
                    .match-card.medium {{ border-left-color: #d97706; }}
                    .match-card.low {{ border-left-color: #2563eb; }}
                    
                    .footer {{
                        margin-top: 50px;
                        padding-top: 20px;
                        border-top: 2px solid #e2e8f0;
                        text-align: center;
                        color: #94a3b8;
                        font-size: 13px;
                    }}
                    
                    .progress-bar {{
                        width: 100%;
                        height: 8px;
                        background: #e2e8f0;
                        border-radius: 4px;
                        overflow: hidden;
                        margin: 8px 0;
                    }}
                    .progress-fill {{
                        height: 100%;
                        background: linear-gradient(90deg, #3498db, #2ecc71);
                        border-radius: 4px;
                        transition: width 0.5s ease;
                    }}
                    
                    .toast {{
                        position: fixed;
                        bottom: 20px;
                        right: 20px;
                        background: #1e293b;
                        color: white;
                        padding: 12px 24px;
                        border-radius: 8px;
                        font-size: 14px;
                        z-index: 999;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    }}
                    
                    @media print {{
                        body {{ background: white; padding: 20px; }}
                        .container {{ box-shadow: none; padding: 20px; }}
                        .section {{ background: white; border: 1px solid #ddd; }}
                        .stat-card {{ border: 1px solid #ddd; }}
                        .heatmap-container {{ border: 1px solid #ddd; }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
            """
        }
    
    def generate(self, report: Report, content: Dict[str, Any]) -> str:
        """
        Generate an HTML report.
        
        Args:
            report: Report object
            content: Report content data
        
        Returns:
            HTML string
        """
        html = []
        
        # Header
        html.append(self._generate_header(report))
        
        # Executive Summary
        if report.statistics:
            html.append(self._generate_summary(report))
        
        # Heatmap
        if report.similarity_matrix and report.document_names:
            html.append(self._generate_heatmap(report))
        
        # Matches
        if report.matches:
            html.append(self._generate_matches(report))
        
        # Detailed Statistics
        if report.statistics:
            html.append(self._generate_statistics(report))
        
        # Footer
        html.append(self._generate_footer(report))
        
        return "\n".join(html)
    
    def _generate_header(self, report: Report) -> str:
        """Generate report header section."""
        status_badge = {
            'pending': '🟡 Pending',
            'generating': '🔄 Generating',
            'completed': '✅ Completed',
            'failed': '❌ Failed'
        }.get(report.status.value, report.status.value)
        
        return f"""
        <div class="header">
            <h1>📊 {report.title}</h1>
            <div class="header-meta">
                <span>📋 Report ID: {report.id[:8]}</span>
                <span>📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
                <span>📁 Type: {report.report_type.value.replace('_', ' ').title()}</span>
                <span>📄 Format: {report.format.value.upper()}</span>
                <span>🔖 Status: {status_badge}</span>
            </div>
            {f'<p style="margin-top:12px;color:#475569;">{report.description}</p>' if report.description else ''}
        </div>
        """
    
    def _generate_summary(self, report: Report) -> str:
        """Generate executive summary section."""
        stats = report.statistics
        
        html = """
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="stats-grid">
        """
        
        summary_items = [
            ('📄 Documents', stats.get('total_documents', 0)),
            ('🔄 Comparisons', stats.get('total_comparisons', 0)),
            ('🎯 Matches', stats.get('total_matches', 0)),
            ('📈 Avg Similarity', f"{stats.get('avg_similarity', 0):.1%}"),
            ('📈 Max Similarity', f"{stats.get('max_similarity', 0):.1%}"),
            ('🔴 High Severity', stats.get('high_severity_count', 0))
        ]
        
        for label, value in summary_items:
            html += f"""
                <div class="stat-card">
                    <div class="stat-value">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _generate_heatmap(self, report: Report) -> str:
        """Generate heatmap section."""
        if not report.similarity_matrix or not report.document_names:
            return ""
        
        img_data = self.visualizer.create_heatmap(
            report.similarity_matrix,
            report.document_names,
            title="Document Similarity Matrix"
        )
        
        return f"""
        <div class="section">
            <h2>📈 Similarity Heatmap</h2>
            <p style="color:#475569;margin-bottom:16px;">
                This heatmap shows pairwise similarity scores between all documents.
                Darker colors indicate higher similarity.
            </p>
            <div class="heatmap-container">
                <img src="data:image/png;base64,{img_data}" alt="Similarity Heatmap" />
            </div>
        </div>
        """
    
    def _generate_matches(self, report: Report) -> str:
        """Generate matches section."""
        matches = report.matches[:100]
        if not matches:
            return ""
        
        html = """
        <div class="section">
            <h2>🔍 Detected Matches</h2>
            <p style="color:#475569;margin-bottom:16px;">
                Found {total} matches across documents.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Source Document</th>
                        <th>Target Document</th>
                        <th>Score</th>
                        <th>Severity</th>
                    </tr>
                </thead>
                <tbody>
        """.format(total=len(matches))
        
        for i, match in enumerate(matches, 1):
            score = match.get('hybrid_score', match.get('score', 0))
            
            if score >= 0.8:
                severity, badge_class = 'High', 'badge-high'
            elif score >= 0.6:
                severity, badge_class = 'Medium', 'badge-medium'
            elif score >= 0.4:
                severity, badge_class = 'Low', 'badge-low'
            else:
                severity, badge_class = 'None', 'badge-none'
            
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{match.get('source_document', 'Unknown')}</td>
                    <td>{match.get('target_document', 'Unknown')}</td>
                    <td>{score:.2%}</td>
                    <td><span class="badge {badge_class}">{severity}</span></td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    def _generate_statistics(self, report: Report) -> str:
        """Generate detailed statistics section."""
        stats = report.statistics
        
        html = """
        <div class="section">
            <h2>📊 Detailed Statistics</h2>
            <div class="stats-grid">
        """
        
        stat_items = [
            ('📄 Total Documents', stats.get('total_documents', 0)),
            ('🔄 Total Comparisons', stats.get('total_comparisons', 0)),
            ('🎯 Total Matches', stats.get('total_matches', 0)),
            ('📈 Average Similarity', f"{stats.get('avg_similarity', 0):.2%}"),
            ('📈 Median Similarity', f"{stats.get('median_similarity', 0):.2%}"),
            ('📈 Max Similarity', f"{stats.get('max_similarity', 0):.2%}"),
            ('📉 Min Similarity', f"{stats.get('min_similarity', 0):.2%}"),
            ('📊 Std Deviation', f"{stats.get('std_similarity', 0):.2%}"),
            ('🔴 High Severity (≥80%)', stats.get('high_severity_count', 0)),
            ('🟡 Medium Severity (50-80%)', stats.get('medium_severity_count', 0)),
            ('🟢 Low Severity (30-50%)', stats.get('low_severity_count', 0)),
            ('⚪ Very Low (<30%)', stats.get('none_severity_count', 0))
        ]
        
        for label, value in stat_items:
            html += f"""
                <div class="stat-card">
                    <div class="stat-value">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _generate_footer(self, report: Report) -> str:
        """Generate report footer."""
        return f"""
            <div class="footer">
                <p>🔍 Report generated by <strong>Semantic Plagiarism Detector</strong></p>
                <p style="margin-top:4px;">
                    Report ID: {report.id} | 
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
                    Version: 1.0.0
                </p>
                <p style="margin-top:8px;font-size:11px;color:#cbd5e1;">
                    This report is for internal use only. Contains confidential analysis results.
                </p>
            </div>
        </body>
        </html>
        """