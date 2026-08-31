"""
PDF report generator for the Report Generation Module
Creates professional PDF reports with tables, charts, and formatting.
"""

import io
import base64
from typing import Dict, Any, List, Optional
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter, landscape
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether, NextPageTemplate,
    Frame, PageTemplate, BaseDocTemplate
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.models.report import Report, ReportConfig
from .visualizations import ReportVisualizer


class PDFGenerator:
    """
    Generates PDF reports with embedded visualizations and professional formatting.
    """
    
    def __init__(self):
        self.visualizer = ReportVisualizer()
        self.styles = self._create_styles()
        self.page_width = A4[0]
        self.page_height = A4[1]
    
    def _create_styles(self):
        """Create custom styles for PDF."""
        styles = getSampleStyleSheet()
        
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            alignment=TA_CENTER,
            spaceAfter=30,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='CustomHeading3',
            parent=styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#555'),
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            fontName='Helvetica'
        ))
        
        styles.add(ParagraphStyle(
            name='CustomSmall',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#7f8c8d'),
            fontName='Helvetica'
        ))
        
        styles.add(ParagraphStyle(
            name='CustomTableHeader',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.white,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='CustomTableCell',
            parent=styles['Normal'],
            fontSize=9,
            fontName='Helvetica'
        ))
        
        return styles
    
    def generate(self, report: Report, content: Dict[str, Any]) -> bytes:
        """
        Generate a PDF report.
        
        Args:
            report: Report object
            content: Report content data
        
        Returns:
            PDF bytes
        """
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
            title=report.title,
            author="Semantic Plagiarism Detector"
        )
        
        story = []
        
        # Title page
        story.append(Paragraph(report.title, self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Header info
        header_info = [
            f"Report ID: {report.id}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Type: {report.report_type.value.replace('_', ' ').title()}",
            f"Format: {report.format.value.upper()}"
        ]
        story.append(Paragraph("<br/>".join(header_info), self.styles['CustomSmall']))
        story.append(Spacer(1, 0.3*inch))
        
        if report.description:
            story.append(Paragraph(report.description, self.styles['CustomBody']))
            story.append(Spacer(1, 0.2*inch))
        
        # Executive Summary
        if report.statistics:
            story.append(PageBreak())
            story.append(Paragraph("Executive Summary", self.styles['CustomHeading2']))
            story.append(self._create_summary_table(report))
            story.append(Spacer(1, 0.2*inch))
        
        # Heatmap
        if report.similarity_matrix and report.document_names:
            story.append(PageBreak())
            story.append(Paragraph("Similarity Heatmap", self.styles['CustomHeading2']))
            story.append(Spacer(1, 0.1*inch))
            
            img_data = self.visualizer.create_heatmap(
                report.similarity_matrix,
                report.document_names,
                title="Document Similarity Matrix"
            )
            
            if img_data:
                try:
                    img_bytes = base64.b64decode(img_data)
                    img = Image(io.BytesIO(img_bytes), width=6*inch, height=5*inch)
                    story.append(KeepTogether([img]))
                    story.append(Spacer(1, 0.1*inch))
                    story.append(
                        Paragraph(
                            "Figure 1: Pairwise similarity scores between all documents.",
                            self.styles['CustomSmall']
                        )
                    )
                except Exception as e:
                    story.append(Paragraph("Heatmap generation failed.", self.styles['CustomBody']))
        
        # Matches
        if report.matches:
            story.append(PageBreak())
            story.append(Paragraph("Detected Matches", self.styles['CustomHeading2']))
            story.append(Spacer(1, 0.1*inch))
            
            match_table = self._create_matches_table(report)
            if match_table:
                story.append(KeepTogether([match_table]))
            
            story.append(Spacer(1, 0.2*inch))
        
        # Detailed Statistics
        if report.statistics:
            story.append(PageBreak())
            story.append(Paragraph("Detailed Statistics", self.styles['CustomHeading2']))
            story.append(Spacer(1, 0.1*inch))
            story.append(self._create_stats_table(report))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Report generated by Semantic Plagiarism Detector", self.styles['CustomSmall']))
        story.append(Paragraph(f"Report ID: {report.id}", self.styles['CustomSmall']))
        
        doc.build(story)
        
        return buffer.getvalue()
    
    def _create_summary_table(self, report: Report) -> Table:
        """Create summary statistics table."""
        stats = report.statistics
        
        data = [['Metric', 'Value']]
        items = [
            ('📄 Total Documents', str(stats.get('total_documents', 0))),
            ('🔄 Total Comparisons', str(stats.get('total_comparisons', 0))),
            ('🎯 Total Matches', str(stats.get('total_matches', 0))),
            ('📈 Average Similarity', f"{stats.get('avg_similarity', 0):.2%}"),
            ('📈 Max Similarity', f"{stats.get('max_similarity', 0):.2%}"),
            ('🔴 High Severity', str(stats.get('high_severity_count', 0))),
            ('🟡 Medium Severity', str(stats.get('medium_severity_count', 0)))
        ]
        
        for label, value in items:
            data.append([label, value])
        
        table = Table(data, colWidths=[3.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        
        return table
    
    def _create_stats_table(self, report: Report) -> Table:
        """Create detailed statistics table."""
        stats = report.statistics
        
        data = [['Metric', 'Value']]
        items = [
            ('📄 Total Documents', str(stats.get('total_documents', 0))),
            ('🔄 Total Comparisons', str(stats.get('total_comparisons', 0))),
            ('🎯 Total Matches', str(stats.get('total_matches', 0))),
            ('📈 Average Similarity', f"{stats.get('avg_similarity', 0):.2%}"),
            ('📈 Median Similarity', f"{stats.get('median_similarity', 0):.2%}"),
            ('📈 Max Similarity', f"{stats.get('max_similarity', 0):.2%}"),
            ('📉 Min Similarity', f"{stats.get('min_similarity', 0):.2%}"),
            ('📊 Std Deviation', f"{stats.get('std_similarity', 0):.2%}"),
            ('🔴 High Severity (≥80%)', str(stats.get('high_severity_count', 0))),
            ('🟡 Medium Severity (50-80%)', str(stats.get('medium_severity_count', 0))),
            ('🟢 Low Severity (30-50%)', str(stats.get('low_severity_count', 0))),
            ('⚪ Very Low (<30%)', str(stats.get('none_severity_count', 0)))
        ]
        
        for label, value in items:
            data.append([label, value])
        
        table = Table(data, colWidths=[3.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        
        return table
    
    def _create_matches_table(self, report: Report) -> Optional[Table]:
        """Create matches table."""
        matches = report.matches[:50]
        if not matches:
            return None
        
        data = [['#', 'Source', 'Target', 'Score', 'Severity']]
        
        for i, match in enumerate(matches, 1):
            score = match.get('hybrid_score', match.get('score', 0))
            
            if score >= 0.8:
                severity = '🔴 High'
            elif score >= 0.6:
                severity = '🟡 Medium'
            elif score >= 0.4:
                severity = '🟢 Low'
            else:
                severity = '⚪ None'
            
            source = match.get('source_document', 'Unknown')[:20]
            target = match.get('target_document', 'Unknown')[:20]
            
            data.append([
                str(i),
                source,
                target,
                f"{score:.2%}",
                severity
            ])
        
        table = Table(data, colWidths=[0.5*inch, 2*inch, 2*inch, 0.8*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
        ]))
        
        return table