"""
Main report generator for the Report Generation Module
Orchestrates the creation of reports in multiple formats.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import uuid

from src.models.report import (
    Report, ReportConfig, ReportRequest, ReportResponse,
    ReportSection, ReportStatus, ReportStatistics, ReportFormat
)
from src.reports.html_generator import HTMLGenerator
from src.reports.pdf_generator import PDFGenerator
from src.reports.csv_exporter import CSVExporter
from src.reports.visualizations import ReportVisualizer


class ReportGenerator:
    """
    Main report generator that orchestrates the creation of reports.
    Supports HTML, PDF, CSV, and JSON formats.
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.html_generator = HTMLGenerator()
        self.pdf_generator = PDFGenerator()
        self.csv_exporter = CSVExporter()
        self.visualizer = ReportVisualizer()
        
        self._reports: Dict[str, Report] = {}
        self._generation_stats: Dict[str, Any] = {
            'total_generated': 0,
            'successful': 0,
            'failed': 0,
            'average_time_ms': 0
        }
    
    def generate_report(self, request: ReportRequest) -> ReportResponse:
        """
        Generate a report based on the request.
        
        Args:
            request: Report request
        
        Returns:
            ReportResponse
        """
        try:
            start_time = time.time()
            
            # Create report
            report = self._create_report_from_request(request)
            report.status = ReportStatus.GENERATING
            report.update_progress(10)
            
            # Build report data
            report.update_progress(20)
            data = self._build_report_data(request)
            report.update_progress(40)
            
            # Generate visualizations
            report.update_progress(50)
            visualizations = self._generate_visualizations(data, request.config)
            report.update_progress(60)
            
            # Generate content
            report.update_progress(70)
            content = self._generate_content(data, visualizations, request.config)
            report.update_progress(80)
            
            # Export
            report.update_progress(85)
            file_path, file_size = self._export_report(report, content, request.format)
            report.update_progress(90)
            
            # Finalize
            report.mark_completed(file_path, file_size)
            report.statistics = data.get('statistics', {})
            report.document_names = data.get('document_names', [])
            report.similarity_matrix = data.get('similarity_matrix', [])
            report.matches = data.get('matches', [])
            report.update_progress(100)
            
            self._reports[report.id] = report
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update stats
            self._generation_stats['total_generated'] += 1
            self._generation_stats['successful'] += 1
            
            return ReportResponse(
                success=True,
                message=f"Report generated successfully in {processing_time:.0f}ms",
                report_id=report.id,
                file_path=file_path,
                file_size=file_size,
                download_url=f"/api/reports/download/{report.id}",
                format=request.format.value,
                record_count=len(data.get('matches', [])),
                report=report
            )
            
        except Exception as e:
            report = self._reports.get(request.analysis_id)
            if report:
                report.mark_failed(str(e))
            
            self._generation_stats['total_generated'] += 1
            self._generation_stats['failed'] += 1
            
            return ReportResponse(
                success=False,
                message=f"Report generation failed: {str(e)}",
                error=str(e)
            )
    
    def _create_report_from_request(self, request: ReportRequest) -> Report:
        """Create a report object from request."""
        title = request.title or f"Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        report = Report(
            title=title,
            description=request.description or "Plagiarism detection analysis report",
            report_type=request.report_type,
            format=request.format,
            metadata={
                'analysis_id': request.analysis_id,
                'document_ids': request.document_ids,
                'include_details': request.include_details,
                'created_at': datetime.now().isoformat(),
                'config': request.config.to_dict() if request.config else None
            }
        )
        
        self._reports[report.id] = report
        return report
    
    def _build_report_data(self, request: ReportRequest) -> Dict[str, Any]:
        """
        Build the report data from analysis results.
        
        This is a placeholder - in production, this would fetch data from
        the analysis module.
        """
        # Sample data structure
        data = {
            'document_names': ['Doc1', 'Doc2', 'Doc3', 'Doc4'],
            'similarity_matrix': [
                [1.0, 0.85, 0.32, 0.12],
                [0.85, 1.0, 0.28, 0.15],
                [0.32, 0.28, 1.0, 0.92],
                [0.12, 0.15, 0.92, 1.0]
            ],
            'matches': [
                {
                    'source_document': 'Doc1',
                    'target_document': 'Doc2',
                    'lexical_score': 0.82,
                    'semantic_score': 0.88,
                    'hybrid_score': 0.85,
                    'severity': 'high'
                },
                {
                    'source_document': 'Doc3',
                    'target_document': 'Doc4',
                    'lexical_score': 0.90,
                    'semantic_score': 0.94,
                    'hybrid_score': 0.92,
                    'severity': 'high'
                },
                {
                    'source_document': 'Doc1',
                    'target_document': 'Doc3',
                    'lexical_score': 0.30,
                    'semantic_score': 0.34,
                    'hybrid_score': 0.32,
                    'severity': 'low'
                }
            ],
            'statistics': {
                'total_documents': 4,
                'total_comparisons': 6,
                'total_matches': 3,
                'average_similarity': 0.523,
                'median_similarity': 0.48,
                'max_similarity': 0.92,
                'min_similarity': 0.12,
                'std_similarity': 0.28,
                'high_severity_count': 2,
                'medium_severity_count': 0,
                'low_severity_count': 1,
                'none_severity_count': 3,
                'avg_processing_time_ms': 150.5
            }
        }
        
        return data
    
    def _generate_visualizations(
        self,
        data: Dict[str, Any],
        config: ReportConfig
    ) -> Dict[str, str]:
        """Generate visualizations for the report."""
        visualizations = {}
        
        if config.include_heatmap and data.get('similarity_matrix'):
            visualizations['heatmap'] = self.visualizer.create_heatmap(
                data['similarity_matrix'],
                data['document_names'],
                title="Document Similarity Matrix"
            )
        
        if config.include_matches and data.get('matches'):
            scores = [m.get('hybrid_score', m.get('score', 0)) for m in data['matches']]
            labels = [f"{m.get('source_document', '')} → {m.get('target_document', '')}" for m in data['matches']]
            
            visualizations['chart'] = self.visualizer.create_similarity_chart(
                scores,
                labels,
                title="Similarity Scores",
                threshold=config.highlight_threshold
            )
            
            visualizations['distribution'] = self.visualizer.create_distribution_chart(
                scores,
                title="Score Distribution"
            )
            
            visualizations['severity'] = self.visualizer.create_severity_distribution(
                data['matches'],
                title="Severity Distribution"
            )
        
        if data.get('statistics'):
            visualizations['summary'] = self.visualizer.create_summary_dashboard(
                data['statistics'],
                data.get('matches', []),
                title="Analysis Summary"
            )
        
        return visualizations
    
    def _generate_content(
        self,
        data: Dict[str, Any],
        visualizations: Dict[str, str],
        config: ReportConfig
    ) -> Dict[str, Any]:
        """Generate report content."""
        content = {
            'sections': [],
            'summary': data.get('statistics', {}),
            'visualizations': visualizations,
            'document_names': data.get('document_names', []),
            'matches': data.get('matches', []),
            'similarity_matrix': data.get('similarity_matrix', [])
        }
        
        # Add summary section
        if config.include_summary_stats:
            content['sections'].append({
                'title': 'Executive Summary',
                'content': self._generate_summary_content(data),
                'data': data.get('statistics', {})
            })
        
        # Add heatmap section
        if config.include_heatmap and 'heatmap' in visualizations:
            content['sections'].append({
                'title': 'Similarity Heatmap',
                'content': 'This heatmap shows the pairwise similarity scores between all documents.',
                'visualization': 'heatmap'
            })
        
        # Add matches section
        if config.include_matches and data.get('matches'):
            content['sections'].append({
                'title': 'Detected Matches',
                'content': f"Found {len(data['matches'])} matches across documents.",
                'matches': data['matches'][:config.max_matches]
            })
        
        # Add statistics section
        if config.include_summary_stats:
            content['sections'].append({
                'title': 'Detailed Statistics',
                'content': self._generate_statistics_table(data),
                'data': data.get('statistics', {})
            })
        
        return content
    
    def _generate_summary_content(self, data: Dict[str, Any]) -> str:
        """Generate summary content."""
        stats = data.get('statistics', {})
        
        return f"""
        <div class="summary-content">
            <p>This report analyzes <strong>{stats.get('total_documents', 0)}</strong> documents with 
            <strong>{stats.get('total_comparisons', 0)}</strong> pairwise comparisons.</p>
            
            <p>The average similarity score is <strong>{stats.get('average_similarity', 0):.2%}</strong>, 
            with a maximum of <strong>{stats.get('max_similarity', 0):.2%}</strong>.</p>
            
            <p>Found <strong>{stats.get('high_severity_count', 0)}</strong> high-severity matches 
            and <strong>{stats.get('medium_severity_count', 0)}</strong> medium-severity matches.</p>
            
            <div class="alert alert-info">
                <strong>💡 Key Insight:</strong> 
                {self._generate_insight(stats)}
            </div>
        </div>
        """
    
    def _generate_insight(self, stats: Dict[str, Any]) -> str:
        """Generate key insight from statistics."""
        high = stats.get('high_severity_count', 0)
        total = stats.get('total_matches', 0)
        
        if high > 0:
            if high / total > 0.5:
                return "More than half of the matches are high severity. Immediate review recommended."
            else:
                return f"{high} high-severity matches detected. These require priority review."
        elif stats.get('average_similarity', 0) > 0.3:
            return "Moderate similarity detected. Review matches to ensure proper attribution."
        else:
            return "Low overall similarity detected. No urgent action required."
    
    def _generate_statistics_table(self, data: Dict[str, Any]) -> str:
        """Generate statistics table."""
        stats = data.get('statistics', {})
        
        rows = [
            ('📄 Total Documents', str(stats.get('total_documents', 0))),
            ('🔄 Total Comparisons', str(stats.get('total_comparisons', 0))),
            ('🎯 Total Matches', str(stats.get('total_matches', 0))),
            ('📈 Average Similarity', f"{stats.get('average_similarity', 0):.2%}"),
            ('📈 Median Similarity', f"{stats.get('median_similarity', 0):.2%}"),
            ('📈 Max Similarity', f"{stats.get('max_similarity', 0):.2%}"),
            ('📉 Min Similarity', f"{stats.get('min_similarity', 0):.2%}"),
            ('📊 Std Deviation', f"{stats.get('std_similarity', 0):.2%}"),
            ('🔴 High Severity (≥80%)', str(stats.get('high_severity_count', 0))),
            ('🟡 Medium Severity (50-80%)', str(stats.get('medium_severity_count', 0))),
            ('🟢 Low Severity (30-50%)', str(stats.get('low_severity_count', 0))),
            ('⚪ Very Low (<30%)', str(stats.get('none_severity_count', 0))),
            ('⏱️ Processing Time', f"{stats.get('avg_processing_time_ms', 0):.0f} ms")
        ]
        
        html = '<table class="stats-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>'
        for label, value in rows:
            html += f'<tr><td>{label}</td><td><strong>{value}</strong></td></tr>'
        html += '</tbody></table>'
        
        return html
    
    def _export_report(
        self,
        report: Report,
        content: Dict[str, Any],
        format: ReportFormat
    ) -> tuple:
        """Export the report to the specified format."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{report.id}_{timestamp}"
        
        if format == ReportFormat.HTML:
            file_path = self.output_dir / f"{filename}.html"
            html_content = self.html_generator.generate(report, content)
            file_path.write_text(html_content, encoding='utf-8')
            file_size = file_path.stat().st_size
        
        elif format == ReportFormat.PDF:
            file_path = self.output_dir / f"{filename}.pdf"
            pdf_bytes = self.pdf_generator.generate(report, content)
            file_path.write_bytes(pdf_bytes)
            file_size = file_path.stat().st_size
        
        elif format == ReportFormat.CSV:
            file_path = self.output_dir / f"{filename}.csv"
            csv_content = self.csv_exporter.export(report, content)
            file_path.write_text(csv_content, encoding='utf-8')
            file_size = file_path.stat().st_size
        
        elif format == ReportFormat.JSON:
            file_path = self.output_dir / f"{filename}.json"
            json_content = json.dumps(report.to_dict(), indent=2, default=str)
            file_path.write_text(json_content, encoding='utf-8')
            file_size = file_path.stat().st_size
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(file_path), file_size
    
    def get_report(self, report_id: str) -> Optional[Report]:
        """Get a report by ID."""
        return self._reports.get(report_id)
    
    def list_reports(self, limit: int = 50, offset: int = 0) -> List[Report]:
        """List all generated reports."""
        reports = list(self._reports.values())
        reports.sort(key=lambda r: r.generated_at, reverse=True)
        return reports[offset:offset + limit]
    
    def delete_report(self, report_id: str) -> bool:
        """Delete a report and its file."""
        report = self._reports.get(report_id)
        if not report:
            return False
        
        # Delete file if exists
        if report.file_path and Path(report.file_path).exists():
            try:
                Path(report.file_path).unlink()
            except:
                pass
        
        del self._reports[report_id]
        return True
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get report generation statistics."""
        return {
            **self._generation_stats,
            'reports_in_memory': len(self._reports),
            'success_rate': (
                self._generation_stats['successful'] / self._generation_stats['total_generated'] * 100
                if self._generation_stats['total_generated'] > 0 else 0
            )
        }
    
    def cleanup_old_reports(self, days: int = 30) -> int:
        """Clean up reports older than specified days."""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        to_delete = []
        
        for report_id, report in self._reports.items():
            if report.generated_at.timestamp() < cutoff:
                to_delete.append(report_id)
        
        for report_id in to_delete:
            self.delete_report(report_id)
        
        return len(to_delete)