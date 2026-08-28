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
Tests for Plagiarism Report Generator.

Comprehensive test suite covering report generation, export,
and visualization functionality.
"""

import json
import os
import tempfile

from src.core.report_generator import (
    PlagiarismReport,
    ReportConfig,
    ReportFormat,
    ReportGenerator,
    ReportSection,
    ReportType,
)


class TestReportConfig:
    """Tests for report configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReportConfig()
        assert config.include_visualizations is True
        assert config.include_raw_data is False
        assert config.include_recommendations is True
        assert config.max_matches_displayed == 50

    def test_custom_config(self):
        """Test custom configuration."""
        config = ReportConfig(
            company_name="Test University",
            include_raw_data=True,
            max_matches_displayed=25,
        )
        assert config.company_name == "Test University"
        assert config.include_raw_data is True
        assert config.max_matches_displayed == 25


class TestReportGenerator:
    """Tests for report generator."""

    def setup_method(self):
        self.config = ReportConfig(include_visualizations=False, include_raw_data=False)
        self.generator = ReportGenerator(self.config)
        self.sample_results = {
            "total_documents": 4,
            "matches": [
                {"doc_a": "a.pdf", "doc_b": "b.pdf", "similarity": 0.92},
                {"doc_a": "a.pdf", "doc_b": "c.pdf", "similarity": 0.75},
                {"doc_a": "b.pdf", "doc_b": "d.pdf", "similarity": 0.45},
            ],
            "flagged": [
                {"doc_a": "a.pdf", "doc_b": "b.pdf", "similarity": 0.92},
            ],
        }

    def test_generate_detailed_report(self):
        """Test detailed report generation."""
        report = self.generator.generate_report(
            self.sample_results, ReportType.DETAILED
        )
        assert isinstance(report, PlagiarismReport)
        assert report.report_type == ReportType.DETAILED
        assert report.summary["total_documents"] == 4
        assert report.summary["total_matches"] == 3
        assert len(report.sections) >= 3

    def test_generate_summary_report(self):
        """Test summary report generation."""
        report = self.generator.generate_report(self.sample_results, ReportType.SUMMARY)
        assert report.report_type == ReportType.SUMMARY
        assert len(report.sections) >= 1

    def test_generate_executive_report(self):
        """Test executive report generation."""
        report = self.generator.generate_report(
            self.sample_results, ReportType.EXECUTIVE
        )
        assert report.report_type == ReportType.EXECUTIVE
        assert len(report.recommendations) > 0

    def test_custom_title(self):
        """Test custom report title."""
        report = self.generator.generate_report(
            self.sample_results, title="Custom Title"
        )
        assert report.title == "Custom Title"

    def test_report_has_metadata(self):
        """Test report includes metadata."""
        report = self.generator.generate_report(self.sample_results)
        assert "config" in report.metadata
        assert "detection_engine" in report.metadata

    def test_empty_results(self):
        """Test report with empty results."""
        report = self.generator.generate_report({})
        assert report.summary["total_documents"] == 0
        assert report.summary["total_matches"] == 0


class TestReportSummary:
    """Tests for report summary generation."""

    def setup_method(self):
        self.generator = ReportGenerator()

    def test_summary_statistics(self):
        """Test summary contains correct statistics."""
        results = {
            "total_documents": 10,
            "matches": [
                {"doc_a": "a.pdf", "doc_b": "b.pdf", "similarity": 0.95},
                {"doc_a": "c.pdf", "doc_b": "d.pdf", "similarity": 0.60},
            ],
        }
        report = self.generator.generate_report(results)
        summary = report.summary
        assert summary["total_documents"] == 10
        assert summary["total_matches"] == 2
        assert summary["severity_distribution"]["critical"] == 1
        assert summary["severity_distribution"]["moderate"] == 1


class TestReportExport:
    """Tests for report export."""

    def setup_method(self):
        self.generator = ReportGenerator()
        self.tmp_dir = tempfile.mkdtemp()

    def test_export_json(self):
        """Test JSON export."""
        report = self.generator.generate_report({"total_documents": 2, "matches": []})
        path = os.path.join(self.tmp_dir, "test_report.json")
        self.generator.export_report(report, ReportFormat.JSON, path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["report_type"] == "summary"

    def test_export_markdown(self):
        """Test Markdown export."""
        report = self.generator.generate_report({"total_documents": 2, "matches": []})
        path = os.path.join(self.tmp_dir, "test_report.md")
        self.generator.export_report(report, ReportFormat.MARKDOWN, path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "Report" in content or "report" in content

    def test_export_text(self):
        """Test text export."""
        report = self.generator.generate_report({"total_documents": 2, "matches": []})
        path = os.path.join(self.tmp_dir, "test_report.txt")
        self.generator.export_report(report, ReportFormat.TEXT, path)
        assert os.path.exists(path)

    def test_export_html(self):
        """Test HTML export."""
        report = self.generator.generate_report({"total_documents": 2, "matches": []})
        path = os.path.join(self.tmp_dir, "test_report.html")
        self.generator.export_report(report, ReportFormat.HTML, path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "<html>" in content


class TestReportSections:
    """Tests for report sections."""

    def setup_method(self):
        self.generator = ReportGenerator()

    def test_overview_section(self):
        """Test overview section content."""
        results = {"total_documents": 5, "matches": [{"similarity": 0.85}]}
        report = self.generator.generate_report(results, ReportType.DETAILED)
        overview = next((s for s in report.sections if s.title == "Overview"), None)
        assert overview is not None
        assert "5" in overview.content

    def test_matches_section(self):
        """Test matches section content."""
        results = {
            "total_documents": 2,
            "matches": [
                {"doc_a": "a.pdf", "doc_b": "b.pdf", "similarity": 0.90},
            ],
        }
        report = self.generator.generate_report(results, ReportType.DETAILED)
        matches_section = next(
            (s for s in report.sections if s.title == "Detected Matches"), None
        )
        assert matches_section is not None

    def test_statistics_section(self):
        """Test statistics section."""
        results = {
            "total_documents": 3,
            "matches": [{"similarity": 0.8}, {"similarity": 0.6}, {"similarity": 0.4}],
        }
        report = self.generator.generate_report(results, ReportType.DETAILED)
        stats_section = next(
            (s for s in report.sections if s.title == "Statistics"), None
        )
        assert stats_section is not None
        assert "mean" in stats_section.data


class TestReportRecommendations:
    """Tests for report recommendations."""

    def setup_method(self):
        self.generator = ReportGenerator()

    def test_high_severity_recommendation(self):
        """Test high severity triggers recommendation."""
        results = {
            "total_documents": 2,
            "matches": [{"similarity": 0.95}, {"similarity": 0.88}],
        }
        report = self.generator.generate_report(results)
        assert any(
            "review" in r.lower() or "immediate" in r.lower()
            for r in report.recommendations
        )

    def test_low_plagiarism_recommendation(self):
        """Test low plagiarism gets positive recommendation."""
        results = {"total_documents": 2, "matches": []}
        report = self.generator.generate_report(results)
        assert any(
            "no immediate action" in r.lower() or "continue" in r.lower()
            for r in report.recommendations
        )


class TestReportSerialization:
    """Tests for report serialization."""

    def test_report_to_dict(self):
        """Test report to_dict."""
        generator = ReportGenerator()
        report = generator.generate_report({"total_documents": 2, "matches": []})
        d = report.to_dict()
        assert "report_id" in d
        assert "sections" in d
        assert isinstance(d["sections"], list)

    def test_section_to_dict(self):
        """Test section to_dict."""
        section = ReportSection(title="Test", content="Content", data={"key": "value"})
        d = section.to_dict()
        assert d["title"] == "Test"
        assert d["data"]["key"] == "value"
