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
tests/core/test_export_min_match_length.py
------------------------------------------
Tests for the min_match_length filter in the export engine (Issue #2474).
Verifies that exports only contain items meeting the length threshold.
"""

from src.core.export_engine import LMSExportEngine
from src.utils.html_report import generate_html_report

INCIDENTS = [
    {
        "doc_a": "doc1.pdf",
        "doc_b": "doc2.pdf",
        "similarity": 0.95,
        "matched_length": 50,
        "matched_text": "The quick brown fox jumps over the lazy dog.",
    },
    {
        "doc_a": "doc3.pdf",
        "doc_b": "doc4.pdf",
        "similarity": 0.80,
        "matched_length": 5,
        "matched_text": "Hello world.",
    },
    {
        "doc_a": "doc5.pdf",
        "doc_b": "doc6.pdf",
        "similarity": 0.70,
        "matched_length": 20,
        "matched_text": "A short match here.",
    },
]


class TestExportMinMatchLength:
    """Tests verifying exports only contain items meeting the length threshold."""

    def test_generate_incident_txt_without_filter(self):
        """Without the filter, all incidents are exported."""
        report = LMSExportEngine.generate_incident_txt(INCIDENTS)
        assert report is not None
        assert "doc1.pdf" in report
        assert "doc3.pdf" in report
        assert "doc5.pdf" in report

    def test_generate_incident_txt_with_min_match_length(self):
        """With min_match_length=20, only incidents with >= 20 matched words are exported."""
        report = LMSExportEngine.generate_incident_txt(INCIDENTS, min_match_length=20)
        assert report is not None
        assert "doc1.pdf" in report
        assert "doc3.pdf" not in report
        assert "doc5.pdf" in report

    def test_generate_incident_txt_with_high_threshold(self):
        """With a threshold higher than all incidents, the report is None."""
        report = LMSExportEngine.generate_incident_txt(INCIDENTS, min_match_length=100)
        assert report is None

    def test_generate_html_report_without_filter(self):
        """Without the filter, all incidents are in the HTML."""
        html = generate_html_report(INCIDENTS)
        assert "doc1.pdf" in html
        assert "doc3.pdf" in html
        assert "doc5.pdf" in html

    def test_generate_html_report_with_min_match_length(self):
        """With min_match_length=20, only incidents with >= 20 matched words are in the HTML."""
        html = generate_html_report(INCIDENTS, min_match_length=20)
        assert "doc1.pdf" in html
        assert "doc3.pdf" not in html
        assert "doc5.pdf" in html

    def test_generate_html_report_with_high_threshold(self):
        """With a threshold higher than all incidents, the empty message is returned."""
        html = generate_html_report(INCIDENTS, min_match_length=100)
        assert "No plagiarism incidents to report." in html

    def test_generate_incident_html_via_export_engine(self):
        """The ExportEngine's HTML generator respects the filter."""
        html = LMSExportEngine.generate_incident_html(INCIDENTS, min_match_length=20)
        assert html is not None
        assert "doc1.pdf" in html
        assert "doc3.pdf" not in html
        assert "doc5.pdf" in html
