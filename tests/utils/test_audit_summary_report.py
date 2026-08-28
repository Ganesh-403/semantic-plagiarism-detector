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
test_audit_summary_report.py
-----------------------------
Unit tests for class-wide Plagiarism Audit Summary Report generation (PDF and HTML formats).
"""

from __future__ import annotations

from io import BytesIO

import pytest

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from src.utils.pdf_report import (
    generate_audit_summary_html,
    generate_audit_summary_pdf,
    generate_audit_summary_report,
    generate_batch_plagiarism_report,
)


def test_generate_batch_plagiarism_report():
    incidents = [
        {
            "incident_id": "INC-001",
            "document_a": "alice.pdf",
            "document_b": "bob.pdf",
            "similarity_score": 0.92,
            "severity_rank": "High",
        },
        {
            "incident_id": "INC-002",
            "document_a": "charlie.pdf",
            "document_b": "david.pdf",
            "similarity_score": 0.81,
            "severity_rank": "Medium",
        },
    ]

    pdf_buffer = generate_batch_plagiarism_report(incidents)  # noqa: F821

    assert isinstance(pdf_buffer, BytesIO)

    pdf_bytes = pdf_buffer.getvalue()
    assert pdf_bytes.startswith(b"%PDF")

    reader = PdfReader(BytesIO(pdf_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert "Batch Plagiarism Investigation Report" in text
    assert "alice.pdf" in text
    assert "bob.pdf" in text
    assert "charlie.pdf" in text
    assert "david.pdf" in text
    assert "92.0%" in text
    assert "81.0%" in text


@pytest.fixture
def sample_audit_data():
    metrics = {
        "total_documents": 25,
        "evaluated_pairs": 300,
        "flagged_incidents": 4,
        "high_severity_count": 2,
        "medium_severity_count": 1,
        "low_severity_count": 1,
        "threshold": 0.59,
        "class_section": "CS-101 Data Structures",
    }
    top_flagged_pairs = [
        {
            "doc_a": "assignment_alice.pdf",
            "doc_b": "assignment_bob.pdf",
            "similarity": 0.945,
        },
        {
            "doc_a": "essay_charlie.docx",
            "doc_b": "essay_david.docx",
            "similarity": 0.912,
        },
        {
            "doc_a": "lab_eve.txt",
            "doc_b": "lab_frank.txt",
            "similarity": 0.785,
        },
        {
            "doc_a": "report_grace.pdf",
            "doc_b": "report_heidi.pdf",
            "similarity": 0.620,
        },
    ]
    return metrics, top_flagged_pairs


def test_generate_audit_summary_html_structure(sample_audit_data):
    metrics, top_pairs = sample_audit_data
    html_output = generate_audit_summary_html(
        metrics=metrics,
        top_flagged_pairs=top_pairs,
        class_section="CS-101 Data Structures",
    )

    assert isinstance(html_output, str)
    assert "<!DOCTYPE html>" in html_output
    assert "Plagiarism Audit Executive Summary" in html_output
    assert "CS-101 Data Structures" in html_output
    assert "25" in html_output  # total docs
    assert "300" in html_output  # total pairs
    assert "assignment_alice.pdf" in html_output
    assert "assignment_bob.pdf" in html_output
    assert "High (≥90%)" in html_output
    assert "Medium (75-89%)" in html_output


def test_generate_audit_summary_html_empty_pairs():
    metrics = {
        "total_documents": 5,
        "evaluated_pairs": 10,
        "flagged_incidents": 0,
        "threshold": 0.60,
    }
    html_output = generate_audit_summary_html(
        metrics=metrics,
        top_flagged_pairs=[],
        class_section="Empty Section",
    )

    assert "No flagged plagiarism pairs found" in html_output
    assert "Empty Section" in html_output


def test_generate_audit_summary_pdf_valid_header(sample_audit_data):
    metrics, top_pairs = sample_audit_data
    pdf_buffer = generate_audit_summary_pdf(
        metrics=metrics,
        top_flagged_pairs=top_pairs,
        class_section="CS-101 Data Structures",
    )

    assert isinstance(pdf_buffer, BytesIO)
    pdf_bytes = pdf_buffer.getvalue()
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500

    reader = PdfReader(BytesIO(pdf_bytes))
    extracted_text = "\n".join([p.extract_text() or "" for p in reader.pages])

    assert "Class Plagiarism Audit Summary Report" in extracted_text
    assert "CS-101 Data Structures" in extracted_text
    assert "assignment_alice.pdf" in extracted_text
    assert "assignment_bob.pdf" in extracted_text


def test_generate_audit_summary_report_helper(sample_audit_data):
    metrics, top_pairs = sample_audit_data

    # Test PDF format output
    pdf_res = generate_audit_summary_report(
        metrics=metrics,
        top_flagged_pairs=top_pairs,
        output_format="pdf",
        class_section="Section A",
    )
    assert isinstance(pdf_res, bytes)
    assert pdf_res.startswith(b"%PDF")

    # Test HTML format output
    html_res = generate_audit_summary_report(
        metrics=metrics,
        top_flagged_pairs=top_pairs,
        output_format="html",
        class_section="Section A",
    )
    assert isinstance(html_res, str)
    assert "Plagiarism Audit Executive Summary" in html_res
