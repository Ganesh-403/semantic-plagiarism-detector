from src.core.export_engine import LMSExportEngine


def test_generate_incident_html_empty():
    """Verify that an empty list of incidents returns None."""
    result = LMSExportEngine.generate_incident_html([])
    assert result is None


def test_generate_incident_html_valid():
    """Verify that a valid list of incidents produces the expected HTML content."""
    incidents = [
        {"doc_a": "essay1.txt", "doc_b": "essay2.txt", "similarity": 0.95},
        {"doc_a": "report_a.pdf", "doc_b": "report_b.pdf", "similarity": 0.85},
        {"doc_a": "doc_x.docx", "doc_b": "doc_y.docx", "similarity": 0.70},
    ]

    html_content = LMSExportEngine.generate_incident_html(incidents)
    assert html_content is not None
    assert isinstance(html_content, str)

    # Verify key structure elements
    assert "<!DOCTYPE html>" in html_content
    assert "Plagiarism Incident Report" in html_content
    assert "Total flagged pairs: 3" in html_content

    # Verify doc names are present
    assert "essay1.txt" in html_content
    assert "report_b.pdf" in html_content
    assert "doc_x.docx" in html_content

    # Verify similarity percentages
    assert "95.0%" in html_content
    assert "85.0%" in html_content
    assert "70.0%" in html_content

    # Verify severity ranks and styling colors are present
    assert "CRITICAL" in html_content
    assert "HIGH" in html_content
    assert "MODERATE" in html_content
    assert "#ff4b4b" in html_content  # CRITICAL color
    assert "#ffa500" in html_content  # HIGH color
    assert "#21c55d" in html_content  # MODERATE color
import csv
import io
import json

from src.core.export_engine import LMSExportEngine


def test_generate_incident_csv_empty():
    """Test that an empty list returns None to prevent empty file downloads."""
    result = LMSExportEngine.generate_incident_csv([])
    assert result is None


def test_generate_incident_csv_valid_data():
    """Test that valid incident data is correctly formatted into a CSV string."""
    incidents = [
        {"doc_a": "student1_hw.pdf", "doc_b": "student2_hw.pdf", "similarity": 0.95},
        {"doc_a": "test_doc.docx", "doc_b": "reference.txt", "similarity": 0.82},
        {"doc_a": "essay1.txt", "doc_b": "essay2.txt", "similarity": 0.75},
    ]

    csv_string = LMSExportEngine.generate_incident_csv(incidents)

    assert csv_string is not None
    assert isinstance(csv_string, str)

    # Parse back the CSV to verify integrity
    reader = csv.DictReader(io.StringIO(csv_string))
    rows = list(reader)

    assert len(rows) == 3

    # Check Header
    assert reader.fieldnames == [
        "Document A",
        "Document B",
        "Similarity Score",
        "Severity Flag",
    ]

    # Check Row 1 (Critical Severity)
    assert rows[0]["Document A"] == "student1_hw.pdf"
    assert rows[0]["Similarity Score"] == "0.9500"
    assert rows[0]["Severity Flag"] == "CRITICAL"

    # Check Row 2 (High Severity)
    assert rows[1]["Severity Flag"] == "HIGH"

    # Check Row 3 (Moderate Severity)
    assert rows[2]["Severity Flag"] == "MODERATE"


def test_generate_incident_csv_missing_keys():
    """Test robustness against missing dictionary keys."""
    incidents = [
        {"doc_a": "missing_b.pdf", "similarity": 0.99},  # Missing doc_b
        {"doc_a": "doc1", "doc_b": "doc2"},  # Missing similarity
    ]

    csv_string = LMSExportEngine.generate_incident_csv(incidents)
    assert csv_string is not None

    reader = csv.DictReader(io.StringIO(csv_string))
    rows = list(reader)

    assert rows[0]["Document B"] == "Unknown"
    assert rows[0]["Similarity Score"] == "0.9900"

    assert rows[1]["Similarity Score"] == "0.0000"
    assert rows[1]["Severity Flag"] == "MODERATE"


def test_generate_incident_json_empty():
    """Test that an empty list returns None for JSON export."""
    result = LMSExportEngine.generate_incident_json([])
    assert result is None


def test_generate_incident_json_valid_data():
    """Test that valid incident data is correctly formatted into a JSON string."""
    incidents = [{"doc_a": "alpha.pdf", "doc_b": "beta.pdf", "similarity": 0.91}]

    json_string = LMSExportEngine.generate_incident_json(incidents)
    assert json_string is not None

    payload = json.loads(json_string)

    assert "metadata" in payload
    assert payload["metadata"]["total_incidents"] == 1
    assert payload["metadata"]["export_format"] == "LMS_JSON_v1"

    assert len(payload["incidents"]) == 1
    assert payload["incidents"][0]["document_a"] == "alpha.pdf"
    assert payload["incidents"][0]["similarity_score"] == 0.91
    assert payload["incidents"][0]["severity_flag"] == "CRITICAL"


def test_build_download_response_sets_safe_headers_for_csv():
    payload, headers = LMSExportEngine.build_download_response(
        "a,b\n1,2\n",
        filename="report.csv",
        content_type="text/csv",
    )

    assert payload == b"a,b\n1,2\n"
    assert headers["Content-Type"] == "text/csv; charset=utf-8"
    assert headers["X-Content-Type-Options"] == "nosniff"


def test_build_download_response_falls_back_for_unexpected_content_type():
    _, headers = LMSExportEngine.build_download_response(
        "hello",
        filename="report.bin",
        content_type="text/html",
    )

    assert headers["Content-Type"] == "application/octet-stream"
    assert headers["X-Content-Type-Options"] == "nosniff"
