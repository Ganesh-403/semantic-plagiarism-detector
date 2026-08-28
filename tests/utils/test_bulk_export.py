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

import csv
import io
import json
import zipfile
from unittest.mock import MagicMock, patch  # noqa: F401

import pytest

from src.utils.bulk_export import (
    ExportFormat,
    export_incidents_csv,
    export_incidents_csv_stream,
    export_incidents_json_stream,
    export_incidents_to_format,
    generate_bulk_reports_zip,
    normalize_csv_headers,
    sanitize_csv_cell_value,
    sanitize_export_filename,
    stream_incidents_csv_chunks,
)


def test_generate_bulk_reports_zip():
    # Flags matching the bulk_export expected schema
    flags = [
        {
            "doc_a": "Alice.pdf",
            "doc_b": "Bob.docx",
            "similarity": 0.85,
            "threshold_at_time_of_flag": 0.5,
        },
        {
            "doc_a": "Charlie.txt",
            "doc_b": "Dave.pdf",
            "similarity": 0.95,
            "threshold_at_time_of_flag": 0.5,
        },
    ]

    # Use default arguments (include all artifact types)
    zip_bytes = generate_bulk_reports_zip(flags)
    assert isinstance(zip_bytes, bytes)

    # Inspect the zip archive
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = zf.namelist()
        # Expect two PDF reports, a summary CSV, and a metadata JSON file
        pdf_names = [n for n in names if n.lower().endswith(".pdf")]
        assert len(pdf_names) == 2

        # New export artifacts
        assert "summary.csv" in names
        assert "metadata.json" in names

        # Verify metadata JSON content
        meta_content = zf.read("metadata.json").decode("utf-8")
        meta = json.loads(meta_content)

        assert "generated_at" in meta
        assert "flags" in meta
        assert len(meta["flags"]) == 2

        input_set = {(f["doc_a"], f["doc_b"]) for f in flags}
        meta_set = {(f["doc_a"], f["doc_b"]) for f in meta["flags"]}
        assert input_set == meta_set


# ---------------------------------------------------------------------------
# Tests for export_incidents_csv_stream (Issue #942)
# ---------------------------------------------------------------------------

_SAMPLE_INCIDENTS = [
    {
        "incident_id": "INC-001",
        "document_a": "alice.pdf",
        "document_b": "bob.pdf",
        "similarity_score": 0.95,
        "severity_rank": "High",
        "review_status": "Pending",
        "date_flagged": "2024-01-15T10:00:00+00:00",
    },
    {
        "incident_id": "INC-002",
        "document_a": "charlie.docx",
        "document_b": "dave.docx",
        "similarity_score": 0.72,
        "severity_rank": "Medium",
        "review_status": "Reviewed",
        "date_flagged": "2024-01-16T08:30:00+00:00",
    },
]


def test_export_incidents_csv_stream_returns_bytes():
    """export_incidents_csv_stream must return UTF-8-SIG encoded bytes."""
    csv_bytes = export_incidents_csv_stream(_SAMPLE_INCIDENTS)

    assert isinstance(csv_bytes, bytes)
    # Must start with UTF-8 BOM (EF BB BF)
    assert csv_bytes.startswith(b"\xef\xbb\xbf")


def test_export_incidents_csv_stream_excel_compatibility():
    """Exported CSV must be readable by Excel on Windows (UTF-8-SIG with BOM)."""
    csv_bytes = export_incidents_csv_stream(_SAMPLE_INCIDENTS)

    # Verify UTF-8-SIG encoding (BOM + valid UTF-8)
    text = csv_bytes.decode("utf-8-sig")
    assert "INC-001" in text
    assert "alice.pdf" in text
    assert "95.00%" in text

    # Verify BOM is present at start (not in decoded text)
    assert csv_bytes[:3] == b"\xef\xbb\xbf"


def test_export_incidents_csv_stream_headers():
    """First row must contain all required column headers."""
    expected_headers = [
        "Incident ID",
        "Doc A",
        "Doc B",
        "Similarity",
        "Severity",
        "Status",
        "Date",
    ]

    csv_bytes = export_incidents_csv_stream(_SAMPLE_INCIDENTS)
    # Decode without BOM for CSV parsing
    text = csv_bytes.decode("utf-8-sig")
    first_line = text.splitlines()[0].strip()
    actual_headers = [h.strip() for h in first_line.split(",")]

    assert actual_headers == expected_headers


def test_export_incidents_csv_stream_row_values():
    """CSV rows must reflect incident field values with correct formatting."""
    csv_bytes = export_incidents_csv_stream(_SAMPLE_INCIDENTS)
    text = csv_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    assert len(rows) == 2

    # First row
    assert rows[0]["Incident ID"] == "INC-001"
    assert rows[0]["Doc A"] == "alice.pdf"
    assert rows[0]["Doc B"] == "bob.pdf"
    assert rows[0]["Similarity"] == "95.00%"
    assert rows[0]["Severity"] == "High"
    assert rows[0]["Status"] == "Pending"
    assert rows[0]["Date"] == "2024-01-15T10:00:00+00:00"

    # Second row
    assert rows[1]["Incident ID"] == "INC-002"
    assert rows[1]["Similarity"] == "72.00%"
    assert rows[1]["Severity"] == "Medium"
    assert rows[1]["Status"] == "Reviewed"


def test_export_incidents_csv_stream_empty_list():
    """An empty incidents list should produce only the header row with BOM."""
    csv_bytes = export_incidents_csv_stream([])
    assert csv_bytes.startswith(b"\xef\xbb\xbf")

    text = csv_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    assert rows == []
    # Rewind and confirm only one line (the header) exists
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 1


def test_export_incidents_csv_stream_non_numeric_similarity():
    """Non-numeric similarity_score should be written as-is without raising."""
    incidents = [
        {
            "incident_id": "INC-X",
            "document_a": "a.pdf",
            "document_b": "b.pdf",
            "similarity_score": "N/A",
            "severity_rank": "Low",
            "review_status": "Pending",
            "date_flagged": "2024-01-17",
        }
    ]

    csv_bytes = export_incidents_csv_stream(incidents)
    assert csv_bytes.startswith(b"\xef\xbb\xbf")

    text = csv_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["Similarity"] == "N/A"


# ---------------------------------------------------------------------------
# Tests for normalize_csv_headers (Issue #1253)
# ---------------------------------------------------------------------------


class TestNormalizeCsvHeaders:
    """Tests for the normalize_csv_headers helper (issue #1253)."""

    def test_returns_list_type(self):
        """The function must always return a list."""
        result = normalize_csv_headers(["header1", "header2"])
        assert isinstance(result, list)

    def test_empty_list_returns_empty(self):
        """An empty input list must return an empty list."""
        result = normalize_csv_headers([])
        assert result == []

    def test_none_input_returns_empty(self):
        """None input must return an empty list."""
        result = normalize_csv_headers(None)
        assert result == []

    def test_non_list_input_returns_empty(self):
        """Non-list input must return an empty list."""
        result = normalize_csv_headers("not a list")
        assert result == []
        result = normalize_csv_headers(123)
        assert result == []

    def test_strips_leading_trailing_whitespace(self):
        """Headers must have leading and trailing whitespace stripped."""
        result = normalize_csv_headers(["  Incident ID ", "  Doc A  ", "  Date  "])
        assert result == ["Incident_ID", "Doc_A", "Date"]

    def test_replaces_spaces_with_underscores(self):
        """Spaces within headers must be replaced with underscores."""
        result = normalize_csv_headers(["First Name", "Last Name", "Full Name"])
        assert result == ["First_Name", "Last_Name", "Full_Name"]

    def test_replaces_special_symbols_with_underscores(self):
        """Invalid symbols must be replaced with underscores."""
        result = normalize_csv_headers(["Doc A!", "similarity@score", "test#header"])
        assert result == ["Doc_A", "similarity_score", "test_header"]

    def test_replaces_punctuation_with_underscores(self):
        """Punctuation marks must be replaced with underscores."""
        result = normalize_csv_headers(["header(1)", "test.value", "data:info"])
        assert result == ["header_1", "test_value", "data_info"]

    def test_collapses_consecutive_underscores(self):
        """Multiple consecutive underscores must be collapsed to one."""
        result = normalize_csv_headers(["foo__bar", "test___header", "a____b"])
        assert result == ["foo_bar", "test_header", "a_b"]

    def test_strips_leading_trailing_underscores(self):
        """Leading and trailing underscores must be stripped."""
        result = normalize_csv_headers(["_header_", "__test__", "___data___"])
        assert result == ["header", "test", "data"]

    def test_preserves_valid_underscores_and_hyphens(self):
        """Valid underscores and hyphens in the middle of headers are preserved."""
        result = normalize_csv_headers(["valid_header", "test-header", "my_var_name"])
        assert result == ["valid_header", "test-header", "my_var_name"]

    def test_preserves_alphanumeric_characters(self):
        """Alphanumeric characters must be preserved."""
        result = normalize_csv_headers(["abc123", "Test456", "data789"])
        assert result == ["abc123", "Test456", "data789"]

    def test_empty_header_uses_fallback(self):
        """Empty headers after normalization must use column_N fallback."""
        result = normalize_csv_headers(["", "   ", "valid_header"])
        assert result[0] == "column_0"
        assert result[1] == "column_1"
        assert result[2] == "valid_header"

    def test_symbols_only_header_uses_fallback(self):
        """Headers consisting only of symbols must use column_N fallback."""
        result = normalize_csv_headers(["@#$%", "!!!", "valid"])
        assert result[0] == "column_0"
        assert result[1] == "column_1"
        assert result[2] == "valid"

    def test_complex_mixed_headers(self):
        """Verify normalization with complex mixed headers."""
        headers = [
            "  Incident ID ",
            "Doc A!!!",
            "similarity@score##",
            "  Review Status  ",
            "date___flagged",
            "",
            "test@#$%header",
        ]
        result = normalize_csv_headers(headers)
        assert result == [
            "Incident_ID",
            "Doc_A",
            "similarity_score",
            "Review_Status",
            "date_flagged",
            "column_5",
            "test_header",
        ]

    def test_non_string_headers_converted(self):
        """Non-string headers must be converted to strings before normalization."""
        result = normalize_csv_headers([123, 45.6, True, None])
        assert result == ["123", "45_6", "True", "column_3"]

    def test_unicode_characters_preserved(self):
        """Unicode characters must be preserved in normalized headers."""
        result = normalize_csv_headers(["caf\u00e9", "na\u00efve", "r\u00e9sum\u00e9"])
        assert result == ["caf\u00e9", "na\u00efve", "r\u00e9sum\u00e9"]

    def test_single_header(self):
        """A single header must be normalized correctly."""
        result = normalize_csv_headers(["  Test Header!  "])
        assert result == ["Test_Header"]

    def test_preserves_original_list(self):
        """The original input list must not be modified."""
        original = ["  Header 1 ", "Header 2!"]
        original_copy = original.copy()
        normalize_csv_headers(original)
        assert original == original_copy

    def test_all_special_characters(self):
        """Headers with all common special characters must be normalized."""
        headers = [
            "test!@#$%^&*()header",
            "data[]{}|\\;:'\"<>?,./value",
            "foo~`+=bar",
        ]
        result = normalize_csv_headers(headers)
        # All special chars become underscores, then collapsed
        for header in result:
            # Each header should only contain alphanumeric, underscore, or hyphen
            assert all(c.isalnum() or c in ("_", "-") for c in header)
            # No consecutive underscores
            assert "__" not in header
            # No leading/trailing underscores
            assert not header.startswith("_")
            assert not header.endswith("_")


def test_create_batch_incident_zip_archive():
    """Verify that create_batch_incident_zip_archive generates a ZIP file with CSV, JSON, and PDF reports."""
    from src.utils.bulk_export import create_batch_incident_zip_archive

    incidents = [
        {
            "incident_id": "INC-001",
            "document_a": "alice.pdf",
            "document_b": "bob.pdf",
            "similarity_score": 0.95,
            "severity_rank": "High",
            "review_status": "Pending",
            "date_flagged": "2024-01-15T10:00:00+00:00",
        },
        {
            "incident_id": "INC-002",
            "document_a": "charlie.docx",
            "document_b": "dave.docx",
            "similarity_score": 0.72,
            "severity_rank": "Medium",
            "review_status": "Reviewed",
            "date_flagged": "2024-01-16T08:30:00+00:00",
        },
    ]

    zip_bytes = create_batch_incident_zip_archive(incidents)
    assert isinstance(zip_bytes, bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = zf.namelist()

        # Check for files
        assert "incidents_summary.csv" in names
        assert "metadata.json" in names

        # Check for PDF reports
        pdf_names = [n for n in names if n.lower().endswith(".pdf")]
        assert len(pdf_names) == 2
        assert "report_INC_001_alice_bob.pdf" in pdf_names
        assert "report_INC_002_charlie_dave.pdf" in pdf_names

        # Verify metadata JSON content
        meta_content = zf.read("metadata.json").decode("utf-8")
        meta = json.loads(meta_content)
        assert "generated_at" in meta
        assert meta["total_incidents"] == 2
        assert meta["incidents"][0]["incident_id"] == "INC-001"

        csv_content = zf.read("incidents_summary.csv").decode("utf-8-sig")
        assert "INC-001" in csv_content
        assert "INC-002" in csv_content
        assert "95.00%" in csv_content
        assert "72.00%" in csv_content


def test_bulk_zip_rejects_path_traversal_in_entry_names():
    """Document / incident metadata must not become ../ or absolute ZIP paths."""
    from src.utils.bulk_export import create_batch_incident_zip_archive

    flags = [
        {
            "doc_a": "../../etc/passwd",
            "doc_b": "/tmp/evil.pdf",
            "similarity": 0.9,
            "threshold_at_time_of_flag": 0.5,
        }
    ]
    zip_bytes = generate_bulk_reports_zip(flags, include_pdf=False)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for name in zf.namelist():
            assert ".." not in name
            assert not name.startswith("/")
            assert not name.startswith("\\")

    incidents = [
        {
            "incident_id": "../../evil",
            "document_a": "../secret.txt",
            "document_b": "/abs/path.docx",
            "similarity_score": 0.8,
        }
    ]
    with patch(
        "src.utils.bulk_export.generate_plagiarism_report",
        return_value=io.BytesIO(b"%PDF-1.4"),
    ):
        zip_bytes = create_batch_incident_zip_archive(incidents)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for name in zf.namelist():
            assert ".." not in name
            assert not name.startswith("/")
            assert not name.startswith("\\")


# ---------------------------------------------------------------------------
# Tests for stream_incidents_csv_chunks (Issue #1511)
# ---------------------------------------------------------------------------


def test_stream_incidents_csv_chunks_default_batch():
    """Verify default batch size and incremental query calls."""
    calls = []

    def mock_query(limit, offset):
        calls.append((limit, offset))
        if offset >= 2500:
            return []

        size = min(limit, 2500 - offset)
        return [{"incident_id": f"INC-{offset+i}"} for i in range(size)]

    chunks = list(stream_incidents_csv_chunks(mock_query))

    assert len(chunks) == 4

    header = chunks[0]
    assert "Incident ID" in header
    assert "Doc A" in header

    assert calls == [(1000, 0), (1000, 1000), (1000, 2000)]


def test_stream_incidents_csv_chunks_custom_batch():
    """Verify custom batch size works."""
    calls = []

    def mock_query(limit, offset):
        calls.append((limit, offset))
        if offset >= 10:
            return []
        size = min(limit, 10 - offset)
        return [{"incident_id": f"INC-{offset+i}"} for i in range(size)]

    chunks = list(stream_incidents_csv_chunks(mock_query, batch_size=3))

    assert calls == [(3, 0), (3, 3), (3, 6), (3, 9)]
    assert len(chunks) == 5


def test_stream_incidents_csv_chunks_empty_results():
    def mock_query(limit, offset):
        return []

    chunks = list(stream_incidents_csv_chunks(mock_query))
    assert len(chunks) == 1
    assert "Incident ID" in chunks[0]


def test_stream_incidents_csv_chunks_memory_efficient():
    """Verify that entire dataset is not accumulated in memory."""

    def mock_query(limit, offset):
        if offset >= 50:
            return []
        return [
            {"incident_id": f"INC-{offset+i}"} for i in range(min(limit, 50 - offset))
        ]

    stream = stream_incidents_csv_chunks(mock_query, batch_size=10)

    header = next(stream)
    assert "Incident ID" in header

    first_batch = next(stream)
    assert "INC-0" in first_batch
    assert "INC-9" in first_batch
    assert "INC-10" not in first_batch


def test_stream_incidents_csv_chunks_escaping():
    def mock_query(limit, offset):
        if offset > 0:
            return []
        return [
            {
                "incident_id": "INC-1",
                "document_a": "comma, in name.pdf",
                "document_b": 'quote" in name.pdf',
                "similarity_score": 0.5,
            }
        ]

    chunks = list(stream_incidents_csv_chunks(mock_query))
    assert len(chunks) == 2
    data = chunks[1]
    assert '"comma, in name.pdf"' in data
    assert 'quote"" in name.pdf' in data


# ---------------------------------------------------------------------------
# Tests for the `delimiter` parameter (European Excel compatibility)
# ---------------------------------------------------------------------------


def test_export_incidents_csv_stream_default_delimiter_is_comma():
    """Without an explicit delimiter, output must remain comma-separated
    (backward compatibility with existing callers)."""
    csv_bytes = export_incidents_csv_stream(_SAMPLE_INCIDENTS)
    text = csv_bytes.decode("utf-8-sig")
    first_line = text.splitlines()[0]

    assert "," in first_line
    assert ";" not in first_line


def test_export_incidents_csv_stream_semicolon_delimiter():
    """delimiter=';' must produce semicolon-delimited CSV, parseable back
    into the original field values."""
    csv_bytes = export_incidents_csv_stream(_SAMPLE_INCIDENTS, delimiter=";")
    text = csv_bytes.decode("utf-8-sig")
    first_line = text.splitlines()[0]

    # Header row uses semicolons, not commas, as the field separator
    assert first_line == "Incident ID;Doc A;Doc B;Similarity;Severity;Status;Date"

    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["Incident ID"] == "INC-001"
    assert rows[0]["Doc A"] == "alice.pdf"
    assert rows[0]["Similarity"] == "95.00%"
    assert rows[1]["Incident ID"] == "INC-002"


def test_export_incidents_csv_stream_tab_delimiter():
    """delimiter='\\t' must produce tab-delimited CSV."""
    csv_bytes = export_incidents_csv_stream(_SAMPLE_INCIDENTS, delimiter="\t")
    text = csv_bytes.decode("utf-8-sig")
    first_line = text.splitlines()[0]

    assert "\t" in first_line
    assert "," not in first_line

    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    rows = list(reader)
    assert rows[0]["Doc A"] == "alice.pdf"


def test_stream_incidents_csv_chunks_semicolon_delimiter():
    """stream_incidents_csv_chunks must honor delimiter=';' across both
    the header chunk and the data chunks."""

    def mock_query(limit, offset):
        if offset > 0:
            return []
        return [
            {
                "incident_id": "INC-1",
                "document_a": "alice.pdf",
                "document_b": "bob.pdf",
                "similarity_score": 0.5,
            }
        ]

    chunks = list(stream_incidents_csv_chunks(mock_query, delimiter=";"))

    assert len(chunks) == 2
    header, data = chunks
    assert header.startswith("Incident ID;Doc A;Doc B")
    assert "INC-1;alice.pdf;bob.pdf" in data


def test_create_batch_incident_zip_archive_semicolon_delimiter():
    """create_batch_incident_zip_archive must forward delimiter into the
    incidents_summary.csv it writes."""
    from src.utils.bulk_export import create_batch_incident_zip_archive

    incidents = [
        {
            "incident_id": "INC-1",
            "document_a": "alice.pdf",
            "document_b": "bob.pdf",
            "similarity_score": 0.5,
        }
    ]

    zip_bytes = create_batch_incident_zip_archive(incidents, delimiter=";")

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        csv_text = zf.read("incidents_summary.csv").decode("utf-8-sig")

    assert "Incident ID;Doc A;Doc B" in csv_text


def test_export_incidents_csv_delimiter_validation():
    """Verify that export_incidents_csv validates delimiter and falls back to comma if invalid (#1735)."""
    # 1. Test valid 1-character delimiter
    csv_bytes = export_incidents_csv(_SAMPLE_INCIDENTS, delimiter=";")
    first_line = csv_bytes.decode("utf-8-sig").splitlines()[0]
    assert ";" in first_line
    assert "," not in first_line

    # 2. Test multi-character delimiter (invalid) -> should fall back to ","
    csv_bytes_multi = export_incidents_csv(_SAMPLE_INCIDENTS, delimiter=";;")
    first_line_multi = csv_bytes_multi.decode("utf-8-sig").splitlines()[0]
    assert "," in first_line_multi
    assert ";" not in first_line_multi

    # 3. Test non-string delimiter (invalid) -> should fall back to ","
    csv_bytes_none = export_incidents_csv(_SAMPLE_INCIDENTS, delimiter=None)
    first_line_none = csv_bytes_none.decode("utf-8-sig").splitlines()[0]
    assert "," in first_line_none


def test_export_incidents_csv_quoting_style():
    """Verify that export_incidents_csv respects custom quoting styles (#1739)."""
    # Test QUOTE_ALL: all fields should be quoted
    csv_bytes_all = export_incidents_csv(_SAMPLE_INCIDENTS, quoting_style=csv.QUOTE_ALL)
    text_all = csv_bytes_all.decode("utf-8-sig")
    first_line_all = text_all.splitlines()[0]
    # Header fields must be quoted
    assert (
        '"Incident ID","Doc A","Doc B","Similarity","Severity","Status","Date"'
        in first_line_all
    )

    # Test QUOTE_MINIMAL: default minimal quoting (normal string without special characters is unquoted)
    csv_bytes_min = export_incidents_csv(
        _SAMPLE_INCIDENTS, quoting_style=csv.QUOTE_MINIMAL
    )
    text_min = csv_bytes_min.decode("utf-8-sig")
    first_line_min = text_min.splitlines()[0]
    # Header fields must not be quoted
    assert "Incident ID,Doc A,Doc B,Similarity,Severity,Status,Date" in first_line_min


# ---------------------------------------------------------------------------
# Tests for sanitize_csv_cell_value (Issue #1744)
# ---------------------------------------------------------------------------


class TestSanitizeCsvCellValue:
    """Tests for CSV cell value sanitizer preventing formula injection (#1744)."""

    def test_prepends_single_quote_for_formula_characters(self):
        assert sanitize_csv_cell_value("=1+1") == "'=1+1"
        assert sanitize_csv_cell_value("+100") == "'+100"
        assert sanitize_csv_cell_value("-50") == "'-50"
        assert sanitize_csv_cell_value("@SUM(A1:A10)") == "'@SUM(A1:A10)"

    def test_safe_strings_unchanged(self):
        assert sanitize_csv_cell_value("normal_text") == "normal_text"
        assert sanitize_csv_cell_value("INC-100") == "INC-100"
        assert sanitize_csv_cell_value("doc_a.pdf") == "doc_a.pdf"

    def test_none_and_empty(self):
        assert sanitize_csv_cell_value(None) == ""
        assert sanitize_csv_cell_value("") == ""

    def test_numeric_values(self):
        assert sanitize_csv_cell_value(123) == "123"
        assert sanitize_csv_cell_value(95.5) == "95.5"

    def test_export_incidents_csv_stream_sanitizes_injection_triggers(self):
        """Verify export_incidents_csv_stream sanitizes cell values starting with formula characters."""
        incidents = [
            {
                "incident_id": "=CMD|' /C calc'!A0",
                "document_a": "+malicious_doc.pdf",
                "document_b": "-subtraction.docx",
                "severity_rank": "@admin",
            }
        ]
        csv_bytes = export_incidents_csv_stream(incidents)
        text = csv_bytes.decode("utf-8-sig")
        assert "'=CMD|' /C calc'!A0" in text
        assert "'+malicious_doc.pdf" in text
        assert "'-subtraction.docx" in text
        assert "'@admin" in text


def test_sanitize_export_filename():
    # Test stripping illegal OS characters
    assert sanitize_export_filename("test<file>.csv") == "testfile.csv"
    assert sanitize_export_filename("test:file|name?.csv") == "testfilename.csv"

    # Test missing extension
    assert sanitize_export_filename("testfile") == "testfile.csv"

    # Test valid filename
    assert sanitize_export_filename("my_valid_file.csv") == "my_valid_file.csv"


# ---------------------------------------------------------------------------
# Tests for ExportFormat Enum (Issue #2008)
# ---------------------------------------------------------------------------


class TestExportFormatEnum:
    """Test suite for the ExportFormat Enum definition and behavior."""

    def test_enum_members_exist(self):
        """Verify all required enum members are defined."""
        assert ExportFormat.CSV.value == "csv"
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.XLSX.value == "xlsx"
        assert ExportFormat.PDF.value == "pdf"

    def test_enum_is_string_subclass(self):
        """Verify ExportFormat inherits from str for seamless integration."""
        assert isinstance(ExportFormat.CSV, str)
        assert ExportFormat.JSON == "json"

    def test_case_insensitive_lookup(self):
        """Verify the _missing_ override allows case-insensitive string matching."""
        assert ExportFormat("CSV") == ExportFormat.CSV
        assert ExportFormat("Json") == ExportFormat.JSON
        assert ExportFormat("XLSX") == ExportFormat.XLSX
        assert ExportFormat("pdf") == ExportFormat.PDF

    def test_whitespace_stripping(self):
        """Verify leading/trailing whitespace is ignored during lookup."""
        assert ExportFormat("  csv  ") == ExportFormat.CSV
        assert ExportFormat("\tjson\n") == ExportFormat.JSON

    def test_invalid_format_raises_value_error(self):
        """Verify unrecognized strings raise a descriptive ValueError."""
        with pytest.raises(ValueError, match="Invalid export format: 'xml'"):
            ExportFormat("xml")

        with pytest.raises(ValueError, match="Invalid export format: 'txt'"):
            ExportFormat("txt")

    def test_get_mime_type(self):
        """Verify MIME type mapping for each format."""
        assert ExportFormat.CSV.get_mime_type() == "text/csv"
        assert ExportFormat.JSON.get_mime_type() == "application/json"
        assert ExportFormat.PDF.get_mime_type() == "application/pdf"
        assert "spreadsheetml" in ExportFormat.XLSX.get_mime_type()

    def test_get_file_extension(self):
        """Verify file extension generation."""
        assert ExportFormat.CSV.get_file_extension() == ".csv"
        assert ExportFormat.JSON.get_file_extension() == ".json"
        assert ExportFormat.XLSX.get_file_extension() == ".xlsx"
        assert ExportFormat.PDF.get_file_extension() == ".pdf"


# ---------------------------------------------------------------------------
# Tests for export_incidents_to_format Dispatcher (Issue #2008)
# ---------------------------------------------------------------------------


class TestExportIncidentsToFormat:
    """Test suite for the export_incidents_to_format dispatcher."""

    @pytest.fixture
    def sample_incidents(self):
        """Provide a standard list of incident dictionaries for testing."""
        return [
            {
                "incident_id": "INC-001",
                "document_a": "alice.pdf",
                "document_b": "bob.pdf",
                "similarity_score": 0.95,
                "severity_rank": "High",
                "review_status": "Pending",
                "date_flagged": "2024-01-15T10:00:00",
            },
            {
                "incident_id": "INC-002",
                "document_a": "charlie.docx",
                "document_b": "dave.docx",
                "similarity_score": 0.72,
                "severity_rank": "Medium",
                "review_status": "Reviewed",
                "date_flagged": "2024-01-16T08:30:00",
            },
        ]

    def test_dispatcher_accepts_enum(self, sample_incidents):
        """Verify dispatcher works when passed an ExportFormat enum directly."""
        result = export_incidents_to_format(sample_incidents, ExportFormat.JSON)
        assert isinstance(result, bytes)
        parsed = json.loads(result.decode("utf-8"))
        assert len(parsed) == 2

    def test_dispatcher_accepts_string(self, sample_incidents):
        """Verify dispatcher works when passed a raw string."""
        result = export_incidents_to_format(sample_incidents, "json")
        assert isinstance(result, bytes)

    def test_dispatcher_csv_output(self, sample_incidents):
        """Verify CSV output is correctly formatted and encoded."""
        result = export_incidents_to_format(sample_incidents, ExportFormat.CSV)

        # Check BOM
        assert result.startswith(b"\xef\xbb\xbf")

        # Parse CSV
        text = result.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["Incident ID"] == "INC-001"
        assert rows[0]["Similarity"] == "95.00%"

    def test_dispatcher_json_output(self, sample_incidents):
        """Verify JSON output is valid and contains all fields."""
        result = export_incidents_to_format(sample_incidents, ExportFormat.JSON)
        parsed = json.loads(result.decode("utf-8"))

        assert parsed[0]["incident_id"] == "INC-001"
        assert parsed[1]["similarity_score"] == 0.72

    def test_dispatcher_invalid_string_raises(self, sample_incidents):
        """Verify dispatcher raises ValueError for invalid format strings."""
        with pytest.raises(ValueError, match="Invalid export format"):
            export_incidents_to_format(sample_incidents, "yaml")

    def test_dispatcher_invalid_type_raises(self, sample_incidents):
        """Verify dispatcher raises TypeError for non-string/non-enum inputs."""
        with pytest.raises(TypeError, match="format must be an ExportFormat"):
            export_incidents_to_format(sample_incidents, 123)

        with pytest.raises(TypeError, match="format must be an ExportFormat"):
            export_incidents_to_format(sample_incidents, ["csv"])

    def test_dispatcher_xlsx_fallback_to_csv(self, sample_incidents):
        """Verify XLSX export falls back to CSV if openpyxl is missing."""
        # Mock pandas.ExcelWriter to raise ImportError
        with patch(
            "pandas.ExcelWriter", side_effect=ImportError("No module named 'openpyxl'")
        ):
            result = export_incidents_to_format(sample_incidents, ExportFormat.XLSX)

        # Should fall back to CSV (starts with BOM)
        assert result.startswith(b"\xef\xbb\xbf")

    def test_dispatcher_pdf_fallback_to_json(self, sample_incidents):
        """Verify PDF export via dispatcher falls back to JSON for incident lists."""
        result = export_incidents_to_format(sample_incidents, ExportFormat.PDF)

        # Should return valid JSON bytes
        parsed = json.loads(result.decode("utf-8"))
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_empty_incidents_list(self):
        """Verify dispatcher handles empty incident lists gracefully."""
        csv_result = export_incidents_to_format([], ExportFormat.CSV)
        json_result = export_incidents_to_format([], ExportFormat.JSON)

        # CSV should just have headers
        assert b"Incident ID" in csv_result

        # JSON should be empty array
        assert json.loads(json_result.decode("utf-8")) == []


# ---------------------------------------------------------------------------
# Tests for Legacy Direct Functions
# ---------------------------------------------------------------------------


class TestLegacyFunctions:
    """Test suite to ensure legacy direct functions still work."""

    def test_csv_stream_basic(self):
        """Verify export_incidents_csv_stream works independently."""
        incidents = [
            {
                "incident_id": "1",
                "document_a": "A",
                "document_b": "B",
                "similarity_score": 0.5,
            }
        ]
        result = export_incidents_csv_stream(incidents)
        assert isinstance(result, bytes)

    def test_json_stream_basic(self):
        """Verify export_incidents_json_stream works independently."""
        incidents = [{"incident_id": "1", "document_a": "A"}]
        result = export_incidents_json_stream(incidents)
        assert json.loads(result.decode("utf-8"))[0]["incident_id"] == "1"


# ---------------------------------------------------------------------------
# Tests for Progress Callback (Issue #3467)
# ---------------------------------------------------------------------------


def test_generate_bulk_reports_zip_progress_callback():
    from src.utils.bulk_export import generate_bulk_reports_zip

    flags = [
        {"doc_a": "A.pdf", "doc_b": "B.pdf", "similarity": 0.5},
        {"doc_a": "C.pdf", "doc_b": "D.pdf", "similarity": 0.6},
    ]
    calls = []

    def progress_cb(current, total):
        calls.append((current, total))

    generate_bulk_reports_zip(flags, include_pdf=False, progress_callback=progress_cb)

    assert calls == [(1, 2), (2, 2)]


def test_create_batch_incident_zip_archive_progress_callback():
    from src.utils.bulk_export import create_batch_incident_zip_archive

    incidents = [
        {"incident_id": "1", "document_a": "A", "document_b": "B"},
        {"incident_id": "2", "document_a": "C", "document_b": "D"},
        {"incident_id": "3", "document_a": "E", "document_b": "F"},
    ]
    calls = []

    def progress_cb(current, total):
        calls.append((current, total))

    # Mock generate_plagiarism_report to speed up test
    with patch("src.utils.bulk_export.generate_plagiarism_report") as mock_report:
        mock_report.return_value = io.BytesIO(b"fake pdf")
        create_batch_incident_zip_archive(incidents, progress_callback=progress_cb)

    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_create_documents_bulk_zip_archive_progress_callback():
    from src.utils.bulk_export import create_documents_bulk_zip_archive

    filenames = ["doc1.pdf", "doc2.pdf"]
    calls = []

    def progress_cb(current, total):
        calls.append((current, total))

    with patch("src.utils.bulk_export.get_all_documents", return_value=[]), patch(
        "src.utils.bulk_export.get_document_word_counts", return_value={}
    ), patch("src.utils.bulk_export._connect") as mock_connect:
        # mock conn
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_connect.return_value.__enter__.return_value = mock_conn

        create_documents_bulk_zip_archive(filenames, progress_callback=progress_cb)

    assert calls == [(1, 2), (2, 2)]


def test_generate_bulk_reports_zip_no_callback():
    from src.utils.bulk_export import generate_bulk_reports_zip

    flags = [
        {"doc_a": "A.pdf", "doc_b": "B.pdf", "similarity": 0.5},
    ]
    # Should not raise any error
    result = generate_bulk_reports_zip(flags, include_pdf=False)
    assert isinstance(result, bytes)
