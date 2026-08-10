import io
import json
import zipfile

from src.utils.bulk_export import (
    export_incidents_csv_stream,
    export_incidents_csv,
    generate_bulk_reports_zip,
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
    import csv as _csv
    import io

    csv_bytes = export_incidents_csv_stream(_SAMPLE_INCIDENTS)
    text = csv_bytes.decode("utf-8-sig")
    reader = _csv.DictReader(io.StringIO(text))
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
    import csv as _csv
    import io

    csv_bytes = export_incidents_csv_stream([])
    assert csv_bytes.startswith(b"\xef\xbb\xbf")

    text = csv_bytes.decode("utf-8-sig")
    reader = _csv.DictReader(io.StringIO(text))
    rows = list(reader)

    assert rows == []
    # Rewind and confirm only one line (the header) exists
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 1


def test_export_incidents_csv_stream_non_numeric_similarity():
    """Non-numeric similarity_score should be written as-is without raising."""
    import csv as _csv
    import io

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
    reader = _csv.DictReader(io.StringIO(text))
    rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["Similarity"] == "N/A"


# ---------------------------------------------------------------------------
# Tests for normalize_csv_headers (Issue #1253)
# ---------------------------------------------------------------------------

from src.utils.bulk_export import normalize_csv_headers


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
        result = normalize_csv_headers(["café", "naïve", "résumé"])
        assert result == ["café", "naïve", "résumé"]

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
        assert "report_INC-001_alicepdf_bobpdf.pdf" in pdf_names
        assert "report_INC-002_charliedocx_davedocx.pdf" in pdf_names

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


# ---------------------------------------------------------------------------
# Tests for stream_incidents_csv_chunks (Issue #1511)
# ---------------------------------------------------------------------------

from src.utils.bulk_export import stream_incidents_csv_chunks

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
        return [{"incident_id": f"INC-{offset+i}"} for i in range(min(limit, 50 - offset))]
        
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
        return [{
            "incident_id": "INC-1",
            "document_a": "comma, in name.pdf",
            "document_b": 'quote" in name.pdf',
            "similarity_score": 0.5
        }]
        
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
    import csv as _csv
    import io as _io

    csv_bytes = export_incidents_csv_stream(_SAMPLE_INCIDENTS, delimiter=";")
    text = csv_bytes.decode("utf-8-sig")
    first_line = text.splitlines()[0]

    # Header row uses semicolons, not commas, as the field separator
    assert first_line == "Incident ID;Doc A;Doc B;Similarity;Severity;Status;Date"

    reader = _csv.DictReader(_io.StringIO(text), delimiter=";")
    rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["Incident ID"] == "INC-001"
    assert rows[0]["Doc A"] == "alice.pdf"
    assert rows[0]["Similarity"] == "95.00%"
    assert rows[1]["Incident ID"] == "INC-002"


def test_export_incidents_csv_stream_tab_delimiter():
    """delimiter='\\t' must produce tab-delimited CSV."""
    import csv as _csv
    import io as _io

    csv_bytes = export_incidents_csv_stream(_SAMPLE_INCIDENTS, delimiter="\t")
    text = csv_bytes.decode("utf-8-sig")
    first_line = text.splitlines()[0]

    assert "\t" in first_line
    assert "," not in first_line

    reader = _csv.DictReader(_io.StringIO(text), delimiter="\t")
    rows = list(reader)
    assert rows[0]["Doc A"] == "alice.pdf"


def test_stream_incidents_csv_chunks_semicolon_delimiter():
    """stream_incidents_csv_chunks must honor delimiter=';' across both
    the header chunk and the data chunks."""
    def mock_query(limit, offset):
        if offset > 0:
            return []
        return [{
            "incident_id": "INC-1",
            "document_a": "alice.pdf",
            "document_b": "bob.pdf",
            "similarity_score": 0.5,
        }]

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
    import csv

    # Test QUOTE_ALL: all fields should be quoted
    csv_bytes_all = export_incidents_csv(_SAMPLE_INCIDENTS, quoting_style=csv.QUOTE_ALL)
    text_all = csv_bytes_all.decode("utf-8-sig")
    first_line_all = text_all.splitlines()[0]
    # Header fields must be quoted
    assert '"Incident ID","Doc A","Doc B","Similarity","Severity","Status","Date"' in first_line_all

    # Test QUOTE_MINIMAL: default minimal quoting (normal string without special characters is unquoted)
    csv_bytes_min = export_incidents_csv(_SAMPLE_INCIDENTS, quoting_style=csv.QUOTE_MINIMAL)
    text_min = csv_bytes_min.decode("utf-8-sig")
    first_line_min = text_min.splitlines()[0]
    # Header fields must not be quoted
    assert "Incident ID,Doc A,Doc B,Similarity,Severity,Status,Date" in first_line_min



