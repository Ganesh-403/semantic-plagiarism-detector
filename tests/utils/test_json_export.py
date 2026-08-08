"""
tests/utils/test_json_export.py
--------------------------------
Comprehensive unit test suite for src/utils/json_export.py.

Tests cover:
- ISO 8601 UTC timestamp (`exported_at`) generation and schema compliance (Issue #1034).
- `export_to_json()` metadata root wrapping, custom metadata merging, data preservation, and configurable indentation (Issue #1250).
- `export_similarity_matrix_to_json()` backward compatibility and pair extraction.
- `export_report_to_json()` and `export_incidents_to_json()` helper outputs.
- `parse_export_json()` parsing and schema validator `validate_json_export_schema()`.
- Data serialization edge cases: NumPy types, NaNs, infinities, Timestamps, and Unicode text.
"""

from datetime import datetime, timezone
import json
import re

import numpy as np
import pandas as pd
import pytest

from src.utils.json_export import (
    build_export_schema_definition,
    export_batch_reports_to_json,
    export_filtered_similarity_matrix_to_json,
    export_incidents_to_json,
    export_report_to_json,
    export_similarity_matrix_to_json,
    export_to_json,
    generate_export_checksum,
    get_export_timestamp,
    json_serializer_fallback,
    parse_export_json,
    validate_json_export_schema,
)

# ISO 8601 UTC Regex Pattern: e.g., 2026-07-31T07:25:00Z
ISO_8601_UTC_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def test_get_export_timestamp_format():
    """Verify that get_export_timestamp() returns a valid ISO 8601 UTC timestamp string ending with Z."""
    ts = get_export_timestamp()
    assert isinstance(ts, str)
    assert ISO_8601_UTC_PATTERN.match(ts), f"Timestamp '{ts}' does not match ISO 8601 UTC pattern YYYY-MM-DDTHH:MM:SSZ"


def test_export_to_json_includes_exported_at_timestamp():
    """Verify that export_to_json() includes exported_at in metadata root (Issue #1034)."""
    sample_data = {"document": "test.txt", "score": 0.88}
    json_output = export_to_json(sample_data)

    parsed = json.loads(json_output)
    assert "metadata" in parsed
    assert "exported_at" in parsed["metadata"]

    exported_at = parsed["metadata"]["exported_at"]
    assert ISO_8601_UTC_PATTERN.match(exported_at)
    assert parsed["data"] == sample_data


def test_export_to_json_preserves_existing_fields():
    """Verify that export_to_json() preserves all original data fields and types."""
    data = {
        "report_id": 1042,
        "author": "Alice Doe",
        "scores": [0.95, 0.42, 0.10],
        "is_flagged": True,
        "details": {"method": "tfidf", "ngram": 3},
    }

    json_str = export_to_json(data)
    result = json.loads(json_str)

    assert result["data"]["report_id"] == 1042
    assert result["data"]["author"] == "Alice Doe"
    assert result["data"]["scores"] == [0.95, 0.42, 0.10]
    assert result["data"]["is_flagged"] is True
    assert result["data"]["details"]["method"] == "tfidf"


def test_export_to_json_with_custom_metadata():
    """Verify that custom metadata fields are merged into root metadata alongside exported_at."""
    data = ["doc1.pdf", "doc2.pdf"]
    custom_meta = {"version": "2.1.0", "environment": "production"}

    json_str = export_to_json(data, metadata=custom_meta)
    result = json.loads(json_str)

    assert "metadata" in result
    assert result["metadata"]["version"] == "2.1.0"
    assert result["metadata"]["environment"] == "production"
    assert "exported_at" in result["metadata"]
    assert result["metadata"]["total_records"] == 2


def test_export_to_json_without_metadata():
    """Verify that setting include_metadata=False returns raw JSON without metadata root."""
    data = {"key": "value", "count": 5}
    json_str = export_to_json(data, include_metadata=False)
    result = json.loads(json_str)

    assert result == data
    assert "metadata" not in result


def test_export_to_json_minified_indent_none():
    """Verify minified single-line JSON generation when indent=None (Issue #1250)."""
    data = {"doc": "A", "score": 0.95}
    json_str = export_to_json(data, include_metadata=False, indent=None)

    assert "\n" not in json_str
    assert json_str == '{"doc": "A", "score": 0.95}'


def test_export_to_json_custom_indentation():
    """Verify custom indentation formatting (Issue #1250)."""
    data = {"doc": "A"}
    json_str = export_to_json(data, include_metadata=False, indent=4)

    assert '\n    "doc": "A"' in json_str


def test_export_similarity_matrix_to_json_valid():
    """Verify that a valid similarity matrix is converted to a clean JSON array of unique pairs."""
    data = [[1.0, 0.85, 0.45], [0.85, 1.0, 0.92], [0.45, 0.92, 1.0]]
    df = pd.DataFrame(
        data, index=["docA", "docB", "docC"], columns=["docA", "docB", "docC"]
    )

    json_str = export_similarity_matrix_to_json(df)
    result = json.loads(json_str)

    assert len(result) == 3
    assert result[0] == {
        "document_1": "docA",
        "document_2": "docB",
        "similarity_score": 0.85,
    }
    assert result[1] == {
        "document_1": "docA",
        "document_2": "docC",
        "similarity_score": 0.45,
    }
    assert result[2] == {
        "document_1": "docB",
        "document_2": "docC",
        "similarity_score": 0.92,
    }
    # Verify pretty-printing with indent=2 (#1614)
    assert "\n" in json_str
    assert '  "document_1"' in json_str


def test_export_similarity_matrix_to_json_with_metadata():
    """Verify export_similarity_matrix_to_json() with include_metadata=True."""
    data = [[1.0, 0.75], [0.75, 1.0]]
    df = pd.DataFrame(data, index=["docA", "docB"], columns=["docA", "docB"])

    json_str = export_similarity_matrix_to_json(df, include_metadata=True)
    result = json.loads(json_str)

    assert "metadata" in result
    assert "exported_at" in result["metadata"]
    assert result["metadata"]["pair_count"] == 1
    assert result["metadata"]["documents_count"] == 2
    assert len(result["pairs"]) == 1


def test_export_similarity_matrix_to_json_empty():
    """Verify that empty or None DataFrame returns empty JSON array or empty metadata structure."""
    assert export_similarity_matrix_to_json(None) == "[]"
    assert export_similarity_matrix_to_json(pd.DataFrame()) == "[]"

    meta_empty = export_similarity_matrix_to_json(None, include_metadata=True)
    parsed = json.loads(meta_empty)
    assert parsed["metadata"]["pair_count"] == 0
    assert parsed["pairs"] == []


def test_export_similarity_matrix_to_json_single_document():
    """Verify that a single document similarity matrix returns an empty array (no pairs)."""
    df = pd.DataFrame([[1.0]], index=["docA"], columns=["docA"])
    assert export_similarity_matrix_to_json(df) == "[]"


def test_export_similarity_matrix_to_json_nan_handling():
    """Verify that NaN similarity scores are gracefully set to 0.0 in JSON export."""
    data = [[1.0, np.nan], [np.nan, 1.0]]
    df = pd.DataFrame(data, index=["docA", "docB"], columns=["docA", "docB"])

    json_str = export_similarity_matrix_to_json(df)
    result = json.loads(json_str)

    assert len(result) == 1
    assert result[0] == {
        "document_1": "docA",
        "document_2": "docB",
        "similarity_score": 0.0,
    }


def test_export_similarity_matrix_to_json_unicode_filenames():
    """Verify that Unicode (UTF-8) characters in filenames are preserved and not escaped."""
    data = [[1.0, 0.75], [0.75, 1.0]]
    df = pd.DataFrame(
        data, index=["📄_doc.txt", "doc_üñ.txt"], columns=["📄_doc.txt", "doc_üñ.txt"]
    )

    json_str = export_similarity_matrix_to_json(df)

    assert "📄_doc.txt" in json_str
    assert "doc_üñ.txt" in json_str

    result = json.loads(json_str)
    assert len(result) == 1
    assert result[0]["document_1"] == "📄_doc.txt"
    assert result[0]["document_2"] == "doc_üñ.txt"


def test_export_report_to_json():
    """Verify export_report_to_json() generates valid JSON with report metadata."""
    report_data = {
        "summary": "Plagiarism analysis completed",
        "top_matches": [{"pair": ["A", "B"], "score": 0.91}],
    }
    json_str = export_report_to_json(report_data, custom_metadata={"author": "Inspector"})

    parsed = json.loads(json_str)
    assert parsed["metadata"]["report_type"] == "plagiarism_analysis"
    assert parsed["metadata"]["author"] == "Inspector"
    assert "exported_at" in parsed["metadata"]
    assert parsed["data"]["summary"] == "Plagiarism analysis completed"


def test_export_incidents_to_json():
    """Verify export_incidents_to_json() formats list of incident records."""
    incidents = [
        {"incident_id": 1, "document_name": "essay1.docx", "status": "resolved"},
        {"incident_id": 2, "document_name": "essay2.pdf", "status": "pending"},
    ]

    json_str = export_incidents_to_json(incidents, session_id="sess_123")
    parsed = json.loads(json_str)

    assert parsed["metadata"]["report_type"] == "incident_log"
    assert parsed["metadata"]["session_id"] == "sess_123"
    assert parsed["metadata"]["total_incidents"] == 2
    assert "exported_at" in parsed["metadata"]
    assert len(parsed["data"]) == 2


def test_parse_export_json_valid_and_invalid():
    """Verify parse_export_json() parses valid JSON and handles invalid input gracefully."""
    valid_json = '{"metadata": {"exported_at": "2026-07-31T07:25:00Z"}, "data": [1, 2, 3]}'
    parsed = parse_export_json(valid_json)
    assert parsed["metadata"]["exported_at"] == "2026-07-31T07:25:00Z"

    assert parse_export_json("") == {}
    assert parse_export_json("invalid json string {") == {}
    assert parse_export_json(None) == {}


def test_validate_json_export_schema():
    """Verify validate_json_export_schema() returns True for compliant export dicts."""
    valid_dict = {
        "metadata": {
            "exported_at": "2026-07-31T07:25:00Z",
            "version": "1.0.0",
        },
        "data": [],
    }
    assert validate_json_export_schema(valid_dict) is True

    invalid_no_meta = {"data": []}
    assert validate_json_export_schema(invalid_no_meta) is False

    invalid_no_timestamp = {"metadata": {"version": "1.0.0"}, "data": []}
    assert validate_json_export_schema(invalid_no_timestamp) is False

    assert validate_json_export_schema(None) is False
    assert validate_json_export_schema("not a dict") is False


def test_generate_export_checksum():
    """Verify generate_export_checksum() computes deterministic 64-char SHA-256 hex string."""
    json_text = '{"metadata": {"exported_at": "2026-07-31T07:25:00Z"}, "data": [1, 2]}'
    checksum = generate_export_checksum(json_text)

    assert isinstance(checksum, str)
    assert len(checksum) == 64
    assert checksum == generate_export_checksum(json_text)


def test_export_batch_reports_to_json():
    """Verify export_batch_reports_to_json() packages multiple reports into batch payload."""
    reports = [{"id": 1, "score": 0.8}, {"id": 2, "score": 0.3}]
    json_str = export_batch_reports_to_json(reports, batch_id="batch_001")
    parsed = json.loads(json_str)

    assert parsed["metadata"]["report_type"] == "batch_plagiarism_analysis"
    assert parsed["metadata"]["batch_id"] == "batch_001"
    assert parsed["metadata"]["batch_size"] == 2
    assert "exported_at" in parsed["metadata"]
    assert len(parsed["data"]) == 2


def test_export_filtered_similarity_matrix_to_json():
    """Verify export_filtered_similarity_matrix_to_json() filters pairs below min similarity threshold."""
    data = [[1.0, 0.85, 0.20], [0.85, 1.0, 0.95], [0.20, 0.95, 1.0]]
    df = pd.DataFrame(data, index=["docA", "docB", "docC"], columns=["docA", "docB", "docC"])

    json_str = export_filtered_similarity_matrix_to_json(df, min_similarity_threshold=0.50)
    parsed = json.loads(json_str)

    assert parsed["metadata"]["min_similarity_threshold"] == 0.50
    assert parsed["metadata"]["filtered_pairs_count"] == 2
    assert len(parsed["data"]) == 2
    assert parsed["data"][0]["similarity_score"] == 0.85
    assert parsed["data"][1]["similarity_score"] == 0.95


def test_build_export_schema_definition():
    """Verify build_export_schema_definition() returns valid JSON Schema object."""
    schema = build_export_schema_definition()
    assert isinstance(schema, dict)
    assert schema["title"] == "PlagiarismDetectorExportReport"
    assert "exported_at" in schema["properties"]["metadata"]["required"]


def test_json_serializer_fallback():
    """Verify custom NumPy, pandas, and datetime serializer function."""
    assert json_serializer_fallback(np.int64(42)) == 42
    assert json_serializer_fallback(np.float64(3.14159)) == 3.14159
    assert json_serializer_fallback(np.nan) == 0.0

    now = datetime(2026, 7, 31, 7, 25, 0, tzinfo=timezone.utc)
    assert json_serializer_fallback(now) == "2026-07-31T07:25:00+00:00"

    ts = pd.Timestamp("2026-07-31 07:25:00")
    assert json_serializer_fallback(ts) == "2026-07-31T07:25:00"

    assert json_serializer_fallback({1, 2, 3}) == [1, 2, 3] or isinstance(json_serializer_fallback({1, 2, 3}), list)


def test_export_to_json_serializes_numpy_types_without_error():
    """Issue: passing NumPy types/datetime through export_to_json() must not raise
    an unhandled TypeError — verifies default=json_serializer_fallback is actually
    wired into json.dumps() end-to-end, not just tested in isolation."""
    data = {
        "count": np.int64(7),
        "score": np.float32(0.85),
        "matrix": np.array([1, 2, 3]),
        "generated_on": datetime(2026, 7, 31, 7, 25, 0, tzinfo=timezone.utc),
    }

    # Must not raise TypeError: Object of type int64 is not JSON serializable
    json_str = export_to_json(data, include_metadata=False)
    parsed = json.loads(json_str)

    assert parsed["count"] == 7
    assert isinstance(parsed["count"], int)
    assert parsed["score"] == pytest.approx(0.85, rel=1e-4)
    assert parsed["matrix"] == [1, 2, 3]
    assert parsed["generated_on"] == "2026-07-31T07:25:00+00:00"


def test_export_to_json_numpy_types_with_metadata_wrapper():
    """Same as above but through the default include_metadata=True path, which
    uses a different json.dumps() call site — both must use the fallback."""
    data = {"total": np.int32(3), "average": np.float64(1.5)}

    json_str = export_to_json(data)
    parsed = json.loads(json_str)

    assert parsed["data"]["total"] == 3
    assert parsed["data"]["average"] == 1.5
    assert "exported_at" in parsed["metadata"]
