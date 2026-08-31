"""
tests/security/test_metadata_forensics.py
-----------------------------------------
Unit tests for Document Provenance and Metadata Forensics.
"""

import pytest
import zipfile
import io
from src.security.metadata_forensics import extract_docx_metadata, extract_pdf_metadata
from src.core.edit_history_analyzer import (
    analyze_edit_velocity,
    analyze_author_mismatch,
    compute_provenance_risk_score,
)


class TestDocxMetadataExtraction:
    """Test suite for DOCX metadata extraction."""

    def test_extract_docx_metadata_valid(self):
        """Verify metadata is extracted from a valid DOCX structure."""
        # Create a minimal mock DOCX (ZIP with core.xml)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            core_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" 
                               xmlns:dc="http://purl.org/dc/elements/1.1/" 
                               xmlns:dcterms="http://purl.org/dc/terms/">
                <dc:creator>John Doe</dc:creator>
                <cp:lastModifiedBy>Jane Smith</cp:lastModifiedBy>
                <cp:revision>5</cp:revision>
            </cp:coreProperties>"""
            zf.writestr("docProps/core.xml", core_xml)

        metadata = extract_docx_metadata(buffer.getvalue())
        assert metadata["creator"] == "John Doe"
        assert metadata["last_modified_by"] == "Jane Smith"
        assert metadata["revision"] == "5"

    def test_extract_docx_metadata_invalid_zip(self):
        """Verify graceful handling of invalid ZIP files."""
        metadata = extract_docx_metadata(b"not a zip file")
        assert metadata["creator"] is None


class TestEditHistoryAnalyzer:
    """Test suite for edit history and velocity analysis."""

    def test_analyze_edit_velocity_normal(self):
        """Verify normal editing velocity is not flagged."""
        result = analyze_edit_velocity(total_editing_time_minutes=60.0, page_count=10)
        assert result["minutes_per_page"] == 6.0
        assert result["is_velocity_anomaly"] is False
        assert result["velocity_score"] == 1.0

    def test_analyze_edit_velocity_suspicious(self):
        """Verify extremely low editing velocity is flagged."""
        result = analyze_edit_velocity(total_editing_time_minutes=5.0, page_count=20)
        assert result["minutes_per_page"] == 0.25
        assert result["is_velocity_anomaly"] is True
        assert result["velocity_score"] < 0.5

    def test_analyze_author_mismatch_match(self):
        """Verify matching authors are not flagged."""
        metadata = {"creator": "Alice", "last_modified_by": "Alice"}
        result = analyze_author_mismatch(metadata)
        assert result["is_author_mismatch"] is False

    def test_analyze_author_mismatch_mismatch(self):
        """Verify mismatched authors are flagged."""
        metadata = {"creator": "Alice", "last_modified_by": "Bob"}
        result = analyze_author_mismatch(metadata)
        assert result["is_author_mismatch"] is True
        assert result["mismatch_score"] > 0.0

    def test_compute_provenance_risk_score_suspicious(self):
        """Verify overall risk score is high for suspicious documents."""
        # Mock DOCX bytes
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            core_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" 
                               xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:creator>EssayMill</dc:creator>
                <cp:lastModifiedBy>Student</cp:lastModifiedBy>
            </cp:coreProperties>"""
            app_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
                <TotalTime>2</TotalTime>
            </Properties>"""
            zf.writestr("docProps/core.xml", core_xml)
            zf.writestr("docProps/app.xml", app_xml)

        result = compute_provenance_risk_score(
            buffer.getvalue(), "docx", estimated_pages=20
        )
        assert result["is_suspicious"] is True
        assert result["risk_score"] > 0.5
