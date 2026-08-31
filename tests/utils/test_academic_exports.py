"""
tests/utils/test_academic_exports.py
------------------------------------
Unit tests for academic export formats (Turnitin XML, PDF/A, LMS Manifest).
"""

import pytest
import xml.etree.ElementTree as ET
from datetime import datetime

from src.utils.export_tii_xml import generate_tii_xml, validate_tii_xml
from src.utils.pdfa_generator import generate_xmp_metadata, prepare_pdfa_metadata
from src.utils.lms_manifest import generate_canvas_manifest


class TestTurnitinXML:
    """Test suite for Turnitin XML generation."""

    def test_generates_valid_xml(self):
        """Verify the output is well-formed XML."""
        data = {
            "document_id": "doc_123",
            "title": "Test Essay",
            "author": "Alice",
            "similarity_score": 25,
            "matches": [{"source": "Internet", "score": 10, "start": 0, "end": 50}],
        }
        xml_str = generate_tii_xml(data)

        assert validate_tii_xml(xml_str) is True

    def test_contains_required_elements(self):
        """Verify the XML contains submission and score elements."""
        data = {"document_id": "1", "similarity_score": 50}
        xml_str = generate_tii_xml(data)

        root = ET.fromstring(xml_str)
        assert root.tag == "originalityReport"
        assert root.find("submission") is not None
        assert root.find("overallSimilarity").text == "50"

    def test_includes_matches(self):
        """Verify match highlights are included."""
        data = {"matches": [{"source": "Wiki", "score": 20, "start": 10, "end": 20}]}
        xml_str = generate_tii_xml(data)
        root = ET.fromstring(xml_str)

        matches = root.find("matches")
        assert matches is not None
        assert len(matches.findall("match")) == 1


class TestPDFA_Metadata:
    """Test suite for PDF/A XMP metadata generation."""

    def test_generates_xmp_packet(self):
        """Verify the output contains the XMP packet wrapper."""
        xmp = generate_xmp_metadata(
            title="Report",
            author="System",
            creation_date=datetime.utcnow(),
            modification_date=datetime.utcnow(),
        )

        assert "<?xpacket begin" in xmp
        assert "<?xpacket end" in xmp
        assert "rdf:RDF" in xmp

    def test_contains_pdfa_identification(self):
        """Verify the PDF/A identification schema is present."""
        xmp = generate_xmp_metadata("T", "A", datetime.utcnow(), datetime.utcnow())

        # Check for PDF/A part and conformance
        assert "pdfaid:part" in xmp
        assert "pdfaid:conformance" in xmp
        assert ">1<" in xmp  # Part 1
        assert ">B<" in xmp  # Conformance B


class TestLMSManifest:
    """Test suite for LMS manifest generation."""

    def test_generates_csv_manifest(self):
        """Verify CSV output contains headers and data."""
        records = [
            {
                "student_id": "S1",
                "assignment_id": "A1",
                "report_filename": "r1.pdf",
                "similarity_score": 10,
            }
        ]
        csv_str = generate_canvas_manifest(records, format="csv")

        lines = csv_str.strip().split("\n")
        assert len(lines) == 2  # Header + 1 record
        assert "student_id" in lines[0]
        assert "S1" in lines[1]

    def test_generates_xml_manifest(self):
        """Verify XML output is well-formed."""
        records = [
            {
                "student_id": "S1",
                "assignment_id": "A1",
                "report_filename": "r1.pdf",
                "similarity_score": 10,
            }
        ]
        xml_str = generate_canvas_manifest(records, format="xml")

        root = ET.fromstring(xml_str)
        assert root.tag == "manifest"
        assert len(root.findall("entry")) == 1
