"""
test_empty_file_upload_validation_issue_3718.py
-----------------------------------------------
Comprehensive unit and regression test suite for Issue #3718:
Handle empty file validation in is_executable_upload.

Verifies:
1. is_executable_upload(b"", filename) returns False immediately for empty byte strings across all filenames.
2. Behavior with empty binary payloads with blocked extensions (.exe, .sh, .bat, .dll).
3. Behavior with empty binary payloads with document extensions (.pdf, .docx, .txt, .csv, .md, .doc).
4. Behavior with empty binary payloads with no extensions, dot-only, or whitespace filenames.
5. Boundary conditions: bytearray(), 1-byte payloads, whitespace-only files.
6. Integration with inspect_upload_safety, validate_upload_payload_safety, classify_upload_threat_vector, and audit_file_stream_safety.
"""

import pytest

from src.security.mime_validator import (
    audit_file_stream_safety,
    batch_validate_upload_payloads,
    classify_upload_threat_vector,
    inspect_upload_safety,
    is_empty_or_whitespace_upload,
    is_executable_upload,
    validate_mime_type,
    validate_upload_payload_safety,
)


class TestEmptyFileIsExecutableUpload:
    """Test suite asserting is_executable_upload handles empty files cleanly."""

    def test_empty_bytes_with_blocked_extensions_returns_false(self):
        """Empty files must return False immediately even if declared extension is in BLOCKED_EXECUTABLE_EXTENSIONS."""
        assert is_executable_upload(b"", "malware.exe") is False
        assert is_executable_upload(b"", "script.sh") is False
        assert is_executable_upload(b"", "payload.bat") is False
        assert is_executable_upload(b"", "app.js") is False
        assert is_executable_upload(b"", "macro.vbs") is False
        assert is_executable_upload(b"", "library.dll") is False

    def test_empty_bytes_with_document_extensions_returns_false(self):
        """Empty files must return False for standard document uploads."""
        assert is_executable_upload(b"", "paper.pdf") is False
        assert is_executable_upload(b"", "thesis.docx") is False
        assert is_executable_upload(b"", "report.doc") is False
        assert is_executable_upload(b"", "data.csv") is False
        assert is_executable_upload(b"", "notes.txt") is False
        assert is_executable_upload(b"", "readme.md") is False

    def test_empty_bytes_with_unusual_filenames(self):
        """Test with extensionless, hidden, or malformed filenames."""
        assert is_executable_upload(b"", "filename_without_extension") is False
        assert is_executable_upload(b"", ".hidden") is False
        assert is_executable_upload(b"", "...") is False
        assert is_executable_upload(b"", "") is False
        assert is_executable_upload(b"", "   ") is False

    def test_empty_bytearray_and_memoryview(self):
        """Test with other zero-length buffer types."""
        assert is_executable_upload(bytearray(), "script.sh") is False
        assert is_executable_upload(bytes(), "exploit.exe") is False

    @pytest.mark.parametrize(
        "ext",
        [
            "exe", "sh", "bat", "js", "vbs", "dll",
            "pdf", "docx", "doc", "txt", "csv", "md",
            "bin", "tar", "gz", "iso", "png", "jpg",
        ],
    )
    def test_all_extensions_parametrized_empty_bytes(self, ext):
        assert is_executable_upload(b"", f"test_file.{ext}") is False


class TestNonEmptyExecutableUploads:
    """Test suite ensuring legitimate executables and scripts continue to be detected."""

    def test_non_empty_blocked_extension(self):
        content = b"echo 'test'"
        assert is_executable_upload(content, "script.sh") is True
        assert is_executable_upload(content, "runner.bat") is True
        assert is_executable_upload(content, "app.exe") is True

    def test_non_empty_magic_bytes_detection(self):
        pe_header = b"MZ\x90\x00\x03\x00\x00\x00"
        shell_header = b"#!/bin/sh\necho 'hacked'"

        # Magic bytes detected even with innocent document names
        assert is_executable_upload(pe_header, "assignment.pdf") is True
        assert is_executable_upload(shell_header, "notes.txt") is True

    def test_safe_document_uploads(self):
        pdf_bytes = b"%PDF-1.4\n1 0 obj\n"
        text_bytes = b"This is a standard academic text document for plagiarism checking."

        assert is_executable_upload(pdf_bytes, "document.pdf") is False
        assert is_executable_upload(text_bytes, "notes.txt") is False


class TestEmptyOrWhitespaceUploadHelper:
    """Test suite for is_empty_or_whitespace_upload helper function."""

    def test_empty_bytes(self):
        assert is_empty_or_whitespace_upload(b"") is True

    def test_whitespace_only_bytes(self):
        assert is_empty_or_whitespace_upload(b"   ") is True
        assert is_empty_or_whitespace_upload(b"\n\t\r ") is True

    def test_content_bytes(self):
        assert is_empty_or_whitespace_upload(b"a") is False
        assert is_empty_or_whitespace_upload(b"  some content  ") is False


class TestInspectUploadSafety:
    """Test suite for inspect_upload_safety integrated pre-ingestion audit."""

    def test_inspect_empty_file(self):
        report = inspect_upload_safety(b"", "empty_document.pdf")
        assert report["filename"] == "empty_document.pdf"
        assert report["extension"] == "pdf"
        assert report["size_bytes"] == 0
        assert report["is_empty"] is True
        assert report["is_executable"] is False
        assert report["is_valid_mime"] is False
        assert report["status"] == "rejected_empty"

    def test_inspect_executable_file(self):
        report = inspect_upload_safety(b"#!/bin/sh\nrm -rf /", "exploit.sh")
        assert report["filename"] == "exploit.sh"
        assert report["extension"] == "sh"
        assert report["is_empty"] is False
        assert report["is_executable"] is True
        assert report["status"] == "rejected_executable"

    def test_inspect_valid_text_file(self):
        content = b"Proper text content for analysis."
        report = inspect_upload_safety(content, "essay.txt")
        assert report["filename"] == "essay.txt"
        assert report["extension"] == "txt"
        assert report["is_empty"] is False
        assert report["is_executable"] is False

    def test_batch_validate_upload_payloads(self):
        uploads = [
            (b"", "empty.txt"),
            (b"MZ\x90\x00...", "disguised.pdf"),
            (b"Valid text content", "sample.txt"),
        ]
        reports = batch_validate_upload_payloads(uploads)
        assert len(reports) == 3
        assert reports[0]["status"] == "rejected_empty"
        assert reports[1]["status"] == "rejected_executable"


class TestThreatVectorAndStreamAuditing:
    """Test suite for threat vector classification and stream auditing."""

    def test_classify_upload_threat_vector_empty(self):
        res = classify_upload_threat_vector(b"", "doc.pdf")
        assert res["threat_category"] == "EMPTY_PAYLOAD"
        assert res["is_dangerous"] is False

    def test_classify_upload_threat_vector_executable(self):
        res = classify_upload_threat_vector(b"MZ\x90\x00", "thesis.pdf")
        assert res["threat_category"] == "MAGIC_BYTE_MISMATCH"
        assert res["is_dangerous"] is True

    def test_classify_upload_threat_vector_extension(self):
        res = classify_upload_threat_vector(b"some content", "file.exe")
        assert res["threat_category"] == "EXECUTABLE_EXTENSION"
        assert res["is_dangerous"] is True

    def test_audit_file_stream_safety_empty(self):
        res = audit_file_stream_safety([], "doc.pdf")
        assert res["is_safe"] is False
        assert res["total_bytes"] == 0

    def test_audit_file_stream_safety_safe_chunks(self):
        chunks = [b"Chunk 1 text content\n", b"Chunk 2 text content\n"]
        res = audit_file_stream_safety(chunks, "doc.txt")
        assert res["is_safe"] is True
        assert res["total_bytes"] > 0

    def test_validate_upload_payload_safety_empty(self):
        is_safe, msg = validate_upload_payload_safety(b"", "sample.pdf")
        assert is_safe is False
        assert "empty" in msg.lower()
