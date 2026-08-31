"""Tests for Windows reserved device name handling in sanitize_filename (Issue #3725).

Tests that Windows reserved device names (CON, PRN, AUX, NUL, COM1-COM9, LPT1-LPT9)
both with and without extensions (e.g. NUL.txt, CON.pdf, COM1.docx) are safely
sanitized by prepending an underscore to prevent Win32 filesystem collisions and
denial-of-service vulnerabilities.
"""

from __future__ import annotations

import pytest

from src.utils.filename import (
    _WINDOWS_RESERVED_NAMES,
    is_windows_reserved_name,
    sanitize_filename,
    sanitize_filename_mapping,
    unique_filename,
)


class TestWindowsReservedDeviceNamesWithExtensions:
    """Test suite verifying Windows reserved device names with various extensions."""

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            # Acceptance Criteria specific cases
            ("NUL.txt", "_NUL.txt"),
            ("CON.pdf", "_CON.pdf"),
            ("COM1.docx", "_COM1.docx"),
            ("AUX.txt", "_AUX.txt"),
            ("PRN.pdf", "_PRN.pdf"),
            # Lowercase variations
            ("nul.txt", "_nul.txt"),
            ("con.pdf", "_con.pdf"),
            ("com1.docx", "_com1.docx"),
            ("aux.txt", "_aux.txt"),
            ("prn.pdf", "_prn.pdf"),
            # Mixed case variations
            ("NuL.Txt", "_NuL.txt"),
            ("cOn.PdF", "_cOn.pdf"),
            ("CoM1.DocX", "_CoM1.docx"),
            ("AuX.TXT", "_AuX.txt"),
            ("PrN.pDf", "_PrN.pdf"),
            # All COM ports (COM1 to COM9) with extensions
            ("COM1.txt", "_COM1.txt"),
            ("COM2.pdf", "_COM2.pdf"),
            ("COM3.docx", "_COM3.docx"),
            ("COM4.xlsx", "_COM4.xlsx"),
            ("COM5.csv", "_COM5.csv"),
            ("COM6.md", "_COM6.md"),
            ("COM7.rtf", "_COM7.rtf"),
            ("COM8.epub", "_COM8.epub"),
            ("COM9.odt", "_COM9.odt"),
            # All LPT ports (LPT1 to LPT9) with extensions
            ("LPT1.txt", "_LPT1.txt"),
            ("LPT2.pdf", "_LPT2.pdf"),
            ("LPT3.docx", "_LPT3.docx"),
            ("LPT4.xlsx", "_LPT4.xlsx"),
            ("LPT5.csv", "_LPT5.csv"),
            ("LPT6.md", "_LPT6.md"),
            ("LPT7.rtf", "_LPT7.rtf"),
            ("LPT8.epub", "_LPT8.epub"),
            ("LPT9.odt", "_LPT9.odt"),
            # Standalone reserved names without extensions
            ("CON", "_CON"),
            ("PRN", "_PRN"),
            ("AUX", "_AUX"),
            ("NUL", "_NUL"),
            ("COM1", "_COM1"),
            ("LPT1", "_LPT1"),
        ],
    )
    def test_reserved_device_names_with_extensions(self, filename: str, expected: str):
        """Verify that reserved device names with standard extensions are prepended with underscore."""
        sanitized = sanitize_filename(filename)
        assert sanitized == expected

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("NUL.tar.gz", "_NUL.tar.gz"),
            ("CON.backup.pdf", "_CON.backup.pdf"),
            ("COM1.v1.0.docx", "_COM1.v1.0.docx"),
            ("aux.test.spec.txt", "_aux.test.spec.txt"),
            ("PRN.old.version.doc", "_PRN.old.version.doc"),
            ("lpt3.draft.final.md", "_lpt3.draft.final.md"),
        ],
    )
    def test_reserved_device_names_with_compound_extensions(self, filename: str, expected: str):
        """Verify that compound or multiple dotted extensions maintain base stem protection."""
        sanitized = sanitize_filename(filename)
        assert sanitized == expected

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("../../NUL.txt", "_NUL.txt"),
            (r"..\\..\\CON.pdf", "_CON.pdf"),
            ("/tmp/uploads/COM1.docx", "_COM1.docx"),
            (r"C:\Windows\System32\PRN.txt", "_PRN.txt"),
            (r"D:\Data\Documents\AUX.pdf", "_AUX.pdf"),
            ("./nested/folder/LPT1.txt", "_LPT1.txt"),
        ],
    )
    def test_reserved_device_names_with_paths_and_traversals(self, filename: str, expected: str):
        """Verify path components and traversal attempts are stripped before reserved name check."""
        sanitized = sanitize_filename(filename)
        assert sanitized == expected

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("<b>NUL</b>.txt", "_NUL.txt"),
            ("<script>CON</script>.pdf", "_CON.pdf"),
            ("<a href='evil'>COM1</a>.docx", "_COM1.docx"),
            ("<span>AUX</span>.txt", "_AUX.txt"),
            ("<i>PRN</i>.csv", "_PRN.csv"),
        ],
    )
    def test_reserved_device_names_with_html_tags(self, filename: str, expected: str):
        """Verify HTML markup is removed prior to checking Windows reserved names."""
        sanitized = sanitize_filename(filename)
        assert sanitized == expected

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("  NUL  .txt", "_NUL.txt"),
            (" \t CON \n .pdf", "_CON.pdf"),
            ("__COM1__.docx", "_COM1.docx"),
            ("..AUX...txt", "_AUX.txt"),
            ("--PRN--.pdf", "_PRN.pdf"),
        ],
    )
    def test_reserved_device_names_with_surrounding_whitespace_and_punctuation(
        self, filename: str, expected: str
    ):
        """Verify surrounding whitespace and punctuation are trimmed properly."""
        sanitized = sanitize_filename(filename)
        assert sanitized == expected


class TestLegitimateFilenamesNotMangled:
    """Test suite ensuring legitimate filenames containing reserved keywords are untouched."""

    @pytest.mark.parametrize(
        "filename",
        [
            "contact.pdf",
            "console.txt",
            "condition.docx",
            "continue.csv",
            "connect.md",
            "context.json",
            "printer.pdf",
            "printing.docx",
            "prno.txt",
            "auxiliary.pdf",
            "auxin.docx",
            "aux_data.txt",
            "nullify.txt",
            "null_value.csv",
            "numeric.pdf",
            "common.docx",
            "communication.pdf",
            "computer.txt",
            "commercial.docx",
            "compact.md",
            "laptop.pdf",
            "option.docx",
            "adoption.txt",
            "description.md",
            "prompt.pdf",
        ],
    )
    def test_words_starting_with_reserved_stems_are_not_prefixed(self, filename: str):
        """Words that begin with or contain reserved stems (like 'contact', 'printer') must not be altered."""
        assert sanitize_filename(filename) == filename

    @pytest.mark.parametrize(
        "stem",
        [
            "con_report.pdf",
            "prn_output.txt",
            "aux_channel.docx",
            "nul_byte.csv",
            "com1_port.json",
            "lpt1_driver.pdf",
        ],
    )
    def test_reserved_stems_with_subsequent_text_are_not_prefixed(self, stem: str):
        """Separated stems like 'con_report' have different tokens and must not trigger leading underscore."""
        assert sanitize_filename(stem) == stem


class TestIsWindowsReservedNameHelper:
    """Direct unit tests for the is_windows_reserved_name function."""

    def test_all_official_reserved_names_detected(self):
        """Ensure all names in _WINDOWS_RESERVED_NAMES are identified."""
        for name in _WINDOWS_RESERVED_NAMES:
            assert is_windows_reserved_name(name) is True
            assert is_windows_reserved_name(name.lower()) is True
            assert is_windows_reserved_name(f"{name}.txt") is True
            assert is_windows_reserved_name(f"{name.lower()}.pdf") is True
            assert is_windows_reserved_name(f"{name}.tar.gz") is True

    def test_none_and_empty_inputs(self):
        """None, empty string, and whitespace return False."""
        assert is_windows_reserved_name("") is False
        assert is_windows_reserved_name(None) is False  # type: ignore[arg-type]
        assert is_windows_reserved_name("   ") is False
        assert is_windows_reserved_name("...") is False

    def test_non_reserved_names_return_false(self):
        """Standard document stems return False."""
        safe_stems = [
            "thesis",
            "report_2026",
            "final_draft",
            "student_submission",
            "plagiarism_analysis",
            "document",
        ]
        for stem in safe_stems:
            assert is_windows_reserved_name(stem) is False
            assert is_windows_reserved_name(f"{stem}.pdf") is False


class TestUniqueFilenameAndMappingWithReservedNames:
    """Tests verifying unique_filename and sanitize_filename_mapping with reserved names."""

    def test_unique_filename_with_reserved_name(self):
        """Verify unique_filename correctly disambiguates multiple reserved filenames."""
        existing = {"_NUL.txt"}
        result = unique_filename("NUL.txt", existing)
        assert result == "_NUL_1.txt"

    def test_unique_filename_multiple_collisions(self):
        """Verify sequential numbering on collision with sanitized reserved names."""
        existing = {"_CON.pdf", "_CON_1.pdf", "_CON_2.pdf"}
        result = unique_filename("CON.pdf", existing)
        assert result == "_CON_3.pdf"

    def test_sanitize_filename_mapping_reserved_names(self):
        """Verify sanitize_filename_mapping safely processes dictionary of files."""
        files = {
            "NUL.txt": b"test content 1",
            "CON.pdf": b"test content 2",
            "COM1.docx": b"test content 3",
            "regular.txt": b"test content 4",
        }
        sanitized_map = sanitize_filename_mapping(files)
        assert "_NUL.txt" in sanitized_map
        assert "_CON.pdf" in sanitized_map
        assert "_COM1.docx" in sanitized_map
        assert "regular.txt" in sanitized_map
        assert sanitized_map["_NUL.txt"] == b"test content 1"
        assert sanitized_map["_CON.pdf"] == b"test content 2"
        assert sanitized_map["_COM1.docx"] == b"test content 3"
        assert sanitized_map["regular.txt"] == b"test content 4"

    def test_sanitize_filename_mapping_collision_resolution(self):
        """Verify duplicate reserved names in mapping are disambiguated."""
        files = {
            "NUL.txt": b"first",
            "nul.txt": b"second",
        }
        sanitized_map = sanitize_filename_mapping(files)
        assert len(sanitized_map) == 2
        assert "_NUL.txt" in sanitized_map
        assert "_nul_1.txt" in sanitized_map or "_NUL_1.txt" in sanitized_map


class TestMaxFilenameLengthConstraintsWithReservedNames:
    """Tests ensuring max_length boundary constraints are preserved when prepending underscore."""

    def test_reserved_name_at_short_max_length(self):
        """Verify reserved name sanitization respects small max_length limits."""
        result = sanitize_filename("NUL.txt", max_length=10)
        assert result == "_NUL.txt"
        assert len(result) <= 10

    def test_reserved_name_at_exact_boundary(self):
        """Verify reserved name with exact length fits within max_length."""
        result = sanitize_filename("COM1.docx", max_length=10)
        assert result == "_COM1.docx"
        assert len(result) == 10

    def test_reserved_name_with_long_compound_extension_truncation(self):
        """Verify long compound extensions are truncated while maintaining safety."""
        result = sanitize_filename("NUL.verylongextensionnamethatexceedslimit.txt", max_length=20)
        assert len(result) <= 20
        assert result.endswith(".txt")
        assert result.startswith("_")


class TestSpecialCharactersAndUnicodeWithReservedNames:
    """Tests ensuring special characters and unicode variations around reserved names are secure."""

    @pytest.mark.parametrize(
        ("raw_input", "expected_prefix"),
        [
            ("NUL\x00.txt", "_NUL.txt"),
            ("CON\r\n.pdf", "_CON.pdf"),
            ("COM1\t.docx", "_COM1.docx"),
            ("PRN\x1f.txt", "_PRN.txt"),
            ("AUX\x7f.pdf", "_AUX.pdf"),
        ],
    )
    def test_control_characters_in_reserved_names_stripped_and_sanitized(
        self, raw_input: str, expected_prefix: str
    ):
        """Control characters are stripped, then the reserved name is sanitized safely."""
        sanitized = sanitize_filename(raw_input)
        assert sanitized == expected_prefix

    @pytest.mark.parametrize(
        "reserved_name",
        [
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        ],
    )
    def test_all_reserved_names_with_common_extensions_never_unsanitized(
        self, reserved_name: str
    ):
        """Parametrized invariant test: for every reserved name and extension, output starts with underscore."""
        extensions = [".pdf", ".docx", ".txt", ".csv", ".xlsx", ".md", ".rtf", ".odt", ".epub"]
        for ext in extensions:
            filename = f"{reserved_name}{ext}"
            sanitized = sanitize_filename(filename)
            assert sanitized.startswith("_"), f"Failed for {filename}: got {sanitized}"
            assert sanitized.endswith(ext)

            lower_filename = f"{reserved_name.lower()}{ext}"
            sanitized_lower = sanitize_filename(lower_filename)
            assert sanitized_lower.startswith("_"), f"Failed for {lower_filename}: got {sanitized_lower}"
            assert sanitized_lower.endswith(ext)


class TestWin32NamespaceAndDevicePrefixEdgeCases:
    """Tests checking Win32 device namespace prefixes and UNC device paths."""

    @pytest.mark.parametrize(
        ("device_path", "expected"),
        [
            (r"\\.\NUL.txt", "_NUL.txt"),
            (r"\\?\CON.pdf", "_CON.pdf"),
            (r"\\.\COM1.docx", "_COM1.docx"),
            (r"\\.\PRN.txt", "_PRN.txt"),
            (r"\\.\AUX.pdf", "_AUX.pdf"),
        ],
    )
    def test_win32_device_prefixes_sanitized_to_safe_stems(
        self, device_path: str, expected: str
    ):
        """Win32 namespace prefixes are stripped down to basename and sanitized."""
        sanitized = sanitize_filename(device_path)
        assert sanitized == expected

