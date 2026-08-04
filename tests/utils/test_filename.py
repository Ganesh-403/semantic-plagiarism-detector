import pytest
from unittest.mock import patch


from src.utils.filename import (_safe_extension, get_file_extension_sanitized,
                                sanitize_filename, sanitize_filename_mapping,
                                unique_filename)

from src.utils.filename import (
    _safe_extension,
    get_file_sha256_hash,
    sanitize_filename,
    sanitize_filename_mapping,
    unique_filename,
)



@pytest.mark.parametrize(
    ("untrusted", "expected"),
    [
        ("<script>alert(1)</script>.pdf", "alert_1.pdf"),
        ("<img src=x onerror=alert(1)>.docx", "document.docx"),
        ("../../grades.pdf", "grades.pdf"),
        (r"..\\..\\grades.pdf", "grades.pdf"),
        ("assignment\x00.pdf", "assignment.pdf"),
        ("  final report  .PDF", "final_report.pdf"),
        ("name&copy;.txt", "name.txt"),
        ("CON.pdf", "_CON.pdf"),
        ("", "document"),
        (None, "document"),
    ],
)
def test_sanitize_filename_security_cases(untrusted, expected):
    assert sanitize_filename(untrusted) == expected



@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("report.PDF", ".pdf"),
        ("essay.Docx", ".docx"),
        ("notes.txt", ".txt"),
        ("archive.TAR.GZ", ".gz"),
        ("no_extension", ""),
        ("", ""),
    ],
)
def test_get_file_extension_sanitized(filename, expected):
    assert get_file_extension_sanitized(filename) == expected


def test_sanitized_filename_contains_no_html_or_path_separators():    result = sanitize_filename(
        '<svg/onload=alert(1)>../../evil "file".pdf'
    )

def test_sanitized_filename_contains_no_html_or_path_separators():
    result = sanitize_filename('<svg/onload=alert(1)>../../evil "file".pdf')


    assert "<" not in result
    assert ">" not in result
    assert '"' not in result
    assert "/" not in result
    assert "\\" not in result
    assert ".." not in result


def test_extension_is_preserved_and_normalized():
    assert sanitize_filename("My File.PdF") == "My_File.pdf"


def test_long_filename_preserves_extension_and_limit():
    result = sanitize_filename("a" * 400 + ".pdf")

    assert len(result) == 150
    assert result.endswith(".pdf")


def test_300_plus_character_filename_truncation_and_hash_uniqueness():
    file1 = "a" * 300 + "_doc1.pdf"
    file2 = "a" * 300 + "_doc2.pdf"

    sanitized1 = sanitize_filename(file1)
    sanitized2 = sanitize_filename(file2)

    assert len(sanitized1) <= 150
    assert len(sanitized2) <= 150
    assert sanitized1.endswith(".pdf")
    assert sanitized2.endswith(".pdf")
    assert sanitized1 != sanitized2


def test_custom_max_length_truncation_with_hash():
    file_input = "b" * 350 + ".docx"
    sanitized = sanitize_filename(file_input, max_length=50)

    assert len(sanitized) <= 50
    assert sanitized.endswith(".docx")


def test_unique_filename_resolves_case_insensitive_collisions():
    existing = {"report.pdf", "report_1.pdf"}

    assert unique_filename("REPORT.PDF", existing) == "REPORT_2.pdf"


def test_mapping_preserves_entries_after_sanitization_collision():
    files = {
        "<b>report</b>.pdf": b"one",
        "report.pdf": b"two",
    }

    result = sanitize_filename_mapping(files)

    assert result == {
        "report.pdf": b"one",
        "report_1.pdf": b"two",
    }


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (".PDF", ".pdf"),
        (".DocX", ".docx"),
        ("file.txt", ".txt"),
        ("no_extension", ""),
    ],
)
def test_get_file_extension_sanitized(filename, expected):
    """get_file_extension_sanitized returns lowercase, well-formed extensions."""
    assert _safe_extension(filename) == expected


def test_get_file_extension_sanitized_handles_empty_filename():
    assert _safe_extension("") == ""


@pytest.mark.parametrize("value", [True, 7.5, "255"])
def test_invalid_max_length_type_is_rejected(value):
    with pytest.raises(TypeError):
        sanitize_filename("file.pdf", max_length=value)


# ---------------------------------------------------------------------------
# Cross-platform path separator compatibility tests  (#870)
# ---------------------------------------------------------------------------


@patch("src.utils.filename.os.name", "nt")
@patch("src.utils.filename.os.path.sep", "\\")
def test_sanitize_filename_windows_style_path_under_nt_mock():
    """Windows absolute path stripped to a bare filename with no separators."""
    result = sanitize_filename(r"C:\Users\attacker\..\secret.pdf")

    assert "\\" not in result
    assert "/" not in result
    assert result == "secret.pdf"


@patch("src.utils.filename.os.name", "nt")
@patch("src.utils.filename.os.path.sep", "\\")
def test_sanitize_filename_deep_windows_traversal_under_nt_mock():
    """Deep Windows traversal collapsed to the leaf filename."""
    result = sanitize_filename(r"D:\work\projects\..\..\sensitive\report.docx")

    assert "\\" not in result
    assert "/" not in result
    assert result == "report.docx"


@patch("src.utils.filename.os.name", "posix")
@patch("src.utils.filename.os.path.sep", "/")
def test_sanitize_filename_posix_absolute_path_under_posix_mock():
    """POSIX absolute path stripped to the leaf filename."""
    result = sanitize_filename("/etc/passwd")

    assert "/" not in result
    assert "\\" not in result
    assert result == "passwd"


@patch("src.utils.filename.os.name", "nt")
@patch("src.utils.filename.os.path.sep", "\\")
def test_sanitize_filename_mixed_separators_under_nt_mock():
    """Input mixing forward and back slashes resolved to a flat filename."""
    result = sanitize_filename("uploads/2024\\report.pdf")

    assert "/" not in result
    assert "\\" not in result
    assert result == "report.pdf"


@patch("src.utils.filename.os.name", "nt")
@patch("src.utils.filename.os.path.sep", "\\")
def test_unique_filename_no_separator_in_output_under_nt_mock():
    """unique_filename() returns a separator-free name even under mocked Windows."""
    existing = {"report.pdf", "report_1.pdf"}
    result = unique_filename(r"C:\Users\user\report.pdf", existing)

    assert "\\" not in result
    assert "/" not in result
    assert result == "report_2.pdf"


@patch("src.utils.filename.os.name", "posix")
@patch("src.utils.filename.os.path.sep", "/")
def test_unique_filename_no_separator_in_output_under_posix_mock():
    """unique_filename() returns a separator-free name even under mocked POSIX."""
    existing = {"report.pdf"}
    result = unique_filename("/home/user/docs/report.pdf", existing)

    assert "/" not in result
    assert "\\" not in result
    assert result == "report_1.pdf"


def test_get_file_sha256_hash_returns_expected_digest():
    file_bytes = b"hello world"

    assert (
        get_file_sha256_hash(file_bytes)
        == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )


def test_get_file_sha256_hash_returns_64_character_hex_digest():
    digest = get_file_sha256_hash(b"test")

    assert len(digest) == 64
    assert digest == digest.lower()
