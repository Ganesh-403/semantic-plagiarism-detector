import pytest

from src.utils.filename import (sanitize_filename, sanitize_filename_mapping,
                                unique_filename)


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


def test_sanitized_filename_contains_no_html_or_path_separators():
    result = sanitize_filename(
        '<svg/onload=alert(1)>../../evil "file".pdf'
    )

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

    assert len(result) == 128
    assert result.endswith(".pdf")


def test_300_plus_character_filename_truncation_and_hash_uniqueness():
    file1 = "a" * 300 + "_doc1.pdf"
    file2 = "a" * 300 + "_doc2.pdf"

    sanitized1 = sanitize_filename(file1)
    sanitized2 = sanitize_filename(file2)

    assert len(sanitized1) <= 128
    assert len(sanitized2) <= 128
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


@pytest.mark.parametrize("value", [True, 7.5, "255"])
def test_invalid_max_length_type_is_rejected(value):
    with pytest.raises(TypeError):
        sanitize_filename("file.pdf", max_length=value)

def test_200_character_filename_is_truncated_preserving_extension():
    filename = "a" * 200 + ".pdf"

    sanitized = sanitize_filename(filename)

    assert len(sanitized) <= 128
    assert sanitized.endswith(".pdf")
