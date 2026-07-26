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

    assert len(result) == 255
    assert result.endswith(".pdf")


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
