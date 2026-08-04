import pytest

from src.utils.filename import (
    InvalidFileExtensionError,
    get_final_extension,
    sanitize_and_validate_filename,
    validate_document_extension,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("document.pdf", ".pdf"),
        ("DOCUMENT.PDF", ".pdf"),
        ("archive.name.docx", ".docx"),
        (r"C:\uploads\report.txt", ".txt"),
        ("../../safe.csv", ".csv"),
    ],
)
def test_get_final_extension_uses_absolute_last_suffix(
    filename,
    expected,
):
    assert get_final_extension(filename) == expected


@pytest.mark.parametrize(
    "filename",
    [
        "document.pdf.exe",
        "report.docx.bat",
        "notes.txt.cmd",
        "paper.PDF.EXE",
        "submission.csv.ps1",
        "archive.zip.jar",
        "image.pdf.js",
        "payload.scr",
    ],
)
def test_executable_final_extensions_are_rejected(filename):
    with pytest.raises(
        InvalidFileExtensionError,
        match="Executable or script",
    ):
        validate_document_extension(filename)


@pytest.mark.parametrize(
    "filename",
    [
        "document.pdf",
        "report.docx",
        "notes.txt",
        "records.csv",
        "bundle.zip",
    ],
)
def test_supported_final_extensions_are_accepted(filename):
    assert validate_document_extension(filename).startswith(".")


def test_unsupported_but_non_executable_extension_is_rejected():
    with pytest.raises(
        InvalidFileExtensionError,
        match="Unsupported final file extension",
    ):
        validate_document_extension("document.png")


def test_extensionless_file_is_rejected():
    with pytest.raises(
        InvalidFileExtensionError,
        match="must have a supported extension",
    ):
        validate_document_extension("README")


def test_custom_allowed_extension_set_is_respected():
    assert validate_document_extension(
        "chapter.epub",
        allowed_extensions={".epub"},
    ) == ".epub"

    with pytest.raises(InvalidFileExtensionError):
        validate_document_extension(
            "chapter.pdf",
            allowed_extensions={".epub"},
        )


def test_validation_occurs_before_sanitization():
    with pytest.raises(InvalidFileExtensionError):
        sanitize_and_validate_filename("document.pdf.exe")


def test_safe_filename_is_sanitized_after_validation():
    assert sanitize_and_validate_filename(
        "<b>Final Report</b>.PDF"
    ) == "Final_Report.pdf"
