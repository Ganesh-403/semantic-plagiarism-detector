import io
import zipfile
from unittest.mock import patch

import pytest

from src.security.mime_validator import (
    is_executable_upload,
    validate_mime_type,
)

CONTENT_TYPES_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Override PartName="{part}" ContentType="{content_type}"/>
</Types>
"""


def build_zip(members: dict[str, bytes | str]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return output.getvalue()


def build_docx() -> bytes:
    return build_zip(
        {
            "[Content_Types].xml": CONTENT_TYPES_TEMPLATE.format(
                part="/word/document.xml",
                content_type=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document.main+xml"
                ),
            ),
            "_rels/.rels": "<Relationships/>",
            "word/document.xml": "<w:document/>",
        }
    )


def build_xlsx() -> bytes:
    return build_zip(
        {
            "[Content_Types].xml": CONTENT_TYPES_TEMPLATE.format(
                part="/xl/workbook.xml",
                content_type=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet.main+xml"
                ),
            ),
            "_rels/.rels": "<Relationships/>",
            "xl/workbook.xml": "<workbook/>",
        }
    )


def test_validate_mime_type_pdf():
    assert (
        validate_mime_type(
            b"%PDF-1.4\n%...\n",
            "test.pdf",
        )
        is True
    )
    assert (
        validate_mime_type(
            b"MZ\x90\x00\x03\x00\x00\x00",
            "malicious.pdf",
        )
        is False
    )


def test_executable_magic_byte_detection():
    """Issue #3720: Test is_executable_upload with files starting with b'MZ' and b'#!/bin/sh'."""
    assert is_executable_upload(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff", "assignment.pdf") is True
    assert is_executable_upload(b"#!/bin/sh\nrm -rf /", "notes.txt") is True
    assert is_executable_upload(b"MZ", "report.docx") is True
    assert is_executable_upload(b"#!/bin/sh", "thesis.md") is True


def test_valid_docx_package_is_accepted():
    assert (
        validate_mime_type(
            build_docx(),
            "report.docx",
        )
        is True
    )


def test_valid_xlsx_package_is_accepted():
    assert (
        validate_mime_type(
            build_xlsx(),
            "report.xlsx",
        )
        is True
    )


def test_standard_zip_is_not_accepted_as_docx_or_xlsx():
    ordinary_zip = build_zip({"notes.txt": "ordinary archive"})

    assert (
        validate_mime_type(
            ordinary_zip,
            "renamed.docx",
        )
        is False
    )
    assert (
        validate_mime_type(
            ordinary_zip,
            "renamed.xlsx",
        )
        is False
    )
    assert (
        validate_mime_type(
            ordinary_zip,
            "archive.zip",
        )
        is True
    )


def test_docx_is_not_accepted_as_xlsx_and_vice_versa():
    assert (
        validate_mime_type(
            build_docx(),
            "wrong.xlsx",
        )
        is False
    )
    assert (
        validate_mime_type(
            build_xlsx(),
            "wrong.docx",
        )
        is False
    )


def test_missing_content_types_is_rejected():
    archive = build_zip({"word/document.xml": "<w:document/>"})

    assert (
        validate_mime_type(
            archive,
            "report.docx",
        )
        is False
    )


def test_missing_main_part_is_rejected():
    archive = build_zip(
        {
            "[Content_Types].xml": CONTENT_TYPES_TEMPLATE.format(
                part="/word/document.xml",
                content_type=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document.main+xml"
                ),
            )
        }
    )

    assert (
        validate_mime_type(
            archive,
            "report.docx",
        )
        is False
    )


def test_malformed_content_types_xml_is_rejected():
    archive = build_zip(
        {
            "[Content_Types].xml": "<Types>",
            "word/document.xml": "<w:document/>",
        }
    )

    assert (
        validate_mime_type(
            archive,
            "report.docx",
        )
        is False
    )


def test_wrong_declared_content_type_is_rejected():
    archive = build_zip(
        {
            "[Content_Types].xml": CONTENT_TYPES_TEMPLATE.format(
                part="/word/document.xml",
                content_type="application/octet-stream",
            ),
            "word/document.xml": "<w:document/>",
        }
    )

    assert (
        validate_mime_type(
            archive,
            "report.docx",
        )
        is False
    )


def test_truncated_zip_is_rejected():
    assert (
        validate_mime_type(
            b"PK\x03\x04truncated",
            "report.docx",
        )
        is False
    )


def test_weak_pk_signature_without_full_magic_bytes_is_rejected():
    assert (
        validate_mime_type(
            b"PK\x05\x06" + b"\x00" * 18,
            "empty.docx",
        )
        is False
    )


def test_too_many_archive_members_are_rejected(monkeypatch):
    monkeypatch.setattr(
        "src.security.mime_validator.MAX_OOXML_ARCHIVE_ENTRIES",
        2,
    )

    assert (
        validate_mime_type(
            build_docx(),
            "report.docx",
        )
        is False
    )


def test_ooxml_validation_does_not_trust_python_magic(monkeypatch):
    called = []

    def mock_check(*args, **kwargs):
        called.append(True)
        return True

    monkeypatch.setattr("src.security.mime_validator._check_magic_bytes", mock_check)
    assert (
        validate_mime_type(
            build_zip({"random.bin": b"123"}),
            "spoofed.docx",
        )
        is False
    )
    assert len(called) == 0


def test_validate_mime_type_text():
    content = b"Hello World! This is essay content."
    assert validate_mime_type(content, "essay.txt") is True
    assert validate_mime_type(content, "essay.md") is True
    assert validate_mime_type(content, "data.csv") is True
    assert (
        validate_mime_type(
            b"\x00\xff\xfe\xffHello",
            "essay.txt",
        )
        is False
    )


def test_validate_mime_type_empty():
    assert validate_mime_type(b"", "empty.pdf") is False


def test_validate_mime_type_unsupported_extension():
    assert (
        validate_mime_type(
            b"some content",
            "file.exe",
        )
        is False
    )


def test_validate_mime_type_magic_fallback(monkeypatch):
    monkeypatch.setattr(
        "src.security.mime_validator._check_magic_bytes",
        lambda *args, **kwargs: None,
    )
    assert (
        validate_mime_type(
            b"%PDF-1.4\n%...\n",
            "test.pdf",
        )
        is True
    )
    assert (
        validate_mime_type(
            b"MZ\x90\x00\x03\x00\x00\x00",
            "malicious.pdf",
        )
        is False
    )


def test_validate_mime_type_accepts_valid_legacy_doc_header(monkeypatch):
    """A .doc file with the complete OLE signature is accepted."""
    valid_doc = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1\x00" * 64
    monkeypatch.setattr(
        "src.security.mime_validator._check_magic_bytes",
        lambda *_args: None,
    )

    assert validate_mime_type(valid_doc, "legacy.doc") is True


def test_validate_mime_type_rejects_invalid_legacy_doc_header(monkeypatch):
    """A renamed payload without OLE bytes must be rejected."""
    monkeypatch.setattr(
        "src.security.mime_validator._check_magic_bytes",
        lambda *_args: True,
    )

    assert (
        validate_mime_type(
            b"not-an-ole-compound-document",
            "malicious.doc",
        )
        is False
    )


def test_validate_mime_type_rejects_truncated_legacy_doc_header(monkeypatch):
    """The old four-byte prefix alone is insufficient."""
    monkeypatch.setattr(
        "src.security.mime_validator._check_magic_bytes",
        lambda *_args: None,
    )

    assert (
        validate_mime_type(
            b"\xd0\xcf\x11\xe0",
            "truncated.doc",
        )
        is False
    )


def test_validate_mime_type_legacy_doc_extension_is_case_insensitive(monkeypatch):
    valid_doc = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1\x00" * 16
    monkeypatch.setattr(
        "src.security.mime_validator._check_magic_bytes",
        lambda *_args: None,
    )

    assert validate_mime_type(valid_doc, "REPORT.DOC") is True


@pytest.mark.parametrize(
    "declared_filename",
    [
        "assignment.pdf",
        "essay.docx",
        "analysis.xlsx",
        "notes.txt",
        "dataset.csv",
        "report.doc",
        "readme.md",
    ],
)
def test_pe_magic_bytes_detected_across_common_document_types(declared_filename):
    """Verify Windows PE header b'MZ' triggers executable detection regardless of declared extension."""
    pe_header = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff"
    assert is_executable_upload(pe_header, declared_filename) is True


@pytest.mark.parametrize(
    "declared_filename",
    [
        "assignment.pdf",
        "essay.docx",
        "analysis.xlsx",
        "notes.txt",
        "dataset.csv",
        "report.doc",
        "readme.md",
    ],
)
def test_shebang_magic_bytes_detected_across_common_document_types(declared_filename):
    """Verify Unix shebang b'#!/bin/sh' triggers executable detection regardless of declared extension."""
    shebang_header = b"#!/bin/sh\necho 'running malicious script'\n"
    assert is_executable_upload(shebang_header, declared_filename) is True


def test_legitimate_documents_not_flagged_as_executable():
    """Verify standard document contents return False in is_executable_upload."""
    assert is_executable_upload(b"%PDF-1.4\n1 0 obj\n", "assignment.pdf") is False
    assert is_executable_upload(b"PK\x03\x04\x14\x00\x00\x00", "essay.docx") is False
    assert is_executable_upload(b"Just plain text essay content.", "notes.txt") is False
    assert is_executable_upload(b"col1,col2,col3\n1,2,3\n", "dataset.csv") is False


def test_ooxml_xml_entities_forbidden():
    """Verify that OOXML archives containing XML DTDs or entities in [Content_Types].xml are rejected."""
    dtd_content_types = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Types [
  <!ENTITY lol "lol">
]>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    archive_with_dtd = build_zip(
        {
            "[Content_Types].xml": dtd_content_types,
            "_rels/.rels": "<Relationships/>",
            "word/document.xml": "<w:document/>",
        }
    )

    assert validate_mime_type(archive_with_dtd, "report.docx") is False


def build_docm() -> bytes:
    return build_zip(
        {
            "[Content_Types].xml": CONTENT_TYPES_TEMPLATE.format(
                part="/word/document.xml",
                content_type="application/vnd.ms-word.document.macroEnabled.main+xml",
            ),
            "_rels/.rels": "<Relationships/>",
            "word/document.xml": "<w:document/>",
        }
    )


def build_xlsm() -> bytes:
    return build_zip(
        {
            "[Content_Types].xml": CONTENT_TYPES_TEMPLATE.format(
                part="/xl/workbook.xml",
                content_type="application/vnd.ms-excel.sheet.macroEnabled.main+xml",
            ),
            "_rels/.rels": "<Relationships/>",
            "xl/workbook.xml": "<workbook/>",
        }
    )


def test_macro_enabled_documents_rejected_by_default():
    # 1. Blocked extension checks
    assert is_executable_upload(b"PK\x03\x04", "report.docm") is True
    assert is_executable_upload(b"PK\x03\x04", "report.xlsm") is True

    # 2. validate_mime_type checks
    assert validate_mime_type(build_docm(), "report.docm") is False
    assert validate_mime_type(build_xlsm(), "report.xlsm") is False

    # 3. Spoofed docx/xlsx carrying macros inside
    spoofed_docx = build_docm()  # built as docm but named as docx
    assert validate_mime_type(spoofed_docx, "spoofed.docx") is False

    spoofed_xlsx = build_xlsm()  # built as xlsm but named as xlsx
    assert validate_mime_type(spoofed_xlsx, "spoofed.xlsx") is False


def test_macro_enabled_documents_allowed_if_configured():
    # 1. Extension check when macros allowed
    assert is_executable_upload(b"PK\x03\x04", "report.docm", allow_macros=True) is False
    assert is_executable_upload(b"PK\x03\x04", "report.xlsm", allow_macros=True) is False

    # 2. validate_mime_type checks
    assert validate_mime_type(build_docm(), "report.docm", allow_macros=True) is True
    assert validate_mime_type(build_xlsm(), "report.xlsm", allow_macros=True) is True

    # 3. Spoofed docx/xlsx carrying macros accepted if macros allowed
    spoofed_docx = build_docm()
    assert validate_mime_type(spoofed_docx, "spoofed.docx", allow_macros=True) is True


def test_ooxml_oversized_content_types_rejected():
    """Verify that OOXML archives with [Content_Types].xml exceeding 2MB are rejected."""
    # Create an oversized [Content_Types].xml (3MB) using comment padding
    padding_size = 3 * 1024 * 1024
    oversized_content = "<!-- " + ("A" * padding_size) + " -->\n" + CONTENT_TYPES_TEMPLATE.format(
        part="/word/document.xml",
        content_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document.main+xml"
        ),
    )

    archive = build_zip(
        {
            "[Content_Types].xml": oversized_content,
            "_rels/.rels": "<Relationships/>",
            "word/document.xml": "<w:document/>",
        }
    )

    from src.security.mime_validator import _validate_ooxml_archive

    assert _validate_ooxml_archive(archive, "docx", "report.docx") is False
    assert validate_mime_type(archive, "report.docx") is False



