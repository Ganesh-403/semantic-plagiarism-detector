import io
import zipfile
from unittest.mock import patch


from src.security.mime_validator import (
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
    assert validate_mime_type(
        b"%PDF-1.4\n%...\n",
        "test.pdf",
    ) is True
    assert validate_mime_type(
        b"MZ\x90\x00\x03\x00\x00\x00",
        "malicious.pdf",
    ) is False


def test_valid_docx_package_is_accepted():
    assert validate_mime_type(
        build_docx(),
        "report.docx",
    ) is True


def test_valid_xlsx_package_is_accepted():
    assert validate_mime_type(
        build_xlsx(),
        "report.xlsx",
    ) is True


def test_standard_zip_is_not_accepted_as_docx_or_xlsx():
    ordinary_zip = build_zip(
        {"notes.txt": "ordinary archive"}
    )

    assert validate_mime_type(
        ordinary_zip,
        "renamed.docx",
    ) is False
    assert validate_mime_type(
        ordinary_zip,
        "renamed.xlsx",
    ) is False
    assert validate_mime_type(
        ordinary_zip,
        "archive.zip",
    ) is True


def test_docx_is_not_accepted_as_xlsx_and_vice_versa():
    assert validate_mime_type(
        build_docx(),
        "wrong.xlsx",
    ) is False
    assert validate_mime_type(
        build_xlsx(),
        "wrong.docx",
    ) is False


def test_missing_content_types_is_rejected():
    archive = build_zip(
        {"word/document.xml": "<w:document/>"}
    )

    assert validate_mime_type(
        archive,
        "report.docx",
    ) is False


def test_missing_main_part_is_rejected():
    archive = build_zip(
        {
            "[Content_Types].xml":
                CONTENT_TYPES_TEMPLATE.format(
                    part="/word/document.xml",
                    content_type=(
                        "application/vnd.openxmlformats-officedocument."
                        "wordprocessingml.document.main+xml"
                    ),
                )
        }
    )

    assert validate_mime_type(
        archive,
        "report.docx",
    ) is False


def test_malformed_content_types_xml_is_rejected():
    archive = build_zip(
        {
            "[Content_Types].xml": "<Types>",
            "word/document.xml": "<w:document/>",
        }
    )

    assert validate_mime_type(
        archive,
        "report.docx",
    ) is False


def test_wrong_declared_content_type_is_rejected():
    archive = build_zip(
        {
            "[Content_Types].xml":
                CONTENT_TYPES_TEMPLATE.format(
                    part="/word/document.xml",
                    content_type="application/octet-stream",
                ),
            "word/document.xml": "<w:document/>",
        }
    )

    assert validate_mime_type(
        archive,
        "report.docx",
    ) is False


def test_truncated_zip_is_rejected():
    assert validate_mime_type(
        b"PK\x03\x04truncated",
        "report.docx",
    ) is False


def test_too_many_archive_members_are_rejected(monkeypatch):
    monkeypatch.setattr(
        "src.security.mime_validator."
        "MAX_OOXML_ARCHIVE_ENTRIES",
        2,
    )

    assert validate_mime_type(
        build_docx(),
        "report.docx",
    ) is False


def test_ooxml_validation_does_not_trust_python_magic():
    with patch(
        "magic.from_buffer",
        return_value="application/zip",
    ) as magic_mock:
        assert validate_mime_type(
            build_zip({"random.bin": b"123"}),
            "spoofed.docx",
        ) is False

    magic_mock.assert_not_called()


def test_validate_mime_type_text():
    content = b"Hello World! This is essay content."
    assert validate_mime_type(content, "essay.txt") is True
    assert validate_mime_type(content, "essay.md") is True
    assert validate_mime_type(content, "data.csv") is True
    assert validate_mime_type(
        b"\x00\xff\xfe\xffHello",
        "essay.txt",
    ) is False


def test_validate_mime_type_empty():
    assert validate_mime_type(b"", "empty.pdf") is False


def test_validate_mime_type_unsupported_extension():
    assert validate_mime_type(
        b"some content",
        "file.exe",
    ) is False


def test_validate_mime_type_magic_fallback():
    with patch(
        "magic.from_buffer",
        side_effect=ImportError("No magic module"),
    ):
        assert validate_mime_type(
            b"%PDF-1.4\n%...\n",
            "test.pdf",
        ) is True
        assert validate_mime_type(
            b"MZ\x90\x00\x03\x00\x00\x00",
            "malicious.pdf",
        ) is False


def test_ooxml_xxe_entity_expansion_safely_handled():
    """Verify that malicious XML entity expansion in [Content_Types].xml is safely handled (Issue #1709)."""
    malicious_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE Types [\n'
        '  <!ENTITY lol "lol">\n'
        '  <!ELEMENT Types (#PCDATA)>\n'
        '  <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">\n'
        ']>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>\n'
        '</Types>'
    )
    docx_bytes = build_zip({
        "[Content_Types].xml": malicious_xml,
        "_rels/.rels": "<Relationships/>",
        "word/document.xml": "<w:document/>",
    })
    # Defusedxml blocks entity expansion / DTD entity resolution gracefully
    result = validate_mime_type(docx_bytes, "malicious_bomb.docx")
    assert result is False

def test_validate_mime_type_accepts_valid_legacy_doc_header(monkeypatch):
    """A .doc file with the complete OLE signature is accepted."""
    valid_doc = (
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        b"\x00" * 64
    )
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

    assert validate_mime_type(
        b"not-an-ole-compound-document",
        "malicious.doc",
    ) is False


def test_validate_mime_type_rejects_truncated_legacy_doc_header(monkeypatch):
    """The old four-byte prefix alone is insufficient."""
    monkeypatch.setattr(
        "src.security.mime_validator._check_magic_bytes",
        lambda *_args: None,
    )

    assert validate_mime_type(
        b"\xd0\xcf\x11\xe0",
        "truncated.doc",
    ) is False


def test_validate_mime_type_legacy_doc_extension_is_case_insensitive(monkeypatch):
    valid_doc = (
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        b"\x00" * 16
    )
    monkeypatch.setattr(
        "src.security.mime_validator._check_magic_bytes",
        lambda *_args: None,
    )

    assert validate_mime_type(valid_doc, "REPORT.DOC") is True
