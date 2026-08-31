"""Regression tests for the OOXML archive exception handler (Issue #2196).

The ``except`` clause of ``_validate_ooxml_archive()`` was dedented out of its
``try:`` block by a bad merge, which made the whole module fail to parse. Every
import of ``src.security.mime_validator`` raised ``SyntaxError``, taking the
upload security gate down with it.

These tests lock in two things:

1. The module imports and compiles at all.
2. Each exception type listed in the handler is actually caught and converted
   into a ``False`` verdict, rather than escaping to the caller. A dedented
   handler still "looks right" when skimming a diff, so the behavioural
   assertions below are what really pin it down.
"""

from __future__ import annotations

import ast
import inspect
import io
import zipfile

import pytest
from defusedxml.common import DefusedXmlException

from src.security import mime_validator
from src.security.mime_validator import (
    OOXML_EXTENSIONS,
    _validate_ooxml_archive,
    validate_mime_type,
)

CONTENT_TYPES_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Override PartName="{part}" ContentType="{content_type}"/>
</Types>
"""

DOCX_MAIN_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
)


def build_zip(members: dict[str, bytes | str]) -> bytes:
    """Build an in-memory ZIP archive from a name -> content mapping."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return output.getvalue()


def build_docx() -> bytes:
    """Build a minimal but structurally valid DOCX package."""
    return build_zip(
        {
            "[Content_Types].xml": CONTENT_TYPES_TEMPLATE.format(
                part="/word/document.xml",
                content_type=DOCX_MAIN_CONTENT_TYPE,
            ),
            "_rels/.rels": "<Relationships/>",
            "word/document.xml": "<w:document/>",
        }
    )


# ── The module must parse and import ─────────────────────────────────────────


def test_module_source_compiles():
    """The module source must be syntactically valid Python.

    Guards against the exact failure mode of #2196: a dedented ``except``
    clause that leaves the preceding ``try:`` without a handler.
    """
    source = inspect.getsource(mime_validator)

    # Raises SyntaxError on the broken revision.
    ast.parse(source)


def test_validate_ooxml_archive_try_block_has_a_handler():
    """``_validate_ooxml_archive`` must contain a ``try``/``except`` pair.

    Parsing the function's AST proves the handler is attached to the ``try``
    rather than sitting at module level next to it.
    """
    source = inspect.getsource(_validate_ooxml_archive)
    tree = ast.parse(source)

    try_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
    assert try_nodes, "_validate_ooxml_archive should guard archive parsing"
    assert any(node.handlers for node in try_nodes), (
        "the try block must have at least one except handler attached"
    )


# ── The handler must actually catch ──────────────────────────────────────────


def test_non_zip_payload_returns_false_instead_of_raising():
    """A ``PK``-prefixed payload that is not a real ZIP must be rejected.

    ``zipfile.ZipFile`` raises ``BadZipFile`` here. Without a reachable
    handler that exception escapes to the caller.
    """
    payload = b"PK\x03\x04" + b"not actually a zip archive"

    assert _validate_ooxml_archive(payload, "docx", "fake.docx") is False


def test_non_zip_payload_rejected_through_public_entry_point():
    """The same rejection must hold through ``validate_mime_type``."""
    payload = b"PK\x03\x04" + b"junk"

    assert validate_mime_type(payload, "fake.docx") is False


def test_truncated_zip_payload_returns_false():
    """A DOCX truncated mid-archive must be rejected, not raise."""
    truncated = build_docx()[: len(build_docx()) // 2]

    assert _validate_ooxml_archive(truncated, "docx", "truncated.docx") is False


def test_malformed_content_types_xml_returns_false():
    """Unparseable ``[Content_Types].xml`` raises ``ParseError`` — must be caught."""
    payload = build_zip(
        {
            "[Content_Types].xml": "<Types><unclosed>",
            "word/document.xml": "<w:document/>",
        }
    )

    assert _validate_ooxml_archive(payload, "docx", "broken-xml.docx") is False


def test_defused_xml_exception_is_caught(monkeypatch):
    """``DefusedXmlException`` from the XML hardening layer must be caught.

    It was added to the handler tuple but, with the handler unreachable, the
    addition never took effect.
    """

    def raise_defused(*_args, **_kwargs):
        raise DefusedXmlException("entity expansion blocked")

    monkeypatch.setattr(
        mime_validator.ElementTree,
        "fromstring",
        raise_defused,
    )

    assert _validate_ooxml_archive(build_docx(), "docx", "bomb.docx") is False


def test_os_error_while_reading_member_is_caught(monkeypatch):
    """``OSError`` raised while reading an archive member must be caught."""
    # Build the payload before patching — writestr() goes through ZipFile.open.
    payload = build_docx()

    def raise_os_error(*_args, **_kwargs):
        raise OSError("simulated read failure")

    monkeypatch.setattr(zipfile.ZipFile, "open", raise_os_error)

    assert _validate_ooxml_archive(payload, "docx", "unreadable.docx") is False


def test_runtime_error_from_testzip_is_caught(monkeypatch):
    """``RuntimeError`` (e.g. encrypted member) must be caught."""

    def raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("File is encrypted, password required for extraction")

    monkeypatch.setattr(zipfile.ZipFile, "testzip", raise_runtime_error)

    assert _validate_ooxml_archive(build_docx(), "docx", "encrypted.docx") is False


def test_handler_logs_a_warning_on_rejection(caplog):
    """Rejections must be logged so operators can see blocked uploads."""
    with caplog.at_level("WARNING", logger=mime_validator.__name__):
        _validate_ooxml_archive(b"PK\x03\x04garbage", "docx", "fake.docx")

    assert any("Invalid OOXML archive" in record.message for record in caplog.records)


# ── The happy path must still work ───────────────────────────────────────────


def test_valid_docx_is_accepted():
    """The fix must not turn valid packages into rejections."""
    assert _validate_ooxml_archive(build_docx(), "docx", "essay.docx") is True


@pytest.mark.parametrize("extension", sorted(OOXML_EXTENSIONS))
def test_unsupported_extension_still_raises_value_error(extension):
    """``ValueError`` for unsupported extensions is deliberate, not caught.

    It signals a programming error rather than a bad upload, so it must keep
    propagating even now that the handler is reachable again.
    """
    assert extension in OOXML_EXTENSIONS

    with pytest.raises(ValueError, match="Unsupported OOXML extension"):
        _validate_ooxml_archive(build_docx(), "pdf", "essay.pdf")
