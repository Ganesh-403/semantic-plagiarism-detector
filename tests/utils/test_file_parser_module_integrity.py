# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Regression tests for the file_parser module structure (Issue #2197).

A bad merge left a stray, bodiless ``def extract_text_from_pdf(...)`` line in
the middle of ``src/utils/file_parser.py``. Python requires an indented block
after a function definition, so the module raised ``IndentationError`` at import
time and everything it exports — ``truncate_filename()``,
``get_file_mime_type_from_bytes()``, ``validate_pdf_page_count()``,
``EncryptedPDFError``, ``extract_pdf_metadata()`` — became unreachable.

The real implementation of ``extract_text_from_pdf()`` was never lost; it still
lives further down the file. These tests therefore assert two things:

1. The module parses, imports, and exposes exactly one definition of each
   public helper.
2. ``extract_text_from_pdf()`` still behaves correctly, and now releases its
   PyMuPDF document handle on the ``EncryptedPDFError`` paths too.
"""

from __future__ import annotations

import ast
import inspect

import fitz
import pytest

from src.utils import file_parser
from src.utils.file_parser import (
    EncryptedPDFError,
    extract_text_from_pdf,
    truncate_filename,
    validate_pdf_page_count,
)

SAMPLE_TEXT = "Semantic similarity detection sample page"
PASSWORD = "secret123"


def make_pdf(page_count: int = 1, text: str = SAMPLE_TEXT) -> bytes:
    """Build a small in-memory PDF with ``page_count`` pages."""
    doc = fitz.open()
    for page_number in range(page_count):
        page = doc.new_page()
        page.insert_text((50, 50), f"{text} {page_number + 1}")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_encrypted_pdf(password: str = PASSWORD) -> bytes:
    """Build an AES-256 encrypted single-page PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), SAMPLE_TEXT)
    pdf_bytes = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw=password,
        owner_pw="owner123",
    )
    doc.close()
    return pdf_bytes


# ── Module structure ─────────────────────────────────────────────────────────


def test_module_source_compiles():
    """The module source must be syntactically valid Python.

    Guards against the exact failure of #2197: a function definition with no
    indented block beneath it.
    """
    source = inspect.getsource(file_parser)

    # Raises IndentationError on the broken revision.
    ast.parse(source)


def test_every_function_has_a_body():
    """No top-level function may be left as a bare signature.

    A stray ``def`` line is what broke the module. This walks the AST and
    fails on any function whose body is empty — which the parser cannot
    even represent, so in practice this asserts nothing regressed into a
    bare ``pass`` stub either.
    """
    tree = ast.parse(inspect.getsource(file_parser))

    stubs = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Pass)
    ]

    assert not stubs, f"functions left as empty stubs: {stubs}"


def test_public_helpers_are_defined_exactly_once():
    """Duplicate definitions silently shadow each other — reject them.

    ``extract_text_from_pdf`` was declared twice: once as the stray bodiless
    signature and once as the real implementation.
    """
    tree = ast.parse(inspect.getsource(file_parser))

    names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    duplicates = sorted({name for name in names if names.count(name) > 1})

    assert not duplicates, f"functions defined more than once: {duplicates}"


def test_fitz_imported_exactly_once():
    """``import fitz`` appeared twice, once mid-module between functions."""
    tree = ast.parse(inspect.getsource(file_parser))

    fitz_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        and any(alias.name == "fitz" for alias in node.names)
    ]

    assert len(fitz_imports) == 1, "fitz should be imported once, in the header block"


def test_collateral_exports_are_importable():
    """Helpers that were collaterally dead must be reachable again."""
    assert truncate_filename("a" * 50, max_len=10) == "a" * 7 + "..."
    assert callable(file_parser.get_file_mime_type_from_bytes)
    assert callable(file_parser.extract_pdf_metadata)
    assert issubclass(EncryptedPDFError, Exception)


# ── extract_text_from_pdf behaviour ──────────────────────────────────────────


def test_extracts_text_from_unprotected_pdf():
    """A plain PDF yields its text and reports ``is_protected`` as False."""
    text, is_protected = extract_text_from_pdf(make_pdf())

    assert SAMPLE_TEXT in text
    assert is_protected is False


def test_pages_are_joined_with_newlines():
    """Multi-page text is concatenated one page per newline-separated block."""
    text, _ = extract_text_from_pdf(make_pdf(page_count=3))

    assert text.count("\n") >= 2
    for page_number in (1, 2, 3):
        assert f"{SAMPLE_TEXT} {page_number}" in text


def test_encrypted_pdf_without_password_raises():
    with pytest.raises(EncryptedPDFError, match="Password required"):
        extract_text_from_pdf(make_encrypted_pdf())


def test_encrypted_pdf_with_wrong_password_raises():
    with pytest.raises(EncryptedPDFError, match="Incorrect password"):
        extract_text_from_pdf(make_encrypted_pdf(), password="not-the-password")


def test_encrypted_pdf_with_correct_password_is_read():
    text, is_protected = extract_text_from_pdf(
        make_encrypted_pdf(),
        password=PASSWORD,
    )

    assert SAMPLE_TEXT in text
    assert is_protected is True


def test_page_limit_is_enforced_before_extraction(monkeypatch):
    """The page-count guard runs first, so oversized PDFs never get opened."""
    monkeypatch.setattr(
        file_parser,
        "validate_pdf_page_count",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("too many pages")),
    )

    with pytest.raises(ValueError, match="too many pages"):
        extract_text_from_pdf(make_pdf())


# ── Document handle lifetime ─────────────────────────────────────────────────


class _ClosingSpy:
    """Wraps ``fitz.open`` and records whether each document was closed."""

    def __init__(self, real_open):
        self._real_open = real_open
        self.opened: list[fitz.Document] = []

    def __call__(self, *args, **kwargs):
        doc = self._real_open(*args, **kwargs)
        self.opened.append(doc)
        return doc

    @property
    def all_closed(self) -> bool:
        return all(doc.is_closed for doc in self.opened)


@pytest.fixture
def closing_spy(monkeypatch):
    spy = _ClosingSpy(fitz.open)
    monkeypatch.setattr(file_parser.fitz, "open", spy)
    return spy


def test_handle_is_closed_on_success(closing_spy):
    extract_text_from_pdf(make_pdf())

    assert closing_spy.opened, "expected the PDF to be opened"
    assert closing_spy.all_closed


def test_handle_is_closed_when_password_is_missing(closing_spy):
    """The missing-password path used to return without closing the document."""
    with pytest.raises(EncryptedPDFError):
        extract_text_from_pdf(make_encrypted_pdf())

    assert closing_spy.opened, "expected the PDF to be opened"
    assert closing_spy.all_closed


def test_handle_is_closed_when_password_is_wrong(closing_spy):
    """The rejected-password path used to leak the document handle too."""
    with pytest.raises(EncryptedPDFError):
        extract_text_from_pdf(make_encrypted_pdf(), password="wrong")

    assert closing_spy.opened, "expected the PDF to be opened"
    assert closing_spy.all_closed


def test_validate_pdf_page_count_still_works():
    """The page-count guard defined just above the stray def is intact."""
    assert validate_pdf_page_count(make_pdf(page_count=2), max_pages=5) == 2

    with pytest.raises(ValueError, match="maximum allowed page limit"):
        validate_pdf_page_count(make_pdf(page_count=4), max_pages=2)
