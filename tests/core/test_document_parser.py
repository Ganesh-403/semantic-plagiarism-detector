import io
import shutil
from unittest.mock import MagicMock, patch

import docx

from src.core.document_parser import (clean_text, extract_text,
                                      extract_text_from_docx,
                                      extract_text_from_pdf,
                                      extract_text_from_txt, extract_texts,
                                      remove_ignore_phrases,
                                      strip_bibliography)

# Skip OCR tests when Tesseract binary is not present on this machine
TESSERACT_AVAILABLE = shutil.which("tesseract") is not None


def _make_pdf_bytes(text: str) -> bytes:
    """Create a minimal in-memory PDF containing the given text."""
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    # Ensure there are enough words to bypass OCR fallback (at least 8 words)
    words = (text + " word" * 10).split()
    c.drawString(50, 150, " ".join(words))
    c.showPage()
    c.save()
    return buf.getvalue()


def _make_docx_bytes(text: str) -> bytes:
    """Create a minimal in-memory DOCX containing the given text."""
    doc = docx.Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@patch("src.core.document_parser._ocr_pdf_page", return_value="")
def test_extract_from_pdf_bytes(mock_ocr):
    pdf_bytes = _make_pdf_bytes(
        "Hello PDF this is a document with enough words to satisfy native text check"
    )
    # For blank page PDF, pdfplumber might return empty string, but it shouldn't error
    result = extract_text_from_pdf(pdf_bytes)
    assert isinstance(result, str)


def test_extract_from_pdf_filters_repeated_headers_page_numbers_and_whitespace():
    # Build mock pages where header/footer lines are REPEATED across pages
    # and page numbers sit on their own dedicated lines (so the filter strips them)
    page_one = MagicMock()
    page_one.extract_text.return_value = (
        "Research Report\n"
        "Introduction\n"
        "This section introduces the topic in detail with enough words.\n"
        "Page 1"
    )
    page_two = MagicMock()
    page_two.extract_text.return_value = (
        "Research Report\n"
        "Body content is written here at length for analysis purposes.\n"
        "Page 2"
    )

    fake_pdf = MagicMock()
    fake_pdf.pages = [page_one, page_two]
    fake_pdf.__enter__ = MagicMock(return_value=fake_pdf)
    fake_pdf.__exit__ = MagicMock(return_value=False)

    with patch("src.core.document_parser.pdfplumber.open", return_value=fake_pdf):
        result = extract_text_from_pdf(io.BytesIO(b"%PDF-fake-pdf"))

    # Repeated header across all pages must be stripped
    assert "Research Report" not in result
    # Standalone page-number lines must be stripped
    assert "Page 1" not in result
    assert "Page 2" not in result
    # Body content must survive
    assert "Introduction" in result
    assert "Body content" in result
    assert "\n\n\n" not in result


def test_extract_from_docx_bytes():
    docx_bytes = _make_docx_bytes("Hello DOCX")
    result = extract_text_from_docx(docx_bytes)
    assert result == "Hello DOCX"


def test_extract_from_txt_bytes():
    txt_bytes = b"Hello TXT"
    result = extract_text_from_txt(txt_bytes)
    assert result == "Hello TXT"


@patch("src.core.document_parser._ocr_pdf_page", return_value="")
def test_extract_text_routing(mock_ocr):
    pdf_bytes = _make_pdf_bytes(
        "Hello PDF this is a document with enough words to satisfy native text check"
    )
    docx_bytes = _make_docx_bytes("Hello DOCX")
    txt_bytes = b"Hello TXT"

    assert isinstance(extract_text(pdf_bytes, "test.pdf"), str)
    assert extract_text(docx_bytes, "test.docx") == "Hello DOCX"
    assert extract_text(txt_bytes, "test.txt") == "Hello TXT"
    # Fallback case
    assert extract_text(txt_bytes, "test.unknown") == "Hello TXT"


def test_extract_texts_mixed():
    docx_bytes = _make_docx_bytes("Hello DOCX")
    txt_bytes = b"Hello TXT"

    mock_file1 = MagicMock()
    mock_file1.name = "doc1.docx"
    mock_file1.read.return_value = docx_bytes

    mock_file2 = MagicMock()
    mock_file2.name = "doc2.txt"
    mock_file2.read.return_value = txt_bytes

    # Mock extract_text to isolate testing of extract_texts structure
    with patch(
        "src.core.document_parser.extract_text",
        side_effect=lambda f, name, **kwargs: f"Parsed {name}",
    ):
        results = extract_texts([mock_file1, mock_file2])

    assert results["doc1.docx"] == "Parsed doc1.docx"
    assert results["doc2.txt"] == "Parsed doc2.txt"


# ---------------------------------------------------------------------------
# strip_bibliography tests (Issue #116)
# ---------------------------------------------------------------------------


class TestStripBibliography:

    def test_strips_references_header(self):
        text = "Some body text.\n\nReferences\n[1] Smith, 2020.\n[2] Jones, 2021."
        result = strip_bibliography(text)
        assert result == "Some body text."
        assert "Smith" not in result

    def test_strips_works_cited(self):
        text = "Analysis complete.\n\nWorks Cited\nDoe, J. (2019). Paper."
        result = strip_bibliography(text)
        assert result == "Analysis complete."

    def test_strips_bibliography_header(self):
        text = "Conclusion drawn.\n\nBibliography\nAdams, B. Book."
        result = strip_bibliography(text)
        assert result == "Conclusion drawn."

    def test_strips_citations_header(self):
        text = "Findings discussed.\n\nCitations\nLee, 2018."
        result = strip_bibliography(text)
        assert result == "Findings discussed."

    def test_strips_reference_list_header(self):
        text = "Summary provided.\n\nReference List\nWang, 2022."
        result = strip_bibliography(text)
        assert result == "Summary provided."

    def test_strips_sources_header(self):
        text = "Method described.\n\nSources\nData from WHO."
        result = strip_bibliography(text)
        assert result == "Method described."

    def test_case_insensitive(self):
        text = "Body here.\n\nREFERENCES\n[1] entry."
        result = strip_bibliography(text)
        assert result == "Body here."

    def test_preserves_normal_text(self):
        text = (
            "Introduction section with enough words to be meaningful.\n\n"
            "Methodology describes the approach used in this study.\n\n"
            "Results show significant improvement over baseline.\n\n"
            "Conclusion summarizes key findings."
        )
        result = strip_bibliography(text)
        assert result == text

    def test_no_bibliography_unchanged(self):
        text = "Just a plain document with no special headers at all."
        assert strip_bibliography(text) == text

    def test_empty_string(self):
        assert strip_bibliography("") == ""

    def test_inline_references_not_stripped(self):
        text = "The references to prior work are important.\nMore text follows."
        result = strip_bibliography(text)
        assert result == text

    def test_bibliography_not_at_start_of_line_not_stripped(self):
        text = "The Bibliography section was reviewed.\nMore text."
        result = strip_bibliography(text)
        assert result == text

    def test_extract_text_strips_bibliography_from_txt(self):
        txt_bytes = b"Body text.\n\nReferences\n[1] Entry one."
        result = extract_text(txt_bytes, "test.txt")
        assert "References" not in result
        assert "Body text" in result

    def test_extract_text_strips_bibliography_from_docx(self):
        docx_bytes = _make_docx_bytes("Body content.\n\nBibliography\nEntry one.")
        result = extract_text(docx_bytes, "test.docx")
        assert "Bibliography" not in result
        assert "Body content" in result


# ---------------------------------------------------------------------------
# remove_ignore_phrases tests (Issue #161)
# ---------------------------------------------------------------------------


class TestRemoveIgnorePhrases:

    def test_removes_single_phrase(self):
        text = (
            "Q1: Explain the theory of relativity. This is my answer about relativity."
        )
        ignore_phrases = "Q1: Explain the theory of relativity"
        result = remove_ignore_phrases(text, ignore_phrases)
        assert "Q1: Explain the theory of relativity" not in result
        assert "This is my answer about relativity" in result

    def test_removes_multiple_phrases(self):
        text = "Q1: First question. My answer to first. Q2: Second question. My answer to second."
        ignore_phrases = "Q1: First question\nQ2: Second question"
        result = remove_ignore_phrases(text, ignore_phrases)
        assert "Q1: First question" not in result
        assert "Q2: Second question" not in result
        assert "My answer to first" in result
        assert "My answer to second" in result

    def test_empty_ignore_phrases_returns_original(self):
        text = "This is my original text."
        ignore_phrases = ""
        result = remove_ignore_phrases(text, ignore_phrases)
        assert result == text

    def test_whitespace_only_ignore_phrases_returns_original(self):
        text = "This is my original text."
        ignore_phrases = "   \n\n   "
        result = remove_ignore_phrases(text, ignore_phrases)
        assert result == text

    def test_none_ignore_phrases_returns_original(self):
        text = "This is my original text."
        result = remove_ignore_phrases(text, "")
        assert result == text

    def test_cleans_extra_whitespace(self):
        text = "Q1: Question text.\n\n\nMy answer here."
        ignore_phrases = "Q1: Question text."
        result = remove_ignore_phrases(text, ignore_phrases)
        assert "Q1: Question text" not in result
        assert "\n\n\n" not in result
        assert "My answer here" in result

    def test_handles_empty_lines_in_ignore_phrases(self):
        text = "Q1: First question. Answer. Q2: Second question. Answer."
        ignore_phrases = "Q1: First question\n\n\nQ2: Second question"
        result = remove_ignore_phrases(text, ignore_phrases)
        assert "Q1: First question" not in result
        assert "Q2: Second question" not in result
        assert "Answer" in result

    def test_case_sensitive_removal(self):
        text = "Q1: Explain the theory. q1: explain the theory."
        ignore_phrases = "Q1: Explain the theory"
        result = remove_ignore_phrases(text, ignore_phrases)
        assert "Q1: Explain the theory" not in result
        assert "q1: explain the theory" in result

    def test_multiple_occurrences_removed(self):
        text = "Instructions: Write in your own words. Paragraph 1. Instructions: Write in your own words. Paragraph 2."
        ignore_phrases = "Instructions: Write in your own words"
        result = remove_ignore_phrases(text, ignore_phrases)
        assert "Instructions: Write in your own words" not in result
        assert "Paragraph 1" in result
        assert "Paragraph 2" in result

    def test_no_match_returns_original(self):
        text = "This is my original text with no matching phrases."
        ignore_phrases = "Q1: Some question\nQ2: Another question"
        result = remove_ignore_phrases(text, ignore_phrases)
        assert result == text


# ---------------------------------------------------------------------------
# clean_text tests
# ---------------------------------------------------------------------------


class TestCleanText:

    def test_collapses_multiple_blank_lines(self):
        text = "Line 1\n\n\n\nLine 2"
        result = clean_text(text)
        assert result == "Line 1\n\nLine 2"

    def test_collapses_multiple_spaces_and_tabs(self):
        text = "Hello     world\t\tPython"
        result = clean_text(text)
        assert result == "Hello world Python"

    def test_replaces_unicode_spaces(self):
        text = "Hello\u00a0World\u200b!"
        result = clean_text(text)
        assert result == "Hello World !"

    def test_removes_spaces_before_newline(self):
        text = "Hello   \nWorld"
        result = clean_text(text)
        assert result == "Hello\nWorld"

    def test_removes_spaces_after_newline(self):
        text = "Hello\n    World"
        result = clean_text(text)
        assert result == "Hello\nWorld"

    def test_strips_leading_and_trailing_whitespace(self):
        text = "   Hello World   \n"
        result = clean_text(text)
        assert result == "Hello World"

    def test_handles_empty_string(self):
        text = ""
        result = clean_text(text)
        assert result == ""

    def test_preserves_normal_text(self):
        text = "This is a normal sentence."
        result = clean_text(text)
        assert result == text

    def test_combines_all_cleaning_steps(self):
        text = "  Hello\t\t\n\n\n  World\u00a0 "
        result = clean_text(text)
        assert result == "Hello\n\nWorld"

    def test_only_whitespace_returns_empty(self):
        text = "   \n\t\n  "
        result = clean_text(text)
        assert result == ""


def test_extract_empty_pdf_gracefully(caplog):
    """Assert that passing an empty/blank PDF returns an empty string gracefully without crashing."""
    import logging

    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.showPage()
    c.save()
    empty_pdf_bytes = buf.getvalue()

    with patch("src.core.document_parser._ocr_pdf_page", return_value=""):
        with caplog.at_level(logging.WARNING):
            result = extract_text_from_pdf(empty_pdf_bytes)

    assert isinstance(result, str)
    assert result.strip() == ""


def test_extract_text_from_doc_success():
    """Test that extract_text_from_doc runs antiword successfully when present."""

    from src.core.document_parser import extract_text_from_doc

    mock_result = MagicMock()
    mock_result.stdout = (
        "This is a test legacy Word Document content extracted by antiword."
    )
    mock_result.returncode = 0

    with patch("shutil.which", return_value="/usr/bin/antiword"):
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = extract_text_from_doc(b"fake doc bytes")
            assert (
                result
                == "This is a test legacy Word Document content extracted by antiword."
            )
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            assert args[0][0] == "antiword"
            assert args[0][1].endswith(".doc")


def test_extract_text_from_doc_missing_antiword():
    """Test that extract_text_from_doc raises RuntimeError if antiword is not installed."""
    import pytest

    from src.core.document_parser import extract_text_from_doc

    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="antiword binary is not installed"):
            extract_text_from_doc(b"fake doc bytes")


def test_extract_text_routing_doc():
    """Test that extract_text routes .doc files to extract_text_from_doc."""
    from src.core.document_parser import extract_text

    mock_result = MagicMock()
    mock_result.stdout = "Legacy Word Doc Content"
    mock_result.returncode = 0

    with patch("shutil.which", return_value="/usr/bin/antiword"):
        with patch("subprocess.run", return_value=mock_result):
            result = extract_text(b"fake bytes", "test_file.doc")
            assert result == "Legacy Word Doc Content"


# ---------------------------------------------------------------------------
# Batch Processing Rate Limiting Tests (Issue #494)
# ---------------------------------------------------------------------------


def test_check_batch_rate_limit_helper():
    """Test check_batch_rate_limit helper raises error when count exceeds 50."""
    import pytest
    from src.core.document_parser import MAX_BATCH_SIZE, check_batch_rate_limit
    from src.errors import PARSER_BATCH_LIMIT_EXCEEDED

    assert MAX_BATCH_SIZE == 50

    # Under limit -> should pass without raising
    check_batch_rate_limit(50)

    # Exceeds limit -> raises ValueError
    with pytest.raises(
        ValueError, match=PARSER_BATCH_LIMIT_EXCEEDED.format(limit=50)
    ):
        check_batch_rate_limit(51)


def test_extract_texts_exceeds_max_batch_size():
    """Test extract_texts raises ValueError when input files exceed 50 documents."""
    import pytest
    from src.core.document_parser import extract_texts
    from src.errors import PARSER_BATCH_LIMIT_EXCEEDED

    files = [MagicMock() for _ in range(51)]
    with pytest.raises(
        ValueError, match=PARSER_BATCH_LIMIT_EXCEEDED.format(limit=50)
    ):
        extract_texts(files)


def test_extract_texts_within_batch_size_limit():
    """Test extract_texts succeeds when input files count is <= 50 documents."""
    from src.core.document_parser import extract_texts

    mock_files = []
    for i in range(5):
        f = MagicMock()
        f.name = f"doc_{i}.txt"
        f.read.return_value = b"sample text"
        mock_files.append(f)

    with patch(
        "src.core.document_parser.extract_text",
        side_effect=lambda data, name, **kwargs: f"Parsed {name}",
    ):
        results = extract_texts(mock_files)

    assert len(results) == 5
    assert results["doc_0.txt"] == "Parsed doc_0.txt"

