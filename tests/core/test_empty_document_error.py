"""
tests/core/test_empty_document_error.py
---------------------------------------
Comprehensive unit tests for the EmptyDocumentError exception and its
integration into the document parsing and UI/CLI pipelines (Issue #2724).
"""

from unittest.mock import patch

import pytest

from src.core.document_parser import extract_text
from src.errors import EmptyDocumentError


class TestEmptyDocumentErrorClass:
    """Test suite for the EmptyDocumentError exception class itself."""

    def test_default_message_formatting(self):
        """Verify the default message includes the filename."""
        err = EmptyDocumentError("blank.pdf")
        assert str(err) == "The document 'blank.pdf' contains no readable text."
        assert err.filename == "blank.pdf"

    def test_custom_message_override(self):
        """Verify a custom message can be provided."""
        custom_msg = "Custom error for test.docx"
        err = EmptyDocumentError("test.docx", message=custom_msg)
        assert str(err) == custom_msg
        assert err.message == custom_msg

    def test_inherits_from_value_error(self):
        """Verify EmptyDocumentError is a subclass of ValueError."""
        assert issubclass(EmptyDocumentError, ValueError)

        err = EmptyDocumentError("test.pdf")
        assert isinstance(err, ValueError)

    def test_can_be_caught_as_value_error(self):
        """Verify it can be caught using a standard ValueError except block."""
        with pytest.raises(ValueError):
            raise EmptyDocumentError("test.pdf")


class TestExtractTextEmptyDocument:
    """Test suite for extract_text raising EmptyDocumentError."""

    @patch("src.core.document_parser.extract_text_from_pdf", return_value="")
    @patch("src.core.document_parser.detect_text_language", return_value="en")
    def test_raises_on_empty_pdf(self, mock_lang, mock_extract):
        """Verify EmptyDocumentError is raised when PDF extraction returns empty."""
        with pytest.raises(EmptyDocumentError) as exc_info:
            extract_text(b"fake pdf bytes", "empty.pdf")

        assert "empty.pdf" in str(exc_info.value)

    @patch(
        "src.core.document_parser.extract_text_from_txt", return_value="   \n\n  \t  "
    )
    @patch("src.core.document_parser.detect_text_language", return_value="en")
    def test_raises_on_whitespace_only_txt(self, mock_lang, mock_extract):
        """Verify EmptyDocumentError is raised when text is only whitespace."""
        with pytest.raises(EmptyDocumentError):
            extract_text(b"   \n\n  \t  ", "whitespace.txt")

    @patch("src.core.document_parser.extract_text_from_docx", return_value="")
    @patch("src.core.document_parser.detect_text_language", return_value="en")
    def test_raises_on_empty_docx(self, mock_lang, mock_extract):
        """Verify EmptyDocumentError is raised for empty DOCX files."""
        with pytest.raises(EmptyDocumentError):
            extract_text(b"fake docx bytes", "blank.docx")

    @patch(
        "src.core.document_parser.extract_text_from_txt",
        return_value="Valid text content here.",
    )
    @patch("src.core.document_parser.detect_text_language", return_value="en")
    def test_does_not_raise_on_valid_text(self, mock_lang, mock_extract):
        """Verify no error is raised when valid text is extracted."""
        result = extract_text(b"Valid text content here.", "valid.txt")
        assert result == "Valid text content here."


class TestPipelineIntegration:
    """Test suite for UI and CLI pipeline integration."""

    def test_streamlit_pipeline_catches_empty_error(self):
        """Verify the Streamlit pipeline catches the error and doesn't crash."""
        # Simulate the pipeline logic
        file_bytes_dict = {"bad.txt": b"", "good.txt": b"A readable document about semantic analysis."}
        raw_texts = {}
        failed_documents = []

        for name, data in file_bytes_dict.items():
            try:
                extracted = extract_text(data, name)
                raw_texts[name] = extracted
            except EmptyDocumentError as ede:
                failed_documents.append({"filename": name, "error": str(ede)})
            except Exception:
                pass

        assert "bad.txt" not in raw_texts
        assert raw_texts["good.txt"] == "A readable document about semantic analysis."
        assert len(failed_documents) == 1
        assert failed_documents[0]["filename"] == "bad.txt"
        assert "no readable text" in failed_documents[0]["error"]

    @patch(
        "src.core.document_parser.extract_text",
        side_effect=EmptyDocumentError("bad.pdf"),
    )
    def test_cli_pipeline_skips_empty_file(self, mock_extract, capsys):
        """Verify the CLI pipeline skips the file and prints a warning."""
        # Simulate CLI logic
        filename = "bad.pdf"
        skipped = []

        try:
            extract_text(b"bytes", filename)
        except EmptyDocumentError as ede:
            import sys

            sys.stderr.write(f"⚠️  Warning: {ede}\n")
            skipped.append(filename)

        assert "bad.pdf" in skipped
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "bad.pdf" in captured.err
