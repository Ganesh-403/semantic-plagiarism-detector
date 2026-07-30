from unittest.mock import patch
from src.security.mime_validator import validate_mime_type

def test_validate_mime_type_pdf():
    # Valid PDF signature
    valid_pdf_bytes = b"%PDF-1.4\n%...\n"
    assert validate_mime_type(valid_pdf_bytes, "test.pdf") is True

    # Invalid PDF (e.g. an executable renamed to .pdf)
    invalid_pdf_bytes = b"MZ\x90\x00\x03\x00\x00\x00"
    assert validate_mime_type(invalid_pdf_bytes, "malicious.pdf") is False

def test_validate_mime_type_docx():
    # Valid zip/docx signature
    valid_docx_bytes = b"PK\x03\x04\x14\x00\x08\x00\x08\x00"
    assert validate_mime_type(valid_docx_bytes, "report.docx") is True

    # Invalid docx signature
    invalid_docx_bytes = b"Not a zip file at all"
    assert validate_mime_type(invalid_docx_bytes, "report.docx") is False

def test_validate_mime_type_text():
    # Valid text bytes
    valid_txt_bytes = "Hello World! This is some essay content.".encode("utf-8")
    assert validate_mime_type(valid_txt_bytes, "essay.txt") is True
    assert validate_mime_type(valid_txt_bytes, "essay.md") is True
    assert validate_mime_type(valid_txt_bytes, "data.csv") is True

    # Binary bytes for text file should fail decoding check
    invalid_txt_bytes = b"\x00\xff\xfe\xffHello"
    assert validate_mime_type(invalid_txt_bytes, "essay.txt") is False

def test_validate_mime_type_empty():
    assert validate_mime_type(b"", "empty.pdf") is False

def test_validate_mime_type_unsupported_extension():
    assert validate_mime_type(b"some content", "file.exe") is False

def test_validate_mime_type_magic_fallback():
    # Test that fallback to headers works even if magic raises an exception
    with patch("magic.from_buffer", side_effect=ImportError("No magic module")):
        valid_pdf_bytes = b"%PDF-1.4\n%...\n"
        assert validate_mime_type(valid_pdf_bytes, "test.pdf") is True
        
        invalid_pdf_bytes = b"MZ\x90\x00\x03\x00\x00\x00"
        assert validate_mime_type(invalid_pdf_bytes, "malicious.pdf") is False
