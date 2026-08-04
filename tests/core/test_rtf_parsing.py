import io

from src.core.document_parser import extract_text_from_rtf


def test_extract_text_from_valid_rtf_bytesio():
    """Test extracting text from a valid RTF BytesIO buffer."""

    rtf_content = (
        r"{\rtf1\ansi"
        r"{\fonttbl{\f0 Arial;}}"
        r"\f0\fs24 Hello World!"
        r"}"
    )

    file = io.BytesIO(rtf_content.encode("utf-8"))

    text = extract_text_from_rtf(file)

    assert text == "Hello World!"


def test_extract_text_from_unicode_escape_rtf():
    """Test parsing RTF containing Unicode escape sequences."""

    rtf_content = (
        r"{\rtf1\ansi\uc1"
        r"Unicode \u8211? Test"
        r"}"
    )

    file = io.BytesIO(rtf_content.encode("utf-8"))

    text = extract_text_from_rtf(file)

    assert isinstance(text, str)
    assert text.strip() != ""
    assert "Unicode" in text
