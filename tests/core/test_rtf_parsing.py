import io

from src.core import document_parser
from src.core.document_parser import extract_text_from_rtf
from src.core.parsers.text_parser import RTF_MAX_FILE_SIZE_BYTES


def test_extract_text_from_valid_rtf_bytesio():
    """Test extracting text from a valid RTF BytesIO buffer."""

    rtf_content = r"{\rtf1\ansi" r"{\fonttbl{\f0 Arial;}}" r"\f0\fs24 Hello World!" r"}"

    file = io.BytesIO(rtf_content.encode("utf-8"))

    text = extract_text_from_rtf(file)

    assert text == "Hello World!"


def test_extract_text_from_unicode_escape_rtf():
    """Test parsing RTF containing Unicode escape sequences."""

    rtf_content = r"{\rtf1\ansi\uc1" r"Unicode \u8211? Test" r"}"

    file = io.BytesIO(rtf_content.encode("utf-8"))

    text = extract_text_from_rtf(file)

    assert isinstance(text, str)
    assert text.strip() != ""
    assert "Unicode" in text


def test_extract_text_from_rtf_rejects_oversized_bytes_before_striprtf(monkeypatch):
    """Oversized RTF payloads are rejected before striprtf is invoked."""

    called = False

    def fail_if_called(_content):
        nonlocal called
        called = True
        raise AssertionError("striprtf must not receive oversized RTF")

    monkeypatch.setattr(document_parser, "rtf_to_text", fail_if_called)

    oversized = b"{\\rtf1 " + b"x" * RTF_MAX_FILE_SIZE_BYTES

    assert document_parser.extract_text_from_rtf(oversized) == ""
    assert called is False


def test_extract_text_from_rtf_accepts_payload_at_size_limit(monkeypatch):
    """RTF payloads exactly at the 10 MB limit remain eligible for parsing."""

    called_with = None

    def fake_rtf_to_text(content):
        nonlocal called_with
        called_with = content
        return "ok"

    monkeypatch.setattr(document_parser, "rtf_to_text", fake_rtf_to_text)

    payload = b"x" * RTF_MAX_FILE_SIZE_BYTES

    assert document_parser.extract_text_from_rtf(payload) == "ok"
    assert called_with == payload.decode("utf-8")


def test_extract_text_from_rtf_rejects_oversized_bytesio_before_reading(monkeypatch):
    """An oversized BytesIO buffer is rejected without consuming it."""

    called = False

    def fail_if_called(_content):
        nonlocal called
        called = True
        raise AssertionError("striprtf must not receive oversized RTF")

    monkeypatch.setattr(document_parser, "rtf_to_text", fail_if_called)

    import io

    stream = io.BytesIO(b"x" * (RTF_MAX_FILE_SIZE_BYTES + 1))
    stream.seek(0)

    assert document_parser.extract_text_from_rtf(stream) == ""
    assert stream.tell() == 0
    assert called is False


def test_extract_text_from_rtf_rejects_oversized_file_path_before_read(
    tmp_path, monkeypatch
):
    """An oversized RTF file on disk is rejected using its file size first."""

    called = False

    def fail_if_called(_content):
        nonlocal called
        called = True
        raise AssertionError("striprtf must not receive oversized RTF")

    monkeypatch.setattr(document_parser, "rtf_to_text", fail_if_called)

    path = tmp_path / "large.rtf"
    with path.open("wb") as handle:
        handle.truncate(RTF_MAX_FILE_SIZE_BYTES + 1)

    assert document_parser.extract_text_from_rtf(str(path)) == ""
    assert called is False
