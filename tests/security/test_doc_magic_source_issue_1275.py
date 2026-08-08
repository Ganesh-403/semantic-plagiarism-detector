from pathlib import Path

SOURCE = Path("src/security/mime_validator.py")
TESTS = Path("tests/security/test_mime_validator.py")


def test_complete_ole_header_is_configured():
    source = SOURCE.read_text(encoding="utf-8")
    assert r'b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"' in source


def test_doc_header_is_checked_before_mime_detection():
    source = SOURCE.read_text(encoding="utf-8")
    doc_check = source.index('if extension == "doc":')
    magic_check = source.index(
        "magic_result = _check_magic_bytes",
        doc_check,
    )
    assert doc_check < magic_check


def test_invalid_doc_header_regression_test_exists():
    source = TESTS.read_text(encoding="utf-8")
    assert "test_validate_mime_type_rejects_invalid_legacy_doc_header" in source
