from pathlib import Path

SOURCE = Path("src/utils/file_parser.py")
TESTS = Path("tests/utils/test_file_parser.py")


def test_required_helper_signature_exists():
    source = SOURCE.read_text(encoding="utf-8")

    assert "def validate_pdf_page_count(" in source
    assert "file_bytes: bytes" in source
    assert "max_pages: int = 500" in source
    assert ") -> int:" in source


def test_required_default_limit_error_exists():
    source = SOURCE.read_text(encoding="utf-8")

    assert '"PDF exceeds maximum allowed page limit "' in source
    assert 'f"({max_pages} pages)"' in source


def test_unit_test_and_extraction_integration_exist():
    source = SOURCE.read_text(encoding="utf-8")
    tests = TESTS.read_text(encoding="utf-8")

    # Extraction validates the already-open document without parsing it twice.
    assert "if doc.page_count > 500:" in source
    assert "class TestPDFPageCountValidation:" in tests
    assert "test_validate_pdf_page_count_rejects_over_default_limit" in tests
