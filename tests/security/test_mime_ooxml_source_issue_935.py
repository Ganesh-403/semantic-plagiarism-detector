from pathlib import Path


MIME_PATH = Path("src/security/mime_validator.py")


def test_ooxml_container_validation_exists():
    source = MIME_PATH.read_text(encoding="utf-8")

    assert "def _validate_ooxml_archive(" in source
    assert '"[Content_Types].xml"' in source
    assert '"word/document.xml"' in source
    assert '"xl/workbook.xml"' in source


def test_ooxml_validation_runs_before_libmagic():
    source = MIME_PATH.read_text(encoding="utf-8")

    ooxml_position = source.index(
        "if extension in OOXML_EXTENSIONS:"
    )
    magic_position = source.index(
        "magic_result = _check_magic_bytes("
    )

    assert ooxml_position < magic_position


def test_xlsx_extension_is_registered():
    source = MIME_PATH.read_text(encoding="utf-8")

    assert '"xlsx": {' in source
    assert (
        "spreadsheetml.sheet"
        in source
    )
