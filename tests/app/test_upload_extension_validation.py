from pathlib import Path


APP_PATH = Path("app/streamlit_app.py")


def test_upload_flow_validates_final_extension():
    source = APP_PATH.read_text(encoding="utf-8")

    assert "validate_document_extension(" in source
    assert "except InvalidFileExtensionError as exc:" in source


def test_upload_rejection_occurs_before_file_is_processed():
    source = APP_PATH.read_text(encoding="utf-8")

    validation = source.index(
        "validate_document_extension("
    )
    insertion = source.index(
        "safe_name = unique_filename(",
        validation,
    )

    assert validation < insertion
