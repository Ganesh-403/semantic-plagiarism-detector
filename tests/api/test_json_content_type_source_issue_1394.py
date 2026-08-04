from pathlib import Path


ASGI_PATH = Path("src/asgi_app.py")
API_TEST_PATH = Path("tests/api/test_app.py")


def test_required_middleware_exists_and_is_registered():
    source = ASGI_PATH.read_text(encoding="utf-8")

    assert "class JSONContentTypeMiddleware(" in source
    assert "Middleware(JSONContentTypeMiddleware)" in source


def test_unsupported_media_type_response_is_defined():
    source = ASGI_PATH.read_text(encoding="utf-8")

    assert "status_code=415" in source
    assert (
        "Unsupported Media Type: Request must be "
        in source
    )
    assert '"application/json"' in source


def test_requested_unit_tests_are_in_api_test_file():
    source = API_TEST_PATH.read_text(encoding="utf-8")

    assert (
        "test_json_middleware_rejects_non_json_post_payload"
        in source
    )
    assert (
        "test_json_middleware_rejects_missing_content_type_with_body"
        in source
    )
    assert (
        "test_json_middleware_accepts_application_json"
        in source
    )
