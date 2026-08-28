# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
    assert "Unsupported Media Type: Request must be " in source
    assert '"application/json"' in source


def test_requested_unit_tests_are_in_api_test_file():
    source = API_TEST_PATH.read_text(encoding="utf-8")

    assert "test_json_middleware_rejects_non_json_post_payload" in source
    assert "test_json_middleware_rejects_missing_content_type_with_body" in source
    assert "test_json_middleware_accepts_application_json" in source
